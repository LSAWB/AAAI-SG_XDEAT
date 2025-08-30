import config
from utils import set_random_seed, get_scheduler
from dataset import CustomDataset
from model import DualTransformer
from trainer import EarlyStopping, ModelTrainer

# ========== Standard Library ==========
import os
import math
import random
from datetime import datetime

# ========== Third-Party Libraries ==========
import numpy as np
import pandas as pd
from tqdm import tqdm
import openml

# ========== PyTorch ==========
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# ========== Sklearn ==========
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    PowerTransformer,
    StandardScaler,
    QuantileTransformer
)

from rtdl_num_embeddings import compute_bins, PiecewiseLinearEmbeddings
from feature_engine.encoding import DecisionTreeEncoder

import json

def main():

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = os.path.join(config.BASE_LOG_DIR, f"{current_time}")
    os.makedirs(base_log_dir, exist_ok=True)
    
    test_accs       = []

    for experiment_number in range(config.EXPERIMENT_REPEAT):
        
        # Set random seed
        set_random_seed(experiment_number)

        # Hyperparameter
        dim_model   = 312
        ffn_factor  = 2.0633896100323157
        dim_ff      = int(dim_model * ffn_factor)
        
        att_dropout = 0.1737031283665752
        res_dropout = 0.06418583581425419
        ffn_dropout = 0.34434081754678947
        
        learning_rate   = 0.00047359936993672335
        weight_decay    = 0.0003093446237321075

        num_layers_cross    = 3
        
        # min_samples_leaf = hparams['ple_min_samples_leaf']
        # min_impurity_decrease = hparams['ple_min_impurity_decrease']

        # tree_kwargs = {
        #     'min_samples_leaf': min_samples_leaf,
        #     'min_impurity_decrease': min_impurity_decrease,
        # }
    
        # Step 1: Load dataset
        df = fetch_openml(data_id=981, as_frame=True)['frame']
        df['label'] = df['Who_Pays_for_Access_Work'].astype(int)
        df = df.drop(columns=['Who_Pays_for_Access_Work'])

        # Step 2: Convert 'Age' to numeric and bin into groups
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Age_group'] = (df['Age'] // 10).astype('Int64')
        df['Age_group'] = df['Age_group'].astype('category')
        df = df.drop(columns=['Age'])  # Drop original age column

        # Step 3: Convert object columns to categorical & fill NA as 'NAN'
        for col in df.columns:
            if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype("category")
                if "NAN" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("NAN")
                df[col] = df[col].fillna("NAN")

        # Step 4: Rare category consolidation
        rare_threshold = len(df) * 0.005
        categorical_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()

        for cat_col in categorical_cols:
            value_counts = df[cat_col].value_counts()
            rare_values = value_counts[value_counts < rare_threshold].index
            df[cat_col] = df[cat_col].apply(lambda x: 'Others_DATA' if x in rare_values else x)

        # Step 5: Update CAT_COLS and NUM_COLS
        CAT_COLS = df.select_dtypes(include=['category', 'object']).columns.tolist()
        NUM_COLS = df.select_dtypes(include=['number']).drop(columns=['label']).columns.tolist()

        for col in CAT_COLS:
            df[col] = df[col].astype(str)
        
        # Step 6: Column stats
        cat_cardinalities = [df[col].nunique() for col in CAT_COLS]
        
        # Step 1: 先從 df 拿索引做切分
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=1/5,
            stratify=df['label'],
            random_state=experiment_number
        )

        # Step 2: 用 index 做第二次切分，但 stratify 要根據原始 df 對應的 label
        train_idx, valid_idx = train_test_split(
            train_idx,
            test_size=1/5,
            stratify=df.loc[train_idx, 'label'],
            random_state=experiment_number
        )

        train_data = df.loc[train_idx].copy()
        valid_data = df.loc[valid_idx].copy()
        test_data  = df.loc[test_idx].copy()
        
        y_train = train_data['label'].values
        y_val   = valid_data['label'].values
        y_test  = test_data['label'].values

        # # === Num Raw: Quantile Transformer ===
        # train_num_array = train_data[NUM_COLS].to_numpy()
        # rng = np.random.default_rng(0)
        # noise = rng.normal(0.0, 1e-5, train_num_array.shape).astype(train_num_array.dtype)

        # quantile_transformer = QuantileTransformer(
        #     n_quantiles=max(min(len(train_num_array) // 30, 1000), 10),
        #     output_distribution='normal',
        #     subsample=10**9
        # ).fit(train_num_array + noise)
        
        # num_train = quantile_transformer.transform(train_data[NUM_COLS])
        # num_val   = quantile_transformer.transform(valid_data[NUM_COLS])
        # num_test  = quantile_transformer.transform(test_data[NUM_COLS])
        
        # num_train_tensor = torch.tensor(num_train, dtype=torch.float32)
        # num_val_tensor   = torch.tensor(num_val, dtype=torch.float32)
        # num_test_tensor  = torch.tensor(num_test, dtype=torch.float32)
        
        # # === Num Tar: Quantile Transformer ===
        # y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze()
        # regression = config.NUM_LABELS == 1
        # bins = compute_bins(
        #     X=num_train_tensor,
        #     y=y_train_tensor,
        #     tree_kwargs=tree_kwargs,
        #     regression=regression,
        # )
        # PLE = PiecewiseLinearEmbeddings(bins, dim_model, activation=True, version='B')

        # === Cat Raw: Ordinal Encoding === 
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ordinal_encoder.fit(train_data[CAT_COLS])
        x_train_cat = ordinal_encoder.transform(train_data[CAT_COLS])
        x_valid_cat = ordinal_encoder.transform(valid_data[CAT_COLS])
        x_test_cat  = ordinal_encoder.transform(test_data[CAT_COLS])
            
        cat_train_tensor = torch.tensor(x_train_cat, dtype=torch.float32)
        cat_val_tensor   = torch.tensor(x_valid_cat, dtype=torch.float32)
        cat_test_tensor  = torch.tensor(x_test_cat, dtype=torch.float32)

        # === Cat Tar: Decesion Tree Classifier / Regressor === 
        dt_encoder = DecisionTreeEncoder(
            variables=CAT_COLS,
            regression=False,
            cv=5,
            # encoding_method='ordered',
            random_state=experiment_number
        )
        dt_encoder.fit(train_data[CAT_COLS], train_data['label'])
        x_train_cat_tar = dt_encoder.transform(train_data[CAT_COLS])
        x_valid_cat_tar = dt_encoder.transform(valid_data[CAT_COLS])
        x_test_cat_tar  = dt_encoder.transform(test_data[CAT_COLS])

        cat_train_tar_tensor = torch.tensor(x_train_cat_tar.to_numpy(), dtype=torch.float32)
        cat_val_tar_tensor   = torch.tensor(x_valid_cat_tar.to_numpy(), dtype=torch.float32)
        cat_test_tar_tensor  = torch.tensor(x_test_cat_tar.to_numpy(), dtype=torch.float32)

        train_dataset = CustomDataset(cat_train_tensor, None, cat_train_tar_tensor, y_train)
        val_dataset   = CustomDataset(cat_val_tensor, None, cat_val_tar_tensor, y_val)
        test_dataset  = CustomDataset(cat_test_tensor, None, cat_test_tar_tensor, y_test)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset,  batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # =========================== Model Training ===========================
        model = DualTransformer(
            ori_cardinalities=cat_cardinalities,
            num_features=len(NUM_COLS),
            cat_features=len(CAT_COLS),
            dim_model=dim_model,
            num_heads=8,
            dim_ff=dim_ff,
            num_layers_cross=num_layers_cross,
            num_labels=config.NUM_LABELS,
            att_dropout=att_dropout,
            res_dropout=res_dropout,
            ffn_dropout=ffn_dropout,
            PLE=None
        ).to(config.DEVICE)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_scheduler(optimizer, config.EPOCHS, config.WARMUP_EPOCHS)

        # Early Stopping
        early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, delta=config.EARLY_STOPPING_DELTA)

        # Trainer
        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            no_of_classes=config.NUM_LABELS,
        )
        
        # Start Training
        best_valid_acc = 0
        print(f"Experiment {experiment_number+1} with random seed {experiment_number}")
        best_model_path = os.path.join(base_log_dir, f"best_model_seed_{experiment_number}.pth")

        for epoch in range(config.EPOCHS):
        
            # Train
            train_loss, train_accuracy = trainer.train(train_loader, epoch)

            # Evaluate
            valid_loss, valid_accuracy = trainer.evaluate(val_loader, epoch)
            
            # Info
            print(f"Epoch: {epoch + 1}/{config.EPOCHS}")
            print(f"Training Accuracy Score: {train_accuracy:.4f}")
            print(f"Validation Accuracy Score: {valid_accuracy:.4f}")

            # Save model
            if valid_accuracy > best_valid_acc:
                save_file = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
                best_valid_acc = valid_accuracy
                torch.save(save_file, best_model_path)

            # Early stopping
            early_stopping(valid_accuracy)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # After training, load the best model and test
        checkpoint = torch.load(best_model_path)

        # Restore model, optimizer, scheduler, and epoch
        model.load_state_dict(checkpoint["model"])
        
        test_loss, test_accuracy = trainer.test(test_loader, epoch)

        print(f"Test ACC Score: {test_accuracy:.4f}")

        # Store the best AUROC for this experiment
        test_accs.append(test_accuracy)

        # Delete the best model file after loading
        if os.path.exists(best_model_path):  
            os.remove(best_model_path)  # Deletes the file
            print(f"Deleted best model checkpoint: {best_model_path}")
        else:
            print(f"Checkpoint file not found: {best_model_path}")

    # After all experiments, calculate the mean and std of AUROCs
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    print(f"Mean ACC: {mean_acc:.4f}, Std Dev ACC: {std_acc:.4f}")

    return mean_acc, std_acc


if __name__ == "__main__":
    main()

