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
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, OrdinalEncoder

# ========== Other ==========
import optuna

import json

from rtdl_num_embeddings import compute_bins, PiecewiseLinearEmbeddings
from feature_engine.encoding import DecisionTreeEncoder

def objective(trial):
    
    experiment_number = 42

    # Set random seed
    set_random_seed(experiment_number)

    # Hyperparameter
    dim_model   = trial.suggest_int('dim_model', 64, 512, step=8)
    ffn_factor  = trial.suggest_float('ffn_factor', 2/3, 8/3)
    dim_ff      = int(dim_model * ffn_factor)
    
    att_dropout = trial.suggest_float('att_dropout', 0.0, 0.5)
    res_dropout = trial.suggest_float('res_dropout', 0.0, 0.2)
    ffn_dropout = trial.suggest_float('ffn_dropout', 0.0, 0.5)
    
    learning_rate   = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay    = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    num_layers_cross    = trial.suggest_int("cross_blocks", 1, 4)
    
    min_samples_leaf = trial.suggest_int("ple_min_samples_leaf", 1, 128)
    min_impurity_decrease = trial.suggest_float("ple_min_impurity_decrease", 1e-9, 0.01, log=True)

    tree_kwargs = {
        'min_samples_leaf': min_samples_leaf,
        'min_impurity_decrease': min_impurity_decrease,
    }

    # Load OpenML Adult dataset 
    df = fetch_openml(data_id=1590, as_frame=True)['frame']
    df['label'] = (df['class'] == '>50K').astype(int)
    df = df.drop(columns=['class'])  # drop the original label column

    CAT_COLS = df.select_dtypes(include=['category', 'object']).columns.tolist()
    NUM_COLS = df.select_dtypes(include=['number']).drop(columns=['label']).columns.tolist()

    # Step 3: Convert object columns to categorical & fill NA as 'NAN'
    for col in CAT_COLS:
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

    for col in NUM_COLS:
        df[col] = df[col].fillna(df[col].median())

    # Handle rare categories (using train_df only to avoid leakage)
    threshold = len(df) * 0.005
    for col in CAT_COLS:
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < threshold].index
        df[col] = df[col].apply(lambda x: 'Others' if x in rare_categories else x)

    # Compute cardinalities
    cat_cardinalities = [df[col].nunique() for col in CAT_COLS]

    # Split training data into train and validation sets
    train_data, test_data = train_test_split(
        df,
        test_size=1/3,
        stratify=df['label'],
        random_state=experiment_number
    )

    train_data, valid_data = train_test_split(
        train_data,
        test_size=1/5,
        stratify=train_data['label'],
        random_state=experiment_number
    )
    
    y_train = train_data['label'].values
    y_val   = valid_data['label'].values
    y_test  = test_data['label'].values

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
        random_state=experiment_number
    )
    dt_encoder.fit(train_data[CAT_COLS], train_data['label'])
    x_train_cat_tar = dt_encoder.transform(train_data[CAT_COLS])
    x_valid_cat_tar = dt_encoder.transform(valid_data[CAT_COLS])
    x_test_cat_tar  = dt_encoder.transform(test_data[CAT_COLS])

    cat_train_tar_tensor = torch.tensor(x_train_cat_tar.to_numpy(), dtype=torch.float32)
    cat_val_tar_tensor   = torch.tensor(x_valid_cat_tar.to_numpy(), dtype=torch.float32)
    cat_test_tar_tensor  = torch.tensor(x_test_cat_tar.to_numpy(), dtype=torch.float32)

    # === Num Raw: Quantile Transformer ===
    train_num_array = train_data[NUM_COLS].to_numpy()
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1e-5, train_num_array.shape).astype(train_num_array.dtype)

    quantile_transformer = QuantileTransformer(
        n_quantiles=max(min(len(train_num_array) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9
    ).fit(train_num_array + noise)
    
    num_train = quantile_transformer.transform(train_data[NUM_COLS])
    num_val   = quantile_transformer.transform(valid_data[NUM_COLS])
    num_test  = quantile_transformer.transform(test_data[NUM_COLS])
    
    num_train_tensor = torch.tensor(num_train, dtype=torch.float32)
    num_val_tensor   = torch.tensor(num_val, dtype=torch.float32)
    num_test_tensor  = torch.tensor(num_test, dtype=torch.float32)
    
    # === Num Tar: Quantile Transformer ===
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze()
    regression = config.NUM_LABELS == 1
    bins = compute_bins(
        X=num_train_tensor,
        y=y_train_tensor,
        tree_kwargs=tree_kwargs,
        regression=regression,
    )
    PLE = PiecewiseLinearEmbeddings(bins, dim_model, activation=False, version='B')

    train_dataset = CustomDataset(cat_train_tensor, num_train_tensor, cat_train_tar_tensor, y_train)
    val_dataset   = CustomDataset(cat_val_tensor, num_val_tensor, cat_val_tar_tensor, y_val)
    test_dataset  = CustomDataset(cat_test_tensor, num_test_tensor, cat_test_tar_tensor, y_test)

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
        PLE=PLE
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
    all_epoch_metrics = []

    for epoch in range(config.EPOCHS):
        
        # Train
        train_loss, train_accuracy = trainer.train(train_loader, epoch)

        # Evaluate
        valid_loss, valid_accuracy = trainer.evaluate(val_loader, epoch)
        
        # Info
        print(f"Epoch: {epoch + 1}/{config.EPOCHS}")
        print(f"Training Accuracy Score: {train_accuracy:.4f}")
        print(f"Validation Accuracy Score: {valid_accuracy:.4f}")

        # === 儲存每個 epoch 的 AUROC ===
        all_epoch_metrics.append({
            "epoch": epoch,
            "train_max_acc": train_accuracy,
            "valid_max_acc": valid_accuracy
        })

        # Save model
        if valid_accuracy > best_valid_acc:
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            best_valid_acc = valid_accuracy
            # torch.save(save_file, best_model_path)

        # Early stopping
        early_stopping(valid_accuracy)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    best_epoch = max(all_epoch_metrics, key=lambda x: x["valid_max_acc"])
    train_acc = best_epoch["train_max_acc"]
    valid_acc = best_epoch["valid_max_acc"]

    return train_acc, valid_acc 


if __name__ == "__main__":
    
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(objective, n_trials=100)

    pareto_results = []
    for t in study.best_trials:
        result = {
            "train_accuarcy": t.values[0],
            "valid_accuarcy": t.values[1],
            "params": t.params
        }
        pareto_results.append(result)

    with open("optuna_pareto_trials.json", "w") as f:
        json.dump(pareto_results, f, indent=4)

    print("✅ Saved Pareto trials to optuna_pareto_trials.json")
