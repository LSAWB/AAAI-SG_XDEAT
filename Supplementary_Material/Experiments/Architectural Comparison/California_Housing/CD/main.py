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

import json


def main():

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = os.path.join(config.BASE_LOG_DIR, f"{current_time}")
    os.makedirs(base_log_dir, exist_ok=True)
    
    valid_rmses      = []
    test_rmses       = []

    for experiment_number in range(config.EXPERIMENT_REPEAT):
        
        # Set random seed
        set_random_seed(experiment_number)

        # Hyperparameter ( ORIGINAL )
        dim_model   = 496
        ffn_factor  = 0.6959785562242603
        dim_ff      = int(dim_model * ffn_factor)
        
        att_dropout = 0.1640106594366585
        res_dropout = 0.0657935216124125
        ffn_dropout = 0.019683174878093623
        
        learning_rate   = 0.00022612519798826525
        weight_decay    = 4.7474845921303904e-06

        num_layers_cross    = 1
        
        min_samples_leaf = 108
        min_impurity_decrease = 0.0026543992421433605

        tree_kwargs = {
            'min_samples_leaf': min_samples_leaf,
            'min_impurity_decrease': min_impurity_decrease,
        }
        
        # 取得資料
        df = fetch_california_housing(as_frame=True).frame
        X = df.drop(columns=["MedHouseVal"])
        y_orig = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

        # Train / Val / Test split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y_orig, test_size=0.2, random_state=experiment_number)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=experiment_number)

        NUM_COLS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        CAT_COLS = []

        # 對 y 做標準化 (僅用 train 的 mean 和 std)
        y_mean = y_train.mean()
        y_std  = y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_val   = (y_val   - y_mean) / y_std
        y_test  = (y_test  - y_mean) / y_std

        train_data = X_train.copy()
        train_data['label'] = y_train

        valid_data = X_val.copy()
        valid_data['label'] = y_val

        test_data = X_test.copy()
        test_data['label'] = y_test
        
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
        
        train_dataset = CustomDataset(None, num_train_tensor, None, y_train)
        val_dataset   = CustomDataset(None, num_val_tensor, None, y_val)
        test_dataset  = CustomDataset(None, num_test_tensor, None, y_test)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset,  batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        # =========================== Model Training ===========================
        model = DualTransformer(
            ori_cardinalities=[],
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
            # no_of_classes=config.NUM_LABELS,
            y_std=y_std
        )
        
        # Start Training
        best_valid_rmse = float("inf")
        print(f"Experiment {experiment_number+1} with random seed {experiment_number}")
        
        best_model_path = os.path.join(base_log_dir, f"best_model_seed_{experiment_number}.pth")

        for epoch in range(config.EPOCHS):
            
            train_loss, train_rmse = trainer.train(train_loader, epoch)
            valid_loss, valid_rmse = trainer.evaluate(val_loader, epoch)

            # Info
            print(f"Epoch: {epoch + 1}/{config.EPOCHS}")
            print(f"Training RMSE: {train_rmse:.4f}")
            print(f"Validation RMSE: {valid_rmse:.4f}")

            # Save model
            if valid_rmse < best_valid_rmse:
                save_file = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
                best_valid_rmse = valid_rmse
                torch.save(save_file, best_model_path)

            # Early stopping
            early_stopping(valid_rmse, model) 
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # After training, load the best model and test
        checkpoint = torch.load(best_model_path)

        # Restore model, optimizer, scheduler, and epoch
        model.load_state_dict(checkpoint["model"])
        
        test_loss, test_rmse = trainer.test(test_loader, epoch)
        print(f"Test RMSE : {test_rmse:.4f}")
        test_rmses.append(test_rmse)
        
        # Delete the best model file after loading
        if os.path.exists(best_model_path):  
            os.remove(best_model_path)  # Deletes the file
            print(f"Deleted best model checkpoint: {best_model_path}")
        else:
            print(f"Checkpoint file not found: {best_model_path}")

    test_rmses = [rmse_tensor.cpu().item() for rmse_tensor in test_rmses]
    print(f"Test RMSE Mean: {np.mean(test_rmses)}")
    print(f"Test RMSE STD : {np.std(test_rmses)}")
    
    return np.mean(test_rmses), np.std(test_rmses)


if __name__ == "__main__":
    main()
