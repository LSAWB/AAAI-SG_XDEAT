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
from sklearn.preprocessing import QuantileTransformer

# ========== Other ==========
import optuna

import json

from rtdl_num_embeddings import compute_bins, PiecewiseLinearEmbeddings


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
    all_epoch_metrics = []

    for epoch in range(config.EPOCHS):
        
        train_loss, train_rmse = trainer.train(train_loader, epoch)
        valid_loss, valid_rmse = trainer.evaluate(val_loader, epoch)

        # Info
        print(f"Epoch: {epoch + 1}/{config.EPOCHS}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {valid_rmse:.4f}")

        # === 儲存每個 epoch ===
        all_epoch_metrics.append({
            "epoch": epoch,
            "train_rmse": train_rmse,
            "valid_rmse": valid_rmse,
        })

        # Save model
        if valid_rmse < best_valid_rmse:
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            best_valid_rmse = valid_rmse
            # torch.save(save_file, best_model_path)

        # Early stopping
        early_stopping(valid_rmse, model) 
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    best_epoch = min(all_epoch_metrics, key=lambda x: x["valid_rmse"])
    train_rmse = best_epoch["train_rmse"]
    valid_rmse = best_epoch["valid_rmse"]

    return train_rmse, valid_rmse 


if __name__ == "__main__":
    
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=100)

    pareto_results = []
    for t in study.best_trials:
        result = {
            "train_min_rmse": t.values[0],
            "valid_min_rmse": t.values[1],
            "params": t.params
        }
        pareto_results.append(result)

    with open("optuna_pareto_trials.json", "w") as f:
        json.dump(pareto_results, f, indent=4)

    print("✅ Saved Pareto trials to optuna_pareto_trials.json")
