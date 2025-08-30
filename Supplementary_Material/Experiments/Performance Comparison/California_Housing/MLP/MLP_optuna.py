# Standard
import os
import math
import json
import random
from datetime import datetime
from typing import Dict, Literal

# General
import numpy as np
import pandas as pd
from tqdm import tqdm

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml, fetch_california_housing

# PyTorch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Hyperparameter Optimization
import optuna

# Tabular Deep Learning Models
from rtdl_revisiting_models import MLP, ResNet, FTTransformer

# Other
import delu


# Scheduler
def get_scheduler(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Seed
def set_random_seed(seed):
    # Set Python random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Dataset
class CustomDataset(Dataset):
    def __init__(self, x, y):
        if isinstance(x, torch.Tensor):
            self.x = x
        elif isinstance(x, np.ndarray):
            self.x = torch.tensor(x, dtype=torch.float32)
        else:  # assume pandas DataFrame
            self.x = torch.tensor(x.to_numpy(), dtype=torch.float32)

        if isinstance(y, torch.Tensor):
            self.labels = y
        elif isinstance(y, np.ndarray):
            self.labels = torch.tensor(y, dtype=torch.float32)
        else:  # assume pandas Series
            self.labels = torch.tensor(y.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]
    
# Early Stopping 
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_rmse, model):
        if self.best_score is None:
            self.best_score = val_rmse
            self.best_model_state = model.state_dict()
        elif val_rmse > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_rmse
            self.best_model_state = model.state_dict()
            self.counter = 0
            

# Model Trainer
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, y_std):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.std    = y_std

    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0.0
        ground_truths, preds_logits = [], []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            self.optimizer.zero_grad()

            # batch = (x, y)
            x, labels = batch
            x = x.to(self.device)
            labels = labels.squeeze(dim=-1).float().to(self.device)

            # Forward pass
            logits = self.model(x).squeeze()  # Only one input: full features
            class_loss = self.criterion(logits.float(), labels)

            total_loss = class_loss
            total_loss.backward()
            train_loss += total_loss.item()

            self.optimizer.step()
            self.scheduler.step()

            preds_logits.append(logits.detach())
            ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, train_loader, train_loss)

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        valid_loss = 0.0
        ground_truths, preds_logits = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating Epoch {epoch+1}", leave=False):

                # batch = (x, y)
                x, labels = batch
                x = x.to(self.device)
                labels = labels.squeeze(dim=-1).float().to(self.device)

                # Forward pass
                logits = self.model(x).squeeze()  # Only one input: full features
                class_loss = self.criterion(logits.float(), labels)
                
                total_loss = class_loss
                valid_loss += total_loss.item()

                preds_logits.append(logits.detach())
                ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, val_loader, valid_loss)  

    
    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0.0
        ground_truths, preds_logits = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}", leave=False):
                
                # batch = (x, y)
                x, labels = batch
                x = x.to(self.device)
                labels = labels.squeeze(dim=-1).float().to(self.device)

                # Forward pass
                logits = self.model(x).squeeze()  # Only one input: full features
                class_loss = self.criterion(logits.float(), labels)
                    
                total_loss = class_loss
                test_loss += total_loss.item()

                preds_logits.append(logits.detach())
                ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, test_loader, test_loss)  


    def _prepare_data(self, data):
        tabular_data, labels = data
        batch_size = len(tabular_data)

        return tabular_data.to(self.device, dtype=torch.long), labels.to(self.device, dtype=torch.float)  


    def _evaluate(self, ground_truths, preds, loader, loss):
        ground_truths = torch.cat(ground_truths)
        preds = torch.cat(preds)

        loss /= len(loader)
        rmse = torch.sqrt(F.mse_loss(preds, ground_truths)) * self.std.item()

        return loss, rmse


def objective(trial):
    
    # Define Hyperparameter
    seed = experiment_number = 42
    n_classes = 1
    batch_size = 256

    warmup_epochs = 10
    epochs = 200

    # Suggested hyperparameters (based on the image you provided)
    n_layers        = trial.suggest_int("n_layers", 1, 8)
    layer_size      = trial.suggest_int("layer_size", 1, 512)
    dropout         = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay    = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    learning_rate   = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        
    # Set random seed
    set_random_seed(seed)

    # 取得資料
    df = fetch_california_housing(as_frame=True).frame
    X = df.drop(columns=["MedHouseVal"])
    y_orig = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

    # Train / Val / Test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_orig, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=seed)

    # 對 y 做標準化 (僅用 train 的 mean 和 std)
    y_mean = y_train.mean()
    y_std  = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val   = (y_val   - y_mean) / y_std
    y_test  = (y_test  - y_mean) / y_std

    # Columns to normalize
    NUM_COLS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

    # === Quantile Transformer ===
    train_num_array = X_train[NUM_COLS].to_numpy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1e-5, train_num_array.shape).astype(train_num_array.dtype)

    quantile_transformer = QuantileTransformer(
        n_quantiles=max(min(len(train_num_array) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9
    ).fit(train_num_array + noise)

    num_train = quantile_transformer.transform(X_train[NUM_COLS])
    num_val   = quantile_transformer.transform(X_val[NUM_COLS])
    num_test  = quantile_transformer.transform(X_test[NUM_COLS])

    # Convert to tensors
    num_train_tensor = torch.tensor(num_train, dtype=torch.float32)
    num_val_tensor   = torch.tensor(num_val, dtype=torch.float32)
    num_test_tensor  = torch.tensor(num_test, dtype=torch.float32)
    y_train_tensor   = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor     = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor    = torch.tensor(y_test, dtype=torch.float32)

    # Dataset and Dataloader (assumes CustomDataset is implemented correctly)
    train_dataset = CustomDataset(num_train_tensor, y_train_tensor)
    val_dataset   = CustomDataset(num_val_tensor, y_val_tensor)
    test_dataset  = CustomDataset(num_test_tensor, y_test_tensor)

    train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        d_in=len(NUM_COLS),
        d_out=n_classes,
        n_blocks=n_layers,
        d_block=layer_size,
        dropout=dropout,
    ).to(device)
    criterion = torch.nn.MSELoss()

    # Define total steps and warmup steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, epochs, warmup_epochs)
    early_stopping = EarlyStopping(patience=16, delta=0)

    # For output path
    best_valid_rmse = float("inf")
    all_epoch_metrics = []

    # Initialize trainer
    trainer = ModelTrainer(model, criterion, optimizer, scheduler, device, y_std)

    for epoch in range(epochs):
        
        train_loss, train_rmse = trainer.train(train_loader, epoch)
        valid_loss, valid_rmse = trainer.evaluate(val_loader, epoch)

        # Info
        print(f"Epoch: {epoch + 1}/{epochs}")
        print(f"Training RMSE: {train_rmse:.4f} | Validation RMSE: {valid_rmse:.4f}")

        # === 儲存每個 epoch 的 AUROC ===
        all_epoch_metrics.append({
            "epoch": epoch,
            "train_rmse": train_rmse,
            "valid_rmse": valid_rmse,
        })

        # Save model if improved
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
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


if __name__ == '__main__':
    
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