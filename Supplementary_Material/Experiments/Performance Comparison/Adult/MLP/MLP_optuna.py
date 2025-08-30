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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, QuantileTransformer
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

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
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.best_model_state = model.state_dict()
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.best_model_state = model.state_dict()
            self.counter = 0


# Model Trainer
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0.0
        ground_truths, preds_logits = [], []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            self.optimizer.zero_grad()

            # batch = (x, y)
            x, labels = batch
            x = x.to(self.device)
            labels = labels.squeeze(dim=-1).long().to(self.device)

            # Forward pass
            logits = self.model(x)  # Only one input: full features
            class_loss = self.criterion(logits.float(), labels)

            total_loss = class_loss
            total_loss.backward()
            train_loss += total_loss.item()

            self.optimizer.step()
            self.scheduler.step()

            # probs = torch.sigmoid(logits).detach()
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
                labels = labels.squeeze(dim=-1).long().to(self.device)

                # Forward pass
                logits = self.model(x)  # Only one input: full features
                class_loss = self.criterion(logits.float(), labels)
                
                total_loss = class_loss
                valid_loss += total_loss.item()

                # probs = torch.sigmoid(logits).detach()
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
                labels = labels.squeeze(dim=-1).long().to(self.device)

                # Forward pass
                logits = self.model(x)  # Only one input: full features
                class_loss = self.criterion(logits.float(), labels)
                    
                total_loss = class_loss
                test_loss += total_loss.item()

                # probs = torch.sigmoid(logits).detach()
                preds_logits.append(logits.detach())
                ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, test_loader, test_loss)  


    def _prepare_data(self, data):
        tabular_data, labels = data
        batch_size = len(tabular_data)

        return tabular_data.to(self.device, dtype=torch.long), labels.to(self.device, dtype=torch.float)  


    def _evaluate(self, ground_truths, preds_logits, loader, loss):
        ground_truths = torch.cat(ground_truths)
        preds_logits = torch.cat(preds_logits)
        loss /= len(loader)

        # Extract predicted labels
        preds_probs = torch.sigmoid(preds_logits)
        preds_labels = preds_probs.argmax(dim=1)

        # Convert tensors to numpy arrays
        y_true = ground_truths.cpu().numpy()
        y_pred = preds_labels.cpu().numpy()

        # Use sklearn to compute accuracy
        accuracy = accuracy_score(y_true, y_pred)

        return loss, accuracy


def objective(trial):
    
    # Define Hyperparameter
    seed = experiment_number = 42
    n_classes = 2
    batch_size = 256

    warmup_epochs = 10
    epochs = 200

    # Suggested hyperparameters (based on the image you provided)
    n_layers        = trial.suggest_int("n_layers", 1, 8)
    layer_size      = trial.suggest_int("layer_size", 1, 512)
    dropout         = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay    = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    learning_rate   = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        
    # Set random seed for reproducibility
    set_random_seed(seed)

    # Load OpenML Adult dataset 
    df = fetch_openml(data_id=1590, as_frame=True)['frame']
    df['label'] = (df['class'] == '>50K').astype(int)
    df = df.drop(columns=['class'])  # drop the original label column

    CAT_COLS = df.select_dtypes(include=['category', 'object']).columns.tolist()
    NUM_COLS = df.select_dtypes(include=['number']).drop(columns=['label']).columns.tolist()

    # Handle missing values in categorical columns
    for col in CAT_COLS:
        if df[col].dtype.name == 'category':
            if 'Missing' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('Missing')
        df[col] = df[col].fillna('Missing')

    # Handle rare categories
    threshold = len(df) * 0.005 
    for col in CAT_COLS:
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < threshold].index
        df[col] = df[col].apply(lambda x: 'Others' if x in rare_categories else x)

    cat_cardinalities = [df[col].nunique() for col in CAT_COLS]

    # Train / Test Split
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

    # === Quantile Transformer ===
    train_num_array = train_data[NUM_COLS].to_numpy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1e-5, train_num_array.shape).astype(train_num_array.dtype)

    quantile_transformer = QuantileTransformer(
        n_quantiles=max(min(len(train_num_array) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9
    ).fit(train_num_array + noise)

    num_train = quantile_transformer.transform(train_data[NUM_COLS])
    num_val   = quantile_transformer.transform(valid_data[NUM_COLS])
    num_test  = quantile_transformer.transform(test_data[NUM_COLS])

    # One-Hot encoding
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    onehot_encoder.fit(train_data[CAT_COLS])
    train_cat_ohe = onehot_encoder.transform(train_data[CAT_COLS])
    valid_cat_ohe = onehot_encoder.transform(valid_data[CAT_COLS])
    test_cat_ohe  = onehot_encoder.transform(test_data[CAT_COLS])

    y_train = train_data['label'].values.astype(np.float32)
    y_valid = valid_data['label'].values.astype(np.float32)
    y_test  = test_data['label'].values.astype(np.float32)

    train_data_encoded = np.concatenate([num_train, train_cat_ohe], axis=1)
    valid_data_encoded = np.concatenate([num_val, valid_cat_ohe], axis=1)
    test_data_encoded  = np.concatenate([num_test,  test_cat_ohe], axis=1)

    # Dataset and Dataloader
    train_dataset = CustomDataset(train_data_encoded, y_train)
    val_dataset   = CustomDataset(valid_data_encoded, y_valid)
    test_dataset  = CustomDataset(test_data_encoded, y_test)
    train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(
        d_in=len(NUM_COLS) + sum(cat_cardinalities),
        d_out=n_classes,
        n_blocks=n_layers,
        d_block=layer_size,
        dropout=dropout,
    ).to(device)

    criterion  = F.cross_entropy

    # Define total steps and warmup steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, epochs, warmup_epochs)
    early_stopping = EarlyStopping(patience=16, delta=0)

    # For output path
    best_valid_acc = 0
    all_epoch_metrics = []

    # Initialize trainer
    trainer = ModelTrainer(model, criterion, optimizer, scheduler, device)

    for epoch in range(epochs):
        
        # Train
        train_loss, train_accuracy = trainer.train(train_loader, epoch)

        # Evaluate
        valid_loss, valid_accuracy = trainer.evaluate(val_loader, epoch)

        # Info
        print(f"Epoch: {epoch + 1}/{epochs}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {valid_accuracy:.4f}")

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
        early_stopping(valid_accuracy, model) 
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    best_epoch = max(all_epoch_metrics, key=lambda x: x["valid_max_acc"])
    train_acc = best_epoch["train_max_acc"]
    valid_acc = best_epoch["valid_max_acc"]

    return train_acc, valid_acc


if __name__ == '__main__':
    
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(objective, n_trials=100)

    pareto_results = []
    for t in study.best_trials:
        result = {
            "train_max_acc": t.values[0],
            "valid_max_acc": t.values[1],
            "params": t.params
        }
        pareto_results.append(result)

    with open("optuna_pareto_trials.json", "w") as f:
        json.dump(pareto_results, f, indent=4)

    print("✅ Saved Pareto trials to optuna_pareto_trials.json")