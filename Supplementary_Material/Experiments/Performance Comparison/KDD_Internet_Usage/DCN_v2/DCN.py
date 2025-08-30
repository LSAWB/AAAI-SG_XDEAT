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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, PowerTransformer
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
import typing as ty


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
    def __init__(self, x_num=None, x_cat=None, y=None):
        if x_num is not None:
            self.x_num = torch.tensor(x_num, dtype=torch.float32)
        else:
            self.x_num = None

        if x_cat is not None:
            self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        else:
            self.x_cat = None

        self.labels = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        output = {}
        if self.x_num is not None:
            output["x_num"] = self.x_num[idx]
        if self.x_cat is not None:
            output["x_cat"] = self.x_cat[idx]
        return output, self.labels[idx]
    

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

# Model 
class CrossLayer(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, x):
        return self.dropout(x0 * self.linear(x)) + x

class DCNv2(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        n_hidden_layers: int,
        n_cross_layers: int,
        hidden_dropout: float,
        cross_dropout: float,
        d_out: int,
        stacked: bool,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
    ) -> None:
        super().__init__()

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.first_linear = nn.Linear(d_in, d)
        self.last_linear = nn.Linear(d if stacked else 2 * d, d_out)

        deep_layers = sum(
            [
                [nn.Linear(d, d), nn.ReLU(True), nn.Dropout(hidden_dropout)]
                for _ in range(n_hidden_layers)
            ],
            [],
        )
        cross_layers = [CrossLayer(d, cross_dropout) for _ in range(n_cross_layers)]

        self.deep_layers = nn.Sequential(*deep_layers)
        self.cross_layers = nn.ModuleList(cross_layers)
        self.stacked = stacked

    def forward(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_linear(x)

        x_cross = x
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x, x_cross)

        if self.stacked:
            return self.last_linear(self.deep_layers(x_cross)).squeeze(1)
        else:
            return self.last_linear(
                torch.cat([x_cross, self.deep_layers(x)], dim=1)
            ).squeeze(1)
            
            
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

            (batch_data, labels) = batch
            
            if "x_num" in batch_data:
                x_num = batch_data["x_num"].to(self.device)
            else:
                x_num = None

            if "x_cat" in batch_data:
                x_cat = batch_data["x_cat"].to(self.device)
            else:
                x_cat = None

            labels = labels.squeeze(dim=-1).long().to(self.device)

            # Forward pass for DCNv2: (x_num, x_cat)
            logits = self.model(x_num, x_cat)
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

                (batch_data, labels) = batch
                
                if "x_num" in batch_data:
                    x_num = batch_data["x_num"].to(self.device)
                else:
                    x_num = None

                if "x_cat" in batch_data:
                    x_cat = batch_data["x_cat"].to(self.device)
                else:
                    x_cat = None
    
                labels = labels.squeeze(dim=-1).long().to(self.device)

                # Forward pass for DCNv2: (x_num, x_cat)
                logits = self.model(x_num, x_cat)
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
                
                (batch_data, labels) = batch
                if "x_num" in batch_data:
                    x_num = batch_data["x_num"].to(self.device)
                else:
                    x_num = None

                if "x_cat" in batch_data:
                    x_cat = batch_data["x_cat"].to(self.device)
                else:
                    x_cat = None
                labels = labels.squeeze(dim=-1).long().to(self.device)

                # Forward pass for DCNv2: (x_num, x_cat)
                logits = self.model(x_num, x_cat)
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


def main(hparams, run_id):
    
    # Define Hyperparameter
    experiment_repeat   = 15
    n_classes           = 2
    batch_size          = 128

    epochs = 200
    warmup_epochs = 10

    test_accs   = []

    # Get current time to name the log directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    current_dir = os.getcwd()
    base_log_dir = os.path.join(current_dir, current_time)
    os.makedirs(base_log_dir, exist_ok=True)

    for experiment_number in range(experiment_repeat):
            
        # Set random seed for reproducibility
        set_random_seed(experiment_number)

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

        # Ordinal encoding
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ordinal_encoder.fit(train_data[CAT_COLS])
        x_train_cat = ordinal_encoder.transform(train_data[CAT_COLS])
        x_valid_cat = ordinal_encoder.transform(valid_data[CAT_COLS])
        x_test_cat  = ordinal_encoder.transform(test_data[CAT_COLS])
        
        x_train_num = train_data[NUM_COLS].values.astype(np.float32)
        x_valid_num = valid_data[NUM_COLS].values.astype(np.float32)
        x_test_num  = test_data[NUM_COLS].values.astype(np.float32)
        
        y_train = train_data['label'].values
        y_valid = valid_data['label'].values
        y_test  = test_data['label'].values
        
        # Dataset and Dataloader
        train_dataset = CustomDataset(None, x_train_cat, y_train)
        val_dataset   = CustomDataset(None, x_valid_cat, y_valid)
        test_dataset  = CustomDataset(None,  x_test_cat,  y_test)
        train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DCNv2(
            d_in=len(NUM_COLS),
            d=hparams['layer_size'],
            n_hidden_layers=hparams['n_hidden_layers'],
            n_cross_layers=hparams['n_cross_layers'],
            hidden_dropout=hparams['hidden_dropout'],
            cross_dropout=hparams['cross_dropout'],
            d_out=n_classes,
            stacked=False,                  # Different Structure but not big difference
            categories=cat_cardinalities,
            d_embedding=hparams['d_embedding'],
        ).to(device)

        learning_rate   = hparams["lr"]
        weight_decay    = hparams["weight_decay"]
        criterion       = F.cross_entropy

        # Define total steps and warmup steps
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_scheduler(optimizer, epochs, warmup_epochs)
        early_stopping = EarlyStopping(patience=16, delta=0)

        # Create a new directory for this experiment
        experiment_log_dir = os.path.join(base_log_dir, str(experiment_number+1))
        os.makedirs(experiment_log_dir, exist_ok=True)

        # For output path
        best_model_path = os.path.join(base_log_dir, f"best_model_seed_{experiment_number}.pth")
        best_valid_acc = 0

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
            early_stopping(valid_accuracy, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # writer.close()

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

if __name__ == '__main__':
    with open("optuna_pareto_trials.json", "r") as f:
        trials = json.load(f)

    results = []
    for run_id, trial in enumerate(trials):
        hparams = trial["params"]
        print(f"\n====== Running trial {run_id} with params: {hparams} ======\n")
        test_mean_acc, test_std_acc = main(hparams, run_id)
        results.append({
            "trial_id": run_id,
            "params": hparams,
            "test_mean_accuarcy": test_mean_acc,
            "test_std_accuarcy": test_std_acc
        })

    with open("optuna_pareto_results.json", "w") as f:
        json.dump(results, f, indent=4)