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
from sklearn.datasets import fetch_openml, fetch_california_housing

# PyTorch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn.init as nn_init
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
        # Handle x_num
        if x_num is not None:
            if isinstance(x_num, torch.Tensor):
                self.x_num = x_num
            elif isinstance(x_num, (np.ndarray, pd.DataFrame)):
                self.x_num = torch.tensor(x_num, dtype=torch.float32)
            else:
                raise TypeError("x_num must be a torch.Tensor, np.ndarray, or pd.DataFrame")
        else:
            self.x_num = None

        # Handle x_cat
        if x_cat is not None:
            if isinstance(x_cat, torch.Tensor):
                self.x_cat = x_cat
            elif isinstance(x_cat, (np.ndarray, pd.DataFrame)):
                self.x_cat = torch.tensor(x_cat, dtype=torch.long)
            else:
                raise TypeError("x_cat must be a torch.Tensor, np.ndarray, or pd.DataFrame")
        else:
            self.x_cat = None

        # Handle y
        if y is not None:
            if isinstance(y, torch.Tensor):
                self.labels = y
            elif isinstance(y, (np.ndarray, pd.Series)):
                self.labels = torch.tensor(y, dtype=torch.float32)
            else:
                raise TypeError("y must be a torch.Tensor, np.ndarray, or pd.Series")
        else:
            raise ValueError("y cannot be None")

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
            
# Model 
class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        n_latent_tokens: int,
        d_token: int,
    ) -> None:
        super().__init__()
        assert n_latent_tokens == 0
        self.n_latent_tokens = n_latent_tokens
        if d_numerical:
            self.weight = nn.Parameter(Tensor(d_numerical + n_latent_tokens, d_token))
            # The initialization is inspired by nn.Linear
            nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = None
            assert categories is not None
        if categories is None:
            self.category_offsets = None
            self.category_embeddings = None
        else:
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

    @property
    def n_tokens(self) -> int:
        return (0 if self.weight is None else len(self.weight)) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        if x_num is None:
            return self.category_embeddings(x_cat + self.category_offsets[None])  # type: ignore[code]
        x_num = torch.cat(
            [
                torch.ones(len(x_num), self.n_latent_tokens, device=x_num.device),
                x_num,
            ],
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]  # type: ignore[code]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],  # type: ignore[code]
                dim=1,
            )
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class AutoInt(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        d_out: int,
    ) -> None:
        assert not prenormalization
        assert activation == 'relu'
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, 0, d_token)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear': nn.Linear(d_token, d_token, bias=False),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = F.relu
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token * n_tokens, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)

        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = layer['linear'](x)
            x = self._end_residual(x, x_residual, layer, 0)
            x = self.activation(x)

        x = x.flatten(1, 2)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


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

            (batch_data, labels) = batch
            
            if "x_num" in batch_data:
                x_num = batch_data["x_num"].to(self.device)
            else:
                x_num = None

            if "x_cat" in batch_data:
                x_cat = batch_data["x_cat"].to(self.device)
            else:
                x_cat = None

            labels = labels.squeeze(dim=-1).float().to(self.device)

            # Forward pass for DCNv2: (x_num, x_cat)
            logits = self.model(x_num, x_cat).squeeze()
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

                labels = labels.squeeze(dim=-1).float().to(self.device)

                # Forward pass for DCNv2: (x_num, x_cat)
                logits = self.model(x_num, x_cat).squeeze()
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

                labels = labels.squeeze(dim=-1).float().to(self.device)

                # Forward pass for DCNv2: (x_num, x_cat)
                logits = self.model(x_num, x_cat).squeeze()
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


    def _evaluate(self, ground_truths, preds, loader, loss):
        ground_truths = torch.cat(ground_truths)
        preds = torch.cat(preds)

        loss /= len(loader)
        rmse = torch.sqrt(F.mse_loss(preds, ground_truths)) * self.std.item()

        return loss, rmse.cpu()
    

def main(hparams, run_id):
    
    # Define Hyperparameter
    experiment_repeat   = 15
    n_classes           = 1
    batch_size          = 256

    epochs = 200
    warmup_epochs = 10

    test_rmses   = []

    # Get current time to name the log directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    current_dir = os.getcwd()
    base_log_dir = os.path.join(current_dir, current_time)
    os.makedirs(base_log_dir, exist_ok=True)

    for experiment_number in range(experiment_repeat):
            
        # Set random seed
        set_random_seed(experiment_number)

        # 取得資料
        df = fetch_california_housing(as_frame=True).frame
        X = df.drop(columns=["MedHouseVal"])
        y_orig = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

        # Train / Val / Test split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y_orig, test_size=0.2, random_state=experiment_number)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=experiment_number)

        # 對 y 做標準化 (僅用 train 的 mean 和 std)
        y_mean = y_train.mean()
        y_std  = y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_val   = (y_val - y_mean) / y_std
        y_test  = (y_test  - y_mean) / y_std

        # Columns to normalize
        NUM_COLS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                    'Population', 'AveOccup', 'Latitude', 'Longitude']

        # === Quantile Transformer ===
        train_num_array = X_train[NUM_COLS].to_numpy()
        rng = np.random.default_rng(experiment_number)
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

        # Dataset and Dataloader
        train_dataset = CustomDataset(num_train_tensor, None, y_train_tensor)
        val_dataset   = CustomDataset(num_val_tensor, None, y_val_tensor)
        test_dataset  = CustomDataset(num_test_tensor, None,  y_test_tensor)
        train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoInt(
            d_numerical=len(NUM_COLS),
            categories=None,
            n_layers=hparams['n_layers'],
            d_token=hparams['d_token'],
            n_heads=2,
            attention_dropout=hparams['attention_dropout'],
            residual_dropout=hparams['residual_dropout'],
            activation="relu",
            prenormalization=False,
            initialization="xavier",
            kv_compression=None,
            kv_compression_sharing=None,
            d_out=n_classes,
        ).to(device)

        learning_rate   = hparams["lr"]
        weight_decay    = hparams["weight_decay"]
        criterion       = torch.nn.MSELoss()

        # Define total steps and warmup steps
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_scheduler(optimizer, epochs, warmup_epochs)
        early_stopping = EarlyStopping(patience=16, delta=0)

        # Create a new directory for this experiment
        experiment_log_dir = os.path.join(base_log_dir, str(experiment_number+1))
        os.makedirs(experiment_log_dir, exist_ok=True)

        # For output path
        best_model_path = os.path.join(base_log_dir, f"best_model_seed_{experiment_number}.pth")
        
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
            print(f"Training RMSE: {train_rmse:.4f}")
            print(f"Validation RMSE: {valid_rmse:.4f}")

            # Save model
            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                save_file = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }
                torch.save(save_file, best_model_path)

            # Early stopping
            early_stopping(valid_rmse, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # writer.close()

        # After training, load the best model and test
        checkpoint = torch.load(best_model_path)

        # Restore model, optimizer, scheduler, and epoch
        model.load_state_dict(checkpoint["model"])
        
        test_loss, test_rmse = trainer.test(test_loader, epoch)

        print(f"Test RMSE: {test_rmse:.4f}")

        # Store the best AUROC for this experiment
        test_rmses.append(test_rmse)

        # Delete the best model file after loading
        if os.path.exists(best_model_path):  
            os.remove(best_model_path)  # Deletes the file
            print(f"Deleted best model checkpoint: {best_model_path}")
        else:
            print(f"Checkpoint file not found: {best_model_path}")

    # After all experiments, calculate the mean and std of AUROCs
    mean_rmse = np.mean(test_rmses)
    std_rmse = np.std(test_rmses)
    print(f"Mean RMSE: {mean_rmse:.4f}, Std Dev RMSE: {std_rmse:.4f}")

    return mean_rmse, std_rmse

if __name__ == '__main__':
    with open("optuna_pareto_trials.json", "r") as f:
        trials = json.load(f)

    results = []
    for run_id, trial in enumerate(trials):
        hparams = trial["params"]
        print(f"\n====== Running trial {run_id} with params: {hparams} ======\n")
        test_mean_rmse, test_std_rmse = main(hparams, run_id)
        results.append({
            "trial_id": run_id,
            "params": hparams,
            "test_mean_rmse": float(test_mean_rmse),
            "test_std_rmse": float(test_std_rmse)
        })

    with open("optuna_pareto_results.json", "w") as f:
        json.dump(results, f, indent=4)