# ========== Standard Library ==========
import os
import math
import random
from datetime import datetime

# ========== Third-Party Libraries ==========
import numpy as np
from tqdm import tqdm

# ========== PyTorch ==========
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pickle
import json
import math
from math import sqrt
from typing import List, Tuple
from torch import Tensor


class _CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_embedding))  # learnable vector of shape [d_embedding]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)  # Xavier-style initialization

    def forward(self, batch_dims: Tuple[int]) -> Tensor:
        if not batch_dims:
            raise ValueError('The input must be non-empty')

        return self.weight.expand(*batch_dims, 1, -1)


class _ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError(
                'For the ReGLU activation, the last input dimension'
                f' must be a multiple of 2, however: {x.shape[-1]=}'
            )
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)
    
    
class ContinuousEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int):
        super().__init__()
        self.n_features     = n_features
        self.d_embedding    = d_embedding

        # Create weight and bias as trainable parameters
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias   = nn.Parameter(torch.empty(n_features, d_embedding))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.d_embedding ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Input should be [B, {self.n_features}], but got {x.shape}")

        x = x.unsqueeze(-1)  
        out = x * self.weight + self.bias  
        return out


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale      = scale
        self.dropout    = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):   # 加入參數
        B, L, H, E  = queries.shape
        _, S, _, D  = values.shape
        scale       = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)         # [B, H, L, S]
        A = torch.softmax(scale * scores, dim=-1)                       # attention scores
        A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, attention_dropout=0.1):
        super(AttentionLayer, self).__init__()
        d_keys          = d_keys or (d_model // n_heads)
        d_values        = d_values or (d_model // n_heads)
        
        self.n_heads    = n_heads
        
        self.inner_attention    = FullAttention(scale=None, attention_dropout=attention_dropout)
        self.query_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection     = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection   = nn.Linear(d_model, d_values * n_heads)
        self.out_projection     = nn.Linear(d_values * n_heads, d_model)
        

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H       = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)


class ASSAAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(ASSAAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

        # Learnable parameters for weight fusion
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))

    def get_alpha_values(self):
        with torch.no_grad():
            w1 = torch.exp(self.a1)
            w2 = torch.exp(self.a2)
            alpha1 = w1 / (w1 + w2)
            alpha2 = w2 / (w1 + w2)
        return alpha1.item(), alpha2.item()
    
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)

        # QK^T / sqrt(d)
        qk_scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale

        # === SSA branch: Squared ReLU(QK^T / sqrt(d) + B) ===
        ssa_scores = F.relu(qk_scores) ** 2
        ssa_scores = self.dropout(ssa_scores)
        ssa_output = torch.einsum("bhls,bshd->blhd", ssa_scores, values)
        
        # === DSA branch: Softmax(QK^T / sqrt(d) + B) ===
        dsa_scores = torch.softmax(qk_scores, dim=-1)
        dsa_scores = self.dropout(dsa_scores)
        dsa_output = torch.einsum("bhls,bshd->blhd", dsa_scores, values)

        # === Adaptive fusion ===
        w1 = torch.exp(self.a1)
        w2 = torch.exp(self.a2)
        alpha1 = w1 / (w1 + w2)
        alpha2 = w2 / (w1 + w2)

        output = alpha1 * ssa_output + alpha2 * dsa_output

        return output.contiguous()


class ASSAAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, attention_dropout=0.1):
        super(ASSAAttentionLayer, self).__init__()
        d_keys      = d_keys or (d_model // n_heads)
        d_values    = d_values or (d_model // n_heads)

        self.n_heads = n_heads

        self.inner_attention    = ASSAAttention(scale=None, attention_dropout=attention_dropout)
        self.query_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection     = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection   = nn.Linear(d_model, d_values * n_heads)
        self.out_projection     = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)
    
class NORMALAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, attention_dropout=0.1,
             residual_dropout=0.1, ffn_dropout=0.1, apply_feature_attn_norm=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.apply_feature_attn_norm = apply_feature_attn_norm
        
        # Dropouts
        self.att_residual_dropout   = nn.Dropout(residual_dropout)
        self.ffn_residual_dropout   = nn.Dropout(residual_dropout)
        self.ffn_dropout            = nn.Dropout(ffn_dropout)
        
        # Feature-wise Module
        self.feature_attention  = AttentionLayer(d_model, n_heads, attention_dropout=attention_dropout)
        self.norm_feature_attn  = nn.LayerNorm(d_model)
        self.norm_feature_ffn   = nn.LayerNorm(d_model)

        self.feature_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff * 2),
            _ReGLU(),
            self.ffn_dropout,
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        
        x_feature = self.norm_feature_attn(x) if self.apply_feature_attn_norm else x
        x = x + self.att_residual_dropout(
            self.feature_attention(x_feature, x_feature, x_feature)
        )

        x_ffn_input = self.norm_feature_ffn(x)
        x = x + self.ffn_residual_dropout(self.feature_ffn(x_ffn_input))
        
        return x    
        
         
class NORMALAttentionLayer(nn.Module):
    def __init__(self, 
                 num_layers: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int = None,
                 attention_dropout: float = 0.1,
                 residual_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,):
        super().__init__()

        self.num_layers         = num_layers

        self.layers = nn.ModuleList([
            NORMALAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
                ffn_dropout=ffn_dropout,
                apply_feature_attn_norm=(i != 0)  # 第一層不用
            )
            for i in range(num_layers)
        ])
        
        self.cls_embedder = _CLSEmbedding(d_embedding=d_model)
        
    def forward(self, x):
        batch, _, d_model = x.shape
        
        cls_token = self.cls_embedder((batch,))
        x = torch.cat([cls_token, x], dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)

        cls_token = x[:, 0] 
        
        return cls_token


class DualTransformer(nn.Module):
    def __init__(self, 
                 num_features: int,
                 dim_model: int,
                 num_heads: int,
                 dim_ff: int,
                 num_layers_cross: int,
                 num_labels: int,
                 att_dropout: float,
                 res_dropout: float,
                 ffn_dropout: float):
        
        super().__init__()

        # === Embedding ===
        self.num_ori_embedding_layer    = ContinuousEmbeddings(n_features=num_features, d_embedding=dim_model)
        self.d_model = dim_model
        
        # === Two-Stage Attention Encoder ===
        self.NORWAL = NORMALAttentionLayer(
            num_layers=num_layers_cross,
            d_model=dim_model,
            n_heads=num_heads,
            d_ff=dim_ff,
            attention_dropout=att_dropout,
            residual_dropout=res_dropout,
            ffn_dropout=ffn_dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.ReLU(),
            nn.Linear(dim_model, num_labels)
        )
                
    def make_parameter_groups(self):
        def get_parameters(module):
            return [] if module is None else list(module.parameters())

        #  no weight decay group
        zero_wd_params = set()

        # === 1. Embedding-related modules
        zero_wd_params.update(get_parameters(self.NORWAL.cls_embedder))

        # === 2. All LayerNorm parameters
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                zero_wd_params.update(module.parameters())
                
        # === 3. All bias parameters
        for name, param in self.named_parameters():
            if name.endswith('.bias'):
                zero_wd_params.add(param)

        # === 4. Create parameter groups
        decay_group = {
            'params': [p for p in self.parameters() if p not in zero_wd_params],
        }

        no_decay_group = {
            'params': list(zero_wd_params),
            'weight_decay': 0.0
        }

        return [decay_group, no_decay_group]


    def forward(self, data):
        
        ori_embedding = self.num_ori_embedding_layer(data)
        embedding = self.NORWAL(ori_embedding)
        logits = self.classifier(embedding)
        
        return logits.squeeze(1)
    

# Seed
def set_random_seed(seed):
    # Set Python random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    def __init__(self, model, criterion, optimizer, device, y_std):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.std    = y_std

    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0.0
        ground_truths, preds_logits = [], []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            self.optimizer.zero_grad()

            x_num, labels = batch
            x_num = x_num.to(self.device)
            labels = labels.squeeze(dim=-1).float().to(self.device)

            logits = self.model(x_num).squeeze()
            class_loss = self.criterion(logits.float(), labels)

            total_loss = class_loss
            total_loss.backward()
            train_loss += total_loss.item()

            self.optimizer.step()

            preds_logits.append(logits.detach())
            ground_truths.append(labels.detach())

        return self._evaluate(ground_truths, preds_logits, train_loader, train_loss)

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        valid_loss = 0.0
        ground_truths, preds_logits = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating Epoch {epoch+1}", leave=False):

                x_num, labels = batch
                x_num = x_num.to(self.device)
                labels = labels.squeeze(dim=-1).float().to(self.device)

                logits = self.model(x_num).squeeze()
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
                
                x_num, labels = batch
                x_num = x_num.to(self.device)
                labels = labels.squeeze(dim=-1).float().to(self.device)

                logits = self.model(x_num).squeeze()
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
    
    
class SyntheticDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

def main():
    
    # Remember to modify them
    DATASET_DIR = '/mnt/nas/users/willie/AIII/Dataset'
    SAVE_DIR = '/home/willie/LAB/2025/AAAI/NEW_ASSA/WITHOUT_ASSA'

    # Ensure save dir exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # List all .pkl files
    dataset_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.pkl')]
    dataset_files.sort()

    # Extract all seeds used
    seeds = sorted({int(f.split("seed")[1].split(".")[0]) for f in dataset_files})

    for seed in seeds:
        results = []
        print(f"\n🔁 Running for seed: {seed}")
        
        # Filter files for this seed only
        seed_files = sorted([f for f in dataset_files if f"seed{seed}" in f])

        for filename in seed_files:
            filepath = os.path.join(DATASET_DIR, filename)
            with open(filepath, 'rb') as f:
                dataset = pickle.load(f)

            X_train, y_train = dataset['X_train'], dataset['y_train']
            X_val, y_val = dataset['X_val'], dataset['y_val']
            X_test, y_test = dataset['X_test'], dataset['y_test']

            y_std = torch.tensor(y_train).float().std()

            train_set = SyntheticDataset(X_train, y_train)
            val_set = SyntheticDataset(X_val, y_val)
            test_set = SyntheticDataset(X_test, y_test)

            train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

            model = DualTransformer(
                num_features=X_train.shape[1],
                dim_model=192,
                num_heads=8,
                dim_ff=256,
                num_layers_cross=3,
                num_labels=1,
                att_dropout=0.2,
                res_dropout=0.0,
                ffn_dropout=0.1,
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            trainer = ModelTrainer(model, criterion, optimizer, device="cuda" if torch.cuda.is_available() else "cpu", y_std=y_std)
            early_stopping = EarlyStopping(patience=10)

            for epoch in range(200):
                train_loss, train_rmse = trainer.train(train_loader, epoch)
                val_loss, val_rmse = trainer.evaluate(val_loader, epoch)

                print(f"[{filename}] Epoch {epoch+1} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
                early_stopping(val_rmse, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            model.load_state_dict(early_stopping.best_model_state)
            test_loss, test_rmse = trainer.test(test_loader, epoch)
            print(f"[{filename}] Test RMSE: {test_rmse:.4f}")

            results.append({
                "filename": filename,
                "ratio": dataset["ratio"],
                "seed": dataset["seed"],
                "test_rmse": round(test_rmse.item(), 6),
                "alphas": [
                    {
                        "layer": i,
                        "alpha1": round(alpha1, 6),
                        "alpha2": round(alpha2, 6)
                    }
                    for i, layer in enumerate(model.NORWAL.layers)
                    if hasattr(layer.feature_attention.inner_attention, "get_alpha_values")
                    for alpha1, alpha2 in [layer.feature_attention.inner_attention.get_alpha_values()]
                ]
            })

        # Save JSON for this seed
        save_path = os.path.join(SAVE_DIR, f"WITHOUT_ASSA_seed{seed}.json")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"✅ Results saved for seed {seed} → {save_path}")

if __name__ == "__main__":
    main()
