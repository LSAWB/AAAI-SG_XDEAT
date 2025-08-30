import numpy as np
import torch
import torch.nn as nn
import pickle
import os

# Configuration
N_TRAIN, N_VAL, N_TEST = 64000, 16000, 20000
INPUT_DIM = 100
USEFUL_DIM_RATIOS = np.round(np.arange(0.1, 1.01, 0.1), 1)  # from 0.1 to 1.0 inclusive
SEEDS = range(10)

# Remember to modify it
SAVE_DIR = "/home/willie/LAB/2025/AAAI/NEW_ASSA/Dataset"

os.makedirs(SAVE_DIR, exist_ok=True)

# MLP Definition
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                a = np.sqrt(1.0 / layer.in_features)
                nn.init.uniform_(layer.bias, -a, a)

    def forward(self, x):
        return self.model(x)

# Generate datasets for each ratio and seed
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate full input once per seed
    X = np.random.randn(N_TRAIN + N_VAL + N_TEST, INPUT_DIM)

    for ratio in USEFUL_DIM_RATIOS:
        useful_dim = int(INPUT_DIM * ratio)
        X_useful = X[:, :useful_dim]

        mlp = MLP(input_dim=useful_dim)
        with torch.no_grad():
            y = mlp(torch.tensor(X_useful, dtype=torch.float32)).squeeze().numpy()

        # Normalize target
        y = (y - y.mean()) / y.std()

        # Split
        X_train, y_train = X[:N_TRAIN], y[:N_TRAIN]
        X_val, y_val = X[N_TRAIN:N_TRAIN + N_VAL], y[N_TRAIN:N_TRAIN + N_VAL]
        X_test, y_test = X[N_TRAIN + N_VAL:], y[N_TRAIN + N_VAL:]
        
        # Save
        file_name = f'dataset_ratio{ratio:.1f}_seed{seed}.pkl'
        with open(os.path.join(SAVE_DIR, file_name), 'wb') as f:
            pickle.dump({
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'ratio': ratio,
                'seed': seed
            }, f)

print("✅ All datasets generated and saved.")
