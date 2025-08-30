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
from sklearn.metrics import root_mean_squared_error

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
import xgboost as xgb

# Other
import delu
import typing as ty


# Seed
def set_random_seed(seed):
    # Set Python random seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def main(hparams, run_id):
    experiment_repeat = 15
    test_rmses = []

    for experiment_number in range(experiment_repeat):
        set_random_seed(experiment_number)

        # Load dataset
        df = fetch_california_housing(as_frame=True).frame
        X = df.drop(columns=["MedHouseVal"])
        y_orig = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y_orig, test_size=0.2, random_state=experiment_number)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=experiment_number)

        # Normalize targets
        y_mean = y_train.mean()
        y_std  = y_train.std()
        y_train_norm = (y_train - y_mean) / y_std
        y_val_norm   = (y_val   - y_mean) / y_std
        y_test_norm  = (y_test  - y_mean) / y_std

        NUM_COLS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

        # XGBRegressor with hyperparameters
        model = xgb.XGBRegressor(
            booster="gbtree",
            max_depth=hparams['max_depth'],
            min_child_weight=hparams['min_child_weight'],
            subsample=hparams['subsample'],
            learning_rate=hparams['learning_rate'],
            colsample_bylevel=hparams['colsample_bylevel'],
            colsample_bytree=hparams['colsample_bytree'],
            gamma=hparams['gamma'],
            reg_lambda=hparams['lambda'],
            reg_alpha=hparams['alpha'],
            verbosity=0,
            n_estimators=2000,
            tree_method="hist",
            n_jobs=8,
            random_state=experiment_number,
            early_stopping_rounds=50,
        )

        model.fit(
            X_train[NUM_COLS], y_train_norm.ravel(),
            eval_set=[(X_val[NUM_COLS], y_val_norm.ravel())],
            verbose=False
        )

        # Predict and evaluate
        y_test_pred_norm = model.predict(X_test[NUM_COLS])
        test_rmse_norm = root_mean_squared_error(y_test_norm.ravel(), y_test_pred_norm)

        # Denormalize RMSE
        test_rmse = test_rmse_norm * y_std
        test_rmses.append(test_rmse)

    mean_rmse = np.mean(test_rmses)
    std_rmse = np.std(test_rmses)
    print(f"📊 Denormalized RMSE → Mean: {mean_rmse:.4f}, Std Dev: {std_rmse:.4f}")

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
            "test_mean_rmse": test_mean_rmse,
            "test_std_rmse": test_std_rmse
        })

    with open("optuna_pareto_results.json", "w") as f:
        json.dump(results, f, indent=4)