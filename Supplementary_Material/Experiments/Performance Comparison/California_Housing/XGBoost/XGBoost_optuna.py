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


def objective(trial):
    
    seed = experiment_number = 42

    set_random_seed(seed)

    # Load dataset
    df = fetch_california_housing(as_frame=True).frame
    X = df.drop(columns=["MedHouseVal"])
    y_orig = df["MedHouseVal"].values.astype(np.float32).reshape(-1, 1)

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_orig, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=seed)

    # Normalize targets (based on train)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_val   = (y_val   - y_mean) / y_std
    y_test  = (y_test  - y_mean) / y_std

    NUM_COLS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

    # === Optuna Hyperparameters (from figure) ===
    params = {
        "booster": "gbtree",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1e2, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1e2, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True), 
        "early_stopping_rounds": 50,           
        "verbosity": 0,
        "n_estimators": 2000,
        "tree_method": "hist",
        "n_jobs": 8,
        "random_state": seed
    }

    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train[NUM_COLS], y_train.ravel(),
        eval_set=[(X_val[NUM_COLS], y_val.ravel())],
        verbose=False
    )

    # Predict on normalized scale
    y_train_pred = model.predict(X_train[NUM_COLS])
    y_val_pred   = model.predict(X_val[NUM_COLS])

    # Compute RMSE on normalized targets
    train_rmse_norm = root_mean_squared_error(y_train.ravel(), y_train_pred)
    val_rmse_norm   = root_mean_squared_error(y_val.ravel(), y_val_pred)

    # Convert RMSE back to original scale
    train_rmse = train_rmse_norm * y_std
    valid_rmse = val_rmse_norm * y_std

    return train_rmse, valid_rmse


if __name__ == '__main__':
    
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=100)

    pareto_results = []
    for t in study.best_trials:
        result = {
            "train_rmse": t.values[0],
            "valid_rmse": t.values[1],
            "params": t.params
        }
        pareto_results.append(result)

    with open("optuna_pareto_trials.json", "w") as f:
        json.dump(pareto_results, f, indent=4)

    print("✅ Saved Pareto trials to optuna_pareto_trials.json")