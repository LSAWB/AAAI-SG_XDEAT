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
    
    # Define Hyperparameter
    seed = experiment_number = 42
        
    # Set random seed for reproducibility
    set_random_seed(seed)

    # --- Load Dataset ---
    df = fetch_openml(data_id=4538, as_frame=True)['frame']
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Phase'])
    df = df.drop(columns=['Phase'])
    df = df.dropna()

    NUM_COLS = df.drop(columns='label').columns.tolist()
    CAT_COLS = []  # No categorical columns
    
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

    y_train = train_data['label'].values.astype(np.float32)
    y_valid = valid_data['label'].values.astype(np.float32)
    y_test  = test_data['label'].values.astype(np.float32)

    X_train = train_data[NUM_COLS]
    X_valid = valid_data[NUM_COLS]
    X_test  = test_data[NUM_COLS]
    
    # === Optuna hyperparameters for XGBClassifier ===
    params = {
        "booster": "gbtree",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1e2, log=True),
        "reg_lambda": trial.suggest_float("lambda", 1e-8, 1e2, log=True),
        "reg_alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True),
        "eval_metric": "merror", 
        "num_class": 5,
        "verbosity": 0,
        "n_estimators": 2000,
        "tree_method": "hist",
        "n_jobs": 8,
        "early_stopping_rounds":50,
        "random_state": seed,
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # Predict probabilities and take argmax for class prediction
    y_train_pred = np.argmax(model.predict_proba(X_train), axis=1)
    y_valid_pred = np.argmax(model.predict_proba(X_valid), axis=1)

    train_acc = accuracy_score(y_train, y_train_pred)
    valid_acc = accuracy_score(y_valid, y_valid_pred)

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