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

    # One-Hot encoding
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    onehot_encoder.fit(train_data[CAT_COLS])
    train_cat_ohe = onehot_encoder.transform(train_data[CAT_COLS])
    valid_cat_ohe = onehot_encoder.transform(valid_data[CAT_COLS])
    test_cat_ohe  = onehot_encoder.transform(test_data[CAT_COLS])

    y_train = train_data['label'].values.astype(np.float32)
    y_valid = valid_data['label'].values.astype(np.float32)
    y_test  = test_data['label'].values.astype(np.float32)

    X_train = np.concatenate([train_data[NUM_COLS], train_cat_ohe], axis=1)
    X_valid = np.concatenate([valid_data[NUM_COLS], valid_cat_ohe], axis=1)
    X_test  = np.concatenate([test_data[NUM_COLS],  test_cat_ohe], axis=1)
    
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
        "objective": "binary:logistic",
        "eval_metric": "error",
        "verbosity": 0,
        "n_estimators": 2000,
        "tree_method": "hist",
        "n_jobs": 8,
        "early_stopping_rounds":50,
        "random_state": seed,
    }

    # Initialize XGBClassifier
    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # Predict and evaluate
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_valid_pred = model.predict_proba(X_valid)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred > 0.5)
    valid_acc = accuracy_score(y_valid, y_valid_pred > 0.5)

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