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

    # One-Hot encoding
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    onehot_encoder.fit(train_data[CAT_COLS])
    train_cat_ohe = onehot_encoder.transform(train_data[CAT_COLS])
    valid_cat_ohe = onehot_encoder.transform(valid_data[CAT_COLS])
    test_cat_ohe  = onehot_encoder.transform(test_data[CAT_COLS])

    y_train = train_data['label'].values
    y_valid = valid_data['label'].values
    y_test  = test_data['label'].values

    X_train = np.concatenate([train_data[NUM_COLS].values, train_cat_ohe], axis=1)
    X_valid = np.concatenate([valid_data[NUM_COLS].values, valid_cat_ohe], axis=1)
    X_test  = np.concatenate([test_data[NUM_COLS].values,  test_cat_ohe], axis=1)
    
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