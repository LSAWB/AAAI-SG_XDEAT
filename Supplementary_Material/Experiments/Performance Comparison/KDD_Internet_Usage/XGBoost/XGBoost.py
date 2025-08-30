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
        
        
def main(hparams, run_id):
    
    # Define Hyperparameter
    experiment_repeat   = 15
    test_accs   = []

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

        # One-Hot encoding
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot_encoder.fit(train_data[CAT_COLS])
        train_cat_ohe = onehot_encoder.transform(train_data[CAT_COLS])
        valid_cat_ohe = onehot_encoder.transform(valid_data[CAT_COLS])
        test_cat_ohe  = onehot_encoder.transform(test_data[CAT_COLS])

        y_train = train_data['label'].values
        y_valid = valid_data['label'].values
        y_test  = test_data['label'].values

        train_data_encoded = np.concatenate([train_data[NUM_COLS].values, train_cat_ohe], axis=1)
        valid_data_encoded = np.concatenate([valid_data[NUM_COLS].values, valid_cat_ohe], axis=1)
        test_data_encoded  = np.concatenate([test_data[NUM_COLS].values,  test_cat_ohe], axis=1)

        # XGBClassifier hyperparameters
        params = {
            "booster": "gbtree",
            "max_depth": hparams['max_depth'],
            "min_child_weight": hparams['min_child_weight'],
            "subsample": hparams['subsample'],
            "learning_rate": hparams['learning_rate'],
            "colsample_bylevel": hparams['colsample_bylevel'],
            "colsample_bytree": hparams['colsample_bytree'],
            "gamma": hparams['gamma'],
            "reg_lambda": hparams['lambda'],  # renamed
            "reg_alpha": hparams['alpha'],    # renamed
            "eval_metric": "error",
            "verbosity": 0,
            "n_estimators": 2000,
            "tree_method": "hist",
            "n_jobs": 8,
            "early_stopping_rounds":50,
            "random_state": experiment_number
        }

        # Initialize and train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            train_data_encoded, y_train,
            eval_set=[(valid_data_encoded, y_valid)],
            verbose=False
        )

        # Predict & evaluate
        y_test_pred = model.predict_proba(test_data_encoded)[:, 1]
        test_acc = accuracy_score(y_test, y_test_pred > 0.5)
        test_accs.append(test_acc)
        
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