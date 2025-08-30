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

    # Get current time to name the log directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    current_dir = os.getcwd()
    base_log_dir = os.path.join(current_dir, current_time)
    os.makedirs(base_log_dir, exist_ok=True)

    for experiment_number in range(experiment_repeat):
            
        # Set random seed for reproducibility
        set_random_seed(experiment_number)

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

        train_data_encoded = np.concatenate([train_data[NUM_COLS], train_cat_ohe], axis=1)
        valid_data_encoded = np.concatenate([valid_data[NUM_COLS], valid_cat_ohe], axis=1)
        test_data_encoded  = np.concatenate([test_data[NUM_COLS],  test_cat_ohe], axis=1)

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
            "reg_lambda": hparams['lambda'], 
            "reg_alpha": hparams['alpha'], 
            "objective": "binary:logistic",
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