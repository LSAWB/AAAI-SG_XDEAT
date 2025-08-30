import os
import torch

# Paths
BASE_LOG_DIR = os.getcwd()

# Experiment Settings ( Seed )
EXPERIMENT_REPEAT = 15

# Training Parameters
EPOCHS          = 200
WARMUP_EPOCHS   = 10

BATCH_SIZE      = 256

NUM_LABELS      = 1

# Early Stopping
EARLY_STOPPING_PATIENCE = 16
EARLY_STOPPING_DELTA = 0

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
