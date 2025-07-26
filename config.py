# config.py

import torch

# --- DEVELOPMENT ---
# Set to True to run on a small subset of data for fast debugging
DEBUG_MODE = True
DEBUG_DATA_SIZE = 100 # Number of samples to use in debug mode

# --- DIRECTORIES ---
DATA_DIR = "data/"
RESULTS_DIR = "results/"
MODEL_DIR = "models/"
LOGS_DIR = "logs/"

# --- FILE PATHS ---
RESULTS_FILE = RESULTS_DIR + "experiment_results.csv"
METRICS_FILE = RESULTS_DIR + "performance_metrics.csv"
BASELINE_CM_FILE = RESULTS_DIR + "baseline_confusion_matrix.png"
HYBRID_CM_FILE = RESULTS_DIR + "hybrid_confusion_matrix.png"
BASELINE_MODEL_PATH = MODEL_DIR + "baseline_bert_classifier.pth"
LOG_FILE = LOGS_DIR + "experiment.log"

# --- DATA PARAMETERS ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- MODEL & TRAINING PARAMETERS ---
MODEL_NAME = "bert-base-uncased"

# This is the corrected logic for device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

NUM_EPOCHS = 1
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 2e-5