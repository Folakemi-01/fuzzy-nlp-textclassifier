# src/runner.py

import logging
import time
import pandas as pd
import os
import torch
import numpy as np

from config import *
from src.data_loader import load_and_split_data
from src.preprocessing import preprocess_text
from src.models.baseline_bert import BaselineBertClassifier
from src.models.fuzzy_bert import FuzzyClassifier

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

class ExperimentRunner:
    def __init__(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        logging.info(f"ExperimentRunner initialized. Using device: {DEVICE}")

    def run(self):
        logging.info("Starting experiment pipeline.")
        
        X_train, X_test, y_train, y_test, target_names = load_and_split_data()
        
        logging.info("Preprocessing text data...")
        X_train_processed = [preprocess_text(text) for text in X_train]
        X_test_processed = [preprocess_text(text) for text in X_test]
        
        baseline_model = BaselineBertClassifier(num_classes=len(target_names))

        # --- THIS IS THE MODIFIED LOGIC ---
        # Check if a trained model already exists.
        if os.path.exists(BASELINE_MODEL_PATH):
            # If it exists, load it to save time.
            logging.info(f"Found existing model. Loading from {BASELINE_MODEL_PATH}")
            baseline_model.model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=torch.device(DEVICE)))
        else:
            # If it does not exist, train it and save it.
            logging.info(f"No model found at {BASELINE_MODEL_PATH}. Starting training...")
            start_time = time.time()
            baseline_model.train(X_train_processed, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
            torch.save(baseline_model.model.state_dict(), BASELINE_MODEL_PATH)
            logging.info(f"Baseline model trained and saved in {time.time() - start_time:.2f} seconds.")
        # --- END OF MODIFIED LOGIC ---

        logging.info("--- Generating Predictions ---")
        start_time = time.time()
        baseline_probabilities = baseline_model.predict_proba(X_test_processed)
        baseline_predictions = np.argmax(baseline_probabilities, axis=1)
        logging.info(f"Prediction generation completed in {time.time() - start_time:.2f} seconds.")
        
        logging.info("--- Initializing and Running Fuzzy Classifier ---")
        start_time = time.time()
        fuzzy_model = FuzzyClassifier(target_names=target_names)
        fuzzy_predictions = fuzzy_model.predict(baseline_probabilities)
        logging.info(f"Hybrid model run completed in {time.time() - start_time:.2f} seconds.")
        
        results_df = pd.DataFrame({
            'text': X_test,
            'true_label': y_test,
            'baseline_prediction': baseline_predictions,
            'hybrid_prediction': fuzzy_predictions
        })
        results_df.to_csv(RESULTS_FILE, index=False)
        logging.info(f"Results saved to {RESULTS_FILE}")
        logging.info("Experiment pipeline finished successfully.")
