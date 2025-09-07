# health_check.py

import sys
import torch
import pandas as pd
from mlxtend.evaluate import mcnemar
from transformers import BertTokenizer, BertModel

def run_check():
    """
    Performs a smoke test on the project to ensure all modules
    and key components can be loaded without errors.
    """
    print("--- Starting Project Health Check ---")

    try:
        # Check 1: Importing your custom modules (with corrected config path)
        print("Checking custom module imports...")
        from config import DEVICE, BASELINE_MODEL_PATH # CORRECTED IMPORT
        from src.data_loader import load_and_split_data, get_class_names
        from src.models.baseline_bert import BaselineBertClassifier
        from src.models.fuzzy_bert import FuzzyClassifier
        print("  [OK] All custom modules imported successfully.")

        # Check 2: Loading data (in debug mode)
        print("\nChecking data loading...")
        X_train, X_test, y_train, y_test, target_names = load_and_split_data()
        print(f"  [OK] Data loaded successfully. Test set size: {len(X_test)} documents.")

        # Check 3: Instantiating models
        print("\nChecking model instantiation...")
        baseline_model = BaselineBertClassifier(num_classes=len(target_names))
        fuzzy_model = FuzzyClassifier(target_names=target_names)
        print("  [OK] Both Baseline and Fuzzy models instantiated successfully.")
        
        # Check 4: Check for trained model file
        print("\nChecking for saved model file...")
        import os
        if os.path.exists(BASELINE_MODEL_PATH):
            print(f"  [OK] Trained model file found at: {BASELINE_MODEL_PATH}")
        else:
            print(f"  [WARNING] Trained model file not found at: {BASELINE_MODEL_PATH}")
            print("           This is not an error if you intend to train from scratch.")

    except ImportError as e:
        print(f"\n[ERROR] Failed to import a module: {e}")
        print("        Ensure your virtual environment is active and all libraries in requirements.txt are installed.")
        return False
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        print("        There may be an issue in your code or file paths.")
        return False

    print("\n-------------------------------------")
    print("✅ Health Check Passed! The project is runnable.")
    print("-------------------------------------")
    return True

if __name__ == "__main__":
    run_check()