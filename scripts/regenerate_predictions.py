import pandas as pd
import numpy as np
import torch
from tqdm import tqdm 

# --- 1. Importing project's functions and classes ---

from src.data_loader import load_and_split_data, get_class_names
from src.models.baseline_bert import BaselineBertClassifier
from src.models.fuzzy_bert import FuzzyClassifier
from config import BASELINE_MODEL_PATH, DEVICE

# --- 2. Use the correct model loading functions ---
def load_baseline_model():
    """Instantiates and loads the trained BaselineBertClassifier."""
    print(f"Loading Baseline BERT model weights from '{BASELINE_MODEL_PATH}'...")
    target_names = get_class_names()
    model = BaselineBertClassifier(num_classes=len(target_names))
    model.model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=torch.device(DEVICE)))
    model.model.eval()
    print("Baseline model loaded successfully.")
    return model

def load_fuzzy_model():
    """Instantiates the FuzzyClassifier."""
    print("Instantiating Fuzzy-Hybrid model...")
    target_names = get_class_names()
    model = FuzzyClassifier(target_names=target_names)
    print("Fuzzy-Hybrid model instantiated successfully.")
    return model

# --- MAIN SCRIPT ---
print("--- Regenerating predictions for the main test set ---")

# 1. Load the data, we only need the test set portion
X_train, X_test, y_train, y_test, class_names = load_and_split_data()

# 2. Load the trained models
baseline_model = load_baseline_model()
fuzzy_model = load_fuzzy_model()

# 3. Generate predictions for the entire test set
baseline_preds = []
fuzzy_preds = []

print(f"\nGenerating predictions for {len(X_test)} test documents...")
# Using tqdm will show a progress bar, which is helpful for long processes
for text in tqdm(X_test):
    # Get probability vector from the baseline model
    prob_vector = baseline_model.predict_proba([text])
    
    # Get the baseline's crisp prediction (index)
    baseline_pred_index = np.argmax(prob_vector, axis=1)[0]
    
    # Get the fuzzy model's prediction (index)
    fuzzy_pred_index = fuzzy_model.predict(prob_vector)[0]
    
    baseline_preds.append(baseline_pred_index)
    fuzzy_preds.append(fuzzy_pred_index)

# 4. Creating the final DataFrame
# We use the numerical labels (0-19) for the McNemar test
df_results = pd.DataFrame({
    'true_label': y_test,
    'baseline_prediction': baseline_preds,
    'fuzzy_prediction': fuzzy_preds
})

# To make it human-readable, we can add the text labels as well
df_results['true_label_name'] = [class_names[i] for i in y_test]
df_results['baseline_prediction_name'] = [class_names[i] for i in baseline_preds]
df_results['fuzzy_prediction_name'] = [class_names[i] for i in fuzzy_preds]


# 5. Save the results to the required file
OUTPUT_FILE = 'mcnemar_input.csv'
df_results.to_csv(OUTPUT_FILE, index=False)

print(f"\nSuccessfully regenerated predictions and saved to '{OUTPUT_FILE}'")
print("\nFirst 5 rows of the new file:")
print(df_results.head())
print("\n--- Regeneration Complete ---")