import pandas as pd
import torch
from sklearn.metrics import classification_report
import numpy as np # Adding for dummy model, you might not need it

#Step 1: Importing project-specific modules 

from src.data_loader import load_full_unsplit_data, get_class_names
from src.models.baseline_bert import BaselineBertClassifier
from src.models.fuzzy_bert import FuzzyClassifier
from config import BASELINE_MODEL_PATH, DEVICE 

# Step 2: Using the CORRECT model loading functions 

def load_baseline_model():
    """
    Instantiates a BaselineBertClassifier and loads its trained weights from disk.
    """
    print(f"Loading Baseline BERT model weights from '{BASELINE_MODEL_PATH}'...")
    try:
        # 1. Getting the number of classes and instantiating the model structure
        target_names = get_class_names()
        model = BaselineBertClassifier(num_classes=len(target_names))
        
        # 2. Loading the saved state dictionary into the model structure
        # Using map_location ensures it works on CPU if a GPU is not available.
        model.model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=torch.device(DEVICE)))
        
        # 3. IMPORTANT: Set the model to evaluation mode
        model.model.eval()
        
        print("Baseline model loaded successfully.")
        return model
        
    except FileNotFoundError:
        print(f"ERROR: Saved baseline model not found at '{BASELINE_MODEL_PATH}'.")
        print("Make sure you have run the main experiment to train and save the model.")
        exit()

def load_fuzzy_model():
    """
    Instantiates a new FuzzyClassifier with its predefined rules.
    This model is not loaded from a file as it has no trained parameters.
    """
    print("Instantiating Fuzzy-Hybrid model...")
    target_names = get_class_names()
    model = FuzzyClassifier(target_names=target_names)
    print("Fuzzy-Hybrid model instantiated successfully.")
    return model

# CONFIGURATION 
CONSENSUS_FILE = 'human_consensus_labels.csv'
RESULTS_FILE = 'focused_analysis_results.csv'

# MAIN SCRIPT 
print("--- Starting Focused Analysis ---")
df_consensus = pd.read_csv(CONSENSUS_FILE)
full_dataset_df = load_full_unsplit_data()
baseline_model = load_baseline_model()
fuzzy_model = load_fuzzy_model()

# Convert 1-based Document_IDs from survey to 0-based DataFrame indices
subset_doc_indices = df_consensus['Document_ID'].astype(int) - 1

#Generating Predictions 
print(f"\nGenerating predictions for {len(subset_doc_indices)} documents...")
predictions = []
class_names = get_class_names()

for doc_index in subset_doc_indices:
    try:
        text_to_predict = full_dataset_df.loc[doc_index]['text']
        
        # 1. Get the 20-dim probability vector from the baseline model
        baseline_prob_vector = baseline_model.predict_proba([text_to_predict])
        
        # 2. Get the baseline's crisp prediction by finding the highest probability
        bert_pred_index = np.argmax(baseline_prob_vector, axis=1)[0]
        
        # 3. Pass the probability vector to the fuzzy model to get its prediction
        fuzzy_pred_index = fuzzy_model.predict(baseline_prob_vector)[0]

        predictions.append({
            'Document_ID': doc_index + 1,
            'BERT_Prediction': class_names[bert_pred_index], # Convert index to class name
            'Fuzzy_Hybrid_Prediction': class_names[fuzzy_pred_index] # Convert index to class name
        })
    except KeyError:
        print(f"Warning: Document index '{doc_index}' not found in the full dataset. Skipping.")
        continue

df_predictions = pd.DataFrame(predictions)

# --- Merge and Save Results ---
df_results = pd.merge(df_consensus, df_predictions, on='Document_ID')
df_results.to_csv(RESULTS_FILE, index=False)
print(f"Full results saved to '{RESULTS_FILE}'")

# --- Calculate and Display Performance Metrics ---
print("\n--- Performance vs. Human Consensus ---")

if not df_results.empty:
    y_true = df_results['Human_Consensus_Label']
    y_pred_bert = df_results['BERT_Prediction']
    y_pred_fuzzy = df_results['Fuzzy_Hybrid_Prediction']
    
    print("\n\n===== BASELINE BERT MODEL REPORT =====")
    print(classification_report(y_true, y_pred_bert, zero_division=0))
    
    print("\n\n===== FUZZY-HYBRID MODEL REPORT =====")
    print(classification_report(y_true, y_pred_fuzzy, zero_division=0))
    
    print("\n--- SUMMARY FOR DISSERTATION TABLE ---")
    report_bert = classification_report(y_true, y_pred_bert, output_dict=True, zero_division=0)
    report_fuzzy = classification_report(y_true, y_pred_fuzzy, output_dict=True, zero_division=0)
    
    print(f"Baseline BERT   | Accuracy: {report_bert['accuracy']:.2f} | F1 (Weighted): {report_bert['weighted avg']['f1-score']:.2f} | Precision (Weighted): {report_bert['weighted avg']['precision']:.2f} | Recall (Weighted): {report_bert['weighted avg']['recall']:.2f}")
    print(f"Fuzzy-Enhanced  | Accuracy: {report_fuzzy['accuracy']:.2f} | F1 (Weighted): {report_fuzzy['weighted avg']['f1-score']:.2f} | Precision (Weighted): {report_fuzzy['weighted avg']['precision']:.2f} | Recall (Weighted): {report_fuzzy['weighted avg']['recall']:.2f}")

else:
    print("No results to display. Please check your document IDs and data loading functions.")

print("\n--- Analysis Complete ---")