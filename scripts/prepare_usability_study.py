import pandas as pd

# --- CONFIGURATION ---

#ChoosING the 5 Document IDs to show your.
DOCUMENT_IDS_TO_SHOW = [10, 12, 50, 8, 33] 

RESULTS_FILE = 'focused_analysis_results.csv'
TEXT_MAPPING_FILE = 'document_text_mapping.csv' 

# --- 1. LOAD THE DATA ---

print("Loading your analysis results and the document text mapping...")
try:
    df_results = pd.read_csv(RESULTS_FILE)
    df_text_map = pd.read_csv(TEXT_MAPPING_FILE)
except FileNotFoundError as e:
    print(f"ERROR: A required file was not found: {e.filename}")
    print("Please ensure both focused_analysis_results.csv and document_text_mapping.csv exist.")
    exit()

# --- 2. GENERATE AND PRINT THE MATERIALS ---

print("\n--- Generating Materials for Usability Study ---")
print("Copy and paste the text below into a new document for your participants.")
print("="*60)

for i, doc_id in enumerate(DOCUMENT_IDS_TO_SHOW):
    
    # Find the result row for the current document ID
    result_row = df_results[df_results['Document_ID'] == doc_id]
    
    # Find the text row from our new mapping file
    text_row = df_text_map[df_text_map['Document_ID'] == doc_id]
    
    if result_row.empty or text_row.empty:
        print(f"\nWARNING: Document ID {doc_id} not found in one of your files. Skipping.")
        continue
        
    # Retrieve all information
    text = text_row['text'].iloc[0]
    human_label = result_row['Human_Consensus_Label'].iloc[0]
    bert_pred = result_row['BERT_Prediction'].iloc[0]
    fuzzy_pred = result_row['Fuzzy_Hybrid_Prediction'].iloc[0]

    # Print the formatted output
    print(f"\nExample Case {i+1}\n")
    print("**Document Text:**")
    print(f'"{text.strip()}"\n') # .strip() cleans up any extra whitespace
    print(f"**Context:** A panel of human users determined the most appropriate category for this text was **{human_label}**.\n")
    print("Please review the predictions made by two different AI systems for this text:\n")
    print(f"* **System A predicted:** {bert_pred}")
    print(f"* **System B predicted:** {fuzzy_pred}")
    print("\n" + "-"*60)

print("\n--- Materials Generated Successfully ---")