# src/data_loader.py

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE, DEBUG_MODE, DEBUG_DATA_SIZE

def load_and_split_data():
    """Loads, filters, and splits the 20 Newsgroups dataset for the main experiment."""
    print("Loading 20 Newsgroups dataset for training/testing...")
    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # Filtering out documents that are too short to be meaningful
    X_full, y_full = [], []
    for doc, label in zip(newsgroups_data.data, newsgroups_data.target):
        if len(doc.split()) > 10:
            X_full.append(doc)
            y_full.append(label)

   
    if DEBUG_MODE:
        print(f"--- RUNNING IN DEBUG MODE: Using {DEBUG_DATA_SIZE} samples ---")
        X, y = X_full[:DEBUG_DATA_SIZE], y_full[:DEBUG_DATA_SIZE]
    else:
        X, y = X_full, y_full

    target_names = newsgroups_data.target_names
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if not DEBUG_MODE else None
    )
    
    print("Data loading and splitting complete.")
    return X_train, X_test, y_train, y_test, target_names

def load_full_unsplit_data():
    """
    Loads the entire 20 Newsgroups dataset into a single pandas DataFrame
    without splitting it. This is useful for looking up specific documents by index.
    """
    print("Loading full unsplit dataset...")
    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    df = pd.DataFrame({
        'text': newsgroups_data.data,
        'label': newsgroups_data.target
    })
    
    # Filtering out short documents, consistent with the main experiment
    df = df[df['text'].str.split().str.len() > 10].reset_index(drop=True)
    return df

def get_class_names():
    """Returns the list of class names from the dataset."""
    return fetch_20newsgroups(subset='all').target_names