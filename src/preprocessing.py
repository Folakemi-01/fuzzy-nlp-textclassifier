# src/preprocessing.py

import re
import nltk
import os

# --- NLTK Setup ---
# Ensures NLTK data is downloaded to a user-writable directory
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True, download_dir=nltk_data_path)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True, download_dir=nltk_data_path)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True, download_dir=nltk_data_path)

# --- Preprocessing Function ---
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Cleans and preprocesses raw text for BERT tokenization."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.tokenize.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)