# tests/test_data_loader.py

import pandas as pd
from src.data_loader import DataLoader

def test_load_data_success():
    """Tests that data is loaded correctly from an existing file."""
    loader = DataLoader(file_path="tests/sample_test_data.csv")
    data = loader.load_data()
    assert data is not None, "Data should not be None on successful load"
    assert isinstance(data, pd.DataFrame), "Loaded data should be a DataFrame"

def test_load_data_file_not_found():
    """Tests that the loader handles a missing file correctly."""
    loader = DataLoader(file_path="non_existent_file.csv")
    data = loader.load_data()
    assert data is None, "Data should be None when file is not found"
