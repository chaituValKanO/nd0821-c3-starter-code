import os
import pandas as pd
from data import process_data

def check_column_integrity(data: pd.DataFrame, expected_colums: list):
    
    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)

def check_shape(data, expected_columns):
    assert data.shape[0] > 1000
    assert data.shape[1] == len(expected_columns)


def check_saved_files():
    assert os.path.isfile("scaler.pkl")
    assert os.path.isfile("logistic_model.pkl")
    assert os.path.isfile("encoder.pkl")