import sys
sys.path.append('../nd0821-c3-starter-code')

import os
import pandas as pd
from starter.code.data import process_data

def test_column_integrity(data: pd.DataFrame, expected_columns: list):
    
    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_columns) == list(these_columns)

def test_shape(data, expected_columns):
    assert data.shape[0] > 1000
    assert data.shape[1] == len(expected_columns)


def test_saved_files():
    assert os.path.isfile("starter/code/scaler.pkl")
    assert os.path.isfile("starter/code/logistic_model.pkl")
    assert os.path.isfile("starter/code/encoder.pkl")
    assert os.path.isfile("starter/code/label_encoder.pkl")