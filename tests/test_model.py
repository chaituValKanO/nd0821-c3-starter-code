import sys
sys.path.append('../nd0821-c3-starter-code')

import os
import json
import numpy as np

from starter.code.model import inference
from starter.code.predict import predict

def test_save_files():
    '''
    This function tests the save files method in model.py
    '''
    assert os.path.isfile("starter/code/scaler.pkl")
    assert os.path.isfile("starter/code/logistic_model.pkl")
    assert os.path.isfile("starter/code/encoder.pkl")
    assert os.path.isfile("starter/code/label_encoder.pkl")

def test_predict(data, target_label):
    X = data.drop(target_label, axis=1)
    assert isinstance(json.loads(predict(X))['prediction'], list)