import joblib

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.code.data import clean_data, process_data
from starter.code.model import compute_model_metrics, inference

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def predict(data):
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    lg_model = joblib.load("logistic_model.pkl")
    lb = joblib.load("label_encoder.pkl")

    data = clean_data(data)

    X_test, _, encoder, scaler, lb = process_data(
        data, categorical_features=cat_features, label=None, training=False,
        encoder=encoder, scaler=scaler, lb=lb)
    
    preds = lb.inverse_transform(inference(lg_model, X_test))
    return {"prediction": preds}

if __name__ == "__main__":
    data = """[{
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
    }]"""

    data = pd.read_json(data, orient='list')
    print(predict(data))