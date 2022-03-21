# Script to train machine learning model.
from base64 import encode
import pandas as pd
from sklearn.model_selection import train_test_split

from data import process_data
from model import slice_census_data, train_model, compute_model_metrics, inference, save_files

data = pd.read_csv("../data/cleaned_census_data.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

for cat_col in cat_features:
    slice_census_data(data, cat_col)

X_train, y_train, encoder, scaler, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, scaler, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, scaler=scaler, lb=lb
)

lg_model = train_model(X_train, y_train)
preds = inference(lg_model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"The test results are Precision={precision}, recall={recall}, fbeta={fbeta}")

save_files(lg_model, scaler, encoder, lb)