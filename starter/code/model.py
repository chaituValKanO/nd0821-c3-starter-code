import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

## Slice testing
def slice_census_data(df, cat_col='education'):
    if os.path.isfile("slice_output.txt"):
        os.remove("slice_output.txt")
        
    for value in df[cat_col].unique():
        df_temp = df[df[cat_col] == value]
        
        mean_hours = df_temp["hours_per_week"].mean()
        stddev_hours = df_temp["hours_per_week"].std()
        
        file_object = open('slice_output.txt', 'a')
        file_object.write(f'\n------Holding Categorical feature {cat_col} and value {value} fixed -------')
        file_object.write(f"\nMean Hours Per Week: {mean_hours}")
        file_object.write(f"\nStd.Dev in Hours Per Week: {stddev_hours}")
        file_object.write('\n------------------------------------------')
        file_object.close()


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lg = LogisticRegression(random_state=123)
    lg.fit(X_train, y_train)

    return lg


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_files(model, scaler, encoder, lb):
    joblib.dump(model, "logistic_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(lb, "label_encoder.pkl")