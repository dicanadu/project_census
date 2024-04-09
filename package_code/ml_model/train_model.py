# Script to train machine learning model.
import os
from sklearn.model_selection import train_test_split
from ml_model.ml.data import process_data, load_data
from ml_model.ml.model import train_model, compute_model_metrics
from ml_model.ml.model import inference
import datetime
import time
import joblib


def train_and_save(X_train,
                   y_train,
                   preprocessor_pipe,
                   path,
                   metric="recall"):
    """"
    Process a logistic regression model optimizing for choosen metric
    saves output to specified folder

    Inputs:
    _______
    X_train: Trainning data
    y_train: Labels
    preprocessor_pipe: pipeline to preprocess numeric and categorical variables
    path: Model path to be saved
    metric: Choose metric to optimize the model.
    ------

    Outputs:
    _______
    model: returns a model trained and saved to specified path.
    """
    model = train_model(preprocessor_pipe, X_train, y_train, metric)
    time_file = datetime.datetime \
        .fromtimestamp(time.time()) \
        .strftime("%Y-%m-%d")
    save_path = os.path.join(path, f"model_{time_file}_{metric}.pkl")
    joblib.dump(model, save_path)
    return model


def performance_testing(model, X_test, y_test, category):
    """"
    Computes metrics for a given category holding fixed a categorical feature

    Inputs:
    _______
    model: Trained model to test data.
    X_test: Test data.
    y_test: Labels.
    category: Given category to fix data.
    ------

    Outputs:
    _______
    results: Dictionary computing every category with its respective metrics
    on recall, precision, and fbeta
    """
    unique_values = X_test[category].unique()
    X_test = X_test.reset_index()
    results = {}
    for value in unique_values:
        X_mask = X_test[X_test[category] == value].index
        X_filtered = X_test.iloc[X_mask]
        y_filtered = y_test[X_mask]
        y_preds = inference(model, X_filtered)
        metrics = compute_model_metrics(y_filtered, y_preds)
        results[value] = {"recall": metrics[0],
                          "precision": metrics[1],
                          "fbeta": metrics[2]}
    return results


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(root, "data", "census.csv")
    model_path = os.path.join(root, "model")

    df = load_data(data_path)
    preprocessor_pipe, X, y = process_data(df, "salary")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        stratify=y)
    # model = train_and_save(X_train,
    #                y_train,
    #                preprocessor_pipe,
    #                path="./ml_model/model/model",
    #                metric="recall")

    last_model = os.listdir(model_path)[-1]
    print(last_model)
    model = joblib.load(os.path.join(model_path, last_model))
    preds = inference(model, X_test)
    scores = compute_model_metrics(y_test, preds)
    scores_fixed = performance_testing(model, X_test, y_test, "race")
    print(scores_fixed)
