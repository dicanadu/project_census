import pytest
import pandas as pd
from ml_model.ml.data import load_data, process_data
from ml_model.ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
import os


@pytest.fixture(scope='session')
def data():
    root = os.path.dirname(__file__)
    file_path = os.path.join(root, "data", "census.csv")
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    return df


@pytest.fixture(scope="session")
def models(data):
    preprocessor_pipe, X, y = process_data(data, "salary")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        stratify=y)
    model_1 = train_model(preprocessor_pipe, X_train, y_train, "accuracy")
    return model_1, X_test, y_test


def test_load_data(data):
    """Test whether data was correctly imported"""
    root = os.path.dirname(__file__)
    file_path = os.path.join(root, "data", "census.csv")
    df_load = load_data(file_path)
    assert df_load.shape == data.shape


def test_train_model(data, models):
    """
    Test that the model used corrected number of features and that scores are
    meaningful to the problem with a greater scorer on 50% for recall
    and precision, and 80% for accuracy
    """
    model_1 = models[0]
    assert len(model_1.feature_names_in_ == 14)
    assert model_1.cv_results_["mean_test_score"].mean() > 0.80


def test_compute_model_metrics(models):
    """
    Check that all metrics are greater thant 0 and less than 1
    """
    model_1, X_test, y_test = models
    y_preds_m1 = inference(model_1, X_test)

    metrics_1 = compute_model_metrics(y_test, y_preds_m1)

    assert all(0 < metric < 1 for metric in metrics_1)


def test_inference(models):
    """
    Check that inference does not return an empty prediction
    """
    model_1, X_test = models[:2]
    y_preds_m1 = inference(model_1, X_test)

    assert y_preds_m1.size > 0
