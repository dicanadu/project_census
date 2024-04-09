from fastapi.testclient import TestClient
from app.main import app
import pytest
import json

client = TestClient(app)


@pytest.fixture(scope="session")
def data():
    negative = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Protective-serv",
        "relationship": "Husband",
        "race": "White",
        "sex": "Female",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"}

    positive = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Protective-serv",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"}
    return negative, positive


def test_api_locally_get_root():
    """Test root get a response 200"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()[:4] == 'Welc'


def test_api_positive_prediction(data):
    """Test positive prediction code 200 and 1 as output"""
    positive = json.dumps(data[1])
    r = client.post("/predict",
                    data=positive)
    assert r.status_code == 200
    assert r.json()["prediction"] == "1"


def test_api_negative_prediction(data):
    """Test negative prediction code 200 and 0 as output"""
    negative = json.dumps(data[0])
    r = client.post("/predict",
                    data=negative)
    assert r.status_code == 200
    assert r.json()["prediction"] == "0"
