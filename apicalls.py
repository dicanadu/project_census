from ml_model.ml.data import load_data
import requests
import json

data_path = "./package_code/data/census.csv"
url = "https://app-income-predictor-303410a66978.herokuapp.com/predict"

example_input = {
    "age": 35,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Protective-serv",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}


def call_api_from_csv(row,
                      api_url="http://127.0.0.1:8000/predict"):
    data = load_data(data_path)
    sample = json.dumps(dict(data.iloc[row, :-1]))
    response = requests.post(api_url, data=sample)
    return response.json()


def call_api_from_dictionary(input,
                             api_url="http://127.0.0.1:8000/predict"):
    sample_input = json.dumps(input)
    response = requests.post(api_url, data=sample_input)
    return response.json()


print(call_api_from_csv(5, url))
print(call_api_from_dictionary(example_input, url))
