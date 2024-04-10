# Income Classificator ML model predictor

## Objective

To test a Logistic Regression model to predict whether salary
of a person is less (0) or above 50k (1)

Data: https://archive.ics.uci.edu/dataset/20/census+income

## How to use this project

Install package running the command:

```bash
pip install package_code/
```

To run, build and slice a ML model, use the command adjusting parameters
metric, save, and slicing:

```bash
#default parameters --metric 'accuracy' --save True --slicing None
python create_model_slicing.py

#To slice on race
python create_model_slicing.py --slicing 'race'
```

Slicing options:
- workclass,
- education,
- marital-status,
- occupation,
- relationship,
- race,
- sex,
- native-country

Models will be saved on directory package_code/model/

## How call API

Run the command inside the app folder and use the url

```bash
uvicorn main:app --reload
```

Later you can use the file apicalls.py to obtain predictions.

The required fields are those stablished by the US census as follows:

```json
data = {
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
    "native-country": "United-States"
}
```
To deploy on heroku once logged and having created an app, use the following
command:

```bash
heroku config:set BUILD_DIR=package_code/app
git subtree push --prefix package_code/app heroku master
```
