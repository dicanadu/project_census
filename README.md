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

Run the command inside the <package_code/app> folder and use the url

```bash
uvicorn uvicorn package_code.app.main:app --reload
```

Later you can use the file apicalls.py to obtain predictions.

```bash
python apicalls.py
```

The required fields are those stablished by the US census as follows:

```json
{
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
To deploy on heroku on git: create an app, add, commit and push to remote

```bash
heroku create app-name --buildpack heroku/python
git init
git add .
git commit -m 'Commit message'
heroku git:remote --app app-name
git push heroku master
```

Procfile and requirements.txt should be on the root

Procfile:
web: uvicorn package_code.app.main:app --host=0.0.0.0 --port=${PORT:-5000}

Alternatively, push only app directory, move Procfile and requirements.txt
to the app directory

Procfile
web: uvicorn main app --host=0.0.0.0 --port=${PORT:-5000}

```bash
heroku config:set BUILD_DIR=package_code/app
git subtree push --prefix package_code/app heroku master
```
