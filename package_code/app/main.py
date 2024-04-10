from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd


app = FastAPI()


class Response(BaseModel):
    age: int = Field(gt=0, lt=150)
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


example = {
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


@app.get("/")
async def root():
    message = 'Welcome to income classification predictor, ARE YOU OVER 50K?'
    return message


@app.post("/predict")
async def predict_income(response: Response):
    data = response.model_dump()
    dataframe = pd.DataFrame([data])
    original_columns = [col.replace("_", "-") for col in dataframe.columns]
    dataframe.columns = original_columns

    model_path = os.path.dirname(__file__)
    model_to_use = [file for file in os.listdir(model_path)
                    if file.endswith(".pkl")][-1]
    full_path = os.path.join(model_path, model_to_use)
    print(full_path)
    model = joblib.load(full_path)

    prediction = model.predict(dataframe)
    return {'prediction': str(prediction[0])}
