from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd


app = FastAPI()


class Response(BaseModel):
    age: int = Field(gt=0, lt=150, example=20)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=10000)
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example=10)
    marital_status: str = Field(alias="marital-status", example="Divorced")
    occupation: str = Field(example="Prof-specialty")
    relationship: str = Field(example="Wife")
    race: str = Field(example="Black")
    sex: str = Field(example="Female")
    capital_gain: int = Field(alias="capital-gain", example=1000)
    capital_loss: int = Field(alias="capital-loss", example=30)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(alias="native-country", example="Mexico")


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
