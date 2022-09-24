"""
Module for api creation
"""
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.prep_data import process_data
from src.model import inference


class Features(BaseModel):
    age: int = Field(example=20)
    workclass: str = Field(example="Private")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(example=13)
    marital_status: str = Field(example="Divorced")
    occupation: str = Field(example="Handlers-cleaners")
    relationship: str = Field(example="Husband")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=0)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(example="United-States")


app = FastAPI()


# Greeting GET method
@app.get("/")
async def greeting():
    return {"greeting": "Hello User!"}


# Inference POST method
@app.post("/inference")
async def predict(features: Features):
    # Load model and encoders
    MODEL_DIR_PATH = "./model"
    MODEL_NAME = "rf_clf.pkl"
    CAT_ENCODER_NAME = "cat_enc.pkl"
    LB_NAME = "lb.pkl"

    with open(os.path.join(MODEL_DIR_PATH, MODEL_NAME), "rb") as model_f:
        clf = pickle.load(model_f)

    with open(os.path.join(MODEL_DIR_PATH, CAT_ENCODER_NAME), "rb") as enc_f:
        encoder = pickle.load(enc_f)

    with open(os.path.join(MODEL_DIR_PATH, LB_NAME), "rb") as lb_f:
        lb = pickle.load(lb_f)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    features_dict = {
        "age": [features.age],
        "workclass": [features.workclass],
        "fnlgt": [features.fnlgt],
        "education": [features.education],
        "education-num": [features.education_num],
        "marital-status": [features.marital_status],
        "occupation": [features.occupation],
        "relationship": [features.relationship],
        "race": [features.race],
        "sex": [features.sex],
        "capital-gain": [features.capital_gain],
        "capital-loss": [features.capital_loss],
        "hours-per-week": [features.hours_per_week],
        "native-country": [features.native_country]
    }

    df = pd.DataFrame.from_dict(features_dict)

    X, _, _, _ = process_data(df,
                              categorical_features=cat_features,
                              training=False,
                              encoder=encoder,
                              lb=lb)

    y_pred = inference(clf, X)

    label = lb.inverse_transform(y_pred)[0]
    return {"label": label}
