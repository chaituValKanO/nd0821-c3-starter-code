from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.encoders import jsonable_encoder
import json

import pandas as pd
import logging

from starter.code.predict import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

class CensusData(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str


app = FastAPI()

@app.get("/")
async def greeting():
    return {"message": "Welcome to FASTAPI inference on census data!"}

@app.post("/predict")
async def make_predictions(record: CensusData):
    logger.info("The post request data: %s", record)
    input_data = jsonable_encoder(record)
    input_data = pd.DataFrame(input_data, index=[0])
    logger.info(f"Input data: {input_data}")
    result = predict(input_data)
    logger.info("The predicted result is %s", result)
    return result

