from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from starter.code.predict import predict

class Record(BaseModel):
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
    salary: float


app = FastAPI()

@app.get("/")
async def greeting():
    return {"message": "Hello!"}

@app.post("/predict/")
async def make_predictions(record: Record):
    return record
    # data = record.dict()
    # data = pd.read_json(data)
    # result = predict(data)
    # return result

