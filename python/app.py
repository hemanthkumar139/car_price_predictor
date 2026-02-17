from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model_package= joblib.load("../models/car_price_model_v2.pkl")
model=model_package["model"]
features=model_package["feature"]

@app.get("/")
def home():
    return {"running":"successfully"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([{
        "Age": data["AGE"],
        "KM": data["KM"],
        "Weight": data["WEIGHT"],
        "HP": data["HP"]
    }])
    df=df[features]
    price = model.predict(df)[0]
    return {"Predicted Price": price}
