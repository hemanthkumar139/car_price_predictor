import joblib
import pandas as pd

model_package=joblib.load("../models/car_price_model_v2.pkl")
rf=model_package["model"]
feature=model_package["feature"]

def predict_price(age,km,weight,hp):
    input_data=pd.DataFrame([[age,km,weight,hp]],columns=feature)
    return rf.predict(input_data)[0]

print("🚗 Car Price Prediction Tool\n")

def get_positive_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("Value cannot be negative. Try again.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a number.")

age = get_positive_input("Enter car age (years): ")
km = get_positive_input("Enter kilometers driven: ")
weight = get_positive_input("Enter car weight (kg): ")
hp = get_positive_input("Enter horsepower (HP): ")

price = predict_price(age, km, weight, hp)

print(f"\n💰 Estimated Car Price: {round(price, 2)}")   


