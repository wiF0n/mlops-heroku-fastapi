"""
Post request to live heroku app
"""

import requests

features = {
    "age": 30,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Doctorate",
    "education_num": 16,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
}

resp = requests.post("https://mlops-heroku-fastapi-dk.herokuapp.com/inference",
                     json=features)

print(f"Response code: {resp.status_code}")
print(f"Response body: {resp.json()}")
