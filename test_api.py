"""
Module for testing api
"""

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture(name="test_client")
def _test_client():
    """
    Run app
    """
    test_client = TestClient(app)
    return test_client


def test_greeting(test_client):
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"greeting": "Hello User!"}


def test_not_implemented(test_client):
    resp = test_client.get("/opica")
    assert resp.status_code != 200


def test_predict_high(test_client):
    resp = test_client.post("/inference",
                            json={
                                "age": 39,
                                "workclass": "State-gov",
                                "fnlgt": 77516,
                                "education": "Bachelors",
                                "education_num": 13,
                                "marital_status": "Never-married",
                                "occupation": "Adm-clerical",
                                "relationship": "Not-in-family",
                                "race": "White",
                                "sex": "Male",
                                "capital_gain": 2174,
                                "capital_loss": 0,
                                "hours_per_week": 40,
                                "native_country": "United-States"
                            })
    assert resp.status_code == 200
    assert resp.json() == {"label": "<=50K"}


def test_predict_low(test_client):
    resp = test_client.post("/inference",
                            json={
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
                            })
    assert resp.status_code == 200
    assert resp.json() == {"label": ">50K"}
