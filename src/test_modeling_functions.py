"""
Module for testing modeling functions
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from prep_data import process_data
import model


@pytest.fixture(name="data")
def _data():
    """
    Prepped data as a fixture
    """
    # Load data
    DATA_PATH = "./data/prepped/census.csv"
    data = pd.read_csv(DATA_PATH)
    return data


@pytest.fixture(name="cat_feats")
def _cat_feats():
    """
    List of categorical features as a fixture
    """

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
    return cat_features


def test_process_data(data, cat_feats):
    """
    Test `process_data` function
    """

    X, y, encoder, lb = process_data(data,
                                     cat_feats,
                                     label="salary",
                                     training=True)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    X_test, y_test, _, _ = process_data(data,
                                        cat_feats,
                                        label="salary",
                                        training=False,
                                        encoder=encoder,
                                        lb=lb)

    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert X.shape[1] == X_test.shape[1]


def test_train_model(data, cat_feats):
    """
    test `train_model` function
    """
    X, y, encoder, lb = process_data(data,
                                     cat_feats,
                                     label="salary",
                                     training=True)

    clf = model.train_model(X, y)

    assert isinstance(clf, RandomForestClassifier)


def test_inference(data, cat_feats):
    """
    test `inference` function
    """
    X, y, encoder, lb = process_data(data,
                                     cat_feats,
                                     label="salary",
                                     training=True)

    clf = model.train_model(X, y)

    y_pred = model.inference(clf, X)

    assert isinstance(y_pred, np.ndarray)


def test_compute_model_metrics(data, cat_feats):
    """
    test `compute_model_metrics` function
    """
    X, y, encoder, lb = process_data(data,
                                     cat_feats,
                                     label="salary",
                                     training=True)

    clf = model.train_model(X, y)

    y_pred = model.inference(clf, X)

    metrics = model.compute_model_metrics(y, y_pred)

    assert isinstance(metrics, tuple)
    assert len(metrics) == 3
