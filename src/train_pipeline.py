"""
Module for running training pipeline
"""
import os
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import prep_data
import model

logging.basicConfig(level=logging.INFO)

# Load data
DATA_PATH = "../data/prepped/census.csv"
data = pd.read_csv(DATA_PATH)

# train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

X_train, y_train, encoder, lb = prep_data.process_data(
    train, categorical_features=cat_features, label="salary", training=True)

X_test, y_test, _, _ = prep_data.process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb)

# Train model
clf = model.train_model(X_train, y_train)

# Check model metrics
y_pred = model.inference(clf, X_test)

model_metrics = model.compute_model_metrics(y_test, y_pred)

logging.info('Precision: %.3f, Recall: %.3f, f1 score: %.3f' % model_metrics)

# Save model and encoders
MODEL_DIR_PATH = "../model"
MODEL_NAME = "rf_clf.pkl"
CAT_ENCODER_NAME = "cat_enc.pkl"
LB_NAME = "lb.pkl"

with open(os.path.join(MODEL_DIR_PATH, MODEL_NAME), "wb") as model_f:
    pickle.dump(clf, model_f)

with open(os.path.join(MODEL_DIR_PATH, CAT_ENCODER_NAME), "wb") as enc_f:
    pickle.dump(encoder, enc_f)

with open(os.path.join(MODEL_DIR_PATH, LB_NAME), "wb") as lb_f:
    pickle.dump(lb, lb_f)
