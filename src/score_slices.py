"""
Module for creating model scores per slice
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import prep_data
import model

# Load data
DATA_PATH = "../data/prepped/census.csv"
data = pd.read_csv(DATA_PATH)

# Load model and encoders
MODEL_DIR_PATH = "../model"
MODEL_NAME = "rf_clf.pkl"
CAT_ENCODER_NAME = "cat_enc.pkl"
LB_NAME = "lb.pkl"

with open(os.path.join(MODEL_DIR_PATH, MODEL_NAME), "rb") as model_f:
    clf = pickle.load(model_f)

with open(os.path.join(MODEL_DIR_PATH, CAT_ENCODER_NAME), "rb") as enc_f:
    encoder = pickle.load(enc_f)

with open(os.path.join(MODEL_DIR_PATH, LB_NAME), "rb") as lb_f:
    lb = pickle.load(lb_f)

# train-test split
train, test = train_test_split(data, test_size=0.20, random_state=42)
test = test.reset_index(drop=True)
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

X_test, y_test, _, _ = prep_data.process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb)

y_pred = model.inference(clf, X_test)

# Loop through categorical features and their levels and compute model score
slice_scores = []
for cat_feat in cat_features:
    for cat_level in test[cat_feat].unique():

        slice_idx = test[test[cat_feat] == cat_level].index
        y_true_slice = y_test[slice_idx]
        y_pred_slice = y_pred[slice_idx]

        precision, recall, fbeta = model.compute_model_metrics(
            y_true_slice, y_pred_slice)

        level_metrics = (f"{cat_feat} - {cat_level}: "
                         f"Precision: {precision:.3f}, "
                         f"Recall: {recall:.3f}, "
                         f"f1 score: {fbeta:.3f}")
        slice_scores.append(level_metrics)
    slice_scores.append("\n")

# save model score file
SLICE_SCORE_PATH = "../model/slice_scores.txt"
with open(SLICE_SCORE_PATH, "w", encoding="UTF-8") as slice_f:
    for slice_score in slice_scores:
        slice_f.write(f"{slice_score}\n")
