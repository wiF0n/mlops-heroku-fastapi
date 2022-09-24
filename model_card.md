# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random forest model (scikit-learn implementation) with default hyperparameters
## Intended Use
Predict whether the salary of a given person will exceed $50K or not.
## Training Data
80% of [Census income data](https://archive.ics.uci.edu/ml/datasets/census+income) (using 42 as a seed for sampling with `train_test_split` function)
## Evaluation Data
20% of [Census income data](https://archive.ics.uci.edu/ml/datasets/census+income) (using 42 as a seed for sampling with `train_test_split` function)
## Metrics
Model was evaluated using precision, recall and f1 score. Values for deployed models are as follows, Precision: 0.758, Recall: 0.645, f1 score: 0.697 
## Ethical Considerations
Dataset has features related to race and gender. 
## Caveats and Recommendations
Models perform differently for different slices of data. Please check [slice scores](./model/slice_scores.txt) to see performance for given slice.
