"""
# Wrap Up
"""
# --- Building Your Own Estimator ---

from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, first_parameter=1, second_parameter=2):
        self.first_parameter = first_parameter
        self.second_parameter = second_parameter
    def fit(self, X, y=None):  # Fit only takes X and y
        print('Fitting the model right here')
        return self
    def transform(self, X):  # Applying transformation to X
        X_transformed = X + 1
        return X_transformed

# NOTE: For a classifier or regressor instead of TransformerMixin inherit ClassifierMixin or RegressorMixin
# And instead of transform function implement a predict function


