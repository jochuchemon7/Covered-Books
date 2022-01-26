"""
# Parameter Selection with Preprocessing, pipelines, pipelines in Grid Searches and General Pipeline Interface
"""

# Example of Non-Pipeline Preprocessing with Cancer Data Set
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
scaler = MinMaxScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc = SVC().fit(X_train_scaled, y_train)
print(f'Test Score: {round(svc.score(X_test_scaled, y_test), 3)}')


# --- Parameter Selection with Preprocessing ---

# Naive GridSearchCV() approach
from sklearn.model_selection import GridSearchCV

param_grid = {'gamma': [.001, .01, .1, 1, 10, 100],
              'C': [.001, .01, .1, 1, 10, 100]}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print(f'Best Cross Validation Score: {round(grid.best_score_, 3)}')
print(f'Best Set Score: {round(grid.score(X_test_scaled, y_test), 3)}')
print(f'Best Parameters: {grid.best_params_}')

# NOTE: Because inside grid the given training set becomes training and test and we scaled the original training and test
# Set the ranges on the original test would be different than the test on grid from the original training scaled

import mglearn
mglearn.plots.plot_improper_processing()

# NOTE: Splitting of the data during cross validation should be done before doing any preprocessing
# CV should be the outermost loop in your processing

# --- Building Pipelines ---  (Most common used is in chaining preprocessing steps)
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])  # A list of tuples (steps) with an assign name
pipe.fit(X_train, y_train)  # First fit is from first step (scaler) and then fit of SVC()
print(f'Test Score: {round(pipe.score(X_test, y_test), 3)}')  # First scales test data and then score with svc


# --- Using Pipelines in Grid Searches ---

# We specify the name of the step the param will go to (svm__parameter)
param_grid = {'svm__gamma': [.001, .01, .1, 1, 10, 100],
              'svm__C': [.001, .01, .1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print(f'Best Cross-Validation Accuracy: {round(grid.best_score_, 2)}')
print(f'Test Set Score: {round(grid.score(X_test, y_test), 3)}')
print(f'Best Parameters: {grid.best_params_}')

# NOTE: Each split in CV the Scaler is refit with only the training splits (fit of scaler only has training)
mglearn.plots.plot_proper_processing()

# -- Illustrating Information Leakage -- (like scaler.fit(X) and not solely with X_train)
import numpy as np

rng = np.random.RandomState(seed=0)
X = rng.normal(size=(100, 10000))  # 100 entries and 10,000 features from Gaussian Distribution
y = rng.normal(size=(100, ))
# NOTE: There is no relation between X and y

from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
X_selected = select.transform(X)
print(f'X_selected.shape: {X_selected.shape}')
print(f'Cross Validation Accuracy (cv only on ridge):'
      f' {round(np.mean(cross_val_score(Ridge(), X_selected, y, cv=5)), 3)}')
# NOTE: The only reason why it gave a good score is bc it by chance some random features had correlation
# bc we fit the feature selection outside the cross-validation the training and test data had correlated features aka (Leaked Information)

pipe = Pipeline([('select', SelectPercentile(score_func=f_regression, percentile=5)),
                 ('ridge', Ridge())])
print(f'Cross Validation Accuracy (Pipeline):'
      f' {round(np.mean(cross_val_score(pipe, X_selected, y, cv=5)), 3)}')

# --- General Pipeline Interface ---

# Implementation of the fit in Pipeline.fit
def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:  # Iterate over all steps except the last one
        X_transformed = estimator.fit_transform(X_transformed, y)  # Fit transform on each step
    self.steps[-1][1].fit(X_transformed, y)  # Fit the last step
    return self

def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:  # Iterator over all steps except the last one
        X_transformed = step[1].transform(X_transformed)  # Transform the data
    return self.steps[-1][1].predict(X_transformed)  # Fit the last step


# -- Convenient Pipeline Creation with make_pipeline --
from sklearn.pipeline import make_pipeline  # Automatically names the steps based on the functions

pipe_long = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC(C=100))])
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))

print(f'Pipeline Steps: {pipe_short.steps}')  # View the name given to the steps

# If multiple steps have the same class name a number is appended
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), PCA(n_components=3), StandardScaler())
print(f'Pipeline Steps: {pipe.steps}')


# -- Accessing Step Attributes --
pipe.fit(cancer.data)
components = pipe.named_steps['pca'].components_
print(f'Components.shape: {components.shape}')

sample_pca = pipe.named_steps['pca']
print(f'Explained Variance: {sample_pca.explained_variance_}')


# -- Accessing Attributes in a Grid-Search Pipeline --
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())  # Pipeline with scaler and model

# Splitting the data into training and test
param_grid = {'logisticregression__C': [.01, .1, 1, 10, 100]}
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

# Accessing the coefficients of Logistic Regression
print(f'Best Estimator: {grid.best_estimator_}')

# Accessing the Logistic Regression step
print(f'Logistic Regression Step: {grid.best_estimator_.named_steps["logisticregression"]}')

# Accessing the Logistic Regression Coefficients
lr = grid.best_estimator_.named_steps['logisticregression']
print(f'Logistic Regression Coefficients: {lr.coef_}')



