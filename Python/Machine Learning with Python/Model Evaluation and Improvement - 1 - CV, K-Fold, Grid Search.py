"""
# CV, K-Fold, Grid Search
"""
import matplotlib.pyplot as plt
import mglearn.plots
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Traditional model evaluation with split data
X, y = make_blobs(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression().fit(X_train, y_train)
print(f'Test set score: {round(logreg.score(X_test, y_test), 2)}')

# --- Cross Validation ---
mglearn.plots.plot_cross_validation()

""" mglearn.plots.plot_cross_validation() """

# -- CV in scikit-learn --
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target)
print(f'Cross Validation Scores: {scores}')

scores = cross_val_score(logreg, iris.data, iris.target, cv=3)  # Changing number of folds
print(f'Cross Validation scores: {scores}')

print(f'Average Cross Validation score: {round(scores.mean(), 2)}')

# --- Stratified K-Fold CV and Other Strategies ---
print(f'Iris labels: \n{iris.target}')  # K-fold cv would not be good on clustered data targets

""" mglearn.plots.plot_stratified_cross_validation() """  # class proportions are the same in each fold

# -- More Control Over CV --
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)  # Using a STANDARD k-fold cv on a classification set

print(f'Cross Validation Scores: \n {cross_val_score(logreg, iris.data, iris.target, cv=kfold)}')

kfold = KFold(n_splits=3)  # UnStratified CV no good for IRIS data distribution
print(f'Cross Validation Scores: \n{cross_val_score(logreg, iris.data, iris.target, cv=kfold)}')

# - With shuffle and seed -
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print(f'Cross Validation Score: \n{cross_val_score(logreg, iris.data, iris.target, cv=kfold)}')

# - Leave-One-Out CV - (Each Single point is tested)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print(f'Number of cv iterations: {len(scores)}')
print(f'Mean Accuracy: {round(scores.mean(), 2)}')

# - Shuffle-Split CV - (Some points are not selected for training or testing)  (StratifiedShuffleSplit as well)
""" mglearn.plots.plot_shuffle_split() """

from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print(f'Cross Validation Score: {scores}')

# - CV with Groups - (training and test contain different groups for when multiple samples of the same group)
from sklearn.model_selection import GroupKFold

X, y = make_blobs(n_samples=12, random_state=0)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]  # Assume first 3 samples belong to the same group
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print(f'Cross Validation Scores: \n{scores}')

""" mglearn.plots.plot_group_kfold() """

# --- Grid Search ---

# - Simple Grid Search -
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print(f'Size of training set: {X_train.shape[0]}  size of the test set: {X_test.shape[0]}')
best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'gamma': gamma, 'C': C}

print(f'Best Score: {round(best_score, 2)}')
print(f'Best Parameters: {best_parameters}')

# - Grid Search with Validation Data - (To avoid using test data on our parameter selection)
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print(f'Size of training set: {X_train.shape[0]}  Size of validation set: {X_valid.shape[0]}   '
      f'Size of test set: {X_test.shape[0]}')
best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_valid, y_valid)
        if score > best_score:
            best_score = score
            best_parameters = {'gamma': gamma, 'C': C}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print(f'Best Score on validation set: {round(best_score, 2)}')
print(f'Best Parameters: {best_parameters}')
print(f'Test set score with best parameters: {round(test_score, 2)}')


# --- Grid Search With Cross-Validation --- (CrossValidation on multiple different parameter combinations)
best_score = 0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'gamma': gamma, 'C': C}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print(f'Best Score: {round(best_score, 2)}')
print(f'Best Parameters: {best_parameters}')
print(f'Test Set Score: {round(test_score, 2)}')

"""
mglearn.plots.plot_cross_val_selection()
mglearn.plots.plot_grid_search_overview()
"""

# -- Grid Search w/ Cross Validation using SKlearn -- (Same as above)
from sklearn.model_selection import GridSearchCV

param_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100],  # Creating a dict with param and values to use
              'C': [0.001, 0.01, 0.1, 1, 10, 100]}
print(f'Parameter Grid: \n{param_grid}')

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Test Set Score: {round(grid_search.score(X_test, y_test), 2)}')

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross Validation Score: {round(grid_search.best_score_, 2)}')  # Mean cv accuracy with cv performed on the training set
print(f'Best Estimator: \n{grid_search.best_estimator_}')

# - Analysing the result of cross-validation - scores for each split on all parameter settings
import pandas as pd

results = pd.DataFrame(grid_search.cv_results_)  # Viewing the results of a grid search
print(results.head())

# Using a heat map for C and gamma
import matplotlib.pyplot as plt
import seaborn as sns
scores = np.array(results.mean_test_score).reshape(6, 6)
sns.heatmap(scores, square=True, annot=True, cbar=False, cmap='viridis',
            xticklabels=param_grid['gamma'], yticklabels=param_grid['C'])
plt.xlabel('gamma')
plt.ylabel('C')

"""
# Or with mglearn
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'], ylabel='C',
                      yticklabels=param_grid['C'], cmap='viridis')
"""

# When ranges are not selected proper (Bad Example)
fig, ax = plt.subplots(1, 3, figsize=(13, 5))

param_grid_linear = {'C': np.linspace(1, 2, 6), 'gamma': np.linspace(1, 2, 6)}
param_grid_one_log = {'C': np.linspace(1, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
param_grid_range = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-7, -2, 6)}

for param_grid, axi in zip([param_grid_linear, param_grid_one_log, param_grid_range], ax.flat):
    grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)
    sns.heatmap(scores, square=True, annot=True, cbar=False, cmap='viridis',
                xticklabels=param_grid['gamma'], yticklabels=param_grid['C'], ax=axi)
    axi.set_xlabel('Gamma')
    axi.set_ylabel('C')

# - Search Over Spaces That are Not Grids -
# Trying different kernels for each dictionary  (Multiple Grids bc not all kernels use the same parameters)
param_grid = [{'kernel': ['rbf'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
print(f'List of grids: \n{param_grid}')


grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation: {round(grid_search.best_score_, 2)}')

results = pd.DataFrame(grid_search.cv_results_)
print(results.T)  # View different parameter settings models

# - Nested Cross Validation - (good for evaluating how well a given model works on a particular dataset)

scores = cross_val_score(GridSearchCV(SVC(), param_grid=param_grid, cv=5), iris.data, iris.target,
                         cv=5)  # CV with the model being a Grid-CV
print(f'Cross Validation Scores: {scores}')
print(f'Mean Cross Validation Score: {round(scores.mean(), 2)}')


# Implementation of the code above (two loops with stratified cv or 5 folds)

def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []
    for trainig_samples, testing_samples in outer_cv.split(X, y):  # CV
        best_params = {}
        best_score = -np.inf
        for parameters in parameter_grid:  # Grid Search
            cv_scores = []
            for inner_train, inner_test in inner_cv.split(X[trainig_samples], y[trainig_samples]):  # CV
                clf = Classifier(**parameters)
                clf.fit(X[inner_train], y[inner_train])
                score = clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = parameters
        clf = Classifier(**best_params)
        clf.fit(X[trainig_samples], y[trainig_samples])
        test_score = round(clf.score(X[testing_samples], y[testing_samples]), 3)
        outer_scores.append(test_score)
    return outer_scores

from sklearn.model_selection import StratifiedKFold, ParameterGrid
scores = nested_cv(iris.data, iris.target, StratifiedKFold(5), StratifiedKFold(5), SVC,
                   ParameterGrid(param_grid=param_grid))
print(f'Cross Validation Scores: {scores}')

