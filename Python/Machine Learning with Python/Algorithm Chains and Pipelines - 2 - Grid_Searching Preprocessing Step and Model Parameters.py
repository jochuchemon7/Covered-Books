"""
# Grid Searching Preprocessing Step and Model Parameters
"""

# - Scaling Data, Computing Polynomial Features and Ridge Regression on Boston Data Set

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

# NOTE: We can adjust the parameters of the preprocessing using the outcome of the supervised task

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# Making our pipeline, param_grid and GridSearchV

pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

param_grid = {'polynomialfeatures__degree': [1, 2, 3],  # Parameter of the Polynomial Features
              'ridge__alpha': [.001, .01, .1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)  # Running Grid Search
grid.fit(X_train, y_train)

# Visualizing the outcome using a heat-map
import matplotlib.pyplot as plt

plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1), cmap='viridis',
            vmin=0)  # reshape to (3, -1) bc of the 3 values in the degree parameter
plt.xlabel('ridge__alpha')
plt.ylabel('polynomialfeatures__degree')
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])), param_grid['polynomialfeatures__degree'])
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
plt.colorbar()
plt.title('Mean Test Score')

print(f'Best Parameters: {grid.best_params_}')
print(f'Test Set Score: {round(grid.score(X_test, y_test), 3)}')

# Running it again without polynomial features for comparison
param_grid = {'ridge__alpha': [.001, .01, .1, 1, 10, 100]}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print(f'Test Score Without Polynomial Features: {round(grid.score(X_test, y_test), 3)}')

# NOTE: Searching over preprocessing parameters together with model parameters is a very powerful strategy

# --- Grid-Searching Which Model To Use ---

# - Comparing SVC and RamForest on the cancer data set -
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

cancer = load_breast_cancer()
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

from sklearn.ensemble import RandomForestClassifier

# Making use of the list of search grids for the two different classifiers and unique parameters

param_grid = [{'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
               'classifier__gamma': [.001, .01, .1, 1, 10, 100],
               'classifier__C': [.001, .01, .1, 1, 10, 100]},
              {'classifier': [RandomForestClassifier(n_estimators=100)], 'preprocessing': [None],
               'classifier__max_features': [1, 2, 3]}]

# NOTE: The classifier name step would take SVC() and RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print(f'Best Params: {grid.best_params_}')
print(f'Best Cross Validation Score: {round(grid.best_score_ , 3)}')
print(f'Test Set Score: {round(grid.score(X_test, y_test) , 3)}')

