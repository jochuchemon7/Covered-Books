"""
# HyperParameters and Model Validation
"""
from sklearn.datasets import load_iris
iris = load_iris()

# ~~ Model Validation ~~
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0,
                                                    train_size=.5)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

# ~~ Model Validation Via Cross-Validation ~~
test_pred = model.fit(X_train, y_train).predict(X_test)
train_pred = model.fit(X_test, y_test).predict(X_train)

print('Pred on Test: ', accuracy_score(y_test, test_pred))
print('Pred on Train: ', accuracy_score(y_train, train_pred))

# ~ Cross Validation ~
from sklearn.model_selection import cross_val_score
print(cross_val_score(model, iris.data, iris.target, cv=5))  # across 5 subsets

# Training on all points but one
from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, iris.data, iris.target, cv=LeaveOneOut())
print(scores)
print(scores.mean())


# --- Selecting the Best Model ---

# ~~ Validation Curves in Scikit-Learn ~~
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + .1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


X, y = make_data(40)

# ~ Visualize the Data ~
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

x_test = np.linspace(-.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(x_test)
    plt.plot(x_test.ravel(), y_test, label=f"degree={degree}")
plt.xlim(-.1, 1)
plt.ylim(-2, 12)
plt.legend(loc='best')

# ~ Visualizing the validation curve with Sklearn.validation_curve ~
from sklearn.model_selection import validation_curve
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y, 'polynomialfeatures__degree',
                                          degree, cv=7)

# Plotting
plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')

# from plot we use 3rd degree
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(x_test)
plt.plot(x_test.ravel(), y_test)
plt.axis(lim)


# --- Learning Curves ---
# new data with factor of five more points
X2, y2 = make_data(200)
plt.scatter(X2, y2)

# ~~ Plot the Validation Curve for Larger Data Set ~~
degree = np.arange(21)
train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2, 'polynomialfeatures__degree',
                                            degree, cv=7)

# plotting
plt.plot(degree, np.median(train_score2, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')
plt.plot(degree, np.median(train_score, 1), color='blue', alpha=.3, linestyle='--')
plt.plot(degree, np.median(val_score, 1), color='red', alpha=.3, linestyle='--')
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')


#Learning Curves in Scikit-Learn
from sklearn.model_selection import learning_curve

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=.0625, right=.95, wspace=.1)

for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree), X, y, cv=7,
                                         train_sizes=np.linspace(.3, 1, 25))
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray', linestyles='--')
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('Score')
    ax[i].set_title(f'Degree={degree}', size=14)
    ax[i].legend(loc='best')


# --- Validation in Practice: Grid Search ---
# Using Grid Search for optimal polynomial model
from sklearn.model_selection import GridSearchCV

# 3-Dimensional grid of model features  (passing all values for attributes for possible combinations)
param_grid = {'polynomialfeatures__degree': np.arange(21),
              'linearregression__fit_intercept': [True, False],
              'linearregression__normalize': [True, False]}
grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
grid.fit(X, y)
print(grid.best_params_)  # Finding best parameters

# Using best parameters
model = grid.best_estimator_
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(x_test)
plt.plot(x_test.ravel(), y_test)
plt.axis(lim)


