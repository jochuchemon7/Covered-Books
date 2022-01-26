import mglearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
###### LINEAR MODELS ########
"""

mglearn.plots.plot_linear_regression_wave()


# make wave dataset
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lm = LinearRegression().fit(X_train, y_train)

print(f'lr.coef: {lm.coef_}')
print(f'lm.intercept_: {lm.intercept_}')

print(f'Training Square Mean Error: {lm.score(X_train, y_train)}')
print(f'Test Square Mean Error: {lm.score(X_test, y_test)}')


# Boston data set (better for many variables)
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lm = LinearRegression().fit(X_train, y_train)

print(f'Training Square Mean Error: {lm.score(X_train, y_train)}')
print(f'Test Square Mean Error: {lm.score(X_test, y_test)}')

# --- Ridge Regression  ---- (coeffs close to 0)  (high alpha -> high regularization -> closer to 0)
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print(f'Training Score: {ridge.score(X_train, y_train)}')
print(f'Test Score: {ridge.score(X_test, y_test)}')

# different alpha variable value (higher alpha more generalization; less accurate on training)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print(f'Ridge10 Training Score: {round(ridge10.score(X_train, y_train), 3)}')
print(f'Ridge10 Test Score: {round(ridge10.score(X_test, y_test), 3)}')

ridge01 = Ridge(alpha=.1).fit(X_train, y_train)
print(f'Ridge01 Training Score: {round(ridge01.score(X_train, y_train), 3)}')
print(f'Ridge01 Test Score: {round(ridge01.score(X_test, y_test), 3)}')

# View the coefficient magnitudes based on the different alpha values
plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lm.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lm.coef_))
plt.ylim(-25, 25)
plt.legend()

mglearn.plots.plot_ridge_n_samples()

# NOTE: Increasing alpha forces coeff closer to 0 (More regularization and More generalization)

# --- Lasso Regression --- (some features are ignore bc some coeffs are 0)

from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
print(f'Training Score: {round(lasso.score(X_train, y_train), 2)}')
print(f'Test Scores: {round(lasso.score(X_test, y_test), 2)}')
print(f'Number of features used: {np.sum(lasso.coef_ != 0)}')


# different alpha values and max_iteration for convergence (less alpha for less restriction)
lasso001 = Lasso(alpha=.01, max_iter=100000).fit(X_train, y_train)
print(f'Training Score: {round(lasso001.score(X_train, y_train), 2)}')
print(f'Test Scores: {round(lasso001.score(X_test, y_test), 2)}')
print(f'Number of features used: {np.sum(lasso001.coef_ != 0)}')

# Alpha too low (Supper less restrictive) results in similar to LinearRegression
lasso00001 = Lasso(alpha=.0001, max_iter=100000).fit(X_train, y_train)
print(f'Training Score: {round(lasso00001.score(X_train, y_train), 2)}')
print(f'Test Score: {round(lasso00001.score(X_test, y_test), 2)}')
print(f'Features used: {np.sum(lasso00001.coef_ != 0)}')

# --- ElasticNet Regression --- (Mixture of both Lasso and Ridge)
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=.1, l1_ratio=.01).fit(X_train, y_train)
print(f'Training Score: {round(elastic_net.score(X_train, y_train), 2)}')
print(f'Test Score: {round(elastic_net.score(X_test, y_test), 2)}')


# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

for model, axi in zip([LinearSVC(), LogisticRegression()], ax.flat):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=axi, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=axi)
    axi.set_title("{}".format(clf.__class__.__name__))
    axi.set_xlabel("Feature 0")
    axi.set_ylabel("Feature 1")
ax[0].legend()

# logistic svm with diff c values for L2 regularization
mglearn.plots.plot_linear_svc_regularization()

# Linear Logistic with breast cancer

cancer = load_breast_cancer()
X, y = cancer['data'], cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
print(f'Train Score: {round(logreg.score(X_train, y_train), 3)}')
print(f'Test Score: {round(logreg.score(X_test, y_test), 3)}')

# With different C value (high C is less regularize and high C is more regularized)
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print(f'Train Score: {round(logreg100.score(X_train, y_train), 3)}')
print(f'Test Score: {round(logreg100.score(X_test, y_test), 3)}')


logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print(f'Train Score: {round(logreg001.score(X_train, y_train), 3)}')
print(f'Test Score: {round(logreg001.score(X_test, y_test), 3)}')

# Comparing coefficient magnitudes
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()


# Using L1 regularization instead

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):

    lr_l1 = LogisticRegression(C=C, penalty="l1", solver='liblinear', max_iter=1e7).fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)



# Multiclass Classification

from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])

linear_svm = LinearSVC().fit(X, y)
print(f'Coefficient shape: {linear_svm.coef_.shape}')
print(f'Intercept shape: {linear_svm.intercept_.shape}')


# Visualize the coeff and intercept
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)

for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))


# Visualize with triangle in the middle
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)

for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# Bernoulli example
X = np.array([[0, 1, 0, 1],
             [1, 0, 1, 1],
             [0, 0, 0, 1],
             [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
    counts[label] = X[y == label].sum(axis=0)
print(f'Feature Counts: {counts}')
