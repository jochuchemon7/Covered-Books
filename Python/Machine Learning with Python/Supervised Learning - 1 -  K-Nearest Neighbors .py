import mglearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import mglearn

# Data sets info
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

cancer.keys()
type(cancer['data'])

print("Sample counts per class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})  )

boston = load_boston()
print('boston shape: ', boston.data.shape)

# Using books helper function
mglearn.plots.plot_knn_classification(n_neighbors=4)


# K-NN

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
p_hat = clf.predict(X_test)
print(p_hat)
print(y_test)
clf.score(X_test, y_test)


fig, ax = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, axi in zip([1, 2, 3], ax.flat):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=axi, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=axi)
    axi.set_title("{} neighbor(s)".format(n_neighbors))
    axi.set_xlabel("feature 0")
    axi.set_ylabel("feature 1")
ax[0].legend(loc=3)


# K-NN with breast cancer data set

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)

train_accuracy = []
test_accuracy = []
neighbors_setting = range(1, 11)
for neighbors in neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors=neighbors).fit(X_train, y_train)
    train_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_setting, train_accuracy, label='training accuracy')
plt.plot(neighbors_setting, test_accuracy, label='test accuracy')
plt.ylabel('accuracy')
plt.xlabel('K Neighbor')
plt.legend()


# K neighbors in regression

mglearn.plots.plot_knn_regression(1)
mglearn.plots.plot_knn_regression(2)
mglearn.plots.plot_knn_regression(3)


# K Neighbors Regression with make_wave data set

from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
print(f'Predictions: {reg.predict(X_test)}')
print(f'R Score Value: {round(reg.score(X_test, y_test), 2)}')

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neighbors, axi in zip([1, 3, 5], ax):
    reg = KNeighborsRegressor(n_neighbors).fit(X_train, y_train)
    axi.plot(line, reg.predict(line))
    axi.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    axi.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    axi.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
        n_neighbors, reg.score(X_train, y_train),reg.score(X_test, y_test)))
    axi.set_xlabel("Feature")
    axi.set_ylabel("Target")
ax[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
