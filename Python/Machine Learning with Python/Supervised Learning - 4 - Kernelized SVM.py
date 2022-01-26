import os
from sklearn.datasets import make_blobs
import graphviz
import mglearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import mglearn
from sklearn.model_selection import train_test_split

# Non linear separable data points
X, y = make_blobs(centers=4, random_state=8)
y = y % 2
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# Linear SVC
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)
mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# Adding third feature and plotting it
from mpl_toolkits.mplot3d import Axes3D, axes3d

X_new = np.hstack([X, X[:, 1:] ** 2])  # third feature as the square of the second feature
figure = plt.figure()
ax = Axes3D(fig=figure, elev=-152, azim=-26)
mask = y == 0

ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)

ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")

# We can now add a hyperplane to separate data points

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]

figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)

ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)

ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature0 ** 2")


# In a 2d plot
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# --- Kernel SVM ---
from sklearn.svm import SVC

X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)  # krbf(x1, x2) = exp (ɣǁx1- x2ǁ2)  & gamma is how close svPoints & C is regularization (importance value of each support point)
sv = svm.support_vectors_
sv_labels = svm.dual_coef_.ravel() > 0  # class label of the support vectors

mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')


# Vary the parameters
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for axi, C in zip(ax, [-1, 0, 3]):
    for a, gamma in zip(axi, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
ax[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"], ncol=4, loc=(.9, 1.2))


# Gaussian SVM
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], stratify=cancer['target'], random_state=0)
svc = SVC()
svc.fit(X_train, y_train)
print(f'Training Accuracy: {round(svc.score(X_train, y_train), 3)}')
print(f'Test Accuracy: {round(svc.score(X_test, y_test), 3)}')

# Plot the min and max values of each feature (They all have different orders of magnitude - No good for SVC)
plt.plot(X_train.min(axis=0), '^', label='min')
plt.plot(X_train.max(axis=0), 'o', label='max')
plt.legend(loc=4)
plt.xlabel('feature index')
plt.ylabel('feature magnitude')
plt.yscale('log')


# Preprocessing svm data (Normalizing to solve the different orders of magnitude problem)

min_on_training = X_train.min(axis=0)  # min on training
range_on_training = (X_train - min_on_training).max(axis=0)  # max of (training - min_training)
X_train_scaled = (X_train - min_on_training) / range_on_training  # (training - min_training) / range
print(f'Minimum for each feature: {X_train_scaled.min(axis=0)}')
print(f'Maximum for each feature: {X_train_scaled.max(axis=0)}')

# Same for test data
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC()
svc.fit(X_train_scaled, y_train)
print(f'Train Accuracy: {round(svc.score(X_train_scaled, y_train), 3)}')
print(f'Test Accuracy: {round(svc.score(X_test_scaled, y_test), 3)}')

# Changing C=100
svc100 = SVC(C=100)
svc100.fit(X_train_scaled, y_train)
print(f'Train Accuracy: {round(svc100.score(X_train_scaled, y_train), 3)}')
print(f'Test Accuracy: {round(svc100.score(X_test_scaled, y_test), 3)}')


# normalized = (x-min(x))/(max(x)-min(x))  ->  (Simpler method)

normalized_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
normalized_test = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))


# C and gamma correlate