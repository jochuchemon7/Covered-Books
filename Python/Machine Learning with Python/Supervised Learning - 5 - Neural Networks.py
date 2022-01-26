import mglearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from IPython.display import display

mglearn.plots.plot_logistic_regression_graph()

display(mglearn.plots.plot_single_hidden_layer_graph())

# Non linear functions before y_hat value in MLP

line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label="tanh")
plt.plot(line, np.maximum(line, 0), label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")

# Example with make_moons dataset

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons


X, y = make_moons(n_samples=100, noise=.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# Reducing the number of hidden units
# (number of hidden units is the number of straight line segments)
mlp10 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[10], random_state=0)
mlp10.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp10, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# [10, 10] hidden_layer_size shape (2 hidden layers, with 10 units each)

mlp1010 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[10, 10], random_state=0)
mlp1010.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp1010, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# Changing activation function from relu to tanh

mlp_tanh = MLPClassifier(solver='lbfgs', activation='tanh', random_state=0, hidden_layer_sizes=[10, 10])
mlp_tanh.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp_tanh, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# Plotting with 4 different alpha values and two hidden layers of 10 and 100 units

fig, ax = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(ax, [10, 100]):
    print(axx, ' ', n_hidden_nodes)
    for axi, alpha in zip(axx, [.0001, .01, .1, 1]):
        mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
        mlp.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=axi)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=axi)
        axi.set_title(f'n_hidden = [{n_hidden_nodes}, {n_hidden_nodes}] \n alpha = {alpha}')

# MLP with different random_states [0-8)
fig, ax = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    mlp = MLPClassifier(solver='lbfgs', random_state=i, hidden_layer_sizes=[100, 100])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=axi)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=axi)


# MLP on breast cancer data set
cancer = load_breast_cancer()
print(f'Cancer data per feature maxima: \n{cancer.data.max(axis=0)}' )


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, train_size=.75)
mlp = MLPClassifier(random_state=42).fit(X_train, y_train)
print(f'Training Accuracy: {round(mlp.score(X_train, y_train), 3)}')
print(f'Test Accuracy: {round(mlp.score(X_test, y_test), 3)}')

# Again but with normalized data (using standard scaler) so the data is scaled for all features

mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0).fit(X_train_scaled, y_train)
print(f'Training Accuracy: {round(mlp.score(X_train_scaled, y_train), 3)}')
print(f'Test Accuracy: {round(mlp.score(X_test_scaled, y_test), 3)}')

# Increasing max_iteration
mlp1000 = MLPClassifier(random_state=0, max_iter=1000).fit(X_train_scaled, y_train)
print(f'Training Accuracy: {round(mlp1000.score(X_train_scaled, y_train), 3)}')
print(f'Test Accuracy: {round(mlp1000.score(X_test_scaled, y_test), 3)}')

# Increasing Alpha for more regularization to deal with overfitting
mlp1000 = MLPClassifier(random_state=0, max_iter=1000, alpha=1).fit(X_train_scaled, y_train)
print(f'Training Accuracy: {round(mlp1000.score(X_train_scaled, y_train), 3)}')
print(f'Test Accuracy: {round(mlp1000.score(X_test_scaled, y_test), 3)}')


# Plot the feature importance
plt.figure(figsize=(20, 5))
plt.imshow(mlp1000.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()


