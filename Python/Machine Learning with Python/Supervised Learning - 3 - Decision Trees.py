import os

import graphviz
import mglearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
import mglearn
from sklearn.model_selection import train_test_split

"""
####### Decision tree with cancer data  ######
"""

from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X, y = cancer['data'], cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
tree = DecisionTreeClassifier().fit(X_train, y_train)  # Fully developed tree
print(f'Training Accuracy: {tree.score(X_train, y_train)}')
print(f'Test Accuracy: {tree.score(X_test, y_test)}')


# Changing the depth (Pre-pruning)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print(f'Training Accuracy: {tree.score(X_train, y_train)}')
print(f'Test Accuracy: {tree.score(X_test, y_test)}')


# Visualizing tree
from sklearn.tree import export_graphviz
import graphviz

export_graphviz(tree, out_file='/home/beto/PycharmProjects/Machine Learning With PY/tree.dot', class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

with open('/home/beto/PycharmProjects/Machine Learning With PY/tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)



# Feature importance
print(f'Feature Importance:\n {tree.feature_importances_}')


def plotting_feature_importance(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


plotting_feature_importance(tree)


# --- Tree Regression vs Linear Regression example ---
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

ram_prices = pd.read_csv(str(os.getcwd()+'/ram_price.csv'))
plt.semilogy(ram_prices.date, ram_prices.price)  # Plot with log scale
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")


data_train = ram_prices[ram_prices.date < 2000]  # Specific data
data_test = ram_prices[ram_prices.date >= 2000]

X_train = data_train.date[:, np.newaxis]
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

X_all = ram_prices.date[:, np.newaxis]  # Predict on all data

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)


price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label="Training data")  # Plotting comparison
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()


# --- Random Forest ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='viridis')
plt.axis('equal')
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)


# Visualize each tree
fig, ax = plt.subplots(2, 3, figsize=(20, 10))

for i, (axi, tree) in enumerate(zip(ax.ravel(), forest.estimators_)):
    axi.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=axi)

mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=ax[-1, -1], alpha=.4)
ax[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

# Random Forest with 100 trees

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,  stratify=cancer.target, train_size=.75)
forest100 = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
print(f'Train Accuracy: {round(forest100.score(X_train, y_train), 3)}')
print(f'Test Accuracy: {round(forest100.score(X_test, y_test), 3)}')

# Plotting feature importance from forest
plotting_feature_importance(forest100)

# Using 1000 trees and all computer cores
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,  stratify=cancer.target, random_state=42)
forest1000 = RandomForestClassifier(n_estimators=1000, n_jobs=-1).fit(X_train, y_train)
print(f'Train Accuracy: {round(forest1000.score(X_train, y_train), 3)}')
print(f'Test Accuracy: {round(forest1000.score(X_test, y_test), 3)}')

# NOTE: max_features=sqrt(n_features) for classification && max_features=log2(n_features) for regression
# Additional; max_leaf_ns, n_estimators, max_depth and max_features

# --- Gradient Boosting --- (Sequential and extra tuning -> learning rate)
from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], stratify=cancer['target'], random_state=0)
gbrf = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
print(f'Training Accuracy: {round(gbrf.score(X_train, y_train), 3)}')
print(f'Test Accuracy: {round(gbrf.score(X_test, y_test), 3)}')

# Changing variables: max_depth and learning_rate; (how strong to correct previous mistakes) (for overfitting)
gbrf1 = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrf1.fit(X_train, y_train)
print(f'Training Accuracy: {round(gbrf1.score(X_train, y_train), 3)}')
print(f'Test Accuracy: {round(gbrf1.score(X_test, y_test), 3)}')

gbrf01 = GradientBoostingClassifier(random_state=0, learning_rate=.01)  # Changing learning rate
gbrf01.fit(X_train, y_train)
print(f'Training Accuracy: {round(gbrf01.score(X_train, y_train), 3)}')
print(f'Test Accuracy: {round(gbrf01.score(X_test, y_test), 3)}')


# Visualizing the feature importance
gbrf = GradientBoostingClassifier(max_depth=1, random_state=0)
gbrf.fit(X_train, y_train)
plotting_feature_importance(gbrf)



