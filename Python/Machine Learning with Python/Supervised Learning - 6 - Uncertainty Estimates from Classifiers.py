from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mglearn


# --- Uncertainty Estimates --- (decision_function and predict_proba)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles

X, y = make_circles(noise=.25, factor=.5, random_state=0)
y_named = np.array(["blue", "red"])[y]  # Rename classes for illustration purposes

X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

# ~ Decision Function ~
print(f'Test Shape: {X_test.shape}')
print(f'Decision function shape: {gbrt.decision_function(X_test).shape}')  # Returns floating point number for each sample

# Positive value preference for positive class and vice-versa
print(f'Decision function: {gbrt.decision_function(X_test)[:6]}')

# Compare prediction with sign of decision function
print(f'Threshold decision function: \n{gbrt.decision_function(X_test) > 0} ')
print(f'Predictions: \n{gbrt.predict(X_test)}')

# NOTE: (Binary Classification the 'negative' class is always the first entry)
print(gbrt.classes_)

# Comparing prediction with decision function and converting ints to class value
greater_than_zero = (gbrt.decision_function(X_test) > 0).astype(int)
pred = gbrt.classes_[greater_than_zero]
print(f'pred is equal to predictions: {np.all(pred == gbrt.predict(X_test))}')


# Range of decision function values
decision_function = gbrt.decision_function(X_test)
print(f'Decision function Max: {round(decision_function.max(), 3)}  Minumum: {round(decision_function.min(), 3)}')


# Plotting Decision Boundary vs Decision Function points

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

mglearn.tools.plot_2d_separator(gbrt, X, ax=ax[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=ax[1], alpha=.4, cm=mglearn.ReBl)

for axi in ax.flat:

    # plot training and test points
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=axi)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=axi)
    axi.set_xlabel("Feature 0")
    axi.set_ylabel("Feature 1")

cbar = plt.colorbar(scores_image, ax=ax.tolist())
ax[0].legend(["Test class 0", "Test class 1", "Train class 0", "Train class 1"], ncol=4, loc=(.1, 1.1))


# --- Predicting Probabilities ---
print(f'Shape of probabilities: {gbrt.predict_proba(X_test).shape}')
print(f'Predicted Probabilities: \n{gbrt.predict_proba(X_test)[:5, ]}')  # Class 0 and 1

# Plotting Decision Boundary vs predict_proba function
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, fill=True, ax=ax[0], alpha=.4, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=ax[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')

for axi in ax.flat:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, ax=axi, markers='^')
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=axi, markers='o')
    axi.set_xlabel('Feature 0')
    axi.set_ylabel('Feature 1')
cbar = plt.colorbar(mappable=scores_image, ax=ax.tolist())
ax[0].legend(['Test Class 0', 'Test Class 1', 'Train Class 0', 'Train Class 1'], ncol=4, loc=(.1, 1.1))


fig, ax = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, fill=True, ax=ax[0], alpha=.4, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=ax[1], alpha=.4, cm=mglearn.ReBl, function='predict_proba')

for axi in ax.flat:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, ax=axi, markers='^')
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=axi, markers='o')
    axi.set_xlabel('Feature 0')
    axi.set_ylabel('Feature 1')
cbar = plt.colorbar(scores_image, ax=ax.tolist())
ax[0].legend(['Test class 0', 'Test class 1'], ncol=2, loc=(.1, 1.1))


# --- Uncertainty In Multiclass Classification ---

from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=.75)

gbc = GradientBoostingClassifier(learning_rate=.01, random_state=0)
gbc.fit(X_train, y_train)

print(f'Decision Function Shape: {gbc.decision_function(X_test).shape}')   # n-samples, n-targets
print(f'Decision Function: \n{gbc.decision_function(X_test)[:5, ]}')

# Max decision function value for each data point on given class (col)   compared to predictions
print(f'Argmax of decision function:\n {np.argmax(gbc.decision_function(X_test), axis=1)}')
print(f'Predictions: \n{gbc.predict(X_test)}')

# Probabilities
print(f'Predicted Probabilities: \n{gbc.predict_proba(X_test)[:6]}')
print((f'Sums: \n{gbc.predict_proba(X_test)[:6].sum(axis=1)}'))  # Sum to 1 for rows

# Compare max probability with predictions
print(f'Argmax Predicted Probabilities: \n{np.argmax(gbc.predict_proba(X_test[:-1]), axis=1)}')
print(f'Predictions: \n{gbc.predict(X_test[:-1])}')

# - decision function from Logistic Regression
from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()
named_target = iris.target_names[y_train]
lm.fit(X_train, named_target)
print(f'Unique classes in the training data: {lm.classes_}')
print(f'Predictions: \n{lm.predict(X_test[:7])}')
argmax_dec_fun = np.argmax(lm.decision_function(X_test), axis=1)
print(f'Argmax of decision function: {argmax_dec_fun[:7]}')
print(f'Combine With classes: \n{named_target[argmax_dec_fun[:7, ]]}')

