"""
# Univariate Nonlinear Transformations AND Automatic, Model-Based and Iterative Feature Selection
"""
import matplotlib.pyplot as plt
import numpy as np


# --- Univariate Nonlinear Transformations ---
# NOTE: Most models work best when each feature is loosely Gaussian distributed

# ~~ Using a 'Count Data' (e.i: 'How often did user A log in') ~~
rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

# View first 10 entries of first feature (All integer and positive)
print(X[:10, ])
print(f'X.shape: {X.shape}')

print(f'Number of Feature Appearances: \n{np.bincount(X[:, 0])}')  # From bins 0-140 from np.unique

# ~~ Visualize the Counts ~~
bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color='k')
plt.ylabel('Number of Appearances')
plt.xlabel('Value')

# All three features
fig, ax = plt.subplots(1, 3, figsize=(12, 7))
for i, axi in enumerate(ax.flat):
    new_bins = np.bincount(X[:, i])
    axi.bar(range(len(new_bins)), new_bins)
    axi.set_title(f'Feature: {i}')

# NOTE: This kind of distribution of many small and few large values is very common in practice

# ~~ Linear Model ~~ (Not so good handled) ~~
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print(f'Test Score: {round(score, 3)}')

# ~ Applying log transformation (add +1 bc the log of 0 is not defined) ~ (Trying to resemble Gaussian pdf)
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

# ~ Visualizing the transformation ~
plt.hist(np.log(X_train_log[:, 0] + 1), bins=25, color='gray', edgecolor='k')
plt.ylabel('Number of Appearances')
plt.xlabel('Value')

# ~ Ridge Model on Log data ~
score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print(f'Test Score: {round(score, 3)}')

# NOTE: All features had the same properties (Not always the case)

# --- Automatic Feature Selection ---

# ~~~ Univariate Statistics ~~~ (Whether there is a statistical significant relationship of feature and target)
# Known as Analysis of Variable (ANOVA) (All features are treated independently NO interaction)

# ~~ Using SelectPercentile and f_classif on cancer data ~~
from sklearn.feature_selection import SelectPercentile
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# Adding Noise
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, train_size=.5,
                                                    random_state=0)
select = SelectPercentile(percentile=50)  # Default score_func is 'f_classif' also 'f_regression' available
select.fit(X_train, y_train)  # selects 50% of features
X_train_selected = select.transform(X_train)

print(f'X_train.shape: {X_train.shape}')
print(f'X_train_selected: {X_train_selected.shape}')  # From 80 to 40 features

# ~ Visualizing Feature Selection ~
mask = select.get_support()
print(mask)
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('Sample Index')

# ~ Using Logistic Regression ~
from sklearn.linear_model import LogisticRegression

X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f'Score with all features: {round(lr.score(X_test, y_test), 3)}')
lr.fit(X_train_selected, y_train)
print(f'Score With Only Selected Features: {round(lr.score(X_test_selected, y_test), 3)}')

# NOTE: Performance was not change even thou 50 features were removed

# --- Model-Based Feature Selection --- (Considers all features at once, so it captures interactions)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Tree base -> feature_importance & Linear models -> coefficients
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
# Features are selected if greater than threshold (median = about half)

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print(f'X_train.shape: {X_train.shape}')
print(f'X_train_l1.shape: {X_train_l1.shape}')

# ~ Visualizing the feature selection ~
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.ylabel('Sample Index')

# ~ Logistic Regression ~
X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print(f'Test Score: {round(score, 3)}')


# --- Iterative Feature Selection --- (Series of models are built, with varying number of features)
# NOTE: 1.start with no features adding one by one or 2.All at once and removing

# ~ Using Recursive Feature Elimination (RFE) ~
# (Start with all, run model remove least important, run model ... until sentinel)
from sklearn.feature_selection import RFE

select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)
select.fit(X_train, y_train)

# ~ Visualize Selected Features ~
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('Sample Index')

# ~ Logistic Regression Test Score ~
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print(f'Test Score: {round(score, 3)}')


