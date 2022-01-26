"""
# Feature Engineering
"""

# --- Categorical Features ---
data = [
 {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
 {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
 {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
 {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

# ~~ Doing a One-Hot-Encoding ~~
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
vec.fit_transform(data)
print(vec.get_feature_names())  # inspecting the feature names

# doing a sparse output
vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)


# --- Text Features ---
sample = ['problem of evil', 'evil queen', 'horizon problem']

# ~~ Word Counts (count of word occurrences) ~~
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
vec = CountVectorizer()
X = vec.fit_transform(sample)
X
print(vec.get_feature_names())

pd.DataFrame(X.toarray(), columns=vec.get_feature_names())  # DF of the words (col) and (1/0) on count

# Term Frequency - Inverse Document Frequency (TF-IDF) (floating values instead of 1/0)
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


# --- Derived Features ---
import matplotlib.pyplot as plt
import numpy as np

# data cannot be describe by a straight line
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)

# still using LinearRegression
from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(X, y)
plt.plot(X, yfit)

# we can add polynomial features to the data
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)  # from 1 degree to 3 degrees (3cols) raise to that power col number
print(X2)  # x, x^2, x^3

# computing linear regression
model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit)


# --- Imputation of Missing Data ---
from numpy import nan
X = np.array([[nan, 0, 3],
              [3, 7, 9],
              [3, 5, 2],
              [4, nan, 6],
              [8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])

# simple mean mean on column
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')  # mean over cols
X2 = imp.fit_transform(X)
print(X2)

# feeding the transformed data
model = LinearRegression().fit(X2, y)
model.predict(X2)


# --- Feature Pipelines ---
from sklearn.pipeline import make_pipeline
model = make_pipeline(SimpleImputer(strategy='mean'),  # mean on columns
                      PolynomialFeatures(degree=2),  # adding columns with ^2 degree
                      LinearRegression())  # LinearRegression
model.fit(X, y)
print(y)  # target
print(model.predict(X))  # predicted y

