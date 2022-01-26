"""
Binning, Discretization, Linear Models, and Trees
"""

# --- Binning, Discretization, Linear Models, and Trees ---

# Linear Regression vs Decision Tree on single input feature data
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)  # False Endpoint = non-inclusive (-3 and 3)

reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label='Decision Tree')

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label='Linear Regression')

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression Output')
plt.xlabel('Input Feature')
plt.legend(loc='best')


# ~ We can do binning place each point into a bin that it would fall ~
bins = np.linspace(-3, 3, 11)
print(f'Bins: {bins}')

# Digitize (record for each data point which bin it fall into)
which_bin = np.digitize(X, bins)
print(f'Data Points:\n {X[:5]}')
print(f'Bin Membership for Data Point:\n {which_bin[:5]}')

# We do a OneHotEncoding on the bin data (Resulting on 10 features)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print(X_binned[:5])
print(f'X_binned.shape: {X_binned.shape}')

# ~~ Building the LinearRegression and Tree Models again with the binned data ~~
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
bins = np.linspace(-3, 3, 11)

lined_binned = encoder.transform(np.digitize(line, bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(lined_binned), label='Linear Regression Binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(lined_binned), label='Decision Tree Binned')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc='best')
plt.ylabel('Input Feature')
plt.xlabel('Regression Output')

# NOTE: Binning works best for linear regression models rather than trees

# --- Interactions and Polynomials --- (adding interaction and polynomial features to the  original data)

# ~~~ Interaction Features ~~~

# Merge the original data with the binned one and forming a 11 feature data
X_combined = np.hstack([X, X_binned])
print(f'X_combined.shape: {X_combined.shape}')

reg = LinearRegression().fit(X_combined, y)

line_combined = np.hstack([line, lined_binned])
plt.plot(line, reg.predict(line_combined), label='Linear Regression Combined')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')

plt.legend(loc='best')
plt.ylabel('Regression Output')
plt.xlabel('Input Feature')
plt.plot(X[:, 0], y, 'o', c='k')


# We want each bin to have their own separate slope, so we add an interaction which indicates which bin a data point belongs
X_product = np.hstack([X_binned, X * X_binned])
print(f'X_product.shape: {X_product.shape}')  # original feature within the bin and zero everywhere else

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([lined_binned, line * lined_binned])
plt.plot(line, reg.predict(line_product), label='Linear Regression Product')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')
plt.plot(X[:, 0], y, 'o', c='k')
plt.legend(loc='best')
plt.ylabel('Regression Output')
plt.xlabel('Input Feature')


# NOTE: Now each bin has it own offset and slope in the model; Binning is a way to expand a continuous feature

# ~~~ Polynomial Features ~~~ (Raising the features to a polynomial degree)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=10, include_bias=False)  # True on bias adds a feature that's constantly 1
poly.fit(X)
X_poly = poly.transform(X)

print(f'X_poly.shape: {X_poly.shape}')

print(f'Entries of X:\n{X[:5]}')
print(f'Entries of X_poly: \n{X_poly[:5]}')

# Getting the semantics of the features by calling the get_feature_names method
print(f'Polynomial Feature Names: \n{poly.get_feature_names()}')

# ~~ Classical model of 'polynomial regression' ~~

# Linear Regression with tenth-degree polynomial features
reg = LinearRegression().fit(X_poly, y)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='Polynomial Linear Regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.legend(loc='best')
plt.ylabel('Regression Output')
plt.xlabel('Input Feature')


# ~~ Comparison; kernel SVM on original data without any transformation ~~
from sklearn.svm import SVR
for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label='SVR Gamma: {}'.format(gamma))
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression Output')
plt.xlabel('Input Feature')
plt.legend(loc='best')


# --- Using the Boston Housing Data Set ---
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

# Rescale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Extract polynomial features and interactions up to a degree of 2
poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print(f'X_train.shape: {X_train.shape}')
print(f'X_train_poly.shape: {X_train_poly.shape}')  # From 13 features to 105 (all possible interactions between two different original features and square of each original feature)
print(f'Polynomial Feature Names: {poly.get_feature_names()}')

# NOTE: features -> 1, x0-x12, x0^2, x0x1-x0x12, x1^2, x1x2-x1x12, x2^2, ... x12^2


# ~~ Performance using Ridge Regression without interactions ~~
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train_scaled, y_train)
print(f'Score Without Interactions: {round(ridge.score(X_test_scaled, y_test), 3)}')
ridge = Ridge().fit(X_train_poly, y_train)
print(f'Score With Interactions: {round(ridge.score(X_test_poly, y_test), 3)}')


# ~~ Using a more complex Random Forest Regressor ~~
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print(f'Score Without Interaction: {round(rf.score(X_test_scaled, y_test), 3)}')
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print(f'Score With Interaction: {round(rf.score(X_test_poly, y_test), 3)}')



