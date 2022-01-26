"""
# Utilizing Expert Knowledge
"""

# --- Utilizing Expert Knowledge --- (Predicting for a given time and day how many people will rent a bike)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mglearn

# For time code letters: https://www.programiz.com/python-programming/datetime/strftime

citibike = mglearn.datasets.load_citibike()
print(f'Citi Bike data: \n{citibike.head()}')

# ~~ Visualizing Rentals For the Whole Month ~~
plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha='left')
plt.plot(citibike, linewidth=1)
plt.xlabel('Date')
plt.ylabel('Rentals')


# NOTE: Training data would be from the first 23 days and test on the remaining (Cannot Shuffle Data)

# Single integer feature as our data representation
X = citibike.index.strftime("%s").astype('int')  # Representing as POSIX timestamp ('%s')
X = X.values.reshape(-1, 1)  # represented in seconds
y = citibike.values


# ~~ Defining a Function to Split the data into Train and Test Sets, Build the Model and Visualize the Result ~~

n_train = 184  # First 184 points for training

def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)

    print(f'Test-Set R^2: {round(regressor.score(X_test, y_test), 3)}')
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90, ha='left')

    plt.plot(range(n_train), y_train, label='Train')
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label='Test')

    plt.plot(range(n_train), y_pred_train, '--', label='Prediction Train')
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label='Prediction Test')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Rentals')


# ~~ Using Random Forest Regressor on POSIX time ~~
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
eval_on_features(X, y, regressor)



# NOTE: It does not work bc test data is outside the range of the feature values
# it predicts the target value of the closest point in the training set


# ~~ We are going to use the time of day and the day of the week as features instead ~~
X_hour = citibike.index.strftime('%-H').astype('int')  # By hour
X_hour = X_hour.values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)

day_of_week = citibike.index.strftime('%w').astype('int').values.reshape(-1, 1)
X_hour_week = np.hstack([X_hour, day_of_week])  # By Hour and day of week
eval_on_features(X_hour_week, y, regressor)  # R^2: .841

# ~~ Using Linear Regression with the modified X_hour_week data ~~
from sklearn.linear_model import LinearRegression
hour = citibike.index.strftime('%-H').astype('int').values.reshape(-1, 1)
day = citibike.index.strftime('%w').astype('int').values.reshape(-1, 1)
merged_data = np.hstack([hour, day])
eval_on_features(merged_data, y, LinearRegression())

# NOTE: Really bad bc we encoded day of the week and time using integers, which are interpreted as categorical variables

# ~~ Using OneHotEncoding to transform the categorical variables ~~
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
encoder = OneHotEncoder()
X_hour_week_onehot = encoder.fit_transform(X_hour_week).toarray()
eval_on_features(X_hour_week_onehot, y, LinearRegression())
eval_on_features(X_hour_week_onehot, y, Ridge())

# ~~ Using Interaction Features, allow the model to learn one coefficient for each combination of day and time of day ~~
# Using PolynomialFeatures on the OneHotEncoded hour and week day data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)  # One coefficient for each day and time

# ~ Plotting the coefficients learned by the model ~ (Not Possible with Random Forest)
hour = ['%02d:00' % i for i in range(0, 24, 3)]
day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
features = day + hour

# We can only keep the coefficients with non-zero values
features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]


# Plotting the coeffient values of the features (product of hour and day) to view importance coefficients
plt.figure(figsize=(15, 15))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel('Feature Name')
plt.ylabel('Feature Magnitude')

