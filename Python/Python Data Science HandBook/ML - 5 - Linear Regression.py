"""
# In Depth: Linear Regression
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# --- In Depth: Linear Regression ---
# simple LR
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - + rng.randn(50)
plt.scatter(x, y)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y, color='blue')
plt.plot(xfit, yfit)

# intercept and slope
print("Model slope: ", model.coef_[0])
print("Model intercept:", model.intercept_)

# Higher Dimensions data
rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = .5 + np.dot(X, [1.5, -2, 1])

model.fit(X, y)
print(model.intercept_)
print(model.coef_)

# ~~ Basis Function Regression ~~
from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])  # Data
poly = PolynomialFeatures(degree=3, include_bias=True)  # Adding columns with higher degree on the rows
poly.fit_transform(x[:, None])

# Same as above but with a pipeline and 7 degrees
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(degree=7), LinearRegression())

# using lr to fit complicated relationships between x and y
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + .1 * rng.randn(50)

poly_model.fit(x[:, None], y)
yfit = poly_model.predict(xfit[:, None])

plt.scatter(x, y)
plt.plot(xfit, yfit)

# + Without  Pipeline +
poly = PolynomialFeatures(degree=7, include_bias=True)
new_x = poly.fit_transform(x[:, None])  # Training data transformation
new_xfit = poly.fit_transform(xfit[:, None])  # Test data transformation

model = LinearRegression().fit(new_x, y)  # fit
pred = model.predict(new_xfit)  # pred

plt.scatter(x, y); plt.plot(xfit, pred)  # plot


# ~~ Gaussian Basis Functions ~~
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, N, width_factor=2.0):
        self._N = N
        self._width_factor = width_factor
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x- y) / width
        return np.exp(-.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        self._centers_ = np.linspace(X.min(), X.max(), self._N)
        self._width_ = self._width_factor * (self._centers_[1] - self._centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self._centers_, self._width_, axis=1)


gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.xlim(0, 10)


# --- Regularization ---

# careful on too many gaussian dimensions

def basis_plot(model, title=None):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    model.fit(x[:, np.newaxis], y)
    ax[0].scatter(x, y)
    ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
    ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))

    if title:
        ax[0].set_title(title)

    ax[1].plot(model.steps[0][1]._centers_, model.steps[1][1].coef_)
    ax[1].set(xlabel='basis location', ylabel='coefficient', xlim=(0, 10))

model = make_pipeline(GaussianFeatures(30), LinearRegression())
basis_plot(model)


# ~~ Ridge Regression (L2) (penalizing sum of squares of the model coeffs) ~~

from sklearn.linear_model import Ridge
model = make_pipeline(GaussianFeatures(30), Ridge(alpha=.1))
basis_plot(model, title='Ridge Regression')

# ~~ Lasso Regularization (L1) (penalizing sum of absolute values of regression coeffs) ~~
from sklearn.linear_model import Lasso  # favors model coefficients to exactly zero
model = make_pipeline(GaussianFeatures(30), Lasso(alpha=.001))
basis_plot(model, title='Lasso Regression')


# --- Example: Predicting Bicycle Traffic ---
import pandas as pd
import os
directory = os.getcwd()
counts = pd.read_csv(str(directory) + '/Data/FremontBridge.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv(str(directory) + '/Data/2763435.csv', index_col='DATE', parse_dates=True)

# compute total daily bicycle traffic
daily = counts.resample('d').sum()  # merge index from hour to day with sum
daily['Total'] = daily.iloc[:, 1:].sum(axis=1)  # single total col
daily = daily[['Total']]  # only total col

# adding binary columns  (Dummy Vars)
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(len(days)):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)  # checking matching day for col

# adding an indicator for holidays as well
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2012', end='2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))  # Adding holiday column (NaN for non)
daily['holiday'].fillna(0, inplace=True)  # 0 for NaN on holiday col

# Dealing with hours of daylight on how they might affect how many people ride
def hours_of_daylight(date, axis=23.44, latitude=47.61):
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days ** 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()


# converting temperatures
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = .5 * (weather['TMIN'] + weather['TMAX'])

# convert precip to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)  # adding if it was a dry day

daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])

# adding a counter from day 1 for how many years have passed
daily['annual'] = (daily.index - daily.index[0]).days / 365

print(daily.head())

# Choosing the columns to use and fit a linear regression
column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday', 'daylight_hrs',
                'PRCP', 'dry day', 'Temp (C)', 'annual']
X = daily[column_names]
y = daily['Total']

# Linear Regression
model = LinearRegression(fit_intercept=False)
X = X.dropna()
y = y[X.index]
model.fit(X, y)
pred = model.predict(X)
daily['predicted'] = pd.Series(pred, index=X.index)
daily[['Total', 'predicted']].plot(alpha=.5)  # Plotting pred and actual

# ~ Looking at the coefficients and how much each feature contributes ~
params = pd.Series(model.coef_, index=X.columns)
print(params)

# ~ Computing the uncertainties using bootstrap resampling of the data ~
from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_ for _ in range(1000)], 0)
print(pd.DataFrame({'effect': params.round(0),
                    'error': err.round(0)}))
