"""
# Working With Time Series
"""
import datetime
import time
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import parser
import pandas as pd
import numpy as np
import os

# = Naive Python dates and times =
print(datetime(year=2021, month=10, day=17))  # datetime type

# or with date-util
date = parser.parse('17th of October 2021')
print(date)

# printing day of the week
print(date.strftime('%A'))

# = NumPy's datetime64 =
date = np.array(['2021-10-17'], dtype=np.datetime64)
print(date); print(date.dtype)

# vectorized operations
print(date + np.arange(12))

# Base Times
np.datetime64('2021-10-17')  # Day-Based-Time
np.datetime64('2021-10-17 12:00')  # Minute-Based-Time
np.datetime64('2021-10-17 12:59:59.50', 'ns')  # Nanosecond-Based-Time

# = Dates and times in Pandas
date = pd.to_datetime('4th of july 2015')
print(date)
print(date.strftime('%A'))  # day of datetime

# numpy vectorized operations (add 12 days)
print(date + pd.to_timedelta(np.arange(12), 'D'))


# = Pandas Time Series: indexing by time =
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04', '2015-07-04', '2015-08-04'])
print(index)
data = pd.Series([0, 1, 2, 3], index=index)
print(data)

# Slice by datetime index
print(data['2014-07-04':'2015-07-04'])

# special indexing by year
print(data['2015'])


# --- Pandas Time Series Data Structures ---
# series of dates yields a DataTimeIndex instead of Timestamp
dates = pd.to_datetime([datetime(2015, 7, 3), '4th of july 2015', '2015-jul-6', '07-07-2015', '20150708'])
print(dates)

# converted to PeriodIndex on daily frequency
print(dates.to_period('D'))

# TimeDeltaIndex created when one dates is subtracted from another
print(dates - dates[0])

# = Regular Sequences: pd.date_range() (similar to range())=
print(pd.date_range('2015-07-03', '2015-07-10'))  # start and end range
print(pd.date_range('2015-07-03', periods=8))  # with number of periods
print(pd.date_range('2015-07-03', periods=8, freq='H'))  # 8 periods with Hourly Frequency

# period_range
print(pd.period_range('2015-07', periods=8, freq='M'))  # 8 monthly periods

# sequence of durations increasing by an hour
print(pd.timedelta_range(0, periods=10, freq='H'))

# frequency specification (hour and minute)
print(pd.timedelta_range(0, periods=9 ,freq='2H30T'))


# --- Resampling, Shifting and Windowing ---
# Financial data
from pandas_datareader import data

goog = data.DataReader('GOOG', start='2004', end='2016', data_source='yahoo')
print(goog.head())

# using closing price
goog = goog['Close']

# visualizing
plt.plot(goog)

# = Resampling and Converting Frequencies =
# resample() and asfreq()  ('BA' -> end of business year)
plt.plot(goog, alpha=.5, linestyle='-')
plt.plot(goog.resample('BA').mean(), linestyle=':')  # average of the previous year
plt.plot(goog.asfreq('BA'), linestyle='--')  # value at the end of the year
plt.legend(['input', 'resample', 'asfreq'], loc='best')

# resample business day data at a daily frequency (for NaN filling)
fig, ax = plt.subplots(2, sharex=True)
data = goog.iloc[:10]

data.asfreq('D').plot(ax=ax[0], marker='o')  # missing rows for missing data
data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
data.asfreq('D', method='ffill').plot(ax=ax[1], style='--o')
ax[1].legend(['back-fill', 'forward-fill'])


# = Time Shifts =
# shift() and tshift()
old_goog = goog.copy()
fig, ax = plt.subplots(nrows=3, ncols=1, sharey=True)
goog = goog.asfreq('D', method='pad')  # apply a frequency

goog.plot(ax=ax[0])
goog.shift(periods=900).plot(ax=ax[1])  # shift data replace empty with NaN
goog.tshift(periods=900).plot(ax=ax[2])  # shifts the index

local_max = pd.to_datetime('2007-11-05')
offset = pd.Timedelta(900, 'D')

ax[0].legend(['input'], loc=2)
ax[0].get_xticklabels()[4].set(weight='heavy', color='red')
ax[0].axvline(local_max, alpha=.3, color='red')

ax[1].legend(['shift(900)'], loc=2)
ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
ax[1].axvline(local_max + offset, alpha=0.3, color='red')

ax[2].legend(['tshift(900)'], loc=2)
ax[2].get_xticklabels()[1].set(weight='heavy', color='red')
ax[2].axvline(local_max + offset, alpha=0.3, color='red')


# shifted vals to compute ROI
ROI = 100 * (goog.tshift(-365) / goog - 1)
ROI.plot()
plt.ylabel('% Return on Investment')

# Rolling Windows (kind of mean of every 365 points)
rolling = goog.rolling(365, center=True)  # one year centered rolling mean and std
data = pd.DataFrame({'input': goog,
                     'one-year rolling_mean': rolling.mean(),
                     'one-year rolling_std': rolling.std()})
ax = data.plot(style=['-', '--', ':'])
ax.lines[0].set_alpha(.3)


# --- Example: Visualizing Seattle Bicycle Counts ---
directory = os.getcwd()
data = pd.read_csv(str(directory + '/Data/FremontBridge.csv'), index_col='Date', parse_dates=True)
print(data.head())

# dropping total and making a new one
data = data.drop(columns=data.columns[0])
data.columns = ['East', 'West']
data['total'] = data.eval('West + East')

# describe
print(data.dropna().describe())

# visualizing the data
data.plot()
plt.ylabel('Hourly Bicycle Count')

# resample by week
weekly = data.resample('W').sum()  # merge by weekly
weekly.plot(style=[':', '--', '-'])
plt.ylabel('Weekly bicycle count')

# 30-day rolling mean
daily = data.resample('D').sum()  # merge by daily
new_daily = daily[:'2016-08-30']
new_daily.rolling(30, center=True).sum().plot(style=[':', '--', '-'])
plt.ylabel('mean hourly count')

# smoother version with Gaussian window (50 days)
new_daily.rolling(50, center=True, win_type='gaussian').sum(std=10).plot(style=[':', '--', '-'])

# = Digging into the data =
# group by for average traffic as a function of the time of day
by_time = data.groupby(data.index.time).mean()  # average for evey hour out of 24
hourly_ticks = 4 * 60 * 60 * np.arange(6)
by_time.plot(xticks=hourly_ticks, style=[':', '--', '-'])

# based on the day of the week
by_weekday = data.groupby(data.index.dayofweek).mean()
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
by_weekday.plot(style=[':', '--', '-'])

# Comparing hourly on weekdays and weekends
weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')  # similar to a ? in c++
by_time = data.groupby([weekend, data.index.time]).mean()  # group by weekday/end and hourly

# Plotting the two panels
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
by_time.loc['Weekday'].plot(ax=ax[0], title='Weekday', xticks=hourly_ticks, style=[':', '--', '-'])
by_time.loc['Weekend'].plot(ax=ax[1], title='Weekend', xticks=hourly_ticks, style=[':', '--', '-'])


# --- High Performance Pandas: eval() and query() ---

# when adding element of two arrays
import time
rng = np.random.RandomState(42)
x = rng.rand(1000000)
y = rng.rand(1000000)
t = time.time()
x+y
runtime = time.time() - t
print(f'runtime: {runtime}')

# above faster than conventional python loop
t = time.time()
np.fromiter((xi + yi for xi, yi in zip(x, y)), dtype=x.dtype, count=len(x))
runtime = time.time() - t
print(f'Runtime: {runtime}')

# when computing compound expressions is less efficient
mask = (x > .5) & (y < .5)

# equivalent to above (every intermediate step is explicitly allocated in memory)
tmp1 = (x > .5)
tmp2 = (y < .5)
mask = tmp1 & tmp2


# Using numexpr gives compound computation without allocating full intermediate arrays
import numexpr
mask_numexpr = numexpr.evaluate('(x > .5) & (y < .5)')
np.allclose(mask, mask_numexpr)

# --- pandas.eval() ---
nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols)) for _ in range(4))

# compute sum of all four typical Pandas approach
t = time.time()
result = df1 + df2 + df3 + df4
runtime = time.time() - t
print(f'Runtime: {runtime}')

# using pd.eval
t = time.time()
pd.eval('df1 + df2 + df3 + df4')
runtime = time.time() - t
print(f'Runtime: {runtime}')

# gives same result
np.allclose(df1 + df2 + df3 + df4, pd.eval('df1 + df2 + df3 + df4'))

# Operations supported by eval
df1, df2, df3, df4, df5 = (pd.DataFrame(rng.randint(0, 1000, (100, 3))) for _ in range(5))
result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
np.allclose(result1, result2)

# comparison operators
result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
result2 = pd.eval('df1 < df2 <= df3 != df4')
np.allclose(result1, result2)

# bitwise operators
result1 = (df1 < .5) & (df2 < .5) | (df3 < df4)
result2 = pd.eval('(df1 < .5) & (df2 < .5) | (df3 < df4)')
np.allclose(result1, result2)

# Object attributes and indices
result1 = df2.T[0] + df3.iloc[1]
result2 = pd.eval('df2.T[0] + df3.iloc[1]')
np.allclose(result1, result2)

# --- DataFrame.eval() for Column-Wise Operations
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
print(df.head())

result1 = (df['A'] + df['B']) / (df['C'] - 1)
result2 = pd.eval("(df.A + df.B) / (df.C - 1)")
np.allclose(result1, result2)

# another way to access columns
result3 = df.eval('(A + B) / (C - 1)')
np.allclose(result1, result3)

# = Assignment in DataFrame.eval() =
print(df.head())

# eval to create new column
df.eval("D = (A + B) / C", inplace=True)
print(df.head())

# existing column can be modified
df.eval("D = (A - B) / C", inplace=True)
print(df.head())

# = Local variables in DataFrame.eval() =
column_mean = df.mean(axis=1)
result1 = df['A'] + column_mean
result2 = df.eval(" A + @column_mean")  # using '@' you can access local variables
np.allclose(result1, result2)

# --- DataFrame.query() Method ---
# consider filtering operation
result1 = df[(df.A < .5) & (df.B < .5)]
result2 = pd.eval("df[(df.A < .5) & (df.B < .5)]")
np.allclose(result1, result2)

# using query()
result2 = df.query("A < .5 and B < .5")
np.allclose(result1, result2)

# query also accepts local vars
Cmean = df['C'].mean()
result1 = df[(df.A < Cmean) & (df.B < Cmean)]
result2 = df.query('A < @Cmean and B < @Cmean')
np.allclose(result1, result2)


