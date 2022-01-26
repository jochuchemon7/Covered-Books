"""
# Pivot Tables and ...
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

titanic = sns.load_dataset('titanic')
print(titanic.head())

# --- Pivot Tables by Hand ---
print(titanic.groupby('sex')[['survived']].mean())  # mean on sexes
print(titanic.groupby(['sex', 'class'])['survived'].mean().unstack())  # mean on sexes and class


# = Pivot Table Syntax =
# equivalent with pivot table
print(titanic.pivot_table('survived', index='sex', columns='class'))

# = Multi-level pivot tables =
# age as a third dimension
age = pd.cut(titanic['age'], [0, 18, 80])  # cut() bins of values
titanic.pivot_table('survived', index=['sex', age], columns='class')

# adding info on the fare paid
fare = pd.qcut(titanic['fare'], 2)  # automatic compute quantiles
print(titanic.pivot_table('survived', index=['sex', age], columns=['class', fare]))


# = Additional pivot table options =
# specifying aggregate functions on data
print(titanic.pivot_table(index='sex', columns='class', aggfunc={'survived': sum, 'fare': 'mean'}))

# computing totals along each grouping (margins=True)
print(titanic.pivot_table('survived', index='sex', columns='class', margins=True))


# --- Example Birthrate Data ---
direct = os.getcwd()
births = pd.read_csv(str(direct + '/Data/births.csv'))
print(births.head())

# add decade column and look at the m/f birthrate of decade
births['decade'] = 10 * (births['year'] // 10)  # (replacing last digit with 0)
print(births.pivot_table('births', index='decade', columns='gender', aggfunc='sum'))

# Visualize the trend
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
plt.ylabel('total births per year')


# = Further Data Exploration =
# estimate of the sample mean .74 is interquartile of Gaussian distribution
quartile = np.percentile(births['births'], [25, 50, 75])
mu = quartile[1]
sig = .74 * (quartile[2] - quartile[0])
print(sig)
old_births = births.copy()

# use query to filter rows with births outside these values
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

# day from float to int
births['day'] = births['day'].astype(int)

# create a datetime index from year, month, day
births.index = pd.to_datetime(10000 * births.year + 100 * births.month  + births.day, format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek


# Plot births by weekday
births.pivot_table('births', index='dayofweek', columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues','Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day')


# = Plotting the mean of number of births by day of the year =
births_by_date = births.pivot_table('births', index=['month', 'day'])
print(births_by_date.head())

# months and days into datetime datatype for index (kinda unstack)
births_by_date.index = [pd.datetime(2012, month, day) for (month, day) in births_by_date.index]
births_by_date.head()

# plotting a time series on the average number of births by date of the year
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)


# --- VECTORIZED STRING OPERATIONS ---

# = Pandas String Operations =
x = np.array([2, 3, 5, 7, 11, 13])
print(x * 2)

# not the same vectorization with string values
data = ['peter', 'Paul', 'MARY', 'gUIDO']
print([name.capitalize() for name in data])  # loops needed

# will break if missing values
data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
print([name.capitalize() for name in data])  # breaks because of none

# PANDAS APPROACH
names = pd.Series(data)
print(names)

# calling single method while skipping missing values
print(names.str.capitalize())


# = TABLES OF PANDAS STRING METHODS =
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])

# many Python built-in string methods are mirrored by Pandas vectorized string
print(monte.str.lower())
print(monte.str.len())
print(monte.str.startswith('T'))
print(monte.str.split())
print(monte.str.match('Te'))  # re.match()
print(monte.str.findall('Terry'))  # re.findall() on each element
print(monte.str.replace('Terry', 'John'))  # replace occurrences
print(monte.str.contains('Eric'))  # re.search() on each element returns boolean
print(monte.str.count('Terry'))  # count occurrences of pattern

# extracting first name
print(monte.str.extract('([A-Za-z]+)'))  # input regular expression

# finding names that start and end with consonant
print(monte.str.findall(r'^[AEIOU].*[aeiou]$'))  # ^ -> starts $ -> ends

# More Miscellaneous methods
print(monte)
print(monte.str.get(0))  # position of element
print(monte.str.slice(start=0, stop=3, step=1))  # slice each element
print(monte.str.slice_replace(start=0, stop=2, repl='__'))  # replace sliced element
print(monte.str.cat(['a', 'b', 'c', 'd', 'e', 'f'], sep=','))  # cat() to each element corresponding
print(monte.str.repeat(repeats=2))  # repeat value
print(monte.str.pad(width=15, side='both', fillchar='-'))  # pretty print
print(monte.str.join('-'))
print(monte.str.get_dummies())  # transform into dummy vars

# = VECTORIZED ITEM ACCESS AND SLICING =
print(monte.str[0:3])  # same as slicing
print(monte.str.split().str.get(-1))  # split name in two and get last name

# example of dummy vars
full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C', 'B|C|D']})
print(full_monte)

# dummy
print(full_monte['info'])
print(full_monte['info'].str.get_dummies())
