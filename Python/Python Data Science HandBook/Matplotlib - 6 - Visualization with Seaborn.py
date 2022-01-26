"""
# Visualization with Seaborn
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
plt.style.use('classic')

# some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# plot the data
plt.plot(x, y)
plt.legend(['A', 'B', 'C', 'D', 'E', 'F'], loc='upper left', ncol=2)

# ++ Using Seaborn ++
sns.set()  # setting the style
plt.plot(x, y)
plt.legend(['A', 'B', 'C', 'D', 'E', 'F'], ncol=2, loc='upper left')

# --- Exploring Seaborn Plots ---

# ~~ Histograms, KDE (kernel density estimation), and densities ~~

# using matplotlib on normal distributions
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col], density=True, alpha=.5)  # plot density histogram

# smooth estimate using a kernel density of the distribution
for col in 'xy':
    sns.kdeplot(data[col], shade=True)

# combining histograms and KDEs
sns.distplot(data['x'])
sns.distplot(data['y'])

# 2-dimensional data set
sns.kdeplot(data=data, x='x', y='y')
sns.kdeplot(data=data, x='x', y='y', fill=True)

# Viewing joint distributions and marginal distributions
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde', fill=True)

# hexagonal based histogram
with sns.axes_style('white'):
    sns.jointplot(data=data, x='x', y='y', kind='hex')

# ~~ Pair Plots ~~
iris = sns.load_dataset('iris')
print(iris.head())

sns.pairplot(iris, hue='species', size=2.5)

# ~~ Faceted Histograms ~~
tips = sns.load_dataset('tips')
print(tips.head())
tips['tip_pct'] = 100 * (tips['tip'] / tips['total_bill'])

grid = sns.FacetGrid(tips, row='sex', col='time', margin_titles=True)
grid.map(plt.hist, 'tip_pct', bins=np.linspace(0, 40, 15))

# ~~ Factor Plots ~~
# view the distribution of a parameter with bins defined by any other parameter  (ei. total_bill by day)
with sns.axes_style(style='ticks'):
    g = sns.factorplot(x='day', y='total_bill', hue='sex', data=tips, kind='box')
    g.set_axis_labels('Day', 'Total Bill')

#  ~~ Joint Distributions ~~
# sns.joinplot() for joint distribution between diff data_sets along with marginal distribution
with sns.axes_style('white'):
    sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')

# ~ joint plot with automatic kernel density estimation and regression ~
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')

# ~~ Bar Plots ~~
planets = sns.load_dataset('planets')
print(planets.head())

# plot with factor plot with kind='count'
with sns.axes_style('white'):
    g = sns.factorplot(x='year', data=planets, kind='count', color='steelblue', aspect=2)
    g.set_xticklabels(step=5)  # with 5 x-tick labels


# plotting by looking at the method of discovery
with sns.axes_style('white'):
    g = sns.factorplot(x='year', data=planets, kind='count', aspect=4, hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')


# --- Example: Exploring Marathon Finishing Times ---
directory = os.getcwd()
data = pd.read_csv(str(directory + '/Data/marathon-data.csv'))
print(data.head())
print(data.dtypes)  # data types from the columns data  (3 are store as 'object' ADT)

# fixing objects ADT for Timedelta for time series
def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return pd.Timedelta(hours=h, minutes=m, seconds=s)



data = pd.read_csv(str(directory+'/Data/marathon-data.csv'),
                   converters={'split': convert_time, 'final': convert_time})  # applying function to cols
print(data.head())
print(data.dtypes)

# Adding split_sec and final_sec columns  (timedelta to seconds int)
data['split_sec'] = data['split'].astype(int) / 1E9
data['final_sec'] = data['final'].astype(int) / 1E9
print(data.head())

# ~ A joint-plot over the data ~

with sns.axes_style('white'):
    g = sns.jointplot(x='split_sec', y='final_sec', data=data, kind='hex')
    g.ax_joint.plot(np.linspace(4000, 16000), np.linspace(8000, 32000), ':k')

# add column to measure the degree each runner negative-splits or positive splits the race
data['split_frac'] = 1 - 2 * data['split_sec'] / data['final_sec']
print(data.head())

# plotting runners with negative splits
sns.displot(x='split_frac', data=data, kde=False)
plt.axvline(x=0, color='black', linestyle='--')

# total people with negative-splits
print(sum(data['split_frac'] < 0))


# ~~ PairGrid ~~
# checking correlation between this split fraction and other variables
g = sns.PairGrid(data=data, vars=['age', 'split_sec', 'final_sec', 'split_frac'], hue='gender',
                 palette='RdBu_r')
g.map(plt.scatter, alpha=.8)
g.add_legend()

# men and women differences
sns.kdeplot(data.split_frac[data.gender == 'M'], label='men', shade=True, color='blue')
sns.kdeplot(data.split_frac[data.gender == 'W'], label='women', shade=True, color='pink')
plt.legend()
plt.xlabel('split_frac')

# ~~ sns.violinplot() ~~
# Looking at the distributions with violin-plot
sns.violinplot(data=data, x='gender', y='split_frac', palette=['lightblue', 'lightpink'])

# new column for decade of age
data['age_dec'] = data.age.map(lambda age: 10 * (age // 10))
print(data.head())

men = (data.gender == 'M')
women = (data.gender == 'W')

# sns.violinplot with hue='gender', and split=True
with sns.axes_style(style=None):
    sns.violinplot(x='age_dec', y='split_frac', hue='gender', data=data, split=True,
                   palette=['lightblue', 'lightpink'], inner='quartile')

# number of 80 year old runners
(data.age > 80).sum()

# using regplot on men with negative splits  ( sns.lmplot() )
g = sns.lmplot(x='final_sec', y='split_frac', col='gender', data=data, markers='.', scatter_kws=dict(color='c'))
g.map(plt.axhline, y=.1, color='k', ls=':')

