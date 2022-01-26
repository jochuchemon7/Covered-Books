"""
# Aggregation and Grouping
"""
import seaborn as sns
import pandas as pd
import numpy as np

planets = sns.load_dataset('planets')
print(planets.shape)
print(planets.head())


# --- Simple Aggregation in Pandas ---
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
print(ser)

# aggregates return single value for a Panda Series
print(ser.sum())
print(ser.mean())

# For data frame
df = pd.DataFrame({'A': rng.rand(5), 'B': rng.rand(5)})
print(df)
print(df.mean())
print(df.mean(axis=0))

# describe() computes several common aggregates while dropping na rows
print(planets.dropna().describe())


# --- GroupBy: Split, Apply, Combine ---
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=['key', 'data'])
print(df)

# groupby() by passing a col name
print(df.groupby('key'))

# when passing an aggregate
print(df.groupby('key').sum())


# = GroupBy Object =

# column indexing
print(planets.groupby('method'))
print(planets.groupby('method')['orbital_period'])

# column indexing with aggregates
print(planets.groupby('method')['orbital_period'].median())

# Iteration over groups
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))

# dispatch methods
print(planets.groupby('method')['year'].describe())

# = Aggregate, filter, transform, apply =
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                  columns=['key', 'data1', 'data2'])
print(df)

# AGGREGATION (multiple methods at once)
print(df.groupby('key').aggregate(['min', np.median, max]))

# specifying the method to columns
print(df.groupby('key').aggregate({'data1': 'min',
                                   'data2': 'max'}))

# FILTERING (drop data based on the group property)
def filter_func(x):
    return x['data2'].std() > 4


print(df); print(df.groupby('key').std())
print(df.groupby('key').filter(func=filter_func))  # Drops all rows from key A

# TRANSFORMATION (output same shape)
print(df.groupby('key').transform(lambda x: x - x.mean()))  # mean of group of keys

# APPLY method (apply function to the group result) similar to transformation
def norm_by_data2(x):
    x['data1'] /= x['data2'].sum()
    return x


print(df)
print(df.groupby('key').apply(norm_by_data2))

# list providing the grouping keys
L = [0, 1, 0, 1, 2, 0]  # assigning the index to a newly created keys (0, 1, 2)
print(df.groupby(L).sum())

# dictionary index to group
df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2)
print(df2.groupby(mapping).sum())

# any python function that inputs the index
print(df2)
print(df2.groupby(str.lower).sum())

# list of valid keys (combine to group on multi-index)
print(df2.groupby([str.lower, mapping]).sum())


# = Grouping Example =

# count discovered planets by method and by decade
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
print(planets.groupby(['method', decade])['number'].sum().unstack().fillna(0))


