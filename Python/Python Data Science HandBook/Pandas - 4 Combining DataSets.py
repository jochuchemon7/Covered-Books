"""
# Combining Datasets: Concat and Append
"""
import pandas as pd
import numpy as np
import os

def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)


print(make_df('ABC', range(3)))


# --- Concatenation of Numpy Arrays ---
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])

# on axis
x = [[1, 2], [3, 4]]
print(np.concatenate([x, x], axis=1))


# = Simple Concatenation with pd.concat =
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
print(pd.concat([ser1, ser2], axis=0))

# = Data Frame concatenation =
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(df1)
print(pd.concat([df1, df2], axis=0))

# on different axis (rbin)
df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
print(pd.concat([df3, df4], axis=1))

# = Duplicate Indices =
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index  # assigning same index
print(y)
print(x)
print(pd.concat([x, y], axis=0))

# ignoring the index to avoid overlapping
print(pd.concat([x, y], ignore_index=True))

# adding MultiIndex keys
print(pd.concat([x, y], keys=['x', 'y']))

# = Concatenation with joins =
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
print(df5)
print(pd.concat([df5, df6]))  # fills NaN

# inner join (intercept columns)
print(pd.concat([df5, df6], join='inner', axis=0))

# = append() method =
print(df1)
print(df2)
print(df1.append(df2))  # append fun on Data Frames


# --- Merge and Join ---

# = Joins =

# one-to-one join
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
print(df1); print(df2)

df3 = pd.merge(df1, df2)  # one-to-one join merge()
print(df3)


# many-to-one join
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
print(df3); print(df4)
pd.merge(df3, df4)

# many-to-many join
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting', 'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux', 'spreadsheets', 'organization']})
print(df1); print(df5)
print(pd.merge(df1, df5))

# = Specification of the Merge Key =
print(df1); print(df2)
print(pd.merge(df1, df2, on='employee'))

# left_on and right_on (when the names of the columns are different)
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(df1); print(df3)
print(pd.merge(df1, df3, left_on='employee', right_on='name'))

# drop a redundant column (axis reverse)
pd.merge(df1, df3, left_on='employee', right_on='name').drop('name', axis=1)


# = left_index and right_index =
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(df1a); print(df2a)
print(pd.merge(df1a, df2a, left_index=True, right_index=True))  # index merge

# using join instead
print(df1a.join(df2a))

# mixing index and on
print(df1a); print(df3)
print(pd.merge(df1a, df3, left_index=True, right_on='name'))

# --- Specifying Set Arithmetic for Joins ---

df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']}, columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']}, columns=['name', 'drink'])
print(df6); print(df7);
print(pd.merge(df6, df7))
print(pd.merge(df6, df7, how='inner'))  # same as above

# outer join
print(pd.merge(df6, df7, how='outer'))

# left join (join over the left entries)
print(pd.merge(df6, df7, how='left'))


# --- Overlapping Column Names ---
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print(df8); print(df9);
print(pd.merge(df8, df9, on="name"))

# specifying the suffixes
print(pd.merge(df8, df9, on='name', suffixes=['_L', '_R']))


# --- Example: US States Data ---
direct = os.getcwd()
pop = pd.read_csv(str(direct + '/Data/state-population.csv'))
areas = pd.read_csv(str(direct + '/Data/state-areas.csv'))
abbrevs = pd.read_csv(str(direct + '/Data/state-abbrevs.csv'))

# heads
print(pop.head())
print(areas.head())
print(abbrevs.head())

# merging data and dropping redundancy
merged = pd.merge(pop, abbrevs, left_on='state/region', right_on='abbreviation', how='outer')
merged = merged.drop(columns='abbreviation')
print(merged.head())

# checking any mismatches (rows with nulls)
print(merged.isnull().any())

# who is null of the population
print(merged[merged['population'].isnull()].head())

# state entries not matching with abrevs
print(merged.loc[merged['state'].isnull(), 'state/region'].unique())

# adding 'Puerto Rico' value to state for abrevation PR
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
print(merged.isnull().any())

# merge merged with areas
final = pd.merge(merged, areas, on='state', how='left')
print(final.head())

# checking for nulls
print(final.isnull().any())

# which regions of null area are there
print(final['state'][final['area (sq. mi)'].isnull()].unique())

# dropping null values on population density of entire USA
final.dropna(inplace=True)
print(final.head())

# selecting  data from 2000s and total population (using query() function)
data2010 = final.query("year == 2010 & ages == 'total'")
print(data2010.head())


# Calculate population density and display
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']

density.sort_values(ascending=False, inplace=True)
print(density.head())
print(density.tail())

