"""
# Hierarchical Indexing
"""
import numpy as np
import pandas as pd

# tuple based index
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956, 18976457, 19378102, 20851820, 25145561]
pop = pd.Series(populations, index=index)
print(pop)


"""
# --- MultiIndex ---
"""

index = pd.MultiIndex.from_tuples(index)
print(index)
print(index.levels)

# re-index
pop = pop.reindex(index)
print(pop)

# Access all data when second index is 2010
print(pop[:, 2010])

# Extra dimension (convert multiplyIndex Series into DataFrame
pop_df = pop.unstack()
print(pop_df)

# Stack is the opposite
print(pop_df.stack())

# Adding another column
pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094, 4687374, 4318033, 5906301, 6879014]})
print(pop_df)
f_u18 = pop_df['under18'] / pop_df['total']
print(f_u18.unstack())


# --- MultiIndex Creation ---
# 'easiest' way
df = pd.DataFrame(np.random.rand(4, 2), index=[['a', 'a', 'b', 'b'], [1, 1, 2, 2]],
                  columns=['data1', 'data2'])
print(df)

# Passing appropriate tuples as keys
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
print(pd.Series(data))

#  = explicit MultiIndex constructors =

print(pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 1, 2, 2]]))
print(pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)]))
print(pd.MultiIndex.from_product([['a', 'b'], [1, 2]]))


# = MultiIndex level names =
pop.index.names = ['state', 'year']
print(pop)

# = MultiIndex for columns (columns can also have MultiIndex) =
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']], names=['subject', 'type'])

data = np.round(np.random.rand(4, 6), 1)
data[:, ::2] *= 10
data += 37

health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data)

# 4-dimensional data
print(health_data['Guido'])


# --- Indexing and Slicing MultiIndex  ---

# = INDEXED SERIES =
print(pop)

# single element
print(pop['California', 2010])

# Using one level
print(pop['California'])

# Partial slicing
print(pop['California': 'Texas'])
print(pop[:, 2000])

# Boolean Masks
print(pop[pop > 22000000])

# fancy indexing
print(pop[['California', 'Texas']])


# = INDEXING DATA FRAMES =
print(health_data)
print(health_data['Guido', 'HR'])  # Single column

# iloc
print(health_data.iloc[:2, :2])

# loc
print(health_data.loc[:, ('Bob', 'HR')])
print(health_data.loc[2013, :])

# IndexSlicing
idx = pd.IndexSlice
print(health_data.loc[idx[:, 1], idx[:, 'HR']])
print(health_data.loc[idx[2013, ], idx['Bob', ]])
print(health_data.loc[idx[2014, ], idx[:, 'Temp']])


# --- Rearranging Multi-Indices ---

# un-sorted new data
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
print(data)

# slicing would not work
try:
    print(data['a':'b'])
except KeyError as e:
    print(type(e))
    print(e)

# sort_index() example
data = data.sort_index()
print(data)

# now we can slice as before
print(data['a':'b'])

# = Stacking and UnStacking Indices =
print(pop)
print(pop.unstack(level=0))
print(pop.unstack(level=1))

# unstacking
print(pop.unstack().stack())


# = Index Setting and Resetting = (turning index labels into columns)
pop_flat = pop.reset_index(name='population')
print(pop_flat)

# Building MultiIndex
print(pop_flat.set_index(keys=['state', 'year']))


# --- Data Aggregations on Multi-Indices ---
print(health_data)

# mean for all years
data_mean = health_data.mean(level='year')
print(data_mean)

# mean on a level from column
print(health_data.mean(axis=1, level='type'))
print(data_mean.mean(axis=1, level='type'))


