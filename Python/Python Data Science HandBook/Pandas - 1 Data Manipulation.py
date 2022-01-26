"""
# Introducing Pandas
"""
import numpy as np
import pandas as pd

# --- Pandas Series Object ---
data = pd.Series([.25, .5, .75, 1])
print(data)
print(f'Values: {data.values}')
print(f'Index: {data.index}')
print(list(data.index))

# accessing data
print(data[0])
print(data[1:3])

# Using strings as index
data = pd.Series([.25, .5, .75, 1], index=['a', 'b', 'c', 'd'])
print(data)
print(data['b'])

# non-sequential indices
data = pd.Series([.25, .5, .75, 1], index=[2, 5, 3, 7])
print(data)
print(data[5])

# = From dictionary to Pandas Series =
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
print(population)
print(population['California'])
print(population['California':'Illinois'])

# other examples
print(pd.Series([2, 4, 6]))
print(pd.Series(5, index=[100, 200, 300]))  # scalar data, repeats to fill index
print(pd.Series({2: 'a', 1: 'b', 3: 'c'}))  # from dict()
print(pd.Series({2: 'a', 1: 'b', 3: 'c'}, index=[3, 2]))  # Picking index/slicing


# --- Data Frames ---
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
print(area)

# Building DF giving col names and align to the index values
states = pd.DataFrame({'population': population, 'area': area})
print(states)
print(f'states.index: {states.index}')  # rowwname
print(f'states.columns: {states.columns}')  # colname

# Specialized Dictionary
print(states['area'])

# DF from single Series
print(pd.DataFrame(population, columns=['population']))

# DF from a list of dicts
data = [{'a': i, 'b': 2 * i} for i in range(3)]
print(pd.DataFrame(data))

# if values are missing they are filled with NaN
print(pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}]))

# from dict to series
print(pd.DataFrame({'population': population, 'area': area}))

# from two-dimensional numpy array
print(pd.DataFrame(np.random.rand(3, 2), columns=['foo', 'bar'], index=['a', 'b', 'c']))

# from numpy structured array
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
A
print(pd.DataFrame(A))


# = Pandas Index Object =
ind = pd.Index([2, 3, 5, 7, 11])
print(ind)
print(ind[1])
print(ind[::2])

print(f'ind.size: {ind.size}')
print(f'ind.shape: {ind.shape}')
print(f'ind.ndim: {ind.ndim}')
print(f'ind.dtype: {ind.dtype}')

"""
ind[1] = 99 would not work (not mutable)
"""

# = Unions, Intersections, differences =
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
print(indA & indB)  # intersect
print(indA | indB)  # Union
print(indA ^ indB)  # symmetric difference


# --- Data Indexing and Selection ---
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data)
print(data['b'])

print('a' in data)
print(data.keys())
print(list(data.items()))

# adding a new index value
data['e'] = 1.25
print(data)

# = Slicing, masking and indexing =
print(data['a':'c'])
print(data[0:2])  # implicit integer index
print(data[(data > .3) & (data < .8)])  # masking
print(data[['a', 'e']])  # fancy indexing


"""
# Best indexers (opinion) - loc, iloc
"""
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
print(data)
print(data[1])  # explicit index when indexing
print(data[1:3])  # implicit index when slicing

# loc attribute
print(data.loc[1])
print(data.loc[1:3])

# iloc attribute
print(data.iloc[1])
print(data.iloc[1:3])


# --- Data Selection in DataFrame ---
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area': area, 'pop': pop})  # dict() of Series
print(data)
print(data['area'])  # Accessing individual Series from DF
print(data.area)  # or with attribute-style access

# adding new column
data['density'] = data['pop'] / data['area']
print(data)

# DF as two-dimensional array
print(data.values)
# transpose data
print(data.T)

print(data.values[0])
print(data['area'])


# Using iloc and loc
print(data.iloc[:3, :2])
print(data.loc[:'Illinois', :'pop'])

# combine masking and fancy indexing
print(data.loc[data.density > 100, ['pop', 'density']])

# Changing values
data.iloc[0, 2] = 90
print(data)

# = Additional Indexing Conventions
print(data['Florida':'Illinois'])
print(data[1:3])  # slices can refer to rows by number rather than by index
print(data[data.density > 100])
