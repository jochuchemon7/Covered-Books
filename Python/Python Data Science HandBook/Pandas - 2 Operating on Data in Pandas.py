"""
# Operating on Data in Pandas
"""
import numpy as np
import pandas as pd

rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
print(ser)

df = pd.DataFrame(rng.randint(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D'])
print(df)

# NumPy ufuncs on Series and DF
print(np.exp(ser))
print(np.sin(df * np.pi / 4))

# Index Alignment
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
print(population/area)  # Gives NaN values

# any missing values are filled with NaN
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
print(A+B)

# replace NaN with other value
A.add(B, fill_value=0)

# index alignment on operations in DataFrame
A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list('AB'))
print(A)
B = pd.DataFrame(rng.randint(0, 10, (3, 3)), columns=list('BAC'))
print(B)

print(A+B)

# filling with the mean
fill = A.stack().mean()
print(A.add(B, fill_value=fill))


# = Operations between DF and Series
A = rng.randint(10, size=(3, 4))
print(A)
print(A - A[0])

# subtraction
df = pd.DataFrame(A, columns=list('QRST'))
print(df - df.iloc[0])

# define axis
print(df.subtract(df['R'], axis=0))

# alignment
halfrow = df.iloc[0, ::2]
print(halfrow)
print(df - halfrow)


# --- Missing Data ---
val1 = np.array([1, None, 3, 4])
print(val1)

# NaN
vals2 = np.array([1, np.nan, 3, 4])
print(vals2)
print(vals2.dtype)
print(vals2.sum())  # nan for any operation

# to ignore nan
print(np.nansum(vals2))
print(np.nanmax(vals2))

# NaN and None in Pandas
print(pd.Series([1, np.nan, 2, None]))

# automatic None to nan and int values to float
x = pd.Series(range(2), dtype=int)
print(x)
x[0] = None
print(x)

# --- Operating on Null Values ---

# detecting null values (None and nan are used inter-changeable
data = pd.Series([1, np.nan, 'hello', None])
print(data.isnull())   # boolean null values
print(data[data.notnull()])  # non null values

# dropping null values
print(data.dropna())

# dropping null values in a DF
df = pd.DataFrame([[1, np.nan, 2], [2, 3, 5], [np.nan, 4, 6]])
print(df)
print(df.dropna())  # drops full rows or columns

# drop NaN along different axis
print(df.dropna(axis='columns'))
print(df.dropna(axis='rows'))

# only dropping when all elements are NaN
df[3] = np.nan
print(df)
print(df.dropna(axis='columns', how='all'))
print(df.dropna(axis='rows', thresh=3))  # min number of non-null values for row/col to be kept

# = Filling Null Values =
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
print(data)

# fill NA entries with single value
print(data.fillna(0))

# forward fill
print(data.fillna(method='ffill'))
# back-fll
print(data.fillna(method='bfill'))

# for data frames
print(df)
print(df.fillna(method='ffill', axis=1))
