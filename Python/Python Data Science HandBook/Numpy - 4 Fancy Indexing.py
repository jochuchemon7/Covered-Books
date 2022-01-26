"""
# Fancy Indexing
"""
import numpy as np
import matplotlib.pyplot as plt

rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)

print([x[3], x[7], x[2]])  # One way of accessing elements
ind = [3, 7, 2]

print(x[ind])  # An alternative

# Fancy indexing shape of the result reflects shape of index arrays
ind = np.array([[3, 7], [4, 5]])
print(x[ind])

# On multiple dimensions
X = np.arange(12).reshape((3, 4))
print(X)
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
print(X[row, col])


# --- Combined Indexing ---
print(X)
print(X[2, [2, 0, 1]])  # Combine fancy and simple indices
print(X[1:, [2, 0, 1]])  # Fancy indexing and slicing

mask = np.array([1, 0, 1, 0], dtype=bool)  # Fancy indexing and masking
print(X[row[:, np.newaxis], mask])


# = Example: Selecting Random Points =
mean = [0, 0]
cov = [[1, 2], [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
print(X.shape)

plt.scatter(X[:, 0], X[:, 1])  # Plotting

# Selecting 20 random points
indices = np.random.choice(X.shape[0], 20, replace=False)
print(indices)

# Slicing
selection = X[indices, :]
print(selection.shape)

# Plotting
plt.scatter(X[:, 0], X[:, 1], alpha=.3)
plt.scatter(selection[:, 0], selection[:, 1], facecolor='none', s=200, color='red')


# = Modifying values with fancy indexing =
x = np.arange(10)
i = np.array([2, 1, 8, 4])

x[i] = 99  # On elements with the passing index
print(x)
x[i] -= 10
print(x)

i = [2, 3, 3, 4, 4, 4]
x = np.zeros(10)
np.add.at(x, i, 1)  # If we wanted to x[i] += 1
print(x)


# = Binning Data Example =
np.random.seed(42)
x = np.random.randn(100)

bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)
i = np.searchsorted(bins, x)  # find appropriate bin for each x
np.add.at(counts, i, 1)  # add 1 to each of the bins

plt.plot(bins, counts, linestyle='--', drawstyle='steps')

# using plt.hist
plt.hist(x, bins, histtype='step')


# --- Fast Sorting in NumPy ---
x = np.array([2, 1, 4, 3, 5])
np.sort(x)

x.sort()  # sort array-in-place
print(x)

x = np.array([2, 1, 4, 5])
i = np.argsort(x)  # Indices of the sorted elements
print(i)

print(x[i])


# Sorting along rows of columns
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)

np.sort(X, axis=0)
np.sort(X, axis=1)

#  = Partial Sorts =
x = np.array([7, 2, 3, 1, 6, 5, 4])
np.partition(x, 3)  # smallest K values to the left of the partition

# --- Example: KNN ---

X = rand.rand(10, 2)
print(X)
plt.scatter(X[:, 0], X[:, 1], s=100)

# for each pair of points, compute differences in their coordinates
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
print(differences.shape)

# square coordinates differences
sq_differences = differences ** 2
print(sq_differences.shape)

# sum coordinate differences to get the squared distance
dist_sq = sq_differences.sum(-1)
print(dist_sq.shape)

# Sort along rows
nearest = np.argsort(dist_sq, axis=1)
print(nearest)

# partition sort
K = 2
nearest_partition = np.argpartition(dist_sq, K+1, axis=1)

# plotting
plt.scatter(X[:, 0], X[:, 1], s=100)
K = 2
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        plt.plot(*zip(X[j], X[i]), color='black')


# --- Structured Data ---
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]


# previous simple array
x = np.zeros(4, dtype=int)
print(x)

# use a compound data type for structured arrays
data = np.zeros(4, dtype={'names': ('name', 'age', 'weight'),
                          'formats': ('U10', 'i4', 'f8')})
print(data.dtype)

# fill the array with our lists
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

# refer by index or name
print(data['name'])  # Get all names
print(data[0])  # first row of data
print(data[-1]['name'])  # name from last row

# Using Boolean masking
print(data[data['age'] < 30]['name'])


# --- Structured Arrays ---
np.dtype()

