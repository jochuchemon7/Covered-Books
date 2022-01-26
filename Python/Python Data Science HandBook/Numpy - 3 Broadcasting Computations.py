"""
# Computation on Arrays: Broadcasting
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# element-by-element operations
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b

# Adding a scalar
a + 5

# add 1D to a 2D
M = np.ones((3, 3))
print(M)
print(M+a)

# more complicated cases (stretches both arrays)
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
print(a+b)

"""
# Broadcasting stretches an array to be of the same shape filling the values with current ones
"""

# Broadcasting example 1
M = np.ones((2, 3))
a = np.arange(3)
print(M)
print(a)
print(M+a)

# Broadcasting example 2
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
print(a)
print(b)
print(a+b)

# Broadcasting example 3
M = np.ones((3, 2))
a = np.arange(3)
print(M)
print(a)
print(M+a)  # Gives error incompatible

print(a[:, np.newaxis].shape)
print(a[:, np.newaxis])  # Reshaping for compatibility
print(M + a[:, np.newaxis])
print(np.logaddexp(M, a[:, np.newaxis]))

# --- Broadcasting in Practice ---
X = np.random.random((10, 3))
X_mean = X.mean(axis=0)
print(X_mean)

# Center the data by subtracting the mean for the 3 columns
X_centered = X - X_mean
print(X_centered.mean(axis=0))


# = Plotting a two dimensional function =
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]  # y = np.linspace(0, 5, 50).reshape((50, 1))

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar()


# --- Comparisons, Masks, and Boolean Logic ---
rainfall = pd.read_csv(str(os.getcwd() + '/Data/Seattle2014.csv'))['PRCP'].values
inches = rainfall / 254
print(inches.shape)

# plotting inches
plt.hist(inches, bins=40)

# = Comparison Operators =
x = np.array([1, 2, 3, 4, 5])
print(x < 3)  # np.less()
print(x > 3)  # np.grater()
print(x <= 3)  # np.less_equal()
print(x >= 3)  # np.grater_equal()
print(x != 3)  # np.not_equal()
print(x == 3)  # np.equal()

# Element-by-element comparison of two arrays
print(2 ** x == x ** 2)

# 2D Example
rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
print(x)
print(x < 6)

# Counting entries
np.count_nonzero(x < 6)
np.sum(x < 6)

# Sum of counts of values less than 6 on axis
np.sum(x < 6, axis=1)
np.sum(x < 6, axis=0)

# are there any values greater than 8
np.any(x > 8)
np.any(x < 0)

# are all values less than 10
np.all(x < 10)
np.all(x == 6)

# all and any with axes
np.all(x < 8, axis=1)

# = Boolean Operators =
np.sum((inches > .5) & (inches < 1))  # two conditions

print(f'Number of days without rain: {np.sum(inches == 0)}')
print(f'Number of days with rain: {np.sum(inches > 0)}')
print(f'Days with more than .5 inches: {np.sum(inches > .5)}')
print(f'Rainy days with < .1 inches: {np.sum(inches < .1)}')


# = Boolean Arrays as Masks =
print(x)
print(x < 5)  # Boolean array
print(x[x < 5])  # selecting the values

# examples
rainy = (inches > 0)
summer = (np.arange(365) - 172 < 92) & (np.arange(365) - 172 > 0)  # Map of summer

print(f'Median precip on rainy days in 2014 (inches): {np.median(inches[rainy])}')
print(f'Median precip on summer days in 2014 (inches): {np.median(inches[summer])}')
print(f'Maximum precip on summer days in 2014 (inches)): {np.max(inches[summer])}')
print(f'Median precip on non-summer rainy days in 2014 (inches): {np.median(inches[rainy & ~summer])}')


# Use | or & when doing a Boolean expression
x = np.arange(10)
print((x > 4) & (x < 8))
