"""
# Numpy Computation
"""
# Simple reciprocal computation
import numpy as np
import time
np.random.seed(0)


def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output


values = np.random.randint(1, 10, 5)
print(compute_reciprocals(values))

# Larger input
big_array = np.random.randint(1, 100, 1000000)
t = time.perf_counter()
compute_reciprocals(big_array)
runtime = time.perf_counter() - t
print(round(runtime, 3), ' seconds')

# Comparing function vs performing an operation on the array
compute_reciprocals(values)
print(1/ values)

# Much faster when operations are done on the array
t = time.time()
(1 / big_array)
runtime = time.time() - t
print(runtime)

np.arange(5) / np.arange(1, 6)  # Between two arrays

x = np.arange(9).reshape((3, 3))  # Multidimensional
print(2 ** x)

# --- Numpy's UFuncs ---
x = np.arange(4)
print(f'x = {x}')
print(f'x + 5 = {x + 5}')
print(f'x - 5 = {x - 5}')
print(f'x * 2 = {x * 2}')
print(f'x / 2 = {x / 2}')
print(f'x // 2 = {x // 2}')

# additional unary ufunc's
print(f'-x = {-x}')
print(f'x ** 2 = {x ** 2}')
print(f'x % 2 = {x % 2}')

# Standard order of operations is respected
print(-(.5*x + 1) ** 2)

# There are all wrappers around specific functions built into numpy
np.add(x, 2)

# Absolute value
x = np.array([-2, -1, 0, 1, 2])
abs(x)
np.abs(x)
np.absolute(x)


# = Trigonometric functions =
theta = np.linspace(0, np.pi, 3)
print(f'theta: {theta}')
print(f'sin(theta): {np.sin(theta)}')
print(f'cos(theta): {np.cos(theta)}')
print(f'tan(theta): {np.tan(theta)}')

# = Inverse Trigonometric functions =
x = [-1, 0 , 1]
print(f'x: {x}')
print(f'arcsin(x): {np.arcsin(x)}')
print(f'arccos(x): {np.arccos(x)}')
print(f'arctan(x): {np.arctan(x)}')

# = Exponents and Logarithms =
x = [1, 2, 3]
print(f'x = {x}')
print(f'e^x = {np.exp(x)}')
print(f'2^x = {np.exp2(x)}')
print(f'3^x = {np.power(3, x)}')

x = [1, 2, 4, 10]
print(f'x = {x}')
print(f'ln(x) = {np.log(x)}')
print(f'log2(x) = {np.log2(x)}')
print(f'log10(x) = {np.log10(x)}')


# = Specialized ufunc's =

from scipy import special

# Gamma functions
x = [1, 5, 10]
print(f'gamma(x) = {special.gamma(x)}')
print(f'ln|gamma(x) = {special.gammaln(x)}')
print(f'beta(x, 2) = {special.beta(x, 2)}')

# Error function (Integral of Gaussian); with complement and inverse
x = np.array([0, .3, .7, 1])
print(f'erf(X) = {special.erf(x)}')
print(f'erfc(x) = {special.erfc(x)}')
print(f'erfinv(x) = {special.erfinv(x)}')


# --- Advance Ufunc features ---

# Specify the array where the result of calculation will be stored
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

# Write results of a computation to every other element of a specify array
y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

# = Aggregates =
x = np.arange(1, 6)
np.add.reduce(x)  # add on all elements  (np.sum)
np.multiply.reduce(x)  # multiply all elements (np.prod)

np.add.accumulate(x)  # To store intermediate steps  (np.cumsum)
np.multiply.accumulate(x)

# = Outer products = (output of all pairs )
x = np.arange(1, 6)
np.multiply.outer(x, x)


# --- Aggregations: Min, Max, etc ---

#  = Summing values =
L = np.random.random(100)
print(np.sum(L))
print(sum(L))

# Comparing summing
big_array = np.random.random(1000000)
t = time.time()
sum(big_array)
runtime1 = time.time() - t

t = time.time()
np.sum(big_array)
runtime2 = time.time() - t
print(f'sum() time: {runtime1} seconds')
print(f'np.sum() time: {runtime2} seconds')


# = Minimum and Maximum =
min(big_array)
max(big_array)

print(f'np.min(big_array): {np.min(big_array)}')
print(f'np.max(big_array): {np.max(big_array)}')

# Other aggregates
print(f'min: {big_array.min()} max: {big_array.max()}  sum: {big_array.sum()}')


# Multidimensional aggregates
M = np.random.random((3, 4))
print(M)
print(M.sum())  # Entire array
print(M.sum(axis=0))  # Sum on columns
print(M.sum(axis=1))  # Sum on rows

# --- Example: Average Height Data Set ---
import os
import pandas as pd
data = pd.read_csv(str(os.getcwd() + '/Data/president_heights.csv'))
heights = np.array(data['height(cm)'])
print(heights)

# Summary Statistics
print(f'Mean Height: {heights.mean()}')
print(f'Standard Deviation: {heights.std()}')
print(f'Minimum height: {heights.min()}')
print(f'Maximum height: {heights.max()}')

# Quantiles
print(f'25th percentile: {np.percentile(heights, 25)}')
print(f'Median: {np.median(heights)}')
print(f'75th percentile: {np.percentile(heights, 75)}')


# Visual representation
import matplotlib.pyplot as plt
plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')

