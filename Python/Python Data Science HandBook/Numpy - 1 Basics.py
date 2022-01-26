"""
# Numpy array Attributes
"""

import numpy as np

np.random.seed(0)

x1 = np.random.randint(10, size=(6))
x2 = np.random.randint(10, size=(3, 3))
x3 = np.random.randint(10, size=(2, 3, 3))

print(f'x3 ndim: {x3.ndim}')
print(f'x3 shape: {x3.shape}')
print(f'x3 size: {x3.size}')

print(f'x3 dtype: {x3.dtype}')

print(f'x3 itemsize: {x3.itemsize} bytes')  # Each element size
print(f'x3 nbytes: {x3.nbytes} bytes')  # Entire array size


# --- Accessing single elements ---

print(x1)
print(x1[0])
print(x1[4])

print(x1[-1])  # Last value
print(x1[-2])  # Second to last value

# multidimensional
print(x2)
print(x2[0, 0])
print(x2[2, 0])
print(x2[2, -1])

x2[0, 0] = 12


# --- slicing ---
x = np.arange(10)
print(x)
print(x[:5])
print(x[5:])
print(x[4:7])

print(x[::2])  # Every other element
print(x[1::2])  # Every other element starting at index 1

print(x[::-1])  # reversed
print(x[5::-2])  # reversed every other starting at index 5

# --- multidimensional subarrays ---
print(x2)
print(x2[:2, :2])  # 2 rows and 2 columns
print(x2[:, ::2])  # all rows and every other column

print(x2[::-1, ::-1])  # reversed

# = slices are views not copies =
print(x2)
x2_sub = x2[:2, :2]
print(x2_sub)
x2_sub[0, 0] = 99
print(x2_sub)
print(x2)

# = Explicitly copying =
x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)
x2_sub_copy[0, 0] = 42
print(x2_sub_copy)
print(x2)


# --- Reshaping Arrays ---
grid = np.arange(1, 10).reshape((3, 3))
print(grid)

x = np.array([1, 2, 3])
print(x)
x.reshape((1, 3))  # one dimensional row into a 1x3 array
x.reshape((3, 1))


# --- Array Concatenation ---
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])

z = np.array([99, 99, 99])
np.concatenate([x, y, z])  # Concatenate more than 2 arrays at once

grid = np.array([[1, 2, 3], [4, 5, 6]])
np.concatenate([grid, grid])  # 2D concat on column
np.concatenate([grid, grid], axis=1)  # concat on row

x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7], [6, 5, 4]])
np.vstack([x, grid])  # Vertical concat

y = np.array([[99], [99]])
np.hstack([y, grid])  # Horizontal concat


# --- Splitting of Arrays ---
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])  # Split at indices 3 and 5
print(x1, x2, x3)

grid = np.arange(16).reshape((4, 4))
print(grid)
upper, lower = np.vsplit(grid, [2])  # Vertical split on row 2
print(upper)
print(lower)

left, right = np.hsplit(grid, [2])  # Horizontal split on column 2
print(left)
print(right)


