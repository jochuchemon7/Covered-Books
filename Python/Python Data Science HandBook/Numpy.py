"""
### NUMPY CHAPTER 2 TUTORIAL
"""
import numpy as np

result = 0
for i in range(100):
    result += i
print(f'result: {result}')


# list (int)
L = list(range(10))
print('L: ', L)
type(L[0])

# List (str)
L2 = [str(s) for s in L]
print('L2: ', L2)
type(L2[0])

# heterogeneous lists
L3 = [12, 'one', True, 38.4]
[print(type(s)) for s in L3]

# Fixed-Types of Arrays
import array
L = list(range(10))
A = array.array('i', L)  # i indicates 'data type'


# Arrays from python lists
np.array([1, 2, 3, 4, 5])
np.array([1, 2, 3, 4, 5], dtype='float32')
np.array([range(i, i+3) for i in [1, 2, 3]])

# Arrays from scratch
np.zeros(10, dtype=int)
np.ones((3, 5), dtype=float)
np.full((3, 3), 7)
np.arange(0, 20, 2)  # from 0 to 20 in steps of 2
np.linspace(0, 1, 5)  # from 0 to 1 in 5 parts

np.random.random((3, 3))  # Uniformly distributed between 0 and 1
np.random.normal(0, 1, (3, 3))  # normal distribution mu = 0 and std = 1

np.random.randint(0, 10, (3, 3))  # from 0 to 10
np.eye(3, 3)

np.empty(3)
