# coding: utf-8
from __future__ import division
from IPython import get_ipython
from backports.shutil_get_terminal_size import get_terminal_size as _get_terminal_size

# # NumPy Basics: Arrays and Vectorized Computation

# In[ ]:

get_ipython().magic(u'matplotlib inline')

# In[ ]:

from numpy.random import randn
import numpy as np
np.set_printoptions(precision=4, suppress=True)


# ## The NumPy ndarray: a multidimensional array object

# In[ ]:

data = randn(2, 3)
data

# In[ ]:

data * 10

# In[ ]:

data + data

# In[ ]:

data.shape

# In[ ]:

data.dtype

# ### Creating ndarrays

# In[ ]:

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1

# In[ ]:

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2

# In[ ]:

arr2.ndim

# In[ ]:

arr2.shape


# In[ ]:

arr1.dtype

# In[ ]:

arr2.dtype


# In[ ]:

np.zeros(10)

# In[ ]:

np.zeros((3, 6))

# In[ ]:

np.empty((2, 3, 2))


# In[ ]:

np.arange(15)

# ### Data Types for ndarrays

# In[ ]:

arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)

# In[ ]:

arr1.dtype

# In[ ]:

arr2.dtype


# In[ ]:

arr = np.array([1, 2, 3, 4, 5])
arr.dtype

# In[ ]:

float_arr = arr.astype(np.float64)
float_arr.dtype

# In[ ]:

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr

# In[ ]:

arr.astype(np.int32)


# In[ ]:

numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)


# In[ ]:

int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype)


# In[ ]:

empty_uint32 = np.empty(8, dtype='u4')
empty_uint32


# ### Operations between arrays and scalars

# In[ ]:

arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr

# In[ ]:

arr * arr

# In[ ]:

arr - arr

# In[ ]:

1 / arr

# In[ ]:

arr ** 0.5


# ### Basic indexing and slicing

# In[ ]:

arr = np.arange(10)
arr

# In[ ]:

arr[5]

# In[ ]:

arr[5:8]

# In[ ]:

arr[5:8] = 12
arr

# In[ ]:

arr_slice = arr[5:8]
arr_slice

# In[ ]:

arr_slice[1] = 12345
arr_slice

# In[ ]:

arr_slice[:] = 64
arr_slice

# In[ ]:

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]

# In[ ]:

arr2d[0][2]
arr2d[0, 2]

# In[ ]:

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d


# In[ ]:

arr3d[0]


# In[ ]:

old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d

# In[ ]:

arr3d[0] = old_values
arr3d


# In[ ]:

arr3d[1, 0]


# #### Indexing with slices

# In[ ]:

arr[1:6]


# In[ ]:

arr2d


# In[ ]:

arr2d[:2]

# In[ ]:

arr2d[:2, 1:]


# In[ ]:

arr2d[1, :2]

# In[ ]:

arr2d[2, :1]


# In[ ]:

arr2d[:, :1]


# In[ ]:

arr2d[:2, 1:] = 0


# ### Boolean indexing

# In[ ]:

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = randn(7, 4)

# In[ ]:
names

# In[ ]:
data

# In[ ]:

names == 'Bob'

# In[ ]:

data[names == 'Bob']

# In[ ]:

data[names == 'Bob', 2:]

# In[ ]:

data[names == 'Bob', 3]

# In[ ]:

names != 'Bob'

# In[ ]:

data[~(names == 'Bob')]


# In[ ]:

mask = (names == 'Bob') | (names == 'Will')
mask

# In[ ]:

data[mask]

# In[ ]:

data[data < 0] = 0
data


# In[ ]:

data[names != 'Joe'] = 7
data


# ### Fancy indexing

# In[ ]:

arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr

# In[ ]:

arr[[4, 3, 0, 6]]

# In[ ]:

arr[[-3, -5, -7]]


# In[ ]:

# more on reshape in Chapter 12
arr = np.arange(32).reshape((8, 4))
arr

# In[ ]:

arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# In[ ]:

arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]

# In[ ]:

arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]


# ### Transposing arrays and swapping axes

# In[ ]:

arr = np.arange(15).reshape((3, 5))
arr

# In[ ]:

arr.T


# In[ ]:

arr = np.random.randn(6, 3)
np.dot(arr.T, arr)


# In[ ]:

arr = np.arange(16).reshape((2, 2, 4))
arr
arr.transpose((1, 0, 2))


# In[ ]:

arr

# In[ ]:

arr.swapaxes(1, 2)


# ## Universal Functions: Fast element-wise array functions

# In[ ]:

arr = np.arange(10)
np.sqrt(arr)

# In[ ]:

np.exp(arr)


# In[ ]:

x = randn(8)
x

# In[ ]:

y = randn(8)
y

# In[ ]:

np.maximum(x, y) # element-wise maximum


# In[ ]:

arr = randn(7) * 5
arr

# In[ ]:

np.modf(arr)


# ## Data processing using arrays

# In[ ]:

points = np.arange(-5, 5, 0.01) # 1000 equally spaced points
xs, ys = np.meshgrid(points, points)
ys


# In[ ]:

from matplotlib.pyplot import imshow, title


# In[ ]:

import matplotlib.pyplot as plt
z = np.sqrt(xs ** 2 + ys ** 2)
z
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")


# In[ ]:

plt.draw()


# ### Expressing conditional logic as array operations

# In[ ]:

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# In[ ]:

result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result


# In[ ]:

result = np.where(cond, xarr, yarr)
result


# In[ ]:

arr = randn(4, 4)
arr

# In[ ]:

np.where(arr > 0, 2, -2)

# In[ ]:

np.where(arr > 0, 2, arr) # set only positive values to 2

# In[ ]:

# Not to be executed

#result = []
#for i in range(n):
#    if cond1[i] and cond2[i]:
#        result.append(0)
#    elif cond1[i]:
#        result.append(1)
#    elif cond2[i]:
#        result.append(2)
#    else:
#        result.append(3)


# In[ ]:

# Not to be executed

#np.where(cond1 & cond2, 0,
#         np.where(cond1, 1,
#                  np.where(cond2, 2, 3)))


# In[ ]:

# Not to be executed

#result = 1 * cond1 + 2 * cond2 + 3 * -(cond1 | cond2)


# ### Mathematical and statistical methods

# In[ ]:

arr = np.random.randn(5, 4) # normally-distributed data
arr

# In[ ]:

arr.mean()

# In[ ]:

np.mean(arr)

# In[ ]:

arr.sum()

# In[ ]:

arr.mean(axis=1)

# In[ ]:

arr.sum(0)


# In[ ]:

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(0)
arr.cumprod(1)


# ### Methods for boolean arrays

# In[ ]:

arr = randn(100)
(arr > 0).sum() # Number of positive values


# In[ ]:

bools = np.array([False, False, True, False])
bools.any()

# In[ ]:

bools.all()

# ### Sorting

# In[ ]:

arr = randn(8)
arr

# In[ ]:

arr.sort()
arr


# In[ ]:

arr = randn(5, 3)
arr

# In[ ]:

arr.sort(1)
arr


# In[ ]:

large_arr = randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))] # 5% quantile


# ### Unique and other set logic

# In[ ]:

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


# In[ ]:

sorted(set(names))


# In[ ]:

values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])


# ## File input and output with arrays

# ### Storing arrays on disk in binary format

# In[ ]:

arr = np.arange(10)
np.save('some_array', arr)

# In[ ]:

arr1 = np.load('some_array.npy')
arr1

# In[ ]:

np.savez('array_archive.npz', a=arr, b=arr)


# In[ ]:

arch = np.load('array_archive.npz')
arch['b']


# In[ ]:

#get_ipython().system(u'rm some_array.npy')
#get_ipython().system(u'rm array_archive.npz')


# ### Saving and loading text files

# In[ ]:

#get_ipython().system(u'cat array_ex.txt')

# In[ ]:

arr = np.loadtxt('array_ex.txt', delimiter=',')
arr

# ## Linear algebra

# In[ ]:

x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print "x =\n", x
print "y =\n", y
z = x.dot(y)  # equivalently np.dot(x, y)
print "x . y =\n", z

# In[ ]:

np.dot(x, np.ones(3))


# In[ ]:

np.random.seed(12345)


# In[ ]:

from numpy.linalg import inv, qr
X = randn(5, 5)
mat = X.T.dot(X)
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)
r

# In[ ]:

from numpy.linalg import solve

A = [[1,1],[1, -1]]
b = [2, 0]

x = solve(A, b)
x

# ## Random number generation

# In[ ]:

samples = np.random.normal(size=(4, 4))
samples


# In[ ]:

from random import normalvariate
from IPython import get_ipython

ipython_shell = get_ipython()
N = 1000000
get_ipython().magic(u'timeit samples = [normalvariate(0, 1) for _ in xrange(N)]')
get_ipython().magic(u'timeit np.random.normal(size=N)')

# In[ ]:

# ## Example: Random Walks
import random
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

plt.plot(walk)

# In[ ]:

np.random.seed(12345)


# In[ ]:

nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
plt.plot(walk)

# In[ ]:

walk.min()
walk.max()


# In[ ]:

(np.abs(walk) >= 10).argmax()


# ### Simulating many random walks at once

# In[ ]:

nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks


# In[ ]:

means = np.mean(walks, axis=1)
print walks.shape
print means
print walks.max()
print walks.min()
plt.hist(means, bins=30)

# In[ ]:

hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum() # Number that hit 30 or -30


# In[ ]:

crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()
crossing_times
plt.hist(crossing_times)

# In[ ]:

steps = np.random.normal(loc=0, scale=0.25,
                         size=(nwalks, nsteps))


# In[ ]:



