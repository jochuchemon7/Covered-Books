"""
# Simple Scatter Plots, Errors and Density and Contour
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import matplotlib as mpl
import numpy as np

# --- Simple Scatter Plots ---
plt.style.use('seaborn-whitegrid')

# using plt.plot
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color='black')

# common marker styles
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker, label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8)

# character codes with line and color codes
plt.plot(x, y, '-ok')

# specify a wide range or properties
plt.plot(x, y, '-p', color='gray', markersize=15, linewidth=4, markerfacecolor='white',
         markeredgecolor='red', markeredgewidth=2)
plt.ylim(-1.2, 1.2)

# = Scatter Plots with plt.scatter
plt.scatter(x, y, marker='o')

# scatter with color and size as variables
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.randn(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis')
plt.colorbar()  # show color scale

# using iris data
iris = load_iris()
features = iris.data.T  # transpose from entry to variable

plt.scatter(features[0], features[1], alpha=.2, s=100*features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])


#  --- Visualizing Errors ---

# = Basic ErrorBars =
x = np.linspace(0, 10, 50)
dy = .8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='.k')

# making error bars lighter than the points
plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)

# = Continuous Errors =
# using plt.fill_between()
a = np.linspace(0, 2*3.14, 50)
b = np.sin(a)
# x-value, lower y & upper y
plt.fill_between(a, b, 0, where=(a > 2) & (a <= 3), color='gray', alpha=.2)
plt.plot(a, b)

# --- Density and Contour Plots ---

# = Visualizing a Three-Dimensional Function Within a Two-Dimensional Plot =

def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)  # Builds two-dimensional grids from one-dimensional arrays
Z = f(X, Y)

# + USING STANDARD LINE-ONLY CONTOUR PLOT +
plt.contour(X, Y, Z, colors='black')  # negative are dashed lined
plt.contour(X, Y, Z, 20, cmap='RdGy')  # more lines and cmap red-gray

# + USING CONTOURF()
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()

# using imshow() to interpret a two-dimensional grid as an image
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy')  # origin changed
plt.colorbar()
plt.axis('image')

# Labeled contours on top of an image
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=.5)
plt.colorbar()

# --- Histograms, Binnings and Density ---

# Simple histogram
plt.style.use('seaborn-white')
data = np.random.randn(1000)
plt.hist(data)

# more customized histogram
plt.hist(data, bins=30, density=True, alpha=.5, histtype='stepfilled', color='steelblue',
         edgecolor=None)

# comparing histograms over distributions
x1 = np.random.normal(0, .8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=.3, density=True, bins=30)  # setting arguments
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)

# to count the entries in a given bin
counts, bin_edges = np.histogram(data, bins=5)
print(counts)
print(bin_edges)

# = Two-Dimensional Histograms and Binnings =

# x and y array from multivariate Gaussian distribution
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T

# = Plotting two dimensional hist =
plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')

# np.histogram2d
counts, x_edges, y_edges = np.histogram2d(x, y, bins=30)
print(counts)
print(x_edges)
print(y_edges)

# = Hexagonal Binnnings =

plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='Count in Bin')

# = Kernel density estimation for evaluating densities in multiple dimensions =
from scipy.stats import gaussian_kde

data = np.vstack([x, y])
kde = gaussian_kde(data)  # fit an array of size [Ndim, Nsamples]

# evaluate on regular grid
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# plot the result on image
plt.imshow(Z.reshape(Xgrid.shape), origin='lower', aspect='auto', extent=[-3.5, 3.5, -6, 6],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label('density')


