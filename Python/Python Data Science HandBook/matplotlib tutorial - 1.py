import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simple plot
x = np.linspace(0, 2, 100)
plt.plot(x, x, label='linear')
plt.plot(x, x ** 2, label='square')
plt.plot(x, x ** 3, label='cubic')
plt.legend()
plt.title('Simple Plot')
plt.xlabel('x label')
plt.ylabel('y label')


# Creating a function
def my_plotter(axes, data1, data2, param_dict):
    out = axes.plot(data1, data2, **param_dict)
    return out

data1, data2, data3, data4 = np.random.randn(4, 100)
fig, ax = plt.subplots(1, 1)
my_plotter(ax, data1, data2, {'marker': 'x'})


# with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2)
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})


# --- PyPlot Tutorial ---

# - 1 -
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()


# - 2 -
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()


# - 3 - (red circles) and x and y axis
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()


# - 4 -
t = np.arange(0, 5, .2)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()


# - 5 -  (s = shape & c = color) attributes

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.rand(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', data=data, c='c', s='d')
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()


# - 6 - (plotting categorical variables) with subplot
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.title('Categorical Plotting')
plt.show()


# - 7 - (multiple figures and axes)
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0, 5, .1)
t2 = np.arange(0, 5, .02)

plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')


# - 8 - (histogram and text)
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=.75)
plt.xlabel('Smarts')
plt.ylabel('probability')
plt.title('Histogram of IQ')
plt.text(50, .025, r'$\mu = 100,\ \sigma = 15$')
plt.grid(True)
plt.show()


# - 9 - (annotating text)
ax = plt.subplot()
t = np.arange(0, 5, .01)
s = np.cos(2 * np.pi * t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=.05),)
plt.ylim(-2, 2)
plt.show()


# -- logarithmic and other nonlinear axes --

y = np.random.normal(loc=.5, scale=.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.title('linear')
plt.yscale('linear')
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.title('log')
plt.yscale('log')
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthresh=.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)

plt.subplots_adjust(top=.92, bottom=.08, left=.1, right=.95, hspace=.25,
                    wspace=.35)

plt.show()
