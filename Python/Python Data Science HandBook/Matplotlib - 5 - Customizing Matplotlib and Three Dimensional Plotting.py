"""
# Customizing Matplotlib
"""
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('classic')

# --- Plot Customization by Hand ---
x = np.random.randn(1000)
plt.hist(x)

# + Adjust by hand to more visually pleasing +
ax = plt.axes(facecolor='#E6E6E6')  # gray background
ax.set_axisbelow(True)

plt.grid(color='w', linestyle='solid')  # solid white grid lines

for spine in ax.spines.values():  # hide axis spines
    spine.set_visible(False)

ax.xaxis.tick_bottom()  # hide top and right ticks
ax.yaxis.tick_left()

ax.tick_params(color='gray', direction='out')  # lighten ticks and labels
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')

ax.hist(x, edgecolor='#E6E6E6', color='#EE6666')  # face and edge color


# --- Changing the Defaults: rcParams ---
from matplotlib import cycler
colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])

# setting global parameters
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

plt.hist(x)

# simple lines with rc parameters
for _ in range(4):
    plt.plot(np.random.rand(10))


# --- StyleSheets ---
# list of styles
print(plt.style.available[:5])  # you can use with to enclose a sheet-style over a plot

# function for two basics types of plot
def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for _ in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')


# ~~ Default Style ~~
plt.style.use('default')
hist_and_lines()  # view default

# FiveThirtyEight style
plt.style.use('fivethirtyeight')
hist_and_lines()

# ggplot style
plt.style.use('ggplot')
hist_and_lines()

# bayesian Methods for Hacker style
plt.style.use('bmh')
hist_and_lines()

# dark background style
plt.style.use('dark_background')
hist_and_lines()

# grayscale style
plt.style.use('grayscale')
hist_and_lines()

# seaborn style
plt.style.use('seaborn')
hist_and_lines()

# --- Three-Dimensional Plotting in Matplotlib ---
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')  # passing '3d' keyword for 3d plotting

# ~~ Three-Dimensional Points and Lines ~~
ax = plt.axes(projection='3d')

# data for 3d line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# data for 3d scatter  points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + .1 * np.random.randn(100)
ydata = np.cos(zdata) + .1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')


# ~~ Three-Dimensional Contour Plots ~~
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


x = np.linspace(-6, 6, 30)  # x and y data
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)  # mesh x and y for z on given function
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# changing the initial viewing angle
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(60, 35)

# ~~ Wireframes and Surface Plots ~~

# Wireframe example

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('WireFrame')

# Surface plot Example
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor=None, rstride=1, cstride=1)
ax.set_title('Surface')


# Partial Polar Grid for surface plot
r = np.linspace(0, 6, 20)
theta = np.linspace(-.9 * np.pi, .8 * np.pi, 40)
r, theta = np.meshgrid(r, theta)

X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor=None)


# ~~ Surface Triangulations ~~

# scatter the data
theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)

ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, c=z, cmap='viridis', linewidth=.5)

# ax.plot.trisurf() places surfaces on sets of triangles formed between adjacent points
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor=None)


# --- Example: Mobious Strip ---
theta = np.linspace(0, 2 * np.pi, 30)
w = np.linspace(-.25, .25, 8)
w, theta = np.meshgrid(w, theta)

# building the strip data
phi = .5 * theta
r = 1 + w * np.cos(phi)  # radius in x-y plane
x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(w * np.sin(phi))

# plotting the object
from matplotlib.tri import Triangulation
tri = Triangulation(np.ravel(w), np.ravel(theta))

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, cmap='viridis', linewidths=.2, triangles=tri.triangles)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)


