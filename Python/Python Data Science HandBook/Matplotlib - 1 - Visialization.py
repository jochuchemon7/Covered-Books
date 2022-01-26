""""
# Visualization with Matplotlib
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# plt.style.use('classic')

# Plotting from a script (you van run myplot.py on the command-line
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()  # only use show() once and most often at the end

# You can save a figure
fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')
fig.savefig('Data/my_figure.png')

# = MATLAB-style interface =
plt.figure()  # plot figure
plt.subplot(2, 1, 1)  # (rows, columns, panel number)
plt.plot(x, np.sin(x))

plt.subplot(2, 1, 2)  # create second panel and set current axis
plt.plot(x, np.cos(x))

# reference current figure
print(plt.gcf())  # get current figure
print(plt.gca())  # get current axis

# = Object-orientated interface =
fig, ax = plt.subplots(2)  # ax is array of to Axes objects

ax[0].plot(x, np.sin(x))  # plot method on appropriate object
ax[1].plot(x, np.cos(x))

# --- Simple Line Plots ---
plt.style.use('seaborn-whitegrid')

# figure and axis
fig = plt.figure()  # can be thought as a single container
ax = plt.axes()

# simple sinusoid
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)

ax.plot(x, np.sin(x))
plt.plot(x, np.sin(x))  # alternatively using pylab interface

# figure with multiple lines
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

# --- Adjusting the Plot: Line Colors and Styles ---

# adjusting the color
plt.plot(x, np.sin(x-0), color='blue')
plt.plot(x, np.sin(x-1), color='g')
plt.plot(x, np.sin(x-2), color='0.75')  # gray scale
plt.plot(x, np.sin(x-3), color='#FFDD44')
plt.plot(x, np.sin(x-4), color=(1.0, 0.2, 0.3))  # FBG tuple values 0-1
plt.plot(x, np.sin(x-5), color='chartreuse')  # HTML color names

# adjusting the line style
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')

# Shorted codes
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--')  # dashed
plt.plot(x, x + 6, linestyle='-.')  # dashdot
plt.plot(x, x + 7, linestyle=':')  # dotted

# linestyle and color codes combined
plt.plot(x, x + 0, '-g')
plt.plot(x, x + 1, '--c')
plt.plot(x, x + 2, '-.k')
plt.plot(x, x + 2, ':r')

# = Adjusting the Plot: Axes Limits =
# simplest way
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

# reversed axis
plt.plot(x, np.sin(x))
plt.xlim(10, 0)
plt.ylim(1.2, -1.2)  # reversed y axis

# using plt.axis()
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5])

# automatic tighten bounds
plt.plot(x, np.sin(x))
plt.axis('tight')

# axis can ensure equal aspect ratio
plt.plot(x, np.sin(x))
plt.axis('equal')

# --- Labeling Plots ---
# titles and axis labels  (position, size and style can be adjust)
plt.plot(x, np.sin(x))
plt.title('A Sine Curve')
plt.xlabel('X')
plt.ylabel('sin(x)')

# specifying label on legends
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')
plt.legend()

# = Matplotlib Gotchas =
# from MATLAB-style to object orientated style methods
"""
plt.xlabel() -> ax.set_xlabel()
plt.ylabel() -> ax.set_ylabel()
plt.xlim() -> ax.set_xlim()
plt.ylim() -> ax.set_ylim()
plt.title() -> ax.set_title()
"""

# example using ax.set instead of calling each function
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2), title='A Simple Plot', ylabel='sin(x)', xlabel='x')
