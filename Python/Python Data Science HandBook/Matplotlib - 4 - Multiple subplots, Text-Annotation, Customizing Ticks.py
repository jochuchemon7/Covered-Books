"""
# Multiple Subplots, Text-Annotation and Customizing Ticks
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
plt.style.use('seaborn-white')


# ++ plt.axes: Subplots by Hand ++
ax1 = plt.axes()
ax2 = plt.axes([.65, .65, .2, .2])  # bottom, left, width, height

# ++ With Object Orientated ++
fig = plt.figure()
ax1 = fig.add_axes([.1, .5, .8, .4], xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([.1, .1, .8, .4], ylim=(-1.2, 1.2))
x = np.linspace(0, 10)
ax1.plot(x, np.sin(x), color='blue')
ax2.plot(x, np.cos(x), color='red')

# ++ plt.subplot: Simple Grids of Subplots ++
for i in range(1, 7):
    plt.subplot(2, 3, i)  # passing shape and index for subplot
    plt.text(.5, .5, str((2, 3, i)), fontsize=18, ha='center')

# adjusting the spacing between
fig = plt.figure()
fig.subplots_adjust(hspace=.5, wspace=.5)  # adjusting while on the figure adt
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)  # uses 1-based index
    ax.text(.5, .5, str((2, 3, i)), ha='center', fontsize=18)

# ++ plt.subplots: The Whole Grid in one Go ++
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')  # shared x and y scale on subplots

# axes are in a two-dim array
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)  # uses conventional 0-based index
for i in range(2):
    for j in range(3):
        ax[i, j].text(.5, .5, str((i, j)), ha='center', fontsize=18)


# ++ plt.GridSpec: More Complicated Arrangements ++
grid = plt.GridSpec(2, 3, wspace=.4, hspace=.3)  # grid of 2 rows and 3 cols
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])  # sets room for 2, 3 but can be used for 2, 2
plt.subplot(grid[1, 2])

# EXAMPLE: MULTI-AXES HISTOGRAM

# normal distributed data
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# set up axes
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, wspace=.3, hspace=.3)
main_ax = fig.add_subplot(grid[:3, 1:])
y_hist = fig.add_subplot(grid[:3, 0], sharey=main_ax, xticks=[])
x_hist = fig.add_subplot(grid[3, 1:], sharex=main_ax, yticks=[])

# scatter points
main_ax.plot(x, y, 'ok', markersize=3, alpha=.2)

# histogram attached
x_hist.hist(x, bins=40, histtype='stepfilled', color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, bins=40, histtype='stepfilled', color='gray', orientation='horizontal')
y_hist.invert_xaxis()


# --- Text and Annotation ---
plt.style.use('seaborn-whitegrid')

# -- Example: Holidays on US Births --
# same as previous in pandas
directory = os.getcwd()
births = pd.read_csv(str(directory + '/Data/births.csv'))
quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sig = quartiles[1], .74 * (quartiles[2] - quartiles[0])

births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births.year + 100 * births.month + births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births', index=[births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day) for (month, day) in births_by_date.index]

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)


# + Same but with ax.text() +
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# labeling the graph
style = dict(size=10, color='gray')

ax.text('2012-01-01', 3950, "New Year's Day", **style)
ax.text('2012-07-04', 4250, 'Independence Day', ha='center', **style)
ax.text('2012-9-4', 4850, 'Labor Day', ha='center', **style)
ax.text('2012-10-31', 4600, 'Halloween', ha='center', **style)
ax.text('2012-11-25', 4450, 'Thanksgiving', ha='center', **style)
ax.text('2012-12-25', 3850, 'Christmas', ha='right', **style)

# labeling the axes
ax.set(title='USA births by day of year (1969-1988)', ylabel='average daily births')

# --- Transforms and Text Position ---

# example using ax.transData, ax.transAxes and fig.transFigure for text position coordinates
fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])

ax.text(1, 5, '. Data: (1, 5)', transform=ax.transData)
ax.text(.5, .1, '. Axes: (.5, .1)', transform=ax.transAxes)
ax.text(.2, .2, '. Figure: (.2, .2)', transform=fig.transFigure)

# --- Arrows and Annotations ---

# simple example of ax.annotate()
fig, ax = plt.subplots()
x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy=(6.8, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=.05))
ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle='->', connectionstyle='angle3,angleA=0,angleB=-90'))


# --- Customizing Ticks ---

# + Major and Minor Ticks +
plt.style.use('seaborn-whitegrid')
ax = plt.axes(xscale='log', yscale='log')

# examining the locators
print(ax.xaxis.get_major_locator())
print(ax.xaxis.get_minor_locator())

# examining the formatters
print(ax.xaxis.get_major_formatter())
print(ax.xaxis.get_minor_formatter())


# ++ Hiding Ticks or Labels ++
ax = plt.axes()
ax.plot(np.random.rand(50))

ax.yaxis.set_major_locator(plt.NullLocator())  # hidden ticks
ax.xaxis.set_major_formatter(plt.NullFormatter())  # hidden ticks labels

# Example Hiding Ticks with image plots  (Showing faces images w/o ticks and tick labels)
fig, ax = plt.subplots(5, 5, figsize=(5, 5))
fig.subplots_adjust(hspace=0, wspace=0)

from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images

for i in range(5):
    for j in range(5):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[10 * i + j], cmap='bone')

# ~~ Reducing or Increasing the Number of Ticks ~~
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)  # crowded labels

# SET MAX NUMBER OF DISPLAYABLE TICKS
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))


# ~~ Fancy Tick Formats ~~

# consider
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), color='blue', label='Sine', lw=3)
ax.plot(x, np.cos(x), color='green', label='Cosine', lw=3)

ax.grid(True)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi)
ax.legend(frameon=False, loc='upper right')

# ++ space ticks and grid lines on multiples of pi ++

fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), color='blue', label='Sine', lw=3)
ax.plot(x, np.cos(x), color='green', label='Cosine', lw=3)
ax.grid(True)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi)
ax.legend(frameon=False, loc='upper right')

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

# ++ changing decimal pi values into pi representation ++
def format_func(value, tick_number):
    N = int(np.round(2 * value / np.pi))  # number of multiples of pi/2
    if N == 0:
        return "0"
    elif N == 1:
        return r'$\pi/2$'
    elif N == 2:
        return r'$\pi$'
    elif N % 2 > 0:
        return r'${0}\pi/2$'.format(N)
    else:
        return r'${0}\pi$'.format(N // 2)


fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), color='blue', label='Sine', lw=3)
ax.plot(x, np.cos(x), color='green', label='Cosine', lw=3)
ax.grid(True)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi)
ax.legend(frameon=False, loc='upper right')

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))  # plt.FuncFormatter takes function





