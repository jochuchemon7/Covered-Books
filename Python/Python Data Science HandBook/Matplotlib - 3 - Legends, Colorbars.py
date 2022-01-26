"""
# Legends, Color bars
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.style.use('classic')

# simple legend implementation
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), linestyle='--', color='red', label='Sine')
ax.plot(x, np.cos(x), linestyle='-', color='blue', label='Consine')
ax.axis('equal')
leg = ax.legend()

# specify the location and turn off the frame
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), linestyle='--', color='red', label='Sine')
ax.plot(x, np.cos(x), linestyle='-', color='blue', label='Consine')
ax.axis('equal')
leg = ax.legend(loc='upper left', frameon=False)

# we can use ncol to specify the number of columns in the legend
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), linestyle='--', color='red', label='Sine')
ax.plot(x, np.cos(x), linestyle='-', color='blue', label='Consine')
ax.axis('equal')
ax.legend(frameon=False, loc='lower center', ncol=2)

# using fancy/rounded box or add a shadow or transparency and padding around the text
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), linestyle='--', color='red', label='Sine')
ax.plot(x, np.cos(x), linestyle='-', color='blue', label='Consine')
ax.axis('equal')
ax.legend(fancybox=True, shadow=True, framealpha=1, borderpad=1)

# --- Choosing Elements for the Legend ---

# Creating multiple lines and legend on lines
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, .5))
lines = plt.plot(x, y)  # able to create multiple lines at once
plt.legend(lines[:2], ['first', 'second'])

# best practice
plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2])
plt.legend(framealpha=1, frameon=True)

# +++ Legend for Size of Points +++

# legend on the size of points
directory = os.getcwd()
cities = pd.read_csv(str(directory + '/Data/california_cities.csv'))
print(cities.shape)

lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

# scatter the points using size and color
plt.scatter(lon, lat, label=None, c=np.log10(population), s=area, linewidths=0, alpha=.5, cmap='viridis')
plt.axis('equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)  # color limits

# legend creation
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=.3, s=area, label=str(area)+' km$^2$')
plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title='City Area')
plt.title('California Cities: Area and Population')


# --- Multiple Legends ---

fig, ax = plt.subplots()
lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)
for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2), linestyle=styles[i], color='black')
ax.axis('equal')

# specify the lines and labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'], loc='lower right', frameon=False)

# create the second legend adn add the artist manually
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['Line C', 'Line D'], loc='upper right', frameon=False)
ax.add_artist(leg)  # add legend ADT

# --- Customizing ColorBars ---

# simple version
plt.style.use('classic')
x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I)
plt.colorbar()  # colorbar()

# Customizing
plt.imshow(I, cmap='gray')  # colormap


# ++ Different Color Maps ++
from matplotlib.colors import LinearSegmentedColormap

# (Functions to view color maps: sequential, divergent, qualitative)

def grayscale_cmap(cmap):
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # RGB to grayscale
    RGB_weight = [.299, .587, .114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return LinearSegmentedColormap.from_list(cmap.name + '_gray', colors, cmap.N)

def view_colormap(cmap):
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2), subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])


view_colormap('jet')  # qualitative color map
view_colormap('viridis')  # sequential (even brightness across)
view_colormap('cubehelix')  # rainbow (another good option to viridis)
view_colormap('RdBu')  # divergent (good for pos and negative values no good greyscale translation)

# ++ Color limits and extensions ++

# image subject to noise
speckles = (np.random.random(I.shape) < .01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')  # extensions to indicate values are above or below the limits
plt.clim(-1, 1)  # set color limits


plt.subplot(1, 1, 1)
plt.imshow(I, cmap='RdBu')
plt.clim(-1, 1)
plt.colorbar()

# +++ Discrete ColorBars +++
plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))  # Representing discrete values with 6 bins
plt.colorbar()
plt.clim(-1, 1)


# --- Example: Handwritten Digits ---

# load and visualize data
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)
fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi, in enumerate(ax.flat):  # first 64 digits
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])

# Two-Dimensional mani-fold learning  (dimensional reduction)
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)  # projecting digits into 2 dimensions

# Plot the results
plt.scatter(projection[:, 0], projection[:, 1], lw=.1,
            c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(label='digit value', ticks=range(6))
plt.clim(-.5, 5.5)


