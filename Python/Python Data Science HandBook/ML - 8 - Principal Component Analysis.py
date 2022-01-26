"""
# In Depth: Principal Component Analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# --- Visualizing two-dimensional data set ---
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1], s=33)
plt.axis('equal')

# ~ Need to learn relationship between X and y ~
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

# Components and explained variance
print(pca.components_)
print(pca.explained_variance_)

# ~ Visualizing as vectors over input data ~
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0, color='black')
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# Plot Data
plt.scatter(X[:, 0], X[:, 1], alpha=.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')


# --- PCA as Dimensionality Reduction ---  (information along the lest important principal axis is removed)
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print(f'Original Shape: {X.shape}')
print(f'Transformed Shape: {X_pca.shape}')


# ~ Inverse transform of the reduce data ~
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=.8, color='blue')
plt.axis('equal')


# --- PCA for visualization: Handwritten digits ---
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

# PCA projection on the 64 dimensional data
pca = PCA(n_components=2)
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

# ~ Plotting the first two principal components ~
plt.scatter(projected[:, 0], projected[:, 1], c=digits.target, edgecolors='none', alpha=.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar()


# --- Choosing the number of components ---
# Looking at the cumulative explained variance ratio
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')


# --- PCA as Noise Filtering ---  (components with larger variance than noise should be unaffected by noise)
def plot_digits(data):
    fig, ax = plt.subplots(4, 10, figsize=(10, 4),
                           subplot_kw={'xticks': [], 'yticks': []},
                           gridspec_kw=dict(wspace=.1, hspace=.1))
    for i, axi in enumerate(ax.flat):
        axi.imshow(data[i].reshape(8, 8), cmap='binary', interpolation='nearest', clim=(0, 16))


plot_digits(digits.data)  # Digits without noise

# ~ Adding Noise ~
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)  # Gaussian random noise added
plot_digits(noisy)  # Digits with noise

# ~ PCA on Noisy Data ~
pca = PCA(n_components=.5).fit(noisy)  # 0-1 range is the percent of variance explained
print(pca.n_components_)

# ~ Compute components and inverse the transformation ~
components = pca.transform(noisy)  # Gather important components
filtered = pca.inverse_transform(components)  # return to original shape
plot_digits(filtered)  # Plotting digits with noise removed


#  ---- Example: Eigenfaces ---
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


# ~~ Look at the principal (150) axes ~~
from sklearn.decomposition import PCA as RandomizedPCA
pca = RandomizedPCA(n_components=150)  # instead of 2914 or (62, 47)
pca.fit(faces.data)

# Visualize the images associated with the first several principal components
fig, ax = plt.subplots(3, 8, figsize=(9, 4), subplot_kw={'xticks': [], 'yticks':[]},
                       gridspec_kw=dict(hspace=.1, wspace=.1))
for i, axi in enumerate(ax.flat):
    axi.imshow(pca.components_[i].reshape(62, 47), cmap='bone')

# Cumulative variance of these components ~
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')


# ~~ Compare input images with images reconstructed from the 150 components ~~
# compute components and transform and inverse transform
pca = RandomizedPCA(n_components=150).fit(faces.data)
transform = pca.transform(faces.data)
projected = pca.inverse_transform(transform)

# Plotting
fig, ax = plt.subplots(2, 10, figsize=(10, 2.5), subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=.1, wspace=.1))
for i in range(10):
    ax[0, i].imshow(faces.images[i], cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
ax[0, 0].set_ylabel('Full-Dim\nInput')
ax[1, 0].set_ylabel('150-Dim\nReconstruction')





