"""
# In Depth: K-Means Clustering
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# --- Introducing K-Means ---  (Clustering algorithms rather than dimensionality reduction)
# ~ Sample Data ~
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=.6, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=30)

# ~ KMeans algorithm ~
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
model.fit(X)
y_kmeans = model.predict(X)

# ~ Visualize KMeans result and centers ~
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='viridis')
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', alpha=.4, s=200)

# --- KMeans Algorithm: Expectation-Maximization (E-M) ---

# ~ Basic Implementation of KMeans algorithm ~
from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    # 1 Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        labels = pairwise_distances_argmin(X, centers)  # 2a Assign labels based on closest center
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])  # 2b find new centers from mean

        if np.all(centers == new_centers):  # Check for convergence
            break
        centers = new_centers

    return centers, labels


centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)

# ~~ Caveats of expectation-maximization ~~

# ~ Using a different random seed can lead to poor results ~
centers, labels = find_clusters(X, 4, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)

# ~ Changing number of clusters to 6 (expected 6 clusters instead of 5) ~
labels = KMeans(n_clusters=6, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)

# ~ K-Means is limited to linear cluster boundaries (Fail classification) ~
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=.05, random_state=0)
labels = KMeans(n_clusters=2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')

# ~ Kernelized K-Means with SpectralClustering ~ (graph of nearest neighbors to computer higher dimension of data)
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(X)  # Non-linear clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')


# --- Example 1: K-Means on digits ---
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

# ~ Clustering ~
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)

# ~ Plotting the cluster centers as the 'typical' digit within the cluster ~
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape((10, 8, 8))
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

# ~ Fixing permuted labels by matching each cluster with true label ~
from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

from sklearn.metrics import accuracy_score
print(accuracy_score(digits.target, labels))

# ~ confusion matrix ~
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_true=digits.target, y_pred=labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')

# ~~ Using manifold algorithm t-SNE to reduce dimensionality and then KMeans on 2d plot~~
# Projecting the data
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0, init='pca')
digits_proj = tsne.fit_transform(digits.data)

# Computing the clusters
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

# Permute the labels (index to value)
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

print(accuracy_score(y_true=digits.target, y_pred=labels))


# --- Example 2: K-Means for color compression ---
from sklearn.datasets import load_sample_image
china = load_sample_image('china.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)
print(china.shape)  # (h, w, RGB)

# ~ Reshape to [n_samples x n_features] ~  (fo a cloud of points in 3D color space)
data = china / 255.
data = data.reshape(427 * 640, 3)
print(data.shape)

# ~ Visualize the pixels in the color space ~
def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data

    # Choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    plt.suptitle(title, size=20)

plot_pixels(data, title='Input color space: 16 million possible colors')


# ~ Reducing 16 million possible colors to 16 colors with KMeans ~
from sklearn.cluster import MiniBatchKMeans  # for large data set (uses data subsets)
kmeans = MiniBatchKMeans(n_clusters=16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors, title='Reduced color space: 16 colors')

# ~ Recoloring the original pixels each pixel is assigned the color of its closest cluster center ~

china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-Color Image', size=16)



