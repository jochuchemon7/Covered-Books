"""
# Unsupervised Learning - 4 - Clustering
"""
import matplotlib.pyplot as plt
import numpy as np
import mglearn

# --- K-Means Clustering ---
mglearn.plots.plot_kmeans_algorithm()
mglearn.plots.plot_kmeans_boundaries()


# ~~ On make_blobs dataset ~~
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(random_state=1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Labels
print(f'Cluster Memberships: \n{kmeans.labels_}')

# Predict X
print(kmeans.predict(X))

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=60, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', alpha=.5, s=80)

# We can also use more or fewer cluster centers
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
labels2 = kmeans.labels_
kmeans = KMeans(n_clusters=5).fit(X)
labels5 = kmeans.labels_
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(X[:, 0], X[:, 1], c=labels2, s=60, cmap='rainbow')
ax[1].scatter(X[:, 0], X[:, 1], c=labels5, s=60, cmap='rainbow')


# ~ Failure cases of K-Means ~

# Plotting K-Means of 3 clusters with legend
X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 1.5, .5], random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
scatter = plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred, s=60, cmap='rainbow', label=y_pred)
legend = ax.legend(*scatter.legend_elements(), loc='best', title='Clusters')
ax.add_artist(legend)

# Plotting where separated data groups are stretched in diagonal (fails on non-spherical clusters)
X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)

transformation = rng.normal(size=(2, 2))  # mu=0 and std=1
X = np.dot(X, transformation)

kmeans = KMeans(n_clusters=3).fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', c='black',
            s=100, linewidths=2, cmap=mglearn.cm3)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')

# ~ K-Means on the Moon data set ~ (Fails on non spherical data shape)
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=.05, random_state=0)

kmeans = KMeans(n_clusters=2).fit(X)
y_pred = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', c='black',
            s=100, linewidths=2, cmap=mglearn.cm3)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')


# ~~ PCA, NMF and K-Means Comparison with Faces Data Set ~~
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape[0], dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people/255.0
print(f'X_people Features: {X_people.shape[1]}')

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

nmf = NMF(n_components=100, random_state=0, max_iter=10000)  # Reduce to 100 components
pca = PCA(n_components=100, random_state=0)
kmeans = KMeans(n_clusters=100, random_state=0)
nmf.fit(X_train)
pca.fit(X_train)
kmeans.fit(X_train)

X_test_pca = pca.transform(X_test)
y_test_kmeans = kmeans.predict(X_test)
X_test_nmf = nmf.transform(X_test)

X_reconstructed_pca = pca.inverse_transform(X_test_pca)
X_reconstructed_kmeans = kmeans.cluster_centers_[y_test_kmeans]
X_reconstructed_nmf = nmf.inverse_transform(X_test_nmf)  # np.dot(X_test_nmf, nmf.components_)

# Extracted Components
image_shape = people.images[0].shape
fig, ax = plt.subplots(3, 5, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
fig.suptitle('First Five Extracted Components ')
for i, axi in enumerate(ax.T):
    axi[0].imshow(kmeans.cluster_centers_[i, :].reshape(image_shape), cmap='gray')
    axi[1].imshow(pca.components_[i, :].reshape(image_shape), cmap='viridis')
    axi[2].imshow(nmf.components_[i, :].reshape(image_shape))
ax[0, 0].set_ylabel('K-Means')
ax[1, 0].set_ylabel('PCA')
ax[2, 0].set_ylabel('NMF')

# Reconstructions
fig, ax = plt.subplots(4, 5, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
fig.suptitle('First Five Test Faces Reconstructed')
for i, axi in enumerate(ax.T):
    axi[0].imshow(X_test[i, :].reshape(image_shape), cmap='gray')
    axi[1].imshow(X_reconstructed_kmeans[i, :].reshape(image_shape), cmap='gray')
    axi[2].imshow(X_reconstructed_pca[i, :].reshape(image_shape), cmap='gray')
    axi[3].imshow(X_reconstructed_nmf[i, :].reshape(image_shape), cmap='gray')
ax[0, 0].set_ylabel('Original Image')
ax[1, 0].set_ylabel('K-Means')
ax[2, 0].set_ylabel('PCA')
ax[3, 0].set_ylabel('NMF')


# ~ Moons Data Set with K-Means using more clusters ~ (Many clusters to cover the variation in a complex dataset)
X, y = make_moons(n_samples=200, noise=.05, random_state=0)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired', s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60,
            marker='^', c=range(kmeans.n_clusters), linewidths=2, cmap='Paired', edgecolors='black')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
print(f'Cluster Membership: {y_pred}')

# Without Using the 10-dimensional representation it would not be possible to separate the two half-moon shape using a linear model

# ~ More expressive representation of the data using the distances to each of the cluster centers are feature with transform method ~

# More expressive representation using distance to each of the cluster centers
distances_features = kmeans.transform(X)
print(f'Distance Features Shape: {distances_features.shape}')
print(f'Distance Features: \n{distances_features}')

