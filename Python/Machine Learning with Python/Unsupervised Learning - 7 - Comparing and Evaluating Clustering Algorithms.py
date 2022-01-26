"""
# Comparing and Evaluating Clustering Algorithms
"""

# --- Evaluating Clustering with Ground Truth --- (ARI and NMI)

# ~~ Adjusted Random Index (ARI) ~~
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score  # ARI
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import mglearn

X, y = make_moons(n_samples=200, noise=.05, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Algo list and random cluster assignment for reference
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]
random_state = np.random.RandomState(seed=0)
random_cluster = random_state.randint(0, 2, len(X))  # Create a random cluster assignment

# Plotting cluster from: Random, KMeans, Agglomerative and DBSCAN
fig, ax = plt.subplots(1, 4, figsize=(15, 3), subplot_kw=dict(xticks=[], yticks=[]))
ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_cluster, cmap=mglearn.cm3, s=60)
ax[0].set_title(f'Random Assignment - ARI: {round(adjusted_rand_score(y, random_cluster), 3)}')

for i in range(1, 4):
    cluster = algorithms[i-1].fit_predict(X_scaled)
    ax[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster, cmap=mglearn.cm3, s=60)
    ax[i].set_title(f'{algorithms[i-1].__class__.__name__} - ARI : {round(adjusted_rand_score(y, cluster), 3)}')


# ~ Using adjusted_rand_score avoids exact matching instead which points are in the same cluster ~
from sklearn.metrics import accuracy_score

cluster1 = [0, 0, 1, 1, 0]
cluster2 = [1, 1, 0, 0, 1]
print(f'Accuracy: {round(accuracy_score(cluster1, cluster2), 3)}')  # 0.0
print(f'ARI: {round(adjusted_rand_score(cluster1, cluster2), 3)}')  # 1.0


# ~~ Evaluating Clustering Without Ground Truth ~~
from sklearn.metrics import silhouette_score

X, y = make_moons(n_samples=200, noise=.05, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

random_state = np.random.RandomState(seed=0)
random_cluster = random_state.randint(0, 2, len(X))
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

fig, ax = plt.subplots(1, 4, figsize=(15, 3), subplot_kw=dict(xticks=[], yticks=[]))
ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_cluster, cmap=mglearn.cm3, s=60)
ax[0].set_title(f'Random Assignment - Silhouette: {round(silhouette_score(X_scaled, random_cluster), 3)}')

for i in range(1, 4):
    cluster = algorithms[i-1].fit_predict(X_scaled)
    ax[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster, cmap=mglearn.cm3, s=60)
    ax[i].set_title(f'{algorithms[i-1].__class__.__name__} : {round(silhouette_score(X_scaled, cluster), 3)}')


# NOTE: an alternative to silhouette is the robustness base clustering metrics

# --- Comparing Algorithms on the Faces DataSets ---
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
image_shape = people.images[0].shape
mask = np.zeros(people.data.shape[0], dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]

# PCA (Dimensional reduction)
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)

# -- Using: DBSCAN (Cluster Algorithm) --
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print(f'Unique Labels {np.unique(labels)}')  # All noise

# DBSCAN min_samples modified
dbscan = DBSCAN(min_samples=3)
labels = dbscan.fit_predict(X_pca)
print(f'Unique Labels: {np.unique(labels)}')

# DBSCAN eps modified
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print(f'Unique Labels: {np.unique(labels)}')

# Only 27 noise points
print('Number of points per cluster: {}'.format(np.bincount(labels + 1)))

noise = X_people[labels == -1]

# Noise faces
fig, axes = plt.subplots(3, 9, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(12, 4))
for image, ax in zip(noise[:27, ], axes.ravel()):
    ax.imshow(image.reshape(image_shape), cmap='gray')


# ~ Finding more clusters among different eps values ~
for eps in [1, 3, 5, 7, 9, 11, 13]:
    print(f'\neps={eps}')
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print(f'Clusters Presented: {np.unique(labels)}')
    print(f'Cluster Size: {np.bincount(labels + 1)}')


# ~ Notice eps = 7 looks most interesting many clusters even thou they are small ~
# Plotting the images in the clusters found with eps=7 and min_sample=3
dbscan = DBSCAN(eps=7, min_samples=3)
labels = dbscan.fit_predict(X_pca)
for cluster in range(max(labels)+1):
    mask = cluster == labels
    n_images = np.sum(mask)
    fig, ax = plt.subplots(1, n_images, figsize=(n_images*1.5, 4),
                           subplot_kw=dict(xticks=[], yticks=[]))
    for image, label, axi in zip(X_people[mask], y_people[mask], ax.flat):
        axi.imshow(image.reshape(image_shape), cmap='gray')
        axi.set_title(people.target_names[label].split()[-1])


# -- Using KMeans (Cluster Algorithm) --

# Using 10 clusters
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
print(f'Cluster sizes  K-Means: {np.bincount(labels_km)}')

# Since using PCA representation we need to inverse_transform to view cluster centers as images
fig, ax = plt.subplots(2, 5, figsize=(12, 4), subplot_kw=dict(xticks=[], yticks=[]))
for center, axi in zip(km.cluster_centers_, ax.flat):
    image = pca.inverse_transform(center)
    axi.imshow(image.reshape(image_shape), cmap='gray')


# NOTE: k-means partitions every point, no concept of noise points. Larger number of clusters finds finer distinctions but harder on manual inspection

# -- Using Agglomerative Algorithm (Cluster Algorithm) --
agg = AgglomerativeClustering(n_clusters=10)
labels_agg = agg.fit_predict(X_pca)
print(f'Cluster Sizes Agglomerative Clustering: {np.bincount(labels_agg)}')

# Measure if the two partitions between KMeans and Agglomerative are similar
print(f'Ari: {adjusted_rand_score(labels_agg, labels_km)}')


# ~~ Plotting the dendrogram ~~
from scipy.cluster.hierarchy import ward, dendrogram
linkage_array = ward(X_pca)
plt.figure(figsize=(20, 5))
dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True)
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')


# ~~ Visualize the 10 clusters, first couple of points in each cluster (show number of points in each cluster to the left)

n_clusters = 10
for cluster in range(n_clusters):
    mask = labels_agg == cluster
    fig, ax = plt.subplots(1, 10, figsize=(5, 2), subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].set_ylabel(np.sum(mask))
    for image, label, asdf, axi in zip(X_people[mask], y_people[mask], labels_agg[mask], ax.flat):
        axi.imshow(image.reshape(image_shape), cmap='gray')
        axi.set_title(people.target_names[label].split()[-1], fontdict={'fontsize': 9})


# NOTE: each row is one cluster; the number to the left is the number of images in each cluster

# ~ Run again Agglomerative with 40 clusters for more homogeneous clusters ~
agglomerative = AgglomerativeClustering(n_clusters=40)
labels_agg = agglomerative.fit_predict(X_pca)
print(f'Cluster sizes agglomerative clustering: {np.bincount(labels_agg)}')

image_shape = people.images[0].shape
n_clusters = 40
for cluster in range(n_clusters):
    mask = labels_agg == cluster
    fig, ax = plt.subplots(1, 15, figsize=(8, 3), subplot_kw=dict(xticks=[], yticks=[]))
    cluster_size = np.sum(mask)
    ax[0].set_ylabel('#{}: {}'.format(cluster, cluster_size))
    for image, label, axi in zip(X_people[mask], y_people[mask], ax.flat):
        axi.imshow(image.reshape(image_shape), cmap='gray')
        axi.set_title(people.target_names[label].split()[-1], fontdict=dict(fontsize=9))
    for i in range(cluster_size, 15):
        ax[i].set_visible(False)


