"""
# Unsupervised Learning - 5 - Agglomerative Clustering
"""
import mglearn

# --- Agglomerative Clustering --- (each point is a cluster and merges the two most similar clusters until reaches desired number of clusters)
# NOTE: No predictions on new test data available
# Ward: (least increase in variance within all clusters)
# Average: (merges two clusters with the smallest average distance between all their points)
# Complete: (merges two clusters with the smallest maximum distance between their points)

mglearn.plots.plot_agglomerative_algorithm()

# ~~ Agglomerative Clustering in the blobs data set ~~
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
scatter = plt.scatter(X[:, 0], X[:, 1], c=assignment, label=assignment)
legend = ax.legend(*scatter.legend_elements(), loc='best', title='Clusters')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')


# ~ Hierarchical Clustering and Dendrograms ~

mglearn.plots.plot_agglomerative()  # Hierarchical view of the clusters created on each step

# ~ Using Scipy to encode the hierarchical cluster and plot the dendrogram ~
from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)

# Apply ward clustering to X, returning the distances bridged when performing agglomerative clustering
linkage_array = ward(X)

# Plotting the dendrogram from the array containing the distances between clusters
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, 'two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, 'three clusters', va='center', fontdict={'size': 15})
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')

# NOTE: each line represents the distance apart the clusters are; However Agglomerative clustering still fails with two-moons data set
