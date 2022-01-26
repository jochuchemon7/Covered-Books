"""
# DBSCAN - Density Based Spacial Clustering of Applications with Noise
"""

# NOTE: Identifies crowded regions of many points close together and points in the dense region are core samples
# if less than min_samples points within distance eps of the starting point then is labeled noise (belonging to no-one) if more then is a core sample and assigned a new cluster label
# At the end you have: core points, boundary points and noise points

# --- DBSCAN ---
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import mglearn

X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print(f'Cluster Membership: {clusters}')  # -1 stands for noise label

# NOTE: eps -> how close points should be if distance is lower or equal to param then points are consider neighbors
# min_samples -> min cluster size
mglearn.plots.plot_dbscan()

# ~~ DBSCAN with Make Moons Data Set ~~
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200, noise=.05, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')



