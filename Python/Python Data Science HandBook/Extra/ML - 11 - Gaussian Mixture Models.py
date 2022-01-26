"""
# In Depth: Gaussian Mixture Models
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


# ~~ Using KMeans on blobs of data ~~
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=400, centers=4, cluster_std=.6, random_state=0)
X = X[:, ::-1]  # Flipping axes

kmean = KMeans(n_clusters=4, random_state=0)
labels = kmean.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis')

# ~~ Visualizing cutoff sphere on each KMeans cluster ~~
from scipy.spatial.distance import cdist

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis', zorder=2)

    # Plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=.5, zorder=1))


kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X)

# ~~ KMeans cluster must be circular so clusters don't become muddled ~~
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))

kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)


# --- Generalizing E-M: Gaussian Mixture Models --- (find a mixture of multidimensional Gaussian pdf)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')

# ~~ Finding probabilistic cluster assignments ~~
probs = gmm.predict_proba(X)
print(probs[:5, ].round(3))

# ~~ Visualizing the uncertainty with the probabilities as the size ~~
size = 50 * probs.max(1) ** 2
plt.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap='viridis')


# ~~ Drawing ellipses on the locations and shapes of GMM clusters ~~
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    # Draw an ellipse with the given position and covariance
    ax = ax or plt.gca()

    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit_predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=20, zorder=2)
    ax.axis('equal')

    w_factor = .2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(position=pos, covariance=covar, alpha=w * w_factor)


gmm = GaussianMixture(n_components=4, random_state=0)
plot_gmm(gmm, X)

# ~~ Applying GMM on the stretched data set ~~
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X_stretched)


# --- GMM as Density Estimation ---
from sklearn.datasets import make_moons
Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1])


# ~~ Trying to fit the moon dataset wit GMM ~~ (fails)
gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
plot_gmm(gmm2, Xmoon)

# ~~ Trying with more components and ignoring the cluster labels ~~ (finds overall distribution of data)
gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
plot_gmm(gmm16, Xmoon, label=False)

# ~~ Creating 400 new points form the 16 component GMM pdf ~~
Xnew, Ynew = gmm16.sample(n_samples=400)
plt.scatter(Xnew[:, 0], Xnew[:, 1])

# NOTE: GMM is a convenient as a flexible means of modeling and arbitrary multidimensional distribution of data

# ~~~ How Many Components ~~ (generative model is inherently a pdf of the dataset)

n_components = range(1, 21)
models = [GaussianMixture(n_components=n, covariance_type='full', random_state=0).fit(Xmoon)
          for n in n_components]

# Computing the Bayesian information criterion (BIC) and Akaike information criterion (AIC)

plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')

# Lower AIC and BIC values for optimization


# --- Example: GMM for Generating New Data --- (Generating new hand written digit data)
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

# Printing the first 100 images
def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=.05, wspace=.05)
    for i, axi in enumerate(ax.flat):
        axi.imshow(data[i].reshape(8, 8), cmap='binary')

plot_digits(digits.data)

# Before GMM to avoid dimensional difficulty we start with invertible dimensionality reduction
from sklearn.decomposition import  PCA
pca = PCA(.99, whiten=True)  # Preserving 99 percent of the variance
data = pca.fit_transform(digits.data)
print(data.shape)

# AIC to get a gauge of the number of
n_components = np.arange(50, 210, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0) for n in n_components]
aics = [model.fit(data).aic(data) for model in models]
plt.plot(n_components, aics)

# Choose 110 for number of components that minimizes the AIC value
gmm = GaussianMixture(n_components=110, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

# Draw samples of 100 new points with the 41 dimensional projected space
X_data_new, Y_data_new = gmm.sample(n_samples=100)
print(X_data_new.shape)

# Finally find the inverse transform of the PCA object
digits_new = pca.inverse_transform(X_data_new)
plot_digits(digits_new)
