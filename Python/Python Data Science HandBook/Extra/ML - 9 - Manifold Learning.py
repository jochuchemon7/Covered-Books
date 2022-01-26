"""
# In Depth: Manifold Learning
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
sns.set()

# --- Manifold Learning: "HELLO" ---

# ~ Generate some two-dimensional data that can be use to define a manifold ~
def make_hello(N=1000, rseed=42):
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(.5, .4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    directory = os.getcwd()
    fig.savefig(str(directory + '/Data/hello.png'))
    plt.close(fig)

    from matplotlib.image import imread
    data = imread(str(directory + '/Data/hello.png'))[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]


# ~ Calling and Visualizing the resulting data ~
X = make_hello(1000)
colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal')


# --- Multidimensional Scaling (MDS) ---
# Trying to rotate the the data
def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)

X2 = rotate(X, 20) + 5
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal')

# ~ Create a distance matrix for N points NxN such that (i, j) contains distance from i and j ~
from sklearn.metrics import pairwise_distances
D = pairwise_distances(X)
print(D.shape)

# Visualize the distance matrix 
plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar()

# Constructing a distance matrix for our rotated and translated data, it is the same as the original
D2 = pairwise_distances(X2)
np.allclose(D, D2)

# ~ Recovering the D-dimensional coordinate representation form NxN ~
from sklearn.manifold import MDS
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal')

# --- MDS as Manifold Learning ----

# 3D dimensional data
def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.rand(dimension, dimension)
    e, V = np.linalg.eig(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])

X3 = random_projection(X, 3)
print(X3.shape)

# ~ Visualize the 3D points ~
from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], **colorize)
ax.view_init(azim=70, elev=50)

# We can compute the distance
model = MDS(n_components=2, random_state=1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal')


# --- NonLinear Embeddings: Where MDS Fails ---

# ~ Function to take input and contorts it into an 'S' shape in three dimensions ~
def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * .75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T


XS = make_hello_s_curve(X)
print(XS.shape)

# ~ Plotting the S shape  (Non Linear Hello; in S-shape (z-axis)) ~
from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter(XS[:, 0], XS[:, 1], XS[:, 2], **colorize)

# ~ Trying with simple MDS algorithm ~
from sklearn.manifold import MDS
model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)
plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis('equal')

# NOTE: The best two-dimensional linear embedding does not wrap around the S-curve

# --- NonLinear Manifolds: Locally Linear Embedding ---

# NOTE: Instead of distance from every point to every point we do nearby points say the nearest 100 points

# LocalLinearEmbedding: with 'modified LLE'
from sklearn.manifold import LocallyLinearEmbedding
model = LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='modified', eigen_solver='dense')
out = model.fit_transform(XS)

fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(.15, -.15)

# NOTE: Manifold advantage lays in their ability to preserve nonlinear relationships in the data
# Generally used after PCA

# - For toy problems (eg S) use LLE
# - For high-dimensional data Isomap
# - For highly clustered data use (t-SNE)  TNSE


# --- Example: Isomap on Faces ---
# Getting data
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=30)
print(faces.data.shape)

# ~ Visualize the Data ~
fig, ax = plt.subplots(nrows=4, ncols=8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='gray')

# ~ Start with PCA to examine explained variance ratio to get an idea of number of linear features requred ~
from sklearn.decomposition import PCA
model = PCA(n_components=100).fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('N components')
plt.ylabel('Variance Explained')  # Data cannot be described linearly with just a few components

# ~~ Using NonLinear Manifolds ~~
from sklearn.manifold import Isomap
model = Isomap(n_components=2)
proj = model.fit_transform(faces.data)
print(proj.shape)

# ~~ Function that will output the image thumbnails at the locations of the projections ~~
from matplotlib import offsetbox

def plot_components(data, model, images=None, ax=None, thumb_frac=.05, cmap='gray'):
    ax = ax or plt.gca()

    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')

    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        show_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - show_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # Don't show points that are too close
                continue
            show_images = np.vstack([show_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), proj[i])
            ax.add_artist(imagebox)


fig, ax = plt.subplots(figsize=(10, 10))
plot_components(faces.data, model=Isomap(n_components=2), images=faces.images[:, ::2, ::2])

# --- Example: Visualizing Structure in Digits ---
from sklearn.datasets import fetch_openml
mnist = fetch_openml('MNIST_784', version=1)
print(mnist.data.shape)

# ~ Plotting digits ~
fig, ax = plt.subplots(nrows=6, ncols=8, subplot_kw={'xticks':[], 'yticks': []},
                       gridspec_kw=dict(hspace=.1, wspace=.1))
for i, axi in enumerate(ax.flat):
    image = mnist.data.iloc[1250 * i]
    image = np.array(image.values).reshape(28, 28)
    axi.imshow(image, cmap='gray_r')

# ~~ Performing Manifold ~~
data = mnist.data[::30]
target = mnist.target[::30]
target = np.array(target, dtype=int)


# ~ Isomap and visualizing ~
model = Isomap(n_components=2)
proj = model.fit_transform(data)
plt.scatter(proj[:, 0], proj[:, 1], c=target, s=5, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-.5, 9.5)

# ~ Gain more info by looking at a single number at a time to reduce the crowdedness ~
from sklearn.manifold import Isomap

# Choose 1/4 of the "1" digits
selected = np.array(mnist.target == str(1), dtype=bool)
data = np.array(mnist.data, dtype=float)
data = data[selected]

fig, ax = plt.subplots(figsize=(10, 10))
model = Isomap(n_neighbors=5, n_components=2, eigen_solver='dense')
plot_components(data=data, model=model, images=data.reshape((-1, 28, 28)), ax=ax, thumb_frac=.05, cmap='gray_r')


