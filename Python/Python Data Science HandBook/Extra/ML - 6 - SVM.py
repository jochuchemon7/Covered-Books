"""
# In Depth: Support Vector Machines
"""
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
sns.set()

# ~ Consider classification task ~

# Data
from sklearn.datasets._samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

# Drawing multiple line separators
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, .65), (.5, 1.6), (-.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1.3, 3.5)

# --- SVM: Maximizing the Margin ---

# Example of adding a margin
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, .65,.33), (.5, 1.6, .55), (-.2, 2.9, .2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=.4)  # Margin
plt.xlim(-1, 3.5)

# ~~ Fitting a SVM ~~
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

# ~~ Function to plot SVM decision boundaries ~~

def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=.5, linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support is True:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='autumn')
plot_svc_decision_function(model)  # Calling created plotting function

print(model.support_vectors_)

# Only points in the margin affect the model
def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=.6)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
fig.subplots_adjust(left=.0625, right=.95, wspace=.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title(f'N = {N}')


# --- Kernel SVM ---
# from In Depth: LM when higher dimensional space defined by polynomials and Gaussian basis functions to fit a non linear
from sklearn.datasets._samples_generator import make_circles
X, y = make_circles(n_samples=100, factor=.1, noise=.1)
clf = SVC(kernel='linear').fit(X, y)

# Poorly perform for nonlinear boundaries
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)


# ~~ Computing a radial basis function ~~
r = np.exp(-(X ** 2).sum(axis=1))  # radial basis function centered on the middle clump

# Visualize the 3D data
from mpl_toolkits import mplot3d
def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

plot_3D(X=X, y=y)

# ++ kernel transformation: compute basis function centered at every point and let SVM shift through results ++

# ~ Applying Radial Basis Function Kernel ~
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=300, lw=1, facecolors='none')

# --- Tuning the SVM: Softening Margins ---
# Overlapping data points
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=0)
plt.scatter(X[:, 0],  X[:, 1], c=y, s=50, cmap='autumn')

# ~ Comparison of C value for margin  (high C = small margin ; low C = larger margin) ~
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=.8)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
fig.subplots_adjust(left=.0625, right=.95, wspace=.1)

for axi, C in zip(ax, [10.0, .1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model=model, ax=axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1], s=300, lw=1, facecolors='none')
    axi.set_title(f'C = {C}', size=14)


# --- Example: Face Recognition ---
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.keys())
print(faces.target_names)
print(faces.images)

# ~ Plotting the faces ~
fig, ax = plt.subplots(nrows=3, ncols=5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])

# Pipeline with preprocessor (PCA reduce to 150 components) and classifier
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

pca = RandomizedPCA(n_components=150, random_state=42, whiten=True)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# Train and Test Data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

# Using Grid Cross-Validation for best parameters
from sklearn.model_selection import GridSearchCV
import time
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [.0001, .0005, .001, .005]}
grid = GridSearchCV(estimator=model, param_grid=param_grid)

t = time.time()
grid.fit(Xtrain, ytrain)
runtime = time.time() - t
print(f'Runtime: {runtime}')
print(f'Best Parameters {grid.best_params_}')


# with the best params we can predict the labels for the test data
model = grid.best_estimator_
yfit = model.predict(Xtest)

# display the test images and their predicted values
fig, ax = plt.subplots(nrows=4, ncols=6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape((62, 47)), cmap='bone')
    axi.set(yticks=[], xticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)


# ~ Classification Report (statistics on the labels) ~
from sklearn.metrics import classification_report
print(classification_report(y_true=ytest, y_pred=yfit, target_names=faces.target_names))

# ~ Confusion Matrix ~
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_true=ytest, y_pred=yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')



