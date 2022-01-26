"""
# Intro to Scikit-Learn
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Data Representation ---
iris = sns.load_dataset('iris')
print(iris.head())
sns.set()
sns.pairplot(iris, hue='species', size=1.5)

# extracting the feature matrix and target values
X_iris = iris.drop('species', axis=1)
print(X_iris.shape)
y_iris = iris['species']
print(y_iris.shape)


# ~~ Supervise Learning Example: LR ~~
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)

# Select the model (linear regression)
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
print(model)

# arranging data into a features matrix and target vector
X = x[:, np.newaxis]
print(X.shape)

# fit to the model
model.fit(X, y)
print(model.coef_)
print(model.intercept_)

# predict data
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

# visualize the results
plt.scatter(x, y)
plt.plot(xfit, yfit)


# ~~ Supervised learning: Iris Classification ~~
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model))


# ~~ Unsupervised Learning: Iris Dimensionality ~~
from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(X_iris)
X_2D = model.transform(X_iris)  # data transformed into 2 dimension

# plotting the results
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot(x='PCA1', y='PCA2', hue='species', data=iris, fit_reg=False)

# ~~ Unsupervised Learning: Iris Clustering ~~
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=3, covariance_type='full')
model.fit(X_iris)
y_gmm = model.predict(X_iris)

# plotting the results
iris['cluster'] = y_gmm
sns.lmplot(x='PCA1', y='PCA2', data=iris, hue='cluster', fit_reg=False)

# subplots
g = sns.FacetGrid(iris, col='cluster', hue='species')
g.map(plt.scatter, 'PCA1', 'PCA2', s=10)
g.add_legend()


# --- Application: Handwritten Digits ---
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.images.shape)

fig, ax = plt.subplots(10, 10, figsize=(8,8), subplot_kw={'xticks': [], 'yticks': []},
                       gridspec_kw=dict(hspace=.4, wspace=.1))
for i, ax in enumerate(ax.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(.05, .05, str(digits.target[i]), transform=ax.transAxes, color='green')

# splitting data
X = digits.data
print(X.shape)

y = digits.target
print(y.shape)

# ~~ Unsupervised Learning: Dimensionality Reduction ~~  (Manifold Algorithm)
# Dimensionality reduction
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
print(data_projected.shape)

# plotting the reduced dimensional data
plt.scatter(x=data_projected[:, 0], y=data_projected[:, 1], c=digits.target, edgecolor='none', alpha=.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-.5, 9.5)

# ~~ Classification on Digits ~~
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model))

# Building a Confusion Matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')

# Plotting with digits with predicted label
fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=.1, wspace=.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(.05, .05, str(y_model[i]), color='green' if (ytest[i] == y_model[i]) else 'red')

