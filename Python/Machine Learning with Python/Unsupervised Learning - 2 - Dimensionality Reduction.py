from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
import mglearn

# --- PCA ---
mglearn.plots.plot_pca_illustration()

# ~ Applying per-class feature histogram on cancer data set ~
cancer = load_breast_cancer()

fig, ax = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]
ax = ax.flat

for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())

ax[0].set_xlabel('Feature Magnitude')
ax[0].set_ylabel('Frequency')
ax[0].legend(['Malignant', 'Benign'], loc='best')
fig.tight_layout()


# Applying PCA to the cancer data (scaled first)

from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)
x_pca = pca.transform(X_scaled)
print(f'Scaled Shape: {X_scaled.shape}')
print(f'Reduced Shape: {x_pca.shape}')

# Plotting the first two components
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(x_pca[:, 0], x_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc='best')
plt.gca().set_aspect('equal')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# The PCA Components  (Each row is a principal component and columns the original feature)
print(f'PCA component shape: {pca.components_.shape}')
print(f'PCA components: \n{pca.components_}')

# ~ Visualize the coefficients using heat map ~
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ['First Component', 'Second Component'])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel('Feature')
plt.ylabel('Principal Components')


# ~ PCA Eigen-faces for feature extraction ~
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
image_shape = people.images[0].shape

# Plotting
fig, ax = plt.subplots(2, 5, figsize=(15, 8), subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(people.images[i], cmap='gray')
    axi.set_title(people.target_names[people.target[i]])

# Shape
print(f'people.images.shape: {people.images.shape}')
print(f'Number of Classes: {len(people.target_names)}')

# Count of images per target name
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if(i + 1) % 3 == 0:
        print()

# Take up to 50 images of each person
mask = np.zeros(people.target.shape[0], dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people/255.0

for target in np.unique(people.target):
    mask[np.where(people.target == target)] = True

# ~ Using KNN to classify images ~ (Looks for the most similar face with 1 Neighbor )
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print(f'Test Score of 1-nn: {round(knn.score(X_test, y_test), 3)}')

# PCA with whitening (Rescales the principal components to have the same scale)
mglearn.plots.plot_pca_whitening()

# ~ Using PCA on the faces data set ~
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)  # We want 100 principal components
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Reduced dimension from 5655 to 100
print(f'X_train.shape: {X_train.shape}')
print(f'X_train_pca.shape: {X_train_pca.shape}')

# Use KNN on the new reduced data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print(f'Test Set Accuracy: {round(knn.score(X_test_pca, y_test), 3)}')

# ~ Plotting the first couple principal components and the shape ~
print(f'pca.components: {pca.components_.shape}')

fig, ax = plt.subplots(3, 5, figsize=(15, 12), subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(pca.components_[i].reshape(people.images[0].shape), cmap='viridis')
    axi.set_title('{}. component'.format((i+1)))

# NOTE: You can also use some Xn numbers to as a coefficients of the PC sort of weights on them

# ~ Restoring the original data from a set number of principal components ~ (using pca.inverse_transform)
image_shape = people.images[0].shape
mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

# NOTE: More details bc more components are being added

# ~ Scatter plot the first 2 principal components of X_train_pca and label them by target ~
mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')


# --- Non_Negative Matrix Factorization --- (Components and coefficients to be non-negative)
# NOTE: Data must be non-negative; it can identify original components for the overall data; more interpretable components

# Apply NMF to synthetic data
mglearn.plots.plot_nmf_illustration()

# NOTE: NMF usually not so good for reconstruction e.i. inverse_transformation
mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)

# ~ Extracting 15 components from the data using Non-Negative Matrix Factorization; NMF) ~
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

# All Components are of positive values
fig, ax = plt.subplots(3, 5, figsize=(15, 12), subplot_kw=dict(xticks=[], yticks=[]))
image_shape = people.images[0].shape
for i, axi in enumerate(ax.flat):
    axi.imshow(nmf.components_[i].reshape(image_shape), cmap='gray')
    axi.set_title("{}. Component".format(i))

# ~~ Images for which these components are particularly strong ~~

# + Sort by 3rd component, plot first 10 images +
compn = 3
inds = np.argsort(X_train_nmf[:, compn])[::-1]  # index of descending sort

fig, ax = plt.subplots(2, 5, figsize=(15, 8), subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in zip(inds, ax.flat):
    axi.imshow(X_train[i].reshape(image_shape), cmap='gray')

# + Sort by 7th component, plot first 10 images + (Faces with a large coefficient for component 7)
compn = 7
inds = np.argsort(X_train_nmf[:, compn])[::-1]

fig, ax = plt.subplots(2, 5, figsize=(15, 13), subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in zip(inds, ax.flat):
    axi.imshow(X_train[i].reshape(image_shape), cmap='gray')

# ~~ Example With Synthetic Data ~~
S = mglearn.datasets.make_signals()
plt.figure(figsize=(7, 1))
plt.plot(S, '-')
plt.xlabel('Time')
plt.ylabel('Signal')

# ~ We want to recover the decomposition of the mixed signal into the original components ~

# Mix data into a 100-dimensional state (add noise)
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print(f'Shape of measurements: {X.shape}')

# Use NMF to recover the three signals
nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print(f'Recovered Signal Shape: {S_.shape}')

# Apply PCA for comparison
pca = PCA(n_components=3)
H = pca.fit_transform(X)
print(f'Recovered Signal Shape: {H.shape}')

# ~ Plotting NMF and PCA results ~
models = [X, S, S_, H]
names = ['Observations (First Three Measurements)', 'True Sources', 'NMF Recovered Signals',
         'PCA Recovered Signals']

fig, ax = plt.subplots(4, 1, figsize=(13, 7), subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.set_title(names[i])
    axi.plot(models[i][:, :3], '-')



