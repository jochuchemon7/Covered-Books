from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mglearn
import pandas

mglearn.plots.plot_scaling()  # StandardScaler, RbustScaler, MinMaxScaler and Normalizer


# --- Applying Data Transformations ---
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)

# ~ Importing the class that implements the preprocessing scaling ~
# (MinMaxScaler; subtracts the training set minimum and divides by the training set range)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled = scaler.fit_transform(X_train)  # Same as both functions combined

# Compared MinMaxScaler transformation
print(f'Transformed Shaped: {X_train_scaled.shape}')
print(f'Per-feature minimum before scaling: \n{X_train.min(axis=0)}')
print(f'per-feature maximum before scaling: \n{X_train.max(axis=0)}')
print(f'per-feature minimum after scaling: \n{X_train_scaled.min(axis=0)}')
print(f'per-feature maximum after scaling: \n{X_train_scaled.max(axis=0)}')

# Scale test data as well
X_test_scaled = scaler.transform(X_test)
print(f'per-feature minimum after scaling: \n{X_test_scaled.min(axis=0)}')
print(f'per-feature maximum after scaling: \n{X_test_scaled.max(axis=0)}')


# --- Scaling Training and Test Data the Same Way (Plotting) --- Plotting scaled data example
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
X_train, X_test = train_test_split(X, random_state=0, test_size=.1)

fig, ax = plt.subplots(1, 3, figsize=(15, 7))

ax[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label='training set', s=60, marker='o')
ax[0].scatter(X_test[:, 0], X_test[:, 1], c=mglearn.cm2(1), label='testing set', s=60, marker='^')
ax[0].legend(loc='upper left')
ax[0].set_title('Original Data')

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

ax[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
ax[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
ax[1].set_title("Scaled Data")

# DO NOT DO THIS!!
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

ax[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label='Training Set', s=60)
ax[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', c=mglearn.cm2(1), label='Test Set', s=60)
ax[2].set_title('Improperly Scaled Data')

for axi in ax:
    axi.set_xlabel('Feature 0')
    axi.set_ylabel('Feature 1')


# --- Effect of Preprocessing on Supervised Learning --- (MinMaxScaler comparison with cancer data set)
from sklearn.svm import SVC

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

svm = SVC(C=100)
svm.fit(X_train, y_train)
print(f'Training Accuracy: {round(svm.score(X_test, y_test), 2)}')


# preprocessing with MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print(f'Test Accuracy: {round(svm.score(X_test_scaled, y_test), 3)}')


# preprocessing with StandardScaler (Every class has a mean of 0 and std of 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm.fit(X_train_scaled, y_train)
print(f'Test Accuracy: {round(svm.score(X_test_scaled, y_test), 2)}')


