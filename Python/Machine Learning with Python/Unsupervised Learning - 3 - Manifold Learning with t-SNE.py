"""
# Manifold Learning with t-SNE
"""

# --- Manifold Learning with t-SNE --- (visualization algo for complex mapping use manifold learning algo)

# NOTE: t-SNE as a manifold algorithm is best used for training visualization by preserving the info indicating which points are neighbors to each other

# ~~ On Hand Written Digit Numbers ~~
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.images.shape)

fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')


# ~ Using PCA for Dimensionality Reduction ~
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(digits.data)

digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683",
          "#4E655E", "#853541", "#3A3120", "#535D8E"]

plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())

for i in range(len(digits.data)):
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color=colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# NOTE: Classes still overlap

# ~ Using t-SNE ~
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)

for i in range(len(digits_tsne)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color=colors[digits.target[i]])

plt.xlabel('t-SNE Feature 0')
plt.ylabel('t-SNE Feature 1')


# NOTE: t-SNE is great for finding a two-dimensional representation of the data that preserves
# the distances between points as best as possible

