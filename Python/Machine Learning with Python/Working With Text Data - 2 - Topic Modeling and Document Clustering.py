"""
#  Topic Modeling and Document Clustering
"""

from sklearn.datasets import load_files

review_train = load_files('aclImdb/train/')
text_train, y_train = review_train.data, review_train.target
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

review_test = load_files('aclImdb/test/')
text_test, y_test = review_test.data, review_test.target
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

# --- Topic Modeling and Document Clustering ---

# -- Latent Dirichlet Allocation (LDA) -- (Find groups of words that appear together frequently)
# Each Document can be understood as a mixture of a subset of topics like components from PCA

# On our data we will remove 20 percent of words and limit the bag-of-words
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=10000, max_df=.15)
X_train = vect.fit_transform(text_train)

# - Using LDA with batch learning method -  (Dimensional Reduction)
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10, learning_method='batch', max_iter=25, random_state=0)
document_topics = lda.fit_transform(X_train)

print(f'lda.components_.shape: {lda.components_.shape}')  # (n_topics, n_words)

# Viewing the most important words for each of the topics
import numpy as np
import mglearn

sorting = np.argsort(lda.components_, axis=1)[:, ::-1]  # Invert rows to make descending sort
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names, sorting=sorting,
                           topics_per_chunk=5, n_words=10)


# - Next Using 100 topics on the LDA -
lda100 = LatentDirichletAllocation(n_components=100, learning_method='batch', max_iter=25, random_state=0)
document_topics100 = lda100.fit_transform(X_train)

# Viewing the words of some topics
topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=topics, feature_names=feature_names, sorting=sorting,
                           topics_per_chunk=7, n_words=20)

# - Confirming topics by intuition by looking at highest ranking words for each topic (e.i. 45) -

music = np.argsort(document_topics100[:, 45])[::-1]  # From high to low vals
for i in music[:10]:
    print(b".".join(text_train[i].split(b".")[:2])+b".\n")  # First two sentences

# - Plotting the name of each topic by the two most common words and the weight of the topic -

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
topic_names = ["{:>2}".format(i) + " ".join(words)
               for i, words in enumerate(feature_names[sorting[:, :2]])]

for col in [0, 1]:
    start = col * 50
    end = (col + 1) * 50
    ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
    ax[col].set_yticks(np.arange(50))
    ax[col].set_yticklabels(topic_names[start:end], ha='left', va='top')
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 2000)
    yax = ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()

