"""
# In Depth: Naive Bayes Classification and Linear Regression
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# --- In Depth: Gaussian Naive Bayes ---
# Assuming data from each label is drawn from a simple Gaussian distribution
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')


# Gaussian with no covariance between dimensions
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

rng = np.random.RandomState(0)  # new data
Xnew = [-6, -14] + [14, 18] * rng.randn(2000, 2)
ynew = model.predict(Xnew)

# plotting new data
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=.1)
plt.axis(lim)

# Probabilistic classification
yprob = model.predict_proba(Xnew)
print(yprob[-8:].round(3))  # last 8 prob round to 2 decimal


# --- Multinomial Naive Bayes ---
# Example: classifying text (probability of observing counts among a number of categories)
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
[print(name) for name in data.target_names]

# select a few categories for training and testing
categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print(train.data[5])

# Using Term Frequency - Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
labels = model.predict(test.data)

# Building a Confusion Matrix for Evaluation
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names,
            yticklabels=train.target_names)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')

# we can send a string and get estimated category for that string
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

print(predict_category('sending a payload to the ISS'))
print(predict_category('discussing islam vs atheism'))
print(predict_category('determining the screen resolution'))




