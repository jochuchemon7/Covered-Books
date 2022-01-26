"""
# Types of Data, Sentiment Analysis and Text Data as Bag of Words
"""

# --- Types of Data Represented as Strings ---

# Categorical Data
# Free Strings That Can Be Semantically Mapped to Categories
# Structured String Data
# Text Data

# --- Example Application: Sentiment Analysis of Movie Reviews ---

# To load files where each sub-folder corresponds to a label like on aclImdb/test/neg and aclImdb/train/neg
import matplotlib.pyplot as plt
from sklearn.datasets import load_files

review_train = load_files('aclImdb/train/')
# load_files return a bunch, containing training text and training labels
text_train, y_train = review_train.data, review_train.target
print(f'Type of text_train: {type(text_train)}')
print(f'Length of Text Train: {len(text_train)}')
print(f'text_train[1]: \n{text_train[1]}')

# Cleaning the text data from HTML line breakers (<br />)
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

# Equal number of neg and pos classes
import numpy as np
print(f'Samples per class (training): {np.bincount(y_train)}')

# Loading the Test data set
review_test = load_files('aclImdb/test/')
text_test, y_test = review_test.data, review_test.target
print(f'Number of documents in test data: {len(text_test)}')
print(f'Samples Per Class (test): {np.bincount(y_test)}')
text_test = [doc.replace(b"<br />", b"") for doc in text_test]

# --- Representing Text Data as a Bag of Words ---

# NOTE: First split each doc into the words by whitespace or punctuation, then collect a vocabulary and then
# Count the frequency of each word

# -- On a toy data set --
bards_words = ["The fool doth think he is wise,", "but the wise man knows himself to be a fool"]

# We use sklearn.feature_extraction.text.CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(bards_words)

print(f'Vocabulary Size: {len(vect.vocabulary_)}')
print(f'Vocabulary Content: \n{vect.vocabulary_}')

# Calling transform for bag-of-words representation for the training data
bag_of_words = vect.transform(bards_words)
print(f'bag_of_words: {repr(bag_of_words)}')

# A 2x13 scipy matrix each row is a data point and one feature for each of the words in the total vocab
# Frequency is store on each element for the given word on the given point
print(f'Dense Representation of bag_of_words: \n{bag_of_words.toarray()}')

values = vect.vocabulary_
values = dict(sorted(values.items(), key=lambda item: item[1]))
print(f'Sorted dictionary by values: \n{values}')

# -- Bag-of-Words for Movie Reviews --

vect = CountVectorizer().fit(text_train)  # Passing all 25,000 point (texts) from train
X_train = vect.transform(text_train)  # The bag of words
print(f'X_train: \n{repr(X_train)}')


# Viewing Details from the X_train
feature_names = vect.get_feature_names()
print(f'Number of Features: {len(feature_names)}')
print(f'First 20 Features: \n{feature_names[:20]}')
print(f'Features 20010 to 20030: \n{feature_names[20010:20030]}')
print(f'Every 2000th feature: \n{feature_names[::2000]}')

# - Using Logistic Regression using Cross-Validation -
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print(f'Mean Cross-Validation Accuracy: {round(np.mean(scores), 3)}')

# - Tuning regularization (C) -
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [.001, .01, .1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print(f'Best Cross Validation Score: {round(grid.best_score_)}')
print(f'Best Parameters: {grid.best_params_}')

# - Assessing generalization performance -
X_test = vect.transform(text_test)
print(f'Test Score: {round(grid.score(X_test, y_test), 3)}')

# NOTE: CountVectorizer regular expression is ('\b\w\w+\b')  -> min of 2 letters/words and sep by boundaries

# Setting the min of data points each word should appear (Removes words that may appear just once)
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print(f'X_train with min_df: {repr(X_train)}')  # From 72,000 down to 27,271

feature_names = vect.get_feature_names()
print(f'First 50 features: \n{feature_names[:50]}')
print(f'Features 20010 to 20030: \n{feature_names[20010:20030]}')
print(f'Every 700th feature: \n{feature_names[::700]}')

# - Grid Search Once More -
grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print(f'Best Cross Validation Score: {round(grid.best_score_, 3)}')

# NOTE: New words from test will be ignored as there are not part of the dictionary

# --- StopWords ---

# We can get rid of words that are too frequent to be informative (e.i more, none, had, today)
# Removing common none emotional descriptive words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print(f'Number of Stop Words: {len(ENGLISH_STOP_WORDS)}')
print(f'Every 10th Stop Word: {list(ENGLISH_STOP_WORDS)[::10]}')

vect = CountVectorizer(min_df=5, stop_words='english').fit(text_train)
X_train = vect.transform(text_train)
print(f'X_train with stop words: \n{repr(X_train)}')

# - Trying again GridSearch -
grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print(f'Best Cross Validation Score: {round(grid.best_score_, 3)}')

# --- Rescaling the Data with tf-idf --- (term frequency-inverse document frequency)
# TfidfVectorizer takes care of the making of the bag_of_words (aka CountVectorizer())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(TfidfVectorizer(min_df=5,  norm=None), LogisticRegression())
param_grid = {'logisticregression__C': [.001,  .01, .1, 1, 10]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(text_train, y_train)
print(f'Best Cross Validation Score: {round(grid.best_score_, 3)}')

# NOTE: Tfidf gives high weight to any term that appears often in a particular document but not in very many documents
# tfidf = tf * log(N+1/Nw +1) +1  -> (N=num of documents inset, Nw=num of documents that word w appears)

vectorizer = grid.best_estimator_.named_steps['tfidfvectorizer']
X_train = vectorizer.transform(text_train)
max_value = X_train.max(axis=0).toarray().ravel()  # Max-value for each of the features over the set
sorted_by_tfidf = max_value.argsort()
feature_names = np.array(vectorizer.get_feature_names())  # Get feature names

print(f'Features with the lowest tfidf: \n{feature_names[sorted_by_tfidf[:20]]}')
print(f'Features with the highest tfidf: \n{feature_names[sorted_by_tfidf[-20:]]}')

# Looking at words with low inverse frequency (Appear frequently and therefore less important)
sorted_by_idf = np.argsort(vectorizer.idf_)
print(f'Features with lowest idf: \n{feature_names[sorted_by_idf[:100]]}')

# NOTE: Some include English stop words

# --- Investigating Model Coefficients --- (Looking at coefficient values of top20 and least20 words)
import mglearn
mglearn.tools.visualize_coefficients(grid.best_estimator_.named_steps['logisticregression'].coef_,
                                     feature_names, n_top_features=40)

# --- Bag-of-Words with More Than One Word (n-Grams) --- (Considering the counts of tokens)
# (e.i -> "It's bad, not good at all" or "It's good, not bad at all")
print(f'Bards_words: \n{bards_words}')

cv = CountVectorizer(ngram_range=(1, 1)).fit(bards_words)  # (min length, max length)
print(f'Vocabulary Size: {len(cv.vocabulary_)}')
print(f'Vocabulary: \n{cv.get_feature_names()}')  # Unigrams

# Only BiGrams
cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
print(f'Vocabulary Size: {len(cv.vocabulary_)}')
print(f'Vocabulary: \n{cv.get_feature_names()}')
print(f'Bards_words: \n{bards_words}')

# No common BiGram between the two points
print(f'Transformed Data (dense): \n{cv.transform(bards_words).toarray()}')

# Using UniGrams, BiGrams and TriGrams (The higher max length the more features are added)
cv = CountVectorizer(ngram_range=(1, 3)).fit(bards_words)
print(f'Vocabulary Size: {len(cv.vocabulary_)}')
print(f'Vocabulary: \n{cv.get_feature_names()}')

# - Finding the best setting of n-gram on text_train with GridSearchCV -
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
param_grid = {'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'logisticregression__C': [.001, .01, .1, 1, 10]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(text_train, y_train)
print(f'Best Cross Validation Score: {round(grid.best_score_, 3)}')
print(f'Best Parameters: \n{grid.best_params_}')


# - HEAT MAP OF SCORES -
import matplotlib.pyplot as plt
import seaborn as sns

scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
sns.heatmap(scores, square=True, annot=True, cmap='viridis', cbar=True,
            xticklabels=param_grid['logisticregression__C'],
            yticklabels=param_grid['tfidfvectorizer__ngram_range'])
plt.xlabel('C')
plt.ylabel('ngram_range')

# - IMPORTANT COEFFICIENT VALUES FOR THE BEST MODEL - (Best estimator with the three UniGram, BiGram and TriGram)
vect = grid.best_estimator_['tfidfvectorizer']  # TfidfVectorizer(min_df=5, ngram_range=(1, 3))
feature_names = np.array(vect.get_feature_names())
coef = grid.best_estimator_['logisticregression'].coef_
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)

# - VISUALIZE ONLY 3 GRAM FEATURES -
mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
mglearn.tools.visualize_coefficients(coef.ravel()[mask], feature_names[mask], n_top_features=40)
