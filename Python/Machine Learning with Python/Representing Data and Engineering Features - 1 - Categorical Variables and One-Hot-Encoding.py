"""
Representing Data and Engineering Features
"""

# --- Categorical Variables ---

# NOTE: Data can come in continuous and/or discrete data forms

# --- One-Hot-Encoding (Dummy Variables) --- (Using Pandas)
import pandas as pd
import os
directory = os.getcwd()
data = pd.read_csv(str(directory + '/Data/adult.data'), header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                          'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                          'hours-per-week', 'native-country', 'income'])
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
print(data.head())


# ~ Check the contents of a column; show the unique values and how they appear ~
print(data.gender.value_counts())

# ~ One-Hot-Encoding ~
print(f'Originial Features: {list(data.columns)}')
data_dummies = pd.get_dummies(data)
print(f'Features after get_dummies: \n{list(data_dummies.columns)}')

print(data_dummies.head())  # Continuous features are not touched

# ~ Slicing features avoiding the target variable ~
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']

X = features.values
y = data_dummies['income_ >50K'].values
print(f'X.shape: {X.shape}, y.shape: {y.shape}')

# ~~ Using Logistic Regression on new Data ~~
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(f'Test Score: {round(logreg.score(X_test, y_test), 3)}')


# --- Numbers Can Encode Categoricals --- (Using scikit OneHotEncoded we can specify columns to be encoded)
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                       'Categorical Feature': ['socks', 'fox', 'socks', 'fox']})
print(demo_df)

# Using pandas get_dummies() (All numbers are treated as continuous and will not create dummy vars)
pd.get_dummies(demo_df)

# Explicit assignment after values changed from int to str
demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])

