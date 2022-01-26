"""
# Metrics for Multi classification
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)

print(f'Accuracy: {round(accuracy_score(y_test, pred), 3)}')
print(f'Confusion Matrix: {confusion_matrix(y_test, pred)}')

sns.heatmap(confusion_matrix(y_test, pred), square=True, annot=True, cmap=plt.cm.gray_r, cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

# F-Score for multiclass setting (The computed class is the positive class and the rest are negative and so on)
from sklearn.metrics import f1_score
print(f'Macro Average f1 Score: {round(f1_score(y_test, pred, average="macro"), 3)}')  # If you care about class equally
print(f'Micro Average f1 Score: {round(f1_score(y_test, pred, average="micro"), 3)}')  # If you care about sample equally

# --- Regression Metrics ---

# --- Using Evaluation Metrics in Model Selection ---

# - On cross_val_score -
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

explicit_accuracy = cross_val_score(SVC(), digits.data, digits.target == 9, scoring='accuracy', cv=3)
roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9, scoring='roc_auc', cv=3)

print(f'Default Scoring: {cross_val_score(SVC(), digits.data, digits.target == 9, cv=3)}')
print(f'Explicit Accuracy Scoring: {explicit_accuracy}')
print(f'AUC Scoring: {roc_auc}')

# - On GridSearchCV -

# Default Accuracy Scoring
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target == 9, random_state=0)

param_grid = {'gamma': [.0001, .01, .1, 1, 10]}
grid = GridSearchCV(estimator=SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)
pred = grid.decision_function(X_test)
print(' --- Grid Search With Accuracy ---')
print(f'Best Parameters: {grid.best_params_}')
print(f'Best Cross-Validation Score (accuracy): {round(grid.best_score_, 3)}')
print(f'Test Set AUC: {round(roc_auc_score(y_test, pred, multi_class="ovo"), 3)}')
print(f'Test Set Accuracy: {round(grid.score(X_test, y_test), 3)}')

# Using AUC Scoring
grid = GridSearchCV(SVC(), param_grid=param_grid, scoring='roc_auc')
grid.fit(X_train, y_train)
pred = grid.decision_function(X_test)
print(' --- Grid Search With ROC_AUC ---')
print(f'Best Parameters: {grid.best_params_}')
print(f'Best Cross-Validation Score (AUC): {round(grid.best_score_, 3)}')
print(f'Test Set AUC: {round(roc_auc_score(y_test, pred, multi_class="ovo"), 3)}')
print(f'Test Set Accuracy: {round(grid.score(X_test, y_test), 3)}')

# NOTE: scoring can also be: (average_precision, f1, f1_macro, f1_micro, f1_weighted)
# For regression: (r2, mean_squared_error, mean_absolute_error)

# Viewing all scoring options
from sklearn.metrics._scorer import SCORERS
print(f'Available Scorers: \n{sorted(SCORERS.keys())}')


