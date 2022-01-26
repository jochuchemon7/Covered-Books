"""
# Evaluation Metrics and Scoring
"""

# --- METRICS FOR BINARY CLASSIFICATION ---

#  - Dealing with Imbalance Data Sets -  (ex: 99 entries of class 1 and 1 entry of class 2)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
import mglearn
digits = load_digits()
y = digits.target == 9  # Making 9:1 imbalanced dataset
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)


# Using DummyClassifier to always predict the majority class (here "Not nine") to see how uninformative accuracy can be
from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_freq = dummy_majority.predict(X_test)
print(f'Unique Predicted Labels: {np.unique(pred_most_freq)}')
print(f'Test Score: {round(dummy_majority.score(X_test, y_test))}')

# Now using Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print(f'Test Score: {round(tree.score(X_test, y_test), 3)}')

# NOTE: Good Score close to constant predictor; 1) something wrong with the tree or 2) accuracy is not a good measure here

# Logistic Regression
from sklearn.linear_model import LogisticRegression

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print(f'Dummy Score: {round(dummy.score(X_test, y_test), 2)}')

logreg = LogisticRegression(C=.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print(f'LogReg Score: {round(logreg.score(X_test, y_test), 2)}')


# -- Confusion Matrices --
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, pred_logreg)
print(f'Confusion Matrix:\n {confusion}')

# NOTE: rows are the TRUE classes and the columns are the PREDICTED classes
mglearn.plots.plot_confusion_matrix_illustration()
mglearn.plots.plot_binary_confusion_matrix()

# Now confusion matrix with the DummyClassifier, Tree and Logistic Regression
print(f'Most Frequent Class:\n {confusion_matrix(y_test, pred_most_freq)}')
print(f'\nDummy Model: \n{confusion_matrix(y_test, pred_dummy)}')
print(f'\nDecision Tree: \n {confusion_matrix(y_test, pred_tree)}')
print(f'\nLogistic Regression: \n {confusion_matrix(y_test, pred_logreg)}')

# Aside of Accuracy : (TN + TP) / (TN+TP+FN+FP) use Precision: TP/(TP+FP) or Recall: TP/(TP+FN)
# NOTE: Recall -> how many positive samples are captured by positive predictions
# NOTE: Precision -> limit number of false positives
# NOTE: F-Score -> 2 * (precision*recall)/(precision+recall)

# F-Score on 9:1 data
from sklearn.metrics import f1_score
print(f'Most Frequent Class: {f1_score(y_test, pred_most_freq)}')
print(f'Dummy Model: {f1_score(y_test, pred_dummy)}')
print(f'Decision Tree: {f1_score(y_test, pred_tree)}')
print(f'Logistic Regression: {f1_score(y_test, pred_logreg)}')

# Precision and Recall scores on Logistic Regression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(f'Logistic Regression Precision Score: {precision_score(y_test, pred_logreg)}')
print(f'Logistic Regression Recall Score: {recall_score(y_test, pred_logreg)}')

# Viewing all with 'classification_report'
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_most_freq))
print(classification_report(y_test, pred_most_freq, target_names=['Not Nine', 'Nine']))
print(classification_report(y_test, pred_dummy, target_names=['Not Nine', 'Nine']))
print(classification_report(y_test, pred_logreg, target_names=['Not Nine', 'Nine']))

# - Taking Uncertainty Into Account (from predict_proba and decision_function) from models -
mglearn.plots.plot_decision_threshold()

from mglearn.datasets import make_blobs
from sklearn.svm import SVC
X, y = make_blobs(n_samples=(400, 50), centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
print(classification_report(y_test, svc.predict(X_test)))

# We are Interested in High Recall to avoid (FN on cancer) so decision_function threshold is decreased

y_pred_lower_threshold = svc.decision_function(X_test) > -.8  # Lowering the threshold from 0 to -.8 for more positive values
y_pred_lower_threshold = y_pred_lower_threshold.astype(int)
print(classification_report(y_test, y_pred_lower_threshold))

# -- Precision-Recall curves and ROC curves --
# NOTE: 'operating point' : Setting a requirement on a classifier like 90% recall

# Precision Recall Curves: See the trade-off between the two when decision_function threshold is changed
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))

# - plotting the precision_recall_curves -
import matplotlib.pyplot as plt
X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))

# Finding threshold closest to zero
close_zero = np.argmin(np.abs(thresholds))
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, c='k',
         label='Threshold Zero', fillstyle='none', mew=2)
plt.plot(precision, recall, label='Precision Recall Curve')
plt.xlabel('Precision')
plt.ylabel('Recall')

# - Precision_Recall_Curve on RandomForestClassifier -
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])
# NOTE: we pass the probability of a sample being class 1 and not class 0

plt.plot(precision, recall, label='svc')
plt.plot(precision[close_zero], recall[close_zero], 'o', c='k', markersize=10, fillstyle='none',
         mew=2, label='Threshold Zero SVC')

plt.plot(precision_rf, recall_rf, label='rf', linestyle='--')
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', markersize=10, c='k',
         fillstyle='none', mew=2, label='Threshold 0.5 rf')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend(loc='best')

print(f'F1 Score of Random Forest: {round(f1_score(y_test, rf.predict(X_test)), 3)}')
print(f'F1 Score of SVC: {round(f1_score(y_test, svc.predict(X_test)), 3)}')

# - Area Under the Curve of the Precision-Recall Curve -  (average_precision_score) (A way to summarize the Precision-Recall curve)
from sklearn.metrics import average_precision_score
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print(f'Average Precision of Random Forest: {round(ap_rf , 2)}')
print(f'Average Precision of SVC: {round(ap_svc, 2)}')

# NOTE: decision_function at random is the avg_precision is the fraction of positive samples in the dataset

# -- Receiver Operating Characteristics (ROC) and AUC --
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
roc_close_zero = np.argmin(np.abs(thresholds))

# NOTE: (TPR->Recall, FPR->fraction of false positives out of all negative samples)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot(fpr[roc_close_zero], tpr[roc_close_zero], 'o', fillstyle='none', c='k', markersize=10,
         mew=2, label='Threshold Zero')
plt.xlabel('FPR')
plt.ylabel('TPR (Recall)')
plt.legend(loc='best')

# NOTE: Choosing a new threshold should not be done on test data but on separate validation set

# - ROC with RandomForest -
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
roc_close_default_rf = np.argmin(np.abs(thresholds_rf - .5))

plt.plot(fpr, tpr, label='ROC Curve SVC')
plt.plot(fpr_rf, tpr_rf, label='ROC Curve RF')
plt.xlabel('FPR')
plt.ylabel('TPR (Recall)')

plt.plot(fpr[roc_close_zero], tpr[roc_close_zero], 'o', markersize=10, fillstyle='none', c='k',
         mew=2, label='Threshold Zero SVC')
plt.plot(fpr_rf[roc_close_default_rf], tpr_rf[roc_close_default_rf], '^', markersize=10, fillstyle='none',
         c='k', mew=2, label='Threshold 0.5 RF')
plt.legend(loc=4)


# - Computing the Area Under the Curve of ROC - (Better choice than accuracy for imbalanced data set)
from sklearn.metrics import roc_auc_score
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
print(f'AUC for Random Forest: {round(rf_auc, 3)}')
print(f'AUC for SVC: {round(svc_auc, 3)}')

# NOTE: auc -> probability that a randomly picked point of the positive class will have a higher score
# according to the classifier than a randomly picked point from the negative class.

# - Back to the Nine and Not Nine Example -
y = digits.target == 9
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

plt.figure()

for gamma in [1, .05, .01]:
    svc = SVC(gamma=gamma).fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test, svc.decision_function(X_test))
    print(f'Gamma: {gamma}   Accuracy: {round(accuracy, 3)}   AUC: {round(auc)}')
    plt.plot(fpr, tpr, label=f'Gamma: {gamma}')
plt.xlabel('FPR')
plt.ylabel('TPR (Recall)')
plt.xlim(-.01, 1)
plt.ylim(0, 1.02)
plt.legend(loc='best')
