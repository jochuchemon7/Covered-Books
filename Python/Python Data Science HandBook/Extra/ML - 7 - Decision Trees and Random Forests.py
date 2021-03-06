"""
# In Depth: Decision Trees and Random Forests
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# ~ Creating a decision tree ~
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')

# --- Fitting a Decision Tree ---
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)


# ~ Function to visualize output ~
def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()

    # plot training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    # Fit estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create color plot with results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=.3, levels=np.arange(n_classes + 1) - .5,
                          cmap=cmap, clim=(y.min(), y.max()), zorder=1)
    ax.set(xlim=xlim, ylim=ylim)


visualize_classifier(DecisionTreeClassifier(), X, y)


# --- Ensembles of Estimators: Random Forests ---
from sklearn.ensemble import BaggingClassifier

# 80% random of the training points
tree = DecisionTreeClassifier()
bag = BaggingClassifier(base_estimator=tree, n_estimators=100, max_samples=.8, random_state=1)

bag.fit(X, y)
visualize_classifier(bag, X, y)


# --- Random Forest Regression ---

# consider fast and slow oscillation data
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)

def model(x, sigma=.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(.5 * x)
    noise = sigma * rng.randn(len(x))
    return slow_oscillation + fast_oscillation + noise


y = model(x)
plt.errorbar(x=x, y=y, yerr=.3, fmt='o')

# ~ Random Forest Regressor ~
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=200)
forest.fit(x[:, None], y)

# fitting and predicting on test data and getting true function values
xfit = np.linspace(0, 10, 1000)
yfit = forest.predict(xfit[:, None])
ytrue = model(xfit, sigma=0)

plt.errorbar(x, y, yerr=.4, fmt='o', alpha=.5)
plt.plot(xfit, yfit, '-r')
plt.plot(xfit, ytrue, '-k', alpha=.5)


# --- Example:: Random Forest for Classifying Digits ---
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.keys())

# ~ Visualize a few data points ~
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, hspace=.05, wspace=.05, bottom=0, top=1)

for i in range(64):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # labeling data
    ax.text(0, 7, str(digits.target[i]))

# ~ Alternative Visualization Technique ~
fig, ax = plt.subplots(nrows=8, ncols=8)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=.05, wspace=.05)

for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    axi.set(xticks=[], yticks=[])
    axi.text(0, 7, str(digits.target[i]))


# ~ Classify digits with Random Forest ~
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

# ~ Classification Report ~
from sklearn.metrics import classification_report
print(classification_report(y_true=ytest, y_pred=ypred))

# ~ Confusion Matrix ~
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_true=ytest, y_pred=ypred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')


# NOTE: You can use the "predict_proba()" method for probability estimates

