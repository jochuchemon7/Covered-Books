"""
# In Depth: Kernel Density Estimation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# --- Motivating KDE: Histograms ---

# Plotting histogram from normal distribution
def make_data(N, f=.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5  # add value of 5 to last 700 numbers
    return x

x = make_data(1000)
hist = plt.hist(x=x, bins=30, density='normal')

# ~ Confirming the area under the histogram is 1 ~
density, bins, patches = hist
widths = bins[1:] - bins[:-1]
(density * widths).sum()

# ~ Having a histogram as a density estimator leads to representations of different quantities ~
x = make_data(20)
bins = np.linspace(-5, 10, 10)

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True,
                       subplot_kw={'xlim': (-4, 9), 'ylim': (-.02, .3)})
fig.subplots_adjust(wspace=.05)
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, density='normal')
    ax[i].plot(x, np.full_like(x, -.01), '|k', markeredgewidth=1)

# The two histograms were from the same histogram density distribution but with different bin location it affects interpretation

# ~~ Histograms as a stack of blocks ~~
fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -.1), '|k', markeredgewidth=1)
for count, edge in zip(*np.histogram(x, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle((edge, i), 1, 1, alpha=.5))
ax.set_xlim(-4, 8)
ax.set_ylim(-.2, 8)

# ~~ Instead of stacking blocks align with the bins we do it align to the points they represent ~~
x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < .5) for xi in x)

plt.fill_between(x_d, density, alpha=.5)
plt.plot(x, np.full_like(x, -.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -.2, 8])

# The blocks center on each individual point

# ~~ Smoothing the 'histogram' with a Gaussian function, using the standard normal curve at each point ~~
from scipy.stats import norm
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)  # Gaussian

plt.fill_between(x_d, density, alpha=.5)
plt.plot(x, np.full_like(x, -.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -.2, 8])


# --- Kernel Density Estimation in Practice ---

# ~~ Replicating a plot with sklearn kernelDensity ~~
from sklearn.neighbors import KernelDensity

kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])

logprob = kde.score_samples(x_d[:, None])  # log of the probability density

plt.fill_between(x_d, np.exp(logprob), alpha=.5)  # with the exponent of the log
plt.plot(x, np.full_like(x, -.1), '|k', markeredgewidth=1)
plt.ylim(-.02, .22)


# ~~ Selecting Bandwidth Via Cross-Validation ~~
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut())
grid.fit(x[:, None])

# Best bandwidth parameter
print(grid.best_params_)


# --- Example: Not So Naive Bayes --- (Generative classification with KDE instead of axis-aligned Gaussian)

# ~~ Building a Class Estimator ~~
from sklearn.base import BaseEstimator, ClassifierMixin  # Almost every estimator has theses inherited

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE

    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        return self  # Return self so we can chain commands

    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# ~~ Using the Custom Estimator ~~
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV

digits = load_digits()
bandwidths = 10 ** np.linspace(0, 2, 100)
grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})
grid.fit(digits.data, digits.target)

scores = grid.cv_results_['mean_test_score']

# Plotting the cross-validation
plt.semilogx(bandwidths, scores)
plt.xlabel('bandwidth')
plt.ylabel('accuracy')
plt.title('KDE Model Performance')

print(grid.best_params_)
print('accuracy = ', grid.best_score_)

# ~ With traditional Gaussian Naive Bayes ~
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

cross_val_score(GaussianNB(), digits.data, digits.target).mean()


