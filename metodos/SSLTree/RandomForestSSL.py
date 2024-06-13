from copy import deepcopy
from scipy.stats import mode
import numpy as np


class RandomForestSSL:
    def __init__(self, estimator=None, n_estimators=10, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.trees = []
        self._random_state = np.random.RandomState(random_state)

    def fit(self, X, y):
        self.trees = []

        labeled_indices = np.where(y != -1)[0]
        unlabeled_indices = np.where(y == -1)[0]

        for b in range(self.n_estimators):
            labeled_sample = self._random_state.choice(labeled_indices, size=len(labeled_indices), replace=True)
            unlabeled_sample = self._random_state.choice(unlabeled_indices, size=len(unlabeled_indices), replace=True)

            sample = np.concatenate((labeled_sample, unlabeled_sample))
            self._random_state.shuffle(sample)

            X_train_b = X[sample]
            y_train_b = y[sample]

            tree = deepcopy(self.estimator)
            tree.fit(X_train_b, y_train_b)
            self.trees.append(tree)

    def predict(self, X):
        y_test_hats = np.empty((len(self.trees), len(X)))
        for i, tree in enumerate(self.trees):
            y_test_hats[i] = tree.predict(X)

        y_test_hats_mode, _ = mode(y_test_hats, axis=0)

        return y_test_hats_mode.flatten()
