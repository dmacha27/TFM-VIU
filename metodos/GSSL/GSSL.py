"""
GSSL.py

Author: David Martínez Acha
Email: dmacha@ubu.es / achacbmb3@gmail.com
Last Modified: 26/06/2024
Description: Graph based Semi-supervised learning methods (graph construction methods and label inference methods)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array

from metodos.GSSL.GraphConstruction import rgcli, gbili
from metodos.GSSL.GraphRegularization import lgc

import warnings


def lgc_dataset_order(X, y):
    """
    Orders the dataset separating labeled and unlabeled instances.

    According to the "Learning with Local and Global Consistency" paper, the first "l" instances correspond to
    labeled points, where x_i for i<l, with l being the number of labeled instances.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The target values. Unlabeled instances are marked with -1.

    Returns
    -------
    X_ordered : array-like or sparse matrix, shape (n_samples, n_features)
        The input samples ordered with labeled instances first followed by unlabeled instances.
    y_ordered : array-like, shape (n_samples,)
        The target values ordered with labeled instances first followed by unlabeled instances.
    n_labeled: int
        The number of labeled instances.
    """

    labeled_indices = y != -1
    X_labeled = X[labeled_indices]
    y_labeled = y[labeled_indices]

    X_unlabeled = X[~labeled_indices]
    y_unlabeled = y[~labeled_indices]

    X_ordered = np.concatenate((X_labeled, X_unlabeled))
    y_ordered = np.hstack((y_labeled, y_unlabeled))

    n_labeled = len(y_labeled)

    return X_ordered, y_ordered, n_labeled


class GSSL(BaseEstimator, ClassifierMixin):
    """
    Graph-based Semi-Supervised Learning Algorithm.

    Parameters
    ----------
    k_e : int, default=5
        Number of neighbors for the kNN graph.
    k_i : int, default=5
        Number of edges to add to the graph.
    nt : int, default=5
        Number of threads.
    alpha : float, default=0.99
        Alpha parameter for label propagation.
    iter_max : int, default=5
        Maximum number of iterations for label propagation.
    threshold : float, default=0.0001
        Convergence threshold for label propagation.

    Attributes
    ----------
    k_e : int
        Number of neighbors for the kNN graph (both GBILI and RGCLI).
    k_i : int
        Number of edges to add to the graph (only RGCLI).
    nt : int
        Number of threads (only RGCLI).
    alpha : float
        Alpha parameter for label propagation.
    iter_max : int
        Maximum number of iterations for label propagation.
    threshold : float
        Convergence threshold for label propagation.
    """

    def __init__(self, k_e=50, k_i=2, nt=4, alpha=0.94, iter_max=10000, threshold=0.00001):
        self.k_e = k_e
        self.k_i = k_i
        self.nt = nt
        self.alpha = alpha
        self.iter_max = iter_max
        self.threshold = threshold

    def fit(self, X, y, method="rgcli"):
        raise NotImplementedError("The fit method should be implemented by subclasses")

    def predict(self, X, method="rgcli"):
        raise NotImplementedError("The predict method should be implemented by subclasses")

    def create_graph(self, X, y, method):
        if method == "rgcli":
            return rgcli(X, y, self.k_e, self.k_i, self.nt)
        else:
            return gbili(X, y, self.k_e)

    def propagate_labels(self, X, y, W):
        return lgc(X, y, W, self.alpha, self.iter_max, self.threshold)


class GSSLTransductive(GSSL):
    """
    Graph-based Semi-Supervised Learning Algorithm. Transductive method.
    """

    def __init__(self, k_e=50, k_i=2, nt=4, alpha=0.94, iter_max=10000, threshold=0.00001):
        super().__init__(k_e, k_i, nt, alpha, iter_max, threshold)
        self.y_pred_ = None

    def fit(self, X, y, method="rgcli"):
        """
        Fits the model. In transductive configuration, the fit methods predicts the labels for the unlabeled samples
        on training set.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values. Unlabeled instances should be marked with -1.
        method : str, optional, default="rgcli"
            The method to create the graph (matrix):
            - "rgcli": Uses the rgcli method.
            - "gbili": Uses the gbili method.

        Returns
        -------
        self : GSSLTransductive
            Self instance of the class.
        """

        X, y, n_labeled = lgc_dataset_order(X, y)

        W = self.create_graph(X, y, method)

        self.y_pred_ = self.propagate_labels(X, y, W)[
                       n_labeled:]  # New labels for the unlabeled samples on training set.

        return self

    def predict(self, X, method="rgcli"):
        """
        In transductive configuration, no new unlabeled instances can be predicted. Transductive methods only uses
        samples of the training set.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values. Unlabeled instances should be marked with -1.
        method : str, optional, default="rgcli"
            The method to create the graph (matrix):
            - "rgcli": Uses the rgcli method.
            - "gbili": Uses the gbili method.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            Predicted labels for unlabeled samples on the training set.
        """

        check_is_fitted(self)

        warnings.warn("Transductive predict method only returns labels for unlabeled subset on the training set. X "
                      "will be ignored")
        return self.y_pred_

    def fit_predict(self, X, y, method="rgcli"):
        """
        Fits the model and predicts the labels for the input data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values. Unlabeled instances should be marked with -1.
        method : str, optional, default="rgcli"
            The method to create the graph (matrix):
            - "rgcli": Uses the rgcli method.
            - "gbili": Uses the gbili method.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            Predicted labels for the input data.
        """

        return self.fit(X, y, method).y_pred_


class GSSLInductive(GSSL):
    """
    Graph-based Semi-Supervised Learning Algorithm. Inductive method.
    """

    def __init__(self, k_e=50, k_i=2, nt=4, alpha=0.94, iter_max=10000, threshold=0.00001):
        super().__init__(k_e, k_i, nt, alpha, iter_max, threshold)
        self.X_ = None
        self.y_ = None

    def fit(self, X, y, method="rgcli"):
        """
        Fits the model. In transductive configuration, the fit method only store the given training set.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values. Unlabeled instances should be marked with -1.
        method : str, optional, default="rgcli"
            The method to create the graph (matrix):
            - "rgcli": Uses the rgcli method.
            - "gbili": Uses the gbili method.

        Returns
        -------
        self : GSSLInductive
            Self instance of the class.
        """

        self.X_, self.y_, _ = lgc_dataset_order(X, y)

        return self

    def predict(self, X, method="rgcli"):
        """
        Predicts the labels for the input data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values. Unlabeled instances should be marked with -1.
        method : str, optional, default="rgcli"
            The method to create the graph (matrix):
            - "rgcli": Uses the rgcli method.
            - "gbili": Uses the gbili method.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            Predicted labels for the input data.
        """

        check_is_fitted(self)
        X = check_array(X)

        X_extend = np.concatenate((self.X_, X))
        y_extend = np.hstack((self.y_, -np.ones((len(X),))))

        W = self.create_graph(X_extend, y_extend, method)
        y_pred = self.propagate_labels(X_extend, y_extend, W)[len(self.X_):]

        return y_pred
