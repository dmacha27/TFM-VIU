"""
GSSL.py

Author: David Martínez Acha
Email: dmacha@ubu.es / achacbmb3@gmail.com
Last Modified: 26/06/2024
Description: Graph based Semi-supervised learning methods (graph construction methods and label inference methods)
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial.distance import minkowski
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import deque
from scipy.spatial.distance import cdist


def gbili(X, y, k=11):
    """
    Graph-based on informativeness of labeled instances (GBILI).

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The target values. Unlabeled instances are marked with -1.
    k : int
        Number of neighbors.

    Returns
    -------
    W : dict
        Dictionary representing the graph edges and weights.
    """

    D = cdist(X, X)

    labeled = np.where(np.array(y) != -1)[0]

    knn = np.argsort(D, axis=1)[:, 1:k + 1]
    m_knn = [[j for j in knn[i] if i in knn[j]] for i in range(len(X))]

    graph = {i: [] for i in range(len(X))}

    for i in range(len(X)):
        min_sum_distance = float('inf')
        min_neighbor = None
        for j in m_knn[i]:
            for l in labeled:
                distance = D[i][j] + D[j][l]
                if distance < min_sum_distance:
                    min_sum_distance = distance
                    min_neighbor = j
        if min_neighbor is not None:
            graph[i].append(min_neighbor)

    component_membership, components_with_labeled = search_components(graph, set(labeled))

    for i in range(len(X)):
        if component_membership[i] in components_with_labeled:
            for k in knn[i]:
                if component_membership[k] in components_with_labeled:
                    graph[i].append(k)

    W = {(i, neighbor): 1 for i in graph for neighbor in graph[i]}

    return W


def search_components(graph, labeled_set):
    """
    Searches connected components in a graph.

    Parameters
    ----------
    graph : dict
        Dictionary representing the graph.
    labeled_set : set
        Set of labeled nodes.

    Returns
    -------
    component_membership : dict
        Dictionary containing the component membership for each node.
    components_with_labeled : set
        Set of components containing at least one labeled node.
    """

    visited = set()
    component_membership = {}
    components_with_labeled = set()

    actual_component = 0

    for i in graph:
        if i in visited:
            continue

        visited.add(i)
        node_queue = deque([i])

        while node_queue:
            current = node_queue.popleft()
            component_membership[current] = actual_component

            if current in labeled_set:
                components_with_labeled.add(actual_component)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    node_queue.append(neighbor)

        actual_component += 1

    return component_membership, components_with_labeled


def rgcli(X, y, k_e=50, k_i=2, nt=1):
    """
    Performs the Regularized Graph-based Consistency-based Labeling and Instance (RGCLI) algorithm.

    This method is based on the following paper:
    "RGCLI: Robust Graph that Considers Labeled Instances for Semi-Supervised Learning"
    by Lilian Berton, Thiago de Paulo Faleiros, Alan Valejo, Jorge Valverde-Rebaza, and Alneu de Andrade Lopes
    Published in Neurocomputing, Volume 226, Pages 238-248, 2017.
    Available at: https://www.sciencedirect.com/science/article/pii/S0925231216314680
    DOI: https://doi.org/10.1016/j.neucom.2016.11.053

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The target values. Unlabeled instances are marked with -1.
    k_e : int
        Number of neighbors for the kNN graph.
    k_i : int
        Number of edges to add to the graph.
    nt : int
        Number of threads.

    Returns
    -------
    W : dict
        Dictionary representing the graph edges and weights.
    """

    labeled = np.where(np.array(y) != -1)[0]

    V, W = list(range(len(X))), dict()

    # SearchKNN
    D = cdist(X, X)
    D_ordered = np.argsort(D, axis=1)

    kNN = D_ordered[:, 1:k_e + 1]
    F = D_ordered[:, -k_e]

    L = np.zeros(len(D), dtype=int)
    D_isin = np.isin(D_ordered, labeled)
    for i in range(len(D)):
        row = D_ordered[i][D_isin[i]]
        L[i] = row[0] if row[0] != i else row[1]
    # End SearchKNN

    T = [V[i * len(V) // nt:(i + 1) * len(V) // nt] for i in range(nt)]

    def SearchRGCLI(T_i):
        for vi in T_i:
            epsilon = dict()
            for vj in kNN[vi]:
                if minkowski(X[vi], X[vj]) <= minkowski(X[vj], X[F[vj]]):
                    e = (vj, vi)
                    epsilon[e] = minkowski(X[vi], X[vj]) + minkowski(X[vj], X[L[vj]])
            E_prime = sorted(epsilon, key=epsilon.get)[:k_i]
            for e in E_prime:
                W[e] = 1

    with ThreadPoolExecutor(max_workers=nt) as executor:
        executor.map(SearchRGCLI, T)

    return W


def lgc(X, y, W_graph, alpha, iter_max, threshold):
    """
    Performs label propagation using the Learning with Local and Global Consistency (LGC) algorithm.

    This method is based on the following paper:
    "Learning with local and global consistency"
    by Dengyong Zhou, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Schölkopf
    Published in Advances in Neural Information Processing Systems, Volume 16, 2003.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The input samples.
    y : array-like, shape (n_samples,)
        The target values. Unlabeled instances are marked with -1.
    W_graph : dict
        Dictionary containing the weighted edges.
    alpha : float
        The alpha parameter for label propagation.
    iter_max : int
        Maximum number of iterations for label propagation.
    threshold : float
        Convergence threshold for label propagation.

    Returns
    -------
    labels : array-like, shape (n_samples,)
        Predicted labels for the input data.
    """

    L = np.sort(np.unique(y[y != -1]))

    F = np.zeros((len(X), len(L)))
    valid_indices = y != -1
    F[valid_indices, np.searchsorted(L, y[valid_indices])] = 1  # Ensure correspondence with L

    Y = np.copy(F)

    W = np.zeros((len(X), len(X)))
    for (i, j), simm in W_graph.items():
        W[i, j] = simm
        W[j, i] = simm

    D = np.diag(W.sum(axis=1) + 1e-5)  # Sum of each row and convert to diagonal matrix

    D_sqrt_inv = np.diag(1 / np.sqrt(np.diagonal(D)))

    S = D_sqrt_inv @ W @ D_sqrt_inv

    F_t = F
    it = 0
    while it < iter_max:
        F_next = np.dot(alpha * S, F_t) + (1 - alpha) * Y
        diff = np.linalg.norm(F_next - F_t)
        F_t = np.copy(F_next)

        if diff <= threshold:
            break
        it += 1

    return L[np.argmax(F_t, axis=1)]


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
    """

    labeled_indices = y != -1
    X_labeled = X[labeled_indices]
    y_labeled = y[labeled_indices]

    X_unlabeled = X[~labeled_indices]
    y_unlabeled = y[~labeled_indices]

    X_ordered = np.concatenate((X_labeled, X_unlabeled))
    y_ordered = np.hstack((y_labeled, y_unlabeled))

    return X_ordered, y_ordered


class GSSLTransductive(BaseEstimator, ClassifierMixin):
    """
    Graph-based Semi-Supervised Learning Algorithm. Transductive method.

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

    def __init__(self, k_e=5, k_i=5, nt=5, alpha=0.99, iter_max=10000, threshold=0.00001):
        self.k_e = k_e
        self.k_i = k_i
        self.nt = nt
        self.alpha = alpha
        self.iter_max = iter_max
        self.threshold = threshold

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

        X, y = lgc_dataset_order(X, y)

        W = rgcli(X, y, self.k_e, self.k_i, self.nt) if method == "rgcli" else gbili(X, y, self.k_e)

        return lgc(X, y, W, self.alpha, self.iter_max, self.threshold)


class GSSLInductive(BaseEstimator, ClassifierMixin):
    """
    Graph-based Semi-Supervised Learning Algorithm. Inductive method.

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
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The input samples used for fitting the model.
    y : array-like, shape (n_samples,)
        The target values used for fitting the model. Unlabeled instances are marked with -1.
    """

    def __init__(self, k_e=5, k_i=5, nt=5, alpha=0.99, iter_max=50, threshold=0.0001):
        self.k_e = k_e
        self.k_i = k_i
        self.nt = nt
        self.alpha = alpha
        self.iter_max = iter_max
        self.threshold = threshold
        self.X = None
        self.y = None

    def fit(self, X, y, method="rgcli"):
        """
        Fits the model and predicts the labels for the input data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values. Unlabeled instances are marked with -1.
        method : str, optional, default="rgcli"
            The method to create the graph (matrix):
            - "rgcli": Uses the rgcli method.
            - "gbili": Uses the gbili method.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = lgc_dataset_order(X, y)

        W = rgcli(X, y, self.k_e, self.k_i, self.nt) if method == "rgcli" else gbili(X, y, self.k_e)

        self.X = X
        self.y = lgc(X, y, W, self.alpha, self.iter_max, self.threshold)

        return self

    def predict(self, X, method="rgcli"):
        """
        Predicts the labels for the input

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.
        method : str, optional, default="rgcli"
            The method to create the graph (matrix):
            - "rgcli": Uses the rgcli method.
            - "gbili": Uses the gbili method.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            Predicted labels for the input data.
        """

        X_extend = np.concatenate((self.X, X))
        y_extend = np.hstack((self.y, -np.ones((len(X)))))

        W = rgcli(X_extend, y_extend, self.k_e, self.k_i, self.nt) if method == "rgcli" else gbili(X_extend, y_extend,
                                                                                                   self.k_e)

        return lgc(X_extend, y_extend, W, self.alpha, self.iter_max, self.threshold)[len(self.X):]
