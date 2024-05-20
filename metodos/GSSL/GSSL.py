"""
GSSL.py

Author: David Martínez Acha
Email: dmacha@ubu.es / achacbmb3@gmail.com
Last Modified: 13/05/2024
Description: Graph based Semi-supervised learning methods (graph construction methods and label inference methods)
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree, KNeighborsClassifier
from collections import deque
from scipy.spatial.distance import cdist


def gbili(X, y, k):
    D = cdist(X, X)

    labeled = np.where(np.array(y) != -1)[0]

    graph = {}
    knn = {i: sorted_neighbors[1:k + 1] for i, sorted_neighbors in enumerate(np.argsort(D, axis=1))}
    m_knn = {i: [j for j in knn[i] if i in knn[j]] for i in range(len(X))}

    for i in range(len(X)):
        min_sum_distance = float('inf')
        min_neighbor = None
        for j in m_knn[i]:
            for l in labeled:
                distance = D[i][j] + D[j][l]
                if distance < min_sum_distance:
                    min_sum_distance = distance
                    min_neighbor = j
        graph[i] = [min_neighbor] if min_neighbor else []

    component_membership, components = search_components(graph)

    components_with_labeled = set()
    for component, members in components.items():
        if any(i in members for i in labeled):
            components_with_labeled.add(component)

    for i in range(len(X)):
        if component_membership[i] in components_with_labeled:
            for k in knn[i]:
                if component_membership[k] in components_with_labeled:
                    graph[i].append(k)

    E = []
    W = {}

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edge = (node, neighbor)
            E.append(edge)
            W[edge] = 1

    return list(range(len(X))), E, W


def search_components(graph):
    visited = set()
    component_membership = {}
    components = {}

    actual_component = 0

    for i in graph:

        if i in visited:
            continue

        visited.add(i)
        node_queue = deque([i])

        while node_queue:
            current = node_queue.popleft()
            component_membership[current] = actual_component

            if actual_component not in components:
                components[actual_component] = [current]
            else:
                components[actual_component].append(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    node_queue.append(neighbor)

        actual_component += 1

    return component_membership, components


def rgcli(X, y, k_e, k_i, nt):
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
    GL : tuple
        Tuple containing the graph vertices, edges, and weights.
    """

    labeled = np.where(np.array(y) != -1)[0]

    V, E, W = list(range(len(X))), [], dict()
    GL = (V, E, W)

    kdtree = KDTree(X)
    l_kdtree = KDTree(X[labeled, :])

    kNN = dict()
    F = dict()
    L = dict()
    T = [V[i * len(V) // nt:(i + 1) * len(V) // nt] for i in range(nt)]

    def SearchKNN(T_i):
        for v in T_i:
            all_neighbors = kdtree.query([X[v]], k=len(X), return_distance=False)[0]

            kNN[v] = all_neighbors[1:k_e + 1]

            labeled_neighbors = l_kdtree.query([X[v]], k=2, return_distance=False)[0]
            L[v] = labeled[labeled_neighbors[labeled_neighbors != v][0]]  # The labeled's domain is all X

            F[v] = all_neighbors[-k_e]

    def SearchRGCLI(T_i):
        for vi in T_i:
            epsilon = dict()
            for vj in kNN[vi]:
                if minkowski(X[vi], X[vj]) <= minkowski(X[vj], X[F[vj]]):
                    e = (vj, vi)
                    epsilon[e] = minkowski(X[vi], X[vj]) + minkowski(X[vj], X[L[vj]])
            E_prime = sorted(epsilon, key=epsilon.get)[:k_i]
            E.extend(E_prime)
            for e in E_prime:
                W[e] = 1

    with ThreadPoolExecutor(max_workers=nt) as executor:
        executor.map(SearchKNN, T)

    with ThreadPoolExecutor(max_workers=nt) as executor:
        executor.map(SearchRGCLI, T)

    return GL


def llgcl(X, y, W_rgcli, alpha, iter_max, threshold):
    """
    Performs label propagation using the Learning with Local and Global Consistency (LLGC) algorithm.

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
    W_rgcli : dict
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
    valid_indices = np.where(y != -1)
    F[valid_indices, np.searchsorted(L, y[valid_indices])] = 1  # Ensure correspondence with L

    Y = np.copy(F)

    W = np.zeros((len(X), len(X)))
    for (i, j), simm in W_rgcli.items():
        W[i, j] = simm
        W[j, i] = simm

    D = np.diag(W.sum(axis=1))  # Sum of each row and convert to diagonal matrix

    D_sqrt_inv = np.diag(1 / np.sqrt(np.diagonal(D)))

    S = np.dot(np.dot(D_sqrt_inv, W), D_sqrt_inv)

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


def llgcl_dataset_order(X, y):
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
    alpha : float, default=0.7
        Alpha parameter for label propagation.
    iter_max : int, default=5
        Maximum number of iterations for label propagation.
    threshold : float, default=0.0001
        Convergence threshold for label propagation.

    Attributes
    ----------
    k_e : int
        Number of neighbors for the kNN graph.
    k_i : int
        Number of edges to add to the graph.
    nt : int
        Number of threads.
    alpha : float
        Alpha parameter for label propagation.
    iter_max : int
        Maximum number of iterations for label propagation.
    threshold : float
        Convergence threshold for label propagation.
    """

    def __init__(self, k_e=5, k_i=5, nt=5, alpha=0.7, iter_max=5, threshold=0.0001):
        self.k_e = k_e
        self.k_i = k_i
        self.nt = nt
        self.alpha = alpha
        self.iter_max = iter_max
        self.threshold = threshold

    def fit_predict(self, X, y):
        """
        Fits the model and predicts the labels for the input data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values. Unlabeled instances are marked with -1.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            Predicted labels for the input data.
        """

        X, y = llgcl_dataset_order(X, y)

        # V, E, W = rgcli(X, y, self.k_e, self.k_i, self.nt)

        V, E, W = gbili(X, y, self.k_e)
        print("DONE")
        return llgcl(X, y, W, self.alpha, self.iter_max, self.threshold)


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
    alpha : float, default=0.7
        Alpha parameter for label propagation.
    iter_max : int, default=5
        Maximum number of iterations for label propagation.
    threshold : float, default=0.0001
        Convergence threshold for label propagation.

    Attributes
    ----------
    k_e : int
        Number of neighbors for the kNN graph.
    k_i : int
        Number of edges to add to the graph.
    nt : int
        Number of threads.
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

    def __init__(self, k_e=5, k_i=5, nt=5, alpha=0.7, iter_max=5, threshold=0.0001):
        self.k_e = k_e
        self.k_i = k_i
        self.nt = nt
        self.alpha = alpha
        self.iter_max = iter_max
        self.threshold = threshold
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Fits the model and predicts the labels for the input data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values. Unlabeled instances are marked with -1.

        Returns
        -------
        self : object
            Returns self.
        """

        X, y = llgcl_dataset_order(X, y)

        V, E, W = rgcli(X, y, self.k_e, self.k_i, self.nt)

        self.X = X
        self.y = llgcl(X, y, W, self.alpha, self.iter_max, self.threshold)

        return self

    def predict(self, X):
        """
        Predicts the labels for the input

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            Predicted labels for the input data.
        """

        X_extend = np.concatenate((self.X, X))
        y_extend = np.hstack((self.y, -np.ones((len(X)))))

        V, E, W = rgcli(X_extend, y_extend, self.k_e, self.k_i, self.nt)

        return llgcl(X_extend, y_extend, W, self.alpha, self.iter_max, self.threshold)[len(self.X):]


if __name__ == '__main__':
    from ucimlrepo import fetch_ucirepo

    fetch = fetch_ucirepo(id=53)

    dataset = fetch.data.features.values
    real_targets = np.ravel(fetch.data.targets)
    labels = np.copy(real_targets)

    num_samples_to_unlabel = int(0.2 * len(labels))
    unlabeled_indices = np.random.choice(len(labels), num_samples_to_unlabel, replace=False)
    labels[unlabeled_indices] = -1

    no_labeled = np.where(labels == -1)[0]
    labeled = np.where(labels != -1)[0]

    dataset_labeled = dataset[labeled, :]
    labels_labeled = labels[labeled]

    dataset_no_labeled = dataset[no_labeled, :]
    labels_no_labeled = labels[no_labeled]

    X = np.concatenate((dataset_labeled, dataset_no_labeled))
    y = np.hstack((labels_labeled, labels_no_labeled))

    trans = GSSLTransductive()

    pred = trans.fit_predict(X, y)

    print("Accuracy GSSL:", accuracy_score(real_targets[no_labeled], pred[len(labeled):]))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(dataset_labeled, labels_labeled)
    print("Accuracy KNN:", accuracy_score(real_targets[no_labeled], knn.predict(dataset_no_labeled)))
