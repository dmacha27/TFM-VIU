"""
GraphConstruction.py

Author: David Martínez Acha
Email: dmacha@ubu.es / achacbmb3@gmail.com
Last Modified: 12/07/2024
Description: Graph construction methods
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial.distance import cdist, minkowski


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
