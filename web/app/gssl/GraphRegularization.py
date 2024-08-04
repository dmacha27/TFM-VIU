"""
GraphRegularization.py

Author: David Martínez Acha
Email: dmacha@ubu.es / achacbmb3@gmail.com
Last Modified: 12/07/2024
Description: Graph regularization methods
"""

import numpy as np


def lgc(X, y, W_graph, alpha=0.90, iter_max=10000, threshold=0.00001):
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

    D_diag = np.diag(W.sum(axis=1) + 1e-5)  # Sum of each row and convert to diagonal matrix

    D_sqrt_inv = np.diag(1 / np.sqrt(np.diagonal(D_diag)))

    S = D_sqrt_inv @ W @ D_sqrt_inv

    F_t_history = [F]
    pred_history = [y]


    F_t = F
    it = 0
    while it < iter_max:
        F_next = np.dot(alpha * S, F_t) + (1 - alpha) * Y
        diff = np.linalg.norm(F_next - F_t)
        F_t = np.copy(F_next)

        F_t_history.append(F_t)
        pred_history.append(L[np.argmax(F_t, axis=1)])
        if diff <= threshold:
            break
        it += 1


    return F, W, D_diag, D_sqrt_inv, S, np.round(np.array(F_t_history), 3), np.array(pred_history), L[
        np.argmax(F_t, axis=1)]
