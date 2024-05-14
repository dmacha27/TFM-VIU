"""
Pruning.py

Author: David Martínez Acha
Email: dmacha@ubu.es / achacbmb3@gmail.com
Last Modified: 13/05/2024
Description: Pruning methods
"""

import numpy as np
import bisect


class CostComplexityPruning:
    """
    Cost Complexity Pruning for decision trees.

    Methods
    -------
    prune(tree, depth, ccp_alpha=0.0):
        Prunes the decision tree with Cost Complexity Pruning algorithm.

    _tree_to_array(tree, depth):
        Converts the decision tree into an array representation.

    _recursive_tree_to_array(node, tree_array, inner_nodes, index):
        Recursively converts the decision tree into an array representation.

    _leaves_of_subtree(tree_array):
        Finds the leaves of each subtree in the array representation of the decision tree.

    _recursive_leaves_of_subtree(tree_array, index, leaves):
        Recursively finds the leaves of each subtree in the array representation of the decision tree.
    """

    @staticmethod
    def prune(tree, depth, ccp_alpha=0.0):
        """
        Prunes the decision tree with the Cost Complexity Pruning algorithm.

        Parameters
        ----------
        tree :
            Decision Tree (root of decision tree).
        depth : int
            The depth of the decision tree.
        ccp_alpha : float, optional
            Regularization parameter used for pruning, by default 0.0.
            Higher values will prune more nodes.

        Returns
        -------
        Node
            The root of the pruned decision tree.
        """

        if ccp_alpha == 0:
            return tree

        tree_array, inner_nodes = CostComplexityPruning._tree_to_array(tree, depth)

        f_T = CostComplexityPruning._leaves_of_subtree(tree_array)  # Leaves of each node

        R_t = lambda x: (1 - max(tree_array[x].probabilities)) * len(tree_array[x].data) / len(
            tree_array[0].data)  # Training error of node x (t)

        R_T_t = lambda x: sum([R_t(l) for l in f_T[x]])  # Training error of subtree x (Tt)

        g_t = lambda x: (R_t(x) - R_T_t(x)) / (len(f_T[x]) - 1)

        alphas = []
        prunings = []

        while len(inner_nodes) > 0:

            training_errors_nodes = [0] * len(inner_nodes)  # R(t)
            training_errors_subtrees = [0] * len(inner_nodes)  # R(Tt)
            objetive_function = [0] * len(inner_nodes)  # g(t)

            for idx, t in enumerate(inner_nodes):
                training_errors_nodes[idx] = R_t(t)
                training_errors_subtrees[idx] = R_T_t(t)
                objetive_function[idx] = g_t(t)

            min_alpha = np.min(objetive_function)
            minimal_g = np.where(objetive_function == min_alpha)[0]
            minimal_g = inner_nodes[minimal_g]

            to_prune = minimal_g[0]
            for t in minimal_g:
                if len(f_T[t]) < len(f_T[to_prune]):
                    to_prune = t

            # Update state
            alphas.append(min_alpha)
            prunings.append(to_prune)
            inner_nodes = inner_nodes[inner_nodes != to_prune]

            old_leaves = f_T[to_prune]
            del f_T[to_prune]

            for key in f_T:
                if old_leaves.issubset(f_T[key]):
                    f_T[key] -= old_leaves
                    f_T[key].add(to_prune)

        match_position = bisect.bisect_left(alphas, ccp_alpha)
        new_leaves = prunings[:match_position if ccp_alpha not in alphas else match_position + 1]  # NEED TO BE REVIEWED

        for t in new_leaves:
            tree_array[t].left = None
            tree_array[t].right = None

        return tree

    @staticmethod
    def _tree_to_array(tree, depth):
        """
        Converts the decision tree into an array representation.

        Parameters
        ----------
        tree :
            Decision Tree (root of decision tree).
        depth : int
            The depth of the decision tree.

        Returns
        -------
        numpy.ndarray
            Array representation of the decision tree.
        numpy.ndarray
            Array of inner node (not leaves) indices.
        """

        tree_array = [None] * (2 ** (depth + 1) - 1)
        inner_nodes = []
        CostComplexityPruning._recursive_tree_to_array(tree, tree_array, inner_nodes, 0)
        return np.array(tree_array), np.array(inner_nodes)

    @staticmethod
    def _recursive_tree_to_array(node, tree_array, inner_nodes, index):
        """
        Recursively converts the decision tree into an array representation.

        Parameters
        ----------
        node :
            The current node of the decision tree.
        tree_array : list
            Array representation of the decision tree.
        inner_nodes : list
            List of inner node indices.
        index : int
            Current index in the array representation.

        """

        if node and index < len(tree_array):
            tree_array[index] = node
            if node.left or node.right:
                inner_nodes.append(index)
            CostComplexityPruning._recursive_tree_to_array(node.left, tree_array, inner_nodes, 2 * index + 1)
            CostComplexityPruning._recursive_tree_to_array(node.right, tree_array, inner_nodes, 2 * index + 2)

    @staticmethod
    def _leaves_of_subtree(tree_array):
        """
        Finds the leaves of each subtree in the array representation of the decision tree.

        Parameters
        ----------
        tree_array : numpy.ndarray
            Array representation of the decision tree.

        Returns
        -------
        dict
            Dictionary containing the leaves of each subtree.
        """

        leaves = {}
        CostComplexityPruning._recursive_leaves_of_subtree(tree_array, 0, leaves)
        return leaves

    @staticmethod
    def _recursive_leaves_of_subtree(tree_array, index, leaves):
        """
        Recursively finds the leaves of each subtree in the array representation of the decision tree.

        Parameters
        ----------
        tree_array : numpy.ndarray
            Array representation of the decision tree.
        index : int
            Current index in the array representation.
        leaves : dict
            Dictionary containing the leaves of each subtree.

        Returns
        -------
        list
            List of indices of leaves in the subtree.
        """

        left_child = 2 * index + 1
        right_child = 2 * index + 2

        if tree_array[left_child].left is None and tree_array[left_child].right is None:
            found_left_leaves = [left_child]
        else:
            found_left_leaves = CostComplexityPruning._recursive_leaves_of_subtree(tree_array, left_child, leaves)

        if tree_array[right_child].left is None and tree_array[right_child].right is None:
            found_right_leaves = [right_child]
        else:
            found_right_leaves = CostComplexityPruning._recursive_leaves_of_subtree(tree_array, right_child, leaves)

        found_left_leaves.extend(found_right_leaves)
        leaves[index] = set(found_left_leaves)

        return found_left_leaves
