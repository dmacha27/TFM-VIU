"""
SSLTree.py

Author: David Martínez Acha
Email: dmacha@ubu.es / achacbmb3@gmail.com
Last Modified: 05/06/2024
Description: Semi-supervised tree (handles both labeled and unlabeled data)
"""

import math

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_random_state

from metodos.SSLTree.utils.Pruning import CostComplexityPruning


class Node:
    """
    A node in a decision tree.

    Attributes
    ----------
    data : array-like
        The subset of data points that belong to this node.

    feature : int
        The index of the feature used for splitting this node.

    val_split : float
        The value used for splitting the feature at this node.

    impurity : float
        The impurity of the node.

    probabilities : array-like
        The class probabilities associated with this node.

    """

    def __init__(self, data, feature, val_split, impurity, probabilities):
        """
        Initializes a Node object with the given data and attributes.

        Parameters
        ----------
        data : array-like
            The subset of data points that belong to this node.

        feature : int
            The index of the feature used for splitting this node.

        val_split : float
            The value used for splitting the feature at this node.

        impurity : float
            The impurity of the node.

        probabilities : array-like
            The class probabilities associated with this node.
        """

        self.data = data
        self.feature = feature
        self.val_split = val_split
        self.impurity = impurity
        self.probabilities = probabilities
        self.left = None
        self.right = None

    def __repr__(self):
        return (f"Node(data={self.data}, feature={self.feature}, val_split={self.val_split}, entropy={self.impurity}, "
                f"probabilities={self.probabilities})")


class SSLTree(ClassifierMixin, BaseEstimator):
    """A decision tree classifier.

    Constructs the tree by computing the dataset's impurity using the method proposed by Levatic et al. (2017).

    Parameters
    ----------
    w : float, default=0.9
        Controls the amount of supervision. Higher values for more supervision.

    criterion : {'entropy', 'gini'}, default='entropy'
        The function to measure the supervised part of the quality of a split.
        - 'entropy': Shannon entropy.
        - 'gini': Gini impurity.

    splitter : {'best', 'random'}, default='best'
        The strategy used to choose the split at each node.
        - 'best': Choose the best split based on impurity.
        - 'random': Choose the best random split.

    max_depth : int, default=4
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    max_features : {'auto', 'sqrt', 'log2', int or float}, default='auto'
        The number of features to consider when looking for the best split:
        - 'auto': All features are considered.
        - 'sqrt': The square root of the total number of features.
        - 'log2': The logarithm base 2 of the total number of features.
        - int: The number of features to consider at each split.
        - float: A fraction of the total number of features to consider at each split.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    ccp_alpha : float, default=0.0
        Regularization parameter used for pruning. Higher values will prune more nodes.

    Attributes
    ----------
    w : float, default=0.9
        The value of the 'w' parameter.

    criterion : {'entropy', 'gini'}
        The function to measure the supervised part of the quality of a split.

    splitter : {'best', 'random'}
        The strategy used to choose the split at each node.

    max_depth : int, default=None
        The maximum depth of the tree.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    max_features : {'auto', 'sqrt', 'log2', int or float}, default='auto'
        The number of features to consider when looking for the best split.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    ccp_alpha : float, default=0.0
        Regularization parameter used for pruning. Higher values will prune more nodes.

    """

    def __init__(self,
                 w=0.9,
                 criterion='entropy',
                 splitter='best',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='auto',
                 random_state=None,
                 ccp_alpha=0.0):

        self.w = w
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha

        self.tree = None
        self.total_var = None
        self.total_impurity = None
        self.classes_ = None
        self.feature_names = None
        self.depth = 0

    def _impurity(self, labels):
        """
        Calculate the impurity measure (either Gini impurity or Shannon entropy) for a set of labels.

        Parameters
        ----------
        labels : array-like
            The array containing the labels for which impurity needs to be calculated.

        Returns
        -------
        float
            The impurity score (Gini impurity or Shannon entropy) for the given set of labels.
        """

        probs = np.unique(labels, return_counts=True)[1] / len(labels)

        if self.criterion == 'gini':
            return 1 - np.sum(probs ** 2)
        else:
            return sum([-p * np.log2(p) for p in probs if p > 0])

    def _var(self, X_i):
        """
        Calculate the variance for a feature.

        Parameters
        ----------
        X_i : array-like
            The array containing the values of a feature.

        Returns
        -------
        float
            The variance of the feature.
        """

        return (np.sum(np.square(X_i)) - np.square(np.sum(X_i)) / len(X_i)) if len(X_i) > 1 else 0

    def _entropy_ssl(self, partitions, selected_features):
        """
        Calculate entropy for the given partitions after splitting the data.

        Parameters
        ----------
        partitions : array-like
            The array containing the two partitions of the data.

        Returns
        -------
        float
            The entropy of the split.
        """

        subsets_labelled = [subset[subset[:, -1] != -1] for subset in partitions]

        total_count_labelled = np.sum([len(subset) for subset in subsets_labelled])
        if total_count_labelled != 0:
            impurity = np.sum(
                [self._impurity(subset[:, -1]) * (len(subset) / total_count_labelled) for subset in
                 subsets_labelled]) / self.total_impurity
        else:
            impurity = 0

        total_count = np.sum([len(subset) for subset in partitions])
        var = 0

        for i in selected_features:
            num = 0
            for subset in partitions:
                num += self._var(subset[:, i]) * (len(subset) / total_count)

            var += (num / self.total_var[i]) if self.total_var[i] else 0

        return self.w * impurity + ((1 - self.w) / (len(selected_features))) * var

    def _split(self, data, feature, feature_val):
        """
        Split the dataset into two subsets based on a feature and its value.

        Parameters
        ----------
        data : numpy.ndarray
            The dataset to be split.
        feature : int
            The index of the feature to split on.
        feature_val : float
            The value of the feature to split on.

        Returns
        -------
        tuple
            A tuple containing two subsets of the original dataset:
            - The subset where the feature values are less than or equal to the feature_val.
            - The subset where the feature values are greater than the feature_val.
        """

        mask = data[:, feature] <= feature_val
        left = data[mask]
        right = data[~mask]

        return left, right

    def _feature_selector(self, num_features):
        """
        Select a subset of features for splitting.

        Parameters
        ----------
        num_features : int
            The total number of features in the dataset.

        Returns
        -------
        numpy.ndarray
            An array containing the indices of the selected features.
        """

        if self.max_features == "auto":
            max_features = num_features
        elif self.max_features == "sqrt":
            max_features = int(math.sqrt(num_features))
        elif self.max_features == "log2":
            max_features = int(math.log2(num_features))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, num_features)
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * num_features)
        else:
            raise ValueError("Invalid value for max_features")

        max_features = max(1, max_features)

        return self.random_state.choice(num_features, max_features, replace=False)

    def _best_split(self, data):
        """
        Find the best split for the dataset.

        Parameters
        ----------
        data : numpy.ndarray
            The dataset to find the best split for.

        Returns
        -------
        tuple
            A tuple containing the following elements:
            - The left subset of the best split.
            - The right subset of the best split.
            - The index of the feature used for the best split.
            - The value of the feature used for the best split.
            - The entropy of the best split.
        """

        best_entropy = float("inf")
        best_feature = -1
        best_feature_val = -1

        selected_features = self._feature_selector(data.shape[1] - 1)
        for feature in selected_features:
            possible_partitions = np.unique(data[:, feature])
            # possible_partitions = (possible_values[:-1] + possible_values[1:]) / 2
            if self.splitter != 'random':
                partition_values = possible_partitions
            else:
                # https://stackoverflow.com/questions/46756606/what-does-splitter-attribute-in-sklearns-decisiontreeclassifier-do
                partition_values = [self.random_state.choice(possible_partitions)]

            for feature_val in partition_values:
                left, right = self._split(data, feature, feature_val)
                entropy = self._entropy_ssl([left, right], selected_features)
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature = feature
                    best_feature_val = feature_val
                    best_left, best_right = left, right

        return best_left, best_right, best_feature, best_feature_val, best_entropy

    def _node_probs(self, data):
        """
        Calculate the probabilities of each label based on their appearance in the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            The dataset to calculate the probabilities of.

        Returns
        -------
        list
            A list containing the probabilities of each label in the dataset.
        """

        labels_in_data = data[:, -1]

        # Only labelled data counts
        total_labels = len(labels_in_data[labels_in_data != -1])
        probs = [0] * len(self.classes_)

        for i, label in enumerate(self.classes_):
            label_appearances = np.where(labels_in_data == label)[0]
            if label_appearances.shape[0] > 0:
                probs[i] = label_appearances.shape[0] / total_labels

        return probs

    def _create_tree(self, data, depth):
        """
        Recursively create a decision tree.

        Parameters
        ----------
        data : numpy.ndarray
            The dataset to build the tree from.
        depth : int
            The current depth of the tree.

        Returns
        -------
        Node
            The root node of the decision tree.
        """

        if self.max_depth is not None and depth > self.max_depth:
            return None

        left_data, right_data, feature, feature_val, entropy = self._best_split(data)

        left_data_labelled = left_data[left_data[:, -1] != -1]
        right_data_labelled = right_data[right_data[:, -1] != -1]

        if len(left_data_labelled) == 0 and len(right_data_labelled) == 0:
            return None

        root = Node(data, feature, feature_val, entropy, self._node_probs(data))

        if self.min_samples_leaf >= len(np.unique(left_data_labelled[:, :-1], axis=0)) and self.min_samples_leaf >= len(
                np.unique(right_data_labelled[:, :-1], axis=0)):
            self.depth = max(self.depth, depth)
            return root

        if 1.0 in root.probabilities:
            self.depth = max(self.depth, depth)
            return root

        # Minimum number of samples required to split an internal node.
        if len(left_data_labelled) >= self.min_samples_split and len(right_data_labelled) >= self.min_samples_split:
            root.left = self._create_tree(left_data, depth + 1)
            root.right = self._create_tree(right_data, depth + 1)
        else:
            root.left = None
            root.right = None

        self.depth = max(self.depth, depth)
        return root

    def fit(self, X, y, feature_names=None):
        """
        Fit the decision tree classifier to the training data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        feature_names : list, optional
            A list containing the names of the features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self.random_state = check_random_state(self.random_state)

        self.feature_names = feature_names
        all_labels = np.unique(y)

        # Unlabelled samples must have -1 label
        self.classes_ = np.sort(all_labels[all_labels != -1])

        if len(y.shape) != 2:
            y = y.reshape(-1, 1)

        data = np.concatenate((X, y), axis=1)

        self.total_impurity = self._impurity(data[data[:, -1] != -1][:, -1])

        if self.total_impurity == 0:
            self.tree = Node(data, -1, 0.0, 0, self._node_probs(data))
            return self

        self.total_var = [self._var(data[:, i]) for i in range(data.shape[1] - 1)]

        self.tree = self._create_tree(data, 0)

        self.tree = CostComplexityPruning.prune(self.tree, self.depth, self.ccp_alpha)

        return self

    def single_predict_proba(self, x):
        """
        Predict class probabilities for an input sample.

        Parameters
        ----------
        x : array-like
            The input sample.

        Returns
        -------
        list
            The class probabilities of the input sample.
        """

        # Starts on root
        node = self.tree

        predictions = [0] * self.classes_
        while node:  # Until leaf is reached
            predictions = node.probabilities
            if x[node.feature] <= node.val_split:
                node = node.left
            else:
                node = node.right

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for multiple input samples.

        Parameters
        ----------
        X : array-like
            The input samples.

        Returns
        -------
        ndarray
            The class probabilities of the input samples.
        """

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 1:
            return self.single_predict_proba(X)

        return np.array([self.single_predict_proba(x) for x in X])

    def single_predict(self, x):
        """
        Predict the class label for an input sample.

        Parameters
        ----------
        x : array-like
            The input sample.

        Returns
        -------
        int
            The predicted class label for the input sample.
        """

        return self.classes_[np.argmax(self.single_predict_proba(x))]

    def predict(self, X):
        """
        Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        list
            The predicted class labels for the input samples.
        """

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 1:
            raise ValueError("Expected 2D array, got 1D array instead:", X, "Use array.reshape(1, -1).")

        predictions = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            predictions[i] = self.single_predict(X[i])

        return predictions

    def text_tree(self, node, depth):
        """
        Generate a textual representation of the decision tree starting from the given node.

        Parameters
        ----------
        node : Node
            The current node of the decision tree.
        depth : int
            The current depth of the tree.

        Returns
        -------
        str
            A textual representation of the decision tree.
        """

        tree = ""
        tree += ("|" + " " * 3) * depth
        tree += "|--- "

        if not node.left or not node.right:
            classes, quantity = np.unique(node.data[:, -1], return_counts=True)
            return tree + "class: " + str(
                self.classes_[np.argmax(node.probabilities)]) + " Classes distribution: " + str(
                classes) + " " + str(quantity) + "\n"
        else:
            tree += ("feature_" + str(node.feature) if not self.feature_names else self.feature_names[
                node.feature]) + " <= " + str(
                node.val_split) + "\n"
            tree += self.text_tree(node.left, depth + 1)

            tree += ("|" + " " * 3) * depth
            tree += "|--- "
            tree += ("feature_" + str(node.feature) if not self.feature_names else self.feature_names[
                node.feature]) + " > " + str(
                node.val_split) + "\n"
            tree += self.text_tree(node.right, depth + 1)

        return tree

    def export_tree(self):
        """
        Export the textual representation of the decision tree.

        Returns
        -------
        str
            A textual representation of the decision tree.
        """
        return self.text_tree(self.tree, 0)
