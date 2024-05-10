class CostComplexityPruning:
    @staticmethod
    def prune(tree, depth):
        tree_array, inner_nodes = CostComplexityPruning._tree_to_array(tree, depth)

        f_T = CostComplexityPruning._leaves_of_subtree(tree_array)  # Leaves of each node

        R_t = lambda x: (1 - max(tree_array[x].probabilities)) * len(tree_array[x].data) / len(
            tree_array[0].data)  # Training error of node x (t)

        R_T_t = lambda x: sum([R_t(l) for l in f_T[x]])  # Training error of subtree x (Tt)

        g_t = lambda x: (R_t(x) - R_T_t(x)) / (len(f_T[x]) - 1)

        for i in range(4):

            training_errors_nodes = [0] * len(inner_nodes)  # R(Tt)
            training_errors_subtrees = [0] * len(inner_nodes)  # R(t)
            objetive_function = [0] * len(inner_nodes)  # g(t)

            for idx, t in enumerate(inner_nodes):
                training_errors_nodes[idx] = R_t(t)
                training_errors_subtrees[idx] = R_T_t(t)
                objetive_function[idx] = g_t(t)

        return inner_nodes

    @staticmethod
    def _tree_to_array(tree, depth):
        tree_array = [None] * (2 ** (depth + 1) - 1)
        inner_nodes = []
        CostComplexityPruning._recursive_tree_to_array(tree, tree_array, inner_nodes, 0)
        return tree_array, inner_nodes

    @staticmethod
    def _recursive_tree_to_array(node, tree_array, inner_nodes, index):
        if node:
            tree_array[index] = node
            if node.left or node.right:
                inner_nodes.append(index)
            CostComplexityPruning._recursive_tree_to_array(node.left, tree_array, inner_nodes, 2 * index + 1)
            CostComplexityPruning._recursive_tree_to_array(node.right, tree_array, inner_nodes, 2 * index + 2)

    @staticmethod
    def _leaves_of_subtree(tree_array):
        leaves = {}
        CostComplexityPruning._recursive_leaves_of_subtree(tree_array, 0, leaves)
        return leaves

    @staticmethod
    def _recursive_leaves_of_subtree(tree_array, index, leaves):
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


if __name__ == '__main__':
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


    leaf_node_1 = Node(data=[1] * 4, feature=None, val_split=None, impurity=0, probabilities=[0, 1])
    leaf_node_2 = Node(data=[0] * 2, feature=None, val_split=None, impurity=0, probabilities=[1, 0])
    leaf_node_3 = Node(data=[0] * 6, feature=None, val_split=None, impurity=0, probabilities=[1, 0])
    leaf_node_4 = Node(data=[1] * 4, feature=None, val_split=None, impurity=0, probabilities=[0, 1])

    internal_node = Node(data=[1] * 4 + [0] * 8, feature=0, val_split=0.5, impurity=0.5, probabilities=[0.5, 0.5])
    internal_node.left = leaf_node_1
    internal_node.right = Node(data=[1] * 4 + [0] * 8, feature=0, val_split=0.5, impurity=0.5,
                               probabilities=[4 / 12, 8 / 12])
    internal_node.right.left = leaf_node_3
    internal_node.right.right = Node(data=[1] * 4 + [0] * 2, feature=0, val_split=0.5, impurity=0.33,
                                     probabilities=[2 / 6, 4 / 6])
    internal_node.right.right.left = leaf_node_2
    internal_node.right.right.right = leaf_node_4

    root = Node(data=[1] * 8 + [0] * 8, feature=0, val_split=0.5, impurity=0.5, probabilities=[0.5, 0.5])
    root.left = leaf_node_1
    root.right = internal_node

    CostComplexityPruning.prune(root, 2)
