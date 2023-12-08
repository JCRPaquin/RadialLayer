"""
The probability of each output node is the product of the probabilities of each interior
node along the path to the output node. The functions here calculate the indices of the
required probabilities; by pre-computing the indices we allow the torch compiler to treat
them as constants.
"""

from copy import deepcopy
from typing import List


def get_indices(max_depth: int) -> List[List[int]]:
    """
    Construct paths from the root of the tree to each output node.
    Used to get the output node distribution of the decision tree.

    :param max_depth: Depth of the decision tree
    :return: List of lists of indexes
    """
    num_nodes = 2 ** max_depth - 1

    expanded = []
    to_expand = [[0]]
    while to_expand:
        head = to_expand.pop()
        depth = len(head)

        if depth >= max_depth:
            expanded.append(head)
            continue

        node_idx = head[-1]
        if node_idx >= num_nodes:
            node_idx -= num_nodes

        within_level_idx = node_idx - (2 ** (depth - 1) - 1)
        left_child_idx = (2 ** depth - 1) + 2 * within_level_idx
        assert left_child_idx + 1 < num_nodes

        to_expand.append(head[:-1] + [num_nodes + node_idx, left_child_idx])
        to_expand.append(head + [left_child_idx + 1])

    expanded = expanded + [
        chain[:-1] + [chain[-1] + num_nodes] for chain in expanded
    ]

    # Sorting is required here, otherwise the probabilities will be assigned to the wrong
    # nodes. Because of the indexing scheme we've used, left-most nodes have higher indices
    # and will be sorted towards the back of the list by default.
    return sorted(expanded, reverse=True)


def path_indices(bin_indices: List[List[int]]) -> List[List[int]]:
    """
    Used to construct the indices for the path probabilities of the interior nodes of
    the decision tree.

    :param bin_indices: List of paths from the root node to each output node
    :return: List of indices to construct path probabilities for each interior node
    """
    paths = dict()
    indices = deepcopy(bin_indices)

    max_depth = len(bin_indices[0])
    num_nodes = 2 ** max_depth - 1

    while indices:
        head = indices.pop()
        if not head:
            continue

        node_idx = head[-1]
        if node_idx >= num_nodes:
            node_idx -= num_nodes
        if node_idx in paths:
            continue

        paths[node_idx] = head[:-1] + [2 * num_nodes] * (max_depth - len(head) + 1)
        indices.append(head[:-1])

    return [paths[idx] for idx in range(num_nodes)]
