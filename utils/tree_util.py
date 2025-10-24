from methods.node import Node
from typing import Any, Dict, Optional


def tree_height(root: Optional[Node]) -> int:
    """
    Compute the height (maximum depth) of a binary tree.

    The height of a tree is defined as the number of nodes
    on the longest path from the root to a leaf. An empty
    tree (None or a sentinel node) has height 0.

    Parameters
    ----------
    root : Optional[Node]
        The root node of the tree. Can be None or a sentinel node
        representing an empty tree.

    Returns
    -------
    int
        Maximum depth of the tree (0 if empty).

    Raises
    ------
    RuntimeError
        If a cycle is detected in the tree structure, indicating a corrupted tree.
    """
    if root is None:
        return 0

    visited = set()

    def _h(n):
        if n is None or n.value is None:
            return 0
        if id(n) in visited:
            raise RuntimeError("Cycle detected in tree!")
        visited.add(id(n))
        return 1 + max(_h(n.left), _h(n.right))

    return _h(root)


def compute_balance_metrics(root: Optional[Node]) -> Dict[str, Any]:
    """
    Compute key balance metrics for a binary tree.

    Metrics computed:
      - `height`: maximum depth of the tree.
      - `avg_depth`: average depth across all non-empty nodes.
      - `max_subtree_imbalance`: maximum difference in height
        between left and right subtrees at any node, indicating
        how unbalanced the tree is locally.

    Parameters
    ----------
    root : Optional[Node]
        The root node of the tree. Can be None or a sentinel node.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
          - "height" : int, maximum depth of the tree.
          - "avg_depth" : float, average depth of all nodes.
          - "max_subtree_imbalance" : int, largest left-right subtree height difference.

    Notes
    -----
    - Depth of root node is counted as 1.
    - Sentinel or empty nodes (value=None) are ignored.
    """
    if root is None or root.value is None:
        return {"height": 0, "avg_depth": 0.0, "max_subtree_imbalance": 0}

    depths = []
    max_imbalance = 0

    def helper(node: Node, depth: int) -> int:
        nonlocal max_imbalance
        if node is None or node.value is None:
            return 0
        left_h = helper(node.left, depth + 1)
        right_h = helper(node.right, depth + 1)
        depths.append(depth)
        max_imbalance = max(max_imbalance, abs(left_h - right_h))
        return 1 + max(left_h, right_h)

    height = helper(root, 1)
    avg_depth = sum(depths) / len(depths) if depths else 0.0
    return {
        "height": height,
        "avg_depth": avg_depth,
        "max_subtree_imbalance": max_imbalance,
    }
