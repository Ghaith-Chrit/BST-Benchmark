import random
from methods.node import Node
from typing import Optional, Iterable
from methods.base import BaseDataStructure


class Treap(BaseDataStructure):
    """
    Treap (randomized binary search tree): Each node has a key (value) and
    a randomly chosen priority. The tree satisfies:
      • BST property on values (left < node < right)
      • Max-heap (or min-heap) property on priorities.
    Expected height is O(log n).

    Public methods:
      - insert(value: str) -> bool       : Insert value; False if duplicate.
      - contains(value: str) -> bool  : Check membership.
      - delete(value: str) -> bool    : Delete value; True if present.
      - validate() -> None            : Verify BST + heap property + parent pointers.
    """

    def __init__(
        self,
        values: Optional[Iterable[str]] = None,
        max_priority: int = 10**6,
    ):
        """
        Initialize an empty Treap or insert initial values.

        Parameters
        ----------
        values : Optional[Iterable[str]]
            Iterable of string values to insert initially.
        """
        self.root: Optional[Node] = None
        self.max_priority = max_priority
        super().__init__(values)

    def _rotate_right(self, y: Node, is_delete: bool = False) -> Node:
        """
        Perform a right rotation around node `y` to maintain RB balance.

        Achieves this by:
          - Moving `y.left` (x) into `y`'s position.
          - Making `x.right` the new left child of `y`.
          - Updating parent pointers of all involved nodes.
          - Adjusting the tree root if necessary.

        This operation mirrors `_left_rotate` and preserves BST ordering.

        Before rotation:
              y
             / \
            x   T3
           / \
         T1  T2

        After rotation:
           x
          / \
        T1    y
             / \
            T2   T3

        Parameters
        ----------
        y : Node
            The pivot node for the rotation.
        is_delete : bool
            Used to track rotations per operation.
        """
        x = y.left
        if x is None:
            return y
        T2 = x.right
        x.right = y
        y.left = T2
        # update parents
        x.parent = y.parent
        y.parent = x
        if T2:
            T2.parent = y
        # adjust root if needed
        if x.parent is None:
            self.root = x
        else:
            if x.parent.left is y:
                x.parent.left = x
            else:
                x.parent.right = x

        if is_delete:
            self.rotations_delete += 1
        else:
            self.rotations_insert += 1

        return x

    def _rotate_left(self, x: Node, is_delete: bool = False) -> Node:
        """
        Perform a left rotation around node `x` to maintain RB balance.

        Achieves this by:
          - Moving `x.right` (y) into `x`'s position.
          - Making `y.left` the new right child of `x`.
          - Updating parent pointers of all involved nodes.
          - Adjusting the tree root if necessary.

        This preserves the BST property while locally restructuring
        the tree to maintain balance after insertions or deletions.

        Before rotation:
          x
         / \
        T1  y
           / \ 
          T2 T3

        After rotation:
             y
            / \
            x   T3
           / \
          T1  T2

        Parameters
        ----------
        x : Node
            The pivot node for the rotation.
        is_delete : bool
            Used to track rotations per operation.
        """
        y = x.right
        if y is None:
            return x
        T2 = y.left
        y.left = x
        x.right = T2
        # update parents
        y.parent = x.parent
        x.parent = y
        if T2:
            T2.parent = x
        if y.parent is None:
            self.root = y
        else:
            if y.parent.left is x:
                y.parent.left = y
            else:
                y.parent.right = y

        if is_delete:
            self.rotations_delete += 1
        else:
            self.rotations_insert += 1

        return y

    def insert(self, value: str) -> bool:
        """
        Insert a value into the treap by generating a random priority,
        then performing rotations to maintain heap property.

        Parameters
        ----------
        value : str
            The value to insert.

        Returns
        -------
        bool
            True if inserted; False if duplicate.
        """
        # normal BST insert
        if self.root is None:
            node = Node(value, priority=random.randint(0, self.max_priority))
            self.root = node
            return True

        cur = self.root
        parent = None
        while cur:
            parent = cur
            if value == cur.value:
                return False  # duplicate
            elif value < cur.value:
                cur = cur.left
            else:
                cur = cur.right

        new_node = Node(
            value, parent=parent, priority=random.randint(0, self.max_priority)
        )

        if value < parent.value:
            parent.left = new_node
        else:
            parent.right = new_node

        # now fix heap property: bubble new_node up via rotations
        cur = new_node
        while cur.parent and getattr(cur.parent, "priority", None) < cur.priority:
            if cur.parent.left is cur:
                self._rotate_right(cur.parent)
            else:
                self._rotate_left(cur.parent)

        return True

    def contains(self, value: str) -> bool:
        """
        Search for a value in the treap (BST search).

        Parameters
        ----------
        value : str
            Value to search.

        Returns
        -------
        bool
            True if found; False otherwise.
        """
        cur = self.root
        while cur:
            if value == cur.value:
                return True
            elif value < cur.value:
                cur = cur.left
            else:
                cur = cur.right
        return False

    def delete(self, value: str) -> bool:
        """
        Delete a value from the treap if present.

        Parameters
        ----------
        value : str
            The value to delete.

        Returns
        -------
        bool
            True if deleted; False if not present.
        """
        # search the node
        cur = self.root
        while cur and cur.value != value:
            if value < cur.value:
                cur = cur.left
            else:
                cur = cur.right
        if cur is None:
            return False

        # “rotate down” the node until it is a leaf, then remove
        while cur.left or cur.right:
            if cur.left is None:
                self._rotate_left(cur, is_delete=True)
            elif cur.right is None:
                self._rotate_right(cur, is_delete=True)
            else:
                # rotate the child with higher priority up
                if cur.left.priority > cur.right.priority:
                    self._rotate_right(cur, is_delete=True)
                else:
                    self._rotate_left(cur, is_delete=True)

        # now cur is leaf, remove it
        if cur.parent is None:
            self.root = None
        else:
            if cur.parent.left is cur:
                cur.parent.left = None
            else:
                cur.parent.right = None
        return True

    def validate(self) -> None:
        """
        Validate Treap invariants: BST property, heap property on priorities,
        and parent pointers.

        Raises
        ------
        AssertionError
            If any invariant is violated.
        """

        def _check(
            node: Optional[Node], min_val: Optional[str], max_val: Optional[str]
        ):
            if node is None:
                return
            # BST constraint
            if min_val is not None:
                assert (
                    node.value > min_val
                ), f"BST violated: {node.value!r} <= {min_val!r}"
            if max_val is not None:
                assert (
                    node.value < max_val
                ), f"BST violated: {node.value!r} >= {max_val!r}"
            # heap constraint
            if node.left:
                assert (
                    getattr(node.left, "priority", None) <= node.priority
                ), f"Heap violated: left child priority {node.left.priority} > node priority {node.priority}"
                assert node.left.parent is node, "Parent pointer wrong for left child"
            if node.right:
                assert (
                    getattr(node.right, "priority", None) <= node.priority
                ), f"Heap violated: right child priority {node.right.priority} > node priority {node.priority}"
                assert node.right.parent is node, "Parent pointer wrong for right child"

            _check(node.left, min_val, node.value)
            _check(node.right, node.value, max_val)

        _check(self.root, None, None)
