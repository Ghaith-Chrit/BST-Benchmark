from methods.node import Node
from typing import Iterable, Optional
from methods.base import BaseDataStructure


class AVLTree(BaseDataStructure):
    """
    AVL Tree using the common Node class.

    Public methods:
      - insert(value: str) -> bool : Inserts a string; returns False if already present.
      - contains(value: str) -> bool : Checks if a string exists in the tree.
      - delete(value: str) -> bool : Deletes a node by value; returns True if deleted.
      - validate() -> None : Checks AVL invariants, raises AssertionError if violated.

    Balances the tree after insertions and deletions by maintaining the
    AVL invariant: for every node, the heights of its left and right subtrees
    differ by at most 1. Rotations (single or double) are used to restore balance.
    """

    def __init__(self, values: Optional[Iterable[str]] = None):
        """
        Initialize an empty AVL tree or insert an initial set of values.

        Parameters
        ----------
        values : Optional[Iterable[str]]
            Iterable of string values to insert initially.
        """
        self.root: Optional[Node] = None
        super().__init__(values)

    def _get_height(self, node: Optional[Node]) -> int:
        """
        Return the height of a node.

        Parameters
        ----------
        node : Optional[Node]
            Node whose height is queried.

        Returns
        -------
        int
            Height of the node; 0 if node is None.
        """
        return node.height if node else 0

    def _update_height(self, node: Optional[Node]) -> None:
        """
        Update the height of a node based on its children's heights.

        Height = 1 + max(left height, right height).

        Called after any insertion, deletion, or rotation to keep node heights correct.

        Parameters
        ----------
        node : Optional[Node]
            Node whose height is updated.
        """
        if node:
            node.height = 1 + max(
                self._get_height(node.left), self._get_height(node.right)
            )

    def _get_balance(self, node: Optional[Node]) -> int:
        """
        Compute the balance factor of a node.

        Balance factor = left subtree height - right subtree height.

        Positive value indicates left-heavy, negative indicates right-heavy.
        Used to detect if rotations are needed.

        Parameters
        ----------
        node : Optional[Node]
            Node whose balance factor is computed.

        Returns
        -------
        int
            Balance factor = left subtree height - right subtree height.
            Positive if left-heavy, negative if right-heavy.
        """
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_right(self, y: Node, is_delete: bool = False) -> Node:
        """
        Perform a right rotation around node y.

        Before rotation:
              y
             / \
            x   T3
           / \
         T1  T2

        After rotation:
           x
         /   \
        T1    y
             / \
            T2  T3

        Parameters
        ----------
        y : Node
            Node around which the rotation is performed.

        Returns
        -------
        Node
            New root of the rotated subtree.
        is_delete : bool
            Used to track rotations per operation.
            
        Notes
        -----
        Promotes y.left (x) to be the new root of the subtree, updates parent
        pointers, and adjusts heights of affected nodes. Used to fix left-heavy
        imbalances (LL or LR cases).
        """
        x = y.left
        if x is None:
            # should not happen if called correctly
            return y

        T2 = x.right

        # rotate
        x.right = y
        y.left = T2

        # update parents
        x.parent = y.parent
        y.parent = x
        if T2:
            T2.parent = y

        # connect x to its new parent
        if x.parent is None:
            self.root = x
        else:
            if x.parent.left is y:
                x.parent.left = x
            else:
                x.parent.right = x

        # update heights
        self._update_height(y)
        self._update_height(x)

        # record rotation metric
        if is_delete:
            self.rotations_delete += 1
        else:
            self.rotations_insert += 1

        return x

    def _rotate_left(self, x: Node, is_delete: bool = False) -> Node:
        """
        Perform a left rotation around node x.

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
            Node around which the rotation is performed.

        Returns
        -------
        Node
            New root of the rotated subtree.
        is_delete : bool
            Used to track rotations per operation.

        Notes
        -----
        Promotes x.right (y) to be the new root of the subtree, updates parent
        pointers, and adjusts heights of affected nodes. Used to fix right-heavy
        imbalances (RR or RL cases).
        """
        y = x.right
        if y is None:
            # should not happen if called correctly
            return x

        T2 = y.left

        # rotate
        y.left = x
        x.right = T2

        # update parents
        y.parent = x.parent
        x.parent = y
        if T2:
            T2.parent = x

        # connect y to its new parent
        if y.parent is None:
            self.root = y
        else:
            if y.parent.left is x:
                y.parent.left = y
            else:
                y.parent.right = y

        # update heights
        self._update_height(x)
        self._update_height(y)

        # record rotation metric
        if is_delete:
            self.rotations_delete += 1
        else:
            self.rotations_insert += 1

        return y

    def insert(self, value: str) -> bool:
        """
        Insert a value into the AVL tree and rebalance.
        Standard BST insertion followed by walking up and rebalancing.

        Parameters
        ----------
        value : str
            Value of the node to insert.

        Returns
        -------
        bool
            True if the value was inserted; False if it was a duplicate.

        Notes
        -----
        Performs standard BST insertion, then walks up the tree to update heights
        and rebalance using rotations (LL, RR, LR, RL) as necessary.
        """
        if self.root is None:
            self.root = Node(value)
            return True

        # BST insert (iterative)
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

        new_node = Node(value, parent=parent)
        if value < parent.value:
            parent.left = new_node
        else:
            parent.right = new_node

        # walk back up and rebalance
        node = new_node.parent
        while node:
            self._update_height(node)
            balance = self._get_balance(node)

            # Left heavy
            if balance > 1:
                if self._get_balance(node.left) < 0:
                    # LR case
                    self._rotate_left(node.left)
                # LL case
                node = self._rotate_right(node)

            # Right heavy
            elif balance < -1:
                if self._get_balance(node.right) > 0:
                    # RL case
                    self._rotate_right(node.right)
                # RR case
                node = self._rotate_left(node)

            # move up
            # If a rotation occurred, node is the new subtree root; continue from its parent
            node = node.parent

        return True

    def contains(self, value: str) -> bool:
        """
        Check if the tree contains a given value.

        Parameters
        ----------
        value : str
            Value to search for.

        Returns
        -------
        bool
            True if the value exists in the tree, else False.

        Notes
        -----
        Standard BST search: traverse left if value < node, right if value > node.
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
        Delete a node by value.
        Uses standard BST delete followed by rebalancing (walking up).
        After deletion, walks up to update heights and rebalance using rotations
        as necessary to restore AVL property.

        Parameters
        ----------
        value : str
            Value of the node to delete.

        Returns
        -------
        bool
            True if the node was deleted, False if not found.

        Notes
        -----
        Performs standard BST deletion:
        - Leaf: remove directly.
        - One child: replace with child.
        - Two children: replace with inorder successor and delete successor.
        """

        def _delete_node(node: Optional[Node], value: str) -> Optional[Node]:
            """Recursively delete node and return new subtree root."""
            if node is None:
                return None
            if value < node.value:
                node.left = _delete_node(node.left, value)
                if node.left:
                    node.left.parent = node
            elif value > node.value:
                node.right = _delete_node(node.right, value)
                if node.right:
                    node.right.parent = node
            else:
                # found the node to delete
                if node.left is None or node.right is None:
                    # one child or none
                    replacement = node.left if node.left else node.right
                    if replacement:
                        replacement.parent = node.parent
                    return replacement
                else:
                    # two children: replace with inorder successor (min in right subtree)
                    succ = node.right
                    while succ.left:
                        succ = succ.left
                    # copy successor's value to node
                    node.value = succ.value
                    # delete successor
                    node.right = _delete_node(node.right, succ.value)
                    if node.right:
                        node.right.parent = node

            # After deletion, update height and rebalance this subtree (if node still exists)
            self._update_height(node)
            balance = self._get_balance(node)

            # Left heavy
            if balance > 1:
                if self._get_balance(node.left) < 0:
                    # LR
                    self._rotate_left(node.left, is_delete=True)
                node = self._rotate_right(node, is_delete=True)

            # Right heavy
            elif balance < -1:
                if self._get_balance(node.right) > 0:
                    # RL
                    self._rotate_right(node.right, is_delete=True)
                node = self._rotate_left(node, is_delete=True)

            return node

        # check existence
        if not self.contains(value):
            return False

        # delete and reassign root (helper ensures parents updated locally)
        self.root = _delete_node(self.root, value)
        if self.root:
            self.root.parent = None
        return True

    def validate(self) -> None:
        """
        Validate AVL tree invariants.

        Checks:
        1. BST ordering.
        2. Heights of nodes are correct.
        3. Balance factor of each node is -1, 0, or 1.

        Raises
        ------
        AssertionError
            If any Red-Black property is violated.
        """

        def _check(node: Optional[Node]) -> int:
            """Recursively check subtree. Returns subtree height."""
            if node is None:
                return 0

            # check left subtree
            left_height = _check(node.left)
            # check right subtree
            right_height = _check(node.right)

            # check BST property
            if node.left:
                assert (
                    node.left.value < node.value
                ), f"BST property violated at {node.value}"
            if node.right:
                assert (
                    node.right.value > node.value
                ), f"BST property violated at {node.value}"

            # check height
            expected_height = 1 + max(left_height, right_height)
            assert node.height == expected_height, f"Height incorrect at {node.value}"

            # check balance factor
            balance = left_height - right_height
            assert -1 <= balance <= 1, f"AVL balance violated at {node.value}"

            return expected_height

        if self.root:
            _check(self.root)
