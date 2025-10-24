from methods.node import Node
from typing import Iterable, Optional
from methods.base import BaseDataStructure


class RBTree(BaseDataStructure):
    """
    Standard Red-Black Tree implementation using a shared Node class and a sentinel nil node.

    Red-Black Tree properties are maintained on all insertions and deletions:
      1. Every node is either red or black.
      2. The root is black.
      3. All leaves (nil) are black.
      4. Red nodes cannot have red children.
      5. Every path from a node to its descendant leaves has the same number of black nodes.

    Public methods:
      - insert(value: str) -> bool : Inserts a string; returns False if already present.
      - contains(value: str) -> bool : Checks if a string exists in the tree.
      - delete(value: str) -> bool : Deletes a node by value; returns True if deleted.
      - validate() -> None : Checks RB invariants, raises AssertionError if violated.
    """

    def __init__(self, values: Optional[Iterable[str]] = None):
        """
        Initialize an empty Red-Black Tree with a sentinel nil node.

        The sentinel `nil` represents all leaves and ensures that leaf checks
        can be done without `None` comparisons. The root initially points to
        `nil`. If `values` is provided, they are inserted in order.

        Parameters
        ----------
        values : Optional[Iterable[str]], default=None
            If provided, all values are inserted into the tree at initialization.
        """

        # create the sentinel nil node (black)
        # nil points to itself to avoid None checks
        self.nil = Node(None, parent=None, color=Node.BLACK)
        self.nil.left = self.nil.right = self.nil.parent = self.nil
        self.root: Node = self.nil
        super().__init__(values)

    def contains(self, value: str) -> bool:
        """
        Check if the tree contains a node with the given value.

        Achieves this by performing a standard BST search:
        starting from the root, repeatedly move left or right depending
        on comparison with the current node until the value is found
        or a leaf is reached.

        Parameters
        ----------
        value : str
            The value to search for.

        Returns
        -------
        bool
            True if the value exists, False otherwise.
        """
        cur = self.root
        while cur is not self.nil:
            if value == cur.value:
                return True
            elif value < cur.value:
                cur = cur.left
            else:
                cur = cur.right
        return False

    def _left_rotate(self, x: Node, is_delete: bool = False) -> None:
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
        if y is self.nil:
            return
        x.right = y.left
        if y.left is not self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self.nil:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

        if is_delete:
            self.rotations_delete += 1
        else:
            self.rotations_insert += 1

    def _right_rotate(self, y: Node, is_delete: bool = False) -> None:
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
        if x is self.nil:
            return
        y.left = x.right
        if x.right is not self.nil:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is self.nil:
            self.root = x
        elif y is y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x

        if is_delete:
            self.rotations_delete += 1
        else:
            self.rotations_insert += 1

    def insert(self, value: str) -> bool:
        """
        Insert a new string into the Red-Black Tree.

        Achieves this by:
          1. Performing a standard BST insertion to find the correct location.
          2. Inserting a new red node at that location.
          3. Calling `_insert_fixup` to restore Red-Black properties, which may:
             - Recolor the new node, parent, and uncle.
             - Perform rotations if necessary.

        Parameters
        ----------
        value : str
            The value to insert.

        Returns
        -------
        bool
            False if the value already exists in the tree, True otherwise.
        """
        node = Node(value, color=Node.RED)
        node.left = node.right = node.parent = self.nil

        y = self.nil
        x = self.root
        # standard BST insert to find spot
        while x is not self.nil:
            y = x
            if node.value == x.value:
                return False
            elif node.value < x.value:
                x = x.left
            else:
                x = x.right

        node.parent = y
        if y is self.nil:
            # tree was empty
            self.root = node
        elif node.value < y.value:
            y.left = node
        else:
            y.right = node

        # fixup to restore RB properties
        self._insert_fixup(node)
        return True

    def _insert_fixup(self, z: Node) -> None:
        """
        Restore Red-Black Tree properties after inserting node z.

        Achieves this by repeatedly checking if z's parent is red.
        There are three primary cases (and their mirror cases):
          - Case 1: Uncle is red: recolor parent, uncle, and grandparent.
          - Case 2: Uncle is black and z is a right child: left-rotate parent.
          - Case 3: Uncle is black and z is a left child: right-rotate grandparent.
        This loop continues until the tree satisfies all RB properties.
        """
        while z.parent.color == Node.RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == Node.RED:
                    # Case 1: uncle red
                    z.parent.color = Node.BLACK
                    y.color = Node.BLACK
                    z.parent.parent.color = Node.RED
                    z = z.parent.parent
                else:
                    # uncle black
                    if z is z.parent.right:
                        # Case 2
                        z = z.parent
                        self._left_rotate(z)
                    # Case 3
                    z.parent.color = Node.BLACK
                    z.parent.parent.color = Node.RED
                    self._right_rotate(z.parent.parent)
            else:
                # symmetric cases
                y = z.parent.parent.left
                if y.color == Node.RED:
                    z.parent.color = Node.BLACK
                    y.color = Node.BLACK
                    z.parent.parent.color = Node.RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = Node.BLACK
                    z.parent.parent.color = Node.RED
                    self._left_rotate(z.parent.parent)
        self.root.color = Node.BLACK

    def _transplant(self, u: Node, v: Node) -> None:
        """
        Replace the subtree rooted at u with the subtree rooted at v.

        Achieves this by updating u's parent to point to v.
        This is used during deletion to move subtrees without breaking the BST property.

        Parameters
        ----------
        u : Node
            Node to be replaced.
        v : Node
            Node that replaces u.
        """
        if u.parent is self.nil:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, node: Node) -> Node:
        """
        Find the minimum node in the subtree rooted at `node`.

        Achieves this by repeatedly following the left child until
        reaching the sentinel `nil`.

        Parameters
        ----------
        node : Node
            The root of the subtree.

        Returns
        -------
        Node
            The node with the smallest value in the subtree.
        """
        while node.left is not self.nil:
            node = node.left
        return node

    def delete(self, value: str) -> bool:
        """
        Delete a node with the given value from the Red-Black Tree.

        Achieves this by:
          1. Performing a standard BST search to find the node.
          2. Replacing the node with its successor (if necessary) using _transplant.
          3. Storing the original color of the removed node to determine if fixup is needed.
          4. Calling `_delete_fixup` if a black node was removed to restore RB properties.

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
        Implements CLRS-style Red-Black deletion with fixup to maintain
        Red-Black properties.
        """
        z = self.root
        # find node z with given value
        while z is not self.nil:
            if value == z.value:
                break
            elif value < z.value:
                z = z.left
            else:
                z = z.right
        else:
            return False  # not found

        y = z
        y_original_color = y.color
        if z.left is self.nil:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is self.nil:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        if y_original_color == Node.BLACK:
            self._delete_fixup(x)

        return True

    def _delete_fixup(self, x: Node) -> None:
        """
        Restore Red-Black properties after deletion.

        Achieves this by:
          - Iteratively examining x's sibling (w) and cases depending on w's color
            and the colors of w's children.
          - Performing recoloring and rotations according to CLRS cases 1-4
            until the black-height property and red-red violations are resolved.

        Parameters
        ----------
        x : Node
            Node that moved into the removed node's original position.
        """
        while x is not self.root and x.color == Node.BLACK:
            if x is x.parent.left:
                w = x.parent.right
                # Case 1
                if w.color == Node.RED:
                    w.color = Node.BLACK
                    x.parent.color = Node.RED
                    self._left_rotate(x.parent, is_delete=True)
                    w = x.parent.right
                # Case 2
                if w.left.color == Node.BLACK and w.right.color == Node.BLACK:
                    w.color = Node.RED
                    x = x.parent
                else:
                    # Case 3
                    if w.right.color == Node.BLACK:
                        w.left.color = Node.BLACK
                        w.color = Node.RED
                        self._right_rotate(w, is_delete=True)
                        w = x.parent.right
                    # Case 4
                    w.color = x.parent.color
                    x.parent.color = Node.BLACK
                    w.right.color = Node.BLACK
                    self._left_rotate(x.parent, is_delete=True)
                    x = self.root
            else:
                # symmetric
                w = x.parent.left
                if w.color == Node.RED:
                    w.color = Node.BLACK
                    x.parent.color = Node.RED
                    self._right_rotate(x.parent, is_delete=True)
                    w = x.parent.left
                if w.right.color == Node.BLACK and w.left.color == Node.BLACK:
                    w.color = Node.RED
                    x = x.parent
                else:
                    if w.left.color == Node.BLACK:
                        w.right.color = Node.BLACK
                        w.color = Node.RED
                        self._left_rotate(w, is_delete=True)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = Node.BLACK
                    w.left.color = Node.BLACK
                    self._right_rotate(x.parent, is_delete=True)
                    x = self.root
        x.color = Node.BLACK

    def validate(self) -> None:
        """
        Validate Red-Black Tree invariants.

        Checks:
        1. Root is black.
        2. No red node has a red child.
        3. Black-height (number of black nodes to leaves) is consistent across all paths.
        4. BST ordering: left child < node < right child.

        Raises
        ------
        AssertionError
            If any Red-Black or BST property is violated.
        """

        def check_node(
            node: Node, min_val: Optional[str] = None, max_val: Optional[str] = None
        ) -> int:
            """
            Recursively check subtree rooted at `node`.

            Parameters
            ----------
            node : Node
                Current node to validate.
            min_val : Optional[str]
                Minimum allowed value for BST property (exclusive).
            max_val : Optional[str]
                Maximum allowed value for BST property (exclusive).

            Returns
            -------
            int
                Black height of the subtree rooted at this node.
            """
            if node is self.nil:
                return 1  # black height of leaf

            # BST ordering
            if min_val is not None:
                assert node.value > min_val, f"BST violated: {node.value} <= {min_val}"
            if max_val is not None:
                assert node.value < max_val, f"BST violated: {node.value} >= {max_val}"

            # Check left and right subtrees
            left_black_height = check_node(node.left, min_val, node.value)
            right_black_height = check_node(node.right, node.value, max_val)

            # Black-height must match
            assert (
                left_black_height == right_black_height
            ), f"Black heights differ at {node.value}"

            # Red node cannot have red children
            if node.color == Node.RED:
                assert (
                    node.left.color == Node.BLACK
                ), f"Red node {node.value} has red left child"
                assert (
                    node.right.color == Node.BLACK
                ), f"Red node {node.value} has red right child"

            # Return black height for this node
            return left_black_height + (1 if node.color == Node.BLACK else 0)

        if self.root is not self.nil:
            assert self.root.color == Node.BLACK, "Root must be black"
        check_node(self.root)
