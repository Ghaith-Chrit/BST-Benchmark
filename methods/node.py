from typing import Optional


class Node:
    """
    Common Node class used by both AVL and Red-Black trees.

    Attributes:
      value (Optional[str]) : the stored string (None is allowed for RB sentinel)
      parent, left, right (Optional[Node]) : Node links (or None / sentinel where appropriate)
      height (Optional[int]) : used by AVL algorithm (leaf = 1)
      color (Optional[bool]) : used by RB tree; use Node.RED/Node.BLACK
      priority (Optional[int]) : used by TREAP
    """

    RED = True
    BLACK = False

    def __init__(
        self,
        value: Optional[str],
        parent: Optional["Node"] = None,
        color: Optional[bool] = None,
        priority: Optional[int] = None,
    ):
        self.value = value
        self.parent = parent
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None
        self.height: int = 1  # meaningful for AVL
        self.color: Optional[bool] = color  # meaningful for RB; True=RED, False=BLACK
        self.priority: Optional[int] = priority

    def __repr__(self):
        return f"Node({self.value!r}, h={self.height}, c={'R' if self.color else 'B' if self.color is not None else 'N'})"
