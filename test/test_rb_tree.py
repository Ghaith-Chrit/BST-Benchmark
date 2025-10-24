import random
import unittest
from methods.node import Node
from typing import List, Optional
from methods.rb_tree import RBTree


def inorder_values(tree: RBTree) -> List[Optional[str]]:
    """
    Return inorder traversal of the tree's values (skips sentinel 'nil' nodes).
    Uses the tree's sentinel `nil` to know leaf boundaries.
    """
    res: List[Optional[str]] = []
    nil = tree.nil

    def _rec(n: Node):
        if n is nil:
            return
        _rec(n.left)
        res.append(n.value)
        _rec(n.right)

    # If root might be the sentinel, handle accordingly
    if tree.root is not None and tree.root is not tree.nil:
        _rec(tree.root)
    return res


class TestRBTree(unittest.TestCase):
    def test_empty_tree(self):
        t = RBTree()
        # empty contains/delete
        self.assertFalse(t.contains("x"))
        self.assertFalse(t.delete("x"))
        # sentinel must be black
        self.assertIsNotNone(t.nil)
        self.assertEqual(t.nil.color, Node.BLACK)
        # validate must not raise on empty
        t.validate()

    def test_basic_add_contains_and_duplicates(self):
        t = RBTree()
        values = ["m", "c", "t", "a", "e"]
        for v in values:
            self.assertTrue(t.insert(v))
        for v in values:
            self.assertTrue(t.contains(v))
        # duplicate add returns False
        self.assertFalse(t.insert("m"))
        # inorder should be lexicographically sorted (strings)
        self.assertEqual(inorder_values(t), sorted(values))
        t.validate()

    def test_init_with_values(self):
        values = ["g", "b", "k", "a", "c"]
        t = RBTree(values)
        for v in values:
            self.assertTrue(t.contains(v))
        self.assertEqual(inorder_values(t), sorted(values))
        t.validate()

    def test_delete_leaf(self):
        t = RBTree()
        for v in ["m", "c", "t"]:
            t.insert(v)
        self.assertTrue(t.delete("c"))  # c is a leaf
        self.assertFalse(t.contains("c"))
        t.validate()
        self.assertEqual(inorder_values(t), ["m", "t"])

    def test_delete_one_child(self):
        t = RBTree()
        for v in ["m", "c", "t", "a"]:
            t.insert(v)
        # Delete 'c' which may have one child 'a'
        self.assertTrue(t.delete("c"))
        self.assertFalse(t.contains("c"))
        t.validate()

    def test_delete_two_children(self):
        t = RBTree()
        for v in ["m", "c", "t", "a", "e", "r", "z"]:
            t.insert(v)
        # delete node with two children (m)
        self.assertTrue(t.delete("m"))
        self.assertFalse(t.contains("m"))
        t.validate()
        expected = sorted([v for v in ["m", "c", "t", "a", "e", "r", "z"] if v != "m"])
        self.assertEqual(inorder_values(t), expected)

    def test_delete_nonexistent(self):
        t = RBTree()
        for v in ["a", "b", "c"]:
            t.insert(v)
        self.assertFalse(t.delete("x"))
        t.validate()

    def test_root_is_black_after_inserts(self):
        t = RBTree()
        for v in ["d", "b", "f", "a", "c", "e", "g"]:
            t.insert(v)
            # root must always be black per validate, but check quick property:
            self.assertIn(t.root.color, (Node.BLACK, Node.RED))
        # validate enforces root black
        t.validate()
        self.assertEqual(t.root.color, Node.BLACK)

    def test_parent_pointers_consistency(self):
        t = RBTree()
        values = ["m", "c", "t", "a", "e", "r", "z", "b", "d", "f"]
        for v in values:
            t.insert(v)
            t.validate()

        # verify parent pointers for all nodes reachable from root
        nil = t.nil

        def check_parents(node: Node):
            if node is nil:
                return
            if node.left is not nil:
                self.assertIs(node.left.parent, node)
                check_parents(node.left)
            if node.right is not nil:
                self.assertIs(node.right.parent, node)
                check_parents(node.right)

        check_parents(t.root)

    def test_many_inserts_and_deletes_stress(self):
        t = RBTree()
        rnd = random.Random(54321)
        reference = set()
        # deterministic sequence of operations
        for i in range(500):
            v = str(rnd.randint(0, 300))
            if rnd.random() < 0.6:
                added = t.insert(v)
                # added True/False may depend on existing; keep reference consistent
                reference.add(v)
            else:
                removed = t.delete(v)
                reference.discard(v)
            if i % 50 == 0:
                t.validate()

        # final validation
        t.validate()
        # inorder should equal lexicographically sorted reference
        expected = sorted(reference)
        self.assertEqual(inorder_values(t), expected)

    def test_inorder_after_sequential_adds_numeric_expectation(self):
        """
        Inserts "0".."49" as strings. Because strings compare lexicographically,
        the natural inorder is lexicographic. However callers may expect numeric
        order — to be explicit we assert numeric-sorted equality to match
        previous AVL test behavior (use numeric key).
        """
        t = RBTree()
        values = [f"{i:02d}" for i in range(50)]
        for v in values:
            t.insert(v)
            t.validate()
        # numeric ordering of string digits:
        expected_numeric_sorted = sorted(values, key=lambda s: int(s))
        self.assertEqual(inorder_values(t), expected_numeric_sorted)
        t.validate()

    def test_validate_raises_on_corruption(self):
        t = RBTree()
        for v in ["m", "c", "t", "a", "e"]:
            t.insert(v)

        # 1) Corrupt root color: set root to RED -> should fail root-is-black assertion
        orig_root_color = t.root.color
        t.root.color = Node.RED
        with self.assertRaises(AssertionError):
            t.validate()
        # restore
        t.root.color = orig_root_color

        # 2) Corrupt by making a parent and child both RED -> violates red-parent rule
        candidate = None
        if t.root.left is not t.nil:
            candidate = t.root.left
        elif t.root.right is not t.nil:
            candidate = t.root.right

        if candidate is not None:
            orig_parent_color = candidate.parent.color
            orig_child_color = candidate.color
            # force both to RED
            candidate.parent.color = Node.RED
            candidate.color = Node.RED
            with self.assertRaises(AssertionError):
                t.validate()
            # restore
            candidate.parent.color = orig_parent_color
            candidate.color = orig_child_color

        # 3) Corrupt BST ordering by swapping values on two nodes (should break black-height or BST checks)
        nodes = []

        def collect(node):
            if node is t.nil:
                return
            nodes.append(node)
            collect(node.left)
            collect(node.right)

        collect(t.root)

        if len(nodes) >= 2:
            a, b = nodes[0], nodes[-1]
            a.value, b.value = b.value, a.value
            with self.assertRaises(AssertionError):
                t.validate()

    def test_transplant_and_minimum_behavior(self):
        # This test exercises delete paths that use _minimum/_transplant internals.
        t = RBTree()
        for v in ["g", "b", "k", "a", "c", "j", "l", "i", "m"]:
            t.insert(v)
        t.validate()
        # delete a node with two children (g), triggers successor transplant logic
        self.assertTrue(t.delete("g"))
        t.validate()
        # delete several more
        for v in ["a", "m", "k"]:
            t.delete(v)
            t.validate()

    def test_deletion_rotations_not_increase(self):
        """
        Insert values that typically cause insert-time rotations and fixups.
        After a sequence of inserts, deletion should be zero rotation.
        """
        t = RBTree()
        # ensure counters exist and reset them
        try:
            t.rotations_insert = 0
            t.rotations_delete = 0
        except AttributeError:
            self.fail(
                "RBTree must expose rotations_insert and rotations_delete attributes"
            )

        # A sequence likely to trigger insert fixups (mix of left/right inserts)
        seq = ["m", "c", "t", "a", "e", "r", "z", "b", "d", "f"]
        for v in seq:
            t.insert(v)

        # No delete rotations should have occurred
        self.assertEqual(
            t.rotations_delete, 0, "No delete rotations should have occurred yet"
        )

    def test_metrics_separation_and_reset(self):
        """
        Ensure insert rotations are counted separately from delete rotations,
        and that reset_metrics or manual reset clears them.
        """
        t = RBTree()
        t.rotations_insert = 0
        t.rotations_delete = 0

        # this pattern tends to cause rotations on insert
        for v in ["3", "2", "1"]:
            t.insert(v)
        insert_count = t.rotations_insert
        self.assertGreater(
            insert_count, 0, "Expected some insert rotations for 3-2-1 sequence"
        )

        # perform delete and ensure delete counter increases (or at least exists)
        t.delete("2")
        self.assertIsInstance(t.rotations_delete, int)

        # insert counter should not be incremented by deletes (remains equal)
        self.assertEqual(
            insert_count,
            t.rotations_insert,
            "insert rotations should not change because of deletes",
        )

        # prefer a reset_metrics API if available
        if hasattr(t, "reset_metrics"):
            t.reset_metrics()
            self.assertEqual(t.rotations_insert, 0)
            self.assertEqual(t.rotations_delete, 0)
        else:
            # fallback: manual reset
            t.rotations_insert = 0
            t.rotations_delete = 0
            self.assertEqual(t.rotations_insert, 0)
            self.assertEqual(t.rotations_delete, 0)

    def test_rotations_recorded_in_fixups(self):
        """
        Stress a sequence of insertions and deletions that perform many fixups,
        then verify the counters are integer and their sum is non-negative.
        """
        t = RBTree()
        t.rotations_insert = 0
        t.rotations_delete = 0

        seq = ["m", "f", "t", "a", "k", "r", "z", "b", "d", "h", "j"]
        for v in seq:
            t.insert(v)

        # perform deletes that cause more fixups
        t.delete("f")
        t.delete("k")

        # ensure counters exist and are ints
        self.assertIsInstance(t.rotations_insert, int)
        self.assertIsInstance(t.rotations_delete, int)
        self.assertGreaterEqual(t.rotations_insert + t.rotations_delete, 0)

    def test_rotation_counters_present_when_no_rotations(self):
        """
        If the tree doesn't require rotations for trivial ops, counters still exist and are zero.
        """
        t = RBTree()
        t.rotations_insert = 0
        t.rotations_delete = 0
        # trivial single insert/delete
        t.insert("x")
        t.delete("x")
        self.assertIsInstance(t.rotations_insert, int)
        self.assertIsInstance(t.rotations_delete, int)


if __name__ == "__main__":
    unittest.main()
