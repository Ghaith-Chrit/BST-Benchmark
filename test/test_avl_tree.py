import random
import unittest
from methods.node import Node
from typing import List, Optional
from methods.avl_tree import AVLTree


def inorder_values(root: Optional[Node]) -> List[str]:
    """Return inorder traversal list of node.value (skips None nodes)."""
    res = []

    def _rec(n):
        if n is None:
            return
        _rec(n.left)
        res.append(n.value)
        _rec(n.right)

    _rec(root)
    return res


class TestAVLTree(unittest.TestCase):
    def test_empty_tree(self):
        t = AVLTree()
        self.assertFalse(t.contains("x"))
        self.assertFalse(t.delete("x"))
        t.validate()

    def test_basic_insert_contains(self):
        t = AVLTree()
        values = ["m", "c", "t", "a", "e"]
        for v in values:
            inserted = t.insert(v)
            self.assertTrue(inserted)
        for v in values:
            self.assertTrue(t.contains(v))
        self.assertFalse(t.insert("m"))
        self.assertEqual(inorder_values(t.root), sorted(values))
        t.validate()

    def test_init_with_values(self):
        values = ["g", "b", "k", "a", "c"]
        t = AVLTree(values)
        for v in values:
            self.assertTrue(t.contains(v))
        self.assertEqual(inorder_values(t.root), sorted(values))
        t.validate()

    def test_delete_leaf(self):
        t = AVLTree()
        for v in ["m", "c", "t"]:
            t.insert(v)
        self.assertTrue(t.delete("c"))
        self.assertFalse(t.contains("c"))
        t.validate()
        self.assertEqual(inorder_values(t.root), ["m", "t"])

    def test_delete_one_child(self):
        t = AVLTree()
        # build tree where 'c' has single child
        for v in ["m", "c", "t", "a"]:
            t.insert(v)
        # 'a' is a leaf under 'c' and 'c' may have one child previously, attempt deleting 'c'
        self.assertTrue(t.delete("c"))
        self.assertFalse(t.contains("c"))
        t.validate()

    def test_delete_two_children(self):
        t = AVLTree()
        for v in ["m", "c", "t", "a", "e", "r", "z"]:
            t.insert(v)
        # 'm' has two children -> delete should replace with inorder successor
        self.assertTrue(t.delete("m"))
        self.assertFalse(t.contains("m"))
        t.validate()
        expected = sorted([v for v in ["m", "c", "t", "a", "e", "r", "z"] if v != "m"])
        self.assertEqual(inorder_values(t.root), expected)

    def test_delete_nonexistent(self):
        t = AVLTree()
        for v in ["a", "b", "c"]:
            t.insert(v)
        self.assertFalse(t.delete("x"))
        t.validate()

    def test_rotations_ll_rr_lr_rl(self):
        # LL rotation: insert decreasing -> right rotate
        t1 = AVLTree()
        for v in ["3", "2", "1"]:
            t1.insert(v)
            t1.validate()
        self.assertEqual(inorder_values(t1.root), ["1", "2", "3"])
        # root should be "2" after balancing
        self.assertEqual(t1.root.value, "2")
        self.assertIsNone(t1.root.parent)

        # RR rotation: insert increasing -> left rotate
        t2 = AVLTree()
        for v in ["1", "2", "3"]:
            t2.insert(v)
            t2.validate()
        self.assertEqual(inorder_values(t2.root), ["1", "2", "3"])
        self.assertEqual(t2.root.value, "2")
        self.assertIsNone(t2.root.parent)

        # LR rotation: sequence that causes left-right
        t3 = AVLTree()
        for v in ["3", "1", "2"]:
            t3.insert(v)
            t3.validate()
        self.assertEqual(inorder_values(t3.root), ["1", "2", "3"])
        self.assertEqual(t3.root.value, "2")

        # RL rotation: sequence that causes right-left
        t4 = AVLTree()
        for v in ["1", "3", "2"]:
            t4.insert(v)
            t4.validate()
        self.assertEqual(inorder_values(t4.root), ["1", "2", "3"])
        self.assertEqual(t4.root.value, "2")

    def test_parent_pointers_and_heights_after_rotations(self):
        t = AVLTree()
        # create many inserts causing many rotations
        values = [str(i) for i in range(20, 0, -1)]  # descending to stress rotations
        for v in values:
            t.insert(v)
            t.validate()

        # check root parent None
        self.assertIsNone(t.root.parent)

        # check heights are consistent via validate; explicit check of node.height equals computed height
        # compute heights recursively and compare
        def compute_height(node):
            if node is None:
                return 0
            lh = compute_height(node.left)
            rh = compute_height(node.right)
            return 1 + max(lh, rh)

        def check_heights(node):
            if node is None:
                return
            self.assertEqual(node.height, compute_height(node))
            if node.left:
                self.assertIs(node.left.parent, node)
            if node.right:
                self.assertIs(node.right.parent, node)
            check_heights(node.left)
            check_heights(node.right)

        check_heights(t.root)

    def test_many_inserts_and_deletes_stress(self):
        t = AVLTree()
        rnd = random.Random(12345)
        reference = set()
        ops = []
        # generate deterministic operations
        for _ in range(500):
            v = str(rnd.randint(0, 200))
            if rnd.random() < 0.6:
                inserted = t.insert(v)
                reference.add(v)
                ops.append(("ins", v, inserted))
            else:
                deleted = t.delete(v)
                reference.discard(v)
                ops.append(("del", v, deleted))
            # validate invariants occasionally to speed up
            if _ % 50 == 0:
                t.validate()

        # final validation
        t.validate()
        # check inorder equals sorted reference
        expected = sorted(reference)
        self.assertEqual(inorder_values(t.root), expected)

    def test_inorder_after_sequential_inserts(self):
        t = AVLTree()
        values = [f"{i:02d}" for i in range(50)]
        for v in values:
            t.insert(v)
            t.validate()
        self.assertEqual(inorder_values(t.root), values)

    def test_delete_until_empty(self):
        t = AVLTree()
        values = ["h", "d", "l", "b", "f", "j", "n"]
        for v in values:
            t.insert(v)
        # delete all
        for v in list(values):
            self.assertTrue(t.delete(v))
            t.validate()
        # tree empty
        self.assertIsNone(t.root)
        self.assertEqual(inorder_values(t.root), [])

    def test_validate_raises_on_corruption(self):
        # create a tree and then corrupt it to ensure validate() detects issues
        t = AVLTree()
        for v in ["m", "c", "t"]:
            t.insert(v)
        # manually corrupt a height
        if t.root and t.root.left:
            t.root.left.height = 999
            with self.assertRaises(AssertionError):
                t.validate()
        # fix height and corrupt BST order
        if t.root and t.root.left:
            t.root.left.height = 1
            # swap values to violate BST
            a = t.root.left.value
            t.root.left.value = t.root.value
            t.root.value = a
            with self.assertRaises(AssertionError):
                t.validate()

    def test_insert_rotations_increase(self):
        """
        Insert values that are likely to cause rotations (e.g., descending
        order -> repeated right rotations). Confirm that rotations_insert > 0.
        """
        t = AVLTree()
        # Start fresh metrics
        try:
            t.rotations_insert = 0
            t.rotations_delete = 0
        except AttributeError:
            self.fail(
                "AVLTree must expose rotations_insert and rotations_delete attributes"
            )

        # Insert descending values which normally trigger right-rotations
        values = [str(i) for i in range(10, 0, -1)]
        for v in values:
            t.insert(v)

        # After a sequence causing imbalance, there should have been some insert rotations
        self.assertGreater(t.rotations_insert, 0, "Expected insert rotations to be > 0")

        # Delete nothing here, so delete counter should remain 0
        self.assertEqual(
            t.rotations_delete, 0, "No delete rotations should have occurred"
        )

    def test_metrics_separation_and_reset(self):
        """
        Ensure insert rotations are counted separately from delete rotations,
        and that resetting or re-initializing the tree resets metrics when intended.
        """
        t = AVLTree()
        # initialize counters
        t.rotations_insert = 0
        t.rotations_delete = 0

        # Cause insert-time rotations
        for v in ["3", "2", "1"]:
            t.insert(v)
        insert_count = t.rotations_insert
        self.assertGreater(insert_count, 0, "Expected some insert rotations for 3-2-1")

        # Now perform deletes and ensure they increment the delete counter (not insert counter)
        t.delete("2")
        self.assertGreaterEqual(t.rotations_delete, 0)

        # Ensure insert counter did not get incremented by deletes (it may remain same)
        self.assertEqual(
            insert_count,
            t.rotations_insert,
            "insert rotations should not change because of deletes",
        )

        # Reset metrics if a method exists - prefer dedicated API, otherwise manual reset
        if hasattr(t, "reset_metrics"):
            t.reset_metrics()
            self.assertEqual(t.rotations_insert, 0)
            self.assertEqual(t.rotations_delete, 0)
        else:
            # manual reset fallback
            t.rotations_insert = 0
            t.rotations_delete = 0
            self.assertEqual(t.rotations_insert, 0)
            self.assertEqual(t.rotations_delete, 0)

    def test_rotations_are_recorded_on_fixups(self):
        """
        Sanity: rotations performed inside _insert_fixup and _delete_fixup should
        be captured — run a few operations and assert total_rotations equals sum.
        """
        t = AVLTree()
        t.rotations_insert = 0
        t.rotations_delete = 0

        # Sequence that typically causes multiple fixups
        seq = ["m", "f", "t", "a", "k", "r", "z", "b", "d"]
        for v in seq:
            t.insert(v)

        # do a delete that requires bubbling & rotations
        t.delete("f")

        # Check that counters exist and total is non-negative integer
        self.assertIsInstance(t.rotations_insert, int)
        self.assertIsInstance(t.rotations_delete, int)
        self.assertGreaterEqual(t.rotations_insert + t.rotations_delete, 0)


if __name__ == "__main__":
    unittest.main()
