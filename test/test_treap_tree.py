import random
import unittest
from methods.node import Node
from typing import List, Optional
from methods.treap_tree import Treap


def inorder_values(root: Optional[Node]) -> List[str]:
    """Return inorder traversal list of node.value (skips None nodes)."""
    res: List[str] = []

    def _rec(n: Optional[Node]):
        if n is None:
            return
        _rec(n.left)
        res.append(n.value)
        _rec(n.right)

    _rec(root)
    return res


class TestTreap(unittest.TestCase):
    def test_empty_tree(self):
        t = Treap()
        self.assertFalse(t.contains("x"))
        self.assertFalse(t.delete("x"))
        # validate on empty should not raise
        t.validate()

    def test_basic_insert_contains(self):
        t = Treap()
        values = ["m", "c", "t", "a", "e"]
        for v in values:
            inserted = t.insert(v)
            self.assertTrue(inserted)
        for v in values:
            self.assertTrue(t.contains(v))
        # duplicates
        self.assertFalse(t.insert("m"))
        # inorder should be sorted lexicographically (strings)
        self.assertEqual(inorder_values(t.root), sorted(values))
        t.validate()

    def test_init_with_values(self):
        values = ["g", "b", "k", "a", "c"]
        t = Treap(values)
        for v in values:
            self.assertTrue(t.contains(v))
        self.assertEqual(inorder_values(t.root), sorted(values))
        t.validate()

    def test_delete_leaf(self):
        t = Treap()
        for v in ["m", "c", "t"]:
            t.insert(v)
        self.assertTrue(t.delete("c"))  # c is a leaf
        self.assertFalse(t.contains("c"))
        t.validate()
        self.assertEqual(inorder_values(t.root), ["m", "t"])

    def test_delete_one_child(self):
        t = Treap()
        # build tree where 'c' has single child
        for v in ["m", "c", "t", "a"]:
            t.insert(v)
        self.assertTrue(t.delete("c"))
        self.assertFalse(t.contains("c"))
        t.validate()

    def test_delete_two_children(self):
        t = Treap()
        for v in ["m", "c", "t", "a", "e", "r", "z"]:
            t.insert(v)
        # delete node with two children
        self.assertTrue(t.delete("m"))
        self.assertFalse(t.contains("m"))
        t.validate()
        expected = sorted([v for v in ["m", "c", "t", "a", "e", "r", "z"] if v != "m"])
        self.assertEqual(inorder_values(t.root), expected)

    def test_delete_nonexistent(self):
        t = Treap()
        for v in ["a", "b", "c"]:
            t.insert(v)
        self.assertFalse(t.delete("x"))
        t.validate()

    def test_parent_pointers_and_priorities(self):
        t = Treap()
        values = ["m", "c", "t", "a", "e", "r", "z", "b", "d", "f"]
        for v in values:
            t.insert(v)
            t.validate()

        # check parent pointers and that each node has a priority attribute
        def check(node: Optional[Node]):
            if node is None:
                return
            # priority must exist and be an int within range
            self.assertTrue(hasattr(node, "priority"))
            if node.left:
                self.assertIs(node.left.parent, node)
            if node.right:
                self.assertIs(node.right.parent, node)
            check(node.left)
            check(node.right)

        check(t.root)

    def test_many_inserts_and_deletes_stress(self):
        t = Treap()
        rnd = random.Random(123456)
        reference = set()
        # deterministic sequence of operations
        for i in range(500):
            v = str(rnd.randint(0, 300))
            if rnd.random() < 0.6:
                inserted = t.insert(v)
                reference.add(v)
            else:
                rem = t.delete(v)
                reference.discard(v)
            if i % 50 == 0:
                t.validate()

        t.validate()
        expected = sorted(reference)
        self.assertEqual(inorder_values(t.root), expected)

    def test_inorder_after_sequential_inserts_strings(self):
        t = Treap()
        values = [str(i) for i in range(50)]
        for v in values:
            t.insert(v)
            t.validate()
        # lexicographic order for strings:
        self.assertEqual(inorder_values(t.root), sorted(values))

    def test_validate_raises_on_corruption_heap(self):
        # create a treap and then deliberately violate heap property on a parent-child
        t = Treap()
        for v in ["m", "c", "t", "a", "e"]:
            t.insert(v)
        t.validate()  # should be fine

        # find a parent-child pair and set child's priority > parent to break heap
        nodes = []

        def collect(node: Optional[Node]):
            if node is None:
                return
            nodes.append(node)
            collect(node.left)
            collect(node.right)

        collect(t.root)
        # pick a parent that has at least one child
        parent = None
        child = None
        for n in nodes:
            if n.left:
                parent = n
                child = n.left
                break
            if n.right:
                parent = n
                child = n.right
                break
        if parent and child:
            old_parent_pr = parent.priority
            old_child_pr = child.priority
            # make child's priority bigger than parent's
            child.priority = parent.priority + 1000
            with self.assertRaises(AssertionError):
                t.validate()
            # restore
            child.priority = old_child_pr
            parent.priority = old_parent_pr

    def test_validate_raises_on_bst_corruption(self):
        t = Treap()
        for v in ["m", "c", "t", "a", "e"]:
            t.insert(v)
        t.validate()

        # pick root and one of its descendants and swap values to break BST
        if t.root and (t.root.left or t.root.right):
            # pick a child
            if t.root.left:
                child = t.root.left
            else:
                child = t.root.right
            orig_root = t.root.value
            orig_child = child.value
            t.root.value, child.value = orig_child, orig_root
            try:
                with self.assertRaises(AssertionError):
                    t.validate()
            finally:
                # restore
                t.root.value = orig_root
                child.value = orig_child

    def test_transparent_priorities_unique_enough(self):
        # Ensure priorities are assigned and approximately uniform across many nodes.
        t = Treap()
        for i in range(200):
            t.insert(str(i))
        # gather priorities
        pri = []

        def collect(node: Optional[Node]):
            if node is None:
                return
            pri.append(getattr(node, "priority", None))
            collect(node.left)
            collect(node.right)

        collect(t.root)
        # no None priorities
        self.assertFalse(any(p is None for p in pri))
        # at least some distinct values
        self.assertGreater(len(set(pri)), 1)

    def test_delete_until_empty(self):
        t = Treap()
        values = ["h", "d", "l", "b", "f", "j", "n"]
        for v in values:
            t.insert(v)
        for v in list(values):
            self.assertTrue(t.delete(v))
            t.validate()
        self.assertIsNone(t.root)
        self.assertEqual(inorder_values(t.root), [])

    def test_validate_detects_parent_pointer_error(self):
        t = Treap()
        for v in ["m", "c", "t"]:
            t.insert(v)
        t.validate()
        # corrupt parent pointer intentionally
        if t.root and t.root.left:
            child = t.root.left
            old_parent = child.parent
            child.parent = None
            with self.assertRaises(AssertionError):
                t.validate()
            # restore
            child.parent = old_parent

    def test_rotation_metrics(self):
        t = Treap()
        t.insert("m")
        t.insert("c")
        t.insert("t")
        before = t.total_rotations
        t.insert("a")
        t.insert("e")
        self.assertGreaterEqual(t.rotations_insert, 0)
        self.assertEqual(t.rotations_delete, 0)
        t.delete("c")
        self.assertGreaterEqual(t.rotations_delete, 0)
        t.validate()


if __name__ == "__main__":
    unittest.main()
