import gc
import time
import random
import statistics
from methods.rb_tree import RBTree
from methods.treap_tree import Treap
from methods.avl_tree import AVLTree
from typing import Iterable, List, Dict, Any, Optional
from utils.tree_util import compute_balance_metrics, tree_height


class TreeBenchmark:
    """
    Benchmark harness for comparing tree-based data structures: AVLTree, RBTree, and Treap.

    The benchmark measures insertion, lookup, and deletion performance, as well as
    tree structural properties such as height, average node depth, subtree imbalance,
    and rotation counts. It can run multiple trials and aggregate results using medians.

    Features
    --------
    - Measures wall-clock time for bulk insert, lookup, and delete operations.
    - Computes structural metrics:
        - `height_after_insert` / `height_after_delete`: maximum depth of the tree.
        - `avg_depth_after_insert` / `avg_depth_after_delete`: average depth of nodes.
        - `max_subtree_imbalance_after_insert` / `max_subtree_imbalance_after_delete`:
          largest difference in height between left and right subtrees at any node.
    - Tracks rotation counts if supported by the tree implementation.
    - Validates tree correctness after insertions and deletions using `validate()`.
    - Supports different workload patterns: random, ascending, descending, and hotspot.
    - Aggregates results across multiple trials using medians.

    Parameters
    ----------
    dataset : Iterable[str]
        The items to insert into the tree.
    queries : Iterable[str]
        Items to query and delete. Should include both present and absent items.
    include : Optional[List[str]], default=None
        List of tree names to benchmark. Choices: ["avl", "rb", "treap"].
        If None, all trees are benchmarked.
    trials : int, default=1
        Number of repeated trials per structure; metrics are aggregated using the median.
    random_seed : Optional[int], default=12345
        Seed for randomization, used in hotspot workloads to generate skewed queries.
    treap_max_priority : int, default=10**6
        Maximum random priority for Treap nodes.

    Methods
    -------
    run(workload: str = "random") -> Dict[str, Dict[str, Any]]
        Run benchmark for each included tree on the specified workload pattern.

        Workloads:
          - "random": dataset and queries shuffled randomly.
          - "ascending": dataset inserted in ascending order.
          - "descending": dataset inserted in descending order.
          - "hotspot": queries are skewed, 80% targeting 10% of dataset keys.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping from tree name to aggregated metrics, including:
                - `insert_sec_median`, `lookup_sec_median`, `delete_sec_median`: median wall-clock times.
                - `insert_ops_per_sec`, `lookup_ops_per_sec`, `delete_ops_per_sec`: median throughput.
                - `rotations_insert_median`, `rotations_delete_median`: median rotation counts.
                - `height_after_insert_median`, `height_after_delete_median`: median max tree height.
                - `avg_depth_after_insert_median`, `avg_depth_after_delete_median`: median average node depth.
                - `max_subtree_imbalance_after_insert_median`, `max_subtree_imbalance_after_delete_median`: median subtree imbalance.
                - `validate_after_insert_all_trials`, `validate_after_delete_all_trials`: boolean indicating correctness across all trials.

    Notes
    -----
    - Depth counting starts at 1 for the root.
    - Supports optional tracking of rotations for trees that implement it.
    - Hotspot workload allows testing performance under skewed access patterns.
    - Useful for empirical analysis of balancing efficiency and operation throughput.
    """

    STRUCTURES = {
        "avl": AVLTree,
        "rb": RBTree,
        "treap": Treap,
    }

    def __init__(
        self,
        dataset: Iterable[str],
        queries: Iterable[str],
        include: Optional[List[str]] = None,
        trials: int = 1,
        random_seed: Optional[int] = 12345,
        treap_max_priority: int = 10**6,
    ):
        self.dataset = list(dataset)
        self.queries = list(queries)
        self.n = len(self.dataset)
        self.q = len(self.queries)
        self.trials = trials
        self.random_seed = random_seed
        self.include = include or list(self.STRUCTURES.keys())
        self.treap_max_priority = treap_max_priority

    def _make_instance(self, name: str):
        """
        Instantiate the requested tree structure.

        Parameters
        ----------
        name : str
            Tree name, one of "avl", "rb", "treap".

        Returns
        -------
        instance
            An instance of the requested tree.

        Raises
        ------
        ValueError
            If the tree name is not recognized.
        """
        if name == "avl":
            return AVLTree()
        if name == "rb":
            return RBTree()
        if name == "treap":
            return Treap(max_priority=self.treap_max_priority)
        raise ValueError(name)

    def _run_single_trial(
        self, name: str, dataset: List[str], queries: List[str]
    ) -> Dict[str, Any]:
        """
        Run a single benchmark trial for a specific tree structure.

        Measures:
            - Bulk insert, lookup, and delete times
            - Rotation counts
            - Height and balance metrics after insert and delete
            - Validation correctness

        Parameters
        ----------
        name : str
            Tree name to benchmark
        dataset : List[str]
            Items to insert
        queries : List[str]
            Items to query and delete

        Returns
        -------
        dict[str, Any]
            Raw trial metrics, including times (ns), counts, height, average depth,
            subtree imbalance, rotations, and validation flags.
        """
        gc.collect()
        inst = self._make_instance(name)

        # Ensure metrics exist (rotations counters), default to 0 if missing
        if not hasattr(inst, "rotations_insert"):
            inst.rotations_insert = 0
        if not hasattr(inst, "rotations_delete"):
            inst.rotations_delete = 0

        # Insert benchmark
        t0 = time.perf_counter_ns()
        for v in dataset:
            inst.insert(v)
        insert_ns = time.perf_counter_ns() - t0

        # validate and measure height after inserts
        try:
            inst.validate()
            validate_after_insert = True
        except AssertionError:
            validate_after_insert = False

        height_after_insert = tree_height(inst.root)
        balance_after_insert = compute_balance_metrics(inst.root)

        # Lookup benchmark
        t0 = time.perf_counter_ns()
        true_positives = 0
        false_positives = 0
        dataset_set = set(dataset)
        for q in queries:
            found = inst.contains(q)
            if q in dataset_set:
                true_positives += 1 if found else 0
            else:
                false_positives += 1 if found else 0
        lookup_ns = time.perf_counter_ns() - t0

        # Delete benchmark (delete query set)
        t0 = time.perf_counter_ns()
        deleted_count = 0
        for q in queries:
            if inst.delete(q):
                deleted_count += 1
        delete_ns = time.perf_counter_ns() - t0

        # validate after deletes
        try:
            inst.validate()
            validate_after_delete = True
        except AssertionError:
            validate_after_delete = False

        height_after_delete = tree_height(inst.root)
        balance_after_delete = compute_balance_metrics(inst.root)

        result = {
            "insert_ns": insert_ns,
            "lookup_ns": lookup_ns,
            "delete_ns": delete_ns,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "deleted_count": deleted_count,
            "rotations_insert": getattr(inst, "rotations_insert", None),
            "rotations_delete": getattr(inst, "rotations_delete", None),
            "height_after_insert": height_after_insert,
            "height_after_delete": height_after_delete,
            "validate_after_insert": validate_after_insert,
            "validate_after_delete": validate_after_delete,
            "avg_depth_after_insert": balance_after_insert["avg_depth"],
            "avg_depth_after_delete": balance_after_delete["avg_depth"],
            "max_subtree_imbalance_after_insert": balance_after_insert[
                "max_subtree_imbalance"
            ],
            "max_subtree_imbalance_after_delete": balance_after_delete[
                "max_subtree_imbalance"
            ],
        }
        return result

    def _aggregate(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple trial results into summary metrics (medians).

        Parameters
        ----------
        trials : List[dict[str, Any]]
            Raw trial results returned by `_run_single_trial`.

        Returns
        -------
        dict[str, Any]
            Aggregated metrics with medians for timings, rotations, height,
            average depth, subtree imbalance, and validation flags.
        """
        insert_times = [t["insert_ns"] for t in trials]
        lookup_times = [t["lookup_ns"] for t in trials]
        delete_times = [t["delete_ns"] for t in trials]
        rotations_insert = [
            t["rotations_insert"] for t in trials if t["rotations_insert"] is not None
        ]
        rotations_delete = [
            t["rotations_delete"] for t in trials if t["rotations_delete"] is not None
        ]
        heights_insert = [t["height_after_insert"] for t in trials]
        heights_delete = [t["height_after_delete"] for t in trials]
        validate_insert = all(t["validate_after_insert"] for t in trials)
        validate_delete = all(t["validate_after_delete"] for t in trials)
        avg_depth_insert = [t["avg_depth_after_insert"] for t in trials]
        avg_depth_delete = [t["avg_depth_after_delete"] for t in trials]
        imbalance_insert = [t["max_subtree_imbalance_after_insert"] for t in trials]
        imbalance_delete = [t["max_subtree_imbalance_after_delete"] for t in trials]

        res = {
            "insert_sec_median": statistics.median(insert_times) / 1e9,
            "lookup_sec_median": statistics.median(lookup_times) / 1e9,
            "delete_sec_median": statistics.median(delete_times) / 1e9,
            "insert_ops_per_sec": (
                (self.n / (statistics.median(insert_times) / 1e9))
                if statistics.median(insert_times) > 0
                else float("inf")
            ),
            "lookup_ops_per_sec": (
                (self.q / (statistics.median(lookup_times) / 1e9))
                if statistics.median(lookup_times) > 0
                else float("inf")
            ),
            "delete_ops_per_sec": (
                (self.q / (statistics.median(delete_times) / 1e9))
                if statistics.median(delete_times) > 0
                else float("inf")
            ),
            "rotations_insert_median": (
                int(statistics.median(rotations_insert)) if rotations_insert else None
            ),
            "rotations_delete_median": (
                int(statistics.median(rotations_delete)) if rotations_delete else None
            ),
            "height_after_insert_median": int(statistics.median(heights_insert)),
            "height_after_delete_median": int(statistics.median(heights_delete)),
            "validate_after_insert_all_trials": validate_insert,
            "validate_after_delete_all_trials": validate_delete,
            "avg_depth_after_insert_median": statistics.median(avg_depth_insert),
            "avg_depth_after_delete_median": statistics.median(avg_depth_delete),
            "max_subtree_imbalance_after_insert_median": statistics.median(
                imbalance_insert
            ),
            "max_subtree_imbalance_after_delete_median": statistics.median(
                imbalance_delete
            ),
        }
        return res

    def run(self, workload: str = "random") -> Dict[str, Dict[str, Any]]:
        """
        Run the benchmark for all included trees on a specified workload pattern.

        Workload options
        ----------------
        - "random": random insertion order and queries
        - "ascending": insert dataset in ascending order
        - "descending": insert dataset in descending order
        - "hotspot": queries are skewed, 80% targeting top 10% of dataset

        Parameters
        ----------
        workload : str, default="random"
            Workload pattern for insertion and queries.

        Returns
        -------
        dict[str, dict[str, Any]]
            Aggregated benchmark metrics for each included tree.
        """
        assert workload in ("random", "ascending", "descending", "hotspot")
        results = {}

        for name in self.include:
            trial_results = []
            for t in range(self.trials):
                # Prepare dataset according to workload
                if workload == "random":
                    ds = list(self.dataset)
                elif workload == "ascending":
                    ds = sorted(
                        self.dataset, key=lambda s: int(s) if s.isdigit() else s
                    )
                elif workload == "descending":
                    ds = sorted(
                        self.dataset,
                        key=lambda s: int(s) if s.isdigit() else s,
                        reverse=True,
                    )
                elif workload == "hotspot":
                    # hotspot: first 10% of dataset are "hot" and appear disproportionately in queries
                    ds = list(self.dataset)

                # queries: for hotspot create skewed queries
                if workload == "hotspot":
                    # create queries: 80% draw from first 10% of dataset, 20% uniformly from all
                    hotspot_size = max(1, int(0.1 * len(self.dataset)))
                    hotspot = list(self.dataset)[:hotspot_size]
                    queries = []
                    rng = random.Random(self.random_seed + t)
                    for _ in range(self.q):
                        if rng.random() < 0.8:
                            queries.append(rng.choice(hotspot))
                        else:
                            queries.append(rng.choice(self.dataset))
                else:
                    queries = list(self.queries)

                # run trial
                trial = self._run_single_trial(name, ds, queries)
                trial_results.append(trial)

            results[name] = self._aggregate(trial_results)

        return results
