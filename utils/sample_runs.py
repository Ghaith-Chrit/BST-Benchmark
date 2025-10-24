import sys
import random
import numpy as np
from utils.benchmark import TreeBenchmark
from utils.plot import plot_scaling_results
from utils.generate_data import generate_strings
from typing import Any, Dict, List, Optional, Tuple
from utils.misc import save_benchmark_results_json, get_file_paths_to_save


def print_benchmark_results(
    results: Dict[str, Dict[str, Any]],
    title: Optional[str] = None,
    query_size: Optional[int] = None,
    dataset_size: Optional[int] = None,
    config: Optional[Dict[str, bool]] = None,
) -> None:
    """
    Print benchmark results in an aligned ASCII table.

    The table dynamically adjusts column widths and includes optional metrics
    such as throughput, false positive rate, failed inserts, and average time per operation.

    Args:
        results (dict[str, dict[str, Any]]): Nested dictionary of results where keys
            are structure names and values are metric dictionaries. Expected metrics include:
                - `insert_sec_median`
                - `lookup_sec_median`
                - `delete_sec_median`
                - `insert_ops_per_sec`
                - `lookup_ops_per_sec`
                - `delete_ops_per_sec`
                - `rotations_insert_median`
                - `rotations_delete_median`
                - `height_after_insert_median`
                - `height_after_delete_median`
                - `validate_after_insert_all_trials`
                - `validate_after_delete_all_trials`
        title (str, optional): Optional title to print centered above the table.
        query_size (int, optional): Number of queries, used to compute average time
            per lookup/delete when `print_avg_per_op` is True. Required for per-op averages.
        dataset_size (int, optional): Dataset size, used to compute average insertion time
            when `print_avg_per_op` is True. Required for per-op averages.
        config (dict[str, bool], optional): Options to toggle metrics. Supported keys:
                - `print_avg_per_op`
                - `print_throughput`
                - `print_rotations`
                - `print_heights`
                - `print_validation`
                - `print_balance`
            Defaults to True for all options if not specified.

    Notes:
        - Time metrics are formatted either as raw seconds or averaged per item, depending on configuration.
        - Throughput metrics are formatted as operations per second.
        - If `results` is empty, prints a notice instead of a table.
    """

    if not results:
        print("No benchmark results to display.")
        return

    config = config or {}
    is_print_avg_per_op = config.get("print_avg_per_op", True)
    is_print_throughput = config.get("print_throughput", True)
    is_print_rotations = config.get("print_rotations", True)
    is_print_heights = config.get("print_heights", True)
    is_print_validation = config.get("print_validation", True)
    is_print_balance = config.get("print_balance", True)

    # Helper to safely extract a metric (value or None)
    def m(metrics: Dict[str, Any], key: str):
        return metrics.get(key, None)

    # If dataset_size/query_size not provided, try to infer from any result row
    sample = next(iter(results.values()))
    dataset_size = dataset_size or sample.get("dataset_size")
    query_size = query_size or sample.get("query_size")

    # formatters
    def fmt_sec(v):
        return "N/A" if v is None else f"{v:.4f}"

    def fmt_avg_insert(v):
        if v is None or not dataset_size:
            return "N/A"
        return f"{(v / dataset_size):.4e}"

    def fmt_avg_lookup(v):
        if v is None or not query_size:
            return "N/A"
        return f"{(v / query_size):.4e}"

    def fmt_ops(v):
        return "N/A" if v is None else f"{v:,.0f}"

    def fmt_num(v):
        return "N/A" if v is None else f"{v}"

    def fmt_frac(v):
        return "N/A" if v is None else f"{v:.4f}"

    # Build rows (list of dicts with stringified fields)
    rows = []
    for name, metrics in results.items():
        row = {"name": name}
        insert_s = m(metrics, "insert_sec_median")
        lookup_s = m(metrics, "lookup_sec_median")
        delete_s = m(metrics, "delete_sec_median")

        row["insert_s"] = fmt_sec(insert_s)
        row["lookup_s"] = fmt_sec(lookup_s)
        row["delete_s"] = fmt_sec(delete_s)

        if is_print_avg_per_op:
            row["insert_per_item"] = fmt_avg_insert(insert_s)
            row["lookup_per_item"] = fmt_avg_lookup(lookup_s)
            row["delete_per_item"] = fmt_avg_lookup(delete_s)

        if is_print_throughput:
            row["insert_ops"] = fmt_ops(m(metrics, "insert_ops_per_sec"))
            row["lookup_ops"] = fmt_ops(m(metrics, "lookup_ops_per_sec"))
            row["delete_ops"] = fmt_ops(m(metrics, "delete_ops_per_sec"))

        if is_print_rotations:
            row["rot_ins"] = fmt_num(m(metrics, "rotations_insert_median"))
            row["rot_del"] = fmt_num(m(metrics, "rotations_delete_median"))

        if is_print_heights:
            row["h_ins"] = fmt_num(m(metrics, "height_after_insert_median"))
            row["h_del"] = fmt_num(m(metrics, "height_after_delete_median"))

        if is_print_balance:
            row["avg_depth_ins"] = fmt_frac(m(metrics, "avg_depth_after_insert_median"))
            row["avg_depth_del"] = fmt_frac(m(metrics, "avg_depth_after_delete_median"))
            row["imbalance_ins"] = fmt_num(
                m(metrics, "max_subtree_imbalance_after_insert_median")
            )
            row["imbalance_del"] = fmt_num(
                m(metrics, "max_subtree_imbalance_after_delete_median")
            )

        if is_print_validation:
            # boolean -> friendly string
            vi = m(metrics, "validate_after_insert_all_trials")
            vd = m(metrics, "validate_after_delete_all_trials")
            row["valid_ins"] = "OK" if vi else ("N/A" if vi is None else "FAIL")
            row["valid_del"] = "OK" if vd else ("N/A" if vd is None else "FAIL")

        rows.append(row)

    # Prepare headers in order
    headers = ["Structure", "Insert (s)", "Lookup (s)", "Delete (s)"]
    if is_print_avg_per_op:
        headers += ["Insert/item", "Lookup/item", "Delete/item"]
    if is_print_throughput:
        headers += ["Insert (ops/s)", "Lookup (ops/s)", "Delete (ops/s)"]
    if is_print_rotations:
        headers += ["Rot-ins", "Rot-del"]
    if is_print_heights:
        headers += ["H-ins", "H-del"]
    if is_print_balance:
        headers += ["AvgDepth(ins)", "AvgDepth(del)", "Imb(ins)", "Imb(del)"]
    if is_print_validation:
        headers += ["Valid(ins)", "Valid(del)"]

    # map header to row key
    mapping = {
        "Structure": lambda r: r["name"],
        "Insert (s)": lambda r: r["insert_s"],
        "Lookup (s)": lambda r: r["lookup_s"],
        "Delete (s)": lambda r: r["delete_s"],
        "Insert/item": lambda r: r.get("insert_per_item", "N/A"),
        "Lookup/item": lambda r: r.get("lookup_per_item", "N/A"),
        "Delete/item": lambda r: r.get("delete_per_item", "N/A"),
        "Insert (ops/s)": lambda r: r.get("insert_ops", "N/A"),
        "Lookup (ops/s)": lambda r: r.get("lookup_ops", "N/A"),
        "Delete (ops/s)": lambda r: r.get("delete_ops", "N/A"),
        "Rot-ins": lambda r: r.get("rot_ins", "N/A"),
        "Rot-del": lambda r: r.get("rot_del", "N/A"),
        "H-ins": lambda r: r.get("h_ins", "N/A"),
        "H-del": lambda r: r.get("h_del", "N/A"),
        "AvgDepth(ins)": lambda r: r.get("avg_depth_ins", "N/A"),
        "AvgDepth(del)": lambda r: r.get("avg_depth_del", "N/A"),
        "Imb(ins)": lambda r: r.get("imbalance_ins", "N/A"),
        "Imb(del)": lambda r: r.get("imbalance_del", "N/A"),
        "Valid(ins)": lambda r: r.get("valid_ins", "N/A"),
        "Valid(del)": lambda r: r.get("valid_del", "N/A"),
    }

    # gather widths
    widths = {}
    for h in headers:
        vals = [mapping[h](r) for r in rows]
        max_val_len = max((len(str(v)) for v in vals), default=0)
        widths[h] = max(len(h), max_val_len)

    # Build header line
    header_line = " | ".join(f"{h:^{widths[h]}}" for h in headers)
    border = "+" + "+".join("-" * (widths[h] + 2) for h in headers) + "+"

    # Print title and table
    print()
    if title:
        print(title.center(len(border)))
    print(border)
    print("| " + header_line + " |")
    print(border.replace("-", "="))

    # Print rows
    for r in rows:
        cells = [
            (
                str(mapping[h](r)).rjust(widths[h])
                if h != "Structure"
                else str(mapping[h](r)).ljust(widths[h])
            )
            for h in headers
        ]
        print("| " + " | ".join(cells) + " |")

    print(border)
    print()


def demo_run(print_cfg: Dict[str, bool]) -> None:
    """
    Run a small demo benchmark for tree-based login checkers and print results.

    This demo generates a modest dataset of pseudo-random strings and a set of queries
    (half from the dataset for true positives, half new for negatives),
    runs the benchmark using default parameters, and prints results using
    `print_benchmark_results`.

    The printed table includes:
      - Insert, Lookup, Delete times (seconds)
      - Per-item average times if `print_avg_per_op` is True
      - Throughput (ops/sec) if `print_throughput` is True
      - Rotation counts for trees if `print_rotations` is True
      - Tree heights after inserts/deletes if `print_heights` is True
      - Validation pass/fail indicators if `print_validation` is True

    Args:
        print_cfg (dict[str, bool]): Configuration dictionary to control which metrics
            are printed. Supported keys:
                - `print_avg_per_op`
                - `print_throughput`
                - `print_rotations`
                - `print_heights`
                - `print_validation`

    Notes:
        - Plotting is skipped because plotting requires results across multiple dataset sizes.
        - This demo uses a small-scale dataset for quick testing.
    """
    print(
        "Demo: generate a small-scale dataset and run the benchmark logic \n"
        "Note: plotting skipped — plotting requires results across multiple dataset sizes.\n"
        "To enable plotting, use a scaling benchmark run and disable demo_run."
    )
    print("=" * 90 + "\n")

    seed = 42
    num_items = 50_000
    num_queries = 25_000

    print(f"Generating {num_items:,} unique elements...")
    dataset, actual_length = generate_strings(num_items, length=12, seed=seed)

    # half queries from dataset (true positives), half new (negatives)
    queries = random.sample(dataset, k=num_queries // 2)
    negatives, _ = generate_strings(
        num_queries - len(queries), length=actual_length, seed=seed + 1
    )
    queries.extend(negatives)
    random.shuffle(queries)

    print("Running benchmark...")
    results = TreeBenchmark(dataset, queries).run()

    # Print results using the updated benchmark printer
    print_benchmark_results(
        results, dataset_size=num_items, query_size=num_queries, config=print_cfg
    )


def scaling_benchmark(
    min_items: int = 1_000,
    max_items: int = 200_000,
    num_steps: int = 8,
    queries_ratio: float = 0.4,
    fix_queries_ratio: bool = True,
    workload: str = "random",
    structures_to_test: Optional[List[str]] = None,
    exclude_structures_above: Optional[List[Tuple[str, int]]] = None,
    num_trials: int = 3,
    seed: int = 42,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run a scaling benchmark for tree-based structures, collecting full performance
    and balance metrics across increasing dataset sizes.

    Metrics collected now include:
      - insert_sec_median / lookup_sec_median / delete_sec_median
      - insert_ops_per_sec / lookup_ops_per_sec / delete_ops_per_sec
      - rotations_insert_median / rotations_delete_median
      - height_after_insert_median / height_after_delete_median
      - avg_depth_after_insert_median / avg_depth_after_delete_median
      - max_subtree_imbalance_after_insert_median / max_subtree_imbalance_after_delete_median
      - validate_after_insert_all_trials / validate_after_delete_all_trials

    Args:
        min_items (int): Starting dataset size.
        max_items (int): Maximum dataset size.
        num_steps (int): Number of dataset sizes to test (logarithmically spaced).
        queries_ratio (float): Ratio of queries to dataset size (e.g., 0.4 means 40% as many queries as items).
        fix_queries_ratio (bool):
            - If True, the number of queries is based on the middle dataset size.
            - If False, queries scale with dataset size.
        workload (str): The workload option to create the query dataset. Check TreeBenchmark for more details.
        structures_to_test (list[str], optional): List of structure names to benchmark.
            If None, all supported structures are tested.
        exclude_structures_above (list(tuple[str, int]), optional): Exclude a structure when
            dataset size exceeds a threshold. Format: [("structure_name", max_allowed_size)].
        num_trials (int): Number of repeated trials per dataset size (median is used).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple:
            - all_results (dict[str, dict[str, list[float]]]): Nested dictionary mapping
              structure_name -> metric_name -> list of values across dataset sizes.
            - dataset_sizes (list[int]): The actual dataset sizes used during benchmarking.

    Notes:
        - Queries consist of at most half true positives (from the dataset) and at least half false
          positives (newly generated data).
        - Dataset sizes are logarithmically spaced and duplicates (due to rounding) are removed.
    """

    # Generate logarithmically spaced dataset sizes
    dataset_sizes = np.logspace(
        np.log10(min_items), np.log10(max_items), num_steps, dtype=int
    )
    # Remove duplicates that may be caused by rounding
    dataset_sizes = sorted(set(dataset_sizes))
    print(
        f"Running scaling benchmark with dataset sizes: {' | '.join(f'{x:,}' for x in dataset_sizes)}"
    )

    # Initialize results structure
    all_results = {}
    metrics_to_track = [
        "insert_sec_median",
        "lookup_sec_median",
        "delete_sec_median",
        "insert_ops_per_sec",
        "lookup_ops_per_sec",
        "delete_ops_per_sec",
        "rotations_insert_median",
        "rotations_delete_median",
        "height_after_insert_median",
        "height_after_delete_median",
        "avg_depth_after_insert_median",
        "avg_depth_after_delete_median",
        "max_subtree_imbalance_after_insert_median",
        "max_subtree_imbalance_after_delete_median",
        "validate_after_insert_all_trials",
        "validate_after_delete_all_trials",
    ]

    for size in dataset_sizes:
        print(f"Benchmarking dataset size: {size:,}")
        dataset, actual_length = generate_strings(size, length=16, seed=seed)

        # Generate queries (half from dataset, half new)
        if fix_queries_ratio:
            middle_idx = len(dataset_sizes) // 2
            num_queries = max(1, int(dataset_sizes[middle_idx] * queries_ratio))
        else:
            num_queries = max(1, int(size * queries_ratio))

        queries = []
        if num_queries > 0:
            # True positives (from dataset)
            num_positives = min(num_queries // 2, len(dataset))
            if num_positives > 0:
                queries.extend(random.sample(dataset, k=num_positives))

            # False positives (new items)
            num_negatives = num_queries - len(queries)
            if num_negatives > 0:
                negatives, _ = generate_strings(
                    num_negatives, length=actual_length, seed=seed + 1
                )
                queries.extend(negatives)

            random.shuffle(queries)

        # Handle structures to exclude based on dataset size
        current_structures_to_test = structures_to_test.copy()
        for structure_name, max_allowed_size in exclude_structures_above or []:
            max_allowed_size = int(float(max_allowed_size))
            if size > max_allowed_size:
                current_structures_to_test.remove(structure_name)

        try:
            benchmark = TreeBenchmark(
                dataset=dataset,
                queries=queries,
                include=current_structures_to_test,
                trials=num_trials,
            )

            results = benchmark.run(workload)

            # Store results
            for structure_name, metrics in results.items():
                if structure_name not in all_results:
                    all_results[structure_name] = {
                        metric: [] for metric in metrics_to_track
                    }

                for metric in metrics_to_track:
                    if metric in metrics:
                        all_results[structure_name][metric].append(metrics[metric])

        except Exception as e:
            sys.exit(f"Error benchmarking size {size}: {e}")

    return all_results, dataset_sizes


def run_scaling_benchmark(
    min_items: int = 1_000,
    max_items: int = 200_000,
    num_steps: int = 8,
    queries_ratio: float = 0.4,
    fix_queries_ratio: bool = True,
    workload: str = "random",
    structures_to_test: Optional[List[str]] = None,
    exclude_structures_above: Optional[List[Tuple[str, int]]] = None,
    num_trials: int = 3,
    seed: int = 42,
    save_result_path: str = None,
    print_cfg: Optional[Dict[str, bool]] = None,
) -> None:
    """
    Run a complete scaling benchmark and optionally save results and plots.

    This function performs a series of benchmarks across increasing dataset sizes,
    prints the results in a formatted table, and optionally saves JSON results and
    plots to the specified path.

    Args:
        min_items (int): Minimum dataset size for the scaling benchmark.
        max_items (int): Maximum dataset size.
        num_steps (int): Number of dataset sizes to evaluate (logarithmically spaced).
        queries_ratio (float): Ratio of queries relative to dataset size.
        fix_queries_ratio (bool):
            - If True, the number of queries is fixed based on the middle dataset size.
            - If False, the number of queries scales with dataset size.
        workload (str): The workload option to create the query dataset. Check TreeBenchmark for more details.
        structures_to_test (list[str], optional): Names of data structures to benchmark.
        bloom_params (tuple[int, float, str, str], optional): Bloom filter parameters
            (capacity, error_rate, first_hash_name, second_hash_name).
        cuckoo_params (tuple[int, int, int, str, str], optional): Cuckoo filter parameters
            (bucket_size, fingerprint_size, max_kicks, first_hash_name, second_hash_name).
        hashtable_params (tuple[int, float, str], optional): Hashtable parameters
            (initial_capacity, max_load_factor, hash_name).
        exclude_structures_above (list(tuple[str, int]), optional): Exclude a structure when
            dataset size exceeds a threshold. Format: [("structure_name", max_allowed_size)].
        num_trials (int): Number of trials per dataset size.
        seed (int): Random seed for reproducibility.
        save_result_path (str, optional): Root path to save JSON results and plots. If None,
            results are not saved.
        print_cfg (dict[str, bool], optional): Configuration dictionary to control which metrics
            are printed and plotted. Keys may include:
                - `print_avg_per_op`
                - `print_throughput`
                - `print_fp`
                - `print_failed_inserts`
            Defaults to True for all options if not specified.

    Notes:
        - If `save_result_path` is provided, results are saved as JSON and PDF.
    """
    print("Scaling Benchmark Run: Performance vs Dataset Size")
    print("=" * 50)

    # This is to allow scientific notations in yaml e.g. 1e9
    min_items = int(float(min_items))
    max_items = int(float(max_items))

    # Run scaling benchmark
    results, dataset_sizes = scaling_benchmark(
        min_items=min_items,
        max_items=max_items,
        num_steps=num_steps,
        queries_ratio=queries_ratio,
        fix_queries_ratio=fix_queries_ratio,
        workload=workload,
        structures_to_test=structures_to_test,
        exclude_structures_above=exclude_structures_above,
        num_trials=num_trials,
        seed=seed,
    )

    query_sizes = []
    if fix_queries_ratio:
        middle_idx = len(dataset_sizes) // 2
        num_queries = max(1, int(dataset_sizes[middle_idx] * queries_ratio))
        query_sizes = [num_queries for _ in range(len(dataset_sizes))]
    else:
        query_sizes = [
            max(1, int(dataset_size * queries_ratio))
            for dataset_size in range(len(dataset_sizes))
        ]

    per_size_results = {}
    for struct_name, metrics in results.items():
        per_size_metrics = {
            k: (v[-1] if isinstance(v, list) else v) for k, v in metrics.items()
        }
        per_size_results[struct_name] = per_size_metrics

    print_benchmark_results(
        per_size_results,
        title=f"Results for the n={dataset_sizes[-1]:,}",
        dataset_size=dataset_sizes[-1],
        query_size=query_sizes[-1],
        config=print_cfg,
    )

    print("Generating plots...")
    if save_result_path is not None:
        file_path_json, file_path_pdf = get_file_paths_to_save(save_result_path)
        plot_scaling_results(
            results, dataset_sizes, query_sizes, file_path_pdf, print_cfg
        )
        save_benchmark_results_json(file_path_json, results, dataset_sizes)
    else:
        plot_scaling_results(results, dataset_sizes, query_sizes, config=print_cfg)
