import numpy as np
from typing import Dict, Optional
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def plot_scaling_results(
    results: Dict[str, Dict[str, list]],
    dataset_sizes: list,
    query_sizes: Optional[list] = None,
    save_path: Optional[str] = None,
    config: Optional[Dict[str, bool]] = None,
) -> None:
    """
    Plot benchmark scaling results for tree structures.

    If a metric has fewer points than `dataset_sizes`, the remaining points are
    linearly extrapolated and displayed as a dashed line.

    Args:
        results (dict[str, dict[str, list]]): Nested dictionary of results from
            `scaling_benchmark()`. Format: `structure_name -> metric_name -> values`.
        dataset_sizes (list[int]): Sizes of datasets corresponding to each result point.
        query_sizes (list[int], optional): Sizes of queries per dataset, used to plot
            average time per operation (required if `plot_avg_per_op` is True in `config`).
        save_path (str, optional): Path to save the plot as a file. If not provided,
            the plot is displayed interactively.
        config (dict[str, bool], optional): Options to control which plots are shown.
            Supported keys:
                - `plot_avg_per_op`
                - `plot_throughput`
                - `plot_rotations`
                - `plot_heights`
                - `plot_balance`
                - `plot_validation`
            Default for all options is True.

    Notes:
        - Average time per operation is computed as `metric / dataset_size` or
          `metric / query_size` when `plot_avg_per_op` is enabled.
        - Metrics include insertion, lookup, delete times, throughput, rotations,
          height, average depth, subtree imbalance, and validation results.
        - Extrapolated points are shown with dashed lines for visual continuity.
        - Uses logarithmic or linear scaling depending on the metric.
    """
    config = config or {}
    is_plot_throughput = config.get("plot_throughput", True)
    is_plot_rotations = config.get("plot_rotations", True)
    is_plot_heights = config.get("plot_heights", True)
    is_plot_subtree_imbalance = config.get("plot_balance", True)
    is_plot_validation = config.get("plot_validation", True)
    is_plot_avg_per_op = (
        config.get("plot_avg_per_op", True)
        and query_sizes is not None
        and len(query_sizes) == len(dataset_sizes)
    )

    plot_configs = [
        ("insert_sec_median", "Insertion Time (s)", "log", "log"),
        ("lookup_sec_median", "Lookup Time (s)", "log", "log"),
        ("delete_sec_median", "Delete Time (s)", "log", "log"),
    ]

    if is_plot_throughput:
        plot_configs.extend(
            [
                ("insert_ops_per_sec", "Insertion Throughput (ops/s)", "log", "linear"),
                ("lookup_ops_per_sec", "Lookup Throughput (ops/s)", "log", "linear"),
                ("delete_ops_per_sec", "Delete Throughput (ops/s)", "log", "linear"),
            ]
        )
    if is_plot_rotations:
        plot_configs.extend(
            [
                ("rotations_insert_median", "Rotations Insert", "log", "linear"),
                ("rotations_delete_median", "Rotations Delete", "log", "linear"),
            ]
        )
    if is_plot_heights:
        plot_configs.extend(
            [
                ("height_after_insert_median", "Height After Insert", "log", "linear"),
                ("height_after_delete_median", "Height After Delete", "log", "linear"),
            ]
        )

    if is_plot_subtree_imbalance:
        plot_configs.extend(
            [
                (
                    "avg_depth_after_insert_median",
                    "Avg Depth After Insert",
                    "log",
                    "linear",
                ),
                (
                    "avg_depth_after_delete_median",
                    "Avg Depth After Delete",
                    "log",
                    "linear",
                ),
                (
                    "max_subtree_imbalance_after_insert_median",
                    "Max Subtree Imbalance Insert",
                    "log",
                    "linear",
                ),
                (
                    "max_subtree_imbalance_after_delete_median",
                    "Max Subtree Imbalance Delete",
                    "log",
                    "linear",
                ),
            ]
        )
    if is_plot_validation:
        plot_configs.extend(
            [
                (
                    "validate_after_insert_all_trials",
                    "Validation After Insert",
                    "linear",
                    "linear",
                ),
                (
                    "validate_after_delete_all_trials",
                    "Validation After Delete",
                    "linear",
                    "linear",
                ),
            ]
        )

    n_rows = 2
    n_plots = len(plot_configs)
    n_cols = (n_plots + 1) // 2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True
    )
    fig.suptitle("Tree Benchmark Scaling", fontsize=16)
    axes_flat = axes.flatten()

    linestyles = ["-", "-.", ":"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    structure_colors = dict(zip(results.keys(), colors))

    for idx, (metric_name, title, x_scale, y_scale) in enumerate(plot_configs):
        ax = axes_flat[idx]
        for i, (struct, metrics) in enumerate(results.items()):
            values = metrics.get(metric_name, [])
            if not values:
                continue

            # convert bool metrics (e.g., validation) to heat map
            if all(isinstance(v, bool) for v in values):
                all_values = [
                    metrics.get(metric_name, []) for metrics in results.values()
                ]
                structures = list(results.keys())
                dataset_len = min(len(v) for v in all_values)

                # Convert boolean to 0/1 array
                data = np.zeros((len(structures), dataset_len), dtype=int)
                for i, values in enumerate(all_values):
                    data[i, :] = [1 if v else 0 for v in values[:dataset_len]]

                # Draw rectangles
                cmap = {0: "red", 1: "green"}
                for i in range(len(structures)):
                    for j in range(dataset_len):
                        rect = plt.Rectangle((j, i), 1, 1, color=cmap[data[i, j]])
                        ax.add_patch(rect)

                ax.set_xticks(np.arange(dataset_len))
                ax.set_xticklabels(dataset_sizes[:dataset_len])
                ax.set_yticks(np.arange(len(structures)))
                ax.set_yticklabels(structures, rotation=45, ha="right")
                ax.set_xlim(0, dataset_len)
                ax.set_ylim(0, len(structures))
                ax.invert_yaxis()

                legend_elements = [
                    Patch(facecolor="green", label="Valid Tree"),
                    Patch(facecolor="red", label="Invalid Tree"),
                ]
                ax.legend(handles=legend_elements, loc="upper right")
                continue

            if is_plot_avg_per_op:
                if metric_name == "insert_sec_median":
                    values = [v / d for v, d in zip(values, dataset_sizes)]
                elif metric_name in ["lookup_sec_median", "delete_sec_median"]:
                    values = [v / q for v, q in zip(values, query_sizes)]

            actual_len = len(values)
            actual_x = dataset_sizes[:actual_len]
            actual_y = values

            style = linestyles[i % len(linestyles)]

            ax.plot(
                actual_x,
                actual_y,
                marker="o",
                label=struct,
                color=structure_colors[struct],
                linestyle=style,
                linewidth=2,
                markersize=6,
            )

            if actual_len < len(dataset_sizes):
                extrap_x = dataset_sizes[actual_len:]
                slope = (
                    (actual_y[-1] - actual_y[-2]) / (actual_x[-1] - actual_x[-2])
                    if len(actual_x) >= 2
                    else 0
                )
                extrap_y = [actual_y[-1] + slope * (x - actual_x[-1]) for x in extrap_x]
                ax.plot(
                    [actual_x[-1]] + extrap_x,
                    [actual_y[-1]] + extrap_y,
                    linestyle="--",
                    color=structure_colors[struct],
                    alpha=0.7,
                    linewidth=1.5,
                )

        if all(isinstance(v, bool) for v in values):
            ax.set_ylabel("")
        else:
            # ax.plot(
            #     dataset_sizes,
            #     dataset_sizes,
            #     linestyle=(0, (3, 5, 1, 5)),
            #     color="black",
            #     linewidth=1.5,
            #     label="y = n",
            # )

            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            ax.set_ylabel(title)
            ax.legend()

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Dataset Size")

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    if n_plots < 4:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
