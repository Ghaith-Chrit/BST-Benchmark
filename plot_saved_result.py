#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from utils.config import load_config
from utils.plot import plot_scaling_results

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot scaling results from JSON")
parser.add_argument(
    "json_path",
    type=str,
    help="Path to the results.json file",
)
parser.add_argument(
    "--config_path",
    type=str,
    default="config/main.yaml",
    required=False,
    help="Path to YAML config file (default: config/main.yaml)",
)
parser.add_argument(
    "--save_path",
    type=str,
    help="Path to save the PDF plot. If not provided will simply show the plot without saving it.",
)

args = parser.parse_args()

file_path = Path(args.json_path)
save_path = Path(args.save_path) if args.save_path else None
config_path = Path(args.config_path)

# Load JSON data
with open(file_path, "r") as f:
    data = json.load(f)

results = data["results"]
dataset_sizes = data["dataset_sizes"]

# Load config
config = load_config(config_path)
print_cfg = config["print_config"]
queries_ratio = config["scaling_benchmark"]["queries_ratio"]
fix_queries_ratio = config["scaling_benchmark"]["fix_queries_ratio"]

# Compute query sizes
if fix_queries_ratio:
    middle_idx = len(dataset_sizes) // 2
    num_queries = max(1, int(dataset_sizes[middle_idx] * queries_ratio))
    query_sizes = [num_queries] * len(dataset_sizes)
else:
    query_sizes = [max(1, int(size * queries_ratio)) for size in dataset_sizes]

# Plot results
plot_scaling_results(
    results,
    dataset_sizes,
    query_sizes,
    config=print_cfg,
    save_path=save_path,
)
