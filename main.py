#!/usr/bin/env python3

from utils.config import load_config
from utils.sample_runs import demo_run, run_scaling_benchmark


def main():
    config = load_config()
    print_cfg = config["print_config"]

    if config["demo_run"]:
        demo_run(print_cfg)
    elif config["scaling_benchmark_run"]:
        scaling_cfg = config["scaling_benchmark"]
        run_scaling_benchmark(print_cfg=print_cfg, **scaling_cfg)


if __name__ == "__main__":
    main()
