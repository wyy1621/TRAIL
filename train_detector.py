"""Train the source RSF and LightGBM stages without final evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.config import load_config, resolve_path
from common.performance import PerformanceRecorder
from detector.pipeline import run


def main() -> None:
    """Invoke the relocated source classifier pipeline in training mode."""
    parser = argparse.ArgumentParser(
        description="Train source-equivalent RSF and LightGBM"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    performance_root = resolve_path(
        config, config["Output"].get("performance_root", "output/performance")
    )
    with PerformanceRecorder(
        entrypoint="train_detector",
        output_root=performance_root,
        config_path=config_path,
    ):
        original = sys.argv
        try:
            sys.argv = [
                original[0],
                "--config",
                str(config_path),
                "--mode",
                "train",
                "--model-name",
                "lightgbm",
            ]
            run()
        finally:
            sys.argv = original


if __name__ == "__main__":
    main()
