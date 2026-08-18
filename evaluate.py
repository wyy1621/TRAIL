"""Run source inference and fixed RSF-LightGBM-ADF evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Callable

import common.source as common_source
from common.config import load_config, resolve_path
from common.performance import PerformanceRecorder
from detector.evaluation import measure_fixed_test_detection, run as run_evaluation
from detector.pipeline import run as run_inference


def _run(function: Callable[[], None], arguments: list[str]) -> None:
    """Call a relocated source CLI while preserving the outer process arguments."""
    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        function()
    finally:
        sys.argv = original


def main() -> None:
    """Generate classifier outputs, then evaluate the fixed paper operating point."""
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed RSF-LightGBM-ADF operating point"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    common_source.RADBENCH_IXP_ROOT = str(
        resolve_path(config, config["DataPipeline"]["ixp_output"])
    )
    common_source._radbench_ixp_cache.clear()

    performance_root = resolve_path(
        config, config["Output"].get("performance_root", "output/performance")
    )
    with PerformanceRecorder(
        entrypoint="evaluate",
        output_root=performance_root,
        config_path=config_path,
    ):
        _run(
            run_inference,
            [
                "--config",
                str(config_path),
                "--mode",
                "infer",
                "--model-name",
                "lightgbm",
            ],
        )
        measure_fixed_test_detection(config_path, classifier="lightgbm")
        _run(
            run_evaluation,
            ["--config", str(config_path), "--clf-name", "lightgbm"],
        )


if __name__ == "__main__":
    main()
