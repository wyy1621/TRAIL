"""Download source-equivalent RADBench updates and prepare IXP inputs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from common.config import load_config, resolve_path
from data_pipeline import download, ixp


def _run_with_args(function: Callable[[], None], arguments: list[str]) -> None:
    """Call a relocated source CLI while preserving the outer process arguments."""
    original = sys.argv
    try:
        sys.argv = [original[0], *arguments]
        function()
    finally:
        sys.argv = original


def main() -> None:
    """Download event updates, pre-event updates, and event-scoped IXP data."""
    parser = argparse.ArgumentParser(
        description="Download source-equivalent RADBench inputs"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["DataPipeline"]
    event_csv = str(resolve_path(config, paths["event_csv"]))
    raw_updates_root = str(resolve_path(config, paths["raw_updates_root"]))
    ixp_output = str(resolve_path(config, paths["ixp_output"]))
    _run_with_args(
        download.main,
        [
            "--event-csv",
            event_csv,
            "--event-type",
            "leak",
            "--root-dir",
            raw_updates_root,
            "--download-before-data",
        ],
    )
    _run_with_args(
        ixp.main,
        [
            "--event-csv",
            event_csv,
            "--event-type",
            "leak",
            "--output-dir",
            ixp_output,
        ],
    )


if __name__ == "__main__":
    main()
