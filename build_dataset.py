"""Build precise GT and no-sparse/nonzero-mask RADBench LMDB datasets."""

from __future__ import annotations

import argparse
import os

from common import RADBenchEvent, read_RADBench_event_list
from common.config import load_config, resolve_path
from common.performance import PerformanceRecorder


def _configure_source_modules(config: dict) -> tuple[object, object]:
    """Inject repository paths into relocated source modules without changing algorithms."""
    import common.source as common_source
    from data_pipeline import features, ground_truth

    raw_paths = config["DataPipeline"]
    paths = {
        key: str(resolve_path(config, value))
        for key, value in raw_paths.items()
    }
    common_source.RADBENCH_IXP_ROOT = paths["ixp_output"]
    common_source._radbench_ixp_cache.clear()
    event_csv = str(paths["event_csv"])
    raw_updates_root = str(paths["raw_updates_root"])
    ground_truth_root = str(paths["ground_truth_root"])

    ground_truth.events = [
        RADBenchEvent(event)
        for event in read_RADBench_event_list(
            event_list_path=event_csv,
            event_type="leak",
        )
    ]
    ground_truth.RIB_STORAGE_DIR = str(paths["rib_root"])
    ground_truth.UPDATE_STORAGE_DIR = str(paths["pre_event_updates_root"])
    ground_truth.ROUTE_TABLE_DIR = str(paths["route_table_root"])
    ground_truth.EVENT_UPDATE_DIR_BASE = raw_updates_root
    ground_truth.OUTPUT_DIR = ground_truth_root
    ground_truth.WITHDRAW_DIR = os.path.join(ground_truth_root, "Withdraws")
    ground_truth.PROPAGATOR_DIR = os.path.join(ground_truth_root, "Propagator")
    for directory in (
        ground_truth.RIB_STORAGE_DIR,
        ground_truth.UPDATE_STORAGE_DIR,
        ground_truth.ROUTE_TABLE_DIR,
        ground_truth.OUTPUT_DIR,
        ground_truth.WITHDRAW_DIR,
        ground_truth.PROPAGATOR_DIR,
    ):
        os.makedirs(directory, exist_ok=True)

    features.RADBENCH_EVENT_INFO_CSV = event_csv
    features.RADBENCH_INPUT_ROOT = raw_updates_root
    features.RADBENCH_GT_ROOT = ground_truth_root
    features.RADBENCH_OUTPUT_ROOT = str(paths["dataset_root"])
    features.RADBENCH_RANDOM_SEED = int(config["seed"])
    return ground_truth, features


def main() -> None:
    """Run the relocated source GT, feature, split, and LMDB pipeline."""
    parser = argparse.ArgumentParser(
        description="Build source-equivalent RADBench LMDBs"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    performance_root = resolve_path(
        config, config["Output"].get("performance_root", "output/performance")
    )
    with PerformanceRecorder(
        entrypoint="build_dataset",
        output_root=performance_root,
        config_path=args.config,
    ):
        ground_truth, features = _configure_source_modules(config)
        for event in ground_truth.events:
            ground_truth.process_event(event)
        features.main()


if __name__ == "__main__":
    main()
