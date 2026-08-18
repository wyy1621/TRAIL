"""Train TimesNet with the source MSE, optimizer, and scheduler implementation."""

from __future__ import annotations

import argparse
import shutil
from common.config import load_config, resolve_path
from common.performance import PerformanceRecorder
from unsupervised.train_impl import main as train_main



def main() -> None:
    """Train TimesNet through the configured epoch and bind its final checkpoint."""
    parser = argparse.ArgumentParser(description="Train source-equivalent TimesNet")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    performance_root = resolve_path(
        config, config["Output"].get("performance_root", "output/performance")
    )

    with PerformanceRecorder(
        entrypoint="train_reconstructor",
        output_root=performance_root,
        config_path=args.config,
    ):
        final_checkpoint = train_main(args.config)

        target = resolve_path(config, config["UnsuperviseModel"]["checkpoint"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_checkpoint, target)


if __name__ == "__main__":
    main()
