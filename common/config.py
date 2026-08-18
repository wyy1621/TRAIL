"""Repository-level YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping and record the directory used for relative paths."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_root"] = str(config_path.parent)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the loaded configuration file."""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_root"]) / path
