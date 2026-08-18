"""Train and run no-sparse pointwise residual classifiers.

Purpose:
    1. Load a no-sparse unsupervised reconstruction checkpoint.
    2. Reconstruct RADBench no-sparse LMDB sequences and cache observed-only
       point residuals as LMDB.
    3. Optionally train/apply Residual-RSF to filter normal-pattern triplets.
    4. Train LR, LightGBM, XGBoost, or a small MLP on retained pointwise
       53-dimensional residual features.
    5. Stream residual cache by triplet key and save pointwise anomaly
       probabilities to inference LMDB.

Key parameters:
    - --config: path to pipeline_no_sparse/config.yaml.
    - --mode: cache / train / infer / all.
    - --unsup-checkpoint, --train-lmdb-path, --infer-lmdb-paths,
      --model-name, --input-source, --input-scale can override config values.

Outputs:
    - pipeline_no_sparse/recon_residual/{checkpoint_tag}/{dataset_tag}.lmdb
    - pipeline_no_sparse/clf_checkpoint/{clf_name}_{input_source}_clf_{checkpoint_tag}/
    - pipeline_no_sparse/clf_inference/{clf_name}_{input_source}_clf_{checkpoint_tag}/{dataset_tag}.lmdb
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
import argparse
import copy
import io
import json
import logging
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import dotenv
import joblib
import lmdb
import numpy as np
import pandas as pd
import torch
import yaml
import zstandard as zstd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PIPELINE_NO_SPARSE_DIR = SCRIPT_PATH.parent
ROOT_DIR = PIPELINE_NO_SPARSE_DIR.parent
CONFIG_ROOT = ROOT_DIR
sys.path.insert(0, str(ROOT_DIR))

from common import GPUAutoSelector, beijing, datetime_format  # noqa: E402
from common.performance import get_active_recorder, measure_if_active  # noqa: E402
from unsupervised.model import build_unsupervised_model, get_unsupervised_reconstruction  # noqa: E402
from detector.rsf import ResidualShapeFilter, build_or_load_rsf, collect_keep_keys  # noqa: E402


logging.Formatter.converter = beijing
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
log = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = ROOT_DIR / 'config.yaml'
DEFAULT_MAP_SIZE = 256 * 1024**3
DEFAULT_COMMIT_INTERVAL = 1000

# File-level overrides. CLI > env > these globals > config/default.
CONFIG_PATH_OVERRIDE = str(DEFAULT_CONFIG_PATH)
MODE_OVERRIDE: str | None = None
UNSUP_CHECKPOINT_OVERRIDE = ''
TRAIN_LMDB_PATH_OVERRIDE: str | None = None
INFER_LMDB_PATHS_OVERRIDE: list[str] | None = None
MODEL_NAME_OVERRIDE: str | None = None
INPUT_SOURCE_OVERRIDE: str | None = None
INPUT_SCALE_OVERRIDE: float | None = None
RSF_ENABLED_OVERRIDE: bool | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse namespace with optional overrides. Values left as None do not
        override env, file globals, or config.
    """
    parser = argparse.ArgumentParser(description='Train and infer no-sparse pointwise classifiers.')
    parser.add_argument('--config', default=None, help='Path to pipeline_no_sparse/config.yaml.')
    parser.add_argument('--mode', choices=['cache', 'train', 'infer', 'all'], default=None)
    parser.add_argument('--unsup-checkpoint', default=None)
    parser.add_argument('--train-lmdb-path', default=None)
    parser.add_argument('--infer-lmdb-paths', nargs='*', default=None)
    parser.add_argument('--model-name', choices=['lr', 'lightgbm', 'xgboost', 'mlp', 'all'], default=None)
    parser.add_argument('--input-source', choices=['residual', 'mse'], default=None)
    parser.add_argument('--input-scale', type=float, default=None)
    parser.add_argument(
        '--rsf',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable or disable triplet-level Residual-RSF before pointwise classification.',
    )
    return parser.parse_args()


def env_list(name: str) -> list[str]:
    value = os.getenv(name, '').strip()
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def first_value(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == '':
            continue
        if isinstance(value, list) and not value:
            continue
        return value
    return None


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root.

    Args:
        path: Absolute path or repository-relative path.

    Returns:
        Absolute normalized path.
    """
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (CONFIG_ROOT / path).resolve()


def sanitize_tag(value: str) -> str:
    return ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in value).strip('_')


def dataset_tag(lmdb_path: str | Path) -> str:
    """Return the compact dataset tag used as the output LMDB filename stem.

    Args:
        lmdb_path: Source LMDB path, usually a directory ending with `.lmdb`.

    Returns:
        Sanitized final path component without the `.lmdb` suffix. For example,
        `DATA/RADBench_v6_NoSparse5_NonzeroMask/Supervise/train_ratio_10.lmdb` becomes
        `train_ratio_10`.
    """
    text = Path(str(lmdb_path).rstrip('/')).name
    if text.endswith('.lmdb'):
        text = text[:-5]
    return sanitize_tag(text)


def checkpoint_tag(checkpoint_path: str | Path) -> str:
    return sanitize_tag(Path(checkpoint_path).stem)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f'Config must be a mapping: {config_path}')
    return config


def resolve_unsup_model_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Build the flat model config accepted by build_unsupervised_model().

    Args:
        config: Full pointwise classifier config.

    Returns:
        Flat unsupervised model config including model_name and model args.
    """
    unsup_cfg = config['UnsuperviseModel']
    model_cfg = {'model_name': unsup_cfg['model_name']}
    if 'model_args' in unsup_cfg:
        model_cfg.update(copy.deepcopy(unsup_cfg['model_args']))
    else:
        for key, value in unsup_cfg.items():
            if key not in {'checkpoint', 'training', 'score'}:
                model_cfg[key] = copy.deepcopy(value)
    if 'win_size' not in model_cfg:
        raise KeyError('UnsuperviseModel missing win_size/model_args.win_size')
    if 'step' not in model_cfg:
        model_cfg['step'] = model_cfg['win_size']
    return model_cfg


def build_effective_config(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Merge CLI/env/global overrides into the config.

    Args:
        args: Parsed CLI namespace.

    Returns:
        `(config_path, effective_config, runtime)` where runtime stores the
        selected mode and classifier list settings.
    """
    config_path_value = first_value(
        args.config,
        os.getenv('POINTWISE_CONFIG_PATH', ''),
        CONFIG_PATH_OVERRIDE,
        str(DEFAULT_CONFIG_PATH),
    )
    global CONFIG_ROOT
    config_path = resolve_repo_path(str(config_path_value))
    CONFIG_ROOT = config_path.parent
    config = load_config(config_path)
    effective = copy.deepcopy(config)

    classifier_cfg = effective['ClassifierModel']
    unsup_cfg = effective['UnsuperviseModel']

    mode = first_value(args.mode, os.getenv('POINTWISE_MODE', ''), MODE_OVERRIDE, 'all')
    unsup_checkpoint = first_value(
        args.unsup_checkpoint,
        os.getenv('POINTWISE_UNSUP_CHECKPOINT', ''),
        UNSUP_CHECKPOINT_OVERRIDE,
        unsup_cfg.get('checkpoint', ''),
    )
    train_lmdb_path = first_value(
        args.train_lmdb_path,
        os.getenv('POINTWISE_TRAIN_LMDB', ''),
        TRAIN_LMDB_PATH_OVERRIDE,
        classifier_cfg.get('train_lmdb_path', ''),
    )
    infer_lmdb_paths = first_value(
        args.infer_lmdb_paths,
        env_list('POINTWISE_INFER_LMDBS'),
        INFER_LMDB_PATHS_OVERRIDE,
        classifier_cfg.get('infer_lmdb_paths', []),
    )
    model_name = first_value(
        args.model_name,
        os.getenv('POINTWISE_MODEL_NAME', ''),
        MODEL_NAME_OVERRIDE,
        classifier_cfg.get('model_name', 'lr'),
    )
    input_source = first_value(
        args.input_source,
        os.getenv('POINTWISE_INPUT_SOURCE', ''),
        INPUT_SOURCE_OVERRIDE,
        classifier_cfg.get('input_source', 'residual'),
    )
    input_scale = first_value(
        args.input_scale,
        os.getenv('POINTWISE_INPUT_SCALE', ''),
        INPUT_SCALE_OVERRIDE,
        classifier_cfg.get('input_scale', 100.0),
    )
    rsf_enabled = first_value(
        args.rsf,
        RSF_ENABLED_OVERRIDE,
        effective.get('RSF', {}).get('enabled', False),
    )

    if not unsup_checkpoint:
        raise ValueError('UnsuperviseModel.checkpoint is required')
    if not train_lmdb_path:
        raise ValueError('ClassifierModel.train_lmdb_path is required')
    if input_source not in {'residual', 'mse'}:
        raise ValueError(f'Unsupported input_source: {input_source}')
    valid_model_names = set(POINTWISE_CLASSIFIER_REGISTRY) | {'all'}
    if model_name not in valid_model_names:
        raise ValueError(f'Unsupported model_name: {model_name}')

    unsup_cfg['checkpoint'] = str(unsup_checkpoint)
    classifier_cfg['train_lmdb_path'] = str(train_lmdb_path)
    classifier_cfg['infer_lmdb_paths'] = [str(path) for path in infer_lmdb_paths]
    classifier_cfg['model_name'] = str(model_name)
    classifier_cfg['input_source'] = str(input_source)
    classifier_cfg['input_scale'] = float(input_scale)
    effective.setdefault('RSF', {})['enabled'] = bool(rsf_enabled)

    runtime = {
        'mode': str(mode),
        'config_path': str(config_path),
        'unsup_checkpoint': str(unsup_checkpoint),
        'train_lmdb_path': str(train_lmdb_path),
        'infer_lmdb_paths': [str(path) for path in infer_lmdb_paths],
        'model_name': str(model_name),
        'input_source': str(input_source),
        'input_scale': float(input_scale),
        'rsf_enabled': bool(rsf_enabled),
    }
    return config_path, effective, runtime


def selected_classifiers(model_name: str) -> list[str]:
    if model_name == 'all':
        return list(POINTWISE_CLASSIFIER_REGISTRY.keys())
    if model_name not in POINTWISE_CLASSIFIER_REGISTRY:
        raise ValueError(f'Unsupported classifier: {model_name}')
    return [model_name]


def validate_basic_config(config: dict[str, Any]) -> None:
    if config.get('Dataset') != 'RADBench':
        raise ValueError(f"Only RADBench no-sparse LMDB is supported, got Dataset={config.get('Dataset')}")
    classifier_cfg = config['ClassifierModel']
    win_size = int(classifier_cfg['win_size'])
    step = int(classifier_cfg.get('step', win_size))
    if step != win_size:
        raise ValueError(f'Pointwise cache requires step == win_size, got win_size={win_size}, step={step}')
    model_cfg = resolve_unsup_model_cfg(config)
    if int(model_cfg['win_size']) != win_size or int(model_cfg.get('step', win_size)) != step:
        raise ValueError('ClassifierModel win_size/step must match UnsuperviseModel model_args')
    if int(classifier_cfg['feature_dim']) != int(model_cfg['feature_dim']):
        raise ValueError('ClassifierModel.feature_dim must match UnsuperviseModel feature_dim')


def validate_classifier_dependencies(classifier_names: list[str]) -> None:
    """Fail fast if a selected external classifier dependency is missing."""
    for clf_name in classifier_names:
        if clf_name not in POINTWISE_CLASSIFIER_REGISTRY:
            raise ValueError(f'Unsupported classifier: {clf_name}')
        POINTWISE_CLASSIFIER_REGISTRY[clf_name].validate_dependency()


def read_pickle(txn: lmdb.Transaction, key: str, *, required: bool = True):
    raw = txn.get(key.encode('ascii'))
    if raw is None:
        if required:
            raise KeyError(f'Missing LMDB metadata key: {key}')
        return None
    return pickle.loads(raw)


def put_pickle(txn: lmdb.Transaction, key: str, value: Any) -> None:
    txn.put(key.encode('ascii'), pickle.dumps(value))


def pack_npz_payload(compressor: zstd.ZstdCompressor, **arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return compressor.compress(buffer.getvalue())


def unpack_npz_payload(decompressor: zstd.ZstdDecompressor, raw: bytes) -> dict[str, np.ndarray]:
    payload = decompressor.decompress(raw)
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


class LmdbWriter:
    """Small commit-batching LMDB writer.

    Args:
        path: Target LMDB directory.
        map_size: LMDB map size in bytes.
        commit_interval: Number of put operations per write transaction.
    """

    def __init__(self, path: Path, map_size: int = DEFAULT_MAP_SIZE, commit_interval: int = DEFAULT_COMMIT_INTERVAL):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(str(path), map_size=map_size, subdir=True, lock=True)
        self.txn = self.env.begin(write=True)
        self.commit_interval = int(commit_interval)
        self.pending_puts = 0

    def put(self, key: str, value: bytes) -> None:
        self.txn.put(key.encode('ascii'), value)
        self.pending_puts += 1
        if self.pending_puts >= self.commit_interval:
            self.commit()

    def put_pickle(self, key: str, value: Any) -> None:
        self.put(key, pickle.dumps(value))

    def commit(self) -> None:
        self.txn.commit()
        self.txn = self.env.begin(write=True)
        self.pending_puts = 0

    def close(self) -> None:
        self.txn.commit()
        self.env.sync()
        self.env.close()


def lmdb_has_metadata(path: Path) -> bool:
    if not path.exists():
        return False
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            return txn.get(b'__metadata__') is not None
    finally:
        env.close()


def _file_identity(path: Path) -> dict[str, Any]:
    """Return a content identity for one checkpoint file."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _lmdb_identity(path: Path) -> dict[str, Any]:
    """Return a low-cost source LMDB version identity and metadata digest."""
    import hashlib

    data_file = path / "data.mdb"
    if not data_file.is_file():
        raise FileNotFoundError(f"LMDB data file not found: {data_file}")
    stat = data_file.stat()
    digest = hashlib.sha256()
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            metadata_keys = [
                b"__event_names__", b"__columns__", b"__seq_len__",
                b"__pos_label_count__", b"__neg_label_count__",
                b"__anomaly_triplets__", b"__mask_type__",
                b"__mask_shape__", b"__mask_columns__",
            ]
            event_names_raw = txn.get(b"__event_names__")
            if event_names_raw is None:
                raise KeyError(f"LMDB metadata missing __event_names__: {path}")
            event_names = pickle.loads(event_names_raw)
            metadata_keys.extend(
                f"__{event_name}_timestamp__".encode("ascii")
                for event_name in event_names
            )
            for key in metadata_keys:
                value = txn.get(key)
                if value is not None:
                    digest.update(key)
                    digest.update(value)
    finally:
        env.close()
    return {
        "path": str(path.resolve()),
        "data_size": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "metadata_sha256": digest.hexdigest(),
    }


def residual_cache_identity(
    config: dict[str, Any],
    source_lmdb_path: str | Path,
) -> dict[str, Any]:
    """Build the model/data/schema identity required to reuse a residual cache."""
    classifier_cfg = config["ClassifierModel"]
    checkpoint_path = resolve_repo_path(config["UnsuperviseModel"]["checkpoint"])
    source_path = resolve_repo_path(source_lmdb_path)
    win_size = int(classifier_cfg["win_size"])
    return {
        "checkpoint": _file_identity(checkpoint_path),
        "source_lmdb": _lmdb_identity(source_path),
        "feature_dim": int(classifier_cfg["feature_dim"]),
        "win_size": win_size,
        "step": int(classifier_cfg.get("step", win_size)),
        "residual_definition": "masked(recon_x - input_x)",
    }


def validate_residual_cache_identity(
    cache_path: Path,
    config: dict[str, Any],
    source_lmdb_path: str | Path,
) -> None:
    """Fail fast when an existing cache belongs to another model or dataset."""
    actual = load_cache_metadata(cache_path).get("identity")
    expected = residual_cache_identity(config, source_lmdb_path)
    if actual != expected:
        raise ValueError(
            "Residual cache identity mismatch. Remove the stale cache explicitly "
            f"before rebuilding: {cache_path}"
        )


def collect_source_metadata(lmdb_path: Path, feature_dim: int) -> dict[str, Any]:
    """Read source RADBench LMDB metadata and data keys.

    Args:
        lmdb_path: Source no-sparse LMDB path.
        feature_dim: Expected feature column count excluding label.

    Returns:
        Metadata dict containing event names, timestamps, columns, and data keys.
    """
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, max_readers=64)
    try:
        with env.begin(write=False) as txn:
            event_names = read_pickle(txn, '__event_names__')
            columns = read_pickle(txn, '__columns__')
            seq_len = read_pickle(txn, '__seq_len__')
            anomaly_triplets = read_pickle(txn, '__anomaly_triplets__', required=False)
            mask_type = read_pickle(txn, '__mask_type__', required=False)
            mask_shape = read_pickle(txn, '__mask_shape__', required=False)
            mask_columns = read_pickle(txn, '__mask_columns__', required=False)
            if mask_shape not in {None, 'T', 'T,D'}:
                raise ValueError(
                    f'Pointwise cache supports legacy T mask or feature-level T,D source mask, '
                    f'got mask_type={mask_type}, mask_shape={mask_shape}: {lmdb_path}'
                )
            if mask_columns is not None and len(mask_columns) != feature_dim:
                raise ValueError(f'__mask_columns__ length={len(mask_columns)}, expected {feature_dim}: {lmdb_path}')
            event_timestamps = {
                event_name: read_pickle(txn, f'__{event_name}_timestamp__')
                for event_name in event_names
            }

            if len(columns) == feature_dim + 1:
                feature_columns = columns[:-1]
            elif len(columns) == feature_dim:
                feature_columns = columns
            else:
                raise ValueError(
                    f'Unexpected __columns__ length={len(columns)}, expected {feature_dim} or {feature_dim + 1}'
                )

            keys: list[str] = []
            for key_bytes, _ in txn.cursor():
                if key_bytes.startswith(b'__') and key_bytes.endswith(b'__'):
                    continue
                key = key_bytes.decode('ascii', errors='ignore')
                if key.startswith('mask_'):
                    continue
                if len(key.rsplit('_', 3)) == 4:
                    keys.append(key)

            keys = sorted(keys)
            if keys:
                sample_key = keys[0]
                event_name = sample_key.rsplit('_', 3)[0]
                sample_timestamps = event_timestamps[event_name]
                sample_mask_raw = txn.get(f'mask_{sample_key}'.encode('ascii'))
                if sample_mask_raw is None:
                    raise KeyError(f'Missing source mask key: mask_{sample_key}')
                sample_mask_size = np.frombuffer(zstd.ZstdDecompressor().decompress(sample_mask_raw), dtype=np.uint8).size
                feature_mask_size = len(sample_timestamps) * feature_dim
                time_mask_size = len(sample_timestamps)
                if sample_mask_size == feature_mask_size:
                    inferred_mask_shape = 'T,D'
                    inferred_mask_type = 'feature_nonzero'
                elif sample_mask_size == time_mask_size:
                    inferred_mask_shape = 'T'
                    inferred_mask_type = 'time_observed_legacy'
                else:
                    raise ValueError(
                        f'mask_{sample_key} size={sample_mask_size}, expected {feature_mask_size} '
                        f'for feature mask or {time_mask_size} for legacy time mask: {lmdb_path}'
                    )
                if mask_shape is not None and mask_shape != inferred_mask_shape:
                    raise ValueError(
                        f'__mask_shape__={mask_shape} conflicts with inferred mask shape {inferred_mask_shape}: {lmdb_path}'
                    )
                mask_shape = inferred_mask_shape
                if mask_type is None:
                    mask_type = inferred_mask_type

    finally:
        env.close()

    return {
        'event_names': event_names,
        'columns': feature_columns,
        'seq_len': seq_len,
        'anomaly_triplets': anomaly_triplets,
        'mask_type': mask_type,
        'mask_shape': mask_shape,
        'mask_columns': mask_columns,
        'event_timestamps': event_timestamps,
        'keys': keys,
    }


def parse_radbench_key(key: str) -> tuple[str, tuple[int, int, int]]:
    parts = key.rsplit('_', 3)
    if len(parts) != 4:
        raise ValueError(f'Invalid RADBench key: {key}')
    return parts[0], (int(parts[1]), int(parts[2]), int(parts[3]))


def read_source_entry(
    txn: lmdb.Transaction,
    decompressor: zstd.ZstdDecompressor,
    key: str,
    event_timestamps: dict[str, np.ndarray],
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one source triplet sequence and feature-level nonzero mask.

    Args:
        txn: Read-only source transaction.
        decompressor: zstd decompressor.
        key: Source triplet key.
        event_timestamps: Event to timestamp array mapping.
        feature_dim: Expected feature count excluding label.

    Returns:
        `(features, labels, feature_mask, timestamps)` for the full event timeline.
    """
    event_name, _ = parse_radbench_key(key)
    timestamps = np.asarray(event_timestamps[event_name], dtype=np.int64)

    raw = txn.get(key.encode('ascii'))
    if raw is None:
        raise KeyError(f'Missing source data key: {key}')
    arr = np.frombuffer(decompressor.decompress(raw), dtype=np.float32).reshape(len(timestamps), -1)
    if arr.shape[1] != feature_dim + 1:
        raise ValueError(f'{key} has {arr.shape[1] - 1} features, expected {feature_dim}')

    mask_raw = txn.get(f'mask_{key}'.encode('ascii'))
    if mask_raw is None:
        raise KeyError(f'Missing source mask key: mask_{key}')
    mask_flat = np.frombuffer(decompressor.decompress(mask_raw), dtype=np.uint8)
    feature_mask_size = len(timestamps) * feature_dim
    time_mask_size = len(timestamps)
    if mask_flat.size == feature_mask_size:
        feature_mask = mask_flat.reshape(len(timestamps), feature_dim)
    elif mask_flat.size == time_mask_size:
        time_mask = mask_flat.reshape(len(timestamps), 1)
        feature_mask = np.repeat(time_mask, feature_dim, axis=1)
    else:
        raise ValueError(
            f'mask_{key} size={mask_flat.size}, expected {feature_mask_size} '
            f'for feature mask shape=({len(timestamps)}, {feature_dim}) or {time_mask_size} '
            f'for legacy time mask shape=({len(timestamps)},)'
        )
    return arr[:, :-1], arr[:, -1], feature_mask, timestamps


def build_and_load_unsup_model(config: dict[str, Any]) -> tuple[torch.nn.Module, str, torch.device, Path]:
    """Build the unsupervised model and load its checkpoint.

    Args:
        config: Effective pointwise classifier config.

    Returns:
        `(model, model_name, device, checkpoint_path)`.
    """
    dotenv.load_dotenv(ROOT_DIR / '.env')
    GPUAutoSelector().select()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_cfg = resolve_unsup_model_cfg(config)
    checkpoint_path = resolve_repo_path(config['UnsuperviseModel']['checkpoint'])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Unsupervised checkpoint not found: {checkpoint_path}')

    model = build_unsupervised_model(model_cfg=model_cfg)
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    log.info('Loaded unsupervised checkpoint: %s', checkpoint_path)
    return model, str(model_cfg['model_name']), device, checkpoint_path


def residual_cache_path(config: dict[str, Any], lmdb_path: str | Path) -> Path:
    output_cfg = config.get('Output', {})
    root = resolve_repo_path(output_cfg.get('residual_cache_root', 'pipeline_no_sparse/recon_residual'))
    ckpt_tag = checkpoint_tag(config['UnsuperviseModel']['checkpoint'])
    return root / ckpt_tag / f'{dataset_tag(lmdb_path)}.lmdb'


def classifier_output_dir(config: dict[str, Any], clf_name: str) -> Path:
    output_cfg = config.get('Output', {})
    root = resolve_repo_path(output_cfg.get('classifier_root', 'pipeline_no_sparse/clf_checkpoint'))
    ckpt_tag = checkpoint_tag(config['UnsuperviseModel']['checkpoint'])
    input_source = config['ClassifierModel']['input_source']
    rsf_tag = config.get('RSF', {}).get('runtime_checkpoint_tag')
    if rsf_tag:
        ckpt_tag = f'{ckpt_tag}_rsf_{sanitize_tag(str(rsf_tag))}'
    return root / f'{clf_name}_{input_source}_clf_{ckpt_tag}'


def inference_lmdb_path(config: dict[str, Any], clf_name: str, source_lmdb_path: str | Path) -> Path:
    output_cfg = config.get('Output', {})
    root = resolve_repo_path(output_cfg.get('inference_root', 'pipeline_no_sparse/clf_inference'))
    ckpt_tag = checkpoint_tag(config['UnsuperviseModel']['checkpoint'])
    rsf_tag = config.get('RSF', {}).get('runtime_checkpoint_tag')
    if rsf_tag:
        ckpt_tag = f'{ckpt_tag}_rsf_{sanitize_tag(str(rsf_tag))}'
    input_source = config['ClassifierModel']['input_source']
    return root / f'{clf_name}_{input_source}_clf_{ckpt_tag}' / f'{dataset_tag(source_lmdb_path)}.lmdb'


def infer_windows(
    model: torch.nn.Module,
    model_name: str,
    device: torch.device,
    windows: list[np.ndarray],
    performance_section: str | None = None,
) -> np.ndarray:
    """Run reconstruction on a batch of windows.

    Args:
        model: Loaded unsupervised model.
        model_name: Model name used by reconstruction extractor.
        device: Torch device.
        windows: List of `(L, D)` float32 windows.
        performance_section: Optional online-equivalent inference section name.

    Returns:
        Residual array `(B, L, D)` on CPU, defined as `recon_x - input_x`.
    """
    section = performance_section or "inference.untracked_reconstructor"
    context = measure_if_active(section, cuda=device.type == "cuda") if performance_section else nullcontext()
    with context:
        window_arr = np.stack(windows).astype(np.float32, copy=False)
        input_x = torch.from_numpy(window_arr).to(device)
        with torch.no_grad():
            output = model(input_x, None, None, None)
            recon_x = get_unsupervised_reconstruction(model_name, output)
            residual = (recon_x - input_x).detach().cpu().numpy().astype(np.float32, copy=False)
    return residual


def empty_triplet_payload(feature_dim: int) -> dict[str, np.ndarray]:
    return {
        'residual': np.empty((0, feature_dim), dtype=np.float32),
        'label': np.empty((0,), dtype=np.int8),
        'timestamp': np.empty((0,), dtype=np.int64),
        'time_index': np.empty((0,), dtype=np.int64),
        'window_index': np.empty((0,), dtype=np.int64),
        'window_time_index': np.empty((0,), dtype=np.uint16),
    }


def append_observed_window(
    key_buffer: dict[str, list[np.ndarray]],
    residual_window: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    time_mask: np.ndarray,
    start: int,
    window_index: int,
) -> int:
    """Append observed time points from one reconstructed window into a triplet buffer.

    Args:
        key_buffer: Per-key payload lists.
        residual_window: `(L, D)` masked residual.
        labels: `(L,)` label window.
        timestamps: `(L,)` timestamp window.
        time_mask: `(L,)` uint8/bool time mask derived from feature-level mask.
        start: Start index of the window in the event timeline.
        window_index: Window index within the triplet sequence.

    Returns:
        Number of observed points appended.
    """
    observed = time_mask.astype(bool)
    if not np.any(observed):
        return 0

    local_idx = np.nonzero(observed)[0]
    n = int(local_idx.size)
    key_buffer['residual'].append(residual_window[observed].astype(np.float32, copy=False))
    key_buffer['label'].append(labels[observed].astype(np.int8, copy=False))
    key_buffer['timestamp'].append(timestamps[observed].astype(np.int64, copy=False))
    key_buffer['time_index'].append((start + local_idx).astype(np.int64, copy=False))
    key_buffer['window_index'].append(np.full(n, int(window_index), dtype=np.int64))
    key_buffer['window_time_index'].append(local_idx.astype(np.uint16, copy=False))
    return n


def concatenate_triplet_buffer(key_buffer: dict[str, list[np.ndarray]], feature_dim: int) -> dict[str, np.ndarray]:
    if not key_buffer['residual']:
        return empty_triplet_payload(feature_dim)
    return {name: np.concatenate(parts, axis=0) for name, parts in key_buffer.items()}


def write_residual_cache(
    config: dict[str, Any],
    source_lmdb_path: str | Path,
    model: torch.nn.Module,
    model_name: str,
    device: torch.device,
    checkpoint_path: Path,
) -> Path:
    """Generate observed-only residual cache for one source LMDB.

    Args:
        config: Effective config.
        source_lmdb_path: Source no-sparse LMDB path.
        model: Loaded unsupervised reconstruction model.
        model_name: Unsupervised model name.
        device: Torch device.
        checkpoint_path: Loaded checkpoint path.

    Returns:
        Path to residual cache LMDB.
    """
    classifier_cfg = config['ClassifierModel']
    training_cfg = classifier_cfg.get('training', {})
    feature_dim = int(classifier_cfg['feature_dim'])
    win_size = int(classifier_cfg['win_size'])
    step = int(classifier_cfg.get('step', win_size))
    batch_size = int(training_cfg.get('batch_size', 512))
    source_path = resolve_repo_path(source_lmdb_path)
    cache_path = residual_cache_path(config, source_path)
    test_lmdb_path = resolve_repo_path(config["Eval"]["test_lmdb_path"])
    performance_section = (
        "inference.reconstructor.test"
        if source_path.resolve() == test_lmdb_path.resolve()
        else None
    )

    if lmdb_has_metadata(cache_path):
        validate_residual_cache_identity(cache_path, config, source_path)
        if performance_section is not None:
            recorder = get_active_recorder()
            if recorder is not None:
                recorder.mark_reused(performance_section)
        log.info('Reuse existing residual cache: %s', cache_path)
        return cache_path
    if cache_path.exists():
        raise RuntimeError(f'Residual cache exists but lacks __metadata__: {cache_path}')

    source_meta = collect_source_metadata(source_path, feature_dim)
    log.info('Building residual cache: source=%s output=%s keys=%d', source_path, cache_path, len(source_meta['keys']))

    compressor = zstd.ZstdCompressor(level=3)
    decompressor = zstd.ZstdDecompressor()
    writer = LmdbWriter(cache_path)
    source_env = lmdb.open(str(source_path), readonly=True, lock=False, readahead=False, max_readers=64)

    pending_windows: list[np.ndarray] = []
    pending_records: list[dict[str, Any]] = []
    key_state: dict[str, dict[str, Any]] = {}
    total_samples = 0
    total_pos = 0
    total_neg = 0
    written_triplets = 0

    def flush_pending() -> None:
        nonlocal total_samples, total_pos, total_neg, written_triplets
        if not pending_windows:
            return

        residual_batch = infer_windows(
            model, model_name, device, pending_windows, performance_section
        )
        for residual_window, record in zip(residual_batch, pending_records, strict=True):
            state = key_state[record['key']]
            feature_mask_window = record['feature_mask']
            context = measure_if_active(performance_section) if performance_section else nullcontext()
            with context:
                masked_residual_window = residual_window * feature_mask_window.astype(np.float32, copy=False)
                time_mask = feature_mask_window.any(axis=-1)
                appended = append_observed_window(
                    state['buffer'],
                    masked_residual_window,
                    record['labels'],
                    record['timestamps'],
                    time_mask,
                    record['start'],
                    record['window_index'],
                )
            state['done'] += 1
            state['observed_count'] += appended

            if state['done'] == state['total']:
                payload = concatenate_triplet_buffer(state['buffer'], feature_dim)
                labels = payload['label']
                pos_count = int(labels.sum())
                sample_count = int(labels.shape[0])
                total_samples += sample_count
                total_pos += pos_count
                total_neg += sample_count - pos_count
                writer.put(record['key'], pack_npz_payload(compressor, **payload))
                written_triplets += 1
                del key_state[record['key']]

        pending_windows.clear()
        pending_records.clear()

    try:
        with source_env.begin(write=False) as txn:
            progress = tqdm(source_meta['keys'], desc=f'Cache {dataset_tag(source_path)}')
            for key in progress:
                features, labels, feature_mask, timestamps = read_source_entry(
                    txn,
                    decompressor,
                    key,
                    source_meta['event_timestamps'],
                    feature_dim,
                )
                starts = list(range(0, max(len(features) - win_size + 1, 0), step))
                key_state[key] = {
                    'total': len(starts),
                    'done': 0,
                    'observed_count': 0,
                    'buffer': {
                        'residual': [],
                        'label': [],
                        'timestamp': [],
                        'time_index': [],
                        'window_index': [],
                        'window_time_index': [],
                    },
                }

                if not starts:
                    payload = empty_triplet_payload(feature_dim)
                    writer.put(key, pack_npz_payload(compressor, **payload))
                    del key_state[key]
                    written_triplets += 1
                    continue

                for window_index, start in enumerate(starts):
                    end = start + win_size
                    pending_windows.append(features[start:end])
                    pending_records.append(
                        {
                            'key': key,
                            'start': start,
                            'window_index': window_index,
                            'labels': labels[start:end],
                            'timestamps': timestamps[start:end],
                            'feature_mask': feature_mask[start:end],
                        }
                    )
                    if len(pending_windows) >= batch_size:
                        flush_pending()
                        progress.set_postfix(samples=total_samples, pos=total_pos, neg=total_neg)

            flush_pending()
    finally:
        source_env.close()

    if key_state:
        raise RuntimeError(f'Unflushed triplet states remain: {len(key_state)}')

    metadata = {
        'created_at': datetime_format(fmt='%Y-%m-%d %H:%M:%S'),
        'identity': residual_cache_identity(config, source_path),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_tag': checkpoint_tag(checkpoint_path),
        'source_lmdb_path': str(source_path),
        'dataset_tag': dataset_tag(source_path),
        'residual_definition': 'masked(recon_x - input_x)',
        'source_mask_type': source_meta['mask_type'],
        'source_mask_shape': source_meta['mask_shape'],
        'time_mask_definition': 'any_nonzero_feature',
        'observed_only': True,
        'feature_dim': feature_dim,
        'win_size': win_size,
        'step': step,
        'sample_count': int(total_samples),
        'positive_count': int(total_pos),
        'negative_count': int(total_neg),
        'triplet_count': int(written_triplets),
    }

    writer.put_pickle('__metadata__', metadata)
    writer.put_pickle('__event_names__', source_meta['event_names'])
    writer.put_pickle('__columns__', source_meta['columns'])
    writer.put_pickle('__seq_len__', source_meta['seq_len'])
    writer.put_pickle('__pos_label_count__', int(total_pos))
    writer.put_pickle('__neg_label_count__', int(total_neg))
    if source_meta['anomaly_triplets'] is not None:
        writer.put_pickle('__anomaly_triplets__', source_meta['anomaly_triplets'])
    for event_name, timestamps in source_meta['event_timestamps'].items():
        writer.put_pickle(f'__{event_name}_timestamp__', timestamps)
    writer.put_pickle('__triplet_count__', int(written_triplets))
    writer.put_pickle('__sample_count__', int(total_samples))
    writer.put_pickle('__source_lmdb_path__', str(source_path))
    writer.put_pickle('__checkpoint_path__', str(checkpoint_path))
    writer.put_pickle('__residual_definition__', 'masked(recon_x - input_x)')
    writer.put_pickle('__source_mask_type__', source_meta['mask_type'])
    writer.put_pickle('__source_mask_shape__', source_meta['mask_shape'])
    writer.put_pickle('__time_mask_definition__', 'any_nonzero_feature')
    writer.close()

    if performance_section is not None:
        recorder = get_active_recorder()
        if recorder is not None:
            recorder.add_count(performance_section, "triplets", written_triplets)
            recorder.add_count(performance_section, "point_samples", total_samples)
    log.info(
        'Residual cache finished: %s samples=%d pos=%d neg=%d triplets=%d',
        cache_path,
        total_samples,
        total_pos,
        total_neg,
        written_triplets,
    )
    return cache_path


def ensure_residual_cache(
    config: dict[str, Any],
    source_lmdb_path: str | Path,
    model_bundle: tuple[torch.nn.Module, str, torch.device, Path] | None,
) -> tuple[Path, tuple[torch.nn.Module, str, torch.device, Path] | None]:
    cache_path = residual_cache_path(config, source_lmdb_path)
    if lmdb_has_metadata(cache_path):
        validate_residual_cache_identity(cache_path, config, source_lmdb_path)
        log.info('Residual cache exists: %s', cache_path)
        return cache_path, model_bundle
    if model_bundle is None:
        model_bundle = build_and_load_unsup_model(config)
    model, model_name, device, checkpoint_path = model_bundle
    return write_residual_cache(config, source_lmdb_path, model, model_name, device, checkpoint_path), model_bundle


def cache_data_keys(cache_path: Path) -> list[str]:
    env = lmdb.open(str(cache_path), readonly=True, lock=False, readahead=False)
    try:
        keys = []
        with env.begin(write=False) as txn:
            for key_bytes, _ in txn.cursor():
                if key_bytes.startswith(b'__') and key_bytes.endswith(b'__'):
                    continue
                keys.append(key_bytes.decode('ascii'))
        return sorted(keys)
    finally:
        env.close()


def load_cache_metadata(cache_path: Path) -> dict[str, Any]:
    env = lmdb.open(str(cache_path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            return read_pickle(txn, '__metadata__')
    finally:
        env.close()


def load_cache_feature_names(cache_path: Path, feature_dim: int) -> list[str]:
    """Read original feature names from residual cache metadata.

    Args:
        cache_path: Residual cache LMDB path.
        feature_dim: Expected feature count.

    Returns:
        Original feature names copied from source LMDB `__columns__`.
    """
    env = lmdb.open(str(cache_path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            feature_names = read_pickle(txn, '__columns__')
    finally:
        env.close()

    feature_names = [str(name) for name in feature_names]
    if len(feature_names) != int(feature_dim):
        raise ValueError(f'Feature name count={len(feature_names)}, expected {feature_dim}: {cache_path}')
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f'Feature names must be unique for tree classifiers: {cache_path}')
    return feature_names


def load_training_matrix(
    cache_path: Path,
    feature_dim: int,
    keep_keys: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load all observed residual samples from a training cache.

    Args:
        cache_path: Residual cache LMDB.
        feature_dim: Expected feature dimension.

    Returns:
        `(residual, label, metadata)` where residual is `(N, D)` float32.
    """
    metadata = load_cache_metadata(cache_path)
    residual_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []

    env = lmdb.open(str(cache_path), readonly=True, lock=False, readahead=False, max_readers=64)
    decompressor = zstd.ZstdDecompressor()
    try:
        with env.begin(write=False) as txn:
            for key in tqdm(cache_data_keys(cache_path), desc='Load train residual'):
                if keep_keys is not None and key not in keep_keys:
                    continue
                raw = txn.get(key.encode('ascii'))
                if raw is None:
                    raise KeyError(f'Missing residual cache key: {key}')
                payload = unpack_npz_payload(decompressor, raw)
                x = payload['residual']
                y = payload['label']
                if x.shape[1] != feature_dim:
                    raise ValueError(f'{key} feature_dim={x.shape[1]}, expected {feature_dim}')
                residual_parts.append(x.astype(np.float32, copy=False))
                label_parts.append(y.astype(np.int8, copy=False))
    finally:
        env.close()

    if not residual_parts:
        raise ValueError('No pointwise training samples remain after optional RSF filtering')
    residual = np.concatenate(residual_parts, axis=0)
    label = np.concatenate(label_parts, axis=0)
    metadata = {**metadata, 'sample_count': int(label.size)}
    return residual, label, metadata


def transform_classifier_input(residual: np.ndarray, input_source: str, input_scale: float) -> np.ndarray:
    """Transform residuals into classifier features.

    Args:
        residual: Raw signed residual `(N, D)`.
        input_source: `residual` or `mse`.
        input_scale: Multiplicative scale before log1p.

    Returns:
        Transformed float32 classifier feature matrix `(N, D)`.
    """
    x = residual.astype(np.float32, copy=True)
    if input_source == 'residual':
        sign = np.sign(x)
        np.abs(x, out=x)
        x *= float(input_scale)
        np.log1p(x, out=x)
        x *= sign
    elif input_source == 'mse':
        np.square(x, out=x)
        x *= float(input_scale)
        np.log1p(x, out=x)
    else:
        raise ValueError(f'Unsupported input_source: {input_source}')

    if not np.all(np.isfinite(x)):
        raise ValueError('Classifier input contains nan or inf after transform')
    return x


class PointwiseClassifierMethod(ABC):
    """Base wrapper for one pointwise classifier method.

    Subclasses own dependency checks, config parsing, estimator construction,
    and estimator input format. The training and inference loops only call this
    common interface.
    """

    name = ''
    config_key = ''
    dependency_module = ''
    use_feature_names = False

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @classmethod
    def validate_dependency(cls) -> None:
        """Import the external package required by this method.

        Raises:
            ImportError: If the selected classifier package is unavailable.
        """
        if not cls.dependency_module:
            return
        try:
            __import__(cls.dependency_module)
        except ImportError as exc:
            raise ImportError(f'{cls.dependency_module} is required when model_name includes {cls.name}') from exc

    @abstractmethod
    def resolve_params(self, positive_count: int, negative_count: int) -> dict[str, Any]:
        """Resolve training-dependent hyperparameters.

        Args:
            positive_count: Number of positive training samples.
            negative_count: Number of negative training samples.

        Returns:
            Estimator parameters written to metadata.
        """

    def resolve_scale_pos_weight(self, params: dict[str, Any], positive_count: int, negative_count: int) -> dict[str, Any]:
        """Replace `scale_pos_weight: auto` with neg/pos for tree models."""
        resolved = copy.deepcopy(params)
        if resolved.get('scale_pos_weight') == 'auto':
            if positive_count <= 0:
                raise ValueError('Cannot compute scale_pos_weight because positive_count is zero')
            resolved['scale_pos_weight'] = float(negative_count) / float(positive_count)
        return resolved

    def prepare_features(self, features: np.ndarray, feature_names: list[str]):
        """Prepare transformed `(N, D)` features for the estimator.

        Tree models keep original feature names to avoid fitted-column warnings;
        linear models use the raw ndarray.
        """
        if self.use_feature_names:
            return pd.DataFrame(features, columns=feature_names, copy=False)
        return features

    @abstractmethod
    def build_estimator(self, resolved_params: dict[str, Any]):
        """Build an unfitted estimator from resolved parameters."""

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
        positive_count: int,
        negative_count: int,
    ):
        """Fit the method-specific estimator.

        Args:
            features: Transformed classifier input `(N, D)`.
            labels: Binary labels `(N,)`.
            feature_names: Original source feature names for DataFrame-based models.
            positive_count: Positive sample count.
            negative_count: Negative sample count.

        Returns:
            `(estimator, resolved_params)`.
        """
        resolved_params = self.resolve_params(positive_count, negative_count)
        estimator = self.build_estimator(resolved_params)
        estimator.fit(self.prepare_features(features, feature_names), labels)
        return estimator, resolved_params

    def predict_probability(self, estimator, features: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """Predict class-1 probabilities for transformed pointwise features.

        Args:
            estimator: Fitted estimator loaded from checkpoint.
            features: Transformed classifier input `(N, D)`.
            feature_names: Original source feature names for DataFrame-based models.

        Returns:
            Float32 anomaly probabilities `(N,)`.
        """
        model_input = self.prepare_features(features, feature_names)
        return estimator.predict_proba(model_input)[:, 1].astype(np.float32, copy=False)


class LR(PointwiseClassifierMethod):
    """Logistic Regression with optional StandardScaler."""

    name = 'lr'
    config_key = 'lr'

    def resolve_params(self, positive_count: int, negative_count: int) -> dict[str, Any]:
        params = copy.deepcopy(self.config['ClassifierModel'][self.config_key])
        use_standard_scaler = bool(params.pop('use_standard_scaler', True))
        return {**params, 'use_standard_scaler': use_standard_scaler}

    def build_estimator(self, resolved_params: dict[str, Any]):
        params = copy.deepcopy(resolved_params)
        use_standard_scaler = bool(params.pop('use_standard_scaler'))
        estimator = LogisticRegression(**params)
        if use_standard_scaler:
            return Pipeline([('scale', StandardScaler()), ('logistic', estimator)])
        return estimator


class LightGBM(PointwiseClassifierMethod):
    """LightGBM pointwise classifier using original feature-name columns."""

    name = 'lightgbm'
    config_key = 'lightgbm'
    dependency_module = 'lightgbm'
    use_feature_names = True

    def resolve_params(self, positive_count: int, negative_count: int) -> dict[str, Any]:
        params = copy.deepcopy(self.config['ClassifierModel'][self.config_key])
        return self.resolve_scale_pos_weight(params, positive_count, negative_count)

    def build_estimator(self, resolved_params: dict[str, Any]):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**resolved_params)


class XGBoost(PointwiseClassifierMethod):
    """XGBoost pointwise classifier using original feature-name columns."""

    name = 'xgboost'
    config_key = 'xgboost'
    dependency_module = 'xgboost'
    use_feature_names = True

    def resolve_params(self, positive_count: int, negative_count: int) -> dict[str, Any]:
        params = copy.deepcopy(self.config['ClassifierModel'][self.config_key])
        return self.resolve_scale_pos_weight(params, positive_count, negative_count)

    def build_estimator(self, resolved_params: dict[str, Any]):
        from xgboost import XGBClassifier

        return XGBClassifier(**resolved_params)


class MLP(PointwiseClassifierMethod):
    """Small PyTorch MLP pointwise classifier built on model.Classifier.PlainMLP.

    The residual cache stores independent observed points as `(N, D)` matrices.
    Existing `PlainMLP` expects `(B, L, D)`, so this wrapper treats each point as
    a one-step sequence `(N, 1, D)` and trains a two-logit classifier.
    """

    name = 'mlp'
    config_key = 'mlp'

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._predict_model: torch.nn.Module | None = None
        self._predict_device: torch.device | None = None
        self._predict_bundle_id: int | None = None

    def resolve_params(self, positive_count: int, negative_count: int) -> dict[str, Any]:
        defaults = {
            'hidden_dims': [32],
            'dropout': 0.2,
            'use_standard_scaler': True,
            'learning_rate': 1.0e-3,
            'weight_decay': 1.0e-3,
            'batch_size': 4096,
            'epochs': 100,
            'validation_fraction': 0.2,
            'patience': 10,
            'min_delta': 1.0e-5,
            'class_weight': 'balanced',
            'device': 'auto',
            'random_state': int(self.config.get('seed', 42)),
            'predict_batch_size': 16384,
        }
        params = {**defaults, **copy.deepcopy(self.config['ClassifierModel'].get(self.config_key, {}))}
        params['hidden_dims'] = [int(value) for value in params['hidden_dims']]
        params['dropout'] = float(params['dropout'])
        params['learning_rate'] = float(params['learning_rate'])
        params['weight_decay'] = float(params['weight_decay'])
        params['batch_size'] = int(params['batch_size'])
        params['epochs'] = int(params['epochs'])
        params['validation_fraction'] = float(params['validation_fraction'])
        params['patience'] = int(params['patience'])
        params['min_delta'] = float(params['min_delta'])
        params['use_standard_scaler'] = bool(params['use_standard_scaler'])
        params['random_state'] = int(params['random_state'])
        params['predict_batch_size'] = int(params['predict_batch_size'])
        if params['batch_size'] <= 0 or params['epochs'] <= 0:
            raise ValueError(f'MLP batch_size and epochs must be positive: {params}')
        if not 0.0 <= params['validation_fraction'] < 1.0:
            raise ValueError(f'MLP validation_fraction must be in [0, 1): {params["validation_fraction"]}')

        class_weight = params.get('class_weight', 'balanced')
        if class_weight == 'balanced':
            if positive_count <= 0:
                raise ValueError('Cannot compute MLP balanced class weight because positive_count is zero')
            params['class_weights'] = [1.0, float(negative_count) / float(positive_count)]
        elif class_weight is None or class_weight == 'none':
            params['class_weights'] = [1.0, 1.0]
        elif isinstance(class_weight, (list, tuple)) and len(class_weight) == 2:
            params['class_weights'] = [float(class_weight[0]), float(class_weight[1])]
        else:
            raise ValueError(f'Unsupported MLP class_weight: {class_weight}')
        return params

    def build_estimator(self, resolved_params: dict[str, Any]):
        raise NotImplementedError('MLP uses a custom PyTorch training loop')

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
        positive_count: int,
        negative_count: int,
    ):
        """Fit a small PlainMLP on transformed pointwise residual features.

        Args:
            features: Transformed classifier input `(N, D)`.
            labels: Binary labels `(N,)`.
            feature_names: Original residual feature names.
            positive_count: Number of positive training samples.
            negative_count: Number of negative training samples.

        Returns:
            `(bundle, resolved_params)` where bundle contains the CPU state dict,
            scaler statistics, and training history.
        """
        resolved_params = self.resolve_params(positive_count, negative_count)
        x, feature_mean, feature_std = self._prepare_training_features(features, resolved_params)
        y = labels.astype(np.int64, copy=False)
        train_indices, val_indices = self._split_indices(y, resolved_params)
        device = self._resolve_device(str(resolved_params['device']))
        model = self._build_model(resolved_params, features.shape[1], device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(resolved_params['learning_rate']),
            weight_decay=float(resolved_params['weight_decay']),
        )
        class_weights = torch.tensor(resolved_params['class_weights'], dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        rng = np.random.default_rng(int(resolved_params['random_state']))

        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []
        progress = tqdm(range(1, int(resolved_params['epochs']) + 1), desc='Train mlp', leave=False)
        for epoch in progress:
            train_loss = self._train_one_epoch(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                train_indices=train_indices,
                batch_size=int(resolved_params['batch_size']),
                rng=rng,
                device=device,
            )
            val_loss = self._evaluate_loss(
                model=model,
                loss_fn=loss_fn,
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                indices=val_indices if val_indices.size else train_indices,
                batch_size=int(resolved_params['batch_size']),
                device=device,
            )
            history.append({'epoch': int(epoch), 'train_loss': float(train_loss), 'val_loss': float(val_loss)})
            progress.set_postfix(train_loss=f'{train_loss:.4g}', val_loss=f'{val_loss:.4g}')

            if val_loss < best_loss - float(resolved_params['min_delta']):
                best_loss = float(val_loss)
                best_epoch = int(epoch)
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(resolved_params['patience']):
                    break

        if best_state is None:
            raise RuntimeError('MLP training did not produce a best state')

        bundle = {
            'model_class': 'PlainMLP',
            'state_dict': best_state,
            'resolved_params': resolved_params,
            'feature_dim': int(features.shape[1]),
            'feature_names': list(feature_names),
            'feature_mean': feature_mean.astype(np.float32, copy=False),
            'feature_std': feature_std.astype(np.float32, copy=False),
            'best_epoch': int(best_epoch),
            'best_val_loss': float(best_loss),
            'history': history,
        }
        return bundle, resolved_params

    def predict_probability(self, estimator, features: np.ndarray, feature_names: list[str]) -> np.ndarray:
        """Predict class-1 probabilities with a saved MLP bundle.

        Args:
            estimator: Bundle returned by `fit` and loaded from joblib.
            features: Transformed classifier input `(N, D)`.
            feature_names: Original source feature names.

        Returns:
            Float32 anomaly probabilities `(N,)`.
        """
        if not isinstance(estimator, dict) or estimator.get('model_class') != 'PlainMLP':
            raise TypeError('MLP estimator must be a PlainMLP bundle dict')
        if list(feature_names) != list(estimator['feature_names']):
            raise ValueError('MLP feature names do not match the training bundle')
        params = estimator['resolved_params']
        x = features.astype(np.float32, copy=False)
        if bool(params['use_standard_scaler']):
            x = (x - estimator['feature_mean']) / estimator['feature_std']
        x = x.astype(np.float32, copy=False)

        device = self._resolve_device(str(params['device']))
        model = self._get_predict_model(estimator, device)
        x_tensor = torch.from_numpy(x)
        batch_size = int(params['predict_batch_size'])
        probs: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, x_tensor.shape[0], batch_size):
                batch = x_tensor[start : start + batch_size].to(device).unsqueeze(1)
                logits = model(batch).squeeze(1)
                prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32, copy=False)
                probs.append(prob)
        if not probs:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(probs, axis=0)

    def _prepare_training_features(
        self,
        features: np.ndarray,
        resolved_params: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize training features when configured.

        Args:
            features: Transformed classifier input `(N, D)`.
            resolved_params: MLP training parameters.

        Returns:
            `(x, feature_mean, feature_std)` as float32 arrays.
        """
        x = features.astype(np.float32, copy=True)
        if bool(resolved_params['use_standard_scaler']):
            feature_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
            feature_std = x.std(axis=0, dtype=np.float64).astype(np.float32)
            feature_std[feature_std < 1.0e-6] = 1.0
            x -= feature_mean
            x /= feature_std
        else:
            feature_mean = np.zeros((features.shape[1],), dtype=np.float32)
            feature_std = np.ones((features.shape[1],), dtype=np.float32)
        if not np.all(np.isfinite(x)):
            raise ValueError('MLP input contains nan or inf after optional standardization')
        return x, feature_mean, feature_std

    def _split_indices(self, labels: np.ndarray, resolved_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Create a deterministic stratified train/validation split.

        Args:
            labels: Binary label vector `(N,)`.
            resolved_params: MLP parameters containing validation_fraction and seed.

        Returns:
            `(train_indices, val_indices)`.
        """
        validation_fraction = float(resolved_params['validation_fraction'])
        all_indices = np.arange(labels.shape[0], dtype=np.int64)
        if validation_fraction <= 0.0:
            return all_indices, np.empty((0,), dtype=np.int64)

        rng = np.random.default_rng(int(resolved_params['random_state']))
        val_parts: list[np.ndarray] = []
        train_parts: list[np.ndarray] = []
        for label_value in (0, 1):
            label_indices = np.flatnonzero(labels == label_value).astype(np.int64, copy=False)
            rng.shuffle(label_indices)
            val_count = int(round(label_indices.shape[0] * validation_fraction))
            if label_indices.shape[0] > 1:
                val_count = min(max(1, val_count), label_indices.shape[0] - 1)
            else:
                val_count = 0
            val_parts.append(label_indices[:val_count])
            train_parts.append(label_indices[val_count:])
        train_indices = np.concatenate(train_parts)
        val_indices = np.concatenate(val_parts)
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        return train_indices, val_indices

    def _resolve_device(self, raw_device: str) -> torch.device:
        """Resolve `auto`, `cpu`, or `cuda` into a torch device."""
        if raw_device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(raw_device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('MLP requested CUDA but torch.cuda.is_available() is false')
        return device

    def _build_model(self, resolved_params: dict[str, Any], feature_dim: int, device: torch.device) -> torch.nn.Module:
        """Build the existing PlainMLP with small pointwise hidden dimensions."""
        from model.Classifier.PlainMLP import PlainMLP

        model = PlainMLP(
            feature_dim=int(feature_dim),
            num_class=2,
            hidden_dims=tuple(int(value) for value in resolved_params['hidden_dims']),
            dropout=float(resolved_params['dropout']),
        )
        return model.to(device)

    def _train_one_epoch(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        train_indices: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> float:
        """Train one MLP epoch and return mean weighted CE loss."""
        model.train()
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        total_loss = 0.0
        total_count = 0
        for start in range(0, shuffled.shape[0], batch_size):
            batch_indices = torch.as_tensor(shuffled[start : start + batch_size], dtype=torch.long)
            batch_x = x_tensor[batch_indices].to(device).unsqueeze(1)
            batch_y = y_tensor[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x).squeeze(1)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(batch_y.shape[0])
            total_count += int(batch_y.shape[0])
        if total_count <= 0:
            raise ValueError('MLP train split is empty')
        return total_loss / float(total_count)

    def _evaluate_loss(
        self,
        *,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        indices: np.ndarray,
        batch_size: int,
        device: torch.device,
    ) -> float:
        """Evaluate mean weighted CE loss for selected indices."""
        if indices.size == 0:
            raise ValueError('MLP evaluation indices are empty')
        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for start in range(0, indices.shape[0], batch_size):
                batch_indices = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long)
                batch_x = x_tensor[batch_indices].to(device).unsqueeze(1)
                batch_y = y_tensor[batch_indices].to(device)
                logits = model(batch_x).squeeze(1)
                loss = loss_fn(logits, batch_y)
                total_loss += float(loss.detach().cpu()) * int(batch_y.shape[0])
                total_count += int(batch_y.shape[0])
        return total_loss / float(total_count)

    def _get_predict_model(self, estimator: dict[str, Any], device: torch.device) -> torch.nn.Module:
        """Lazily rebuild and cache the PlainMLP model for inference."""
        bundle_id = id(estimator)
        if self._predict_model is not None and self._predict_bundle_id == bundle_id and self._predict_device == device:
            return self._predict_model
        model = self._build_model(estimator['resolved_params'], int(estimator['feature_dim']), device)
        model.load_state_dict(estimator['state_dict'])
        model.eval()
        self._predict_model = model
        self._predict_device = device
        self._predict_bundle_id = bundle_id
        return model


POINTWISE_CLASSIFIER_REGISTRY: dict[str, type[PointwiseClassifierMethod]] = {
    LR.name: LR,
    LightGBM.name: LightGBM,
    XGBoost.name: XGBoost,
    MLP.name: MLP,
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def write_summary(
    path: Path,
    *,
    clf_name: str,
    config: dict[str, Any],
    resolved_params: dict[str, Any],
    train_cache_path: Path,
    train_metadata: dict[str, Any],
    classifier_path: Path,
    inference_paths: list[Path],
    elapsed_seconds: float,
) -> None:
    lines = [
        'No-Sparse Pointwise Classifier Summary',
        '',
        f'classifier: {clf_name}',
        f'input_source: {config["ClassifierModel"]["input_source"]}',
        f'input_scale: {config["ClassifierModel"]["input_scale"]}',
        f'unsup_checkpoint: {resolve_repo_path(config["UnsuperviseModel"]["checkpoint"])}',
        f'train_lmdb_path: {resolve_repo_path(config["ClassifierModel"]["train_lmdb_path"])}',
        f'train_cache_path: {train_cache_path}',
        f'rsf_enabled: {bool(config.get("RSF", {}).get("runtime_checkpoint_path"))}',
        f'rsf_checkpoint_path: {config.get("RSF", {}).get("runtime_checkpoint_path")}',
        f'classifier_path: {classifier_path}',
        f'train_sample_count: {train_metadata["sample_count"]}',
        f'train_positive_count: {train_metadata["positive_count"]}',
        f'train_negative_count: {train_metadata["negative_count"]}',
        f'classifier_params: {json.dumps(resolved_params, ensure_ascii=False)}',
        f'elapsed_seconds: {elapsed_seconds:.2f}',
        '',
        'inference_outputs:',
    ]
    if inference_paths:
        lines.extend(f'- {path}' for path in inference_paths)
    else:
        lines.append('- <not generated in this mode>')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def classifier_identity(
    config: dict[str, Any],
    clf_name: str,
    train_cache_path: Path,
    feature_names: list[str],
    positive_count: int,
    negative_count: int,
) -> dict[str, Any]:
    """Build the exact training identity required to reuse a classifier."""
    method = POINTWISE_CLASSIFIER_REGISTRY[clf_name](config)
    rsf_path = config.get("RSF", {}).get("runtime_checkpoint_path")
    return {
        "clf_name": clf_name,
        "resolved_params": method.resolve_params(positive_count, negative_count),
        "input_source": config["ClassifierModel"]["input_source"],
        "input_scale": float(config["ClassifierModel"]["input_scale"]),
        "feature_names": feature_names,
        "train_residual_identity": load_cache_metadata(train_cache_path).get("identity"),
        "rsf_metadata": (
            json.loads((Path(rsf_path) / "metadata.json").read_text(encoding="utf-8"))
            if rsf_path else None
        ),
    }


def validate_classifier_identity(
    config: dict[str, Any],
    clf_name: str,
    output_dir: Path,
    train_cache_path: Path,
    feature_names: list[str],
    positive_count: int,
    negative_count: int,
) -> None:
    """Fail fast when a saved classifier belongs to another configuration."""
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Classifier metadata not found: {metadata_path}")
    actual = json.loads(metadata_path.read_text(encoding="utf-8")).get("identity")
    expected = classifier_identity(
        config, clf_name, train_cache_path, feature_names, positive_count, negative_count
    )
    if actual != expected:
        raise ValueError(
            "Classifier identity mismatch. Remove the stale classifier explicitly "
            f"before retraining: {output_dir}"
        )


def train_classifiers(
    config_path: Path,
    config: dict[str, Any],
    train_cache_path: Path,
    classifier_names: list[str],
    rsf_keep_keys: set[str] | None = None,
) -> dict[str, Path]:
    """Train selected classifiers and save them.

    Args:
        config_path: Effective config file path.
        config: Effective config.
        train_cache_path: Observed-only residual cache for training.
        classifier_names: Classifier names to train.

    Returns:
        Mapping from classifier name to checkpoint directory.
    """
    validate_classifier_dependencies(classifier_names)
    feature_dim = int(config['ClassifierModel']['feature_dim'])
    feature_names = load_cache_feature_names(train_cache_path, feature_dim)
    residual, label, train_metadata = load_training_matrix(train_cache_path, feature_dim, rsf_keep_keys)
    x = transform_classifier_input(
        residual,
        config['ClassifierModel']['input_source'],
        float(config['ClassifierModel']['input_scale']),
    )
    y = label.astype(np.int64, copy=False)
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError(f'Training labels must contain both classes, got pos={positive_count}, neg={negative_count}')

    output_dirs: dict[str, Path] = {}
    start_time = time.perf_counter()
    for clf_name in classifier_names:
        output_dir = classifier_output_dir(config, clf_name)
        classifier_path = output_dir / 'classifier.joblib'
        if output_dir.exists() and any(output_dir.iterdir()):
            if classifier_path.exists():
                validate_classifier_identity(
                    config, clf_name, output_dir, train_cache_path,
                    feature_names, positive_count, negative_count,
                )
                log.warning('Classifier output directory already exists; skip training and reuse checkpoint: %s', output_dir)
                recorder = get_active_recorder()
                if recorder is not None:
                    recorder.mark_reused("training.classifier")
                output_dirs[clf_name] = output_dir
                continue
            raise FileExistsError(
                f'Classifier output directory already exists and is not empty, '
                f'but classifier checkpoint is missing: {classifier_path}'
            )

        log.info('Training classifier %s on %d samples', clf_name, len(y))
        method = POINTWISE_CLASSIFIER_REGISTRY[clf_name](config)
        fit_start = time.perf_counter()
        with measure_if_active("training.classifier"):
            classifier, resolved_params = method.fit(
                x, y, feature_names, positive_count, negative_count
            )
        log.info('Finished training classifier %s in %.2fs', clf_name, time.perf_counter() - fit_start)

        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, classifier_path)

        metadata = {
            'created_at': datetime_format(fmt='%Y-%m-%d %H:%M:%S'),
            'identity': classifier_identity(
                config, clf_name, train_cache_path, feature_names,
                positive_count, negative_count,
            ),
            'clf_name': clf_name,
            'input_source': config['ClassifierModel']['input_source'],
            'input_scale': float(config['ClassifierModel']['input_scale']),
            'unsup_checkpoint': str(resolve_repo_path(config['UnsuperviseModel']['checkpoint'])),
            'train_lmdb_path': str(resolve_repo_path(config['ClassifierModel']['train_lmdb_path'])),
            'train_cache_path': str(train_cache_path),
            'rsf_enabled': bool(config.get('RSF', {}).get('runtime_checkpoint_path')),
            'rsf_checkpoint_path': config.get('RSF', {}).get('runtime_checkpoint_path'),
            'infer_lmdb_paths': [str(resolve_repo_path(path)) for path in config['ClassifierModel'].get('infer_lmdb_paths', [])],
            'classifier_params': resolved_params,
            'feature_names': feature_names,
            'train_sample_count': int(len(y)),
            'train_positive_count': positive_count,
            'train_negative_count': negative_count,
        }
        write_json(output_dir / 'metadata.json', metadata)
        shutil.copyfile(config_path, output_dir / 'config.yaml')
        shutil.copyfile(SCRIPT_PATH, output_dir / SCRIPT_PATH.name)
        write_summary(
            output_dir / 'summary.txt',
            clf_name=clf_name,
            config=config,
            resolved_params=resolved_params,
            train_cache_path=train_cache_path,
            train_metadata={
                'sample_count': int(len(y)),
                'positive_count': positive_count,
                'negative_count': negative_count,
            },
            classifier_path=classifier_path,
            inference_paths=[
                inference_lmdb_path(config, clf_name, lmdb_path)
                for lmdb_path in config['ClassifierModel'].get('infer_lmdb_paths', [])
            ],
            elapsed_seconds=time.perf_counter() - start_time,
        )
        output_dirs[clf_name] = output_dir
        log.info('Classifier saved: %s', classifier_path)

    return output_dirs


def load_classifier_checkpoint(config: dict[str, Any], clf_name: str):
    output_dir = classifier_output_dir(config, clf_name)
    classifier_path = output_dir / 'classifier.joblib'
    if not classifier_path.exists():
        raise FileNotFoundError(f'Classifier checkpoint not found: {classifier_path}')
    metadata_path = output_dir / 'metadata.json'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Classifier metadata not found: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    train_cache_path = residual_cache_path(config, config['ClassifierModel']['train_lmdb_path'])
    validate_classifier_identity(
        config, clf_name, output_dir, train_cache_path,
        list(metadata['feature_names']),
        int(metadata['train_positive_count']),
        int(metadata['train_negative_count']),
    )
    return joblib.load(classifier_path), output_dir


def copy_inference_metadata(
    residual_cache_path: Path,
    writer: LmdbWriter,
    metadata: dict[str, Any],
) -> None:
    env = lmdb.open(str(residual_cache_path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            for key in ['__event_names__', '__pos_label_count__', '__neg_label_count__', '__anomaly_triplets__']:
                raw = txn.get(key.encode('ascii'))
                if raw is not None:
                    writer.put(key, raw)
            event_names = read_pickle(txn, '__event_names__')
            for event_name in event_names:
                raw = txn.get(f'__{event_name}_timestamp__'.encode('ascii'))
                if raw is not None:
                    writer.put(f'__{event_name}_timestamp__', raw)
    finally:
        env.close()

    writer.put_pickle('__metadata__', metadata)
    writer.put_pickle('__triplet_count__', int(metadata['triplet_count']))
    writer.put_pickle('__sample_count__', int(metadata['sample_count']))
    writer.put_pickle('__source_lmdb_path__', metadata['source_lmdb_path'])
    writer.put_pickle('__residual_cache_path__', metadata['residual_cache_path'])
    writer.put_pickle('__classifier_checkpoint_path__', metadata['classifier_checkpoint_path'])
    writer.put_pickle('__prediction_definition__', metadata['prediction_definition'])


def inference_identity(
    config: dict[str, Any],
    classifier_dir: Path,
    residual_cache_path: Path,
    source_lmdb_path: str | Path,
) -> dict[str, Any]:
    """Build the model/cache/data identity required to reuse inference output."""
    return {
        "classifier": _file_identity(classifier_dir / "classifier.joblib"),
        "classifier_identity": json.loads(
            (classifier_dir / "metadata.json").read_text(encoding="utf-8")
        ).get("identity"),
        "residual_identity": load_cache_metadata(residual_cache_path).get("identity"),
        "source_lmdb": _lmdb_identity(resolve_repo_path(source_lmdb_path)),
        "rsf_checkpoint_path": config.get("RSF", {}).get("runtime_checkpoint_path"),
    }


def run_inference(
    config: dict[str, Any],
    clf_name: str,
    classifier,
    classifier_dir: Path,
    residual_cache_path: Path,
    source_lmdb_path: str | Path,
    rsf_keep_keys: set[str] | None = None,
) -> Path:
    """Stream residual cache through a classifier and write probability LMDB.

    Args:
        config: Effective config.
        clf_name: Classifier name.
        classifier: Loaded fitted classifier.
        classifier_dir: Classifier checkpoint directory.
        residual_cache_path: Residual cache LMDB.
        source_lmdb_path: Original source LMDB path.

    Returns:
        Inference LMDB path.
    """
    output_path = inference_lmdb_path(config, clf_name, source_lmdb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = resolve_repo_path(source_lmdb_path)
    test_path = resolve_repo_path(config["Eval"]["test_lmdb_path"])
    performance_section = (
        "inference.classifier.test"
        if source_path.resolve() == test_path.resolve()
        else None
    )
    if lmdb_has_metadata(output_path):
        actual = load_cache_metadata(output_path).get('identity')
        expected = inference_identity(
            config, classifier_dir, residual_cache_path, source_lmdb_path
        )
        if actual != expected:
            raise ValueError(
                'Inference identity mismatch. Remove the stale output explicitly '
                f'before rerunning: {output_path}'
            )
        if performance_section is not None:
            recorder = get_active_recorder()
            if recorder is not None:
                recorder.mark_reused(performance_section)
        log.warning('Inference LMDB already exists; skip inference and reuse cache: %s', output_path)
        return output_path
    if output_path.exists():
        raise RuntimeError(f'Inference LMDB exists but lacks __metadata__: {output_path}')

    input_source = config['ClassifierModel']['input_source']
    input_scale = float(config['ClassifierModel']['input_scale'])
    feature_dim = int(config['ClassifierModel']['feature_dim'])
    feature_names = load_cache_feature_names(residual_cache_path, feature_dim)
    method = POINTWISE_CLASSIFIER_REGISTRY[clf_name](config)

    keys = cache_data_keys(residual_cache_path)
    decompressor = zstd.ZstdDecompressor()
    compressor = zstd.ZstdCompressor(level=3)
    cache_env = lmdb.open(str(residual_cache_path), readonly=True, lock=False, readahead=False, max_readers=64)
    writer = LmdbWriter(output_path)

    sample_count = 0
    positive_count = 0
    negative_count = 0
    try:
        with cache_env.begin(write=False) as txn:
            for key in tqdm(keys, desc=f'Infer {clf_name} {dataset_tag(source_lmdb_path)}'):
                raw = txn.get(key.encode('ascii'))
                if raw is None:
                    raise KeyError(f'Missing residual cache key: {key}')
                payload = unpack_npz_payload(decompressor, raw)
                residual = payload['residual']
                labels = payload['label']
                if rsf_keep_keys is not None and key not in rsf_keep_keys:
                    residual = residual[:0]
                    labels = labels[:0]
                if residual.shape[0] == 0:
                    prob = np.empty((0,), dtype=np.float32)
                else:
                    context = measure_if_active(performance_section) if performance_section else nullcontext()
                    with context:
                        features = transform_classifier_input(residual, input_source, input_scale)
                        prob = method.predict_probability(classifier, features, feature_names)

                sample_count += int(labels.shape[0])
                pos = int(labels.sum())
                positive_count += pos
                negative_count += int(labels.shape[0]) - pos
                writer.put(
                    key,
                    pack_npz_payload(
                        compressor,
                        prob=prob,
                        label=labels.astype(np.int8, copy=False),
                        timestamp=payload['timestamp'][:labels.shape[0]].astype(np.int64, copy=False),
                        time_index=payload['time_index'][:labels.shape[0]].astype(np.int64, copy=False),
                        window_index=payload['window_index'][:labels.shape[0]].astype(np.int64, copy=False),
                        window_time_index=payload['window_time_index'][:labels.shape[0]].astype(np.uint16, copy=False),
                    ),
                )
    finally:
        cache_env.close()

    metadata = {
        'created_at': datetime_format(fmt='%Y-%m-%d %H:%M:%S'),
        'identity': inference_identity(
            config, classifier_dir, residual_cache_path, source_lmdb_path
        ),
        'clf_name': clf_name,
        'source_lmdb_path': str(resolve_repo_path(source_lmdb_path)),
        'dataset_tag': dataset_tag(source_lmdb_path),
        'residual_cache_path': str(residual_cache_path),
        'classifier_checkpoint_path': str(classifier_dir / 'classifier.joblib'),
        'input_source': input_source,
        'input_scale': input_scale,
        'sample_count': int(sample_count),
        'positive_count': int(positive_count),
        'negative_count': int(negative_count),
        'triplet_count': int(len(keys)),
        'prediction_definition': 'probability of class 1',
        'rsf_enabled': rsf_keep_keys is not None,
        'rsf_checkpoint_path': config.get('RSF', {}).get('runtime_checkpoint_path'),
    }
    copy_inference_metadata(residual_cache_path, writer, metadata)
    writer.close()
    if performance_section is not None:
        recorder = get_active_recorder()
        if recorder is not None:
            recorder.add_count(performance_section, "triplets", len(keys))
            recorder.add_count(performance_section, "point_samples", sample_count)
    log.info('Inference LMDB saved: %s samples=%d', output_path, sample_count)
    return output_path


def prepare_rsf(
    config: dict[str, Any],
    runtime: dict[str, Any],
    train_cache_path: Path,
    model_bundle: tuple[torch.nn.Module, str, torch.device, Path] | None,
) -> tuple[ResidualShapeFilter, tuple[torch.nn.Module, str, torch.device, Path] | None]:
    """Build or load RSF and attach its identity tag to downstream outputs.

    Args:
        config: Effective pipeline configuration.
        runtime: Resolved CLI/config runtime values.
        train_cache_path: Residual cache for RSF and pointwise training data.
        model_bundle: Lazily loaded unsupervised model bundle.

    Returns:
        Fitted RSF and the possibly initialized unsupervised model bundle.
    """
    rsf_cfg = config['RSF']
    val_lmdb_path = str(rsf_cfg['val_lmdb_path'])
    val_cache_path, model_bundle = ensure_residual_cache(config, val_lmdb_path, model_bundle)
    rsf, checkpoint_dir = build_or_load_rsf(
        unsup_checkpoint=resolve_repo_path(config['UnsuperviseModel']['checkpoint']),
        train_cache_path=train_cache_path,
        train_lmdb_path=resolve_repo_path(runtime['train_lmdb_path']),
        val_cache_path=val_cache_path,
        val_lmdb_path=resolve_repo_path(val_lmdb_path),
        checkpoint_root=resolve_repo_path(
            rsf_cfg.get('checkpoint_root', 'pipeline_no_sparse/rsf/checkpoints')
        ),
        config=rsf_cfg,
    )
    rsf_cfg['runtime_checkpoint_tag'] = checkpoint_dir.name.rsplit('_', 1)[-1]
    rsf_cfg['runtime_checkpoint_path'] = str(checkpoint_dir)
    log.info('Using RSF checkpoint: %s threshold=%.8f', checkpoint_dir, rsf.threshold)
    return rsf, model_bundle


def run() -> None:
    args = parse_args()
    config_path, config, runtime = build_effective_config(args)
    validate_basic_config(config)

    seed = int(config.get('seed', 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    classifier_names = selected_classifiers(runtime['model_name'])
    mode = runtime['mode']
    if mode not in {'cache', 'train', 'infer', 'all'}:
        raise ValueError(f'Unsupported mode: {mode}')
    if mode in {'train', 'infer', 'all'}:
        validate_classifier_dependencies(classifier_names)

    log.info('Pointwise classifier mode=%s classifiers=%s', mode, classifier_names)
    model_bundle: tuple[torch.nn.Module, str, torch.device, Path] | None = None

    train_cache_path: Path | None = None
    if mode in {'cache', 'train', 'all'} or runtime['rsf_enabled']:
        train_cache_path, model_bundle = ensure_residual_cache(config, runtime['train_lmdb_path'], model_bundle)

    if mode == 'cache':
        for lmdb_path in runtime['infer_lmdb_paths']:
            _, model_bundle = ensure_residual_cache(config, lmdb_path, model_bundle)
        return

    rsf: ResidualShapeFilter | None = None
    train_rsf_keep_keys: set[str] | None = None
    if runtime['rsf_enabled']:
        if train_cache_path is None:
            raise RuntimeError('RSF requires the training residual cache')
        rsf, model_bundle = prepare_rsf(config, runtime, train_cache_path, model_bundle)
        if mode in {'train', 'all'}:
            train_rsf_keep_keys = collect_keep_keys(
                rsf,
                train_cache_path,
                resolve_repo_path(runtime['train_lmdb_path']),
                config=config['RSF'],
            )
            log.info('RSF retained %d training triplets', len(train_rsf_keep_keys))

    classifier_dirs: dict[str, Path] = {}
    if mode in {'train', 'all'}:
        if train_cache_path is None:
            raise RuntimeError('train_cache_path was not prepared')
        classifier_dirs = train_classifiers(
            config_path,
            config,
            train_cache_path,
            classifier_names,
            train_rsf_keep_keys,
        )

    if mode == 'train':
        return

    if mode == 'infer':
        classifier_dirs = {}

    if mode in {'infer', 'all'}:
        for clf_name in classifier_names:
            classifier, classifier_dir = load_classifier_checkpoint(config, clf_name)
            classifier_dirs[clf_name] = classifier_dir

        for lmdb_path in runtime['infer_lmdb_paths']:
            infer_cache_path, model_bundle = ensure_residual_cache(config, lmdb_path, model_bundle)
            infer_rsf_keep_keys = None
            if rsf is not None:
                source_path = resolve_repo_path(lmdb_path)
                test_path = resolve_repo_path(config["Eval"]["test_lmdb_path"])
                infer_rsf_keep_keys = collect_keep_keys(
                    rsf,
                    infer_cache_path,
                    source_path,
                    config=config['RSF'],
                    performance_section=(
                        "inference.rsf.test"
                        if source_path.resolve() == test_path.resolve()
                        else None
                    ),
                )
                log.info('RSF retained %d triplets for %s', len(infer_rsf_keep_keys), lmdb_path)
            for clf_name in classifier_names:
                classifier, classifier_dir = load_classifier_checkpoint(config, clf_name)
                run_inference(
                    config,
                    clf_name,
                    classifier,
                    classifier_dir,
                    infer_cache_path,
                    lmdb_path,
                    infer_rsf_keep_keys,
                )


if __name__ == '__main__':
    run()
