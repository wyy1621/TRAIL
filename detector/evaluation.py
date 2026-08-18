"""Evaluate a fixed pointwise classifier operating point on RADBench.

Purpose:
    1. Read the test classifier inference LMDB.
    2. Apply one configured point threshold and one configured ADF density threshold.
    3. Write fixed-point metrics, summaries, and probability CDF diagnostics.

Key parameters:
    - --config: path to the repository config.
    - --clf-name: lr / lightgbm / xgboost / mlp / all.
    - --max-processes: overrides Eval.max_processes for message scanning.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import gc
import io
import json
import logging
import math
import multiprocessing as mp
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import lmdb
import matplotlib
import numpy as np
import yaml
import zstandard as zstd
from matplotlib import ticker
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_PATH = Path(__file__).resolve()
PIPELINE_NO_SPARSE_DIR = SCRIPT_PATH.parent
ROOT_DIR = PIPELINE_NO_SPARSE_DIR.parent
CONFIG_ROOT = ROOT_DIR
sys.path.insert(0, str(ROOT_DIR))

from common import beijing, compute_binary_metrics, decodeMRT, iter_announce_update_parts, radbench_message_key  # noqa: E402
from common.performance import get_active_recorder  # noqa: E402
from detector.eval_common import collect_radbench_target_files, load_radbench_gt_msg_set  # noqa: E402
from detector.eval_utils import extract_clean_triplets  # noqa: E402
from detector.rsf import rsf_checkpoint_dir  # noqa: E402


logging.Formatter.converter = beijing
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
log = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = ROOT_DIR / 'config.yaml'
SUPPORTED_CLASSIFIERS = ('lr', 'lightgbm', 'xgboost', 'mlp')
DATASET_ORDER = ('Test',)
LEVELS = ('point', 'triplet', 'message')
ADF_WINDOW_SIZE = 10
ADF_WINDOW_STEP = 1

METRIC_FIELDNAMES = [
    'classifier',
    'dataset',
    'scope',
    'event_name',
    'point_threshold',
    'adf_enabled',
    'adf_density_threshold',
    'adf_min_count',
    'adf_window_size',
    'adf_window_step',
    'point_tp',
    'point_fp',
    'point_tn',
    'point_fn',
    'point_precision',
    'point_recall',
    'point_f1',
    'point_fpr',
    'triplet_tp',
    'triplet_fp',
    'triplet_tn',
    'triplet_fn',
    'triplet_precision',
    'triplet_recall',
    'triplet_f1',
    'triplet_fpr',
    'message_tp',
    'message_fp',
    'message_tn',
    'message_fn',
    'message_precision',
    'message_recall',
    'message_f1',
    'message_fpr',
    'point_sample_count',
    'point_positive_count',
    'point_pred_positive_count',
    'triplet_count',
    'triplet_gt_count',
    'triplet_pred_positive_count',
    'message_count',
    'message_gt_count',
    'message_pred_positive_count',
    'raw_alert_point_count',
    'kept_alert_point_count',
    'raw_pred_triplet_count',
    'kept_pred_triplet_count',
]

_G_SLOT_MASKS: dict[int, dict[tuple[int, int, int], int]] | None = None
_G_GT_MSG_SET: set[tuple[str, str, str]] | None = None
_G_CASE_COUNT = 0


def parse_args() -> argparse.Namespace:
    """Parse fixed-point evaluation command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate one fixed pointwise operating point.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml.")
    parser.add_argument("--clf-name", choices=[*SUPPORTED_CLASSIFIERS, "all"], default=None)
    parser.add_argument("--output-root", default=None, help="Override Eval.output_root.")
    parser.add_argument("--max-processes", type=int, default=None, help="Override Eval.max_processes.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config.

    Args:
        config_path: YAML config path.

    Returns:
        Parsed config mapping.
    """
    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f'Config must be a mapping: {config_path}')
    return config


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve absolute or repository-relative path.

    Args:
        path: Path string or Path object.

    Returns:
        Absolute normalized path.
    """
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (CONFIG_ROOT / path).resolve()


def sanitize_tag(value: str) -> str:
    """Convert arbitrary text to a path-safe tag.

    Args:
        value: Raw tag text.

    Returns:
        Path-safe tag containing alnum, dash, and underscore characters.
    """
    return ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in value).strip('_')


def dataset_tag(lmdb_path: str | Path) -> str:
    """Return inference dataset tag from a source LMDB path.

    Args:
        lmdb_path: Source LMDB path.

    Returns:
        Sanitized final component without `.lmdb`.
    """
    text = Path(str(lmdb_path).rstrip('/')).name
    if text.endswith('.lmdb'):
        text = text[:-5]
    return sanitize_tag(text)


def checkpoint_tag(checkpoint_path: str | Path) -> str:
    """Return checkpoint stem tag used by no-sparse classifier outputs."""
    return sanitize_tag(Path(checkpoint_path).stem)


def classifier_run_tag(config: dict[str, Any]) -> str:
    """Return the classifier output tag, including RSF identity when enabled."""
    tag = checkpoint_tag(config['UnsuperviseModel']['checkpoint'])
    rsf_tag = config.get('RSF', {}).get('runtime_checkpoint_tag')
    if rsf_tag:
        tag = f'{tag}_rsf_{sanitize_tag(str(rsf_tag))}'
    return tag


def selected_classifiers(config: dict[str, Any], override: str | None) -> list[str]:
    """Resolve classifier names from CLI override or config.

    Args:
        config: Effective no-sparse config.
        override: Optional CLI classifier name.

    Returns:
        List of classifier names to evaluate.
    """
    model_name = override or str(config['ClassifierModel'].get('model_name', 'lr'))
    if model_name == 'all':
        return list(SUPPORTED_CLASSIFIERS)
    if model_name not in SUPPORTED_CLASSIFIERS:
        raise ValueError(f'Unsupported classifier: {model_name}')
    return [model_name]


def inference_lmdb_path(config: dict[str, Any], clf_name: str, source_lmdb_path: str | Path) -> Path:
    """Resolve no-sparse classifier inference LMDB path.

    Args:
        config: Effective no-sparse config.
        clf_name: Classifier name.
        source_lmdb_path: Original source LMDB path.

    Returns:
        Expected inference LMDB directory path.
    """
    output_cfg = config.get('Output', {})
    root = resolve_repo_path(output_cfg.get('inference_root', 'pipeline_no_sparse/clf_inference'))
    ckpt_tag = classifier_run_tag(config)
    input_source = config['ClassifierModel']['input_source']
    return root / f'{clf_name}_{input_source}_clf_{ckpt_tag}' / f'{dataset_tag(source_lmdb_path)}.lmdb'


def classifier_output_dir(config: dict[str, Any], clf_name: str, output_root: Path) -> Path:
    """Resolve threshold-eval output directory for one classifier."""
    rsf_tag = config.get('RSF', {}).get('runtime_checkpoint_tag')
    if rsf_tag:
        output_root = output_root / f'rsf_{sanitize_tag(str(rsf_tag))}'
    ckpt_tag = checkpoint_tag(config['UnsuperviseModel']['checkpoint'])
    input_source = config['ClassifierModel']['input_source']
    return output_root / f'{clf_name}_{input_source}_clf_{ckpt_tag}'


def classifier_eval_output_dir(
    config: dict[str, Any],
    clf_name: str,
    output_root: Path,
) -> Path:
    """Resolve the fixed-point output directory for one classifier."""
    return classifier_output_dir(config, clf_name, output_root)


def parse_bool(value: Any, name: str) -> bool:
    """Parse a config boolean and fail fast on ambiguous values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('true', '1', 'yes', 'y', 'on'):
            return True
        if lowered in ('false', '0', 'no', 'n', 'off'):
            return False
    raise ValueError(f'{name} must be a boolean, got {value!r}')


def parse_finite_float(value: Any, name: str) -> float:
    """Parse a required numeric scalar and reject sequence-like values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a numeric scalar, got {type(value).__name__}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return parsed


def compute_adf_min_count(density_threshold: float, window_size: int) -> int:
    """Convert an ADF density threshold into the required alert count."""
    return int(math.ceil(float(density_threshold) * int(window_size) - 1e-12))


def build_eval_cases(point_threshold: float, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the single configured evaluation case.

    Args:
        point_threshold: Fixed anomaly-probability threshold.
        runtime: Resolved fixed-point evaluation options.

    Returns:
        A one-element case list used by the vectorized metric core.
    """
    if bool(runtime["adf_enabled"]):
        density_threshold = float(runtime["adf_density_threshold"])
        return [
            {
                "combo_index": 0,
                "point_case_index": 0,
                "threshold": float(point_threshold),
                "adf_enabled": True,
                "density_case_index": 0,
                "adf_density_threshold": density_threshold,
                "adf_min_count": compute_adf_min_count(
                    density_threshold,
                    runtime["adf_window_size"],
                ),
                "adf_window_size": int(runtime["adf_window_size"]),
                "adf_window_step": int(runtime["adf_window_step"]),
            }
        ]
    return [
        {
            "combo_index": 0,
            "point_case_index": 0,
            "threshold": float(point_threshold),
            "adf_enabled": False,
            "density_case_index": "",
            "adf_density_threshold": "",
            "adf_min_count": "",
            "adf_window_size": "",
            "adf_window_step": "",
        }
    ]


def read_pickle(txn: lmdb.Transaction, key: str, *, required: bool = True) -> Any:
    """Read a pickled LMDB metadata value.

    Args:
        txn: Open read transaction.
        key: ASCII metadata key.
        required: Whether a missing key is an error.

    Returns:
        Unpickled value, or None when optional and missing.
    """
    raw = txn.get(key.encode('ascii'))
    if raw is None:
        if required:
            raise KeyError(f'Missing LMDB metadata key: {key}')
        return None
    return pickle.loads(raw)


def unpack_npz_payload(decompressor: zstd.ZstdDecompressor, raw: bytes) -> dict[str, np.ndarray]:
    """Decompress and unpack one zstd-compressed NPZ payload.

    Args:
        decompressor: zstd decompressor.
        raw: Compressed LMDB value.

    Returns:
        Mapping of NPZ array names to numpy arrays.
    """
    payload = decompressor.decompress(raw)
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def parse_radbench_key(key: str) -> tuple[str, tuple[int, int, int]]:
    """Parse `{event_name}_{as1}_{as2}_{as3}` key.

    Args:
        key: RADBench data key.

    Returns:
        `(event_name, triplet)`.
    """
    parts = key.rsplit('_', 3)
    if len(parts) != 4:
        raise ValueError(f'Invalid RADBench key: {key}')
    return parts[0], (int(parts[1]), int(parts[2]), int(parts[3]))


def load_data_keys(lmdb_path: Path) -> list[str]:
    """Load sorted data keys from an inference LMDB.

    Args:
        lmdb_path: Inference LMDB directory path.

    Returns:
        Sorted non-metadata data keys.
    """
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, max_readers=64)
    try:
        keys: list[str] = []
        with env.begin(write=False) as txn:
            for key_bytes, _ in txn.cursor():
                if key_bytes.startswith(b'__') and key_bytes.endswith(b'__'):
                    continue
                key = key_bytes.decode('ascii')
                if len(key.rsplit('_', 3)) == 4:
                    keys.append(key)
        return sorted(keys)
    finally:
        env.close()


def normalize_triplet_set(values: Any) -> set[tuple[int, int, int]]:
    """Normalize raw triplet metadata into integer triplet tuples."""
    return {tuple(map(int, triplet)) for triplet in values or []}


def load_inference_metadata(lmdb_path: Path) -> dict[str, Any]:
    """Load event and GT metadata from an inference LMDB.

    Args:
        lmdb_path: Inference LMDB path.

    Returns:
        Metadata dict containing event names, timestamps, and GT triplet sets.
    """
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, max_readers=64)
    try:
        with env.begin(write=False) as txn:
            event_names = [str(name) for name in read_pickle(txn, '__event_names__')]
            anomaly_triplets_raw = read_pickle(txn, '__anomaly_triplets__')
            if not isinstance(anomaly_triplets_raw, dict):
                raise TypeError(f'__anomaly_triplets__ must be a RADBench event mapping: {lmdb_path}')
            event_timestamps = {
                event_name: [int(ts) for ts in read_pickle(txn, f'__{event_name}_timestamp__')]
                for event_name in event_names
            }
    finally:
        env.close()

    gt_triplets = {
        event_name: normalize_triplet_set(anomaly_triplets_raw.get(event_name, set()))
        for event_name in event_names
    }
    return {
        'event_names': event_names,
        'event_timestamps': event_timestamps,
        'gt_triplets': gt_triplets,
    }


def threshold_mask(eligible_count: int) -> int:
    """Build a bitmask for thresholds with index `< eligible_count`."""
    if eligible_count <= 0:
        return 0
    return (1 << int(eligible_count)) - 1


def add_mask_to_counter(counter: np.ndarray, mask: int) -> None:
    """Increment threshold counters selected by a bitmask.

    Args:
        counter: One-dimensional int64 counter array.
        mask: Integer bitmask where bit i maps to threshold i.

    Returns:
        None. The counter is modified in place.
    """
    while mask:
        bit = mask & -mask
        counter[bit.bit_length() - 1] += 1
        mask ^= bit


def new_confusion_counter(case_count: int) -> dict[str, np.ndarray]:
    """Create vectorized confusion counters for all thresholds."""
    return {
        'tp': np.zeros(case_count, dtype=np.int64),
        'fp': np.zeros(case_count, dtype=np.int64),
        'tn': np.zeros(case_count, dtype=np.int64),
        'fn': np.zeros(case_count, dtype=np.int64),
    }


def write_csv_rows(csv_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write normalized CSV rows.

    Args:
        csv_path: Output CSV path.
        rows: Row dictionaries.
        fieldnames: Stable output columns.

    Returns:
        None. The CSV file is overwritten.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, '') for field in fieldnames} for row in rows)


def metric_values(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    """Compute unweighted binary metric values from confusion counts."""
    metrics = compute_binary_metrics(int(tp), int(fp), int(tn), int(fn))
    return {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'fpr': float(metrics['fpr']),
    }


def validate_binary_labels(labels: np.ndarray, key: str) -> np.ndarray:
    """Validate and normalize point labels.

    Args:
        labels: Raw label array.
        key: LMDB key used in error messages.

    Returns:
        int8 label array containing only 0 and 1.
    """
    labels = labels.astype(np.int8, copy=False).reshape(-1)
    unique = np.unique(labels)
    if np.any((unique != 0) & (unique != 1)):
        raise ValueError(f'{key} contains non-binary labels: {unique.tolist()}')
    return labels


def read_probability_payload(
    txn: lmdb.Transaction,
    decompressor: zstd.ZstdDecompressor,
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read probability, label, and timestamp arrays for one triplet key.

    Args:
        txn: Read-only LMDB transaction.
        decompressor: zstd decompressor.
        key: Inference LMDB data key.

    Returns:
        `(prob, labels, timestamps)` arrays for observed points.
    """
    raw = txn.get(key.encode('ascii'))
    if raw is None:
        raise KeyError(f'Missing inference key: {key}')
    payload = unpack_npz_payload(decompressor, raw)
    for field in ('prob', 'label', 'timestamp'):
        if field not in payload:
            raise KeyError(f'Missing `{field}` in inference payload: {key}')
    prob = payload['prob'].astype(np.float64, copy=False).reshape(-1)
    labels = validate_binary_labels(payload['label'], key)
    timestamps = payload['timestamp'].astype(np.int64, copy=False).reshape(-1)
    if prob.shape[0] != labels.shape[0] or prob.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f'{key} payload length mismatch: prob={prob.shape[0]}, '
            f'label={labels.shape[0]}, timestamp={timestamps.shape[0]}'
        )
    if not np.all(np.isfinite(prob)):
        raise ValueError(f'{key} contains nan or inf probabilities')
    return prob, labels, timestamps


def update_point_counts(counter: dict[str, np.ndarray], prob: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> None:
    """Update observed point-level confusion counters.

    Args:
        counter: Mutable confusion counter for one event.
        prob: Probability array for one triplet.
        labels: Binary label array for the same points.
        thresholds: Increasing threshold vector.

    Returns:
        None. The counter is modified in place.
    """
    if prob.size == 0:
        return
    eligible = np.searchsorted(thresholds, prob, side='right')
    positive = labels == 1
    negative = ~positive
    for idx in range(len(thresholds)):
        pred = eligible > idx
        counter['tp'][idx] += int(np.count_nonzero(pred & positive))
        counter['fp'][idx] += int(np.count_nonzero(pred & negative))
        counter['fn'][idx] += int(np.count_nonzero(~pred & positive))
        counter['tn'][idx] += int(np.count_nonzero(~pred & negative))


def update_slot_masks(
    slot_masks: dict[int, dict[tuple[int, int, int], int]],
    triplet: tuple[int, int, int],
    prob: np.ndarray,
    timestamps: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    """Update per-minute triplet threshold bitmasks for message evaluation.

    Args:
        slot_masks: Mutable `minute_ts -> triplet -> threshold bitmask` map.
        triplet: Current AS triplet.
        prob: Probability array for observed points.
        timestamps: Minute timestamps for observed points.
        thresholds: Increasing threshold vector.

    Returns:
        None. The slot map is modified in place.
    """
    if prob.size == 0:
        return
    eligible = np.searchsorted(thresholds, prob, side='right')
    positive_indices = np.nonzero(eligible > 0)[0]
    for point_idx in positive_indices:
        minute_ts = int(timestamps[point_idx]) // 60 * 60
        mask = threshold_mask(int(eligible[point_idx]))
        triplet_masks = slot_masks[minute_ts]
        triplet_masks[triplet] = triplet_masks.get(triplet, 0) | mask


def compute_position_max_window_counts(
    positions: np.ndarray,
    seq_len: int,
    window_size: int,
    window_step: int,
) -> np.ndarray:
    """Compute each alert position's best covering-window alert count.

    Args:
        positions: Unique sorted alert indices in the event minute sequence.
        seq_len: Full event sequence length.
        window_size: ADF sliding-window length.
        window_step: ADF sliding-window step.

    Returns:
        Integer array aligned with `positions`; each value is the maximum alert
        count among valid windows covering that alert point.
    """
    if positions.size == 0 or seq_len < window_size:
        return np.zeros(positions.shape[0], dtype=np.int16)

    alert = np.zeros(seq_len, dtype=np.int16)
    alert[positions] = 1
    cumsum = np.concatenate(([0], np.cumsum(alert, dtype=np.int32)))
    all_start_counts = cumsum[window_size:] - cumsum[:-window_size]
    start_counts = all_start_counts[::window_step]
    max_start_index = int(all_start_counts.shape[0] - 1)
    max_counts = np.zeros(positions.shape[0], dtype=np.int16)

    for idx, pos_raw in enumerate(positions):
        pos = int(pos_raw)
        left = max(0, pos - window_size + 1)
        right = min(pos, max_start_index)
        left_slot = (left + window_step - 1) // window_step
        right_slot = right // window_step
        if left_slot <= right_slot:
            max_counts[idx] = int(np.max(start_counts[left_slot:right_slot + 1]))
    return max_counts


def apply_fixed_adf(
    prob: np.ndarray,
    timestamps: np.ndarray,
    *,
    event_timestamp_index: dict[int, int],
    sequence_length: int,
    point_threshold: float,
    density_threshold: float,
    window_size: int,
    window_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the deployed fixed point threshold and ADF to one triplet.

    Args:
        prob: Point anomaly probabilities for one triplet.
        timestamps: Timestamps aligned with ``prob``.
        event_timestamp_index: Minute timestamp to event sequence position.
        sequence_length: Full event sequence length.
        point_threshold: Fixed classifier operating point.
        density_threshold: Fixed ADF density operating point.
        window_size: ADF window length.
        window_step: ADF window start step.

    Returns:
        ``(candidate_indices, retained_indices, minute_timestamps)``.
    """
    if prob.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"Fixed ADF length mismatch: prob={prob.shape[0]}, timestamps={timestamps.shape[0]}"
        )
    minute_timestamps = np.empty(timestamps.shape[0], dtype=np.int64)
    positions = np.empty(timestamps.shape[0], dtype=np.int64)
    for point_idx, timestamp in enumerate(timestamps):
        minute_ts = normalize_minute_timestamp(int(timestamp))
        if minute_ts not in event_timestamp_index:
            raise ValueError(f"Point timestamp {minute_ts} not found in event metadata")
        minute_timestamps[point_idx] = minute_ts
        positions[point_idx] = event_timestamp_index[minute_ts]

    candidate_indices = np.nonzero(prob >= float(point_threshold))[0]
    if candidate_indices.size == 0:
        return candidate_indices, candidate_indices.copy(), minute_timestamps

    candidate_positions = positions[candidate_indices]
    order = np.argsort(candidate_positions)
    candidate_positions = candidate_positions[order]
    candidate_indices = candidate_indices[order]
    if np.unique(candidate_positions).size != candidate_positions.size:
        raise ValueError("Duplicate alert positions found for one triplet")

    max_window_counts = compute_position_max_window_counts(
        candidate_positions,
        seq_len=sequence_length,
        window_size=window_size,
        window_step=window_step,
    )
    min_count = compute_adf_min_count(density_threshold, window_size)
    retained_indices = candidate_indices[max_window_counts >= min_count]
    return candidate_indices, retained_indices, minute_timestamps


def add_labeled_point_mask(counter: dict[str, np.ndarray], label: int, mask: int) -> None:
    """Increment TP or FP point counters for every set bit in `mask`."""
    if int(label) == 1:
        add_mask_to_counter(counter['tp'], mask)
    else:
        add_mask_to_counter(counter['fp'], mask)


def normalize_minute_timestamp(timestamp: int) -> int:
    """Normalize a timestamp to the minute bucket used by message evaluation."""
    return int(timestamp) // 60 * 60


def build_event_timestamp_index(event_name: str, timestamps: list[int]) -> dict[int, int]:
    """Build a minute timestamp to sequence-index map for one RADBench event."""
    if not timestamps:
        raise ValueError(f'Missing timestamp metadata for event={event_name}')
    ts_to_index = {normalize_minute_timestamp(ts): idx for idx, ts in enumerate(timestamps)}
    if len(ts_to_index) != len(timestamps):
        raise ValueError(f'Duplicate minute timestamps found in metadata for event={event_name}')
    return ts_to_index

def analyze_inference_dataset_with_adf(
    *,
    dataset_name: str,
    classifier: str,
    inference_path: Path,
    thresholds: np.ndarray,
    runtime: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze one classifier inference LMDB with ADF density filtering.

    Args:
        dataset_name: `Val` or `Test`.
        classifier: Classifier name for progress display.
        inference_path: Classifier inference LMDB.
        thresholds: Increasing point-threshold vector.
        runtime: Resolved ADF runtime settings.
        cases: Ordered point-threshold and density-threshold cases.

    Returns:
        Dataset state with ADF-filtered point, triplet, and message slot masks.
    """
    metadata = load_inference_metadata(inference_path)
    event_names = metadata['event_names']
    gt_triplets: dict[str, set[tuple[int, int, int]]] = metadata['gt_triplets']
    case_count = len(thresholds)
    density_thresholds = [float(runtime['adf_density_threshold'])]
    density_count = len(density_thresholds)
    combo_count = len(cases)
    expected_combo_count = case_count * density_count
    if combo_count != expected_combo_count:
        raise ValueError(f'ADF case count mismatch: {combo_count} != {expected_combo_count}')

    window_size = int(runtime['adf_window_size'])
    window_step = int(runtime['adf_window_step'])
    event_ts_to_index = {
        event_name: build_event_timestamp_index(event_name, metadata['event_timestamps'][event_name])
        for event_name in event_names
    }

    point_counts = {event_name: new_confusion_counter(combo_count) for event_name in event_names}
    triplet_counts = {event_name: new_confusion_counter(combo_count) for event_name in event_names}
    all_triplets = {event_name: set() for event_name in event_names}
    slot_masks = {event_name: defaultdict(dict) for event_name in event_names}
    cdf_samples = {event_name: {0: [], 1: []} for event_name in event_names}
    raw_alert_counts = {event_name: np.zeros(combo_count, dtype=np.int64) for event_name in event_names}
    kept_alert_counts = {event_name: np.zeros(combo_count, dtype=np.int64) for event_name in event_names}
    raw_triplet_counts = {event_name: np.zeros(combo_count, dtype=np.int64) for event_name in event_names}
    kept_triplet_counts = {event_name: np.zeros(combo_count, dtype=np.int64) for event_name in event_names}
    point_positive_counts = {event_name: 0 for event_name in event_names}
    point_negative_counts = {event_name: 0 for event_name in event_names}

    keys = load_data_keys(inference_path)
    decompressor = zstd.ZstdDecompressor()
    env = lmdb.open(str(inference_path), readonly=True, lock=False, readahead=False, max_readers=64)
    try:
        with env.begin(write=False) as txn:
            progress = tqdm(keys, desc=f'Analyze {classifier} {dataset_name} with ADF')
            for key in progress:
                event_name, triplet = parse_radbench_key(key)
                if event_name not in point_counts:
                    raise KeyError(f'{key} belongs to event not listed in __event_names__: {event_name}')
                prob, labels, timestamps = read_probability_payload(txn, decompressor, key)
                all_triplets[event_name].add(triplet)

                if prob.size:
                    cdf_samples[event_name][0].append(prob[labels == 0].astype(np.float32, copy=False))
                    cdf_samples[event_name][1].append(prob[labels == 1].astype(np.float32, copy=False))
                    point_positive_counts[event_name] += int(np.count_nonzero(labels == 1))
                    point_negative_counts[event_name] += int(np.count_nonzero(labels == 0))
                else:
                    continue

                if case_count != 1 or density_count != 1:
                    raise ValueError("Fixed ADF evaluation requires one point and density threshold")
                candidate_indices, retained_indices, minute_timestamps = apply_fixed_adf(
                    prob,
                    timestamps,
                    event_timestamp_index=event_ts_to_index[event_name],
                    sequence_length=len(metadata['event_timestamps'][event_name]),
                    point_threshold=float(thresholds[0]),
                    density_threshold=float(density_thresholds[0]),
                    window_size=window_size,
                    window_step=window_step,
                )
                if candidate_indices.size:
                    raw_alert_counts[event_name][0] += int(candidate_indices.size)
                    raw_triplet_counts[event_name][0] += 1

                retained_triplet_mask = 0
                for point_idx in retained_indices:
                    minute_ts = int(minute_timestamps[point_idx])
                    triplet_masks = slot_masks[event_name][minute_ts]
                    triplet_masks[triplet] = 1
                    add_labeled_point_mask(point_counts[event_name], int(labels[point_idx]), 1)
                    kept_alert_counts[event_name][0] += 1
                    retained_triplet_mask = 1

                if retained_triplet_mask:
                    if triplet in gt_triplets.get(event_name, set()):
                        triplet_counts[event_name]['tp'][0] += 1
                    else:
                        triplet_counts[event_name]['fp'][0] += 1
                    kept_triplet_counts[event_name][0] += 1
    finally:
        env.close()

    for event_name in event_names:
        positive_count = int(point_positive_counts[event_name])
        negative_count = int(point_negative_counts[event_name])
        point_counts[event_name]['fn'] = np.full(combo_count, positive_count, dtype=np.int64) - point_counts[event_name]['tp']
        point_counts[event_name]['tn'] = np.full(combo_count, negative_count, dtype=np.int64) - point_counts[event_name]['fp']
        if np.any(point_counts[event_name]['fn'] < 0) or np.any(point_counts[event_name]['tn'] < 0):
            raise ValueError(f'Invalid ADF point confusion counts for event={event_name}, dataset={dataset_name}')

        event_gt_triplets = gt_triplets.get(event_name, set())
        gt_count = len(event_gt_triplets)
        non_gt_count = len(all_triplets[event_name] - event_gt_triplets)
        triplet_counts[event_name]['fn'] = np.full(combo_count, gt_count, dtype=np.int64) - triplet_counts[event_name]['tp']
        triplet_counts[event_name]['tn'] = np.full(combo_count, non_gt_count, dtype=np.int64) - triplet_counts[event_name]['fp']
        if np.any(triplet_counts[event_name]['fn'] < 0) or np.any(triplet_counts[event_name]['tn'] < 0):
            raise ValueError(f'Invalid ADF triplet confusion counts for event={event_name}, dataset={dataset_name}')

    return {
        'dataset_name': dataset_name,
        'event_names': event_names,
        'event_timestamps': metadata['event_timestamps'],
        'gt_triplets': gt_triplets,
        'all_triplets': all_triplets,
        'point_counts': point_counts,
        'triplet_counts': triplet_counts,
        'slot_masks': {event: dict(value) for event, value in slot_masks.items()},
        'cdf_samples': cdf_samples,
        'raw_alert_counts': raw_alert_counts,
        'kept_alert_counts': kept_alert_counts,
        'raw_triplet_counts': raw_triplet_counts,
        'kept_triplet_counts': kept_triplet_counts,
        'cases': cases,
    }


def analyze_inference_dataset(
    *,
    dataset_name: str,
    classifier: str,
    inference_path: Path,
    thresholds: np.ndarray,
    runtime: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze one classifier inference LMDB except raw message scanning.

    Args:
        dataset_name: `Val` or `Test`.
        classifier: Classifier name for progress display.
        inference_path: Classifier inference LMDB.
        thresholds: Increasing candidate thresholds.
        runtime: Resolved runtime options.
        cases: Ordered metric cases.

    Returns:
        Dataset state containing point counts, triplet counts, CDF samples,
        message slot masks, and metadata needed for later row construction.
    """
    if bool(runtime['adf_enabled']):
        return analyze_inference_dataset_with_adf(
            dataset_name=dataset_name,
            classifier=classifier,
            inference_path=inference_path,
            thresholds=thresholds,
            runtime=runtime,
            cases=cases,
        )

    metadata = load_inference_metadata(inference_path)
    event_names = metadata['event_names']
    gt_triplets: dict[str, set[tuple[int, int, int]]] = metadata['gt_triplets']
    case_count = len(thresholds)
    if len(cases) != case_count:
        raise ValueError(f'Raw threshold case count mismatch: {len(cases)} != {case_count}')

    point_counts = {event_name: new_confusion_counter(case_count) for event_name in event_names}
    triplet_counts = {event_name: new_confusion_counter(case_count) for event_name in event_names}
    all_triplets = {event_name: set() for event_name in event_names}
    slot_masks = {event_name: defaultdict(dict) for event_name in event_names}
    cdf_samples = {event_name: {0: [], 1: []} for event_name in event_names}

    keys = load_data_keys(inference_path)
    decompressor = zstd.ZstdDecompressor()
    env = lmdb.open(str(inference_path), readonly=True, lock=False, readahead=False, max_readers=64)
    try:
        with env.begin(write=False) as txn:
            progress = tqdm(keys, desc=f'Analyze {classifier} {dataset_name}')
            for key in progress:
                event_name, triplet = parse_radbench_key(key)
                if event_name not in point_counts:
                    raise KeyError(f'{key} belongs to event not listed in __event_names__: {event_name}')
                prob, labels, timestamps = read_probability_payload(txn, decompressor, key)
                all_triplets[event_name].add(triplet)

                if prob.size:
                    cdf_samples[event_name][0].append(prob[labels == 0].astype(np.float32, copy=False))
                    cdf_samples[event_name][1].append(prob[labels == 1].astype(np.float32, copy=False))

                update_point_counts(point_counts[event_name], prob, labels, thresholds)
                update_slot_masks(slot_masks[event_name], triplet, prob, timestamps, thresholds)

                if prob.size:
                    eligible = int(np.searchsorted(thresholds, float(prob.max()), side='right'))
                    mask = threshold_mask(eligible)
                    if mask:
                        if triplet in gt_triplets.get(event_name, set()):
                            add_mask_to_counter(triplet_counts[event_name]['tp'], mask)
                        else:
                            add_mask_to_counter(triplet_counts[event_name]['fp'], mask)
    finally:
        env.close()

    raw_alert_counts: dict[str, np.ndarray] = {}
    kept_alert_counts: dict[str, np.ndarray] = {}
    raw_triplet_counts: dict[str, np.ndarray] = {}
    kept_triplet_counts: dict[str, np.ndarray] = {}
    for event_name in event_names:
        event_gt_triplets = gt_triplets.get(event_name, set())
        gt_count = len(event_gt_triplets)
        non_gt_count = len(all_triplets[event_name] - event_gt_triplets)
        triplet_counts[event_name]['fn'] = np.full(case_count, gt_count, dtype=np.int64) - triplet_counts[event_name]['tp']
        triplet_counts[event_name]['tn'] = np.full(case_count, non_gt_count, dtype=np.int64) - triplet_counts[event_name]['fp']
        if np.any(triplet_counts[event_name]['fn'] < 0) or np.any(triplet_counts[event_name]['tn'] < 0):
            raise ValueError(f'Invalid triplet confusion counts for event={event_name}, dataset={dataset_name}')

        alert_counts = point_counts[event_name]['tp'] + point_counts[event_name]['fp']
        pred_triplet_counts = triplet_counts[event_name]['tp'] + triplet_counts[event_name]['fp']
        raw_alert_counts[event_name] = alert_counts.copy()
        kept_alert_counts[event_name] = alert_counts.copy()
        raw_triplet_counts[event_name] = pred_triplet_counts.copy()
        kept_triplet_counts[event_name] = pred_triplet_counts.copy()

    return {
        'dataset_name': dataset_name,
        'event_names': event_names,
        'event_timestamps': metadata['event_timestamps'],
        'gt_triplets': gt_triplets,
        'all_triplets': all_triplets,
        'point_counts': point_counts,
        'triplet_counts': triplet_counts,
        'slot_masks': {event: dict(value) for event, value in slot_masks.items()},
        'cdf_samples': cdf_samples,
        'raw_alert_counts': raw_alert_counts,
        'kept_alert_counts': kept_alert_counts,
        'raw_triplet_counts': raw_triplet_counts,
        'kept_triplet_counts': kept_triplet_counts,
        'cases': cases,
    }


def init_message_worker(slot_masks, gt_msg_set, case_count: int) -> None:
    """Initialize forked worker globals for message-level evaluation."""
    global _G_SLOT_MASKS
    global _G_GT_MSG_SET
    global _G_CASE_COUNT
    _G_SLOT_MASKS = slot_masks
    _G_GT_MSG_SET = gt_msg_set
    _G_CASE_COUNT = int(case_count)


def new_message_partial(case_count: int) -> dict[str, Any]:
    """Create an empty message-level partial accumulator."""
    return {
        'total_messages': 0,
        'total_gt_messages': 0,
        'tp': np.zeros(case_count, dtype=np.int64),
        'fp': np.zeros(case_count, dtype=np.int64),
    }


def process_update_file_for_messages(item: tuple[str, str, str]) -> dict[str, Any]:
    """Process one raw RADBench update file for all thresholds.

    Args:
        item: `(event_name, time_str, filepath)` tuple.

    Returns:
        Partial message counters for all thresholds.
    """
    if _G_SLOT_MASKS is None or _G_GT_MSG_SET is None:
        raise RuntimeError('Message worker context is not initialized.')

    _event_name, _time_str, filepath = item
    partial = new_message_partial(_G_CASE_COUNT)
    messages = decodeMRT(filepath)

    for parts in iter_announce_update_parts(messages):
        msg_ts_str = parts[1]
        as_path_str = parts[6]
        if not as_path_str:
            continue

        partial['total_messages'] += 1
        is_gt_message = radbench_message_key(parts) in _G_GT_MSG_SET
        if is_gt_message:
            partial['total_gt_messages'] += 1

        triplets_in_msg = extract_clean_triplets(as_path_str, parts[4])
        if not triplets_in_msg:
            continue

        minute_ts = int(msg_ts_str) // 60 * 60
        pred_triplets_at_ts = _G_SLOT_MASKS.get(minute_ts)
        if not pred_triplets_at_ts:
            continue

        active_mask = 0
        for triplet in triplets_in_msg:
            active_mask |= pred_triplets_at_ts.get(triplet, 0)
        if not active_mask:
            continue

        if is_gt_message:
            add_mask_to_counter(partial['tp'], active_mask)
        else:
            add_mask_to_counter(partial['fp'], active_mask)

    return partial


def evaluate_event_messages(
    *,
    event_name: str,
    slot_masks: dict[int, dict[tuple[int, int, int], int]],
    event_timestamps: dict[str, np.ndarray],
    updates_root: Path,
    gt_msg_root: Path,
    max_processes: int,
    case_count: int,
    dataset_name: str,
    event_csv: Path,
) -> dict[str, Any]:
    """Evaluate message-level metrics for one RADBench event.

    Args:
        event_name: RADBench event name.
        slot_masks: `minute_ts -> triplet -> threshold bitmask` predictions.
        event_timestamps: Dataset event timestamp metadata.
        updates_root: Raw RADBench update root.
        gt_msg_root: Ground-truth message root.
        max_processes: Maximum worker count.
        case_count: Number of thresholds.
        dataset_name: `Val` or `Test` for progress display.

    Returns:
        Message counters for all thresholds.
    """
    update_files = collect_radbench_target_files([event_name], event_timestamps, str(updates_root))
    gt_msg_set = load_radbench_gt_msg_set(event_name, str(gt_msg_root), str(event_csv))
    merged = new_message_partial(case_count)
    if not update_files:
        log.warning('[%s] event=%s has no matching RADBench update files', dataset_name, event_name)
        return merged

    worker_count = max(1, min(int(max_processes), len(update_files)))
    gc.collect()
    if hasattr(gc, 'freeze'):
        gc.freeze()
    try:
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=worker_count,
            initializer=init_message_worker,
            initargs=(slot_masks, gt_msg_set, case_count),
        ) as pool:
            partials = pool.imap_unordered(process_update_file_for_messages, update_files)
            for partial in tqdm(
                partials,
                total=len(update_files),
                desc=f'[{dataset_name}] eval messages event={event_name}',
                leave=False,
            ):
                merged['total_messages'] += int(partial['total_messages'])
                merged['total_gt_messages'] += int(partial['total_gt_messages'])
                merged['tp'] += partial['tp']
                merged['fp'] += partial['fp']
    finally:
        if hasattr(gc, 'unfreeze'):
            gc.unfreeze()

    log.info(
        '[%s] event=%s total_messages=%d gt_messages=%d update_files=%d workers=%d',
        dataset_name,
        event_name,
        merged['total_messages'],
        merged['total_gt_messages'],
        len(update_files),
        worker_count,
    )
    return merged


def evaluate_dataset_messages(
    state: dict[str, Any],
    updates_root: Path,
    gt_msg_root: Path,
    max_processes: int,
    case_count: int,
    event_csv: Path,
) -> dict[str, dict[str, Any]]:
    """Evaluate message-level counters for all events in one dataset.

    Args:
        state: Dataset state returned by `analyze_inference_dataset`.
        updates_root: Raw RADBench update root.
        gt_msg_root: Ground-truth message root.
        max_processes: Maximum worker count.
        case_count: Number of thresholds.

    Returns:
        Event to message counter mapping.
    """
    results = {}
    for event_name in state['event_names']:
        results[event_name] = evaluate_event_messages(
            event_name=event_name,
            slot_masks=state['slot_masks'][event_name],
            event_timestamps=state['event_timestamps'],
            updates_root=updates_root,
            gt_msg_root=gt_msg_root,
            max_processes=max_processes,
            case_count=case_count,
            dataset_name=state['dataset_name'],
            event_csv=event_csv,
        )
    return results


def finalize_message_counter(counter: dict[str, Any]) -> dict[str, np.ndarray | int]:
    """Build full message-level confusion arrays from TP/FP counters."""
    tp = counter['tp']
    fp = counter['fp']
    total_messages = int(counter['total_messages'])
    total_gt_messages = int(counter['total_gt_messages'])
    fn = np.full_like(tp, total_gt_messages) - tp
    tn = np.full_like(tp, total_messages) - tp - fp - fn
    if np.any(fn < 0) or np.any(tn < 0):
        raise ValueError('Invalid message-level confusion counts')
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total_messages': total_messages,
        'total_gt_messages': total_gt_messages,
    }


def sum_confusion_counters(counters: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Sum vectorized confusion counters across events."""
    if not counters:
        raise ValueError('No counters to sum')
    return {
        field: sum((counter[field] for counter in counters), np.zeros_like(counters[0][field]))
        for field in ('tp', 'fp', 'tn', 'fn')
    }


def build_rows_for_scope(
    *,
    classifier: str,
    dataset_name: str,
    scope: str,
    event_name: str,
    cases: list[dict[str, Any]],
    point_counter: dict[str, np.ndarray],
    triplet_counter: dict[str, np.ndarray],
    message_counter: dict[str, np.ndarray | int],
    raw_alert_counts: np.ndarray,
    kept_alert_counts: np.ndarray,
    raw_triplet_counts: np.ndarray,
    kept_triplet_counts: np.ndarray,
) -> list[dict[str, Any]]:
    """Build metric CSV rows for one event or overall scope.

    Args:
        classifier: Classifier name.
        dataset_name: `Val` or `Test`.
        scope: `event` or `overall`.
        event_name: Event name or `OVERALL`.
        cases: Ordered raw-threshold or ADF threshold-density cases.
        point_counter: Point-level confusion arrays.
        triplet_counter: Triplet-level confusion arrays.
        message_counter: Message-level confusion arrays plus totals.
        raw_alert_counts: Raw threshold alert counts before ADF.
        kept_alert_counts: Alert counts retained after ADF.
        raw_triplet_counts: Raw predicted triplet counts before ADF.
        kept_triplet_counts: Predicted triplet counts retained after ADF.

    Returns:
        Long-form metric rows for all cases.
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        idx = int(case['combo_index'])
        point = metric_values(
            point_counter['tp'][idx],
            point_counter['fp'][idx],
            point_counter['tn'][idx],
            point_counter['fn'][idx],
        )
        triplet = metric_values(
            triplet_counter['tp'][idx],
            triplet_counter['fp'][idx],
            triplet_counter['tn'][idx],
            triplet_counter['fn'][idx],
        )
        message = metric_values(
            message_counter['tp'][idx],
            message_counter['fp'][idx],
            message_counter['tn'][idx],
            message_counter['fn'][idx],
        )
        rows.append(
            {
                'classifier': classifier,
                'dataset': dataset_name,
                'scope': scope,
                'event_name': event_name,
                'point_case_index': int(case['point_case_index']),
                'point_threshold': f'{float(case["threshold"]):.10g}',
                'adf_enabled': bool(case['adf_enabled']),
                'density_case_index': case['density_case_index'],
                'adf_density_threshold': case['adf_density_threshold'],
                'adf_min_count': case['adf_min_count'],
                'adf_window_size': case['adf_window_size'],
                'adf_window_step': case['adf_window_step'],
                'point_tp': point['tp'],
                'point_fp': point['fp'],
                'point_tn': point['tn'],
                'point_fn': point['fn'],
                'point_precision': point['precision'],
                'point_recall': point['recall'],
                'point_f1': point['f1'],
                'point_fpr': point['fpr'],
                'triplet_tp': triplet['tp'],
                'triplet_fp': triplet['fp'],
                'triplet_tn': triplet['tn'],
                'triplet_fn': triplet['fn'],
                'triplet_precision': triplet['precision'],
                'triplet_recall': triplet['recall'],
                'triplet_f1': triplet['f1'],
                'triplet_fpr': triplet['fpr'],
                'message_tp': message['tp'],
                'message_fp': message['fp'],
                'message_tn': message['tn'],
                'message_fn': message['fn'],
                'message_precision': message['precision'],
                'message_recall': message['recall'],
                'message_f1': message['f1'],
                'message_fpr': message['fpr'],
                'point_sample_count': int(point['tp'] + point['fp'] + point['tn'] + point['fn']),
                'point_positive_count': int(point['tp'] + point['fn']),
                'point_pred_positive_count': int(point['tp'] + point['fp']),
                'triplet_count': int(triplet['tp'] + triplet['fp'] + triplet['tn'] + triplet['fn']),
                'triplet_gt_count': int(triplet['tp'] + triplet['fn']),
                'triplet_pred_positive_count': int(triplet['tp'] + triplet['fp']),
                'message_count': int(message_counter['total_messages']),
                'message_gt_count': int(message_counter['total_gt_messages']),
                'message_pred_positive_count': int(message['tp'] + message['fp']),
                'raw_alert_point_count': int(raw_alert_counts[idx]),
                'kept_alert_point_count': int(kept_alert_counts[idx]),
                'raw_pred_triplet_count': int(raw_triplet_counts[idx]),
                'kept_pred_triplet_count': int(kept_triplet_counts[idx]),
            }
        )
    return rows


def build_metric_rows(
    *,
    classifier: str,
    cases: list[dict[str, Any]],
    state: dict[str, Any],
    message_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build event and overall rows for one dataset.

    Args:
        classifier: Classifier name.
        cases: Ordered raw-threshold or ADF threshold-density cases.
        state: Dataset state from inference analysis.
        message_results: Event message counters.

    Returns:
        Long-form metric rows for event and overall scopes.
    """
    dataset_name = state['dataset_name']
    combo_count = len(cases)
    rows: list[dict[str, Any]] = []
    finalized_messages = {event: finalize_message_counter(message_results[event]) for event in state['event_names']}

    for event_name in state['event_names']:
        rows.extend(
            build_rows_for_scope(
                classifier=classifier,
                dataset_name=dataset_name,
                scope='event',
                event_name=event_name,
                cases=cases,
                point_counter=state['point_counts'][event_name],
                triplet_counter=state['triplet_counts'][event_name],
                message_counter=finalized_messages[event_name],
                raw_alert_counts=state['raw_alert_counts'][event_name],
                kept_alert_counts=state['kept_alert_counts'][event_name],
                raw_triplet_counts=state['raw_triplet_counts'][event_name],
                kept_triplet_counts=state['kept_triplet_counts'][event_name],
            )
        )

    overall_point = sum_confusion_counters([state['point_counts'][event] for event in state['event_names']])
    overall_triplet = sum_confusion_counters([state['triplet_counts'][event] for event in state['event_names']])
    overall_message = {
        'tp': sum((finalized_messages[event]['tp'] for event in state['event_names']), np.zeros(combo_count, dtype=np.int64)),
        'fp': sum((finalized_messages[event]['fp'] for event in state['event_names']), np.zeros(combo_count, dtype=np.int64)),
        'tn': sum((finalized_messages[event]['tn'] for event in state['event_names']), np.zeros(combo_count, dtype=np.int64)),
        'fn': sum((finalized_messages[event]['fn'] for event in state['event_names']), np.zeros(combo_count, dtype=np.int64)),
        'total_messages': sum(int(finalized_messages[event]['total_messages']) for event in state['event_names']),
        'total_gt_messages': sum(int(finalized_messages[event]['total_gt_messages']) for event in state['event_names']),
    }
    overall_raw_alert = sum(state['raw_alert_counts'].values(), np.zeros(combo_count, dtype=np.int64))
    overall_kept_alert = sum(state['kept_alert_counts'].values(), np.zeros(combo_count, dtype=np.int64))
    overall_raw_triplet = sum(state['raw_triplet_counts'].values(), np.zeros(combo_count, dtype=np.int64))
    overall_kept_triplet = sum(state['kept_triplet_counts'].values(), np.zeros(combo_count, dtype=np.int64))
    rows.extend(
        build_rows_for_scope(
            classifier=classifier,
            dataset_name=dataset_name,
            scope='overall',
            event_name='OVERALL',
            cases=cases,
            point_counter=overall_point,
            triplet_counter=overall_triplet,
            message_counter=overall_message,
            raw_alert_counts=overall_raw_alert,
            kept_alert_counts=overall_kept_alert,
            raw_triplet_counts=overall_raw_triplet,
            kept_triplet_counts=overall_kept_triplet,
        )
    )
    return rows


def plot_cdf_by_event(cdf_samples: dict[str, dict[int, list[np.ndarray]]], dataset_name: str, plot_path: Path) -> None:
    """Plot per-event empirical CDFs for Normal and Abnormal points.

    Args:
        cdf_samples: Event to label to probability-array lists.
        dataset_name: `Val` or `Test`.
        plot_path: Output PNG path.

    Returns:
        None. The plot is written to disk.
    """
    event_names = list(cdf_samples.keys())
    if not event_names:
        raise ValueError(f'No CDF samples found for dataset={dataset_name}')
    col_count = min(3, max(1, len(event_names)))
    row_count = int(math.ceil(len(event_names) / col_count))
    styles = {
        0: ('Normal', '#0072B2'),
        1: ('Abnormal', '#D55E00'),
    }

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'savefig.dpi': 300}):
        fig, axes = plt.subplots(
            row_count,
            col_count,
            figsize=(3.15 * col_count, 2.35 * row_count + 0.25),
            squeeze=False,
        )
        for idx, event_name in enumerate(event_names):
            ax = axes[idx // col_count][idx % col_count]
            for label_value, (label_name, color) in styles.items():
                arrays = [arr for arr in cdf_samples[event_name][label_value] if arr.size]
                if not arrays:
                    continue
                values = np.sort(np.concatenate(arrays).astype(np.float64, copy=False))
                y = np.arange(1, values.size + 1, dtype=np.float64) / float(values.size)
                ax.plot(values, y, color=color, linewidth=1.4, label=f'{label_name} (n={values.size})')
            ax.set_title(event_name, fontsize=8.5, pad=4)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis='both', color='#E1E1E1', linewidth=0.55, alpha=0.8)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
            ax.tick_params(width=0.7, length=3.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(frameon=False, fontsize=7.2, loc='lower right')

        for idx in range(len(event_names), row_count * col_count):
            axes[idx // col_count][idx % col_count].axis('off')

        fig.suptitle(f'{dataset_name} pointwise probability CDF by event', y=0.995, fontsize=10.0)
        fig.text(0.5, 0.025, 'Anomaly probability', ha='center', va='center', fontsize=9.0)
        fig.text(0.012, 0.5, 'CDF', ha='center', va='center', rotation='vertical', fontsize=9.0)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.095, top=0.915, wspace=0.20, hspace=0.34)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    log.info('Saved CDF plot: %s', plot_path)


def sort_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort fixed-point metric rows for stable CSV and summary output."""
    dataset_rank = {"Val": 0, "Test": 1}
    return sorted(
        rows,
        key=lambda row: (
            row["classifier"],
            dataset_rank.get(row["dataset"], 99),
            row["scope"] != "overall",
            row["event_name"],
        ),
    )


def format_count_line(row: dict[str, Any], level: str) -> str:
    """Format one event/overall metric row for summary text."""
    label = row['event_name'] if row['scope'] == 'event' else 'OVERALL'
    return (
        f'  {label:<48}  '
        f'TP={int(row[f"{level}_tp"]):>8}  '
        f'FP={int(row[f"{level}_fp"]):>8}  '
        f'TN={int(row[f"{level}_tn"]):>8}  '
        f'FN={int(row[f"{level}_fn"]):>8}  '
        f'P/R/F1/FPR='
        f'{float(row[f"{level}_precision"]):.6f}/'
        f'{float(row[f"{level}_recall"]):.6f}/'
        f'{float(row[f"{level}_f1"]):.6f}/'
        f'{float(row[f"{level}_fpr"]):.6f}'
    )


def write_fixed_summary(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    metrics_csv_path: Path,
    classifier: str,
    input_source: str,
    runtime: dict[str, Any],
) -> None:
    """Write the configured operating point and all Val/Test metrics."""
    sep = "=" * 72
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(f"{sep}\n")
        file_obj.write(f"  Classifier          : {classifier}\n")
        file_obj.write(f"  Input Source        : {input_source}\n")
        file_obj.write(f"  Source CSV          : {metrics_csv_path}\n")
        file_obj.write(f"  Point Threshold     : {float(runtime['point_threshold']):.10g}\n")
        file_obj.write(f"  ADF Enabled         : {bool(runtime['adf_enabled'])}\n")
        if runtime["adf_enabled"]:
            file_obj.write(
                f"  ADF Density         : {float(runtime['adf_density_threshold']):.10g}\n"
            )
            file_obj.write(
                f"  ADF Min Count       : "
                f"{compute_adf_min_count(runtime['adf_density_threshold'], runtime['adf_window_size'])}\n"
            )
            file_obj.write(f"  ADF Window Size     : {int(runtime['adf_window_size'])}\n")
            file_obj.write(f"  ADF Window Step     : {int(runtime['adf_window_step'])}\n")
        file_obj.write(f"{sep}\n")
        for dataset_name in DATASET_ORDER:
            dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
            if not dataset_rows:
                raise ValueError(f"No metric rows found for dataset={dataset_name}")
            for level in LEVELS:
                file_obj.write(f"  [{dataset_name} {level.title()} Level]\n")
                for row in dataset_rows:
                    file_obj.write(format_count_line(row, level) + "\n")
                file_obj.write(f"{sep}\n")
    log.info("Saved fixed-point summary: %s", output_path)


def resolve_runtime(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate the fixed operating point and evaluation paths."""
    if config.get("Dataset") != "RADBench":
        raise ValueError(f"Only RADBench is supported, got Dataset={config.get('Dataset')}")
    eval_cfg = config.get("Eval")
    if not isinstance(eval_cfg, dict):
        raise KeyError("Missing Eval config block in config.yaml")

    val_lmdb = resolve_repo_path(eval_cfg["val_lmdb_path"])
    test_lmdb = resolve_repo_path(eval_cfg["test_lmdb_path"])
    updates_root = resolve_repo_path(eval_cfg["updates_root"])
    gt_msg_root = resolve_repo_path(eval_cfg["gt_msg_root"])
    output_root = resolve_repo_path(args.output_root or eval_cfg.get("output_root", "output/evaluation"))
    max_processes_value = args.max_processes if args.max_processes is not None else eval_cfg.get("max_processes", 8)
    max_processes = min(16, int(max_processes_value))
    adf_enabled = parse_bool(eval_cfg["adf_enabled"], "Eval.adf_enabled")
    point_threshold = parse_finite_float(eval_cfg["point_threshold"], "Eval.point_threshold")
    adf_density_threshold = parse_finite_float(
        eval_cfg["adf_density_threshold"],
        "Eval.adf_density_threshold",
    )
    adf_window_size = int(eval_cfg.get("adf_window_size", ADF_WINDOW_SIZE))
    adf_window_step = int(eval_cfg.get("adf_window_step", ADF_WINDOW_STEP))
    event_csv = resolve_repo_path(config["DataPipeline"]["event_csv"])

    if point_threshold < 0.0 or point_threshold > 1.0:
        raise ValueError(f"Eval.point_threshold must be in [0, 1], got {point_threshold}")
    if adf_density_threshold <= 0.0 or adf_density_threshold > 1.0:
        raise ValueError(
            "Eval.adf_density_threshold must be in (0, 1], "
            f"got {adf_density_threshold}"
        )
    if max_processes < 1:
        raise ValueError(f"Eval.max_processes must be >= 1, got {max_processes}")
    if adf_window_size < 1:
        raise ValueError(f"Eval.adf_window_size must be >= 1, got {adf_window_size}")
    if adf_window_step < 1:
        raise ValueError(f"Eval.adf_window_step must be >= 1, got {adf_window_step}")

    rsf_cfg = config.get("RSF", {})
    rsf_enabled = parse_bool(rsf_cfg.get("enabled", False), "RSF.enabled")
    if rsf_enabled:
        unsup_checkpoint = resolve_repo_path(config["UnsuperviseModel"]["checkpoint"])
        train_lmdb = resolve_repo_path(config["ClassifierModel"]["train_lmdb_path"])
        rsf_val_lmdb = resolve_repo_path(rsf_cfg["val_lmdb_path"])
        checkpoint_root = resolve_repo_path(
            rsf_cfg.get("checkpoint_root", "output/rsf")
        )
        residual_root = resolve_repo_path(config["Output"]["residual_cache_root"])
        checkpoint_stem = checkpoint_tag(config["UnsuperviseModel"]["checkpoint"])
        train_cache = residual_root / checkpoint_stem / f"{dataset_tag(train_lmdb)}.lmdb"
        val_cache = residual_root / checkpoint_stem / f"{dataset_tag(rsf_val_lmdb)}.lmdb"
        checkpoint_dir = rsf_checkpoint_dir(
            unsup_checkpoint,
            train_lmdb,
            rsf_val_lmdb,
            checkpoint_root,
            rsf_cfg,
            train_cache_path=train_cache,
            val_cache_path=val_cache,
        )
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                "RSF checkpoint does not exist; run train_detector.py first: "
                f"{checkpoint_dir}"
            )
        rsf_cfg["runtime_checkpoint_tag"] = checkpoint_dir.name.rsplit("_", 1)[-1]
        rsf_cfg["runtime_checkpoint_path"] = str(checkpoint_dir)

    for candidate_path, description in [
        (val_lmdb, "Eval.val_lmdb_path"),
        (test_lmdb, "Eval.test_lmdb_path"),
        (updates_root, "Eval.updates_root"),
        (gt_msg_root, "Eval.gt_msg_root"),
    ]:
        if not candidate_path.exists():
            raise FileNotFoundError(f"{description} does not exist: {candidate_path}")

    return {
        "val_lmdb": val_lmdb,
        "test_lmdb": test_lmdb,
        "updates_root": updates_root,
        "gt_msg_root": gt_msg_root,
        "output_root": output_root,
        "max_processes": max_processes,
        "adf_enabled": adf_enabled,
        "adf_density_threshold": adf_density_threshold,
        "adf_window_size": adf_window_size,
        "adf_window_step": adf_window_step,
        "rsf_enabled": rsf_enabled,
        "rsf_checkpoint_path": rsf_cfg.get("runtime_checkpoint_path"),
        "point_threshold": point_threshold,
        "event_csv": event_csv,
    }


def _inference_identity(path: Path) -> Any:
    """Read the identity stored by the classifier inference stage."""
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            metadata = read_pickle(txn, '__metadata__')
    finally:
        env.close()
    identity = metadata.get('identity')
    if identity is None:
        raise ValueError(f'Inference metadata lacks identity: {path}')
    return identity


def _file_sha256(path: Path) -> str:
    """Hash one small evaluation control file such as the event CSV."""
    import hashlib

    digest = hashlib.sha256()
    with path.open('rb') as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest(root: Path) -> list[dict[str, Any]]:
    """Build a stable size/mtime manifest for evaluation data files."""
    return [
        {
            'path': str(path.relative_to(root)),
            'size': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns,
        }
        for path in sorted(item for item in root.rglob('*') if item.is_file())
    ]


def evaluation_identity(
    runtime: dict[str, Any],
    test_inference: Path,
) -> dict[str, Any]:
    """Build the complete fixed-point evaluation identity."""
    return {
        "test_inference_identity": _inference_identity(test_inference),
        "point_threshold": float(runtime["point_threshold"]),
        "adf_enabled": bool(runtime["adf_enabled"]),
        "adf_density_threshold": float(runtime["adf_density_threshold"]),
        "adf_window_size": int(runtime["adf_window_size"]),
        "adf_window_step": int(runtime["adf_window_step"]),
        "updates_root": str(runtime["updates_root"].resolve()),
        "updates_manifest": _directory_manifest(runtime["updates_root"]),
        "gt_msg_root": str(runtime["gt_msg_root"].resolve()),
        "gt_manifest": _directory_manifest(runtime["gt_msg_root"]),
        "event_csv": str(runtime["event_csv"].resolve()),
        "event_csv_sha256": _file_sha256(runtime["event_csv"]),
    }


def measure_fixed_test_detection(config_path: str | Path, classifier: str = "lightgbm") -> None:
    """Measure fixed threshold, ADF, and in-memory alert construction on test data.

    Probability LMDB reading and decompression intentionally occur outside the
    measured block because they only support research-stage cache reuse.
    """
    global CONFIG_ROOT
    config_path = Path(config_path).resolve()
    CONFIG_ROOT = config_path.parent
    config = load_config(config_path)
    args = argparse.Namespace(output_root=None, max_processes=None)
    runtime = resolve_runtime(args, config)
    inference_path = inference_lmdb_path(config, classifier, runtime["test_lmdb"])
    if not inference_path.exists():
        raise FileNotFoundError(f"Test inference LMDB not found: {inference_path}")

    metadata = load_inference_metadata(inference_path)
    event_names = metadata["event_names"]
    event_indices = {
        event_name: build_event_timestamp_index(
            event_name, metadata["event_timestamps"][event_name]
        )
        for event_name in event_names
    }
    traffic_seconds = sum(
        len(metadata["event_timestamps"][event_name]) * 60
        for event_name in event_names
    )
    recorder = get_active_recorder()
    if recorder is not None:
        recorder.set_summary(
            test_event_count=len(event_names),
            test_traffic_duration_seconds=int(traffic_seconds),
            point_threshold=float(runtime["point_threshold"]),
            adf_enabled=bool(runtime["adf_enabled"]),
            adf_density_threshold=float(runtime["adf_density_threshold"]),
            adf_window_size=int(runtime["adf_window_size"]),
            adf_window_step=int(runtime["adf_window_step"]),
        )

    keys = load_data_keys(inference_path)
    decompressor = zstd.ZstdDecompressor()
    env = lmdb.open(str(inference_path), readonly=True, lock=False, readahead=False, max_readers=64)
    retained_alert_count = 0
    raw_alert_count = 0
    resource_context = (
        recorder.track_resources("inference.threshold_adf_alert.test")
        if recorder is not None
        else nullcontext()
    )
    try:
        with resource_context, env.begin(write=False) as txn:
            for key in tqdm(keys, desc="Measure fixed Test ADF"):
                event_name, triplet = parse_radbench_key(key)
                prob, _labels, timestamps = read_probability_payload(txn, decompressor, key)
                compute_started = time.perf_counter()
                if runtime["adf_enabled"]:
                    candidate_indices, retained_indices, minute_timestamps = apply_fixed_adf(
                        prob,
                        timestamps,
                        event_timestamp_index=event_indices[event_name],
                        sequence_length=len(metadata["event_timestamps"][event_name]),
                        point_threshold=float(runtime["point_threshold"]),
                        density_threshold=float(runtime["adf_density_threshold"]),
                        window_size=int(runtime["adf_window_size"]),
                        window_step=int(runtime["adf_window_step"]),
                    )
                else:
                    minute_timestamps = np.asarray(
                        [normalize_minute_timestamp(int(value)) for value in timestamps],
                        dtype=np.int64,
                    )
                    candidate_indices = np.nonzero(
                        prob >= float(runtime["point_threshold"])
                    )[0]
                    retained_indices = candidate_indices
                raw_alert_count += int(candidate_indices.size)
                alert_records = [
                    (event_name, triplet, int(minute_timestamps[index]))
                    for index in retained_indices
                ]
                retained_alert_count += len(alert_records)
                if recorder is not None:
                    recorder.add_elapsed(
                        "inference.threshold_adf_alert.test",
                        time.perf_counter() - compute_started,
                    )
    finally:
        env.close()

    if recorder is not None:
        recorder.add_count("inference.threshold_adf_alert.test", "triplets", len(keys))
        recorder.add_count(
            "inference.threshold_adf_alert.test", "raw_alerts", raw_alert_count
        )
        recorder.add_count(
            "inference.threshold_adf_alert.test", "retained_alerts", retained_alert_count
        )


def evaluate_classifier(config: dict[str, Any], runtime: dict[str, Any], classifier: str) -> None:
    """Evaluate one classifier at the configured fixed operating point."""
    input_source = str(config["ClassifierModel"]["input_source"])
    out_dir = classifier_eval_output_dir(config, classifier, runtime["output_root"])
    metrics_csv_path = out_dir / "metrics.csv"

    test_inference = inference_lmdb_path(config, classifier, runtime["test_lmdb"])
    if not test_inference.exists():
        raise FileNotFoundError(
            f"Test inference LMDB not found for classifier={classifier}: {test_inference}"
        )

    identity = evaluation_identity(runtime, test_inference)
    run_metadata_path = out_dir / "run_metadata.json"
    if metrics_csv_path.exists():
        if not run_metadata_path.is_file():
            raise FileNotFoundError(f"Evaluation metadata not found: {run_metadata_path}")
        actual = json.loads(run_metadata_path.read_text(encoding="utf-8")).get("identity")
        if actual != identity:
            raise ValueError(
                "Evaluation identity mismatch. Remove the stale output explicitly "
                f"before rerunning: {out_dir}"
            )
        log.warning("Evaluation results already exist; skip evaluation and reuse output: %s", out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_path.write_text(
        json.dumps({"identity": identity}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    thresholds = np.asarray([runtime["point_threshold"]], dtype=np.float64)
    cases = build_eval_cases(runtime["point_threshold"], runtime)
    if len(cases) != 1:
        raise ValueError(f"Fixed evaluation must build exactly one case, got {len(cases)}")

    all_rows: list[dict[str, Any]] = []
    for dataset_name, inference_path in [
        ("Test", test_inference),
    ]:
        state = analyze_inference_dataset(
            dataset_name=dataset_name,
            classifier=classifier,
            inference_path=inference_path,
            thresholds=thresholds,
            runtime=runtime,
            cases=cases,
        )
        plot_cdf_by_event(
            state["cdf_samples"],
            dataset_name,
            out_dir / "cdf" / f"{dataset_name.lower()}_cdf_by_event.png",
        )
        message_results = evaluate_dataset_messages(
            state,
            runtime["updates_root"],
            runtime["gt_msg_root"],
            runtime["max_processes"],
            1,
            runtime["event_csv"],
        )
        all_rows.extend(
            build_metric_rows(
                classifier=classifier,
                cases=cases,
                state=state,
                message_results=message_results,
            )
        )

    all_rows = sort_metric_rows(all_rows)
    write_csv_rows(metrics_csv_path, all_rows, METRIC_FIELDNAMES)
    log.info("Saved fixed-point metrics CSV: %s", metrics_csv_path)
    write_fixed_summary(
        rows=all_rows,
        output_path=out_dir / "summary" / "fixed_point_summary.txt",
        metrics_csv_path=metrics_csv_path,
        classifier=classifier,
        input_source=input_source,
        runtime=runtime,
    )


def run() -> None:
    """Run fixed-point evaluation for the selected classifiers."""
    args = parse_args()
    global CONFIG_ROOT
    config_path = resolve_repo_path(args.config)
    CONFIG_ROOT = config_path.parent
    config = load_config(config_path)
    runtime = resolve_runtime(args, config)
    classifiers = selected_classifiers(config, args.clf_name)
    log.info(
        "Fixed-point evaluation started: classifiers=%s point_threshold=%.10g "
        "max_processes=%d adf_enabled=%s adf_density_threshold=%.10g",
        classifiers,
        runtime["point_threshold"],
        runtime["max_processes"],
        runtime["adf_enabled"],
        runtime["adf_density_threshold"],
    )
    for classifier in classifiers:
        evaluate_classifier(config, runtime, classifier)
    log.info("Fixed-point evaluation finished: %s", runtime["output_root"])


if __name__ == '__main__':
    run()
