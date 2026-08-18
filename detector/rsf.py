"""Train and apply the Residual-RSF filter documented in ``rsf/README.md``.

Inputs are observed-only residual cache entries and source LMDB feature masks.
Outputs are triplet keep decisions plus separately saved StandardScaler and
LogisticRegression checkpoints. Checkpoint identity binds the unsupervised
checkpoint content, training/validation LMDB paths, and RSF configuration.
"""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lmdb
import numpy as np
import zstandard as zstd

from common.performance import get_active_recorder, measure_if_active
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RSF_FEATURE_COLUMNS = [
    "mask_active_ratio", "mask_active_span_ratio", "mask_longest_run_ratio",
    "residual_densest_window_ratio", "residual_effective_support_ratio",
    "mask_dominant_freq_ratio", "residual_dominant_freq_ratio",
    "residual_roughness", "residual_sign_change_rate", "residual_extrema_density",
]
RSF_VERSION = 1
EPSILON = 1.0e-8


def _read_pickle(txn: lmdb.Transaction, key: str) -> Any:
    raw = txn.get(key.encode("ascii"))
    if raw is None:
        raise KeyError(f"LMDB missing metadata key: {key}")
    return pickle.loads(raw)


def _unpack_payload(decompressor: zstd.ZstdDecompressor, raw: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(decompressor.decompress(raw)), allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _event_name(key: str) -> str:
    parts = key.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid RADBench key: {key}")
    return parts[0]


def _longest_true_run(values: np.ndarray) -> int:
    edges = np.diff(np.concatenate(([False], values, [False])).astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def _dominant_frequency_ratio(values: np.ndarray) -> float:
    non_dc = np.square(np.abs(np.fft.rfft(values)))[1:]
    total = float(non_dc.sum())
    return float(non_dc.max() / total) if total > 0.0 else 0.0


def extract_rsf_features(
    cached_residual: np.ndarray,
    time_index: np.ndarray,
    source_feature_mask: np.ndarray,
    *,
    reconstructed_length: int,
    window_size: int,
) -> np.ndarray | None:
    """Extract the fixed-order 10-dimensional feature vector for one triplet.

    Args:
        cached_residual: Observed-only signed residual, shape ``(N, D)``.
        time_index: Strictly increasing indices, shape ``(N,)``.
        source_feature_mask: Full-event feature mask, shape ``(T, D)``.
        reconstructed_length: Dynamic timeline length covered by complete
            unsupervised reconstruction windows.
        window_size: Residual concentration window length.

    Returns:
        Float64 vector of shape ``(10,)``; ``None`` means prefilter rejection.
    """
    mask = np.asarray(source_feature_mask, dtype=bool)
    residual = np.asarray(cached_residual, dtype=np.float64)
    indices = np.asarray(time_index, dtype=np.int64)
    if mask.ndim != 2 or residual.ndim != 2 or residual.shape[1] != mask.shape[1]:
        raise ValueError(f"Invalid RSF shapes: residual={residual.shape}, mask={mask.shape}")
    length = int(reconstructed_length)
    if length <= 0 or length > mask.shape[0]:
        raise ValueError(
            f"Invalid reconstructed_length={length} for source timeline={mask.shape[0]}"
        )
    if length < window_size:
        raise ValueError(f"RSF timeline length={length}, requires window_size={window_size}")
    mask = mask[:length]
    if indices.size and (np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= length):
        raise ValueError("RSF time_index must be strictly increasing and inside reconstructed timeline")
    if residual.shape[0] != indices.size:
        raise ValueError("RSF residual row count does not match time_index")
    observed = mask.any(axis=1)
    if not np.any(observed):
        return None
    if indices.size and not np.all(observed[indices]):
        raise ValueError("Residual time_index contains minutes without active source features")

    residual_full = np.zeros(mask.shape, dtype=np.float64)
    residual_full[indices] = residual
    count = mask.sum(axis=1)
    mse = np.zeros(length, dtype=np.float64)
    mse[observed] = np.where(mask[observed], np.square(residual_full[observed]), 0.0).sum(axis=1) / count[observed]
    intensity = np.zeros(length, dtype=np.float64)
    intensity[observed] = np.log1p(mse[observed])
    active_intensity = intensity[observed]
    median = float(np.median(active_intensity))
    q25, q75 = np.percentile(active_intensity, [25.0, 75.0])
    scale = max(float((q75 - q25) / 1.349), float(np.std(active_intensity)), EPSILON)
    normalized = np.zeros(length, dtype=np.float64)
    normalized[observed] = (active_intensity - median) / scale
    positive = np.maximum(normalized, 0.0)
    positive_sum = float(positive.sum())

    active_ratio = float(observed.mean())
    active_indices = np.flatnonzero(observed)
    active_span_ratio = float((active_indices[-1] - active_indices[0] + 1) / length)
    longest_run_ratio = float(_longest_true_run(observed) / length)
    if positive_sum > 0.0:
        window_sum = np.convolve(positive, np.ones(window_size), mode="valid")
        densest_window_ratio = float(window_sum.max() / (positive_sum + EPSILON))
        effective_support_ratio = float(positive_sum**2 / (float(np.square(positive).sum()) + EPSILON) / length)
    else:
        densest_window_ratio = effective_support_ratio = 0.0

    difference = np.diff(normalized)
    adjacent = observed[:-1] & observed[1:]
    roughness = float(np.abs(difference[adjacent]).mean()) if np.any(adjacent) else 0.0
    triple = observed[:-2] & observed[1:-1] & observed[2:]
    left_difference, right_difference = difference[:-1], difference[1:]
    valid_direction = triple & (left_difference != 0) & (right_difference != 0)
    flip_count = int((triple & (left_difference * right_difference < 0)).sum())
    features = np.asarray([
        active_ratio, active_span_ratio, longest_run_ratio, densest_window_ratio,
        effective_support_ratio,
        _dominant_frequency_ratio(observed.astype(np.float64) - active_ratio),
        _dominant_frequency_ratio(positive), roughness,
        float(flip_count / max(int(valid_direction.sum()), 1)),
        float(flip_count / max(int(triple.sum()), 1)),
    ])
    if not np.all(np.isfinite(features)):
        raise ValueError("RSF features contain nan or inf")
    return features


def reconstruction_coverage_length(event_length: int, win_size: int, step: int) -> int:
    """Return the prefix length covered by the cache's complete windows.

    Args:
        event_length: Actual event timeline length in minutes.
        win_size: Unsupervised reconstruction window length.
        step: Window start interval used while generating the residual cache.

    Returns:
        End index of the last complete reconstruction window, or zero when no
        complete window can be generated.
    """
    if event_length < win_size:
        return 0
    last_start = ((event_length - win_size) // step) * step
    return last_start + win_size


@dataclass(frozen=True)
class ResidualShapeFilter:
    """Fitted scaler, LR classifier, and validation-derived keep threshold."""
    scaler: StandardScaler
    classifier: LogisticRegression
    threshold: float
    metadata: dict[str, Any]


def _collect_triplet_features(
    residual_cache_path: Path,
    source_lmdb_path: Path,
    *,
    feature_dim: int,
    window_size: int,
    performance_section: str | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load aligned residual/mask pairs and return features and triplet labels."""
    cache_env = lmdb.open(str(residual_cache_path), readonly=True, lock=False, readahead=False)
    source_env = lmdb.open(str(source_lmdb_path), readonly=True, lock=False, readahead=False)
    decompressor = zstd.ZstdDecompressor()
    keys: list[str] = []
    rows: list[np.ndarray] = []
    labels: list[int] = []
    valid_flags: list[bool] = []
    try:
        with cache_env.begin() as cache_txn, source_env.begin() as source_txn:
            cache_metadata = _read_pickle(cache_txn, "__metadata__")
            recon_win_size = int(cache_metadata["win_size"])
            recon_step = int(cache_metadata["step"])
            events = [str(value) for value in _read_pickle(cache_txn, "__event_names__")]
            event_lengths = {event: len(_read_pickle(cache_txn, f"__{event}_timestamp__")) for event in events}
            anomaly_triplets = _read_pickle(cache_txn, "__anomaly_triplets__")
            anomaly_keys = {f"{event}_{q[0]}_{q[1]}_{q[2]}" for event, triplets in anomaly_triplets.items() for q in triplets}
            data_keys = sorted(key.decode("ascii") for key in cache_txn.cursor().iternext(keys=True, values=False) if not key.startswith(b"__"))
            for key in tqdm(data_keys, desc=f"Extract RSF {residual_cache_path.stem}"):
                raw = cache_txn.get(key.encode("ascii"))
                if raw is None:
                    raise KeyError(f"Residual cache missing key: {key}")
                payload = _unpack_payload(decompressor, raw)
                event_length = event_lengths[_event_name(key)]
                mask_raw = source_txn.get(f"mask_{key}".encode("ascii"))
                if mask_raw is None:
                    raise KeyError(f"Source LMDB missing mask_{key}")
                mask_flat = np.frombuffer(decompressor.decompress(mask_raw), dtype=np.uint8)
                expected = event_length * feature_dim
                if mask_flat.size != expected:
                    raise ValueError(f"RSF requires T,D mask: mask_{key} size={mask_flat.size}, expected={expected}")
                if performance_section is None:
                    features = extract_rsf_features(
                        payload["residual"], payload["time_index"], mask_flat.reshape(event_length, feature_dim),
                        reconstructed_length=reconstruction_coverage_length(
                            event_length, recon_win_size, recon_step
                        ),
                        window_size=window_size,
                    )
                else:
                    with measure_if_active(performance_section):
                        features = extract_rsf_features(
                            payload["residual"], payload["time_index"], mask_flat.reshape(event_length, feature_dim),
                            reconstructed_length=reconstruction_coverage_length(
                                event_length, recon_win_size, recon_step
                            ),
                            window_size=window_size,
                        )
                payload_label, metadata_label = int(np.any(payload["label"] == 1)), int(key in anomaly_keys)
                if payload_label != metadata_label:
                    raise ValueError(f"Triplet label mismatch for {key}")
                keys.append(key)
                rows.append(features if features is not None else np.zeros(len(RSF_FEATURE_COLUMNS)))
                labels.append(metadata_label)
                valid_flags.append(features is not None)
    finally:
        cache_env.close()
        source_env.close()
    return keys, np.asarray(rows), np.asarray(labels, dtype=np.int8), np.asarray(valid_flags, dtype=bool)


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def _lmdb_identity(path: Path) -> dict[str, Any]:
    """Return source LMDB version identity without hashing the full data file."""
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


def _residual_identity(path: Path) -> Any:
    """Read and require the reconstruction identity from a residual cache."""
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            raw = txn.get(b"__metadata__")
            if raw is None:
                raise KeyError(f"Residual cache metadata not found: {path}")
            metadata = pickle.loads(raw)
    finally:
        env.close()
    identity = metadata.get("identity")
    if identity is None:
        raise ValueError(f"Residual cache metadata lacks identity: {path}")
    return identity


def _identity(
    unsup_checkpoint: Path,
    train_cache_path: Path,
    train_lmdb_path: Path,
    val_cache_path: Path,
    val_lmdb_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": RSF_VERSION,
        "unsup_checkpoint": _file_identity(unsup_checkpoint),
        "train_residual_identity": _residual_identity(train_cache_path),
        "val_residual_identity": _residual_identity(val_cache_path),
        "train_lmdb": _lmdb_identity(train_lmdb_path),
        "val_lmdb": _lmdb_identity(val_lmdb_path),
        "feature_dim": int(config["feature_dim"]),
        "window_size": int(config.get("window_size", 100)),
        "lr": config.get("lr", {}),
        "feature_columns": RSF_FEATURE_COLUMNS,
    }


def rsf_checkpoint_dir(
    unsup_checkpoint: Path,
    train_lmdb_path: Path,
    val_lmdb_path: Path,
    checkpoint_root: Path,
    config: dict[str, Any],
    train_cache_path: Path | None = None,
    val_cache_path: Path | None = None,
) -> Path:
    """Resolve the identity-tagged RSF checkpoint directory."""
    if train_cache_path is None or val_cache_path is None:
        raise ValueError("RSF checkpoint identity requires train and val residual caches")
    identity = _identity(
        unsup_checkpoint, train_cache_path, train_lmdb_path,
        val_cache_path, val_lmdb_path, config,
    )
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return checkpoint_root / f"{unsup_checkpoint.stem}_{tag}"


def build_or_load_rsf(
    *, unsup_checkpoint: Path, train_cache_path: Path, train_lmdb_path: Path,
    val_cache_path: Path, val_lmdb_path: Path, checkpoint_root: Path,
    config: dict[str, Any],
) -> tuple[ResidualShapeFilter, Path]:
    """Train or safely reuse RSF and return it with its checkpoint directory."""
    identity = _identity(
        unsup_checkpoint, train_cache_path, train_lmdb_path,
        val_cache_path, val_lmdb_path, config,
    )
    output_dir = rsf_checkpoint_dir(
        unsup_checkpoint, train_lmdb_path, val_lmdb_path, checkpoint_root, config,
        train_cache_path=train_cache_path, val_cache_path=val_cache_path,
    )
    scaler_path, lr_path, metadata_path = output_dir / "scaler.joblib", output_dir / "lr.joblib", output_dir / "metadata.json"
    if output_dir.exists():
        if not (scaler_path.exists() and lr_path.exists() and metadata_path.exists()):
            raise FileExistsError(f"Incomplete RSF checkpoint directory: {output_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity:
            raise ValueError(f"RSF checkpoint identity mismatch: {output_dir}")
        recorder = get_active_recorder()
        if recorder is not None and recorder.entrypoint == "train_detector":
            recorder.mark_reused("training.rsf")
        return ResidualShapeFilter(joblib.load(scaler_path), joblib.load(lr_path), float(metadata["threshold"]), metadata), output_dir

    feature_dim = int(config["feature_dim"])
    window_size = int(config.get("window_size", 100))
    _, train_x, train_y, train_valid = _collect_triplet_features(train_cache_path, train_lmdb_path, feature_dim=feature_dim, window_size=window_size)
    _, val_x, val_y, val_valid = _collect_triplet_features(val_cache_path, val_lmdb_path, feature_dim=feature_dim, window_size=window_size)
    invalid_val_anomaly = int(np.sum((val_y == 1) & ~val_valid))
    if invalid_val_anomaly:
        raise ValueError(f"Validation has {invalid_val_anomaly} anomaly triplets rejected by RSF prefilter")
    fit_y = train_y[train_valid]
    if not np.any(fit_y == 0) or not np.any(fit_y == 1):
        raise ValueError("RSF training triplets must contain both classes after prefilter")
    recorder = get_active_recorder()
    measure_training = recorder is not None and recorder.entrypoint == "train_detector"
    context = measure_if_active("training.rsf") if measure_training else nullcontext()
    with context:
        scaler = StandardScaler().fit(train_x[train_valid])
    scaled_train_x = scaler.transform(train_x[train_valid])
    lr_params = {"solver": "lbfgs", "class_weight": "balanced", "C": 1.0, "max_iter": 5000, "random_state": 42, **config.get("lr", {})}
    context = measure_if_active("training.rsf") if measure_training else nullcontext()
    with context:
        classifier = LogisticRegression(**lr_params).fit(scaled_train_x, fit_y)
    val_probability = classifier.predict_proba(scaler.transform(val_x[val_y == 1]))[:, 1]
    if val_probability.size == 0:
        raise ValueError("RSF validation data has no anomaly triplets")
    threshold = float(val_probability.min())
    metadata = {
        "identity": identity, "threshold": threshold,
        "threshold_definition": "minimum validation anomaly keep probability",
        "train_triplet_count": int(train_y.size), "train_prefilter_count": int(train_valid.sum()),
        "validation_triplet_count": int(val_y.size), "validation_anomaly_count": int((val_y == 1).sum()),
        "lr_params": lr_params,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(scaler, scaler_path)
    joblib.dump(classifier, lr_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return ResidualShapeFilter(scaler, classifier, threshold, metadata), output_dir


def collect_keep_keys(
    rsf: ResidualShapeFilter,
    residual_cache_path: Path,
    source_lmdb_path: Path,
    *,
    config: dict[str, Any],
    performance_section: str | None = None,
) -> set[str]:
    """Return triplet keys retained by a fitted RSF for one dataset."""
    keys, features, _, valid = _collect_triplet_features(
        residual_cache_path, source_lmdb_path, feature_dim=int(config["feature_dim"]),
        window_size=int(config.get("window_size", 100)),
        performance_section=performance_section,
    )
    keep = np.zeros(len(keys), dtype=bool)
    if np.any(valid):
        if performance_section is None:
            probability = rsf.classifier.predict_proba(rsf.scaler.transform(features[valid]))[:, 1]
            keep[valid] = probability >= rsf.threshold
        else:
            with measure_if_active(performance_section):
                probability = rsf.classifier.predict_proba(rsf.scaler.transform(features[valid]))[:, 1]
                keep[valid] = probability >= rsf.threshold
    selected_keys = {key for key, selected in zip(keys, keep, strict=True) if selected}
    if performance_section is not None:
        recorder = get_active_recorder()
        if recorder is not None:
            recorder.add_count(performance_section, "triplets", len(keys))
            recorder.add_count(performance_section, "retained_triplets", len(selected_keys))
    return selected_keys
