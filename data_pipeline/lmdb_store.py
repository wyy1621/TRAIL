import bisect
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import copy
import datetime
import glob
import json
import logging
import os
import time
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import beijing, RADBenchEvent, data_completion, df_fill_by_timestamp, judge_triplet_anomaly, read_RADBench_event_list, transform_nonzero_only
logging.Formatter.converter = beijing
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s', force=True)
log = logging.getLogger(__name__)



test_time = 0

import zstandard
import lmdb
import pickle


def should_persist_workers(num_workers: int) -> bool:
    """根据 worker 数决定是否启用 persistent_workers。"""
    return False


class BgpLMDBDataset(Dataset):
    def __init__(self, lmdb_file_path, win_size, step=100, allowed_keys=None):
        """
        Args:
            lmdb_file_path (str): Path to the LMDB file.
            win_size (int): Size of the sliding window.
            step (int): Step size for the sliding window.
        """
        self.lmdb_file_path = lmdb_file_path
        self.win_size = win_size
        self.step = step
        self.env = None
        
        # Open temporary LMDB environment to read metadata
        temp_env = lmdb.open(lmdb_file_path, readonly=True, lock=False)
        try:
            with temp_env.begin() as txn:
                # Read metadata
                self.timestamps = pickle.loads(txn.get(b'__timestamps__'))
                self.feature_names = pickle.loads(txn.get(b'__columns__'))
                self.pos_label_count = pickle.loads(txn.get(b'__pos_label_count__'))
                self.neg_label_count = pickle.loads(txn.get(b'__neg_label_count__'))
                
                # Count keys excluding special metadata keys
                self.keys = []
                cursor = txn.cursor()
                for key, _ in cursor:
                    # Filter out metadata keys which start and end with __
                    if not (key.startswith(b'__') and key.endswith(b'__')):
                        try:
                            self.keys.append(key.decode('ascii'))
                        except UnicodeDecodeError:
                            continue # Skip non-ascii keys if any
        finally:
            temp_env.close()
            
        self.total_timestamps_len = len(self.timestamps)
        
        # Calculate number of windows per triplet
        if self.total_timestamps_len < self.win_size:
            self.windows_per_triplet = 0
        else:
            self.windows_per_triplet = (self.total_timestamps_len - self.win_size) // self.step + 1
            
        # Filter to only the requested subset of triplet keys (e.g., for threshold search sampling)
        if allowed_keys is not None:
            log.info(f"Filtering to allowed keys: {len(allowed_keys)} keys")    
            allowed_set = set(allowed_keys)
            self.keys = [k for k in self.keys if k in allowed_set]

        self.total_samples = len(self.keys) * self.windows_per_triplet
        
        log.info(f"Initialized BgpLMDBDataset from {lmdb_file_path}")
        log.info(f"Total triplets: {len(self.keys)}")
        log.info(f"Timestamps length: {self.total_timestamps_len}")
        log.info(f"Windows per triplet: {self.windows_per_triplet}")
        log.info(f"Total samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def _init_env(self):
        self.env = lmdb.open(self.lmdb_file_path, readonly=True, lock=False)
        self.decompressor = zstandard.ZstdDecompressor()
        self._cached_triplet_key = None
        self._cached_full_arr = None
        self._cached_triplet_arr = None

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of bound")

        if self.env is None:
            self._init_env()

        # Determine which triplet and which window
        key_idx = idx // self.windows_per_triplet
        window_idx = idx % self.windows_per_triplet

        start = window_idx * self.step
        end = start + self.win_size

        triplet_key = self.keys[key_idx]

        if self._cached_triplet_key != triplet_key:
            with self.env.begin(write=False) as txn:
                val_compressed = txn.get(triplet_key.encode('ascii'))
                if val_compressed is None:
                    raise ValueError(f"Key {triplet_key} not found in LMDB")

                val_bytes = self.decompressor.decompress(val_compressed)
                self._cached_triplet_key = triplet_key
                self._cached_full_arr = np.frombuffer(val_bytes, dtype=np.float32).reshape(self.total_timestamps_len, -1)
                self._cached_triplet_arr = np.array([int(x) for x in triplet_key.split('_')], dtype=np.int32)

        full_arr = self._cached_full_arr
        data_window = full_arr[start:end, :-1].copy()
        label_window = full_arr[start:end, -1].copy()
        timestamp_window = self.timestamps[start:end]

        return self._cached_triplet_arr.copy(), np.array(timestamp_window), torch.from_numpy(data_window), torch.from_numpy(label_window)

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


class BgpLMDBDatasetCropped(Dataset):
    """从 ``convert_bgp_lmdb_to_cropped`` 生成的裁剪 LMDB 读取数据集。

    每条记录是一个预先截取好的滑窗片段，仅保留含异常点的窗口。
    每条记录都有各自的时间戳和特征+标签数据，且所有窗口大小相同（win_size）。

    Key 格式::

        {A}_{B}_{C}_{window_idx}   例如 ``12345_67890_11111_0``

    Value 存储格式（经 zstd 压缩）::

        [  int64 timestamps  (win_size,)  ]
        [  float32 data+label (win_size, num_features+1)  ]

    元数据键: ``__columns__``, ``__win_size__``, ``__step__``,
    ``__pos_label_count__``, ``__neg_label_count__``

    ``__getitem__`` 返回::

        triplet_arr       np.ndarray  (3,)  int32     — ASN 三元组
        timestamps        np.ndarray  (win_size,) int64  — Unix 时间戳
        data              torch.Tensor (win_size, num_features) float32
        label             torch.Tensor (win_size,) float32
    """

    def __init__(self, lmdb_file_path: str):
        self.lmdb_file_path = lmdb_file_path
        self.env = None

        temp_env = lmdb.open(lmdb_file_path, readonly=True, lock=False)
        try:
            with temp_env.begin() as txn:
                self.feature_names = pickle.loads(txn.get(b'__columns__'))
                self.win_size = pickle.loads(txn.get(b'__win_size__'))
                self.step = pickle.loads(txn.get(b'__step__'))
                pos_raw = txn.get(b'__pos_label_count__')
                neg_raw = txn.get(b'__neg_label_count__')
                self.pos_label_count = pickle.loads(pos_raw) if pos_raw is not None else 0
                self.neg_label_count = pickle.loads(neg_raw) if neg_raw is not None else 0

                self.keys = []
                cursor = txn.cursor()
                for key_bytes, _ in cursor:
                    if key_bytes.startswith(b'__') and key_bytes.endswith(b'__'):
                        continue
                    try:
                        self.keys.append(key_bytes.decode('ascii'))
                    except UnicodeDecodeError:
                        continue
        finally:
            temp_env.close()

        # feature_names 首列为 timestamp，其余为特征列（最后一列 label 存于 float32 数组末位）
        self.num_features = len(self.feature_names)
        self.total_samples = len(self.keys)

        log.info(f"Initialized BgpLMDBDatasetCropped from {lmdb_file_path}")
        log.info(f"Total samples: {self.total_samples}, win_size={self.win_size}, step={self.step}")
        log.info(f"Pos labels: {self.pos_label_count}, Neg labels: {self.neg_label_count}")

    def __len__(self):
        return self.total_samples

    def _init_env(self):
        self.env = lmdb.open(self.lmdb_file_path, readonly=True, lock=False)
        self.decompressor = zstandard.ZstdDecompressor()
        self._cached_key = None
        self._cached_full_arr = None
        self._cached_triplet_arr = None

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of bound")

        if self.env is None:
            self._init_env()

        key_str = self.keys[idx]

        if self._cached_key != key_str:
            with self.env.begin(write=False) as txn:
                val_compressed = txn.get(key_str.encode('ascii'))
                if val_compressed is None:
                    raise ValueError(f"Key {key_str} not found in LMDB")

                val_bytes = self.decompressor.decompress(val_compressed)

                # 解包合并矩阵: (win_size, 1 + num_features + 1) float64
                # 列布局: [timestamp | features... | label]
                num_cols = len(self.feature_names)
                self._cached_key = key_str
                self._cached_full_arr = np.frombuffer(val_bytes, dtype=np.float64).reshape(self.win_size, num_cols)

                # 从 key 解析三元组: "A_B_C_N" -> rsplit 取最后一个 '_' 分隔符
                triplet_str = key_str.rsplit('_', 1)[0]   # "A_B_C"
                self._cached_triplet_arr = np.array([int(x) for x in triplet_str.split('_')], dtype=np.int32)

        full_arr = self._cached_full_arr
        timestamps = full_arr[:, 0].astype(np.int64)            # (win_size,)
        data_window = full_arr[:, 1:-1].astype(np.float32)      # (win_size, num_features)
        label_window = full_arr[:, -1].astype(np.float32)       # (win_size,)

        return (
            self._cached_triplet_arr.copy(),
            timestamps,
            torch.from_numpy(data_window),
            torch.from_numpy(label_window),
        )

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


class RADBenchLMDBDataset(Dataset):
    def __init__(self, lmdb_file_path, win_size, step=100, allowed_keys=None, return_mask=False):
        """RADBench LMDB reader for event-prefixed keys.

        Expected metadata keys:
        - __seq_len__
        - __columns__
        - __event_names__
        - __{event_name}_timestamp__ (per event)
        - __pos_label_count__, __neg_label_count__

        Data keys format:
        - {event_name}_A_B_C
        - mask_{event_name}_A_B_C

        Args:
            lmdb_file_path: LMDB 路径。
            win_size: 滑动窗口长度。
            step: 滑动窗口步长。
            allowed_keys: optional collection of key strings to restrict the dataset to.
            return_mask: 是否在返回值末尾追加 feature-level nonzero mask。
        """
        self.lmdb_file_path = lmdb_file_path
        self.win_size = win_size
        self.step = step
        self.env = None
        self._allowed_keys = set(allowed_keys) if allowed_keys is not None else None
        self.return_mask = return_mask

        self.seq_len = 0
        self.feature_names = []
        self.pos_label_count = 0
        self.neg_label_count = 0
        self.event_names = []
        self.event_timestamps = {}
        self.event_triplets = {}
        self.event_windows_per_triplet = {}
        self.active_event_names = []
        self.event_sample_offsets = []
        self.mask_type = None
        self.mask_shape = None
        self.mask_columns = []

        temp_env = lmdb.open(lmdb_file_path, readonly=True, lock=False)
        try:
            with temp_env.begin() as txn:
                seq_len_raw = txn.get(b'__seq_len__')
                if seq_len_raw is not None:
                    self.seq_len = pickle.loads(seq_len_raw)

                cols_raw = txn.get(b'__columns__')
                if cols_raw is not None:
                    self.feature_names = pickle.loads(cols_raw)

                pos_raw = txn.get(b'__pos_label_count__')
                neg_raw = txn.get(b'__neg_label_count__')
                if pos_raw is not None:
                    self.pos_label_count = pickle.loads(pos_raw)
                if neg_raw is not None:
                    self.neg_label_count = pickle.loads(neg_raw)

                mask_type_raw = txn.get(b'__mask_type__')
                mask_shape_raw = txn.get(b'__mask_shape__')
                mask_columns_raw = txn.get(b'__mask_columns__')
                if mask_type_raw is not None:
                    self.mask_type = pickle.loads(mask_type_raw)
                if mask_shape_raw is not None:
                    self.mask_shape = pickle.loads(mask_shape_raw)
                if mask_columns_raw is not None:
                    self.mask_columns = pickle.loads(mask_columns_raw)
                if self.return_mask and self.mask_shape not in {None, 'T', 'T,D'}:
                    raise ValueError(
                        f'RADBench no-sparse return_mask=True supports legacy T mask or feature-level T,D mask, '
                        f'got mask_type={self.mask_type}, mask_shape={self.mask_shape}'
                    )

                event_names_raw = txn.get(b'__event_names__')
                if event_names_raw is not None:
                    self.event_names = pickle.loads(event_names_raw)

                # Load per-event timestamps from metadata if available.
                for event_name in self.event_names:
                    key = f'__{event_name}_timestamp__'.encode('ascii')
                    ts_raw = txn.get(key)
                    if ts_raw is not None:
                        self.event_timestamps[event_name] = pickle.loads(ts_raw)

                triplet_buffer = defaultdict(list)

                # Scan data keys once and only keep compact triplet metadata.
                cursor = txn.cursor()
                for key, _ in cursor:
                    if key.startswith(b'__') and key.endswith(b'__'):
                        continue

                    try:
                        key_str = key.decode('ascii')
                    except UnicodeDecodeError:
                        continue

                    if key_str.startswith('mask_'):
                        continue

                    if self._allowed_keys is not None and key_str not in self._allowed_keys:
                        continue

                    parts = key_str.rsplit('_', 3)
                    if len(parts) != 4:
                        continue

                    event_name = parts[0]
                    triplet_buffer[event_name].append((
                        int(parts[1]),
                        int(parts[2]),
                        int(parts[3]),
                    ))

                if not self.event_names:
                    self.event_names = sorted(list(triplet_buffer.keys()))

                total_samples = 0
                for event_name in self.event_names:
                    timestamps = self.event_timestamps.get(event_name)
                    if timestamps is None:
                        continue

                    triplets = triplet_buffer.get(event_name)
                    if not triplets:
                        continue

                    if len(timestamps) < self.win_size:
                        windows_per_triplet = 0
                    else:
                        windows_per_triplet = (len(timestamps) - self.win_size) // self.step + 1

                    if windows_per_triplet == 0:
                        continue

                    self.active_event_names.append(event_name)
                    self.event_sample_offsets.append(total_samples)
                    self.event_triplets[event_name] = np.asarray(triplets, dtype=np.int32)
                    self.event_windows_per_triplet[event_name] = windows_per_triplet
                    total_samples += len(triplets) * windows_per_triplet
        finally:
            temp_env.close()

        self.total_samples = total_samples

        log.info(f"Initialized RADBenchLMDBDataset from {lmdb_file_path}")
        log.info(f"Events: {len(self.active_event_names)}, Total samples: {self.total_samples}")
        log.info(f"Feature columns: {len(self.feature_names)}, seq_len: {self.seq_len}")

    def __len__(self):
        return self.total_samples

    def get_triplet_keys(self) -> list[str]:
        """返回当前 dataset 实际可访问的 RADBench triplet full keys。"""
        keys: list[str] = []
        for event_name in self.active_event_names:
            for triplet_arr in self.event_triplets[event_name]:
                keys.append(f'{event_name}_{triplet_arr[0]}_{triplet_arr[1]}_{triplet_arr[2]}')
        return keys

    def _init_env(self):
        self.env = lmdb.open(self.lmdb_file_path, readonly=True, lock=False)
        self.decompressor = zstandard.ZstdDecompressor()
        self._cached_key = None
        self._cached_full_arr = None
        self._cached_full_mask = None
        self._cached_triplet_arr = None

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError('Index out of bound')

        if self.env is None:
            self._init_env()

        event_pos = bisect.bisect_right(self.event_sample_offsets, idx) - 1
        if event_pos < 0:
            raise IndexError('Index out of bound')

        event_name = self.active_event_names[event_pos]
        event_start = self.event_sample_offsets[event_pos]
        local_idx = idx - event_start
        windows_per_triplet = self.event_windows_per_triplet[event_name]
        triplet_idx = local_idx // windows_per_triplet
        window_idx = local_idx % windows_per_triplet

        timestamps = self.event_timestamps[event_name]
        triplet_arr = self.event_triplets[event_name][triplet_idx]

        start = window_idx * self.step
        end = start + self.win_size
        key_str = f'{event_name}_{triplet_arr[0]}_{triplet_arr[1]}_{triplet_arr[2]}'

        if self._cached_key != key_str:
            with self.env.begin(write=False) as txn:
                val_compressed = txn.get(key_str.encode('ascii'))
                if val_compressed is None:
                    raise ValueError(f'Key {key_str} not found in LMDB')

                val_bytes = self.decompressor.decompress(val_compressed)
                self._cached_key = key_str
                self._cached_full_arr = np.frombuffer(val_bytes, dtype=np.float32).reshape(len(timestamps), -1)
                self._cached_triplet_arr = triplet_arr.copy()

                if self.return_mask:
                    mask_key = f'mask_{key_str}'.encode('ascii')
                    mask_compressed = txn.get(mask_key)
                    if mask_compressed is None:
                        raise ValueError(f'Mask key {mask_key.decode("ascii")} not found in LMDB')
                    mask_bytes = self.decompressor.decompress(mask_compressed)
                    feature_dim = self._cached_full_arr.shape[1] - 1
                    mask_flat = np.frombuffer(mask_bytes, dtype=np.uint8)
                    feature_mask_size = len(timestamps) * feature_dim
                    time_mask_size = len(timestamps)
                    if mask_flat.size == feature_mask_size:
                        self._cached_full_mask = mask_flat.reshape(len(timestamps), feature_dim)
                    elif mask_flat.size == time_mask_size:
                        time_mask = mask_flat.reshape(len(timestamps), 1)
                        self._cached_full_mask = np.repeat(time_mask, feature_dim, axis=1)
                    else:
                        raise ValueError(
                            f'Mask key {mask_key.decode("ascii")} size={mask_flat.size}, '
                            f'expected {feature_mask_size} for feature mask shape=({len(timestamps)}, {feature_dim}) '
                            f'or {time_mask_size} for legacy time mask shape=({len(timestamps)},)'
                        )

        full_arr = self._cached_full_arr
        data_window = full_arr[start:end, :-1].copy()
        label_window = full_arr[start:end, -1].copy()
        timestamp_window = timestamps[start:end]

        item = (
            event_name,
            self._cached_triplet_arr.copy(),
            np.array(timestamp_window),
            torch.from_numpy(data_window),
            torch.from_numpy(label_window),
        )
        if not self.return_mask:
            return item

        mask_window = self._cached_full_mask[start:end].copy()
        return item + (torch.from_numpy(mask_window),)

    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()

