import ctypes
import ctypes.util
import datetime
import logging
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import os
import pickle

import duckdb
from duckdb_appender import Appender
import pytz
import lmdb
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

def beijing(sec, what):
    beijing_time = datetime.datetime.now(ZoneInfo('Asia/Shanghai')) # 返回北京时间
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s', force=True)
log = logging.getLogger(__name__)


class _NVMLUtilization(ctypes.Structure):
    """NVML utilization.gpu / utilization.memory 返回结构。"""

    _fields_ = [
        ('gpu', ctypes.c_uint),
        ('memory', ctypes.c_uint),
    ]


class _NVMLMemoryInfo(ctypes.Structure):
    """NVML 显存信息返回结构，单位为 bytes。"""

    _fields_ = [
        ('total', ctypes.c_ulonglong),
        ('free', ctypes.c_ulonglong),
        ('used', ctypes.c_ulonglong),
    ]


class GPUAutoSelector:
    """在脚本启动前自动选择最空闲单卡，并更新 CUDA_VISIBLE_DEVICES。"""

    _has_run = False

    def __init__(
        self,
        enabled: bool = True,
        sample_count: int = 10,
        sample_interval_seconds: float = 0.2,
        max_avg_utilization: float = 30.0,
        min_free_memory_gb: float = 10.0,
        prompt_on_failure: bool = True,
    ) -> None:
        """初始化 GPU 自动选卡器。

        Args:
            enabled: 是否启用自动选卡
            sample_count: 采样次数
            sample_interval_seconds: 采样间隔秒数
            max_avg_utilization: 候选卡平均占用率上限，单位百分比
            min_free_memory_gb: 候选卡当前空闲显存下限，单位 GiB
            prompt_on_failure: 无候选卡时是否提示用户手动选择
        """
        if sample_count <= 0:
            raise ValueError('sample_count must be greater than 0')
        if sample_interval_seconds <= 0:
            raise ValueError('sample_interval_seconds must be greater than 0')

        self.enabled = enabled
        self.sample_count = sample_count
        self.sample_interval_seconds = sample_interval_seconds
        self.max_avg_utilization = max_avg_utilization
        self.min_free_memory_gb = min_free_memory_gb
        self.prompt_on_failure = prompt_on_failure

    @staticmethod
    def _check_nvml_result(nvml: ctypes.CDLL, result: int, function_name: str) -> None:
        """检查 NVML 调用结果，失败时抛出包含原始 NVML 错误信息的异常。

        Args:
            nvml: 已加载的 NVML 动态库
            result: NVML 函数返回码
            function_name: 发生错误的 NVML 函数名
        """
        if result == 0:
            return

        nvml.nvmlErrorString.restype = ctypes.c_char_p
        error_text = nvml.nvmlErrorString(result)
        decoded_error = error_text.decode('utf-8', errors='replace') if error_text else f'error_code={result}'
        raise RuntimeError(f'{function_name} failed: {decoded_error}')

    def _load_nvml(self, library_path: str) -> ctypes.CDLL:
        """加载 NVML 动态库并配置本类用到的函数签名。

        Args:
            library_path: ctypes 解析到的 NVML 动态库路径

        Returns:
            配置好函数签名的 NVML 动态库对象
        """
        nvml = ctypes.CDLL(library_path)

        nvml.nvmlInit_v2.restype = ctypes.c_int
        nvml.nvmlShutdown.restype = ctypes.c_int

        nvml.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        nvml.nvmlDeviceGetCount_v2.restype = ctypes.c_int

        nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int

        nvml.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NVMLUtilization)]
        nvml.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int

        nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NVMLMemoryInfo)]
        nvml.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int

        return nvml

    def _query_gpu_snapshot(self) -> list[dict[str, float | str]]:
        """查询当前每张 GPU 的占用率与空闲显存。

        Returns:
            每张 GPU 的索引、当前利用率和空闲显存
        """
        command = [
            'nvidia-smi',
            '--query-gpu=index,utilization.gpu,memory.free',
            '--format=csv,noheader,nounits',
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        snapshot: list[dict[str, float | str]] = []
        for raw_line in result.stdout.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 3:
                raise ValueError(f'Unexpected nvidia-smi output line: {line}')
            snapshot.append(
                {
                    'index': parts[0],
                    'utilization': float(parts[1]),
                    'free_memory_gb': float(parts[2]) / 1024.0,
                }
            )
        return snapshot

    def _collect_stats(self) -> list[dict[str, float | str]]:
        """按固定采样次数统计每张 GPU 的平均占用和空闲显存。"""
        nvml_library = ctypes.util.find_library('nvidia-ml')
        if nvml_library is not None:
            return self._collect_stats_with_nvml(nvml_library)

        if shutil.which('nvidia-smi') is None:
            raise RuntimeError('neither NVML nor nvidia-smi is available')

        log.warning('NVML library is not available. Fall back to nvidia-smi subprocess sampling.')
        return self._collect_stats_with_nvidia_smi()

    def _collect_stats_with_nvidia_smi(self) -> list[dict[str, float | str]]:
        """通过重复调用 nvidia-smi 采样 GPU 状态。

        Returns:
            每张 GPU 的平均占用率、最新空闲显存和平均空闲显存
        """
        utilization_samples: dict[str, list[float]] = defaultdict(list)
        free_memory_samples: dict[str, list[float]] = defaultdict(list)

        for sample_idx in range(self.sample_count):
            for gpu_info in self._query_gpu_snapshot():
                gpu_index = str(gpu_info['index'])
                utilization_samples[gpu_index].append(float(gpu_info['utilization']))
                free_memory_samples[gpu_index].append(float(gpu_info['free_memory_gb']))
            if sample_idx + 1 < self.sample_count:
                time.sleep(self.sample_interval_seconds)

        return self._build_stats(utilization_samples, free_memory_samples)

    def _collect_stats_with_nvml(self, library_path: str) -> list[dict[str, float | str]]:
        """通过 NVML in-process API 采样 GPU 状态，避免重复启动 nvidia-smi。

        Args:
            library_path: NVML 动态库路径

        Returns:
            每张 GPU 的平均占用率、最新空闲显存和平均空闲显存
        """
        nvml = self._load_nvml(library_path)
        initialized = False
        utilization_samples: dict[str, list[float]] = defaultdict(list)
        free_memory_samples: dict[str, list[float]] = defaultdict(list)

        try:
            self._check_nvml_result(nvml, nvml.nvmlInit_v2(), 'nvmlInit_v2')
            initialized = True

            device_count = ctypes.c_uint()
            self._check_nvml_result(
                nvml,
                nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count)),
                'nvmlDeviceGetCount_v2',
            )

            handles: list[tuple[str, ctypes.c_void_p]] = []
            for gpu_index in range(device_count.value):
                handle = ctypes.c_void_p()
                self._check_nvml_result(
                    nvml,
                    nvml.nvmlDeviceGetHandleByIndex_v2(gpu_index, ctypes.byref(handle)),
                    'nvmlDeviceGetHandleByIndex_v2',
                )
                handles.append((str(gpu_index), handle))

            for sample_idx in range(self.sample_count):
                for gpu_index, handle in handles:
                    utilization = _NVMLUtilization()
                    memory_info = _NVMLMemoryInfo()
                    self._check_nvml_result(
                        nvml,
                        nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization)),
                        'nvmlDeviceGetUtilizationRates',
                    )
                    self._check_nvml_result(
                        nvml,
                        nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory_info)),
                        'nvmlDeviceGetMemoryInfo',
                    )
                    utilization_samples[gpu_index].append(float(utilization.gpu))
                    free_memory_samples[gpu_index].append(float(memory_info.free) / (1024.0 ** 3))

                if sample_idx + 1 < self.sample_count:
                    time.sleep(self.sample_interval_seconds)
        finally:
            if initialized:
                nvml.nvmlShutdown()

        return self._build_stats(utilization_samples, free_memory_samples)

    @staticmethod
    def _build_stats(
        utilization_samples: dict[str, list[float]],
        free_memory_samples: dict[str, list[float]],
    ) -> list[dict[str, float | str]]:
        """根据采样序列计算每张 GPU 的统计量。

        Args:
            utilization_samples: GPU index 到利用率采样值的映射
            free_memory_samples: GPU index 到空闲显存采样值的映射，单位 GiB

        Returns:
            每张 GPU 的平均占用率、最新空闲显存和平均空闲显存
        """
        stats: list[dict[str, float | str]] = []
        for gpu_index in sorted(utilization_samples.keys(), key=int):
            free_samples = free_memory_samples[gpu_index]
            stats.append(
                {
                    'index': gpu_index,
                    'avg_utilization': float(sum(utilization_samples[gpu_index]) / len(utilization_samples[gpu_index])),
                    'latest_free_memory_gb': float(free_samples[-1]),
                    'avg_free_memory_gb': float(sum(free_samples) / len(free_samples)),
                }
            )
        return stats

    def select(self) -> str | None:
        """执行自动选卡逻辑，并在成功时写入 CUDA_VISIBLE_DEVICES。"""
        if GPUAutoSelector._has_run:
            return os.environ.get('CUDA_VISIBLE_DEVICES')
        log.info('Starting GPU auto selection...')
        GPUAutoSelector._has_run = True

        if not self.enabled:
            log.info('GPU auto selection disabled.')
            return os.environ.get('CUDA_VISIBLE_DEVICES')

        default_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            collect_start_time = time.perf_counter()
            stats = self._collect_stats()
            log.info('GPU stats collection finished in %.2fs.', time.perf_counter() - collect_start_time)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else str(exc)
            log.warning('Skip GPU auto selection because nvidia-smi failed: %s', stderr)
            return default_visible_devices
        except (OSError, RuntimeError) as exc:
            log.warning('Skip GPU auto selection because GPU stats collection failed: %s', exc)
            return default_visible_devices

        if not stats:
            log.warning('Skip GPU auto selection because no GPU was reported.')
            return default_visible_devices

        sorted_stats = sorted(
            stats,
            key=lambda item: (
                float(item['avg_utilization']),
                -float(item['latest_free_memory_gb']),
                int(str(item['index'])),
            ),
        )
        log.info(
            'GPU auto selection config: sample_count=%s, sample_interval=%.1fs, max_avg_utilization=%.1f, min_free_memory_gb=%.1f, default CUDA_VISIBLE_DEVICES=%s',
            self.sample_count,
            self.sample_interval_seconds,
            self.max_avg_utilization,
            self.min_free_memory_gb,
            default_visible_devices,
        )
        for stat in sorted_stats:
            log.info(
                'GPU %s avg_utilization=%.2f%% latest_free_memory=%.2f GiB avg_free_memory=%.2f GiB',
                stat['index'],
                float(stat['avg_utilization']),
                float(stat['latest_free_memory_gb']),
                float(stat['avg_free_memory_gb']),
            )

        candidates = [
            stat
            for stat in sorted_stats
            if float(stat['avg_utilization']) < self.max_avg_utilization
            and float(stat['latest_free_memory_gb']) > self.min_free_memory_gb
        ]
        if candidates:
            selected_gpu = str(candidates[0]['index'])
            os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
            log.info('Selected GPU %s and set CUDA_VISIBLE_DEVICES=%s.', selected_gpu, selected_gpu)
            return selected_gpu

        if not self.prompt_on_failure:
            raise RuntimeError(
                'No GPU satisfies the selection rule. '
                f'Current CUDA_VISIBLE_DEVICES={default_visible_devices!r}.'
            )

        prompt_lines = [
            'No GPU satisfies the selection rule:',
            f'avg_utilization < {self.max_avg_utilization} and free_memory > {self.min_free_memory_gb} GiB.',
            'Current GPU stats:',
        ]
        for stat in sorted_stats:
            prompt_lines.append(
                f"  GPU {stat['index']}: avg_utilization={float(stat['avg_utilization']):.2f}% "
                f"latest_free_memory={float(stat['latest_free_memory_gb']):.2f} GiB"
            )
        default_text = default_visible_devices if default_visible_devices is not None else '<keep current setting>'
        prompt_lines.append(f'Choose one GPU index manually [default: {default_text}]: ')
        prompt_text = '\n'.join(prompt_lines)

        try:
            user_choice = input(prompt_text).strip()
        except EOFError as exc:
            raise RuntimeError(
                'No GPU satisfies the selection rule and interactive GPU selection is unavailable. '
                f'Current CUDA_VISIBLE_DEVICES={default_visible_devices!r}.'
            ) from exc

        if not user_choice:
            log.warning('No GPU selected manually. Keep CUDA_VISIBLE_DEVICES=%s.', default_visible_devices)
            return default_visible_devices

        valid_gpu_indices = {str(stat['index']) for stat in sorted_stats}
        if user_choice not in valid_gpu_indices:
            raise ValueError(f'Invalid GPU index: {user_choice}. Available GPUs: {sorted(valid_gpu_indices, key=int)}')

        os.environ['CUDA_VISIBLE_DEVICES'] = user_choice
        log.info('Manually selected GPU %s and set CUDA_VISIBLE_DEVICES=%s.', user_choice, user_choice)
        return user_choice


class AnomalyStorage:
    """DuckDB 异常存储器：内部 buffer 聚合 + Appender 批量写入。"""

    TABLE_NAME = 'anomalies'

    def __init__(self, db_path: str = 'bgp_anomalies.duckdb', buffer_size_limit: int = 100000, read_only: bool = False):
        self.db_path = db_path
        self.buffer_size_limit = buffer_size_limit
        self.read_only = read_only
        self.buffer: List[Tuple[int, int, int, int]] = []

        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.conn = duckdb.connect(db_path, read_only=read_only)
        if not read_only:
            self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                timestamp UBIGINT NOT NULL,
                as_1 UINTEGER NOT NULL,
                as_2 UINTEGER NOT NULL,
                as_3 UINTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)"
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('status', 'writing')"
        )

    def add_anomaly(self, timestamp: int, as_1: int, as_2: int, as_3: int) -> None:
        if self.read_only:
            raise RuntimeError('Cannot write in read-only mode.')
        self.buffer.append((int(timestamp), int(as_1), int(as_2), int(as_3)))
        if len(self.buffer) >= self.buffer_size_limit:
            self.flush()

    def add_batch(self, rows: Iterable[Tuple[int, int, int, int]]) -> None:
        if self.read_only:
            raise RuntimeError('Cannot write in read-only mode.')
        self.buffer.extend(rows)
        if len(self.buffer) >= self.buffer_size_limit:
            self.flush()

    def flush(self) -> None:
        if self.read_only or not self.buffer:
            return

        with Appender(self.conn, schema="main", table=self.TABLE_NAME) as appender:
            for row in self.buffer:
                appender.append_row(row)
        self.buffer.clear()

    def mark_complete(self) -> None:
        if self.read_only:
            return
        self.flush()
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('status', 'complete')"
        )
        self.conn.execute("CHECKPOINT")

    def query_by_timestamps(self, timestamps: Iterable[int]) -> List[Tuple[int, int, int, int]]:
        ts_list = [int(ts) for ts in timestamps]
        if not ts_list:
            return []

        if not self.read_only:
            self.flush()

        if len(ts_list) == 1:
            return self.conn.execute(
                f"SELECT timestamp, as_1, as_2, as_3 FROM {self.TABLE_NAME} WHERE timestamp = ?",
                [ts_list[0]],
            ).fetchall()

        placeholders = ','.join(['?'] * len(ts_list))
        return self.conn.execute(
            f"SELECT timestamp, as_1, as_2, as_3 FROM {self.TABLE_NAME} WHERE timestamp IN ({placeholders})",
            ts_list,
        ).fetchall()

    def query_slot_triplets(self, time_str: str) -> dict:
        dt = datetime.datetime.strptime(time_str, "%Y%m%d.%H%M").replace(
            tzinfo=datetime.timezone.utc
        )
        slot_start = int(dt.timestamp())
        slot_end = slot_start + 300

        rows = self.conn.execute(
            f"SELECT timestamp, as_1, as_2, as_3 FROM {self.TABLE_NAME} WHERE timestamp >= ? AND timestamp < ?",
            [slot_start, slot_end],
        ).fetchall()

        slot_predictions = {}
        for ts, as_1, as_2, as_3 in rows:
            ts_key = int(ts)
            if ts_key not in slot_predictions:
                slot_predictions[ts_key] = set()
            slot_predictions[ts_key].add((as_1, as_2, as_3))
        return slot_predictions

    def close(self) -> None:
        if not self.read_only:
            self.flush()
        self.conn.close()

    @staticmethod
    def remove_files(db_path: str) -> None:
        for suffix in ('', '.wal'):
            target = f"{db_path}{suffix}"
            if os.path.exists(target):
                os.remove(target)

    @staticmethod
    def is_complete(db_path: str) -> bool:
        if not os.path.exists(db_path):
            return False
        try:
            conn = duckdb.connect(db_path, read_only=True)
            try:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key='status'"
                ).fetchone()
            finally:
                conn.close()
        except Exception:
            return False
        return row is not None and row[0] == 'complete'

    @classmethod
    def query_slot_triplets_from_file(cls, db_path: str, time_str: str) -> dict:
        storage = cls(db_path=db_path, read_only=True)
        try:
            return storage.query_slot_triplets(time_str)
        finally:
            storage.close()

    @classmethod
    def load_unique_triplets_from_file(cls, db_path: str) -> set[tuple[int, int, int]]:
        if not os.path.exists(db_path):
            return set()
        storage = cls(db_path=db_path, read_only=True)
        try:
            rows = storage.conn.execute(
                f"SELECT DISTINCT as_1, as_2, as_3 FROM {cls.TABLE_NAME}"
            ).fetchall()
            return {tuple(map(int, row)) for row in rows}
        finally:
            storage.close()


def compute_binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    weigh_for_n = (tp + fn) / (tn + fp) if (tn + fp) > 0 else 1.0
    fp_weighted = fp * weigh_for_n
    precision_weighted = tp / (tp + fp_weighted) if (tp + fp_weighted) > 0 else 0
    f1_weighted = 2 * precision_weighted * recall / (precision_weighted + recall) if (precision_weighted + recall) > 0 else 0
    return {
        'recall': recall,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'precision_weighted': precision_weighted,
        'f1_weighted': f1_weighted,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


@dataclass
class CurveStyle:
    """CDF 曲线样式定义。"""

    label: str = 'Curve'
    color: str = '#1976D2'
    linewidth: float = 1.6
    linestyle: str = '-'
    alpha: float = 1.0
    marker_size: float = 18.0
    drawstyle: str | None = None


class _CDFCurveState:
    """单条 CDF 曲线的累积状态。"""

    def __init__(self) -> None:
        self.raw_chunks: list[np.ndarray] = []
        self.mode = 'exact'
        self.count = 0
        self.raw_min: float | None = None
        self.raw_max: float | None = None
        self.bin_edges: np.ndarray | None = None
        self.bin_counts: np.ndarray | None = None
        self.underflow_raw: list[np.ndarray] = []
        self.underflow_t: list[np.ndarray] = []
        self.overflow_raw: list[np.ndarray] = []
        self.overflow_t: list[np.ndarray] = []
        self.precomputed_x_plot: np.ndarray | None = None
        self.precomputed_y: np.ndarray | None = None


class CDFPlotter:
    """公共 CDF 绘图器，支持分批添加数据、预计算 CDF 与 exact/bin 自动切换。"""

    def __init__(
        self,
        x_scale: str = 'linear',
        switch_bin_threshold: int = 100_000,
        bin_count: int = 1000,
        y_min: float = -0.05,
        y_max: float = 1.05,
        annotate_min_max: bool = True,
        grid: bool = True,
    ) -> None:
        if x_scale not in {'linear', 'log_positive', 'signed_log1p'}:
            raise ValueError(f'Unsupported x_scale: {x_scale}')
        if switch_bin_threshold <= 0:
            raise ValueError('switch_bin_threshold must be positive')
        if bin_count <= 1:
            raise ValueError('bin_count must be greater than 1')
        if y_min >= y_max:
            raise ValueError('y_min must be smaller than y_max')

        self.x_scale = x_scale
        self.switch_bin_threshold = int(switch_bin_threshold)
        self.bin_count = int(bin_count)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.annotate_min_max = bool(annotate_min_max)
        self.grid = bool(grid)
        self._styles: dict[str, CurveStyle] = {}
        self._states: dict[str, _CDFCurveState] = {}

    def register_curve(self, curve_name: str, style: CurveStyle | None = None) -> None:
        if curve_name in self._styles:
            raise ValueError(f'Curve already registered: {curve_name}')
        style = style or CurveStyle(label=curve_name)
        if not style.label:
            style = CurveStyle(
                label=curve_name,
                color=style.color,
                linewidth=style.linewidth,
                linestyle=style.linestyle,
                alpha=style.alpha,
                marker_size=style.marker_size,
                drawstyle=style.drawstyle,
            )
        self._styles[curve_name] = style
        self._states[curve_name] = _CDFCurveState()

    def add_data(self, curve_name: str, values: Iterable[float] | np.ndarray) -> None:
        """向指定曲线添加数据，输入可为 iterable 或 ndarray，内部按需切换 exact/bin。"""
        state = self._get_state(curve_name)

        if isinstance(values, np.ndarray):
            raw_values = np.asarray(values).reshape(-1)
            if raw_values.size == 0:
                return
            if state.mode == 'exact' and raw_values.size > self.switch_bin_threshold:
                self._add_large_ndarray_direct_to_binned(curve_name, state, raw_values)
                return
            for raw_chunk_view in self._iter_ndarray_chunks(raw_values):
                raw = self._prepare_raw_chunk(curve_name, raw_chunk_view)
                if raw.size == 0:
                    continue
                self._ingest_prepared_chunk(state, raw)
            return

        raw = self._prepare_raw_chunk(curve_name, values)
        if raw.size == 0:
            return
        self._ingest_prepared_chunk(state, raw)

    def set_precomputed_cdf(
        self,
        curve_name: str,
        x_raw: Iterable[float] | np.ndarray,
        y: Iterable[float] | np.ndarray,
        *,
        count: int,
        raw_min: float | None = None,
        raw_max: float | None = None,
    ) -> None:
        """Set precomputed CDF points for a registered curve.

        Args:
            curve_name: Registered curve name.
            x_raw: Raw x-axis CDF positions.
            y: Cumulative probabilities aligned with `x_raw`.
            count: Number of samples represented by this CDF.
            raw_min: Optional raw minimum used for axis ticks and endpoint metadata.
            raw_max: Optional raw maximum used for axis ticks and endpoint metadata.
        """
        if count < 0:
            raise ValueError(f'count must be non-negative, got {count}')

        state = self._get_state(curve_name)
        raw = np.asarray(x_raw, dtype=np.float64).reshape(-1)
        cdf = np.asarray(y, dtype=np.float64).reshape(-1)
        if raw.size != cdf.size:
            raise ValueError(f'x_raw and y must have the same length for curve {curve_name}')

        finite_mask = np.isfinite(raw) & np.isfinite(cdf)
        raw = raw[finite_mask]
        cdf = cdf[finite_mask]
        if raw.size == 0:
            state.mode = 'precomputed'
            state.count = int(count)
            state.raw_chunks = []
            state.bin_edges = None
            state.bin_counts = None
            state.underflow_raw = []
            state.underflow_t = []
            state.overflow_raw = []
            state.overflow_t = []
            state.precomputed_x_plot = np.array([], dtype=np.float64)
            state.precomputed_y = np.array([], dtype=np.float64)
            state.raw_min = raw_min
            state.raw_max = raw_max
            return

        if np.any(cdf < -1e-12) or np.any(cdf > 1.0 + 1e-12):
            raise ValueError(f'precomputed CDF values must be in [0, 1] for curve {curve_name}')
        self._validate_raw_values(curve_name, raw)

        x_plot = self._transform_x(raw)
        order = np.argsort(x_plot)
        x_plot = x_plot[order]
        cdf = np.clip(cdf[order], 0.0, 1.0)
        if np.any(np.diff(cdf) < -1e-12):
            raise ValueError(f'precomputed CDF values must be non-decreasing for curve {curve_name}')
        x_plot, cdf = self._collapse_duplicate_cdf_points(x_plot, cdf)

        state.mode = 'precomputed'
        state.count = int(count)
        state.raw_chunks = []
        state.bin_edges = None
        state.bin_counts = None
        state.underflow_raw = []
        state.underflow_t = []
        state.overflow_raw = []
        state.overflow_t = []
        state.precomputed_x_plot = x_plot
        state.precomputed_y = cdf
        state.raw_min = float(raw_min) if raw_min is not None else float(np.min(raw))
        state.raw_max = float(raw_max) if raw_max is not None else float(np.max(raw))

    def get_count(self, curve_name: str) -> int:
        return self._get_state(curve_name).count

    def get_raw_min(self, curve_name: str) -> float | None:
        return self._get_state(curve_name).raw_min

    def get_raw_max(self, curve_name: str) -> float | None:
        return self._get_state(curve_name).raw_max

    def get_percentile(self, curve_name: str, q: float) -> float | None:
        state = self._get_state(curve_name)
        if state.count == 0:
            return None
        if not (0 <= q <= 100):
            raise ValueError(f'Percentile must be in [0, 100], got {q}')

        if state.mode == 'precomputed':
            if state.precomputed_x_plot is None or state.precomputed_y is None or state.precomputed_y.size == 0:
                return None
            target = float(q) / 100.0
            idx = int(np.searchsorted(state.precomputed_y, target, side='left'))
            idx = min(max(idx, 0), state.precomputed_y.size - 1)
            return self._inverse_transform_scalar(float(state.precomputed_x_plot[idx]))

        if state.mode == 'exact':
            values = self._get_exact_values(state)
            return float(np.percentile(values, q)) if values.size > 0 else None

        target_rank = max(1, int(np.ceil((q / 100.0) * state.count)))
        return self._estimate_percentile_from_binned(state, target_rank)

    def get_percentile_label(self, curve_name: str, q: float, fmt: str = '.3e') -> str:
        value = self.get_percentile(curve_name, q)
        if value is None:
            return 'N/A'
        return format(float(value), fmt)

    def plot(
        self,
        ax,
        *,
        xlabel: str,
        ylabel: str = 'Cumulative Probability',
        legend_loc: str = 'lower right',
        extra_vlines: list[dict[str, Any]] | None = None,
        show_legend: bool = True,
        annotate_right_endpoint_value: bool = False,
    ) -> None:
        has_curve = False
        visible_curves: list[dict[str, Any]] = []

        for curve_name, style in self._styles.items():
            x_plot, y_plot = self._build_curve_xy(curve_name)
            if x_plot.size == 0:
                continue
            has_curve = True
            plot_kwargs = {
                'color': style.color,
                'linewidth': style.linewidth,
                'linestyle': style.linestyle,
                'alpha': style.alpha,
                'label': f'{style.label} (n={self.get_count(curve_name):,})',
            }
            if style.drawstyle is not None:
                plot_kwargs['drawstyle'] = style.drawstyle
            ax.plot(x_plot, y_plot, **plot_kwargs)
            visible_curves.append(
                {
                    'curve_name': curve_name,
                    'style': style,
                    'x_plot': x_plot,
                    'y_plot': y_plot,
                }
            )

        if visible_curves:
            self._draw_endpoint_markers(ax, visible_curves)
            if annotate_right_endpoint_value:
                self._annotate_right_endpoint_values(ax, visible_curves)

        if self.annotate_min_max and visible_curves:
            annotation_points: list[dict[str, Any]] = []
            for curve_info in visible_curves:
                annotation_points.extend(
                    self._collect_curve_annotation_points(
                        curve_info['curve_name'],
                        curve_info['style'],
                        curve_info['y_plot'],
                    )
                )
            self._annotate_points(ax, annotation_points)

        if extra_vlines:
            for line_cfg in extra_vlines:
                x_raw = float(line_cfg['x'])
                x_plot = self._transform_x(np.array([x_raw], dtype=np.float64))[0]
                ax.axvline(
                    x_plot,
                    color=line_cfg.get('color', '#2E7D32'),
                    linestyle=line_cfg.get('linestyle', '--'),
                    linewidth=line_cfg.get('linewidth', 1.6),
                    alpha=line_cfg.get('alpha', 1.0),
                    label=line_cfg.get('label'),
                )

        if has_curve:
            visible_curve_names = [curve_info['curve_name'] for curve_info in visible_curves]
            tick_positions, tick_labels = self._build_global_tick_positions_and_labels(visible_curve_names)
            if tick_positions:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(self.y_min, self.y_max)
        if self.grid:
            ax.grid(True)
        if show_legend and has_curve:
            ax.legend(loc=legend_loc)

    def _get_state(self, curve_name: str) -> _CDFCurveState:
        if curve_name not in self._states:
            raise KeyError(f'Curve not registered: {curve_name}')
        return self._states[curve_name]

    def _validate_raw_values(self, curve_name: str, raw: np.ndarray) -> None:
        if self.x_scale == 'log_positive':
            invalid_count = int(np.sum(raw <= 0))
            if invalid_count > 0:
                raise ValueError(
                    f'Curve {curve_name} contains {invalid_count} non-positive values under log_positive scale'
                )

    def _update_raw_min_max(self, state: _CDFCurveState, raw: np.ndarray) -> None:
        raw_min = float(np.min(raw))
        raw_max = float(np.max(raw))
        state.raw_min = raw_min if state.raw_min is None else min(state.raw_min, raw_min)
        state.raw_max = raw_max if state.raw_max is None else max(state.raw_max, raw_max)

    def _iter_ndarray_chunks(self, values: np.ndarray) -> Iterable[np.ndarray]:
        """按阈值大小切分 ndarray，避免一次性处理超大数组。"""
        chunk_size = max(self.switch_bin_threshold, 1)
        total = int(values.size)
        for start in range(0, total, chunk_size):
            yield values[start:start + chunk_size]

    def _prepare_raw_chunk(self, curve_name: str, values: Iterable[float] | np.ndarray) -> np.ndarray:
        """把输入标准化为单个可处理 chunk，并完成 finite/坐标合法性过滤。"""
        raw = np.asarray(values, dtype=np.float64).reshape(-1)
        if raw.size == 0:
            return np.array([], dtype=np.float64)
        finite_mask = np.isfinite(raw)
        if not np.any(finite_mask):
            return np.array([], dtype=np.float64)
        raw = raw[finite_mask]
        self._validate_raw_values(curve_name, raw)
        return raw

    def _ingest_prepared_chunk(self, state: _CDFCurveState, raw: np.ndarray) -> None:
        """写入已经完成清洗的单个 chunk。"""
        self._update_raw_min_max(state, raw)
        state.count += int(raw.size)

        if state.mode == 'exact':
            state.raw_chunks.append(raw)
            if state.count > self.switch_bin_threshold:
                self._switch_curve_to_binned(state)
            return

        self._add_to_binned_state(state, raw)

    def _resolve_bin_range(self, t_min: float, t_max: float) -> tuple[float, float]:
        """归一化 bin 上下界，避免所有点重合时无法建箱。"""
        if np.isclose(t_min, t_max):
            delta = max(1e-9, abs(t_min) * 1e-6 + 1e-9)
            t_min -= delta
            t_max += delta
        return t_min, t_max

    def _compute_transformed_range_from_chunks(self, chunks: Iterable[np.ndarray]) -> tuple[float, float] | None:
        """分块计算 transformed 坐标范围，避免整段 concatenate。"""
        t_min: float | None = None
        t_max: float | None = None
        has_value = False
        for raw in chunks:
            if raw.size == 0:
                continue
            transformed = self._transform_x(raw)
            chunk_min = float(np.min(transformed))
            chunk_max = float(np.max(transformed))
            t_min = chunk_min if t_min is None else min(t_min, chunk_min)
            t_max = chunk_max if t_max is None else max(t_max, chunk_max)
            has_value = True
        if not has_value or t_min is None or t_max is None:
            return None
        return self._resolve_bin_range(t_min, t_max)

    def _initialize_binned_state(self, state: _CDFCurveState, t_min: float, t_max: float) -> None:
        """根据 transformed 坐标范围初始化 bin 状态。"""
        state.bin_edges = np.linspace(t_min, t_max, self.bin_count + 1, dtype=np.float64)
        state.bin_counts = np.zeros(self.bin_count, dtype=np.int64)
        state.mode = 'binned'

    def _migrate_exact_chunks_to_binned(self, state: _CDFCurveState) -> None:
        """把历史 exact chunks 逐块迁移到 bin，避免一次性拼接。"""
        raw_chunks = state.raw_chunks
        state.raw_chunks = []
        for raw in raw_chunks:
            if raw.size == 0:
                continue
            self._add_to_binned_state(state, raw)

    def _add_large_ndarray_direct_to_binned(
        self,
        curve_name: str,
        state: _CDFCurveState,
        values: np.ndarray,
    ) -> None:
        """超大 ndarray 直接切到 bin 模式写入，避免先 exact 暂存导致 OOM。"""
        current_range = self._compute_transformed_range_from_chunks(state.raw_chunks)
        new_range = self._compute_transformed_range_from_chunks(
            self._prepare_raw_chunk(curve_name, raw_chunk_view)
            for raw_chunk_view in self._iter_ndarray_chunks(values)
        )
        if new_range is None:
            return

        if current_range is None:
            t_min, t_max = new_range
        else:
            t_min = min(current_range[0], new_range[0])
            t_max = max(current_range[1], new_range[1])
            t_min, t_max = self._resolve_bin_range(t_min, t_max)

        self._initialize_binned_state(state, t_min, t_max)
        self._migrate_exact_chunks_to_binned(state)
        for raw_chunk_view in self._iter_ndarray_chunks(values):
            raw = self._prepare_raw_chunk(curve_name, raw_chunk_view)
            if raw.size == 0:
                continue
            self._update_raw_min_max(state, raw)
            state.count += int(raw.size)
            self._add_to_binned_state(state, raw)

    def _get_exact_values(self, state: _CDFCurveState) -> np.ndarray:
        if not state.raw_chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(state.raw_chunks).astype(np.float64, copy=False)

    def _switch_curve_to_binned(self, state: _CDFCurveState) -> None:
        """将 exact 状态按块迁移到 binned，避免一次性拼接全部原始数据。"""
        transformed_range = self._compute_transformed_range_from_chunks(state.raw_chunks)
        if transformed_range is None:
            return
        t_min, t_max = transformed_range
        self._initialize_binned_state(state, t_min, t_max)
        self._migrate_exact_chunks_to_binned(state)

    def _add_to_binned_state(self, state: _CDFCurveState, raw: np.ndarray) -> None:
        if state.bin_edges is None or state.bin_counts is None:
            raise RuntimeError('Binned state is not initialized')

        transformed = self._transform_x(raw)
        left = state.bin_edges[0]
        right = state.bin_edges[-1]

        under_mask = transformed < left
        over_mask = transformed > right
        mid_mask = ~(under_mask | over_mask)

        if np.any(under_mask):
            state.underflow_raw.append(raw[under_mask].astype(np.float64, copy=False))
            state.underflow_t.append(transformed[under_mask].astype(np.float64, copy=False))
        if np.any(over_mask):
            state.overflow_raw.append(raw[over_mask].astype(np.float64, copy=False))
            state.overflow_t.append(transformed[over_mask].astype(np.float64, copy=False))
        if np.any(mid_mask):
            counts, _ = np.histogram(transformed[mid_mask], bins=state.bin_edges)
            state.bin_counts += counts.astype(np.int64, copy=False)

    def _estimate_percentile_from_binned(self, state: _CDFCurveState, target_rank: int) -> float | None:
        under_raw = self._concat_or_empty(state.underflow_raw)
        under_t = self._concat_or_empty(state.underflow_t)
        if under_raw.size > 0:
            order = np.argsort(under_t)
            if target_rank <= under_raw.size:
                return float(under_raw[order[target_rank - 1]])
        rank_offset = under_raw.size

        if state.bin_counts is not None and state.bin_edges is not None:
            cumulative = np.cumsum(state.bin_counts)
            if cumulative.size > 0 and target_rank <= rank_offset + cumulative[-1]:
                bin_idx = int(np.searchsorted(cumulative, target_rank - rank_offset, side='left'))
                bin_left = float(state.bin_edges[bin_idx])
                bin_right = float(state.bin_edges[bin_idx + 1])
                raw_left = self._inverse_transform_scalar(bin_left)
                raw_right = self._inverse_transform_scalar(bin_right)
                return float((raw_left + raw_right) / 2.0)
            if cumulative.size > 0:
                rank_offset += int(cumulative[-1])

        over_raw = self._concat_or_empty(state.overflow_raw)
        over_t = self._concat_or_empty(state.overflow_t)
        if over_raw.size > 0:
            order = np.argsort(over_t)
            over_index = target_rank - rank_offset - 1
            if 0 <= over_index < over_raw.size:
                return float(over_raw[order[over_index]])
        return state.raw_max

    def _build_curve_xy(self, curve_name: str) -> tuple[np.ndarray, np.ndarray]:
        state = self._get_state(curve_name)
        if state.count == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        if state.mode == 'precomputed':
            if state.precomputed_x_plot is None or state.precomputed_y is None:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
            return state.precomputed_x_plot, state.precomputed_y
        if state.mode == 'exact':
            values = self._get_exact_values(state)
            return self._compute_exact_ecdf(values)
        return self._compute_binned_cdf(state)

    @staticmethod
    def _collapse_duplicate_cdf_points(
        x_plot: np.ndarray,
        y_plot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """合并相同 x 的 CDF 点，只保留该 x 的最终累计概率。

        Args:
            x_plot: 已按 x 升序排列的横轴坐标。
            y_plot: 与 x_plot 对齐的累计概率。

        Returns:
            去重后的 `(x_plot, y_plot)`，用于在唯一 x 点之间直接连线。
        """
        if x_plot.size <= 1:
            return x_plot, y_plot
        keep_mask = np.empty(x_plot.size, dtype=bool)
        keep_mask[:-1] = x_plot[:-1] != x_plot[1:]
        keep_mask[-1] = True
        return x_plot[keep_mask], y_plot[keep_mask]

    def _compute_exact_ecdf(self, raw_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if raw_values.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        transformed = self._transform_x(raw_values)
        order = np.argsort(transformed)
        x_plot = transformed[order]
        y_plot = np.arange(1, raw_values.size + 1, dtype=np.float64) / raw_values.size
        return self._collapse_duplicate_cdf_points(x_plot, y_plot)

    def _compute_binned_cdf(self, state: _CDFCurveState) -> tuple[np.ndarray, np.ndarray]:
        total = state.count
        if total == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        cumulative = 0

        under_t = self._concat_or_empty(state.underflow_t)
        if under_t.size > 0:
            under_sorted = np.sort(under_t)
            under_y = (np.arange(1, under_sorted.size + 1, dtype=np.float64) + cumulative) / total
            x_parts.append(under_sorted)
            y_parts.append(under_y)
            cumulative += under_sorted.size

        if state.bin_counts is not None and state.bin_edges is not None and np.any(state.bin_counts > 0):
            positive_bins = np.flatnonzero(state.bin_counts > 0)
            bin_x = state.bin_edges[positive_bins + 1]
            bin_y = (cumulative + np.cumsum(state.bin_counts[positive_bins], dtype=np.int64)) / total
            x_parts.append(bin_x.astype(np.float64, copy=False))
            y_parts.append(bin_y.astype(np.float64, copy=False))
            cumulative = int(cumulative + np.sum(state.bin_counts[positive_bins]))

        over_t = self._concat_or_empty(state.overflow_t)
        if over_t.size > 0:
            over_sorted = np.sort(over_t)
            over_y = (np.arange(1, over_sorted.size + 1, dtype=np.float64) + cumulative) / total
            x_parts.append(over_sorted)
            y_parts.append(over_y)

        if not x_parts:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        return self._collapse_duplicate_cdf_points(np.concatenate(x_parts), np.concatenate(y_parts))

    def _draw_endpoint_markers(self, ax, visible_curves: list[dict[str, Any]]) -> None:
        """Draw same-color markers at the first and last visible CDF points."""
        for curve_info in visible_curves:
            x_plot = curve_info['x_plot']
            y_plot = curve_info['y_plot']
            if x_plot.size == 0 or y_plot.size == 0:
                continue
            style = curve_info['style']
            ax.scatter(
                [x_plot[0], x_plot[-1]],
                [y_plot[0], y_plot[-1]],
                color=style.color,
                edgecolors=style.color,
                s=style.marker_size,
                zorder=5,
            )

    def _annotate_right_endpoint_values(self, ax, visible_curves: list[dict[str, Any]]) -> None:
        """Annotate each curve's right endpoint with its raw x value."""
        for idx, curve_info in enumerate(visible_curves):
            x_plot = curve_info['x_plot']
            y_plot = curve_info['y_plot']
            if x_plot.size == 0 or y_plot.size == 0:
                continue
            style = curve_info['style']
            raw_value = self._inverse_transform_scalar(float(x_plot[-1]))
            offset_y = 7.0 if idx % 2 == 0 else -10.0
            vertical_alignment = 'bottom' if offset_y > 0 else 'top'
            ax.annotate(
                self._format_value_label(raw_value),
                xy=(float(x_plot[-1]), float(y_plot[-1])),
                xytext=(-4.0, offset_y),
                textcoords='offset points',
                ha='right',
                va=vertical_alignment,
                color=style.color,
                fontsize=7,
                fontweight='bold',
                zorder=6,
                bbox={
                    'boxstyle': 'round,pad=0.15',
                    'facecolor': 'white',
                    'edgecolor': 'none',
                    'alpha': 0.75,
                },
            )

    def _collect_curve_annotation_points(
        self,
        curve_name: str,
        style: CurveStyle,
        y_plot: np.ndarray,
    ) -> list[dict[str, Any]]:
        state = self._get_state(curve_name)
        points: list[dict[str, Any]] = []
        if state.raw_min is not None:
            min_x = float(self._transform_x(np.array([state.raw_min], dtype=np.float64))[0])
            min_y = float(y_plot[0]) if y_plot.size > 0 else 0.0
            points.append(
                {
                    'kind': 'min',
                    'x': min_x,
                    'y': min_y,
                    'text': f'min={self._format_value_label(state.raw_min)}',
                    'color': style.color,
                    'marker_size': style.marker_size,
                }
            )
        if state.raw_max is not None:
            max_x = float(self._transform_x(np.array([state.raw_max], dtype=np.float64))[0])
            points.append(
                {
                    'kind': 'max',
                    'x': max_x,
                    'y': 1.0,
                    'text': f'max={self._format_value_label(state.raw_max)}',
                    'color': style.color,
                    'marker_size': style.marker_size,
                }
            )
        return points

    def _annotate_points(self, ax, annotation_points: list[dict[str, Any]]) -> None:
        if not annotation_points:
            return
        sorted_points = sorted(annotation_points, key=lambda item: (float(item['x']), float(item['y'])))
        for idx, point in enumerate(sorted_points):
            place_above = (idx % 2 == 0)
            offset_y = 5.0 if place_above else -6.0
            vertical_alignment = 'bottom' if place_above else 'top'
            ax.annotate(
                point['text'],
                xy=(point['x'], point['y']),
                xytext=(0, offset_y),
                textcoords='offset points',
                fontsize=6,
                color=point['color'],
                ha='center',
                va=vertical_alignment,
            )

    def _build_tick_positions_and_labels(self, curve_name: str) -> list[tuple[float, str]]:
        state = self._get_state(curve_name)
        if state.raw_min is None or state.raw_max is None:
            return []
        raw_min = float(state.raw_min)
        raw_max = float(state.raw_max)
        if raw_min > raw_max:
            raw_min, raw_max = raw_max, raw_min

        if self.x_scale == 'linear':
            if np.isclose(raw_min, raw_max):
                candidates = np.array([raw_min], dtype=np.float64)
            else:
                candidates = np.linspace(raw_min, raw_max, 5, dtype=np.float64)
        elif self.x_scale == 'log_positive':
            min_exp = int(np.floor(np.log10(raw_min)))
            max_exp = int(np.ceil(np.log10(raw_max)))
            candidates = np.array([10.0 ** exp for exp in range(min_exp, max_exp + 1)], dtype=np.float64)
            candidates = candidates[(candidates >= raw_min) & (candidates <= raw_max)]
            if candidates.size == 0:
                candidates = np.array([raw_min, raw_max], dtype=np.float64)
        else:
            candidates = self._build_signed_log1p_raw_ticks(raw_min, raw_max)

        transformed = self._transform_x(candidates)
        return [(float(pos), self._format_value_label(raw)) for pos, raw in zip(transformed, candidates)]

    def _build_global_tick_positions_and_labels(self, curve_names: list[str]) -> tuple[list[float], list[str]]:
        raw_mins: list[float] = []
        raw_maxs: list[float] = []
        for curve_name in curve_names:
            state = self._get_state(curve_name)
            if state.raw_min is None or state.raw_max is None:
                continue
            raw_mins.append(float(state.raw_min))
            raw_maxs.append(float(state.raw_max))
        if not raw_mins or not raw_maxs:
            return [], []

        raw_min = min(raw_mins)
        raw_max = max(raw_maxs)
        if raw_min > raw_max:
            raw_min, raw_max = raw_max, raw_min

        if self.x_scale == 'linear':
            if np.isclose(raw_min, raw_max):
                candidates = np.array([raw_min], dtype=np.float64)
            else:
                candidates = np.linspace(raw_min, raw_max, 5, dtype=np.float64)
        elif self.x_scale == 'log_positive':
            min_exp = int(np.floor(np.log10(raw_min)))
            max_exp = int(np.ceil(np.log10(raw_max)))
            candidates = np.array([10.0 ** exp for exp in range(min_exp, max_exp + 1)], dtype=np.float64)
            candidates = candidates[(candidates >= raw_min) & (candidates <= raw_max)]
            if candidates.size == 0:
                candidates = np.array([raw_min, raw_max], dtype=np.float64)
        else:
            candidates = self._build_signed_log1p_raw_ticks(raw_min, raw_max)

        transformed = self._transform_x(candidates)
        tick_pairs = [(float(pos), self._format_value_label(raw)) for pos, raw in zip(transformed, candidates)]
        return self._merge_tick_pairs(tick_pairs)

    def _build_signed_log1p_raw_ticks(self, raw_min: float, raw_max: float) -> np.ndarray:
        base_candidates = np.array(
            [
                -1e6, -1e4, -1e2, -10.0, -1.0, -1e-2, -1e-4,
                0.0,
                1e-4, 1e-2, 1.0, 10.0, 1e2, 1e4, 1e6,
            ],
            dtype=np.float64,
        )
        candidates = base_candidates[(base_candidates >= raw_min) & (base_candidates <= raw_max)]
        if candidates.size == 0:
            candidates = np.array([raw_min, raw_max], dtype=np.float64)
        if raw_min <= 0 <= raw_max and not np.any(np.isclose(candidates, 0.0)):
            candidates = np.sort(np.concatenate([candidates, np.array([0.0], dtype=np.float64)]))
        if not np.any(np.isclose(candidates, raw_min)):
            candidates = np.sort(np.concatenate([candidates, np.array([raw_min], dtype=np.float64)]))
        if not np.any(np.isclose(candidates, raw_max)):
            candidates = np.sort(np.concatenate([candidates, np.array([raw_max], dtype=np.float64)]))
        return candidates

    def _merge_tick_pairs(self, tick_pairs: list[tuple[float, str]]) -> tuple[list[float], list[str]]:
        merged: dict[float, str] = {}
        for pos, label in tick_pairs:
            rounded_pos = round(float(pos), 10)
            merged[rounded_pos] = label
        positions = sorted(merged.keys())
        labels = [merged[pos] for pos in positions]
        return positions, labels

    def _transform_x(self, raw_values: np.ndarray) -> np.ndarray:
        raw_values = np.asarray(raw_values, dtype=np.float64)
        if self.x_scale == 'linear':
            return raw_values
        if self.x_scale == 'log_positive':
            return np.log10(raw_values)
        return np.sign(raw_values) * np.log1p(np.abs(raw_values))

    def _inverse_transform_scalar(self, value: float) -> float:
        if self.x_scale == 'linear':
            return float(value)
        if self.x_scale == 'log_positive':
            return float(10.0 ** value)
        sign = 1.0 if value >= 0 else -1.0
        return float(sign * (np.exp(abs(value)) - 1.0))

    @staticmethod
    def _concat_or_empty(chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(chunks).astype(np.float64, copy=False)

    @staticmethod
    def _format_value_label(value: float) -> str:
        return f'{float(value):.2e}'


def decodeMRT(filepath) -> list[str]:
    decoded_data = os.popen(f'bgpdump -q -m "{filepath}"').read()
    lines = decoded_data.strip('\n').split("\n")
    result = []
    for line in lines:
        if not line:
            continue
        parts = line.split('|', 6)  # maxsplit=6，只切到 prefix 字段，避免切割长 AS Path
        if len(parts) > 5 and ':' in parts[5]:  # parts[5] 是 PREFIX，含 ':' 则为 IPv6，跳过
            continue
        result.append(line)
    return result


def iter_announce_update_parts(messages: Iterable[str]):
    for msg in messages:
        if not msg:
            continue
        parts = msg.strip().split('|')
        if len(parts) < 7 or parts[2] != 'A':
            continue
        yield parts


def build_consecutive_triplets(clean_as_path: list[int]) -> set[tuple[int, int, int]]:
    """Build all consecutive AS triplets from a cleaned AS path.

    Args:
        clean_as_path: AS path after `AsPathDataClean`, containing ordered AS integers.

    Returns:
        Every consecutive `(hops[i], hops[i + 1], hops[i + 2])` triplet. Returns
        an empty set when the path has fewer than three AS hops.
    """
    if not clean_as_path or len(clean_as_path) < 3:
        return set()

    return {
        tuple(clean_as_path[idx:idx + 3])
        for idx in range(len(clean_as_path) - 2)
    }


def radbench_message_key(parts: list[str]) -> tuple[str, str, str]:
    """Build a RADBench message key for exact GT message matching.

    Args:
        parts: bgpdump `BGP4MP` line split by `|`.

    Returns:
        `(timestamp, prefix, as_path)` key.
    """
    if len(parts) < 7:
        raise ValueError(f'RADBench message key requires at least 7 fields, got {len(parts)}')
    return parts[1], parts[5], parts[6]


RADBENCH_IXP_ROOT = os.path.join('DATA', 'Processed', 'IXP')
_radbench_ixp_cache: dict[tuple[str, str], frozenset[int]] = {}


def load_radbench_ixp_asns(event_name: str, ixp_root: str | None = None) -> frozenset[int]:
    """加载单个 RADBench 事件的 IXP ASN 集合。

    Args:
        event_name: RADBench 事件名。
        ixp_root: `prepare_radbench_ixps.py` 生成的事件 IXP 目录。

    Returns:
        事件对应的 IXP ASN 集合。
    """
    if not event_name:
        raise ValueError('event_name must be non-empty when loading RADBench IXP ASNs')
    if ixp_root is None:
        ixp_root = RADBENCH_IXP_ROOT

    cache_key = (os.path.abspath(ixp_root), event_name)
    if cache_key in _radbench_ixp_cache:
        return _radbench_ixp_cache[cache_key]

    filepath = os.path.join(ixp_root, f'{event_name}.txt')
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f'RADBench IXP ASN file not found for event {event_name}: {filepath}. '
            'Run prepare_radbench_ixps.py first.'
        )

    with open(filepath, 'r', encoding='utf-8') as f:
        tokens = f.read().split()
    try:
        ixp_asns = frozenset(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(f'Invalid ASN token in {filepath}') from exc

    if not ixp_asns:
        raise RuntimeError(f'RADBench IXP ASN file is empty: {filepath}')
    _radbench_ixp_cache[cache_key] = ixp_asns
    return ixp_asns


def to_bin(IP):
    list1 = IP.split('.')
    list2 = []
    for item in list1:
        try:
            item = bin(int(item))  # ---0b11000000 0b10101000 0b1100000 0b1011110 ----

            # cut the first 2 bin :0b.
            item = item[2:]
        except:
            print(IP)

        list2.append(item.zfill(8))  # --['11000000', '10101000', '01100000', '01011110']--
    v2 = ''.join(list2)  # ----11000000101010000110000001011110----
    # print(v2)
    return v2

def is_sub_pfx(father_prefix, sub_prefix):
    father_pfx, f_mask = father_prefix.split('/')
    sub_pfx, sub_mask = sub_prefix.split('/')
    f_mask = int(f_mask)
    sub_mask = int(sub_mask)
    if (f_mask > sub_mask):
        return False
    elif (f_mask % 8 == 0):  # 
        block = int(f_mask / 8)
        list1 = father_pfx.split('.')
        list2 = sub_pfx.split('.')
        return (list1[0:block] == list2[0:block])
    else:  # 
        father_IP_bin = to_bin(father_pfx)
        sub_IP_bin = to_bin(sub_pfx)
        father_IP_bin = father_IP_bin[0:f_mask]
        sub_IP_bin = sub_IP_bin[0:f_mask]
        if father_IP_bin == sub_IP_bin:
            return True
        else:
            return False

def edit_distance(l1, l2):  # 
        """
        :param l1: list 1
        :param l2: list 2
        :return: edit distance between l1 and l2
        """
        rows = len(l1) + 1
        cols = len(l2) + 1

        dist = [[0 for x in range(cols)] for x in range(rows)]

        for i in range(1, rows):
            dist[i][0] = i
        for i in range(1, cols):
            dist[0][i] = i

        for col in range(1, cols):
            for row in range(1, rows):
                if l1[row - 1] == l2[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                     dist[row][col - 1] + 1,  # insertion
                                     dist[row - 1][col - 1] + cost)  # substitution
        return dist[row][col], rows - cols

class DataFileHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        # self.static_features = {}
        if not os.path.exists(self.file_path):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w') as file:
                file.write('')

    def read(self):
        with open(self.file_path, 'r') as file:
            return file.readlines()

    def write_line(self, data:dict):
        if os.path.getsize(self.file_path) == 0:
            # If the file is empty, write the header
            with open(self.file_path, 'w') as file:
                file.write(','.join(data.keys()) + '\n')
        # data = dict(sorted(data.items(), key=lambda item: item[0]))  # 按照键排序
        with open(self.file_path, 'a') as file:
            file.writelines([','.join(map(str, data.values())) + '\n'])


def delete_empty_files_and_dirs(directory, suffix='.csv'):
    """
    Deletes empty files and directories in the specified directory.
    :param directory: The directory to clean up.
    """
    empty_files = []
    empty_dirs = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            if not name.endswith(suffix):
                continue
            file_path = os.path.join(root, name)
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                empty_files.append(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            if not os.listdir(dir_path):  # Check if the directory is empty
                os.rmdir(dir_path)  # Remove the empty directory
                empty_dirs.append(dir_path)
    return empty_dirs, empty_files

def save_distribution_img(df:pd.DataFrame, filepath):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    sns.set(style="whitegrid")

    # 创建画布
    feature_num = len(df.columns)
    column_num = min(3, feature_num)
    row_num = feature_num // column_num if feature_num % column_num == 0 else feature_num // column_num + 1
    fig, axes = plt.subplots(row_num, column_num, figsize=(30, 18))
    axes = axes.flatten()
    for feature_name, ax in zip(df.columns, axes):
        log.info(f'drawing {feature_name}')
        # ax.set_title(f'{feature_name} distribution')
        # if feature_name in ['asCountry', ]:
        #     continue
        data = df[feature_name]
        sns.histplot(data, kde=True, ax=ax, color='blue', bins=30)
    # 自动调整布局
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300)

def scalar_ignore_columnname_transform(X:pd.DataFrame, rename_dict, scaler):
    """临时修改列名后执行 transform，再恢复列名"""
    origin_name, target_name = list(rename_dict.items())[0]
    X_renamed = X.rename(columns=rename_dict)
    X_renamed[[target_name]] = scaler.transform(X_renamed[[target_name]])
    return X_renamed.rename(columns={target_name: origin_name})

# dataframe根据开始结束时间戳补全
def df_fill_by_timestamp(df: pd.DataFrame, start_timestamp: int, end_timestamp: int, time_step: int) -> pd.DataFrame:
    """
    根据开始和结束时间戳补全DataFrame，填充缺失的时间戳行。
    :param df: 原始DataFrame，必须包含'timestamp'列。
    :param start_timestamp: 开始时间戳。
    :param end_timestamp: 结束时间戳。
    :param time_step: 时间步长
    :return: 补全后的DataFrame。
    """
    df['timestamp'] = df['timestamp'].astype(int)
    full_timestamps = np.arange(start_timestamp, end_timestamp + 1, time_step)
    # 将原始数据设置为以 timestamp 为索引
    # df.set_index('timestamp', inplace=True)
    df = df.set_index('timestamp').reindex(full_timestamps).fillna(0).reset_index()
    # timeline_df.rename(columns={'index': 'timestamp'}, inplace=True)
    return df

def build_timestamp_observed_mask(df: pd.DataFrame, total_timestamps) -> np.ndarray:
    """构造时间点级真实观测 mask。

    Args:
        df: 补全前的特征表，必须包含 timestamp 列。
        total_timestamps: 完整时间轴，输出 mask 与该顺序一致。

    Returns:
        np.ndarray: uint8 数组，shape 为 (len(total_timestamps),)，1 表示原始数据中存在该时间点，0 表示补全产生。
    """
    if 'timestamp' not in df.columns:
        raise KeyError("Input dataframe missing required column: timestamp")

    full_timestamps = np.asarray(total_timestamps, dtype=np.int64)
    observed_timestamps = df['timestamp'].to_numpy(dtype=np.int64, copy=False)
    return np.isin(full_timestamps, observed_timestamps).astype(np.uint8, copy=False)


def transform_nonzero_only(data:np.array, scaler):
    # data = np.asarray(data)
    data = np.asarray(data, dtype=np.float64)  # 确保数据类型为float64
    non_zero_mask = data != 0
    if np.any(non_zero_mask):
        data[non_zero_mask] = scaler.transform(data[non_zero_mask].reshape(-1, 1)).flatten()
    return data

def datetime_format(fmt='%Y%m%d-%H%M%S'):
    """
    返回一个格式化函数，将字符串转换为指定格式的datetime对象。
    :param fmt: 日期时间格式字符串，默认为'%Y%m%d-%H%M%S'。
    :return: 格式化函数。
    """
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(fmt)

def read_RADBench_event_list(event_list_path='anomaly-event-info.csv', event_type='leak'):
    """读取事件列表
    Args:
        event_list_path (str): 事件列表路径
        type (str): 事件类型[leak, hijack, outage, ]
    """
    events = pd.read_csv(event_list_path)
    if event_type:
        events = events[events['event_type'] == event_type]
    results = []
    for index, row in events.iterrows():
        event_type = row['event_type']
        event_name = row['event_name']
        event_start_time = row['start_time'].strip()
        event_end_time = row['end_time'].strip()
        prefix = row['prefix']
        hijacked_prefix = row['hijacked_prefix']
        hijack_as = row['hijack_as']
        victim_as = row['vicitim_as']
        outage = row['outage_as']
        leaker_as = row['leak_as']
        propagator_as = row['propagator_as']
        hijack_as = int(hijack_as) if not pd.isna(hijack_as) else hijack_as
        victim_as = int(victim_as) if not pd.isna(victim_as) else victim_as
        leaker_as = int(leaker_as) if not pd.isna(leaker_as) else leaker_as
        propagator_as = list(map(int, propagator_as.split(' '))) if not pd.isna(propagator_as) else []

        # if evnet_type=='leak':
        event_start_time = datetime.datetime.strptime(event_start_time, "%Y/%m/%d %H:%M").strftime("%Y-%m-%d %H:%M")
        if event_end_time != 'unknown':
            event_end_time = datetime.datetime.strptime(event_end_time, "%Y/%m/%d %H:%M").strftime("%Y-%m-%d %H:%M")
        data = {
            'event_name': event_name,
            'event_type': event_type,
            'event_start_time': event_start_time,
            'event_end_time': "" if event_end_time=='unknown' else event_end_time,
            'prefix': prefix,
            'hijacked_prefix': hijacked_prefix,
            'hijack_as': hijack_as,
            'victim_as': victim_as,
            'outage': outage,
            'leaker_as': leaker_as,
            'propagator_as': propagator_as
        }
        # yield data
        results.append(data)
    return results

class RADBenchEvent:
    def __init__(self, event_info):
        self.name = event_info['event_name']
        self.type = event_info['event_type']
        self.start_time = event_info['event_start_time']
        self.end_time = event_info['event_end_time']
        self.leaker_as = event_info['leaker_as']
        self.propagator_as = event_info['propagator_as']
        
    def __str__(self):
        return f"Event: {self.name}, Type: {self.type}, Leaker AS: {self.leaker_as}, Start: {self.start_time}"
    
def judge_triplet_anomaly(triplet, leaker, propagators:list):
    """判断三元组是否异常

    Args:
        triplet (list): (as1, as2, as3)
        leaker (int): 泄漏者AS
        propagators (list): 传播者AS列表

    Returns:
        _type_: _description_
    """
    assert len(triplet) == 3
    assert isinstance(triplet[0], int)
    if leaker in triplet and (any([prop in triplet for prop in propagators]) if propagators else True):  # leaker propagator都在三元组中就算异常
    # if leaker in triplet:
        return True
    else:
        return False
    


def get_RADBench_event_label_TimestampBased(event:RADBenchEvent, feature_path):
    """基于时间戳计算事件标签

    Args:
        event (RADBenchEvent): RADBench事件对象
    """
    event_start_time = datetime.datetime.strptime(event.start_time, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
    event_end_time = datetime.datetime.strptime(event.end_time, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc) if event.end_time else None
    with open(f'{feature_path}/{event.name}/timestamps.txt', 'r') as f:
        small_time_window, start_time, end_time = map(int, f.read().strip().split(' '))
    label=[]
    for timestamp in range(start_time, end_time + small_time_window, small_time_window):
        if event_start_time.timestamp() - small_time_window <= timestamp <= (event_end_time.timestamp() if event_end_time else float('inf')):
            label.append(1)
        else:
            label.append(0)
    return label, start_time, end_time, small_time_window

def calc_pointwise_metrics(pred_label, true_label):
    """计算point-wise指标
    Args:
        pred_label (list): 预测标签
        true_label (list): 真实标签
    Returns:
        dict: point-wise指标
    """
    assert len(pred_label) == len(true_label), "预测标签和真实标签长度不一致"

    report = classification_report(y_true=true_label, y_pred=pred_label,
                        target_names=['Normal', 'Abnomarl'], labels=range(2))
    # 混淆矩阵：https://zhuanlan.zhihu.com/p/350664406
    cm = confusion_matrix(y_true=true_label, y_pred=pred_label)
    TN, FP, FN, TP = cm.ravel()
    precision = TP/(TP+FP) if TP+FP else -1
    recall = TP/(TP+FN) if TP+FN else -1
    F1_score = 2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else -1
    FPR = FP/(FP+TN)

    metric_str = f"{'='*50}\nTN={TN}, FP={FP}, FN={FN}, TP={TP}\nprecision={precision}, recall={recall}, F1-Score={F1_score}, FPR={FPR}\n\n{report}\n{'='*50}"

    return metric_str

def symmetric_log_transform(X):
    """
    应用对称对数变换 f(x) = sign(x) * log1p(abs(x))
    """
    return np.sign(X) * np.log1p(np.abs(X))

def data_completion(df:pd.DataFrame, total_timestamps):
    df = df[df['timestamp'].isin(total_timestamps)].copy()
    df = df.set_index('timestamp').reindex(total_timestamps).reset_index()

    ffill_cols = [col for col in df.columns if col.startswith(('AS_MPL', 'AS_AVG_PL',)) and col != 'label']
    other_cols = [col for col in df.columns if col not in ffill_cols and col != 'timestamp']

    df[ffill_cols] = df[ffill_cols].replace(0, np.nan).ffill().bfill().fillna(0)
    df[other_cols] = df[other_cols].fillna(0)
    return df

def check_asn(asn):
    if (asn == 23456 or asn == 0):
        return False
    elif (asn >= 401309):
        return False
    elif (asn >= 64496 and asn <= 131071):
        return False
    elif (asn >= 155962 and asn <= 196607):
        return False
    elif (asn >= 216476 and asn <= 262143):
        return False
    elif (asn >= 273821 and asn <= 327679):
        return False
    elif (asn >= 329728 and asn <= 393215):
        return False
    elif (asn >= 402333):
        return False
    else:
        return True

def AsPathDataClean(hops, ixp_asns: set[int] | frozenset[int] | None = None):
    """清洗 AS_PATH，并可在最前面移除 IXP ASN。

    Args:
        hops: 原始 AS_PATH token 序列。
        ixp_asns: 需要从路径中删除的 IXP / Route Server ASN 集合。

    Returns:
        清洗后的 ASN 列表；遇到 AS_SET、非法 ASN 或环路时返回 None。
    """
    cleaned = []
    ixp_asn_set = ixp_asns if ixp_asns else frozenset()
    for item in hops:
        if isinstance(item, str) and ('{' in item or '}' in item):  # AS_SET，return None
            return None
        try:
            as_num = int(item)
        except ValueError:
            return None

        # PathProb 口径：先移除 IXP/Route Server ASN，再做 prepending 压缩和环路检测。
        if as_num in ixp_asn_set:
            continue

        if not check_asn(as_num):
            return None

        # 去除连续重复AS
        if not cleaned or cleaned[-1] != as_num:
            cleaned.append(as_num)

    if not cleaned:
        return None

    # DataClean 2. 环路检测
    if len(set(cleaned)) != len(cleaned):
        return None

    return cleaned

def iter_decoded_lines(decoded_dir: str) -> Iterable[Tuple[int, str, str]]:
    """Yield (timestamp, msg_type, as_path) from decoded update files."""
    files = [
        f
        for f in os.listdir(decoded_dir)
        if os.path.isfile(os.path.join(decoded_dir, f))
        and f.startswith("rrc00_updates.")
        and (f.endswith(".txt") or f.endswith(".gz"))
    ]
    files.sort()
    for fname in files:
        path = os.path.join(decoded_dir, fname)
        with open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("|")
                if len(parts) < 7:
                    continue
                try:
                    ts = int(parts[1])
                except ValueError:
                    continue
                yield ts, parts[2], parts[6]

def compute_precise_labels(
    decoded_dir: str,
    target_triplets: List[Tuple[int, int, int]],
    coarse_label: np.ndarray,
    timestamps: np.ndarray,
    time_window: int,
    ) -> dict[Tuple[int, int, int], np.ndarray]:
    """Compute precise labels for the given triplet list."""

    base_ts = int(timestamps[0])
    end_ts = base_ts + time_window * len(timestamps)
    labels = {triplet: np.zeros(len(timestamps), dtype=int) for triplet in target_triplets}
    target_set = set(target_triplets)

    for ts, msg_type, as_path in tqdm(
        iter_decoded_lines(decoded_dir), desc="Scanning updates", total=None
    ):
        if ts < base_ts or ts >= end_ts:
            continue
        if msg_type != "A":
            continue
        idx = (ts - base_ts) // time_window
        hops_raw = as_path.split(" ")
        try:
            hops = AsPathDataClean(hops_raw)
        except Exception:
            continue
        if hops is None:
            continue
        for i in range(1, len(hops) - 1):
            triplet = (hops[i - 1], hops[i], hops[i + 1])
            if triplet not in target_set:
                continue
            if labels[triplet][idx]:
                continue
            labels[triplet][idx] = 1
    for triplet in target_triplets:
        labels[triplet] = labels[triplet] * coarse_label

    return labels
    
if __name__ == "__main__":
    get_RADBench_event_label_TimestampBased()
