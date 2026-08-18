"""Generate RADBench no-sparse LMDB data with feature-level nonzero masks.

Purpose:
    Build an independent RADBench no-sparse dataset whose mask marks only
    observed, nonzero raw feature values for reconstruction learning.

Key parameters:
    - RADBENCH_OUTPUT_ROOT: output dataset root.
    - SPARSE_TRIPLET_ROW_THRESHOLD: minimum raw rows retained per triplet CSV.

Outputs:
    - RawFeature/, Scalers/, Unsupervise/, Supervise/, and Test/ under
      RADBENCH_OUTPUT_ROOT.
    - LMDB `mask_{key}` values are uint8 arrays with shape `(T, D)`, where
      D is the feature dimension excluding label.
"""

import copy
import datetime
import logging
import os
import pickle
import random
import re
import shutil
import threading
import time
import types
from collections import defaultdict
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from queue import Queue as ThreadQueue

import dotenv
import joblib
import lmdb
import numpy as np
import pandas as pd
import pytricia
import zstandard
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, StandardScaler
from tqdm import tqdm

from common.performance import get_active_recorder

from common import (
    AsPathDataClean,
    DataFileHandler,
    beijing,
    build_consecutive_triplets,
    data_completion,
    decodeMRT,
    delete_empty_files_and_dirs,
    edit_distance,
    load_radbench_ixp_asns,
    read_RADBench_event_list,
)


logging.Formatter.converter = beijing
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

dotenv.load_dotenv()

TRAIN_VAL_EVENT_NUM = 8
RADBENCH_EVENT_INFO_CSV = "radbench_leaks.csv"
RADBENCH_INPUT_ROOT = "DATA/Raw/RADBench"
RADBENCH_GT_ROOT = "DATA/Processed/GroundTruth"
SPARSE_TRIPLET_ROW_THRESHOLD = 5
RADBENCH_OUTPUT_ROOT = "DATA/RADBench"
RADBENCH_RANDOM_SEED = 42
LMDB_MASK_TYPE = "feature_nonzero"
LMDB_MASK_SHAPE = "T,D"


def seed_everything(seed: int):
    """设置 Python 和 NumPy 随机种子。

    Args:
        seed (int): 输入随机种子。

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)


def parse_event_time(time_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)


def parse_update_datetime(filename: str):
    match = re.search(r"\d{8}\.\d{4}", filename)
    if not match:
        return None
    try:
        return datetime.datetime.strptime(match.group(), "%Y%m%d.%H%M")
    except ValueError:
        return None


def floor_to_5min(dt: datetime.datetime) -> datetime.datetime:
    return dt.replace(minute=dt.minute - (dt.minute % 5), second=0, microsecond=0)


def expected_update_iter(start_time: datetime.datetime, end_time: datetime.datetime):
    current = floor_to_5min(start_time)
    while current <= end_time:
        year = current.strftime("%Y")
        month = current.strftime("%m")
        date_str = current.strftime("%Y%m%d")
        time_str = current.strftime("%H%M")
        filename = f"rrc00_updates.{date_str}.{time_str}.gz"
        url = f"http://data.ris.ripe.net/rrc00/{year}.{month}/updates.{date_str}.{time_str}.gz"
        yield filename, url
        current += datetime.timedelta(minutes=5)


def get_last_gt_timestamp(event_name: str, gt_root: str | None = None):
    if gt_root is None:
        gt_root = RADBENCH_GT_ROOT
    gt_file = os.path.join(gt_root, f"{event_name}.txt")
    if not os.path.exists(gt_file):
        return None

    max_ts = None
    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            try:
                ts = int(parts[1])
                if max_ts is None or ts > max_ts:
                    max_ts = ts
            except ValueError:
                continue

    if max_ts is None:
        return None
    return datetime.datetime.fromtimestamp(max_ts, datetime.timezone.utc)


def infer_event_end_time(event: dict):
    if event.get("event_end_time"):
        return parse_event_time(event["event_end_time"])

    gt_end = get_last_gt_timestamp(event["event_name"])
    if gt_end is not None:
        return gt_end

    return parse_event_time(event["event_start_time"]) + datetime.timedelta(hours=6)


def ensure_updates_complete(start_time: datetime.datetime, end_time: datetime.datetime, save_dir: str):
    """检查并补全给定时间范围内的 updates 文件（5 分钟粒度）。本地已有的文件直接跳过。"""
    os.makedirs(save_dir, exist_ok=True)

    # 快速检查：若时间范围内所有文件本地均已存在，直接返回
    all_expected = list(expected_update_iter(start_time, end_time))
    if all_expected and all(os.path.exists(os.path.join(save_dir, fname)) for fname, _ in all_expected):
        log.info("All updates already present locally, skip download: %s", save_dir)
        return

    max_retries = 3
    temp_dir = os.path.join(save_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)

    def download_task(url: str, target_path: str):
        if os.path.exists(target_path):
            return

        tmp_path = os.path.join(temp_dir, os.path.basename(target_path))
        cmd = f"wget -q -O {tmp_path} {url}"
        ret = os.system(cmd)

        if ret != 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return

        if os.path.exists(tmp_path):
            shutil.move(tmp_path, target_path)

    for attempt in range(1, max_retries + 1):
        missing = []
        for filename, url in expected_update_iter(start_time, end_time):
            target_path = os.path.join(save_dir, filename)
            if not os.path.exists(target_path):
                missing.append((url, target_path))

        if not missing:
            log.info("All updates are ready: %s", save_dir)
            break

        log.info(
            "Downloading missing updates for %s (attempt %d): %d files",
            save_dir,
            attempt,
            len(missing),
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(download_task, url, path) for url, path in missing]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {os.path.basename(save_dir)}"):
                pass

    final_missing = [
        filename
        for filename, _ in all_expected
        if not os.path.exists(os.path.join(save_dir, filename))
    ]
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    if final_missing:
        preview = ", ".join(final_missing[:10])
        raise RuntimeError(
            f"Incomplete update window after download: {save_dir}. "
            f"missing={len(final_missing)}/{len(all_expected)}. First missing files: {preview}"
        )


def build_window_anomaly_map(
    event_name: str,
    leaker_as,
    propagator_as_list,
    time_window: int,
    gt_root: str | None = None,
    ixp_asns: set[int] | frozenset[int] | None = None,
):
    """从 GT 精确消息构造按时间窗口组织的异常三元组映射。

    过滤规则：
    1) AS_PATH 必须包含 leaker；
    2) AS_PATH 必须包含至少一个 propagator；

    只提取两种三元组形式：
    1) [x, propagator, leaker]
    2) [propagator, leaker, x]

    Args:
        ixp_asns: 需要在 AS_PATH clean 最前面移除的 IXP / Route Server ASN 集合。
    """
    anomaly_map = defaultdict(set)
    if gt_root is None:
        gt_root = RADBENCH_GT_ROOT

    if leaker_as is None or pd.isna(leaker_as):
        return anomaly_map

    gt_file = os.path.join(gt_root, f"{event_name}.txt")
    if not os.path.exists(gt_file):
        log.warning("GT file not found: %s", gt_file)
        return anomaly_map

    leaker = int(leaker_as)
    propagators = set(int(x) for x in propagator_as_list) if propagator_as_list else set()

    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 7 or parts[2] != "A":
                continue

            try:
                ts = int(parts[1])
            except ValueError:
                continue

            hops = AsPathDataClean(parts[6].split(" "), ixp_asns=ixp_asns)
            if not hops or len(hops) < 3:
                continue

            if leaker not in hops:
                continue

            if propagators and not any(p in hops for p in propagators):
                continue

            window_ts = ts - (ts % time_window)

            for i in range(len(hops) - 1):
                cur_as = hops[i]
                if cur_as not in propagators:
                    continue
                if i + 1 >= len(hops) or hops[i + 1] != leaker:
                    continue

                if i > 0:
                    left_triplet = (hops[i - 1], hops[i], hops[i + 1])
                    anomaly_map[window_ts].add(left_triplet)

                if i + 2 < len(hops):
                    right_triplet = (hops[i], hops[i + 1], hops[i + 2])
                    anomaly_map[window_ts].add(right_triplet)

    log.info(
        "Anomaly map built for %s: %d windows, %d triplets",
        event_name,
        len(anomaly_map),
        sum(len(v) for v in anomaly_map.values()),
    )
    return anomaly_map


class FeatureExtractor:
    def __init__(
        self,
        RawDataPath,
        OutputFeaturePath,
        time_window=60,
        date_range=None,
        pct_change=False,
        anomaly_window_map=None,
        event_name="",
        ixp_asns: set[int] | frozenset[int] | None = None,
        performance_section: str | None = None,
    ):
        """特征提取器（CF 主流程 + RADBench 标注微调）。

        Args:
            RawDataPath (str): 原始 updates 数据目录
            OutputFeaturePath (str): 输出目录，含 timeline_feature 和 timestamps.txt
            time_window (int, optional): 特征计算滑动时间窗口（秒）
            date_range (tuple, optional): (start_date, end_date)，筛选 updates 文件
            pct_change (bool, optional): 是否对部分列做环比变化
            anomaly_window_map (dict, optional): window_ts -> set(triplet)
            event_name (str, optional): 事件名，仅用于日志定位
            ixp_asns (set, optional): 需要在 AS_PATH clean 最前面移除的 IXP / Route Server ASN 集合
            performance_section (str, optional): 在线等价预处理性能区间名。
        """
        self.RawDataPath = RawDataPath
        self.OutputFeaturePath = OutputFeaturePath
        self.time_window = time_window
        self.start_date, self.end_date = date_range if date_range else (None, None)
        self.pct_change_flag = pct_change
        self.pct_prefixes = ("AS_TRANSIT", "TRIPLET")
        self.anomaly_window_map = anomaly_window_map if anomaly_window_map else defaultdict(set)
        self.event_name = event_name
        self.ixp_asns = ixp_asns if ixp_asns else frozenset()
        self.performance_section = performance_section
        self.all_anomaly_triplets = set()

        os.makedirs(self.OutputFeaturePath, exist_ok=True)

        self.current_window_ts = None
        self.timestamp_list = []

        self.total_time = 0
        self.is_sub_prefix_time = 0
        self.edit_distance_time = 0

        self.__init_features__()

    def __init_timewindow_features__(self):
        """初始化时间窗口内特征变量。

        这些变量在每个窗口 flush 后会重置。
        """
        # features within timewindow, 下一个时间窗口就清空
        self.TripletSet = set()
        self.TRIPLET_APPEAR_NUM = defaultdict(int)  # triplet出现计数, TRIPLET_APPEAR_NUM[triplet]=x

        self.AS_ANN_NUM = defaultdict(int)  # AS宣告次数, AS_ANN_NUM[ASN]=x
        # self.AS_ANN_PREFIX = defaultdict(set) # AS宣告的prefix, AS_ANN_PREFIX[ASN]=set(prefix), 源作者代码中的victim_prefix
        self.AS_ANN_PREFIX = defaultdict(pytricia.PyTricia)  # AS宣告的prefix, AS_ANN_PREFIX[ASN]=set(prefix), 源作者代码中的victim_prefix
        # self.AS_NEW_SUB_PREFIX = defaultdict(set) # 新子前缀, AS_NEW_SUB_PREFIX[ASN]=set(prefix)
        self.AS_NEW_SUB_PREFIX_ANN_NUM = defaultdict(int)  # AS新子前缀宣告次数, AS_NEW_SUB_PREFIX_ANN_NUM[ASN]=x
        self.AS_PEER_NUM = defaultdict(int)  # AS的peer数量, AS_PEER_NUM[ASN]=x
        self.AS_PEER_PATH = defaultdict(lambda: defaultdict(list))  # origin AS的到peer的路径, AS_PEER_PATH[ASN][peer]=path(list形态)
        self.AS_PEERS_INCREASE = defaultdict(int)  # AS的peer增加次数, AS_PEERS_INCREASE[ASN]=x
        self.AS_PL = defaultdict(lambda: defaultdict(int))  # AS_PATH长度计数, AS_PATH_LEN_CNT[ASN][len_path]=x
        self.AS_MPL = defaultdict(int)  # AS最长路径长度, AS_MPL[ASN]=x
        self.AS_DIFF_ANN = defaultdict(int)  # AS差异宣告as path次数, AS_DIFF_ANN[ASN]=x
        self.AS_DIFF_ANN_SHORTER = defaultdict(lambda: defaultdict(int))  # AS差异宣告as path变短次数, AS_DIFF_ANN_SHORTER[ASN][len_cut]=x
        self.AS_DIFF_ANN_LONGER = defaultdict(lambda: defaultdict(int))  # AS差异宣告as path变长次数, AS_DIFF_ANN_LONGER[ASN][len_cut]=x
        self.AS_DIFF_PATH_ED = defaultdict(lambda: defaultdict(float))  # AS差异宣告as path编辑距离, AS_DIFF_PATH_ED[ASN][ed]=x
        self.AS_DUPLICATE_ANN = defaultdict(int)  # AS重复宣告次数, AS_DUPLICATE_ANN[ASN]=x

        # self.AS_MOAS_NUM = defaultdict(int) # AS MOAS宣告次数, AS_MOAS_NUM[ASN]=x

        self.AS_WITHDRAW_COUNT = defaultdict(int)  # AS撤回次数, AS_WITHDRAW_NUM[ASN]=x
        self.AS_WITHDRAW_PREFIX = defaultdict(set)  # AS撤回的prefix, AS_WITHDRAW_PREFIX[ASN]=set(prefix)

        # 25.8.25 新增
        self.AS_TRANSIT_NUM = defaultdict(int)  # AS 在路径上作为transit的出现次数, AS_TRANSIT_NUM[ASN]=x
        self.AS_TRANSIT_PREFIX = defaultdict(set)  # AS 在路径上作为transit的prefix, AS_TRANSIT_PREFIX[ASN]=set(origin_AS)
        self.AS_TRANSIT_ORIGIN_AS = defaultdict(set)  # AS 在路径上作为transit的origin AS, AS_TRANSIT_ORIGIN_AS[ASN]=set(origin_AS)
        self.TRIPLET_APPEAR_NUM = defaultdict(int)  # triplet出现计数, TRIPLET_APPEAR_NUM[triplet]=x
        self.TRIPLET_TRANSIT_PREFIX = defaultdict(set)  # triplet在路径上作为transit的prefix, TRIPLET_TRANSIT_PREFIX[triplet]=set(origin_AS)
        self.TRIPLET_TRANSIT_ORIGIN_AS = defaultdict(set)  # triplet路径上的origin AS, TRIPLET_ORIGIN_AS[triplet]=set(origin_AS)
        self.TRIPLET_WITHDRAW_COUNT = defaultdict(int)  # triplet撤回的次数, TRIPLET_WITHDRAW_COUNT[triplet]=x
        self.TRIPLET_WITHDRAW_PREFIX = defaultdict(set)  # triplet撤回的prefix, TRIPLET_WITHDRAW_PREFIX[triplet]=set(prefix)

        # 优化 Withdraw 处理的倒排索引
        # prefix -> set(origin AS)
        self.PREFIX_TO_ORIGIN = defaultdict(set)
        # prefix -> set(triplet)
        self.PREFIX_TO_TRIPLET = defaultdict(set)

    def __init_global_features__(self):
        """初始化全局不会清空的特征变量。"""
        self.AllFeatures = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def __init_features__(self):
        """初始化所有特征变量。"""
        self.__init_timewindow_features__()
        self.__init_global_features__()

    def calc_extra_features(self):
        """计算额外聚合特征。"""
        self.AS_AVG_PL = defaultdict(float)  # AS平均路径长度, AS_AVG_PL[ASN]=avg_len
        self.AS_AVG_DIFF_ED = defaultdict(float)  # AS平均差异编辑距离, AS_AVG_DIFF_ED[ASN]=ed
        self.AS_AVG_DIFF_ANN_SHORTER = defaultdict(float)  # AS平均差异宣告变短次数, AS_AVG_DIFF_ANN_SHORTER[ASN]=x
        self.AS_AVG_DIFF_ANN_LONGER = defaultdict(float)  # AS平均差异宣告变长次数, AS_AVG_DIFF_ANN_LONGER[ASN]=x

        def calc_avg(d: dict):
            """计算字典的平均值"""
            if not d:
                return 0.0
            s = 0
            c = 0
            for key, value in d.items():
                if isinstance(value, (int, float)):
                    s += key * value
                    c += value
            return s / c if c else 0.0

        for asn in self.AS_PL:
            self.AS_AVG_PL[asn] = calc_avg(self.AS_PL[asn])
        for asn in self.AS_DIFF_PATH_ED:
            self.AS_AVG_DIFF_ED[asn] = calc_avg(self.AS_DIFF_PATH_ED[asn])
        for asn in self.AS_DIFF_ANN_SHORTER:
            self.AS_AVG_DIFF_ANN_SHORTER[asn] = calc_avg(self.AS_DIFF_ANN_SHORTER[asn])
        for asn in self.AS_DIFF_ANN_LONGER:
            self.AS_AVG_DIFF_ANN_LONGER[asn] = calc_avg(self.AS_DIFF_ANN_LONGER[asn])

    def extract_timeline_feature_per_msg(self, msg: list[str]):
        # todo: 性能优化 leak-20200721-Commercial_Conecte_leak/rrc00_updates.20200719.1625
        prefix = msg[5]
        # 跳过 ipv6 前缀
        if ":" in prefix:
            return

        ############################################
        # drop: ['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix']
        if msg[2] == "A":
            hops = AsPathDataClean(msg[6].split(" "), ixp_asns=self.ixp_asns)
            if not hops:
                return

            len_path = len(hops)
            if len_path <= 0:
                return

            origin = hops[-1]
            if not isinstance(origin, int):
                return

            self.AS_ANN_NUM[origin] += 1

            for triplet in build_consecutive_triplets(hops):
                # Skip if any element in the triplet is not an integer (e.g., AS_SET)
                if any(not isinstance(hop, int) for hop in triplet):
                    continue

                self.TripletSet.add(triplet)
                self.TRIPLET_APPEAR_NUM[triplet] += 1
                self.TRIPLET_TRANSIT_PREFIX[triplet].add(prefix)
                self.TRIPLET_TRANSIT_ORIGIN_AS[triplet].add(origin)
                self.PREFIX_TO_TRIPLET[prefix].add(triplet)

            for hop in hops[:-1]:
                if not isinstance(hop, int):
                    continue
                self.AS_TRANSIT_NUM[hop] += 1
                self.AS_TRANSIT_PREFIX[hop].add(prefix)
                self.AS_TRANSIT_ORIGIN_AS[hop].add(origin)

            t = time.time()
            # 使用 Pytricia 优化：检查前缀是否存在，不存在再判断是否是已存在前缀的子前缀
            if not self.AS_ANN_PREFIX[origin].has_key(prefix):
                # .get_key(prefix) 返回覆盖该前缀的最具体前缀（父前缀）
                if self.AS_ANN_PREFIX[origin].get_key(prefix) is not None:
                    self.AS_NEW_SUB_PREFIX_ANN_NUM[origin] += 1
                self.AS_ANN_PREFIX[origin][prefix] = True
            self.PREFIX_TO_ORIGIN[prefix].add(origin)
            self.is_sub_prefix_time += time.time() - t

            peer = hops[0]
            if peer not in self.AS_PEER_PATH[origin]:
                self.AS_PEER_NUM[origin] += 1
                self.AS_PEERS_INCREASE[origin] += 1
                self.AS_PEER_PATH[origin][peer] = hops
                self.AS_PL[origin][len_path] += 1
                if len_path > self.AS_MPL[origin]:
                    self.AS_MPL[origin] = len_path
            else:
                if hops != self.AS_PEER_PATH[origin][peer]:
                    self.AS_DIFF_ANN[origin] += 1
                    t = time.time()
                    ed, len_cut = edit_distance(self.AS_PEER_PATH[origin][peer], hops)
                    self.edit_distance_time += time.time() - t
                    self.AS_PEER_PATH[origin][peer] = hops
                    if len_cut > 0:
                        self.AS_DIFF_ANN_SHORTER[origin][len_cut] += 1
                    else:
                        self.AS_DIFF_ANN_LONGER[origin][len_cut] += 1
                    self.AS_DIFF_PATH_ED[origin][ed] += 1
                    self.AS_PL[origin][len_path] += 1
                    if len_path > self.AS_MPL[origin]:
                        self.AS_MPL[origin] = len_path
                else:
                    self.AS_DUPLICATE_ANN[origin] += 1

        elif msg[2] == "W":
            # 使用倒排索引 (O(1) lookup) 代替全量遍历 (O(N))

            # 1) Update AS Withdraw features
            # 只遍历曾经宣告过该 prefix 的 Origin AS
            if prefix in self.PREFIX_TO_ORIGIN:
                for origin in self.PREFIX_TO_ORIGIN[prefix]:
                    self.AS_WITHDRAW_COUNT[origin] += 1
                    self.AS_WITHDRAW_PREFIX[origin].add(prefix)

            # 2) Update Triplet Withdraw features
            # 只遍历曾经宣告过该 prefix 的 Triplet
            if prefix in self.PREFIX_TO_TRIPLET:
                for triplet in self.PREFIX_TO_TRIPLET[prefix]:
                    self.TRIPLET_WITHDRAW_COUNT[triplet] += 1
                    self.TRIPLET_WITHDRAW_PREFIX[triplet].add(prefix)
        ############################################

    def extract_timeline_feature(self):
        self.current_window_ts = None
        all_files = sorted(
            [
                i
                for i in os.listdir(self.RawDataPath)
                if os.path.isfile(os.path.join(self.RawDataPath, i))
                and i.startswith("rrc")
                and (i.endswith(".gz") or i.endswith("txt"))
            ]
        )

        if self.start_date and self.end_date:
            filtered_files = []
            for filename in all_files:
                file_datetime = parse_update_datetime(filename)
                if file_datetime is None:
                    continue
                if self.start_date <= file_datetime <= self.end_date:
                    filtered_files.append(filename)
        else:
            filtered_files = all_files

        # Using a producer-consumer model for background decoding
        decode_queue = ThreadQueue(maxsize=5)

        def producer(file_list, queue):
            for filename in file_list:
                file_path = os.path.join(self.RawDataPath, filename)
                started = time.perf_counter()
                msgs = decodeMRT(file_path)
                elapsed = time.perf_counter() - started
                recorder = get_active_recorder()
                if recorder is not None and self.performance_section is not None:
                    recorder.add_elapsed(self.performance_section, elapsed)
                    recorder.add_count(self.performance_section, "five_minute_files", 1)
                queue.put((filename, msgs))
            queue.put(None)

        decoder_thread = threading.Thread(target=producer, args=(filtered_files, decode_queue))
        decoder_thread.start()

        for _ in tqdm(range(len(filtered_files)), desc=f"Event: {os.path.basename(self.RawDataPath)}"):
            item = decode_queue.get()
            if item is None:
                break

            filename, msgs = item
            compute_started = time.perf_counter()
            excluded_io_seconds = 0.0
            for msg in tqdm(msgs, desc=f"Processing {filename}", leave=False):
                msg = msg.strip().split("|")
                if len(msg) < 6:
                    continue

                try:
                    timestamp = int(msg[1])
                except ValueError:
                    continue

                self.extract_timeline_feature_per_msg(msg)

                if timestamp % self.time_window != 0:
                    continue

                if self.current_window_ts is None:
                    self.current_window_ts = timestamp
                    continue

                if timestamp - self.current_window_ts < self.time_window:
                    continue

                self.timestamp_list.append(self.current_window_ts)
                # 保存时间戳
                io_started = time.perf_counter()
                with open(os.path.join(self.OutputFeaturePath, "timestamps.txt"), "w") as ts_file:
                    if self.timestamp_list:
                        ts_file.write(
                            f"{self.time_window} {self.timestamp_list[0]} {self.timestamp_list[-1]}"
                        )
                excluded_io_seconds += time.perf_counter() - io_started

                # 一个时间窗口结束，计算并保存特征数据
                self.calc_extra_features()
                candidate_anomalies = self.anomaly_window_map.get(self.current_window_ts, set())

                triple_feature_kv = {
                    "AS_NEW_SUB_PREFIX_ANN_NUM": self.AS_NEW_SUB_PREFIX_ANN_NUM,
                    "AS_PEER_NUM": self.AS_PEER_NUM,
                    "AS_PEERS_INCREASE": self.AS_PEERS_INCREASE,
                    "AS_MPL": self.AS_MPL,
                    "AS_ANN_NUM": self.AS_ANN_NUM,
                    "AS_DUPLICATE_ANN": self.AS_DUPLICATE_ANN,
                    "AS_WITHDRAW_PREFIX_COUNT": lambda x: len(self.AS_WITHDRAW_PREFIX[x]),
                    "AS_WITHDRAW_COUNT": self.AS_WITHDRAW_COUNT,
                    "AS_DIFF_ANN": self.AS_DIFF_ANN,
                    "AS_AVG_PL": self.AS_AVG_PL,
                    "AS_AVG_DIFF_ED": self.AS_AVG_DIFF_ED,
                    "AS_AVG_DIFF_ANN_SHORTER": self.AS_AVG_DIFF_ANN_SHORTER,
                    "AS_AVG_DIFF_ANN_LONGER": self.AS_AVG_DIFF_ANN_LONGER,
                    "AS_TRANSIT_NUM": self.AS_TRANSIT_NUM,
                    "AS_TRANSIT_PREFIX_COUNT": lambda x: len(self.AS_TRANSIT_PREFIX[x]),
                    "AS_TRANSIT_ORIGIN_AS_COUNT": lambda x: len(self.AS_TRANSIT_ORIGIN_AS[x]),
                    "TRIPLET_APPEAR_NUM": self.TRIPLET_APPEAR_NUM,
                    "TRIPLET_PREFIX_NUM": lambda x: len(self.TRIPLET_TRANSIT_PREFIX[x]),
                    "TRIPLET_ORIGIN_AS_NUM": lambda x: len(self.TRIPLET_TRANSIT_ORIGIN_AS[x]),
                    "TRIPLET_WITHDRAW_PREFIX_COUNT": lambda x: len(self.TRIPLET_WITHDRAW_PREFIX[x]),
                    "TRIPLET_WITHDRAW_COUNT": lambda x: self.TRIPLET_WITHDRAW_COUNT[x],
                }

                for triplet in self.TripletSet:
                    final_feature = {"timestamp": self.current_window_ts}
                    for k, v in triple_feature_kv.items():
                        if k.startswith("TRIPLET_"):
                            if isinstance(v, (types.LambdaType, types.FunctionType)):
                                feature_value = v(triplet)
                            else:
                                feature_value = v[triplet]
                            final_feature[k] = feature_value
                        else:
                            for i in range(3):
                                feature_name = f"{k}_{i}"
                                if isinstance(v, (types.LambdaType, types.FunctionType)):
                                    feature_value = v(triplet[i])
                                else:
                                    feature_value = v[triplet[i]]
                                final_feature[feature_name] = feature_value

                    is_anomaly = 1 if triplet in candidate_anomalies else 0
                    if is_anomaly:
                        self.all_anomaly_triplets.add(triplet)
                    final_feature["label"] = is_anomaly

                    # 保存特征数据
                    out_filename = ",".join(str(x) for x in triplet) + ".csv"
                    filepath = os.path.join(
                        self.OutputFeaturePath,
                        "timeline_feature",
                        str(triplet[1]),
                        out_filename,
                    )
                    io_started = time.perf_counter()
                    DataFileHandler(filepath).write_line(final_feature)
                    excluded_io_seconds += time.perf_counter() - io_started

                self.__init_timewindow_features__()
                self.current_window_ts = timestamp

            compute_seconds = max(
                0.0, time.perf_counter() - compute_started - excluded_io_seconds
            )
            self.total_time += compute_seconds
            recorder = get_active_recorder()
            if recorder is not None and self.performance_section is not None:
                recorder.add_elapsed(self.performance_section, compute_seconds)
                recorder.add_count(self.performance_section, "messages", len(msgs))

        decoder_thread.join()

        # 清理空白文件和文件夹
        delete_empty_files_and_dirs(os.path.join(self.OutputFeaturePath, "timeline_feature"), suffix=".csv")

        # 保存本事件遇到的所有异常三元组
        if self.all_anomaly_triplets:
            with open(os.path.join(self.OutputFeaturePath, "AnomalyTriplets.txt"), "w") as f:
                for triplet in sorted(self.all_anomaly_triplets):
                    f.write(f"{triplet[0]}_{triplet[1]}_{triplet[2]}\n")

        if self.pct_change_flag:
            self.pct_change()

    def pct_change(self):
        """对时间序列特征进行环比变化转换。"""
        base_dir = os.path.join(self.OutputFeaturePath, "timeline_feature")
        if not os.path.exists(base_dir):
            log.info("timeline_feature not found: %s", base_dir)
            return

        csv_files = []
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if fname.endswith(".csv"):
                    csv_files.append(os.path.join(root, fname))

        if not csv_files:
            log.info("No CSV files found under timeline_feature.")
            return

        for fp in tqdm(csv_files, desc="Applying pct_change to timeline features"):
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                log.warning("Failed to read %s: %s", fp, e)
                continue

            if df is None or df.empty:
                continue

            if "timestamp" in df.columns:
                # 按时间排序保证环比顺序正确
                df = df.sort_values("timestamp").reset_index(drop=True)

            cols = [c for c in df.columns if c.startswith(self.pct_prefixes)]
            if not cols:
                continue

            for c in cols:
                # 转为数值，无法解析的置为 NaN
                df[c] = pd.to_numeric(df[c], errors="coerce")
                s = df[c].pct_change()
                # 将无穷和缺失值替换为 0；包含第一行 NaN 的情况
                s = s.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                df[c] = s

            df.to_csv(fp, index=False)

def load_radbench_leak_events(csv_path: str = RADBENCH_EVENT_INFO_CSV):
    """读取 RADBench 泄漏事件列表。

    Args:
        csv_path (str): 输入事件元数据 CSV 路径。

    Returns:
        list[dict]: 按 CSV 原始顺序返回的 leak 事件列表。
    """
    return read_RADBench_event_list(event_list_path=csv_path, event_type="leak")


def split_events_for_datasets(events: list[dict]):
    # if len(events) < 9:
    #     raise ValueError(f"Need at least 9 leak events, got {len(events)}")

    supervised_events = events[:TRAIN_VAL_EVENT_NUM]
    test_events = events[TRAIN_VAL_EVENT_NUM:]
    return supervised_events, test_events


def move_files(base_dir, files, dest_root, max_workers=32):
    """多线程复制文件（保持相对目录结构）。"""
    if not files:
        return

    os.makedirs(dest_root, exist_ok=True)

    tasks = []
    dirs_to_create = set()
    # 1. Prepare paths and identify directories
    for src in files:
        rel_path = os.path.relpath(src, base_dir)
        dest = os.path.join(dest_root, rel_path)
        dirs_to_create.add(os.path.dirname(dest))
        tasks.append((src, dest))

    # 2. Create directories
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    # 3. Parallel Copy
    def _copy(task):
        src, dest = task
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            log.warning("Failed to copy %s to %s: %s", src, dest, e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_copy, tasks), total=len(tasks), desc=f"Copying files to {dest_root}"))


def get_total_timestamps(ts_path):
    if not os.path.exists(ts_path):
        log.warning("timestamps.txt not found: %s", ts_path)
        return None

    with open(ts_path, "r") as f:
        parts = f.read().strip().split()
        if len(parts) < 3:
            return None
        window = int(parts[0])
        start_ts = int(parts[1])
        end_ts = int(parts[2])

    return list(range(start_ts, end_ts + window, window))


def center_crop_timestamps(timestamps: list[int], seq_len: int):
    if len(timestamps) <= seq_len:
        return timestamps
    extra = len(timestamps) - seq_len
    head = extra // 2
    return timestamps[head : head + seq_len]


def compute_min_seq_len(event_timestamps_map: dict[str, list[int]]):
    lengths = [len(v) for v in event_timestamps_map.values() if v]
    if not lengths:
        return 0
    return min(lengths)


def collect_event_csv_files(raw_feature_root: str, event_names: list[str]):
    """收集事件下的全部特征 CSV，并按路径排序保证顺序稳定。

    Args:
        raw_feature_root (str): 输入 raw feature 根目录。
        event_names (list[str]): 输入事件名列表。

    Returns:
        list[str]: 排序后的 CSV 路径列表。
    """
    all_csvs = []
    for event_name in event_names:
        feature_dir = os.path.join(raw_feature_root, event_name, "timeline_feature")
        if not os.path.isdir(feature_dir):
            log.warning("Missing feature dir: %s", feature_dir)
            continue
        for root, _, files in os.walk(feature_dir):
            for f in sorted(files):
                if f.endswith(".csv"):
                    all_csvs.append(os.path.join(root, f))
    all_csvs.sort()
    return all_csvs


def extract_event_name_from_path(file_path: str, event_names: list[str]):
    normalized = file_path.replace("\\", "/")
    for name in event_names:
        if f"/{name}/" in normalized:
            return name
    return None


def process_file_completion(file_path, total_timestamps):
    try:
        df = pd.read_csv(file_path)
        if df is None or df.empty:
            return None
        return data_completion(df, total_timestamps)
    except Exception as e:
        log.error("Error processing %s: %s", file_path, e)
        return None


def build_feature_nonzero_mask(
    df: pd.DataFrame,
    total_timestamps: list[int] | np.ndarray,
    feature_columns: list[str],
) -> np.ndarray:
    """Build a feature-level mask for observed nonzero raw feature values.

    Args:
        df: Raw triplet CSV dataframe before time-axis completion.
        total_timestamps: Final event timeline written to LMDB.
        feature_columns: Feature columns in the exact LMDB feature order; label is excluded.

    Returns:
        uint8 mask with shape `(T, D)`. `1` means the raw CSV contains that
        timestamp and the corresponding raw feature value is nonzero.
    """
    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe missing required column: timestamp")

    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Input dataframe missing feature columns required by nonzero mask: {missing_columns}")

    full_timestamps = np.asarray(total_timestamps, dtype=np.int64)
    mask = np.zeros((len(full_timestamps), len(feature_columns)), dtype=np.uint8)
    if len(full_timestamps) == 0 or not feature_columns:
        return mask

    real_df = df[df["timestamp"].isin(full_timestamps)]
    if real_df.empty:
        return mask

    positions = pd.Index(full_timestamps).get_indexer(real_df["timestamp"].to_numpy(dtype=np.int64, copy=False))
    valid_positions = positions >= 0
    if not np.any(valid_positions):
        return mask

    real_df = real_df.iloc[np.nonzero(valid_positions)[0]]
    positions = positions[valid_positions]
    values = real_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    feature_mask = (np.isfinite(values) & (values != 0)).astype(np.uint8, copy=False)
    np.maximum.at(mask, positions, feature_mask)
    return mask


def process_file_completion_with_nonzero_mask(
    file_path: str,
    total_timestamps: list[int],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Read one triplet CSV, complete the timeline, and return nonzero feature mask.

    Args:
        file_path: Input triplet CSV path.
        total_timestamps: Final cropped event timeline.
        feature_columns: Feature columns in LMDB order, excluding label.

    Returns:
        `(completed_df, nonzero_mask)` where `nonzero_mask` is uint8 with
        shape `(T, D)` and marks observed raw nonzero feature values.
    """
    df = pd.read_csv(file_path)
    if df is None or df.empty:
        return None

    nonzero_mask = build_feature_nonzero_mask(df, total_timestamps, feature_columns)
    completed_df = data_completion(df, total_timestamps)
    return completed_df, nonzero_mask


def count_real_label_points(df: pd.DataFrame, total_timestamps) -> tuple[int, int]:
    """统计补全前真实时间点的正负标签数量。

    Args:
        df: 原始三元组 CSV 读入后的 DataFrame，必须包含 timestamp 列。
        total_timestamps: 当前事件最终写入 LMDB 的时间轴。

    Returns:
        tuple[int, int]: `(pos_count, neg_count)`，只统计原始 CSV 中真实存在且落在
        `total_timestamps` 内的行；若没有 label 列，则这些真实行全部按负样本计数。
    """
    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe missing required column: timestamp")

    real_df = df[df["timestamp"].isin(total_timestamps)]
    if real_df.empty:
        return 0, 0

    if "label" not in real_df.columns:
        return 0, int(len(real_df))

    labels = real_df["label"].to_numpy(copy=False)
    pos_count = int(np.count_nonzero(labels == 1))
    neg_count = int(np.count_nonzero(labels == 0))
    return pos_count, neg_count


def standardization_worker(file_queue, result_queue, ts_dict, feature_columns: list[str]):
    """Worker process for nonzero standardization parameter calculation.

    Consumes `(file_path, ts_key)` from file_queue and returns
    `(completed_dataframe, nonzero_feature_mask)` or None to result_queue.
    """
    while True:
        item = file_queue.get()
        if item is None:
            result_queue.put(None)
            break

        file_path, ts_key = item
        try:
            timestamps = ts_dict[ts_key]
            result = process_file_completion_with_nonzero_mask(file_path, timestamps, feature_columns)
            if result is not None:
                result_queue.put(result)
        except Exception as e:
            log.error("Worker process error for %s: %s", file_path, e)
            result_queue.put(None)


_LMDB_PREPROCESS_CONTEXT: dict | None = None
_LMDB_ZSTD_COMPRESSOR: zstandard.ZstdCompressor | None = None


def get_lmdb_feature_name(column: str) -> str:
    """返回列名对应的 scaler 特征名。

    Args:
        column: CSV/LMDB 中的特征列名。

    Returns:
        str: 去除三元组位置后缀后的基础特征名；普通列保持原名。
    """
    if column.endswith(("_0", "_1", "_2")):
        base, suffix = column.rsplit("_", 1)
        if suffix in ("0", "1", "2"):
            return base
    return column


def build_lmdb_feature_groups(columns: list[str], scalar_dict: dict[str, Pipeline]) -> dict[str, list[str]]:
    """按 scaler 特征名聚合 LMDB 写入列。

    Args:
        columns: 去除 timestamp 后、即将写入 LMDB 的列名列表。
        scalar_dict: {feature_name: sklearn Pipeline} 标准化器字典。

    Returns:
        dict[str, list[str]]: {feature_name: columns}，用于批量 transform 同一组特征列。
    """
    feature_groups = defaultdict(list)
    for column in columns:
        if column == "label":
            continue

        feature_name = get_lmdb_feature_name(column)
        if feature_name in scalar_dict:
            feature_groups[feature_name].append(column)

    return dict(feature_groups)


def init_lmdb_preprocess_worker(context: dict) -> None:
    """初始化 LMDB 写入前的多进程预处理上下文。

    Args:
        context: 只读上下文字典，包含 scaler、事件时间轴、写入列和特征分组。

    Returns:
        None
    """
    global _LMDB_PREPROCESS_CONTEXT, _LMDB_ZSTD_COMPRESSOR
    _LMDB_PREPROCESS_CONTEXT = context
    _LMDB_ZSTD_COMPRESSOR = zstandard.ZstdCompressor(level=1)


def process_lmdb_write_entry(entry: tuple[str, str]) -> tuple[str, bytes, bytes, int, int, str, float] | None:
    """读取并预处理单个三元组 CSV，生成可直接写入 LMDB 的压缩数据。

    Args:
        entry: (file_path, event_name)，分别表示 CSV 路径和所属 RADBench 事件名。

    Returns:
        tuple[str, bytes, bytes, int, int, str, float] | None: LMDB key、压缩后的特征矩阵、
        压缩后的feature-level nonzero mask、正负标签时间点数、事件名和在线等价计算时间。
        空 CSV 返回 None。
    """
    if _LMDB_PREPROCESS_CONTEXT is None or _LMDB_ZSTD_COMPRESSOR is None:
        raise RuntimeError("LMDB preprocess worker context has not been initialized")

    file_path, event_name = entry
    df = pd.read_csv(file_path)
    if df is None or df.empty:
        return None

    timestamps = _LMDB_PREPROCESS_CONTEXT["event_timestamps_map"][event_name]
    columns = _LMDB_PREPROCESS_CONTEXT["columns"]
    mask_columns = _LMDB_PREPROCESS_CONTEXT["mask_columns"]

    missing_raw_columns = [column for column in columns if column not in df.columns]
    if missing_raw_columns:
        raise KeyError(f"{file_path} missing columns required by LMDB schema: {missing_raw_columns}")

    pos_count, neg_count = count_real_label_points(df, timestamps)
    compute_started = time.perf_counter()
    nonzero_mask = build_feature_nonzero_mask(df, timestamps, mask_columns)
    df = data_completion(df, timestamps)

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{file_path} missing columns required by LMDB schema after completion: {missing_columns}")

    scalar_dict = _LMDB_PREPROCESS_CONTEXT["scalar_dict"]
    scaled_columns = [column for columns in _LMDB_PREPROCESS_CONTEXT["feature_groups"].values() for column in columns]
    if scaled_columns:
        # Scaler outputs floating-point values, so these columns must be float before
        # assigning transformed data back into the DataFrame.
        df[scaled_columns] = df[scaled_columns].astype(np.float64)

    for feature_name, feature_columns in _LMDB_PREPROCESS_CONTEXT["feature_groups"].items():
        values = df.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=False)
        transformed = scalar_dict[feature_name].transform(values.reshape(-1, 1)).reshape(values.shape)
        df.loc[:, feature_columns] = transformed

    data_arr = df.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
    online_compute_seconds = time.perf_counter() - compute_started
    cdata = _LMDB_ZSTD_COMPRESSOR.compress(data_arr.tobytes())
    cmask = _LMDB_ZSTD_COMPRESSOR.compress(nonzero_mask.tobytes())

    basename = os.path.basename(file_path)
    triplet_part = basename.replace(".csv", "").replace(",", "_")
    key = f"{event_name}_{triplet_part}"
    return key, cdata, cmask, pos_count, neg_count, event_name, online_compute_seconds


def calculate_standardization_params(target_folders_with_timestamps, scalar_save_path):
    """target_folders_with_timestamps: [(folder_path, event_ts_map, event_names), ...]"""
    all_tasks = []

    for folder, event_ts_map, event_names in target_folders_with_timestamps:
        for root, _, files in os.walk(folder):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                file_path = os.path.join(root, file)
                event_name = extract_event_name_from_path(file_path, event_names)
                if event_name is None or event_name not in event_ts_map:
                    continue
                timestamps = event_ts_map[event_name]
                if not timestamps:
                    continue
                all_tasks.append((file_path, timestamps))

    if not all_tasks:
        log.warning("No files found for standardization.")
        return

    sample_path, sample_ts = all_tasks[0]
    sample_df = process_file_completion(sample_path, sample_ts)
    if sample_df is None:
        log.error("Failed to read sample file.")
        return

    timeline_feature_names = defaultdict(list)
    for col in sample_df.columns:
        if col in ["timestamp", "label"]:
            continue
        if not col.endswith(("_0", "_1", "_2")):
            timeline_feature_names[col].append(col)
            continue
        base, suffix = col.rsplit("_", 1)
        if suffix in ("0", "1", "2"):
            timeline_feature_names[base].append(col)
    timeline_feature_names = {k: sorted(set(v)) for k, v in timeline_feature_names.items()}
    mask_columns = [c for c in sample_df.columns if c not in ["timestamp", "label"]]
    mask_column_index = {column: idx for idx, column in enumerate(mask_columns)}

    pipeline_steps_dict = defaultdict(list)
    std_dict = defaultdict(StandardScaler)

    for feature_name in timeline_feature_names:
        if feature_name == "AS_AVG_DIFF_ANN_LONGER":
            pipeline_steps_dict[feature_name].append(("pos", FunctionTransformer(np.negative)))
            pipeline_steps_dict[feature_name].append(("log1p", FunctionTransformer(np.log1p)))
            pipeline_steps_dict[feature_name].append(("neg", FunctionTransformer(np.negative)))
        else:
            pipeline_steps_dict[feature_name].append(("log1p", FunctionTransformer(np.log1p)))

        if feature_name in ["AS_AVG_PL", "AS_MPL"]:
            std_dict[feature_name] = StandardScaler()
        else:
            std_dict[feature_name] = MaxAbsScaler()

    prev_scalar_dict = {
        feature_name: Pipeline(steps=pipeline_steps_dict[feature_name])
        for feature_name in timeline_feature_names
    }

    # Producer-Consumer with Multiprocessing Queue to control memory usage
    manager = Manager()
    ts_lookup = {}
    for _, ts in all_tasks:
        ts_lookup[id(ts)] = ts
    shared_ts_dict = manager.dict(ts_lookup)

    file_queue = Queue()
    result_queue = Queue(maxsize=128)

    for file_path, ts in all_tasks:
        file_queue.put((file_path, id(ts)))

    num_workers = 8
    for _ in range(num_workers):
        file_queue.put(None)

    workers = []
    for _ in range(num_workers):
        p = Process(target=standardization_worker, args=(file_queue, result_queue, shared_ts_dict, mask_columns))
        p.start()
        workers.append(p)

    total_expected_messages = len(all_tasks) + num_workers

    fit_count_by_feature = defaultdict(int)

    with tqdm(total=len(all_tasks), desc="Calculating Std Params") as pbar:
        for _ in range(total_expected_messages):
            result = result_queue.get()
            if result is None:
                continue

            try:
                df, nonzero_mask = result
                if nonzero_mask.ndim != 2:
                    raise ValueError(f"nonzero_mask should be 2-D, got shape={nonzero_mask.shape}")
                if nonzero_mask.shape[0] != len(df) or nonzero_mask.shape[1] != len(mask_columns):
                    raise ValueError(
                        f"nonzero_mask shape={nonzero_mask.shape} does not match df rows={len(df)} "
                        f"and feature columns={len(mask_columns)}"
                    )
                if not np.any(nonzero_mask):
                    continue

                for feature_name in timeline_feature_names:
                    feature_parts = []
                    for feature_col in timeline_feature_names[feature_name]:
                        col_mask = nonzero_mask[:, mask_column_index[feature_col]].astype(bool, copy=False)
                        if not np.any(col_mask):
                            continue
                        feature_values = df.loc[:, feature_col].to_numpy(dtype=np.float64, copy=False)
                        feature_parts.append(feature_values[col_mask])

                    if not feature_parts:
                        continue

                    feature_data = np.concatenate(feature_parts).astype(np.float64, copy=False)
                    transformed_data = prev_scalar_dict[feature_name].transform(feature_data.reshape(-1, 1))
                    std_dict[feature_name].partial_fit(transformed_data)
                    fit_count_by_feature[feature_name] += int(feature_data.size)
            except Exception as e:
                log.error("Main process fitting error: %s", e)
            finally:
                pbar.update(1)

    for p in workers:
        p.join()

    missing_fit_features = [feature_name for feature_name in timeline_feature_names if fit_count_by_feature[feature_name] == 0]
    if missing_fit_features:
        raise RuntimeError(f"No nonzero training values found for scaler features: {missing_fit_features}")

    scalar_dict = {}
    for feature_name in timeline_feature_names:
        scalar_dict[feature_name] = Pipeline(
            steps=pipeline_steps_dict[feature_name] + [("std", copy.deepcopy(std_dict[feature_name]))]
        )

    os.makedirs(scalar_save_path, exist_ok=True)
    for feature_name in timeline_feature_names:
        joblib.dump(scalar_dict[feature_name], os.path.join(scalar_save_path, f"{feature_name}.pkl"))

    log.info("Standardization parameters saved to %s", scalar_save_path)


def collect_anomaly_triplets_files(event_names: list[str], raw_feature_root: str) -> dict[str, str]:
    """收集各事件的 AnomalyTriplets.txt 路径（仅存在的文件）。"""
    result = {}
    for event_name in event_names:
        path = os.path.join(raw_feature_root, event_name, "AnomalyTriplets.txt")
        if os.path.exists(path):
            result[event_name] = path
        else:
            log.warning("AnomalyTriplets.txt not found for event %s: %s", event_name, path)
    return result


def write_to_lmdb(
    target_folders,
    lmdb_path,
    scalar_save_path,
    event_timestamps_map,
    seq_len,
    event_names,
    negative_sample_ratio: float = 0.0,
    anomaly_triplets_files: dict[str, str] | None = None,
):
    """读取三元组 CSV，补全标准化后写入 LMDB，同时写入非零特征级 mask。

    Args:
        target_folders: 输入 timeline_feature 目录列表。
        lmdb_path: 输出 LMDB 路径。
        scalar_save_path: 标准化参数目录。
        event_timestamps_map: 每个事件的完整时间轴。
        seq_len: LMDB 序列长度元数据。
        event_names: 参与写入的事件名列表。
        negative_sample_ratio: 正负样本比例目标（0 表示不采样，保留全部负样本）。
        anomaly_triplets_files: {event_name: AnomalyTriplets.txt 路径}。

    Returns:
        None。每个三元组数据 key 写入完整序列，每个 mask key 写入 shape 为 `(T, D)`
        的 uint8 feature-level nonzero mask。
    """
    if os.path.exists(lmdb_path):
        if os.path.basename(lmdb_path) == "test.lmdb":
            recorder = get_active_recorder()
            if recorder is not None:
                for event_name in event_names:
                    recorder.mark_reused(f"preprocess.test.sequence.{event_name}")
        log.info("LMDB already exists, skipping write: %s", lmdb_path)
        return

    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    if os.path.basename(lmdb_path) == "test.lmdb":
        performance_group = "test"
    elif "Unsupervise" in Path(lmdb_path).parts:
        performance_group = "before3day"
    else:
        performance_group = "train"

    scalar_files = sorted(f for f in os.listdir(scalar_save_path) if f.endswith(".pkl"))
    scalar_dict = {}
    for f in scalar_files:
        feature_name = f[:-4]
        scalar_dict[feature_name] = joblib.load(os.path.join(scalar_save_path, f))

    all_entries = []
    for folder in target_folders:
        for root, _, files in os.walk(folder):
            for file in sorted(files):
                if not file.endswith(".csv"):
                    continue
                file_path = os.path.join(root, file)
                event_name = extract_event_name_from_path(file_path, event_names)
                if event_name is None:
                    continue
                if event_name not in event_timestamps_map:
                    continue
                all_entries.append((file_path, event_name))
    all_entries.sort()

    if not all_entries:
        log.warning("No files to write to LMDB: %s", lmdb_path)
        return

    # --- 采样逻辑 ---
    final_anomaly_triplets: dict[str, set[tuple[int, int, int]]] | None = None

    if anomaly_triplets_files:
        # 按事件聚合异常三元组，避免不同事件同名 triplet 互相污染
        anomaly_set_by_event: dict[str, set[str]] = defaultdict(set)
        for ev, fpath in anomaly_triplets_files.items():
            with open(fpath, "r") as f:
                for line in f:
                    parts = line.strip().split("_")
                    if len(parts) == 3:
                        anomaly_set_by_event[ev].add(f"{parts[0]},{parts[1]},{parts[2]}")

        pos_entries = []
        neg_entries = []
        for file_path, event_name in all_entries:
            triplet_str = os.path.basename(file_path).replace(".csv", "")
            if triplet_str in anomaly_set_by_event.get(event_name, set()):
                pos_entries.append((file_path, event_name))
            else:
                neg_entries.append((file_path, event_name))

        log.info("Found %d positive files and %d negative files.", len(pos_entries), len(neg_entries))

        if negative_sample_ratio > 0 and neg_entries:
            # 只用原始 CSV 中真实存在的标签点计算采样比例，避免补全点放大负样本数。
            total_pos_points = 0
            neg_in_pos = 0
            for fp, event_name in tqdm(pos_entries, desc="Scanning positive files"):
                df = pd.read_csv(fp)
                if df is None or df.empty:
                    continue
                pos_c, neg_c = count_real_label_points(df, event_timestamps_map[event_name])
                total_pos_points += pos_c
                neg_in_pos += neg_c

            neg_entries_with_counts = []
            total_neg_available = 0
            for fp, event_name in tqdm(neg_entries, desc="Scanning negative files"):
                df = pd.read_csv(fp)
                neg_c = 0
                if df is not None and not df.empty:
                    _, neg_c = count_real_label_points(df, event_timestamps_map[event_name])
                neg_entries_with_counts.append((fp, event_name, neg_c))
                total_neg_available += neg_c

            total_neg_target = int(total_pos_points * negative_sample_ratio)
            additional_neg_needed = max(0, total_neg_target - neg_in_pos)

            log.info(
                "Real anomaly timesteps: %d, real neg in pos files: %d, "
                "target real neg: %d, additional real neg needed: %d, available real neg: %d.",
                total_pos_points, neg_in_pos, total_neg_target, additional_neg_needed, total_neg_available,
            )

            if additional_neg_needed <= 0:
                neg_entries = []
                log.info("Positive files already satisfy target real negative ratio; keeping no extra negative files.")
            elif total_neg_available <= additional_neg_needed:
                log.info("Negative files have fewer real points than target, keeping all.")
            else:
                neg_entries_sorted = sorted(neg_entries_with_counts, key=lambda x: os.path.getsize(x[0]))
                n = len(neg_entries_sorted)
                idx1 = int(n * 0.1)
                idx2 = idx1 + int(n * 0.4)
                parts_list = [neg_entries_sorted[:idx1], neg_entries_sorted[idx1:idx2], neg_entries_sorted[idx2:]]
                sampled = []
                sampled_neg_points = 0

                while sampled_neg_points < additional_neg_needed and any(parts_list):
                    for part_idx in random.sample(range(len(parts_list)), len(parts_list)):
                        part = parts_list[part_idx]
                        while part:
                            fp, event_name, neg_c = part.pop(random.randrange(len(part)))
                            if neg_c <= 0:
                                continue
                            sampled.append((fp, event_name))
                            sampled_neg_points += neg_c
                            break
                        if sampled_neg_points >= additional_neg_needed:
                            break

                neg_entries = sampled
                log.info(
                    "Stratified sampled negative files to %d (ratio %.1f, real neg %d/%d).",
                    len(neg_entries), negative_sample_ratio, sampled_neg_points, additional_neg_needed,
                )

        all_entries = pos_entries + neg_entries
        random.shuffle(all_entries)

        # 构建 final_anomaly_triplets（按事件分组）
        final_anomaly_triplets = defaultdict(set)
        for fp, ev in pos_entries:
            triplet_str = os.path.basename(fp).replace(".csv", "")
            t_parts = triplet_str.split(",")
            if len(t_parts) == 3:
                try:
                    final_anomaly_triplets[ev].add(tuple(map(int, t_parts)))
                except ValueError:
                    pass
        final_anomaly_triplets = dict(final_anomaly_triplets)

    # --- 写入 LMDB ---
    sample_df = None
    for sample_file, sample_event in all_entries:
        sample_df = process_file_completion(sample_file, event_timestamps_map[sample_event])
        if sample_df is not None:
            break

    if sample_df is None:
        log.warning("No readable CSV files to write to LMDB: %s", lmdb_path)
        return

    env = lmdb.open(lmdb_path, map_size=500 * 1024**3)
    sample_cols = [c for c in sample_df.columns if c != "timestamp"]
    mask_columns = [c for c in sample_cols if c != "label"]
    feature_groups = build_lmdb_feature_groups(sample_cols, scalar_dict)
    worker_timestamps_map = {
        event_name: np.asarray(timestamps, dtype=np.int64)
        for event_name, timestamps in event_timestamps_map.items()
    }
    worker_context = {
        "scalar_dict": scalar_dict,
        "event_timestamps_map": worker_timestamps_map,
        "columns": sample_cols,
        "mask_columns": mask_columns,
        "feature_groups": feature_groups,
    }

    num_workers = max(1, min(8, os.cpu_count() or 1, len(all_entries)))
    commit_batch_size = 4096
    task_chunksize = max(1, min(128, len(all_entries) // (num_workers * 8) or 1))
    log.info(
        "Writing %d CSV files to %s with %d preprocess workers, chunksize=%d, commit_batch=%d",
        len(all_entries), lmdb_path, num_workers, task_chunksize, commit_batch_size,
    )

    txn = env.begin(write=True)
    pbar = None
    total_pos = 0
    total_neg = 0
    count = 0

    try:
        # Write Metadata
        txn.put(b"__seq_len__", pickle.dumps(int(seq_len)))
        txn.put(b"__columns__", pickle.dumps(sample_cols))
        txn.put(b"__event_names__", pickle.dumps(event_names))
        txn.put(b"__mask_type__", pickle.dumps(LMDB_MASK_TYPE))
        txn.put(b"__mask_shape__", pickle.dumps(LMDB_MASK_SHAPE))
        txn.put(b"__mask_columns__", pickle.dumps(mask_columns))
        for event_name, timestamps in event_timestamps_map.items():
            # Event-specific timestamps metadata
            meta_key = f"__{event_name}_timestamp__".encode("ascii")
            txn.put(meta_key, pickle.dumps(timestamps))

        pbar = tqdm(total=len(all_entries), desc=f"Writing to LMDB {lmdb_path}")
        recorder = get_active_recorder()
        resource_context = (
            recorder.track_resources(f"preprocess.{performance_group}.sequence_resources")
            if recorder is not None
            else nullcontext()
        )
        with resource_context, ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_lmdb_preprocess_worker,
            initargs=(worker_context,),
        ) as executor:
            for item in executor.map(process_lmdb_write_entry, all_entries, chunksize=task_chunksize):
                pbar.update(1)
                if item is None:
                    continue

                key, val, mask_val, pos_c, neg_c, event_name, online_compute_seconds = item
                recorder = get_active_recorder()
                if recorder is not None:
                    recorder.add_elapsed(
                        f"preprocess.{performance_group}.sequence.{event_name}",
                        online_compute_seconds,
                    )
                    recorder.add_count(
                        f"preprocess.{performance_group}.sequence.{event_name}",
                        "triplets",
                        1,
                    )
                txn.put(key.encode("ascii"), val)
                txn.put(f"mask_{key}".encode("ascii"), mask_val)
                total_pos += pos_c
                total_neg += neg_c
                count += 1

                if count % commit_batch_size == 0:
                    # Commit periodically to reduce long transaction pressure
                    txn.commit()
                    txn = env.begin(write=True)

        # Label stats metadata
        txn.put(b"__pos_label_count__", pickle.dumps(int(total_pos)))
        txn.put(b"__neg_label_count__", pickle.dumps(int(total_neg)))
        if final_anomaly_triplets is not None:
            txn.put(b"__anomaly_triplets__", pickle.dumps(final_anomaly_triplets))
        txn.commit()
        txn = None

    except Exception:
        if txn is not None:
            txn.abort()
        raise
    finally:
        if pbar is not None:
            pbar.close()
        env.close()

    log.info("Finished writing %d items to LMDB %s", count, lmdb_path)


def load_event_timestamps_map(raw_feature_root: str, event_names: list[str]):
    event_ts_map = {}
    for event_name in event_names:
        ts_path = os.path.join(raw_feature_root, event_name, "timestamps.txt")
        timestamps = get_total_timestamps(ts_path)
        if timestamps:
            event_ts_map[event_name] = timestamps
    return event_ts_map


def split_files_7_3(file_list: list[str]):
    random.shuffle(file_list)
    split_idx = int(len(file_list) * 0.7)
    return file_list[:split_idx], file_list[split_idx:]


def prepare_dataset_splits(supervised_events: list[dict], test_events: list[dict]):
    """准备监督/无监督/测试数据集切分目录。

    Args:
        supervised_events (list[dict]): 输入监督事件列表。
        test_events (list[dict]): 输入测试事件列表。

    Returns:
        dict: 各 split 目录与事件名信息。
    """
    supervised_event_names = [e["event_name"] for e in supervised_events]
    test_event_names = [e["event_name"] for e in test_events]

    supervised_raw_root = os.path.join(RADBENCH_OUTPUT_ROOT, "RawFeature")
    unsupervised_raw_root = os.path.join(RADBENCH_OUTPUT_ROOT, "RawFeature", "Before3Day")

    supervised_csv = collect_event_csv_files(supervised_raw_root, supervised_event_names)
    unsupervised_csv = collect_event_csv_files(unsupervised_raw_root, supervised_event_names)
    test_csv = collect_event_csv_files(supervised_raw_root, test_event_names)

    sup_train, sup_val = split_files_7_3(supervised_csv)
    unsup_train, unsup_val = split_files_7_3(unsupervised_csv)

    sup_train_dir = os.path.join(RADBENCH_OUTPUT_ROOT, "Supervise", "Train")
    sup_val_dir = os.path.join(RADBENCH_OUTPUT_ROOT, "Supervise", "Val")
    unsup_train_dir = os.path.join(RADBENCH_OUTPUT_ROOT, "Unsupervise", "Train")
    unsup_val_dir = os.path.join(RADBENCH_OUTPUT_ROOT, "Unsupervise", "Val")
    test_dir = os.path.join(RADBENCH_OUTPUT_ROOT, "Test")

    if os.path.exists(sup_train_dir):
        log.info("Split output already exists, skip: %s", sup_train_dir)
    else:
        os.makedirs(sup_train_dir, exist_ok=True)
        move_files(supervised_raw_root, sup_train, sup_train_dir)

    if os.path.exists(sup_val_dir):
        log.info("Split output already exists, skip: %s", sup_val_dir)
    else:
        os.makedirs(sup_val_dir, exist_ok=True)
        move_files(supervised_raw_root, sup_val, sup_val_dir)

    if os.path.exists(unsup_train_dir):
        log.info("Split output already exists, skip: %s", unsup_train_dir)
    else:
        os.makedirs(unsup_train_dir, exist_ok=True)
        move_files(unsupervised_raw_root, unsup_train, unsup_train_dir)

    if os.path.exists(unsup_val_dir):
        log.info("Split output already exists, skip: %s", unsup_val_dir)
    else:
        os.makedirs(unsup_val_dir, exist_ok=True)
        move_files(unsupervised_raw_root, unsup_val, unsup_val_dir)

    if os.path.exists(test_dir):
        log.info("Split output already exists, skip: %s", test_dir)
    else:
        os.makedirs(test_dir, exist_ok=True)
        move_files(supervised_raw_root, test_csv, test_dir)

    return {
        "supervised_event_names": supervised_event_names,
        "test_event_names": test_event_names,
        "sup_train_dir": sup_train_dir,
        "sup_val_dir": sup_val_dir,
        "unsup_train_dir": unsup_train_dir,
        "unsup_val_dir": unsup_val_dir,
        "test_dir": test_dir,
        "supervised_raw_root": supervised_raw_root,
        "unsupervised_raw_root": unsupervised_raw_root,
    }


def get_earliest_file_datetime(directory: str):
    """返回目录下 rrc update 文件中时间戳最早的 datetime，若目录不存在或无匹配文件则返回 None。"""
    if not os.path.isdir(directory):
        return None
    min_dt = None
    for fname in os.listdir(directory):
        if not (fname.startswith("rrc") and (fname.endswith(".gz") or fname.endswith(".txt"))):
            continue
        dt = parse_update_datetime(fname)
        if dt is None:
            continue
        if min_dt is None or dt < min_dt:
            min_dt = dt
    return min_dt


def prepare_event_raw_features(event: dict, mode: str, time_window: int = 60):
    """生成单个事件的 raw feature 目录。

    Args:
        event (dict): 输入事件元数据。
        mode (str): 输入模式，支持 unsupervise_before3d/supervise/test。
        time_window (int): 输入时间窗口大小（秒）。

    Returns:
        None
    """
    event_name = event["event_name"]
    group = {
        "unsupervise_before3d": "before3day",
        "supervise": "train",
        "test": "test",
    }.get(mode)
    if group is None:
        raise ValueError(f"Unknown mode: {mode}")
    performance_section = f"preprocess.{group}.raw.{event_name}"

    if mode == "unsupervise_before3d":
        raw_data_path = os.path.join(RADBENCH_INPUT_ROOT, "Before3Day", event_name)
        output_feature_path = os.path.join(RADBENCH_OUTPUT_ROOT, "RawFeature", "Before3Day", event_name)
        # 以主事件目录中最早文件时间 t 为基准，取 before3d 区间 [t-3d, t-5min]
        main_event_dir = os.path.join(RADBENCH_INPUT_ROOT, event_name)
        t = get_earliest_file_datetime(main_event_dir)
        if t is None:
            raise FileNotFoundError(f"Cannot find any update files in main event dir: {main_event_dir}")
        date_range = (t - datetime.timedelta(days=3), t - datetime.timedelta(minutes=5))
        ensure_updates_complete(date_range[0], date_range[1], raw_data_path)
    elif mode in ("supervise", "test"):
        raw_data_path = os.path.join(RADBENCH_INPUT_ROOT, event_name)
        output_feature_path = os.path.join(RADBENCH_OUTPUT_ROOT, "RawFeature", event_name)
        date_range = None  # 目录内文件已预选，直接读取所有文件
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 目录已存在时复用 RawFeature，并继续执行原地低频三元组过滤，保证续跑满足 no-sparse 约束。
    if os.path.exists(output_feature_path):
        recorder = get_active_recorder()
        if recorder is not None:
            recorder.mark_reused(performance_section)
        log.info("Raw feature directory already exists, skip feature extraction: %s", output_feature_path)
        filter_sparse_triplets_in_raw_feature(
            output_feature_path,
            performance_section=f"preprocess.{group}.sparse.{event_name}",
        )
        return

    ixp_asns = load_radbench_ixp_asns(event_name)
    anomaly_window_map = build_window_anomaly_map(
        event_name=event_name,
        leaker_as=event.get("leaker_as"),
        propagator_as_list=event.get("propagator_as", []),
        time_window=time_window,
        ixp_asns=ixp_asns,
    )

    os.makedirs(output_feature_path, exist_ok=True)
    extractor = FeatureExtractor(
        RawDataPath=raw_data_path,
        OutputFeaturePath=output_feature_path,
        time_window=time_window,
        date_range=date_range,
        pct_change=False,
        anomaly_window_map=anomaly_window_map,
        event_name=event_name,
        ixp_asns=ixp_asns,
        performance_section=performance_section,
    )
    recorder = get_active_recorder()
    resource_context = (
        recorder.track_resources(performance_section)
        if recorder is not None
        else nullcontext()
    )
    with resource_context:
        extractor.extract_timeline_feature()
    filter_sparse_triplets_in_raw_feature(
        output_feature_path,
        performance_section=f"preprocess.{group}.sparse.{event_name}",
    )


def prepare_all_event_features(supervised_events: list[dict], test_events: list[dict]):
    """直接从 RADBench 原始 update 生成并过滤 RawFeature。

    Args:
        supervised_events (list[dict]): 需要生成监督与 before3day RawFeature 的事件。
        test_events (list[dict]): 需要生成测试 RawFeature 的事件。

    Returns:
        None。RawFeature 输出到 RADBENCH_OUTPUT_ROOT，并移除低频三元组。
    """
    for event in supervised_events:
        log.info("Preparing unsupervised before3day raw feature: %s", event["event_name"])
        prepare_event_raw_features(event, mode="unsupervise_before3d")

    for event in supervised_events:
        log.info("Preparing supervised raw feature: %s", event["event_name"])
        prepare_event_raw_features(event, mode="supervise")

    for event in test_events:
        log.info("Preparing test raw feature: %s", event["event_name"])
        prepare_event_raw_features(event, mode="test")


def count_csv_data_rows(csv_path: str) -> int:
    """统计 RawFeature CSV 的原始数据行数。

    Args:
        csv_path (str): 输入三元组 CSV 文件路径。

    Returns:
        int: CSV 中除表头外的数据行数。
    """
    with open(csv_path, "rb") as f:
        line_count = sum(1 for _ in f)
    return max(line_count - 1, 0)


def parse_triplet_from_csv_path(csv_path: str) -> tuple[int, int, int]:
    """从 `A,B,C.csv` 文件名解析 AS 三元组。

    Args:
        csv_path (str): 输入三元组 CSV 文件路径。

    Returns:
        tuple[int, int, int]: 解析得到的 AS 三元组。
    """
    parts = os.path.basename(csv_path).replace(".csv", "").split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid triplet CSV filename: {csv_path}")
    return tuple(map(int, parts))


def load_anomaly_triplets_file(path: str) -> set[tuple[int, int, int]]:
    """读取 AnomalyTriplets.txt 中的异常三元组。

    Args:
        path (str): 输入 AnomalyTriplets.txt 文件路径。

    Returns:
        set[tuple[int, int, int]]: 文件中的异常三元组集合。
    """
    if not os.path.exists(path):
        log.warning("AnomalyTriplets.txt not found, use empty anomaly set: %s", path)
        return set()

    anomaly_triplets = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("_")
            if len(parts) != 3:
                continue
            anomaly_triplets.add(tuple(map(int, parts)))
    return anomaly_triplets


def write_sparse_filter_summary(
    event_dir: str,
    total_triplets: int,
    kept_triplets: set[tuple[int, int, int]],
    anomaly_triplets: set[tuple[int, int, int]],
):
    """写入人类可读的低频三元组过滤统计。

    Args:
        event_dir (str): 事件 RawFeature 目录。
        total_triplets (int): 过滤前 CSV 三元组总数。
        kept_triplets (set[tuple[int, int, int]]): 保留下来的三元组集合。
        anomaly_triplets (set[tuple[int, int, int]]): 原始异常三元组集合。

    Returns:
        None。
    """
    kept_anomaly_triplets = anomaly_triplets & kept_triplets
    summary_path = os.path.join(event_dir, "SparseFilterSummary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Sparse triplet filter summary\n")
        f.write(f"event_dir: {event_dir}\n")
        f.write(f"row_threshold: {SPARSE_TRIPLET_ROW_THRESHOLD}\n")
        f.write(f"total_triplets: {total_triplets}\n")
        f.write(f"kept_triplets: {len(kept_triplets)}\n")
        f.write(f"removed_sparse_triplets: {total_triplets - len(kept_triplets)}\n")
        f.write(f"source_anomaly_triplets: {len(anomaly_triplets)}\n")
        f.write(f"kept_anomaly_triplets: {len(kept_anomaly_triplets)}\n")
        f.write(f"removed_anomaly_triplets: {len(anomaly_triplets - kept_triplets)}\n")


def filter_sparse_triplets_in_raw_feature(
    event_dir: str, performance_section: str | None = None
):
    """在事件 RawFeature 目录内原地删除低频三元组并重写 AnomalyTriplets.txt。

    Args:
        event_dir (str): 事件 RawFeature 目录，内部必须包含 timeline_feature。
        performance_section (str, optional): 仅累计阈值判断计算的性能区间名。

    Returns:
        None。函数会删除观测行数小于 SPARSE_TRIPLET_ROW_THRESHOLD 的三元组 CSV。
    """
    if not os.path.isdir(event_dir):
        raise FileNotFoundError(f"RawFeature event directory not found: {event_dir}")

    feature_dir = os.path.join(event_dir, "timeline_feature")
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"timeline_feature directory not found: {feature_dir}")

    csv_files = []
    for root, _, files in os.walk(feature_dir):
        for file in sorted(files):
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    csv_files.sort()

    kept_triplets = set()

    def _filter_in_place(csv_path: str):
        row_count = count_csv_data_rows(csv_path)
        compute_started = time.perf_counter()
        triplet = parse_triplet_from_csv_path(csv_path)
        keep = row_count >= SPARSE_TRIPLET_ROW_THRESHOLD
        recorder = get_active_recorder()
        if recorder is not None and performance_section is not None:
            recorder.add_elapsed(
                performance_section, time.perf_counter() - compute_started
            )
            recorder.add_count(performance_section, "triplets", 1)
        if not keep:
            os.remove(csv_path)
            return triplet, False
        return triplet, True

    max_workers = min(8, os.cpu_count() or 1)
    batch_size = max_workers * 512
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(csv_files), desc=f"Filtering {os.path.basename(event_dir)}") as pbar:
            for start in range(0, len(csv_files), batch_size):
                batch = csv_files[start:start + batch_size]
                futures = [executor.submit(_filter_in_place, path) for path in batch]
                for future in as_completed(futures):
                    triplet, keep = future.result()
                    if keep:
                        kept_triplets.add(triplet)
                    pbar.update(1)

    delete_empty_files_and_dirs(feature_dir, suffix=".csv")

    anomaly_triplets = load_anomaly_triplets_file(os.path.join(event_dir, "AnomalyTriplets.txt"))
    target_anomaly_path = os.path.join(event_dir, "AnomalyTriplets.txt")
    with open(target_anomaly_path, "w", encoding="utf-8") as f:
        for triplet in sorted(anomaly_triplets & kept_triplets):
            f.write(f"{triplet[0]}_{triplet[1]}_{triplet[2]}\n")

    write_sparse_filter_summary(
        event_dir=event_dir,
        total_triplets=len(csv_files),
        kept_triplets=kept_triplets,
        anomaly_triplets=anomaly_triplets,
    )
    log.info(
        "Finished sparse triplet filtering: %s, total=%d, kept=%d, removed=%d",
        event_dir, len(csv_files), len(kept_triplets), len(csv_files) - len(kept_triplets),
    )


def get_lmdb_name(base_name: str, ratio: float) -> str:
    if ratio == 0:
        return f"{base_name}.lmdb"
    else:
        return f"{base_name}_ratio_{int(ratio)}.lmdb"


def main():
    """生成 RADBench 数据集目录与 LMDB。

    Args:
        None

    Returns:
        None
    """
    seed_everything(RADBENCH_RANDOM_SEED)
    log.info("Using RADBench raw update root: %s", RADBENCH_INPUT_ROOT)
    log.info("Using nonzero-mask RADBench output root: %s", RADBENCH_OUTPUT_ROOT)
    log.info("Using sparse triplet row threshold: %d", SPARSE_TRIPLET_ROW_THRESHOLD)
    log.info("Using RADBench event csv: %s", RADBENCH_EVENT_INFO_CSV)
    log.info("Using random seed: %d", RADBENCH_RANDOM_SEED)


    events = load_radbench_leak_events(RADBENCH_EVENT_INFO_CSV)
    supervised_events, test_events = split_events_for_datasets(events)
    recorder = get_active_recorder()
    if recorder is not None:
        recorder.set_summary(
            preprocess_expected_events={
                "before3day": [event["event_name"] for event in supervised_events],
                "train": [event["event_name"] for event in supervised_events],
                "test": [event["event_name"] for event in test_events],
            }
        )

    prepare_all_event_features(supervised_events, test_events)

    split_info = prepare_dataset_splits(supervised_events, test_events)

    supervised_event_names = split_info["supervised_event_names"]
    test_event_names = split_info["test_event_names"]

    sup_ts_map = load_event_timestamps_map(split_info["supervised_raw_root"], supervised_event_names)
    unsup_ts_map = load_event_timestamps_map(split_info["unsupervised_raw_root"], supervised_event_names)
    test_ts_map = load_event_timestamps_map(split_info["supervised_raw_root"], test_event_names)

    sup_seq_len = compute_min_seq_len(sup_ts_map)
    unsup_seq_len = compute_min_seq_len(unsup_ts_map)
    test_seq_len = compute_min_seq_len(test_ts_map)

    if sup_seq_len == 0 or unsup_seq_len == 0 or test_seq_len == 0:
        raise RuntimeError(
            f"Invalid seq_len detected: supervise={sup_seq_len}, unsupervise={unsup_seq_len}, test={test_seq_len}. "
            "Please check raw feature extraction outputs and timestamps.txt files."
        )

    sup_ts_map = {k: center_crop_timestamps(v, sup_seq_len) for k, v in sup_ts_map.items()}
    unsup_ts_map = {k: center_crop_timestamps(v, unsup_seq_len) for k, v in unsup_ts_map.items()}
    test_ts_map = {k: center_crop_timestamps(v, test_seq_len) for k, v in test_ts_map.items()}

    scalar_save_path = os.path.join(RADBENCH_OUTPUT_ROOT, "Scalers")
    if os.path.exists(scalar_save_path):
        log.info("Scaler directory already exists, skip recalculation: %s", scalar_save_path)
    else:
        calculate_standardization_params(
            [
                (split_info["unsup_train_dir"], unsup_ts_map, supervised_event_names),
                (split_info["sup_train_dir"], sup_ts_map, supervised_event_names),
            ],
            scalar_save_path=scalar_save_path,
        )

    if not os.path.exists(scalar_save_path) or not any(x.endswith(".pkl") for x in os.listdir(scalar_save_path)):
        raise RuntimeError(
            f"Scaler files are missing under {scalar_save_path}. "
            "Please prepare scalers first or remove the empty directory and rerun."
        )

    unsup_base = os.path.join(RADBENCH_OUTPUT_ROOT, "Unsupervise")
    sup_base = os.path.join(RADBENCH_OUTPUT_ROOT, "Supervise")
    test_base = os.path.join(RADBENCH_OUTPUT_ROOT, "Test")
    neg_sample_ratio = 10.0

    sup_anomaly_files = collect_anomaly_triplets_files(supervised_event_names, split_info["supervised_raw_root"])
    test_anomaly_files = collect_anomaly_triplets_files(test_event_names, split_info["supervised_raw_root"])

    write_to_lmdb(
        [split_info["unsup_train_dir"]],
        os.path.join(unsup_base, get_lmdb_name("train", 0)),
        scalar_save_path,
        unsup_ts_map,
        unsup_seq_len,
        supervised_event_names,
    )
    write_to_lmdb(
        [split_info["unsup_val_dir"]],
        os.path.join(unsup_base, get_lmdb_name("val", 0)),
        scalar_save_path,
        unsup_ts_map,
        unsup_seq_len,
        supervised_event_names,
    )
    write_to_lmdb(
        [split_info["sup_train_dir"]],
        os.path.join(sup_base, get_lmdb_name("train", 0)),
        scalar_save_path,
        sup_ts_map,
        sup_seq_len,
        supervised_event_names,
        negative_sample_ratio=0,
        anomaly_triplets_files=sup_anomaly_files,
    )
    write_to_lmdb(
        [split_info["sup_train_dir"]],
        os.path.join(sup_base, get_lmdb_name("train", neg_sample_ratio)),
        scalar_save_path,
        sup_ts_map,
        sup_seq_len,
        supervised_event_names,
        negative_sample_ratio=neg_sample_ratio,
        anomaly_triplets_files=sup_anomaly_files,
    )
    write_to_lmdb(
        [split_info["sup_val_dir"]],
        os.path.join(sup_base, get_lmdb_name("val", 0)),
        scalar_save_path,
        sup_ts_map,
        sup_seq_len,
        supervised_event_names,
        negative_sample_ratio=0,
        anomaly_triplets_files=sup_anomaly_files,
    )
    # write_to_lmdb(
    #     [split_info["sup_val_dir"]],
    #     os.path.join(sup_base, get_lmdb_name("val", neg_sample_ratio)),
    #     scalar_save_path,
    #     sup_ts_map,
    #     sup_seq_len,
    #     supervised_event_names,
    #     negative_sample_ratio=neg_sample_ratio,
    #     anomaly_triplets_files=sup_anomaly_files,
    # )
    write_to_lmdb(
        [split_info["test_dir"]],
        os.path.join(test_base, get_lmdb_name("test", 0)),
        scalar_save_path,
        test_ts_map,
        test_seq_len,
        test_event_names,
        negative_sample_ratio=0.0,
        anomaly_triplets_files=test_anomaly_files,
    )


if __name__ == "__main__":
    main()
