import datetime
import csv
import multiprocessing as mp
import os
import pickle
from collections import defaultdict

import lmdb
from tqdm import tqdm

from common import AnomalyStorage, AsPathDataClean, build_consecutive_triplets, compute_binary_metrics, decodeMRT, iter_announce_update_parts, load_radbench_ixp_asns, radbench_message_key


DEFAULT_RADBENCH_EVENT_INFO_CSV = 'radbench_leaks.csv'
_radbench_leak_event_cache: dict[str, dict[str, tuple[int, set[int]]]] = {}


def _resolve_repo_file(path: str) -> str:
    """解析仓库内配置文件路径。

    Args:
        path: 绝对路径或相对当前工作目录 / 仓库根目录的路径。

    Returns:
        已存在文件的绝对路径。
    """
    if os.path.isabs(path):
        resolved = path
    elif os.path.exists(path):
        resolved = os.path.abspath(path)
    else:
        resolved = os.path.join(os.path.dirname(__file__), path)

    if not os.path.exists(resolved):
        raise FileNotFoundError(f'RADBench event CSV not found: {path}')
    return resolved


def _load_radbench_leak_event_metadata(event_csv: str) -> dict[str, tuple[int, set[int]]]:
    """读取 RADBench route leak 事件的 leaker 与人工标注 propagator。

    Args:
        event_csv: RADBench 事件元数据 CSV。

    Returns:
        `{event_name: (leaker_as, propagator_as_set)}` 映射。
    """
    csv_path = _resolve_repo_file(event_csv)
    if csv_path in _radbench_leak_event_cache:
        return _radbench_leak_event_cache[csv_path]

    metadata = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('event_type') != 'leak':
                continue
            event_name = row.get('event_name', '').strip()
            leak_as_raw = row.get('leak_as', '').strip()
            propagator_raw = row.get('propagator_as', '').strip()
            if not event_name or not leak_as_raw:
                continue
            propagators = {int(asn) for asn in propagator_raw.split()} if propagator_raw else set()
            metadata[event_name] = (int(leak_as_raw), propagators)

    _radbench_leak_event_cache[csv_path] = metadata
    return metadata


def _has_annotated_propagator_left_of_leaker(
    parts: list[str],
    leaker_as: int,
    propagator_as_set: set[int],
    ixp_asns: set[int] | frozenset[int],
) -> bool:
    """判断 GT 消息是否匹配人工标注的 propagator-leaker 邻接。

    Args:
        parts: bgpdump `BGP4MP` line split by `|`.
        leaker_as: 事件标注的 leaker AS。
        propagator_as_set: 事件标注的 propagator AS 集合。
        ixp_asns: 需要从 AS_PATH 中移除的 IXP / Route Server ASN 集合。

    Returns:
        True 表示清洗后的 AS_PATH 中 leaker 左侧一跳是标注 propagator。
    """
    if len(parts) < 7 or parts[2] != 'A':
        return False

    clean_as_path = AsPathDataClean(parts[6].split(), ixp_asns=ixp_asns)
    if not clean_as_path:
        return False
    try:
        leaker_idx = clean_as_path.index(leaker_as)
    except ValueError:
        return False
    if leaker_idx == 0:
        return False
    return clean_as_path[leaker_idx - 1] in propagator_as_set


def load_slot_triplet_anomaly_map(filepath):
    """读取 CF 异常标注，返回 {time_str -> set(triplet_tuple)}。"""
    anomaly_map = defaultdict(set)
    if not os.path.exists(filepath):
        return anomaly_map

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 2 or not parts[0].startswith('rrc_'):
                continue
            time_str = parts[0].replace('rrc_', '')
            for triplet_str in parts[1].split(','):
                triplet = tuple(triplet_str.split('_'))
                if len(triplet) == 3:
                    anomaly_map[time_str].add(triplet)
    return anomaly_map


def query_prediction_slot_triplets(db_path, time_str):
    """按 5 分钟槽位从 DuckDB 查询预测 triplet。"""
    if not db_path:
        return defaultdict(set)
    return AnomalyStorage.query_slot_triplets_from_file(db_path, time_str)


def iter_time_range_update_files(min_ts, max_ts, updates_dir, collector='rrc00'):
    """按 5 分钟步进枚举 CF update 文件。"""
    current_dt = datetime.datetime.fromtimestamp(min_ts, tz=datetime.timezone.utc)
    end_dt = datetime.datetime.fromtimestamp(max_ts, tz=datetime.timezone.utc)
    current_dt = current_dt.replace(
        minute=current_dt.minute - (current_dt.minute % 5),
        second=0,
        microsecond=0,
    )

    target_files = []
    while current_dt <= end_dt:
        time_str = current_dt.strftime('%Y%m%d.%H%M')
        filepath = os.path.join(updates_dir, f'{collector}_updates.{time_str}.gz')
        if os.path.exists(filepath):
            target_files.append((time_str, filepath))
        current_dt += datetime.timedelta(minutes=5)
    return target_files


def process_cf_update_file_core(item, pred_db_path, gt_anomalies):
    """处理单个 CF update 文件，返回消息级 tp/fp/tn/fn。"""
    time_str, filepath = item
    msgs = decodeMRT(filepath)
    pred_anomalies_for_time = query_prediction_slot_triplets(pred_db_path, time_str)
    gt_triplets_for_time = gt_anomalies.get(time_str, set())

    local_tp = local_fp = local_tn = local_fn = 0
    for parts in iter_announce_update_parts(msgs):
        msg_ts = int(parts[1])
        as_path_str = parts[6]
        if not as_path_str:
            continue

        as_path_raw = as_path_str.split()

        is_anomaly_gt = 0
        if gt_triplets_for_time:
            for i in range(len(as_path_raw) - 2):
                if tuple(as_path_raw[i:i + 3]) in gt_triplets_for_time:
                    is_anomaly_gt = 1
                    break

        is_anomaly_pred = 0
        as_path_clean = AsPathDataClean(as_path_raw)
        if as_path_clean and len(as_path_clean) >= 3:
            pred_triplets_at_ts = pred_anomalies_for_time.get(msg_ts // 60 * 60, set())
            if pred_triplets_at_ts:
                for i in range(len(as_path_clean) - 2):
                    if tuple(as_path_clean[i:i + 3]) in pred_triplets_at_ts:
                        is_anomaly_pred = 1
                        break

        if is_anomaly_gt == 1 and is_anomaly_pred == 1:
            local_tp += 1
        elif is_anomaly_gt == 0 and is_anomaly_pred == 1:
            local_fp += 1
        elif is_anomaly_gt == 1 and is_anomaly_pred == 0:
            local_fn += 1
        else:
            local_tn += 1

    return local_tp, local_fp, local_tn, local_fn


def run_cf_raw_update_evaluation(target_files, worker_fn, task_timeout=300, max_processes=6):
    """并行执行 CF raw update 评估并聚合消息级混淆矩阵。"""
    tp = fp = tn = fn = 0
    num_processes = min(max_processes, len(target_files)) if target_files else 1
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=num_processes) as pool:
        futures = [(item, pool.apply_async(worker_fn, (item,))) for item in target_files]
        progress = tqdm(total=len(target_files), desc='Processing Raw Updates & Calculating Metrics')
        for item, future in futures:
            try:
                local_tp, local_fp, local_tn, local_fn = future.get(timeout=task_timeout)
                tp += local_tp
                fp += local_fp
                tn += local_tn
                fn += local_fn
            except mp.TimeoutError:
                pass
            finally:
                progress.update(1)
        progress.close()
    return tp, fp, tn, fn


def load_triplet_label_maps_from_lmdb(lmdb_path, is_radbench):
    """从测试 LMDB 读取全量 triplet 与 GT triplet。"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    try:
        with env.begin() as txn:
            all_triplets_map = defaultdict(set)
            anomaly_triplets_raw = txn.get(b'__anomaly_triplets__')
            if anomaly_triplets_raw is None:
                raise ValueError(f'__anomaly_triplets__ metadata not found in {lmdb_path}')
            anomaly_triplets = pickle.loads(anomaly_triplets_raw)

            if is_radbench:
                event_names_raw = txn.get(b'__event_names__')
                if event_names_raw is None:
                    raise ValueError(f'__event_names__ metadata not found in {lmdb_path}')
                event_names = pickle.loads(event_names_raw)
                gt_triplets_map = defaultdict(set)
                for event_name in event_names:
                    gt_triplets_map[event_name] = set(anomaly_triplets.get(event_name, set()))
                for key_b in txn.cursor().iternext(keys=True, values=False):
                    if key_b.startswith(b'__') and key_b.endswith(b'__'):
                        continue
                    key_str = key_b.decode('ascii')
                    parts = key_str.rsplit('_', 3)
                    if len(parts) != 4:
                        continue
                    event_name = parts[0]
                    triplet = tuple(map(int, parts[1:]))
                    all_triplets_map[event_name].add(triplet)
            else:
                gt_triplets_map = {None: set(anomaly_triplets)}
                for key_b in txn.cursor().iternext(keys=True, values=False):
                    if key_b.startswith(b'__') and key_b.endswith(b'__'):
                        continue
                    key_str = key_b.decode('ascii')
                    parts = key_str.split('_')
                    if len(parts) != 3:
                        continue
                    all_triplets_map[None].add(tuple(map(int, parts)))
        return dict(all_triplets_map), gt_triplets_map
    finally:
        env.close()


def load_unique_predicted_triplets_from_duckdb(db_path_or_map, is_radbench):
    """从 DuckDB 读取唯一预测 triplet。"""
    if is_radbench:
        pred_triplets_map = {}
        for event_name, db_path in db_path_or_map.items():
            pred_triplets_map[event_name] = AnomalyStorage.load_unique_triplets_from_file(db_path)
        return pred_triplets_map
    return {None: AnomalyStorage.load_unique_triplets_from_file(db_path_or_map)}


def compute_triplet_metrics(all_triplets_map, gt_triplets_map, pred_triplets_map):
    """计算 triplet 级别指标。"""
    event_metrics = {}
    total_tp = total_fp = total_tn = total_fn = 0
    groups = set(all_triplets_map.keys()) | set(gt_triplets_map.keys()) | set(pred_triplets_map.keys())
    for group in groups:
        all_triplets = all_triplets_map.get(group, set())
        gt_triplets = gt_triplets_map.get(group, set())
        pred_triplets = pred_triplets_map.get(group, set())
        tp = len(pred_triplets & gt_triplets)
        fp = len(pred_triplets - gt_triplets)
        fn = len(gt_triplets - pred_triplets)
        tn = len(all_triplets - (pred_triplets | gt_triplets))
        metrics = compute_binary_metrics(tp, fp, tn, fn)
        event_metrics[group] = metrics
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
    overall_metrics = compute_binary_metrics(total_tp, total_fp, total_tn, total_fn)
    return event_metrics, overall_metrics


def build_triplet_metrics_for_single_db(test_lmdb_path, pred_db_path):
    """计算 CF 单库 triplet 指标，并返回唯一预测 triplet 数。"""
    all_triplets_map, gt_triplets_map = load_triplet_label_maps_from_lmdb(test_lmdb_path, is_radbench=False)
    pred_triplets_map = load_unique_predicted_triplets_from_duckdb(pred_db_path, is_radbench=False)
    _, overall_triplet_metrics = compute_triplet_metrics(all_triplets_map, gt_triplets_map, pred_triplets_map)
    predicted_triplet_count = len(pred_triplets_map.get(None, set()))
    return overall_triplet_metrics, predicted_triplet_count


def load_radbench_gt_msg_set(event_name, gt_dir, event_csv=DEFAULT_RADBENCH_EVENT_INFO_CSV):
    """读取单个 RADBench 事件的 GT 消息集合，并按标注 propagator 过滤。

    Args:
        event_name: RADBench 事件名。
        gt_dir: GT message 根目录。
        event_csv: RADBench 事件元数据 CSV，提供 `leak_as` 与 `propagator_as`。

    Returns:
        仅包含 leaker 左侧一跳为标注 propagator 的 `(timestamp, prefix, as_path)` 集合。
    """
    metadata = _load_radbench_leak_event_metadata(event_csv)
    if event_name not in metadata:
        raise KeyError(f'RADBench leak event not found in {event_csv}: {event_name}')

    leaker_as, propagator_as_set = metadata[event_name]
    if not propagator_as_set:
        raise ValueError(f'RADBench leak event has no annotated propagator_as: {event_name}')

    gt_path = os.path.join(gt_dir, f'{event_name}.txt')
    ixp_asns = load_radbench_ixp_asns(event_name)
    gt_set = set()
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split('|')
            if _has_annotated_propagator_left_of_leaker(
                parts,
                leaker_as=leaker_as,
                propagator_as_set=propagator_as_set,
                ixp_asns=ixp_asns,
            ):
                gt_set.add(radbench_message_key(parts))
    return gt_set


def get_radbench_test_event_metadata(test_lmdb_path):
    """读取 RADBench 测试集事件名与时间戳元数据。"""
    temp_env = lmdb.open(test_lmdb_path, readonly=True, lock=False)
    with temp_env.begin() as txn:
        event_names = pickle.loads(txn.get(b'__event_names__'))
        event_timestamps = {}
        for event_name in event_names:
            ts_raw = txn.get(f'__{event_name}_timestamp__'.encode('ascii'))
            if ts_raw is not None:
                event_timestamps[event_name] = pickle.loads(ts_raw)
    temp_env.close()
    return event_names, event_timestamps


def collect_radbench_target_files(event_names, event_timestamps, updates_dir):
    """按事件枚举 RADBench raw update 文件。"""
    target_files = []
    for event_name in event_names:
        ts_list = event_timestamps.get(event_name, [])
        if not ts_list:
            continue
        min_ts = min(ts_list)
        max_ts = max(ts_list)
        current_dt = datetime.datetime.fromtimestamp(min_ts, tz=datetime.timezone.utc)
        end_dt = datetime.datetime.fromtimestamp(max_ts, tz=datetime.timezone.utc)
        current_dt = current_dt.replace(
            minute=current_dt.minute - (current_dt.minute % 5),
            second=0,
            microsecond=0,
        )
        while current_dt <= end_dt:
            time_str = current_dt.strftime('%Y%m%d.%H%M')
            filepath = os.path.join(updates_dir, event_name, f'rrc00_updates.{time_str}.gz')
            if os.path.exists(filepath):
                target_files.append((event_name, time_str, filepath))
            current_dt += datetime.timedelta(minutes=5)
    return target_files


def process_radbench_update_file_core(item, pred_db_paths, gt_msg_sets):
    """处理单个 RADBench update 文件，返回消息级 tp/fp/tn/fn。"""
    event_name, time_str, filepath = item
    msgs = decodeMRT(filepath)
    pred_db_path = pred_db_paths.get(event_name, '')
    pred_anomalies_for_time = query_prediction_slot_triplets(pred_db_path, time_str)
    gt_msg_set = gt_msg_sets.get(event_name, set())
    ixp_asns = load_radbench_ixp_asns(event_name)

    local_tp = local_fp = local_tn = local_fn = 0
    for parts in iter_announce_update_parts(msgs):
        msg_ts_str = parts[1]
        prefix_str = parts[5]
        as_path_str = parts[6]
        if not as_path_str:
            continue

        as_path_raw = as_path_str.split()

        is_anomaly_gt = 1 if radbench_message_key(parts) in gt_msg_set else 0
        is_anomaly_pred = 0
        as_path_clean = AsPathDataClean(as_path_raw, ixp_asns=ixp_asns)
        if as_path_clean and len(as_path_clean) >= 3:
            msg_ts_int = int(msg_ts_str)
            pred_triplets_at_ts = pred_anomalies_for_time.get(msg_ts_int // 60 * 60, set())
            if pred_triplets_at_ts:
                triplets_in_msg = build_consecutive_triplets(as_path_clean)
                if triplets_in_msg & pred_triplets_at_ts:
                    is_anomaly_pred = 1

        if is_anomaly_gt == 1 and is_anomaly_pred == 1:
            local_tp += 1
        elif is_anomaly_gt == 0 and is_anomaly_pred == 1:
            local_fp += 1
        elif is_anomaly_gt == 1 and is_anomaly_pred == 0:
            local_fn += 1
        else:
            local_tn += 1

    return local_tp, local_fp, local_tn, local_fn


def run_radbench_event_raw_update_evaluation(event_names, target_files, worker_fn, task_timeout=300, max_processes=6):
    """并行执行 RADBench raw update 评估并按事件聚合混淆矩阵。"""
    event_cm = {event_name: [0, 0, 0, 0] for event_name in event_names}
    num_processes = min(max_processes, len(target_files)) if target_files else 1
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=num_processes) as pool:
        futures = [(item, pool.apply_async(worker_fn, (item,))) for item in target_files]
        progress = tqdm(total=len(target_files), desc='Processing Raw Updates & Calculating Metrics')
        for item, future in futures:
            event_name = item[0]
            try:
                local_tp, local_fp, local_tn, local_fn = future.get(timeout=task_timeout)
                event_cm[event_name][0] += local_tp
                event_cm[event_name][1] += local_fp
                event_cm[event_name][2] += local_tn
                event_cm[event_name][3] += local_fn
            except mp.TimeoutError:
                pass
            finally:
                progress.update(1)
        progress.close()
    return event_cm


def build_radbench_triplet_metrics(test_lmdb_path, pred_db_paths):
    """计算 RADBench triplet 指标，并返回唯一预测 triplet 总数。"""
    all_triplets_map, gt_triplets_map = load_triplet_label_maps_from_lmdb(test_lmdb_path, is_radbench=True)
    pred_triplets_map = load_unique_predicted_triplets_from_duckdb(pred_db_paths, is_radbench=True)
    event_triplet_metrics, overall_triplet_metrics = compute_triplet_metrics(
        all_triplets_map,
        gt_triplets_map,
        pred_triplets_map,
    )
    predicted_triplet_count = sum(len(triplets) for triplets in pred_triplets_map.values())
    return event_triplet_metrics, overall_triplet_metrics, predicted_triplet_count
