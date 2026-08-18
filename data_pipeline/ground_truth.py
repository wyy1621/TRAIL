import os
import datetime
import logging
import csv
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
from tqdm import tqdm
import dotenv
import pytricia
import json
from concurrent.futures import ThreadPoolExecutor

from common import beijing, read_RADBench_event_list, RADBenchEvent
from data_pipeline.ris_download import set_base_params, getURLList, downloadByParams

logging.Formatter.converter = beijing
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s', force=True)
log = logging.getLogger(__name__)

dotenv.load_dotenv()

events = read_RADBench_event_list(event_list_path='radbench_leaks.csv', event_type='leak')
events = [RADBenchEvent(event) for event in events]

# Configuration
RIB_STORAGE_DIR = "DATA/Raw/RIB"
UPDATE_STORAGE_DIR = "DATA/Raw/PreEventUpdates"
ROUTE_TABLE_DIR = "DATA/Processed/RouteTables"
EVENT_UPDATE_DIR_BASE = "DATA/Raw/RADBench"
OUTPUT_DIR = "DATA/Processed/GroundTruth"
WITHDRAW_DIR = os.path.join(OUTPUT_DIR, "Withdraws")
PROPAGATOR_DIR = os.path.join(OUTPUT_DIR, "Propagator")
UPDATE_DECODE_WORKERS = 8
COMPRESSED_MRT_MAGIC = {
    ".gz": b"\x1f\x8b",
    ".bz2": b"BZh",
}

# "MERGED": Download all RIBs in 24h and merge
# "SIMULATION": Download initial RIB (T-24h) + Updates (T-24h to T), simulate accumulation
RIB_GENERATION_MODE = "SIMULATION" 

os.makedirs(RIB_STORAGE_DIR, exist_ok=True)
os.makedirs(UPDATE_STORAGE_DIR, exist_ok=True)
os.makedirs(ROUTE_TABLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WITHDRAW_DIR, exist_ok=True)
os.makedirs(PROPAGATOR_DIR, exist_ok=True)


def is_complete_compressed_mrt(filepath: str) -> bool:
    """Check whether a downloaded compressed MRT file is valid enough to reuse.

    Args:
        filepath: Local `.gz` or `.bz2` MRT file path.

    Returns:
        True when the path is a non-empty regular file whose magic bytes match
        its compression suffix; otherwise False.
    """
    suffix = os.path.splitext(filepath)[1].lower()
    expected_magic = COMPRESSED_MRT_MAGIC.get(suffix)
    if expected_magic is None or not os.path.isfile(filepath):
        return False

    try:
        if os.path.getsize(filepath) <= 0:
            return False
        with open(filepath, "rb") as file_obj:
            return file_obj.read(len(expected_magic)) == expected_magic
    except OSError as exc:
        log.warning("Failed to validate compressed MRT file %s: %s", filepath, exc)
        return False


def list_invalid_downloads(directory: str, filenames: list[str]) -> list[str]:
    """List missing or invalid compressed MRT downloads.

    Args:
        directory: Directory containing downloaded MRT files.
        filenames: Expected local filenames.

    Returns:
        Expected filenames that are absent, empty, or have invalid magic bytes.
    """
    return [
        filename
        for filename in filenames
        if not is_complete_compressed_mrt(os.path.join(directory, filename))
    ]


def list_event_update_files(event_name):
    """List raw RADBench update files for one event.

    Args:
        event_name: RADBench event name.

    Returns:
        Sorted absolute file paths. The order is the processing order used by labeling.
    """
    event_dir = os.path.join(EVENT_UPDATE_DIR_BASE, event_name)
    if not os.path.isdir(event_dir):
        log.error(f"Update directory not found: {event_dir}")
        return []

    files = [
        os.path.join(event_dir, filename)
        for filename in os.listdir(event_dir)
        if filename.startswith("rrc00_updates")
        and (filename.endswith(".txt") or filename.endswith(".gz") or filename.endswith(".bz2"))
    ]
    files.sort()
    return files


def iter_mrt_lines(filepath):
    """Stream decoded MRT lines without materializing the full bgpdump output.

    Args:
        filepath: Raw MRT file path (.gz/.bz2), or a decoded .txt file path.

    Yields:
        Decoded bgpdump text lines without trailing newlines.
    """
    if filepath.endswith(".txt"):
        with open(filepath, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield line
        return

    process = subprocess.Popen(
        ["bgpdump", "-q", "-m", filepath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError(f"Failed to open bgpdump streams for {filepath}")

    completed = False
    try:
        for line in process.stdout:
            line = line.rstrip("\n")
            if line:
                yield line
        completed = True
    finally:
        if not completed and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        stderr = process.stderr.read()
        return_code = process.wait()
        if completed and return_code != 0:
            message = stderr.strip() or "no stderr"
            raise RuntimeError(f"bgpdump failed for {filepath}: {message}")


def decode_update_file(filepath):
    """Decode one update file into bgpdump text lines.

    Args:
        filepath: Raw MRT update file path, or a decoded text file path.

    Returns:
        A list of decoded update lines in the original file order.
    """
    return list(iter_mrt_lines(filepath))


def iter_decoded_update_files(filepaths, max_workers=UPDATE_DECODE_WORKERS):
    """Decode update files concurrently while yielding results in file order.

    Args:
        filepaths: Sorted update file paths.
        max_workers: Number of decoding threads. It is capped at 16 to match the host limit.

    Yields:
        Tuples of (filepath, decoded_lines), ordered exactly as the input file list.
    """
    if not filepaths:
        return

    worker_count = min(max(1, int(max_workers)), 16, len(filepaths))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        next_submit_idx = 0
        futures = {}

        while next_submit_idx < worker_count:
            futures[next_submit_idx] = executor.submit(decode_update_file, filepaths[next_submit_idx])
            next_submit_idx += 1

        for consume_idx in range(len(filepaths)):
            decoded_lines = futures.pop(consume_idx).result()
            if next_submit_idx < len(filepaths):
                futures[next_submit_idx] = executor.submit(decode_update_file, filepaths[next_submit_idx])
                next_submit_idx += 1
            yield filepaths[consume_idx], decoded_lines


def prepare_merged_rib(event_name, event_start_str):
    """
    Download RIBs in [T-24h, T], decode, merge into one file using MERGED format.
    Returns path to merged RIB file.
    """
    merged_path = os.path.join(RIB_STORAGE_DIR, f"{event_name}.txt")
    if os.path.exists(merged_path) and os.path.getsize(merged_path) > 0:
        log.info(f"Merged RIB already exists for {event_name}: {merged_path}")
        return merged_path

    try:
        dt = datetime.datetime.strptime(event_start_str, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        dt = datetime.datetime.strptime(event_start_str, "%Y/%m/%d %H:%M").replace(tzinfo=datetime.timezone.utc)

    # 24h window before event
    end_dt = dt
    start_dt = end_dt - datetime.timedelta(hours=24)
    
    start_s = start_dt.strftime("%Y-%m-%d-%H:%M")
    end_s = end_dt.strftime("%Y-%m-%d-%H:%M")
    
    log.info(f"Preparing RIBs for {event_name} from {start_s} to {end_s}...")
    
    # 1. Download
    bp = set_base_params(start_s, end_s, data_types="BGP:RIBS", collector="rrc00", minio_flag=False)
    urllist = getURLList(bp)
    
    target_urls = []
    # Identify expected filenames to track them
    expected_files = []
    
    for _, url in urllist:
        if "bview" in url or "rib" in url:
            filename = url.split('/')[-1]
            # Compatibility downloader saves as rrc00_{filename}
            saved_filename = f"rrc00_{filename}"
            expected_files.append(saved_filename)
            
            local_path = os.path.join(RIB_STORAGE_DIR, saved_filename)
            if not is_complete_compressed_mrt(local_path):
                if os.path.exists(local_path):
                    log.warning(f"Existing RIB file is invalid and will be redownloaded: {local_path}")
                target_urls.append((_, url))
            else:
                log.info(f"Valid RIB file already exists, skipping download: {saved_filename}")
            
    if not expected_files:
         log.error(f"No RIB URLs found between {start_s} and {end_s}")
         return None

    if target_urls:
        log.info(f"Downloading {len(target_urls)} missing RIB files...")
        downloadByParams(
            base_type=bp.base_type,
            urllist=target_urls,
            destination=RIB_STORAGE_DIR,
            minio_enable=False,
        ).start_on()
    else:
        log.info("All required RIB files are already present locally.")

    invalid_files = list_invalid_downloads(RIB_STORAGE_DIR, expected_files)
    if invalid_files:
        preview = ", ".join(invalid_files[:10])
        raise RuntimeError(
            f"Missing or invalid RIB files after download: "
            f"{len(invalid_files)}/{len(expected_files)}. First files: {preview}"
        )
    
    # 2. Merge
    # Use a dictionary to store unique paths for each prefix
    # data[prefix] = set(as_path_str)
    # Using a simple dict might be memory intensive but for < 1M prefixes * ~5 paths it should fit in RAM of a decent server.
    master_rib = {}
    
    log.info(f"Merging RIB files for {event_name}...")
    
    # Check downloaded files in RIB_STORAGE_DIR
    found_files = sorted(
        os.path.join(RIB_STORAGE_DIR, filename)
        for filename in expected_files
        if is_complete_compressed_mrt(os.path.join(RIB_STORAGE_DIR, filename))
    )
    
    if not found_files:
        log.error("No RIB files found after download attempt.")
        return None
        
    for i, raw_path in enumerate(found_files):
        log.info(f"Processing {i+1}/{len(found_files)}: {raw_path}...")
        try:
            for line in tqdm(iter_mrt_lines(raw_path), desc=f"Decoding {os.path.basename(raw_path)}"):
                parts = line.strip().split('|')
                # TABLE_DUMP2|ts|B|peer_ip|peer_as|prefix|aspath|...
                if len(parts) < 7: continue
                if not parts[0].startswith("TABLE_DUMP"): continue
                
                prefix = parts[5]
                if ':' in prefix: continue
                if prefix == '0.0.0.0/0': continue
                
                as_path = parts[6]
                if not as_path: continue
                
                if prefix not in master_rib:
                    master_rib[prefix] = set()
                master_rib[prefix].add(as_path)
            
            # Note: We keep the raw file as requested by user
            
        except Exception as e:
            log.error(f"Failed to decode/process {raw_path}: {e}")
    
    # 3. Save Merged
    log.info(f"Saving merged RIB to {merged_path}...")
    with open(merged_path, 'w') as f:
        for prefix, paths in master_rib.items():
            for path in paths:
                f.write(f"MERGED|{prefix}|{path}\n")
                
    del master_rib # Free memory
    return merged_path

def prepare_simulated_rib(event_name, event_start_str):
    """
    1. Download latest RIB before T-24h.
    2. Download Updates in [T-24h, T].
    3. Simulate routing table (accumulation mode):
       - Load Initial RIB.
       - Replay Updates: For every Announcement, add path to set. Ignore Withdrawals.
       - Save result to DATA/RouteTable/{event_name}.txt
    """
    route_table_path = os.path.join(ROUTE_TABLE_DIR, f"{event_name}.txt")
    if os.path.exists(route_table_path) and os.path.getsize(route_table_path) > 0:
        log.info(f"Route Table already exists for {event_name}: {route_table_path}")
        return route_table_path

    try:
        dt = datetime.datetime.strptime(event_start_str, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        dt = datetime.datetime.strptime(event_start_str, "%Y/%m/%d %H:%M").replace(tzinfo=datetime.timezone.utc)

    # Time window
    # end_dt = dt
    end_dt = dt - datetime.timedelta(hours=1) # End 1h before event start
    start_dt = dt - datetime.timedelta(hours=25) # Start 25h before event start (to ensure we get a RIB before T-24h)
    
    start_s = start_dt.strftime("%Y-%m-%d-%H:%M")
    end_s = end_dt.strftime("%Y-%m-%d-%H:%M")
    
    log.info(f"[SIMULATION] Preparing data for {event_name} ({start_s} to {end_s})...")
    
    # --- Step 1: Download Initial RIB (Nearest before start_dt) ---
    # We search in [start_dt - 8h, start_dt] to be safe
    rib_search_start = (start_dt - datetime.timedelta(hours=8)).strftime("%Y-%m-%d-%H:%M")
    
    bp_rib = set_base_params(rib_search_start, start_s, data_types="BGP:RIBS", collector="rrc00", minio_flag=False)
    rib_urllist = getURLList(bp_rib)
    
    rib_url = None
    # Pick the last one (latest before start_dt)
    valid_ribs = [(_, u) for _, u in rib_urllist if "bview" in u or "rib" in u]
    if valid_ribs:
        rib_url = valid_ribs[-1]
    
    if not rib_url:
        log.error(f"No initial RIB found before {start_s}")
        return None
        
    rib_filename = rib_url[1].split('/')[-1]
    saved_rib_filename = f"rrc00_{rib_filename}"
    local_rib_path = os.path.join(RIB_STORAGE_DIR, saved_rib_filename)
    
    if not is_complete_compressed_mrt(local_rib_path):
        if os.path.exists(local_rib_path):
            log.warning(f"Existing initial RIB is invalid and will be redownloaded: {local_rib_path}")
        log.info(f"Downloading Initial RIB: {rib_filename}")
        downloadByParams(
            base_type=bp_rib.base_type, # Dummy, not used by downloadByParams really if list provided
            urllist=[rib_url],
            destination=RIB_STORAGE_DIR,
            minio_enable=False
        ).start_on()
    else:
        log.info(f"Valid initial RIB exists locally: {saved_rib_filename}")

    if not is_complete_compressed_mrt(local_rib_path):
        raise RuntimeError(f"Initial RIB is missing or invalid after download: {local_rib_path}")

    # --- Step 2: Download Updates [T-24h, T] ---
    bp_upd = set_base_params(start_s, end_s, data_types="BGP:UPDATES", collector="rrc00", minio_flag=False)
    upd_urllist = getURLList(bp_upd)
    
    target_upd_urls = []
    expected_upd_files = []
    
    for _, url in upd_urllist:
        fname = url.split('/')[-1]
        saved_fname = f"rrc00_{fname}"
        expected_upd_files.append(saved_fname)

        local_path = os.path.join(UPDATE_STORAGE_DIR, saved_fname)
        if not is_complete_compressed_mrt(local_path):
            if os.path.exists(local_path):
                log.warning(f"Existing update file is invalid and will be redownloaded: {local_path}")
            target_upd_urls.append((_, url))
            
    if target_upd_urls:
        log.info(f"Downloading {len(target_upd_urls)} update files...")
        downloadByParams(
            base_type=bp_upd.base_type,
            urllist=target_upd_urls,
            destination=UPDATE_STORAGE_DIR,
            minio_enable=False
        ).start_on()
    else:
        log.info("All update files exist locally.")

    invalid_update_files = list_invalid_downloads(UPDATE_STORAGE_DIR, expected_upd_files)
    if invalid_update_files:
        preview = ", ".join(invalid_update_files[:10])
        raise RuntimeError(
            f"Missing or invalid update files after download: "
            f"{len(invalid_update_files)}/{len(expected_upd_files)}. First files: {preview}"
        )
        
    # --- Step 3: Simulation ---
    master_rib = {} # prefix -> set(paths)
    
    # A. Load Initial RIB
    log.info(f"Loading Initial RIB {local_rib_path}...")
    try:
        for line in tqdm(iter_mrt_lines(local_rib_path), desc="Processing Initial RIB"):
            parts = line.strip().split('|')
            if len(parts) < 7: continue
            if not parts[0].startswith("TABLE_DUMP"): continue
            
            prefix = parts[5]
            if ':' in prefix or prefix == '0.0.0.0/0': continue
            
            as_path = parts[6]
            if not as_path: continue
            
            if prefix not in master_rib:
                master_rib[prefix] = set()
            master_rib[prefix].add(as_path)
    except Exception as e:
        log.error(f"Failed to process initial RIB: {e}")
        return None
        
    # B. Apply Updates
    # Sort files by time
    sorted_upd_files = sorted(
        os.path.join(UPDATE_STORAGE_DIR, filename)
        for filename in expected_upd_files
        if is_complete_compressed_mrt(os.path.join(UPDATE_STORAGE_DIR, filename))
    )  # Lexicographical sort works for rrc filenames because they contain timestamps.
    
    log.info(f"Simulating updates from {len(sorted_upd_files)} files...")
    for upd_path in tqdm(sorted_upd_files, desc="Replaying Updates"):
        try:
            for line in iter_mrt_lines(upd_path):
                parts = line.strip().split('|')
                # BGP4MP|ts|A|...|prefix|aspath|...
                if len(parts) < 6: continue
                if parts[0] != "BGP4MP": continue
                
                type_ = parts[2]
                prefix = parts[5]
                
                if ':' in prefix or prefix == '0.0.0.0/0': continue
                
                if type_ == 'A':
                    if len(parts) < 7: continue
                    as_path = parts[6]
                    if not as_path: continue
                    
                    if prefix not in master_rib:
                         master_rib[prefix] = set()
                    master_rib[prefix].add(as_path) # Add new path
                # Withdrawals are IGNORED as per requirement "record all possible paths"
                
        except Exception as e:
            log.error(f"Error processing update {upd_path}: {e}")
            
    # --- Step 4: Save Result ---
    log.info(f"Saving simulated Route Table to {route_table_path}...")
    with open(route_table_path, 'w') as f:
        for prefix, paths in master_rib.items():
            for path in paths:
                f.write(f"MERGED|{prefix}|{path}\n")
                
    del master_rib
    return route_table_path

def load_rib_data(rib_path):
    """
    Load RIB data into memory using Pytricia.
    Supports both TABLE_DUMP2 (raw) and MERGED formats.
    Returns: pytricia.PyTricia tree {prefix: set(as_path_str)}
    """
    log.info(f"Loading RIB data from {rib_path}...")
    rib_tree = pytricia.PyTricia()
    try:
        with open(rib_path, 'r') as f:
            for line in tqdm(f, desc="Parsing RIB"):
                parts = line.strip().split('|')
                
                prefix = None
                as_path = None
                
                # Check format
                if len(parts) >= 3 and parts[0] == "MERGED":
                    # MERGED|prefix|path
                    prefix = parts[1]
                    as_path = parts[2]
                elif len(parts) > 6 and parts[0].startswith("TABLE_DUMP"):
                     # TABLE_DUMP2...
                    prefix = parts[5]
                    as_path = parts[6]
                else:
                    continue

                if not prefix or not as_path:
                    continue
                
                # Filter IPv6
                if ':' in prefix:
                    continue

                if prefix=='0.0.0.0/0': continue
                
                if prefix not in rib_tree:
                    rib_tree[prefix] = set()
                rib_tree[prefix].add(as_path)
    except Exception as e:
        log.error(f"Error reading RIB file: {e}")
    return rib_tree

def process_event(event):
    event_name = event.name
    start_time_str = event.start_time
    leaker_as_raw = event.leaker_as
    
    target_as = None
    if leaker_as_raw and not pd.isna(leaker_as_raw):
        target_as = str(int(leaker_as_raw))
        # target_as = int(leaker_as_raw)
        
    if not target_as:
        log.warning(f"No Leaker AS defined for {event_name}. Skipping.")
        return

    output_file = os.path.join(OUTPUT_DIR, f"{event_name}.txt")
    if os.path.exists(output_file):
        log.info(f"Output file already exists for {event_name}, skipping processing.")
        return

    log.info(f"Processing {event_name} (Target AS: {target_as})")

    # 1. Prepare RIB based on mode
    rib_source_path = None
    
    if RIB_GENERATION_MODE == "MERGED":
        rib_source_path = prepare_merged_rib(event_name, start_time_str)
    elif RIB_GENERATION_MODE == "SIMULATION":
        rib_source_path = prepare_simulated_rib(event_name, start_time_str)
    else:
        log.error(f"Unknown RIB_GENERATION_MODE: {RIB_GENERATION_MODE}")
        return

    if not rib_source_path:
        log.error("Could not obtain RIB data. Skipping event.")
        return
        
    rib_data = load_rib_data(rib_source_path)
    
    # 2. Process Updates
    try:
        event_start_dt = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
    except:
        event_start_dt = datetime.datetime.strptime(start_time_str, "%Y/%m/%d %H:%M").replace(tzinfo=datetime.timezone.utc)
        
    search_start_ts = (event_start_dt - datetime.timedelta(minutes=10)).timestamp()
    if event.end_time:
        event_end_dt = datetime.datetime.strptime(event.end_time, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
    else:
        event_end_dt = event_start_dt + datetime.timedelta(hours=6)
    event_end_dt = event_end_dt + datetime.timedelta(minutes=10)
    search_end_ts = event_end_dt.timestamp()
    
    anomalies = []
    withdrawals = []
    anomalous_prefixes = set()
    missing_parent_prefixes = set()
    leaker_in_rib_cache = {}
    propagator_prefixes = {}
    peer_propagator_skipped = 0

    update_filepaths = list_event_update_files(event_name)
    if not update_filepaths:
        log.error(f"No update files found for {event_name}.")
        return

    decoded_file_iter = iter_decoded_update_files(update_filepaths)
    for _, lines in tqdm(decoded_file_iter, total=len(update_filepaths), desc=f"Scanning updates for {event_name}"):
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) < 6: continue
            if parts[0] != "BGP4MP": continue
                
            try:
                ts = int(parts[1])
            except: continue
                
            if not (search_start_ts <= ts <= search_end_ts):
                continue
            
            type_ = parts[2]
            prefix = parts[5]
            
            # Skip IPv6
            if ':' in prefix:
                continue
            
            if type_ == "A":
                if len(parts) < 7: continue
                as_path_str = parts[6]
                as_path_list = as_path_str.split(' ')
                
                if target_as not in as_path_list:
                    continue
                
                is_anomaly = False
                
                # Check against RIB for context
                parent_prefix = rib_data.get_key(prefix)
                
                if parent_prefix is None:
                    missing_parent_prefixes.add(prefix)
                    continue

                # Condition 2: Origin is leaker
                if as_path_list and as_path_list[-1] == target_as:
                    is_origin_in_rib = False
                    for rp in rib_data[parent_prefix]:
                        rp_list = rp.split(' ')
                        if rp_list and rp_list[-1] == target_as:
                            is_origin_in_rib = True
                            break
                    if not is_origin_in_rib:
                        is_anomaly = True
                else:
                    # Condition 1: Check against RIB (minimal parent prefix)
                    # parent_prefix is already fetched above
                    if parent_prefix in leaker_in_rib_cache:
                        leaker_in_rib = leaker_in_rib_cache[parent_prefix]
                    else:
                        rib_paths = rib_data[parent_prefix]
                        leaker_in_rib = False
                        for rp in rib_paths:
                            if target_as in rp.split(' '):
                                leaker_in_rib = True
                                break
                        leaker_in_rib_cache[parent_prefix] = leaker_in_rib
                        
                    if not leaker_in_rib:
                        is_anomaly = True

                if is_anomaly:
                    # Exclude announcements where the leaker's left neighbor is the observation peer.
                    # Input: BGP4MP parts and AS_PATH list. Output: skip peer-originated propagator cases.
                    peer_as = parts[4]
                    leaker_idx = as_path_list.index(target_as)
                    propagator = as_path_list[leaker_idx - 1] if leaker_idx > 0 else None
                    if propagator == peer_as:
                        peer_propagator_skipped += 1
                        continue

                    anomalies.append(line.strip())
                    anomalous_prefixes.add(prefix)
                    
                    # Record Propagator (AS to the left of leaker)
                    if propagator is not None:
                        if propagator not in propagator_prefixes:
                            propagator_prefixes[propagator] = set()
                        propagator_prefixes[propagator].add(prefix)
            
            elif type_ == "W":
                if prefix in anomalous_prefixes:
                    withdrawals.append(line.strip())
                    anomalous_prefixes.remove(prefix) # Avoid duplicate withdraws if multiple anomalies for same prefix

    if anomalies:
        log.info(f"Found {len(anomalies)} anomalies. Saving to {OUTPUT_DIR}...")
        with open(output_file, 'w') as f:
            for a in anomalies:
                f.write(a + '\n')
    else:
        log.info(f"No anomalies found for {event_name}.")
        
    if withdrawals:
        wd_file = os.path.join(WITHDRAW_DIR, f"{event_name}.txt")
        log.info(f"Found {len(withdrawals)} related withdrawals. Saving to {WITHDRAW_DIR}...")
        with open(wd_file, 'w') as f:
            for w in withdrawals:
                f.write(w + '\n')
    log.info(f"{len(anomalous_prefixes)} prefixes remain anomalous at end of scan")
    if missing_parent_prefixes:
        log.info(f"{len(missing_parent_prefixes)} prefixes had no parent in RIB (skipped checking against RIB).")
    if peer_propagator_skipped:
        log.info(f"Skipped {peer_propagator_skipped} anomalies whose propagator is the observation peer.")

    if propagator_prefixes:
        log.info(f"[{event_name}] Propagators detected (Left of Leaker):")
        
        # Calculate unique prefix counts
        propagator_stats = {p: len(prefixes) for p, prefixes in propagator_prefixes.items()}
        sorted_props = sorted(propagator_stats.items(), key=lambda x: x[1], reverse=True)
        
        for p, count in sorted_props:
            log.info(f"  AS{p}: {count} prefixes")
            
        # Save to JSON
        prop_file = os.path.join(PROPAGATOR_DIR, f"{event_name}.json")
        with open(prop_file, 'w') as f:
            json.dump(dict(sorted_props), f, indent=4)
        log.info(f"Propagator stats saved to {prop_file}")

def analyze_propagator_message_count(event):
    """统计传播者对应的异常消息条数。输入：RADBenchEvent。输出：传播者消息占比或条数字典 JSON。"""
    event_name = event.name
    leaker_as_raw = event.leaker_as

    # Control output mode: True = Percentage, False = Raw Message Count
    CALC_PERCENTAGE = False

    if not leaker_as_raw or pd.isna(leaker_as_raw):
        return

    target_as = str(int(leaker_as_raw))
    input_file = os.path.join(OUTPUT_DIR, f"{event_name}.txt")

    if not os.path.exists(input_file):
        return

    if CALC_PERCENTAGE:
        save_dir = os.path.join(OUTPUT_DIR, "Propagator_Message_Percentage")
    else:
        save_dir = os.path.join(OUTPUT_DIR, "Propagator_Message_Count")
    os.makedirs(save_dir, exist_ok=True)

    propagator_message_counts = {}

    try:
        with open(input_file, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 7:
                    continue

                as_path_str = parts[6]
                as_path_list = as_path_str.split(' ')

                if target_as in as_path_list:
                    try:
                        # Find the first occurrence (leftmost) of leaker
                        idx = as_path_list.index(target_as)
                        if idx > 0:
                            propagator = as_path_list[idx - 1]
                            propagator_message_counts[propagator] = propagator_message_counts.get(propagator, 0) + 1
                    except ValueError:
                        pass
    except Exception as e:
        log.error(f"Error reading anomaly file {input_file}: {e}")
        return

    sorted_stats = sorted(propagator_message_counts.items(), key=lambda x: x[1], reverse=True)
    total_message_count = sum(propagator_message_counts.values())

    result = {}
    if CALC_PERCENTAGE and total_message_count > 0:
        for p, count in sorted_stats:
            result[p] = count / total_message_count
    else:
        result = dict(sorted_stats)

    output_path = os.path.join(save_dir, f"{event_name}.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    log.info(f"[{event_name}] Propagator message analysis saved to {output_path}")

def visualize_results():
    log.info("Generating visualization...")
    vis_dir = os.path.join(OUTPUT_DIR, "Visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    num_events = len(events)
    if num_events == 0:
        log.warning("No events to visualize.")
        return

    # Grid Layout Configuration
    COLS = 3
    # Logical rows of events
    event_rows = (num_events + COLS - 1) // COLS
    
    # Figure size: Width 24, Height based on rows
    # Increase height per row to 8 to allow more spacing
    fig, axes = plt.subplots(event_rows * 2, COLS, figsize=(24, 8 * event_rows), constrained_layout={"h_pad": 0.2, "w_pad": 0.1})
    
    # Ensure axes is always 2D array [rows, cols]
    if event_rows * 2 == 1: 
        axes = np.array([axes])
    if len(axes.shape) == 1:
        # If 1D, it could be [row] or [col] depending on subplots params
        axes = axes.reshape(-1, COLS)

    for i, event in enumerate(tqdm(events, desc="Visualizing")):
        event_name = event.name
        
        # Determine Grid Position
        col_idx = i % COLS
        row_idx = i // COLS
        
        # Top plot (Timeline) index = row_idx * 2
        # Bottom plot (Counts) index = row_idx * 2 + 1
        ax1 = axes[row_idx * 2, col_idx]
        ax2 = axes[row_idx * 2 + 1, col_idx]
        
        # 1. Gather Data & Time Range
        event_dir = os.path.join(EVENT_UPDATE_DIR_BASE, event_name)
        
        # Calculate min_ts from event start time
        try:
            evt_start_dt = datetime.datetime.strptime(event.start_time, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
            min_ts = (evt_start_dt - datetime.timedelta(hours=3)).timestamp()
        except:
            # Fallback
            min_ts = 0
            
        max_ts = None
        
        # Scan files for max_ts only
        if os.path.exists(event_dir):
            files = sorted([f for f in os.listdir(event_dir) if f.startswith("rrc00_updates") and (f.endswith(".txt") or f.endswith(".gz"))])
            if files:
                # Max from last file
                try:
                    fpath = os.path.join(event_dir, files[-1])
                    if fpath.endswith(".gz"):
                        last_line = None
                        for line in iter_mrt_lines(fpath):
                            last_line = line
                        if last_line: max_ts = float(last_line.split('|')[1])
                    else:
                        with open(fpath) as f: lines = f.readlines()
                        if lines: max_ts = float(lines[-1].split('|')[1])
                except: pass
        
        if min_ts == 0 and max_ts is not None: min_ts = max_ts - 3600*6 # Fallback if event start parsing failed
        if max_ts is None: max_ts = min_ts + 3600*6
        
        # Pad range slightly
        data_start_dt = datetime.datetime.fromtimestamp(min_ts, tz=datetime.timezone.utc)
        data_end_dt = datetime.datetime.fromtimestamp(max_ts, tz=datetime.timezone.utc)
        
        # Align to minute boundaries
        plot_start_ts = (min_ts // 60) * 60
        plot_end_ts = (max_ts // 60) * 60 + 60
        
        # 2. Load Anomalies and Withdrawals
        label_file = os.path.join(OUTPUT_DIR, f"{event_name}.txt") 
        anomaly_ts_list = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) > 1:
                        try:
                            anomaly_ts_list.append(float(parts[1]))
                        except: pass

        withdraw_file = os.path.join(WITHDRAW_DIR, f"{event_name}.txt")
        withdraw_ts_list = []
        if os.path.exists(withdraw_file):
            with open(withdraw_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) > 1:
                        try:
                            withdraw_ts_list.append(float(parts[1]))
                        except: pass

        # 3. Binning (1 Minute)
        bins = np.arange(plot_start_ts, plot_end_ts + 60, 60)
        # Handle case where bins is empty or too small
        if len(bins) < 2:
            bins = np.array([plot_start_ts, plot_start_ts + 60])
            
        bin_counts, bin_edges = np.histogram(anomaly_ts_list, bins=bins)
        withdrawal_counts, _ = np.histogram(withdraw_ts_list, bins=bins)
        
        bin_binary = (bin_counts > 0).astype(int)
        
        bin_centers = bin_edges[:-1]
        bin_dt = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc) for ts in bin_centers]
        
        # 4. Plotting
        # --- Top Plot: Timeline ---
        ax1.set_title(event_name, fontsize=11, fontweight='bold')
        
        # Ground Truth Shader
        try:
            report_start = datetime.datetime.strptime(event.start_time, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
            if event.end_time and event.end_time != 'unknown':
                report_end = datetime.datetime.strptime(event.end_time, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
            else:
                report_end = data_end_dt 
            
            ax1.axvspan(report_start, report_end, color='green', alpha=0.3, label='Reported Label')
        except: pass
        
        # Precise Label Bars
        width_days = 1.0 / (24*60) # 1 minute in Matplotlib date units
        ax1.bar(bin_dt, bin_binary, width=width_days, color='red', alpha=0.6, label='Precise Label', align='edge')
        
        # Formatting Ax1
        if bin_dt:
            ax1.set_xlim(bin_dt[0], bin_dt[-1])
        ax1.set_ylim(0, 1.1)
        ax1.set_yticks([]) # Binary, no need for numbers
        ax1.legend(loc='upper right', fontsize='small')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=datetime.timezone.utc))
        
        # --- Bottom Plot: Counts ---
        ax2.bar(bin_dt, bin_counts, width=width_days, color='blue', alpha=0.7, label='Count', align='edge')
        ax2.bar(bin_dt, withdrawal_counts, width=width_days, color='orange', alpha=0.7, label='Withdraw', align='edge', bottom=bin_counts)
        
        # Formatting Ax2
        ax2.set_yscale('symlog')
        if bin_dt:
            ax2.set_xlim(bin_dt[0], bin_dt[-1])
        ax2.legend(loc='upper right', fontsize='small')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=datetime.timezone.utc))
        ax2.set_ylabel("Anomalies")
        
    # Hide Empty Subplots
    for j in range(num_events, event_rows * COLS):
        row_idx = j // COLS
        col_idx = j % COLS
        axes[row_idx * 2, col_idx].set_visible(False)
        axes[row_idx * 2 + 1, col_idx].set_visible(False)

    out_path = os.path.join(vis_dir, "all_events_visualization.png")
    plt.savefig(out_path, dpi=100)
    log.info(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    for event in events:
        process_event(event)
        print("\n")
    visualize_results()
    for event in events:
        analyze_propagator_message_count(event)
