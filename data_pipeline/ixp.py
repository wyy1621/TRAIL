"""Prepare per-event RADBench IXP ASN sets.

用途：
    为每个 RADBench leak 事件生成 AS_PATH clean 使用的 IXP ASN 集合。若最终
    `DATA/RADBench_IXPs/{event}.txt` 已存在且非空，直接跳过该事件；否则在线
    下载 CAIDA as-relationships 与 PeeringDB，解析并合并 as-relationships 头部的
    `# IXP ASes` 与 PeeringDB Route Server ASN。当事件时间早于 CAIDA PeeringDB
    v2 JSON 可用日期时，只使用 as-relationships。原链接下载失败时，跳过
    对应快照或事件并继续处理后续任务。

关键参数：
    - `--event-csv`：RADBench 事件元数据 CSV，默认 `anomaly-event-info-v5.csv`。
    - `--output-dir`：事件 IXP ASN 输出目录，默认 `DATA/RADBench_IXPs`。

输出：
    - `DATA/RADBench_IXPs/{event}.txt`：一行空格分隔 ASN，按数值升序排列。
"""

from __future__ import annotations

import argparse
import bz2
import csv
import datetime as dt
import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm


def beijing(sec, what):
    """Return Beijing local time tuple for logging formatter."""
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).timetuple()


logging.Formatter.converter = beijing
logging.basicConfig(
    level=logging.WARNING,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

DEFAULT_EVENT_CSV = "radbench_leaks.csv"
DEFAULT_OUTPUT_DIR = "DATA/Processed/IXP"
AS_REL_BASE_URL = "https://publicdata.caida.org/datasets/as-relationships"
END_DATE = 6
PEERINGDB_JSON_START_DATE = dt.date(2018, 3, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare per-event RADBench IXP ASN sets")
    parser.add_argument("--event-csv", default=DEFAULT_EVENT_CSV)
    parser.add_argument("--event-type", default="leak", help="Event type to process. Use 'all' for every event.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_event_start_time(raw_time: str) -> dt.datetime:
    """Parse RADBench CSV start time.

    Args:
        raw_time: Time string in `YYYY/M/D H:M` format.

    Returns:
        Naive UTC datetime used only for month/date matching.
    """
    return dt.datetime.strptime(raw_time.strip(), "%Y/%m/%d %H:%M")


def load_events(event_csv: Path, event_type: str) -> list[dict[str, str]]:
    """Load target RADBench events from CSV.

    Args:
        event_csv: RADBench event metadata CSV path.
        event_type: Event type filter. `all` disables filtering.

    Returns:
        Ordered event dictionaries from the CSV.
    """
    if not event_csv.exists():
        raise FileNotFoundError(f"Event CSV not found: {event_csv}")

    with event_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if event_type != "all":
        rows = [row for row in rows if row.get("event_type") == event_type]
    if not rows:
        raise RuntimeError(f"No events loaded from {event_csv} with event_type={event_type}")
    return rows


def parse_asn_tokens(text: str) -> set[int]:
    """Extract ASN integers from whitespace/comment text.

    Args:
        text: File content or header suffix containing ASN tokens.

    Returns:
        Set of parsed ASN integers.
    """
    return {int(token) for token in re.findall(r"\d+", text)}


def parse_as_relationship_ixps(as_rel_file: Path) -> set[int]:
    """Parse CAIDA `# IXP ASes:` header from one as-relationship file.

    Args:
        as_rel_file: CAIDA as-rel/as-rel2 file.

    Returns:
        IXP ASN set from the header.
    """
    with as_rel_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# IXP ASes:"):
                ixps = parse_asn_tokens(line.split(":", 1)[1])
                if not ixps:
                    raise RuntimeError(f"Empty IXP ASes header in {as_rel_file}")
                return ixps
    raise RuntimeError(f"Missing '# IXP ASes:' header in {as_rel_file}")


def event_as_relationship_snapshot_date(start_time: dt.datetime) -> dt.date:
    """Return the CAIDA as-relationship snapshot date for one event.

    Args:
        start_time: RADBench event start time.

    Returns:
        First day of the event month, matching the monthly CAIDA snapshots used
        by the existing baseline data.
    """
    return dt.date(start_time.year, start_time.month, 1)


def as_relationship_candidates(snapshot_date: dt.date) -> list[tuple[str, str]]:
    """Build CAIDA URLs for one as-relationship snapshot.

    Args:
        snapshot_date: Snapshot date.

    Returns:
        Candidate `(filename, remote_bz2_url)` pairs, preferring serial-2.
    """
    date_str = snapshot_date.strftime("%Y%m%d")
    return [
        (
            f"{date_str}.as-rel2.txt",
            f"{AS_REL_BASE_URL}/serial-2/{date_str}.as-rel2.txt.bz2",
        ),
        (
            f"{date_str}.as-rel.txt",
            f"{AS_REL_BASE_URL}/serial-1/{date_str}.as-rel.txt.bz2",
        ),
    ]


def download_as_relationship_file(snapshot_date: dt.date, *, temp_root: Path) -> Path | None:
    """Download and decompress one CAIDA as-relationship snapshot.

    Args:
        snapshot_date: Required monthly snapshot date.
        temp_root: Temporary directory for downloaded intermediate files.

    Returns:
        Decompressed as-relationship txt file path under `temp_root`, or `None`
        when every download candidate fails.
    """
    errors = []
    candidates = as_relationship_candidates(snapshot_date)
    progress = tqdm(
        candidates,
        desc=f"as-rel {snapshot_date:%Y%m%d}",
        unit="candidate",
        leave=False,
    )
    for filename, url in progress:
        temp_bz2 = temp_root / f"{filename}.bz2"
        output_path = temp_root / filename
        try:
            progress.set_postfix_str(filename)
            if not download_url(url, temp_bz2):
                errors.append(f"{url}: download failed")
                continue
            raw = bz2.decompress(temp_bz2.read_bytes())
            if not raw.strip():
                raise RuntimeError(f"Downloaded empty as-relationship file from {url}")
            output_path.write_bytes(raw)
            return output_path
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, EOFError, RuntimeError) as exc:
            errors.append(f"{url}: {exc}")
            continue

    return None


def expected_peeringdb_dates(month: str) -> list[dt.date]:
    """Return PathProb-style PeeringDB dates for one `YYYYMM` month."""
    year = int(month[:4])
    mon = int(month[4:6])
    return [dt.date(year, mon, day) for day in range(1, END_DATE)]


def peeringdb_url(snapshot_date: dt.date) -> str:
    return (
        "https://publicdata.caida.org/datasets/peeringdb-v2/"
        f"{snapshot_date:%Y}/{snapshot_date:%m}/"
        f"peeringdb_2_dump_{snapshot_date:%Y}_{snapshot_date:%m}_{snapshot_date:%d}.json"
    )


def download_url_once(url: str, output_path: Path) -> None:
    """Download one URL directly without using any proxy.

    Args:
        url: Source URL.
        output_path: Destination file path.

    Returns:
        None.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"User-Agent": "BgpUnsupervisedV2/prepare-radbench-ixps"})
    with opener.open(request, timeout=120) as response, output_path.open("wb") as f:
        f.write(response.read())


def remove_partial_file(output_path: Path) -> None:
    """Remove a possibly incomplete download file.

    Args:
        output_path: Path that may contain an incomplete download.

    Returns:
        None.
    """
    if output_path.exists():
        output_path.unlink()


def download_url(url: str, output_path: Path) -> bool:
    """Download one original URL directly without using any proxy.

    Args:
        url: Source URL.
        output_path: Destination file path.

    Returns:
        True when the file is downloaded successfully, otherwise False.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_url_once(url, output_path)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        remove_partial_file(output_path)
        log.debug("Failed to download original URL %s directly: %s", url, exc)
        return False


def extract_route_server_asns_from_json(json_path: Path) -> set[int]:
    """Extract PeeringDB Route Server ASN set from CAIDA PeeringDB v2 JSON.

    Args:
        json_path: Downloaded CAIDA PeeringDB v2 JSON file.

    Returns:
        Route Server ASN set from `data["net"]["data"]`.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ixps = {
        int(item["asn"])
        for item in data["net"]["data"]
        if item.get("info_type") == "Route Server"
    }
    if not ixps:
        raise RuntimeError(f"No Route Server ASN extracted from {json_path}")
    return ixps


def write_asn_set(asns: set[int], output_path: Path) -> None:
    """Write ASN set as one space-separated line.

    Args:
        asns: ASN set.
        output_path: Destination txt file.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(" ".join(str(asn) for asn in sorted(asns)))
        f.write("\n")


def download_peeringdb_route_server_asns(snapshot_date: dt.date, *, temp_root: Path) -> set[int] | None:
    """Download one PeeringDB snapshot and extract Route Server ASNs.

    Args:
        snapshot_date: PeeringDB snapshot date.
        temp_root: Temporary directory for downloaded intermediate files.

    Returns:
        Route Server ASN set from the downloaded snapshot, or `None` when the
        snapshot cannot be downloaded directly from the original URL.
    """
    json_path = temp_root / f"peeringdb_2_dump_{snapshot_date:%Y_%m_%d}.json"
    url = peeringdb_url(snapshot_date)
    if not download_url(url, json_path):
        return None
    return extract_route_server_asns_from_json(json_path)


def load_peeringdb_route_server_asns(month: str, *, temp_root: Path) -> set[int]:
    """Download and merge Route Server ASN sets for one event month.

    Args:
        month: Event month in `YYYYMM`.
        temp_root: Temporary directory for downloaded intermediate files.

    Returns:
        Union of freshly downloaded Route Server ASN sets from the month days 1-5.
        Returns an empty set when the month predates CAIDA PeeringDB v2 JSON snapshots.
    """
    snapshot_dates = expected_peeringdb_dates(month)
    if max(snapshot_dates) < PEERINGDB_JSON_START_DATE:
        return set()

    ixps: set[int] = set()
    active_dates = [snapshot_date for snapshot_date in snapshot_dates if snapshot_date >= PEERINGDB_JSON_START_DATE]
    skipped_count = 0
    progress = tqdm(active_dates, desc=f"peeringdb {month}", unit="snapshot", leave=False)
    for snapshot_date in progress:
        route_server_asns = download_peeringdb_route_server_asns(snapshot_date, temp_root=temp_root)
        if route_server_asns is None:
            skipped_count += 1
            progress.set_postfix(download_skipped=skipped_count, ixps=len(ixps))
            continue
        ixps.update(route_server_asns)
        progress.set_postfix(download_skipped=skipped_count, ixps=len(ixps))
    return ixps


def main() -> None:
    """Generate `DATA/RADBench_IXPs/{event}.txt` files."""
    args = parse_args()
    event_csv = Path(args.event_csv)
    output_dir = Path(args.output_dir)

    events = load_events(event_csv, args.event_type)

    skipped_events: list[tuple[str, str]] = []
    progress = tqdm(events, desc="radbench-events", unit="event")
    for event in progress:
        event_name = event["event_name"]
        progress.set_postfix(event=event_name[:32])
        output_path = output_dir / f"{event_name}.txt"
        if output_path.exists() and output_path.stat().st_size > 0:
            progress.set_postfix(event=event_name[:32], status="exists")
            continue

        start_time = parse_event_start_time(event["start_time"])
        month = start_time.strftime("%Y%m")

        with tempfile.TemporaryDirectory(prefix=f"radbench_ixp_{event_name}_") as temp_dir:
            temp_root = Path(temp_dir)
            as_rel_snapshot_date = event_as_relationship_snapshot_date(start_time)
            as_rel_file = download_as_relationship_file(
                as_rel_snapshot_date,
                temp_root=temp_root,
            )
            if as_rel_file is None:
                skipped_events.append((event_name, f"as-rel {as_rel_snapshot_date:%Y%m%d} download failed"))
                progress.set_postfix(event=event_name[:32], status="skipped")
                continue
            as_rel_ixps = parse_as_relationship_ixps(as_rel_file)
            peeringdb_ixps = load_peeringdb_route_server_asns(
                month,
                temp_root=temp_root,
            )
            merged_ixps = as_rel_ixps | peeringdb_ixps
            if not merged_ixps:
                raise RuntimeError(f"No IXP ASN collected for event {event_name}")

            write_asn_set(merged_ixps, output_path)
            progress.set_postfix(
                event=event_name[:32],
                total=len(merged_ixps),
                as_rel=len(as_rel_ixps),
                peeringdb=len(peeringdb_ixps),
            )

    if skipped_events:
        tqdm.write("Skipped RADBench IXP events:")
        for event_name, reason in skipped_events:
            tqdm.write(f"  {event_name}: {reason}")


if __name__ == "__main__":
    main()
