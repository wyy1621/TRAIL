"""Download RADBench event updates and before-event update data.

Purpose:
    Download RIPE RIS BGP update MRT gzip files for RADBench events. By default
    the script downloads event-window updates first. Before-event data is a
    separate optional task enabled by --download-before-data. The script does
    not decode MRT files and does not use the local bgpdownloader package.

Key parameters:
    --event-csv: RADBench event metadata CSV. Default is anomaly-event-info-v6.csv.
    --root-dir: RADBench raw update root. Default is /data/RouteLeak/RADBench.
    --event-prefix-minutes: Minutes included before event start for the event window. Default is 6 hours.
    --event-suffix-minutes: Minutes included after known event end for the event window. Default is 6 hours.
    --download-before-data: Also download Before3Day data after event updates finish.
    --workers: Parallel download workers, capped at 16.
    --retries: Retry count for the direct download URL.

Outputs:
    {root-dir}/{event_name}/rrc00_updates.YYYYMMDD.HHMM.gz
    {root-dir}/Before3Day/{event_name}/rrc00_updates.YYYYMMDD.HHMM.gz
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


DEFAULT_EVENT_CSV = "radbench_leaks.csv"
DEFAULT_ROOT_DIR = "DATA/Raw/RADBench"
DEFAULT_COLLECTOR = "rrc00"
DEFAULT_BEFORE_DIR_NAME = "Before3Day"
DEFAULT_EVENT_PREFIX_MINUTES = 6 * 60
DEFAULT_EVENT_SUFFIX_MINUTES = 6 * 60
DOWNLOAD_BEFORE3D = False
USER_AGENT = "Wget/1.21.1"
GZIP_MAGIC = b"\x1f\x8b"


def beijing(sec: float, what: object) -> time.struct_time:
    """Return Beijing local time for logging.

    Args:
        sec: Unused timestamp argument required by logging.Formatter.converter.
        what: Unused tuple argument required by logging.Formatter.converter.

    Returns:
        Beijing local time tuple.
    """
    del sec, what
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).timetuple()


logging.Formatter.converter = beijing
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RADBenchEvent:
    """RADBench event metadata needed for update downloads."""

    event_name: str
    event_type: str
    start_time: dt.datetime
    end_time: dt.datetime | None


@dataclass(frozen=True)
class DownloadItem:
    """One expected remote update file and its local filename."""

    url: str
    filename: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Download configuration.
    """
    parser = argparse.ArgumentParser(description="Download RADBench event updates, with optional before-event updates.")
    parser.add_argument("--event-csv", default=DEFAULT_EVENT_CSV, help="RADBench event metadata CSV path.")
    parser.add_argument("--event-type", default="leak", help="Event type filter. Use 'all' to disable filtering.")
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR, help="RADBench raw update root directory.")
    parser.add_argument("--collector", default=DEFAULT_COLLECTOR, help="RIPE RIS collector. RADBench defaults to rrc00.")
    parser.add_argument(
        "--before-dir-name",
        default=DEFAULT_BEFORE_DIR_NAME,
        help="Directory name under root-dir for before-event data.",
    )
    parser.add_argument(
        "--event-prefix-minutes",
        type=int,
        default=DEFAULT_EVENT_PREFIX_MINUTES,
        help="Minutes included before event start for event-window updates.",
    )
    parser.add_argument(
        "--event-suffix-minutes",
        type=int,
        default=DEFAULT_EVENT_SUFFIX_MINUTES,
        help="Minutes included after known event end for event-window updates.",
    )
    parser.add_argument(
        "--before-days",
        type=int,
        default=3,
        help="Days of before-event updates to download.",
    )
    parser.add_argument(
        "--before-gap-minutes",
        type=int,
        default=5,
        help="Gap between before-event end and event-window start.",
    )
    parser.add_argument(
        "--unknown-duration-hours",
        type=int,
        default=6,
        help="Final event-window duration used when end_time is unknown.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers. The effective value is capped at 16.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retry count for the direct download URL.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds.")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024, help="Download chunk size in bytes.")
    parser.add_argument(
        "--event-name",
        action="append",
        default=None,
        help="Download only the named event. Can be passed multiple times.",
    )
    parser.add_argument("--skip-event-updates", action="store_true", help="Skip event-window updates.")
    parser.add_argument(
        "--download-before-data",
        action="store_true",
        default=DOWNLOAD_BEFORE3D,
        help="Download before-event data after all event-window updates finish.",
    )
    return parser.parse_args()


def parse_csv_time(raw_time: str) -> dt.datetime:
    """Parse RADBench CSV time as UTC.

    Args:
        raw_time: Time string in YYYY/M/D H:M or YYYY-MM-DD HH:MM format.

    Returns:
        Timezone-aware UTC datetime.
    """
    cleaned = raw_time.strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return dt.datetime.strptime(cleaned, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unsupported RADBench time format: {raw_time!r}")


def load_events(event_csv: str, event_type: str, event_names: list[str] | None) -> list[RADBenchEvent]:
    """Load target RADBench events from CSV.

    Args:
        event_csv: Event metadata CSV path.
        event_type: Event type filter. `all` disables filtering.
        event_names: Optional selected event names.

    Returns:
        Ordered target events.
    """
    csv_path = Path(event_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Event CSV not found: {csv_path}")

    required_columns = {"event_type", "event_name", "start_time", "end_time"}
    target_type = None if event_type.lower() == "all" else event_type
    events: list[RADBenchEvent] = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty event CSV: {csv_path}")
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise KeyError(f"{csv_path} missing required columns: {sorted(missing_columns)}")

        for row in reader:
            row_event_type = row["event_type"].strip()
            if target_type is not None and row_event_type != target_type:
                continue

            end_raw = row["end_time"].strip()
            end_time = None if end_raw == "unknown" or not end_raw else parse_csv_time(end_raw)
            events.append(
                RADBenchEvent(
                    event_name=row["event_name"].strip(),
                    event_type=row_event_type,
                    start_time=parse_csv_time(row["start_time"]),
                    end_time=end_time,
                )
            )

    if event_names:
        selected = set(event_names)
        events = [event for event in events if event.event_name in selected]
        missing_names = sorted(selected - {event.event_name for event in events})
        if missing_names:
            raise ValueError(f"Event names not found in {csv_path}: {missing_names}")

    if not events:
        raise RuntimeError(f"No events loaded from {csv_path} with event_type={event_type}")
    return events


def floor_to_5min(value: dt.datetime) -> dt.datetime:
    """Floor a datetime to the RIPE RIS 5-minute update granularity.

    Args:
        value: Input datetime.

    Returns:
        Datetime rounded down to a 5-minute boundary.
    """
    return value.replace(minute=value.minute - (value.minute % 5), second=0, microsecond=0)


def build_update_plan(start_time: dt.datetime, end_time: dt.datetime, collector: str) -> list[DownloadItem]:
    """Build expected RIPE RIS update URLs and local filenames.

    Args:
        start_time: Inclusive window start.
        end_time: Inclusive window end.
        collector: RIPE RIS collector name, such as rrc00.

    Returns:
        Expected download items in chronological order.
    """
    if end_time < start_time:
        raise ValueError(f"end_time must be greater than or equal to start_time: {start_time} -> {end_time}")

    current = floor_to_5min(start_time)
    plan: list[DownloadItem] = []
    while current <= end_time:
        year_month = current.strftime("%Y.%m")
        date_time = current.strftime("%Y%m%d.%H%M")
        remote_name = f"updates.{date_time}.gz"
        filename = f"{collector}_{remote_name}"
        url = f"http://data.ris.ripe.net/{collector}/{year_month}/{remote_name}"
        plan.append(DownloadItem(url=url, filename=filename))
        current += dt.timedelta(minutes=5)

    if not plan:
        raise RuntimeError(f"No update filenames generated for {collector}: {start_time} -> {end_time}")
    return plan


def is_complete_gzip(path: Path) -> bool:
    """Check whether a local gzip file looks complete enough to skip.

    Args:
        path: Local gzip file path.

    Returns:
        True when the file exists, is non-empty, and has gzip magic bytes.
    """
    if not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
        with path.open("rb") as file_obj:
            return file_obj.read(2) == GZIP_MAGIC
    except OSError:
        return False


def list_missing_gz(destination: Path, plan: list[DownloadItem]) -> list[DownloadItem]:
    """List update files absent or invalid in a local destination.

    Args:
        destination: Local directory to inspect.
        plan: Expected download items.

    Returns:
        Missing or invalid download items. Valid existing `.gz` files are skipped.
    """
    missing = []
    for item in plan:
        if not item.filename.endswith(".gz"):
            raise RuntimeError(f"Expected gzip update file, got: {item.filename}")
        target_path = destination / item.filename
        if is_complete_gzip(target_path):
            continue
        if target_path.exists():
            log.warning("Existing update file is invalid and will be redownloaded: %s", target_path)
        missing.append(item)
    return missing


def validate_downloaded_gzip(path: Path, expected_length: int | None) -> None:
    """Validate a newly downloaded gzip file before finalizing it.

    Args:
        path: Temporary downloaded file path.
        expected_length: Content-Length from HTTP response, if provided.

    Returns:
        None.
    """
    file_size = path.stat().st_size
    if file_size <= 0:
        raise RuntimeError(f"Downloaded empty file: {path}")
    if expected_length is not None and file_size != expected_length:
        raise RuntimeError(f"Downloaded size mismatch for {path}: got {file_size}, expected {expected_length}")
    with path.open("rb") as file_obj:
        if file_obj.read(2) != GZIP_MAGIC:
            raise RuntimeError(f"Downloaded file does not have gzip magic bytes: {path}")


def open_url_without_proxy(request: urllib.request.Request, timeout: int):
    """Open a URL request while ignoring system proxy environment variables.

    Args:
        request: Prepared urllib request.
        timeout: Per-request timeout in seconds.

    Returns:
        HTTP response object usable as a context manager.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def cleanup_buffer_dir(buffer_dir: Path) -> None:
    """Remove temporary download files after a successful window download.

    Args:
        buffer_dir: Temporary download directory created under the destination.

    Returns:
        None.
    """
    if not buffer_dir.is_dir():
        return
    for tmp_path in buffer_dir.glob("*.tmp.*"):
        try:
            tmp_path.unlink()
        except OSError as exc:
            log.warning("Failed to remove temporary download file %s: %s", tmp_path, exc)
    try:
        buffer_dir.rmdir()
    except OSError:
        # Keep the directory if it contains unexpected files or another worker just touched it.
        pass


def download_one(
    item: DownloadItem,
    destination: Path,
    buffer_dir: Path,
    retries: int,
    timeout: int,
    chunk_size: int,
) -> None:
    """Download one missing update file with retries and atomic finalization.

    Args:
        item: Remote URL and local filename.
        destination: Local output directory.
        buffer_dir: Temporary download directory under the same filesystem as destination.
        retries: Retry count for the direct download URL.
        timeout: Per-request timeout in seconds.
        chunk_size: Stream read chunk size in bytes.

    Returns:
        None.
    """
    target_path = destination / item.filename
    if is_complete_gzip(target_path):
        return

    errors: list[str] = []
    for attempt in range(1, retries + 1):
        tmp_path = buffer_dir / f"{item.filename}.tmp.{os.getpid()}.{attempt}"
        try:
            request = urllib.request.Request(item.url, headers={"User-Agent": USER_AGENT})
            with open_url_without_proxy(request, timeout=timeout) as response:
                status = getattr(response, "status", None)
                if status is not None and status >= 400:
                    raise RuntimeError(f"HTTP status {status}")
                length_header = response.headers.get("Content-Length")
                expected_length = int(length_header) if length_header and length_header.isdigit() else None
                with tmp_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        output_file.write(chunk)

            validate_downloaded_gzip(tmp_path, expected_length)
            if is_complete_gzip(target_path):
                tmp_path.unlink(missing_ok=True)
                return
            tmp_path.replace(target_path)
            return
        except (OSError, RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            tmp_path.unlink(missing_ok=True)
            error = f"direct attempt {attempt}/{retries}: {exc}"
            errors.append(error)
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"Failed to download {item.url} -> {target_path}. Errors: {' | '.join(errors)}")


def download_missing_items(
    destination: Path,
    missing_items: list[DownloadItem],
    workers: int,
    retries: int,
    timeout: int,
    chunk_size: int,
) -> None:
    """Download missing update files concurrently.

    Args:
        destination: Local output directory.
        missing_items: Missing files to download.
        workers: Requested parallel workers, capped at 16.
        retries: Retry count for the direct download URL.
        timeout: Per-request timeout in seconds.
        chunk_size: Stream read chunk size in bytes.

    Returns:
        None.
    """
    if not missing_items:
        return

    buffer_dir = destination / ".download_tmp"
    buffer_dir.mkdir(parents=True, exist_ok=True)
    worker_count = min(max(1, workers), 16, len(missing_items))
    errors: list[str] = []
    item_iter = iter(missing_items)
    future_to_item = {}
    submitted = 0
    completed = 0
    executor = ThreadPoolExecutor(max_workers=worker_count)

    def submit_next() -> bool:
        """Submit one pending download task if available."""
        nonlocal submitted
        try:
            item = next(item_iter)
        except StopIteration:
            return False
        future = executor.submit(
            download_one,
            item,
            destination,
            buffer_dir,
            retries,
            timeout,
            chunk_size,
        )
        future_to_item[future] = item
        submitted += 1
        return True

    try:
        for _ in range(worker_count):
            submit_next()

        with tqdm(total=len(missing_items), desc=f"Downloading {destination.name}", unit="file") as pbar:
            while future_to_item:
                done, _ = wait(future_to_item, return_when=FIRST_COMPLETED)
                for future in done:
                    item = future_to_item.pop(future)
                    completed += 1
                    try:
                        future.result()
                    except RuntimeError as exc:
                        errors.append(str(exc))
                        tqdm.write(f"Failed {completed}/{len(missing_items)}: {item.filename}")
                    pbar.update(1)
                    submit_next()
    except KeyboardInterrupt:
        for future in future_to_item:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        log.warning(
            "Interrupted by user. Cancelled pending downloads; currently running downloads may finish. "
            "Temporary files remain under %s and are ignored on the next run.",
            buffer_dir,
        )
        raise
    else:
        executor.shutdown(wait=True)

    if errors:
        raise RuntimeError("Some update files failed to download: " + " || ".join(errors[:10]))
    cleanup_buffer_dir(buffer_dir)


def ensure_update_window_complete(
    destination: Path,
    start_time: dt.datetime,
    end_time: dt.datetime,
    collector: str,
    description: str,
    args: argparse.Namespace,
) -> None:
    """Download missing update files and verify the local window is complete.

    Args:
        destination: Local output directory.
        start_time: Inclusive update window start.
        end_time: Inclusive update window end.
        collector: RIPE RIS collector name.
        description: Human-readable window label for logs.
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    destination.mkdir(parents=True, exist_ok=True)
    plan = build_update_plan(start_time, end_time, collector)
    missing = list_missing_gz(destination, plan)

    if not missing:
        log.info("Complete locally, skip download: %s (%d files)", destination, len(plan))
        return

    missing_preview = ", ".join(item.filename for item in missing[:10])
    log.info("Missing or invalid files for %s: %s", destination, missing_preview)
    log.info(
        "Downloading %s: %s -> %s, missing=%d/%d, destination=%s",
        description,
        start_time.isoformat(),
        end_time.isoformat(),
        len(missing),
        len(plan),
        destination,
    )
    download_missing_items(
        destination=destination,
        missing_items=missing,
        workers=args.workers,
        retries=args.retries,
        timeout=args.timeout,
        chunk_size=args.chunk_size,
    )

    still_missing = list_missing_gz(destination, plan)
    if still_missing:
        preview = ", ".join(item.filename for item in still_missing[:10])
        raise RuntimeError(
            f"Incomplete update window after download: {destination}. "
            f"missing={len(still_missing)}/{len(plan)}. First missing files: {preview}"
        )

    log.info("Finished and verified: %s (%d files)", destination, len(plan))


def get_event_update_window(
    event: RADBenchEvent,
    prefix_minutes: int,
    suffix_minutes: int,
    unknown_duration_hours: int,
) -> tuple[dt.datetime, dt.datetime]:
    """Return the event update download window.

    Args:
        event: RADBench event metadata.
        prefix_minutes: Minutes included before event start.
        suffix_minutes: Minutes included after known event end.
        unknown_duration_hours: Final duration used when event end is unknown.

    Returns:
        `(start_time, end_time)` for event-window updates.
    """
    start_time = event.start_time - dt.timedelta(minutes=prefix_minutes)
    if event.end_time is None:
        end_time = event.start_time + dt.timedelta(hours=unknown_duration_hours)
    else:
        end_time = event.end_time + dt.timedelta(minutes=suffix_minutes)
    return start_time, end_time


def get_before_update_window(
    event_update_start: dt.datetime,
    before_days: int,
    before_gap_minutes: int,
) -> tuple[dt.datetime, dt.datetime]:
    """Return the before-event update download window.

    Args:
        event_update_start: Start time of the event update window.
        before_days: Number of days before the event window to download.
        before_gap_minutes: Gap before the event update window.

    Returns:
        `(start_time, end_time)` for before-event updates.
    """
    aligned_event_start = floor_to_5min(event_update_start)
    end_time = aligned_event_start - dt.timedelta(minutes=before_gap_minutes)
    start_time = end_time - dt.timedelta(days=before_days)
    return start_time, end_time


def download_event_updates(event: RADBenchEvent, args: argparse.Namespace) -> None:
    """Download event-window updates for one RADBench event.

    Args:
        event: RADBench event metadata.
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    event_start, event_end = get_event_update_window(
        event,
        prefix_minutes=args.event_prefix_minutes,
        suffix_minutes=args.event_suffix_minutes,
        unknown_duration_hours=args.unknown_duration_hours,
    )
    root_dir = Path(args.root_dir)
    ensure_update_window_complete(
        destination=root_dir / event.event_name,
        start_time=event_start,
        end_time=event_end,
        collector=args.collector,
        description=f"event updates for {event.event_name}",
        args=args,
    )


def download_before_updates(event: RADBenchEvent, args: argparse.Namespace) -> None:
    """Download before-event updates for one RADBench event.

    Args:
        event: RADBench event metadata.
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    event_start, _ = get_event_update_window(
        event,
        prefix_minutes=args.event_prefix_minutes,
        suffix_minutes=args.event_suffix_minutes,
        unknown_duration_hours=args.unknown_duration_hours,
    )
    before_start, before_end = get_before_update_window(
        event_start,
        before_days=args.before_days,
        before_gap_minutes=args.before_gap_minutes,
    )
    root_dir = Path(args.root_dir)
    ensure_update_window_complete(
        destination=root_dir / args.before_dir_name / event.event_name,
        start_time=before_start,
        end_time=before_end,
        collector=args.collector,
        description=f"before-event updates for {event.event_name}",
        args=args,
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate download CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None.
    """
    if args.skip_event_updates and not args.download_before_data:
        raise ValueError("At least one task must be enabled: event updates or --download-before-data.")
    if args.retries <= 0:
        raise ValueError("--retries must be greater than 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")
    if args.event_prefix_minutes < 0:
        raise ValueError("--event-prefix-minutes must be greater than or equal to 0")
    if args.event_suffix_minutes < 0:
        raise ValueError("--event-suffix-minutes must be greater than or equal to 0")
    if args.before_gap_minutes < 0:
        raise ValueError("--before-gap-minutes must be greater than or equal to 0")


def run_download_tasks(args: argparse.Namespace) -> None:
    """Run enabled RADBench download tasks.

    Args:
        args: Parsed and validated command-line arguments.

    Returns:
        None.
    """
    events = load_events(args.event_csv, args.event_type, args.event_name)
    log.info("Loaded %d target events from %s", len(events), args.event_csv)

    if not args.skip_event_updates:
        log.info("Starting event-window update downloads for all target events.")
        for event in events:
            log.info("Processing event-window updates: %s", event.event_name)
            download_event_updates(event, args)

    if args.download_before_data:
        log.info("Starting before-event update downloads for all target events.")
        for event in events:
            log.info("Processing before-event updates: %s", event.event_name)
            download_before_updates(event, args)

    log.info("All enabled RADBench update download tasks are complete.")


def main() -> None:
    """Parse arguments and run RADBench update downloads."""
    args = parse_args()
    validate_args(args)
    run_download_tasks(args)


if __name__ == "__main__":
    main()
