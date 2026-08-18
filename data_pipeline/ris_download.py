"""Download the RRC00 RIB and update files required by precise GT labeling.

The original repository routes these calls through a general-purpose downloader
covering many collectors and data products. Precise GT only uses three small
interfaces for RRC00; this module implements those interfaces directly while
preserving URL enumeration and local filename semantics.
"""

from __future__ import annotations

import datetime as dt
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


HTTPS = "HTTPS"
_GZIP_MAGIC = b"\x1f\x8b"
_MAX_WORKERS = 10
_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class BaseParams:
    """Parameters consumed by the source precise-label implementation."""

    start_time: dt.datetime
    end_time: dt.datetime
    bgp_data_type: str
    bgp_collectors: list[str]
    base_type: str = "BGP"
    minio_flag: bool = False


def set_base_params(
    start_time: str,
    end_time: str,
    data_types: str,
    collector: str,
    minio_flag: bool,
) -> BaseParams:
    """Parse the source downloader arguments into RRC00 download parameters.

    Args:
        start_time: Inclusive UTC lower bound in YYYY-mm-dd-HH:MM format.
        end_time: Inclusive UTC upper bound in the same format.
        data_types: BGP:RIBS, BGP:UPDATES, or BGP:ALL.
        collector: Collector name; precise GT requires rrc00.
        minio_flag: Retained for call compatibility. MinIO is not used.

    Returns:
        Parsed parameters used by getURLList.
    """
    base_type, separator, subtype = data_types.partition(":")
    if base_type != "BGP" or not separator:
        raise ValueError(f"Unsupported data type: {data_types}")
    if collector.lower() != "rrc00":
        raise ValueError("Precise RADBench GT requires collector rrc00")

    datetime_format = "%Y-%m-%d-%H:%M"
    start = dt.datetime.strptime(start_time, datetime_format)
    end = dt.datetime.strptime(end_time, datetime_format)
    if start > end:
        raise ValueError(f"start_time must not exceed end_time: {start_time} > {end_time}")

    normalized_subtype = subtype.upper() or "ALL"
    if normalized_subtype not in {"RIBS", "UPDATES", "ALL"}:
        raise ValueError(f"Unsupported BGP subtype: {subtype}")
    return BaseParams(
        start_time=start,
        end_time=end,
        bgp_data_type=normalized_subtype,
        bgp_collectors=["rrc00"],
        minio_flag=minio_flag,
    )


def getURLList(params: BaseParams) -> list[tuple[str, str]]:
    """Enumerate the same inclusive RRC00 archive interval as URLGetter.

    Args:
        params: Parsed RRC00 time range and data subtype.

    Returns:
        (HTTPS, url) tuples ordered chronologically, with RIBs before updates
        when ALL is requested.
    """
    if params.bgp_collectors != ["rrc00"]:
        raise ValueError("Precise RADBench GT requires collector rrc00")

    urls: list[tuple[str, str]] = []
    if params.bgp_data_type in {"RIBS", "ALL"}:
        day = params.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= params.end_time:
            for hour in (0, 8, 16):
                point = day.replace(hour=hour)
                if params.start_time <= point <= params.end_time:
                    urls.append(
                        (
                            HTTPS,
                            f"https://data.ris.ripe.net/rrc00/{point:%Y.%m}/"
                            f"bview.{point:%Y%m%d.%H%M}.gz",
                        )
                    )
            day += dt.timedelta(days=1)

    if params.bgp_data_type in {"UPDATES", "ALL"}:
        point = params.start_time.replace(second=0, microsecond=0)
        remainder = point.minute % 5
        if remainder:
            point += dt.timedelta(minutes=5 - remainder)
        while point <= params.end_time:
            urls.append(
                (
                    HTTPS,
                    f"https://data.ris.ripe.net/rrc00/{point:%Y.%m}/"
                    f"updates.{point:%Y%m%d.%H%M}.gz",
                )
            )
            point += dt.timedelta(minutes=5)
    return urls


def _is_valid_gzip(path: Path) -> bool:
    """Return whether a local archive is non-empty and has gzip magic bytes."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as file_obj:
        return file_obj.read(len(_GZIP_MAGIC)) == _GZIP_MAGIC


def _download_one(url: str, destination: Path) -> None:
    """Download one RRC00 archive atomically and validate its container."""
    target = destination / f"rrc00_{url.rsplit('/', 1)[-1]}"
    if _is_valid_gzip(target):
        return

    temporary = target.with_name(f"{target.name}.part")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Wget/1.21.1"})
            with opener.open(request, timeout=60) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if not _is_valid_gzip(temporary):
                raise RuntimeError(f"Downloaded file is not a valid gzip archive: {url}")
            temporary.replace(target)
            return
        except (OSError, RuntimeError) as exc:
            last_error = exc
            if temporary.exists():
                temporary.unlink()
            if attempt < _MAX_ATTEMPTS:
                time.sleep(attempt)

    raise RuntimeError(f"Failed to download {url} after {_MAX_ATTEMPTS} attempts") from last_error


class downloadByParams:
    """Compatibility wrapper used by the unmodified precise-GT call sites."""

    def __init__(
        self,
        base_type: str,
        minio_enable: bool,
        urllist: list[tuple[str, str]],
        destination: str,
    ) -> None:
        if base_type != "BGP":
            raise ValueError(f"Unsupported base type: {base_type}")
        if minio_enable:
            raise ValueError("The RRC00 compatibility downloader does not use MinIO")
        if any(protocol != HTTPS for protocol, _ in urllist):
            raise ValueError("Only HTTPS RRC00 downloads are supported")
        self.urllist = urllist
        self.destination = Path(destination)

    def start_on(self) -> None:
        """Download all requested files and fail if any worker fails."""
        self.destination.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = [
                executor.submit(_download_one, url, self.destination)
                for _, url in self.urllist
            ]
            for future in futures:
                future.result()
