"""Online-equivalent performance accounting for TRAIL research entrypoints.

The recorder measures only explicitly marked computation blocks. Repository
cache I/O stays unchanged and remains outside those blocks.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import psutil


log = logging.getLogger(__name__)
_ACTIVE_RECORDER: "PerformanceRecorder | None" = None


def _beijing_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def _cuda_sync() -> None:
    """Synchronize CUDA when available so GPU timings are complete."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_active_recorder() -> "PerformanceRecorder | None":
    """Return the recorder owned by the current research entrypoint."""
    return _ACTIVE_RECORDER


class PerformanceRecorder:
    """Accumulate computation timings and resource peaks for one entrypoint.

    Args:
        entrypoint: Stable name of the invoking research script.
        output_root: Directory receiving the unique JSON report.
        config_path: Effective YAML configuration path.
        sample_interval_seconds: RSS sampling interval.

    Outputs:
        A JSON report written during context exit.
    """

    def __init__(
        self,
        *,
        entrypoint: str,
        output_root: str | Path,
        config_path: str | Path,
        sample_interval_seconds: float = 0.1,
    ) -> None:
        self.entrypoint = entrypoint
        self.output_root = Path(output_root)
        self.config_path = str(Path(config_path).resolve())
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.started_at = _beijing_now()
        self.run_id = self.started_at.strftime("%Y%m%d-%H%M%S-%f")
        self.sections: dict[str, dict[str, Any]] = {}
        self.summary: dict[str, Any] = {}
        self._active_sections: dict[str, int] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._sampler: threading.Thread | None = None
        self._finished = False
        self.report_path: Path | None = None

    def __enter__(self) -> "PerformanceRecorder":
        global _ACTIVE_RECORDER
        if _ACTIVE_RECORDER is not None:
            raise RuntimeError("Only one PerformanceRecorder may be active per process")
        _ACTIVE_RECORDER = self
        self._sampler = threading.Thread(
            target=self._sample_loop,
            name="trail-performance-sampler",
            daemon=True,
        )
        self._sampler.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        global _ACTIVE_RECORDER
        try:
            if exc_value is None:
                self.finish(status="completed")
            else:
                self.summary["error"] = {
                    "type": type(exc_value).__name__,
                    "message": str(exc_value),
                }
                self.finish(status="failed")
        finally:
            _ACTIVE_RECORDER = None
        return False

    @contextmanager
    def measure(self, section: str, *, cuda: bool = False) -> Iterator[None]:
        """Measure one necessary computation block and accumulate repeated calls."""
        if cuda:
            _cuda_sync()
            self._reset_gpu_peak()
        self._activate(section)
        with self._lock:
            excluded_before = float(self._section(section)["excluded_seconds"])
        start = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                _cuda_sync()
            elapsed = time.perf_counter() - start
            with self._lock:
                excluded_after = float(self._section(section)["excluded_seconds"])
            elapsed = max(0.0, elapsed - (excluded_after - excluded_before))
            gpu_peak = self._gpu_peak_bytes() if cuda else 0
            self._deactivate(section, elapsed, gpu_peak, increment_call=True)

    def add_elapsed(self, section: str, elapsed_seconds: float, *, calls: int = 1) -> None:
        """Add timing measured inside a worker or a fine-grained hot loop."""
        with self._lock:
            item = self._section(section)
            item["status"] = "executed"
            item["compute_seconds"] += float(elapsed_seconds)
            item["calls"] += int(calls)

    def subtract_elapsed(self, section: str, elapsed_seconds: float) -> None:
        """Accumulate logging or persistence time excluded by an outer measure."""
        with self._lock:
            if section not in self.sections:
                return
            item = self.sections[section]
            item["excluded_seconds"] += float(elapsed_seconds)

    @contextmanager
    def track_resources(self, section: str) -> Iterator[None]:
        """Attribute sampled resource peaks without adding wall-clock time."""
        self._activate(section)
        try:
            yield
        finally:
            self._deactivate(section, 0.0, 0, increment_call=False)

    def mark_reused(self, section: str) -> None:
        """Record that a cache prevented the computation block from executing."""
        with self._lock:
            item = self._section(section)
            item["status"] = "reused"

    def add_count(self, section: str, name: str, value: int) -> None:
        """Add an integer input/output count to a section."""
        with self._lock:
            counts = self._section(section).setdefault("counts", {})
            counts[name] = int(counts.get(name, 0)) + int(value)

    def set_summary(self, **values: Any) -> None:
        """Attach derived totals and fixed operating-point metadata."""
        self.summary.update(values)

    def finish(self, *, status: str) -> Path:
        """Stop resource sampling, print a summary, and persist one JSON report."""
        if self._finished:
            if self.report_path is None:
                raise RuntimeError("Finished recorder has no report path")
            return self.report_path
        self._stop_event.set()
        if self._sampler is not None:
            self._sampler.join(timeout=max(1.0, self.sample_interval_seconds * 4))
        self._sample_once()
        ended_at = _beijing_now()
        self._derive_summary()
        report = {
            "schema_version": 1,
            "run_id": self.run_id,
            "entrypoint": self.entrypoint,
            "status": status,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "config_path": self.config_path,
            "hardware": self._hardware_snapshot(),
            "sections": self.sections,
            "summary": self.summary,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.report_path = self.output_root / f"{self.entrypoint}_{self.run_id}.json"
        self.report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._finished = True
        aggregate_path = (
            self._write_aggregate_if_complete() if status == "completed" else None
        )
        if aggregate_path is not None:
            self.summary["aggregate_report_path"] = str(aggregate_path)
        self._log_summary(status)
        return self.report_path

    def _section(self, name: str) -> dict[str, Any]:
        return self.sections.setdefault(
            name,
            {
                "status": "executed",
                "compute_seconds": 0.0,
                "calls": 0,
                "excluded_seconds": 0.0,
                "peak_rss_bytes": 0,
                "peak_gpu_reserved_bytes": 0,
            },
        )

    def _activate(self, section: str) -> None:
        first_call = False
        with self._lock:
            item = self._section(section)
            item["status"] = "executed"
            first_call = item["calls"] == 0 and section not in self._active_sections
            self._active_sections[section] = self._active_sections.get(section, 0) + 1
        if first_call:
            rss = self._process_tree_rss()
            with self._lock:
                item = self._section(section)
                item["peak_rss_bytes"] = max(int(item["peak_rss_bytes"]), rss)

    def _deactivate(
        self,
        section: str,
        elapsed: float,
        gpu_peak: int,
        *,
        increment_call: bool,
    ) -> None:
        with self._lock:
            item = self._section(section)
            item["compute_seconds"] += float(elapsed)
            if increment_call:
                item["calls"] += 1
            item["peak_gpu_reserved_bytes"] = max(
                int(item["peak_gpu_reserved_bytes"]), int(gpu_peak)
            )
            remaining = self._active_sections.get(section, 0) - 1
            if remaining > 0:
                self._active_sections[section] = remaining
            else:
                self._active_sections.pop(section, None)

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._sample_once()

    def _sample_once(self) -> None:
        rss = self._process_tree_rss()
        with self._lock:
            for section in self._active_sections:
                item = self._section(section)
                item["peak_rss_bytes"] = max(int(item["peak_rss_bytes"]), rss)

    @staticmethod
    def _process_tree_rss() -> int:
        process = psutil.Process()
        processes = [process]
        try:
            processes.extend(process.children(recursive=True))
        except psutil.Error:
            pass
        total = 0
        for candidate in processes:
            try:
                total += int(candidate.memory_info().rss)
            except psutil.Error:
                continue
        return total

    @staticmethod
    def _reset_gpu_peak() -> None:
        import torch

        if not torch.cuda.is_available():
            return
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)

    @staticmethod
    def _gpu_peak_bytes() -> int:
        import torch

        if not torch.cuda.is_available():
            return 0
        return max(
            int(torch.cuda.max_memory_reserved(index))
            for index in range(torch.cuda.device_count())
        )

    @staticmethod
    def _hardware_snapshot() -> dict[str, Any]:
        import torch

        gpu_names = []
        if torch.cuda.is_available():
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "total_memory_bytes": int(psutil.virtual_memory().total),
            "gpu_count": len(gpu_names),
            "gpu_names": gpu_names,
            "torch_num_threads": int(torch.get_num_threads()),
            "pid": os.getpid(),
        }

    def _derive_summary(self) -> None:
        """Build totals only when every required component actually executed."""
        training_names = [
            "training.reconstructor",
            "training.rsf",
            "training.classifier",
        ]
        training_components = {
            name.removeprefix("training."): float(self.sections[name]["compute_seconds"])
            for name in training_names
            if name in self.sections
        }
        if training_components:
            self.summary["training_component_seconds"] = training_components
        present_training_names = [
            name for name in training_names if name in self.sections
        ]
        if len(present_training_names) == len(training_names) and all(
            self.sections[name]["status"] == "executed" for name in training_names
        ):
            self.summary["training_total_seconds"] = sum(
                float(self.sections[name]["compute_seconds"]) for name in training_names
            )
        elif any(
            self.sections[name]["status"] != "executed"
            for name in present_training_names
        ):
            self.summary["training_status"] = "incomplete"

        preprocess_test = sum(
            float(item["compute_seconds"])
            for name, item in self.sections.items()
            if name.startswith("preprocess.test.") and item["status"] == "executed"
        )
        preprocess_test_items = {
            name: item["status"]
            for name, item in self.sections.items()
            if name.startswith("preprocess.test.")
        }
        if preprocess_test:
            self.summary["test_preprocess_seconds"] = preprocess_test
        if preprocess_test_items:
            self.summary["test_preprocess_component_status"] = preprocess_test_items
        expected_test_events = self.summary.get("preprocess_expected_events", {}).get(
            "test", []
        )
        expected_test_sections = [
            f"preprocess.test.{stage}.{event_name}"
            for event_name in expected_test_events
            for stage in ("raw", "sparse", "sequence")
        ]
        if expected_test_sections:
            expected_status = {
                name: self.sections.get(name, {}).get("status", "missing")
                for name in expected_test_sections
            }
            self.summary["test_preprocess_expected_status"] = expected_status
            if any(status != "executed" for status in expected_status.values()):
                self.summary["test_preprocess_status"] = "incomplete"

        inference_names = [
            "inference.reconstructor.test",
            "inference.rsf.test",
            "inference.classifier.test",
            "inference.threshold_adf_alert.test",
        ]
        statuses = {
            name: self.sections.get(name, {}).get("status", "missing")
            for name in inference_names
        }
        if any(name in self.sections for name in inference_names):
            self.summary["inference_component_status"] = statuses
        if all(status == "executed" for status in statuses.values()):
            inference_seconds = sum(
                float(self.sections[name]["compute_seconds"]) for name in inference_names
            )
            self.summary["inference_seconds"] = inference_seconds
            if preprocess_test:
                total = preprocess_test + inference_seconds
                self.summary["online_equivalent_processing_time_seconds"] = total
                traffic = self.summary.get("test_traffic_duration_seconds")
                if isinstance(traffic, (int, float)) and traffic > 0 and total > 0:
                    self.summary["realtime_speedup"] = float(traffic) / total
        elif any(name in self.sections for name in inference_names):
            self.summary["inference_status"] = "incomplete"

        if self.sections:
            self.summary["peak_rss_bytes"] = max(
                int(item["peak_rss_bytes"]) for item in self.sections.values()
            )
            self.summary["peak_gpu_reserved_bytes"] = max(
                int(item["peak_gpu_reserved_bytes"]) for item in self.sections.values()
            )

    def _write_aggregate_if_complete(self) -> Path | None:
        """Write a cross-entrypoint snapshot when all four experiment reports exist."""
        latest: dict[str, dict[str, Any]] = {}
        required = {
            "build_dataset",
            "train_reconstructor",
            "train_detector",
            "evaluate",
        }
        for candidate in sorted(self.output_root.glob("*.json")):
            if candidate.name.startswith("aggregate_"):
                continue
            try:
                report = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.error("Failed to read performance report %s: %s", candidate, exc)
                raise
            if report.get("config_path") != self.config_path or report.get("status") != "completed":
                continue
            entrypoint = report.get("entrypoint")
            if entrypoint in required:
                latest[str(entrypoint)] = {"path": str(candidate), "report": report}
        if set(latest) != required:
            return None

        build_summary = latest["build_dataset"]["report"].get("summary", {})
        recon_summary = latest["train_reconstructor"]["report"].get("summary", {})
        detector_summary = latest["train_detector"]["report"].get("summary", {})
        eval_summary = latest["evaluate"]["report"].get("summary", {})
        if (
            build_summary.get("test_preprocess_status") == "incomplete"
            or recon_summary.get("training_status") == "incomplete"
            or detector_summary.get("training_status") == "incomplete"
        ):
            return None
        preprocess_seconds = build_summary.get("test_preprocess_seconds")
        inference_seconds = eval_summary.get("inference_seconds")
        recon_seconds = recon_summary.get("training_component_seconds", {}).get("reconstructor")
        rsf_seconds = detector_summary.get("training_component_seconds", {}).get("rsf")
        classifier_seconds = detector_summary.get("training_component_seconds", {}).get("classifier")
        values = [preprocess_seconds, inference_seconds, recon_seconds, rsf_seconds, classifier_seconds]
        if not all(isinstance(value, (int, float)) for value in values):
            return None

        detection_seconds = float(preprocess_seconds) + float(inference_seconds)
        traffic_seconds = eval_summary.get("test_traffic_duration_seconds")
        aggregate = {
            "schema_version": 1,
            "created_at": _beijing_now().isoformat(),
            "config_path": self.config_path,
            "component_reports": {name: item["path"] for name, item in latest.items()},
            "training_component_seconds": {
                "reconstructor": float(recon_seconds),
                "rsf": float(rsf_seconds),
                "classifier": float(classifier_seconds),
            },
            "training_total_seconds": float(recon_seconds) + float(rsf_seconds) + float(classifier_seconds),
            "test_preprocess_seconds": float(preprocess_seconds),
            "inference_seconds": float(inference_seconds),
            "online_equivalent_processing_time_seconds": detection_seconds,
            "test_traffic_duration_seconds": traffic_seconds,
            "realtime_speedup": (
                float(traffic_seconds) / detection_seconds
                if isinstance(traffic_seconds, (int, float)) and traffic_seconds > 0 and detection_seconds > 0
                else None
            ),
            "peak_rss_bytes": max(
                int(item["report"].get("summary", {}).get("peak_rss_bytes", 0))
                for item in latest.values()
            ),
            "peak_gpu_reserved_bytes": max(
                int(item["report"].get("summary", {}).get("peak_gpu_reserved_bytes", 0))
                for item in latest.values()
            ),
        }
        path = self.output_root / f"aggregate_{self.run_id}.json"
        path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Aggregate performance JSON: %s", path)
        return path

    def _log_summary(self, status: str) -> None:
        """Print compact entrypoint totals while event details remain in JSON."""
        log.info("Performance report status=%s entrypoint=%s", status, self.entrypoint)
        for group in ("before3day", "train", "test"):
            total = sum(
                float(item["compute_seconds"])
                for name, item in self.sections.items()
                if name.startswith(f"preprocess.{group}.")
                and item["status"] == "executed"
            )
            if total > 0:
                log.info("Performance preprocess.%-12s %.6fs", group, total)
        for name in (
            "training.reconstructor",
            "training.rsf",
            "training.classifier",
            "inference.reconstructor.test",
            "inference.rsf.test",
            "inference.classifier.test",
            "inference.threshold_adf_alert.test",
        ):
            if name not in self.sections:
                continue
            item = self.sections[name]
            log.info(
                "Performance %-48s status=%s compute=%.6fs",
                name,
                item["status"],
                float(item["compute_seconds"]),
            )
        if "online_equivalent_processing_time_seconds" in self.summary:
            log.info(
                "Performance online-equivalent total: %.6fs speedup=%s",
                float(self.summary["online_equivalent_processing_time_seconds"]),
                self.summary.get("realtime_speedup", "n/a"),
            )
        log.info(
            "Performance peak RAM=%.3fGiB peak GPU=%.3fGiB",
            int(self.summary.get("peak_rss_bytes", 0)) / 1024**3,
            int(self.summary.get("peak_gpu_reserved_bytes", 0)) / 1024**3,
        )
        log.info("Performance JSON: %s", self.report_path)


@contextmanager
def exclude_if_active(section: str) -> Iterator[None]:
    """Exclude logging or persistence nested inside an active measured block."""
    recorder = get_active_recorder()
    started = time.perf_counter()
    try:
        yield
    finally:
        if recorder is not None:
            recorder.subtract_elapsed(section, time.perf_counter() - started)


@contextmanager
def measure_if_active(section: str, *, cuda: bool = False) -> Iterator[None]:
    """Measure a block when an entrypoint recorder is active; otherwise no-op."""
    recorder = get_active_recorder()
    if recorder is None:
        yield
        return
    with recorder.measure(section, cuda=cuda):
        yield
