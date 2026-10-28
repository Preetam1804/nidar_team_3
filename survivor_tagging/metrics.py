"""Runtime metrics for the survivor_tagging module.

Collects, per mission:
* per-stage latency samples (projection, association, publish)
* tag lifecycle event counts (created, confirmed, removed, withdrawn)
* position-error histogram (only populated when ground truth is
  available, i.e. in the simulator / bench playback)
* legacy false-tag / duplicate counters

Everything is plain Python; formatted into a benchmark report at the end
without needing plotting libraries.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field


@dataclass
class Histogram:
    edges: list[float]
    counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.edges) + 1)   # +1 for overflow

    def add(self, value: float) -> None:
        idx = 0
        for e in self.edges:
            if value <= e:
                break
            idx += 1
        self.counts[idx] += 1

    def as_dict(self) -> dict:
        return {"edges": list(self.edges), "counts": list(self.counts)}

    def median_bin(self) -> float | None:
        total = sum(self.counts)
        if total == 0:
            return None
        acc = 0
        for c, e in zip(self.counts, [self.edges[0]] + self.edges):
            acc += c
            if acc >= total / 2.0:
                return e
        return self.edges[-1]


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.monotonic)
    ticks: int = 0
    detections_ingested: int = 0
    pose_ingested: int = 0
    latency: dict[str, list[float]] = field(default_factory=lambda: {
        "project_ms": [], "associate_ms": [], "publish_ms": [], "tick_ms": [],
    })
    tag_events: dict[str, int] = field(default_factory=lambda: {
        "created": 0, "confirmed": 0, "removed": 0, "withdrawn": 0, "published": 0,
    })
    err_hist: Histogram = field(default_factory=lambda: Histogram(
        edges=[0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]))
    # end-to-emit latency measured in real time in the pipeline
    e2e_latency_s: list[float] = field(default_factory=list)
    # CPU and RAM resource monitoring
    start_cpu_time: float = field(default_factory=time.process_time)
    ram_samples_mb: list[float] = field(default_factory=list)
    peak_ram_mb: float = 0.0
    # sim-specific
    correct_tags: int = 0
    incorrect_tags: int = 0
    duplicates: int = 0

    def record_latency(self, stage: str, seconds: float) -> None:
        hist = self.latency.get(stage)
        if hist is not None:
            hist.append(seconds * 1000.0)

    def record_tag_event(self, name: str) -> None:
        if name in self.tag_events:
            self.tag_events[name] += 1

    def record_error(self, metres: float) -> None:
        self.err_hist.add(metres)

    def sample_resources(self) -> None:
        """Sample current process memory and peak allocation."""
        ram_mb = 0.0
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            cur, peak = tracemalloc.get_traced_memory()
            ram_mb = cur / (1024.0 * 1024.0)
            self.peak_ram_mb = max(self.peak_ram_mb, peak / (1024.0 * 1024.0))
        except Exception:
            pass

        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            ru_max = usage.ru_maxrss
            # Linux reports in KiB, macOS in bytes
            ru_mb = ru_max / (1024.0 * 1024.0) if ru_max > 1e7 else ru_max / 1024.0
            self.peak_ram_mb = max(self.peak_ram_mb, ru_mb)
            if ram_mb == 0.0:
                ram_mb = ru_mb
        except Exception:
            pass

        try:
            import psutil
            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
            ram_mb = rss_mb
            self.peak_ram_mb = max(self.peak_ram_mb, rss_mb)
        except Exception:
            pass

        if ram_mb > 0.0:
            self.ram_samples_mb.append(round(ram_mb, 2))

    # ------------------------------------------------------------------
    @staticmethod
    def _summary(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        s = sorted(values)
        return {
            "n": len(s),
            "mean_ms": round(statistics.mean(s), 3),
            "p50_ms": round(s[len(s) // 2], 3),
            "p95_ms": round(s[int(0.95 * (len(s) - 1))], 3),
            "max_ms": round(s[-1], 3),
        }

    def latency_summary(self) -> dict:
        return {k: self._summary(v) for k, v in self.latency.items()}

    def summary(self) -> dict:
        self.sample_resources()
        total = self.tag_events.get("created", 0)
        confirmed = self.tag_events.get("confirmed", 0)
        removed = self.tag_events.get("removed", 0) + self.tag_events.get("withdrawn", 0)
        scoreable = self.correct_tags + self.incorrect_tags
        cpu_total_s = max(0.0, time.process_time() - self.start_cpu_time)
        wall_total_s = max(1e-6, time.monotonic() - self.started_at)
        cpu_ms_per_tick = round((cpu_total_s / self.ticks) * 1000.0, 3) if self.ticks else 0.0
        cpu_util_pct = round((cpu_total_s / wall_total_s) * 100.0, 2)
        curr_ram = self.ram_samples_mb[-1] if self.ram_samples_mb else 0.0
        peak_ram = max(self.peak_ram_mb, curr_ram)
        return {
            "ticks": self.ticks,
            "detections_ingested": self.detections_ingested,
            "pose_ingested": self.pose_ingested,
            "tag_events": dict(self.tag_events),
            "track_churn": {
                "created": total,
                "removed_or_withdrawn": removed,
                "churn_per_confirmed":
                    round(removed / confirmed, 3) if confirmed else 0.0,
            },
            "latency": self.latency_summary(),
            "resources": {
                "cpu_user_sys_s": round(cpu_total_s, 4),
                "cpu_ms_per_tick": cpu_ms_per_tick,
                "cpu_util_pct": cpu_util_pct,
                "ram_current_mb": round(curr_ram, 2),
                "ram_peak_mb": round(peak_ram, 2),
            },
            "position_error_hist": self.err_hist.as_dict(),
            "sim_scoring": {
                "correct": self.correct_tags,
                "incorrect": self.incorrect_tags,
                "correct_rate":
                    round(self.correct_tags / scoreable, 4) if scoreable else None,
                "duplicates": self.duplicates,
            },
        }
