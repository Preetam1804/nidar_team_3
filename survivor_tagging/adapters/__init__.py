"""Adapters for real and synthetic data sources.

* InMemoryProviders    - used by tests and the discrete-event simulator.
* JSONLPlaybackAdapter - replays recorded mission logs (or a wrapper for
                         a live JSON-lines socket at integration time).
* Stub providers       - return None (no data) until real modules exist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .. import schemas
from ..providers import DetectionsProvider, MapProvider, PoseProvider

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# In-memory providers (tests / simulator)
# ----------------------------------------------------------------------

class InMemoryDetectors(DetectionsProvider):
    """Feeds a list of DetectionFrames manually; pops as consumed."""

    def __init__(self) -> None:
        self._frames: list[schemas.DetectionFrame] = []
        self._consumed: set[str] = set()

    def push(self, frame: schemas.DetectionFrame) -> None:
        self._frames.append(frame)

    def latest_frame(self) -> schemas.DetectionFrame | None:
        if not self._frames:
            return None
        return self._frames[0]

    def ack(self, frame_id: str) -> None:
        if self._frames:
            fid = self._frames[0].frame_id
            self._consumed.add(str(fid))
            self._frames.pop(0)


class InMemoryPose(PoseProvider):
    """Yields the latest pushed PoseSample."""

    def __init__(self) -> None:
        self._sample: schemas.PoseSample | None = None

    def push(self, s: schemas.PoseSample) -> None:
        self._sample = s

    def latest(self) -> schemas.PoseSample | None:
        return self._sample


class InMemoryMap(MapProvider):
    """Yields the latest pushed OccupancyGridMsg."""

    def __init__(self) -> None:
        self._map: schemas.OccupancyGridMsg | None = None

    def push(self, m: schemas.OccupancyGridMsg) -> None:
        self._map = m

    def latest(self) -> schemas.OccupancyGridMsg | None:
        return self._map


# ----------------------------------------------------------------------
# JSONL playback (recorded logs / live JSON-lines stream)
# ----------------------------------------------------------------------

class JSONLPlaybackAdapter:
    """Replays events from a JSONL file {source, stamp, payload} rows.

    One row per line, each shaped like:
        {"source": "detections"|"pose"|"map", "payload": {...}}

    Rows are handed to an in-memory provider each time advance() is called
    with a stamp that has passed the row's own stamp, giving deterministic
    replay for benchmarking.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._rows: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("JSONL parse error at %s: %s", self.path, exc)
                    continue
                self._rows.append(row)
        self._rows.sort(key=lambda r: r.get("stamp", 0.0))
        self._cursor = 0
        self.detectors = InMemoryDetectors()
        self.pose = InMemoryPose()
        self.map = InMemoryMap()

    def advance_to(self, stamp: float) -> None:
        """Feed all rows with row stamp <= stamp into the providers."""
        while self._cursor < len(self._rows):
            row = self._rows[self._cursor]
            if float(row.get("stamp", 0.0)) > stamp:
                break
            self._dispatch(row)
            self._cursor += 1

    def _dispatch(self, row: dict[str, Any]) -> None:
        src = row.get("source")
        payload = row.get("payload")
        if src == "detections":
            try:
                frame = schemas.parse_detection_frame(payload)
            except schemas.SchemaError as exc:
                log.warning("Skipping bad detection row: %s", exc)
                return
            self.detectors.push(frame)
        elif src == "pose":
            try:
                sample = schemas.parse_pose(payload)
            except schemas.SchemaError as exc:
                log.warning("Skipping bad pose row: %s", exc)
                return
            self.pose.push(sample)
        elif src == "map":
            try:
                m = schemas.parse_map(payload)
            except schemas.SchemaError as exc:
                log.warning("Skipping bad map row: %s", exc)
                return
            self.map.push(m)
        else:
            log.warning("Unknown JSONL source %r", src)


# ----------------------------------------------------------------------
# Stubs for missing real modules
# ----------------------------------------------------------------------

class StubProvider(DetectionsProvider, PoseProvider):
    """Returns no data until real modules arrive at integration time."""

    def latest(self) -> schemas.PoseSample | None:
        return None

    def latest_frame(self) -> schemas.DetectionFrame | None:
        return None

    def ack(self, frame_id: str) -> None:
        return None


class NullMap(MapProvider):
    def latest(self) -> schemas.OccupancyGridMsg | None:
        return None
