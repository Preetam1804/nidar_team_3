"""Map-frame multi-detection tracker with dedup and lifecycle.

Tracks survive across camera frames because the survivor does not move
and the drone does.  Association therefore happens in the map frame
(never camera-frame IoU, which is meaningless once the camera rotates).

Lifecycle per plan v3:
  TENTATIVE  - one detection observed, not yet confirmed
  CONFIRMED  - >= confirm_hits detections within the last confirm_window
  REMOVED    - no hit within max_missed_ticks (tag withdrawn)

Association policy:
  For each new projected detection, find the nearest existing track.
  If the distance is within gate_dist_m:
      - within hit_radius_m: counts as a hit on that track
      - else: a new track is created only if no track matches; this
        bounds duplicate-tag drift for survivors seen from slightly
        different viewpoints
  If the nearest track is further than gate_dist_m: create a new track.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

from . import schemas

TENTATIVE = schemas.TAG_STATE_TENTATIVE
CONFIRMED = schemas.TAG_STATE_CONFIRMED
REMOVED = schemas.TAG_STATE_REMOVED


@dataclass
class Track:
    track_id: str
    x: float               # map-frame running position estimate
    y: float
    conf: float            # max detection confidence seen
    hits: int
    misses: int
    last_seen_stamp: float
    first_seen_stamp: float
    state: str = TENTATIVE
    recent_hits: list[float] = field(default_factory=list)   # stamps of hits
    edge_zone: bool = False        # last detection was in the edge zone
    low_ray: bool = False          # last detection had a low ray
    depth_used: bool = False       # LiDAR refinement applied (future use)
    snapped: bool = False          # last detection required occupancy snap

    def age_s(self, now: float) -> float:
        return now - self.first_seen_stamp

    def hit_fraction(self, window: int) -> float:
        """Fraction of the last `window` detections that were hits."""
        if window <= 0:
            return 1.0
        return min(1.0, self.hits / window)


from typing import Callable


class Tracker:
    """Owns the set of active survivor tracks in the map frame."""

    def __init__(
        self,
        cfg: dict,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        t = cfg["tracker"]
        self.gate_dist_m: float = float(t["gate_dist_m"])
        self.hit_radius_m: float = float(t["hit_radius_m"])
        self.confirm_hits: int = int(t["confirm_hits"])
        self.confirm_window: int = int(t["confirm_window"])
        self.max_missed_ticks: int = int(t["max_missed_ticks"])
        self.max_tracks: int = int(t["max_map_units"])

        self._tracks: dict[str, Track] = {}
        self._removed: list[str] = []
        self._on_event = on_event

    # ------------------------------------------------------------------
    def active_tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.state == CONFIRMED]

    def removed_track_ids(self) -> list[str]:
        return list(self._removed)

    # ------------------------------------------------------------------
    def ingest(
        self,
        projected: list[tuple[float, float, float, float, bool, bool, bool]],
        now: float,
    ) -> None:
        """Ingest already-projected detections for this tick.

        Each element is (x, y, conf, edge_zone_flag, low_ray_flag,
        snapped_flag) in the map frame.
        """
        if len(self._tracks) >= self.max_tracks and projected:
            # Drop tracks with the worst misses first under track pressure.
            self._evict_worst(now)
        for x, y, conf, edge, low, snapped in projected:
            self._ingest_one(x, y, conf, edge, low, snapped, now)
        self._age(now)

    def _ingest_one(
        self,
        x: float, y: float, conf: float,
        edge: bool, low: bool, snapped: bool,
        now: float,
    ) -> None:
        nearest_id, nearest_dist = self._nearest(x, y)
        if nearest_id is not None and nearest_dist <= self.hit_radius_m:
            self._update_track(nearest_id, x, y, conf, edge, low, snapped, now)
            return
        if nearest_id is not None and nearest_dist <= self.gate_dist_m:
            # Ambiguous: close enough to gate but not a hit.  Do not open
            # a competing track this close to an existing one; this is
            # the duplicate-suppression policy per plan section C.
            # Still refresh the track's "seen-but-not-hit" timer so a
            # survivor drifting slowly between shots is not dropped.
            self._tracks[nearest_id].misses += 1
            self._tracks[nearest_id].conf = max(self._tracks[nearest_id].conf, conf)
            return
        self._create(x, y, conf, edge, low, snapped, now)

    def _nearest(self, x: float, y: float) -> tuple[str | None, float]:
        best_id: str | None = None
        best_d = float("inf")
        for tid, tr in self._tracks.items():
            d = math.hypot(tr.x - x, tr.y - y)
            if d < best_d:
                best_d = d
                best_id = tid
        return best_id, best_d

    def _create(
        self,
        x: float, y: float, conf: float,
        edge: bool, low: bool, snapped: bool,
        now: float,
    ) -> Track:
        tr = Track(
            track_id=f"tag_{uuid.uuid4().hex[:12]}",
            x=x, y=y, conf=conf, hits=1, misses=0,
            last_seen_stamp=now, first_seen_stamp=now,
            state=TENTATIVE,
            recent_hits=[now],
            edge_zone=edge, low_ray=low,
            snapped=snapped,
        )
        self._tracks[tr.track_id] = tr
        if self._on_event:
            self._on_event("created")
        return tr

    def _update_track(
        self,
        tid: str, x: float, y: float, conf: float,
        edge: bool, low: bool, snapped: bool,
        now: float,
    ) -> None:
        tr = self._tracks[tid]
        # Running average keeps the tag stable as more shots come in.
        w = 1.0 / (1.0 + float(tr.hits - 1)) if tr.hits > 0 else 1.0
        tr.x = tr.x + w * (x - tr.x)
        tr.y = tr.y + w * (y - tr.y)
        tr.conf = max(tr.conf, conf)
        tr.hits += 1
        tr.misses = 0
        tr.last_seen_stamp = now
        tr.recent_hits.append(now)
        if len(tr.recent_hits) > self.confirm_window:
            tr.recent_hits.pop(0)
        tr.edge_zone = edge
        tr.low_ray = low
        tr.snapped = tr.snapped or snapped
        if tr.state != CONFIRMED and len(tr.recent_hits) >= self.confirm_hits:
            tr.state = CONFIRMED
            if self._on_event:
                self._on_event("confirmed")

    def _age(self, now: float) -> None:
        for tid in list(self._tracks.keys()):
            tr = self._tracks[tid]
            if (now - tr.last_seen_stamp) > self.max_missed_ticks:
                tr.state = REMOVED
                self._removed.append(tid)
                self._tracks.pop(tid)
                if self._on_event:
                    self._on_event("removed")

    def _evict_worst(self, now: float) -> None:
        if not self._tracks:
            return
        worst_id, worst = max(
            self._tracks.items(), key=lambda kv: kv[1].misses
        )
        if worst.misses == 0 and len(self._tracks) < self.max_tracks:
            return
        worst.state = REMOVED
        self._removed.append(worst_id)
        self._tracks.pop(worst_id)
        if self._on_event:
            self._on_event("removed")

    def withdraw_if_occupied(
        self,
        occupancy_fn,
    ) -> list[str]:
        """Withdraw tracks whose map cell has become OCCUPIED.

        occupancy_fn(col, row) -> bool (True when the cell is occupied).
        Withdrawn track ids are returned.
        """
        withdrawn = []
        for tid in list(self._tracks.keys()):
            tr = self._tracks[tid]
            col = int(math.floor(tr.x))
            row = int(math.floor(tr.y))
            if occupancy_fn(col, row):
                tr.state = REMOVED
                self._removed.append(tid)
                self._tracks.pop(tid)
                withdrawn.append(tid)
        return withdrawn
