"""Pipeline orchestration for the survivor_tagging module.

The pipeline is provider-driven: each tick() pulls whatever is fresh
from the injected providers, projects detections into the map frame,
updates the tracker, evaluates tag lifecycle, and emits GCS payloads.

No transport is imported here — Teams 1/2/5 arrive through the
providers, so their internals can change freely without touching this
file (per plan section E).
"""

from __future__ import annotations

import logging
import math
import time

from . import grid as grid_mod
from . import metrics as metrics_mod
from . import projection as proj_mod
from . import schemas
from . import tags as tags_mod
from . import tracker as tracker_mod
from .providers import DetectionsProvider, MapProvider, PoseProvider

log = logging.getLogger(__name__)


class SurvivorTaggingPipeline:
    """Team 3's survivor localisation + map tagging pipeline."""

    def __init__(
        self,
        cfg: dict,
        detections_provider: DetectionsProvider,
        pose_provider: PoseProvider,
        map_provider: MapProvider,
    ) -> None:
        self.cfg = cfg
        self.detections = detections_provider
        self.pose = pose_provider
        self.map_provider = map_provider

        self.intr = proj_mod.intrinsics_from_cfg(cfg)
        cam = cfg["camera"]
        self.img_w: int = int(cam["width"])
        self.img_h: int = int(cam["height"])
        self.floor_z: float = float(cfg["projection"]["floor_plane_z"])
        self.edge_zone_frac: float = float(cam["edge_zone_frac"])
        self.max_floor_dist_m: float = float(cfg["projection"]["max_floor_dist_m"])
        self.min_floor_dist_m: float = float(cfg["projection"]["min_floor_dist_m"])

        self.metrics = metrics_mod.Metrics()
        self.tracker = tracker_mod.Tracker(cfg, on_event=self.metrics.record_tag_event)

        tgc = cfg["tags"]
        self.min_publish_conf: float = float(tgc["min_publish_conf"])
        self.max_pose_age_s: float = float(tgc["max_pose_age_s"])
        self.cpu_ram_sample_every_n: int = int(
            cfg.get("metrics", {}).get("cpu_ram_sample_every_n", 100)
        )

        self.snap_radius: int = 2
        self._last_map: schemas.OccupancyGridMsg | None = None

    # ------------------------------------------------------------------
    def tick(self) -> list[dict]:
        """One pipeline step; returns GCS payload dicts emitted this tick."""
        t0 = time.monotonic()
        emitted: list[dict] = []

        map_msg = self.map_provider.latest()
        if map_msg is not None:
            self._last_map = map_msg

        frame = self.detections.latest_frame()
        if frame is None:
            self.metrics.ticks += 1
            self.metrics.record_latency("tick_ms", time.monotonic() - t0)
            return emitted

        pose = self.pose.latest()
        if pose is not None:
            self.metrics.pose_ingested += 1
        if pose is None:
            log.warning(
                "pose missing — skipping frame at stamp %.3f", frame.stamp
            )
            self.metrics.ticks += 1
            self.metrics.record_latency("tick_ms", time.monotonic() - t0)
            return emitted

        map_msg = self._last_map
        projected: list[tuple] = []
        t_proj = time.monotonic()
        self.metrics.detections_ingested += len(frame.detections)
        for det in frame.detections:
            try:
                hit = proj_mod.project_detection(
                    self.intr, pose.position, pose.orientation, det,
                    self.img_w, self.img_h, self.floor_z, self.edge_zone_frac,
                )
            except Exception as exc:
                # Malformed or infeasible inputs are logged and skipped,
                # never crash the run (per plan section E).
                log.warning("projection failed for det %s: %s", det, exc)
                continue
            if not hit.hit:
                continue
            if not (self.min_floor_dist_m <= hit.dist_m <= self.max_floor_dist_m):
                log.debug(
                    "det at %.3f m outside floor-dist bounds, skipping",
                    hit.dist_m,
                )
                continue

            snapped = False
            if map_msg is not None:
                spec = grid_mod.grid_from_msg(map_msg)
                col, row = spec.world_to_cell(hit.x, hit.y)
                if col >= 0:
                    ccol, crow, sn = grid_mod.snap_to_free(
                        map_msg, col, row, self.snap_radius
                    )
                    snapped = sn
                else:
                    # Projected outside the map — skip this detection.
                    continue

            projected.append(
                (hit.x, hit.y, det.conf, hit.edge_zone, hit.low_ray, snapped)
            )
        self.metrics.record_latency("project_ms", time.monotonic() - t_proj)

        # -- association ---------------------------------------------------
        t1 = time.monotonic()
        self.tracker.ingest(projected, now=pose.stamp)
        self.metrics.record_latency("associate_ms", time.monotonic() - t1)
        self.detections.ack(frame_id=frame.frame_id)

        # -- occupancy-based integrity check -------------------------------
        withdrawn: list[str] = []
        if map_msg is not None:
            def _occ(col: int, row: int) -> bool:
                state = grid_mod.cell_state(map_msg, col, row)
                return state == schemas.OCCUPIED

            withdrawn = self.tracker.withdraw_if_occupied(_occ)
        for tid in withdrawn:
            self.metrics.record_tag_event("withdrawn")
            log.info("tag %s withdrawn: cell mapped occupied", tid)

        # -- build publish payloads ----------------------------------------
        t2 = time.monotonic()
        pose_age_s = max(0.0, frame.stamp - pose.stamp)
        for tr in self.tracker.confirmed_tracks():
            sigma = pose.sigma_xy
            conf = tags_mod.fused_confidence(
                tr.conf, sigma, pose_age_s, tr.edge_zone, tr.low_ray, tr.snapped,
                max_pose_age_s=self.max_pose_age_s,
            )
            if conf < self.min_publish_conf:
                continue
            if pose_age_s > self.max_pose_age_s and tr.conf < 0.5:
                continue
            cell = self._cell_for(tr.x, tr.y, map_msg)
            if cell is None:
                continue
            flags = {
                "snapped": tr.snapped,
                "depth_used": tr.depth_used,
                "low_ray": tr.low_ray,
                "edge_zone": tr.edge_zone,
                "pose_age_s": round(pose_age_s, 3),
                "pose_sigma_xy_m": round(sigma, 4),
            }
            payload = tags_mod.build_tag_payload(
                tr.track_id, cell, (tr.x, tr.y), conf,
                flags=flags, state=schemas.TAG_STATE_CONFIRMED,
                stamp=tr.last_seen_stamp,
            )
            emitted.append(payload)
            self.metrics.record_tag_event("published")
        self.metrics.record_latency("publish_ms", time.monotonic() - t2)

        self.metrics.ticks += 1
        self.metrics.record_latency("tick_ms", time.monotonic() - t0)
        if self.metrics.ticks % self.cpu_ram_sample_every_n == 0:
            self.metrics.sample_resources()
        return emitted

    # ------------------------------------------------------------------
    def _cell_for(
        self, x: float, y: float, map_msg
    ) -> tuple[int, int] | None:
        if map_msg is None:
            # No Team 1 map yet: fall back to the configured grid geometry
            # (origin at (0,0), res_m from config) so the primary
            # floor-plane path still tags cells independently of mapping.
            g = self.cfg["grid"]
            col = int(math.floor(x / float(g["res_m"])))
            row = int(math.floor(y / float(g["res_m"])))
            if 0 <= col < int(g["w"]) and 0 <= row < int(g["h"]):
                return (col, row)
            return None
        spec = grid_mod.grid_from_msg(map_msg)
        col, row = spec.world_to_cell(x, y)
        if col < 0:
            return None
        return (col, row)
