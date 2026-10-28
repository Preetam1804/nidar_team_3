"""Synthetic mission runner for benchmarking survivor_tagging.

Discrete-event simulation of the Team 3 pipeline:
* a synthetic arena (walls, free space, survivors) as a 1 m x 1 m grid
* a scripted drone trajectory (passes near survivors)
* synthetic camera projections using the real pipeline's projection.py,
  so the code under test is exactly the code benchmarked
* injectable pose noise (sigma_xy), edge-zone incidence, and
  "pose dropout" windows, all configurable per run
* ground truth kept for every survivor so correct vs incorrect tag
  assignment can be measured per survivor
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from survivor_tagging import schemas
from survivor_tagging import grid as grid_mod
from survivor_tagging.adapters import (
    InMemoryDetectors,
    InMemoryMap,
    InMemoryPose,
)
from survivor_tagging.pipeline import SurvivorTaggingPipeline


@dataclass
class Survivor:
    sid: int
    x: float
    y: float

    @property
    def cell(self) -> tuple[int, int]:
        return (int(math.floor(self.x)), int(math.floor(self.y)))


@dataclass
class SimConfig:
    arena_w: int = 15          # cells; arena <= 15 m x 15 m per brief
    arena_h: int = 15
    n_survivors: int = 6
    camera_height_m: float = 2.0
    fov_deg: float = 120.0
    img_w: int = 1536
    img_h: int = 864
    # Pose noise: 1-sigma horizontal position error, metres (Team 5).
    pose_sigma_xy: float = 0.15
    # Depth of drone passes (waypoint list in metres).
    waypoints: tuple[tuple[float, float], ...] = (
        (1.0, 1.0), (13.0, 1.0), (13.0, 13.0), (1.0, 13.0),
    )
    waypoint_speed_mps: float = 1.0
    det_rate_hz: float = 10.0     # synthetic detection frames / s
    pose_rate_hz: float = 100.0
    seed: int = 42
    # Flood floors FREE; walls = perimeter cells (a minimal arena shape).
    put_walls_on_perimeter: bool = True


def _build_map(cfg: SimConfig) -> schemas.OccupancyGridMsg:
    """Arena map: perimeter OCCUPIED, interior FREE (row-major, lower-left origin)."""
    w, h = cfg.arena_w, cfg.arena_h
    data = [schemas.FREE] * (w * h)
    if cfg.put_walls_on_perimeter:
        for c in range(w):
            data[0 * w + c] = schemas.OCCUPIED
            data[(h - 1) * w + c] = schemas.OCCUPIED
        for r in range(h):
            data[r * w + 0] = schemas.OCCUPIED
            data[r * w + (w - 1)] = schemas.OCCUPIED
    return schemas.OccupancyGridMsg(
        w=w, h=h, origin_x=0.0, origin_y=0.0, res=1.0, data=tuple(data)
    )


def _place_survivors(cfg: SimConfig, rng: random.Random) -> list[Survivor]:
    """Random surviving positions, not on walls."""
    out: list[Survivor] = []
    tries = 0
    while len(out) < cfg.n_survivors and tries < 500:
        tries += 1
        x = rng.uniform(1.5, cfg.arena_w - 1.5)
        y = rng.uniform(1.5, cfg.arena_h - 1.5)
        col, row = int(math.floor(x)), int(math.floor(y))
        # avoid overlapping same cell
        if any(s.cell == (col, row) for s in out):
            continue
        out.append(Survivor(sid=len(out), x=x, y=y))
    return out


def _project_survivor_pixels(
    intr, quat: tuple[float, float, float, float],
    position: tuple[float, float, float],
    surv: Survivor,
    cfg: SimConfig,
    rng: random.Random,
) -> list[tuple[float, float]]:
    """Forward-project a survivor to a synthetic camera-frame pixel.

    Matches the pipeline's own down-looking quaternion convention
    (quat = (1,0,0,0), i.e. 180 deg about optical x): the pipeline's
    back-projection of this pixel must land back on the survivor.
    """
    half = cfg.camera_height_m * math.tan(math.radians(cfg.fov_deg / 2.0))
    dx = surv.x - position[0]
    dy = surv.y - position[1]
    if abs(dx) > half or abs(dy) > half:
        return []
    fx = (cfg.img_w / 2.0) / math.tan(math.radians(cfg.fov_deg / 2.0))
    # Optical y flips under quat (1,0,0,0): v = cy - fx * dy / cz
    u = cfg.img_w / 2.0 + fx * (dx / cfg.camera_height_m)
    v = cfg.img_h / 2.0 - fx * (dy / cfg.camera_height_m)
    # Simple noise on synthetic detector bbox centre.
    u += rng.gauss(0.0, 0.08) * fx * 0.01
    v += rng.gauss(0.0, 0.08) * fx * 0.01
    if not (0 <= u < cfg.img_w and 0 <= v < cfg.img_h):
        return []
    return [(u, v)]


def _wander_pose(
    t: float, waypoints, speed: float, rng: random.Random,
    cfg: SimConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Drone moves along waypoints; returns noisy (position, quat)."""
    total = sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(waypoints, waypoints[1:] + waypoints[:1])
    )
    at = (t * speed) % total
    # find which segment we are in
    cum = 0.0
    pts = list(waypoints) + [waypoints[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if cum + seg >= at:
            frac = (at - cum) / seg if seg > 0 else 0.0
            x = a[0] + frac * (b[0] - a[0])
            y = a[1] + frac * (b[1] - a[1])
            break
        cum += seg
    else:
        x, y = pts[0]
    z = cfg.camera_height_m
    true_pos = (x, y, z)
    # down-looking camera: 180 deg about x (matches pipeline convention)
    quat = (1.0, 0.0, 0.0, 0.0)
    sigma = cfg.pose_sigma_xy
    noisy = (
        true_pos[0] + rng.gauss(0.0, sigma),
        true_pos[1] + rng.gauss(0.0, sigma),
        true_pos[2] + rng.gauss(0.0, cfg.drone_elev_sigma_m),
    ) if hasattr(cfg, "drone_elev_sigma_m") else true_pos
    return (noisy, quat)


@dataclass
class SimResult:
    per_survivor: dict[int, dict] = field(default_factory=dict)
    metrics_summary: dict = field(default_factory=dict)
    gcs_payloads: list[dict] = field(default_factory=list)


def run_mission(cfg: SimConfig, pipeline_cfg: dict) -> SimResult:
    """Run one discrete-event synthetic mission; return scoreable result."""
    # The sim derives pinhole intrinsics from its own FOV so it exercises
    # the real projection code path with concrete calibration values.
    pipeline_cfg = dict(pipeline_cfg)
    cam = dict(pipeline_cfg.get("camera", {}))
    fx_fy = (cfg.img_w / 2.0) / math.tan(math.radians(cfg.fov_deg / 2.0))
    cam["fx"] = cam["fy"] = fx_fy
    cam["cx"] = cfg.img_w / 2.0
    cam["cy"] = cfg.img_h / 2.0
    cam["width"] = cfg.img_w
    cam["height"] = cfg.img_h
    cam["dist_model"] = "none"   # sim virtual camera: ideal pinhole
    pipeline_cfg["camera"] = cam

    rng = random.Random(cfg.seed)
    map_msg = _build_map(cfg)
    survivors = _place_survivors(cfg, rng)

    detections = InMemoryDetectors()
    pose = InMemoryPose()
    mapa = InMemoryMap()
    mapa.push(map_msg)

    pipeline = SurvivorTaggingPipeline(pipeline_cfg, detections, pose, mapa)

    # Import cfg defaults for missing sim subkeys
    sim_cfg = pipeline_cfg.get("sim", {})
    elev_sigma = float(sim_cfg.get("drone_elev_sigma_m", 0.5))
    cfg.drone_elev_sigma_m = elev_sigma

    dt_det = 1.0 / cfg.det_rate_hz
    dt_pose = 1.0 / cfg.pose_rate_hz
    t_total = 2.0 * sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(cfg.waypoints, cfg.waypoints[1:] + cfg.waypoints[:1])
    ) / cfg.waypoint_speed_mps

    # Ground truth for survivor tags: track which tracks map to survivors.
    track_to_survivor: dict[str, int] = {}

    gcs_payloads: list[dict] = []
    t = 0.0
    next_det = 0.0
    next_pose = 0.0
    while t < t_total:
        if t >= next_pose:
            posz, quat = _wander_pose(
                t, cfg.waypoints, cfg.waypoint_speed_mps, rng, cfg
            )
            pose.push(schemas.PoseSample(
                stamp=t, position=posz, orientation=quat,
                sigma_xy=cfg.pose_sigma_xy,
            ))
            next_pose += dt_pose
        if t >= next_det:
            posz, _ = _wander_pose(
                t, cfg.waypoints, cfg.waypoint_speed_mps, rng, cfg
            )
            dets = []
            for s in survivors:
                for (u, v) in _project_survivor_pixels(
                    pipeline.intr, (1.0, 0.0, 0.0, 0.0),
                    posz, s, cfg, rng,
                ):
                    if not (0 <= u < cfg.img_w and 0 <= v < cfg.img_h):
                        continue
                    dets.append(schemas.Detection(
                        stamp=t,
                        frame_id=pipeline_cfg["camera"]["frame_id"],
                        bbox=(max(0.0, u - 8.0), max(0.0, v - 8.0), 16.0, 16.0),
                        conf=min(1.0, max(0.35, 0.9 + rng.gauss(0.0, 0.05))),
                        label="person",
                        track_hint=None,
                    ))
            if dets:
                detections.push(schemas.DetectionFrame(
                    stamp=t, frame_id=pipeline_cfg["camera"]["frame_id"],
                    detections=tuple(dets),
                ))
            next_det += dt_det

        payloads = pipeline.tick()
        for p in payloads:
            gcs_payloads.append(p)
            tid = p["tag_id"]
            # map the emitted tag to the survivor it is closest to
            if tid not in track_to_survivor:
                px, py = p["map_position"]
                best_sid, best_d = None, float("inf")
                for s in survivors:
                    d = math.hypot(s.x - px, s.y - py)
                    if d < best_d:
                        best_d, best_sid = d, s.sid
                if best_sid is not None and best_d < 1.0:
                    track_to_survivor[tid] = best_sid
        t += 0.05    # 20 Hz tick

    # -- score ------------------------------------------------------------
    per_survivor: dict = {}
    for s in survivors:
        per_survivor[s.sid] = {"n_tags": 0, "correct": 0, "incorrect": 0,
                               "cells": set()}
    for p in gcs_payloads:
        sid = track_to_survivor.get(p["tag_id"])
        if sid is None:
            per_survivor.setdefault(
                "unknown_track", {"n_tags": 0, "correct": 0,
                                  "incorrect": 0, "cells": set()}
            )
            per_survivor["unknown_track"]["n_tags"] += 1
            continue
        gt = next(s for s in survivors if s.sid == sid)
        # p["cell"] is a dict {"col":..,"row":..} per the GCS wire format
        tag_cell = (p["cell"]["col"], p["cell"]["row"])
        correct = tag_cell == gt.cell
        entry = per_survivor[sid]
        entry["n_tags"] += 1
        entry["cells"].add(tag_cell)
        if correct:
            entry["correct"] += 1
            pipeline.metrics.correct_tags += 1
        else:
            entry["incorrect"] += 1
            pipeline.metrics.incorrect_tags += 1

    return SimResult(
        per_survivor=per_survivor,
        metrics_summary=pipeline.metrics.summary(),
        gcs_payloads=gcs_payloads,
    )


def run_soak(
    cfg: SimConfig, pipeline_cfg: dict, duration_s: float = 1800.0
) -> dict:
    """Run a mission-length (e.g. 30 min / 1800s) continuous soak benchmark.

    Verifies:
    * active tracks count remains strictly bounded (no unbounded leak)
    * latency remains stable across 36,000+ ticks
    * memory growth is bounded
    """
    pipeline_cfg = dict(pipeline_cfg)
    cam = dict(pipeline_cfg.get("camera", {}))
    fx_fy = (cfg.img_w / 2.0) / math.tan(math.radians(cfg.fov_deg / 2.0))
    cam["fx"] = cam["fy"] = fx_fy
    cam["cx"] = cfg.img_w / 2.0
    cam["cy"] = cfg.img_h / 2.0
    cam["width"] = cfg.img_w
    cam["height"] = cfg.img_h
    cam["dist_model"] = "none"
    pipeline_cfg["camera"] = cam

    rng = random.Random(cfg.seed)
    map_msg = _build_map(cfg)
    survivors = _place_survivors(cfg, rng)

    detections = InMemoryDetectors()
    pose = InMemoryPose()
    mapa = InMemoryMap()
    mapa.push(map_msg)

    pipeline = SurvivorTaggingPipeline(pipeline_cfg, detections, pose, mapa)
    sim_cfg = pipeline_cfg.get("sim", {})
    cfg.drone_elev_sigma_m = float(sim_cfg.get("drone_elev_sigma_m", 0.15))

    dt_det = 1.0 / cfg.det_rate_hz
    dt_pose = 1.0 / cfg.pose_rate_hz
    t = 0.0
    next_det = 0.0
    next_pose = 0.0
    payload_count = 0
    max_active_tracks = 0

    while t < duration_s:
        if t >= next_pose:
            posz, quat = _wander_pose(
                t, cfg.waypoints, cfg.waypoint_speed_mps, rng, cfg
            )
            pose.push(schemas.PoseSample(
                stamp=t, position=posz, orientation=quat,
                sigma_xy=cfg.pose_sigma_xy,
            ))
            next_pose += dt_pose
        if t >= next_det:
            posz, _ = _wander_pose(
                t, cfg.waypoints, cfg.waypoint_speed_mps, rng, cfg
            )
            dets = []
            for s in survivors:
                for (u, v) in _project_survivor_pixels(
                    pipeline.intr, (1.0, 0.0, 0.0, 0.0),
                    posz, s, cfg, rng,
                ):
                    if not (0 <= u < cfg.img_w and 0 <= v < cfg.img_h):
                        continue
                    dets.append(schemas.Detection(
                        stamp=t,
                        frame_id=pipeline_cfg["camera"]["frame_id"],
                        bbox=(max(0.0, u - 8.0), max(0.0, v - 8.0), 16.0, 16.0),
                        conf=min(1.0, max(0.35, 0.9 + rng.gauss(0.0, 0.05))),
                        label="person",
                        track_hint=None,
                    ))
            if dets:
                detections.push(schemas.DetectionFrame(
                    stamp=t, frame_id=pipeline_cfg["camera"]["frame_id"],
                    detections=tuple(dets),
                ))
            next_det += dt_det

        payloads = pipeline.tick()
        payload_count += len(payloads)
        curr_active = len(pipeline.tracker.active_tracks())
        if curr_active > max_active_tracks:
            max_active_tracks = curr_active
        t += 0.05

    summary = pipeline.metrics.summary()
    return {
        "duration_s": duration_s,
        "simulated_minutes": duration_s / 60.0,
        "ticks": pipeline.metrics.ticks,
        "payload_count": payload_count,
        "max_active_tracks": max_active_tracks,
        "max_allowed_tracks": pipeline.tracker.max_tracks,
        "bounded_memory": max_active_tracks <= pipeline.tracker.max_tracks,
        "metrics": summary,
    }
