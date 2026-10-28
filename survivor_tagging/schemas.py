"""Canonical message schemas for the survivor_tagging module.

All cross-team messages are plain dictionaries in a fixed JSON shape.
This module defines those shapes, validates them strictly, and converts
them into typed dataclasses the core modules consume.

Schema rules
------------
* Unknown extra keys are rejected (typos surface loudly, not silently).
* All required numeric types are checked (int/float, finite).
* Timestamps are seconds since the shared mission clock (float).
* Frames are named explicitly; values are never passed frame-implicitly.

Fact source: "Team 3 — Survivor Localisation and Map Tagging" in
NIDAR_Team_Subproblem_Assignments.pdf and the integration plan in the
same document.  Message naming uses the source-team numbering from the
assignments PDF (Team 1 mapping, Team 2 detection, Team 5 pose).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


class SchemaError(ValueError):
    """Raised when an incoming message does not match its schema."""


# --------------------------------------------------------------------------
# Generic validation helpers
# --------------------------------------------------------------------------

def _require_dict(msg: Any, kind: str) -> dict:
    if not isinstance(msg, dict):
        raise SchemaError(f"{kind}: expected a dict, got {type(msg).__name__}")
    return msg


def _require_keys(d: dict, keys: Iterable[str], kind: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise SchemaError(f"{kind}: missing required key(s) {missing}")


def _reject_unknown(d: dict, allowed: set[str], kind: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise SchemaError(f"{kind}: unknown key(s) {sorted(unknown)}")


def _require_finite_number(value: Any, name: str, kind: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{kind}.{name}: expected a number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise SchemaError(f"{kind}.{name}: value must be finite, got {v}")
    return v


def _require_short_list(value: Any, name: str, kind: str,
                        length: int, element: str = "number") -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SchemaError(f"{kind}.{name}: expected a list of {length} {element}s")
    return [
        _require_finite_number(v, f"{name}[{i}]", kind)
        for i, v in enumerate(value)
    ]


def _require_confidence(value: Any, kind: str, name: str = "conf") -> float:
    v = _require_finite_number(value, name, kind)
    if not 0.0 <= v <= 1.0:
        raise SchemaError(f"{kind}.{name}: confidence must be in [0, 1], got {v}")
    return v


def _require_string_in(value: Any, allowed: set[str], name: str, kind: str) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{kind}.{name}: expected a string")
    if value not in allowed:
        raise SchemaError(f"{kind}.{name}: {value!r} is not one of {sorted(allowed)}")
    return value


def _require_positive_int(value: Any, name: str, kind: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{kind}.{name}: expected an int, got {type(value).__name__}")
    if value <= 0:
        raise SchemaError(f"{kind}.{name}: must be > 0, got {value}")
    return value


# --------------------------------------------------------------------------
# Team 2 -> Team 3 : detection message
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """A single detection in camera-frame pixel coordinates."""
    stamp: float            # seconds on the shared mission clock
    frame_id: str           # camera optical frame, e.g. "cam3noir_optical"
    bbox: tuple[float, float, float, float]   # (px, py, width, height), pixels
    conf: float             # detection confidence in [0, 1]
    label: str              # detection class label, e.g. "person"
    track_hint: str | None = None  # optional detector-internal id


def parse_detection(msg: dict) -> Detection:
    kind = "detections"
    d = _require_dict(msg, kind)
    allowed = {"stamp", "frame_id", "bbox", "conf", "label", "track_hint"}
    _require_keys(d, ("stamp", "frame_id", "bbox", "conf", "label"), kind)
    _reject_unknown(d, allowed, kind)

    stamp = _require_finite_number(d["stamp"], "stamp", kind)
    if stamp < 0:
        raise SchemaError(f"{kind}.stamp: must be >= 0, got {stamp}")
    frame_id = d["frame_id"]
    if not isinstance(frame_id, str) or not frame_id:
        raise SchemaError(f"{kind}.frame_id: expected a non-empty string")

    bbox = _require_short_list(d["bbox"], "bbox", kind, 4)
    if min(bbox) < 0:
        raise SchemaError(f"{kind}.bbox: pixel values must be >= 0, got {bbox}")

    conf = _require_confidence(d["conf"], kind)
    label = d["label"]
    if not isinstance(label, str) or not label:
        raise SchemaError(f"{kind}.label: expected a non-empty string")

    hint = d.get("track_hint")
    if hint is not None and not isinstance(hint, (str, int)):
        raise SchemaError(f"{kind}.track_hint: expected a string or int id")

    return Detection(
        stamp=stamp,
        frame_id=frame_id,
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        conf=conf,
        label=label,
        track_hint=str(hint) if hint is not None else None,
    )


@dataclass(frozen=True)
class DetectionFrame:
    """A batch of detections sharing one camera stamp and frame id."""
    stamp: float
    frame_id: str
    detections: tuple[Detection, ...]


def parse_detection_frame(msg: dict) -> DetectionFrame:
    kind = "detection_frame"
    d = _require_dict(msg, kind)
    allowed = {"stamp", "frame_id", "detections"}
    _require_keys(d, ("stamp", "frame_id", "detections"), kind)
    _reject_unknown(d, allowed, kind)
    stamp = _require_finite_number(d["stamp"], "stamp", kind)
    fid = d["frame_id"]
    if not isinstance(fid, str) or not fid:
        raise SchemaError(f"{kind}.frame_id: expected a non-empty string")

    dets_raw = d["detections"]
    if not isinstance(dets_raw, (list, tuple)):
        raise SchemaError(f"{kind}.detections: expected a list")
    dets = [
        parse_detection({**det, "stamp": stamp, "frame_id": fid})
        for det in dets_raw
    ]
    return DetectionFrame(stamp=stamp, frame_id=fid, detections=tuple(dets))


# --------------------------------------------------------------------------
# Team 5 -> Team 3 : pose message
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PoseSample:
    """One pose sample in the map frame.

    position is [x, y, z] in the map frame ( metres).
    orientation is [qx, qy, qz, qw] (unit quaternion, map<-body).
    sigma_xy is the 1-sigma horizontal position standard deviation in metres.
    """
    stamp: float
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    sigma_xy: float


def parse_pose(msg: dict) -> PoseSample:
    kind = "pose"
    d = _require_dict(msg, kind)
    allowed = {"stamp", "position", "orientation", "sigma_xy"}
    _require_keys(d, ("stamp", "position", "orientation", "sigma_xy"), kind)
    _reject_unknown(d, allowed, kind)

    stamp = _require_finite_number(d["stamp"], "stamp", kind)
    if stamp < 0:
        raise SchemaError(f"{kind}.stamp: must be >= 0, got {stamp}")

    pos = _require_short_list(d["position"], "position", kind, 3)
    quat = _require_short_list(d["orientation"], "orientation", kind, 4)
    sigma = _require_finite_number(d["sigma_xy"], "sigma_xy", kind)
    if sigma < 0:
        raise SchemaError(f"{kind}.sigma_xy: must be >= 0, got {sigma}")

    norm = math.sqrt(sum(q * q for q in quat))
    if abs(norm - 1.0) > 1e-3:
        raise SchemaError(f"{kind}.orientation: quaternion norm {norm} far from unit")

    return PoseSample(
        stamp=stamp,
        position=(pos[0], pos[1], pos[2]),
        orientation=(quat[0], quat[1], quat[2], quat[3]),
        sigma_xy=sigma,
    )


# --------------------------------------------------------------------------
# Team 1 -> Team 3 : occupancy map message
# --------------------------------------------------------------------------

UNKNOWN = -1      # never seen
FREE = 0          # explored, no obstacle seen
OCCUPIED = 100    # wall / obstacle seen

# 1-sigma values below this are "sharp pose"; above, confidence degrades hard.
# Tunable, not a schema constraint.
POSE_SHARP_SIGMA = 0.02


@dataclass(frozen=True)
class OccupancyGridMsg:
    """Row-major occupancy grid in the map frame.

    Cell (col, row) is at index row * w + col.  Cell (0, 0) is the map's
    lower-left corner in ROS convention; its centre is at
    (origin_x, origin_y) in map-frame metres.

    Values are -1 (unknown), 0 (free), or 100 (occupied) but arrive as
    ROS occupancy-grid probabilities (0..100, higher = more likely
    occupied); we threshold them at load into 3 states.
    """
    w: int
    h: int
    origin_x: float
    origin_y: float
    res: float
    data: tuple[int, ...]     # 3-state values (UNKNOWN / FREE / OCCUPIED)


def parse_map(msg: dict) -> OccupancyGridMsg:
    kind = "map"
    d = _require_dict(msg, kind)
    allowed = {"w", "h", "origin_x", "origin_y", "res", "data"}
    _require_keys(d, ("w", "h", "origin_x", "origin_y", "res", "data"), kind)
    _reject_unknown(d, allowed, kind)

    w = _require_positive_int(d["w"], "w", kind)
    h = _require_positive_int(d["h"], "h", kind)
    ox = _require_finite_number(d["origin_x"], "origin_x", kind)
    oy = _require_finite_number(d["origin_y"], "origin_y", kind)
    res = _require_finite_number(d["res"], "res", kind)
    if res <= 0:
        raise SchemaError(f"{kind}.res: must be > 0, got {res}")
    raw = d["data"]
    if not isinstance(raw, (list, tuple)):
        raise SchemaError(f"{kind}.data: expected a list of length w * h")
    if len(raw) != w * h:
        raise SchemaError(
            f"{kind}.data: length {len(raw)} != w*h = {w * h}"
        )
    data = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SchemaError(f"{kind}.data: cell values must be numeric, got {type(v).__name__}")
        fv = int(round(float(v)))
        if fv < -1 or fv > 100:
            raise SchemaError(f"{kind}.data: cell value {v} outside [-1, 100]")
        data.append(_classify_cell(fv))
    return OccupancyGridMsg(w=w, h=h, origin_x=ox, origin_y=oy, res=res, data=tuple(data))


def _classify_cell(prob: int) -> int:
    """Convert a ROS occupancy-grid probability to 3-state."""
    if prob <= UNKNOWN:
        return UNKNOWN
    if prob >= 65:      # ROS map_server convention: > 65 = occupied
        return OCCUPIED
    return FREE


# --------------------------------------------------------------------------
# Team 3 -> GCS : tag payload (canonical, v1)
# --------------------------------------------------------------------------

TAG_STATE_TENTATIVE = "TENTATIVE"
TAG_STATE_CONFIRMED = "CONFIRMED"
TAG_STATE_REMOVED = "REMOVED"


@dataclass(frozen=True)
class TagUpdate:
    """The published tag payload sent to the GCS.

    Field names are frozen for v1: tag_id, cell, map_position, confidence,
    flags, state, stamp.  Any change bumps the wire-version constant.
    """
    tag_id: str
    cell: tuple[int, int]          # (col, row)
    map_position: tuple[float, float]   # map frame metres
    confidence: float
    flags: dict[str, Any] = field(default_factory=dict)
    state: str = TAG_STATE_CONFIRMED
    stamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tag_id": self.tag_id,
            "cell": {"col": self.cell[0], "row": self.cell[1]},
            "map_position": [self.map_position[0], self.map_position[1]],
            "confidence": self.confidence,
            "flags": dict(self.flags),
            "state": self.state,
            "stamp": self.stamp,
        }


def make_tag_update(
    tag_id: str,
    cell: tuple[int, int],
    map_position: tuple[float, float],
    confidence: float,
    *,
    flags: dict[str, Any] | None = None,
    state: str = TAG_STATE_CONFIRMED,
    stamp: float = 0.0,
) -> TagUpdate:
    """Build a validated TagUpdate; confidence is clamped to [0, 1]."""
    if not isinstance(tag_id, str) or not tag_id:
        raise SchemaError("tag_id: expected a non-empty string")
    for i in (0, 1):
        if isinstance(cell[i], bool) or not isinstance(cell[i], int):
            raise SchemaError(f"cell[{i}]: expected an int")
    mp = tuple(float(v) for v in map_position)
    conf = max(0.0, min(1.0, float(confidence)))
    state = _require_string_in(
        state, {TAG_STATE_TENTATIVE, TAG_STATE_CONFIRMED, TAG_STATE_REMOVED},
        "state", "tag_update",
    )
    return TagUpdate(
        tag_id=tag_id,
        cell=cell,
        map_position=(mp[0], mp[1]),
        confidence=conf,
        flags=dict(flags or {}),
        state=state,
        stamp=float(stamp),
    )
