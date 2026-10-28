"""Tag confidence fusion and GCS payload construction.

Confidence fusion (per plan v3): each tag's published confidence is a
product of independent quality factors

    detection conf x pose sharpness x edge-zone factor x
    low-ray factor x pose-age factor x snap factor

All factors are in [0, 1]; 1.0 means "no penalty".  Factors are
deliberately simple and monotone so the benchmark report can sweep each
one independently.

IR-only vs daylight regimes: the fusion math itself is regime-agnostic;
regime-specific confidence floors are the detector's (Team 2's)
calibration concern and are handled by the provider layer.
"""

from __future__ import annotations

import math

from . import schemas
from .schemas import make_tag_update


def pose_sharpness(sigma_xy: float, sharp_sigma: float) -> float:
    """1.0 at sigma <= sharp_sigma, decaying to a 0.3 floor at 10x.

    A 0.3 floor reflects plan section C: degraded pose reduces tag
    confidence but never silently zeroes a valid detection.
    """
    if sigma_xy <= sharp_sigma:
        return 1.0
    rel = (sigma_xy - sharp_sigma) / (9.0 * sharp_sigma)   # 0..1 over 9x
    if rel >= 1.0:
        return 0.3
    return 1.0 - 0.7 * rel


def pose_age_factor(age_s: float, max_age_s: float) -> float:
    """1.0 at age 0, linear to 0.5 at max_age_s, floor 0.5 above.

    Deliberately does NOT drop to zero: a stale, flagged, low-confidence
    tag beats silently dropping it (per plan section C).
    """
    if age_s <= 0.0:
        return 1.0
    if age_s >= max_age_s:
        return 0.5
    return 1.0 - 0.5 * (age_s / max_age_s)


def edge_zone_factor(edge_zone: bool, penalty: float = 0.75) -> float:
    return penalty if edge_zone else 1.0


def low_ray_factor(low_ray: bool, penalty: float = 0.7) -> float:
    return penalty if low_ray else 1.0


def snap_factor(snapped: bool, penalty: float = 0.9) -> float:
    return penalty if snapped else 1.0


def fused_confidence(
    det_conf: float,
    sigma_xy: float,
    pose_age_s: float,
    edge_zone: bool,
    low_ray: bool,
    snapped: bool,
    *,
    sharp_sigma: float = 0.05,
    max_pose_age_s: float = 0.5,
    edge_penalty: float = 0.75,
    low_ray_penalty: float = 0.7,
    snap_penalty: float = 0.9,
) -> float:
    """Fuse detection confidence with quality factors into [0, 1]."""
    p = pose_sharpness(sigma_xy, sharp_sigma)
    a = pose_age_factor(pose_age_s, max_pose_age_s)
    e = edge_zone_factor(edge_zone, edge_penalty)
    l = low_ray_factor(low_ray, low_ray_penalty)
    s = snap_factor(snapped, snap_penalty)
    fused = det_conf * p * a * e * l * s
    return max(0.0, min(1.0, fused))


def build_tag_payload(
    tag_id: str,
    cell: tuple[int, int],
    map_position: tuple[float, float],
    fused_conf: float,
    flags: dict | None = None,
    state: str = schemas.TAG_STATE_CONFIRMED,
    stamp: float = 0.0,
) -> dict:
    """Build the canonical GCS JSON payload for one tag."""
    updates = make_tag_update(
        tag_id=tag_id,
        cell=cell,
        map_position=map_position,
        confidence=fused_conf,
        flags=flags or {},
        state=state,
        stamp=stamp,
    )
    return updates.to_dict()
