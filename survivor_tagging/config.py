"""Configuration loading for the survivor_tagging module.

Every tunable lives in a YAML file or an environment-variable override.
No device ids, ports, paths, frame names, or thresholds are hardcoded
into module logic.

Environment overrides follow the pattern

    SURVIVOR_TAGGING_<PATH> = value

where PATH is a dotted key path inside the YAML with dots replaced by
underscores, upper-cased.  Example::

    SURVIVOR_TAGGING_projection_floor_plane_z=0.0

Values are parsed as YAML scalars first (so "0.5" becomes float, "true"
becomes bool).  If the key does not already exist in the YAML the
override is still accepted (it lands in a `_env_extras` dict so handlers
can opt in to it explicitly) — that keeps adapter-specific config from
needing a code change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "SURVIVOR_TAGGING_"

_DEFAULTS: dict[str, Any] = {
    # ---- Coordinate grid (Team 1 map) --------------------------------
    "grid": {
        "res_m": 1.0,                 # score grid cell size; ROS map may differ
        "w": 15,
        "h": 15,
        "unknown_val": -1,
        "free_val": 0,
        "occupied_val": 100,
    },
    # ---- Camera ------------------------------------------------------
    "camera": {
        "frame_id": "cam3noir_optical",
        "width": 1536,
        "height": 864,
        # Default intrinsic matrix for the Pi Camera 3 NoIR Wide is applied
        # externally on calibration; zeros mean "must be set from config file".
        "fx": 0.0,
        "fy": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        "dist_model": "radial_k1k2_p12",   # options: radial_k1k2_p12 | fisheye | none
        "dist_params": [],
        # Maximum horizontal distance from image centre that we treat as a
        # "wide-FOV edge zone" (weight down confidence near edges).
        "edge_zone_frac": 0.85,
    },
    # ---- Projection / floor plane ------------------------------------
    "projection": {
        "floor_plane_z": 0.0,               # map-frame ground height, metres
        "min_ray_elevation_deg": 0.1,       # ray nearly parallel to ground => no intersection
        "min_floor_angle_deg": 2.0,         # reject rays that hit too far under the drone
        "max_floor_dist_m": 15.0,           # arena cap per mission brief
        "min_floor_dist_m": 0.2,
    },
    # ---- Tracker --------------------------------------------------------
    "tracker": {
        "gate_dist_m": 1.5,          # max map-frame association distance
        "hit_radius_m": 0.5,         # max distance for a candidate to count as a hit
        "confirm_hits": 2,           # hits required before a tag becomes CONFIRMED
        "confirm_window": 3,         # within N recent detections
        "max_missed_ticks": 8,       # ticks without hits before a track is removed
        "max_map_units": 64,         # active track cap
    },
    # ---- Tag lifecycle --------------------------------------------------
    "tags": {
        "min_confirm_conf": 0.5,     # below this a tag stays TENTATIVE
        "min_publish_conf": 0.5,     # below this a tag is not published at all
        "max_pose_age_s": 0.5,       # older than this and pose quality drops hard
        "min_active_s": 0.2,         # never publish a tag younger than this
    },
    # ---- Logging / metrics ----------------------------------------------
    "metrics": {
        "latency_hist_bins_s": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        "err_hist_bins_m": [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        "cpu_ram_sample_every_n": 100,
    },
    # ---- Simulator fidelity --------------------------------------------
    "sim": {
        "arena_w": 15.0,
        "arena_h": 15.0,
        "fov_deg": 120.0,
        "survivor_height_m": 1.7,
        "camera_height_m": 2.0,
        "drone_speed_mps": 1.0,
        "drone_elev_sigma_m": 0.15,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _split_env_key(key: str) -> list[str]:
    """SURVIVOR_TAGGING_PROJECTION_FLOOR_PLANE_Z -> ['projection', 'floor_plane_z']"""
    body = key[len(ENV_PREFIX):].lower()
    parts = [p.strip() for p in body.split("_") if p.strip()]
    return parts


def _env_overrides() -> tuple[dict, dict[str, Any]]:
    """Collect SURVIVOR_TAGGING_* env overrides.

    Returns (tree, extras) where tree matches existing YAML keys and extras
    are env keys that do not resolve to existing YAML paths (kept so
    adapters can use them without code changes).
    """
    tree: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for env_key, env_val in sorted(os.environ.items()):
        if not env_key.startswith(ENV_PREFIX):
            continue
        parts = _split_env_key(env_key)
        if not parts:
            continue
        parsed = _parse_scalar(env_val)
        # greedy longest-prefix match against known top-level group names
        node = tree
        placed = False
        for i in range(len(parts), 0, -1):
            candidate = "_".join(parts[:i]).lower()
            if candidate in _DEFAULTS:
                remaining = "_".join(parts[i:]).lower()
                if not remaining:
                    # override a whole group — replace it entirely
                    node[candidate] = parsed
                    placed = True
                    break
                # the remainder rejoins into the dotted leaf name,
                # e.g. SEARCH_TAGS_MIN_PUBLISH_CONF -> min_publish_conf
                node.setdefault(candidate, {})[remaining] = parsed
                placed = True
                break
        if not placed:
            extras[".".join(parts)] = parsed
    return tree, extras


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load YAML config, merge defaults, then apply env overrides."""
    cfg = dict(_DEFAULTS)
    if path is not None:
        p = Path(path)
        if p.exists():
            user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if not isinstance(user, dict):
                raise ValueError(f"Config file {p} must contain a YAML mapping")
            cfg = _deep_merge(cfg, user)
    env_tree, env_extras = _env_overrides()
    if env_tree:
        cfg = _deep_merge(cfg, env_tree)
    if env_extras:
        cfg.setdefault("_env_extras", {}).update(env_extras)
    return cfg


def default_config_path() -> Path:
    """Path of config.yaml shipped next to the package root."""
    return Path(__file__).resolve().parent.parent / "config.yaml"
