"""Projection from camera-frame pixels to map-frame floor points.

Primary path (per plan v3): pixel -> undistort -> ray in camera
optical frame -> rotate into map frame via Team 5's orientation ->
intersect the floor plane z = floor_plane_z -> clamp to arena extent.

Confidence-zone weighting (per plan v3) down-weights two regions where
projection error inflates:
* near the image edges, where wide-FOV radial distortion is strongest;
* near-horizontal rays, where a small orientation error maps onto a
  large ground footprint.

LiDAR refinement is NOT in this path: the approved plan gates it
behind an optional calibrated extrinsic that does not exist yet.

Distortion models supported (native numpy/math, no OpenCV dependency):
* "none"            - ideal pinhole
* "radial_k1k2_p12" - Brown-Conrady, [k1, k2, p1, p2], iterated inversion
* "fisheye"         - equidistant, theta-style inversion
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import schemas


class ProjectionError(ValueError):
    """Raised when a pixel cannot be projected to a floor point."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole model with selectable distortion model and parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    model: str                      # none | radial_k1k2_p12 | fisheye
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ProjectionError(
                "CameraIntrinsics: fx and fy must be > 0 (got "
                f"fx={self.fx}, fy={self.fy}); the module refuses to "
                "project with unset intrinsics."
            )
        if self.model not in ("none", "radial_k1k2_p12", "fisheye"):
            raise ProjectionError(f"Unknown distortion model: {self.model!r}")
        if self.model == "radial_k1k2_p12" and len(self.params) < 4:
            raise ProjectionError(
                f"radial_k1k2_p12 needs [k1, k2, p1, p2], got {self.params}"
            )
        if self.model == "fisheye" and len(self.params) < 4:
            raise ProjectionError(
                f"fisheye needs [k1, k2, k3, k4], got {self.params}"
            )


@dataclass(frozen=True)
class FloorHit:
    """Result of a floor-plane projection for one pixel."""
    hit: bool
    x: float = 0.0
    y: float = 0.0
    dist_m: float = 0.0          # horizontal distance camera->hit point
    elevation_deg: float = 0.0   # ray depression below the horizontal
    edge_zone: bool = False
    low_ray: bool = False        # near-horizontal ray: pose error amplified


def intrinsics_from_cfg(cfg: dict) -> CameraIntrinsics:
    """Build CameraIntrinsics from the config['camera'] subtree."""
    c = cfg["camera"]
    return CameraIntrinsics(
        fx=float(c["fx"]),
        fy=float(c["fy"]),
        cx=float(c["cx"]),
        cy=float(c["cy"]),
        model=str(c.get("dist_model", "radial_k1k2_p12")),
        params=tuple(float(p) for p in c.get("dist_params", [])),
    )


# --------------------------------------------------------------------------
# Distortion
# --------------------------------------------------------------------------

def _undistort_radial(
    xn0: float, yn0: float,
    k1: float, k2: float, p1: float, p2: float,
) -> tuple[float, float]:
    """Invert Brown-Conrady radial+tangential distortion.

    Forward model (what most calibration tools output) is:
        r2        = xn^2 + yn^2
        x_dist    = xn * (1 + k1*r2 + k2*r2^2)
                    + p1*(r2 + 2*xn^2) + 2*p2*xn*yn
        y_dist    = yn * (1 + k1*r2 + k2*r2^2)
                    + p2*(r2 + 2*yn^2) + 2*p1*xn*yn
    We invert it iteratively: the ideal pinhole point (xn, yn) that,
    when distorted, maps back to (xn0, yn0).  A fixed-point iteration
    converges fast for realistic k magnitudes (|k1| < ~1).
    """
    # Start from the distorted point itself — close enough in practice.
    xn = xn0
    yn = yn0
    for _ in range(12):
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        if abs(radial) < 1e-12:
            raise ProjectionError("Distortion model is singular at this point")
        dx_t = p1 * (r2 + 2.0 * xn * xn) + 2.0 * p2 * (xn * yn)
        dy_t = p2 * (r2 + 2.0 * yn * yn) + 2.0 * p1 * (xn * yn)
        xn_next = (xn0 - dx_t) / radial
        yn_next = (yn0 - dy_t) / radial
        if abs(xn_next - xn) < 1e-12 and abs(yn_next - yn) < 1e-12:
            xn, yn = xn_next, yn_next
            break
        xn, yn = xn_next, yn_next
    return (xn, yn)


def _undistort_fisheye(
    xn0: float, yn0: float,
    k1: float, k2: float, k3: float, k4: float,
) -> tuple[float, float]:
    """Invert an OpenCV-style equidistant fisheye model.

    Forward model:
        theta = acos(1 / sqrt(1 + xn^2 + yn^2))     # angle to optical axis
        theta_d = theta * (1 + k1*th^2 + k2*th^4 + k3*th^6 + k4*th^8)
        u = fx * (theta_d / r) * xn + cx, similarly for v
    We invert numerically: given the observed pinhole-relative radius
    r0 = sqrt(xn0^2 + yn0^2), find theta such that
        r0 = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    then reconstruct the unit-sphere direction.
    """
    r0 = math.hypot(xn0, yn0)
    if r0 < 1e-9:
        return (0.0, 0.0)
    # Bisection on theta in [0, pi/2].
    def f(theta: float) -> float:
        t2 = theta * theta
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - r0
    lo, hi = 0.0, math.pi / 2.0
    if f(hi) < 0:
        return (xn0, yn0)      # out of model range, treat as ideal
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)
    scale = theta / r0
    return (xn0 * scale, yn0 * scale)


def normalized_coords(
    intr: CameraIntrinsics, u: float, v: float
) -> tuple[float, float]:
    """Distorted pixel -> undistorted normalized image coords."""
    xn0 = (u - intr.cx) / intr.fx
    yn0 = (v - intr.cy) / intr.fy
    if intr.model == "none":
        return (xn0, yn0)
    if intr.model == "fisheye":
        k1, k2, k3, k4 = intr.params[:4]
        return _undistort_fisheye(xn0, yn0, k1, k2, k3, k4)
    # radial_k1k2_p12
    k1, k2, p1, p2 = intr.params[:4]
    return _undistort_radial(xn0, yn0, k1, k2, p1, p2)


def pixel_to_ray(intr: CameraIntrinsics, u: float, v: float) -> tuple[float, float, float]:
    """Unit ray in the camera optical frame (z forward)."""
    xn, yn = normalized_coords(intr, u, v)
    norm = math.sqrt(xn * xn + yn * yn + 1.0)
    return (xn / norm, yn / norm, 1.0 / norm)


# --------------------------------------------------------------------------
# Quaternion rotation (camera frame -> map frame)
# --------------------------------------------------------------------------

def quat_rotate(
    q: tuple[float, float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate v by unit quaternion q = (qx, qy, qz, qw)."""
    qx, qy, qz, qw = q
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * v[2] - qz * v[1])
    ty = 2.0 * (qz * v[0] - qx * v[2])
    tz = 2.0 * (qx * v[1] - qy * v[0])
    # v' = v + q.w*t + cross(q.xyz, t)
    return (
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    )


# --------------------------------------------------------------------------
# Floor-plane intersection
# --------------------------------------------------------------------------

def ray_to_floor(
    ray_camera: tuple[float, float, float],
    quat_map_from_camera: tuple[float, float, float, float],
    cam_pos_map: tuple[float, float, float],
    floor_z: float,
) -> FloorHit:
    """Intersect a camera-frame ray with the map-frame floor plane."""
    rx, ry, rz = quat_rotate(quat_map_from_camera, ray_camera)
    elevation_deg = math.degrees(math.asin(max(-1.0, min(1.0, -rz))))
    low_ray = elevation_deg < 5.0
    if abs(rz) < 1e-9:
        return FloorHit(hit=False, elevation_deg=elevation_deg, low_ray=low_ray)
    t = (floor_z - cam_pos_map[2]) / rz
    if t <= 0:
        return FloorHit(hit=False, elevation_deg=elevation_deg, low_ray=low_ray)
    x = cam_pos_map[0] + t * rx
    y = cam_pos_map[1] + t * ry
    dist = math.hypot(x - cam_pos_map[0], y - cam_pos_map[1])
    return FloorHit(
        hit=True, x=x, y=y, dist_m=dist,
        elevation_deg=elevation_deg,
        low_ray=low_ray,
    )


def edge_zone_flag(
    w: int,
    h: int,
    u: float,
    v: float,
    edge_zone_frac: float,
) -> bool:
    """True if the pixel lies in the outer edge_zone_frac of the frame."""
    dx = abs(u - w / 2.0)
    dy = abs(v - h / 2.0)
    return dx > edge_zone_frac * (w / 2.0) or dy > edge_zone_frac * (h / 2.0)


def project_pixel_to_floor(
    intr: CameraIntrinsics,
    cam_pos_map: tuple[float, float, float],
    quat_map_from_cam: tuple[float, float, float, float],
    u: float,
    v: float,
    image_w: int,
    image_h: int,
    floor_z: float,
    edge_zone_frac: float,
) -> FloorHit:
    """Full chain: pixel -> FloorHit on the map floor plane."""
    ray = pixel_to_ray(intr, u, v)
    hit = ray_to_floor(ray, quat_map_from_cam, cam_pos_map, floor_z)
    if not hit.hit:
        return hit
    return FloorHit(
        hit=True, x=hit.x, y=hit.y, dist_m=hit.dist_m,
        elevation_deg=hit.elevation_deg,
        edge_zone=edge_zone_flag(image_w, image_h, u, v, edge_zone_frac),
        low_ray=hit.low_ray,
    )


def project_detection(
    intr: CameraIntrinsics,
    cam_pos_map: tuple[float, float, float],
    quat_map_from_cam: tuple[float, float, float, float],
    det: schemas.Detection,
    image_w: int,
    image_h: int,
    floor_z: float,
    edge_zone_frac: float,
) -> FloorHit:
    """Project a detection's bbox centre pixel to a map-frame floor point."""
    px = det.bbox[0] + det.bbox[2] * 0.5
    py = det.bbox[1] + det.bbox[3] * 0.5
    return project_pixel_to_floor(
        intr, cam_pos_map, quat_map_from_cam, px, py,
        image_w, image_h, floor_z, edge_zone_frac,
    )
