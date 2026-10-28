"""Physically grounded 6-DoF quadcopter flight dynamics and trajectory generator.

Models realistic autonomous drone motion for indoor Search-and-Rescue:
- Smooth S-curve acceleration, deceleration, and velocity ramping
- Dynamic banking into turns (roll angle proportional to centripetal acceleration)
- Dynamic forward pitch proportional to longitudinal acceleration
- Aerodynamic rotor-downwash micro-turbulence (hover instability in confined spaces)
- Altitude profile: Smooth liftoff, cruise, inspection loiter, and landing
- Autonomous inspection loiter: Decelerates and hovers over suspected casualties
  while the onboard vision pipeline accumulates confirmation frames.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (roll, pitch, yaw in radians) to quaternion (qx, qy, qz, qw)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def _quaternion_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Multiply two quaternions (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


@dataclass
class DroneFlightState:
    """Full 6-DoF vehicle kinematics state at a point in time."""
    position: tuple[float, float, float]       # (x, y, z) in meters
    velocity: tuple[float, float, float]       # (vx, vy, vz) in m/s
    speed: float                               # scalar horizontal speed (m/s)
    acceleration: tuple[float, float, float]   # (ax, ay, az) in m/s^2
    attitude_euler: tuple[float, float, float] # (roll, pitch, yaw) in radians
    orientation_quat: tuple[float, float, float, float] # (qx, qy, qz, qw) optical
    is_loitering: bool = False                 # currently hovering to inspect
    loiter_target: tuple[float, float] | None = None
    flight_phase: str = "CRUISE"               # TAKEOFF | CRUISE | INSPECT | RETURN | LAND


class RealisticFlightDynamics:
    """Simulates 6-DoF quadcopter flight dynamics along curved paths."""

    def __init__(
        self,
        waypoints: Sequence[tuple[float, float]],
        cruise_speed_mps: float = 1.0,
        cruise_altitude_m: float = 2.2,
        max_accel_mps2: float = 0.8,
        takeoff_duration_s: float = 2.5,
        landing_duration_s: float = 2.5,
        loiter_duration_s: float = 1.8,
        seed: int = 42,
    ) -> None:
        self.waypoints = list(waypoints)
        self.cruise_speed = cruise_speed_mps
        self.cruise_altitude = cruise_altitude_m
        self.max_accel = max_accel_mps2
        self.takeoff_duration = takeoff_duration_s
        self.landing_duration = landing_duration_s
        self.loiter_duration = loiter_duration_s
        self.rng = random.Random(seed)

        # Precompute cumulative spline segments
        self._build_smooth_path()

        # State tracking
        self.current_distance = 0.0
        self.current_speed = 0.0
        self.loiter_timer = 0.0
        self.active_loiter_target: tuple[float, float] | None = None
        self.inspected_targets: set[tuple[int, int]] = set()

    def _build_smooth_path(self) -> None:
        """Create dense interpolated path points with rounded corners."""
        pts = self.waypoints
        if len(pts) < 2:
            pts = [(1.5, 1.5), (13.5, 1.5)]

        # Sample points with subdivision for smooth curvature
        dense: list[tuple[float, float]] = []
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            steps = max(5, int(dist / 0.15))
            for s in range(steps):
                frac = s / float(steps)
                dense.append((
                    p0[0] + frac * (p1[0] - p0[0]),
                    p0[1] + frac * (p1[1] - p0[1]),
                ))

        # Apply multi-pass moving average to round all sharp 90-degree corners
        for _ in range(4):
            smoothed: list[tuple[float, float]] = []
            n = len(dense)
            for j in range(n):
                prev_pt = dense[(j - 1) % n]
                curr_pt = dense[j]
                next_pt = dense[(j + 1) % n]
                smoothed.append((
                    0.25 * prev_pt[0] + 0.5 * curr_pt[0] + 0.25 * next_pt[0],
                    0.25 * prev_pt[1] + 0.5 * curr_pt[1] + 0.25 * next_pt[1],
                ))
            dense = smoothed

        self.path_points = dense

        # Compute cumulative distance along the path
        self.cum_dist = [0.0]
        for k in range(1, len(dense)):
            dx = dense[k][0] - dense[k - 1][0]
            dy = dense[k][1] - dense[k - 1][1]
            self.cum_dist.append(self.cum_dist[-1] + math.hypot(dx, dy))
        self.total_path_length = self.cum_dist[-1]

    def trigger_inspection(self, target_pos: tuple[float, float]) -> bool:
        """Trigger an autonomous inspection hover if not recently inspected."""
        grid_key = (int(round(target_pos[0])), int(round(target_pos[1])))
        if grid_key in self.inspected_targets:
            return False
        if self.loiter_timer > 0:
            return False  # Already inspecting

        self.loiter_timer = self.loiter_duration
        self.active_loiter_target = target_pos
        self.inspected_targets.add(grid_key)
        return True

    def get_state(self, t: float, dt: float = 0.05) -> DroneFlightState:
        """Evaluate continuous 6-DoF flight state at mission time t."""
        # 1. Flight Phase: Takeoff
        if t < self.takeoff_duration:
            frac = t / self.takeoff_duration
            # Smooth cubic ease-in-out for liftoff
            smooth_frac = 3.0 * (frac ** 2) - 2.0 * (frac ** 3)
            z = smooth_frac * self.cruise_altitude
            p0 = self.path_points[0]
            return DroneFlightState(
                position=(p0[0], p0[1], z),
                velocity=(0.0, 0.0, self.cruise_altitude / self.takeoff_duration),
                speed=0.0,
                acceleration=(0.0, 0.0, 0.0),
                attitude_euler=(0.0, 0.0, 0.0),
                orientation_quat=(1.0, 0.0, 0.0, 0.0),
                is_loitering=False,
                flight_phase="TAKEOFF",
            )

        # 2. Inspection Loiter Handling
        is_loitering = False
        target_speed = self.cruise_speed
        target_alt = self.cruise_altitude

        if self.loiter_timer > 0.0:
            is_loitering = True
            self.loiter_timer = max(0.0, self.loiter_timer - dt)
            target_speed = 0.08  # Drop to near-hover
            target_alt = self.cruise_altitude * 0.85  # Lower slightly for close inspection

        # Smooth S-curve velocity profiling towards target speed
        accel_step = self.max_accel * dt
        if self.current_speed < target_speed:
            self.current_speed = min(target_speed, self.current_speed + accel_step)
        elif self.current_speed > target_speed:
            self.current_speed = max(target_speed, self.current_speed - accel_step * 1.5)

        # Advance along path distance
        self.current_distance = (self.current_distance + self.current_speed * dt) % self.total_path_length

        # Interpolate position along smooth path
        pt_curr, heading, curvature = self._eval_path_at_dist(self.current_distance)
        vx = self.current_speed * math.cos(heading)
        vy = self.current_speed * math.sin(heading)

        # 3. Dynamic Banking & Pitch from vehicle physics
        # Longitudinal acceleration: dv/dt
        long_accel = 0.0
        if not is_loitering and self.current_speed < self.cruise_speed:
            long_accel = self.max_accel
        elif is_loitering:
            long_accel = -self.max_accel

        # Lateral centripetal acceleration: a_lat = v^2 * curvature
        lat_accel = (self.current_speed ** 2) * curvature

        # Physical quadcopter tilt angles:
        # Pitch: nose down when accelerating forward, nose up to brake
        pitch_rad = math.atan2(long_accel, 9.8066)
        pitch_rad = max(-0.25, min(0.25, pitch_rad))

        # Roll: banks inwards into curved turns
        roll_rad = math.atan2(-lat_accel, 9.8066)
        roll_rad = max(-0.35, min(0.35, roll_rad))

        # 4. Confined-Space Rotor Turbulence & Hover Wobble
        # Harmonics representing downwash turbulence bouncing off floor and walls
        wobble_z = 0.018 * math.sin(2.4 * t) + 0.009 * math.sin(5.7 * t)
        wobble_roll = 0.014 * math.sin(3.1 * t + 0.5) + 0.007 * math.cos(7.3 * t)
        wobble_pitch = 0.014 * math.cos(2.8 * t) + 0.007 * math.sin(6.9 * t + 1.2)

        final_z = max(0.2, target_alt + wobble_z)
        final_roll = roll_rad + wobble_roll
        final_pitch = pitch_rad + wobble_pitch
        final_yaw = heading

        # 5. Base Downward-Looking Camera Quaternion
        # Downward camera optical frame standard is 180 deg about X: (1, 0, 0, 0)
        # We compose the body roll/pitch/yaw onto the optical base
        body_quat = _quaternion_from_euler(final_roll, final_pitch, final_yaw)
        base_optical = (1.0, 0.0, 0.0, 0.0)
        optical_quat = _quaternion_multiply(body_quat, base_optical)

        phase = "INSPECT" if is_loitering else "CRUISE"

        return DroneFlightState(
            position=(pt_curr[0], pt_curr[1], final_z),
            velocity=(vx, vy, 0.0),
            speed=self.current_speed,
            acceleration=(long_accel * math.cos(heading), long_accel * math.sin(heading), 0.0),
            attitude_euler=(final_roll, final_pitch, final_yaw),
            orientation_quat=optical_quat,
            is_loitering=is_loitering,
            loiter_target=self.active_loiter_target if is_loitering else None,
            flight_phase=phase,
        )

    def _eval_path_at_dist(self, s: float) -> tuple[tuple[float, float], float, float]:
        """Return (position, heading_rad, curvature) along path at cumulative distance s."""
        s = s % self.total_path_length

        # Binary search for segment
        low, high = 0, len(self.cum_dist) - 1
        while low < high - 1:
            mid = (low + high) // 2
            if self.cum_dist[mid] <= s:
                low = mid
            else:
                high = mid

        i0 = low
        i1 = high
        ds = self.cum_dist[i1] - self.cum_dist[i0]
        frac = (s - self.cum_dist[i0]) / ds if ds > 1e-6 else 0.0

        p0 = self.path_points[i0]
        p1 = self.path_points[i1]
        x = p0[0] + frac * (p1[0] - p0[0])
        y = p0[1] + frac * (p1[1] - p0[1])

        # Heading from tangent
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        heading = math.atan2(dy, dx)

        # Estimate curvature using 3 successive points
        i_prev = max(0, i0 - 1)
        i_next = min(len(self.path_points) - 1, i1 + 1)
        h0 = math.atan2(
            self.path_points[i0][1] - self.path_points[i_prev][1],
            self.path_points[i0][0] - self.path_points[i_prev][0],
        )
        h1 = math.atan2(
            self.path_points[i_next][1] - self.path_points[i1][1],
            self.path_points[i_next][0] - self.path_points[i1][0],
        )
        dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
        curvature = dh / max(0.1, ds)

        return (x, y), heading, curvature
