"""Unit tests for the realistic flight dynamics and tactical SAR search planner."""

import math
import pytest
from taggingsim.flight_dynamics import RealisticFlightDynamics, _quaternion_from_euler
from taggingsim.search_planner import generate_sar_trajectory, TACTICAL_SAR_WAYPOINTS


def test_quaternion_euler_conversion():
    # Identity (0, 0, 0) -> (0, 0, 0, 1)
    qx, qy, qz, qw = _quaternion_from_euler(0.0, 0.0, 0.0)
    assert math.isclose(qw, 1.0, abs_tol=1e-5)
    assert math.isclose(math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw), 1.0, abs_tol=1e-5)


def test_flight_dynamics_takeoff_and_cruise():
    waypoints = [(1.5, 1.5), (5.0, 5.0), (10.0, 5.0), (1.5, 1.5)]
    dynamics = RealisticFlightDynamics(
        waypoints=waypoints,
        cruise_speed_mps=1.0,
        cruise_altitude_m=2.0,
        takeoff_duration_s=2.0,
    )

    # Initial state at t = 0 (on ground)
    s0 = dynamics.get_state(0.0)
    assert s0.flight_phase == "TAKEOFF"
    assert math.isclose(s0.position[2], 0.0, abs_tol=1e-3)
    assert s0.speed == 0.0

    # Mid-takeoff (t = 1.0s)
    s_mid = dynamics.get_state(1.0)
    assert s_mid.flight_phase == "TAKEOFF"
    assert 0.0 < s_mid.position[2] < 2.0

    # Cruise phase (t = 5.0s)
    # Step through simulation to allow speed ramp-up
    t = 0.0
    dt = 0.1
    s_cruise = s0
    while t < 5.0:
        s_cruise = dynamics.get_state(t, dt=dt)
        t += dt

    assert s_cruise.flight_phase == "CRUISE"
    assert math.isclose(s_cruise.position[2], 2.0, abs_tol=0.2)  # within turbulence band
    assert s_cruise.speed > 0.5


def test_flight_dynamics_inspection_loiter():
    waypoints = [(1.5, 1.5), (5.0, 1.5), (5.0, 5.0), (1.5, 1.5)]
    dynamics = RealisticFlightDynamics(
        waypoints=waypoints,
        cruise_speed_mps=1.0,
        loiter_duration_s=1.5,
    )

    # Step to cruise
    for t in [1.0, 2.0, 3.0, 4.0]:
        dynamics.get_state(t, dt=0.5)

    # Trigger inspection hover
    triggered = dynamics.trigger_inspection((3.5, 3.0))
    assert triggered is True

    # State should now be in INSPECT phase with lower speed
    s_inspect = dynamics.get_state(4.1, dt=0.1)
    assert s_inspect.flight_phase == "INSPECT"
    assert s_inspect.is_loitering is True


def test_sar_search_planner_waypoints():
    pts = generate_sar_trajectory()
    assert len(pts) >= 4
    # Starts and ends at helipad
    assert pts[0] == (1.5, 1.5)
    assert pts[-1] == (1.5, 1.5)

    # All points are safely within 15m arena boundaries
    for x, y in pts:
        assert 1.0 <= x <= 14.0
        assert 1.0 <= y <= 14.0
