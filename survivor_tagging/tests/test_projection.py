import math

import pytest

from survivor_tagging import projection as proj
from survivor_tagging.projection import (
    CameraIntrinsics, ProjectionError, normalized_coords, pixel_to_ray,
    quat_rotate, ray_to_floor,
)


def down_quat() -> tuple:
    """Camera mounted looking straight down at the map floor.

    Optical frame: z forward (toward floor), x right, y down (toward
    ground plane).  A 180-degree rotation about the optical x-axis maps
    optical +z to map -z, optical +y to map -y (pointing earthward).
    """
    return (1.0, 0.0, 0.0, 0.0)


class TestIntrinsics:
    def test_zero_fx_rejected(self):
        with pytest.raises(ProjectionError):
            CameraIntrinsics(fx=0.0, fy=700, cx=1, cy=1, model="none")

    def test_bad_model_rejected(self):
        with pytest.raises(ProjectionError):
            CameraIntrinsics(fx=700, fy=700, cx=1, cy=1, model="bogus")

    def test_radial_needs_4_params(self):
        with pytest.raises(ProjectionError):
            CameraIntrinsics(fx=700, fy=700, cx=1, cy=1, model="radial_k1k2_p12",
                             params=(0.1, 0.2))


class TestDistortion:
    def test_identity_when_none(self):
        intr = CameraIntrinsics(fx=700, fy=700, cx=512, cy=384, model="none")
        xn, yn = normalized_coords(intr, 512.0, 384.0)
        assert abs(xn) < 1e-9
        assert abs(yn) < 1e-9

    def test_mild_k1_roundtrip(self):
        """For a mild k1, forward-project the ideal point, then invert."""
        k1 = 0.1
        k2 = 0.0
        intr = CameraIntrinsics(fx=700, fy=700, cx=512, cy=384,
                                model="radial_k1k2_p12", params=(k1, k2, 0, 0))
        xn_true = 0.2
        yn_true = -0.15
        r2 = xn_true**2 + yn_true**2
        xn_dist = xn_true * (1 + k1 * r2 + k2 * r2**2)
        yn_dist = yn_true * (1 + k1 * r2 + k2 * r2**2)
        u = 512 + 700 * xn_dist
        v = 384 + 700 * yn_dist
        xn_inv, yn_inv = normalized_coords(intr, u, v)
        assert abs(xn_inv - xn_true) < 1e-6
        assert abs(yn_inv - yn_true) < 1e-6

    def test_fisheye_grows_toward_edge(self):
        intr = CameraIntrinsics(fx=700, fy=700, cx=512, cy=384,
                                model="fisheye", params=(0.2, 0.05, 0.01, 0.001))
        u = 512 + 700 * 1.0
        v = 384
        xn, yn = normalized_coords(intr, u, v)
        assert abs(yn) < 1e-9
        assert 0.3 < xn < 2.0


class TestQuat:
    def test_identity(self):
        v = (1.0, 2.0, 3.0)
        assert quat_rotate((0, 0, 0, 1), v) == pytest.approx(v)

    def test_90deg_yaw(self):
        s = math.sqrt(0.5)
        v = (1.0, 0.0, 0.0)
        assert quat_rotate((0, 0, s, s), v) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)

    def test_normalize(self):
        q = (0.1, 0.2, 0.3, 0.4)
        n = math.sqrt(sum(c * c for c in q))
        qn = tuple(c / n for c in q)
        v = (1.0, 0.0, 0.0)
        vr = quat_rotate(qn, v)
        # rotation preserves full 3-norm of the vector
        assert math.sqrt(sum(c * c for c in vr)) == pytest.approx(1.0, abs=1e-6)


class TestFloor:
    def test_straight_down_hit(self):
        intr = CameraIntrinsics(fx=700, fy=700, cx=512, cy=384, model="none")
        ray = pixel_to_ray(intr, 512, 384)   # centre ray, straight along optical z
        hit = ray_to_floor(ray, down_quat(), (5.0, 5.0, 2.0), 0.0)
        assert hit.hit is True
        assert hit.x == pytest.approx(5.0, abs=1e-6)
        assert hit.y == pytest.approx(5.0, abs=1e-6)
        assert hit.dist_m == pytest.approx(0.0, abs=1e-6)

    def test_ray_up_no_hit(self):
        # rz>0 after rotation: points toward ceiling from the camera pose
        ray = (0.0, 0.0, 1.0)
        quat_up = (0.0, 0.0, 0.0, 1.0)  # optical z aligned with map +z (up)
        hit = ray_to_floor(ray, quat_up, (5.0, 5.0, 2.0), 0.0)
        assert hit.hit is False

    def test_ray_horizontal_no_hit(self):
        # yaw-only rotation about map z never changes rz, so a purely
        # horizontal optical ray stays horizontal: rz == 0.0 -> no hit
        ray = (1.0, 0.0, 0.0)
        hit = ray_to_floor(ray, down_quat(), (5.0, 5.0, 2.0), 0.0)
        assert hit.hit is False

    def test_offset_pixel(self):
        intr = CameraIntrinsics(fx=700, fy=700, cx=512, cy=384, model="none")
        # A pixel off-centre should produce a ground point off-centre too.
        ray = pixel_to_ray(intr, 512 + 350, 384)
        hit = ray_to_floor(ray, down_quat(), (5.0, 5.0, 2.0), 0.0)
        assert hit.hit is True
        # 500 pixels off-centre at 700px focal length: x_n = 0.5 -> at z=2 m
        # the ground offset is roughly 2.0 * 0.5 / 1.0 * ... ~ 1 m
        assert 0.5 < abs(hit.x - 5.0) < 2.5


class TestEdgeFlag:
    def test_centre_not_edge(self):
        assert proj.edge_zone_flag(100, 100, 50, 50, 0.85) is False

    def test_corner_is_edge(self):
        assert proj.edge_zone_flag(100, 100, 5, 5, 0.85) is True

    def test_mid_frame_threshold(self):
        assert proj.edge_zone_flag(100, 100, 93, 50, 0.85) is True
