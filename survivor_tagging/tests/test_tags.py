import pytest

from survivor_tagging import tags


class TestPoseSharpness:
    def test_sharp_full(self):
        assert tags.pose_sharpness(0.02, 0.02) == 1.0

    def test_decay(self):
        v = tags.pose_sharpness(0.03, 0.02)
        assert 0.3 < v < 1.0

    def test_far_grace_floor(self):
        assert tags.pose_sharpness(1.0, 0.02) == 0.3


class TestPoseAge:
    def test_fresh(self):
        assert tags.pose_age_factor(0.0, 0.5) == 1.0

    def test_mid(self):
        v = tags.pose_age_factor(0.25, 0.5)
        assert 0.7 < v < 0.8

    def test_floor(self):
        assert tags.pose_age_factor(5.0, 0.5) == 0.5


class TestFactors:
    def test_edge_zone(self):
        assert tags.edge_zone_factor(True, 0.75) == 0.75
        assert tags.edge_zone_factor(False, 0.75) == 1.0

    def test_low_ray(self):
        assert tags.low_ray_factor(True, 0.7) == 0.7
        assert tags.low_ray_factor(False, 0.7) == 1.0

    def test_snap(self):
        assert tags.snap_factor(True, 0.9) == 0.9
        assert tags.snap_factor(False, 0.9) == 1.0


class TestFused:
    def test_all_perfect(self):
        v = tags.fused_confidence(0.9, 0.02, 0.0, False, False, False)
        assert v == pytest.approx(0.9)

    def test_edge_zone_reduces(self):
        clean = tags.fused_confidence(0.9, 0.02, 0.0, False, False, False)
        edge = tags.fused_confidence(0.9, 0.02, 0.0, True, False, False)
        assert edge < clean

    def test_stale_goes_down_not_zero(self):
        clean = tags.fused_confidence(0.9, 0.02, 0.0, False, False, False)
        stale = tags.fused_confidence(0.9, 0.02, 5.0, False, False, False)
        assert stale < clean
        assert stale > 0.0

    def test_clamped(self):
        v = tags.fused_confidence(1.0, 0.02, 0.0, False, False, False)
        assert 0.0 <= v <= 1.0


class TestPayloadShape:
    def test_payload_fields(self):
        d = tags.build_tag_payload(
            "t1", (2, 3), (2.5, 3.5), 0.7, flags={"snapped": True}, stamp=1.0,
        )
        assert set(d.keys()) == {
            "tag_id", "cell", "map_position", "confidence",
            "flags", "state", "stamp",
        }
        assert d["cell"] == {"col": 2, "row": 3}
        assert d["state"] == "CONFIRMED"
