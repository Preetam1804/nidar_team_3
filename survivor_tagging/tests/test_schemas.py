import math

import pytest

from survivor_tagging import schemas
from survivor_tagging.schemas import SchemaError


class TestDetection:
    def test_valid_minimal(self):
        d = schemas.parse_detection({
            "stamp": 1.0, "frame_id": "cam", "bbox": [10, 10, 20, 20],
            "conf": 0.9, "label": "person",
        })
        assert d.bbox == (10.0, 10.0, 20.0, 20.0)
        assert d.label == "person"

    def test_missing_field(self):
        # stamp is required
        with pytest.raises(SchemaError):
            schemas.parse_detection({
                "frame_id": "cam", "bbox": [0, 0, 20, 20],
                "conf": 0.9, "label": "person",
            })
        # conf is required
        with pytest.raises(SchemaError):
            schemas.parse_detection({
                "stamp": 1.0, "frame_id": "cam", "bbox": [0, 0, 20, 20],
                "label": "person",
            })

    def test_unknown_key_rejected(self):
        with pytest.raises(SchemaError):
            schemas.parse_detection({
                "stamp": 1.0, "frame_id": "cam", "bbox": [0, 0, 20, 20],
                "conf": 0.9, "label": "person", "extra_key": 1,
            })

    def test_confidence_bounds(self):
        with pytest.raises(SchemaError):
            schemas.parse_detection({
                "stamp": 1.0, "frame_id": "cam", "bbox": [0, 0, 20, 20],
                "conf": 1.5, "label": "person",
            })

    def test_nan_rejected(self):
        with pytest.raises(SchemaError):
            schemas.parse_detection({
                "stamp": float("nan"), "frame_id": "cam",
                "bbox": [0, 0, 20, 20], "conf": 0.9, "label": "person",
            })


class TestPose:
    def test_valid(self):
        p = schemas.parse_pose({
            "stamp": 5.0, "position": [1.0, 2.0, 1.5],
            "orientation": [0, 0, 0, 1], "sigma_xy": 0.1,
        })
        assert p.position == (1.0, 2.0, 1.5)

    def test_non_unit_quaternion_rejected(self):
        with pytest.raises(SchemaError):
            schemas.parse_pose({
                "stamp": 5.0, "position": [1, 2, 1.5],
                "orientation": [0, 0, 0, 2], "sigma_xy": 0.1,
            })

    def test_negative_sigma_rejected(self):
        with pytest.raises(SchemaError):
            schemas.parse_pose({
                "stamp": 5.0, "position": [1, 2, 1.5],
                "orientation": [0, 0, 0, 1], "sigma_xy": -0.1,
            })


class TestMap:
    def _map(self, data):
        return {
            "w": 3, "h": 2, "origin_x": 0.0, "origin_y": 0.0, "res": 1.0,
            "data": data,
        }

    def test_valid(self):
        m = schemas.parse_map(self._map([0] * 6))
        assert len(m.data) == 6

    def test_length_mismatch(self):
        with pytest.raises(SchemaError):
            schemas.parse_map(self._map([0] * 5))

    def test_unknown_negative_one(self):
        m = schemas.parse_map(self._map([-1] * 6))
        assert all(v == schemas.UNKNOWN for v in m.data)

    def test_out_of_range(self):
        with pytest.raises(SchemaError):
            schemas.parse_map(self._map([-2] * 6))
        with pytest.raises(SchemaError):
            schemas.parse_map(self._map([101] * 6))


class TestTagUpdate:
    def test_valid_roundtrip(self):
        tu = schemas.make_tag_update(
            "t1", (3, 4), (3.5, 4.5), 0.8,
            flags={"snapped": False}, stamp=1.0,
        )
        d = tu.to_dict()
        assert d["tag_id"] == "t1"
        assert d["cell"] == {"col": 3, "row": 4}
        assert d["confidence"] == 0.8

    def test_conf_clamped(self):
        tu = schemas.make_tag_update("t1", (0, 0), (0.5, 0.5), 2.0)
        assert tu.confidence == 1.0

    def test_bad_state(self):
        with pytest.raises(SchemaError):
            schemas.make_tag_update("t1", (0, 0), (0.5, 0.5), 0.5,
                                    state="WEIRD")
