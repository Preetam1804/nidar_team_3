"""Integration tests: full pipeline over in-memory providers."""
import math

import pytest

from survivor_tagging import schemas
from survivor_tagging.adapters import (
    InMemoryDetectors, InMemoryMap, InMemoryPose,
)
from survivor_tagging.pipeline import SurvivorTaggingPipeline


def free_arena(w=15, h=15):
    data = [schemas.FREE] * (w * h)
    return schemas.OccupancyGridMsg(
        w=w, h=h, origin_x=0.0, origin_y=0.0, res=1.0, data=tuple(data)
    )


def make_pipeline(cfg):
    det, pose, mp = InMemoryDetectors(), InMemoryPose(), InMemoryMap()
    return SurvivorTaggingPipeline(cfg, det, pose, mp), det, pose, mp


def down_pose(stamp, x, y, z=2.0, sigma=0.02):
    # quat (1,0,0,0) = 180 deg about x: optical z points toward the floor
    return schemas.PoseSample(
        stamp=stamp, position=(x, y, z),
        orientation=(1.0, 0.0, 0.0, 0.0),
        sigma_xy=sigma,
    )


def bbox_for(cfg, world_x, world_y, cam_x, cam_y, cam_z=2.0, half_span=0.15):
    """Pixel bbox for a survivor at world (world_x, world_y) under a
    camera looking straight down (optical z toward floor, identity in-plane)."""
    intr = cfg["camera"]
    fx = intr["fx"]
    # ground half-span covered by (width/2) px:
    #   x_n = (width/2)/fx, ground_m = z * x_n
    ground_half = cam_z * (intr["width"] / 2.0) / fx
    u_per_m = (intr["width"] / 2.0) / ground_half
    v_per_m = (intr["height"] / 2.0) / ground_half
    u = intr["width"] / 2.0 + (world_x - cam_x) * u_per_m
    v = intr["height"] / 2.0 - (world_y - cam_y) * v_per_m
    px = max(0.0, u - 10)
    py = max(0.0, v - 10)
    return (px, py, 20.0, 20.0)


class TestHappyPath:
    def test_confirm_and_emit_tag(self, cfg):
        pl, det, pose, mp = make_pipeline(cfg)
        mp.push(free_arena())
        t = 1.0
        pose.push(down_pose(t, 5.0, 5.0))
        det.push(schemas.DetectionFrame(
            stamp=t, frame_id=cfg["camera"]["frame_id"],
            detections=(schemas.Detection(
                stamp=t, frame_id=cfg["camera"]["frame_id"],
                bbox=bbox_for(cfg, 5.5, 5.0, 5.0, 5.0),
                conf=0.9, label="person",
            ),),
        ))
        pl.tick()      # TENTATIVE
        det.push(schemas.DetectionFrame(
            stamp=t + 0.1, frame_id=cfg["camera"]["frame_id"],
            detections=(schemas.Detection(
                stamp=t + 0.1, frame_id=cfg["camera"]["frame_id"],
                bbox=bbox_for(cfg, 5.5, 5.0, 5.0, 5.1),
                conf=0.9, label="person",
            ),),
        ))
        out = pl.tick()
        assert out, "expected a confirmed tag to be emitted"
        p = out[0]
        assert p["state"] == "CONFIRMED"
        assert p["cell"]["col"] == 5
        assert p["cell"]["row"] == 4 or p["cell"]["row"] == 5

    def test_duplicate_detections_single_tag(self, cfg):
        pl, det, pose, mp = make_pipeline(cfg)
        mp.push(free_arena())
        for i in range(5):
            t = 1.0 + 0.1 * i
            pose.push(down_pose(t, 5.0 + 0.01 * i, 5.0))
            det.push(schemas.DetectionFrame(
                stamp=t, frame_id="cam",
                detections=(schemas.Detection(
                    stamp=t, frame_id="cam",
                    bbox=bbox_for(cfg, 5.5, 5.0, 5.0 + 0.01 * i, 5.0),
                    conf=0.9, label="person",
                ),),
            ))
            pl.tick()
        # one track from five near-identical detections at same map spot
        conf_tracks = pl.tracker.confirmed_tracks()
        assert len(conf_tracks) == 1


class TestFailureModes:
    def test_no_pose_no_output(self, cfg):
        pl, det, pose, mp = make_pipeline(cfg)
        mp.push(free_arena())
        det.push(schemas.DetectionFrame(
            stamp=1.0, frame_id="cam",
            detections=(schemas.Detection(
                stamp=1.0, frame_id="cam", bbox=(700, 380, 20, 20),
                conf=0.9, label="person",
            ),),
        ))
        assert pl.tick() == []

    def test_no_map_still_projects(self, cfg):
        """No occupancy map should not block tagging (snap disabled)."""
        pl, det, pose, mp = make_pipeline(cfg)
        t = 1.0
        pose.push(down_pose(t, 5.0, 5.0))
        det.push(schemas.DetectionFrame(
            stamp=t, frame_id="cam",
            detections=(schemas.Detection(
                stamp=t, frame_id="cam",
                bbox=bbox_for(cfg, 5.5, 5.0, 5.0, 5.0),
                conf=0.9, label="person",
            ),),
        ))
        pl.tick()
        det.push(schemas.DetectionFrame(
            stamp=t + 0.1, frame_id="cam",
            detections=(schemas.Detection(
                stamp=t + 0.1, frame_id="cam",
                bbox=bbox_for(cfg, 5.5, 5.0, 5.0, 5.1),
                conf=0.9, label="person",
            ),),
        ))
        out = pl.tick()
        assert out, "tags should still be emitted without a map"

    def test_pose_dropout_then_recovery(self, cfg):
        pl, det, pose, mp = make_pipeline(cfg)
        mp.push(free_arena())

        def push_det(t, cam_x):
            det.push(schemas.DetectionFrame(
                stamp=t, frame_id="cam",
                detections=(schemas.Detection(
                    stamp=t, frame_id="cam",
                    bbox=bbox_for(cfg, 5.5, 5.0, cam_x, 5.0),
                    conf=0.9, label="person",
                ),),
            ))

        push_det(1.0, 5.0)
        pose.push(down_pose(1.0, 5.0, 5.0))
        pl.tick()
        push_det(1.1, 5.0)
        pose.push(down_pose(1.1, 5.0, 5.0))
        pl.tick()
        n_confirmed_after_start = len(pl.tracker.confirmed_tracks())
        assert n_confirmed_after_start >= 1
        # pose stops arriving — the next frame is tagged at reduced
        # confidence with pose_age flagged (plan: stale-but-flagged beats
        # silently dropped); it must NOT publish at full confidence.
        push_det(2.0, 5.0)
        out = pl.tick()
        if out:
            p = out[0]
            assert p["confidence"] <= 0.9 * 0.55   # age factor applied
            assert p["flags"]["pose_age_s"] > 0.5
        # pose recovers — pipeline keeps working
        pose.push(down_pose(3.0, 5.0, 5.0))
        push_det(3.1, 5.0)
        pl.tick()
        assert len(pl.tracker.confirmed_tracks()) >= 1

    def test_malformed_detection_does_not_crash(self, cfg):
        pl, det, pose, mp = make_pipeline(cfg)
        mp.push(free_arena())
        pose.push(down_pose(1.0, 5.0, 5.0))
        bad = {"stamp": "not-a-number", "frame_id": "cam",
               "bbox": [1, 2, 3, 4], "conf": 0.5, "label": "person"}
        with pytest.raises(schemas.SchemaError):
            schemas.parse_detection(bad)
        # the pipeline itself catches projection errors internally; feed
        # something absurd but valid: huge bbox, out-of-frame, still safe
        det.push(schemas.DetectionFrame(
            stamp=1.0, frame_id="cam",
            detections=(schemas.Detection(
                stamp=1.0, frame_id="cam",
                bbox=(1e6, 1e6, 1.0, 1.0), conf=0.9, label="person",
            ),),
        ))
        assert pl.tick() == []   # no crash, no output

    def test_unpublished_below_min_conf(self, cfg):
        cfg = dict(cfg)
        cfg["tags"] = dict(cfg["tags"])
        cfg["tags"]["min_publish_conf"] = 0.99
        pl, det, pose, mp = make_pipeline(cfg)
        mp.push(free_arena())
        t = 1.0
        pose.push(down_pose(t, 5.0, 5.0))
        for i in range(5):
            det.push(schemas.DetectionFrame(
                stamp=t + 0.1 * i, frame_id="cam",
                detections=(schemas.Detection(
                    stamp=t + 0.1 * i, frame_id="cam",
                    bbox=bbox_for(cfg, 5.5, 5.0, 5.0, 5.0),
                    conf=0.9, label="person",
                ),),
            ))
            out = pl.tick()
        # All tags below 0.99 publish threshold, so nothing published
        assert out == []


class TestSoakSignature:
    def test_many_ticks_no_growth(self, cfg):
        """Cheap stand-in for the soak: many ticks, track count bounded."""
        pl, det, pose, mp = make_pipeline(cfg)
        mp.push(free_arena())
        # one persistent survivor; same cell seen repeatedly
        for i in range(400):
            t = i * 0.1
            pose.push(down_pose(t, 5.0 + 0.01 * i % 1.0, 5.0))
            det.push(schemas.DetectionFrame(
                stamp=t, frame_id="cam",
                detections=(schemas.Detection(
                    stamp=t, frame_id="cam",
                    bbox=bbox_for(cfg, 5.5, 5.0, 5.0 + 0.01 * i % 1.0, 5.0),
                    conf=0.9, label="person",
                ),),
            ))
            pl.tick()
        assert len(pl.tracker.active_tracks()) <= 4
        assert pl.metrics.ticks == 400

    def test_run_soak_bounded_memory(self, cfg):
        from taggingsim.sim import SimConfig, run_soak
        simcfg = SimConfig(seed=42, n_survivors=4)
        report = run_soak(simcfg, cfg, duration_s=120.0)
        assert report["bounded_memory"] is True
        assert report["max_active_tracks"] <= report["max_allowed_tracks"]
        assert report["ticks"] == 2401


class TestPlaybackAdapter:
    def test_jsonl_playback_roundtrip(self, tmp_path, cfg):
        import json
        from survivor_tagging.adapters import JSONLPlaybackAdapter
        from taggingsim.cli import run_playback_mode

        log_file = tmp_path / "test_mission.jsonl"
        rows = [
            {
                "source": "map", "stamp": 0.0,
                "payload": {
                    "w": 15, "h": 15, "origin_x": 0.0, "origin_y": 0.0, "res": 1.0,
                    "data": [0] * 225,
                },
            },
            {
                "source": "pose", "stamp": 0.0,
                "payload": {
                    "stamp": 0.0, "position": [5.0, 5.0, 2.0],
                    "orientation": [1.0, 0.0, 0.0, 0.0], "sigma_xy": 0.05,
                },
            },
            {
                "source": "detections", "stamp": 0.0,
                "payload": {
                    "stamp": 0.0, "frame_id": "cam3noir_optical",
                    "detections": [{
                        "stamp": 0.0, "frame_id": "cam3noir_optical",
                        "bbox": [760.0, 424.0, 16.0, 16.0], "conf": 0.95,
                        "label": "person",
                    }],
                },
            },
            {
                "source": "detections", "stamp": 0.1,
                "payload": {
                    "stamp": 0.1, "frame_id": "cam3noir_optical",
                    "detections": [{
                        "stamp": 0.1, "frame_id": "cam3noir_optical",
                        "bbox": [760.0, 424.0, 16.0, 16.0], "conf": 0.95,
                        "label": "person",
                    }],
                },
            },
        ]
        with log_file.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        res = run_playback_mode(str(log_file), cfg)
        assert res["mode"] == "playback"
        assert res["total_rows"] == 4
        assert res["metrics"]["ticks"] >= 3
