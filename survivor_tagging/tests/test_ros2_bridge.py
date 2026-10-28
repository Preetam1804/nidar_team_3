"""Tests for the ROS 2 adapter provider buffers."""

import pytest
from survivor_tagging import schemas
from survivor_tagging.adapters.ros2_bridge import (
    ROS2DetectorsProvider,
    ROS2PoseProvider,
    ROS2MapProvider,
)


def test_ros2_providers_buffering():
    det = ROS2DetectorsProvider()
    pose = ROS2PoseProvider()
    mp = ROS2MapProvider()

    assert det.latest_frame() is None
    assert pose.latest() is None
    assert mp.latest() is None

    p_sample = schemas.PoseSample(
        stamp=10.0,
        position=(1.0, 2.0, 2.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        sigma_xy=0.15,
    )
    pose.update(p_sample)
    assert pose.latest() == p_sample

    d_frame = schemas.DetectionFrame(
        stamp=10.0,
        frame_id="cam3noir_optical",
        detections=(
            schemas.Detection(
                stamp=10.0,
                frame_id="cam3noir_optical",
                bbox=(100.0, 100.0, 50.0, 50.0),
                conf=0.9,
                label="person",
            ),
        ),
    )
    det.push(d_frame)
    assert det.latest_frame() == d_frame

    m_msg = schemas.OccupancyGridMsg(
        w=2, h=2, origin_x=0.0, origin_y=0.0, res=1.0, data=(0, 0, 100, 0)
    )
    mp.update(m_msg)
    assert mp.latest() == m_msg
