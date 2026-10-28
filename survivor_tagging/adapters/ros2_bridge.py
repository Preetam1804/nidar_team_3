"""ROS 2 Jazzy Integration Node for Survivor Localisation & Map Tagging.

Bridges:
- Team 5: PX4 SITL pose (/fmu/out/vehicle_local_position, /fmu/out/vehicle_attitude)
- Team 2: Camera bounding boxes (/camera/detections or vision_msgs/Detection2DArray)
- Team 1: Occupancy grid (/map or nav_msgs/OccupancyGrid)
- Team 4 / GCS: Emits canonical tag payloads (/survivor_tags/gcs)
- Gazebo & RViz2: Visual 3D markers (/survivor_tags/markers & Gazebo spawn service)
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from typing import Any

import sys
from pathlib import Path

# Support running as a standalone script: python3 ros2_bridge.py
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from survivor_tagging import schemas
from survivor_tagging.config import load_config
from survivor_tagging.pipeline import SurvivorTaggingPipeline
from survivor_tagging.providers import DetectionsProvider, MapProvider, PoseProvider

log = logging.getLogger(__name__)

# Attempt importing rclpy and ROS 2 message definitions
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from std_msgs.msg import String, Header
    from nav_msgs.msg import OccupancyGrid, Odometry
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    HAVE_ROS2 = True
    ROS2_IMPORT_ERROR = ""
except ImportError as exc:
    HAVE_ROS2 = False
    ROS2_IMPORT_ERROR = str(exc)
    Node = object  # type: ignore


class ROS2DetectorsProvider(DetectionsProvider):
    """Buffers detection frames from ROS 2."""
    def __init__(self) -> None:
        self._latest: schemas.DetectionFrame | None = None

    def push(self, frame: schemas.DetectionFrame) -> None:
        self._latest = frame

    def latest_frame(self) -> schemas.DetectionFrame | None:
        return self._latest

    def ack(self, frame_id: str) -> None:
        pass


class ROS2PoseProvider(PoseProvider):
    """Holds the latest 6-DoF pose from PX4 or Odometry."""
    def __init__(self) -> None:
        self._latest: schemas.PoseSample | None = None

    def update(self, pose: schemas.PoseSample) -> None:
        self._latest = pose

    def latest(self) -> schemas.PoseSample | None:
        return self._latest


class ROS2MapProvider(MapProvider):
    """Holds the latest occupancy grid from Team 1 SLAM."""
    def __init__(self) -> None:
        self._latest: schemas.OccupancyGridMsg | None = None

    def update(self, m: schemas.OccupancyGridMsg) -> None:
        self._latest = m

    def latest(self) -> schemas.OccupancyGridMsg | None:
        return self._latest


class SurvivorTaggingNode(Node):  # type: ignore
    """ROS 2 Node orchestrating the Survivor Tagging Pipeline at 20 Hz."""

    def __init__(self, config_path: str | None = None) -> None:
        if not HAVE_ROS2:
            raise RuntimeError("ROS 2 (rclpy) is not installed in the active environment.")

        super().__init__("survivor_tagging_node")
        self.cfg = load_config(config_path)

        # Calibrate camera if 0.0 guard is set
        cam = dict(self.cfg.get("camera", {}))
        if cam.get("fx", 0.0) == 0.0:
            w = cam.get("width", 1536)
            fov = float(self.cfg.get("sim", {}).get("fov_deg", 120.0))
            fx_fy = (w / 2.0) / math.tan(math.radians(fov / 2.0))
            cam["fx"] = cam["fy"] = fx_fy
            cam["cx"] = w / 2.0
            cam["cy"] = cam.get("height", 864) / 2.0
            cam["dist_model"] = "none"
            self.cfg["camera"] = cam

        self.det_provider = ROS2DetectorsProvider()
        self.pose_provider = ROS2PoseProvider()
        self.map_provider = ROS2MapProvider()

        self.pipeline = SurvivorTaggingPipeline(
            self.cfg, self.det_provider, self.pose_provider, self.map_provider
        )

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 1. Odometry / Pose Subscribers
        self.create_subscription(
            Odometry,
            "/model/airmouse/odometry",
            self._on_odometry,
            qos_best_effort,
        )

        # 2. Team 1 SLAM Map Subscriber
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self._on_occupancy_grid,
            qos_transient,
        )

        # 3. Team 2 Detections (JSON string or ROS msg)
        self.create_subscription(
            String,
            "/camera/detections",
            self._on_detection_json,
            qos_best_effort,
        )

        # 4. Publishers
        self.pub_gcs = self.create_publisher(String, "/survivor_tags/gcs", 10)
        self.pub_markers = self.create_publisher(
            MarkerArray, "/survivor_tags/markers", 10
        )

        # Pipeline ticker at 20 Hz (50 ms)
        self.timer = self.create_timer(0.05, self._tick_pipeline)
        self.spawned_beacons: set[str] = set()

        self.get_logger().info(
            "NiDAR AirMouse — Survivor Tagging ROS 2 Node initialized at 20 Hz."
        )

    def _on_odometry(self, msg: Odometry) -> None:
        """Handle incoming drone odometry from Gazebo or PX4 bridge."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Invert or align coordinate frame for camera looking down:
        # Default down-looking optical orientation is (1, 0, 0, 0)
        pose_sample = schemas.PoseSample(
            stamp=stamp,
            position=(float(pos.x), float(pos.y), float(pos.z)),
            orientation=(float(ori.x), float(ori.y), float(ori.z), float(ori.w)),
            sigma_xy=float(self.cfg.get("sim", {}).get("pose_sigma_xy", 0.15)),
        )
        self.pose_provider.update(pose_sample)

    def _on_occupancy_grid(self, msg: OccupancyGrid) -> None:
        """Handle incoming occupancy grid from Team 1."""
        info = msg.info
        occ_msg = schemas.OccupancyGridMsg(
            w=int(info.width),
            h=int(info.height),
            origin_x=float(info.origin.position.x),
            origin_y=float(info.origin.position.y),
            res=float(info.resolution),
            data=tuple(int(v) for v in msg.data),
        )
        self.map_provider.update(occ_msg)

    def _on_detection_json(self, msg: String) -> None:
        """Handle detection message from Team 2 vision pipeline."""
        try:
            payload = json.loads(msg.data)
            frame = schemas.parse_detection_frame(payload)
            self.det_provider.push(frame)
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse detection: {exc}")

    def _tick_pipeline(self) -> None:
        """Run one pipeline tick, publish GCS tags, and update 3D markers."""
        emitted = self.pipeline.tick()
        if not emitted:
            return

        marker_array = MarkerArray()
        for idx, tag in enumerate(emitted):
            tid = tag["tag_id"]
            pos = tag["map_position"]
            cell = tag["cell"]
            col = cell["col"] if isinstance(cell, dict) else cell[0]
            row = cell["row"] if isinstance(cell, dict) else cell[1]
            conf = tag["confidence"]

            # Publish GCS JSON payload
            msg = String()
            msg.data = json.dumps(tag)
            self.pub_gcs.publish(msg)

            # RViz 3D Pin Marker
            m_pin = Marker()
            m_pin.header.frame_id = "map"
            m_pin.header.stamp = self.get_clock().now().to_msg()
            m_pin.ns = "survivors"
            m_pin.id = idx * 2
            m_pin.type = Marker.CYLINDER
            m_pin.action = Marker.ADD
            m_pin.pose.position.x = float(pos[0])
            m_pin.pose.position.y = float(pos[1])
            m_pin.pose.position.z = 0.5
            m_pin.scale.x = 0.15
            m_pin.scale.y = 0.15
            m_pin.scale.z = 1.0
            m_pin.color.r = 0.0
            m_pin.color.g = 0.95
            m_pin.color.b = 0.55
            m_pin.color.a = 0.9
            marker_array.markers.append(m_pin)

            # RViz Text Tag
            m_txt = Marker()
            m_txt.header.frame_id = "map"
            m_txt.header.stamp = self.get_clock().now().to_msg()
            m_txt.ns = "survivor_labels"
            m_txt.id = idx * 2 + 1
            m_txt.type = Marker.TEXT_VIEW_FACING
            m_txt.action = Marker.ADD
            m_txt.pose.position.x = float(pos[0])
            m_txt.pose.position.y = float(pos[1])
            m_txt.pose.position.z = 1.25
            m_txt.scale.z = 0.35
            m_txt.color.r = 1.0
            m_txt.color.g = 1.0
            m_txt.color.b = 1.0
            m_txt.color.a = 1.0
            m_txt.text = f"TAG: {tid[-6:]}\n[{col},{row}] ({conf*100:.1f}%)"
            marker_array.markers.append(m_txt)

            # Spawn physical beacon in Gazebo Harmonic if not already spawned
            if tid not in self.spawned_beacons:
                self.spawned_beacons.add(tid)
                self._spawn_gazebo_beacon(tid, pos[0], pos[1])
                self.get_logger().info(
                    f"★ [SURVIVOR CONFIRMED] Tag: {tid} | Cell: {cell} | Pos: ({pos[0]:.2f}, {pos[1]:.2f}) | Conf: {conf*100:.1f}%"
                )

        if marker_array.markers:
            self.pub_markers.publish(marker_array)

    def _spawn_gazebo_beacon(self, tag_id: str, x: float, y: float) -> None:
        """Call Gazebo Harmonic CLI / service to spawn a visual 3D beacon model."""
        cmd = [
            "gz", "service", "-s", "/world/survivor_arena/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req",
            f'sdf_filename: "gazebo/models/survivor_beacon/model.sdf", name: "beacon_{tag_id[-6:]}", pose: {{position: {{x: {x:.2f}, y: {y:.2f}, z: 0.05}}}}'
        ]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def main(args: list[str] | None = None) -> None:
    if not HAVE_ROS2:
        print(f"[ERROR] rclpy is required to run the survivor tagging ROS 2 node. Reason: {ROS2_IMPORT_ERROR}")
        return

    rclpy.init(args=args)
    node = SurvivorTaggingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
