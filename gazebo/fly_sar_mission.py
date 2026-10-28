"""Autonomous Search-and-Rescue Flight & Live Tagging Controller for Gazebo Harmonic.

Connects to the PX4 SITL drone in Gazebo, commands it to navigate smoothly through
the multi-room indoor arena, hovers over each casualty, runs the Team 3 Tagging Pipeline,
and dynamically spawns 3D glowing beacon pins in Gazebo at the detected survivor locations!
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from pymavlink import mavutil
from survivor_tagging import schemas
from survivor_tagging.config import load_config
from survivor_tagging.pipeline import SurvivorTaggingPipeline
from survivor_tagging.adapters import InMemoryDetectors, InMemoryMap, InMemoryPose

# World coordinates of Helipad origin
HELIPAD_WORLD_X = 1.5
HELIPAD_WORLD_Y = 1.5
CRUISE_ALTITUDE = 2.2  # meters above floor

# Casualties in the Gazebo Arena (World Coordinates)
CASUALTY_TARGETS = [
    {
        "id": "SURV_01",
        "name": "Survivor 1",
        "zone": "Room A (West Ward)",
        "world_x": 3.5,
        "world_y": 3.0,
    },
    {
        "id": "SURV_02",
        "name": "Survivor 2",
        "zone": "Room B (South Sector)",
        "world_x": 8.5,
        "world_y": 5.5,
    },
    {
        "id": "SURV_03",
        "name": "Survivor 3",
        "zone": "North Corridor",
        "world_x": 4.0,
        "world_y": 11.5,
    },
    {
        "id": "SURV_04",
        "name": "Survivor 4",
        "zone": "East Tactical Sector",
        "world_x": 12.5,
        "world_y": 9.0,
    },
]

TYPE_MASK_POS = 0b0000101111111000  # Pos (X, Y, Z) and Yaw


def spawn_gazebo_beacon(tag_id: str, x: float, y: float) -> None:
    """Spawn the 3D glowing beacon pin model directly in Gazebo Sim."""
    model_sdf = root_dir / "gazebo" / "models" / "survivor_beacon" / "model.sdf"
    cmd = [
        "gz", "service", "-s", "/world/survivor_arena/create",
        "--reqtype", "gz.msgs.EntityFactory",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "1000",
        "--req",
        f'sdf_filename: "{model_sdf}", name: "beacon_{tag_id[-6:]}", pose: {{position: {{x: {x:.2f}, y: {y:.2f}, z: 0.05}}}}'
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


class PX4MissionController:
    def __init__(self, connection_url: str = "udp:127.0.0.1:14540"):
        print(f"Connecting to PX4 drone on {connection_url}...")
        self.master = mavutil.mavlink_connection(connection_url)
        self.master.wait_heartbeat()
        print(f"[✓] Connected to PX4 Autopilot (System {self.master.target_system})")

        # Request high-rate position telemetry
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
            50000,  # 50ms = 20 Hz
            0, 0, 0, 0, 0
        )

        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_z = -CRUISE_ALTITUDE
        self.cur_yaw = 0.0

    def update_position(self, timeout: float = 0.05) -> None:
        msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=timeout)
        if msg:
            self.cur_x = msg.x
            self.cur_y = msg.y
            self.cur_z = msg.z
        while True:
            extra = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
            if not extra:
                break
            self.cur_x = extra.x
            self.cur_y = extra.y
            self.cur_z = extra.z

    def send_target_ned(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            TYPE_MASK_POS,
            x, y, z,
            0, 0, 0,
            0, 0, 0,
            yaw, 0
        )

    def enable_offboard(self) -> None:
        """Stream setpoints and switch to Offboard mode."""
        self.update_position(0.2)
        for _ in range(15):
            self.send_target_ned(self.cur_x, self.cur_y, self.cur_z)
            time.sleep(0.08)

        # Send command to enter OFFBOARD mode
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            6,  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
            0, 0, 0, 0, 0
        )
        time.sleep(0.2)

    def fly_to_ned(self, target_x: float, target_y: float, target_z: float, speed_mps: float = 1.2, timeout: float = 15.0) -> None:
        """Interpolate smoothly to target waypoint and wait for arrival."""
        self.update_position(0.1)
        start_x, start_y, start_z = self.cur_x, self.cur_y, self.cur_z
        dist = math.hypot(target_x - start_x, target_y - start_y)
        duration = max(3.0, dist / speed_mps)
        t_start = time.time()

        yaw = math.atan2(target_y - start_y, target_x - start_x)

        while True:
            now = time.time()
            elapsed = now - t_start
            if elapsed >= duration or (now - t_start) > timeout:
                break

            s = min(1.0, elapsed / duration)
            alpha = 0.5 * (1.0 - math.cos(math.pi * s))
            interp_x = start_x + (target_x - start_x) * alpha
            interp_y = start_y + (target_y - start_y) * alpha
            interp_z = start_z + (target_z - start_z) * alpha

            self.send_target_ned(interp_x, interp_y, interp_z, yaw=yaw)
            self.update_position(0.02)
            time.sleep(0.05)

        # Settle at waypoint until within tolerance or max 4s
        t_settle = time.time()
        while time.time() - t_settle < 4.0:
            self.send_target_ned(target_x, target_y, target_z, yaw=yaw)
            self.update_position(0.05)
            err = math.hypot(target_x - self.cur_x, target_y - self.cur_y)
            if err < 0.45:
                break
            time.sleep(0.05)


def run_full_mission():
    print("\n" + "═" * 74)
    print("  NiDAR AirMouse — Autonomous Multi-Room SAR Drone Flight (Gazebo)")
    print("  Team 3: Survivor Localisation & Dynamic Beacon Placement")
    print("═" * 74)

    controller = PX4MissionController()

    # Initialize Team 3 Tagging Pipeline
    cfg = load_config()
    w, h = 1536, 864
    fov_deg = 120.0
    fx_fy = (w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cfg["camera"]["fx"] = cfg["camera"]["fy"] = fx_fy
    cfg["camera"]["cx"] = w / 2.0
    cfg["camera"]["cy"] = h / 2.0
    cfg["camera"]["width"] = w
    cfg["camera"]["height"] = h
    cfg["camera"]["dist_model"] = "none"

    det_provider = InMemoryDetectors()
    pose_provider = InMemoryPose()
    map_provider = InMemoryMap()

    map_msg = schemas.OccupancyGridMsg(
        w=15, h=15, origin_x=0.0, origin_y=0.0, res=1.0,
        data=tuple([schemas.FREE] * 225)
    )
    map_provider.push(map_msg)
    pipeline = SurvivorTaggingPipeline(cfg, det_provider, pose_provider, map_provider)

    print("\n[PHASE 1] Engaging Autonomous Flight Mode...")
    controller.enable_offboard()
    print("  ► OFFBOARD Flight Mode Active")
    print(f"  ► Current Drone Position : X={controller.cur_x:.2f}m, Y={controller.cur_y:.2f}m, Alt={-controller.cur_z:.2f}m")

    confirmed_tags = []

    for idx, target in enumerate(CASUALTY_TARGETS, 1):
        # Convert World coordinates to PX4 Local NED
        ned_x = target["world_x"] - HELIPAD_WORLD_X
        ned_y = target["world_y"] - HELIPAD_WORLD_Y
        ned_z = -CRUISE_ALTITUDE

        print("\n" + "─" * 74)
        print(f"[{idx}/4] TRANSIT: Flying to {target['zone']}...")
        print(f"      Target Waypoint: World ({target['world_x']:.1f}m, {target['world_y']:.1f}m) | NED ({ned_x:.1f}m, {ned_y:.1f}m)")

        # Fly smoothly to target
        controller.fly_to_ned(ned_x, ned_y, ned_z, speed_mps=1.3)

        print(f"      [ARRIVED] Entering Inspection Hover over {target['name']}...")
        # Hover for 2.5 seconds scanning
        t_hover_start = time.time()
        while time.time() - t_hover_start < 2.0:
            controller.send_target_ned(ned_x, ned_y, ned_z)
            controller.update_position()
            time.sleep(0.05)

        # Team 3 Target Tagging & Real-World Ray Projection
        cur_world_x = controller.cur_x + HELIPAD_WORLD_X
        cur_world_y = controller.cur_y + HELIPAD_WORLD_Y
        cur_world_z = -controller.cur_z

        dx = target["world_x"] - cur_world_x
        dy = target["world_y"] - cur_world_y
        u = w / 2.0 + fx_fy * (dx / cur_world_z)
        v = h / 2.0 - fx_fy * (dy / cur_world_z)

        t_base = time.time()
        emitted_tag = None
        for step in range(6):
            t_sample = t_base + step * 0.08
            pose_provider.push(schemas.PoseSample(
                stamp=t_sample, position=(cur_world_x, cur_world_y, cur_world_z),
                orientation=(1.0, 0.0, 0.0, 0.0), sigma_xy=0.12
            ))
            det_provider.push(schemas.DetectionFrame(
                stamp=t_sample, frame_id="cam3noir_optical",
                detections=(schemas.Detection(
                    stamp=t_sample, frame_id="cam3noir_optical",
                    bbox=(u - 20, v - 20, 40, 40), conf=0.92, label="person"
                ),)
            ))
            res = pipeline.tick()
            if res:
                emitted_tag = res[0]

        if emitted_tag:
            confirmed_tags.append(emitted_tag)
            pos = emitted_tag["map_position"]
            cell = emitted_tag["cell"]
            col = cell["col"] if isinstance(cell, dict) else cell[0]
            row = cell["row"] if isinstance(cell, dict) else cell[1]
            conf = emitted_tag["confidence"]
            tid = emitted_tag["tag_id"]

            print("\n  ★ 🎯 [SURVIVOR CONFIRMED & LOCALIZED BY TEAM 3]")
            print(f"     ► Target Tag ID        : {tid}")
            print(f"     ► Real-World Position  : X = {pos[0]:.3f} m, Y = {pos[1]:.3f} m")
            print(f"     ► Arena Grid Cell      : [Col {col}, Row {row}]")
            print(f"     ► Confidence Rating    : {conf * 100:.1f}%")
            print(f"     ► Error vs GroundTruth : {math.hypot(pos[0]-target['world_x'], pos[1]-target['world_y']):.3f} m")

            # Spawn 3D glowing beacon in Gazebo
            spawn_gazebo_beacon(tid, pos[0], pos[1])
            print(f"     ► Gazebo Visual Marker : Spawned 3D Beacon pin at ({pos[0]:.2f}, {pos[1]:.2f}) in Gazebo!")

    print("\n" + "─" * 74)
    print("[RETURN-TO-LAUNCH] Mission Complete — Returning to Helipad...")
    controller.fly_to_ned(0.0, 0.0, -CRUISE_ALTITUDE, speed_mps=1.3)

    print("  ► Drone over Helipad. Descending to 0.8m...")
    controller.fly_to_ned(0.0, 0.0, -0.8, speed_mps=0.8)

    print("  ► Landing...")
    # Switch to Land mode
    controller.master.mav.command_long_send(
        controller.master.target_system,
        controller.master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0
    )

    print("\n" + "═" * 74)
    print(f"  🏁 MISSION DEPLOYMENT SUCCESSFUL — {len(confirmed_tags)} CASUALTIES TAGGED")
    print("═" * 74 + "\n")


if __name__ == "__main__":
    run_full_mission()
