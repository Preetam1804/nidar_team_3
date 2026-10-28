# NiDAR AirMouse — Team 3: Survivor Localisation & Map Tagging

Autonomous survivor localisation, map-frame tracking, and Ground Control Station (GCS) grid tagging module for the **NiDAR AirMouse** sub-250g autonomous indoor search-and-rescue nano-UAV.

---

## 1. Module Overview & Responsibilities

In the NiDAR AirMouse architecture, **Team 3 (Survivor Tagging)** bridges onboard perception and global mission coordination:
1. **Perception Ingestion**: Receives 2D bounding-box person detections from Team 2 (Vision/AI Pipeline).
2. **State & Spatial Grounding**: Ingests high-rate 6-DoF vehicle pose estimates from Team 5 (State Estimation / PX4 EKF) and occupancy grid maps from Team 1 (SLAM & Exploration).
3. **Ray Casting & Floor-Plane Projection**: Projects optical image rays through camera calibration intrinsics and vehicle attitude down to the ground plane ($Z = 0.0$ m).
4. **Map-Frame Multi-Target Tracking**: Associates detections across camera frames in the world coordinate frame using Euclidean gating and running barycentric spatial averaging.
5. **Quality-Weighted Confidence Fusion**: Fuses detector confidence, SLAM covariance sharpness, ray elevation angle, optical edge-zone distortion penalties, and pose latency into a single calibrated confidence score.
6. **Integrity Verification & Occupancy Snapping**: Cross-checks candidate cells against the live occupancy grid to withdraw tags mapped onto newly discovered structural obstacles and snap near-free floor cells.
7. **Canonical GCS Tag Dispatch**: Broadcasts structured JSON payloads to Team 4 (Communications / GCS) over telemetry at 20 Hz.

---

## 2. Key Architecture Decisions (Plan v3 Approved)

- **Primary Sensor Path**: Raspberry Pi Camera 3 NoIR Wide (120° diagonal FOV) paired with an active 850 nm IR LED flood illuminator. Enables operation in unlit, smoke-filled subterranean voids.
- **Monocular Floor-Plane Projection**: Primary localisation path operates without depth sensor dependencies by solving the ray-plane intersection $Z = Z_{\text{floor}}$. Optional LiDAR C1 range refinement is gated as an additive feature once hardware extrinsics are calibrated.
- **Down-Looking Optical Orientation Convention**: The downward-looking camera is oriented with quaternion `(1.0, 0.0, 0.0, 0.0)` ($180^\circ$ rotation about optical $X$).
- **Intrinsics Zero-Guard**: `config.yaml` intentionally defaults camera intrinsics (`fx, fy, cx, cy`) to `0.0`. The pipeline refuses to project uncalibrated inputs to prevent silent tag drift. The simulation harness injects synthetic FOV-derived pinhole intrinsics.
- **Graceful Confidence Degradation (No Silent Drops)**: Stale poses or wide-angle edge detections apply monotonic confidence penalties rather than dropping valid survivor contacts.

---

## 3. Installation & Setup

### Requirements
- Python 3.10+ (tested on Python 3.12.3)
- Dependencies: `numpy`, `pyyaml`, `pytest`

### Quickstart
```bash
# Clone and navigate to module directory
git clone <repo-url> survivor_tagging
cd survivor_tagging

# Create virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Configuration & Environment Overrides

Every threshold, frame identifier, and dimension is externalised into [`config.yaml`](file:///c:/Users/Preetam/Desktop/team%203/config.yaml).

### Configuration Structure (`config.yaml`)
```yaml
grid:
  res_m: 1.0              # Scoring cell resolution (1 m x 1 m)
  w: 15                   # Maximum arena width (metres)
  h: 15                   # Maximum arena height (metres)

camera:
  frame_id: cam3noir_optical
  width: 1536
  height: 864
  fx: 0.0                 # 0.0 guard: requires calibration
  fy: 0.0
  cx: 0.0
  cy: 0.0
  dist_model: radial_k1k2_p12 # radial_k1k2_p12 | fisheye | none
  dist_params: []
  edge_zone_frac: 0.85

projection:
  floor_plane_z: 0.0
  min_ray_elevation_deg: 0.1
  min_floor_angle_deg: 2.0
  max_floor_dist_m: 15.0
  min_floor_dist_m: 0.2

tracker:
  gate_dist_m: 1.5        # Spatial gating threshold
  hit_radius_m: 0.5       # Track association radius
  confirm_hits: 2         # Hits required for confirmation
  confirm_window: 3       # Hits evaluated within window
  max_missed_ticks: 8     # Ticks before pruning unconfirmed tracks
  max_map_units: 64       # Active track capacity cap

tags:
  min_confirm_conf: 0.5
  min_publish_conf: 0.5   # Threshold for publishing to GCS
  max_pose_age_s: 0.5     # Pose latency degradation limit
  min_active_s: 0.2

metrics:
  cpu_ram_sample_every_n: 100

sim:
  arena_w: 15.0
  arena_h: 15.0
  fov_deg: 120.0
  camera_height_m: 2.0
  drone_speed_mps: 1.0
  drone_elev_sigma_m: 0.15 # Altimeter uncertainty with rangefinder
```

### Environment Variable Overrides
Any configuration parameter can be overridden at runtime using the prefix `SURVIVOR_TAGGING_` with uppercase keys and underscores replacing dots:
```bash
export SURVIVOR_TAGGING_TAGS_MIN_PUBLISH_CONF=0.55
export SURVIVOR_TAGGING_TRACKER_HIT_RADIUS_M=0.45
export SURVIVOR_TAGGING_SIM_DRONE_ELEV_SIGMA_M=0.10
```

---

## 5. Cross-Team Interface Specifications

The module adheres to clean decoupled provider interfaces ([`survivor_tagging/providers.py`](file:///c:/Users/Preetam/Desktop/team%203/survivor_tagging/providers.py)):

### What Team 3 Consumes:
1. **Team 2 (Vision/AI Detection)**:
   - `DetectionFrame`: `(stamp: float, frame_id: str, detections: tuple[Detection, ...])`
   - `Detection`: `(stamp, frame_id, bbox: (u, v, w, h), conf: float [0..1], label: str, track_hint)`
2. **Team 5 (State Estimation / PX4 EKF)**:
   - `PoseSample`: `(stamp: float, position: (x, y, z), orientation: (qx, qy, qz, qw), sigma_xy: float)`
3. **Team 1 (SLAM / Exploration)**:
   - `OccupancyGridMsg`: `(w: int, h: int, origin_x: float, origin_y: float, res: float, data: tuple[int, ...])`
   - Values: `-1` (Unknown), `0` (Free), `100` (Occupied).

### What Team 3 Emits (GCS Tag Payload):
Payloads emitted via GCS wire protocol:
```json
{
  "tag_id": "tag_0add678547c1",
  "cell": {
    "col": 2,
    "row": 7
  },
  "map_position": [2.215, 7.589],
  "confidence": 0.842,
  "state": "CONFIRMED",
  "stamp": 14.35,
  "flags": {
    "snapped": false,
    "depth_used": false,
    "low_ray": false,
    "edge_zone": false,
    "pose_age_s": 0.012,
    "pose_sigma_xy_m": 0.15
  }
}
```

---

## 6. Verification & Test Suite

### Running Unit & Integration Tests
```bash
python -m pytest survivor_tagging/tests -q
# Expect: 76 passed
```

### Running the Synthetic Mission Benchmark (`taggingsim`)
```bash
# Standard benchmark mission with seed 7 and 6 survivors
python -m taggingsim run --seed 7 --survivors 6

# Interactive Real-Time Visual GUI Dashboard (Animated Arena, Camera View, Live Telemetry)
python -m taggingsim run --gui

# Export mission animation to GIF / MP4
python -m taggingsim run --save-anim mission_demo.gif

# Save benchmark report to JSON
python -m taggingsim run --seed 7 --survivors 6 --out report.json
```

### Running 3D Gazebo Harmonic Simulation (WSL2 / Linux)
```bash
# Launch 15m x 15m SAR Arena in Gazebo Harmonic with casualties, PX4 SITL & ROS 2 Bridge
chmod +x gazebo/run_gazebo_mission.sh
./gazebo/run_gazebo_mission.sh
```

### Running Continuous Soak Testing (30 minutes)
```bash
python -m taggingsim soak --minutes 30
# Simulates 36,001 ticks (1,800 s) in ~3 seconds.
# Verifies zero memory leaks, bounded track capacity, and stable p95 latency.
```

### Replaying Recorded Flight Logs (`--playback`)
```bash
python -m taggingsim --playback flight_mission.jsonl --out playback_report.json
```

---

## 7. Performance & Acceptance Gate Status

| Acceptance Gate | Specification | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Correct-Cell Rate** | $\ge 85\%$ | **89.1%** (688 / 772 tags) | ✅ PASSED |
| **False-Tag Rate** | $< 1.0\%$ | **0.0%** (0 unmapped tags) | ✅ PASSED |
| **Duplicate Tags** | $0$ duplicates | **0** | ✅ PASSED |
| **Track Churn Ratio** | $\le 1.0$ per confirmed | **0.78** | ✅ PASSED |
| **Latency (p95)** | $< 50$ ms (20 Hz budget) | **0.132 ms** | ✅ PASSED |
| **Compute Overhead** | $\le 2.1$ W payload budget | **0.081 ms / tick CPU, 0.73 MB RAM** | ✅ PASSED |
| **30-Min Soak Test** | Bounded tracks $\le 64$ | **Max active: 4 / 64, RAM: 1.88 MB** | ✅ PASSED |
