# NiDAR AirMouse — Team 3 Experiment & Validation Plan

**Document**: Experiment Plan & Flight Validation Protocols  
**Module**: `survivor_tagging` (Team 3 — Survivor Localisation & Map Tagging)  
**Applicability**: Phase 2 (Simulation & Bench Playback) & Phase 3 (HIL & Flight Trials)  

---

## 1. Objectives & Validation Scope

The objective of this experiment plan is to establish a systematic, repeatable verification framework for the **NiDAR AirMouse Team 3 Survivor Localisation and Tagging Module**.

The plan validates:
1. **Geometric Localisation Accuracy**: Precision of monocular ray-plane projection down to the floor plane under realistic flight dynamics and sensor noise.
2. **Track Integrity & Multi-Target Association**: Prevention of false tracks, duplicate tag emissions, and unneeded track churn across overlapping viewpoints.
3. **Robustness & Degradation Management**: Graceful handling of pose dropouts, high covariance, optical edge distortion, and obstacle occlusions.
4. **Computational Budget Adherence**: Compliance with the sub-250g nano-UAV compute envelope ($\le 2.1\text{ W}$ vision payload, $\le 50\text{ ms}$ tick latency).
5. **Long-Duration Reliability**: Zero memory leaks or unbounded track growth during 30-minute search missions.

---

## 2. Test Architecture & Equipment

### 2.1 Simulation & Bench Testbed (Phase 2)
- **Harness**: `taggingsim` discrete-event synthetic mission simulator.
- **Data Adapters**: `InMemoryDetectors`, `InMemoryPose`, `InMemoryMap`, `JSONLPlaybackAdapter`.
- **Operating System**: Linux Cloud VM / Ubuntu 22.04 / Windows 11.
- **Python Runtime**: Python 3.10+ (tested 3.12.3).

### 2.2 Hardware-in-the-Loop & Flight Rig (Phase 3)
- **Avionics Platform**: NiDAR AirMouse nano-drone (quadcopter, sub-250g MTOW).
- **Compute Unit**: Raspberry Pi Zero 2W / CM4 (Quad-core Cortex-A53).
- **Vision Sensor**: Raspberry Pi Camera 3 NoIR Wide (120° diagonal FOV) with calibrated radial distortion.
- **Active Illumination**: 850 nm IR LED flood illuminator (subterranean darkness operation).
- **Flight Controller**: Pixhawk / PX4 Autopilot running state estimation EKF.
- **Ranging Sensor**: Downward-facing micro-LiDAR / ToF rangefinder (e.g. TFmini-S or VL53L1X).

---

## 3. Experiment Matrix

| Experiment ID | Title | Objective | Harness / Command | Acceptance Gate |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Baseline Mission Benchmark | Verify baseline cell accuracy, zero duplicates, and latency | `python -m taggingsim run --seed 7 --survivors 6` | Rate $\ge 85\%$, False $< 1\%$, Churn $\le 1.0$ |
| **EXP-02** | Altimeter Noise Sweep | Quantify impact of vertical position uncertainty ($\sigma_z$) | `taggingsim` parameter sweep: $\sigma_z \in [0.0, 0.5]\text{ m}$ | Rate $\ge 85\%$ for $\sigma_z \le 0.20\text{ m}$ |
| **EXP-03** | SLAM Covariance Sweep | Evaluate spatial gating under horizontal drift ($\sigma_{xy}$) | `taggingsim` parameter sweep: $\sigma_{xy} \in [0.05, 0.35]\text{ m}$ | Rate $\ge 80\%$ for $\sigma_{xy} \le 0.20\text{ m}$ |
| **EXP-04** | Edge-Zone Distortion | Verify wide-FOV edge penalty suppression | `taggingsim` with varying `edge_zone_frac` & publish confidence | Zero spurious tracks outside arena |
| **EXP-05** | Map Conflict Withdrawal | Verify withdrawal of tags overlapping occupied cells | `pytest survivor_tagging/tests/test_tracker.py` | Overlapping tag withdrawn immediately |
| **EXP-06** | 30-Minute Continuous Soak | Stress test memory stability, track capping, and latency | `python -m taggingsim soak --minutes 30` | Active tracks $\le 64$, Zero memory leak |
| **EXP-07** | Deterministic Playback | Validate recorded flight mission replay | `python -m taggingsim --playback <file.jsonl>` | $100\%$ log frame ingestion, zero dropouts |
| **EXP-08** | Flight HIL Ground Truth | Compare flight tags against optical motion-capture ground truth | Vicon/OptiTrack arena flight trial | Mean horizontal error $\le 0.50\text{ m}$ |

---

## 4. Detailed Experiment Protocols

### EXP-01: Baseline Mission Benchmark
- **Procedure**:
  1. Load default [`config.yaml`](file:///c:/Users/Preetam/Desktop/team%203/config.yaml).
  2. Execute `python -m taggingsim run --seed 7 --survivors 6 --out exp01_report.json`.
  3. Validate summary report for correct tags, incorrect tags, track churn, and latencies.
- **Pass Criteria**:
  - Correct-cell rate $\ge 85.0\%$.
  - False-tag rate $< 1.0\%$.
  - Duplicate count $= 0$.
  - Churn per confirmed tag $\le 1.0$.
  - p95 tick latency $< 50\text{ ms}$.

### EXP-02: Altimeter Uncertainty & Rangefinder Failure Sweep
- **Rationale**: Monocular floor-plane projection scales horizontal coordinates inversely with altitude. A $20\%$ altitude error produces a $20\%$ horizontal ray shift.
- **Procedure**:
  1. Sweep `sim.drone_elev_sigma_m` through $[0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]\text{ m}$.
  2. Measure correct-cell rate and false-tag creation count for each altitude noise setting.
- **Expected Outcome**:
  - For downward rangefinder ($\sigma_z \le 0.10\text{ m}$): correct rate $> 95\%$.
  - For fused EKF ($\sigma_z = 0.15\text{ m}$): correct rate $\ge 89\%$.
  - When rangefinder is disabled and barometer drifts ($\sigma_z = 0.50\text{ m}$): correct rate drops to $\sim 83\%$.

### EXP-03: Visual SLAM Covariance Sweep
- **Procedure**:
  1. Sweep `pose_sigma_xy` through $[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]\text{ m}$.
  2. Confirm tracker association filter fuses detections into unified tracks up to $\sigma_{xy} = 0.20\text{ m}$.
- **Pass Criteria**:
  - Duplicate tags remain $0$ across all sweeps.
  - Track count does not bifurcate into split tracks.

### EXP-04: Camera Edge-Zone & Low-Ray Suppression
- **Procedure**:
  1. Inject synthetic detections in the outer $15\%$ of the camera frame (`edge_zone = True`).
  2. Verify that confidence fusion penalises the candidate tag with `edge_penalty = 0.75`.
  3. Verify that low rays ($< 5^\circ$ ground elevation) apply `low_ray_penalty = 0.70`.
- **Pass Criteria**:
  - Marginal detections below `min_publish_conf = 0.50` are held as `TENTATIVE` and not broadcast to GCS until the drone centers overhead.

### EXP-05: Occupancy Grid Conflict & Structural Withdrawal
- **Procedure**:
  1. Tag a confirmed survivor at $(X=3.5, Y=2.5)$ in a FREE cell.
  2. Ingest updated map message from Team 1 marking cell $(3, 2)$ as OCCUPIED (wall/rubble discovered).
  3. Tick pipeline.
- **Pass Criteria**:
  - Track is immediately moved to `REMOVED`.
  - Tag withdrawal event is recorded and logged.

### EXP-06: 30-Minute Continuous Mission Soak
- **Procedure**:
  1. Execute `python -m taggingsim soak --minutes 30 --out exp06_soak.json`.
  2. Monitor process memory consumption via `Metrics.sample_resources()`.
  3. Verify active tracks count at every tick.
- **Pass Criteria**:
  - Total elapsed ticks: $36,001$.
  - Peak memory growth $< 5\text{ MB}$ over baseline.
  - Peak active tracks strictly bounded $\le 64$ (`max_map_units`).
  - Maximum single-tick latency $< 50\text{ ms}$.

### EXP-07: Log Playback & Deterministic Replay
- **Procedure**:
  1. Ingest recorded flight log file `mission.jsonl`.
  2. Execute `python -m taggingsim --playback mission.jsonl --out playback_results.json`.
  3. Repeat run with identical inputs and verify bit-for-bit identical output payloads.
- **Pass Criteria**:
  - Output payloads, tag IDs, confidence scores, and cells are 100% deterministic.

---

## 5. Phase 3 Flight Test Procedure & Data Recording

When transitioning to flight testing in the subterranean tunnel arena:
1. **Pre-Flight Camera Calibration**:
   - Team 2 records a checkerboard calibration video to produce `fx, fy, cx, cy` and distortion coefficients `k1, k2, p1, p2`.
   - Update `camera.fx, fy, cx, cy` in `config.yaml`.
2. **Pre-Flight Sensor Health Check**:
   - Confirm Team 5 PX4 publishes `pose` at $\ge 50\text{ Hz}$.
   - Confirm Team 1 publishes `map` metadata and resolution ($1.0\text{ m}$).
   - Confirm downward LiDAR reports non-zero altitude.
3. **In-Flight Data Logging**:
   - Run lightweight JSONL logger logging Team 2 detections and Team 5 poses.
4. **Post-Flight Scoring**:
   - Feed logged flight data into `taggingsim --playback flight.jsonl`.
   - Compare emitted tag coordinates against surveyed survivor mannequin markers.

---

## 6. Sign-Off & Verification Schedule

| Milestone | Deliverable | Status |
| :--- | :--- | :--- |
| **Phase 2 Simulation** | 76 unit/integration tests, synthetic benchmark harness, soak runner | ✅ Complete |
| **Phase 2 Benchmark** | Acceptance gates verified ($\ge 85\%$ accuracy, zero duplicates) | ✅ Complete |
| **Phase 3 HIL Prep** | Log playback adapter and standardized schemas frozen | ✅ Complete |
| **Phase 3 Flight Trials** | Motion-capture arena trials and subterranean search flights | Scheduled |
