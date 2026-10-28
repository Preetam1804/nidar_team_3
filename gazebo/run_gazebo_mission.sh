#!/usr/bin/env bash
# ==============================================================================
# NiDAR AirMouse — Team 3: Gazebo Harmonic Survivor Tagging Demonstration
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "  NiDAR AirMouse — Team 3: Survivor Localisation & Tagging Simulation"
echo "  Environment: Gazebo Harmonic + ROS 2 Jazzy + PX4 SITL"
echo "======================================================================"

# 1. Source ROS 2 & PX4 Workspaces
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
    echo "[✓] Sourced ROS 2 Jazzy"
fi

if [ -f "$HOME/px4_ros2_ws/install/setup.bash" ]; then
    source "$HOME/px4_ros2_ws/install/setup.bash"
    echo "[✓] Sourced px4_ros2_ws"
fi

# 2. Setup Gazebo Model & Python Paths
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export GZ_SIM_RESOURCE_PATH="$REPO_DIR/gazebo/models:$GZ_SIM_RESOURCE_PATH"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH"

WORLD_FILE="$REPO_DIR/gazebo/worlds/survivor_arena.sdf"

if [ ! -f "$WORLD_FILE" ]; then
    echo "[ERROR] World file not found at: $WORLD_FILE"
    exit 1
fi

# Link world into PX4 worlds directory if PX4 exists
if [ -d "$HOME/PX4-Autopilot/Tools/simulation/gz/worlds" ]; then
    ln -sf "$WORLD_FILE" "$HOME/PX4-Autopilot/Tools/simulation/gz/worlds/survivor_arena.sdf"
    echo "[✓] Linked survivor_arena.sdf into PX4-Autopilot worlds"
fi

# 3. Check arguments or run default
if [ "$1" == "--px4" ]; then
    echo "======================================================================"
    echo "  Starting PX4 SITL with downward-camera drone in Survivor Arena..."
    echo "  Once the 'pxh>' shell appears, type:"
    echo "    commander takeoff"
    echo "======================================================================"
    cd "$HOME/PX4-Autopilot"
    PX4_GZ_MODEL_POSE="1.5,1.5,0.2" PX4_GZ_WORLD=survivor_arena make px4_sitl gz_x500_mono_cam_down
else
    echo "[✓] Starting Gazebo Harmonic with Survivor Arena (15m x 15m)..."
    echo "    Casualties at: (3.5, 3.0), (8.5, 5.5), (4.0, 11.5), (12.5, 9.0)"
    
    # Run Gazebo Sim with our arena
    gz sim -r "$WORLD_FILE" &
    GZ_PID=$!
    echo "[✓] Gazebo Sim started (PID: $GZ_PID)"

    # Run Team 3 ROS 2 Tagging Bridge Node
    echo "[✓] Starting Survivor Tagging ROS 2 Node..."
    python3 "$REPO_DIR/survivor_tagging/adapters/ros2_bridge.py" &
    NODE_PID=$!

    trap "kill $GZ_PID $NODE_PID 2>/dev/null || true" EXIT
    wait $GZ_PID
fi
