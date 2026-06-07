#!/bin/bash
# prompt.sh — send one LLM prompt and auto-start the plotter if not already running.
#
# Usage:
#   ./prompt.sh "Move the robot forward 2 meters"
#   ./prompt.sh "Go to x=3, y=1 as fast as possible"
#
# The plotter starts on the FIRST call (so recording begins when experiments start)
# and keeps running for all subsequent prompts in the same session.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 \"<prompt text>\""
    exit 1
fi

PROMPT="$1"
WS="$(cd "$(dirname "$0")" && pwd)"

# Source ROS2 workspace
source "$WS/install/setup.bash"

# Start plotter if not already running
if ! ros2 node list 2>/dev/null | grep -q "/turtle_plotter"; then
    echo "[prompt.sh] Starting plotter (first prompt of session)..."
    ros2 run trajectory_publisher turtle_plotter.py \
        --ros-args -p live_plot:=false &
    PLOTTER_PID=$!
    echo "[prompt.sh] Plotter PID=$PLOTTER_PID — waiting 2s for it to come up..."
    sleep 2
fi

# Send the prompt (double-quote the YAML value so single quotes in prompts work;
# escape any double quotes that appear inside the prompt text)
ESCAPED="${PROMPT//\"/\\\"}"
echo "[prompt.sh] Sending: \"$ESCAPED\""
ros2 service call /LlmPrompt llm_pack_interface/srv/String \
    "{ prompt: \"$ESCAPED\" }"
