#!/bin/bash
# plot_latest.sh — plot the most recent recorded run after all experiments finish.
#
# Usage:
#   ./plot_latest.sh          # opens window
#   ./plot_latest.sh --save   # saves plot.png inside the run dir

WS="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/comparison_logs"

# Find the latest run_* directory
LATEST=$(ls -dt "$LOG_DIR"/run_*/ 2>/dev/null | head -1)

if [ -z "$LATEST" ]; then
    echo "No run directories found in $LOG_DIR"
    exit 1
fi

echo "Plotting: $LATEST"
python3 "$WS/src/trajectory_publisher/scripts/plot_run.py" "$LATEST" "$@"
