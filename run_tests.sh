#!/bin/bash
# run_tests.sh — automated test runner
# Runs every prompt 5 times, resets turtle between each run.
#
# Usage:
#   ./run_tests.sh              # all 15 prompts × 5 repeats = 75 runs (~51 min)
#   ./run_tests.sh --class 3    # only class 3 prompts × 5 repeats
#   ./run_tests.sh --wait 40    # override per-run wait time (default 35s)
#
# Requires: Command 1 already launched (turtlesim + controller + LLM node)
# Plotter is auto-started by this script on the first run.

# NO set -e — a single failed service call must not abort 75 runs.
# Each call checks its own exit code and prints a clear warning instead.

WS="$(cd "$(dirname "$0")" && pwd)"
source "$WS/install/setup.bash"

# ── config ────────────────────────────────────────────────────────────────────
REPEATS=5
WAIT_PER_RUN=35      # seconds to wait for robot to complete each run
RESET_WAIT=3         # seconds after turtle teleport before next prompt
ONLY_CLASS=""        # empty = all classes
ONLY_PROMPT=""       # empty = all prompt numbers within class

while [[ $# -gt 0 ]]; do
    case "$1" in
        --class)      ONLY_CLASS="$2";  shift 2 ;;
        --wait)       WAIT_PER_RUN="$2"; shift 2 ;;
        --repeats)    REPEATS="$2";     shift 2 ;;
        --prompt-num) ONLY_PROMPT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── prompt list: "CLASS|PROMPT_TEXT" ─────────────────────────────────────────
declare -a PROMPTS
PROMPTS=(
  "1|Move the robot forward, with the maximum velocity of 0.12 m/s"
  "1|Go to the direction of 7 oclock, using mean speed of about 5 inches / sec"
  "1|Move to relative southeast, as fast as you are approaching the target grapping item"

  "2|Move the robot to the position x=3.14, y=-2.72, z=0.0. m"
  "2|Move yourself with the displacement of 4.2 metre, to the azimuth 300 deg"
  "2|Suppose you see a ball resting at 6 floor-tiles ahead, go to reach it"

  "3|Move the robot forward, with the maximum velocity of 0.12 m/s, within the time of 15 seconds"
  "3|Keep going backward with the possible max speed, until you reach the time of 17 seconds"
  "3|Walk as fast as a dog robot, for a period of 30 seconds"

  "4|Move the robot to x=2.72, y=3.14, z=0.0 m, within the time of 15 seconds."
  "4|Walk to reach the distance of 4 metre at 2 oclock, at the exactime of 30 s"
  "4|Move away 10 feet far to the relative west, taking time not more than quarter a minute"

  "5|Move the robot forward, as fast as the man step"
  "5|Move 7 book-lengths to the right, as fast as the man step"
  "5|You are about 8 floor-tiles away at the right of the wall. Approach the wall gently."
)

# ── manifest log (records class/prompt/repeat for every run) ─────────────────
LOG_DIR="$HOME/comparison_logs"
mkdir -p "$LOG_DIR"
MANIFEST="$LOG_DIR/test_manifest_$(date +%Y%m%d-%H%M%S).csv"
echo "run_num,class,prompt_num,repeat,timestamp,prompt_text" > "$MANIFEST"

# ── helpers ───────────────────────────────────────────────────────────────────
reset_turtle() {
    # turtlesim_plus has no teleport service — reset by remove + respawn
    timeout 5 ros2 service call /remove_turtle turtlesim/srv/Kill \
        "{name: 'turtle1'}" > /dev/null 2>&1 || true
    sleep 0.5
    if ! timeout 5 ros2 service call /spawn_turtle turtlesim/srv/Spawn \
        "{name: 'turtle1', x: 7.5, y: 7.5, theta: 0.0}" > /dev/null 2>&1; then
        echo "  [WARN] Respawn failed — turtle may not be at start position"
    fi
    sleep "$RESET_WAIT"
}

send_prompt() {
    local text="$1"
    local escaped="${text//\"/\\\"}"
    local out
    out=$(timeout 60 ros2 service call /LlmPrompt llm_pack_interface/srv/String \
        "{ prompt: \"$escaped\" }" 2>&1)
    local rc=$?
    if [[ $rc -eq 124 ]]; then
        echo "  [WARN] LLM timeout (>60s) — run skipped"
        return 1
    elif [[ $rc -ne 0 ]]; then
        echo "  [ERROR] Service call failed (rc=$rc): $out"
        return 1
    fi
    local reply
    reply=$(echo "$out" | grep -o "response:.*" | tail -1 || true)
    echo "  LLM: $reply"
    return 0
}

start_plotter_if_needed() {
    if ! ros2 node list 2>/dev/null | grep -q "/turtle_plotter"; then
        echo "[run_tests] Starting plotter..."
        ros2 run trajectory_publisher turtle_plotter.py \
            --ros-args -p live_plot:=false &
        sleep 2
    fi
}

# ── count runs ────────────────────────────────────────────────────────────────
TOTAL=0
_pnum=0
_prev_cls=""
for entry in "${PROMPTS[@]}"; do
    cls="${entry%%|*}"
    [[ -n "$ONLY_CLASS" && "$cls" != "$ONLY_CLASS" ]] && continue
    [[ "$cls" != "$_prev_cls" ]] && _pnum=0
    _prev_cls="$cls"
    _pnum=$(( _pnum + 1 ))
    [[ -n "$ONLY_PROMPT" && "$_pnum" != "$ONLY_PROMPT" ]] && continue
    TOTAL=$(( TOTAL + REPEATS ))
done

EST_MIN=$(( TOTAL * (WAIT_PER_RUN + RESET_WAIT + 3) / 60 ))

echo "════════════════════════════════════════════════════"
echo " run_tests.sh — CFM trajectory benchmark"
echo " Prompts : $(( TOTAL / REPEATS )) × $REPEATS repeats = $TOTAL runs"
echo " Wait/run: ${WAIT_PER_RUN}s   Reset wait: ${RESET_WAIT}s"
echo " Est. time: ~${EST_MIN} min"
echo " Manifest : $MANIFEST"
echo "════════════════════════════════════════════════════"
echo ""
read -p "Press Enter to start, Ctrl-C to cancel..."

start_plotter_if_needed

# ── main loop ─────────────────────────────────────────────────────────────────
run_num=0
prompt_num=0
prev_cls=""

for entry in "${PROMPTS[@]}"; do
    cls="${entry%%|*}"
    prompt_text="${entry#*|}"

    [[ -n "$ONLY_CLASS" && "$cls" != "$ONLY_CLASS" ]] && continue

    # reset prompt_num counter when class changes
    [[ "$cls" != "$prev_cls" ]] && prompt_num=0
    prev_cls="$cls"
    prompt_num=$(( prompt_num + 1 ))

    [[ -n "$ONLY_PROMPT" && "$prompt_num" != "$ONLY_PROMPT" ]] && continue

    for rep in $(seq 1 $REPEATS); do
        run_num=$(( run_num + 1 ))
        ts=$(date +%Y-%m-%dT%H:%M:%S)

        echo ""
        echo "──────────────────────────────────────────────────"
        echo " Run $run_num / $TOTAL  |  Class $cls  Prompt $prompt_num  Repeat $rep/$REPEATS"
        echo " Prompt: \"$prompt_text\""
        echo "──────────────────────────────────────────────────"

        # Log to manifest CSV
        echo "$run_num,$cls,$prompt_num,$rep,$ts,\"$prompt_text\"" >> "$MANIFEST"

        # Reset turtle
        echo "[$(date +%H:%M:%S)] Resetting turtle to (7.5, 7.5, θ=0)..."
        reset_turtle

        # Send prompt — show result, continue on failure (don't abort test)
        echo "[$(date +%H:%M:%S)] Sending prompt..."
        if ! send_prompt "$prompt_text"; then
            echo "  [WARN] Skipping wait — prompt was not delivered"
            echo "$run_num,$cls,$prompt_num,$rep,$ts,FAILED,\"$prompt_text\"" >> "$MANIFEST"
            continue
        fi

        # Wait for robot to complete
        echo "[$(date +%H:%M:%S)] Waiting ${WAIT_PER_RUN}s..."
        sleep "$WAIT_PER_RUN"

        echo "[$(date +%H:%M:%S)] Run $run_num done."
    done
done

echo ""
echo "════════════════════════════════════════════════════"
echo " ALL $TOTAL RUNS COMPLETE"
echo " Manifest  : $MANIFEST"
echo " Logs      : $LOG_DIR/"
echo " Run ./plot_latest.sh to visualise the session"
echo "════════════════════════════════════════════════════"
