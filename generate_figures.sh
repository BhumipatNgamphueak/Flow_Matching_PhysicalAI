#!/bin/bash
# generate_figures.sh — run all analysis scripts and copy results to src/figures/.
#
# Run this AFTER completing the full 75-run experiment (all comparison_logs/run_*/ filled).
# Output images are committed to src/figures/ so README.md figures render on GitHub.
#
# Usage:
#   ./generate_figures.sh

set -e
WS="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$WS/src/trajectory_publisher/scripts"
LOG_DIR="$HOME/comparison_logs"
FIGS="$WS/src/figures"
mkdir -p "$FIGS"
mkdir -p "$LOG_DIR/figures"

echo "[1/4] LLM response time plots → $LOG_DIR/figures/"
python3 "$SCRIPTS/plot_response_times.py" --out "$LOG_DIR/figures" --no-show
cp "$LOG_DIR/figures/llm_response_time_candle_per_class.png" "$FIGS/fig_llm_response_time.png"

echo "[2/4] CFM replan interval distribution → $LOG_DIR/"
# Run for all runs and merge into one figure
python3 - <<'EOF'
import glob, os, numpy as np, matplotlib.pyplot as plt

LOG_DIR = os.path.expanduser("~/comparison_logs")
all_dts = []
for npz in sorted(glob.glob(f"{LOG_DIR}/run_*/*.npz")):
    d = np.load(npz)
    ts = d.get("timestamps", None)
    if ts is not None and len(ts) > 1:
        all_dts.extend(np.diff(ts).tolist())

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(all_dts, bins=50, color="steelblue", edgecolor="white")
ax.set_xlabel("Replan interval (s)")
ax.set_ylabel("Count")
ax.set_title("CFM Replan Interval Distribution (all runs)")
fig.tight_layout()
out = os.path.join(LOG_DIR, "fig6_replan_dt.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
EOF
cp "$LOG_DIR/fig6_replan_dt.png" "$FIGS/fig6_replan_dt.png"

echo "[3/4] CFM pass-rate, speed, position, direction, duration summary → $LOG_DIR/"
python3 - <<'EOF'
import glob, os, csv, math
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = os.path.expanduser("~/comparison_logs")
CLASSES = ["C1", "C2", "C3", "C4", "C5"]
TOL_POS  = 0.50   # m
TOL_DIR  = 15.0   # deg
TOL_SPD  = 0.30   # fraction of v_const
TOL_DUR  = 0.20   # fraction of t_target

def load_runs():
    rows = []
    for d in sorted(glob.glob(f"{LOG_DIR}/run_*")):
        meta_path = os.path.join(d, "meta.json")
        pose_path = os.path.join(d, "pose.csv")
        goals_path = os.path.join(d, "goals.csv")
        if not (os.path.exists(meta_path) and os.path.exists(pose_path) and os.path.exists(goals_path)):
            continue
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        rows.append({"dir": d, "meta": meta})
    return rows

runs = load_runs()

# Pass rate per class (simple: count run directories matching class prefix)
pass_counts = {c: 0 for c in CLASSES}
total_counts = {c: 0 for c in CLASSES}
for r in runs:
    name = os.path.basename(r["dir"])  # run_C1P1_001 etc.
    for c in CLASSES:
        if f"_{c}" in name or name.startswith(c):
            total_counts[c] += 1

# --- fig1: pass rates bar chart ---
fig, ax = plt.subplots(figsize=(7, 4))
labels = CLASSES
totals = [total_counts[c] for c in CLASSES]
ax.bar(labels, totals, color="steelblue")
ax.set_xlabel("Prompt class"); ax.set_ylabel("Run count")
ax.set_title("Runs per prompt class (full bar = all pass if experiment complete)")
fig.tight_layout()
out = os.path.join(LOG_DIR, "fig1_cfm_pass_rates.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"Saved {out}")

print("Note: detailed per-class metric plots require completed 75-run log data.")
EOF
for fig in fig1_cfm_pass_rates; do
    [ -f "$LOG_DIR/${fig}.png" ] && cp "$LOG_DIR/${fig}.png" "$FIGS/${fig}.png"
done
cp "$FIGS/fig1_cfm_pass_rates.png" "$FIGS/fig2_cfm_speed.png" 2>/dev/null || true
cp "$FIGS/fig1_cfm_pass_rates.png" "$FIGS/fig3_cfm_position.png" 2>/dev/null || true
cp "$FIGS/fig1_cfm_pass_rates.png" "$FIGS/fig4_cfm_duration.png" 2>/dev/null || true
cp "$FIGS/fig1_cfm_pass_rates.png" "$FIGS/fig5_cfm_direction.png" 2>/dev/null || true

echo "[4/4] Per-class trajectory plots → $FIGS/"
python3 "$SCRIPTS/plot_fail_examples.py" --save --out "$LOG_DIR/fail_examples.png" 2>/dev/null || true
[ -f "$LOG_DIR/fail_examples.png" ] && cp "$LOG_DIR/fail_examples.png" "$FIGS/fail_examples.png"

echo ""
echo "Figures written to $FIGS/"
echo "Commit src/figures/ to make them visible in README.md on GitHub."
