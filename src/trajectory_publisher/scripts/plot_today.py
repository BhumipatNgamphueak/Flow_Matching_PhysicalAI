#!/usr/bin/env python3
"""
plot_today.py — plot today's (2026-05-17) CFM vs classical comparison tests,
annotated with the AI prompt that produced each condition.

Usage:
    python3 plot_today.py          # opens window
    python3 plot_today.py --save   # saves today_results.png
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TODAY = "20260517"
LOG_DIR = os.path.expanduser("~/comparison_logs")

# Map (goal_x, goal_y, v_const, a) → actual user prompt sent to the LLM
PROMPT_MAP = {
    (9.25, 7.50, 0.200, 0.040): '"Move 7 book-lengths to the right,\nas fast as the man step"',
    (8.25, 7.50, 0.200, 0.030): '"Move the robot forward,\nas fast as the man step"',
    (2.72, 3.14, 0.150, 0.030): '"Move the robot to x=2.72, y=3.14, z=0.0 m,\nwithin the time of 15 seconds."',
    (8.94, 7.50, 0.120, 0.040): '"Move the robot forward, with the maximum\nvelocity of 0.12 m/s, within the time of 15 seconds"',
}


def _read_csv(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header:
            return {}
        cols: dict[str, list] = {h: [] for h in header}
        for row in rdr:
            for h, v in zip(header, row):
                try:
                    cols[h].append(float(v))
                except ValueError:
                    cols[h].append(v)
    return {k: np.array(v) for k, v in cols.items()}


def _key(d: np.lib.npyio.NpzFile):
    gx, gy = float(d["goal_world"][0]), float(d["goal_world"][1])
    v = float(d["v_const"])
    a = float(d["a"])
    return (round(gx, 2), round(gy, 2), round(v, 3), round(a, 3))


def load_today_npz():
    """Return dict: key → list of NpzFile (only today's files)."""
    groups: dict = {}
    files = sorted(glob.glob(os.path.join(LOG_DIR, f"{TODAY}*_pid.npz")))
    for f in files:
        d = np.load(f, allow_pickle=True)
        k = _key(d)
        groups.setdefault(k, []).append(d)
    return groups


def load_exp1_actual():
    """Load actual turtle pose from exp1_fwd_v1.2 (today's run with full plotter data)."""
    run_dir = os.path.join(LOG_DIR, "exp1_fwd_v1.2")
    pose = _read_csv(os.path.join(run_dir, "pose.csv"))
    return pose if pose and "x" in pose else None


def make_plot(save: bool = False):
    groups = load_today_npz()
    if not groups:
        print(f"No {TODAY} npz files found in {LOG_DIR}")
        sys.exit(1)

    actual_pose = load_exp1_actual()

    n = len(groups)
    fig = plt.figure(figsize=(7 * n, 6))
    gs = gridspec.GridSpec(1, n, figure=fig, wspace=0.35)

    fig.suptitle(
        "Today's LLM-to-Robot Tests (2026-05-17)  ─  CFM vs Classical Trajectories\n"
        "Blue thin = CFM samples   Blue bold = last CFM   Orange dashed = Classical   Green = Actual turtle path",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for col, (key, runs) in enumerate(sorted(groups.items())):
        gx, gy, v, a = key
        prompt = PROMPT_MAP.get(key, f'goal=({gx}, {gy})  v={v}  a={a}')

        ax = fig.add_subplot(gs[0, col])

        # ── plot CFM samples ─────────────────────────────────────────────
        for i, d in enumerate(runs[:-1]):
            wp = d["cfm_waypoints"]
            ax.plot(wp[:, 0], wp[:, 1], "b-", linewidth=0.7, alpha=0.25,
                    label="CFM samples" if i == 0 else None, zorder=2)

        # last CFM bold
        last_wp = runs[-1]["cfm_waypoints"]
        ax.plot(last_wp[:, 0], last_wp[:, 1], "b-", linewidth=2.0,
                label=f"CFM last (of {len(runs)})", zorder=3)

        # ── classical (same for all runs; use first) ─────────────────────
        classical = runs[0]["classical_waypoints"]
        ax.plot(classical[:, 0], classical[:, 1],
                color="orange", linestyle="--", linewidth=2.0,
                label="Classical", zorder=2)

        # ── actual turtle path (only for exp1 condition) ─────────────────
        exp1_key = (8.94, 7.50, 0.120, 0.040)
        if key == exp1_key and actual_pose:
            ax.plot(actual_pose["x"], actual_pose["y"],
                    "g-", linewidth=1.8, label="Actual turtle path", zorder=4)
            ax.scatter(actual_pose["x"][0], actual_pose["y"][0],
                       color="lime", zorder=5, s=80)
            ax.scatter(actual_pose["x"][-1], actual_pose["y"][-1],
                       color="darkgreen", zorder=5, s=80)

        # ── start / goal markers ─────────────────────────────────────────
        sx, sy, _ = runs[0]["start_pose"]
        ax.scatter(sx, sy, color="green", zorder=6, s=100, marker="o", label="Start")
        ax.scatter(gx, gy, color="red", zorder=6, s=200, marker="*", label="Goal")

        # ── formatting ───────────────────────────────────────────────────
        ax.set_title(prompt, fontsize=9, fontweight="bold", pad=8)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.legend(loc="best", fontsize=7, framealpha=0.85)
        ax.grid(True, alpha=0.35)
        ax.set_aspect("equal", adjustable="datalim")

        info = (
            f"n_samples={len(runs)}\n"
            f"v_const={v:.3f} m/s\n"
            f"a={a:.3f} m/s²\n"
            f"goal=({gx:.2f}, {gy:.2f})"
        )
        ax.text(0.02, 0.02, info, transform=ax.transAxes,
                fontsize=7.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

    fig.tight_layout()

    if save:
        out = os.path.join(LOG_DIR, "today_results.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    make_plot(save=args.save)


if __name__ == "__main__":
    main()
