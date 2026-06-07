#!/usr/bin/env python3
"""
compare_cfm_vs_classical.py  —  CFM (Flow Matching) vs Classical Trapezoid comparison.

Shows for each experiment:
  • XY trajectory: CFM plan, Classical plan, Actual turtle execution
  • Speed profile: commanded velocity vs time + ideal trapezoidal speed envelope
  • Distance to goal over time
  • Summary metrics table

Usage:
    python3 compare_cfm_vs_classical.py           # opens window
    python3 compare_cfm_vs_classical.py --save    # saves cfm_vs_classical.png
"""

from __future__ import annotations
import argparse, csv, math, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

LOG_DIR = os.path.expanduser("~/comparison_logs")

EXPERIMENTS = [
    {
        "name":   "exp4_goal_5_2_9s",
        "label":  'Goal (5.0, 2.0)\nv=0.20 m/s  a=0.04 m/s²\n[CFM SUCCESS ✓]',
        "prompt": '"Move the robot to x=2.72, y=3.14\n within 15 s" → re-run variant',
    },
    {
        "name":   "exp2_goal_5_2",
        "label":  'Goal (5.0, 2.0)\nv=0.15 m/s  a=0.03 m/s²\n[CFM FAIL ✗]',
        "prompt": '"Move the robot to x=5, y=2"',
    },
    {
        "name":   "exp1_fwd_v1.2",
        "label":  'Goal (8.94, 7.5) forward\nv=0.12 m/s  a=0.04 m/s²',
        "prompt": '"Move the robot forward, max v=0.12 m/s\n within 15 s"',
    },
    {
        "name":   "exp5_fwd_man_step",
        "label":  'Goal (8.25, 7.5) forward\nv=0.20 m/s  a=0.03 m/s²',
        "prompt": '"Move the robot forward,\n as fast as the man step"',
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────
def read_csv(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rdr = csv.reader(f)
        hdr = next(rdr, None)
        if not hdr:
            return {}
        cols: dict[str, list] = {h: [] for h in hdr}
        for row in rdr:
            for h, v in zip(hdr, row):
                try:
                    cols[h].append(float(v))
                except ValueError:
                    pass
    return {k: np.array(v) for k, v in cols.items()}


def split_by_publish(df: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Split long-format setpoint CSV into list of (N,3) arrays, one per publish."""
    if not df or "publish_time" not in df:
        return []
    t = df["publish_time"]
    if len(t) == 0:
        return []
    out = []
    for pt in np.unique(t):
        mask = t == pt
        out.append(np.column_stack([df["x"][mask], df["y"][mask], df["theta"][mask]]))
    return out


def trapezoid_speed_profile(v_const: float, a: float, distance: float,
                            dt: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Ideal trapezoidal speed vs time for given params."""
    t_accel = v_const / a
    d_accel = 0.5 * a * t_accel ** 2
    if 2 * d_accel >= distance:
        v_peak = math.sqrt(a * distance)
        t_ramp = v_peak / a
        t1, t2, t3 = t_ramp, t_ramp, 2 * t_ramp
    else:
        v_peak = v_const
        d_const = distance - 2 * d_accel
        t1 = t_accel
        t2 = t_accel + d_const / v_const
        t3 = t2 + t_accel

    ts = np.arange(0, t3 + dt, dt)
    vs = np.where(ts <= t1, a * ts,
          np.where(ts <= t2, v_peak,
          np.where(ts <= t3, v_peak - a * (ts - t2), 0.0)))
    return ts, vs


def load_experiment(exp: dict) -> dict:
    d = os.path.join(LOG_DIR, exp["name"])
    pose    = read_csv(os.path.join(d, "pose.csv"))
    cmd     = read_csv(os.path.join(d, "cmd_vel.csv"))
    goals   = read_csv(os.path.join(d, "goals.csv"))
    cfm_df  = read_csv(os.path.join(d, "cfm_setpoints.csv"))
    cls_df  = read_csv(os.path.join(d, "classical_setpoints.csv"))

    gx = float(goals["goal_x"][-1]) if goals and "goal_x" in goals and len(goals["goal_x"]) else None
    gy = float(goals["goal_y"][-1]) if goals and "goal_y" in goals and len(goals["goal_y"]) else None
    v_const = float(goals["v_const"][-1]) if goals and "v_const" in goals and len(goals["v_const"]) else None
    a       = float(goals["a"][-1])       if goals and "a"       in goals and len(goals["a"])       else None

    cfm_publishes = split_by_publish(cfm_df)
    cls_waypoints = np.column_stack([cls_df["x"], cls_df["y"], cls_df["theta"]]) if cls_df and "x" in cls_df else None

    # distance-to-goal time series
    dist_to_goal = None
    if pose and gx is not None:
        dist_to_goal = np.sqrt((pose["x"] - gx) ** 2 + (pose["y"] - gy) ** 2)

    # trapezoid speed envelope (using classical distance as reference)
    trap_t = trap_v = None
    if cls_waypoints is not None and v_const and a:
        dx = cls_waypoints[-1, 0] - cls_waypoints[0, 0]
        dy = cls_waypoints[-1, 1] - cls_waypoints[0, 1]
        dist = math.hypot(dx, dy)
        trap_t, trap_v = trapezoid_speed_profile(v_const, a, dist)

    return dict(
        pose=pose, cmd=cmd, goals=goals,
        cfm_publishes=cfm_publishes, cls_waypoints=cls_waypoints,
        dist_to_goal=dist_to_goal,
        gx=gx, gy=gy, v_const=v_const, a=a,
        trap_t=trap_t, trap_v=trap_v,
        **exp,
    )


# ── main figure ───────────────────────────────────────────────────────────────
def make_plot(save: bool = False):
    data = [load_experiment(e) for e in EXPERIMENTS]
    n = len(data)

    fig = plt.figure(figsize=(6 * n, 15))
    outer = gridspec.GridSpec(4, n, figure=fig,
                              hspace=0.55, wspace=0.38,
                              height_ratios=[2.2, 1.5, 1.5, 0.9])

    fig.suptitle(
        "CFM (Flow Matching) vs Classical Trapezoidal  —  Comparison across 4 Experiments\n"
        "CFM thin = every resample   CFM bold = final plan   Orange = Classical plan   Green = Actual turtle path",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── legend handles (shared) ───────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color="royalblue", lw=0.8, alpha=0.35, label="CFM resamples"),
        Line2D([0], [0], color="royalblue", lw=2.2, label="CFM final plan"),
        Line2D([0], [0], color="darkorange", lw=2.0, linestyle="--", label="Classical (trapezoid) plan"),
        Line2D([0], [0], color="green", lw=1.8, label="Actual turtle path"),
        Line2D([0], [0], color="royalblue", lw=2.0, linestyle="-.", label="CFM commanded speed"),
        Line2D([0], [0], color="darkorange", lw=2.0, linestyle="--", label="Ideal trapezoid speed envelope"),
        Line2D([0], [0], color="green", lw=1.8, label="Actual dist to goal"),
    ]

    for col, d in enumerate(data):
        pose = d["pose"]
        cmd  = d["cmd"]
        gx, gy = d["gx"], d["gy"]

        # ── ROW 0: XY trajectory ─────────────────────────────────────────
        ax_xy = fig.add_subplot(outer[0, col])

        # CFM resamples (thin)
        for i, wp in enumerate(d["cfm_publishes"][:-1]):
            ax_xy.plot(wp[:, 0], wp[:, 1], color="royalblue", lw=0.7,
                       alpha=0.25, label="CFM resamples" if i == 0 else None, zorder=2)
        # CFM final (bold)
        if d["cfm_publishes"]:
            last = d["cfm_publishes"][-1]
            ax_xy.plot(last[:, 0], last[:, 1], color="royalblue", lw=2.2,
                       label=f"CFM ({len(d['cfm_publishes'])} samples)", zorder=3)
        # Classical plan
        if d["cls_waypoints"] is not None:
            cw = d["cls_waypoints"]
            ax_xy.plot(cw[:, 0], cw[:, 1], color="darkorange", lw=2.0,
                       linestyle="--", label="Classical (trapezoid)", zorder=2)
        # Actual turtle path
        if pose and "x" in pose:
            ax_xy.plot(pose["x"], pose["y"], color="green", lw=1.6,
                       label="Actual turtle path", zorder=4)
            ax_xy.scatter(pose["x"][0], pose["y"][0], color="lime", s=80, zorder=6)
            ax_xy.scatter(pose["x"][-1], pose["y"][-1], color="darkgreen", s=80, zorder=6)
        # Goal
        if gx is not None:
            ax_xy.scatter(gx, gy, color="red", s=250, marker="*", zorder=7, label="Goal")

        ax_xy.set_title(d["label"], fontsize=9, fontweight="bold", pad=6)
        ax_xy.set_xlabel("X (m)", fontsize=8)
        ax_xy.set_ylabel("Y (m)", fontsize=8)
        ax_xy.legend(fontsize=6.5, loc="best", framealpha=0.8)
        ax_xy.grid(True, alpha=0.3)
        ax_xy.set_aspect("equal", adjustable="datalim")

        # ── ROW 1: Speed profile ─────────────────────────────────────────
        ax_v = fig.add_subplot(outer[1, col])

        if cmd and "time" in cmd:
            t0_cmd = cmd["time"][0]  # cmd starts after goal arrives
            ax_v.plot(cmd["time"] - t0_cmd, cmd["linear_x"],
                      color="royalblue", lw=1.5, alpha=0.85,
                      label="CFM commanded speed")

        if d["trap_t"] is not None:
            ax_v.plot(d["trap_t"], d["trap_v"],
                      color="darkorange", lw=2.2, linestyle="--",
                      label="Ideal trapezoid envelope")

        if d["v_const"]:
            ax_v.axhline(d["v_const"], color="gray", lw=0.8, linestyle=":",
                         label=f"v_const={d['v_const']:.3f}")

        ax_v.set_xlabel("Time since goal (s)", fontsize=8)
        ax_v.set_ylabel("Linear vel (m/s)", fontsize=8)
        ax_v.set_title("Speed Profile", fontsize=9, fontweight="bold")
        ax_v.legend(fontsize=6.5, loc="upper right", framealpha=0.8)
        ax_v.grid(True, alpha=0.3)
        if d["v_const"]:
            ax_v.set_ylim(-0.02, d["v_const"] * 1.6)

        # ── ROW 2: Distance to goal ───────────────────────────────────────
        ax_d = fig.add_subplot(outer[2, col])

        if d["dist_to_goal"] is not None and pose and "time" in pose:
            # find when goal was received (first cmd_vel timestamp)
            t0_goal = float(cmd["time"][0]) if cmd and "time" in cmd else 0.0
            t_rel = pose["time"] - t0_goal
            ax_d.plot(t_rel, d["dist_to_goal"], color="green", lw=1.5,
                      label="Dist to goal (actual)")

        # classical expected duration
        if d["cls_waypoints"] is not None and d["trap_t"] is not None:
            t_cls = d["trap_t"][-1]
            ax_d.axvline(t_cls, color="darkorange", lw=1.5, linestyle="--",
                         label=f"Classical done at {t_cls:.1f} s")

        ax_d.axhline(0.05, color="gray", lw=0.8, linestyle=":",
                     label="Tolerance 0.05 m")

        final_dist = d["dist_to_goal"][-1] if d["dist_to_goal"] is not None else float("nan")
        ax_d.set_xlabel("Time since goal (s)", fontsize=8)
        ax_d.set_ylabel("Distance to goal (m)", fontsize=8)
        ax_d.set_title(f"Distance to Goal  (final: {final_dist:.3f} m)", fontsize=9, fontweight="bold")
        ax_d.legend(fontsize=6.5, loc="upper right", framealpha=0.8)
        ax_d.grid(True, alpha=0.3)
        ax_d.set_ylim(bottom=0)

        # ── ROW 3: Metrics summary box ────────────────────────────────────
        ax_m = fig.add_subplot(outer[3, col])
        ax_m.axis("off")

        n_cfm = len(d["cfm_publishes"])
        t_total = pose["time"][-1] if pose and "time" in pose else float("nan")
        cls_duration = d["trap_t"][-1] if d["trap_t"] is not None else float("nan")
        final_dist_str = f"{final_dist:.3f}" if not math.isnan(final_dist) else "N/A"
        reached = "YES ✓" if (not math.isnan(final_dist) and final_dist < 0.10) else "NO ✗"
        max_cmd = cmd["linear_x"].max() if cmd and "linear_x" in cmd else float("nan")

        rows = [
            ["", "CFM", "Classical"],
            ["Plan type", "Flow Matching", "Trapezoidal"],
            ["Resamples", str(n_cfm), "1 (analytical)"],
            ["Plan duration", "50 wpts × 40ms", f"{cls_duration:.1f} s"],
            ["Actual total time", f"{t_total:.1f} s", "(not executed)"],
            ["Final dist to goal", final_dist_str + " m", "0.000 m (ideal)"],
            ["Reached goal (<0.1m)", reached, "YES (by design)"],
            ["Max cmd speed", f"{max_cmd:.3f} m/s", f"{d['v_const']:.3f} m/s"],
        ]

        tbl = ax_m.table(
            cellText=rows[1:],
            colLabels=rows[0],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        # header style
        for j in range(3):
            tbl[0, j].set_facecolor("#2c4a7c")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        # CFM col
        for i in range(1, len(rows)):
            tbl[i, 1].set_facecolor("#ddeeff")
        # Classical col
        for i in range(1, len(rows)):
            tbl[i, 2].set_facecolor("#fff0dd")

    # ── Prompt annotation at top of each column ───────────────────────────
    for col, d in enumerate(data):
        x_pos = (col + 0.5) / n
        fig.text(x_pos, 0.965,
                 f'LLM prompt: {d["prompt"]}',
                 ha="center", va="top", fontsize=7.5,
                 style="italic", color="#333333",
                 transform=fig.transFigure)

    if save:
        out = os.path.join(LOG_DIR, "cfm_vs_classical.png")
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
