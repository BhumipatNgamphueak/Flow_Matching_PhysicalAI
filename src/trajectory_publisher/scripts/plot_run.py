#!/usr/bin/env python3
"""
plot_run.py — offline plot of a single recorded run.

Usage:
    python3 plot_run.py <run_dir>           # e.g. ~/comparison_logs/run_20260509-220500
    python3 plot_run.py <run_dir> --save    # write plot.png next to the CSVs

Expects the following CSVs in <run_dir>:
    pose.csv              turtle pose time-series
    cmd_vel.csv           cmd_vel time-series
    cfm_setpoints.csv     long-format CFM setpoint publishes
    classical_setpoints.csv  long-format classical setpoint publishes
    goals.csv             (optional) goal events from /traj_context
"""

from __future__ import annotations

import argparse
import os
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt


def _read_csv(path: str) -> dict[str, np.ndarray]:
    """Read a CSV into a dict of float-typed numpy arrays. Returns empty arrays if file missing."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
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


def _last_publish(long_df: dict[str, np.ndarray]) -> np.ndarray | None:
    """Pull the LAST publish from a long-format setpoint CSV → (N, 3) [x, y, theta]."""
    if not long_df or 'publish_time' not in long_df:
        return None
    t = long_df['publish_time']
    if len(t) == 0:
        return None
    last_t = t[-1]
    mask = t == last_t
    return np.stack([long_df['x'][mask],
                     long_df['y'][mask],
                     long_df['theta'][mask]], axis=1)


def _all_publishes(long_df: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Split a long-format setpoint CSV into a list of (N, 3) arrays, one per publish."""
    if not long_df or 'publish_time' not in long_df:
        return []
    t = long_df['publish_time']
    if len(t) == 0:
        return []
    uniq_times = np.unique(t)
    out = []
    for pt in uniq_times:
        mask = t == pt
        out.append(np.stack([long_df['x'][mask],
                             long_df['y'][mask],
                             long_df['theta'][mask]], axis=1))
    return out


def make_plot(run_dir: str, save: bool = False):
    pose      = _read_csv(os.path.join(run_dir, 'pose.csv'))
    cmd       = _read_csv(os.path.join(run_dir, 'cmd_vel.csv'))
    cfm_df    = _read_csv(os.path.join(run_dir, 'cfm_setpoints.csv'))
    classical_df = _read_csv(os.path.join(run_dir, 'classical_setpoints.csv'))
    goals     = _read_csv(os.path.join(run_dir, 'goals.csv'))

    if not pose or 'time' not in pose:
        print(f'No pose.csv data in {run_dir} — nothing to plot.')
        return

    cfm_all       = _all_publishes(cfm_df)
    classical_last = _last_publish(classical_df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    title = f'Run: {os.path.basename(os.path.abspath(run_dir))}'
    # Use world-frame goal if available (new schema has goal_x_world);
    # fall back to goal_x_body (still new schema) then goal_x (old schema pre-refactor)
    if goals and 'goal_x_world' in goals:
        _gx_col, _gy_col = 'goal_x_world', 'goal_y_world'
    elif goals and 'goal_x_body' in goals:
        _gx_col, _gy_col = 'goal_x_body', 'goal_y_body'
    else:
        _gx_col, _gy_col = 'goal_x', 'goal_y'
    if goals and _gx_col in goals and len(goals[_gx_col]) and not np.all(np.isnan(goals[_gx_col].astype(float))):
        gx, gy = goals[_gx_col][-1], goals[_gy_col][-1]
        v = goals['v_const'][-1] if 'v_const' in goals else float('nan')
        a = goals['a'][-1] if 'a' in goals else float('nan')
        title += f'   |   goal=({gx:.2f}, {gy:.2f})   v_const={v:.3f}   a={a:.3f}'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax_pos, ax_vel, ax_xy, ax_theta = (axes[0, 0], axes[0, 1],
                                       axes[1, 0], axes[1, 1])

    # ── [0,0] Position vs time ───────────────────────────────────────────
    ax_pos.plot(pose['time'], pose['x'], 'b-', linewidth=1.5, label='x (turtle)')
    ax_pos.plot(pose['time'], pose['y'], 'r-', linewidth=1.5, label='y (turtle)')
    _valid_goals = (goals and _gx_col in goals and len(goals[_gx_col]) > 0
                    and not np.isnan(float(goals[_gx_col][-1])))
    if _valid_goals:
        ax_pos.axhline(float(goals[_gx_col][-1]), color='blue', linestyle=':',
                       alpha=0.5, linewidth=1.0, label=f'goal x={float(goals[_gx_col][-1]):.2f}')
        ax_pos.axhline(float(goals[_gy_col][-1]), color='red', linestyle=':',
                       alpha=0.5, linewidth=1.0, label=f'goal y={float(goals[_gy_col][-1]):.2f}')
        if goals and len(goals.get('time', [])) > 1:
            for gt in goals['time']:
                ax_pos.axvline(gt, color='gray', linestyle=':', alpha=0.4)
    ax_pos.set_xlabel('Time (s)'); ax_pos.set_ylabel('Position (m)')
    ax_pos.set_title('Position vs Time')
    ax_pos.legend(loc='best', fontsize=8); ax_pos.grid(True, alpha=0.4)

    # ── [0,1] Linear velocity vs time ────────────────────────────────────
    ax_vel.plot(pose['time'], pose['linear_velocity'],
                'b-', linewidth=1.5, label='Actual (pose)')
    if cmd and 'time' in cmd:
        ax_vel.plot(cmd['time'], cmd['linear_x'],
                    'r--', linewidth=1.5, label='Commanded')
    ax_vel.set_xlabel('Time (s)'); ax_vel.set_ylabel('Linear Velocity (m/s)')
    ax_vel.set_title('Linear Velocity vs Time')
    ax_vel.legend(loc='upper right'); ax_vel.grid(True, alpha=0.4)

    # ── [1,0] XY trajectory ──────────────────────────────────────────────
    # Plot every CFM setpoint as a thin trace (resample history) + last one bold.
    for k, cfm_traj in enumerate(cfm_all[:-1]):
        ax_xy.plot(cfm_traj[:, 0], cfm_traj[:, 1],
                   'b-', linewidth=0.8, alpha=0.30,
                   label='CFM history' if k == 0 else None,
                   zorder=2)
    if cfm_all:
        last = cfm_all[-1]
        ax_xy.plot(last[:, 0], last[:, 1],
                   'b--', linewidth=1.8, alpha=0.95,
                   label=f'CFM setpoint (last of {len(cfm_all)})',
                   zorder=3)
    if classical_last is not None:
        ax_xy.plot(classical_last[:, 0], classical_last[:, 1],
                   color='orange', linestyle=':', linewidth=2.0,
                   alpha=0.90, label='Classical setpoint', zorder=2)
    ax_xy.plot(pose['x'], pose['y'], 'g-', linewidth=1.8,
               label='Turtle actual', zorder=4)
    ax_xy.scatter(pose['x'][0], pose['y'][0],
                  color='green', zorder=5, s=80, label='Start')
    ax_xy.scatter(pose['x'][-1], pose['y'][-1],
                  color='red', zorder=5, s=80, label='End')
    if _valid_goals:
        gx_vals = goals[_gx_col].astype(float)
        gy_vals = goals[_gy_col].astype(float)
        mask = ~(np.isnan(gx_vals) | np.isnan(gy_vals))
        if mask.any():
            ax_xy.scatter(gx_vals[mask], gy_vals[mask],
                          marker='*', color='black', s=200, zorder=6,
                          label='Goal(s)')
    ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('XY Trajectory — Setpoints vs Actual')
    ax_xy.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax_xy.grid(True, alpha=0.4)
    ax_xy.set_aspect('equal', adjustable='datalim')

    # ── [1,1] Heading vs time ────────────────────────────────────────────
    ax_theta.plot(pose['time'], pose['theta'],
                  color='purple', linewidth=1.5, label='θ (rad)')
    ax_theta.set_xlabel('Time (s)'); ax_theta.set_ylabel('Theta (rad)')
    ax_theta.set_title('Heading vs Time')
    ax_theta.legend(loc='upper right'); ax_theta.grid(True, alpha=0.4)

    fig.tight_layout(pad=3.0)

    if save:
        out_path = os.path.join(run_dir, 'plot.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved {out_path}')
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', help='Directory containing the CSV files')
    p.add_argument('--save', action='store_true',
                   help='Write plot.png next to CSVs instead of opening a window')
    args = p.parse_args()
    run_dir = os.path.expanduser(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f'Not a directory: {run_dir}', file=sys.stderr)
        sys.exit(1)
    make_plot(run_dir, save=args.save)


if __name__ == '__main__':
    main()
