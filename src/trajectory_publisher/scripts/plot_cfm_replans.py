#!/usr/bin/env python3
"""
plot_cfm_replans.py
--------------------
Plots every CFM replan trajectory for each run, overlaid on one XY figure.
Shows whether replanning consistently converges toward the goal or scatters.

Usage:
  python3 plot_cfm_replans.py                         # latest run dir
  python3 plot_cfm_replans.py ~/comparison_logs/run_X # specific run
  python3 plot_cfm_replans.py --manifest <csv> --run N  # specific run number from manifest
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # no display needed — saves to PNG
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def latest_run_dir():
    dirs = sorted(glob.glob(os.path.expanduser('~/comparison_logs/run_*')))
    if not dirs:
        sys.exit('No run dirs found in ~/comparison_logs/')
    return dirs[-1]


def plot_run(run_dir: str, tmin: float = None, tmax: float = None):
    cfm_path  = os.path.join(run_dir, 'cfm_setpoints.csv')
    pose_path = os.path.join(run_dir, 'pose.csv')
    cls_path  = os.path.join(run_dir, 'classical_setpoints.csv')
    goal_path = os.path.join(run_dir, 'goals.csv')

    if not os.path.exists(cfm_path):
        sys.exit(f'cfm_setpoints.csv not found in {run_dir}')

    cfm  = pd.read_csv(cfm_path)
    pose = pd.read_csv(pose_path) if os.path.exists(pose_path) else None
    cls  = pd.read_csv(cls_path)  if os.path.exists(cls_path)  else None
    goals = pd.read_csv(goal_path) if os.path.exists(goal_path) else None

    # Optional time window filter
    if tmin is not None:
        cfm  = cfm[cfm['publish_time']  >= tmin]
        if pose is not None: pose = pose[pose['time'] >= tmin]
    if tmax is not None:
        cfm  = cfm[cfm['publish_time']  <= tmax]
        if pose is not None: pose = pose[pose['time'] <= tmax]

    # Each unique publish_time is one plan; resample_idx groups them
    # Re-index resample_idx to be contiguous after filtering
    cfm = cfm.copy()
    cfm['resample_idx'] = pd.factorize(cfm['resample_idx'])[0]
    plans = cfm.groupby('resample_idx')
    n_plans = len(plans)

    # Color each plan from blue (first) → red (last)
    cmap   = cm.get_cmap('coolwarm', n_plans)
    colors = [cmap(i / max(n_plans - 1, 1)) for i in range(n_plans)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'CFM Replan Trajectories\n{os.path.basename(run_dir)}',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.set_title(f'All {n_plans} CFM Plans (blue=first → red=last)')

    # Classical baseline (first published plan only)
    if cls is not None and len(cls):
        first_pub = cls['publish_time'].min()
        cls0 = cls[cls['publish_time'] == first_pub].sort_values('wp_idx')
        ax.plot(cls0['x'].values, cls0['y'].values,
                color='orange', lw=2, ls='--', label='Classical baseline', zorder=3)

    # All CFM plans
    for idx, (rid, grp) in enumerate(plans):
        grp = grp.sort_values('wp_idx')
        xs, ys = grp['x'].values, grp['y'].values
        alpha = 0.4 + 0.6 * (idx / max(n_plans - 1, 1))
        ax.plot(xs, ys, color=colors[idx], lw=1.2,
                alpha=alpha, zorder=2,
                label=f'Plan {int(rid)}' if n_plans <= 10 else None)
        ax.scatter(xs[0],  ys[0],  color=colors[idx], s=20, zorder=4, alpha=alpha)
        ax.scatter(xs[-1], ys[-1], color=colors[idx], s=40, marker='x', zorder=4, alpha=alpha)

    # Actual turtle path
    if pose is not None and len(pose):
        ax.plot(pose['x'].values, pose['y'].values, 'g-', lw=2, label='Turtle actual', zorder=5)
        ax.scatter(pose['x'].iloc[0],  pose['y'].iloc[0],
                   color='green', s=80, zorder=6, label='Start')
        ax.scatter(pose['x'].iloc[-1], pose['y'].iloc[-1],
                   color='red',   s=80, zorder=6, label='End')

    # Goal marker from classical endpoint (most reliable)
    if cls is not None and len(cls):
        first_pub = cls['publish_time'].min()
        first_plan = cls[cls['publish_time'] == first_pub]
        gx, gy = first_plan['x'].iloc[-1], first_plan['y'].iloc[-1]
        ax.scatter(gx, gy, color='gold', s=200, marker='*',
                   zorder=7, label='Goal', edgecolors='black', linewidths=0.8)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    if n_plans <= 10:
        ax.legend(fontsize=7, loc='best')
    else:
        ax.legend(fontsize=7, loc='best',
                  handles=[ax.lines[0]] if ax.lines else [],
                  labels=[f'{n_plans} CFM plans (blue→red)'])

    # ── Right panel: endpoint scatter — where does each plan END? ────────────
    ax2 = axes[1]
    ax2.set_title('CFM Plan Endpoints vs Goal\n(shows consistency of replans)')

    ends_x, ends_y, end_colors = [], [], []
    for idx, (rid, grp) in enumerate(plans):
        ends_x.append(grp['x'].iloc[-1])
        ends_y.append(grp['y'].iloc[-1])
        end_colors.append(colors[idx])

    sc = ax2.scatter(ends_x, ends_y, c=list(range(n_plans)), cmap='coolwarm',
                     s=80, zorder=5, edgecolors='k', linewidths=0.5,
                     label='Plan endpoint')
    plt.colorbar(sc, ax=ax2, label='Replan index (0=first)')

    if cls is not None and len(cls):
        ax2.scatter(gx, gy, color='gold', s=300, marker='*',
                    zorder=7, label='Goal', edgecolors='black', linewidths=0.8)

    # Start point
    if pose is not None and len(pose):
        ax2.scatter(pose['x'].iloc[0], pose['y'].iloc[0],
                    color='green', s=120, marker='D', zorder=6, label='Start')

    # Distance from each endpoint to goal
    if cls is not None and len(cls):
        dists = [np.hypot(ex - gx, ey - gy) for ex, ey in zip(ends_x, ends_y)]
        mean_d = np.mean(dists)
        std_d  = np.std(dists)
        ax2.set_title(f'CFM Endpoints  |  dist to goal: mean={mean_d:.3f}m  std={std_d:.3f}m')

    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()

    tag = f'_t{tmin:.0f}-{tmax:.0f}' if (tmin or tmax) else ''
    out = os.path.join(run_dir, f'cfm_replans{tag}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', nargs='?', default=None,
                    help='Path to run directory (default: latest)')
    ap.add_argument('--tmin', type=float, default=None,
                    help='Only show plans with publish_time >= tmin (s)')
    ap.add_argument('--tmax', type=float, default=None,
                    help='Only show plans with publish_time <= tmax (s)')
    args = ap.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    print(f'Plotting: {run_dir}')
    plot_run(run_dir, tmin=args.tmin, tmax=args.tmax)


if __name__ == '__main__':
    main()
