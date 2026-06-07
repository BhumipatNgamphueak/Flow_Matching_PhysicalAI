#!/usr/bin/env python3
"""
plot_fail_examples.py — 2 fail-case examples per prompt class vs groundtruth.

Creates a 5-column × 2-row figure (10 panels total).
Each panel shows one run: actual turtle path, last CFM plan, classical baseline, goal.

Usage:
    python3 plot_fail_examples.py                        # auto-find latest run dir
    python3 plot_fail_examples.py --run-dir ~/comparison_logs/run_20260523-214156
    python3 plot_fail_examples.py --manifest ~/comparison_logs/test_manifest_*.csv
    python3 plot_fail_examples.py --save                 # write fail_examples.png
    python3 plot_fail_examples.py --min-error 0.5        # override threshold (default 0.3m)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import glob
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')   # will switch to interactive below if --save not set
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LOG_DIR = os.path.expanduser('~/comparison_logs')

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_csv_arrays(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header:
            return {}
        cols: dict[str, list] = {h.strip(): [] for h in header}
        for row in rdr:
            for h, v in zip(header, row):
                h = h.strip()
                try:
                    cols[h].append(float(v))
                except (ValueError, TypeError):
                    cols[h].append(float('nan'))
    return {k: np.array(v) for k, v in cols.items()}


def _find_latest_run_dir() -> str:
    dirs = sorted(glob.glob(os.path.join(LOG_DIR, 'run_*/')))
    if not dirs:
        raise FileNotFoundError(f'No run_* dirs in {LOG_DIR}')
    return dirs[-1].rstrip('/')


def _find_latest_manifest() -> str:
    files = sorted(glob.glob(os.path.join(LOG_DIR, 'test_manifest_*.csv')))
    if not files:
        raise FileNotFoundError(f'No test_manifest_*.csv in {LOG_DIR}')
    return files[-1]


# ── data loading ─────────────────────────────────────────────────────────────

def load_session(run_dir: str, manifest_path: str, min_error: float):
    """
    Returns a list of dicts, one per run, with all data needed for plotting.
    Each dict has: run_num, cls, prompt_num, repeat, prompt_text, error,
                   pose (N,3)[time,x,y], cfm (M,2)[x,y], classical (K,2)[x,y],
                   goal_world (2,)[gx,gy]
    """
    import json

    # Session start from meta.json
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        session_start = datetime.fromisoformat(meta['start_time'])
    else:
        session_start = datetime.fromisoformat('2026-05-23T21:41:56')

    # Manifest
    with open(manifest_path, newline='') as f:
        manifest = list(csv.DictReader(f))
    for r in manifest:
        r['t_start'] = (datetime.fromisoformat(r['timestamp']) - session_start).total_seconds()
    for i, r in enumerate(manifest):
        r['t_end'] = manifest[i + 1]['t_start'] if i + 1 < len(manifest) else r['t_start'] + 60.0

    # Load big CSVs as numpy arrays for fast slicing
    print('Loading pose.csv...', end=' ', flush=True)
    pose_d = _load_csv_arrays(os.path.join(run_dir, 'pose.csv'))
    print('done')

    print('Loading cfm_setpoints.csv...', end=' ', flush=True)
    cfm_d = _load_csv_arrays(os.path.join(run_dir, 'cfm_setpoints.csv'))
    print('done')

    print('Loading classical_setpoints.csv...', end=' ', flush=True)
    cls_d = _load_csv_arrays(os.path.join(run_dir, 'classical_setpoints.csv'))
    print('done')

    print('Loading goals.csv...', end=' ', flush=True)
    goals_d = _load_csv_arrays(os.path.join(run_dir, 'goals.csv'))
    print('done')

    pose_t  = pose_d.get('time', np.array([]))
    pose_x  = pose_d.get('x',    np.array([]))
    pose_y  = pose_d.get('y',    np.array([]))

    cfm_pt  = cfm_d.get('publish_time', np.array([]))
    cfm_x   = cfm_d.get('x',           np.array([]))
    cfm_y   = cfm_d.get('y',           np.array([]))

    cls_pt  = cls_d.get('publish_time', np.array([]))
    cls_x   = cls_d.get('x',           np.array([]))
    cls_y   = cls_d.get('y',           np.array([]))

    goals_t  = goals_d.get('time',          np.array([]))
    goals_gxw = goals_d.get('goal_x_world', np.array([]))
    goals_gyw = goals_d.get('goal_y_world', np.array([]))

    runs = []
    for r in manifest:
        t0, t1 = r['t_start'], r['t_end']

        # Pose segment
        pmask = (pose_t >= t0) & (pose_t < t1)
        if not pmask.any():
            continue
        px, py = pose_x[pmask], pose_y[pmask]

        # Goal: last goal event in window
        gmask = (goals_t >= t0) & (goals_t < t1)
        gx_vals = goals_gxw[gmask]
        gy_vals = goals_gyw[gmask]
        valid = ~(np.isnan(gx_vals) | np.isnan(gy_vals))
        if valid.any():
            goal_world = np.array([gx_vals[valid][-1], gy_vals[valid][-1]])
        else:
            goal_world = np.array([float('nan'), float('nan')])

        # Final error
        if np.isnan(goal_world).any():
            error = float('nan')
        else:
            error = float(np.sqrt((px[-1] - goal_world[0])**2 + (py[-1] - goal_world[1])**2))

        # Last CFM publish in window
        cfm_mask = (cfm_pt >= t0) & (cfm_pt < t1)
        if cfm_mask.any():
            last_cfm_t = cfm_pt[cfm_mask].max()
            fm = cfm_mask & (cfm_pt == last_cfm_t)
            cfm_xy = np.stack([cfm_x[fm], cfm_y[fm]], axis=1)
        else:
            cfm_xy = np.empty((0, 2))

        # Last classical publish in window
        cls_mask = (cls_pt >= t0) & (cls_pt < t1)
        if cls_mask.any():
            last_cls_t = cls_pt[cls_mask].max()
            cm = cls_mask & (cls_pt == last_cls_t)
            classical_xy = np.stack([cls_x[cm], cls_y[cm]], axis=1)
        else:
            classical_xy = np.empty((0, 2))

        runs.append({
            'run_num':     int(r['run_num']),
            'cls':         int(r['class']),
            'prompt_num':  int(r['prompt_num']),
            'repeat':      int(r['repeat']),
            'prompt_text': r['prompt_text'],
            'error':       error,
            'pose_xy':     np.stack([px, py], axis=1),
            'cfm_xy':      cfm_xy,
            'classical_xy': classical_xy,
            'goal_world':  goal_world,
        })

    return runs


# ── example selection ─────────────────────────────────────────────────────────

def pick_examples(runs: list[dict], n_per_class: int = 2,
                  min_error: float = 0.3) -> dict[int, list[dict]]:
    """
    For each class pick n_per_class examples.
    Strategy:
      - Prefer runs with largest error (most dramatic fail).
      - Fall back to any run if fewer than n available above threshold.
    Returns dict: cls → list of run dicts (length n_per_class).
    """
    by_class: dict[int, list[dict]] = {}
    for r in runs:
        by_class.setdefault(r['cls'], []).append(r)

    selected: dict[int, list[dict]] = {}
    for cls in sorted(by_class.keys()):
        pool = sorted(by_class[cls], key=lambda x: -x['error']
                      if not np.isnan(x['error']) else 0)
        fails = [r for r in pool if not np.isnan(r['error']) and r['error'] >= min_error]
        chosen = fails[:n_per_class]
        # pad with best-available if not enough fails
        if len(chosen) < n_per_class:
            extras = [r for r in pool if r not in chosen]
            chosen += extras[:n_per_class - len(chosen)]
        selected[cls] = chosen[:n_per_class]
    return selected


# ── plotting ─────────────────────────────────────────────────────────────────

CLASS_LABELS = {
    1: 'Class 1\nDirection only',
    2: 'Class 2\nGoal position',
    3: 'Class 3\nDirection + time',
    4: 'Class 4\nGoal + time',
    5: 'Class 5\nInformal language',
}

def _abbrev(text: str, n: int = 42) -> str:
    return text if len(text) <= n else text[:n - 1] + '…'


def _to_robot_frame(xy: np.ndarray, ox: float, oy: float) -> np.ndarray:
    """Translate (N,2) world-frame coordinates to robot frame (origin at start)."""
    out = xy.copy()
    out[:, 0] -= ox
    out[:, 1] -= oy
    return out


def make_fail_plot(selected: dict[int, list[dict]], save_path: Optional[str] = None):
    n_classes = len(selected)
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_classes,
                             figsize=(4.5 * n_classes, 4.5 * n_rows),
                             squeeze=False)

    fig.suptitle('CFM vs Classical Baseline — 2 Fail Examples per Prompt Class\n'
                 '(robot frame: start = origin)',
                 fontsize=13, fontweight='bold', y=1.01)

    # Column headers
    for col, cls in enumerate(sorted(selected.keys())):
        axes[0, col].set_title(CLASS_LABELS.get(cls, f'Class {cls}'),
                               fontsize=10, fontweight='bold', pad=8)

    for col, cls in enumerate(sorted(selected.keys())):
        examples = selected[cls]
        for row, run in enumerate(examples):
            ax = axes[row, col]

            pose = run['pose_xy']
            cfm  = run['cfm_xy']
            cls_ = run['classical_xy']

            # Origin = start of classical (= pose at context-receipt time,
            # which is where the robot was when the goal was issued).
            # Fall back to first pose point if classical is empty.
            if len(cls_) > 0:
                ox, oy = cls_[0, 0], cls_[0, 1]
            else:
                ox, oy = pose[0, 0], pose[0, 1]

            # Goal = classical endpoint (direct start → goal, most reliable)
            if len(cls_) > 1:
                goal_local = np.array([cls_[-1, 0] - ox, cls_[-1, 1] - oy])
            else:
                goal_local = np.array([float('nan'), float('nan')])

            # Transform to robot frame
            pose_r = _to_robot_frame(pose, ox, oy)
            cfm_r  = _to_robot_frame(cfm,  ox, oy) if len(cfm) > 0 else cfm
            cls_r  = _to_robot_frame(cls_,  ox, oy) if len(cls_) > 0 else cls_

            # Recompute error in robot frame (same distance, just shifted)
            err_str = f'err={run["error"]:.2f}m' if not np.isnan(run['error']) else 'err=?'

            # Actual path
            if len(pose_r) > 1:
                ax.plot(pose_r[:, 0], pose_r[:, 1],
                        color='#2ca02c', linewidth=1.8, zorder=4, label='Actual')
            ax.scatter(0, 0, color='limegreen', s=70, zorder=6, label='Start')
            ax.scatter(pose_r[-1, 0], pose_r[-1, 1],
                       color='red', marker='X', s=90, zorder=6, label='End')

            # CFM last setpoint
            if len(cfm_r) > 1:
                ax.plot(cfm_r[:, 0], cfm_r[:, 1],
                        color='#1f77b4', linestyle='--', linewidth=1.5,
                        zorder=3, alpha=0.85, label='CFM plan')

            # Classical baseline (always from origin → goal)
            if len(cls_r) > 1:
                ax.plot(cls_r[:, 0], cls_r[:, 1],
                        color='#ff7f0e', linestyle=':', linewidth=2.0,
                        zorder=2, alpha=0.90, label='Classical')

            # Goal marker — endpoint of classical baseline
            if not np.isnan(goal_local).any():
                ax.scatter(goal_local[0], goal_local[1],
                           marker='*', color='black', s=220, zorder=7, label='Goal')

            ax.set_title(f'{_abbrev(run["prompt_text"], 40)}\n'
                         f'Run {run["run_num"]} rep{run["repeat"]}  {err_str}',
                         fontsize=7.5, pad=4)

            ax.set_xlabel('X body (m)', fontsize=8)
            ax.set_ylabel('Y body (m)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.35)
            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
            ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)
            ax.set_aspect('equal', adjustable='datalim')

            # Only show legend on first panel
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=7, framealpha=0.85)

    # Shared legend at bottom
    legend_elements = [
        mpatches.Patch(color='#2ca02c',  label='Actual turtle path'),
        mpatches.Patch(color='#1f77b4',  label='CFM plan (last resample)'),
        mpatches.Patch(color='#ff7f0e',  label='Classical baseline (start → goal)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                   markersize=12, label='Goal (= classical endpoint)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                   markersize=8, label='Start (robot frame origin)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(pad=2.0)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    else:
        matplotlib.use('TkAgg')
        plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir',   default=None,
                   help='Path to run directory (default: latest run_* dir)')
    p.add_argument('--manifest',  default=None,
                   help='Path to test_manifest_*.csv (default: latest)')
    p.add_argument('--save',      action='store_true',
                   help='Save to <log_dir>/fail_examples.png instead of opening window')
    p.add_argument('--out',       default=None,
                   help='Output path (overrides --save default location)')
    p.add_argument('--min-error', type=float, default=0.3,
                   help='Minimum final error (m) to count as a fail (default 0.3)')
    p.add_argument('--n',         type=int, default=2,
                   help='Examples per class (default 2)')
    args = p.parse_args()

    run_dir  = args.run_dir  or _find_latest_run_dir()
    manifest = args.manifest or _find_latest_manifest()

    print(f'Run dir : {run_dir}')
    print(f'Manifest: {manifest}')

    runs     = load_session(run_dir, manifest, args.min_error)
    selected = pick_examples(runs, n_per_class=args.n, min_error=args.min_error)

    print('\nSelected runs:')
    for cls in sorted(selected.keys()):
        for r in selected[cls]:
            print(f'  Class {cls}  run {r["run_num"]:>2}  repeat {r["repeat"]}'
                  f'  err={r["error"]:.3f}m  "{_abbrev(r["prompt_text"], 55)}"')

    if args.save or args.out:
        out = args.out or os.path.join(run_dir, 'fail_examples.png')
        make_fail_plot(selected, save_path=out)
    else:
        import matplotlib
        matplotlib.use('TkAgg')
        make_fail_plot(selected, save_path=None)


if __name__ == '__main__':
    main()
