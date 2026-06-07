#!/usr/bin/env python3
"""
plot_cfm_gen_times.py — CFM trajectory-generation time statistics
(the model's sampling latency, EXCLUDING LLM response time).

Source: the controller logs every CFM sample as
    [INFO] [<epoch>] ... [CFM-3D] Trajectory <label> in <X> ms ...
These are extracted from ~/.ros/log/python3_*.log for a given run window,
then mapped to the prompt that was active at each event time.

Produces:
  cfm_gen_time_candle_all.png       single candle, all CFM samples
  cfm_gen_time_candle_per_class.png candle per prompt class
  llm_vs_cfm_time.png               LLM response vs CFM gen time (log scale)

Usage:
  python3 plot_cfm_gen_times.py            # uses the 2026-05-24 run defaults
"""

from __future__ import annotations
import argparse, csv, glob, os, re, datetime
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

LOG_DIR = os.path.expanduser('~/comparison_logs')

CLASS_DESC = {
    'C1': 'Direction + Speed', 'C2': 'Absolute Position', 'C3': 'Speed + Duration',
    'C4': 'Position + Duration', 'C5': 'Ambiguous',
}
CLASS_COLORS = {
    'C1': '#4C72B0', 'C2': '#55A868', 'C3': '#C44E52', 'C4': '#8172B3', 'C5': '#CCB974',
}

_CFM_PAT = re.compile(
    r'\[INFO\]\s*\[(\d+\.\d+)\].*\[CFM-3D\] Trajectory\s+(\S+)\s+in\s+([\d.]+)\s*ms')


def extract_cfm_events(win_start, win_end):
    events = []
    for f in glob.glob(os.path.expanduser('~/.ros/log/python3_*.log')):
        try:
            for line in open(f, errors='ignore'):
                if 'CFM-3D' not in line:
                    continue
                m = _CFM_PAT.search(line)
                if not m:
                    continue
                ts, label, ms = float(m.group(1)), m.group(2), float(m.group(3))
                if win_start <= ts <= win_end:
                    events.append((ts, label, ms))
        except Exception:
            pass
    events.sort()
    return events


def map_events_to_prompts(events, goals_csv, manifest_csv):
    """Assign each CFM event a (class, prompt) using goal time windows."""
    goals = list(csv.DictReader(open(goals_csv)))
    manifest = list(csv.DictReader(open(manifest_csv)))
    if not events:
        return []
    # launch_start_epoch ≈ first_event_epoch − first_goal_relative_time
    first_goal_rel = float(goals[0]['time'])
    launch_epoch = events[0][0] - first_goal_rel
    # goal boundaries in epoch
    bounds = []
    for i, g in enumerate(goals):
        t0 = launch_epoch + float(g['time'])
        t1 = launch_epoch + float(goals[i + 1]['time']) if i + 1 < len(goals) else float('inf')
        cls = f"C{manifest[i]['class']}" if i < len(manifest) else 'C?'
        pn = f"P{manifest[i]['prompt_num']}" if i < len(manifest) else 'P?'
        bounds.append((t0, t1, cls, pn))
    out = []
    for ts, label, ms in events:
        cls, pn = 'C?', 'P?'
        for t0, t1, c, p in bounds:
            if t0 <= ts < t1:
                cls, pn = c, p
                break
        out.append({'epoch': ts, 'label': label, 'gen_ms': ms,
                    'class': cls, 'prompt_num': pn})
    return out


def _draw_candle(ax, x, vals, color, width=0.5):
    vals = np.asarray(vals, float)
    vmin, vmax = vals.min(), vals.max()
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    mean = vals.mean()
    ax.plot([x, x], [vmin, vmax], color='black', lw=1.0, zorder=2)
    cap = width * 0.25
    ax.plot([x-cap, x+cap], [vmin, vmin], color='black', lw=1.0, zorder=2)
    ax.plot([x-cap, x+cap], [vmax, vmax], color='black', lw=1.0, zorder=2)
    ax.add_patch(plt.Rectangle((x-width/2, q1), width, max(q3-q1, 1e-6),
                               facecolor=color, edgecolor='black', lw=1.0,
                               alpha=0.85, zorder=3))
    ax.plot([x-width/2, x+width/2], [med, med], color='black', lw=1.6, zorder=4)
    ax.scatter([x], [mean], marker='D', s=24, color='white',
               edgecolor='black', lw=0.8, zorder=5)
    return vmin, vmax, mean, med, q1, q3


def plot_all_candle(rows, out_dir, show):
    vals = np.array([r['gen_ms'] for r in rows], float)
    fig, ax = plt.subplots(figsize=(6, 7.5))
    vmin, vmax, mean, med, q1, q3 = _draw_candle(ax, 0, vals, '#C44E52', width=0.5)
    ax.scatter(np.random.normal(0, 0.05, len(vals)), vals, s=10,
               color='black', alpha=0.20, zorder=6)
    for name, val in [('max', vmax), ('Q3', q3), ('mean', mean),
                      ('median', med), ('Q1', q1), ('min', vmin)]:
        ax.annotate(f'{name} = {val:.1f} ms', xy=(0.27, val), xytext=(0.34, val),
                    va='center', fontsize=9,
                    arrowprops=dict(arrowstyle='-', color='#999', lw=0.6))
    ax.set_xlim(-0.8, 1.2); ax.set_xticks([0])
    ax.set_xticklabels([f'All CFM samples\n(n={len(vals)})'], fontsize=10)
    ax.set_ylabel('CFM generation time (ms)')
    ax.set_title('CFM Trajectory Generation Time — All Samples\n'
                 f'mean {mean:.1f} ± {vals.std(ddof=1):.1f} ms  '
                 '(wick=min/max, box=IQR, line=median, ◇=mean)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    out = os.path.join(out_dir, 'cfm_gen_time_candle_all.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); print(f'Saved {out}')
    plt.show() if show else plt.close(fig)


def plot_per_class_candle(rows, out_dir, show):
    per_class = OrderedDict()
    for r in rows:
        per_class.setdefault(r['class'], []).append(r['gen_ms'])
    # keep only known classes, in order
    classes = [c for c in CLASS_DESC if c in per_class]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ymax = 0
    for x, c in enumerate(classes):
        _, vmax, *_ = _draw_candle(ax, x, per_class[c], CLASS_COLORS[c], width=0.5)
        ymax = max(ymax, vmax)
        v = np.asarray(per_class[c])
        ax.scatter(np.full(len(v), x) + np.random.normal(0, 0.03, len(v)),
                   v, s=8, color='black', alpha=0.20, zorder=6)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f'{c}\n{CLASS_DESC[c]}' for c in classes], fontsize=9)
    ax.set_ylabel('CFM generation time (ms)')
    ax.set_title('CFM Generation Time by Prompt Class '
                 '(should be ~constant — fixed model forward pass)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, ymax * 1.12)
    out = os.path.join(out_dir, 'cfm_gen_time_candle_per_class.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); print(f'Saved {out}')
    plt.show() if show else plt.close(fig)


def plot_llm_vs_cfm(rows, llm_csv, out_dir, show):
    cfm = np.array([r['gen_ms'] for r in rows], float)
    llm = None
    if os.path.exists(llm_csv):
        llm = np.array([float(r['thinking_ms'])
                        for r in csv.DictReader(open(llm_csv))
                        if r.get('thinking_ms')], float)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    data, labels, colors = [], [], []
    if llm is not None:
        data.append(llm); labels.append(f'LLM response\n(n={len(llm)})'); colors.append('#4C72B0')
    data.append(cfm); labels.append(f'CFM generation\n(n={len(cfm)})'); colors.append('#C44E52')

    for x, (d, col) in enumerate(zip(data, colors)):
        _draw_candle(ax, x, d, col, width=0.45)
        ax.scatter(np.full(len(d), x) + np.random.normal(0, 0.03, len(d)),
                   d, s=8, color='black', alpha=0.15, zorder=6)
    ax.set_yscale('log')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('time (ms, log scale)')
    title = 'LLM Response vs CFM Generation Time'
    if llm is not None:
        title += f'\nLLM mean {llm.mean():.0f} ms  vs  CFM mean {cfm.mean():.1f} ms  ' \
                 f'(CFM ~{llm.mean()/cfm.mean():.0f}× faster)'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, which='both')
    out = os.path.join(out_dir, 'llm_vs_cfm_time.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); print(f'Saved {out}')
    plt.show() if show else plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default=os.path.join(LOG_DIR, 'run_20260524-174419'))
    p.add_argument('--manifest', default=os.path.join(LOG_DIR, 'test_manifest_20260524-174424.csv'))
    p.add_argument('--llm', default=os.path.join(LOG_DIR, 'llm_prompts_20260524.csv'))
    p.add_argument('--out', default=os.path.join(LOG_DIR, 'figures'))
    p.add_argument('--win-start', default='2026-05-24 17:40:00')
    p.add_argument('--win-end',   default='2026-05-24 21:40:00')
    p.add_argument('--tz-offset', type=int, default=7, help='local UTC offset hours')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()
    if not args.show:
        matplotlib.use('Agg')

    tz = datetime.timezone(datetime.timedelta(hours=args.tz_offset))
    ws = datetime.datetime.strptime(args.win_start, '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz).timestamp()
    we = datetime.datetime.strptime(args.win_end, '%Y-%m-%d %H:%M:%S').replace(tzinfo=tz).timestamp()

    events = extract_cfm_events(ws, we)
    print(f'CFM gen events in window: {len(events)}')
    if not events:
        print('No CFM events found — check the run window / log files.')
        return

    rows = map_events_to_prompts(events, os.path.join(args.run, 'goals.csv'), args.manifest)
    # save mapped CSV
    out_dir = args.out; os.makedirs(out_dir, exist_ok=True)
    mapped_csv = os.path.join(LOG_DIR, 'cfm_gen_times.csv')
    with open(mapped_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'label', 'gen_ms', 'class', 'prompt_num'])
        w.writeheader(); w.writerows(rows)
    print(f'Saved {mapped_csv}')

    vals = np.array([r['gen_ms'] for r in rows])
    print(f'CFM gen time: mean={vals.mean():.1f}ms std={vals.std(ddof=1):.1f}ms '
          f'min={vals.min():.1f} max={vals.max():.1f}')

    plot_all_candle(rows, out_dir, args.show)
    plot_per_class_candle(rows, out_dir, args.show)
    plot_llm_vs_cfm(rows, args.llm, out_dir, args.show)


if __name__ == '__main__':
    main()
