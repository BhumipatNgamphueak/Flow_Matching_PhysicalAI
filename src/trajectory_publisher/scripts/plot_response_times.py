#!/usr/bin/env python3
"""
plot_response_times.py — LLM response-time statistics per prompt / per class.

Reads the LLM call log written by llm_node (timestamp, thinking_ms, prompt, ...)
and the test manifest (class, prompt_num, prompt_text), then produces:

  1. Per-prompt bar chart  — mean thinking time ± std, colored by class
  2. Per-class box plot     — distribution of thinking times within each class

Usage:
  python3 plot_response_times.py \
      --llm    ~/comparison_logs/llm_prompts_20260524.csv \
      --manifest ~/comparison_logs/test_manifest_20260524-174424.csv \
      --out    ~/comparison_logs/figures \
      [--show]

If --manifest is omitted, prompts are grouped in order (3 per class assumed).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 5 evaluation classes (matches README.md)
CLASS_DESC = {
    'C1': 'Direction + Speed',
    'C2': 'Absolute Position',
    'C3': 'Speed + Duration',
    'C4': 'Position + Duration',
    'C5': 'Ambiguous',
}
CLASS_COLORS = {
    'C1': '#4C72B0', 'C2': '#55A868', 'C3': '#C44E52',
    'C4': '#8172B3', 'C5': '#CCB974',
}


def _read_csv_dicts(path: str) -> list[dict]:
    with open(os.path.expanduser(path)) as f:
        return list(csv.DictReader(f))


def _short(prompt: str, n: int = 38) -> str:
    p = prompt.strip()
    return (p[:n] + '…') if len(p) > n else p


def build_prompt_table(llm_rows, manifest_rows):
    """Returns ordered list of dicts:
       {class, prompt_num, label, prompt, times[ms]} — one per unique prompt."""
    # Map prompt-text → (class, prompt_num) using manifest
    text_to_cp = {}
    if manifest_rows:
        for r in manifest_rows:
            key = r['prompt_text'].strip()
            if key not in text_to_cp:
                text_to_cp[key] = (f"C{r['class']}", f"P{r['prompt_num']}")

    grouped: "OrderedDict[str, dict]" = OrderedDict()
    for r in llm_rows:
        prompt = r['prompt'].strip()
        try:
            ms = float(r['thinking_ms'])
        except (KeyError, ValueError):
            continue
        if prompt not in grouped:
            cls, pn = text_to_cp.get(prompt, ('C?', '?'))
            grouped[prompt] = {
                'class': cls, 'prompt_num': pn,
                'prompt': prompt, 'times': [],
            }
        grouped[prompt]['times'].append(ms)

    table = list(grouped.values())
    # sort by class then prompt_num for stable ordering
    def sort_key(d):
        c = d['class'][1:] if d['class'].startswith('C') and d['class'][1:].isdigit() else '9'
        p = d['prompt_num'][1:] if d['prompt_num'].startswith('P') and d['prompt_num'][1:].isdigit() else '9'
        return (int(c) if c.isdigit() else 9, int(p) if p.isdigit() else 9)
    table.sort(key=sort_key)
    for i, d in enumerate(table, 1):
        d['idx'] = i
        d['label'] = f"{d['class']}.{d['prompt_num']}"
    return table


def _draw_candle(ax, x, vals, color, width=0.6):
    """Draw one candlestick at position x for the sample `vals`:
       thin wick = min→max, thick body = Q1→Q3, line = median, dot = mean."""
    vals = np.asarray(vals, dtype=float)
    vmin, vmax = vals.min(), vals.max()
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    mean = vals.mean()
    # wick (min → max)
    ax.plot([x, x], [vmin, vmax], color='black', linewidth=1.0, zorder=2)
    # small caps at min/max
    cap = width * 0.25
    ax.plot([x - cap, x + cap], [vmin, vmin], color='black', linewidth=1.0, zorder=2)
    ax.plot([x - cap, x + cap], [vmax, vmax], color='black', linewidth=1.0, zorder=2)
    # body (Q1 → Q3)
    ax.add_patch(plt.Rectangle((x - width / 2, q1), width, max(q3 - q1, 1e-6),
                               facecolor=color, edgecolor='black',
                               linewidth=1.0, alpha=0.85, zorder=3))
    # median line
    ax.plot([x - width / 2, x + width / 2], [med, med],
            color='black', linewidth=1.6, zorder=4)
    # mean marker
    ax.scatter([x], [mean], marker='D', s=22, color='white',
               edgecolor='black', linewidth=0.8, zorder=5)
    return vmin, vmax


def plot_per_prompt_candle(table, out_dir, show):
    fig, ax = plt.subplots(figsize=(15, 7.5))
    xs = np.arange(len(table))
    ymax = 0
    for x, d in zip(xs, table):
        color = CLASS_COLORS.get(d['class'], '#888888')
        _, vmax = _draw_candle(ax, x, d['times'], color)
        ymax = max(ymax, vmax)

    ax.set_xticks(xs)
    ax.set_xticklabels([d['label'] for d in table], fontsize=9)
    ax.set_ylabel('LLM response time (ms)')
    ax.set_title('LLM Response Time per Prompt — candlestick '
                 '(wick=min/max, box=IQR, line=median, ◇=mean, n=5)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, ymax * 1.12)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASS_DESC]
    leg_labels = [f'{c}: {CLASS_DESC[c]}' for c in CLASS_DESC]
    ax.legend(handles, leg_labels, loc='upper left', fontsize=9, framealpha=0.9)

    for x, d in zip(xs, table):
        ax.annotate(_short(d['prompt'], 32),
                    xy=(x, 0), xytext=(0, -26),
                    textcoords='offset points', rotation=35,
                    ha='right', va='top', fontsize=6.5, color='#444444')

    fig.subplots_adjust(bottom=0.30)
    out = os.path.join(out_dir, 'llm_response_time_candle_per_prompt.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show() if show else plt.close(fig)


def plot_per_class_candle(table, out_dir, show):
    per_class: "OrderedDict[str, list]" = OrderedDict()
    for d in table:
        per_class.setdefault(d['class'], []).extend(d['times'])
    classes = list(per_class.keys())

    fig, ax = plt.subplots(figsize=(10, 6.5))
    xs = np.arange(len(classes))
    ymax = 0
    for x, c in zip(xs, classes):
        vmin, vmax = _draw_candle(ax, x, per_class[c],
                                  CLASS_COLORS.get(c, '#888888'), width=0.5)
        ymax = max(ymax, vmax)
        # overlay raw points
        v = np.asarray(per_class[c])
        ax.scatter(np.full(len(v), x) + np.random.normal(0, 0.03, len(v)),
                   v, s=14, color='black', alpha=0.35, zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels([f'{c}\n{CLASS_DESC.get(c, "")}' for c in classes], fontsize=9)
    ax.set_ylabel('LLM response time (ms)')
    ax.set_title('LLM Response Time by Prompt Class — candlestick '
                 '(wick=min/max, box=IQR, line=median, ◇=mean, n=15)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, ymax * 1.12)

    out = os.path.join(out_dir, 'llm_response_time_candle_per_class.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show() if show else plt.close(fig)


def plot_overall_candle(table, out_dir, show):
    """A single candlestick aggregating ALL prompts/repeats into one candle."""
    all_times = []
    for d in table:
        all_times.extend(d['times'])
    vals = np.asarray(all_times, dtype=float)

    vmin, vmax = vals.min(), vals.max()
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    mean = vals.mean()
    std  = vals.std(ddof=1)

    fig, ax = plt.subplots(figsize=(6, 7.5))
    x = 0
    _draw_candle(ax, x, vals, '#4C72B0', width=0.5)
    # overlay all raw points with jitter
    ax.scatter(np.full(len(vals), x) + np.random.normal(0, 0.05, len(vals)),
               vals, s=16, color='black', alpha=0.30, zorder=6)

    # annotate the five-number summary + mean on the right
    labels = [
        ('max',    vmax),
        ('Q3',     q3),
        ('median', med),
        ('mean',   mean),
        ('Q1',     q1),
        ('min',    vmin),
    ]
    for name, val in labels:
        ax.annotate(f'{name} = {val:.0f} ms',
                    xy=(x + 0.27, val), xytext=(x + 0.34, val),
                    textcoords='data', va='center', fontsize=9,
                    color='#222222',
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.6))

    ax.set_xlim(-0.8, 1.1)
    ax.set_xticks([x])
    ax.set_xticklabels([f'All prompts\n(n={len(vals)})'], fontsize=10)
    ax.set_ylabel('LLM response time (ms)')
    ax.set_title('LLM Response Time — All Prompts Combined\n'
                 f'mean {mean:.0f} ± {std:.0f} ms   (wick=min/max, box=IQR, '
                 'line=median, ◇=mean)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, vmax * 1.10)

    out = os.path.join(out_dir, 'llm_response_time_candle_all.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    plt.show() if show else plt.close(fig)


def plot_per_prompt(table, out_dir, show):
    labels  = [d['label'] for d in table]
    means   = [np.mean(d['times']) for d in table]
    stds    = [np.std(d['times'], ddof=1) if len(d['times']) > 1 else 0 for d in table]
    colors  = [CLASS_COLORS.get(d['class'], '#888888') for d in table]
    ns      = [len(d['times']) for d in table]

    fig, ax = plt.subplots(figsize=(14, 7))
    xs = np.arange(len(table))
    bars = ax.bar(xs, means, yerr=stds, capsize=4, color=colors,
                  edgecolor='black', linewidth=0.5, alpha=0.9)

    # value labels on bars
    for x, m, s, n in zip(xs, means, stds, ns):
        ax.text(x, m + s + 100, f'{m:.0f}', ha='center', va='bottom', fontsize=8)
        ax.text(x, 150, f'n={n}', ha='center', va='bottom', fontsize=7, color='white')

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel('LLM response time (ms)')
    ax.set_title('Gemini LLM Response Time per Prompt  (mean ± std, n=5 each)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # class legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c])
               for c in CLASS_DESC]
    leg_labels = [f'{c}: {CLASS_DESC[c]}' for c in CLASS_DESC]
    ax.legend(handles, leg_labels, loc='upper left', fontsize=9, framealpha=0.9)

    # second x-axis row: short prompt text under each bar
    for x, d in zip(xs, table):
        ax.annotate(_short(d['prompt'], 30),
                    xy=(x, 0), xytext=(0, -28),
                    textcoords='offset points', rotation=35,
                    ha='right', va='top', fontsize=6.5, color='#444444')

    fig.subplots_adjust(bottom=0.30)
    out = os.path.join(out_dir, 'llm_response_time_per_prompt.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_per_class(table, out_dir, show):
    # aggregate times per class
    per_class: "OrderedDict[str, list]" = OrderedDict()
    for d in table:
        per_class.setdefault(d['class'], []).extend(d['times'])

    classes = list(per_class.keys())
    data    = [per_class[c] for c in classes]
    colors  = [CLASS_COLORS.get(c, '#888888') for c in classes]

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    medianprops=dict(color='black'))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    # overlay individual points
    for i, d in enumerate(data, 1):
        jitter = np.random.normal(0, 0.04, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=14,
                   color='black', alpha=0.4, zorder=3)

    ax.set_xticklabels([f'{c}\n{CLASS_DESC.get(c, "")}' for c in classes],
                       fontsize=9)
    ax.set_ylabel('LLM response time (ms)')
    ax.set_title('Gemini LLM Response Time by Prompt Class',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    out = os.path.join(out_dir, 'llm_response_time_per_class.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_summary(table):
    print('\nResponse-time summary (ms):')
    print(f'{"label":8s} {"n":>3s} {"mean":>8s} {"std":>7s} {"min":>7s} {"max":>7s}  prompt')
    all_times = []
    for d in table:
        t = np.array(d['times']); all_times.extend(t)
        print(f'{d["label"]:8s} {len(t):3d} {t.mean():8.0f} '
              f'{(t.std(ddof=1) if len(t)>1 else 0):7.0f} {t.min():7.0f} {t.max():7.0f}  '
              f'{_short(d["prompt"], 45)}')
    a = np.array(all_times)
    print(f'\nOVERALL  n={len(a)}  mean={a.mean():.0f}ms  std={a.std(ddof=1):.0f}ms  '
          f'min={a.min():.0f}ms  max={a.max():.0f}ms')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--llm', default='~/comparison_logs/llm_prompts_20260524.csv',
                   help='LLM call log CSV (timestamp, thinking_ms, prompt, ...)')
    p.add_argument('--manifest', default='~/comparison_logs/test_manifest_20260524-174424.csv',
                   help='Test manifest CSV (class, prompt_num, prompt_text)')
    p.add_argument('--out', default='~/comparison_logs/figures',
                   help='Output directory for figures')
    p.add_argument('--show', action='store_true', help='Show windows instead of just saving')
    args = p.parse_args()

    if not args.show:
        matplotlib.use('Agg')

    llm_rows = _read_csv_dicts(args.llm)
    manifest_rows = []
    mpath = os.path.expanduser(args.manifest)
    if os.path.exists(mpath):
        manifest_rows = _read_csv_dicts(mpath)
    else:
        print(f'(manifest not found at {mpath}; grouping prompts in order)')

    table = build_prompt_table(llm_rows, manifest_rows)
    if not table:
        print('No LLM rows with thinking_ms found — nothing to plot.')
        return

    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print_summary(table)
    # Single candle: all prompts combined (requested)
    plot_overall_candle(table, out_dir, args.show)
    # Candlestick charts: per-prompt + all-prompt (per class)
    plot_per_prompt_candle(table, out_dir, args.show)
    plot_per_class_candle(table, out_dir, args.show)
    # Bar / box variants (kept for comparison)
    plot_per_prompt(table, out_dir, args.show)
    plot_per_class(table, out_dir, args.show)


if __name__ == '__main__':
    main()
