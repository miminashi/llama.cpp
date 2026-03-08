#!/usr/bin/env python3
"""Generate graphs for combined 1-A (GPU-native sort) + 1-B (Gate+Up merge) benchmark."""

import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
from collections import defaultdict

OUTPUT_DIR = "/home/ubuntu/projects/llama.cpp/report/image"

# 1-A only data (from gpu_native_sort report)
data_1a = {
    'pp128':  {'baseline': 224.4, 'modified': 231.6, 'pct': 3.2},
    'pp512':  {'baseline': 357.9, 'modified': 383.7, 'pct': 7.2},
    'pp2048': {'baseline': 398.2, 'modified': 431.3, 'pct': 8.3},
}

# 1-B only data (from gate_up_merge report)
data_1b = {
    'pp128':  {'baseline': 224.4, 'modified': 266.3, 'pct': 18.7},
    'pp512':  {'baseline': 357.5, 'modified': 431.3, 'pct': 20.6},
    'pp2048': {'baseline': 398.9, 'modified': 484.6, 'pct': 21.5},
}


def parse_csv(csv_path):
    """Parse llama-bench CSV output, return dict of {metric: [avg_ts values]}."""
    results = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_prompt = int(row['n_prompt'])
                n_gen = int(row['n_gen'])
                avg_ts = float(row['avg_ts'])
                key = f"pp{n_prompt}" if n_gen == 0 else f"tg{n_gen}"
                results[key].append(avg_ts)
            except (ValueError, KeyError):
                continue
    return results


def make_bar_chart(data_combined, output_path):
    """Create 4-condition PP throughput comparison bar chart."""
    metrics = ['pp128', 'pp512', 'pp2048']
    x = np.arange(len(metrics))
    width = 0.2

    baseline_vals = [data_1a[m]['baseline'] for m in metrics]
    a_only_vals = [data_1a[m]['modified'] for m in metrics]
    b_only_vals = [data_1b[m]['modified'] for m in metrics]
    combined_vals = [data_combined.get(m, {}).get('mean', 0) for m in metrics]
    combined_errs = [data_combined.get(m, {}).get('std', 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - 1.5*width, baseline_vals, width, label='Baseline', color='#4472C4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*width, a_only_vals, width, label='1-A: GPU-native sort', color='#ED7D31', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*width, b_only_vals, width, label='1-B: Gate+Up merge', color='#70AD47', edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x + 1.5*width, combined_vals, width, label='1-A + 1-B combined',
                   color='#FFC000', edgecolor='black', linewidth=0.5,
                   yerr=combined_errs, capsize=3, error_kw={'linewidth': 1})

    ax.set_xlabel('Prompt Size', fontsize=12)
    ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax.set_title('PP Throughput: Baseline vs Individual vs Combined Optimizations\n(Qwen3.5-35B-A3B Q4_K_M, 4GPU 2C+2R)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def make_improvement_chart(data_combined, data_baseline, output_path):
    """Create cumulative improvement stacked chart."""
    metrics = ['pp128', 'pp512', 'pp2048']

    pct_1a = [data_1a[m]['pct'] for m in metrics]
    pct_1b = [data_1b[m]['pct'] for m in metrics]

    # Theoretical multiplicative: (1 + a%) * (1 + b%) - 1
    pct_theory = [(1 + a/100) * (1 + b/100) * 100 - 100 for a, b in zip(pct_1a, pct_1b)]

    # Actual combined (use today's measured baseline for accurate comparison)
    pct_actual = []
    for m in metrics:
        if m in data_combined and m in data_baseline and data_combined[m]['mean'] > 0:
            actual = (data_combined[m]['mean'] / data_baseline[m]['mean'] - 1) * 100
            pct_actual.append(actual)
        else:
            pct_actual.append(0)

    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - 1.5*width, pct_1a, width, label='1-A: GPU-native sort only', color='#ED7D31', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*width, pct_1b, width, label='1-B: Gate+Up merge only', color='#70AD47', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*width, pct_theory, width, label='Theoretical (1-A × 1-B)', color='#A5A5A5', edgecolor='black', linewidth=0.5, linestyle='--')
    bars4 = ax.bar(x + 1.5*width, pct_actual, width, label='Actual combined', color='#FFC000', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Prompt Size', fontsize=12)
    ax.set_ylabel('Improvement vs Baseline (%)', fontsize=12)
    ax.set_title('Cumulative Improvement: Individual vs Theoretical vs Actual\n(Qwen3.5-35B-A3B Q4_K_M, 4GPU 2C+2R)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'+{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    baseline_csv = sys.argv[1] if len(sys.argv) > 1 else '/tmp/bench_combined_baseline.csv'
    modified_csv = sys.argv[2] if len(sys.argv) > 2 else '/tmp/bench_combined_modified.csv'

    print(f"Parsing baseline: {baseline_csv}")
    raw_b = parse_csv(baseline_csv)
    data_baseline = {}
    for key, vals in raw_b.items():
        data_baseline[key] = {'mean': np.mean(vals), 'std': np.std(vals), 'n': len(vals)}
        print(f"  {key}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, n={len(vals)}")

    print(f"Parsing modified: {modified_csv}")
    raw_m = parse_csv(modified_csv)
    data_combined = {}
    for key, vals in raw_m.items():
        data_combined[key] = {'mean': np.mean(vals), 'std': np.std(vals), 'n': len(vals)}
        print(f"  {key}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, n={len(vals)}")

    make_bar_chart(data_combined, f"{OUTPUT_DIR}/2026-03-08_combined_1a1b_pp_comparison.png")
    make_improvement_chart(data_combined, data_baseline, f"{OUTPUT_DIR}/2026-03-08_combined_1a1b_improvement.png")
