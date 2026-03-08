#!/usr/bin/env python3
"""Generate graphs for gate+up merge benchmark results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from ABAB benchmark (4 rounds, 5 reps each)
data = {
    'pp128':  {'baseline': [224.54, 224.29, 224.60, 224.31], 'fused': [266.45, 265.94, 266.50, 266.37]},
    'pp512':  {'baseline': [358.28, 355.58, 357.89, 358.09], 'fused': [431.77, 430.22, 432.26, 430.78]},
    'pp2048': {'baseline': [399.05, 398.64, 398.66, 399.06], 'fused': [485.91, 484.33, 484.89, 483.40]},
    'tg32':   {'baseline': [35.55, 35.29, 35.64, 34.91],     'fused': [35.13, 34.98, 35.36, 35.38]},
    'tg128':  {'baseline': [35.66, 35.56, 35.65, 35.67],     'fused': [35.57, 35.59, 35.55, 35.52]},
}

outdir = '/home/ubuntu/projects/llama.cpp/report/image'

# --- Graph 1: PP throughput comparison ---
fig, ax = plt.subplots(figsize=(10, 6))
pp_keys = ['pp128', 'pp512', 'pp2048']
x = np.arange(len(pp_keys))
width = 0.35

baseline_means = [np.mean(data[k]['baseline']) for k in pp_keys]
baseline_stds  = [np.std(data[k]['baseline']) for k in pp_keys]
fused_means    = [np.mean(data[k]['fused']) for k in pp_keys]
fused_stds     = [np.std(data[k]['fused']) for k in pp_keys]

bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
               label='Separate (gate_exps + up_exps)', color='#4C72B0', capsize=5)
bars2 = ax.bar(x + width/2, fused_means, width, yerr=fused_stds,
               label='Fused (gate_up_exps)', color='#DD8452', capsize=5)

ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
ax.set_title('Qwen3.5-35B-A3B Q4_K_M: Gate+Up Merge PP Performance\n(4GPU: 2 CUDA + 2 RDMA)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(pp_keys, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars1, baseline_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, fused_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{outdir}/2026-03-08_gate_up_merge_pp_comparison.png', dpi=150)
plt.close()

# --- Graph 2: Improvement percentage ---
fig, ax = plt.subplots(figsize=(10, 6))
all_keys = ['pp128', 'pp512', 'pp2048', 'tg32', 'tg128']
improvements = [(np.mean(data[k]['fused']) - np.mean(data[k]['baseline'])) / np.mean(data[k]['baseline']) * 100
                for k in all_keys]
colors = ['#DD8452' if imp > 1 else '#A0A0A0' for imp in improvements]

bars = ax.bar(range(len(all_keys)), improvements, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Improvement (%)', fontsize=12)
ax.set_title('Gate+Up Merge: Performance Improvement\n(Fused vs Separate, Qwen3.5-35B-A3B Q4_K_M, 4GPU)', fontsize=13)
ax.set_xticks(range(len(all_keys)))
ax.set_xticklabels(all_keys, fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

for bar, imp in zip(bars, improvements):
    va = 'bottom' if imp >= 0 else 'top'
    offset = 0.5 if imp >= 0 else -0.5
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
            f'{imp:+.1f}%', ha='center', va=va, fontsize=10, fontweight='bold')

# Add significance annotations
sig_labels = ['***', '***', '***', 'ns', 'ns']
for i, sig in enumerate(sig_labels):
    y_pos = max(improvements[i], 0) + 2.5 if improvements[i] >= 0 else improvements[i] - 2
    ax.text(i, y_pos, sig, ha='center', va='bottom', fontsize=9, color='#555555')

plt.tight_layout()
plt.savefig(f'{outdir}/2026-03-08_gate_up_merge_improvement.png', dpi=150)
plt.close()

print("Graphs saved:")
print(f"  {outdir}/2026-03-08_gate_up_merge_pp_comparison.png")
print(f"  {outdir}/2026-03-08_gate_up_merge_improvement.png")
