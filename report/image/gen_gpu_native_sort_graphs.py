#!/usr/bin/env python3
"""Generate benchmark comparison graphs for GPU-native IDs sorting optimization."""

import matplotlib.pyplot as plt
import numpy as np

# Data from ABAB x4 benchmark
pp_sizes = ['pp128', 'pp512', 'pp2048']

baseline_means = [224.39, 357.92, 398.21]
modified_means = [231.60, 383.70, 431.33]

baseline_samples = {
    'pp128':  [224.46, 224.43, 224.20, 224.46],
    'pp512':  [357.16, 358.08, 358.50, 357.93],
    'pp2048': [398.58, 396.67, 398.56, 399.02],
}
modified_samples = {
    'pp128':  [231.41, 231.68, 231.60, 231.71],
    'pp512':  [383.75, 383.79, 383.62, 383.64],
    'pp2048': [431.40, 431.40, 431.31, 431.21],
}

baseline_stds = [np.std(baseline_samples[k], ddof=1) for k in pp_sizes]
modified_stds = [np.std(modified_samples[k], ddof=1) for k in pp_sizes]

improvements = [(m - b) / b * 100 for b, m in zip(baseline_means, modified_means)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Graph 1: PP throughput comparison
x = np.arange(len(pp_sizes))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_means, width, label='Baseline (CPU sort)',
                color='#5B9BD5', yerr=baseline_stds, capsize=4)
bars2 = ax1.bar(x + width/2, modified_means, width, label='GPU-native sort',
                color='#ED7D31', yerr=modified_stds, capsize=4)

ax1.set_ylabel('Throughput (t/s)')
ax1.set_title('Prompt Processing Throughput\nQwen3.5-35B-A3B Q4_K_M, 4GPU (2C+2R)')
ax1.set_xticks(x)
ax1.set_xticklabels(pp_sizes)
ax1.legend()
ax1.set_ylim(0, max(modified_means) * 1.15)

for bar, val in zip(bars1, baseline_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, modified_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# Graph 2: Improvement percentage
colors = ['#70AD47' if imp > 0 else '#FF6B6B' for imp in improvements]
bars3 = ax2.bar(pp_sizes, improvements, color=colors, width=0.5)
ax2.set_ylabel('Improvement (%)')
ax2.set_title('PP Improvement: GPU-native sort vs CPU sort')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_ylim(0, max(improvements) * 1.3)

for bar, val in zip(bars3, improvements):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'+{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-08_gpu_native_sort_pp_comparison.png',
            dpi=150, bbox_inches='tight')
print("Saved graph")
