#!/usr/bin/env python3
"""Generate pp16384 ubatch sweep and GPU scaling graphs."""

import matplotlib.pyplot as plt
import numpy as np

# Phase A: 2C2R (4GPU) ubatch sweep
ub_sizes = [512, 2048, 4096, 8192, 16384]
ts_2c2r = [345.27, 410.70, 411.36, 412.32, 412.17]
sd_2c2r = [0.81, 1.38, 0.75, 0.78, 1.13]

# Phase A': 4C4R (8GPU) ubatch sweep (3 points)
ub_4c4r = [512, 2048, 8192]
ts_4c4r = [299.89, 397.79, 399.13]
sd_4c4r = [2.34, 0.58, 0.33]

# Phase B: overlap results
ts_2c2r_overlap = {2048: 417.61}
sd_2c2r_overlap = {2048: 0.40}
ts_4c4r_overlap = {2048: 407.52}
sd_4c4r_overlap = {2048: 0.49}

plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,

})

# === Graph 1: ubatch size vs PP performance ===
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.errorbar(ub_sizes, ts_2c2r, yerr=sd_2c2r, marker='o', linewidth=2,
            capsize=4, label='2C2R (4GPU)', color='#2196F3')
ax.errorbar(ub_4c4r, ts_4c4r, yerr=sd_4c4r, marker='s', linewidth=2,
            capsize=4, label='4C4R (8GPU)', color='#FF5722')

# Overlap points
ax.errorbar([2048], [ts_2c2r_overlap[2048]], yerr=[sd_2c2r_overlap[2048]],
            marker='D', markersize=9, linewidth=0, capsize=4,
            label='2C2R + overlap', color='#1565C0', zorder=5)
ax.errorbar([2048], [ts_4c4r_overlap[2048]], yerr=[sd_4c4r_overlap[2048]],
            marker='D', markersize=9, linewidth=0, capsize=4,
            label='4C4R + overlap', color='#BF360C', zorder=5)

ax.set_xscale('log', base=2)
ax.set_xticks(ub_sizes)
ax.set_xticklabels([str(x) for x in ub_sizes])
ax.set_xlabel('ubatch size')
ax.set_ylabel('pp16384 throughput (t/s)')
ax.set_title('Qwen3.5-35B-A3B Q4_K_M — pp16384 ubatch sweep\n(RDMA, flash attention)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(250, 450)

# Annotate key values
ax.annotate(f'{ts_2c2r[0]:.1f}', (ub_sizes[0], ts_2c2r[0]),
            textcoords='offset points', xytext=(8, -12), fontsize=9, color='#2196F3')
ax.annotate(f'{ts_2c2r[1]:.1f}', (ub_sizes[1], ts_2c2r[1]),
            textcoords='offset points', xytext=(8, 5), fontsize=9, color='#2196F3')
ax.annotate(f'{ts_2c2r_overlap[2048]:.1f}', (2048, ts_2c2r_overlap[2048]),
            textcoords='offset points', xytext=(8, 5), fontsize=9, color='#1565C0')
ax.annotate(f'{ts_4c4r[0]:.1f}', (ub_4c4r[0], ts_4c4r[0]),
            textcoords='offset points', xytext=(8, -12), fontsize=9, color='#FF5722')

plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-07_pp16384_ubatch_sweep.png')
plt.close()

# === Graph 2: GPU scaling efficiency ===
fig, ax = plt.subplots(figsize=(8, 5))

# Compute scaling ratio for common ubatch sizes
common_ub = [512, 2048, 8192]
scaling = [ts_4c4r[i] / ts_2c2r[ub_sizes.index(common_ub[i])] for i in range(len(common_ub))]

# Add overlap scaling
overlap_scaling = ts_4c4r_overlap[2048] / ts_2c2r_overlap[2048]

ax.bar(range(len(common_ub)), scaling, width=0.5, color='#4CAF50', alpha=0.8, label='baseline')
ax.bar([len(common_ub)], [overlap_scaling], width=0.5, color='#FF9800', alpha=0.8, label='overlap (ub=2048)')

ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='ideal 2.0x')
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='1.0x (no scaling)')

xtick_labels = [f'ub={x}' for x in common_ub] + ['ub=2048\n(overlap)']
ax.set_xticks(range(len(common_ub) + 1))
ax.set_xticklabels(xtick_labels)
ax.set_ylabel('4C4R / 2C2R speed ratio')
ax.set_title('Qwen3.5-35B-A3B — GPU scaling efficiency (4→8 GPU)\npp16384')
ax.legend(loc='upper right')
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 2.5)

for i, v in enumerate(scaling):
    ax.text(i, v + 0.03, f'{v:.2f}x', ha='center', fontsize=10, fontweight='bold')
ax.text(len(common_ub), overlap_scaling + 0.03, f'{overlap_scaling:.2f}x',
        ha='center', fontsize=10, fontweight='bold')

plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-07_pp16384_gpu_scaling.png')
plt.close()

print("Graphs saved successfully.")
