#!/usr/bin/env python3
"""Generate graphs for GPU-native sort merge report."""

import matplotlib.pyplot as plt
import numpy as np

# Data: CUDA 4GPU comparison
pp_sizes = ['pp128', 'pp512', 'pp2048']

# Normal GGUF
upstream_normal = [216.72, 340.06, 329.86]
merge_normal = [230.36, 385.87, 372.86]

# Fused GGUF
upstream_fused = [260.25, 417.00, 404.25]
merge_fused = [273.20, 462.43, 447.58]

# TG data
tg_upstream_normal = 47.18
tg_upstream_fused = 47.37
tg_merge_normal = 47.20
tg_merge_fused = 47.44

# --- Graph 1: PP throughput comparison ---
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(pp_sizes))
width = 0.2

bars1 = ax.bar(x - 1.5*width, upstream_normal, width, label='upstream + normal GGUF', color='#a0a0a0')
bars2 = ax.bar(x - 0.5*width, upstream_fused, width, label='upstream + fused GGUF (1-B only)', color='#6baed6')
bars3 = ax.bar(x + 0.5*width, merge_normal, width, label='merge + normal GGUF (1-A only)', color='#fd8d3c')
bars4 = ax.bar(x + 1.5*width, merge_fused, width, label='merge + fused GGUF (1-A + 1-B)', color='#e6550d')

ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
ax.set_title('PP Throughput: upstream vs GPU-native sort + Gate+Up merge\n(Qwen3.5-35B-A3B Q4_K_M, CUDA 4GPU P100)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(pp_sizes, fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-08_merge_gpu_native_sort_pp_comparison.png', dpi=150)
plt.close()

# --- Graph 2: Improvement rates vs upstream baseline ---
fig, ax = plt.subplots(figsize=(10, 6))

# 1-A only improvement (merge normal vs upstream normal)
imp_1a = [(m/u - 1)*100 for m, u in zip(merge_normal, upstream_normal)]
# 1-B only improvement (upstream fused vs upstream normal)
imp_1b = [(f/n - 1)*100 for f, n in zip(upstream_fused, upstream_normal)]
# 1-A + 1-B combined (merge fused vs upstream normal)
imp_combined = [(m/u - 1)*100 for m, u in zip(merge_fused, upstream_normal)]

x = np.arange(len(pp_sizes))
width = 0.25

bars1 = ax.bar(x - width, imp_1a, width, label='1-A: GPU-native sort', color='#fd8d3c')
bars2 = ax.bar(x, imp_1b, width, label='1-B: Gate+Up merge', color='#6baed6')
bars3 = ax.bar(x + width, imp_combined, width, label='1-A + 1-B combined', color='#e6550d')

ax.set_ylabel('Improvement vs upstream (%)', fontsize=12)
ax.set_title('PP Improvement Rate by Optimization\n(vs upstream/master baseline, Qwen3.5-35B-A3B Q4_K_M, CUDA 4GPU)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(pp_sizes, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'+{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-08_merge_gpu_native_sort_improvement.png', dpi=150)
plt.close()

# --- Graph 3: Pre-merge vs post-merge regression check (RDMA 4GPU) ---
fig, ax = plt.subplots(figsize=(10, 6))

tests = ['pp128', 'pp512', 'pp2048', 'tg32']
pre_merge = [272.09, 459.07, 518.91, 36.12]
post_merge = [271.52, 457.90, 467.76, 36.68]
changes = [(p/q - 1)*100 for p, q in zip(post_merge, pre_merge)]

x = np.arange(len(tests))
width = 0.35

bars1 = ax.bar(x - width/2, pre_merge, width, label='Pre-merge (1-A only)', color='#6baed6')
bars2 = ax.bar(x + width/2, post_merge, width, label='Post-merge (1-A + upstream)', color='#e6550d')

ax.set_ylabel('Throughput (tokens/s)', fontsize=12)
ax.set_title('Upstream Merge Regression Check\n(Qwen3.5-35B-A3B fused GGUF, 2C+2R RDMA)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(tests, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for i, (b1, b2) in enumerate(zip(bars1, bars2)):
    h2 = b2.get_height()
    color = '#d62728' if changes[i] < -2 else '#2ca02c' if changes[i] > 2 else '#333333'
    ax.annotate(f'{changes[i]:+.1f}%', xy=(b2.get_x() + b2.get_width()/2, h2),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-08_merge_gpu_native_sort_regression.png', dpi=150)
plt.close()

print("All graphs generated successfully.")
