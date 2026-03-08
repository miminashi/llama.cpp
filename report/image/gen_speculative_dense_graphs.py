#!/usr/bin/env python3
"""Generate speculative decoding benchmark graphs for Dense (27B) + MoE comparison."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# === Data ===

# 27B Dense main model (2GPU, baseline tg=10.5)
dense_27b = {
    'drafts': ['0.8B', '2B', '4B', '9B'],
    'draft_sizes': [0.8, 2, 4, 9],
    'dm8_tg':  [9.39, 9.12, 8.26, 7.62],
    'dm16_tg': [13.16, 12.56, 10.92, 9.73],
    'dm8_accept':  [1.79, 4.00, 0.95, 0.62],
    'dm16_accept': [0.36, 0.53, 0.36, 0.42],
    'baseline': 10.5,
}

# 35B-A3B MoE main model (2GPU, baseline tg=39.1) - from previous report
moe_35b = {
    'drafts': ['0.8B', '2B'],
    'draft_sizes': [0.8, 2],
    'dm8_tg':  [33.9, 30.4],
    'dm16_tg': [34.2, 30.1],
    'dm8_accept':  [0.83, 0.62],
    'dm16_accept': [0.36, 0.36],
    'baseline': 39.1,
}

# 122B-A10B MoE main model (7GPU, baseline tg=18.9) - from previous report
moe_122b = {
    'drafts': ['0.8B', '2B', '4B'],
    'draft_sizes': [0.8, 2, 4],
    'dm8_tg':  [20.6, 19.1, 15.6],
    'dm16_tg': [22.8, 20.8, 16.6],
    'dm8_accept':  [1.66, 3.50, 1.29],
    'dm16_accept': [0.74, 0.53, 0.31],
    'baseline': 18.9,
}

# Draft standalone speeds
draft_speeds = {0.8: 106.4, 2: 77.5, 4: 42.4, 9: 29.7}

def speedup(tg, baseline):
    return (tg / baseline - 1) * 100

# === Graph 1: Speedup vs Draft Model Size ===
fig, ax = plt.subplots(figsize=(10, 6))

# 27B Dense
x = dense_27b['draft_sizes']
y8 = [speedup(t, dense_27b['baseline']) for t in dense_27b['dm8_tg']]
y16 = [speedup(t, dense_27b['baseline']) for t in dense_27b['dm16_tg']]
ax.plot(x, y8, 'o-', color='#2196F3', label='27B Dense (2GPU) max=8', markersize=8)
ax.plot(x, y16, 's-', color='#1565C0', label='27B Dense (2GPU) max=16', markersize=8)

# 35B-A3B MoE
x = moe_35b['draft_sizes']
y8 = [speedup(t, moe_35b['baseline']) for t in moe_35b['dm8_tg']]
y16 = [speedup(t, moe_35b['baseline']) for t in moe_35b['dm16_tg']]
ax.plot(x, y8, '^--', color='#FF9800', label='35B-A3B MoE (2GPU) max=8', markersize=8, alpha=0.7)
ax.plot(x, y16, 'v--', color='#E65100', label='35B-A3B MoE (2GPU) max=16', markersize=8, alpha=0.7)

# 122B-A10B MoE
x = moe_122b['draft_sizes']
y8 = [speedup(t, moe_122b['baseline']) for t in moe_122b['dm8_tg']]
y16 = [speedup(t, moe_122b['baseline']) for t in moe_122b['dm16_tg']]
ax.plot(x, y8, 'D--', color='#4CAF50', label='122B-A10B MoE (7GPU) max=8', markersize=8, alpha=0.7)
ax.plot(x, y16, 'p--', color='#1B5E20', label='122B-A10B MoE (7GPU) max=16', markersize=8, alpha=0.7)

ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Draft Model Size (B params)', fontsize=12)
ax.set_ylabel('Speedup (%)', fontsize=12)
ax.set_title('Speculative Decoding Speedup: Dense vs MoE Main Models', fontsize=14)
ax.legend(fontsize=9, loc='lower left')
ax.set_xticks([0.8, 2, 4, 9])
ax.set_xticklabels(['0.8B', '2B', '4B', '9B'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-07_speculative_dense_speedup.png', dpi=150)
plt.close()

# === Graph 2: Accept Rate vs Draft Model Size ===
fig, ax = plt.subplots(figsize=(10, 6))

# 27B Dense
x = dense_27b['draft_sizes']
ax.plot(x, dense_27b['dm8_accept'], 'o-', color='#2196F3', label='27B Dense max=8', markersize=8)
ax.plot(x, dense_27b['dm16_accept'], 's-', color='#1565C0', label='27B Dense max=16', markersize=8)

# 35B-A3B MoE
x = moe_35b['draft_sizes']
ax.plot(x, moe_35b['dm8_accept'], '^--', color='#FF9800', label='35B-A3B MoE max=8', markersize=8, alpha=0.7)
ax.plot(x, moe_35b['dm16_accept'], 'v--', color='#E65100', label='35B-A3B MoE max=16', markersize=8, alpha=0.7)

# 122B-A10B MoE
x = moe_122b['draft_sizes']
ax.plot(x, moe_122b['dm8_accept'], 'D--', color='#4CAF50', label='122B-A10B MoE max=8', markersize=8, alpha=0.7)
ax.plot(x, moe_122b['dm16_accept'], 'p--', color='#1B5E20', label='122B-A10B MoE max=16', markersize=8, alpha=0.7)

ax.set_xlabel('Draft Model Size (B params)', fontsize=12)
ax.set_ylabel('Accept Rate (%)', fontsize=12)
ax.set_title('Accept Rate: Dense vs MoE Main Models (all <5%)', fontsize=14)
ax.legend(fontsize=9)
ax.set_xticks([0.8, 2, 4, 9])
ax.set_xticklabels(['0.8B', '2B', '4B', '9B'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-07_speculative_dense_accept_rate.png', dpi=150)
plt.close()

# === Graph 3: Speed Ratio vs Speedup Scatter ===
fig, ax = plt.subplots(figsize=(10, 6))

def plot_scatter(data, label_prefix, color, marker):
    for i, draft_size in enumerate(data['draft_sizes']):
        ratio = draft_speeds[draft_size] / data['baseline']
        for dm, tg_list, dm_label in [(8, data['dm8_tg'], 'max=8'), (16, data['dm16_tg'], 'max=16')]:
            sp = speedup(tg_list[i], data['baseline'])
            m = marker if dm == 8 else 's'
            ax.scatter(ratio, sp, color=color, marker=m, s=100,
                      alpha=0.8, zorder=5)
            ax.annotate(f'{data["drafts"][i]}\n{dm_label}',
                       (ratio, sp), textcoords="offset points",
                       xytext=(8, 5), fontsize=7, color=color)

plot_scatter(dense_27b, '27B Dense', '#1565C0', 'o')
plot_scatter(moe_35b, '35B-A3B MoE', '#E65100', '^')
plot_scatter(moe_122b, '122B-A10B MoE', '#1B5E20', 'D')

# Legend with proxy artists
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0', markersize=10, label='27B Dense (2GPU)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#E65100', markersize=10, label='35B-A3B MoE (2GPU)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#1B5E20', markersize=10, label='122B-A10B MoE (7GPU)'),
]

ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Speed Ratio (draft tg / main tg)', fontsize=12)
ax.set_ylabel('Speedup (%)', fontsize=12)
ax.set_title('Speed Ratio vs Speedup: All Configurations', fontsize=14)
ax.legend(handles=legend_elements, fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-07_speculative_dense_speed_ratio.png', dpi=150)
plt.close()

print("Graphs generated successfully.")
