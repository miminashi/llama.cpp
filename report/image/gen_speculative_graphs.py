#!/usr/bin/env python3
"""Generate speculative decoding benchmark graphs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from Phase 2 screening
# 35B-A3B baselines: pp=79.0, tg=39.1
# 122B-A10B baselines: pp=38.5, tg=18.9

# Draft model names and data
drafts_35b = ['0.8B', '0.8B', '2B', '2B']
maxvals_35b = [8, 16, 8, 16]
tg_35b = [33.9, 34.2, 30.4, 30.1]
accept_35b = [0.83, 0.36, 0.62, 0.36]
speedup_35b = [(t/39.1 - 1)*100 for t in tg_35b]

drafts_122b = ['0.8B', '0.8B', '2B', '2B', '4B', '4B']
maxvals_122b = [8, 16, 8, 16, 8, 16]
tg_122b = [20.6, 22.8, 19.1, 20.8, 15.6, 16.6]
accept_122b = [1.66, 0.74, 3.50, 0.53, 1.29, 0.31]
speedup_122b = [(t/18.9 - 1)*100 for t in tg_122b]

# Draft model standalone speeds
draft_names_all = ['0.8B', '2B', '4B', '9B', '35B-A3B']
draft_tg = [106.4, 77.5, 42.4, 29.7, 39.1]
speed_ratio_35b = [t/39.1 for t in draft_tg[:2]]  # only 0.8B, 2B
speed_ratio_122b = [t/18.9 for t in draft_tg]

# ============================================================
# Graph 1: Speedup vs Draft Model Size (grouped bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Group by draft model, show max=8 and max=16 side by side
# For 35B main
drafts_unique = ['0.8B', '2B']
x = np.arange(len(drafts_unique))
width = 0.18

# 35B max=8
vals_35b_m8 = [speedup_35b[0], speedup_35b[2]]
# 35B max=16
vals_35b_m16 = [speedup_35b[1], speedup_35b[3]]
# 122B max=8
vals_122b_m8 = [speedup_122b[0], speedup_122b[2]]
# 122B max=16
vals_122b_m16 = [speedup_122b[1], speedup_122b[3]]

bars1 = ax.bar(x - 1.5*width, vals_35b_m8, width, label='35B-A3B, max=8', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, vals_35b_m16, width, label='35B-A3B, max=16', color='#1565C0', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, vals_122b_m8, width, label='122B-A10B, max=8', color='#FF9800', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, vals_122b_m16, width, label='122B-A10B, max=16', color='#E65100', alpha=0.8)

# Add 4B bars for 122B only
x_4b = np.array([2])
ax.bar(x_4b + 0.5*width, [speedup_122b[4]], width, color='#FF9800', alpha=0.8)
ax.bar(x_4b + 1.5*width, [speedup_122b[5]], width, color='#E65100', alpha=0.8)

ax.set_xlabel('Draft Model', fontsize=12)
ax.set_ylabel('Speedup vs Baseline (%)', fontsize=12)
ax.set_title('Speculative Decoding: Speedup by Draft Model and Configuration', fontsize=13)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['0.8B', '2B', '4B'])
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)

plt.tight_layout()
plt.savefig('report/image/2026-03-07_speculative_speedup.png', dpi=150)
plt.close()

# ============================================================
# Graph 2: Accept Rate vs Draft Model Size
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(3)  # 0.8B, 2B, 4B
width = 0.18

# 35B accept rates (max=8 and 16)
ax.bar(x[:2] - 1.5*width, [accept_35b[0], accept_35b[2]], width,
       label='35B-A3B, max=8', color='#2196F3', alpha=0.8)
ax.bar(x[:2] - 0.5*width, [accept_35b[1], accept_35b[3]], width,
       label='35B-A3B, max=16', color='#1565C0', alpha=0.8)

# 122B accept rates
ax.bar(x + 0.5*width, [accept_122b[0], accept_122b[2], accept_122b[4]], width,
       label='122B-A10B, max=8', color='#FF9800', alpha=0.8)
ax.bar(x + 1.5*width, [accept_122b[1], accept_122b[3], accept_122b[5]], width,
       label='122B-A10B, max=16', color='#E65100', alpha=0.8)

ax.set_xlabel('Draft Model', fontsize=12)
ax.set_ylabel('Accept Rate (%)', fontsize=12)
ax.set_title('Speculative Decoding: Draft Token Accept Rate', fontsize=13)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['0.8B', '2B', '4B'])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report/image/2026-03-07_speculative_accept_rate.png', dpi=150)
plt.close()

# ============================================================
# Graph 3: Speed Ratio vs Speedup Scatter
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# All data points with speed ratio and speedup
# For 35B main model
for i, (d, m, su) in enumerate(zip(drafts_35b, maxvals_35b, speedup_35b)):
    sr = draft_tg[['0.8B','2B'].index(d)] / 39.1
    color = '#2196F3' if m == 8 else '#1565C0'
    marker = 'o' if m == 8 else 's'
    ax.scatter(sr, su, color=color, marker=marker, s=100, zorder=5,
               edgecolors='black', linewidths=0.5)
    ax.annotate(f'{d}\nmax={m}\n(35B)', (sr, su),
                textcoords="offset points", xytext=(10, 5), fontsize=8,
                color='#1565C0')

# For 122B main model
for i, (d, m, su) in enumerate(zip(drafts_122b, maxvals_122b, speedup_122b)):
    draft_idx = ['0.8B','2B','4B'].index(d)
    sr = draft_tg[draft_idx] / 18.9
    color = '#FF9800' if m == 8 else '#E65100'
    marker = 'o' if m == 8 else 's'
    ax.scatter(sr, su, color=color, marker=marker, s=100, zorder=5,
               edgecolors='black', linewidths=0.5)
    ax.annotate(f'{d}\nmax={m}\n(122B)', (sr, su),
                textcoords="offset points", xytext=(10, 5), fontsize=8,
                color='#E65100')

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even')
ax.set_xlabel('Speed Ratio (draft tg / target tg)', fontsize=12)
ax.set_ylabel('Speedup (%)', fontsize=12)
ax.set_title('Speed Ratio vs Speculative Decoding Speedup', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Add legend for markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=10, label='35B-A3B max=8'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#1565C0', markersize=10, label='35B-A3B max=16'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', markersize=10, label='122B-A10B max=8'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#E65100', markersize=10, label='122B-A10B max=16'),
    Line2D([0], [0], color='red', linestyle='--', label='Break-even'),
]
ax.legend(handles=legend_elements, fontsize=9)

plt.tight_layout()
plt.savefig('report/image/2026-03-07_speculative_speed_ratio.png', dpi=150)
plt.close()

print("All graphs generated successfully.")
