import matplotlib.pyplot as plt
import numpy as np

pp_sizes = ['pp128', 'pp512', 'pp2048']

pre_merge = [272.09, 459.07, 518.91]
post_merge = [271.52, 457.90, 467.76]
scratch_copy = [272.27, 459.52, 519.86]

x = np.arange(len(pp_sizes))
width = 0.25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

bars1 = ax1.bar(x - width, post_merge, width, label='Post-merge (fusion disabled)', color='#e74c3c', alpha=0.85)
bars2 = ax1.bar(x, scratch_copy, width, label='Scratch copy fusion (fix)', color='#2ecc71', alpha=0.85)
bars3 = ax1.bar(x + width, pre_merge, width, label='Pre-merge (baseline)', color='#3498db', alpha=0.85)

ax1.set_xlabel('Prompt Size')
ax1.set_ylabel('Throughput (t/s)')
ax1.set_title('PP Throughput: RDMA 2C+2R (Qwen3.5 35B A3B Q4_K_M fused)')
ax1.set_xticks(x)
ax1.set_xticklabels(pp_sizes)
ax1.legend(loc='upper left')
ax1.set_ylim(0, 600)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

improvement = [(s - p) / p * 100 for s, p in zip(scratch_copy, post_merge)]
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in improvement]
bars = ax2.bar(pp_sizes, improvement, color=colors, alpha=0.85)
ax2.set_xlabel('Prompt Size')
ax2.set_ylabel('Improvement (%)')
ax2.set_title('Scratch Copy Fusion vs Fusion Disabled')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

for bar, val in zip(bars, improvement):
    height = bar.get_height()
    ax2.annotate(f'{val:+.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('report/image/2026-03-08_topk_moe_scratch_copy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Done")
