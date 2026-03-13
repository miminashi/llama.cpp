import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 35B data
labels_35b = ['pp128', 'pp512', 'pp2048', 'tg32']
unfused_35b = [231.61, 383.57, 431.09, 35.58]
fused_35b = [273.81, 457.83, 515.56, 36.07]

# 122B data
labels_122b = ['pp128', 'pp16384', 'tg32']
unfused_122b = [109.80, 158.21, 17.90]
fused_122b = [109.63, 158.00, 17.68]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 35B chart
x = np.arange(len(labels_35b))
w = 0.35
bars1 = axes[0].bar(x - w/2, unfused_35b, w, label='unfused (20.49 GiB)', color='#4c72b0')
bars2 = axes[0].bar(x + w/2, fused_35b, w, label='fused (19.71 GiB)', color='#dd8452')
axes[0].set_ylabel('Tokens/s')
axes[0].set_title('Qwen3.5-35B-A3B Q4_K_M (4GPU: 2C+2R)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_35b)
axes[0].legend()
for bar, val in zip(bars1, unfused_35b):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, fused_35b):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
# Add percentage annotations
for i, (u, f) in enumerate(zip(unfused_35b, fused_35b)):
    pct = (f - u) / u * 100
    axes[0].text(i, max(u, f) + 25, f'+{pct:.1f}%', ha='center', fontsize=9, fontweight='bold', color='green')

# 122B chart
x2 = np.arange(len(labels_122b))
bars3 = axes[1].bar(x2 - w/2, unfused_122b, w, label='unfused (71.27 GiB)', color='#4c72b0')
bars4 = axes[1].bar(x2 + w/2, fused_122b, w, label='fused (71.27 GiB)', color='#dd8452')
axes[1].set_ylabel('Tokens/s')
axes[1].set_title('Qwen3.5-122B-A10B Q4_K_M (11GPU: 7C+4R)')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(labels_122b)
axes[1].legend()
for bar, val in zip(bars3, unfused_122b):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars4, fused_122b):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}', ha='center', va='bottom', fontsize=8)
for i, (u, f) in enumerate(zip(unfused_122b, fused_122b)):
    pct = (f - u) / u * 100
    color = 'green' if pct >= 0 else 'red'
    sign = '+' if pct >= 0 else ''
    axes[1].text(i, max(u, f) + 8, f'{sign}{pct:.1f}%', ha='center', fontsize=9, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('/home/ubuntu/projects/llama.cpp/report/image/2026-03-09_unfused_vs_fused.png', dpi=150)
print('Graph saved.')
