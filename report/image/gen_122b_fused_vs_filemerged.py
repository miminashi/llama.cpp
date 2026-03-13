import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

labels = ['pp128', 'pp16384', 'tg32']
file_merged = [109.80, 158.21, 17.90]
gate_up_fused = [117.06, 170.17, 18.09]
file_merged_err = [1.15, 0.49, 0.02]
gate_up_fused_err = [1.32, 0.54, 0.01]

x = np.arange(len(labels))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x_pp = np.arange(2)
bars1 = ax1.bar(x_pp - width/2, file_merged[:2], width, yerr=file_merged_err[:2],
                label='file-merged (unfused)', color='#4A90D9', capsize=5)
bars2 = ax1.bar(x_pp + width/2, gate_up_fused[:2], width, yerr=gate_up_fused_err[:2],
                label='Gate+Up fused', color='#E8524A', capsize=5)

ax1.set_ylabel('tokens/s')
ax1.set_title('Qwen3.5-122B-A10B Q4_K_M 11GPU (7C+4R)\nPrompt Processing')
ax1.set_xticks(x_pp)
ax1.set_xticklabels(labels[:2])
ax1.legend()
ax1.set_ylim(0, max(gate_up_fused[:2]) * 1.2)

for bar, val, pct in zip([bars1[0], bars1[1], bars2[0], bars2[1]],
                          file_merged[:2] + gate_up_fused[:2],
                          [0, 0, 6.6, 7.6]):
    label = f'{val:.1f}'
    if pct > 0:
        label += f'\n(+{pct:.1f}%)'
    ax1.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=9)

bars3 = ax2.bar(x[0] - width/2, file_merged[2], width, yerr=file_merged_err[2],
                label='file-merged (unfused)', color='#4A90D9', capsize=5)
bars4 = ax2.bar(x[0] + width/2, gate_up_fused[2], width, yerr=gate_up_fused_err[2],
                label='Gate+Up fused', color='#E8524A', capsize=5)

ax2.set_ylabel('tokens/s')
ax2.set_title('Qwen3.5-122B-A10B Q4_K_M 11GPU (7C+4R)\nToken Generation')
ax2.set_xticks([x[0]])
ax2.set_xticklabels([labels[2]])
ax2.legend()
ax2.set_ylim(0, max(gate_up_fused[2], file_merged[2]) * 1.4)

for bar, val, pct in zip([bars3[0], bars4[0]], [file_merged[2], gate_up_fused[2]], [0, 1.1]):
    label = f'{val:.2f}'
    if pct > 0:
        label += f'\n(+{pct:.1f}%)'
    ax2.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('report/image/2026-03-09_122b_fused_vs_filemerged.png', dpi=150, bbox_inches='tight')
print('Saved')
