#!/usr/bin/env python3
"""Tier 1 benchmark graphs for Qwen3.5-35B and 122B experiments."""

import matplotlib.pyplot as plt
import numpy as np

# ---- Graph 1: 1-A ubatch saturation curve (35B, 4GPU, pp=ub) ----
ub_35b = [512, 1024, 2048, 4096, 8192]
pp_35b = [386.14, 459.81, 504.02, 487.65, 467.41]
pp_35b_err = [0.91, 0.14, 2.95, 2.01, 1.19]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(ub_35b, pp_35b, yerr=pp_35b_err, marker='o', linewidth=2,
            capsize=4, label='Qwen3.5-35B (4GPU, 2C+2R)')
ax.set_xscale('log', base=2)
ax.set_xticks(ub_35b)
ax.set_xticklabels([str(x) for x in ub_35b])
ax.set_xlabel('ubatch size')
ax.set_ylabel('Prompt throughput (t/s)')
ax.set_title('ubatch Size vs PP Throughput (Qwen3.5-35B, 4GPU)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('report/image/2026-03-08_tier1_ubatch_35b.png', dpi=150)
plt.close()

# ---- Graph 2: 1-C GPU scaling (122B) ----
gpus = [6, 7, 8, 11]
configs = ['6C', '7C', '4C+4R', '7C+4R']
pp128_122b = [103.37, 103.41, 102.41, 102.55]
pp512_122b = [164.19, 163.45, 162.19, 160.69]
tg32_122b = [18.98, 18.96, 17.40, 17.70]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(gpus, pp128_122b, 'o-', linewidth=2, label='pp128')
ax1.plot(gpus, pp512_122b, 's-', linewidth=2, label='pp512')
ax1.set_xlabel('GPU count')
ax1.set_ylabel('Prompt throughput (t/s)')
ax1.set_title('GPU Scaling: PP (Qwen3.5-122B)')
ax1.set_xticks(gpus)
ax1.set_xticklabels([f'{g}\n({c})' for g, c in zip(gpus, configs)])
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(gpus, tg32_122b, 'o-', linewidth=2, color='tab:green', label='tg32')
ax2.set_xlabel('GPU count')
ax2.set_ylabel('Generation throughput (t/s)')
ax2.set_title('GPU Scaling: TG (Qwen3.5-122B)')
ax2.set_xticks(gpus)
ax2.set_xticklabels([f'{g}\n({c})' for g, c in zip(gpus, configs)])
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('report/image/2026-03-08_tier1_gpu_scaling_122b.png', dpi=150)
plt.close()

# ---- Graph 3: 1-D ubatch saturation curve (122B, 11GPU) ----
ub_122b = [256, 512, 1024, 2048, 4096, 8192]
pp_122b = [130.70, 161.56, 199.37, 231.50, 257.97, 265.23]
pp_122b_err = [0.16, 0.26, 0.16, 0.50, 0.37, 0.21]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(ub_122b, pp_122b, yerr=pp_122b_err, marker='o', linewidth=2,
            capsize=4, label='122B (11GPU, 7C+4R)', color='tab:blue')
# Mark saturation point
ax.axvline(x=8192, color='tab:red', linestyle=':', alpha=0.5, label='saturation (ub=8192)')
ax.annotate('OOM', xy=(16384, 265), fontsize=10, color='tab:red',
            ha='center', va='bottom')
ax.plot(16384, 265, 'x', color='tab:red', markersize=12, markeredgewidth=2)
ax.set_xscale('log', base=2)
ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384])
ax.set_xticklabels(['256', '512', '1024', '2048', '4096', '8192', '16384'])
ax.set_xlabel('ubatch size')
ax.set_ylabel('Prompt throughput (t/s)')
ax.set_title('ubatch Saturation Curve (Qwen3.5-122B, 11GPU)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('report/image/2026-03-08_tier1_ubatch_122b.png', dpi=150)
plt.close()

# ---- Graph 4: 1-D ubatch comparison (35B vs 122B, normalized) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: absolute throughput
ax1.errorbar(ub_122b, pp_122b, yerr=pp_122b_err, marker='o', linewidth=2,
             capsize=4, label='122B (11GPU, 7C+4R)')
ub_35b_full = [512, 1024, 2048, 4096, 8192]
pp_35b_full = [386.14, 459.81, 504.02, 487.65, 467.41]
pp_35b_full_err = [0.91, 0.14, 2.95, 2.01, 1.19]
ax1.errorbar(ub_35b_full, pp_35b_full, yerr=pp_35b_full_err, marker='s', linewidth=2,
             capsize=4, label='35B (4GPU, 2C+2R)', linestyle='--')
ax1.set_xscale('log', base=2)
ax1.set_xticks([256, 512, 1024, 2048, 4096, 8192])
ax1.set_xticklabels(['256', '512', '1024', '2048', '4096', '8192'])
ax1.set_xlabel('ubatch size')
ax1.set_ylabel('Prompt throughput (t/s)')
ax1.set_title('Absolute Throughput: 35B vs 122B')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: normalized to ub=512 baseline
norm_122b = [x / 161.56 for x in pp_122b]
norm_35b = [x / 386.14 for x in pp_35b_full]
ax2.plot(ub_122b, norm_122b, 'o-', linewidth=2, label='122B (11GPU)')
ax2.plot(ub_35b_full, norm_35b, 's--', linewidth=2, label='35B (4GPU)')
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xscale('log', base=2)
ax2.set_xticks([256, 512, 1024, 2048, 4096, 8192])
ax2.set_xticklabels(['256', '512', '1024', '2048', '4096', '8192'])
ax2.set_xlabel('ubatch size')
ax2.set_ylabel('Normalized throughput (ub=512 = 1.0)')
ax2.set_title('Normalized Saturation: 35B vs 122B')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('report/image/2026-03-08_tier1_ubatch_comparison.png', dpi=150)
plt.close()

print("All graphs saved to report/image/")
