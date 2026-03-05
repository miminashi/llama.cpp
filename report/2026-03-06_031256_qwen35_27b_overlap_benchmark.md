# Qwen3.5-27B CUDA/RDMA Overlap ベンチマーク

- **実施日時**: 2026年3月6日 03:12
- **ワークツリー**: `.worktree/cuda-rdma-overlap` (commit `64a67bf25`)

## 前提・目的

CUDA/RDMA Overlap の効果がモデル特性（MoE/dense, モデルサイズ）でどう変わるかを検証する。
既に3モデルの結果が得られており、4つ目のデータポイントとして **Qwen3.5-27B** (dense, 27B active) を追加する。

- **背景**: overlap 効果はモデルの compute/communication バランスに依存
  - MoE (少ない active params) → compute-bound → overlap 効果小
  - Dense (全パラメータ active) → communication-bound → overlap 効果大
- **目的**: Dense だが小サイズ (16GB) のモデルでの overlap 効果を計測
- **参照レポート**:
  - [Qwen3.5-122B overlap](report/2026-03-06_014808_qwen35_122b_overlap_benchmark.md)
  - [CUDA/RDMA overlap GLM-4.7](report/2026-03-05_163449_cuda_rdma_overlap_benchmark.md)
  - [CUDA/RDMA overlap GPU scaling](report/2026-03-05_232127_cuda_rdma_overlap_gpu_scaling_benchmark.md)

## テスト構成

| 項目 | 値 |
|------|-----|
| モデル | Qwen3.5-27B Q4_K_M (dense, 27B params, ~16GB) |
| GPU 構成 | 7 CUDA + 4 RDMA (11GPU) |
| バイナリ | `.worktree/cuda-rdma-overlap/build/bin/llama-bench` |
| オプション | `-ngl 999 -fa 1` |

### A/B 条件

| 条件 | 環境変数 |
|------|----------|
| A (Baseline) | `GGML_RDMA_SERVERS=192.168.100.2:50051` |
| B (Overlap) | `GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1 GGML_RDMA_SERVERS=192.168.100.2:50051` |

## 再現方法

### 正確性テスト
```bash
gpu-lock.sh run GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/cuda-rdma-overlap/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-27B-GGUF_Qwen3.5-27B-Q4_K_M.gguf \
  -ngl 999 -fa 1 -f /tmp/overlap_test_prompt.txt -n 64 --log-file /tmp/llama-cli.log
```

### pp20000 ベンチマーク (ABAB × 2 pairs, r=5)
```bash
# A: Baseline
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/cuda-rdma-overlap/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-27B-GGUF_Qwen3.5-27B-Q4_K_M.gguf \
  -ngl 999 -fa 1 -p 20000 -n 0 -r 5 -o csv

# B: Overlap
gpu-lock.sh run GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/cuda-rdma-overlap/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-27B-GGUF_Qwen3.5-27B-Q4_K_M.gguf \
  -ngl 999 -fa 1 -p 20000 -n 0 -r 5 -o csv
```

### tg32 ベンチマーク (ABAB × 2 pairs, r=5)
```bash
# 同じコマンドで -p 0 -n 32 に変更
```

## 結果

### 正確性テスト
Overlap モードで正常な推論出力を確認。pp=113.9 t/s, tg=9.6 t/s (llama-cli, ~20000 token prompt)。

### pp20000 A/B 比較

| Run | Baseline (t/s) | Overlap (t/s) | Diff |
|:---:|:-:|:-:|:-:|
| Pair 1 | 119.395 | 119.882 | +0.487 |
| Pair 2 | 119.368 | 120.093 | +0.725 |
| **Mean** | **119.382** | **119.988** | **+0.606** |

- **効果**: **+0.51%** (t=5.09, p=0.123, ns)
- 方向は一貫して正だが、n=2 pairs のため統計的有意に至らず

### tg32 退行チェック

| Run | Baseline (t/s) | Overlap (t/s) | Diff |
|:---:|:-:|:-:|:-:|
| Pair 1 | 10.079 | 10.113 | +0.034 |
| Pair 2 | 10.082 | 10.111 | +0.029 |
| **Mean** | **10.081** | **10.112** | **+0.032** |

- **効果**: **+0.31%** (t=12.60, p=0.050)
- 退行なし

## 全モデル CUDA/RDMA Overlap 効果マップ

| モデル | サイズ | Active Params | タイプ | pp20000 効果 | tg32 効果 |
|:------:|:------:|:------:|:------:|:------:|:------:|
| Qwen3.5-35B-A3B | 19.8GB | 3B | MoE | -0.2% (ns) | — |
| Qwen3.5-122B-A10B | 71.9GB | 10B | MoE | +0.70% (ns) | +0.26% (ns) |
| **Qwen3.5-27B** | **16GB** | **27B** | **Dense** | **+0.51% (ns)** | **+0.31%** |
| GLM-4.7 IQ2_M | ~40GB | 355B | Dense | +27.7%*** | neutral |

## 分析

### Overlap 効果の決定要因

Qwen3.5-27B は dense モデルだが、overlap 効果は **+0.51%** と GLM-4.7 (+27.7%) に比べて非常に小さい。

**主な原因: モデルサイズの小ささ**

- Qwen3.5-27B は 16GB と小さく、11GPU (176GB VRAM) に対して余裕がある
- 各デバイスに割り当てられるレイヤー数が少なく、RDMA 転送量も少ない
- RDMA 通信時間が短いため、CUDA 計算とオーバーラップしても改善幅が小さい

**Overlap 効果のスケーリング法則**:

| 因子 | 効果への影響 | 根拠 |
|------|:---:|------|
| モデルタイプ (dense vs MoE) | **大** | Dense は全パラメータが通信対象 → communication-bound |
| Active パラメータ数 | **大** | Active params が大きいほど RDMA 転送量が増加 |
| モデルサイズ (総量) | **中** | サイズが大きいほど各デバイスの負荷が増え overlap 余地が拡大 |
| GPU 数 | 小 | GPU 増加はレイヤー分散を薄くし、per-device 通信量を減少 |

### 結論

Overlap 効果は **dense + 大規模モデル** で最大化される。Dense であってもモデルサイズが小さければ（Qwen3.5-27B: 16GB）効果は限定的。GLM-4.7 (40GB, 355B active) のような **大規模 dense モデル** が overlap の主要ターゲットとなる。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c1ca6d5fd (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 32°C | 33°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1502018) |

GGML_RDMA 環境変数:
- (none)
