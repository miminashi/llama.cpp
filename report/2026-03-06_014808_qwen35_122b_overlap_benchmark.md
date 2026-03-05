# Qwen3.5-122B-A10B CUDA/RDMA Overlap ベンチマーク

- **実施日時**: 2026年3月6日 01:48
- **ワークツリー**: `.worktree/cuda-rdma-overlap` (commit `64a67bf25`)

## 前提・目的

前回のスケーリングベンチマーク ([2026-03-05 GPU スケーリング](2026-03-05_232127_cuda_rdma_overlap_gpu_scaling_benchmark.md)) で CUDA/RDMA Overlap の効果がモデル特性に依存することを確認:
- Qwen3.5-35B-A3B (MoE, 3B active, 19.8GB) → compute-bound → overlap 効果なし (-0.2%)
- GLM-4.7 (dense, 355B active, ~40GB) → communication-bound → overlap **+27.7%**

**Qwen3.5-122B-A10B** (MoE, 10B active, Q4_K_M, ~71.9GB) は両者の中間的モデル:
- MoE だが active パラメータが 3.3 倍 (10B vs 3B) → 計算量増
- モデルサイズが 3.6 倍 (71.9GB vs 19.8GB) → RDMA 転送データ量増
- compute/communication 比率は 35B-A3B と GLM-4.7 の間と予想

## テスト構成

| 項目 | 値 |
|------|------|
| モデル | Qwen3.5-122B-A10B Q4_K_M (~71.9GB, 3 shard) |
| GPU 構成 | 7 CUDA + 4 RDMA = 11GPU (176GB VRAM) |
| ベンチマーク | llama-bench (pp20000, tg32) |
| 手法 | ABAB Paired Design (2 pairs, r=5 each) |

### A/B 条件

| 条件 | 環境変数 |
|------|---------|
| A (Baseline) | `GGML_RDMA_SERVERS=192.168.100.2:50051` |
| B (Overlap) | `GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1 GGML_RDMA_SERVERS=192.168.100.2:50051` |

## 正確性テスト

~20,000 tokens 入力 (6500 行の `x = 1` + 質問) で Overlap モードの正常動作を確認:
- ガベージ/NaN/クラッシュ: なし
- 応答: 意味のある推論出力 (Thinking Process 分析)
- 速度: pp = 149.9 t/s, tg = 15.1 t/s (llama-cli 表示)

## ベンチマーク結果

### pp20000 (Prompt Processing)

| Pair | Baseline (A) | Overlap (B) | 差分 |
|:----:|:------------:|:-----------:|:----:|
| 1 | 136.72 | 137.39 | +0.67 |
| 2 | 136.43 | 137.67 | +1.24 |
| **平均** | **136.58** | **137.53** | **+0.95 (+0.70%)** |

- 対応あり t 検定: t=3.36, p=0.184 (有意でない, df=1)
- stddev: A1=0.79, B1=0.32, A2=0.50, B2=0.32

### tg32 (Token Generation)

| Pair | Baseline (A) | Overlap (B) | 差分 |
|:----:|:------------:|:-----------:|:----:|
| 1 | 17.54 | 17.68 | +0.13 |
| 2 | 17.38 | 17.71 | +0.34 |
| **平均** | **17.46** | **17.69** | **+0.23 (+1.35%)** |

- 対応あり t 検定: t=2.33, p=0.259 (有意でない, df=1)
- tg 退行なし (graph reuse によるフォールバック)

## モデル間比較

| モデル | サイズ | Active | タイプ | pp Overlap 効果 | GPU 構成 |
|:------:|:------:|:------:|:------:|:---------------:|:--------:|
| Qwen3.5-35B-A3B | 19.8GB | 3B | MoE | **-0.2%** (ns) | 2C+2R |
| **Qwen3.5-122B-A10B** | **71.9GB** | **10B** | **MoE** | **+0.70%** (ns) | **7C+4R** |
| GLM-4.7 IQ2_M | ~40GB | 355B (dense) | Dense | **+27.7%*** | 7C+4R |

## 考察

### Overlap 効果がほぼ中立 (+0.7%)

Qwen3.5-122B-A10B は 71.9GB と大きいが、MoE アーキテクチャにより active パラメータは 10B にとどまる。GPU 計算時間が支配的 (compute-bound) なため、CUDA/RDMA 並列化で隠蔽できる通信時間の割合が小さい。

### MoE vs Dense の特性差

- **MoE (35B-A3B, 122B-A10B)**: モデルサイズに対して計算量が少ない → compute 比率が高い → overlap 効果なし
- **Dense (GLM-4.7)**: 全パラメータが計算に参加 → communication 比率が高い → overlap で大幅改善

### 122B が 35B より微改善する理由

モデルサイズ 3.6 倍 (19.8GB → 71.9GB) により RDMA 転送量が増加。active パラメータは 3.3 倍 (3B → 10B) だが、MoE のルーティングオーバーヘッドも増加。結果として communication 比率がわずかに改善し、overlap の恩恵が微小に現れた可能性がある。ただし統計的に有意ではない。

### 結論

CUDA/RDMA Overlap は **communication-bound なモデル (dense, 大規模) でのみ有効**。MoE モデルはサイズが大きくても compute-bound のため効果は限定的。GLM-4.7 のような dense モデルが overlap の主要ターゲット。

## 再現方法

```bash
# 正確性テスト
python3 -c "print('Explain this code:\n' + 'x = 1\n' * 6500 + 'What does this code do? Answer in one sentence.')" > /tmp/overlap_test_prompt.txt

gpu-lock.sh run GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/cuda-rdma-overlap/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-122B-A10B-GGUF_Q4_K_M_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  -ngl 999 -fa 1 -f /tmp/overlap_test_prompt.txt -n 64 --log-file /tmp/llama-cli.log

# pp20000 ベンチマーク (A: Baseline)
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/cuda-rdma-overlap/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-122B-A10B-GGUF_Q4_K_M_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  -ngl 999 -fa 1 -p 20000 -n 0 -r 5 -o csv

# pp20000 ベンチマーク (B: Overlap)
gpu-lock.sh run GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/cuda-rdma-overlap/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-122B-A10B-GGUF_Q4_K_M_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  -ngl 999 -fa 1 -p 20000 -n 0 -r 5 -o csv

# tg32 ベンチマーク (同様に A/B で実行)
# -p 0 -n 32 に変更
```

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
