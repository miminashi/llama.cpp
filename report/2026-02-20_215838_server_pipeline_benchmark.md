# サーバーサイドパイプライン A/B ベンチマーク

- **実施日時**: 2026年2月20日 21:58
- **ワークツリー**: `.worktree/server-pipeline`
- **ブランチ**: `feature/server-pipeline`
- **コミット**: `09549aca0`
- **参照レポート**: [サーバーサイドパイプライン実装レポート](2026-02-20_203618_server_pipeline_implementation.md)

## 前提・目的

サーバーサイドパイプライン実装 (first-immediate, rest-batched) の性能効果を GLM-4.7 IQ2_M 11GPU 構成で定量評価する。

- **背景**: RDMA バックエンドの Generation フェーズで、D0 即送信・D1-D3 をバッチ化することで IB send/recv オーバーヘッド (~1.3ms) を削減する
- **目的**: パイプライン ON/OFF の A/B ベンチマークにより、pp128 と tg32 の変化を統計的に検証する
- **期待効果**: tg32 +1% (~1.3ms 削減)、pp128 中立〜小幅改善

## 実験条件

- **モデル**: GLM-4.7 IQ2_M (355B.A32B, 2.7 bpw, ~114GB)
- **構成**: 7 CUDA (1号機) + 4 RDMA (2号機) = 11GPU
- **分割**: `-sm layer`
- **実験計画**: ABAB Paired Design, n=5 ペア
- **条件A (OFF)**: `GGML_RDMA_NO_PIPELINE=1` (パイプライン無効)
- **条件B (ON)**: デフォルト (パイプライン有効)
- **サーバー再起動**: 不要 (パイプラインはクライアント側バッファリング、env var はプロセス起動時評価)

## 再現方法

```bash
DEV='CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]'
MODEL=/tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf
BENCH=build/bin/llama-bench

# デプロイ + サーバー再起動
gpu-lock.sh run bash scripts/rdma-deploy.sh
gpu-lock.sh run bash scripts/rdma-server.sh restart

# 条件A (OFF)
gpu-lock.sh run GGML_RDMA_NO_PIPELINE=1 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 $BENCH -m $MODEL -dev "$DEV" \
  -ngl 999 -sm layer -r 1 -p 128 -n 32 -o csv

# 条件B (ON)
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 $BENCH -m $MODEL -dev "$DEV" \
  -ngl 999 -sm layer -r 1 -p 128 -n 32 -o csv
```

ABAB 順序で 5 ペア (計 10 回) 実行。

## 生データ

### pp128 (tokens/sec)

| Pair | A (OFF) | B (ON) | Diff (B-A) |
|:----:|--------:|-------:|-----------:|
| 1 | 23.393 | 23.341 | -0.052 |
| 2 | 23.415 | 23.374 | -0.041 |
| 3 | 23.364 | 23.347 | -0.017 |
| 4 | 23.405 | 23.310 | -0.095 |
| 5 | 23.366 | 23.320 | -0.046 |

### tg32 (tokens/sec)

| Pair | A (OFF) | B (ON) | Diff (B-A) |
|:----:|--------:|-------:|-----------:|
| 1 | 7.770 | 7.810 | +0.040 |
| 2 | 7.783 | 7.807 | +0.024 |
| 3 | 7.773 | 7.814 | +0.041 |
| 4 | 7.685 | 7.815 | +0.130 |
| 5 | 7.752 | 7.731 | -0.021 |

## 統計分析

### pp128

| 統計量 | 値 |
|--------|---:|
| Mean A (OFF) | 23.389 t/s |
| Mean B (ON) | 23.338 t/s |
| Mean diff (B-A) | -0.050 t/s |
| SD of diffs | 0.028 |
| t-statistic | -3.96 |
| df | 4 |
| p-value (two-tailed) | **0.017** |
| % change | **-0.21%** |
| Cohen's d | -1.77 |
| 95% CI | [-0.085, -0.015] |

**有意** (p = 0.017 < 0.05)。パイプライン ON で pp128 に **-0.21%** の微小回帰。

### tg32

| 統計量 | 値 |
|--------|---:|
| Mean A (OFF) | 7.753 t/s |
| Mean B (ON) | 7.795 t/s |
| Mean diff (B-A) | +0.043 t/s |
| SD of diffs | 0.055 |
| t-statistic | +1.75 |
| df | 4 |
| p-value (two-tailed) | **0.156** |
| % change | **+0.55%** |
| Cohen's d | +0.78 |
| 95% CI | [-0.025, +0.111] |

**非有意** (p = 0.156 > 0.05)。改善傾向 (+0.55%) だが CI がゼロを含む。

### ドリフトチェック

A 条件の時系列推移:

| Pair | pp128 A | tg32 A |
|:----:|--------:|-------:|
| 1 | 23.393 | 7.770 |
| 2 | 23.415 | 7.783 |
| 3 | 23.364 | 7.773 |
| 4 | 23.405 | 7.685 |
| 5 | 23.366 | 7.752 |

- pp128: 安定 (range 0.051, <0.3%)
- tg32: Pair 4 の A が 7.685 で他より低い (サーマルスロットリングの可能性)。この外れ値が Pair 4 の差分 (+0.130) を膨張させている

## 結果サマリー

| Metric | A (OFF) | B (ON) | % Change | p-value | Cohen's d | 有意? |
|--------|:-------:|:------:|:--------:|:-------:|:---------:|:-----:|
| pp128 | 23.389 | 23.338 | **-0.21%** | 0.017 | -1.77 | Yes (回帰) |
| tg32 | 7.753 | 7.795 | **+0.55%** | 0.156 | +0.78 | No |

## 考察

1. **pp128 の微小回帰 (-0.21%)**: 統計的に有意だが実用上無視可能な水準 (0.05 t/s)。パイプラインバッファのメモリ確保・解放オーバーヘッドが原因と推定。pp128 では RDMA Write が支配的でコマンドバッチ化の効果がないため、純粋にオーバーヘッドが見える。

2. **tg32 の改善傾向 (+0.55%)**: 期待 +1% に対し +0.55%。Cohen's d = 0.78 (中〜大効果) だが、Pair 4 外れ値の影響でばらつきが大きく非有意。Pair 4 を除くと差分は {+0.040, +0.024, +0.041, -0.021} で mean = +0.021 (+0.27%) にまで縮小する。

3. **期待効果との乖離**: 理論上の ~1.3ms 削減に対し、実測では 0.55% ≈ 0.7ms 程度。パイプラインメッセージのパースオーバーヘッドや、共有接続でのメッセージバッファリング自体のコストが削減効果を相殺していると考えられる。

## 判定

パイプラインの効果は tg32 で期待より小さく非有意、pp128 で微小回帰が有意。

**推奨**: デフォルト ON を維持。pp128 回帰は実用上無視可能 (-0.05 t/s)、tg32 改善は追加検証 (n=10 以上) で確認の余地あり。効果が確認できない場合はデフォルト OFF への変更を検討。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `f5a631535 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 37°C | 37°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 85689) |

GGML_RDMA 環境変数: (none)
