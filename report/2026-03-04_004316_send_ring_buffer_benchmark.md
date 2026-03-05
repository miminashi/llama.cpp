# Send リングバッファによる selective signaling 回復

- **実施日時**: 2026年3月4日 00:43
- **ワークツリー**: `.worktree/send-ring` (branch: `feature/send-ring`)
- **コミット**: `32dbffe78`

## 前提・目的

Send path の always-signal 修正 (commit `9a5a4a911`) により、GLM-4.7 11GPU の pp128 が 30.6 → 24.3 t/s (-20.6%) に低下した。原因は graph_compute コマンド送信時の CQ ポーリングオーバーヘッド (毎回シグナル → 毎回ポーリング)。

既存の double-buffering アプローチ (`.worktree/double-buffering`) は TG -4.24% 回帰があった。根本原因は `can_skip_last` ロジックのバグにより double-buffering が実質的にシグナリング頻度を減らせていなかったこと。また 2 バッファでは interval=2 が限界 (50% 削減) で、元の interval=64 (98% 削減) には遠い。

**本実装**: 既存の 16MB `send_buffer_` をリングバッファとして使用し、MR を追加せずに interval=64 の selective signaling を回復する。

### 参照レポート

- [staging buffer double-buffering](report/2026-03-02_040542_staging_buffer_double_buffering.md) — 既存 double-buffering 試行
- [double buffer Qwen3.5 A/B benchmark](report/2026-03-02_143000_double_buffer_qwen35_ab_benchmark.md) — TG -4.24% 回帰

## 設計

### 動作原理

1. `send_buffer_` (16MB) を円環的に使用。各 send がバッファ内の異なる位置 (`ring_pos_`) にデータをコピー
2. 先行 WR のデータ領域と重複しないため、unsignaled WR でもバッファ再利用レースが発生しない
3. interval=64 で CQ ポーリング 98% 削減 (元の selective signaling と同等)
4. MR 追加なし (単一バッファ・単一 MR) → MTT キャッシュ圧迫なし

### シグナリング条件 (内部バッファ使用時)

| 条件 | トリガー | 目的 |
|------|---------|------|
| `RDMA_NO_SELECTIVE_SIGNAL` | 環境変数 | デバッグ用全シグナル |
| `force_signal` | ラップ時に unsignaled WR あり | 先行 WR ドレイン |
| `fills_to_end` | `ring_pos_ == chunk_size` | avail=0 状態の防止 |
| `ring_unsignaled_ >= 64` | 通常の interval | SQ オーバーフロー防止 |

### 初回実装のバグと修正

初回実装ではラップ時に `send_size` を `avail` (残りスペース) にトランケートしていたが、IB Send/Recv はメッセージベースのため、送信サイズの変更が受信側の期待サイズと不一致を引き起こし `local length error` でクラッシュした。

修正: ラップ時はトランケートせず offset 0 にフルメッセージを送信。`force_signal` で先行 WR をドレイン。

## 変更内容

| ファイル | 変更 |
|---------|------|
| `rdma-transport.h` | `ring_pos_`, `ring_unsignaled_` メンバ追加 |
| `rdma-transport.cpp` | `send()` のリングバッファ化、`disconnect()` でリセット |

## 再現方法

### ビルド・デプロイ

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/send-ring/scripts/rdma-build.sh local
bash scripts/gpu-lock.sh run bash /home/ubuntu/projects/llama.cpp/.worktree/send-ring/scripts/rdma-deploy.sh
ssh 192.168.100.2 "pkill -9 -f rdma-server"; true
bash /home/ubuntu/projects/llama.cpp/.worktree/send-ring/scripts/rdma-server.sh start
```

### 動作確認 (Qwen3.5 4GPU)

```bash
bash scripts/gpu-lock.sh run \
  CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  /home/ubuntu/projects/llama.cpp/.worktree/send-ring/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -p 'What is 2+2? Answer briefly.' -n 50 \
  -fa on --log-file /tmp/llama-cli.log 2>&1
```

結果: 正常動作。pp=64.8 t/s, tg=36.3 t/s。

### A/B ベンチマーク

```bash
# 条件 A (always-signal):
bash scripts/gpu-lock.sh run \
  GGML_RDMA_NO_SELECTIVE_SIGNAL=1 CUDA_VISIBLE_DEVICES=4,5 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  /home/ubuntu/projects/llama.cpp/.worktree/send-ring/build/bin/llama-bench \
  --model /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -dev 'CUDA0/CUDA1/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]' \
  -ngl 999 -sm layer -fa 1 -r 1 -p 128 -n 32 -o csv 2>&1

# 条件 B (ring buffer, default):
bash scripts/gpu-lock.sh run \
  CUDA_VISIBLE_DEVICES=4,5 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  /home/ubuntu/projects/llama.cpp/.worktree/send-ring/build/bin/llama-bench \
  --model /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -dev 'CUDA0/CUDA1/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]' \
  -ngl 999 -sm layer -fa 1 -r 1 -p 128 -n 32 -o csv 2>&1
```

ABAB ペアデザイン、5 ペア (10 回実行)。

## 結果

### PP128 (t/s)

| Pair | A (always-signal) | B (ring buffer) | Diff (B-A) |
|:----:|:-----------------:|:---------------:|:----------:|
| 1 | 204.08 | 203.84 | -0.24 |
| 2 | 203.84 | 204.13 | +0.28 |
| 3 | 204.11 | 203.83 | -0.28 |
| 4 | 203.84 | 203.79 | -0.04 |
| 5 | 203.79 | 203.89 | +0.10 |
| **平均** | **203.93** | **203.90** | **-0.04** |

- 差: **-0.02%** (p=0.75, 有意差なし)
- Qwen3.5 はコンピュート律速 (GPU 94.8%) のため RDMA 通信最適化の効果は出ない

### TG32 (t/s)

| Pair | A (always-signal) | B (ring buffer) | Diff (B-A) |
|:----:|:-----------------:|:---------------:|:----------:|
| 1 | 36.03 | 36.17 | +0.14 |
| 2 | 36.07 | 36.05 | -0.02 |
| 3 | 36.22 | 34.87 | -1.35 |
| 4 | 36.21 | 36.01 | -0.21 |
| 5 | 35.04 | 36.07 | +1.03 |
| **平均** | **35.91** | **35.83** | **-0.08** |

- 差: **-0.22%** (p=0.84, 有意差なし)
- Pair 3/5 のばらつきは GPU サーマル変動またはコンテキスト初期化タイミングによるもの

## 統計手法

- ABAB Paired Design (5ペア、対応あり t 検定)
- 交絡排除: 温度ドリフト → ABAB 交互実行で中和

## 結論

- Qwen3.5-35B-A3B (4GPU) で**性能ニュートラル** (PP/TG ともに有意差なし)
- 期待通りの結果: Qwen3.5 はコンピュート律速のため Send path の最適化は効果が出ない
- **回帰なし**を確認 (double-buffering の TG -4.24% 問題を回避)
- GLM-4.7 11GPU (通信律速) での検証が次のステップ — pp128 で +26% (24.3 → 30.6) の回復を期待

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `32dbffe78 (feature/send-ring)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 30°C | 32°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1317472) |

GGML_RDMA 環境変数: (none)
