# Signal Interval 修正 + Qwen3.5 A/B ベンチマーク

- **実施日時**: 2026年3月2日 04:51
- **ワークツリー**: `.worktree/double-buffering`
- **コミット**: `fa8f7c6a6` (fix(rdma): limit signal interval to 2 for double-buffered paths)
- **前回レポート**: [report/2026-03-02_040542_staging_buffer_double_buffering.md](2026-03-02_040542_staging_buffer_double_buffering.md)

## 前提・目的

前回の double-buffering 実装 (`ec2fb5a89`) により staging/send バッファが 2 個に増設されたが、signal interval の制御に正確性バグが残っていた:

1. **RDMA Write** (`set_tensor`): signal 条件が `is_last` のみ。3 チャンク以上の転送で unsignaled WR が 2 個以上飛行中になり、バッファ A が DMA 完了前に再利用される
2. **Send**: `RDMA_SIGNAL_INTERVAL=64` だが内部バッファは 2 つ。GLM-4.7 の Send/Recv フォールバック (> 4GB テンソル) で多数の 16MB チャンク送信時にバッファ再利用レース発生

### 修正内容

- **RDMA Write** (`ggml-rdma.cpp`): `unsignaled_count` を追加し、2 チャンクごとに signal (最大 1 unsignaled WR が飛行中)
- **Send** (`rdma-transport.cpp`): 内部バッファ使用時は `RDMA_IB_SIGNAL_INTERVAL=2`、外部 MR/inline 時は従来の `RDMA_SIGNAL_INTERVAL=64` を使用

### 安全性の根拠

IB 順序保証: WR A (unsignaled) → WR B (signaled) → wait(B) → A も完了保証。バッファ 2 個で interval=2 なら、バッファ A の DMA 中にバッファ B を使用 → B を signal → wait → A 完了 → A を安全に再利用。

## 再現方法

### ビルド・デプロイ

```bash
bash .worktree/double-buffering/scripts/rdma-build.sh local
gpu-lock.sh run bash .worktree/double-buffering/scripts/rdma-deploy.sh
bash .worktree/double-buffering/scripts/rdma-server.sh start
```

### 正常性テスト

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=5,6 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/double-buffering/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -p "Hello, how are you?" -n 50 --log-file /tmp/llama-cli.log
```

結果: pp=54.6 t/s, tg=33.6 t/s で正常動作確認。

### A/B ベンチマーク

ABAB Paired Design 5 ペア + ウォームアップ 1 回:
- **A**: `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` (selective signaling OFF = 全 WR signaled)
- **B**: デフォルト (selective signaling ON + double-buffering + signal interval=2)

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=5,6 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/double-buffering/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -dev 'CUDA0/CUDA1/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -fa 1 -r 1 -p 128 -n 32 -o csv
```

## 結果

### pp128 (t/s)

| Pair | A (signaling OFF) | B (signaling ON) | B-A |
|------|-------------------:|------------------:|------:|
| 1 | 202.52 | 202.63 | +0.11 |
| 2 | 202.96 | 202.52 | -0.44 |
| 3 | 202.12 | 202.64 | +0.52 |
| 4 | 202.56 | 202.45 | -0.11 |
| 5 | 202.35 | 202.79 | +0.44 |
| **平均** | **202.50** | **202.61** | **+0.10** |

差分: **+0.05%** (有意差なし)

### tg32 (t/s)

| Pair | A (signaling OFF) | B (signaling ON) | B-A |
|------|-------------------:|------------------:|------:|
| 1 | 33.82 | 33.83 | +0.01 |
| 2 | 33.83 | 34.10 | +0.27 |
| 3 | 33.60 | 33.89 | +0.29 |
| 4 | 33.74 | 33.67 | -0.07 |
| 5 | 33.80 | 34.53 | +0.73 |
| **平均** | **33.76** | **34.01** | **+0.25** |

差分: **+0.74%** (B5 が外れ値の可能性。統計的に有意ではない見込み)

## 考察

### Qwen3.5 6GPU で差が出ない理由

1. **RDMA Write (set_tensor)**: Qwen3.5 のテンソルは通常 16MB 以下で single-chunk 転送。`is_last=true` で常に signaled されるため、`unsignaled_count` は発火しない
2. **Send (graph_compute)**: コマンドメッセージは小さく (< 256B)、inline 送信される。内部バッファを使わないため `RDMA_IB_SIGNAL_INTERVAL` は適用されない

### GLM-4.7 での期待効果

GLM-4.7 は > 4GB のバッファを持つため Send/Recv フォールバックを使用。16MB チャンクでの連続送信時に signal interval=2 が発火し、バッファ再利用レースを防止する。これにより:
- **正確性**: レース由来のゴミ文字出力 (`>>9-A($%,...`) を防止
- **性能**: always-signal (interval=1) より 50% の signaled WR を削減し、pp128 改善の可能性

### 修正の性質

本修正は**正確性修正** (correctness fix) であり、性能最適化ではない。double-buffering でバッファ数を 2 に増やしたのに signal interval がバッファ数に連動していなかった設計上のバグを修正。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `fa8f7c6a6 (feature/double-buffering)` | -- |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 34C | 35C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | -- | running (PID 1161251) |
| テスト GPU 構成 | CUDA5,6 | RDMA0-3 |
| モデル | Qwen3.5-35B-A3B UD-Q4_K_M | -- |

## 次のステップ

- GLM-4.7 IQ2_M 11GPU (7C+4R) でのベンチマーク: Send/Recv フォールバックパスの signal interval=2 の効果測定
- GLM-4.7 で always-signal (interval=1) vs interval=2 の pp128 比較: selective signaling 回復度合いの確認
