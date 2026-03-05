# Staging Buffer Double-Buffering による PP 性能改善

- **実施日時**: 2026年3月2日 04:05
- **ワークツリー**: `.worktree/double-buffering`
- **ブランチ**: `feature/double-buffering`
- **コミット**: `ec2fb5a89`

## 前提・目的

### 背景

バッファ再利用レースコンディション修正 (commit `e5700e0f8`) により、RDMA Write の staging buffer を常時シグナルに変更した。これにより selective signaling が実質無効化され、pp128 が GLM-4.7 IQ2_M 11GPU で ~29.4 → ~24.3 t/s に低下 (-17.3%)。

**根本原因**: `rdma_staging_buffer` が単一の再利用バッファであるため、unsignaled RDMA Write の DMA 完了前に次の `memcpy` がバッファを上書き → データ破損。常時シグナルは安全だが DMA と memcpy のパイプライン化を阻害。

同様に `rdma_connection::send()` の内部バッファ `send_buffer_` も単一再利用バッファで、`uses_internal_buffer` フラグによる常時シグナルが必要だった。

### 目的

Staging buffer と send buffer を double-buffering 化し、バッファ A の DMA 中にバッファ B に memcpy することで、selective signaling を安全に再有効化する。

### 参照レポート

- [RDMA バッファレースコンディション修正](report/2026-02-23_020143_rdma_write_signal_race_condition_fix.md)
- [Send バッファレースコンディション修正](report/2026-02-23_094039_send_buffer_race_condition_fix.md)

## 実装内容

### 1. `rdma_staging_buffer` double-buffering (`rdma-memory.h/cpp`)

- `info_` を `info_[2]` (配列) に変更
- コンストラクタで 2 つのバッファを確保
- `get_buffer()` が `info_[current_idx_]` を返す
- `advance()` メソッドで `current_idx_ ^= 1` によりバッファ切替
- サイズ超過時は両方を再確保 (サイズ統一)

### 2. `set_tensor` RDMA Write ループ (`ggml-rdma.cpp`)

- 常時 `signaled=true` を `is_last` ベースの selective signaling に復帰
- `rdma_write()` 後に `ctx->staging->advance()` でバッファ切替
- 最終チャンクのみ signaled (IB 順序保証で先行 WR も完了)

### 3. `send_buffer_` double-buffering (`rdma-transport.h/cpp`)

- `send_buffer_` を `send_buffers_[2]`、`send_mr_` を `send_mrs_[2]` に変更
- `setup_qp()` で両方初期化・登録
- `send()` で `send_buf_idx_` を使い交互バッファ選択
- posting 後に `send_buf_idx_ ^= 1` で切替
- signal 条件から `uses_internal_buffer` を削除

## 再現方法

### ビルド・デプロイ

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/double-buffering/scripts/rdma-build.sh local
bash scripts/gpu-lock.sh run bash /home/ubuntu/projects/llama.cpp/.worktree/double-buffering/scripts/rdma-deploy.sh
bash /home/ubuntu/projects/llama.cpp/.worktree/double-buffering/scripts/rdma-server.sh restart
```

### 正常性テスト

```bash
bash scripts/gpu-lock.sh run CUDA_VISIBLE_DEVICES=5,6 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  /home/ubuntu/projects/llama.cpp/.worktree/double-buffering/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -p "Hello, how are you?" -n 50 --log-file /tmp/llama-cli.log
```

### A/B ベンチマーク

ABAB 交互実行、5 ペア (+ ウォームアップ 1 回破棄):

```bash
# Condition A: selective signaling OFF (always signal)
bash scripts/gpu-lock.sh run GGML_RDMA_NO_SELECTIVE_SIGNAL=1 CUDA_VISIBLE_DEVICES=5,6 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/double-buffering/build/bin/llama-bench \
  --model /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -dev 'CUDA0/CUDA1/RDMA0[...]/RDMA1[...]/RDMA2[...]/RDMA3[...]' \
  -ngl 999 -sm layer -fa 1 --repetitions 1 --n-prompt 128 --n-gen 32 -o csv

# Condition B: selective signaling ON (default, double-buffering active)
bash scripts/gpu-lock.sh run CUDA_VISIBLE_DEVICES=5,6 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/double-buffering/build/bin/llama-bench \
  --model /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -dev 'CUDA0/CUDA1/RDMA0[...]/RDMA1[...]/RDMA2[...]/RDMA3[...]' \
  -ngl 999 -sm layer -fa 1 --repetitions 1 --n-prompt 128 --n-gen 32 -o csv
```

## 交絡チェック

- [x] **単一変数の分離**: 同一バイナリ + `GGML_RDMA_NO_SELECTIVE_SIGNAL` 環境変数トグルのみ
- [x] **ホットパスのログ出力**: `RDMA_LOG_DBG` はデバッグモード時のみ (`GGML_RDMA_DEBUG`)
- [x] **分布の単峰性**: pp128 の A 条件は 202.66-203.39 (range 0.73)、B 条件は 209.61-210.20 (range 0.59)、明確に単峰
- [x] **サーバー再起動不要**: `GGML_RDMA_NO_SELECTIVE_SIGNAL` はクライアント側のみ影響 (サーバーは Send/Recv の小さいインライン制御メッセージのみ)

## 結果

### 正常性テスト

llama-cli で Qwen3.5-35B-A3B (6GPU) の推論が正常に完了。NaN・ゴミ文字なし。

- pp: 59.8 t/s
- tg: 33.9 t/s

### A/B ベンチマーク生データ

**構成**: Qwen3.5-35B-A3B UD-Q4_K_M, 6GPU (CUDA5,6 + RDMA0-3), pp128/tg32, `-fa 1`

#### pp128 (t/s)

| Pair | A (signal OFF) | B (signal ON) | diff (B-A) |
|:----:|:--------------:|:-------------:|:----------:|
| 1 | 203.39 | 210.20 | +6.82 |
| 2 | 203.08 | 210.10 | +7.02 |
| 3 | 202.72 | 209.75 | +7.02 |
| 4 | 202.66 | 209.93 | +7.27 |
| 5 | 202.66 | 209.61 | +6.95 |

A 平均: 202.90 ± 0.32 t/s
B 平均: 209.92 ± 0.23 t/s

| 指標 | 値 |
|------|:---:|
| 差分平均 | +7.02 t/s (+3.46%) |
| t(4) | 95.78 |
| p 値 | 4.4 × 10⁻⁸ |
| Cohen's d | 42.8 (圧倒的に大) |
| 95% CI | [+6.82, +7.22] t/s |
| 全ペア正の効果 | 5/5 (100%) |

**判定**: p < 0.05 かつ効果 > 0.5% → **有効**

#### tg32 (t/s)

| Pair | A (signal OFF) | B (signal ON) | diff (B-A) |
|:----:|:--------------:|:-------------:|:----------:|
| 1 | 34.87 | 33.26 | -1.61 |
| 2 | 34.00 | 33.45 | -0.55 |
| 3 | 33.92 | 34.04 | +0.13 |
| 4 | 33.45 | 33.81 | +0.36 |
| 5 | 33.98 | 33.95 | -0.03 |

A 平均: 34.04 ± 0.51 t/s
B 平均: 33.70 ± 0.33 t/s

| 指標 | 値 |
|------|:---:|
| 差分平均 | -0.34 t/s (-1.00%) |
| t(4) | -0.69 |
| p 値 | 0.53 |
| Cohen's d | -0.31 (小) |
| 全ペア負の効果 | 3/5 (60%) |

**判定**: p ≥ 0.05 → **効果なし** (予想通り: tg は set_tensor を使用しない)

> **注記**: tg32 の pair 1 の A 値 (34.87) は他より高い。ウォームアップ直後の残留効果の可能性あるが、IQR 法の外れ値閾値内 (Q3+1.5*IQR=34.52+1.5*0.89=35.86) なので除外せず。

## 考察

### pp128 改善のメカニズム

1. pp128 処理中、モデルのテンソルデータを RDMA Write でリモート GPU に転送 (`set_tensor`)
2. 常時シグナル (A 条件): 各チャンクで memcpy → post RDMA Write → wait_for_completion → 次のチャンク (直列)
3. Selective signaling + double-buffering (B 条件): memcpy(buf_0) → post unsignaled Write → memcpy(buf_1) → post signaled Write → wait → ... (パイプライン化)
4. 最終チャンクのみ signaled で DMA 完了を保証 (IB の WR 順序保証)

### MoE モデルでの効果量

- GLM-4.7 Dense (11GPU) での過去の selective signaling 効果: +26% (23.35 → 29.44 t/s)
- Qwen3.5-35B MoE (6GPU) での今回の効果: +3.46% (202.90 → 209.92 t/s)
- MoE モデルは pp128 で 200+ t/s と高速 → RDMA Write 待ち時間の相対的割合が小さい
- Dense モデル (GLM-4.7) では RDMA Write がボトルネックの大きな割合を占めるため、効果がより大きい

### tg への影響なし

tg (token generation) は `graph_compute` (Send/Recv) が主体で、`set_tensor` (RDMA Write) は使用しない。double-buffering は Send にも適用したが、graph_compute の Send データは常に < 16MB (single chunk、`is_last=true`) で常時 signaled のため、効果なし。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `ec2fb5a89 (feature/double-buffering)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 34°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1153174) |

GGML_RDMA 環境変数: (none)

**テスト GPU 構成**: Node 1 CUDA5-6 + Node 2 RDMA0-3 (6GPU)
**CUDA0-4 は別の llama-server プロセスが使用中**（テスト時に干渉なし: 異なる GPU デバイス）
