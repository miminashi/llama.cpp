# Qwen3.5-35B-A3B PP性能向上調査レポート

- **実施日時**: 2026年3月7日 05:00〜06:07
- **ワークツリー**: `.worktree/qwen35-pp-opt2` (MMVQ実験用、結果は退行のため不採用)
- **モデル**: `unsloth/Qwen3.5-35B-A3B-GGUF` (Q4_K_M, 19.8GB)
- **テスト構成**: 2GPU ローカル (CUDA4,5) / 4GPU (2C+2R)

## 前提・目的

Qwen3.5-35B-A3B の PP (prompt processing) 性能について、RDMA 以外の観点から改善余地を調査する。過去のプロファイリングで server-side GPU compute が 94.8% と判明しており、RDMA 最適化の効果が限定的であったため。

### Qwen3.5-35B-A3B アーキテクチャ

| コンポーネント | 値 |
|---|---|
| アーキテクチャ | `qwen35moe` (三重ハイブリッド) |
| レイヤー数 | 40 |
| embedding_length | 2048 |
| expert_count | 256 |
| expert_used_count | **8** |
| expert_feed_forward_length | 512 |
| Delta Net レイヤー | 30/40 (full_attn_interval=4) |
| SSM Conv | 30/40 |
| Chunk size (delta net) | 64 |

## 調査結果

### 1. MUL_MAT_ID カーネルパスの特定

**n_expert_used = 8** であり、`MMVQ_MMID_MAX_BATCH_SIZE = 4` を超える。P100 (cc=600) では:

| チェック | 結果 | 理由 |
|---|---|---|
| MMVQ (ne2 ≤ 4) | 不可 | n_expert_used=8 > 4 |
| MMQ | 不可 | `turing_mma_available` = false (cc=600 < 750) |
| MMF | 不可 | `volta_mma_available` = false (cc=600 ≠ 700) |
| **cuBLAS fallback** | **使用** | 上記すべて不可 |

ただし、PP 時の `ne2` は `n_tokens`（ubatch サイズ）であり、TG 時の `ne2=1` ≤ 4 なので TG は MMVQ を使用。PP では ne2 > 8 のため cuBLAS fallback が確定。

cuBLAS fallback の処理フロー:
1. `cudaStreamSynchronize` — ids テンソルを GPU→CPU コピー待ち
2. CPU 上でエキスパートごとにトークンをソート
3. `cudaStreamSynchronize` — ソート済み ids を CPU→GPU コピー待ち
4. `get_rows_cuda` — GPU 上で src1 データをソート
5. エキスパートごとに `ggml_cuda_mul_mat` を呼び出し (tokens_per_expert ≤ 8 なら MMVQ、> 8 なら cuBLAS)
6. 結果をアンソート

### 2. GPU 数最適化ベンチマーク (2GPU vs 4GPU)

ABAB Paired Design, r=5。

| PP サイズ | 2GPU (t/s) | 4GPU (t/s) | 差分 | p値 |
|---|---:|---:|---:|---|
| pp128 | **216.9** | 216.2 | -0.3% | 0.051 (ns) |
| pp512 | **344.1** | 340.5 | -1.1% | 0.019 (*) |
| pp2048 | 331.7 | **354.6** | +6.9% | 0.000 (***) |
| tg32 | **39.7** | 34.5 | -13.1% | 0.000 (***) |

**結論**: 2GPU が pp128/pp512/tg32 で優位。4GPU は pp2048 のみ有利（PCIe 負荷分散効果）。tg は RDMA ラウンドトリップレイテンシにより -13%。

### 3. ssm_conv 共有メモリ最適化

commit `1e38a7a6f` ("CUDA: use shared mem for ssm_conv") は既に upstream merge (`e674c4209`) に含まれており、現行ベースラインに反映済み。

nvprof で `ssm_conv_long_token_f32` カーネルの使用を確認 (PP160: 3ms, 30 calls)。PP 全体に占める割合は < 1% であり、ssm_conv は PP のボトルネックではない。

### 4. PP GPU 計算プロファイリング (nvprof)

2GPU, PP~160 tokens (187.8 t/s) での nvprof GPU 時間内訳:

| カーネル | 時間 (ms) | 割合 | 説明 |
|---|---:|---:|---|
| maxwell_hgemm_128x64_tn | 232 | 52% | cuBLAS FP16 matmul (MoE per-expert) |
| dequantize_block_q4_K | 65 | 15% | Q4_K→FP16 重み逆量子化 |
| gemmSN_TN variants | 53 | 12% | cuBLAS small gemv (少トークン expert) |
| convert_unary f32↔f16 | 34 | 8% | 型変換 |
| k_get_rows_float | 26 | 6% | GPU 上トークンソート |
| dequantize_block_q5_K | 23 | 5% | Q5_K→FP16 重み逆量子化 |
| maxwell_sgemm | 8 | 2% | cuBLAS FP32 matmul |
| **合計** | **446** | **100%** | |

**MoE cuBLAS fallback が PP GPU 時間の 91% を占める。** 主要コストは matmul 自体 (64%) と weight dequant (20%)。

### 5. MMVQ MUL_MAT_ID バイパス実験

cuBLAS fallback を回避し、MMVQ の `is_multi_token_id` パスで全トークンを処理する変更を実装・検証。

**変更内容**: `ggml_cuda_mul_mat_id` で quantized + MMQ/MMF 不可の場合、ne2 チェックをバイパスして MMVQ を直接呼び出し。

| Config | Baseline (t/s) | MMVQ (t/s) | 差分 |
|---|---:|---:|---:|
| pp128 | 216.7 | 186.7 | **-13.8%** |
| pp512 | 341.0 | 198.6 | **-41.7%** |
| pp2048 | 329.3 | 222.9 | **-32.3%** |
| tg32 | 39.7 | 40.0 | +0.8% |

**結果: PP で大幅退行。** 原因:
- MMVQ `is_multi_token_id` は `c_ncols_dst=1` で各トークンを独立処理（1×hidden の mat-vec）
- cuBLAS fallback はトークンをエキスパートごとにバッチ化し、`ncols_dst=4〜8` の MMVQ（PP128 時）や cuBLAS batched matmul（PP512+ 時）で処理
- cuBLAS fallback の方がウェイトデータの再利用効率が高い

**結論: この変更は不採用。cuBLAS fallback は CPU sorting + sync のオーバーヘッドにもかかわらず、バッチ化により MMVQ 単独よりも効率的。**

### 6. ubatch サイズチューニング

2GPU ローカル構成, r=3。

| ub | pp128 (t/s) | pp512 (t/s) | pp2048 (t/s) |
|---:|---:|---:|---:|
| 32 | 120.6 | 122.7 | 120.7 |
| 64 | 168.2 | 171.3 | 167.1 |
| 128 | 217.0 | 220.1 | 213.6 |
| 256 | 217.2 | 280.1 | 269.5 |
| 512 (default) | **216.8** | **343.4** | 331.2 |
| 1024 | — | 342.5 | 415.6 |
| 2048 | — | — | **497.1** |
| 4096 (-b 4096) | — | — | **563.7** (pp4096) |

**主要発見: `-ub` をプロンプト長に合わせると PP が大幅向上。**

| PP サイズ | 最適 ub | vs default (ub=512) |
|---|---|---|
| pp128 | 512 (default) | ±0% |
| pp512 | 512 (default) | ±0% |
| pp2048 | 2048 | **+50.0%** |
| pp4096 | 4096 (-b 4096) | **+70.2%** (vs ub=512 推定) |

**メカニズム**: ubatch が大きいほど MoE FFN でのエキスパートあたりのトークン数が増加:
- ub=512, pp2048: 4 ubatches × 512 tokens → 平均 16 tokens/expert
- ub=2048, pp2048: 1 ubatch × 2048 tokens → 平均 64 tokens/expert
- 大きいバッチは cuBLAS の行列乗算効率を向上させる

**VRAM 使用量**: ub=2048 で compute メモリが CUDA1 で 525→2100 MiB に増加するが、P100 16GB では十分余裕あり。

出力正常性は ub=2048 で確認済み（delta net chunking は ubatch 内で CS=64 のチャンク処理を行うため、大きい ubatch でも正常動作）。

## まとめ

### 有効な最適化

| 最適化 | 効果 | 条件 |
|---|---|---|
| **ubatch = prompt length** | PP +50〜70% | pp > 512 の場合 |
| 2GPU 構成 (RDMA なし) | tg +15%, pp128 +0.3% | 19.8GB モデル |

### 無効と判明した最適化

| 試行 | 結果 | 原因 |
|---|---|---|
| MMVQ MUL_MAT_ID バイパス | PP -14〜42% | per-token 処理はバッチ化に劣る |
| ssm_conv shared mem | 既に含有 | upstream merge 済み |
| 4GPU 拡張 | PP -1%, tg -13% | RDMA overhead > compute benefit |
| MMVQ_MMID_MAX_BATCH_SIZE=8 | PP 効果なし | PP の ne2 = n_tokens >> 8 |

### 今後の可能性

1. **GPU-side expert sorting**: cudaStreamSynchronize 2回を排除 (CPU sorting をGPU sorting に置換)
2. **cuBLAS batched GEMM**: per-expert ループを 1回の batched GEMM 呼び出しに統合
3. **MMVQ multi-column multi_token_id**: `c_ncols_dst=1` を expert ごとの batched ncols_dst に変更

これらはいずれも大規模な変更が必要であり、upstream 側での対応が望ましい。

## 再現方法

### 2GPU vs 4GPU ベンチマーク
```bash
CUDA_VISIBLE_DEVICES=4,5 llama-bench \
  -m <model_path> -ngl 999 -sm layer -fa 1 -r 5 -p 128 -n 0 -o csv
```

### ubatch チューニング
```bash
CUDA_VISIBLE_DEVICES=4,5 llama-bench \
  -m <model_path> -ngl 999 -sm layer -fa 1 -ub 2048 -r 3 -p 2048 -n 0 -o csv
```

### nvprof プロファイリング
```bash
CUDA_VISIBLE_DEVICES=4,5 nvprof --print-gpu-summary llama-cli \
  -m <model_path> -ngl 999 -fa 1 -f <prompt_file> -n 1 --log-file /tmp/llama-cli.log
```

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `e674c4209 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 36°C | 34°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
