# Gate+Up Fusion 後のカーネルプロファイリング (Phase 1-3)

- **実施日時**: 2026年3月12日 06:19
- **ワークツリー**: `.worktree/merge-overlap` (branch: `merge/cuda-rdma-overlap`)
- **参照レポート**:
  - [PP/TG 最適化ロードマップ](2026-03-12_004808_pp_tg_optimization_roadmap.md)
  - [CUDA/RDMA Overlap マージレポート](2026-03-12_061237_cuda_rdma_overlap_merge.md)
  - [Unfused vs Fused ベンチマーク](2026-03-09_034923_unfused_vs_fused_benchmark.md)

## 前提・目的

Gate+Up fusion + GPU-native sort 適用後の Qwen3.5-35B-A3B (fused Q4_K_M) カーネル時間配分を取得し、Phase 2 以降の最適化施策の効果見積もりの基盤データとする。

- **背景**: Gate+Up fusion によりレイヤーあたりの `mul_mat_id` が 3→2 に削減された。fusion 前は dequant が GPU 時間の約 20% を占めていた。fusion 後の実際の比率を測定する必要がある
- **目的**: 以下の情報を取得する
  1. Dequant 比率 (fusion 前の 20% からの変化)
  2. Expert loop (mul_mat_id GEMM) の時間配分
  3. Delta-net (recurrent) 層の計算比率
  4. データ変換オーバーヘッド
- **前提条件**: 2 CUDA GPU (CUDA4/CUDA5) でローカル実行 (RDMA 不使用)

## 再現方法

```bash
# pp128 プロファイリング
nvprof --print-gpu-summary --csv \
  .worktree/merge-overlap/build/bin/llama-bench \
  -m models/Qwen3.5-35B-A3B-Q4_K_M-fused.gguf \
  -ngl 999 -fa 1 -t 1 -p 128 -n 0 -r 3 \
  -dev 'CUDA4/CUDA5' -o csv

# pp2048 プロファイリング
nvprof --print-gpu-summary --csv \
  .worktree/merge-overlap/build/bin/llama-bench \
  -m models/Qwen3.5-35B-A3B-Q4_K_M-fused.gguf \
  -ngl 999 -fa 1 -t 1 -p 2048 -n 0 -r 3 \
  -dev 'CUDA4/CUDA5' -o csv
```

## 結果

### ベンチマーク性能

| pp size | t/s | 構成 |
|---------|----:|------|
| pp128 | 262.0 ± 2.3 | 2 CUDA (CUDA4/CUDA5) |
| pp2048 | 436.7 ± 1.1 | 2 CUDA (CUDA4/CUDA5) |

### pp2048 カーネル時間配分 (推論のみ、HtoD memcpy 除外)

HtoD memcpy (モデルロード) が全体の 17.5% (3.81s) を占めるため、推論カーネルのみの配分を算出。

| カテゴリ | Time (s) | 推論比率 | 主要カーネル |
|---------|-------:|------:|------|
| **MoE GEMM (FP16)** | 6.394 | 35.5% | `maxwell_hgemm_128x64_tn` (60158 calls, avg 0.106ms) |
| **データ変換** | 3.192 | 17.7% | convert_unary (6.7%), k_get_rows (5.8%), cpy_scalar (3.7%), quantize_q8_1 (1.6%) |
| **Dequant** | 2.399 | 13.3% | q4_K (11.6%), q6_K (1.3%), q5_K (0.5%) |
| **mul_mat_vec_q** | 2.167 | 12.0% | Q4_K int=1~8 (9.9%), Q6_K int=1~8 (2.1%) |
| **Delta-net** | 1.791 | 10.0% | gemmSN_TN (5.3%), pad_f32 (1.4%), batch_trsm (0.7%), tri_kernel (0.9%), ssm_conv (0.4%), etc. |
| **MoE GEMM (FP32)** | 0.596 | 3.3% | `maxwell_sgemm_128x64_tn` (17280 calls) |
| **Flash Attention** | 0.348 | 1.9% | flash_attn_tile (1.5%), combine (0.1%) |
| **Elementwise ops** | ~1.0 | 5.5% | op_mul (1.5%), rms_norm (0.8%), op_add (0.9%), etc. |
| **その他** | ~0.1 | 0.8% | memset, memcpy DtoD/DtoH |

### pp128 vs pp2048 の比較

pp128 では HtoD memcpy が 68.7% を占め、推論カーネルの比率が小さい。推論部分のみの比較:

| カテゴリ | pp128 比率 | pp2048 比率 | 変化 |
|---------|----------:|----------:|------|
| MoE GEMM (FP16) | ~28% | 35.5% | pp2048 で GEMM 支配が強まる |
| Dequant | ~12.5% | 13.3% | ほぼ同等 |
| mul_mat_vec_q | ~18% | 12.0% | pp128 では sparse expert が多い |
| Delta-net | ~8% | 10.0% | ほぼ同等 |

### MoE Expert カーネルの詳細

`mul_mat_vec_q` の `int=N` テンプレートパラメータは、各 expert に割り当てられたトークン数を示す:

| Tokens/Expert | Q4_K Time (s) | Q6_K Time (s) | Calls |
|:---:|-------:|-------:|------:|
| 1 | 0.300 | 0.112 | 36,252 |
| 2 | 0.240 | 0.060 | 22,494 |
| 3 | 0.244 | 0.053 | 15,994 |
| 4 | 0.216 | 0.060 | 12,267 |
| 5 | 0.176 | 0.044 | 9,268 |
| 6 | 0.178 | 0.043 | 7,782 |
| 7 | 0.178 | 0.044 | 6,792 |
| 8 | 0.177 | 0.042 | 5,808 |

- tokens=1~3 (expert にトークンが 1~3 個) が呼び出しの 63% を占める
- tokens ≥ 9 は GEMM パス (`maxwell_hgemm`) に移行

### Delta-net カーネル内訳

| カーネル | Time (s) | 比率 | 説明 |
|---------|-------:|------:|------|
| gemmSN_TN_kernel_half | 0.951 | 5.3% | 小行列 batched GEMM (recurrence) |
| pad_f32 | 0.251 | 1.4% | テンソルパディング |
| batch_trsm_right_kernel | 0.120 | 0.7% | 三角行列ソルブ (recurrence) |
| tri_kernel (lower) | 0.111 | 0.6% | 三角マスク |
| ssm_conv_long_token_f32 | 0.076 | 0.4% | SSM 畳み込み |
| concat_f32_non_cont | 0.084 | 0.5% | テンソル結合 |
| tri_kernel (upper) | 0.055 | 0.3% | 三角マスク |
| l2_norm_f32 | 0.028 | 0.2% | L2 正規化 |
| cpy_scalar_transpose | 0.027 | 0.2% | 転置コピー |
| diag_kernel | 0.003 | <0.1% | 対角行列 |
| cumsum_kernel | 0.003 | <0.1% | 累積和 |
| **合計** | **1.709** | **9.5%** | |

## Phase 2 への影響評価

### Phase 2-1: CUDA 12.6 + cublasSgemmGroupedBatched

**ターゲット**: MoE GEMM (35.5%) + mul_mat_vec_q (12.0%) = **47.5%**

`maxwell_hgemm` が 60158 回呼ばれており (avg 0.106ms)、expert ループのカーネルラウンチオーバーヘッドが存在する。Grouped Batched GEMM で複数 expert の GEMM を 1 カーネルラウンチにバッチ化できれば:

- **楽観的見積もり**: カーネルラウンチ削減で 5-8% の pp 改善
- **保守的見積もり**: P100 では hgemm 自体がボトルネック（compute bound）のため 2-4%
- **リスク**: CUDA 12.6 アップグレードに伴うドライバ更新

### Phase 2-2: Delta-net 層の最適化

**ターゲット**: Delta-net 合計 **10.0%** (gemmSN 5.3% + 補助カーネル 4.7%)

- `gemmSN_TN_kernel_half`: cublas の小行列バッチ GEMM。既に cublas で最適化済みの可能性が高い
- `pad_f32` (1.4%): パディング削除やカーネル融合で削減可能性あり
- `batch_trsm` (0.7%): 三角ソルブ、cublas の実装依存
- **楽観的見積もり**: pad_f32 削減 + gemmSN 最適化で 3-5% の pp 改善
- **保守的見積もり**: cublas カーネルの置換は困難、pad_f32 のみ 1-2%

### 予想外の発見: データ変換オーバーヘッド (17.7%)

`convert_unary` (float↔half, 6.7%) と `k_get_rows_float` (5.8%) が合計 12.5% を占める。これは:
- `k_get_rows_float`: expert token gathering (mul_mat_id の前段)
- `convert_unary`: 型変換 (Q4_K dequant → FP16 → GEMM → FP32 → 次の演算)

**潜在的改善**: 型変換チェーンの短縮、k_get_rows の最適化は Phase 2 以降の新たな候補。

### Dequant 比率の変化

| 時期 | Dequant 比率 | 備考 |
|------|----------:|------|
| Fusion 前 | ~20% | Gate+Up 別テンソル (mul_mat_id x3/layer) |
| Fusion 後 | 13.3% | Gate+Up 融合 (mul_mat_id x2/layer) |

**-6.7 ポイントの削減**を確認。fusion により dequant が約 1/3 削減された。

## 結論

1. **Dequant 13%**: fusion 前の 20% から低下を確認。さらなる dequant 削減の余地は限定的
2. **MoE GEMM 38.8% が最大ボトルネック**: Phase 2-1 (Grouped GEMM) の主要ターゲット
3. **Delta-net 10%**: Phase 2-2 の最適化候補。gemmSN (cublas) は置換困難だが pad_f32 に余地あり
4. **データ変換 17.7%**: 新たに発見された改善候補。convert_unary + k_get_rows の最適化は未計画
5. **mul_mat_vec_q 12%**: sparse expert パス。tokens=1~3 が 63% を占める。Grouped GEMM と共に最適化可能

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `453f4c7c1 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 24°C | 30°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1979336) |
