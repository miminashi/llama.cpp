# Qwen3.5 PP性能改善 チーム探求レポート

- **実施日時**: 2026年3月8日 05:54
- **ワークツリー**: N/A (調査・分析のみ)
- **対象モデル**: Qwen3.5-35B-A3B Q4_K_M (21GB), Qwen3.5-122B-A10B Q4_K_M (72GB)
- **リソース**: P100 x11 (7C+4R), 176GB VRAM, 100GbE RDMA

## 前提・目的

これまでの20施策 ([振り返りレポート](2026-03-08_042629_pp_optimization_retrospective.md) 参照) を踏まえ、Qwen3.5のPP性能にまだ改善の余地がないか、3名のエージェントチームで探求する。

- **背景**: Qwen3.5-35B-A3B は compute-bound (GPU計算 94.8%, RDMA通信 3.3%)。P100 (cc 6.0) の構造的制約 (DP4A/TensorCore/CUDA Graph 不可) により `mul_mat_id` が最遅フォールバックパスに落ちている
- **目的**: 過去に試していない新しいアプローチ、特に P100×11 / 176GB VRAM という特殊環境を活かした「非定石」な手法を発見する
- **参照レポート**:
  - [PP性能改善施策 振り返りレポート](2026-03-08_042629_pp_optimization_retrospective.md)
  - [Qwen3.5 PP性能改善 探索レポート](2026-03-08_051455_pp_optimization_exploration.md)
  - [Qwen3.5 PP性能改善 総合調査レポート](2026-03-07_162952_qwen35_pp_optimization_comprehensive.md)

## 調査体制

| エージェント | 担当 | 主な成果 |
|---|---|---|
| **literature-researcher** | 文献調査 (論文・フレームワーク実装) | MoE最適化の最新手法と P100 適用性の整理 |
| **code-analyst** | コード分析 (mul_mat_id, CUDA kernels) | フォールバックパスの詳細コスト分析、FP16パス確認 |
| **strategy-integrator** | 戦略統合 (過去レポート照合・評価) | 20施策との重複排除、見積もり矛盾の解決 |

## 重要な新発見

### 1. P100は既にFP16 HGEMMを使用している

**nvprofデータ**: `maxwell_hgemm_128x64_tn` = GPU時間の52%

**コード確認** (`ggml-cuda.cu` L1286-1338):
- `fast_fp16_hardware_available(cc=600)` = true
- Q4_K_M → F16 dequant → `cublasGemmEx(CUDA_R_16F, COMPUTE_16F)` → P100では `maxwell_hgemm` が呼ばれる
- つまり「F16 HGEMMパスの追加」は新しい提案ではなく、**既存の動作**

**意味**: 改善のターゲットは GEMM 自体ではなく、**dequant コスト (全体の20%)** と **カーネルラウンチ + stream sync オーバーヘッド**。

### 2. PP128での混在パス（MMVQ / cuBLASフォールバック）

**コード確認** (`ggml-cuda.cu` L2268-2423):
- `MMVQ_MAX_BATCH_SIZE = 8`: tokens_per_expert ≤ 8 → MMVQ パス (GPU native, sync不要)
- PP128, 256 experts, 8 active → 平均 4 tokens/expert
- **大半の expert は MMVQ パスに入る**が、人気 expert (10-20 tokens) は **cuBLAS フォールバック**

### 3. Stream sync は 160-240回/ubatch

- 2回/mul_mat_id op × (40 layers × 2-3 ops) = **160-240回/ubatch**
- ただし sync は expert ループの**外側**にのみ存在 (ソート前後)
- expert ループ内は sync なしで連続カーネルラウンチ → GPU pipeline は途切れない
- sync + CPU sort = ~70-140 μs/op × 120 ops = **8.4-16.8 ms/ubatch** (PP128全体の1.3-2.6%)

### 4. 余剰 VRAM (155GB) で全 expert 重みの F16 キャッシュが可能

- Qwen3.5-35B Q4_K_M (21GB) → F16 展開 ≈ 75GB
- 155GB 余剰に完全に収まる
- nvprof: dequant = GPU時間の20% → キャッシュで排除可能
- 文献に前例なし — **176GB VRAM で 21GB モデルという構成が特殊すぎて想定されていない**

### 5. CUDA 12.0 の制約

- `cublasSgemmGroupedBatched` (可変サイズ対応) は **CUDA 12.4+ 必要 → 利用不可**
- `cublasGemmBatchedEx` (同一サイズ) は **利用可能** だがパディング必要
- Triton カーネル全般 (vLLM fused_moe 等) は **cc ≥ 7.0 必要 → 利用不可**

### 6. upstream に GPU-native expert sort が既存

- `mmid.cu` の `mm_ids_helper` カーネル: 1ワープ/expert で ids_src1, ids_dst, expert_bounds を並列計算
- **MMQ/MMF パスからのみ呼ばれ、フォールバックパスでは未使用**
- フォールバックパスに統合すれば stream sync ×2回 + CPU sort を排除可能

## エージェント間の議論ハイライト

### 議論1: GPU-native sort の効果見積もり

- **literature-researcher**: 「stream sync排除で10-30%改善」
- **strategy-integrator** (初期): 「sync は expert ループの外側のみ。GPU pipeline は途切れない。実測は 1.3-2.6%」
- **code-analyst**: 「2回の sync + CPU sort = ~70-140μs/op。120 ops で 8.4-16.8ms。PP128全体 ~640ms の 1.3-2.6%」
- **strategy-integrator** (修正): 「CPU triple-loop sort のコストを過小評価していた。O(256×128×8) = 262K iterations で ~6-12ms/ubatch。sync排除より **CPU sort排除の方が大きい**」
- **結論**: **+3-5%** が修正後の見積もり。`mm_ids_helper` (shared memory, 1ワープ/expert) をフォールバックパスに統合するだけで実装可能

### 議論2: F16 dequantキャッシュの効果

- **code-analyst**: 「dequant = GPU時間の20%。キャッシュで +15-25%」
- **strategy-integrator**: 「nvprof データ (dequant 65ms q4_K + 23ms q5_K / 446ms total) と一致。キャッシュで排除されるのはステップ1 (Q4_K_M→F16 dequant) のみ。ステップ2-4 (src1変換, HGEMM, dst変換) は不変。精密見積もり: **+15-20%**。加えて `ggml_cuda_pool_alloc` の確保/解放も排除される」
- **literature-researcher**: 「文献に前例なし。vLLM/TensorRT-LLM は fused dequant-GEMM (Marlin) だが P100 では Tensor Core 不可で使えない。分離 dequant の環境だからこそキャッシュが合理的」
- **結論**: 155GB 余剰 VRAM という特殊環境でのみ可能な手法。**+15-20%** が見込める最有力施策

### 議論3: cuBLAS Batched GEMM

- **code-analyst**: 「`cublasSgemmGroupedBatched` は CUDA 12.4+ 必要。CUDA 12.0 では不可」
- **literature-researcher**: 「CUTLASS 2.x の Grouped GEMM は Pascal 非推奨。`cublasGemmBatchedEx` はパディング必要」
- **strategy-integrator**: 「expert ループは既に GPU pipelined。Batched にしても削減されるのはラウンチオーバーヘッドのみ。**3-8%**」
- **結論**: 利用可能な API は `cublasGemmBatchedEx` (同一サイズ, パディング必要)。効果は限定的

### 議論4: 122B モデルのスケーリング

- **全エージェント一致**: 35B (active 3B) で負のスケーリングだった原因は GPU あたり計算量不足。122B (active 10B) なら 3.3 倍の計算密度 → 8-11GPU で正のスケーリングに転じる可能性大
- **コスト: ゼロ** (ベンチマーク実行のみ)

## P100 で使えないことが確定した手法

| 手法 | 不可理由 | 確認元 |
|------|---------|--------|
| Triton カーネル全般 (vLLM fused_moe, FusedXpert) | cc ≥ 7.0 必要 | literature |
| CUDA Graph | cc ≥ 8.0 (Ampere) 必要 | 過去レポート |
| Tensor Core 融合 (Marlin, DeepGEMM) | cc ≥ 7.0 必要 | literature |
| DP4A / INT8 カーネル (MMQ) | cc ≥ 6.1 必要, -48% 退行実証済み | 過去レポート |
| `cublasSgemmGroupedBatched` | CUDA 12.4+ 必要, 環境は 12.0 (**12.4+にアップグレードすれば利用可能**, P100は12.8まで対応) | code-analyst |
| CUTLASS 2.x Grouped GEMM | Pascal 非推奨化済み | literature |
| F16 HGEMM パス追加 | **既に使用中** | code-analyst + nvprof |

## 最終提案リスト (優先度順)

### Tier 0: コスト=ゼロ、即実験可能

| # | 施策 | 期待効果 | 備考 |
|---|------|:-------:|------|
| **0-A** | **F16 GGUF モデルでベンチマーク (35B, 4GPU)** | **PP +15-20%** | dequant排除。コード変更不要 |
| **0-B** | Qwen3.5-122B-A10B ベースライン (11GPU) | スケーリング回復 | Active 10B (35Bの3.3倍) → GPU追加が有効化する可能性大 |
| **0-C** | 122B GPU数スイープ (6/7/8/11GPU) | スケーリング曲線取得 | 72GBモデル → 最低5GPU必要 |
| **0-D** | 122B ubatchスイープ (-ub 256〜2048) | 最適ubatch特定 | 35Bと同様に ub=2048 で飽和するか確認 |

**0-A の詳細 (最有力施策 — コード変更不要に格上げ)**:

チーム調査では「dequant キャッシュ」(Tier 2, 実装1週間) として提案されたが、**F16 GGUF モデルを直接使えばコード変更なしで同等の効果が得られる**。

- **現状**: Q4_K_M (21GB) → 毎回 F16 に dequant → cublasHgemm。dequant = GPU 時間の 20%
- **改善**: F16 GGUF (69GB) → dequant 不要 → cublasHgemm 直接。176GB VRAM に十分収まる
- **期待効果**: PP **+15-20%** (dequant 20% 排除 + pool_alloc オーバーヘッド排除)

**F16 GGUF の作成方法**: unsloth は BF16 GGUF のみ提供 (F16 なし)。P100 は BF16 ネイティブ非対応 (Ampere 以降) のため、F16 への変換が必要:

```bash
# 方法1: BF16 GGUF → F16 GGUF (簡単)
build/bin/llama-quantize Qwen3.5-35B-A3B-BF16.gguf Qwen3.5-35B-A3B-F16.gguf F16

# 方法2: HF safetensors → F16 GGUF (より高精度)
python convert_hf_to_gguf.py --outtype f16 <HF_model_dir>
```

BF16→F16 変換は指数部の範囲が狭くなる (8→5 bit) が、NN 重みは通常 ±1 未満でオーバーフローの懸念はない。仮数部は 7→10 bit に増え精度は向上する。

**Q4_K_M と F16 の比較**:

| | Q4_K_M | F16 | 差異 |
|---|---|---|---|
| ディスク/VRAM | 21GB | 69GB | +48GB |
| ランタイム dequant | 毎ubatch 88ms (20%) | **不要** | -88ms |
| GEMM 精度 | Q4 → F16 (量子化誤差) | F16 ネイティブ | F16 の方が高精度 |
| 推論品質 | Q4_K_M 相当 | **大幅向上** (元のF16精度) | 品質も改善 |

**0-B/C/D の根拠**: 35B で 4→8GPU が全条件で負のスケーリングだった原因は active params (3B) の少なさ。122B (active 10B) なら GPU 追加でスケーリングが改善するはず。コスト=ゼロのため最優先で実験すべき。

### Tier 1: 低コスト、確実な改善

| # | 施策 | 期待効果 | 実装コスト | リスク | 過去検証 |
|---|------|:-------:|:---------:|:-----:|---------|
| **1-A** | GPU-native IDs sorting | PP +3-5% | 低 | 低 | 未実施 (新規) |
| **1-B** | Gate+Up マージ GGUF | PP +2-5% | 低 (GGUF再変換) | 低 | 未実施 |

**1-A の詳細**: `mmid.cu` の `mm_ids_helper` カーネルが upstream に既存。フォールバックパスでも呼べるように改修すれば、`cudaStreamSynchronize` ×2回 + CPU triple-loop sort を排除可能。

**1-B の詳細**: `ffn_gate_exps` + `ffn_up_exps` → `ffn_gate_up_exps` に結合し `mul_mat_id` 3→2回/layer に削減。コードは対応済み (`create_tensor_gate_up_exps`)。unsloth GGUF は分離格納のため `convert_hf_to_gguf.py` で再変換が必要。

### Tier 2: 中コスト、大きな改善期待

| # | 施策 | 期待効果 | 実装コスト | リスク | 過去検証 |
|---|------|:-------:|:---------:|:-----:|---------|
| **2-B** | cuBLAS Batched GEMM化 | PP +3-5% | 中 | 中 | 未実施 |

> **注**: 旧 Tier 2-A「F16 dequantキャッシュ」は Tier 0-A「F16 GGUFモデル使用」に格上げ。F16 GGUF を直接使えばコード変更不要で同等の効果が得られるため。

**2-B の詳細**:
- expert ループの個別 GEMM を `cublasGemmBatchedEx(CUDA_R_16F)` で一括化
- カーネルラウンチオーバーヘッド削減
- パディングが必要 (tokens_per_expert が不均一、最大値に合わせてゼロパディング → FLOPS 浪費)
- `cublasSgemmGroupedBatched` (可変サイズ対応) は CUDA 12.4+ 必要 (現環境 12.0)。ただし P100 は CUDA 12.8 まで対応しており、**toolkit アップグレードで利用可能になる** → パディング不要の Grouped Batched GEMM が選択肢に入る
- F16 GGUF と組み合わせれば F16 重みを直接 batched GEMM に渡せる

### Tier 2-B: delta-net Strided Batched GEMM (追加発見)

| # | 施策 | 期待効果 | 実装コスト | リスク | 過去検証 |
|---|------|:-------:|:---------:|:-----:|---------|
| **2-C** | delta-net Strided Batched GEMM | PP +4-9% | 中 | 中 | 未実施 (新規) |

**詳細**:
- delta-net chunking の 64×64 小行列 GEMM (~420回/ubatch) を `cublasHgemmStridedBatched` で一括化
- nvprof データからの精密見積もり:
  - delta-net attention = GPU時間の 7.6-13.7% (保守的: 34ms, 楽観的: 61ms / 446ms total)
  - 2-3x 高速化で PP **+4-9%**
- literature-researcher の初期推定 (37-50%) は「レイヤー数比率」と「計算時間比率」の混同による過大評価。MoE FFN が 91% を占め、delta-net attention は最大 14%
- FLA (Flash Linear Attention) は Triton (cc 7.0+) 必須で P100 不可のため、batched GEMM アプローチが現実的

### Tier 3: 高コスト、追加改善

| # | 施策 | 期待効果 | 実装コスト | リスク |
|---|------|:-------:|:---------:|:-----:|
| **3-A** | cuBLAS Batched GEMM (MoE expert一括化) | PP +3-5% | 中 | 中 |
| **3-B** | Expert multi-stream 並列実行 | PP +5-15% | 中-高 | 中 |
| **3-C** | Expert 重み複製 (HarMoEny) | PP +10-20% | 高 | 高 |

**3-A**: expert ループの個別 GEMM を一括化。CUDA 12.0 では `cublasGemmBatchedEx(CUDA_R_16F)` (同一サイズ必須、パディング要)。**CUDA 12.4+ にアップグレードすれば `cublasSgemmGroupedBatched`** (可変サイズ対応、パディング不要) が利用可能に。P100 は CUDA 12.8 まで対応しているため toolkit アップグレードは可能。

**3-B**: expert ループの各 GEMM を複数 CUDA stream で並列実行。llama.cpp upstream PR #16991 の Q/K/V 並列化の延長。ggml の単一 stream モデル変更が必要。

**3-C**: 頻出 expert を複数 GPU に複製してロードバランシング (HarMoEny 論文: 最大41%改善)。11GPU の余剰 VRAM を活用。ルーティングパターン依存のため効果は不確実。

### Tier 4: 不採用

| 施策 | 不採用理由 | 実証 |
|------|-----------|------|
| MMQ 強制有効化 | P100 に DP4A 不可、-48% 退行 | 実測 |
| F16 HGEMM パス追加 | **既に使用中** | nvprof `maxwell_hgemm` 52% + コード確認 |
| 8GPU スケーリング (35B) | 全条件で負のスケーリング | ABAB実測 |
| MMVQ bypass | -14〜-42% 退行 | 実測 |
| CUDA Graph | P100 cc 6.0 不可 | アーキテクチャ制約 |
| Triton カーネル (vLLM fused_moe等) | cc ≥ 7.0 必要 | アーキテクチャ制約 |
| Delta-Net FLA | Triton (cc ≥ 7.0) 依存 | アーキテクチャ制約 |
| ~~`cublasSgemmGroupedBatched`~~ | ~~CUDA 12.4+ 必要 (環境は12.0)~~ → **P100は12.8まで対応、toolkitアップグレードで利用可能** | Tier 3-A に移動 |
| モデルレプリカ | llama.cpp 設計変更必要、コスト>>>利得 | 探索レポート結論 |
| Expert Parallelism | allreduce コスト > compute 節約 | 過去試行で断念 |
| Recurrent 層分離配置 | layer-split で自然発生、追加効果なし | 構造分析 |
| メモリレイアウト最適化 | compute-bound で帯域幅は非ボトルネック | nvprof確認 |

## 組み合わせ効果の分析

過去の知見: Per-device + Pipeline の組み合わせで **+25.9%** (単独では両方中立/悪化)。Ring buffer + Pipeline は相殺 (0%)。組み合わせの予測は慎重に行う必要がある。

### 相乗効果が期待できる組み合わせ

| 組み合わせ | 累積期待効果 | 根拠 |
|-----------|:----------:|------|
| **F16 GGUF + GPU-native sort + Gate+Up マージ** | **+21-32%** | 独立した改善の乗算 (1.175 × 1.04 × 1.035 ≈ 1.27) |
| **F16 GGUF + Batched GEMM** | F16重み常駐 + 一括GEMM | F16 重みを直接 batched GEMM に渡す → dequant + pool_alloc 完全排除 |
| **GPU-native sort + Gate+Up マージ** | +5-10% (加算的) | sort回数: 120→80回 (3→2 ops/layer) |

**F16 GGUF + GPU-native sort の相乗効果** (strategy-integrator 発見):
- F16 GGUF により src0 が既に F16 → dequant スキップ
- GPU-native sort により stream sync 不要
- → expert loop が「GPU-native sort → get_rows → cublasHgemm(FP16, sync不要)」の **完全GPU常駐パイプライン** になる
- CPU介入が完全排除され、GPU utilization が大幅向上する可能性

### 相殺リスクがある組み合わせ

| 組み合わせ | リスク |
|-----------|-------|
| Batched GEMM + multi-stream | GPUリソース競合 |
| delta-net batched + F16キャッシュ | 独立 (F16キャッシュは MoE のみ対象) |

### 理論的最大改善 (全Tier 1-2 適用)

```
GPU-native sort (+3-5%) × Gate+Up (+2-5%) × F16キャッシュ (+15-20%) × delta-net batched (+4-9%)
= 累積 +27-44%
```

## 推奨実行順序 (3段階戦略)

```
Stage 1 (即日〜数日, コストゼロ):
  0-A: F16 GGUF 作成 + 35B ベンチマーク (PP +15-20% 期待, コード変更不要)
       BF16 GGUF ダウンロード → llama-quantize で F16 変換 → llama-bench
  Gate+Up マージ状態の GGUF メタデータ確認 (10分)
  0-B: 122Bモデルのベースライン取得 (11GPU)
  0-C/D: 122B GPU数スイープ + ubatchスイープ

Stage 2 (1-2週, 低コスト):
  1-A: GPU-native sort (mmid.cu の mm_ids_helper をフォールバックパスに統合)
  1-B: Gate+Up マージ GGUF (未マージの場合のみ)
  → Stage 1+2 累積: +20-30%

Stage 3 (2-4週, 中コスト, 追加改善):
  2-C: delta-net Strided Batched GEMM
  3-A: cuBLAS Batched GEMM (MoE expert一括化)
  → Stage 3 追加: +7-14%
```

## 文献参照

| ソース | URL | 関連提案 |
|--------|-----|---------|
| cuBLAS Grouped GEMM API | https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/ | 2-B |
| FusedXpert (SC'25) | https://sc25.supercomputing.org/proceedings/posters/poster_files/post173s2-file3.pdf | Triton, P100不可 |
| Static Batching Framework | https://arxiv.org/abs/2501.16103 | 2-B 概念参考 |
| HarMoEny (Expert 複製) | https://arxiv.org/abs/2506.12417 | 非定石参考 |
| GatedDeltaNet (NVlabs) | https://github.com/NVlabs/GatedDeltaNet | 3-A 参考 |
| Marlin (fused dequant-GEMM) | https://github.com/IST-DASLab/marlin | P100不可 (Tensor Core必要) |
| P100 HGEMM | https://github.com/hma02/cublasHgemm-P100 | FP16確認 |
| vLLM MoE kernels | https://docs.vllm.ai/en/latest/design/moe_kernel_features/ | Triton, P100不可 |
| KTransformers (SOSP'25) | https://dl.acm.org/doi/10.1145/3731569.3764843 | 参考 |
| llama.cpp Issue #12859 | https://github.com/ggml-org/llama.cpp/issues/12859 | mul_mat_id 改善議論 |

## 過去20施策との重複チェック

| 新提案 | 過去の関連施策 | 重複? |
|--------|-------------|-------|
| F16 dequantキャッシュ | なし | 新規 |
| GPU-native sort | なし | 新規 |
| delta-net batched GEMM | なし | 新規 |
| cuBLAS Batched GEMM | E-19 Gate+Up (部分的関連) | 別施策 |
| Expert multi-stream | B-11 GDR並列 (関連) | 異なるレベルの並列化 |
| Expert重み複製 | なし | 新規 |
| 122Bモデル | E-20 8GPU拡張 (関連) | 異なるモデルで再検証 |
| Gate+Up GGUF | E-19 と同一 | 概念は同一、未実施 |

**重複: 1件 (Gate+Up)。それ以外はすべて新規。**

## 結論

**全13施策を評価し、8施策を有望 (Tier 0-3)、12施策を不採用 (Tier 4) としました。**

P100×11 / 176GB VRAM という特殊環境で Qwen3.5 の PP 改善余地は**まだある**。3段階戦略で理論的最大 **+27-44%** の PP 改善が見込まれる。

### 最も費用対効果の高い施策

1. **F16 GGUF モデル使用 (+15-20%, コード変更不要)** — BF16 GGUF から `llama-quantize` で F16 に変換するだけ。69GB は 176GB VRAM に余裕で収まる。dequant (GPU時間の20%) を完全排除。チーム調査では「dequant キャッシュ実装 (1週間)」として提案されたが、**F16 GGUF を直接使えば同等の効果がコードゼロで得られる**
2. **122B モデル実験** (コストゼロ) — active params 3.3 倍により GPU スケーリングが回復し、11GPU の有効活用が期待できる
3. **GPU-native sort (+3-5%)** — upstream 既存の `mm_ids_helper` カーネルをフォールバックパスに統合するだけで、stream sync + CPU sort を排除

### 組み合わせの核心

F16 GGUF + GPU-native sort の組み合わせは、expert loop から CPU 介入を完全排除し「**完全 GPU 常駐パイプライン**」を実現する。これは単純な加算以上の相乗効果をもたらす可能性がある。
