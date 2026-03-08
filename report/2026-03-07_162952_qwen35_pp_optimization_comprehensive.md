# Qwen3.5-35B-A3B PP性能改善 総合調査レポート

- **実施日時**: 2026年3月7日 16:29
- **ワークツリー**: N/A (調査・分析のみ、コード変更なし)

## 前提・目的

Qwen3.5-35B-A3B (MoE, 256 experts / 8 active) の Prompt Processing (PP) 性能を改善する方法を、3名のエージェントチームによる並行調査 + 交差検証で徹底的に洗い出す。

### 背景

- **モデル**: Qwen3.5-35B-A3B (35B total, 3B active, Q4_K_M)
- **ハードウェア**: P100 (cc 6.0) × 4-8GPU, 2ノード RDMA 接続
- **テスト構成**: Node1 CUDA4,5 + Node2 RDMA0,1 (4GPU) が標準
- **現在の性能**: PP128 ~200 t/s (4GPU), PP2048 ~51 t/s (overlap有効時)
- **既知のボトルネック**: server-side GPU compute が 94.8%, RDMA 通信は 3.3%

### 参照レポート

- [RDMA サーバーオーバーヘッド分析](report/2026-03-06_091203_rdma_server_overhead_analysis.md)
- [Qwen3.5 PP 最適化調査](report/2026-03-07_060701_qwen35_pp_optimization_investigation.md)
- [Qwen3.5 PP16384 全最適化](report/2026-03-07_074630_qwen35_pp16384_all_optimizations.md)

## 調査体制

| エージェント | 担当 | 調査範囲 |
|-------------|------|---------|
| Agent A | コード分析 | split 実行ループ, set_tensor/get_tensor, ディスパッチオーバーヘッド |
| Agent B | 設定・パラメータ | ubatch, FA, スレッド数, RDMA/CUDA 環境変数, 量子化 |
| Agent C | アーキテクチャ | Expert parallelism, upstream MoE 最適化, CUDA カーネル |
| Discussion-AC | 交差検証 | Agent A vs C の提案の技術的実現可能性 |
| Discussion-B | 深掘り | Agent B の主張の検証, 見落とし環境変数 |
| Discussion-Novel | 新規探索 | 他エージェントが見落とした改善手法 |

## 核心的発見: P100 は MoE にとって最悪のアーキテクチャ

全エージェントの調査結果を横断して見える最重要の構造的事実:

| 機能 | 必要な cc | P100 (cc 6.0) | 影響 |
|------|----------|---------------|------|
| DP4A (MMQ カーネル) | >= 6.1 | 不可 | `mul_mat_id` がフォールバックパスに落ちる |
| Tensor Core (MMA) | >= 7.0 | 不可 | cuBLAS FP16 HMMA 不可、FP32 FFMA のみ |
| CUDA Graph | >= 8.0 (Ampere) | 不可 | カーネルラウンチバッチ化不可 |

`mul_mat_id` (MoE expert 計算) の PP パスは以下の最遅フォールバックを使用:
1. `cudaStreamSynchronize` で GPU パイプラインを完全ドレイン
2. expert routing 結果を Device → Host コピー
3. CPU 上で expert ごとにトークンをソート
4. expert ごとにループで cuBLAS GEMM を個別呼び出し

**40 MoE レイヤー × ubatch あたり 2-3 回の `mul_mat_id` = 80-120 回の stream sync per ubatch**

この構造的制約は RDMA バックエンドの最適化では対処できない。

## 改善提案の総合評価

### Tier 1: 即座に適用可能 (コード変更不要)

#### 1-A. ubatch サイズチューニング (`-ub 2048`)

| 項目 | 内容 |
|------|------|
| 期待改善 | **pp2048: +50%, pp16384: +19%** (実証済み) |
| 実装コスト | なし (パラメータ変更のみ) |
| リスク | なし |
| 状態 | **既に検証完了** |

- ub=2048 で飽和 (ub=4096/8192 は追加利得 <0.4%)
- メカニズム: 256 experts / 8 active で tokens/expert が ub=512→2048 で 16→64 に増加し cuBLAS タイル充填率が飽和点に到達
- pp128 には効果なし (ub=128 < デフォルト ub=512)
- 推奨コマンド: `-b 16384 -ub 2048`

#### 1-B. PCIe トポロジ活用 (2C+2R vs 4C)

| 項目 | 内容 |
|------|------|
| 期待改善 | **pp2048: +15.8%**, pp128: -0.6%, tg32: -9.7% |
| 実装コスト | なし (GPU 構成変更のみ) |
| リスク | tg が 10% 劣化 |
| 状態 | **既に検証完了** |

- Node1 GPU3-6 が同一 PCIe スイッチ (PIX) にあり帯域幅競合
- 2C+2R はノード間でメモリサブシステムを分散し PCIe 競合を解消
- pp サイズ依存: pp ≤ 512 は 4C ローカルが有利、pp ≥ 2048 は 2C+2R が有利

#### 1-C. CUDA/RDMA Overlap

| 項目 | 内容 |
|------|------|
| 期待改善 | **pp16384: +1.7% (4GPU), +2.4% (8GPU)** |
| 実装コスト | なし (環境変数のみ) |
| リスク | なし |
| 状態 | **既に実装・検証完了** |

- `GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1`
- compute-bound (94.8%) のため隠蔽可能な通信時間が少なく効果は小さい

### Tier 2: 低コストで試行可能

#### 2-A. Gate+Up 重みマージ GGUF

| 項目 | 内容 |
|------|------|
| 期待改善 | **+2-5%** (Agent C の +15-25% は交差検証で過大評価と判明) |
| 実装コスト | GGUF 再変換のみ (コード変更不要) |
| リスク | 低 (フォールバックパスあり) |
| 状態 | 未検証 |

- `ffn_gate_exps` + `ffn_up_exps` → `ffn_gate_up_exps` に結合し `mul_mat_id` 呼び出しを 3→2 に削減
- コードは対応済み (`llama-model.cpp:2991` の `create_tensor_gate_up_exps`)
- 現在の unsloth GGUF は分離格納 → `convert_hf_to_gguf.py` で再変換が必要
- **計算量は不変** (行列サイズが 2倍になる)。削減されるのはカーネルラウンチ + D2H/H2D sync 1回分

**過大評価の根拠**: Agent C は「mul_mat_id 呼び出し 33% 削減 = 性能 15-25% 改善」と推定したが、Discussion-AC の検証で「計算量自体は変わらず、削減されるのは sync オーバーヘッドのみ」と修正。P100 のフォールバックパスでは sync 1回あたり 5-20us × 40レイヤー = 0.2-0.8ms の削減に過ぎない。

#### 2-B. MMQ 強制有効化 (ビルドフラグ変更)

| 項目 | 内容 |
|------|------|
| 期待改善 | **+5-20%** (stream sync 排除による、要実測) |
| 実装コスト | ビルドフラグ変更のみ |
| リスク | 中 (P100 に DP4A 命令がないため、ソフトウェアエミュレーションの性能が未知) |
| 状態 | **未検証 (新発見)** |

Discussion-Novel が発見した最も興味深い提案:

- 現在のビルド: `CMAKE_CUDA_ARCHITECTURES=native` → `sm_60` のみ
- `CMAKE_CUDA_ARCHITECTURES="60;61-virtual"` + `GGML_CUDA_FORCE_MMQ=ON` でビルドすると、MMQ カーネルが PTX JIT で有効化される可能性
- MMQ が有効になれば `mul_mat_id` の stream sync + CPU sort が完全に排除される
- **核心の問い**: P100 でソフトウェアエミュレートされる DP4A の MMQ が、現在のフォールバック (sync + per-expert cuBLAS) より速いか？

```
現在のフォールバック: sync(5-20us) + sort(CPU) + expert×8 cuBLAS calls
MMQ パス: GPU 内で完結、sync なし、DP4A ソフトウェアエミュレーション
```

stream sync の回数 (80-120回/ubatch) を考えると、sync 排除だけで 0.4-2.4ms の改善が見込める。さらに GPU パイプラインが途切れなくなることで cuBLAS のスループットも向上する可能性がある。

**検証方法**:
```bash
rm -rf build && cmake -B build -DGGML_CUDA=ON -DGGML_RDMA=ON -DCMAKE_CUDA_ARCHITECTURES="60;61-virtual" -DGGML_CUDA_FORCE_MMQ=ON && cmake --build build -j16
```

### Tier 3: 中程度の実装コスト

#### 3-A. tg コマンドバッチング

| 項目 | 内容 |
|------|------|
| 期待改善 | tg: +3-5% (PP には影響なし) |
| 実装コスト | 中 (RDMA プロトコル変更) |
| リスク | 低 |

- 複数 split コマンドを 1つの IB メッセージにまとめ、tg の RDMA RTT を削減
- tg のみの改善だが、tg の RDMA 比 -10% 損失の部分回復に有効

#### 3-B. Shared Expert + MoE Expert 並列化

| 項目 | 内容 |
|------|------|
| 期待改善 | 理論上 +10-16% (Amdahl's law: Shared Expert は FFN の ~20%) |
| 実装コスト | 高 (ggml の単一ストリームモデル変更が必要) |
| リスク | 高 (アーキテクチャレベルの変更) |

- Shared Expert FFN と MoE FFN は同じ入力を共有し出力は独立 → 理論上並列実行可能
- 現在は ggml が単一 CUDA ストリームで逐次実行
- マルチストリーム対応は ggml-cuda バックエンドの根本改造が必要

### Tier 4: 適用不可能 / 効果なし

| 提案 | 不可の理由 |
|------|-----------|
| CUDA Graph | P100 cc=600 < Ampere cc=800 で無効化 |
| Expert Parallelism | allreduce 通信コスト > 計算節約。過去の試行で断念済み |
| `GGML_SCHED_MAX_COPIES` 変更 | compute-bound で パイプライン深度増加は無効 |
| スレッド数 (`-t`) 変更 | 全レイヤー GPU オフロードで CPU 演算は最小限 |
| 量子化フォーマット変更 | Q4_K_M は compute/quality バランス良好。Q8_0 は VRAM 不足 |
| CUDA 環境変数 | 有効なものは発見されず (詳細は下表) |

#### 検証した CUDA 環境変数

| 環境変数 | 結果 |
|----------|------|
| `GGML_CUDA_NO_PINNED` | 逆効果 (H2D 転送低下) |
| `GGML_CUDA_REGISTER_HOST` | layer split では H2D が少なく影響なし |
| `GGML_CUDA_GRAPH_OPT` | P100 では CUDA Graph 自体が無効 |
| `GGML_CUDA_DISABLE_FUSION` | 逆効果 (MoE topk fusion が無効になる) |
| `GGML_OP_OFFLOAD_MIN_BATCH` | layer split のスケジューリングに影響なし |
| `GGML_CUDA_PEER_MAX_BATCH_SIZE` | layer split では使用されない |

## エージェント間の議論ハイライト

### 議論 1: Gate+Up マージの改善幅

- **Agent C**: 「mul_mat_id 呼び出し 33% 削減で +15-25%」
- **Discussion-AC**: 「計算量は不変。行列サイズが 2倍になるだけ。削減されるのは sync オーバーヘッド 1回分/レイヤーのみ。修正後見積もり: +2-5%」
- **結論**: Agent C は「呼び出し回数削減 = 同比率の性能改善」と誤って仮定。実際は GEMM 計算が支配的で、sync オーバーヘッド削減の効果は限定的

### 議論 2: CUDA Graph の適用可能性

- **Agent A**: 「MoE の動的 expert 選択で静的グラフキャプチャが困難。+5-15%」
- **Agent C**: 「cudaStreamSynchronize が原因で無効化。+5-15%」
- **Discussion-AC**: 「P100 (cc 600) では CUDA Graph 自体がアーキテクチャ制約で完全に無効化。改善幅は 0%」
- **結論**: 両エージェントとも P100 の cc チェック (`cc < GGML_CUDA_CC_AMPERE`) を見落としていた

### 議論 3: Expert 数の訂正

- **初期仮定**: 64 experts / 4 active
- **Discussion-B の訂正**: **256 experts / 8 active** (既存レポートから確認)
- **影響**: tokens/expert の計算が変わるが、ubatch 飽和点の結論は同じ (ub=2048 で 64 tokens/expert が cuBLAS タイル充填の飽和点)

### 議論 4: MMQ 強制有効化の実現可能性

- **Discussion-Novel**: 「`CMAKE_CUDA_ARCHITECTURES=61-virtual` でビルドすれば MMQ が有効化され stream sync が排除される。+5-20%」
- **懸念**: P100 に物理的な DP4A 命令がないため、PTX JIT でソフトウェアエミュレーションになる。エミュレーションの性能オーバーヘッドが sync 排除の利得を上回る可能性
- **結論**: 実測が必要。ビルドフラグ変更のみで試行可能なためコストは低い

## 推奨アクションプラン

### 即座に実行 (コスト: 極低)

1. **ubatch チューニング** — pp2048 以上では `-b 16384 -ub 2048` を標準設定に
2. **pp サイズ別 GPU 構成選択** — pp ≤ 512: ローカル GPU、pp ≥ 2048: 2C+2R

### 検証実験 (コスト: 低)

3. **MMQ 強制有効化ビルド** — `CMAKE_CUDA_ARCHITECTURES="60;61-virtual"` + `GGML_CUDA_FORCE_MMQ=ON` で A/B テスト
4. **Gate+Up マージ GGUF** — `convert_hf_to_gguf.py` で再変換し A/B テスト

### 将来検討 (コスト: 高)

5. **Shared Expert 並列化** — ggml マルチストリーム対応が前提
6. **mul_mat_id バッチ GEMM** — upstream CUDA バックエンド改造

## PP 改善の理論的上限

現在の PP128 ~200 t/s (4GPU) に対し:

| 最適化 | 理論改善幅 | 累積 t/s |
|--------|----------|---------|
| ベースライン | — | 200 |
| Gate+Up マージ | +2-5% | 204-210 |
| MMQ 強制 (楽観) | +5-20% | 214-252 |
| Shared Expert 並列 | +10-16% | 235-292 |
| **理論上限 (全適用)** | | **~250-290** |

ただし Shared Expert 並列化は ggml アーキテクチャ変更が必要なため、現実的な上限は **~210-250 t/s** (MMQ 強制の効果次第)。

## 結論

Qwen3.5-35B-A3B の PP 性能は **P100 の MoE フォールバックパス** によって構造的に制限されている。RDMA バックエンドのオーバーヘッドは 0.02% で最適化の余地はない。

**最もコスト効率の高い次のアクション**は:
1. **MMQ 強制有効化ビルドの検証** (ビルドフラグ変更のみ、stream sync 排除で +5-20% の可能性)
2. **Gate+Up マージ GGUF の検証** (モデル再変換のみ、+2-5%)

これらは両方ともコード変更不要で試行可能であり、合計で +7-25% の改善余地がある。
