# Pipeline Parallelism Phase 1: Async Ubatch Dispatch 実装レポート

- **実施日時**: 2026年3月2日 23:31
- **ワークツリー**: `.worktree/pipeline-parallelism`
- **ブランチ**: `feature/pipeline-parallelism`
- **コミット**: `8243337ba`

## 前提・目的

Qwen3.5 6GPU のプロファイリング（[参照](report/2026-03-02_195335_pp_profiling_qwen35_optimization.md)）で pp ボトルネックがサーバー側 GPU 計算（94.8%）であることが判明。Pipeline Parallelism でデバイス間のアイドル時間を最小化し、通信バウンドモデルでの pp 高速化を目指す。

### 背景: Copy Slot メカニズム

`ggml_backend_sched` は `n_copies = GGML_SCHED_MAX_COPIES(4)` のバッファコピーを確保し、`ggml_backend_sched_alloc_graph()` 呼び出しごとに copy slot をローテーションする:

```cpp
sched->cur_copy = sched->next_copy;
sched->next_copy = (sched->next_copy + 1) % sched->n_copies;
```

各 copy slot はクロスバックエンド入力転送用の独立メモリを持つ。異なる copy slot の ubatch は、前の ubatch の入力転送メモリと競合しないため、GPU 0 が ubatch 1 を処理中に GPU 1 が ubatch 0 を処理する並列実行が可能になる。

### 問題: Graph Reuse が Copy Slot ローテーションを阻害

Graph reuse が有効な場合、`sched_reset` と `alloc_graph` がスキップされるため copy slot がローテーションせず、全 ubatch が同一 copy slot を使用する。これにより、クロスバックエンド転送が直列化され、パイプライン並列の効果が得られない。

## 実装

### 変更内容（1ファイル、6行追加/1行削除）

**`src/llama-context.cpp` — `process_ubatch()`**:

```cpp
// 変更前
if (!graph_reuse_disable && res->can_reuse(gparams)) {

// 変更後
const bool pipeline_pp = cparams.pipeline_parallel && ubatch.n_tokens > 1;
if (!graph_reuse_disable && !pipeline_pp && res->can_reuse(gparams)) {
```

`pipeline_parallel` が有効かつ prompt 処理（`n_tokens > 1`）の場合、graph reuse を無効化して copy slot ローテーションを強制する。

### 計画からの変更点

当初計画では Phase A（全 ubatch dispatch）→ Phase B（synchronize）→ Phase C（結果収集）の3フェーズ構造を予定していたが、コード解析で以下の問題を発見し、既存のループ構造をそのまま活用する方針に変更した:

1. **`gallocr` は計算テンソルを copy slot に関係なく同一 GPU メモリに割り当てる**: Copy slot は**クロスバックエンド入力コピーテンソル**のみ分離する。`t_logits` 等の計算結果テンソルは全 ubatch で同一メモリを共有する
2. **Phase C で過去の ubatch の結果を読み取れない**: 全 ubatch を dispatch すると、最後の ubatch が計算結果メモリを上書きしており、Phase C 時点では最後の ubatch の結果のみ有効
3. **既存の `tensor_get_async` パターンが正しい**: 各 `process_ubatch` 直後に `tensor_get_async` を呼ぶことで、GPU ストリームの順序保証により前の ubatch の結果読み取りが次の ubatch の計算前に実行される

結論: **graph reuse 無効化のみが必要な変更であり、decode ループの構造変更は不要**。

### なぜ動作するか

GPU ストリーム上の操作順序:
1. Ubatch 0: compute → tensor_get_async（GPU アドレスを即座にキャプチャ）
2. Ubatch 1: sched_reset → alloc_graph（copy slot 1）→ set_inputs → compute
3. ストリーム順序保証: step 1 の tensor_get は step 2 の compute の前に完了

Copy slot ローテーションにより:
- Ubatch 0 のクロスバックエンド転送（copy 0）と ubatch 1 の転送（copy 1）が別メモリを使用
- GPU 0 が ubatch 1 を処理中に GPU 1 が ubatch 0 を処理可能

## 検証

### 正確性テスト

```bash
CUDA_VISIBLE_DEVICES=4,5 llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1' -sm layer -ngl 999 -fa 1 \
  -ub 32 -p "Explain RDMA in detail" -n 64 \
  --single-turn --simple-io --log-file /tmp/llama-cli.log
```

**結果**: 正常な英語テキスト出力（ガーベジ文字なし）。pp=62.9 t/s, tg=40.0 t/s。

### 性能テスト（Qwen3.5-35B-A3B, 2GPU CUDA-only）

| テスト | Baseline (graph reuse ON) | Pipeline (graph reuse OFF for pp) | 差 |
|--------|:------------------------:|:----------------------------------:|:----:|
| pp128 (4 ubatches) | 108.70 t/s | 108.75 t/s | +0.05% |
| pp512 (16 ubatches) | 109.76 t/s | 109.07 t/s | -0.63% |
| tg32 | 39.55 t/s | 40.00 t/s | +1.1% |

- **Baseline**: `feature/rdma-backend` (commit `09a531964`)
- **Pipeline**: `feature/pipeline-parallelism` (commit `8243337ba`)
- 条件: `CUDA_VISIBLE_DEVICES=4,5`, `-sm layer -ngl 999 -fa 1 -ub 32`, 各3回

### 分析

- **pp**: 差は測定誤差範囲内（±1%未満）。Qwen3.5 はプロファイリングで compute-bound（94.8%）であることが確認済みのため、通信最適化の効果が出ないのは想定通り
- **tg**: `pipeline_pp` 条件は `n_tokens > 1` でのみトリガーされるため、tg（n_tokens == 1）は影響なし
- **pp512 の -0.63%**: graph build の追加オーバーヘッド（~1ms/ubatch × 16 = ~16ms、total ~4.66s に対して 0.34%）の可能性

### 今後のテスト

通信バウンドモデル（GLM-4.7 IQ2_M、11GPU RDMA 構成）でのテストが必要。RDMA バックエンドでは `async` と `events` の capability が報告されない可能性があり、その場合 `pipeline_parallel=false` となりパイプラインが無効化される。RDMA 統合は別フェーズで対応。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `8243337ba (feature/pipeline-parallelism)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 40°C | 42°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1222762) |
