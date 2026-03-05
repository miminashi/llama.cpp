# 非RDMA PP性能改善: 調査レポート

- **実施日時**: 2026年3月2日 20:07
- **ワークツリー**: なし（コード調査のみ、`feature/rdma-backend` ブランチ上で実施）
- **ブランチ**: `feature/rdma-backend`
- **コミット**: `c1ca6d5fd`

## 前提・目的

前回のプロファイリング ([pp_profiling_qwen35_optimization](2026-03-02_195335_pp_profiling_qwen35_optimization.md)) で、Qwen3.5 6GPU の pp ボトルネックは **サーバー側 GPU 計算** (graph_compute の 94.8%) であり、RDMA 通信最適化の余地がないことが判明した。

- **背景**: RDMA レベルの最適化 (selective signaling, double-buffering 等) は通信バウンドの構成 (GLM-4.7 11GPU) では効果的だが、計算バウンドの構成 (Qwen3.5 6GPU) では pp に影響しない
- **目的**: RDMA 以外の手法で pp 改善が可能かを調査し、各候補の実現可能性・インパクト・変更箇所を特定する
- **前提条件**: upstream マージは目指さないため、互換性を壊す変更も許容
- **対象環境**: Node 1 (P100×7, CC=600) + Node 2 (P100×4, CC=600)

## llama.cpp PP アーキテクチャ概要

### 推論ループ (`src/llama-context.cpp`)

1. **decode** (`:1466`) — バッチ全体を受け取る
2. **memory->init_batch** (`:1510`) — `n_ubatch` サイズで ubatch に分割
3. **process_ubatch** (`:1070`) — 各 ubatch に対しグラフ構築・計算
   - `model.build_graph()` (`:1096`) — 計算グラフ構築
   - `ggml_backend_sched_alloc_graph()` (`:1106`) — バッファ割当
   - `res->set_inputs()` (`:1117`) — 入力テンソル設定
   - `graph_compute()` (`:1122`) — グラフ計算実行
4. **graph_compute** (`:2055`) — `ggml_backend_sched_graph_compute_async()` を呼出

### スケジューラ (`ggml/src/ggml-backend.cpp`)

- **Split 実行** (`ggml_backend_sched_compute_splits`, `:1445`):
  - 各 split のテンソルコピー → `ggml_backend_graph_compute_async()` → イベント記録
  - **split は逐次実行** — 前の split の計算完了を待ってから次の split を開始
- **Pipeline parallelism インフラ** (`:714-718`):
  - `n_copies` (最大 `GGML_SCHED_MAX_COPIES=4`, `:672`) のバッファが事前確保される
  - `parallel=true` で有効化されるが、現在は `cb_eval` コールバック経由でのみ利用
- **graph_optimize** (`:1326`): split ごとに呼ばれ、CUDA Graph 用のノード最適化を実行

### CUDA Graph とストリーム最適化 (`ggml/src/ggml-cuda/ggml-cuda.cu`)

- **CUDA Graph** (`:3918`): `graph_compute` 内で capture/replay
- **graph_optimize** (`:4002`): QKV 並列ストリーム (attn_norm からの fan-out 検出)
- **互換性チェック** (`:2870-2902`): split buffer, `MUL_MAT_ID` 等で無効化

## 最適化候補の詳細分析

### 候補一覧

| # | 最適化 | PP改善見込み | 実装難度 | 対象モデル |
|---|-------|------------|---------|-----------|
| 1 | CUDA Graph 有効化 (P100) | 5-15% | 低 | GLM-4.7 (Dense) |
| 2 | Concurrent QKV Streams | 10-20% | 低-中 | Dense モデル |
| 3 | Pipeline Parallelism (マイクロバッチ) | 30-60% | 高 | 全モデル (長プロンプト) |
| 4 | MoE CUDA Graph 対応 | 5-15% | 中 | Qwen3.5 (MoE) |
| 5 | n_ubatch チューニング | 0-10% | なし | 全モデル |
| 6 | FULL_GRAPH flush 非同期化 | 0-5% | 低 | 全モデル |

---

### 1. CUDA Graph 有効化 (P100) — 推奨度: ★★★

**現状分析**

`ggml-cuda.cu:3906` で CC < Ampere (800) の場合、CUDA Graph が無効化される:

```cpp
// ggml/src/ggml-cuda/ggml-cuda.cu:3906
if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
    if (!graph->disable_due_to_gpu_arch) {
        GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
    }
    graph->disable_due_to_gpu_arch = true;
}
```

P100 は CC=600 (`GGML_CUDA_CC_PASCAL`, `common.cuh:47`) で、API レベルでは CUDA Graph をサポートする (CUDA 10.0+, CC 3.5+)。

**変更内容**

`ggml-cuda.cu:3906` の条件を CC >= PASCAL に緩和:

```diff
- if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
+ if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_PASCAL) {
```

**期待効果**

- CUDA Graph の capture/replay により、カーネル起動オーバーヘッドを削減
- PP 時の多数のカーネル呼び出し (n_nodes ≈ 1065) で効果が大きい
- 期待改善: **5-15%** (カーネル起動時間の割合に依存)

**制約・リスク**

- `cudaGraphExecUpdate()` は Ampere+ で効率化されたが、P100 でも基本的な capture/replay は動作する
- P100 では graph update が毎回 recreate になる可能性 → 2回目以降で判断
- **MoE (mul_mat_id) では別途無効化** (`ggml-cuda.cu:2886-2890`) → Qwen3.5 には効果なし

**対象モデル**: GLM-4.7 (Dense) のサーバー側単一 GPU 計算

---

### 2. Concurrent QKV Streams — 推奨度: ★★★

**現状分析**

`ggml-cuda.cu:4002-4244` に QKV 並列ストリーム実行の完全な実装がある。`attn_norm` ノードからの fan-out (Q, K, V の3分岐) を検出し、各分岐を別ストリームで並列実行する。

ただし2つの条件で制限される:

```cpp
// ggml/src/ggml-cuda/ggml-cuda.cu:4014-4017
static bool enable_graph_optimization = [] {
    const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
    return env != nullptr && atoi(env) == 1;
}();

// ggml/src/ggml-cuda/ggml-cuda.cu:4026
if (!use_cuda_graph || ggml_backend_cuda_get_device_count() != 1) {
    return;
}
```

1. **環境変数 `GGML_CUDA_GRAPH_OPT=1`** が必要 (デフォルト無効)
2. **`device_count == 1`** — マルチ GPU 環境では無効化

**変更内容**

`device_count` チェックを削除:

```diff
// ggml/src/ggml-cuda/ggml-cuda.cu:4026
- if (!use_cuda_graph || ggml_backend_cuda_get_device_count() != 1) {
+ if (!use_cuda_graph) {
```

**原理**

`graph_optimize` はバックエンドコンテキスト (`cuda_ctx`) 内で動作し、他デバイスには依存しない。`device_count` チェックは保守的な制限であり、各デバイスのバックエンドは独立して `graph_optimize` を呼ぶ (`ggml-backend.cpp:1326`)。

**期待効果**

- Q, K, V の計算 (主に `mul_mat`) を3ストリームで並列実行
- PP ではこれらが独立した計算であり、GPU の SM 占有率が低い場合に並列化の効果大
- 期待改善: **10-20%** (P100 の SM 数 = 56 に対する Q/K/V 個別の SM 占有率に依存)

**前提条件**

- 候補1 (CUDA Graph 有効化) が先に必要 — `use_cuda_graph` が false だと early return
- `GGML_CUDA_GRAPH_OPT=1` 環境変数の設定が必要

**対象モデル**: Dense モデル (GLM-4.7 等)。MoE では CUDA Graph 自体が無効化されるため効果なし。

---

### 3. Pipeline Parallelism (マイクロバッチ) — 推奨度: ★★ (高インパクトだが高難度)

**現状分析**

llama.cpp のスケジューラ (`ggml-backend.cpp:1445-1629`) は split を**逐次実行**する。各 split の計算が完了してから次の split に入力をコピーし計算を開始する:

```
Time →
Device 0: [=== compute ===]
Device 1:                   [copy][=== compute ===]
Device 2:                                         [copy][=== compute ===]
...
```

Pipeline parallelism では、プロンプトをマイクロバッチに分割し、各デバイスが異なるマイクロバッチを並列処理する:

```
Time →
Device 0: [= mb0 =][= mb1 =][= mb2 =][= mb3 =]
Device 1:          [= mb0 =][= mb1 =][= mb2 =][= mb3 =]
Device 2:                   [= mb0 =][= mb1 =][= mb2 =][= mb3 =]
...
```

**既存インフラ**

- `GGML_SCHED_MAX_COPIES=4` (`ggml-backend.cpp:672`) — 4個のバッファコピーが事前確保済み
- `n_copies` (`ggml-backend.cpp:715`) — `parallel=true` で有効化
- テンソルコピーとイベントのパイプライン処理インフラが存在

**理論的改善**

- N=6 stages (devices), M=4 micro-batches: 合計時間 = `(N+M-1)/M` ステージ分
  - 逐次: 6 ステージ → パイプライン: `(6+4-1)/4` = 2.25 ステージ → **62.5% 高速化**
- KV cache 依存: 同一デバイス内で逐次処理すれば自然に満たされる

**必要な変更**

| ファイル | 変更内容 |
|---------|---------|
| `src/llama-context.cpp` | decode ループのマイクロバッチ分割・パイプライン化 |
| `ggml/src/ggml-backend.cpp` | 複数 split の同時実行サポート (事前にコピーを重ね合わせ) |
| KV cache | マイクロバッチ間のポジション管理 |
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 非同期 graph_compute の結果を次マイクロバッチの入力送信と重ね合わせ |

**リスク**

- アーキテクチャ全体の改修が必要 — バグが入りやすい
- KV cache の正しいポジション管理が複雑
- RDMA バックエンドとの統合が必要
- 短プロンプトでは分割オーバーヘッドが支配的になる可能性

**対象モデル**: 全モデル (特に長プロンプトで効果大)

---

### 4. MoE CUDA Graph 対応 — 推奨度: ★★

**現状分析**

`ggml-cuda.cu:2886-2890` で `MUL_MAT_ID` ノードが存在すると CUDA Graph が無効化される:

```cpp
// ggml/src/ggml-cuda/ggml-cuda.cu:2885-2890
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
if (node->op == GGML_OP_MUL_MAT_ID && (!ggml_is_quantized(node->src[0]->type) || node->ne[2] > MMVQ_MMID_MAX_BATCH_SIZE)) {
    // under these conditions, the mul_mat_id operation will need to synchronize the stream, so we cannot use CUDA graphs
    // TODO: figure out a way to enable for larger batch sizes, without hurting performance
    // ref: https://github.com/ggml-org/llama.cpp/pull/18958
    use_cuda_graph = false;
}
```

`MMVQ_MMID_MAX_BATCH_SIZE` は `mmvq.cuh:4` で **4** と定義。PP (batch_size=128) では `ne[2] > 4` → 常に CUDA Graph 無効。

**MUL_MAT_ID の実装** (`mmid.cu:1-165`)

`mm_ids_helper` カーネル (`mmid.cu:28`) は expert routing を計算する。このカーネル自体はストリーム同期を必要としないが、後続の `ggml_cuda_mul_mat_id` (`ggml-cuda.cu:2268-2310`) で大バッチサイズ時に MMVQ/MMQ ではなくフォールバックパスを使用し、そこでストリーム同期が発生する (`ggml-cuda.cu:2311`):

```cpp
// ggml-cuda.cu:2309-2311
// note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
cudaStream_t stream = ctx.stream();
```

**変更方針**

2つのアプローチが考えられる:

A. **MoE 部分だけ CUDA Graph 外にする**: Graph capture を MoE ノード前後で分割し、MoE 以外の部分 (attention, norm, FFN の非 MoE 部分) を capture
B. **MUL_MAT_ID のストリーム同期を排除**: `mm_ids_helper` → `mul_mat` の一連の処理を capture-friendly に書き換え

**期待効果**: 5-15% (MoE モデルの非 MoE 部分のカーネル起動削減)

**対象モデル**: Qwen3.5 (MoE) のサーバー側計算

---

### 5. n_ubatch チューニング — 推奨度: ★★★ (即座に検証可能)

**現状分析**

`n_ubatch` は llama.cpp がプロンプトをサブバッチに分割するサイズ。デフォルトは `n_batch` (通常 512):

```cpp
// src/llama-context.cpp:156
cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);
```

PP 中のループ (`llama-context.cpp:1560`):

```cpp
do {
    const auto & ubatch = mctx->get_ubatch();
    // ... process_ubatch() for each ubatch
} while (mctx->next());
```

`n_ubatch` がプロンプト長より小さい場合、複数回の `process_ubatch` が呼ばれ、それぞれで完全な graph_compute サイクルが発生する。

**検証方法**

パラメータのみの変更で即座に検証可能。コード変更不要:

```bash
llama-bench -m model.gguf -ngl 999 -ub 64 ...   # 小さい ubatch
llama-bench -m model.gguf -ngl 999 -ub 128 ...
llama-bench -m model.gguf -ngl 999 -ub 256 ...
llama-bench -m model.gguf -ngl 999 -ub 512 ...  # デフォルト
```

**期待効果**

- 大きい `n_ubatch`: graph_compute の呼び出し回数削減、ただし GPU メモリ使用量増加
- 小さい `n_ubatch`: メモリ節約、ただし追加の graph build/compute オーバーヘッド
- 最適値はモデル・GPU メモリ・プロンプト長に依存
- 期待改善: **0-10%** (既にデフォルトが妥当な場合は効果なし)

**対象モデル**: 全モデル

---

### 6. FULL_GRAPH flush 非同期化 — 推奨度: ★

**現状分析**

RDMA バックエンドの `graph_compute` パスには3つのモードがある:

| モード | FLUSH | COMPUTE | 統合 |
|--------|-------|---------|------|
| recompute | ✅ 統合済み (FLUSH_AND_RECOMPUTE_ASYNC) | 非同期 | 1コマンド |
| compute_update | ✅ 統合済み (FLUSH_AND_COMPUTE_UPDATE_ASYNC) | 非同期 | 1コマンド |
| **FULL_GRAPH** | ❌ 同期 (FLUSH_ALL_STAGING) | 別コマンド | 未統合 |

FULL_GRAPH path (`ggml-rdma.cpp:1576-1612`) では flush が同期コマンドのため、サーバーが前デバイスの計算を完了するまでクライアントがブロックされる。

**変更内容**

新コマンド `RDMA_CMD_FLUSH_AND_GRAPH_COMPUTE_ASYNC` を新設し、flush + graph compute を1コマンドに統合。

**効果の限界**

前回のプロファイリングで確認済み:
- クライアント側のブロッキング: 321.64 ms → 削減可能
- **しかしサーバー側の計算合計は変わらない** (~300 ms)
- 結果収集 (`get_tensor`/`synchronize`) で同じ時間待つ
- **PP スループットには実質影響なし** — サーバー計算がクリティカルパス

**対象モデル**: 全モデル (ただし効果は限定的)

## 推奨実装順序

### Phase 1: 即座に検証 (コード変更なし)

1. **候補5: n_ubatch チューニング** — パラメータ変更のみで効果を確認
   - `-ub 64`, `-ub 128`, `-ub 256`, `-ub 512` の比較
   - GLM-4.7 11GPU と Qwen3.5 6GPU の両方で検証

### Phase 2: 低コスト実装 (1行変更)

2. **候補1: CUDA Graph 有効化 (P100)** — CC チェックの1行変更
   - `ggml-cuda.cu:3906` の条件を `GGML_CUDA_CC_PASCAL` に緩和
   - まず local ビルドで crash しないことを確認
   - GLM-4.7 サーバー側で A/B ベンチマーク

3. **候補2: Concurrent QKV Streams** — `device_count` チェックの1行変更
   - 候補1 が成功した場合のみ (CUDA Graph が前提)
   - `GGML_CUDA_GRAPH_OPT=1` + 候補1 の組み合わせで検証

### Phase 3: 中程度の実装

4. **候補4: MoE CUDA Graph 対応** — Qwen3.5 特化の改善
   - MoE 部分の CUDA Graph 分割が必要
   - 候補1 が動作確認後に着手

### Phase 4: 大規模改修 (将来課題)

5. **候補3: Pipeline Parallelism** — 最も高インパクトだが高リスク
   - アーキテクチャ改修が必要、POC から段階的に進める

### 優先度マトリクス

```
                高インパクト
                    │
        ┌───────────┼───────────┐
        │     3     │     2     │
        │ Pipeline  │  QKV      │
高難度 ─┤           │  Streams  ├─ 低難度
        │     4     │   1  5    │
        │ MoE Graph │ CUDA  ub  │
        └───────────┼───────────┘
                    │
                低インパクト
```

**推奨**: Phase 1 (候補5) → Phase 2 (候補1→2) の順で進める。これにより最小限の変更で最大の効果を検証できる。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c1ca6d5fd (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 39°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1203493) |
