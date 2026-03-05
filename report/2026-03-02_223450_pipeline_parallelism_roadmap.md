# Pipeline Parallelism (マイクロバッチ) 実装ロードマップ

- **実施日時**: 2026年3月2日 22:34
- **ワークツリー**: なし（コード調査・設計のみ、`feature/rdma-backend` ブランチ上で実施）
- **ブランチ**: `feature/rdma-backend`
- **コミット**: `c1ca6d5fd`

## 前提・目的

### 背景

[pp_profiling_qwen35_optimization](2026-03-02_195335_pp_profiling_qwen35_optimization.md) のプロファイリングにより、Qwen3.5 6GPU の pp ボトルネックは**サーバー側 GPU 計算** (graph_compute の 94.8%) であることが判明。RDMA 通信最適化 (selective signaling, double-buffering 等) は通信バウンドの構成でのみ効果的であり、計算バウンドの構成では pp スループットを改善できない。

[non_rdma_pp_optimization_survey](2026-03-02_200704_non_rdma_pp_optimization_survey.md) で6つの最適化候補を分析した結果、Pipeline Parallelism が最も高いインパクト（理論上 30-60% 改善）を持つことが判明。ただし実装難度も最高であるため、段階的な実装ロードマップが必要。

### 目的

- Qwen3.5 35B-A3B (MoE, 3B active) と Qwen3.5 27B (dense) の PP スループットを大幅に改善する
- 計算バウンド構成において、デバイス間のパイプライン並列性を活用し、GPU idle 時間を最小化する
- 既存の RDMA バックエンドと統合し、マルチノード構成でも動作させる

### 対象モデル

| モデル | タイプ | Active Params | テスト構成 |
|--------|--------|:------------:|-----------|
| Qwen3.5-35B-A3B | MoE | 3B | 4GPU (2C + 2R) |
| Qwen3.5-27B | Dense | 27B | 4GPU (2C + 2R) |

> **注記**: プロファイリングデータ (94.8% GPU compute 等) は過去の 6GPU (2C+4R) 構成での測定結果を参照。今後の実装・検証は 4GPU (2C+2R) 構成で実施する。

### 参照レポート

- [PP プロファイリング: Qwen3.5 6GPU](2026-03-02_195335_pp_profiling_qwen35_optimization.md)
- [非RDMA PP性能改善 調査](2026-03-02_200704_non_rdma_pp_optimization_survey.md)
- [n_ubatch チューニング](2026-03-02_205900_n_ubatch_tuning_benchmark.md)
- [CUDA Graph P100 ベンチマーク](2026-03-02_214717_cuda_graph_p100_benchmark.md)
- [MoE CUDA Graph 実現可能性](2026-03-02_222455_moe_cuda_graph_feasibility.md)

---

## 1. Pipeline Parallelism の原理

### 現在の逐次実行パターン

現在のスケジューラ (`ggml_backend_sched_compute_splits`, `ggml-backend.cpp:1445`) は split を逐次実行する。各 split は 1 つのバックエンド（= 1 GPU）に対応し、前の split の計算完了を待ってから次の split に入力をコピーして計算を開始する:

```
Time →
               ┌─ split 0 ─┐
Device 0 (GPU): [== compute ==]
                             ┌─ split 1 ─┐
Device 1 (GPU):               [copy][== compute ==]
                                                    ┌─ split 2 ─┐
Device 2 (GPU):                                      [copy][== compute ==]

Total time = T_compute × N_devices + T_copy × (N_devices - 1)
```

Qwen3.5 6GPU のプロファイリングでは、この逐次パターンにより pp graph_compute の **94.8%** がサーバー GPU 計算待ち (flush wait) に費やされていた。

### Pipeline Parallelism の実行モデル

プロンプト (128 トークン) を M 個のマイクロバッチに分割し、各デバイスが異なるマイクロバッチをパイプライン的に処理する:

```
Time →
Device 0: [= mb0 =][= mb1 =][= mb2 =][= mb3 =]
Device 1:          [= mb0 =][= mb1 =][= mb2 =][= mb3 =]
Device 2:                   [= mb0 =][= mb1 =][= mb2 =][= mb3 =]
Device 3:                            [= mb0 =][= mb1 =][= mb2 =][= mb3 =]
          |<──── fill ────>|<── steady state ──>|<──── drain ────>|
```

### 理論的改善値

パイプライン並列の合計時間:

```
T_pipeline = (N + M - 1) × T_stage
T_sequential = N × M × T_stage

Speedup = (N × M) / (N + M - 1)
```

N=4 (デバイス), M=4 (マイクロバッチ) の場合:

| 構成 | 逐次時間 | パイプライン時間 | 高速化率 |
|------|---------|----------------|---------|
| N=4, M=2 | 8 stages | 5 stages | **1.60x** (+60%) |
| N=4, M=4 | 16 stages | 7 stages | **2.29x** (+129%) |
| N=4, M=8 | 32 stages | 11 stages | **2.91x** (+191%) |

ただし実際には:
- パイプライン fill/drain のオーバーヘッド
- マイクロバッチ間のデータ転送コスト
- KV キャッシュの順序制約による同期コスト
- マイクロバッチサイズの縮小による GPU 効率低下

現実的な期待値: **30-60%** 改善 (M=4, オーバーヘッド込み)

---

## 2. 現状分析: 既存インフラの棚卸し

### 完成済み (~60%)

| コンポーネント | ファイル:行 | 状態 | 詳細 |
|--------------|-----------|------|------|
| バッファ多重コピー | `ggml-backend.cpp:715` | **完成** | `n_copies = GGML_SCHED_MAX_COPIES(4)` で 4 コピー確保 |
| テンソルコピー配列 | `ggml-backend.cpp:698` | **完成** | `hv_tensor_copies[hash_size][n_backends][n_copies]` |
| コピーローテーション | `ggml-backend.cpp:1773-1774` | **完成** | `cur_copy = next_copy; next_copy = (next_copy + 1) % n_copies` |
| イベント同期 | `ggml-backend.cpp:718` | **完成** | `events[MAX_BACKENDS][MAX_COPIES]` |
| イベント wait/record | `ggml-backend.cpp:1466-1623` | **完成** | split 間でのイベントベース同期 |
| パイプライン有効化条件 | `llama-context.cpp:309-336` | **完成** | multi-device + layer split + offload_kqv + async/events 対応 |
| フォールバック | `llama-context.cpp:474-477` | **完成** | pipeline_parallel でメモリ不足時に自動無効化 |
| RDMA 非同期コマンド | `ggml-rdma.cpp:1487-1604` | **完成** | `RDMA_ASYNC_COMPUTE` で graph_compute を非同期送信 |

### 未実装 (~40%)

| コンポーネント | ファイル:行 | 状態 | 必要な変更 |
|--------------|-----------|------|-----------|
| Split 並列実行 | `ggml-backend.cpp:1453` | **逐次** | `for (split_id = 0...)` ループが逐次 → オーバーラップに |
| マイクロバッチ分割 | `llama-context.cpp:1560` | **未実装** | decode ループでの ubatch 分割・パイプライン化 |
| KV キャッシュ並列管理 | `llama-batch.cpp:289-321` | **制約** | 位置整合性バリデーションが厳密な順序制約 |
| RDMA per-device 追跡 | `ggml-rdma.cpp:1446-1650` | **未実装** | 非同期 graph_compute の完了を device 別に追跡 |
| 結果収集パイプライン | `llama-context.cpp:1580` | **逐次** | `process_ubatch` の結果をマイクロバッチ間で重ね合わせ |

### 既存パイプラインインフラの詳細

`ggml_backend_sched` 構造体 (`ggml-backend.cpp:685-739`) に既にパイプラインの骨格がある:

```cpp
// pipeline parallelism support
int n_copies;   // = 4 (parallel=true) or 1
int cur_copy;   // 現在使用中のコピーインデックス
int next_copy;  // 次に使用するコピーインデックス
ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
```

`alloc_graph` (`ggml-backend.cpp:1773`) でコピーがローテーションされる:

```cpp
sched->cur_copy = sched->next_copy;
sched->next_copy = (sched->next_copy + 1) % sched->n_copies;
```

しかし、`compute_splits` (`ggml-backend.cpp:1445-1629`) は `cur_copy` のみを使用し、split は依然として逐次実行される。パイプラインインフラは「バッファのダブルバッファリング」のために存在しているが、**split 間のオーバーラップ実行**には使われていない。

---

## 3. アーキテクチャ設計

### 3.1 レイヤー分割モデルにおけるパイプラインの動作

llama.cpp のレイヤー分割 (`-sm layer`) では、モデルのレイヤーが各デバイスに順番に割り当てられる。1 つの ubatch の計算は Device 0 → Device 1 → ... → Device N-1 の順に逐次的にパイプラインを通過する:

```
Layer 0-14:  Device 0 (CUDA0)
Layer 15-29: Device 1 (CUDA1)
Layer 30-44: Device 2 (RDMA0)
Layer 45-59: Device 3 (RDMA1)
```

### 3.2 パイプライン実行のタイムライン

pp128, M=4 マイクロバッチ (各 32 トークン) の場合:

```
Step:     0    1    2    3    4    5    6
         ─────────────────────────────────
Dev 0:  [mb0][mb1][mb2][mb3]
Dev 1:       [mb0][mb1][mb2][mb3]
Dev 2:            [mb0][mb1][mb2][mb3]
Dev 3:                 [mb0][mb1][mb2][mb3]
         ─────────────────────────────────
Steps:    7 steps (vs 16 steps sequential)
```

### 3.3 マイクロバッチ分割戦略

#### 分割レベル

ubatch レベル (既存の `n_ubatch` メカニズムを活用):

```
Prompt (128 tokens)
  ├── ubatch 0: tokens 0-31   (pos 0-31)
  ├── ubatch 1: tokens 32-63  (pos 32-63)
  ├── ubatch 2: tokens 64-95  (pos 64-95)
  └── ubatch 3: tokens 96-127 (pos 96-127)
```

現在の `process_ubatch` ループ (`llama-context.cpp:1560-1580`) を修正し、複数の ubatch を同時にパイプラインに投入する。

#### マイクロバッチサイズの選択

| pp サイズ | M=2 (64tok) | M=4 (32tok) | M=8 (16tok) |
|----------|:-----------:|:-----------:|:-----------:|
| 128 | 最も安全 | 推奨 | GPU 効率低下 |
| 256 | 推奨 | 推奨 | 効果的 |
| 512 | 推奨 | 最も効果的 | 効果的 |

推奨: `M = min(4, pp_size / 32)` — 各マイクロバッチが最低 32 トークンを維持。

### 3.4 KV キャッシュ位置管理

#### 問題

KV キャッシュの位置バリデーション (`llama-batch.cpp:289-321`) は**厳密な連続性制約**を持つ:

```cpp
if (seq_pos_min(s) != p0 + 1) {
    ok = false;  // 位置が連続していなければエラー
}
```

パイプライン実行では、同一シーケンスの異なるマイクロバッチが異なるデバイスで同時に処理される。Device 0 が mb1 (pos 32-63) を処理している間、Device 2 はまだ mb0 (pos 0-31) を処理中。各デバイスの KV キャッシュは異なるタイミングで更新される。

#### 解決策

**レイヤー分割ではデバイスごとに独立した KV キャッシュスロットを持つ**ため、この問題は自然に解決される:

1. 各デバイスは自分のレイヤーの KV キャッシュのみを管理
2. 同一デバイス上では mb0 → mb1 → mb2 → mb3 の順に逐次処理
3. 位置の連続性はデバイス内で自然に保証される

```
Device 0 (Layer 0-9):
  mb0 (pos 0-31)  → KV cache updated for pos 0-31
  mb1 (pos 32-63) → KV cache updated for pos 32-63  ← pos 31+1 = 32 ✓
  mb2 (pos 64-95) → KV cache updated for pos 64-95
  mb3 (pos 96-127)→ KV cache updated for pos 96-127

Device 1 (Layer 10-19):
  [idle]           ← mb0 hasn't arrived yet
  mb0 (pos 0-31)  → KV cache updated for pos 0-31
  mb1 (pos 32-63) → KV cache updated for pos 32-63
  mb2 (pos 64-95) → KV cache updated for pos 64-95
```

#### KV キャッシュ `init_batch` の変更

現在の `memory->init_batch()` (`llama-context.cpp:1510`) は一括で KV スロットを確保する。パイプライン化後は各マイクロバッチに対して位置情報を事前に設定し、1回の `init_batch` で全マイクロバッチ分のスロットを確保する必要がある。

---

## 4. 実装フェーズ

### Phase 0: Qwen3.5 27B (Dense) プロファイリング

**目的**: Pipeline Parallelism の効果を正確に予測するためのベースラインデータ取得。

**理由**: MoE モデル (Qwen3.5 35B-A3B) は active パラメータが 3B と少なく、計算バウンドの度合いが dense モデルと異なる。Dense モデルでの通信/計算比率を定量化することで、パイプラインの theoretical speedup をより正確に見積もれる。

**作業内容**:

1. Qwen3.5 27B を 4GPU (2C+2R) で実行 (暫定テスト構成に準拠)
2. `GGML_RDMA_PROFILE=1` でプロファイリング
3. pp128 の FULL_GRAPH path における flush wait / serialize / send の比率を測定
4. TG の recompute path のレイテンシを測定

**変更ファイル**: なし (計測のみ)

**成果物**: プロファイリングレポート (各 split の計算時間・通信時間の内訳)

**期間**: 1日

---

### Phase 1: スケジューラの Split オーバーラップ (CUDA-only)

**目的**: `ggml_backend_sched_compute_splits` を改修し、異なるマイクロバッチの split を並列実行可能にする。

**原理**: 同一マイクロバッチの split は Device 0 → Device 1 → ... の順に依存関係があるが、異なるマイクロバッチの split は異なるデバイス間で独立に実行可能。

#### 1.1 split 実行ループの改修

現在 (`ggml-backend.cpp:1453`):

```cpp
for (int split_id = 0; split_id < sched->n_splits; split_id++) {
    // copy inputs → compute → record event  (逐次)
}
```

改修後:

```cpp
// Pipeline scheduler: 各 step で異なるマイクロバッチの異なる split を並列実行
for (int step = 0; step < n_devices + n_microbatches - 1; step++) {
    for (int mb = 0; mb < n_microbatches; mb++) {
        int device = step - mb;
        if (device < 0 || device >= n_devices) continue;

        // wait for dependency: same device, previous microbatch
        if (mb > 0) {
            event_wait(events[device][mb-1]);
        }
        // wait for dependency: previous device, same microbatch
        if (device > 0) {
            event_wait(events[device-1][mb]);
        }

        // copy inputs & compute
        copy_inputs(split[device], mb);
        graph_compute_async(backend[device], &split[device].graph);

        // record completion
        event_record(events[device][mb]);
    }
}
```

#### 1.2 テンソルコピーの多重化

既存の `n_copies=4` インフラを活用。各マイクロバッチが異なる `copy_id` を使用:

```cpp
// 現在
struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

// 改修後
struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, microbatch_id % sched->n_copies);
```

#### 1.3 必要な変更

| ファイル | 関数 | 変更内容 |
|---------|------|---------|
| `ggml/src/ggml-backend.cpp:1445-1629` | `ggml_backend_sched_compute_splits` | パイプライン split スケジューリング |
| `ggml/src/ggml-backend.cpp:1768-1785` | `ggml_backend_sched_alloc_graph` | マイクロバッチ数分のコピーを管理 |
| `ggml/src/ggml-backend.cpp:685-739` | `ggml_backend_sched` | パイプライン状態 (マイクロバッチ数、進行状況) を追加 |
| `ggml/include/ggml-backend.h` | 公開 API | `ggml_backend_sched_set_pipeline_depth()` 追加 |

#### 1.4 この Phase での制約

- **CUDA バックエンドのみ**で検証 (RDMA なし)
- **単一 ubatch** の split をパイプライン化 (decode ループは変更しない)
- テスト: 1 号機の CUDA4,5 で Qwen3.5 を local CUDA のみで実行 (RDMA なしでパイプラインの基本動作を検証)

**期間**: 3-5日

---

### Phase 2: マイクロバッチ分割とKV キャッシュのパイプライン対応

**目的**: `decode` ループを改修し、複数の ubatch をパイプラインに同時投入可能にする。

#### 2.1 decode ループの改修

現在 (`llama-context.cpp:1560`):

```cpp
do {
    const auto & ubatch = mctx->get_ubatch();
    const auto * res = process_ubatch(ubatch, ...);
    // 結果処理 ...
} while (mctx->next());
```

改修後:

```cpp
// Phase 1: 全マイクロバッチのグラフを構築・割当
std::vector<pipeline_stage> stages;
do {
    const auto & ubatch = mctx->get_ubatch();
    auto * gf = build_and_alloc_graph(ubatch, ...);
    stages.push_back({ubatch, gf});
} while (mctx->next());

// Phase 2: パイプライン実行
pipeline_execute(stages);

// Phase 3: 結果収集
for (auto & stage : stages) {
    collect_results(stage);
}
```

#### 2.2 KV キャッシュの事前スロット確保

`memory->init_batch()` で全マイクロバッチ分の KV スロットを一括確保:

```cpp
// 現在: init_batch が n_ubatch サイズで分割
mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);

// 改修後: パイプライン深度を考慮した分割
// n_ubatch を pipeline_depth で割って micro_ubatch_size を決定
int micro_ubatch = cparams.n_ubatch / pipeline_depth;
mctx = memory->init_batch(*balloc, micro_ubatch, output_all);
```

#### 2.3 位置バリデーションの調整

`llama-batch.cpp:289-321` のバリデーションは変更不要。レイヤー分割では各デバイスが自分の ubatch を順番に処理するため、位置の連続性は自然に保証される。ただし、`init_batch` の内部で ubatch を小さく分割するため、分割ロジックの整合性を確認する必要がある。

#### 2.4 必要な変更

| ファイル | 関数 | 変更内容 |
|---------|------|---------|
| `src/llama-context.cpp:1560-1580` | `decode` (ubatch ループ) | パイプライン実行への書き換え |
| `src/llama-context.cpp:1070-1132` | `process_ubatch` | 非同期版の分離 (graph build + compute を分割) |
| `src/llama-context.cpp:2055-2079` | `graph_compute` | パイプライン版のディスパッチ |
| `src/llama-context.cpp:309-340` | pipeline 初期化 | パイプライン深度パラメータ追加 |
| `src/llama-kv-cache.cpp` | KV スロット管理 | マイクロバッチ対応 (変更は最小限) |

**期間**: 5-7日

---

### Phase 3: RDMA バックエンド統合

**目的**: パイプライン実行を RDMA バックエンドで動作させる。

#### 3.1 RDMA 非同期コマンドの活用

現在の RDMA バックエンドは `RDMA_ASYNC_COMPUTE` で非同期 graph_compute をサポートしている (`ggml-rdma.cpp:1487-1604`):

```cpp
if (RDMA_ASYNC_COMPUTE) {
    bool status = send_rdma_cmd_async(ctx->conn.get(), RDMA_CMD_FLUSH_AND_RECOMPUTE_ASYNC,
                                      input.data(), input.size(), ...);
    // 応答を待たずに返る
}
```

パイプライン実行では、Device 0 に mb0 を送信した直後に Device 1 に mb0 のデータを転送開始する必要がある。`send_rdma_cmd_async` はこのパターンに適合する。

#### 3.2 Per-Device 完了追跡

現在の課題: 非同期コマンドの完了を device 別に追跡する仕組みがない。

```cpp
// 現在: send_rdma_cmd_async は fire-and-forget
// synchronize() で全デバイスの完了を待つ

// 必要: device 別の完了追跡
struct rdma_pipeline_state {
    std::atomic<bool> compute_done[MAX_DEVICES][MAX_MICROBATCHES];
    // or: per-device completion events
};
```

#### 3.3 FULL_GRAPH flush の非同期化

パイプライン実行では FULL_GRAPH path の同期 flush がボトルネックになる (`ggml-rdma.cpp:1576-1581`)。Phase 3 では flush + graph_compute を 1 コマンドに統合する:

```cpp
// 新コマンド: RDMA_CMD_FLUSH_AND_GRAPH_COMPUTE_ASYNC
// recompute path と同様に、flush と graph compute を 1 つの非同期コマンドに統合
send_rdma_cmd_async(conn, RDMA_CMD_FLUSH_AND_GRAPH_COMPUTE_ASYNC,
                    input.data(), input.size(), ...);
```

#### 3.4 サーバー側の並列コマンド処理

現在のサーバーは単一コネクション・単一スレッドでコマンドを順次処理する。パイプライン実行では、異なるデバイスへのコマンドが独立に到着する可能性がある。

**オプション A**: 既存の単一コネクション + コマンドキュー
- クライアントが全マイクロバッチのコマンドを連続送信
- サーバーが FIFO で処理
- **最もシンプルだが、パイプラインの恩恵が限定的**

**オプション B**: Per-device コネクション (既存の `GGML_RDMA_PER_DEVICE_CONN`)
- 各デバイスが独立したコネクションを持つ
- サーバーが各コネクションを独立スレッドで処理
- **完全なパイプライン並列性を実現**
- ただし per-device conn は tg32 +1.76% だが pp128 -1.35% の実績

推奨: **オプション B** (per-device conn) をベースに、パイプライン用にチューニング。

#### 3.5 必要な変更

| ファイル | 関数 | 変更内容 |
|---------|------|---------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp:1446-1650` | `graph_compute` | パイプライン用非同期送信 + per-device 完了追跡 |
| `ggml/src/ggml-rdma/ggml-rdma.cpp:1576-1612` | FULL_GRAPH path | flush + graph を 1 コマンドに統合 |
| `ggml/src/ggml-rdma/ggml-rdma.cpp:762` | `send_rdma_cmd_async` | 完了コールバック/イベント追加 |
| `ggml/src/ggml-rdma/rdma-server.cpp` | サーバー | per-device 並列コマンド処理 |

**期間**: 5-7日

---

### Phase 4: 最適化・チューニング

**目的**: パイプライン実行の性能を最大化するためのチューニング。

#### 4.1 マイクロバッチサイズの自動選択

```cpp
// heuristic: バランスの取れたマイクロバッチサイズを選択
int optimal_microbatch_size(int pp_size, int n_devices) {
    // GPU 効率を維持するため、最低 16 トークン/マイクロバッチ
    int max_microbatches = pp_size / 16;
    // パイプラインの効果は n_devices まで
    int target = std::min(max_microbatches, n_devices);
    // n_copies (=4) の制約
    target = std::min(target, GGML_SCHED_MAX_COPIES);
    return pp_size / target;
}
```

#### 4.2 CUDA Stream とイベントの最適化

- Split 間のデータ転送を compute stream とは別の copy stream で実行
- `cudaStreamWaitEvent` で最小限の同期
- CUDA Graph との共存 (Phase 1 で CUDA Graph が有効化されている場合)

#### 4.3 プロファイリングとボトルネック特定

- パイプラインの各ステージの実行時間を計測
- バブル (idle) 時間の可視化
- 最適なマイクロバッチ数の実験的決定

#### 4.4 必要な変更

| ファイル | 変更内容 |
|---------|---------|
| `src/llama-context.cpp` | `pipeline_depth` パラメータ + 自動選択ロジック |
| `ggml/src/ggml-backend.cpp` | パイプラインプロファイリング出力 |
| `common/arg.cpp` | `-pd` (pipeline depth) CLI オプション |

**期間**: 3-5日

---

## 5. 各フェーズの変更ファイル・関数の全体図

```
Phase 0 (プロファイリング)
  └── 変更なし (計測のみ)

Phase 1 (Split オーバーラップ)
  ├── ggml/src/ggml-backend.cpp
  │   ├── ggml_backend_sched_compute_splits()  ← 主要変更
  │   ├── ggml_backend_sched_alloc_graph()     ← コピー管理
  │   └── struct ggml_backend_sched            ← パイプライン状態追加
  └── ggml/include/ggml-backend.h
      └── ggml_backend_sched_set_pipeline_depth()  ← API 追加

Phase 2 (マイクロバッチ + KV キャッシュ)
  ├── src/llama-context.cpp
  │   ├── decode()             ← ubatch ループ改修
  │   ├── process_ubatch()     ← 非同期版分離
  │   ├── graph_compute()      ← パイプラインディスパッチ
  │   └── 初期化              ← pipeline_depth 設定
  ├── src/llama-kv-cache.cpp
  │   └── スロット管理         ← マイクロバッチ対応
  └── src/llama-batch.cpp
      └── (変更なし、検証のみ)

Phase 3 (RDMA 統合)
  ├── ggml/src/ggml-rdma/ggml-rdma.cpp
  │   ├── graph_compute()        ← パイプライン非同期送信
  │   ├── FULL_GRAPH path        ← flush+graph 統合
  │   └── send_rdma_cmd_async()  ← 完了追跡追加
  └── ggml/src/ggml-rdma/rdma-server.cpp
      └── コマンド処理           ← per-device 並列化

Phase 4 (チューニング)
  ├── src/llama-context.cpp     ← auto pipeline depth
  ├── ggml/src/ggml-backend.cpp ← プロファイリング
  └── common/arg.cpp            ← CLI オプション
```

---

## 6. リスクと制約

### 6.1 高リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| KV キャッシュの整合性破壊 | 推論結果が不正 | Phase 2 で段階的に検証。レイヤー分割の特性を活用し、per-device 順序を保証 |
| RDMA 非同期コマンドの競合 | サーバークラッシュ | Phase 3 で per-device conn を使用し、コマンドの独立性を保証 |
| メモリ不足 (n_copies × バッファ) | OOM | パイプライン深度を動的に調整。フォールバック機構 (既存の `pipeline_parallel=false`) を活用 |

### 6.2 中リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| マイクロバッチの GPU 効率低下 | 改善率が理論値を大きく下回る | 最低 32 トークン/マイクロバッチを維持。Phase 0 でプロファイリング |
| パイプライン fill/drain オーバーヘッド | 短プロンプトで逆効果 | pp128 以下ではパイプライン無効化 |
| graph reuse の無効化 | 各マイクロバッチで graph rebuild | パイプライン内では同一構造のグラフを再利用可能 (ubatch サイズが同一) |

### 6.3 低リスク

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| TG への影響 | tg 性能劣化 | TG ではパイプライン無効 (n_tokens=1 ではパイプラインの意味がない) |
| upstream マージの困難さ | メンテナンスコスト | upstream を目指さない前提なので許容 |

---

## 7. 理論的性能見積もり

### 参考: 過去のプロファイリングデータ (Qwen3.5 35B-A3B, 6GPU)

6GPU (2C+4R) でのプロファイリングデータ (`pp_profiling_qwen35_optimization.md`):

- 逐次 pp128 合計: 339.38 ms (graph_compute のみ)
- flush wait (=GPU 計算): 321.64 ms
- 平均ステージ時間: ~53.6 ms (321.64 / 6)

### Qwen3.5 35B-A3B (MoE, 4GPU テスト構成)

4GPU (2C+2R) での見積もり。ステージ数が減るため、パイプラインの理論的高速化率は 6GPU より控えめになる。

pp128 を 4 micro-ubatch (各 32tok) に分割:

```
4GPU の場合 (N=4, M=4):
  - 各 micro-ubatch の stage 時間: T_stage (Phase 0 で計測)
  - パイプライン合計: (4+4-1) × T_stage = 7 × T_stage
  - 逐次合計: 4 × 4 × T_stage = 16 × T_stage
  - 理論的高速化: 16 / 7 = 2.29x
```

**注意**: 実際にはマイクロバッチサイズの縮小により GPU 効率が変化する。MoE モデルでは active パラメータが少ないため、小バッチでも GPU 効率があまり落ちない可能性がある。

現実的な見積もり (4GPU):

| 指標 | Pipeline M=4 (理論) | Pipeline M=4 (現実的) |
|------|:------------------:|:--------------------:|
| pp128 | 2.29x | **1.3-1.6x** |
| pp512 | 2.91x | **1.8-2.2x** |

> 具体的な t/s 値は Phase 0 の 4GPU プロファイリング後に確定する。

### Qwen3.5 27B (Dense, 4GPU テスト構成)

Dense モデルは MoE より計算密度が高いため、パイプラインの恩恵が大きい:

- 各ステージの計算時間が長い → fill/drain オーバーヘッドの相対的影響が小さい
- GPU 効率がマイクロバッチサイズの影響を受けやすい → トレードオフ

Phase 0 のプロファイリング結果次第で見積もりを更新する。

### GLM-4.7 (Dense, 11GPU)

| 指標 | 逐次 (現在) | Pipeline M=4 (理論) | Pipeline M=4 (現実的) |
|------|:-----------:|:------------------:|:--------------------:|
| pp128 (t/s) | ~24.3 | ~72.9 (3.0x) | **35-48** (1.4-2.0x) |

11GPU でのパイプライン効果は 4GPU より大きい (N が大きいため fill/drain のステージ数の比率が改善)。

---

## 8. 全体スケジュール

| Phase | 期間 | 前提条件 | 主な成果物 |
|-------|------|---------|-----------|
| **Phase 0** | 1日 | なし | プロファイリングレポート |
| **Phase 1** | 3-5日 | Phase 0 | CUDA-only split オーバーラップ |
| **Phase 2** | 5-7日 | Phase 1 | マイクロバッチ + KV キャッシュ対応 |
| **Phase 3** | 5-7日 | Phase 2 | RDMA バックエンド統合 |
| **Phase 4** | 3-5日 | Phase 3 | チューニング・最適化 |
| **合計** | **17-25日** | | |

### マイルストーン

1. **Phase 0 完了**: Qwen3.5 27B のプロファイリングデータ取得 → Pipeline の効果見積もり確定
2. **Phase 1 完了**: CUDA-only で pp128 の改善を確認 → パイプラインの基本動作検証
3. **Phase 2 完了**: マイクロバッチ分割が正しく動作 → KV キャッシュの整合性確認
4. **Phase 3 完了**: RDMA マルチノードでパイプライン動作 → 最終目標に向けた統合テスト
5. **Phase 4 完了**: チューニング完了 → 最終性能数値の確定
