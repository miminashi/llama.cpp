# ggml_backend_sched 逐次処理ボトルネック分析

- **実施日時**: 2026年2月19日 09:07
- **ワークツリー**: `.worktree/rdma-backend`

## 1. 前提・目的

### 背景

RDMA バックエンドの Generation 速度は RPC 比で約 -10% の劣位がある (GLM-4.7 IQ2_M, 11GPU 構成)。

| バックエンド | Prompt (t/s) | Generation (t/s) |
|-------------|:------------:|:----------------:|
| **RDMA (GDR budget 12GB)** | **6.4** | **6.8** |
| RPC (TCP) | 5.4 | 7.5 |

RDMA は Prompt で +19%、Generation で -10%。この非対称性の根本原因を体系的に分析する。

### 目的

2026-02-10〜02-18 にかけて実施した Generation 速度の最適化 6 件が、いずれも `ggml_backend_sched_compute_splits()` の逐次処理モデルにより e2e 効果が限定的または皆無であった。これらに共通するメカニズムを文書化し、今後の改善方向を示す。

### 参照レポート

| # | 日付 | レポート | 関連最適化 |
|---|------|---------|-----------|
| 1 | 02-10 | [rdma_async_compute_optimization](2026-02-10_214841_rdma_async_compute_optimization.md) | Async compute |
| 2 | 02-11 | [parallel_compute_dispatch](2026-02-11_214242_parallel_compute_dispatch.md) | Parallel compute dispatch |
| 3 | 02-13 | [rdma_synchronize_implementation](2026-02-13_053658_rdma_synchronize_implementation.md) | Two-phase loop |
| 4 | 02-14 | [deferred_copy_clean_ab_benchmark](2026-02-14_082220_deferred_copy_clean_ab_benchmark.md) | Deferred copy |
| 5 | 02-18 | [generation_optimization_deep_investigation](2026-02-18_070912_generation_optimization_deep_investigation.md) | 総合分析 |
| 6 | 02-18 | [unimplemented_optimizations_inventory](2026-02-18_120000_unimplemented_optimizations_inventory.md) | 未実装最適化の棚卸 |
| 7 | 02-18 | [shared_pd_per_device_connection](2026-02-18_130000_shared_pd_per_device_connection.md) | PD 共有 per-device 接続 |
| 8 | 02-18 | [server_push_rdma_write_imm](2026-02-18_200000_server_push_rdma_write_imm.md) | Server-Push |

---

## 2. ggml_backend_sched の逐次処理モデル

### 2.1 コード構造

`ggml_backend_sched_compute_splits()` (`ggml/src/ggml-backend.cpp:1443-1627`) はスケジューラの中核ループであり、計算グラフを分割 (split) 単位で逐次処理する。

```
for split_id in 0..n_splits:
    ① input_copy: 前の split の出力 → 今の split のバックエンドへコピー
    ② graph_compute_async: 今の split のグラフを計算
    ③ event_record: 完了イベントを記録
```

#### input_copy 内の暗黙的同期

ステップ①の `input_copy` では、前の split の計算結果を次の split のバックエンドにコピーするために、**前の split の完了を待つ必要がある**。

```cpp
// L1472-1476: イベントがない場合は全面同期
if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
    ggml_backend_event_wait(split_backend, sched->events[...]);
} else {
    ggml_backend_synchronize(split_backend);  // ← 暗黙的バリア
}
```

```cpp
// L1566-1571: async copy 不可の場合も同期
ggml_backend_synchronize(input_backend);  // ← input 側の同期
```

RDMA バックエンドでは `event_record` / `event_wait` が未実装 (`NULL`) のため、常に `ggml_backend_synchronize()` フォールバックが呼ばれる。現在の RDMA `synchronize` 実装は no-op (`ggml-rdma.cpp:1393-1396`) だが、これは RDMA 操作が同期的であるため問題にならない — 真の同期は `get_tensor` の Send/Recv 応答待ちで暗黙的に行われる。

### 2.2 Layer Split のデータ依存チェーン

Layer split では各 split の出力が次の split の入力となる直列依存チェーンを形成する:

```
Split 0 (CUDA0) → Split 1 (CUDA1) → ... → Split 6 (CUDA6) → Split 7 (RDMA0) → ... → Split 10 (RDMA3)
```

Split N+1 の `input_copy` は Split N の計算結果に依存するため、**物理的に並列化不可能**。これはスケジューラの制限ではなく、layer split の本質的な性質である。

### 2.3 Generation フェーズの 1 トークンタイムライン

Generation フェーズ (tg) では 1 トークンあたり 1 回のフォワードパスが全デバイスを逐次に通過する。GLM-4.7 IQ2_M (7C+4R) の実測タイムラインは以下の通り:

```
時間 (ms)  0                           80                         133
           ├───── CUDA0-6 計算 ─────────┤─── RDMA0-3 逐次処理 ──────┤
           │      ~80ms (ローカル)       │    ~53ms (リモート)        │
           │                            │                           │
           │ [ローカル GPU は               │ D0: send→compute→recv    │
           │  ggml_backend_sched が        │     ~13ms                │
           │  内部的に高速処理]              │ D1: send→compute→recv    │
           │                            │     ~12ms                │
           │                            │ D2: send→compute→recv    │
           │                            │     ~13ms                │
           │                            │ D3: send→compute→recv    │
           │                            │     ~12ms                │
           │                            │ (+ IB 往復 ~0.5ms/device) │
```

1 トークンの合計: ~133ms → ~7.5 t/s (理論値)。実測 6.8 t/s との差はオーバーヘッドによる。

RDMA の 4 デバイス分の処理 (~53ms) はこのタイムラインの 40% を占める。ここが RPC との差が生まれる領域である。

---

## 3. RPC とのアーキテクチャ比較

### 3.1 プロトコル設計の違い

| 特性 | RPC | RDMA |
|------|-----|------|
| 接続モデル | **1 GPU = 1 ソケット** | **N GPU = 1 QP** (共有接続) |
| graph_compute | Fire-and-forget (応答なし) | Async: fire-and-forget / Sync: 応答待ち |
| get_tensor | Send/Recv (暗黙同期) | RDMA Read (GDR) or Send/Recv |
| synchronize | **No-op** | No-op |
| event_record/wait | **NULL** | **NULL** |

### 3.2 RPC の暗黙的パイプライン

RPC (`ggml-rpc.cpp:868-888`) の graph_compute は `send_rpc_cmd()` の引数なしオーバーロード (応答を読まない) を使用し、送信後即座にリターンする:

```cpp
// RPC: graph_compute は応答を読まない
static enum ggml_status ggml_backend_rpc_graph_compute(...) {
    ...
    bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
    // ↑ send only, no recv — fire-and-forget
    return GGML_STATUS_SUCCESS;
}
```

`ggml_backend_sched` が Split N → Split N+1 を逐次処理する際、RPC では:

1. Split N (RDMA device D0) の graph_compute → TCP 送信のみで即座にリターン
2. Split N の input_copy で `ggml_backend_synchronize(split_backend)` → **no-op**
3. Split N+1 (RDMA device D1) の graph_compute → TCP 送信のみで即座にリターン
4. ...全デバイスの graph_compute が「見かけ上」瞬時に完了
5. 最終 split 後の get_tensor で **初めて応答を待つ** (TCP recv)

TCP のカーネルバッファに graph_compute コマンドが複数蓄積され、サーバー側では FIFO で順次処理される。**スケジューラは逐次だが、TCP バッファリングにより結果的にパイプライン効果が発生する**。

### 3.3 RDMA でパイプライン効果が得られない理由

RDMA (共有接続モード) では全デバイスが同一 QP を共有する。Async compute (fire-and-forget) を実装済みだが:

1. graph_compute (async) → D0 の計算コマンド送信、即座にリターン
2. **get_tensor (D0)** → `compute_pending_` が true なので Send/Recv フォールバック。サーバーの FIFO コマンドキューで D0 計算完了を**暗黙的に待つ**
3. graph_compute (async) → D1 の計算コマンド送信
4. **get_tensor (D1)** → 同様に D1 計算完了を待つ

問題: `ggml_backend_sched` はデバイスごとに `graph_compute` → `get_tensor` をペアで呼ぶ。RDMA では get_tensor の応答が compute 完了後にしか返らないため、D1 のコマンドは D0 の get_tensor 完了まで送信されない。**4 デバイスが完全に逐次化される**。

RPC では各デバイスに独立ソケットがあり、D0 の get_tensor 応答を待っている間も D1 のソケットは独立に処理可能。ただし、`ggml_backend_sched` の逐次処理のため、RPC でもこの「真の」並列性は活用されておらず、TCP バッファリングによる暗黙的パイプラインのみが寄与している。

### 3.4 RPC 優位の定量的見積もり

RDMA デバイスあたりの処理内訳 (実測値):
- GPU 計算: ~12ms
- IB 往復 (コマンド送受信): ~0.5ms
- get_tensor データ転送: ~0.3ms

RPC では graph_compute の IB 往復が不要 (TCP バッファリング) なため:
- 4 デバイス × 0.5ms = **~2ms の削減**
- 133ms 中の 2ms ≈ **1.5%** の改善

しかし実測差は ~10% (6.8 vs 7.5 t/s)。残りの差は RDMA の `compute_pending_` による Send/Recv フォールバック (RDMA Read より遅い) と、共有 QP のシリアライゼーションに起因すると推定される。

---

## 4. スケジューラ制限を受けた 6 件の最適化

本セクションが本レポートの核心である。各最適化について、概要 → 実測結果 → スケジューラがどう阻んだか → 教訓を述べる。

### 概要表

| # | 最適化 | マイクロ効果 | e2e 効果 | 制限メカニズム |
|---|--------|:----------:|:-------:|-------------|
| 1 | Async compute | graph_compute 24.5x 高速化 | 0% | get_tensor で暗黙同期、時間が移動しただけ |
| 2 | Parallel compute dispatch | server dispatch 13→2ms | 0% | クライアント逐次送信で D1-D3 が未送信 |
| 3 | Server-Push (RDMA Write IMM) | get_tensor RT 削減 | +0.74% | 0.5ms/device の削減、逐次 53ms 中の ~2ms |
| 4 | Two-phase loop (synchronize) | N/A | 0% (p=0.69-1.00) | Layer split データ依存で Phase 1 内同期 |
| 5 | PD 共有 per-device connection | MTT 問題解消 | **-2.96%** | 並列性恩恵なし + deferred copy 喪失 |
| 6 | Backend interface 拡張 (Plan B) | 未実施 | N/A | upstream 障壁 + データ依存は API で解消不可 |

---

### 4.1 Async Compute (fire-and-forget graph_compute)

**実施日**: 2026-02-10 | **レポート**: [rdma_async_compute_optimization](2026-02-10_214841_rdma_async_compute_optimization.md)

#### 概要

graph_compute でサーバー応答を待たず即座にリターンする fire-and-forget 方式。サーバーは ASYNC コマンド受信後、応答を返さずに計算を開始する。

#### 実測結果

| 指標 | Sync (従来) | Async (新) |
|------|:----------:|:---------:|
| graph_compute 平均 | 17.2 ms | **0.7 ms** |
| graph_compute 合計 (200回) | 3,437 ms | 141 ms |
| get_tensor 合計 | 364 ms | **3,528 ms** |
| **Generation (t/s)** | **6.6** | **6.6** |

graph_compute は **24.5 倍高速化**したが、Generation 速度は変化なし。

#### スケジューラがどう阻んだか

```
Sync:   graph_compute(17ms) → get_tensor(2ms)   = 19ms/device
Async:  graph_compute(0.7ms) → get_tensor(18ms)  = 19ms/device  ← 合計は同じ
```

Async ではクライアントが graph_compute 後に即座に get_tensor を呼ぶ。get_tensor は `compute_pending_=true` により RDMA Read を使えず、Send/Recv フォールバックでサーバーのコマンドキューを通る。サーバーは FIFO で処理するため、GET_TENSOR コマンドは先行する ASYNC 計算の完了後に処理される。

結果: graph_compute で節約した時間が get_tensor の待ち時間に**そのまま移動**しただけ。

#### 教訓

FIFO コマンドキューを共有する限り、個別コマンドの高速化は全体スループットに寄与しない。改善には「全デバイスの graph_compute を先に送り、その後で全デバイスの get_tensor をまとめて行う」パイプライン化が必要だが、これは `ggml_backend_sched` の逐次モデルでは実現できない。

---

### 4.2 Parallel Compute Dispatch (サーバー側並列ディスパッチ)

**実施日**: 2026-02-11 | **レポート**: [parallel_compute_dispatch](2026-02-11_214242_parallel_compute_dispatch.md)

#### 概要

サーバー側にデバイスごとのワーカースレッドを配置し、ASYNC コマンドを受信即座にワーカーに振り分けて並列計算を実現する。

#### 実測結果

| Config | pp128 (t/s) | tg32 (t/s) |
|--------|:---:|:---:|
| PARALLEL_COMPUTE=0 | 23.25 | 7.72 |
| PARALLEL_COMPUTE=1 | 23.19 | 7.73 |

e2e 改善なし。サーバーログでは各 ASYNC ディスパッチは 1.9-2.9ms で完了していた (従来の 13ms から大幅短縮)。

#### スケジューラがどう阻んだか

サーバー側は正しく並列化されたが、クライアントが以下の順序でコマンドを送る:

```
ASYNC D0 → GET_TENSOR D0 (12ms 待ち) → ASYNC D1 → GET_TENSOR D1 (12ms 待ち) → ...
```

D0 の GET_TENSOR 応答を待っている間、D1 の ASYNC コマンドはまだ送信されていない。サーバーのワーカースレッド D1-D3 は**アイドル状態のまま**。

#### 教訓

サーバー側の並列化はクライアント側の逐次送信ボトルネックを解消できない。「送信を先に全部行い、受信を後で全部行う」Two-phase パターンが必要だが、これは 4.4 で検証・否定された。

---

### 4.3 Server-Push (RDMA Write with IMM)

**実施日**: 2026-02-18 | **レポート**: [server_push_rdma_write_imm](2026-02-18_200000_server_push_rdma_write_imm.md)

#### 概要

graph_compute 完了後、サーバーが自発的に RDMA Write with Immediate Data で計算結果をクライアントにプッシュする。クライアントは CQ から IMM 通知をポーリングするだけで、GET_TENSOR の Send/Recv 往復を省略できる。

#### 実測結果 (GLM-4.7 IQ2_M, 7C+4R, n=15 paired)

| 指標 | Push ON | Push OFF | 差分 |
|------|:-------:|:--------:|:----:|
| tg32 (t/s) | 7.763 ± 0.012 | 7.705 ± 0.007 | **+0.057 (+0.74%)** |
| p 値 | — | — | 1.28 × 10⁻¹⁰ |
| Cohen's d | — | — | 4.30 |

統計的に有意 (p < 0.05 かつ > 0.5%) だが、実用的には小さい。

#### スケジューラがどう阻んだか

Server-Push はデバイスあたり約 0.5ms の GET_TENSOR 往復を削減する。しかし:

- 4 デバイス × 0.5ms = **~2ms/token** の削減
- 1 トークンの合計処理時間 ~133ms 中の 2ms = **~1.5%**
- 実測 +0.74% はこの理論値の半分程度

根本的に、`ggml_backend_sched` がデバイスを逐次処理する限り、per-device の通信レイテンシ削減は加算的にしか効かない。GPU 計算時間 (~12ms/device) がドミナントなため、0.5ms の削減はごく小さい割合にとどまる。

#### 教訓

Server-Push は正しい方向性だが、逐次スケジューリング下では効果が加算的で、乗算的 (パイプライン) 効果は得られない。全デバイスの計算結果を一括プッシュし、クライアントが一括受信するパイプラインと組み合わせて初めて真価を発揮する。

---

### 4.4 Two-Phase Loop (synchronize 実装)

**実施日**: 2026-02-13 | **レポート**: [rdma_synchronize_implementation](2026-02-13_053658_rdma_synchronize_implementation.md)

#### 概要

`ggml_backend_sched_compute_splits()` のループを 2 フェーズに分割する試み:
- **Phase 1**: 全 split の input_copy + graph_compute_async (計算を全て投入)
- **Phase 2**: 全 split の event_wait + get_tensor (結果を全て回収)

これにより全デバイスの計算をパイプライン化する狙い。

#### 実測結果

| 構成 | 条件 | pp128 (t/s) | tg32 (t/s) |
|------|------|:---:|:---:|
| GLM-4.7, 7C+4R | 2-phase | — | — |
| GLM-4.7, 7C+4R | default | — | — |
| p 値 | — | p=0.69 | p=1.00 |

完全に効果なし。

#### スケジューラがどう阻んだか

Layer split ではデータ依存が直列チェーンを形成する:

```
Split 0 出力 → Split 1 入力 → Split 1 出力 → Split 2 入力 → ...
```

Phase 1 で Split 1 の input_copy を行おうとすると、Split 0 の計算結果が必要。`ggml_backend_synchronize(input_backend)` が呼ばれ、Split 0 の完了を待つ。**Phase 1 内で逐次化が復活し、元のループと同じ動作に退化する**。

```
Phase 1:
  Split 0: input_copy → graph_compute_async
  Split 1: input_copy → synchronize(Split 0) ← ここで待つ → graph_compute_async
  Split 2: input_copy → synchronize(Split 1) ← ここで待つ → graph_compute_async
  ...
  ↑ 結局逐次処理と同じ
```

#### 教訓

Layer split のデータ依存チェーンは「投入を先に全部行う」アプローチでは解消できない。Two-phase が有効なのは row split (各デバイスが独立部分を計算) のケースのみだが、P100 には NVLink がなく row split は実用的でない。

---

### 4.5 PD 共有 Per-device Connection

**実施日**: 2026-02-18 | **レポート**: [shared_pd_per_device_connection](2026-02-18_130000_shared_pd_per_device_connection.md)

#### 概要

従来 ConnectX-4 の MTT キャッシュ制限で不可能だった per-device 接続を、PD (Protection Domain) を共有することで MR の重複登録を排除し実現する。各デバイスが独立 QP を持つことでコマンド送受信の並列化を狙う。

#### 実測結果 (GLM-4.7 IQ2_M, 7C+4R, n=15 paired)

| 指標 | Per-device | Shared (default) | 差分 |
|------|:----------:|:----------------:|:----:|
| tg32 (t/s) | 7.200 ± 0.000 | 7.420 ± 0.041 | **-0.220 (-2.96%)** |
| p 値 | — | — | 7.30 × 10⁻¹² |
| Cohen's d | — | — | -5.31 |

**有意な性能悪化**。Per-device は共有接続より 3% 遅い。

#### スケジューラがどう阻んだか

独立 QP を持っても、`ggml_backend_sched` がデバイスを逐次処理する限り並列性は活用されない:

1. **並列性恩恵なし**: graph_compute(D0) → get_tensor(D0) → graph_compute(D1) → ... の逐次パターンは変わらない
2. **Deferred copy 喪失 (-1.45%)**: 異なる接続間では `cpy_tensor` が `src_ctx->conn != dst_ctx->conn` で false を返し、deferred copy が無効化される
3. **リソースオーバーヘッド**: 4× (QP + CQ + recv_buffer + send_buffer) の追加メモリ/RNIC リソース
4. **スレッド競合**: サーバー側 4 スレッドが GPU リソースと CUDA コンテキストを競合

#### 教訓

Per-device 接続はクライアント側のパイプライン化とセットでなければ無意味。さらに、deferred copy との排他関係が予想外の性能低下を招いた。

---

### 4.6 Backend Interface 拡張 (Plan B: graph_compute_begin/finish)

**実施日**: 2026-02-18 (分析のみ) | **レポート**: [unimplemented_optimizations_inventory](2026-02-18_120000_unimplemented_optimizations_inventory.md)

#### 概要

`ggml_backend_i` に `graph_compute_begin()` / `graph_compute_finish()` を追加し、スケジューラが全デバイスの計算を先に投入してから結果を回収する Two-phase パターンを可能にする API 提案。

#### 未実施の理由

1. **upstream 障壁**: `ggml_backend_i` は llama.cpp の共有インフラであり、RDMA 固有の API 変更は upstream に受け入れられにくい
2. **データ依存の本質**: API を変えても layer split のデータ依存チェーンは解消されない (4.4 Two-phase loop と同じ問題)
3. **限定的恩恵**: Row split でのみ有効だが、P100 (NVLink なし) では row split が実用的でない

#### 教訓

問題の本質は API の不足ではなく、layer split のデータ依存構造にある。スケジューラの API を変更しても、入力データが前の split の出力である限り、逐次処理は避けられない。

---

## 5. 根本的制約の分析

### 5.1 Layer Split: 並列化不可能な直列チェーン

Transformer モデルの layer split では、レイヤー N の出力がレイヤー N+1 の入力となる。これは計算グラフ上の真のデータ依存であり、いかなるスケジューリング変更でも解消できない:

```
Layer 0-8 (CUDA0)
    ↓ output tensor
Layer 9-16 (CUDA1)
    ↓ output tensor
...
Layer 49-56 (RDMA0)
    ↓ output tensor
Layer 57-63 (RDMA3)
    ↓ final output
```

唯一の例外は**同一サーバー内の連続スプリット** (RDMA0 → RDMA1 → RDMA2 → RDMA3) であり、これらの間のデータ転送はネットワーク往復なしに実行可能 (deferred copy がこれを実現)。

### 5.2 Row Split が使えない理由

Row split (テンソル並列) では各デバイスがテンソルの異なる次元を並列計算するため、デバイス間のデータ依存はない。しかし:

- **AllReduce が必要**: 各 Attention/FFN レイヤー後に全デバイスの部分結果を集約
- **P100 に NVLink なし**: AllReduce は PCIe 経由、バンド幅 ~12 GB/s (双方向)
- **シングルノード検証済み**: Row split は layer split より遅い (CLAUDE.md に記載)
- **InfiniBand AllReduce**: 理論的には 100 Gbps (~12.5 GB/s) だが、RDMA バックエンドに未実装

### 5.3 スケジューラ変更の限界

| 変更案 | 効果 | 阻害要因 |
|--------|:----:|---------|
| Two-phase loop | なし | Layer split データ依存で Phase 1 内同期 |
| graph_compute_begin/finish API | なし | 同上 |
| Per-device connection | 悪化 | 並列性未活用 + deferred copy 喪失 |
| イベント実装 | 微小 | 同期粒度の改善のみ、データ依存は残存 |

**結論: `ggml_backend_sched` の変更だけでは Generation 速度の RPC 比劣位を解消できない。**

---

## 6. 改善提案

### 提案 A: Server-Side Pipeline (サーバー内パイプライン)

#### 概要

連続する同一サーバー向けスプリット (RDMA0 → RDMA1 → RDMA2 → RDMA3) を**単一のバッチコマンド**として送信し、サーバーが内部で compute → internal_copy → compute チェーンを自律的に実行する。

#### メカニズム

現状:
```
Client: ASYNC(D0) → GET(D0) → ASYNC(D1) → GET(D1) → ASYNC(D2) → GET(D2) → ASYNC(D3) → GET(D3)
         0.7ms      12ms        0.7ms      12ms        0.7ms      12ms        0.7ms      12ms
合計: ~53ms (IB 往復 8 回)
```

改善案:
```
Client: BATCH_COMPUTE(D0,D1,D2,D3) → BATCH_GET(D0,D1,D2,D3)
         0.7ms                         ~48ms + 転送
合計: ~50ms (IB 往復 2 回)
```

サーバー側では:
1. D0 compute (12ms)
2. D0→D1 internal cudaMemcpyPeer (ネットワーク不要)
3. D1 compute (12ms)
4. D1→D2 internal cudaMemcpyPeer
5. ... (D2, D3 同様)
6. 全結果を一括返送

#### 期待効果

- IB 往復削減: (4-1) × 2 × 0.5ms = **~3ms**
- get_tensor のバッチ化による応答待ち最適化: **~1ms**
- 合計: **~4ms/token** (133ms 中 ~3%)

#### 実装複雑度: 高

- 新コマンド `RDMA_CMD_BATCH_COMPUTE` の設計
- サーバー側の内部 split 間コピーロジック
- Deferred copy が部分的にこれを実現済み (compute は未包含)

---

### 提案 B: Per-device Connection + Cross-connection Deferred Copy

#### 概要

4.5 で per-device 接続が -3% に悪化した主因は deferred copy の喪失 (-1.45%)。同一サーバーの複数接続を論理グループとして管理し、cross-connection でも deferred copy を許可することでこの問題を解消する。

#### メカニズム

```cpp
// 現状: cpy_tensor は同一接続でのみ deferred copy
if (src_ctx->conn != dst_ctx->conn) return false;  // 異接続 → 不可

// 改善: 同一サーバー (同一 PD) の接続ならば deferred copy を許可
if (src_ctx->conn->pd_ != dst_ctx->conn->pd_) return false;  // 異 PD → 不可
// 同一 PD → deferred copy 可能
```

Per-device 接続の独立 QP + deferred copy の恩恵を両立:
- 独立 QP: Server-Push で各デバイスの結果を並列プッシュ可能
- Deferred copy: サーバー内部コピーの +1.45% を維持

#### 期待効果

- Deferred copy 回復: **+1.45%**
- Server-Push との相乗: 独立 QP で複数デバイスが並列プッシュ → **+1-3%** (推定)
- 合計: **+3-5%**

#### 実装複雑度: 中-高

- `deferred_copy_list` のキーを `conn` から `pd` に変更
- サーバー側で cross-connection deferred copy の実行パスを追加
- Server-Push との統合テスト

---

### 提案 C: RDMA イベント実装

#### 概要

`ggml_backend_sched` の既存イベント機構 (`event_record` / `event_wait`, L1464-1476, L1619-1622) を活用する。RDMA バックエンドではこれらが未実装 (`NULL`) のため、常に `ggml_backend_synchronize()` フォールバックが呼ばれている。

#### メカニズム

イベント実装により:
- `event_record`: 計算完了のマーカーを記録 (軽量操作)
- `event_wait`: 別バックエンドで前のバックエンドの計算完了を待つ (必要な場合のみブロック)

これにより input_copy の同期がより細粒度になり、不要な同期を排除できる可能性がある。

#### 期待効果: ~1%

- 現状の `ggml_backend_synchronize()` は no-op なため、実質的な同期オーバーヘッドは get_tensor の Send/Recv に包含されている
- イベント実装の直接的効果は限定的だが、将来的な最適化 (Two-phase loop の再検討等) の基盤となる

#### 実装複雑度: 中

- `rdma_event` クラスの設計 (タイムスタンプ or カウンターベース)
- `event_record` / `event_wait` の RDMA 実装
- `ggml_backend_sched` との統合テスト

---

### 提案の優先度

| 優先度 | 提案 | 期待効果 | 複雑度 | 前提条件 |
|:------:|------|:--------:|:------:|---------|
| 1 | **A: Server-Side Pipeline** | +3% | 高 | なし |
| 2 | **B: Cross-conn Deferred Copy** | +3-5% | 中-高 | Per-device conn (ConnectX-4 PD 共有で可能) |
| 3 | **C: RDMA イベント** | +1% | 中 | なし |

提案 A と B は独立に実装可能。B は A と組み合わせるとさらに効果的 (Server-Side Pipeline + per-device push の両方の恩恵)。C は A/B の基盤として有用。

---

## 7. 結論

### 主要知見

1. **Layer split のデータ依存チェーンが根本的ボトルネック**: Split N+1 は Split N の出力に依存するため、デバイス間の並列化は物理的に不可能。6 件の最適化はいずれもこの壁に阻まれた。

2. **RPC の暗黙的パイプライン**: RPC は「1 GPU = 1 ソケット」のアーキテクチャにより、TCP バッファリングを通じた暗黙的パイプライン効果を持つ。RDMA の共有 QP モデルではこの効果が得られない。

3. **マイクロ最適化の罠**: graph_compute 24.5x 高速化 (async) やサーバー並列ディスパッチなど、個別操作の劇的な改善が e2e 性能にまったく寄与しないケースが複数発生した。FIFO キュー + 逐次スケジューリング下では、ボトルネックの移動が起きるだけ。

4. **副作用による性能悪化**: Per-device 接続のように、一見有望な構造変更が deferred copy の喪失という副作用で -3% の悪化を招くケースがある。相互依存する最適化の組み合わせ評価が重要。

5. **RDMA の真の優位領域**: Prompt 処理では RDMA Write (ゼロコピー、9.8 GB/s) により RPC 比 +19%。Generation の劣位 (-10%) にもかかわらず、Prompt-heavy ワークロードでは RDMA が有利。

### 推奨アクション

| 優先度 | アクション | 理由 |
|:------:|-----------|------|
| **短期** | 現状維持 (deferred copy + server-push ON) | +2.2% (1.45% + 0.74%) の着実な改善 |
| **中期** | 提案 B: Cross-conn deferred copy | Per-device の並列性 + deferred copy の両立 |
| **中期** | 提案 A: Server-Side Pipeline | IB 往復削減で ~3% 改善 |
| **長期** | ConnectX-6+ への移行 | MTT 制限解消、per-device 接続のフル活用 |
| **対象外** | `ggml_backend_sched` の変更 | Layer split のデータ依存は API 変更で解消不可 |
