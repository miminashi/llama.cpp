# RDMA synchronize 実装 + Two-phase loop

- **実施日時**: 2026年2月13日 05:36
- **ワークツリー**: `.worktree/rdma-synchronize`
- **参照レポート**: [report/2026-02-13_044038_rdma_backend_comprehensive_review.md](report/2026-02-13_044038_rdma_backend_comprehensive_review.md)

## 前提・目的

### 背景

前回のレビューレポートで提案された案A「Two-phase loop」の実装可能性を調査し、実装を行った。

調査の結果、**スケジューラ側の Two-phase loop はレイヤー分割のデータ依存により効果がない**ことが判明。
代わりに、**RDMA バックエンドの `synchronize` を正しく実装する**ことで、`get_tensor` の RDMA Read 高速パスを有効化するという実質的な改善を実装した。

### Two-phase loop が効かない理由

スケジューラ (`ggml_backend_sched_compute_splits`, ggml-backend.cpp) は split を逐次処理する:

1. Split N の入力コピー (`input_copy`) → Split N の `graph_compute`
2. Split N+1 の入力コピー → Split N+1 の `graph_compute`
3. ...

レイヤー分割では、Split N+1 の入力は Split N の出力（活性化テンソル）に依存する。
Phase 1 で全 split を dispatch しようとしても、`input_copy` 時に `ggml_backend_synchronize(input_backend)` が呼ばれ、Split N の計算完了を待つ。結果として Phase 1 が逐次実行に退化し、現状と同じ動作になる。

### synchronize 修正が有効な理由

- **現在**: `synchronize` が NO-OP → `compute_pending_` がクリアされない → `get_tensor` が常に遅い Send/Recv パスを使用
- **修正後**: `synchronize` が `compute_pending_` をドレイン → `get_tensor` が高速 RDMA Read パスを使用可能（GDR バッファ時）
- スケジューラは既に `ggml_backend_synchronize(input_backend)` を呼んでいる — 修正すれば自然に効く

## 修正内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | `RDMA_CMD_SYNC` 追加, `synchronize` 実装, サーバー側ハンドラ追加 |
| `ggml/include/ggml-rdma.h` | `RDMA_PROTO_PATCH_VERSION` 1→2 |
| `ggml/src/ggml-backend.cpp` | Two-phase loop 追加 (`GGML_SCHED_TWO_PHASE=1` で有効化) |

### 1. RDMA_CMD_SYNC コマンド (ggml-rdma.cpp)

```cpp
RDMA_CMD_GRAPH_COMPUTE_ASYNC,            // Async: no response sent
RDMA_CMD_SYNC,                           // Drain pending async commands; server responds with empty ack
RDMA_CMD_COUNT,
```

### 2. synchronize 実装 (ggml-rdma.cpp)

```cpp
static void ggml_backend_rdma_synchronize(ggml_backend_t backend) {
    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)backend->context;
    if (ctx->conn && ctx->conn->compute_pending_.load(std::memory_order_acquire)) {
        send_rdma_cmd(ctx->conn.get(), RDMA_CMD_SYNC, nullptr, 0, nullptr);
    }
}
```

**動作**: `compute_pending_` が true の場合のみ SYNC コマンドを送信。サーバーは FIFO コマンドキューなので、SYNC への応答が返った時点で全 async コマンドが完了済み。`send_rdma_cmd` の recv パス (L706) で `compute_pending_` が自動的に false にクリアされる。

### 3. サーバー側ハンドラ (ggml-rdma.cpp)

```cpp
case RDMA_CMD_SYNC: {
    send_rsp_empty();
    break;
}
```

コマンドキューが FIFO のため、SYNC の前に投入された全 async コマンドはこの時点で完了している。空の応答を返すだけでよい。

### 4. Two-phase loop (ggml-backend.cpp)

`ggml_backend_sched_compute_splits_two_phase()` 関数を追加:
- Phase 1: 各 split の input_copy + graph_compute_async（元コードと同じ）
- Phase 2: 各 split の synchronize
- `GGML_SCHED_TWO_PHASE=1` 環境変数で有効化（デフォルト無効）
- `callback_eval` 使用時は従来パスにフォールバック

レイヤー分割では Phase 1 がデータ依存で逐次に退化するため、実質的な効果はない。
将来の row split 実装時に独立 split の並列 dispatch が可能になる。

## 期待される効果

| シナリオ | 効果 |
|---------|------|
| レイヤー分割 + GDR 有効 | **小改善**: synchronize → RDMA Read パス有効化 (~0.1-0.3ms/split 改善) |
| レイヤー分割 + GDR 無効 | **効果なし**: RDMA Read は GDR MR のみ |
| Row split (将来) | **大改善**: 独立 split の並列 dispatch が可能 |
| Two-phase env var | **効果なし〜同等**: レイヤー分割ではデータ依存で逐次に退化 |

### synchronize の効果メカニズム

スケジューラの split 処理フロー:

```
Split 0 (CUDA0):
  graph_compute_async → [compute_pending_ = true]

Split 1 (RDMA0):
  synchronize(CUDA0) → [compute_pending_ = false] (SYNC sent, response received)
  input_copy: get_tensor(CUDA0 output) → RDMA Read パス使用可能 (GDR時)
  graph_compute_async → [compute_pending_ = true]

Split 2 (CUDA1):
  synchronize(RDMA0) → [compute_pending_ = false] (SYNC sent)
  input_copy: get_tensor(RDMA0 output) → RDMA Read パス使用可能 (GDR時)
  ...
```

以前は `synchronize` が NO-OP だったため、`compute_pending_` が true のまま残り、`get_tensor` が Send/Recv フォールバックを使用していた。

## 再現方法

### ビルド

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-synchronize
bash scripts/rdma-build.sh local
```

### 動作確認 (後日)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

プロファイリング出力で `get_tensor` が RDMA Read パスを使用しているか確認する。

### Two-phase loop テスト (後日)

```bash
GGML_SCHED_TWO_PHASE=1 GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## ビルド結果

- ビルド: **成功** (rdma-synchronize ワークツリー)
- 実行テスト: 後日実施予定 (GPU 使用中)
