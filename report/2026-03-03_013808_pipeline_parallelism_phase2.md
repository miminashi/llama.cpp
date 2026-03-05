# Pipeline Parallelism Phase 2: Async RDMA graph_compute + Functional Events

- **実施日時**: 2026年3月3日 01:38
- **ワークツリー**: `.worktree/pipeline-parallelism`
- **ブランチ**: `feature/pipeline-parallelism`
- **コミット**: `fd2f7d962`

## 前提・目的

### 背景

Phase 1 (`8243337ba`) で graph reuse 無効化、Phase 1.5 (`cdf988202`) で RDMA event caps (no-op) を実装し、`pipeline_parallel = true` のゲート条件を通過済み。しかし RDMA の `graph_compute` がブロッキングのため、スケジューラが ubatch 間のパイプラインオーバーラップを実現できていなかった。

### 目的

RDMA の `event_record` / `event_wait` / `event_synchronize` / `synchronize` を機能実装し、ubatch N+1 の CUDA splits が ubatch N の RDMA compute と並行実行できるようにする。

### 前提条件

- Phase 1 (graph reuse 無効化) 適用済み
- Phase 1.5 (event caps: event_new/event_free 実装) 適用済み
- `GGML_RDMA_ASYNC_COMPUTE=1` (デフォルト) で `graph_compute` が非ブロッキング

### 参照

- [Phase 1 レポート](2026-03-02_233154_pipeline_parallelism_phase1.md)
- [Phase 1.5 レポート](2026-03-03_004144_pipeline_parallelism_phase1.5_event_caps.md)

## 実装内容

変更ファイル: `ggml/src/ggml-rdma/ggml-rdma.cpp` (1ファイル, +19/-6行)

### 1. `drain_pending_compute()` ヘルパー関数

```cpp
static void drain_pending_compute(rdma_connection * conn) {
    if (!conn->compute_pending_.load(std::memory_order_acquire)) {
        return;
    }
    send_rdma_cmd(conn, RDMA_CMD_FLUSH_ALL_STAGING, nullptr, 0, nullptr);
}
```

`FLUSH_ALL_STAGING` with 0 entries をバリアとして使用。サーバーは FIFO でコマンドを処理するため、このコマンドの応答が返った時点で先行する async compute を含む全コマンドが完了している。

### 2. `synchronize()` — no-op → drain

```cpp
static void ggml_backend_rdma_synchronize(ggml_backend_t backend) {
    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)backend->context;
    drain_pending_compute(ctx->conn.get());
}
```

### 3. `event_record()` — バックエンドコンテキストを保存

```cpp
static void ggml_backend_rdma_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    event->context = (void *)backend->context;
}
```

### 4. `event_wait()` — pending compute をドレイン

```cpp
static void ggml_backend_rdma_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    if (event->context == nullptr) return;
    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)event->context;
    drain_pending_compute(ctx->conn.get());
}
```

### 5. `event_synchronize()` — 同上

```cpp
static void ggml_backend_rdma_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    if (event->context == nullptr) return;
    ggml_backend_rdma_context * ctx = (ggml_backend_rdma_context *)event->context;
    drain_pending_compute(ctx->conn.get());
}
```

## 動作原理

`compute_splits` はイベントで ubatch 間の同期を管理する:

```
現在 (graph_compute がブロック):
  ub0: [C0][C1][R0-block][R1-block]
  ub1:                              [C0][C1][R0-block][R1-block]

Phase 2 (async graph_compute + functional events):
  ub0: [C0][C1][R0-send][R1-send]
  ub1: [C0][C1][R0-drain+send][R1-drain+send]  ← CUDA splits が RDMA compute と並行
```

- **RDMA_ASYNC_COMPUTE=1** (デフォルト): `graph_compute` は `send_rdma_cmd_async()` で即座にリターン。`event_record` がコンテキストを保存。次の ubatch で `event_wait` → `drain_pending_compute` で完了を確認。
- **RDMA_ASYNC_COMPUTE=0**: `graph_compute` がブロッキング。`compute_pending_` は常に false。`drain_pending_compute` は即リターン。デグレなし。
- **共有接続** (デフォルト): R0/R1 が同一接続を共有。R0 の drain は R1 の pending も暗黙的にドレイン (FIFO)。

## 再現方法

### ビルド・デプロイ

```bash
bash .worktree/pipeline-parallelism/scripts/rdma-build.sh local
bash .worktree/pipeline-parallelism/scripts/rdma-deploy.sh
gpu-lock.sh run bash .worktree/pipeline-parallelism/scripts/rdma-server.sh restart
```

### 正確性テスト — Qwen3.5-35B-A3B (MoE, 2C+2R)

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/pipeline-parallelism/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "Explain RDMA in detail" -n 64 \
  --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### 正確性テスト — Qwen3.5-27B (Dense, 2C+2R)

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/pipeline-parallelism/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-27B-GGUF:Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "Explain RDMA in detail" -n 64 \
  --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### フォールバック確認 — RDMA_ASYNC_COMPUTE=0

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  GGML_RDMA_ASYNC_COMPUTE=0 \
  .worktree/pipeline-parallelism/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "Hello" -n 16 --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## テスト結果

### Qwen3.5-35B-A3B (MoE, 2C+2R, 4GPU)

| 条件 | Prompt (t/s) | Generation (t/s) | 出力 |
|------|:---:|:---:|------|
| Phase 2 (ASYNC_COMPUTE=1) | **55.8** | **36.1** | 正常 |
| フォールバック (ASYNC_COMPUTE=0) | 47.6 | 36.5 | 正常 |

- **pp 改善: +17.2%** (47.6 → 55.8) — CUDA splits と RDMA compute のオーバーラップ効果
- tg は同等 (tg は ubatch=1 なのでパイプライン化の恩恵なし)

### Qwen3.5-27B (Dense, 2C+2R, 4GPU)

| 条件 | Prompt (t/s) | Generation (t/s) | 出力 |
|------|:---:|:---:|------|
| Phase 2 (ASYNC_COMPUTE=1) | **19.0** | **10.3** | 正常 |
| フォールバック (ASYNC_COMPUTE=0) | 13.4 | 10.7 | 正常 |

- **pp 改善: +41.8%** (13.4 → 19.0) — Dense モデルはレイヤー数が多く RDMA split 比率が高いため改善幅が大きい
- tg は同等 (ubatch=1)

### MoE vs Dense の改善幅の差

MoE (+17%) と Dense (+42%) で改善幅が大きく異なる理由:

- **MoE (Qwen3.5-35B-A3B)**: Active パラメータが 3B と少なく、各 RDMA split の計算時間が短い → CUDA split との並行化の余地が相対的に小さい
- **Dense (Qwen3.5-27B)**: 全 27B パラメータが活性化され、各 RDMA split の計算時間が長い → CUDA split の実行と大きくオーバーラップ可能

### 全テスト合格

| テスト | モデル | ASYNC_COMPUTE | 結果 |
|--------|--------|:---:|:---:|
| 正確性 | Qwen3.5-35B-A3B (MoE) | 1 | PASS |
| 正確性 | Qwen3.5-27B (Dense) | 1 | PASS |
| フォールバック | Qwen3.5-35B-A3B (MoE) | 0 | PASS |
| フォールバック | Qwen3.5-27B (Dense) | 0 | PASS |

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `fd2f7d962 (feature/pipeline-parallelism)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1239454) |

## 結論

- RDMA イベント/synchronize を機能実装し、パイプラインパラレリズムが動作することを確認
- **MoE で pp +17%、Dense で pp +42%** の改善を観測
- tg はパイプライン化の対象外 (ubatch=1) のため変化なし
- `RDMA_ASYNC_COMPUTE=0` フォールバックは両モデルで正常動作を確認
- 変更量は最小限 (1ファイル, +19/-6行) で、既存の `RDMA_ASYNC_COMPUTE` メカニズムを活用
