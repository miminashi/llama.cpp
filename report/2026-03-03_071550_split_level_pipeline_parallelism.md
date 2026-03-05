# Split-level Pipeline Parallelism 実装レポート

- **実施日時**: 2026年3月3日 07:15
- **ワークツリー**: `.worktree/pipeline-splits` (branch: `feature/pipeline-splits`)
- **コミット**: `bc70d9d23`

## 前提・目的

Pipeline Parallelism Phase 1-4 (先行実装) の A/B ベンチマーク結果は PP +2.18% と理論値 (+30-60%) に大きく届かなかった。Gap Analysis により、`ggml-backend.cpp` の `compute_splits` が異なるデバイスで異なる ubatch を同時処理できないことが根本原因と特定された。

本実装は「Split-level Pipeline Parallelism」として、以下を目的とする:
1. RDMA `event_wait` の no-op 化によるクライアントサイドの CPU ブロッキング排除
2. サーバーサイドでの cross-device 同期機構の実装
3. パイプラインデコードループによる ubatch 処理の最適化

### 参照レポート
- [Pipeline Parallelism Phase 1-4 A/B Benchmark](report/2026-03-03_034054_pipeline_parallelism_ab_benchmark.md)
- [Pipeline Parallelism Gap Analysis](report/2026-03-03_050236_pipeline_parallelism_gap_analysis.md)
- [PP Profiling: Qwen3.5 Optimization](report/2026-03-02_195335_pp_profiling_qwen35_optimization.md)

## 実装内容

### Phase 1: Per-copy Split Snapshots (スキップ)

`ggml-backend.cpp` の per-copy split snapshots は実装調査の結果スキップした。理由:
- `compute_splits` は `alloc_graph` で設定された現在のスケジューラ状態を使用
- テンソル構造体は次の `split_graph` 呼び出しまで有効
- `gallocr` は各 copy slot に個別のメモリを割り当て

### Phase 2: Non-blocking RDMA Event Operations (`ggml-rdma.cpp`)

`GGML_RDMA_PIPELINE=1` 設定時、RDMA バックエンドの `event_wait` を no-op にする。

```cpp
static void ggml_backend_rdma_event_wait(...) {
    if (RDMA_PIPELINE) return;  // no-op
    // 既存の drain_pending_compute
}
```

### Phase 3: Server-side Cross-device Synchronization

#### クライアントサイド
- `pipeline_wait_entry` 構造体: `{src_device, expected_seq}` で cross-device 依存を追跡
- `pipeline_conn_state`: 接続ごとの pending waits を管理
- `g_device_compute_seq[64]`: デバイスごとの compute シーケンス番号 (atomic)
- `build_pipeline_payload()`: 既存の async コマンドを pipeline コマンドにラップ
- `cpy_tensor_async()`: cross-connection copy 時に pipeline wait を追加
- 3つの async dispatch パス (recompute, compute_update, full_graph) で `RDMA_CMD_PIPELINE_GRAPH_COMPUTE_ASYNC` を使用

#### サーバーサイド
- `pipeline_graph_compute()`: wait entries を解析し、依存デバイスの compute 完了を待機してから実行
- `pipeline_compute_seq_[64]` + mutex + condition_variable で cross-device 同期
- compute 完了後にシーケンス番号をインクリメント + `notify_all()`

#### ワイヤフォーマット
```
| n_waits(4B) | waits(N*8B) | device(4B) | sub_cmd(4B) | <original async payload> |
```

### Phase 4: Pipeline Decode Loop (`llama-context.cpp`)

`GGML_RDMA_PIPELINE=1` かつ `pipeline_parallel` かつ `n_tokens_all > n_ubatch` の場合に発動。

**初期実装 (バグあり)**: 全 ubatch の graph を事前ビルド → 一括 dispatch。`sched->splits` が最後の ubatch のものに上書きされ、不正なテンソルポインタで計算される問題。

**修正版**: alloc + set_inputs + compute を各 ubatch ごとに逐次実行し、出力抽出のみ最後にまとめる。
- 各 ubatch に専用の `llm_graph_result` スロットを割り当て (テンソル構造体の延命)
- backend ポインタを alloc 後・reset 前に保存
- `ggml_backend_sched_synchronize` 後に一括で出力抽出

## 検証結果

### 1. CUDA-only 正確性テスト (qwen2.5-0.5b, 2GPU CUDA4,5)

| 条件 | 出力 | PP (t/s) | TG (t/s) |
|------|------|:--------:|:--------:|
| ベースライン (`-ub 4`) | "Whispers of the ocean, a vast and deep..." | 1171.8 | 198.9 |
| Pipeline (`-ub 4`) | "Whispers of the ocean, a vast and deep..." | 1103.3 | 202.6 |

出力は完全一致。パイプラインパスの正確性を確認。

### 2. RDMA 正確性・性能テスト (Qwen3.5-35B-A3B, 4GPU CUDA4,5+RDMA0,1)

| 条件 | 出力冒頭 | PP (t/s) | TG (t/s) |
|------|---------|:--------:|:--------:|
| ベースライン (`-ub 128`) | "Thinking Process: 1. Analyze..." | 152.0 | 36.0 |
| Pipeline (`-ub 128`) | "Thinking Process: 1. Analyze..." | 152.4 | 36.4 |

出力は完全一致。**性能差なし** (PP +0.3%, TG +1.1%)。

### 性能差なしの原因分析

Qwen3.5-35B-A3B (MoE, 3B active) は **compute-bound** であることが既知:
- PP 計算の 94.8% がサーバーサイド GPU 計算時間
- RDMA 通信は 3.3% のみ → event_wait no-op による改善余地がない
- 共有接続 (デフォルト) ではサーバーが FIFO 逐次処理 → パイプライン効果なし

## 発見事項と制約

### 1. `sched->splits` の共有状態問題

`ggml_backend_sched` の splits は全 ubatch で共有されるグローバル状態。事前に全 ubatch の graph を alloc しても、最後の ubatch の splits だけが残る。これが Phase 1 (per-copy split snapshots) が本来必要な理由。

**回避策**: alloc + compute を各 ubatch ごとにペアリングすることで splits の整合性を保つ。ただし、これにより全 ubatch の同時 dispatch ができなくなり、パイプライン効果が制限される。

### 2. 真のパイプライン効果に必要な条件

| 条件 | 現状 | 必要な対応 |
|------|------|-----------|
| Per-copy split snapshots | 未実装 | `ggml-backend.cpp` に split snapshot/restore 機構を追加 |
| Per-device connections | `GGML_RDMA_PER_DEVICE_CONN=1` で利用可能 | ConnectX-4 では MTT キャッシュ制限あり |
| Communication-bound モデル | Qwen3.5 は compute-bound | GLM-4.7 (11GPU) での検証が必要 |
| サーバーサイド並列化 | FIFO 逐次処理 | Per-device connection + Phase 3 で実現可能 |

### 3. クライアント・サーバー互換性

Pipeline-splits コードは新しいコマンド (`RDMA_CMD_FLUSH_AND_GRAPH_COMPUTE_ASYNC`, `RDMA_CMD_PIPELINE_GRAPH_COMPUTE_ASYNC`) を含む。これらは `feature/pipeline-parallelism` 以降のブランチで追加されたもので、`feature/rdma-backend` のサーバーとは非互換。テスト時はクライアントとサーバーのバイナリバージョンを一致させる必要がある。

## 再現方法

### ビルド
```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/pipeline-splits/scripts/rdma-build.sh local
```

### サーバー起動 (2号機)
```bash
scp build/bin/rdma-server build/bin/lib*.so* 192.168.100.2:/tmp/pipeline-splits-bin/
ssh 192.168.100.2 "LD_LIBRARY_PATH=/tmp/pipeline-splits-bin \
  nohup /tmp/pipeline-splits-bin/rdma-server -H 0.0.0.0 -p 50051 &"
```

### 正確性テスト (CUDA-only)
```bash
GGML_RDMA_PIPELINE=1 CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-cli -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -c 256 -ub 4 -n 10 -p 'The capital of France is' \
  --flash-attn on --no-warmup --single-turn --simple-io \
  --log-file /tmp/llama-cli.log --seed 42
```

### RDMA テスト
```bash
GGML_RDMA_PIPELINE=1 CUDA_VISIBLE_DEVICES=4,5 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -ub 128 -n 20 \
  -p 'Write a poem about the ocean...' \
  --flash-attn on --no-warmup --single-turn --simple-io \
  --log-file /tmp/llama-cli.log --seed 42
```

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | Phase 2 (event_wait no-op), Phase 3 (pipeline data structures, pipeline command, cpy_tensor_async, server-side pipeline_graph_compute) |
| `src/llama-context.cpp` | Phase 4 (pipeline decode loop, pipeline result pre-allocation) |
| `src/llama-context.h` | `gf_pipeline_results` メンバ追加 |

## 結論

Split-level Pipeline Parallelism の Phase 2-4 を実装し、正確性を検証した。Qwen3.5-35B-A3B (compute-bound) では性能差は観測されなかった。

真のパイプライン効果を得るには:
1. `ggml-backend.cpp` に per-copy split snapshots を実装 (全 ubatch の事前 dispatch を可能にする)
2. Per-device connections でサーバーサイド並列化を有効にする
3. GLM-4.7 等の communication-bound モデルで検証する

現時点での実装は、将来の per-copy split snapshots 実装に向けた基盤として機能する。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `bc70d9d23 (feature/pipeline-splits)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 1257458, port 50052; PID 1282951, port 50051) |
