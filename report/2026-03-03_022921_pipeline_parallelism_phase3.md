# パイプライン並列化 Phase 3: FULL_GRAPH 同期 flush 排除 + プロファイリング

- **実施日時**: 2026年3月3日 02:29
- **ワークツリー**: `.worktree/pipeline-parallelism`
- **ブランチ**: `feature/pipeline-parallelism`
- **コミット**: `1a4c7225c`

## 前提・目的

Phase 2 (`fd2f7d962`) で RDMA event/synchronize を実装し、ubatch 間パイプラインオーバーラップを実現した。しかし、パイプラインモードでは copy slot ローテーションにより `graph_cache::is_cached()` が常に false を返し、毎 ubatch で FULL_GRAPH パスが実行される。

FULL_GRAPH パスの問題点:
1. **同期 flush** (`send_rdma_cmd(RDMA_CMD_FLUSH_ALL_STAGING)`) — ブロッキングラウンドトリップ。サーバーの前回 async compute 完了を待つ
2. **フルグラフ再シリアライズ** (`serialize_graph()`) — 毎回全ノードを走査 (~1ms/回)

対照的に RECOMPUTE パスは flush+compute を1コマンドに統合し `send_rdma_cmd_async()` で fire-and-forget（ブロッキングなし）。

**目的**: FULL_GRAPH パスでも flush + graph compute を1つの async コマンドに統合し、同期 flush を排除する。

### 参照レポート

- [Phase 1: パイプライン並列化実装](report/2026-03-02_233154_pipeline_parallelism_phase1.md)
- [Phase 1.5: Event caps 実装](report/2026-03-03_004144_pipeline_parallelism_phase1.5_event_caps.md)
- [Phase 2: RDMA events 実装](report/2026-03-03_013808_pipeline_parallelism_phase2.md)

## 変更内容

### Part 1: FLUSH_AND_GRAPH_COMPUTE_ASYNC 統合コマンド

**変更ファイル**: `ggml/src/ggml-rdma/ggml-rdma.cpp`

1. **新 enum 値**: `RDMA_CMD_FLUSH_AND_GRAPH_COMPUTE_ASYNC` を追加
2. **クライアント FULL_GRAPH パス**: RDMA_ASYNC_COMPUTE=1 時、flush + copies + graph data を1つの async コマンドに統合。sync フォールバック (RDMA_ASYNC_COMPUTE=0) は既存動作を維持
3. **サーバー `flush_and_graph_compute` メソッド**: copies/flush/graph data をパースし、flush → deferred copies → graph_compute を順次実行
4. **サーバーハンドラ**: switch case に `RDMA_CMD_FLUSH_AND_GRAPH_COMPUTE_ASYNC` を追加

Wire format:
```
| n_copies(4B) | copy_entries(N*48B) | n_flush(4B) | flush_entries(M*24B) | graph_data... |
```

### Part 2: パイプラインプロファイリング

**変更ファイル**: `src/llama-context.cpp`

`process_ubatch` に `GGML_RDMA_PROFILE` 環境変数ゲートのタイミング計測を追加:
- `build_graph`, `alloc_graph`, `set_inputs`, `graph_compute` の各フェーズの所要時間
- ubatch 番号、パイプラインモードフラグ、トークン数を出力

## 正確性テスト結果

### MoE (Qwen3.5-35B-A3B, 4GPU, -ub 32)

| テスト条件 | pp (t/s) | tg (t/s) | 結果 |
|-----------|:--------:|:--------:|:----:|
| Async (デフォルト) | 56.3 | 36.1 | 正常出力 |
| Sync (RDMA_ASYNC_COMPUTE=0) | 47.3 | 36.4 | 正常出力 |
| Profiling (GGML_RDMA_PROFILE=1) | 60.0 | 36.2 | 正常出力 + プロファイル出力 |

## プロファイリング分析

### Prompt Processing (pipeline_pp=1)

| ubatch | n_tokens | build (ms) | alloc (ms) | inputs (ms) | compute (ms) | 備考 |
|:------:|:--------:|:----------:|:----------:|:-----------:|:------------:|:----:|
| #1 | 2 | 3.12 | 4.00 | 0.03 | 370.43 | 初回 (model init含む) |
| #2 | 2 | 2.04 | 2.05 | 0.01 | 42.30 | |
| #3 | 16 | 2.58 | 5.33 | 0.01 | 205.34 | |

### Token Generation (pipeline_pp=0)

| ubatch | build (ms) | alloc (ms) | inputs (ms) | compute (ms) |
|:------:|:----------:|:----------:|:-----------:|:------------:|
| #4+ | 0.00 (reuse) | 0.00 | 0.01 | ~19.8 |

### RDMA graph_compute 内訳

| パス | serialize (ms) | send (ms) | 回数 | 備考 |
|:---:|:-------------:|:---------:|:----:|:----:|
| FULL_GRAPH async | 0.64-2.03 | 0.10-0.70 | 8 | n_flush=0 (GDR直接書き込み) |
| RECOMPUTE async | 0.25-0.38 (pre_send) | 0.01 | 52+ | graph 再利用 |

**key finding**: 今回のテスト環境では `n_flush=0`（全バッファが GDR 対応で staging flush 不要）。つまり FULL_GRAPH パスの同期 flush は空操作だったが、ブロッキング send_rdma_cmd + recv のラウンドトリップ自体が除去された。

### 残存ボトルネック

1. **serialize_graph**: 毎 FULL_GRAPH 呼び出しで 0.6-2.0 ms。graph_cache パイプライン対応で排除可能
2. **build_graph + alloc_graph**: 毎 pp ubatch で 2-5 ms。スケジューラ fast-path で排除可能
3. **compute (tg)**: ~19.8 ms/ubatch — サーバー GPU 計算時間がドミナント

## 再現方法

### ビルド・デプロイ

```bash
bash .worktree/pipeline-parallelism/scripts/rdma-build.sh local
bash .worktree/pipeline-parallelism/scripts/rdma-deploy.sh
gpu-lock.sh run bash .worktree/pipeline-parallelism/scripts/rdma-server.sh restart
```

### 正確性テスト

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/pipeline-parallelism/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "Explain RDMA in detail" -n 64 \
  --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### プロファイリングテスト

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  GGML_RDMA_PROFILE=1 \
  .worktree/pipeline-parallelism/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "Explain RDMA in detail" -n 32 \
  --single-turn --simple-io --log-file /tmp/llama-cli.log 2>&1
```

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c1ca6d5fd (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 34°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 1247498) |

## まとめ

Phase 3 で FULL_GRAPH パスの同期 flush を排除した。実装は正常動作を確認。今回のテスト環境では GDR が有効で `n_flush=0` だったため flush 自体の影響は軽微だが、ブロッキング send/recv ラウンドトリップの除去は完了した。

Phase 2 との A/B 性能比較は未実施（Phase 2 バイナリとの交互テストが必要）。正確性テストでは pp=56-60 t/s を記録しており、Phase 2 の pp=55.8 t/s と同等以上。

次ステップ候補:
1. **A/B 性能比較**: Phase 2 vs Phase 3 の定量評価
2. **Phase 4: graph_cache パイプライン対応**: serialize_graph の毎 ubatch 実行を排除
3. **Phase 5: build_graph/alloc_graph fast-path**: スケジューラレベルのパイプライン最適化
