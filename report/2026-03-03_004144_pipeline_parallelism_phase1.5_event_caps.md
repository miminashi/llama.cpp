# Pipeline Parallelism Phase 1.5: RDMA Event Caps 実装レポート

- **実施日時**: 2026年3月3日 00:41
- **ワークツリー**: `.worktree/pipeline-parallelism`
- **ブランチ**: `feature/pipeline-parallelism`
- **コミット**: `cdf988202` (Phase 1: `8243337ba` の上)

## 前提・目的

Phase 1 (`8243337ba`) で `process_ubatch()` の graph reuse 無効化を実装済みだが、RDMA バックエンドが `caps.async = false`, `caps.events = false` を報告するため、CUDA+RDMA 混在環境では `pipeline_parallel = false` となりパイプライン並列が無効化されていた。

- **背景**: `llama-context.cpp` line 309-334 のゲート条件が、全 GPU バックエンドに `async=true`, `events=true` を要求する
- **目的**: RDMA バックエンドに no-op イベント実装を追加し、ゲート条件を通過させてパイプライン並列を有効化する
- **根拠**: RDMA の `graph_compute` は同期的（サーバー応答を待つ）であるため、イベント操作はすべて「常に完了済み」として扱えば正しい

## 変更内容

### `ggml/src/ggml-rdma/ggml-rdma.cpp` (1ファイルのみ、+31/-7行)

1. **No-op イベント関数 5 つを追加**
   - `ggml_backend_rdma_event_record` — backend interface 用 (no-op)
   - `ggml_backend_rdma_event_wait` — backend interface 用 (no-op)
   - `ggml_backend_rdma_device_event_new` — `new ggml_backend_event { dev, nullptr }` を返す
   - `ggml_backend_rdma_device_event_free` — `delete event`
   - `ggml_backend_rdma_device_event_synchronize` — no-op (同期操作は既に完了済み)

2. **Backend interface struct に `event_record`, `event_wait` を設定** (NULL → 関数ポインタ)

3. **Device interface struct に `event_new`, `event_free`, `event_synchronize` を設定** (NULL → 関数ポインタ)

4. **Device caps を更新**: `async = true`, `events = true`

## 再現方法

### ビルド・デプロイ

```bash
bash .worktree/pipeline-parallelism/scripts/rdma-build.sh local
bash .worktree/pipeline-parallelism/scripts/rdma-deploy.sh
bash .worktree/pipeline-parallelism/scripts/rdma-server.sh restart
```

### 正確性テスト — Qwen3.5-35B-A3B (MoE, 4GPU: 2C+2R)

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/pipeline-parallelism/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "Explain RDMA in detail" -n 64 \
  --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### 正確性テスト — Qwen3.5-27B (Dense, 4GPU: 2C+2R)

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/pipeline-parallelism/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-27B-GGUF:Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "Explain RDMA in detail" -n 64 \
  --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## 検証結果

### 正確性テスト

| モデル | タイプ | PP (t/s) | TG (t/s) | 出力品質 |
|--------|--------|:--------:|:--------:|---------|
| Qwen3.5-35B-A3B | MoE | 56.3 | 36.4 | 正常（ガーベジなし） |
| Qwen3.5-27B | Dense | 18.9 | 10.2 | 正常（ガーベジなし） |

- 両モデルとも Pipeline parallelism 有効状態で正常な推論出力を確認
- RDMA デバイスへのモデル分散も正常に動作（メモリ分配ログで確認）

### Pipeline parallelism ゲート通過の確認

Event caps 追加前は `pipeline_parallel = false` に設定されていた。追加後は RDMA デバイスが `async=true, events=true` を報告するため、ゲート条件を通過して `pipeline_parallel = true` となる。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c1ca6d5fd (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 1231407) |

## 次のステップ

- 性能テスト（Pipeline ON vs OFF の A/B 比較）は Phase 2 以降で実施
- Qwen3.5 は compute-bound なので大きな pp 改善は期待しない（ゲート通過の動作確認が主目的）
- GLM-4.7 11GPU での検証は 16GPU 拡張時に実施予定
