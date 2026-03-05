# パイプライン並列化バグ修正 (Bug 1 修正 + Bug 2 調査)

- **実施日時**: 2026年3月4日 13:42
- **ワークツリー**: `.worktree/pipeline-splits`
- **ブランチ**: `feature/pipeline-splits`
- **コミット**: `a423ade64`

## 前提・目的

パイプライン並列化のスケーリングベンチマーク (pp20000, Qwen3.5-35B-A3B) で発見された 2 つのバグの修正:

- **Bug 1**: `GGML_RDMA_PIPELINE=1` で pp20000 がクラッシュ (`ggml_cuda_mul_mat_id` CUDA error)
- **Bug 2**: `GGML_RDMA_PARALLEL=1` (pipeline + per-device combined) で 3+ RDMA デバイス使用時にゴミ出力

関連レポート:
- [パイプラインスケーリングベンチマーク](report/2026-03-04_120833_pipeline_scaling_pp20000_benchmark.md)
- [パイプライン + per-device 複合ベンチマーク](report/2026-03-03_145744_pipeline_perdevice_combined_benchmark.md)

## 根本原因分析

### Bug 1: gallocr バッファ再利用によるクラッシュ

`llama-context.cpp` のパイプライン処理ループは ubatch をチャンク (n_copies=4) 単位で処理していた:

```
Phase 1: for ci in 0..3:  reset → build → alloc_graph → set_inputs  (4回)
Phase 2: for ci in 0..3:  compute_copy(copy_id)                     (4回)
Phase 3: synchronize + extract
```

Phase 1 の各 `ggml_gallocr_alloc_graph` が同じ計算バッファプールからテンソルデータポインタを割り当てるため、N 回目の alloc が N-1 回目の割り当てを上書き。Phase 2 で dispatch する時点では最後の ubatch のデータのみ有効で、先行 ubatch の入力データは破損済み。サーバーが破損データで MoE 計算を実行し CUDA error。

### Bug 2: intra-ubatch 同期の問題 (未解決)

当初は Bug 1 と同じ inter-ubatch バッファ再利用が原因と推定したが、修正後も再現。

調査の結果、Bug 2 の根本原因は inter-ubatch ではなく **intra-ubatch** (単一 ubatch 内) の同期問題:

- `GGML_RDMA_PARALLEL` は `event_wait` を NO-OP にし、代わりに `pipeline_wait` 機構で cross-device 依存関係を管理
- `pipeline_wait` は 2 RDMA デバイスでは正常動作するが、3+ デバイスで破綻
- 修正には `ggml-rdma.cpp` の pipeline_wait/event_wait 機構の改修が必要

## 修正内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/llama-context.h` | `gf_pipeline_results` メンバ削除 (-3 行) |
| `src/llama-context.cpp` | パイプライン分岐ブロック削除 + per-ubatch sync 追加 (-171 行, +7 行) |

### 修正の要点

1. **チャンク化マルチ ubatch パイプライン処理を削除** (164 行のコード削除)
2. **標準パスに per-ubatch 同期を追加**:

```cpp
const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);

// Pipeline mode: synchronize between ubatches to prevent buffer reuse races
// (event_wait is NO-OP in pipeline mode, so explicit sync is needed)
static const bool RDMA_PIPELINE_MODE = !!getenv("GGML_RDMA_PIPELINE") || !!getenv("GGML_RDMA_PARALLEL");
if (RDMA_PIPELINE_MODE && cparams.pipeline_parallel) {
    ggml_backend_sched_synchronize(sched.get());
}
```

### 保持される最適化

- **ubatch 内パイプライン並列化**: pipeline_wait + deferred copies による cross-device 依存関係管理
- **非同期コマンドディスパッチ**: RDMA graph_compute は fire-and-forget

### 失われる最適化

- **ubatch 間パイプライン並列化**: ubatch N の計算と ubatch N+1 の通信のオーバーラップ (バグの原因)

## 検証結果

### Bug 1 修正確認: pp20000 クラッシュ解消

| テスト条件 | 結果 |
|-----------|------|
| `GGML_RDMA_PIPELINE=1` pp20000 (2C+2R, 4GPU) | **PASS**: 280.96 t/s (クラッシュなし) |
| `GGML_RDMA_PIPELINE=1` pp128 (2C+2R, 4GPU) | **PASS**: 201.94 t/s |
| `GGML_RDMA_PIPELINE=1` tg32 (2C+2R, 4GPU) | **PASS**: 36.14 t/s |

### Bug 2 テスト: 3+ RDMA ゴミ出力

| テスト条件 | 結果 |
|-----------|------|
| `GGML_RDMA_PARALLEL=1` 2C+2R (4GPU) | **PASS**: 正常な日本語出力 |
| `GGML_RDMA_PARALLEL=1` 3C+3R (6GPU) | **FAIL**: ゴミ出力 (未修正) |

### 回帰テスト: baseline (環境変数なし)

| テスト条件 | 結果 |
|-----------|------|
| Baseline pp128 (2C+2R, 4GPU) | **PASS**: 202.05 t/s |
| Baseline tg32 (2C+2R, 4GPU) | **PASS**: 35.66 t/s |

## 再現方法

### ビルド・デプロイ

```bash
bash .worktree/pipeline-splits/scripts/rdma-build.sh local
bash scripts/gpu-lock.sh run bash .worktree/pipeline-splits/scripts/rdma-deploy.sh
bash scripts/gpu-lock.sh run bash .worktree/pipeline-splits/scripts/rdma-server.sh restart
```

### Bug 1 テスト

```bash
bash scripts/gpu-lock.sh run GGML_RDMA_PIPELINE=1 CUDA_VISIBLE_DEVICES=4,5 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/pipeline-splits/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -dev 'CUDA0/CUDA1/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]' \
  -ngl 999 -sm layer -fa 1 -r 1 -p 20000 -n 32 -o csv
```

### Bug 2 テスト

```bash
bash scripts/gpu-lock.sh run GGML_RDMA_PARALLEL=1 CUDA_VISIBLE_DEVICES=4,5,6 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/pipeline-splits/build/bin/llama-cli \
  --model /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051]' \
  -ngl 999 -sm layer -fa 1 -p '日本語で美しい詩を書いてください。' -n 128 \
  --log-file /tmp/llama-cli.log
```

## Bug 2 の今後の調査方針

Bug 2 の根本原因は `ggml-rdma.cpp` の intra-ubatch 同期機構にある:

1. **`event_wait` NO-OP**: `RDMA_PIPELINE` 有効時、`event_wait` が無条件で NO-OP になるが、per-device 接続では各デバイスが独立した FIFO を持つため FIFO 順序保証が成立しない
2. **`pipeline_wait` の 3+ デバイス問題**: `cpy_tensor_async` による deferred copy + pipeline_wait は 2 デバイスでは正常だが、3+ デバイスで依存関係チェーンが正しく機能しない
3. **考えられる修正案**:
   - `event_wait` を per-device 接続時には NO-OP にしない (同一バックエンド型チェック付き)
   - pipeline_wait のシーケンス番号管理を 3+ デバイスチェーンに対応させる
   - 最もシンプルな修正: combined モードでは `event_wait` を有効化し pipeline_wait を無効化

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `a423ade64 (feature/pipeline-splits)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 1336372) |
