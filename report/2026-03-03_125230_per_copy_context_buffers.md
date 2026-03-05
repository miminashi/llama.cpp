# Per-copy Context Buffers による Pipeline Dispatch 修正

- **実施日時**: 2026年3月3日 12:52
- **ワークツリー**: `.worktree/pipeline-splits`
- **ブランチ**: `feature/pipeline-splits`
- **コミット**: `73a2fa063`

## 前提・目的

Per-copy Split Snapshots (commit `c07c4bdd4`) で実装した pipeline dispatch が segfault で動作しなかった問題を修正する。

- **背景**: pipeline decode loop で全 ubatch を事前ビルド → 一括 dispatch するため、`alloc_graph` 時に splits を copy slot ごとにスナップショット保存する機構を実装した。しかし実行時にクラッシュした。
- **目的**: segfault の根本原因を特定し、修正する
- **参照レポート**: [pipeline_parallelism_phase4](report/2026-03-03_031241_pipeline_parallelism_phase4.md), [split_level_pipeline_parallelism](report/2026-03-03_071550_split_level_pipeline_parallelism.md)

## 根本原因

### 問題1: Copy Slot のオーバーフロー

9 ubatch に対して 4 copy slot しかないため、5番目以降の ubatch が copy slot 0-3 のスナップショットを上書きしていた。

**修正**: pipeline decode loop を chunked 処理に変更。`n_copies` (=4) 個の ubatch ごとにビルド → dispatch → 同期を繰り返す。

### 問題2: sched->ctx のテンソル構造体の消失 (根本原因)

chunked 処理にしても依然 segfault。診断の結果、`split_graph()` が呼ばれるたびに `sched->ctx` が `ggml_free()` → `ggml_init()` で再作成され、共有の `context_buffer` が上書きされることが判明。

スナップショットの `graph.nodes[]` や `input_cpys[]` は `sched->ctx` 内のテンソル構造体へのポインタだが、次の `split_graph()` 呼び出しで `context_buffer` が再利用されるとポインタが dangling になる。

**重要な発見**: `ggml_free(ctx)` は `ggml_context` メタデータ構造体のみを解放し、外部提供の `mem_buffer` (= `context_buffer`) の内容自体は解放しない。したがって、copy slot ごとに別の `context_buffer` を使えばテンソル構造体はそれぞれのバッファ内で永続する。

### 修正方法: Per-copy Context Buffers

`ggml_backend_sched` に `copy_context_buffers[GGML_SCHED_MAX_COPIES]` を追加。`alloc_graph()` 内で `split_graph()` を呼ぶ前に `sched->context_buffer` を当該 copy slot 用のバッファに差し替え、完了後に元に戻す。

```cpp
// alloc_graph() 内
char * saved_context_buffer = nullptr;
if (sched->n_copies > 1 && sched->copy_context_buffers[sched->cur_copy] != nullptr) {
    saved_context_buffer = sched->context_buffer;
    sched->context_buffer = sched->copy_context_buffers[sched->cur_copy];
}
ggml_backend_sched_split_graph(sched, graph);
if (saved_context_buffer != nullptr) {
    sched->context_buffer = saved_context_buffer;
}
```

**メモリオーバーヘッド**: ~400KB × 4 copy slots = ~1.6MB (無視可能)

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-backend.cpp` | per-copy context buffer の allocate/free/swap、snapshot_splits の簡略化 |
| `src/llama-context.cpp` | pipeline decode loop を chunked 処理 (n_copies 単位) に変更 |

## 再現方法

### ビルド
```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/pipeline-splits/scripts/rdma-build.sh local
bash /home/ubuntu/projects/llama.cpp/.worktree/pipeline-splits/scripts/rdma-deploy.sh
```

### CUDA-only 正確性テスト
```bash
# Pipeline OFF
CUDA_VISIBLE_DEVICES=4,5 build/bin/llama-cli \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -fa 1 -c 256 -n 30 --seed 42 -ub 4 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log

# Pipeline ON
GGML_RDMA_PIPELINE=1 CUDA_VISIBLE_DEVICES=4,5 build/bin/llama-cli \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -fa 1 -c 256 -n 30 --seed 42 -ub 4 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### RDMA 正確性テスト
```bash
# Pipeline OFF
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  LD_LIBRARY_PATH=build/bin build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log

# Pipeline ON
GGML_RDMA_PIPELINE=1 GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  LD_LIBRARY_PATH=build/bin build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## テスト結果

### CUDA-only 正確性テスト (qwen2.5-0.5b, 2GPU, seed=42, -ub 4)

| 条件 | 出力 | 結果 |
|------|------|------|
| Pipeline OFF | "The capital of France is Paris." | OK |
| Pipeline ON | "The capital of France is Paris." | OK (一致) |
| Pipeline OFF (長文) | 素数リスト (2,3,5,7,11,13,17...) | OK |
| Pipeline ON (長文) | 素数リスト (2,3,5,7,11,13,17...) | OK (一致) |

### RDMA 正確性テスト (Qwen3.5-35B-A3B, 4GPU: CUDA4,5 + RDMA0,1, seed=42)

| 条件 | 出力 | Prompt t/s | Gen t/s | 結果 |
|------|------|:----------:|:-------:|------|
| Pipeline OFF | "Thinking Process: 1. Analyze..." | 53.7 | 36.1 | OK |
| Pipeline ON | "Thinking Process: 1. Analyze..." | 57.6 | 35.7 | OK (一致) |

### A/B ベンチマーク (Qwen3.5-35B-A3B, 4GPU, 探索的比較 5ペア)

**注**: Qwen3.5 は compute-bound (GPU 計算が 94.8%) のため大きな差は期待しない。リグレッション確認が目的。

#### Prompt (t/s)

| Pair | OFF (A) | ON (B) | Diff |
|:----:|:-------:|:------:|:----:|
| 1 | 115.4 | 116.5 | +1.1 |
| 2 | 116.2 | 116.1 | -0.1 |
| 3 | 114.9 | 116.1 | +1.2 |
| 4 | 116.3 | 116.1 | -0.2 |
| 5 | 116.5 | 116.8 | +0.3 |
| **平均** | **115.86** | **116.32** | **+0.46 (+0.40%)** |

#### Generation (t/s)

| Pair | OFF (A) | ON (B) | Diff |
|:----:|:-------:|:------:|:----:|
| 1 | 36.3 | 36.2 | -0.1 |
| 2 | 36.4 | 36.6 | +0.2 |
| 3 | 36.3 | 36.5 | +0.2 |
| 4 | 36.4 | 36.5 | +0.1 |
| 5 | 36.4 | 35.4 | -1.0 |
| **平均** | **36.36** | **36.24** | **-0.12 (-0.33%)** |

#### 判定

- **Prompt**: +0.40% — ノイズ範囲内、有意差なし
- **Generation**: -0.33% — ノイズ範囲内、リグレッションなし

Qwen3.5 は compute-bound のため期待通り。Communication-bound モデル (GLM-4.7) でのテストが必要だが、per-device connections との組み合わせが前提 (今回はスコープ外)。

## 発見事項

1. **`ggml_free()` の動作**: 外部提供バッファの内容は解放しない — per-copy バッファ戦略の根拠
2. **gallocr と copy slot**: `gallocr` は各 copy slot のテンソルに別のデータ領域を割り当てる。context buffer が別であれば、テンソル構造体 (metadata) とテンソルデータ (backend buffer) の両方が copy slot 間で独立
3. **Chunked pipeline**: `n_copies` (通常4) を超える ubatch 数でも安全に処理可能。9 ubatch → 3 chunk (4+4+1) で正常動作
4. **Pipeline パスの有効化条件**: `GGML_RDMA_PIPELINE=1` + `pipeline_parallel=true` (n_copies>1, RDMA backend あり) + `n_tokens_all > n_ubatch` (マルチ ubatch)

## 今後の課題

1. **Communication-bound モデルでの検証**: GLM-4.7 + per-device connections で pipeline の効果を測定
2. **Per-device connections との統合**: 現在の pipeline は shared connection で全デバイスを逐次処理。真のパイプライン効果には per-device connections が必要
3. **RDMA event 活用**: 現在 RDMA backend の `event_wait` は no-op。inter-device 並列性のためにはデバイス間の適切な同期が必要

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `73a2fa063 (feature/pipeline-splits)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 31°C | 33°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1257458) |
