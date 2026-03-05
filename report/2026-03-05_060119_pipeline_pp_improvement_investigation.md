# Pipeline 並列化による PP 改善調査

- **実施日時**: 2026年3月5日 06:01
- **ワークツリー**: `.worktree/pipeline-complete` (branch: `feature/pipeline-complete`, commit: `34542217f`)
- **参照レポート**:
  - [pipeline-complete 8GPU ベンチマーク](report/2026-03-05_031342_pipeline_complete_8gpu_qwen35_retest.md)
  - [per-device 単独 8GPU ベンチマーク](report/2026-03-05_033222_per_device_conn_alone_8gpu_benchmark.md)
  - [PP プロファイリング (Qwen3.5 6GPU)](report/2026-03-02_195335_pp_profiling_qwen35_optimization.md)

## 前提・目的

### 背景

Pipeline 並列化 (PIPELINE) と Send ring buffer の組み合わせが GLM-4.7 (communication-bound モデル) で PP 改善を実現できるか検証する。

| 時点 | Qwen3.5 4GPU pp128 | GLM-4.7 11GPU pp128 |
|------|:---:|:---:|
| Send always-signal 適用前 (merge/validated) | — | ~30.6 t/s |
| Send always-signal 適用後 (feature/rdma-backend) | ~189 t/s | ~24.3 t/s |
| Ring buffer + PIPELINE (pipeline-complete) | ~200 t/s | **未検証** |

### 仮説

GLM-4.7 では ring buffer による Send selective signaling 回復 + PIPELINE の FIFO wait 解消が相乗効果を発揮し、pp128 を ~30.6 t/s 付近まで回復できる。

## 再現方法

### サーバーデプロイ

pipeline-complete サーバーのデプロイが必要（`RDMA_CMD_PIPELINE_GRAPH_COMPUTE_ASYNC` (cmd 29) を feature/rdma-backend サーバーは持たないため）。

```bash
gpu-lock.sh run bash scripts/rdma-server.sh stop
bash .worktree/pipeline-complete/scripts/rdma-deploy.sh
bash .worktree/pipeline-complete/scripts/rdma-server.sh start
```

### Qwen3.5 4GPU A/B ベンチマーク

```bash
# A: Baseline (ring buffer のみ)
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  .worktree/pipeline-complete/build/bin/llama-bench \
  -m ~/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -fa 1 -t 4 -p 128 -n 32 -r 1 -o csv

# B: PIPELINE=1
GGML_RDMA_PIPELINE=1 GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  .worktree/pipeline-complete/build/bin/llama-bench \
  -m ~/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -fa 1 -t 4 -p 128 -n 32 -r 1 -o csv
```

### GLM-4.7 11GPU A/B ベンチマーク

```bash
# A/B 同様に CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 で実行
# モデル: ~/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf
```

### プロファイリング

```bash
GGML_RDMA_PROFILE=1 GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  .worktree/pipeline-complete/build/bin/llama-bench \
  -m <GLM-4.7 model> -ngl 999 -fa 1 -t 4 -p 128 -n 0 -r 1 -o csv
```

## 結果

### 1. Qwen3.5 4GPU: PIPELINE 効果 (ABAB × 5 pairs)

| メトリクス | Baseline (A) | PIPELINE=1 (B) | 差分 | p値 |
|-----------|:---:|:---:|:---:|:---:|
| **pp128** | 199.3 t/s | 199.2 t/s | **-0.02%** (ns) | 0.895 |
| **tg32** | 33.56 t/s | 33.83 t/s | **+0.82%** | 0.024 |

前回結果と一致: Qwen3.5 (compute-bound) では PIPELINE は PP に効果なし。

### 2. GLM-4.7 11GPU: PIPELINE 効果 (ABAB × 5 pairs)

| メトリクス | Baseline (A) | PIPELINE=1 (B) | 差分 | p値 |
|-----------|:---:|:---:|:---:|:---:|
| **pp128** | 24.167 t/s | 24.172 t/s | **+0.02%** (ns) | 0.482 |
| **tg32** | 8.455 t/s | 8.434 t/s | **-0.25%** (ns) | 0.210 |

**仮説は否定された**: GLM-4.7 でも PIPELINE は PP にまったく効果なし。

### 3. GDR 無効テスト

| 条件 | pp128 | tg32 |
|------|:---:|:---:|
| GDR 有効 (デフォルト) | 24.169 | 8.447 |
| GDR 無効 (`GGML_RDMA_NO_GDR=1`) | 24.169 | 8.449 |

GDR の有無は PP/TG に影響なし。GDR sync-before-recv は PP ボトルネックではない。

### 4. プロファイリング結果 (GLM-4.7 11GPU pp128)

#### PIPELINE=0 (Baseline)

```
graph_compute #1: 0.65 ms (dispatch) → cmd=23 drain: 510.70 ms (server compute)
graph_compute #2: 0.67 ms             → cmd=23 drain: 558.50 ms
graph_compute #3: 0.60 ms             → cmd=23 drain: 480.81 ms
graph_compute #4: 0.56 ms             → cmd=23 drain: 294.31 ms
─────────────────────────────────────────────────────────────────
RDMA total: 1846 ms, dispatch total: 2.5 ms, server wait: 1843 ms
ubatch total: 5212 ms → CUDA: ~3366 ms (7 devices)
```

#### PIPELINE=1

```
graph_compute #1: 0.69 ms (dispatch, no block)
graph_compute #2: 511.70 ms (cmd=30 send blocks 510.97 ms)
graph_compute #3: 559.72 ms (cmd=30 send blocks 559.08 ms)
graph_compute #4: 480.87 ms (cmd=30 send blocks 480.35 ms)
+ final cmd=23 drain: 293.15 ms
─────────────────────────────────────────────────────────────────
RDMA total: 1846 ms (PIPELINE=0 と完全一致)
ubatch total: 5209 ms
```

## 分析

### 根本原因: Layer-split の依存関係

Layer-split モデルでは各デバイスが前のデバイスの出力に依存するため、デバイス計算は**逐次実行が必須**:

```
D0 → D1 → D2 → D3 (シリアル、並列化不可)
```

PIPELINE の `pipeline_waits` はサーバーに前デバイスの完了待ちを指示する。結果として:
- **PIPELINE=0**: クライアントの `event_wait` → `drain_pending_compute` でブロック
- **PIPELINE=1**: サーバーの `pipeline_waits` で同等のブロック、クライアント send がブロック

ブロッキングの**場所**が移動するだけで、**合計時間は同一** (1846ms)。

### Ring buffer が GLM-4.7 PP を回復しない理由

| 要因 | 詳細 |
|------|------|
| **graph_compute dispatch は <1ms** | Send CQ polling overhead が無視可能 → selective signaling の改善余地ゼロ |
| **ボトルネックは server-side GPU compute** | 各デバイス 294-558ms の計算待ち (合計 1843ms / 5212ms = 35%) |
| **RDMA Write always-signal は無関係** | GDR 有効時は RDMA Write 不使用、GDR 無効でも PP 変化なし |
| **Send ring buffer の効果対象が違う** | Ring buffer は data send (大サイズ) に効く。graph_compute command send (小サイズ, inline) には無関係 |

### 30.6 t/s と 24.17 t/s の差 (~26%) の正体

merge/validated (30.6 t/s) と現在 (24.17 t/s) の差は Send selective signaling ではない。差の根源は今回の調査では特定できなかったが、以下の可能性が高い:

1. **異なる `event_wait` 実装**: merge/validated では `drain_pending_compute` が呼ばれないか、異なるタイミングで呼ばれていた可能性
2. **異なるコードパス**: always-signal fix に伴うコード変更が、signaling 以外の部分でも overhead を追加した可能性
3. **測定条件の差異**: merge/validated 時点のベンチマークと現在の条件が異なる可能性

### PP 改善の残余オプション

| 案 | 実現可能性 | 期待効果 | 備考 |
|---|:---:|:---:|---|
| **Split 並列ループ** (ggml-backend.cpp 改修) | 低 | +30-60% (理論) | Layer-split は並列化不可。Row-split のみ恩恵あるが P100 では非効率 |
| **Ubatch 並列 dispatch** | 中 | 小 | pp128 は単一 ubatch のため恩恵なし。pp512+ では効果あり得る |
| **Server-side compute 高速化** | 低 | 未知 | GPU ハードウェア制約 (P100 16GB) |
| **CUDA/RDMA overlap** | 高 | +35% (理論) | 7 CUDA (3366ms) と 4 RDMA (1846ms) の並列化。ggml-backend.cpp の split ループ改修が必要 |

**CUDA/RDMA overlap** が最も有望: 現在は 7 CUDA splits + 4 RDMA splits が逐次実行 (3366 + 1846 = 5212ms)。もし CUDA と RDMA を並列化できれば max(3366, 1846) ≈ 3366ms → pp128 ≈ 38 t/s (+57%)。ただし ggml-backend.cpp の根本改修が必要。

## 結論

1. **Ring buffer は GLM-4.7 PP を回復しない**: graph_compute dispatch (<1ms) が insignificant なため
2. **PIPELINE は GLM-4.7 PP に効果なし**: Layer-split のデバイス依存関係がデバイス間並列化を阻止
3. **GDR は PP に影響なし**: sync-before-recv は PP ボトルネックではない
4. **PP ボトルネックは server-side GPU compute の逐次実行**: 各デバイス 300-560ms の計算が直列
5. **30.6 → 24.17 の差は未解明**: Send selective signaling ではなく、event_wait やコードパスの変更に起因する可能性

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c1ca6d5fd (dirty) (feature/rdma-backend)` | — |
| テストバイナリ | `.worktree/pipeline-complete` (`34542217f`) | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 28°C | 27°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
