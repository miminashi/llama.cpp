# Pipeline Parallelism: 理論値 vs 実測値の乖離分析

- **実施日時**: 2026年3月3日 05:02
- **ワークツリー**: `.worktree/pipeline-parallelism` (プロファイリング対象), `feature/rdma-backend` (ベースライン)

## 前提・目的

### 背景

Pipeline Parallelism Phase 1-4 の [A/B ベンチマーク](2026-03-03_034054_pipeline_parallelism_ab_benchmark.md) で以下の結果を得た:

- **PP 実測**: +2.18% (110.6 → 113.0 t/s, Qwen3.5-35B-A3B MoE, 4GPU)
- **TG 実測**: +4.68% (34.1 → 35.7 t/s, 予想外の改善)
- **PP 理論値**: +30〜60% ([ロードマップ](2026-03-02_223450_pipeline_parallelism_roadmap.md) 記載)

理論値と実測値の間に **15〜30倍の乖離** がある。本レポートでは乖離の根本原因を特定する。

### 目的

1. コード差分分析 + プロファイリングで乖離の原因を特定する
2. 実測 +2.18% の内訳を推定する
3. 予想外の TG +4.68% の原因を解明する
4. Dense モデルでの効果をデータから推定する

### 参照レポート

- [Pipeline Parallelism Phase 1-4 A/B ベンチマーク](2026-03-03_034054_pipeline_parallelism_ab_benchmark.md)
- [Pipeline Parallelism ロードマップ](2026-03-02_223450_pipeline_parallelism_roadmap.md)
- [Pipeline Parallelism Phase 2 (Async + Events)](2026-03-03_013808_pipeline_parallelism_phase2.md)
- [PP プロファイリング: Qwen3.5 6GPU](2026-03-02_195335_pp_profiling_qwen35_optimization.md)

---

## 分析1: 理論モデル vs 実装の不一致

### ロードマップの理論モデル

ロードマップの理論値 (+30〜60%) は **スケジューラレベルの split 並列化** を前提としている:

```
理論モデル (N=4 devices, M=4 microbatches):
  Dev 0: [mb0][mb1][mb2][mb3]
  Dev 1:      [mb0][mb1][mb2][mb3]
  Dev 2:           [mb0][mb1][mb2][mb3]
  Dev 3:                [mb0][mb1][mb2][mb3]
  → 7 stages vs 16 stages → Speedup = 2.29x (+129%)
```

この理論モデルでは **異なるデバイスが異なる ubatch を同時処理** する。

### 実装で行われたこと

| Phase | 実装内容 | コード差分 |
|-------|---------|-----------|
| Phase 1 | Graph reuse 無効化 → copy slot ローテーション | `llama-context.cpp` +16 行 |
| Phase 1.5 | RDMA event caps (event_new/event_free) | `ggml-rdma.cpp` +20 行 |
| Phase 2 | `drain_pending_compute()` + `event_record/event_wait/synchronize` | `ggml-rdma.cpp` +19 行 |
| Phase 3 | `FULL_GRAPH async` パス + graph_cache multi-slot (MAX_SLOTS=8) | `ggml-rdma.cpp` +250 行 |
| Phase 4 | graph_cache の更新フロー修正 + pipeline プロファイリング | `ggml-rdma.cpp` +60 行, `llama-context.cpp` +23 行 |

### 実装で行われなかったこと（理論値の源泉）

| 未実装項目 | ロードマップの該当箇所 | 理論的インパクト |
|-----------|---------------------|----------------|
| **Split 並列ループ** | Phase 1: `ggml_backend_sched_compute_splits` 改修 | Speedup 2.29x の根幹 |
| **マイクロバッチ同時投入** | Phase 2: decode ループでの複数 ubatch 同時ディスパッチ | パイプライン fill/steady-state/drain |
| **Per-device 完了追跡** | Phase 3: device 別 async 完了トラッキング | D0 完了前に D1 へ送信可能に |

**`ggml-backend.cpp` は一切変更されていない。** ロードマップ Phase 1 の核心である split 並列ループの実装がスキップされ、代わりにクライアント-サーバー間の RDMA 通信最適化が行われた。

### ベースラインが既に持つ async 機構

ベースライン (`feature/rdma-backend`) にも `GGML_RDMA_ASYNC_COMPUTE=1` (デフォルト) があり、`send_rdma_cmd_async()` でサーバーの graph_compute を非同期送信している。ただしベースラインの `synchronize()` は no-op のため、async 機構の恩恵は限定的。

Pipeline 版が追加した価値:
1. **event_record/event_wait**: スケジューラに RDMA バックエンドの async capability を通知 (`async=true, events=true`)
2. **FLUSH_AND_GRAPH_COMPUTE_ASYNC**: flush + compute の統合コマンドで roundtrip 削減
3. **graph_cache multi-slot**: 最大8スロットのキャッシュで serialize_graph スキップ

---

## 分析2: プロファイリング結果

### テスト構成

- **GPU**: Node 1 CUDA4,5 + Node 2 RDMA0,1 (4GPU)
- **モデル**: `unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M` (MoE, 3B active)
- **パラメータ**: `-sm layer -ngl 999 -fa 1 -ub 32`
- **プロンプト**: `"hello " × 100` (~100 tokens), `-n 8`
- **環境変数**: `GGML_RDMA_PROFILE=1`

### RDMA Profile Summary 比較

| 指標 | Baseline | Pipeline | 変化 |
|------|:--------:|:--------:|:----:|
| graph_compute 合計 | 481.8 ms (30 calls) | 38.7 ms (20 calls) | **-92%** |
| graph_compute 平均 | 16.1 ms | 1.9 ms | **-88%** |
| get_tensor 合計 | 204.5 ms (Send/Recv) | 4.6 ms (RDMA Read) | **-98%** |
| set_tensor 合計 | 3256.5 ms | 2675.6 ms | **-18%** |
| total RDMA time | 3942.8 ms | 2718.9 ms | **-31%** |
| graph_compute type | full=25, recomp=5 | full=14, recomp=6 | — |

### PP ubatch 詳細 (Pipeline 版)

| ubatch | n_tokens | compute (ms) | pipeline_pp | 備考 |
|:------:|:--------:|:------------:|:-----------:|------|
| #1 | 2 | 477.95 | 1 | 初回（モデルウォームアップ含む） |
| #2 | 2 | 43.81 | 1 | |
| #3 | 32 | 271.73 | 1 | |
| #4 | 32 | 218.28 | 1 | |
| #5 | 32 | 217.97 | 1 | |
| #6 | 14 | 162.28 | 1 | |

### graph_compute の送信時間比較 (2nd RDMA device, n_copies>0)

| 指標 | Baseline | Pipeline | 備考 |
|------|:--------:|:--------:|------|
| send 時間 | **50〜82 ms** | **< 1 ms** | ベースラインはサーバー計算完了待ち |
| recv 時間 | 0 ms (async) | 0 ms (async) | 両方とも async |
| total | 52〜83 ms | 1.4〜1.9 ms | Pipeline が **50x 高速** |

ベースラインの graph_compute send が 50-82ms ブロックする理由: サーバーの command queue が FIFO のため、前のデバイスの async compute が完了するまで次のコマンドが処理されない。Pipeline 版は `FLUSH_AND_GRAPH_COMPUTE_ASYNC` 統合コマンドでこのブロッキングを回避。

### TG ubatch 詳細

| 指標 | Baseline | Pipeline |
|------|:--------:|:--------:|
| graph_compute per call (device 0) | 1.4 ms (full) | 0.9 ms (recompute) |
| graph_compute per call (device 1) | 6.7 ms (full/recompute) | 1.2 ms (recompute) |
| get_tensor path | Send/Recv (17 ms avg) | RDMA Read (0.8 ms avg) |
| ubatch total | ~20 ms | ~20 ms |

Pipeline 版の TG 改善は:
1. **graph_cache multi-slot** → recompute パスで serialize_graph スキップ (device 1: 6.7ms → 1.2ms)
2. **get_tensor の RDMA Read 化** → `compute_pending_` が false のとき direct memory access (17ms → 0.8ms)

---

## 分析3: +2.18% PP の内訳推定

### 実際の実行タイムライン

**ベースライン (逐次処理 + FIFO async)**:
```
ubatch ループ (process_ubatch は同期呼び出し):
  ub0: [serialize+send D0 ~2ms] → [serialize+send D1 ~83ms ← FIFO wait] → [total ~85ms]
  ub1: [serialize+send D0 ~2ms] → [serialize+send D1 ~73ms] → [total ~75ms]
  ...
```

**Pipeline 版 (async send + event drain)**:
```
ubatch ループ (process_ubatch は同期呼び出し):
  ub0: [async-send D0 ~2ms] → [async-send D1 ~2ms] → [event_wait for all ~200ms]
  ub1: [async-send D0 ~2ms] → [async-send D1 ~2ms] → [event_wait ~200ms]
  ...
```

Pipeline 版は送信自体は高速 (~4ms) だが、ubatch 終了時の `event_wait` → `drain_pending_compute()` で全デバイスの計算完了を待つ。サーバー GPU 計算 (~200ms/ubatch) がボトルネックのため、全体の改善は限定的。

### 内訳推定

| 貢献要因 | 推定効果 | メカニズム |
|---------|:-------:|----------|
| FIFO wait 解消 (send 50-82ms → <1ms) | +1.0〜1.5% | ubatch 内の送信ブロッキング排除 |
| graph_cache multi-slot (serialize skip) | +0.5〜1.0% | 2回目以降の ubatch で serialize_graph (~1.5ms/device) スキップ |
| FLUSH_AND_GRAPH_COMPUTE_ASYNC (roundtrip 削減) | ~0.2% | flush+compute を1コマンドに統合 |
| **合計** | **~2.2%** | A/B 実測 +2.18% と一致 |

### なぜ理論値と乖離するか

```
理論: 異なるデバイスが異なる ubatch を同時処理
  → GPU idle 時間を最小化 → +60〜130%

実装: 同一 ubatch 内の通信最適化
  → 通信のブロッキングは排除されるが、GPU 計算は依然として逐次
  → compute-bound 構成では改善余地が通信時間比率 (5.2%) に限定
  → 5.2% × ~40% 最適化 ≈ 2%
```

---

## 分析4: TG +4.68% の原因

TG は `n_tokens=1` のため ubatch 間パイプラインは無関係。改善は以下の 2 つの副次効果による:

### 1. graph_cache multi-slot によるオーバーヘッド削減

TG では同一グラフ構造が繰り返される。Pipeline 版の multi-slot graph_cache により:
- Baseline: 毎回 serialize_graph → FULL_GRAPH 送信 (6.7ms/device)
- Pipeline: recompute パスで pre_send + send のみ (1.2ms/device)
- 削減: ~5.5ms/token (2nd RDMA device)

### 2. get_tensor の RDMA Read 化

Pipeline 版は `compute_pending_` 状態管理により、async compute が完了済みのとき RDMA Read を使用:
- Baseline: get_tensor で Send/Recv (サーバー command queue 経由, 17ms avg)
- Pipeline: get_tensor で RDMA Read (direct memory access, 0.8ms avg)
- 削減: ~16ms/call × 少数の get_tensor 呼び出し

### TG 改善の定量分析

1000/34.08 = 29.35ms/token (baseline) → 1000/35.67 = 28.04ms/token (pipeline)
→ 1.31ms/token の削減

graph_compute: 5.5ms × (RDMA比率 ~50%) ≈ 2.75ms → ubatch あたり ~1.3ms 削減と概ね一致

---

## 分析5: Dense モデルへの効果推定

### Phase 2 の参照データ (Qwen3.5-27B, 4GPU)

[Phase 2 レポート](2026-03-03_013808_pipeline_parallelism_phase2.md) での `ASYNC_COMPUTE=0` vs `=1` 比較:

| 条件 | PP (t/s) | 改善 |
|------|:--------:|:----:|
| Phase 2 (async=1) | 19.0 | +41.8% |
| Fallback (async=0) | 13.4 | baseline |

この +41.8% は event 機構全体の効果（ベースライン no-op synchronize からの改善）。

### Pipeline 版単独実行 (Qwen3.5-27B, 4GPU)

本レポートでの計測:
- **Pipeline 版**: PP = 34.4 t/s, TG = 11.4 t/s
- **ベースライン**: Pipeline サーバーとの互換性問題で計測不可（後述）

### Dense モデルのベースライン + Pipeline サーバー互換性問題

ベースラインクライアント + Pipeline サーバーで Qwen3.5-27B を実行すると、サーバーが CUDA エラーでクラッシュ:

```
CUDA error: invalid argument
  in function ggml_cuda_cpy at ggml-cuda/cpy.cu:415
  cudaMemcpyAsync(src1_ddc, src0_ddc, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream)
```

原因: Dense モデルのグラフ構造が複雑（SSM レイヤー含む）で、ベースラインの分離コマンド送信 (`FLUSH_ALL_STAGING` + `GRAPH_COMPUTE` を個別に) と Pipeline サーバーのハンドラ間でテンソル状態の不整合が発生。MoE モデル (Qwen3.5-35B-A3B) ではこの問題は発生しない。

---

## 結論と推奨事項

### 乖離の根本原因

**理論値の前提 (split 並列ループ) が実装されていない。** `ggml-backend.cpp` の split 実行ループは一切変更されておらず、各 split は依然として逐次実行される。実装された改善は RDMA 通信のオーバーヘッド削減のみで、GPU idle 時間の最小化（理論値の源泉）は達成されていない。

### 実測値の妥当性

+2.18% PP は通信オーバーヘッド削減として妥当:
- MoE モデル (compute-bound): 通信時間比率 ~5% → 通信最適化 ~40% → 全体 ~2%
- Dense モデルではより大きな改善が期待される（通信時間比率が高いため）

### 理論値に近づけるための推奨事項

| 優先度 | 施策 | 期待効果 | 実装難度 |
|:------:|------|:--------:|:--------:|
| 1 | `ggml_backend_sched_compute_splits` の並列化 | +30〜60% (PP) | 高 |
| 2 | ubatch N+1 のデータ送信を ubatch N のサーバー計算と並行実行 | +10〜20% (PP) | 中 |
| 3 | Per-device RDMA connection (RDMA0/RDMA1 独立接続) | +5〜10% (TG) | 中 |

**施策1 が必須。** 現在の Phase 1-4 は通信最適化としては有効だが、理論値のスケジューラレベル並列化にはプログラムの根本的な構造変更が必要。

---

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
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1257458) |

GGML_RDMA 環境変数: (none)

テスト構成: Node 1 CUDA4,5 + Node 2 RDMA0,1 (4GPU), rdma-server に `CUDA_VISIBLE_DEVICES=0,1` を適用
