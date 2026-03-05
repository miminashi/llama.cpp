# PP性能向上: パイプライン並列 vs エキスパート並列 方向性分析

- **実施日時**: 2026年3月5日 11:34
- **種別**: 分析レポート（実験なし、既存データの統合分析）
- **参照レポート**:
  - [Pipeline PP 改善調査](report/2026-03-05_060119_pipeline_pp_improvement_investigation.md)
  - [PP プロファイリング Qwen3.5 6GPU](report/2026-03-02_195335_pp_profiling_qwen35_optimization.md)
  - [Expert Parallelism Phase 10](report/2026-02-24_043136_expert_parallelism_phase10_deferred_set_tensor.md)
  - [Expert Parallelism Phase 11](report/2026-02-25_043933_expert_parallelism_phase11_parallel_dispatch.md)
  - [Send Ring Buffer ベンチマーク](report/2026-03-04_004316_send_ring_buffer_benchmark.md)
  - [Pipeline Complete 8GPU Retest](report/2026-03-05_031342_pipeline_complete_8gpu_qwen35_retest.md)

## 前提・目的

GLM-4.7 11GPU (7C+4R) の pp128 = 24.17 t/s を改善するための方向性を決定する。候補は3つ:

1. **CUDA/RDMA overlap** — ggml-backend.cpp の split loop 改修による CUDA/RDMA 並行実行
2. **エキスパート並列 (EP) 再挑戦** — MoE モデルでのデバイス間エキスパート分散
3. **30.6→24.3 退行調査** — 原因不明の 26% 退行の解明

---

## 1. 現状の PP ボトルネック

### GLM-4.7 11GPU (7 CUDA + 4 RDMA)

プロファイリング結果 (pipeline-complete, pp128):

| フェーズ | 時間 (ms) | 割合 |
|---------|----------|------|
| CUDA デバイス計算 (7台) | 3,366 | 64.6% |
| RDMA デバイス計算 (4台, 逐次) | 1,846 | 35.4% |
| graph_compute dispatch | <3 | <0.1% |
| **合計** | **5,212** | 100% |

- 各 RDMA デバイスの計算: 294–558 ms (逐次実行、合計 1,846 ms)
- Layer-split の依存関係により **デバイス間並列化は不可能** (D0→D1→D2→D3)
- PIPELINE=1 はブロッキング箇所を移動するだけで合計時間は同一

### Qwen3.5 6GPU (2 CUDA + 4 RDMA)

| カテゴリ | 時間 (ms) | 割合 |
|---------|----------|------|
| サーバー側 GPU 計算 (flush 待ち) | 321.64 | **94.8%** |
| RDMA 通信 (serialize + send) | 11.29 | 3.3% |
| その他 (drain, snapshot) | 6.45 | 1.9% |

Qwen3.5 MoE は完全に計算バウンドであり、RDMA 最適化の余地は事実上ない。

---

## 2. エキスパート並列 (EP) の評価

### 実績

11フェーズにわたる実装で機能的に動作し、正しい推論出力を確認済み。

| 指標 | EP=11 | ベースライン (Layer-split) | 比率 |
|------|:-----:|:------------------------:|:----:|
| pp (t/s) | 0.3 | 24.17 | **99% 劣化** |
| tg (t/s) | 0.8 | 8.45 | **91% 劣化** |

Phase 7 (USAGE_WEIGHTS 修正) が最大の改善を記録: pp +200%, tg +3900%。しかし絶対値は依然として実用水準に程遠い。

### 構造的問題: graph_compute 往復回数

| 方式 | 1トークンあたりの往復回数 | 理由 |
|------|:---:|------|
| Layer-split (PP) | ~30 | デバイス数 × split 数 |
| EP (現状) | ~366 | 62 MoE レイヤー × エキスパート数 × デバイス数 |

各往復 ~1.5ms のため、EP では 1 トークンあたり ~550ms が通信オーバーヘッドとなり、tg 上限は ~1.8 t/s。

### GDR レースコンディション (Phase 11 で発見)

並列ディスパッチ時に `conn->recv()` でブロック中のサーバーに対し、クライアント RNIC が GDR 経由で同じ GPU メモリに RDMA Write を実行 → メモリ破損 (`illegal memory access`)。

- GDR 無効 (`GGML_RDMA_GDR_BUDGET_GB=0`) なら並列化可能だが、往復回数の問題は残る
- 根本解決には per-device write fencing が必要

### EP 改善のロードマップ

| ステージ | 施策 | 期待される tg | 実装コスト |
|---------|------|:-----------:|:---------:|
| (a) GDR レース解消 | per-device write fencing | ~3.6 t/s (57%劣化) | 中 |
| (b) MoE 一括コマンド | 62 MoE レイヤーを 1 コマンドに統合 | ~8 t/s (ベースライン同等) | **高** (プロトコル大幅変更) |
| (c) サーバー側ルーティング | エキスパート選択をサーバーで実行 | >8 t/s | **最大** |

(a) だけでは実用水準に到達しない。(b) でようやくベースライン同等だが、RPC プロトコルの大幅な変更が必要。

### EP の適用範囲

EP の恩恵は **MoE モデルかつ VRAM が不足する場合** に限定される。Qwen3.5 はすでに Layer-split で pp 200 t/s を達成しており EP は不要。GLM-4.7 は dense モデルのため EP の対象外。EP が有効になるのは、将来的に大規模 MoE モデル (例: GLM-4.7 Q4_K_M) を限られた GPU 数で動かす場合に限られる。

---

## 3. CUDA/RDMA Overlap の評価

### 原理

現在の split loop は CUDA splits と RDMA splits を逐次実行している:

```
[CUDA splits: 3366ms] → [RDMA splits: 1846ms] = 5212ms
```

CUDA/RDMA overlap は ubatch 間でこれを並行化する:

```
ubatch N:   [CUDA splits: 3366ms]
ubatch N+1:            [RDMA splits: 1846ms]  ← CUDA と並行
                       ────────────────────
                       合計: max(3366, 1846) ≈ 3366ms
```

理論的改善: 5212ms → 3366ms = **pp128 ≈ 38 t/s (+57%)**

### 既存インフラの活用

| コンポーネント | 状態 | 説明 |
|-------------|:----:|------|
| per-copy context buffers | ✅ 実装済み | ubatch ごとに独立した tensor struct バッファ |
| split snapshots | ✅ 実装済み | 各 ubatch の split 構造を保存 |
| per-device connections | ✅ 実装済み | デバイスごとの独立接続 |
| split loop 改修 | ❌ 未実装 | **核心部分**: ggml-backend.cpp で CUDA/RDMA を並行ディスパッチ |

### リスク

| リスク | 影響 | 緩和策 |
|-------|------|--------|
| ggml-backend.cpp の深い変更 | スレッド安全性、upstream との乖離 | RDMA バックエンド側でのフック |
| TG には効果なし | ubatch が 1 つのため overlap 不可 | TG 改善は別施策 |
| テストが GLM-4.7 限定 | テスト時間が長い | Qwen3.5 は計算バウンドで効果なし |

### パイプライン並列の既存成果

| 構成 | モデル | PP 改善 | TG 改善 |
|------|-------|:------:|:------:|
| PIPELINE + per-device | Qwen3.5 4GPU | **+25.9%** | +1.8% |
| PIPELINE + per-device | GLM-4.7 11GPU | -0.33% | +1.17% |
| Ring buffer のみ | GLM-4.7 11GPU | 0% | 0% |
| PIPELINE のみ | Qwen3.5 4GPU | 0% | +0.82% |

GLM-4.7 で PIPELINE が効かない原因は Layer-split のデバイス依存関係。CUDA/RDMA overlap はデバイス間ではなく **バックエンド種別間** の並列化であり、この制約を回避できる。

---

## 4. 30.6→24.3 退行調査

### 現状

| 時点 | pp128 (t/s) | 備考 |
|------|:-----------:|------|
| merge/validated | ~30.6 | always-signal fix 前 |
| feature/rdma-backend | ~24.3 | always-signal fix 後 |
| + Ring buffer | 24.17 | 回復せず |
| + PIPELINE=1 | 24.17 | 回復せず |
| + GDR 無効 | 24.17 | 回復せず |

### 調査済みの仮説

| 仮説 | 結果 |
|------|------|
| Send selective signaling の喪失 | ❌ Ring buffer で回復しても変化なし |
| GDR sync overhead | ❌ GDR 無効でも変化なし |
| RDMA Write always-signal | ❌ GDR 有効時は RDMA Write 不使用 |
| Pipeline wait のブロッキング | ❌ PIPELINE=1 でも変化なし |

### 残る可能性

1. `event_wait` / `drain_pending_compute` の実装差異
2. always-signal fix に伴うコードパスの副次的変更
3. merge/validated 時点との測定条件差異

### リスク評価

既に主要な仮説を検証・棄却済み。原因特定の見通しが立っておらず、工数が読めない。ただし、もし解明できれば **+26% の即時回復** という高いリターンがある。

---

## 5. 総合比較

| 項目 | EP 再挑戦 | CUDA/RDMA overlap | 退行調査 |
|------|:--------:|:-----------------:|:-------:|
| **理論的 PP 改善** | -99% → -57% (a のみ) | **+57%** | **+26%** |
| **実装難易度** | 高 (プロトコル変更) | 中 (backend.cpp 改修) | 不明 |
| **既存インフラ** | EP 実装済み、GDR レースが壁 | split snapshot 等の基盤あり | — |
| **テスト容易性** | Qwen3.5 可 (高速) | GLM-4.7 必須 (低速) | GLM-4.7 必須 |
| **TG 効果** | 劣化 | なし | +26% (全体的) |
| **適用範囲** | MoE モデルのみ | 全 layer-split モデル | 全モデル |
| **実用化まで** | (b) プロトコル変更が必須 | split loop 改修のみ | 原因特定次第 |

---

## 6. 推奨

### 最優先: CUDA/RDMA overlap

**理由**:
1. **理論的改善幅が最大** (+57%) — 24.17 → 38 t/s の可能性
2. **既存インフラが活用可能** — per-copy context buffers, split snapshots, per-device connections はすべて実装済み
3. **全 layer-split モデルに汎用適用** — GLM-4.7 だけでなく将来の大規模モデルにも有効
4. **実装範囲が比較的限定的** — ggml-backend.cpp の split loop 改修が核心

### 副次推奨: 退行調査の並行検討

CUDA/RDMA overlap とは独立した問題のため、もし原因が見つかれば **相乗効果** で 38 × 1.26 ≈ 48 t/s も理論的に可能。ただし原因特定の見通しが不透明なため、メインの工数は CUDA/RDMA overlap に集中すべき。

### EP について

EP は構造的問題 (往復回数 366 回/トークン) が根本的であり、GDR レース解消 (a) だけでは実用水準 (tg ~3.6 t/s) に達しない。MoE 一括コマンド (b) でようやくベースライン同等だが、プロトコル大幅変更が必要。**CUDA/RDMA overlap 成功後の次フェーズ** として、(b) のアプローチを検討するのが合理的。

---

## まとめ

```
優先度: CUDA/RDMA overlap >> 退行調査 >> EP 再挑戦

CUDA/RDMA overlap:
  - 理論値: pp128 ≈ 38 t/s (+57%)
  - 必要作業: ggml-backend.cpp split loop で CUDA/RDMA 並行ディスパッチ
  - 基盤: per-copy buffers ✅, split snapshots ✅, per-device conn ✅

退行調査:
  - 理論値: pp128 ≈ 30.6 t/s (+26%)
  - 状況: 主要仮説は棄却済み、原因不明
  - CUDA/RDMA overlap と独立、並行検討可

EP 再挑戦:
  - 現状: 99% 劣化、GDR レースと往復回数が二重の壁
  - MoE 一括コマンドが必要だがプロトコル変更大
  - CUDA/RDMA overlap 成功後の次フェーズとして検討
```
