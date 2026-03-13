# Qwen3.5 35B/122B PP/TG 性能向上ロードマップ

- **作成日時**: 2026年3月12日 00:48

## 前提・目的

前回の探求レポート (2026-03-08) 以降、Gate+Up fusion と GPU-native sort が実装・マージされ、PP 性能が大幅に改善された (+24.8% ~ +34.6%)。また、F16 GGUF が実測で却下され (-15% pp, -33% tg)、前回の最有力施策 (Tier 0-A) が無効化された。TG 最適化 (combined-send, pre-post recv, doorbell batching) も既にマージ済みだが、RDMA は TG で RPC に -3.5% ~ -6.1% 劣位のまま。

本ロードマップは、これらの変化を踏まえて残りの改善余地を体系的に評価し、PP/TG 両方の性能向上計画を策定するものである。

### 参照レポート
- [前回探求レポート](2026-03-08_055421_pp_improvement_exploration_team.md)
- [Gate+Up fused モデルベンチマーク](2026-03-09_180920_122b_gate_up_fused_model.md)
- [unfused vs fused ベンチマーク](2026-03-09_034923_unfused_vs_fused_benchmark.md)
- [RDMA vs RPC ベンチマーク](2026-03-10_113211_rdma_vs_rpc_upstream_benchmark.md)

## 現在のベースライン (2026-03-11, feature/rdma-backend HEAD)

| モデル | 構成 | pp128 | pp512 | pp2048 | pp16384 | tg32 |
|--------|------|:-----:|:-----:|:------:|:-------:|:----:|
| **35B** (Q4_K_M fused) | 4GPU (2C+2R) | 273.8 | 457.8 | 515.6 | — | 36.1 |
| **122B** (Q4_K_M fused) | 11GPU (7C+4R) | 117.1 | — | — | 170.2 | 18.1 |

### RDMA vs RPC 比較

| モデル | pp128 | pp2048/pp16384 | tg32 |
|--------|:-----:|:--------------:|:----:|
| 35B | RDMA +8.3% | RDMA +38.9% | **RDMA -6.1%** |
| 122B | RDMA +4.9% | RDMA +18.1% | **RDMA -3.5%** |

## 実装済み施策の整理

### feature/rdma-backend にマージ済み
1. **Gate+Up fusion**: 35B pp +18-21%, 122B pp +6.6-7.6%
2. **GPU-native mul_mat_id sorting**: pp +3.2-8.3%
3. **TopK MoE scratch copy fusion**: pp2048 RDMA +11.1% 回復
4. **Combined-send** (header+data 一括): RNR NAK 削減
5. **Pre-post recv**: サーバー側事前バッファ
6. **Doorbell batching + max_inline 512**: PCIe doorbell 半減
7. **Pipeline parallelism + Per-device connections**: 環境変数トグル
8. **Send ring buffer**: 効果中立
9. **Deferred copy**: IB round-trip 排除
10. **GDR budget system**: MTT キャッシュオーバーフロー回避

### 未マージの実装済みワークツリー
- **CUDA/RDMA Overlap** (`.worktree/cuda-rdma-overlap`): pp2048 +28.7% (GLM-4.7), Qwen3.5 MoE では中立

### 却下された施策
- **F16 GGUF**: 実測 pp -15%, tg -33% (メモリ帯域ボトルネック)
- **MMQ 強制**: P100 dp4a 非対応, -48%
- **CUDA Graph / Triton / Tensor Core**: P100 アーキテクチャ制約

## ロードマップ

### Phase 1: 低リスク・即実行 (1-3日)

#### 1-1. CUDA/RDMA Overlap のマージ
- **内容**: `feature/cuda-rdma-overlap` (commit `64a67bf25`) を feature/rdma-backend にマージ
- **効果**: Qwen3.5 MoE では pp 中立だが退行もなし。`GGML_RDMA_CUDA_OVERLAP=1` の環境変数ゲート付き (デフォルト無効)
- **理由**: 16GPU 拡張時 (RDMA デバイス 4→8) で overlap 余地が拡大。マージしておくことで将来の利得基盤になる
- **検証**: 122B 11GPU (7C+4R) で pp128/pp16384/tg32 のデグレテスト

#### 1-2. topk_moe scratch copy fusion のマージ確認
- **内容**: commit `94b58c301` が feature/rdma-backend に含まれているか確認
- **効果**: RDMA パスの pp2048 で fusion 無効化を回避 (+11.1% 回復)
- **状態**: ✅ 確認済み — feature/rdma-backend に含まれている

#### 1-3. 現状プロファイリング (Gate+Up fusion 後)
- **内容**: Gate+Up fusion + GPU-native sort 適用後の nvprof / CUDA profiling を実施
- **目的**: dequant 比率、expert loop 時間、sync オーバーヘッドの現在値を取得
- **対象**: Qwen3.5-35B 4GPU (2C+2R) の pp128, pp512, pp2048

### Phase 2: 中リスク・PP改善 (1-2週)

#### 2-1. CUDA 12.4+ アップグレード + cublasSgemmGroupedBatched
- **内容**: CUDA toolkit を 12.0 → 12.6 LTS にアップグレードし、`cublasSgemmGroupedBatched` を `mul_mat_id` の expert ループに適用
- **効果**: pp +3-8% (expert GEMM のカーネルラウンチオーバーヘッド削減)
- **前提**: Phase 1-3 のプロファイリングで expert ループのオーバーヘッドが有意であることを確認
- **リスク**: CUDA アップグレードでドライバ更新 (535→550+) が必要になる可能性。nvidia-peermem との互換性要確認

#### 2-2. Delta-net 層の最適化調査
- **内容**: Qwen3.5 の recurrent (delta-net) 層のプロファイリングと最適化候補の特定
- **前提**: Phase 1-3 のプロファイリングで delta-net 層の計算比率を確認
- **候補**: `cublasHgemmStridedBatched` で小行列 GEMM をバッチ化

### Phase 3: TG 改善 (2-3週)

#### 3-1. TG レイテンシの詳細プロファイリング
- **内容**: TG の hot path のレイテンシ分解 (各 RDMA verb の所要時間、CQ polling 回数、サーバー応答時間)
- **目的**: RDMA の TG 劣位 (-3.5% ~ -6.1%) の正確な原因を特定

#### 3-2. TG graph_compute コマンドの最適化
- **候補**: Server-side TG pipeline、コマンドバッチング、graph キャッシュ改善
- **効果**: tg +2-5% (推定)

### Phase 4: 高リスク・追加改善 (1ヶ月+)

#### 4-1. GDR per-device write fencing
- **内容**: GDR race condition を解決し、parallel dispatch + GDR の共存を実現
- **効果**: expert parallelism の復活で pp +5-15%

#### 4-2. Expert multi-stream 並列実行
- **内容**: 同一 GPU 上の複数 expert を異なる CUDA stream で並列実行
- **効果**: pp +5-15%

#### 4-3. ggml_backend_sched の並列 split 実行
- **内容**: upstream の `ggml_backend_sched_compute_splits()` を非同期化
- **効果**: TG +10-20%

## 施策間の依存関係

```
Phase 1-1 (Overlap) ─────────────────── 独立
Phase 1-2 (scratch copy) ────────────── 独立 (✅完了)
Phase 1-3 (プロファイリング) ─────────── Phase 2-1, 2-2, 3-1 の前提条件
Phase 2-1 (CUDA 12.6 + Grouped GEMM) ─ Phase 1-3 に依存
Phase 2-2 (Delta-net) ────────────────── Phase 1-3 に依存
Phase 3-1 (TG プロファイリング) ──────── 独立
Phase 3-2 (TG 最適化) ────────────────── Phase 3-1 に依存
Phase 4-1 (GDR fencing) ──────────────── Phase 1-1 と組み合わせで効果最大
Phase 4-2 (Multi-stream) ─────────────── 独立
Phase 4-3 (sched 並列化) ─────────────── Phase 4-2 の上位互換
```

## 理論的最大改善 (全Phase適用)

### PP (122B 11GPU pp16384, ベース 170.2 t/s)
```
Phase 1 (Overlap)          : +0-3%
Phase 2-1 (Grouped GEMM)   : +3-8%
Phase 2-2 (Delta-net)      : +4-9%
Phase 4-1 (GDR + parallel) : +5-10%
Phase 4-2 (Multi-stream)   : +10-16%
累積 (乗算): +24-55% → ~210-264 t/s
現実的見込み: +15-30% → ~196-221 t/s
```

### TG (122B 11GPU tg32, ベース 18.1 t/s → RPC 18.9 t/s)
```
Phase 3-2 (コマンド最適化) : +2-5%
Phase 4-1 (GDR fencing)    : +1-3%
Phase 4-3 (sched 並列化)   : +5-10%
累積 (乗算): +8-19% → ~19.5-21.5 t/s
目標: RPC 比パリティ以上 (18.9 t/s)
```

## 16GPU 拡張時の考慮

| 施策 | 16GPU への影響 |
|------|:---:|
| Overlap (1-1) | **効果拡大**: RDMA デバイス 4→8 で overlap 余地 2x |
| Grouped GEMM (2-1) | 中立: GPU ローカル最適化 |
| TG コマンドバッチ (3-2) | **効果拡大**: 8 RDMA デバイスで RTT 削減効果 2x |
| GDR fencing (4-1) | **効果拡大**: 8 デバイスの並列性で最大効果 |
| sched 並列化 (4-3) | **効果拡大**: 16 split の並列化 |

## 前回レポートからの変更点

| 項目 | 前回 (2026-03-08) | 今回 |
|------|:---:|:---:|
| F16 GGUF (Tier 0-A) | 最有力施策 | **却下** (実測 pp -15%, tg -33%) |
| Gate+Up fusion | 提案段階 | **マージ済み** (35B pp +18-21%) |
| GPU-native sort | 提案段階 | **マージ済み** (pp +3.2-8.3%) |
| TopK scratch copy fusion | 提案段階 | **マージ済み** (pp2048 +11.1%) |
| Combined-send/Pre-post recv/Doorbell | 提案段階 | **マージ済み** (TG 最適化) |
| CUDA 12.6 Grouped GEMM | 未検討 | **新規追加** (Phase 2-1) |
| Delta-net 最適化 | 未検討 | **新規追加** (Phase 2-2) |
| TG プロファイリング | 未検討 | **新規追加** (Phase 3-1) |

## 調査体制

本ロードマップは以下の 6 エージェントの調査結果を統合して策定した:

### 分析エージェント
- **fusion-analyst**: Gate+Up fusion の効果分析と残存最適化余地の評価
- **bottleneck-analyst**: 現在のボトルネック分析 (PP/TG 各パスの時間分解)
- **strategy-researcher**: CUDA アーキテクチャ制約 (P100) 下での最適化戦略調査

### 計画エージェント
- **pp-optimizer**: PP 改善施策の優先順位付けと実装計画
- **tg-rdma-optimizer**: TG 改善施策の評価と RDMA レイテンシ分析
- **strategy-integrator**: 全施策の統合、依存関係整理、ロードマップ策定
