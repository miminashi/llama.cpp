# RDMA バックエンド 振り返りレポート v6

- **実施日時**: 2026年2月21日 03:31
- **ワークツリー**: `feature/rdma-backend` (メインリポジトリ)
- **前回レトロスペクティブ**: [2026-02-19_190406_rdma_backend_retrospective_v5.md](2026-02-19_190406_rdma_backend_retrospective_v5.md)

---

## 前回レトロスペクティブ (v5) からの主要変更点

v5 (02/19 19:04) 以降の約2日間で **6 本のレポート** (#83〜#88) が追加され、コード変更 1 件 (Selective Signaling マージ) を実施した。

- **検証**: 3件 (row split 11GPU, merge-validated A/B, llama-cli ABAB 追試)
- **実装**: 1件 (server-side pipeline)
- **ベンチマーク**: 1件 (server-side pipeline A/B)
- **マージ**: 1件 (selective signaling を feature/rdma-backend に cherry-pick)

---

## v5→v6 統合サマリーテーブル (全 6 項目)

| # | 内容 | レポート | ワークツリー | 効果 | マージ状況 |
|---|------|---------|------------|------|-----------|
| 1 | Row split 11GPU RDMA ベンチマーク | [#83](2026-02-19_193000_row_split_11gpu_rdma_benchmark.md) | `rdma-row-split` | layer split が全条件で優位 (tg +13%) | - (検証) |
| 2 | Selective signaling + server-push A/B 検証 | [#84](2026-02-20_132916_merge_validated_improvements_benchmark.md) | `merge-validated` | SS: pp +26%, tg +0.4% (ns) / Per-dev: pp -1.35%, tg +1.76% | マージ判断材料 |
| 3 | Server-side pipeline 実装 | [#85](2026-02-20_203618_server_pipeline_implementation.md) | `server-pipeline` | (実装のみ) | 未マージ |
| 4 | llama-cli ABAB 追試テスト | [#86](2026-02-20_212013_llama_cli_abab_replication.md) | `merge-validated` | 短プロンプトでは改善再現されず (検証) | - |
| 5 | Server-side pipeline A/B ベンチマーク | [#87](2026-02-20_215838_server_pipeline_benchmark.md) | `server-pipeline` | pp -0.21% (有意), tg +0.55% (非有意) | 未マージ |
| 6 | Selective signaling マージ確認 | [#88](2026-02-21_030543_selective_signaling_merge.md) | `feature/rdma-backend` | pp 30.15 t/s, tg 7.75 t/s | **マージ済み** |

- **マージ状況の凡例**:
  - `-`: コード変更なし (検証/調査のみ)
  - `マージ済み`: コード変更が feature/rdma-backend に取り込まれている
  - `未マージ`: ワークツリーにコード変更があるが feature/rdma-backend には未取り込み

---

## v5→v6 の重要な知見

### 知見1: Selective Signaling マージの意義

Selective signaling を `feature/rdma-backend` にマージした (#88)。これは v5 以降で唯一のコード変更であり、以下の結果を確認:

| 指標 | マージ前 (ベースライン) | マージ後 | 変化 |
|------|:---:|:---:|:---:|
| pp128 | 23.35 t/s | **30.15 t/s** | **+29.1%** |
| tg32 | 7.75 t/s | **7.75 t/s** | ±0% |

pp128 の +29% は、GLM-4.7 MoE の Prompt 処理で数百回の RDMA Write が発生するため、CQ ポーリング削減の効果が劇的に現れたもの。tg32 は 1 トークンごとの逐次処理で Write 回数が少なく、効果がほぼゼロ。

### 知見2: 通信最適化の到達点確認

tg32 ≈ 128ms/token のうち GPU 計算が 93.8% (~120ms)、通信オーバーヘッドが 6.2% (~8ms)。通信層の残余はプロトコル残余 ~0.5-1ms のみで、これ以上の通信最適化は理論的上限に近い。

v5 で指摘した「RDMA バックエンド側でできる改善は限界に達した」という結論は、v6 の追加検証で再確認された:

- **Server-side pipeline** (#87): tg +0.55% (非有意) — 期待 +1% に対し半分で、統計的にも不確実
- **llama-cli 追試** (#86): llama-bench で確認された改善が短プロンプトでは再現されず — 改善が特定条件依存であることを示唆

### 知見3: Server-push と per-device connections のトレードオフ確定

#84 の A/B テストで、per-device connections + server-push は **pp128 で -1.35% の退行、tg32 で +1.76% の改善** というトレードオフが確定した。デフォルトは selective signaling のみ (shared connection) とし、tg 最重視時のみ `GGML_RDMA_PER_DEVICE_CONN=1` を opt-in で使用する方針。

---

## 性能サマリー更新

### GLM-4.7 RDMA vs RPC 比較 (更新版)

| バックエンド | pp128 (t/s) | tg32 (t/s) | 備考 |
|-------------|:------------:|:----------:|------|
| **RDMA (v6, SS マージ後)** | **30.15** | **7.75** | Selective signaling ON (デフォルト) |
| RDMA (v5, server-push) | 7.4 | 7.76 | Server-Push ON (ワークツリー) |
| RDMA (v4, fprintf 修正後) | 7.4 | 7.4〜7.5 | — |
| RDMA (v3, GDR budget 12GB) | 6.4 | 6.8 (ばらつき大) | 二峰性あり |
| RDMA (GDR 無効) | 6.4 | 6.0 | — |
| RPC (TCP) | 5.4 | 7.5 | — |

- **v6 時点**: RDMA は Prompt で RPC 比 **+458%** (30.15 vs 5.4)、Generation で RPC 比 **+3.3%** (7.75 vs 7.5)
- v5 まで pp128 は 7.4 t/s (selective signaling 未マージ) だったが、v6 で **30.15 t/s** に跳躍

### Step ごとのベストスコア (更新版)

| Step | 構成 | モデル | pp (t/s) | tg (t/s) | ローカル比 (tg) |
|------|------|--------|:--------:|:--------:|:--------------:|
| Step 1 完了 | RDMA 1+1 | qwen2.5-0.5b | — | 148.5 | 75.7% |
| Step 2 完了 | 11GPU (7C+4R) | gpt-oss-120b | 3〜11 | 0.7〜1.1 | 1.5〜2.3% |
| Step 3 完了 | RDMA 1+1 | gpt-oss-20b | 61.09 | 58.47 | **91.0%** |
| Step 4 完了 (GDR) | RDMA 1+1 | gpt-oss-20b | **403.69** | 59.52 | **92.6%** |
| Step 4 完了 (クラスタ) | 11GPU (7C+4R) | gpt-oss-120b | 201.02 | 36.92 | 84.7% |
| Step 5 (v4 時点) | 11GPU (7C+4R) | GLM-4.7 IQ2_M | 7.4 | 7.4〜7.5 | — |
| Step 5 (v5 時点) | 11GPU (7C+4R) | GLM-4.7 IQ2_M | 7.4 | 7.65〜7.76 | — |
| **Step 5 (v6 時点)** | **11GPU (7C+4R)** | **GLM-4.7 IQ2_M** | **30.15** | **7.75** | **—** |

---

## 今後の最適化提言カタログ

全レポート (#1〜#88) から抽出した提言を 4 カテゴリに分類する。これが本レポートの核心であり、今後の開発方針を体系的に整理する。

### A. アーキテクチャレベル変更 (llama.cpp upstream)

#### A-1: `ggml_backend_sched` 逐次処理の改善

- **概要**: `ggml_backend_sched_compute_splits()` が各デバイスに対して `graph_compute → get_tensor` を逐次実行する構造を改善する。Two-phase loop、server-side pipeline 等の形態が考えられるが、layer split のデータ依存 (レイヤー N+1 の入力がレイヤー N の出力に依存) により、単純な並列化は不可能
- **期待効果**: 理論上は +50-100% tg だが、データ依存の壁により **実現困難**
- **実装複雑度**: **高** (upstream API 変更が必要)
- **前提条件**: llama.cpp upstream の設計承認
- **出典**: [#74](2026-02-19_090721_ggml_backend_sched_bottleneck_analysis.md), [#70](2026-02-18_070912_generation_optimization_deep_investigation.md), [#71](2026-02-18_120000_unimplemented_optimizations_inventory.md)
- **結論**: v5 で検証した Two-phase loop、parallel dispatch、server-side pipeline はいずれもこの壁に衝突して効果なし。根本的なスケジューラ再設計 (例: row split + EP 対応) が必要

#### A-2: 計算グラフ最適化

- **概要**: tg32 ≈ 128ms/token のうち 93.8% が GPU 計算。通信最適化が限界に到達した今、次の改善は計算側にある。カーネル融合 (Flash Attention、GQA 最適化)、レイヤー枝刈り、量子化改善等
- **期待効果**: 計算側 10% 改善 → 全体 9.4% 改善
- **実装複雑度**: **高** (upstream 機能依存)
- **前提条件**: llama.cpp upstream の最適化進展
- **出典**: [#88](2026-02-21_030543_selective_signaling_merge.md), [#74](2026-02-19_090721_ggml_backend_sched_bottleneck_analysis.md)

#### A-3: RDMA event (`event_record`/`event_wait`) 実装

- **概要**: 現在 RDMA バックエンドは `event_record`/`event_wait` が NULL で、常に `ggml_backend_synchronize()` にフォールバック。イベントベースの細粒度同期を実装することで、prompt バッチ処理 (`n_copies > 1`) でのパイプライン化が可能になる
- **期待効果**: pp +30-50%、tg ±0% (バッチ=1 では効果なし)
- **実装複雑度**: **中**
- **前提条件**: なし
- **出典**: [#74](2026-02-19_090721_ggml_backend_sched_bottleneck_analysis.md)

### B. 並列化戦略の変更

#### B-1: Expert Parallelism (EP)

- **概要**: GLM-4.7 の 160 Expert を GPU に分散し、トークンごとの Expert 選択に基づいてルーティング。All-to-All 通信で Expert 間のトークン転送を実現
- **期待効果**: 14-20 t/s (現 7.75 t/s から **+80-158%**)
- **実装複雑度**: **高** (3-5 週間)
  - Expert パーティショニングロジック
  - All-to-All 通信 (ggml 未実装)
  - `build_moe_ffn()` 大規模リファクタ
  - `MUL_MAT_ID` 分散バージョン
- **前提条件**: なし (基盤的な変更)
- **出典**: [#58](2026-02-12_070600_hybrid_parallelism_research.md)
- **備考**: GLM-4.7 は トークンあたり 8/160 Expert のみ使用 (~95% スパース性) → EP による VRAM 効率が非常に高い

#### B-2: TP=2 + PP=8 ハイブリッド並列化

- **概要**: ノード内 2GPU でテンソル並列 (TP=2)、ノード間でパイプライン並列 (PP)。P100 の PCIe P2P を利用した AllReduce
- **期待効果**: 12-13 t/s (現 7.75 t/s から **+55-68%**)
- **実装複雑度**: **中〜高** (2-4 週間)
- **前提条件**: upstream PR #19378 の設計参照 (ただし MoE 未対応)
- **出典**: [#58](2026-02-12_070600_hybrid_parallelism_research.md)
- **備考**: PR #19378 は MoE を明示的に非サポート。FFN 部分には EP が別途必要

#### B-3: Speculative Decoding

- **概要**: ドラフトモデルまたは N-gram ベースの投機的デコーディング
- **期待効果**: MoE では **限定的〜逆効果**
  - N-gram: 受理率 4-10.6%、tg -2〜-7.8% (**退行**)
  - Draft model (qwen2.5-0.5b): 受理率 50-60%、RDMA tg +3.9% (ただし高分散)
- **実装複雑度**: **低** (既存機能)
- **出典**: [#59](2026-02-12_134623_ngram_speculative_decoding_test.md), [#60](2026-02-12_160000_draft_speculative_decoding_test.md)
- **結論**: MoE の Expert ルーティングが確率的であるため、投機的デコーディングの受理率が低い。**非推奨**

### C. ハードウェア変更

#### C-1: 16GPU 拡張 (8+8)

- **概要**: 1号機に 1 枚、2号機に 4 枚の P100 を追加し、8+8 = 16GPU 構成にする
- **期待効果**: GLM-4.7 Q4_K_M (~195 GB) が 16GPU (256 GB VRAM) に収容可能に。スケーリングによる速度向上
- **実装複雑度**: **低** (ハードウェア追加 + 設定変更)
- **前提条件**: 物理 GPU 追加
- **出典**: [#61](2026-02-13_044038_rdma_backend_comprehensive_review.md), CLAUDE.md
- **備考**: Step 5 の最終目標。11GPU 事前検証は完了済み

#### C-2: ConnectX-6+ RNIC アップグレード

- **概要**: ConnectX-4 の MTT キャッシュ制限 (~10-16 GB) を解消し、per-device connections + deferred copy の両立を可能にする
- **期待効果**: per-device connections のデフォルト有効化 → tg +3-5%。Cross-connection deferred copy も実現可能
- **実装複雑度**: **低** (ハードウェア交換 + ソフトウェアは既存)
- **前提条件**: ConnectX-6+ ハードウェア調達
- **出典**: [#70](2026-02-18_070912_generation_optimization_deep_investigation.md), [#72](2026-02-18_130000_shared_pd_per_device_connection.md)

#### C-3: Cross-connection deferred copy

- **概要**: per-device connections 間で PD を共有し、deferred copy を接続間でも有効化する。現在は per-device 有効時に deferred copy が無効化されるため -1.45% の退行が発生
- **期待効果**: +3-5% (deferred copy 回復 +1.45% + server-push 並列化 +1-3%)
- **実装複雑度**: **中〜高**
  - `deferred_copy_list` のキーを `conn` から `pd` に変更
  - 接続間 deferred copy の実行パス追加
- **前提条件**: ConnectX-6+ (ConnectX-4 でも PD 共有は可能だが MTT 制限で実用的でない)
- **出典**: [#74](2026-02-19_090721_ggml_backend_sched_bottleneck_analysis.md)

### D. Opt-in 機能の再評価 (条件付き)

#### D-1: Per-device connections + server-push (`GGML_RDMA_PER_DEVICE_CONN=1`)

- **概要**: デバイスごとの独立 RDMA 接続 + サーバー主導 RDMA Write。現在は opt-in
- **効果**: pp128 -1.35% (退行) / tg32 +1.76% (改善) のトレードオフ
- **再評価条件**: ConnectX-6+ アップグレード時に MTT キャッシュ制限が解消されれば、pp128 退行が縮小〜解消される可能性
- **出典**: [#84](2026-02-20_132916_merge_validated_improvements_benchmark.md)

#### D-2: Server-side pipeline

- **概要**: D0 即送信、D1-D3 バッチ化による IB send/recv オーバーヘッド削減
- **効果**: pp128 -0.21% (有意) / tg32 +0.55% (非有意)
- **再評価条件**: n=10 以上のサンプルサイズで再検証。Cohen's d = 0.78 (中〜大効果) が真の効果であれば n=10 で有意になる可能性がある
- **出典**: [#87](2026-02-20_215838_server_pipeline_benchmark.md)

### 提言の優先度マトリクス

| 優先度 | 提言 | 期待効果 (tg) | 実装複雑度 | 前提条件 |
|:------:|------|:------------:|:----------:|---------|
| **1** | C-1: 16GPU 拡張 | スケーリング | 低 | GPU 追加 |
| **2** | B-1: Expert Parallelism | +80-158% | 高 | なし |
| **3** | B-2: TP=2 + PP | +55-68% | 中〜高 | EP と補完的 |
| 4 | C-2: ConnectX-6 | +3-5% | 低 | ハードウェア |
| 5 | C-3: Cross-conn deferred copy | +3-5% | 中〜高 | ConnectX-6 |
| 6 | A-3: RDMA event | pp +30-50% | 中 | なし |
| 7 | D-2: Pipeline 再検証 | +0.55%? | 低 | n=10 テスト |
| — | A-1: sched 改善 | 理論上大 | 高 | 再設計必要 |
| — | A-2: 計算グラフ | 計算側改善 | 高 | upstream 依存 |
| ✕ | B-3: Speculative decoding | ±0% | — | MoE 非推奨 |

---

## コード統計 (更新版)

### ファイル別行数

| ファイル | v5 行数 | v6 行数 | 差分 | 役割 |
|---------|-------:|-------:|----:|------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 3,735 | 3,745 | **+10** | メイン実装 |
| `ggml/src/ggml-rdma/rdma-transport.cpp` | 968 | 978 | **+10** | RDMA 接続管理 |
| `ggml/src/ggml-rdma/rdma-gdr.cpp` | 444 | 444 | 0 | GPUDirect RDMA |
| `ggml/src/ggml-rdma/rdma-memory.cpp` | 307 | 307 | 0 | ホストメモリ管理 |
| `ggml/src/ggml-rdma/rdma-transport.h` | 187 | 187 | 0 | トランスポート層ヘッダ |
| `ggml/src/ggml-rdma/rdma-gdr.h` | 116 | 116 | 0 | GPUDirect ヘッダ |
| `ggml/src/ggml-rdma/rdma-memory.h` | 109 | 109 | 0 | メモリ管理ヘッダ |
| `ggml/src/ggml-rdma/CMakeLists.txt` | 70 | 70 | 0 | ビルド設定 |
| `ggml/include/ggml-rdma.h` | 79 | 79 | 0 | 公開 C API |
| `tools/rdma/rdma-server.cpp` | 283 | 283 | 0 | サーバーエントリポイント |
| **RDMA コア合計** | **6,298** | **6,318** | **+20** | **10 ファイル** |

v5→v6 では Selective Signaling のマージにより `ggml-rdma.cpp` (RDMA Write チャンキング) と `rdma-transport.cpp` (IB Send シグナリング) に各 +10 行が追加された。

### ヘルパースクリプト (変更なし)

| スクリプト | v5 行数 | v6 行数 | 差分 |
|-----------|-------:|-------:|----:|
| `scripts/rdma-build.sh` | 48 | 48 | 0 |
| `scripts/rdma-deploy.sh` | 27 | 27 | 0 |
| `scripts/rdma-server.sh` | 77 | 77 | 0 |
| `scripts/rdma-env-check.sh` | 360 | 360 | 0 |
| `scripts/gpu-lock.sh` | 157 | 157 | 0 |
| `scripts/task-board.sh` | 369 | 369 | 0 |
| **合計** | **1,038** | **1,038** | **0** |

### コミット・レポート統計

| 指標 | v5 | v6 | 差分 |
|------|---:|---:|-----:|
| RDMA 関連コミット数 (ソースコード変更) | 18 | 19 | +1 |
| 全コミット数 (feature/rdma-backend) | 54 | 60 | +6 |
| レポート数 (MD ファイル) | 82 | 89 | +7 |
| git diff insertions (vs master) | 31,056 | 32,881 | +1,825 |
| 変更ファイル数 (vs master) | 122 | 130 | +8 |

---

## 新規ワークツリー一覧 (v5 以降: 2件)

| ワークツリー | ブランチ | 目的 | 備考 |
|-------------|---------|------|------|
| `merge-validated` | `merge/validated-improvements` | Server-push + Selective signaling 統合テスト | Cherry-pick 検証用、SS は rdma-backend にマージ済み |
| `server-pipeline` | `feature/server-pipeline` | Server-side pipeline 実装 | 未マージ (効果非有意) |

### ワークツリー全体サマリー (更新版)

```
メイン:          1 ワークツリー (rdma-backend)
コード変更なし:  2 ワークツリー (調査のみ)
マージ済み:      1 ワークツリー (fix-bimodal-fprintf)
未マージ:       21 ワークツリー (実験/開発コード)
合計:           25 ワークツリー (master 除く)
```

---

## 新規レポート一覧 (#83-#89)

| # | 日付 | ファイル | タイトル | 区分 |
|---|------|---------|---------|:----:|
| 83 | 02/19 | [2026-02-19_193000](2026-02-19_193000_row_split_11gpu_rdma_benchmark.md) | Row split 11GPU RDMA ベンチマーク | 検証 |
| 84 | 02/20 | [2026-02-20_132916](2026-02-20_132916_merge_validated_improvements_benchmark.md) | 検証済み改善ワークツリーのマージとベンチマーク | 検証 |
| 85 | 02/20 | [2026-02-20_203618](2026-02-20_203618_server_pipeline_implementation.md) | サーバーサイドパイプライン実装 | 実装 |
| 86 | 02/20 | [2026-02-20_212013](2026-02-20_212013_llama_cli_abab_replication.md) | llama-cli ABAB 追試テスト | 検証 |
| 87 | 02/20 | [2026-02-20_215838](2026-02-20_215838_server_pipeline_benchmark.md) | サーバーサイドパイプライン A/B ベンチマーク | 検証 |
| 88 | 02/21 | [2026-02-21_030543](2026-02-21_030543_selective_signaling_merge.md) | Selective Signaling マージ・最適化到達点 | マージ |
| 89 | 02/21 | [2026-02-21_033117](2026-02-21_033117_rdma_backend_retrospective_v6.md) | 振り返りレポート v6 (本レポート) | — |

### レポート数とファイル数

- report/ ディレクトリには通し番号付き .md レポート **89 本** + 補助 .md **1 本** + .html **2 本** = 計 **92 ファイル**
- 通し番号は .md レポートのみを対象 (#1〜#89)

---

## v5 以降のコミット (feature/rdma-backend ブランチ)

| コミット | 内容 |
|---------|------|
| `f07146517` | docs: add benchmark reports and permissions reference |
| `dc66a55dc` | docs: extract procedural sections from CLAUDE.md into Claude Code skills |
| `9fb5b7e64` | docs: fix CLAUDE.md contradictions with current code state |
| `2b2c975ac` | docs: add inline env var example to gpu-lock.sh usage and update layer split benchmarks |
| `79bad6600` | feat: support inline env vars in gpu-lock.sh (KEY=VALUE before command) |
| `184b99ba2` | feat: add process detection fallback to gpu-lock.sh and strengthen CLAUDE.md |
| `cfd75866c` | docs: expand GPU lock scope and add GDR troubleshooting to CLAUDE.md |
| `5dea2e773` | docs: clarify worktree merge policy — commit locally, no merge to rdma-backend |
| `310682c72` | feat: add pre-experiment environment check script (rdma-env-check.sh) |
| `1e256325d` | docs: add GPUDirect RDMA (GDR) setup guide to CLAUDE.md |
| `c33d1b86a` | feat: add multi-session workflow infrastructure (task board + GPU lock) |
| `6b44098f8` | docs: expand sample size guidelines in stats skill with effect-size table and pilot run procedure |
| `0b9ef987f` | docs: fix report path in CLAUDE.md to match actual location |
| `a718746e4` | docs: add RDMA backend retrospective v5 report |
| `f5a631535` | docs: add SD explanation and rename stats skill README to benchmark statistics guide |
| `20e808adb` | **feat: add selective signaling for chunked RDMA Write and Send** |
| `f0f656a88` | docs: add benchmark reports and selective signaling merge report |

---

## 残タスク (更新版)

| # | タスク | v5 状態 | v6 状態 | 優先度 | ブロッカー |
|---|--------|---------|---------|:------:|-----------|
| 1〜3 | (Step 1-3 タスク) | 完了 | 完了 | — | — |
| 4 | 16GPU ハードウェア準備 | 待ち | **待ち** (変更なし) | 高 | 物理 GPU 追加 |
| 5 | Q3_K_M/Q4_K_M ダウンロード・テスト | 未着手 | **未着手** (16GPU 待ち) | 高 | タスク 4 |
| 6 | 16GPU 全体テスト | 未着手 | **未着手** (タスク 4,5 依存) | 高 | タスク 4, 5 |
| 7 | RDMA Generation 速度改善 | 限界到達 | **限界到達** (変更なし) | 低 | アーキテクチャ変更 |
| 8 | ~~Server-Push + Selective signaling マージ~~ | マージ待ち | **Selective signaling マージ済み** | — | 完了 |

### タスク #8 の更新

v5 時点で「Server-Push + Selective signaling マージ」として待機していたタスクの結論:

| ワークツリー | v5 推奨 | v6 結論 |
|-------------|---------|---------|
| `rdma-selective-signaling` | 推奨 | **マージ済み** (`20e808adb`) |
| `rdma-server-push` | 推奨 | **Opt-in 維持** (`GGML_RDMA_PER_DEVICE_CONN=1` で有効化) |

Server-push は per-device connections と不可分であり、pp128 退行 (-1.35%) とのトレードオフがあるため、デフォルトマージではなく opt-in として維持する判断を確定。

---

## プロジェクト全体の振り返り

### 22日間の成果総括

2026年1月31日の実装開始から2月21日まで、**22日間** で RDMA バックエンドを `ゼロから安定動作+通信最適化の実質的上限到達` まで進めた。

### 性能進化の軌跡

```
Day 1  (02/01): 0.3 t/s    — 初回動作 (naive protocol)
Day 4  (02/04): 148.5 t/s  — プロトコル最適化完了 (495×)
Day 5  (02/05): 59.5 t/s   — GPUDirect RDMA (20b, 92.6% of local)
Day 7  (02/07): 6.8 t/s    — GLM-4.7 47B on 11 GPUs
Day 10 (02/09): 7.2 t/s    — 最良値 (安定化後) ← v3 時点
Day 14 (02/13): 7.5 t/s    — 二峰性修正 + deferred copy ← v4 時点
Day 20 (02/19): 7.76 t/s   — server-push + selective signaling ← v5 時点
Day 22 (02/21): 7.75 t/s   — selective signaling マージ (pp128: 30.15 t/s) ← v6 時点
```

### v5→v6 の総括

| 項目 | v5 時点 | v6 時点 | 変化 |
|------|---------|---------|------|
| pp128 (feature/rdma-backend) | 23.35 t/s | **30.15 t/s** | **+29.1%** |
| tg32 (feature/rdma-backend) | 7.75 t/s | **7.75 t/s** | ±0% |
| RDMA vs RPC (pp128) | +37% | **+458%** | Selective signaling マージ |
| RDMA vs RPC (tg32) | +3.5% | **+3.3%** | 微減 (測定誤差範囲) |
| マージ済みコード変更 | deferred copy, GDR budget | + **selective signaling** | +20 行 |
| 新規ワークツリー | — | 2件 | merge-validated, server-pipeline |
| 新規レポート | — | 7件 (#83-#89) | 検証3, 実装1, ベンチマーク1, マージ1, 振り返り1 |

**通信レイヤーの最適化は実質的な上限に到達した。** 次の大きな改善ステップは:

1. **16GPU 拡張** — ハードウェア追加によるスケーリングと Q4 量子化モデルへの対応
2. **Expert Parallelism** — MoE 特化の並列化で理論上 +80-158% の tg 改善
3. **ConnectX-6 RNIC** — MTT 制限解消で per-device connections のデフォルト有効化

---

## 全レポート一覧 (更新版)

v5 の 82 本 → v6 含め **89 本** (+ 補助ドキュメント 3 本)

| # | 日付 | ファイル | タイトル | Step |
|---|------|---------|---------|:----:|
| 1 | 01/31 | [2026-01-31_214933](2026-01-31_214933_gpudirect_rdma_feasibility.md) | GPUDirect RDMA 実装可能性調査 | 1 |
| 2 | 01/31 | [2026-01-31_222100](2026-01-31_222100_gpudirect_rdma_backend_implementation.md) | GPUDirect RDMA バックエンド実装 | 1 |
| 3 | 02/01 | [2026-02-01_104744](2026-02-01_104744_rdma_backend_test.md) | RDMA バックエンド実行テスト・修正 | 1 |
| 4 | 02/03 | [2026-02-03_134920](2026-02-03_134920_pcie_topology_verification.md) | PCIeトポロジー検証 | 1 |
| 5 | 02/03 | [2026-02-03_165600](2026-02-03_165600_rdma_backend_test.md) | RDMA バックエンド テスト・修正 (続) | 1 |
| 6 | 02/03 | [2026-02-03_183742](2026-02-03_183742_get_alloc_size_optimization.md) | get_alloc_size パフォーマンス最適化 | 1 |
| 7 | 02/03 | [2026-02-03_190500](2026-02-03_190500_rdma_timeout_fix.md) | RDMAタイムアウト問題修正 | 1 |
| 8 | 02/04 | [2026-02-04_040100](2026-02-04_040100_rdma_graph_compute_buffer_fix.md) | RDMAグラフ計算時バッファサイズ問題の修正 | 1 |
| 9 | 02/04 | [2026-02-04_052200](2026-02-04_052200_rdma_connection_lifetime_fix.md) | RDMA接続ライフタイム修正 | 1 |
| 10 | 02/04 | [2026-02-04_060500](2026-02-04_060500_rdma_protocol_performance_optimization.md) | RDMAプロトコル最適化 (0.3→2.7 t/s) | 1 |
| 11 | 02/04 | [2026-02-04_065100](2026-02-04_065100_rdma_graph_diff_update_optimization.md) | グラフ差分更新最適化 (2.7→47.2 t/s) | 1 |
| 12 | 02/04 | [2026-02-04_072500](2026-02-04_072500_rdma_adaptive_rsp_persistent_staging.md) | 適応的レスポンス + 永続ステージング | 1 |
| 13 | 02/04 | [2026-02-04_110225](2026-02-04_110225_rdma_benchmark_gpt_oss.md) | gpt-oss-20b/120b ベンチマーク | 2 |
| 14 | 02/04 | [2026-02-04_184824](2026-02-04_184824_rdma_120b_cluster_fix.md) | 120b クラスタ修正 (supports_buft) | 2 |
| 15 | 02/04 | [2026-02-04_214044](2026-02-04_214044_cpu_staged_vs_gpudirect_rdma_comparison.md) | CPU-staged vs GPUDirect 比較 | 3 |
| 16 | 02/04 | [2026-02-04_224611](2026-02-04_224611_rdma_batch_flush_optimization.md) | バッチフラッシュ最適化 | 3 |
| 17 | 02/05 | [2026-02-05_003452](2026-02-05_003452_step3_flush_compute_optimization.md) | FLUSH+COMPUTE統合 & プロファイリング | 3 |
| 18 | 02/05 | [2026-02-05_101500](2026-02-05_101500_step4_gpudirect_rdma_enabled.md) | GPUDirect RDMA 有効化 | 4 |
| 19 | 02/05 | [2026-02-05_121023](2026-02-05_121023_step4_gpudirect_120b_cluster_test.md) | GPUDirect 120b 11GPU クラスタテスト | 4 |
| 20 | 02/05 | [2026-02-05_133246](2026-02-05_133246_language_speed_comparison.md) | 日本語/英語プロンプト速度差調査 | 4 |
| 21 | 02/05 | [2026-02-05_150903](2026-02-05_150903_language_speed_benchmark_v2.md) | 言語速度比較ベンチマーク v2 | 4 |
| 22 | 02/05 | [2026-02-05_162500](2026-02-05_162500_rdma_vs_rpc_benchmark.md) | RDMA vs RPC 比較 | 4 |
| 23 | 02/05 | [2026-02-05_170441](2026-02-05_170441_gpu_combination_benchmark.md) | GPU組み合わせベンチマーク | 4 |
| 24 | 02/05 | [2026-02-05_175811](2026-02-05_175811_gpu_topology_benchmark.md) | GPUトポロジ考慮ベンチマーク | 4 |
| 25 | 02/05 | [2026-02-05_192000](2026-02-05_192000_gpu_combination_variance_analysis.md) | 性能ばらつき分析 | 4 |
| 26 | 02/05 | [2026-02-05_210745](2026-02-05_210745_rdma_variance_deep_investigation.md) | 性能ばらつき深掘り調査 | 4 |
| 27 | 02/05 | [2026-02-05_225700](2026-02-05_225700_rnr_nak_root_cause_fix.md) | RNR NAK 根本原因特定・修正 | 4 |
| 28 | 02/06 | [2026-02-06_000500](2026-02-06_000500_rdma_vs_rpc_benchmark_v2.md) | RDMA vs RPC v2 (RNR修正後) | 4 |
| 29 | 02/06 | [2026-02-06_202900](2026-02-06_202900_persistent_boot_config_guide.md) | init.sh 永続化マニュアル | Infra |
| 30 | 02/06 | [2026-02-06_213701](2026-02-06_213701_rdma_optimization_benchmark.md) | 性能最適化ベンチマーク (5手法評価) | 4 |
| 31 | 02/07 | [2026-02-07_003000](2026-02-07_003000_glm47_iq2m_11gpu_test.md) | GLM-4.7 IQ2_M 11GPU テスト | 5 |
| 32 | 02/07 | [2026-02-07_095600](2026-02-07_095600_mmap_rdma_write_bugfix.md) | mmap + RDMA Write バグ修正 | 4/5 |
| 33 | 02/07 | [2026-02-07_104703](2026-02-07_104703_rdma_backend_retrospective.md) | 振り返りレポート v1 | — |
| 34 | 02/07 | [2026-02-07_124327](2026-02-07_124327_glm47_iq2m_rdma_vs_rpc_output_quality.md) | GLM-4.7 IQ2_M RDMA vs RPC 出力品質比較 | 5 |
| 35 | 02/07 | [2026-02-07_173027](2026-02-07_173027_build_procedure_investigation.md) | ビルド手順の安定化調査 | — |
| 36 | 02/07 | [2026-02-07_175849](2026-02-07_175849_rdma_backend_technical_architecture.md) | RDMA バックエンド技術アーキテクチャ | — |
| 37 | 02/07 | [2026-02-07_192500](2026-02-07_192500_p100_pcie_p2p_transfer_spec.md) | P100 PCIe P2P転送仕様 | — |
| 38 | 02/07 | [2026-02-07_195500](2026-02-07_195500_multi_rdma_device_output_corruption_fix.md) | マルチRDMAデバイス出力破損バグ修正 | 5 |
| 39 | 02/07 | [2026-02-07_202700](2026-02-07_202700_cpy_tensor_fix_comprehensive_test.md) | cpy_tensor 修正の包括的テスト | 5 |
| 40 | 02/07 | [2026-02-07_203000](2026-02-07_203000_rdma_backend_retrospective_v2.md) | 振り返りレポート v2 | — |
| 41 | 02/07 | [2026-02-07_215841](2026-02-07_215841_gpudirect_rdma_timeout_fix.md) | GPUDirect RDMA タイムアウト修正 | 5 |
| 42 | 02/07 | [2026-02-07_235500](2026-02-07_235500_rdma_performance_improvement_research.md) | RDMA 性能改善調査 | 5 |
| 43 | 02/08 | [2026-02-08_013936](2026-02-08_013936_p100_compute_performance_research.md) | P100 計算性能ボトルネック分析 | 5 |
| 44 | 02/08 | [2026-02-08_020500](2026-02-08_020500_server_robustness_and_per_device_connections.md) | サーバー堅牢化と Per-device 接続 | 5 |
| 45 | 02/08 | [2026-02-08_140512](2026-02-08_140512_rdma_completion_timeout_fix.md) | RDMA Completion タイムアウト修正 | 5 |
| 46 | 02/08 | [2026-02-08_163000](2026-02-08_163000_server_command_recv_timeout_fix.md) | サーバー側コマンド受信タイムアウト修正 | 5 |
| 47 | 02/08 | [2026-02-08_213000](2026-02-08_213000_graph_reserve_oom_segfault_fix.md) | graph_reserve OOM Segfault 修正 | 5 |
| 48 | 02/09 | [2026-02-09_025900](2026-02-09_025900_fit_params_oom_fallback.md) | fit_params OOM フォールバック | 5 |
| 49 | 02/09 | [2026-02-09_030930](2026-02-09_030930_rdma_backend_retrospective_v3.md) | 振り返りレポート v3 | — |
| 50 | 02/09 | [2026-02-09_084919](2026-02-09_084919_github_deploy_key_setup.md) | GitHub Deploy Key セットアップ | Infra |
| 51 | 02/10 | [2026-02-10_214841](2026-02-10_214841_rdma_async_compute_optimization.md) | Async compute 最適化 | 5 |
| 52 | 02/10 | [2026-02-10_222348](2026-02-10_222348_doorbell_batching_max_inline.md) | Doorbell batching + max_inline | 実験 |
| 53 | 02/11 | [2026-02-11_040659](2026-02-11_040659_prepost_recv_rnr_nak.md) | Pre-posted recv (RNR NAK) | 実験 |
| 54 | 02/11 | [2026-02-11_045619](2026-02-11_045619_combined_send_rnr_nak.md) | Combined send (ヘッダ統合) | 実験 |
| 55 | 02/11 | [2026-02-11_095500](2026-02-11_095500_hotpath_memory_optimization.md) | Hotpath memory 最適化 | 実験 |
| 56 | 02/11 | [2026-02-11_133000](2026-02-11_133000_xdev_cache_optimization.md) | Cross-device cache 最適化 | 実験 |
| 57 | 02/11 | [2026-02-11_214242](2026-02-11_214242_parallel_compute_dispatch.md) | Parallel compute dispatch | 5 |
| 58 | 02/12 | [2026-02-12_070600](2026-02-12_070600_hybrid_parallelism_research.md) | Hybrid parallelism research | 調査 |
| 59 | 02/12 | [2026-02-12_134623](2026-02-12_134623_ngram_speculative_decoding_test.md) | Ngram speculative decoding test | 実験 |
| 60 | 02/12 | [2026-02-12_160000](2026-02-12_160000_draft_speculative_decoding_test.md) | Draft model speculative decoding test | 実験 |
| 61 | 02/13 | [2026-02-13_044038](2026-02-13_044038_rdma_backend_comprehensive_review.md) | RDMA バックエンド包括的レビュー | 調査 |
| 62 | 02/13 | [2026-02-13_053056](2026-02-13_053056_deferred_copy_optimization.md) | Deferred copy 最適化 | 5 |
| 63 | 02/13 | [2026-02-13_053658](2026-02-13_053658_rdma_synchronize_implementation.md) | RDMA synchronize 実装 | 5 |
| 64 | 02/13 | [2026-02-13_090026](2026-02-13_090026_deferred_copy_statistical_benchmark.md) | Deferred copy 統計的ベンチマーク | 5 |
| 65 | 02/13 | [2026-02-13_153100](2026-02-13_153100_bimodal_tg_investigation.md) | 二峰性 tg 分布調査 | バグ |
| 66 | 02/13 | [2026-02-13_185255](2026-02-13_185255_rdma_backend_retrospective_v4.md) | 振り返りレポート v4 | — |
| 67 | 02/14 | [2026-02-14_024402](2026-02-14_024402_synchronize_bimodal_fix_and_benchmark.md) | synchronize 二峰性修正マージ & 効果検証 | 検証 |
| 68 | 02/14 | [2026-02-14_055914](2026-02-14_055914_synchronize_confound_free_remeasurement.md) | synchronize 交絡因子排除再測定 | 検証 |
| 69 | 02/14 | [2026-02-14_082220](2026-02-14_082220_deferred_copy_clean_ab_benchmark.md) | Deferred copy 再評価 (二峰性修正後) | 検証 |
| 70 | 02/18 | [2026-02-18_070912](2026-02-18_070912_generation_optimization_deep_investigation.md) | Generation 最適化 深堀り調査 | 調査 |
| 71 | 02/18 | [2026-02-18_120000](2026-02-18_120000_unimplemented_optimizations_inventory.md) | 未実施最適化の棚卸し | 調査 |
| 72 | 02/18 | [2026-02-18_130000](2026-02-18_130000_shared_pd_per_device_connection.md) | PD 共有 Per-device connection | 実装 |
| 73 | 02/18 | [2026-02-18_200000](2026-02-18_200000_server_push_rdma_write_imm.md) | Server-Push (RDMA Write with IMM) | 実装 |
| 74 | 02/19 | [2026-02-19_090721](2026-02-19_090721_ggml_backend_sched_bottleneck_analysis.md) | ggml_backend_sched ボトルネック分析 | 調査 |
| 75 | 02/19 | [2026-02-19_104648](2026-02-19_104648_hugepage_staging_benchmark.md) | Hugepage ステージングバッファ | 実装 |
| 76 | 02/19 | [2026-02-19_120000](2026-02-19_120000_selective_signaling.md) | Selective signaling | 実装 |
| 77 | 02/19 | [2026-02-19_125638](2026-02-19_125638_nvidia_peermem_load_failure.md) | nvidia-peermem ロード失敗調査 | 調査 |
| 78 | 02/19 | [2026-02-19_133220](2026-02-19_133220_nvidia_peermem_kernel_downgrade.md) | nvidia-peermem 復旧 (カーネルダウングレード) | Infra |
| 79 | 02/19 | [2026-02-19_160000](2026-02-19_160000_hugepage_gdr_benchmark.md) | Hugepage + GDR バジェット拡大 | 検証 |
| 80 | 02/19 | [2026-02-19_163700](2026-02-19_163700_row_vs_layer_split_benchmark.md) | Row vs Layer split ベンチマーク | 検証 |
| 81 | 02/19 | [2026-02-19_200000](2026-02-19_200000_row_split_fit_params_fix.md) | Row split fit_params 修正 | 実装 |
| 82 | 02/19 | [2026-02-19_190406](2026-02-19_190406_rdma_backend_retrospective_v5.md) | 振り返りレポート v5 | — |
| **83** | **02/19** | [2026-02-19_193000](2026-02-19_193000_row_split_11gpu_rdma_benchmark.md) | **Row split 11GPU RDMA ベンチマーク** | **検証** |
| **84** | **02/20** | [2026-02-20_132916](2026-02-20_132916_merge_validated_improvements_benchmark.md) | **検証済み改善のマージとベンチマーク** | **検証** |
| **85** | **02/20** | [2026-02-20_203618](2026-02-20_203618_server_pipeline_implementation.md) | **サーバーサイドパイプライン実装** | **実装** |
| **86** | **02/20** | [2026-02-20_212013](2026-02-20_212013_llama_cli_abab_replication.md) | **llama-cli ABAB 追試テスト** | **検証** |
| **87** | **02/20** | [2026-02-20_215838](2026-02-20_215838_server_pipeline_benchmark.md) | **サーバーサイドパイプライン A/B ベンチマーク** | **検証** |
| **88** | **02/21** | [2026-02-21_030543](2026-02-21_030543_selective_signaling_merge.md) | **Selective Signaling マージ・最適化到達点** | **マージ** |
| **89** | **02/21** | [2026-02-21_033117](2026-02-21_033117_rdma_backend_retrospective_v6.md) | **振り返りレポート v6 (本レポート)** | **—** |

### 補助ドキュメント

| ファイル | 形式 | 説明 |
|---------|------|------|
| [2026-02-07_175849_rdma_backend_technical_architecture.html](2026-02-07_175849_rdma_backend_technical_architecture.html) | HTML | 技術アーキテクチャ (Mermaid図レンダリング版) |
| [2026-02-07_llama_cli_inference_pipeline.md](2026-02-07_llama_cli_inference_pipeline.md) | MD | llama-cli 推論パイプライン解説 |
| [2026-02-07_llama_cli_inference_pipeline.html](2026-02-07_llama_cli_inference_pipeline.html) | HTML | llama-cli 推論パイプライン (Mermaid図レンダリング版) |

---

## 量子化レベル選択 (変更なし)

| 量子化 | bpw | 推定サイズ | 16GPU (256GB) | 11GPU (176GB) |
|--------|----:|----------:|:-------------:|:-------------:|
| IQ2_M | 2.7 | 114 GB | 収容可能 | 収容可能 (品質△) |
| Q3_K_M | ~3.5 | ~150 GB | 収容可能 | 収容不可 |
| Q4_K_M | ~4.5 | ~195 GB | 収容可能 | 収容不可 |
