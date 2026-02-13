# RDMA バックエンド 振り返りレポート v4

- **実施日時**: 2026年2月13日 18:52
- **ワークツリー**: `.worktree/rdma-backend`
- **前回レトロスペクティブ**: [2026-02-09_030930_rdma_backend_retrospective_v3.md](2026-02-09_030930_rdma_backend_retrospective_v3.md)

---

## 前回レトロスペクティブ (v3) からの主要変更点

v3 (02/09 03:09) 以降の約4日間で **17 本のレポート** (#50〜#66) が追加され、4 機能の実装、2 バグ修正、10 件の調査・実験が完了した。

### 実装された機能 (4件)

| # | 機能 | 概要 | コミット | ファイル |
|---|------|------|---------|---------|
| 1 | **Async compute (fire-and-forget)** | `graph_compute` を非同期化。クライアント側 graph_compute 24.5× 高速化 (17.2ms → 0.7ms)。ただし get_tensor の暗黙的同期で e2e 改善なし | `e26e01a0e` | `ggml-rdma.cpp` |
| 2 | **initiator_depth 16** | ConnectX-4 `max_qp_rd_atom=16` に合わせ `initiator_depth`/`responder_resources` を 1→16 に拡張。将来のバッチ RDMA Write 最適化を有効化 | `e26e01a0e` | `ggml-rdma.cpp`, `rdma-transport.cpp` |
| 3 | **Parallel compute dispatch** | サーバー側 per-device worker threads (`compute_dispatcher`) で graph_compute をデバイス並列ディスパッチ。サーバーは正しく並列化するが、クライアント `ggml_backend_sched` が逐次のため e2e 改善なし | 未マージ (`e98da14fd`) | `ggml-rdma.cpp` |
| 4 | **Deferred copy** | `cpy_tensor` を同一接続 RDMA バッファ間で再有効化。クロスデバイスコピーを遅延エントリとして記録し、ASYNC コマンドに同梱。サーバーが D2H+H2D をローカル実行 (IB ネットワーク不要) | `1d2c9cbdd` | `ggml-rdma.cpp` |

### 修正されたバグ (2件)

| # | バグ | 修正内容 | コミット |
|---|------|---------|---------|
| 15 | **ホットパス fprintf による二峰性 tg** | `get_tensor()` ホットパスの always-on `fprintf(stderr, ...)` が stdio バッファフラッシュで ~1700ms スパイクを発生。`RDMA_LOG_DBG` (GGML_RDMA_DEBUG=1 で有効) に置換 | `5b6e0bf56` |
| 16 | **Deferred copy + flush_all_staging 競合** | バウンディングボックスマージで stale staging データが deferred copy 先を上書き。RECOMPUTE/COMPUTE_UPDATE パスで `flush_all_staging` → `execute_deferred_copies` の実行順序を修正 | `1d2c9cbdd` |

### 完了した調査・実験 (10件)

| # | 調査 | 結論 | レポート |
|---|------|------|---------|
| 1 | **Doorbell batching + max_inline 拡張** | 2 send → 1 post_send で ~10µs/token 削減。GPU compute ~17ms に対し無視可能。改善なし | [#52](2026-02-10_222348_doorbell_batching_max_inline.md) |
| 2 | **Pre-posted recv (RNR NAK 削減)** | ASYNC compute 前にサーバー側 recv をプレポスト。RNR NAK ~40K→~41K で変化なし。NAK の主因はヘッダ↔データ間隔 | [#53](2026-02-11_040659_prepost_recv_rnr_nak.md) |
| 3 | **Combined send (ヘッダ+データ統合)** | 1 バッファ統合送信 (≤16MB)。Send/Recv 回数半減、RNR NAK 6% 減。ただし recv バッファオーバーヘッドで tg 低下 | [#54](2026-02-11_045619_combined_send_rnr_nak.md) |
| 4 | **Hotpath memory allocation 最適化** | vector reserve、per-connection user_data (global mutex 除去)、graph cache buffer 再利用。コード品質向上、e2e 改善なし | [#55](2026-02-11_095500_hotpath_memory_optimization.md) |
| 5 | **Cross-device transfer cache 最適化** | GPU allocation cache + pinned host buffer。`fix_xdev` 0.09ms→0.02ms。layer split では xdev テンソルが稀で e2e 改善なし | [#56](2026-02-11_133000_xdev_cache_optimization.md) |
| 6 | **Hybrid parallelism research** | GLM-4.7 on P100×16: 純 PP=8-9 t/s、TP=2+PP=8=12-13 t/s、EP=14-20 t/s。推奨: 1) Speculative, 2) Compute overlap, 3) EP | [#58](2026-02-12_070600_hybrid_parallelism_research.md) |
| 7 | **Ngram speculative decoding test** | GLM-4.7 (MoE) で acceptance rate 4.2-10.6%。Expert routing の確率的性質により予測困難。MoE モデルには不適 | [#59](2026-02-12_134623_ngram_speculative_decoding_test.md) |
| 8 | **Draft model speculative decoding test** | GLM-4-9B draft: 56-60% acceptance, tg −2〜−6%。qwen2.5-0.5b draft: 50-52%, tg +3.9%。RDMA で微増、RPC で大幅低下 | [#60](2026-02-12_160000_draft_speculative_decoding_test.md) |
| 9 | **RDMA backend comprehensive review** | 5 セクション: メモリリーク (低リスク)、Generation ボトルネック (two-phase loop + synchronize)、ビルドスクリプト、テストギャップ、16GPU 準備状況 | [#61](2026-02-13_044038_rdma_backend_comprehensive_review.md) |
| 10 | **Bimodal tg distribution investigation** | tg が 5.5 t/s (46%) と 6.7 t/s (54%) の二峰性分布。根本原因: ホットパス fprintf の stdio バッファフラッシュスパイク (~1700ms)。修正後 7.4-7.5 t/s 安定 | [#65](2026-02-13_153100_bimodal_tg_investigation.md) |

### v3→v4 統合サマリーテーブル (全17項目)

| # | 修正/実験内容 | レポート | ワークツリー | tg 改善 | マージ状況 |
|---|-------------|---------|------------|--------|-----------|
| 1 | Async compute (fire-and-forget) | [#51](2026-02-10_214841_rdma_async_compute_optimization.md) | `rdma-backend` | なし (e2e) | - |
| 2 | initiator_depth 16 | [#51](2026-02-10_214841_rdma_async_compute_optimization.md) | `rdma-backend` | なし | - |
| 3 | Doorbell batching + max_inline | [#52](2026-02-10_222348_doorbell_batching_max_inline.md) | `opt-doorbell-batch` | なし | 未マージ |
| 4 | Pre-posted recv | [#53](2026-02-11_040659_prepost_recv_rnr_nak.md) | `opt-prepost-recv` | なし | 未マージ |
| 5 | Combined send | [#54](2026-02-11_045619_combined_send_rnr_nak.md) | `opt-combined-send` | 低下 | 未マージ |
| 6 | Hotpath memory 最適化 | [#55](2026-02-11_095500_hotpath_memory_optimization.md) | `rdma-hotpath` | なし | 未マージ |
| 7 | Cross-device cache | [#56](2026-02-11_133000_xdev_cache_optimization.md) | `rdma-xdev-cache` | なし | 未マージ |
| 8 | Parallel compute dispatch | [#57](2026-02-11_214242_parallel_compute_dispatch.md) | `rdma-parallel-compute` | なし (e2e) | 未マージ |
| 9 | Hybrid parallelism research | [#58](2026-02-12_070600_hybrid_parallelism_research.md) | `hybrid-split-research` | (調査) | - |
| 10 | Ngram speculative decoding | [#59](2026-02-12_134623_ngram_speculative_decoding_test.md) | `ngram-spec-test` | −7.8% | 未マージ |
| 11 | Draft model speculative decoding | [#60](2026-02-12_160000_draft_speculative_decoding_test.md) | `draft-spec-test` | +3.9% (qwen) | 未マージ |
| 12 | Comprehensive review | [#61](2026-02-13_044038_rdma_backend_comprehensive_review.md) | `rdma-backend` | (調査) | - |
| 13 | Deferred copy | [#62](2026-02-13_053056_deferred_copy_optimization.md) | `rdma-deferred-copy` | **+13%** (20b) / **+5.9%** (GLM) | 機能コードマージ済み (`1d2c9cbdd`)、開発スクリプト未マージ |
| 14 | synchronize 実装 | [#63](2026-02-13_053658_rdma_synchronize_implementation.md) | `rdma-synchronize` | 未テスト | 未マージ |
| 15 | Deferred copy 統計検証 | [#64](2026-02-13_090026_deferred_copy_statistical_benchmark.md) | `rdma-deferred-copy` | (検証) | (#13 と同一ワークツリー) |
| 16 | 二峰性 tg 調査 | [#65](2026-02-13_153100_bimodal_tg_investigation.md) | `bimodal-investigation` | (調査) | - |
| 17 | 二峰性 tg 修正 (fprintf) | [#65](2026-02-13_153100_bimodal_tg_investigation.md) | `fix-bimodal-fprintf` | **+10%** (安定化) | マージ済み (`5b6e0bf56`) |

- **マージ状況の凡例**:
  - `-`: rdma-backend で直接作業、またはコード変更なし (マージ対象なし)
  - `マージ済み`: ワークツリーのコード変更が rdma-backend に取り込まれている
  - `未マージ`: ワークツリーにコード変更があるが rdma-backend には未取り込み
  - レポートは全て rdma-backend に作成されるため、マージ状況には含めない

---

## 作成されたレポート (v3 以降: 17件)

| # | 日付 | ファイル | タイトル | 区分 |
|---|------|---------|---------|:----:|
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
| 66 | — | [2026-02-13_185255](2026-02-13_185255_rdma_backend_retrospective_v4.md) | 振り返りレポート v4 (本レポート) | — |

*v3 の #49 の後に作成された新規レポートは #50〜#65 の 16 本 + 本レポート (#66) = 計 17 本。*

---

## v3 以降のコミット (7件)

| コミット | 内容 |
|---------|------|
| `e26e01a0e` | feat(rdma): add async graph_compute and increase initiator_depth to 16 |
| `253845a9a` | docs: add worktree rules and fix build/deploy scripts for worktree support |
| `055fad0f4` | docs: add optimization experiment reports and hybrid parallelism research |
| `63db2af0e` | docs: add ngram and draft model speculative decoding test reports |
| `1d2c9cbdd` | feat(rdma): implement deferred copy for server-local cross-device transfers |
| `5b6e0bf56` | fix(rdma): remove always-on fprintf from hot paths to fix bimodal tg |
| `49e511f1b` | docs: add comprehensive review, synchronize research, and update hybrid parallelism report |

---

## Step 別ステータス (更新版)

| Step | 内容 | 状態 | v3 からの変更 |
|------|------|:----:|:-------------:|
| Step 1 | 基本 RDMA 通信 | ✅ 完了 | なし |
| Step 2 | マルチノードクラスタ安定化 | ✅ 完了 | なし |
| Step 3 | RDMA 性能最適化 | ✅ 完了 | なし |
| Step 4 | GPUDirect RDMA 有効化 | ✅ 完了 | なし |
| Step 5 | GLM-4.7 Q4 on 16 P100s | 🔧 進行中 | 大幅進捗 (下記) |

### Step 5 の進捗更新

| マイルストーン | v3 時点 | v4 時点 |
|---------------|--------|--------|
| GLM-4.7 IQ2_M 動作 | **安定動作** | **安定動作** (二峰性修正で 7.4-7.5 t/s) |
| Async compute | 未実装 | **実装済み** (graph_compute 24.5× 高速化、e2e なし) |
| Deferred copy | 未実装 | **実装済み** (gpt-oss-20b +13% tg、GLM-4.7 +5.9% tg) |
| Parallel compute dispatch | 未実装 | **実装済み** (サーバー正常、クライアント sched 制約で e2e なし) |
| synchronize 実装 | 未実装 | **実装済み** (RDMA_CMD_SYNC、GDR RDMA Read パス有効化) |
| 二峰性 tg 修正 | 未発見 | **修正済み** (fprintf 除去 → 7.4 t/s 安定) |
| 最適化実験 (6件) | 未着手 | **完了** (doorbell, pre-posted recv, combined send, hotpath, xdev cache, speculative) |
| Hybrid parallelism 調査 | 未着手 | **完了** (PP/TP/EP 比較、speculative 評価) |
| 16GPU ハードウェア | 待ち | 待ち (変更なし) |
| Q3_K_M/Q4_K_M テスト | 未着手 | 未着手 (16GPU 待ち) |

---

## 性能サマリー (更新版)

### Step ごとのベストスコア

| Step | 構成 | モデル | pp (t/s) | tg (t/s) | ローカル比 (tg) |
|------|------|--------|:--------:|:--------:|:--------------:|
| Step 1 完了 | RDMA 1+1 | qwen2.5-0.5b | — | 148.5 | 75.7% |
| Step 2 完了 | 11GPU (7C+4R) | gpt-oss-120b | 3〜11 | 0.7〜1.1 | 1.5〜2.3% |
| Step 3 完了 | RDMA 1+1 | gpt-oss-20b | 61.09 | 58.47 | **91.0%** |
| Step 4 完了 (GDR) | RDMA 1+1 | gpt-oss-20b | **403.69** | 59.52 | **92.6%** |
| Step 4 完了 (クラスタ) | 11GPU (7C+4R) | gpt-oss-120b | 201.02 | 36.92 | 84.7% |
| Step 5 (v3 時点) | 11GPU (7C+4R) | GLM-4.7 IQ2_M | 6.3〜10.6 | 5.5〜7.2 | — |
| **Step 5 (v4 時点)** | **11GPU (7C+4R)** | **GLM-4.7 IQ2_M** | **7.4** | **7.4〜7.5** | **—** |

### GLM-4.7 IQ2_M 最新性能データ (11GPU: 7C+4R)

#### v3 以降の新規テスト結果

| テスト | 条件 | pp (t/s) | tg (t/s) | 出典 |
|--------|------|:--------:|:--------:|------|
| Async compute | seed=42, 50tok | 6.3 | 6.6 | [#51](2026-02-10_214841_rdma_async_compute_optimization.md) |
| Deferred copy (統計, n=20) | seed=42, paired t-test | 7.1±0.5 | **6.7±0.8** | [#64](2026-02-13_090026_deferred_copy_statistical_benchmark.md) |
| 二峰性修正後 (n=10) | seed=42, stderr→/dev/null | **7.4** | **7.4〜7.5** | [#65](2026-02-13_153100_bimodal_tg_investigation.md) |

#### 二峰性問題と修正

v3 時点で tg が 5.5〜7.2 t/s と大きくばらついていた原因を特定:
- **根本原因**: `get_tensor()` ホットパスの `fprintf(stderr, ...)` が stdio バッファフラッシュで ~1700ms スパイクを発生
- **証拠**: stderr→/dev/null で 7.4 t/s 安定 (10/10)、stderr→file で 5.5 vs 6.7 の二峰性
- **修正** (`5b6e0bf56`): always-on fprintf → `RDMA_LOG_DBG` (GGML_RDMA_DEBUG=1 でのみ有効)
- **結果**: tg **7.4〜7.5 t/s 安定** (二峰性完全解消)

### gpt-oss-20b deferred copy の効果

| 構成 | 条件 | tg (t/s) | 改善率 |
|------|------|:--------:|:------:|
| 1C+2R | deferred copy 無効 | 50.9 | — |
| 1C+2R | **deferred copy 有効** | **57.7** | **+13.4%** |

### GLM-4.7 RDMA vs RPC 比較 (更新版)

| バックエンド | Prompt (t/s) | Generation (t/s) | 備考 |
|-------------|:------------:|:----------------:|------|
| **RDMA (v4, fprintf 修正後)** | **7.4** | **7.4〜7.5** | 安定 |
| RDMA (v3, GDR budget 12GB) | 6.4 | 6.8 (ばらつき大) | 二峰性あり |
| RDMA (GDR 無効) | 6.4 | 6.0 | — |
| RPC (TCP) | 5.4 | 7.5 | — |

- **v4 時点**: RDMA は Prompt で RPC 比 **+37%**、Generation でも RPC と **同等** (7.4-7.5 vs 7.5)
- v3 までは Generation で RPC 比 -10% だったが、fprintf 修正で **ほぼ同等** に改善

---

## 既知のバグと対応状況 (更新版)

v3 の 14 件 + 新規 2 件 = 合計 **16 件**。すべて修正済みまたはワークアラウンドあり。

| # | バグ | v3 状態 | v4 状態 |
|---|------|---------|---------|
| 1 | GPUDirect timeout on large models | 修正済み | 修正済み (変更なし) |
| 2 | GLM-4.7 IQ2_M 出力品質 | 修正済み | 修正済み (変更なし) |
| 3 | RDMA デバイス逐次実行 | 設計上の制約 | **設計上の制約** (parallel dispatch 実装、クライアント制約残存) |
| 4 | cmake --build 無出力 | 解決 | 解決 (変更なし) |
| 5 | mmap + RDMA Write overflow | 修正済み | 修正済み (変更なし) |
| 6 | RNR NAK 2秒スパイク | 修正済み | 修正済み (変更なし) |
| 7 | supports_buft クロスデバイス | 修正済み | 修正済み (変更なし) |
| 8 | マルチRDMAデバイス出力破損 | 修正済み | 修正済み (**deferred copy で cpy_tensor 再有効化**) |
| 9 | get_tensor stale data | 修正済み | 修正済み (変更なし) |
| 10 | RDMA Completion タイムアウト | 修正済み | 修正済み (変更なし) |
| 11 | サーバーコマンド受信タイムアウト | 修正済み | 修正済み (変更なし) |
| 12 | graph_reserve OOM Segfault | 修正済み | 修正済み (変更なし) |
| 13 | fit_params OOM フォールバック | 修正済み | 修正済み (変更なし) |
| 14 | Per-device 接続 MTT オーバーフロー | ワークアラウンド | ワークアラウンド (変更なし) |
| **15** | **ホットパス fprintf 二峰性 tg** | **(未発見)** | **修正済み** (fprintf → RDMA_LOG_DBG `5b6e0bf56`) |
| **16** | **Deferred copy flush 競合** | **(未実装)** | **修正済み** (flush → copy 実行順序 `1d2c9cbdd`) |

---

## ワークツリー一覧とマージ状況

全 **18 ワークツリー** (master 除く) の状態:

### メインブランチ

| ワークツリー | ブランチ | 目的 | 備考 |
|-------------|---------|------|------|
| `rdma-backend` | `feature/rdma-backend` | メイン開発ブランチ | HEAD (全コミットの統合先) |

### コード変更なし (2件)

| ワークツリー | ブランチ | 目的 | 備考 |
|-------------|---------|------|------|
| `bimodal-investigation` | `bimodal-investigation` | 二峰性 tg 調査 | 原因特定 → `fix-bimodal-fprintf` での修正に直結 |
| `hybrid-split-research` | `research/hybrid-split` | Hybrid parallelism 調査 | 推奨アクションが #10/#11 に直結 |

### マージ済み (1件)

| ワークツリー | ブランチ | 目的 | 備考 |
|-------------|---------|------|------|
| `fix-bimodal-fprintf` | `fix/bimodal-fprintf` | fprintf バグ修正 | `5b6e0bf56` でマージ済み、tg +10% 安定化 |

### 未マージ (14件)

| ワークツリー | ブランチ | 目的 | 未マージの理由 |
|-------------|---------|------|---------------|
| `opt-all` | `opt-all` | 最適化実験統合 | 実験コード、本採用見送り |
| `opt-combined-send` | `opt-combined-send` | ヘッダ+データ統合送信 | 実験コード、本採用見送り |
| `opt-cq-backoff` | `opt-cq-backoff` | CQ ポーリングバックオフ | 実験コード、本採用見送り |
| `opt-doorbell-batch` | `opt-doorbell-batch` | Doorbell batching | 実験コード、本採用見送り |
| `opt-prealloc-buf` | `opt-prealloc-buf` | バッファプリアロケーション | 実験コード、本採用見送り |
| `opt-prepost-recv` | `opt-prepost-recv` | Pre-posted recv | 実験コード、本採用見送り |
| `opt-uncommitted` | `opt-uncommitted` | 最適化実験インフラ | 実験インフラ、本採用見送り |
| `rdma-hotpath` | `feature/rdma-hotpath` | ホットパスメモリ最適化 | 実験コード、本採用見送り |
| `rdma-deferred-copy` | `feature/rdma-deferred-copy` | Deferred copy 実装 | 機能コードマージ済み (`1d2c9cbdd`)、開発スクリプト未マージ |
| `rdma-synchronize` | `feature/rdma-synchronize` | synchronize 実装 | 開発コード、未テスト |
| `rdma-xdev-cache` | `feature/rdma-xdev-cache` | Cross-device cache | 実験コード、本採用見送り |
| `rdma-parallel-compute` | `feature/rdma-parallel-compute` | Parallel compute dispatch | e2e 改善なし (クライアント sched 逐次制約) |
| `draft-spec-test` | `draft-spec-test` | Draft model speculative test | テストコード、未採用 |
| `ngram-spec-test` | `ngram-spec-test` | Ngram speculative test | テストコード、未採用 |

### マージ状況サマリー

```
メイン:          1 ワークツリー (rdma-backend)
コード変更なし:  2 ワークツリー (調査のみ)
マージ済み:      1 ワークツリー (fix-bimodal-fprintf)
未マージ:       14 ワークツリー (実験/開発コード、本採用見送りまたは未テスト)
```

---

## 残タスク (更新版)

| # | タスク | v3 状態 | v4 状態 | 優先度 | ブロッカー |
|---|--------|---------|---------|:------:|-----------|
| 1〜3 | (Step 1-3 タスク) | 完了 | 完了 | — | — |
| 4 | 16GPU ハードウェア準備 | 待ち | **待ち** (変更なし) | 高 | 物理 GPU 追加 |
| 5 | Q3_K_M/Q4_K_M ダウンロード・テスト | 未着手 | **未着手** (16GPU 待ち) | 高 | タスク 4 |
| 6 | 16GPU 全体テスト | 未着手 | **未着手** (タスク 4,5 依存) | 高 | タスク 4, 5 |
| 7 | マルチ RDMA デバイス非同期化 | 未着手 | **調査完了** (two-phase loop 設計) | 低 | なし |
| 8 | RDMA 性能改善 Phase 1-3 | 調査完了・未実装 | **Phase 1 実装済み、Phase 2-3 未実装** | 中 | なし |
| **9** | **Two-phase loop (e2e 改善の鍵)** | (未計画) | **設計完了・synchronize 実装済み** | 中 | `ggml_backend_sched` 変更 |

### タスク #8 RDMA 性能改善 Phase 進捗

| Phase | 内容 | v3 状態 | v4 状態 |
|-------|------|---------|---------|
| Phase 1 | graph_compute fire-and-forget | 未実装 | **実装済み** (`e26e01a0e`) — e2e なし |
| Phase 2 | RDMA Write Immediate + SRQ | 未実装 | 未実装 |
| Phase 3 | パイプライン化 (非同期 graph_compute) | 未実装 | 未実装 |

### タスク #9 Two-phase loop の必要性

- **問題**: 現在のクライアントは graph_compute → get_tensor を **デバイスごとに逐次**実行
- **原因**: `ggml_backend_sched` が各スプリットで compute → get_tensor を直列呼び出し
- **解決策**: Phase 1 で全デバイスに ASYNC dispatch → Phase 2 で全 get_tensor を回収
- **現状**: `RDMA_CMD_SYNC` と `synchronize()` を実装済み。`ggml_backend_sched` 側の変更が必要
- **効果見込み**: Layer split では小さい (各デバイスの計算が依存関係あり)。Row split では大きい (~50% 並列化)

---

## コード統計 (更新版)

### ファイル別行数

| ファイル | v3 行数 | v4 行数 | 差分 | 役割 |
|---------|-------:|-------:|----:|------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 3,323 | 3,735 | +412 | メイン実装 (async, deferred copy, synchronize) |
| `ggml/src/ggml-rdma/rdma-transport.cpp` | 968 | 968 | 0 | RDMA 接続管理 |
| `ggml/src/ggml-rdma/rdma-gdr.cpp` | 444 | 444 | 0 | GPUDirect RDMA |
| `ggml/src/ggml-rdma/rdma-memory.cpp` | 307 | 307 | 0 | ホストメモリ管理 |
| `ggml/src/ggml-rdma/rdma-transport.h` | 186 | 187 | +1 | トランスポート層ヘッダ |
| `ggml/src/ggml-rdma/rdma-gdr.h` | 116 | 116 | 0 | GPUDirect ヘッダ |
| `ggml/src/ggml-rdma/rdma-memory.h` | 109 | 109 | 0 | メモリ管理ヘッダ |
| `ggml/src/ggml-rdma/CMakeLists.txt` | 70 | 70 | 0 | ビルド設定 |
| `ggml/include/ggml-rdma.h` | — | 79 | — | 公開 C API |
| `tools/rdma/rdma-server.cpp` | — | 283 | — | サーバーエントリポイント |
| **RDMA コア合計** | **5,523** | **6,298** | **+775** | **10 ファイル** |

### ヘルパースクリプト

| スクリプト | v3 行数 | v4 行数 | 差分 |
|-----------|-------:|-------:|----:|
| `scripts/rdma-build.sh` | 39 | 40 | +1 |
| `scripts/rdma-deploy.sh` | 20 | 21 | +1 |
| `scripts/rdma-server.sh` | 60 | 60 | 0 |
| **合計** | **119** | **121** | **+2** |

### コミット・レポート統計

| 指標 | v3 | v4 | 差分 |
|------|---:|---:|-----:|
| RDMA 関連コミット数 (ソースコード変更) | 25 | 18* | — |
| 全コミット数 (feature/rdma-backend) | — | 32** | — |
| レポート数 (MD ファイル) | 49 | 66 | +17 |
| git diff insertions (vs master) | 19,168 | 25,143 | +5,975 |
| 変更ファイル数 (vs master) | 66 | 94 | +28 |

*\* RDMA ソースディレクトリに対するコミットのみ*
*\*\* docs コミット含む全体*

---

## プロジェクト全体の振り返り

### 14日間の成果総括

2026年1月31日の実装開始から2月13日まで、**14日間** で RDMA バックエンドを `ゼロから安定動作+最適化` まで進めた。

| マイルストーン | 日付 | 成果 |
|---------------|------|------|
| 初回実装 | 01/31 | RDMA バックエンド骨格 (feat commit `2c54d7603`) |
| 初回動作 | 02/01 | qwen2.5-0.5b で初の分散推論成功 (0.3 t/s) |
| プロトコル最適化 | 02/04 | 0.3 → 148.5 t/s (495 倍高速化) |
| マルチノード安定化 | 02/04 | gpt-oss-120b が 11GPU クラスタで動作 |
| GPUDirect RDMA | 02/05 | GPU VRAM 直接 RDMA 転送 (pp128: 6.6 倍高速) |
| RNR NAK 修正 | 02/05 | 2 秒スパイク解消、RDMA が RPC を上回る |
| GLM-4.7 動作 | 02/07 | 470 億パラメータモデルが 11GPU で推論 |
| バグ修正完了 (v3) | 02/09 | 全 14 件のバグが修正済み |
| **Async compute** | **02/10** | **graph_compute 24.5× 高速化** |
| **最適化実験 (6件)** | **02/10〜11** | **IB レベル最適化の限界を確認** |
| **Deferred copy** | **02/13** | **サーバーローカル D2H+H2D、gpt-oss-20b tg +13%** |
| **二峰性 tg 修正** | **02/13** | **7.4-7.5 t/s 安定 (RDMA ≈ RPC)** |

### 性能進化の軌跡

```
Day 1  (02/01): 0.3 t/s    — 初回動作 (naive protocol)
Day 4  (02/04): 148.5 t/s  — プロトコル最適化完了 (495×)
Day 5  (02/05): 59.5 t/s   — GPUDirect RDMA (20b, 92.6% of local)
Day 7  (02/07): 6.8 t/s    — GLM-4.7 47B on 11 GPUs
Day 10 (02/09): 7.2 t/s    — 最良値 (安定化後) ← v3 時点
Day 14 (02/13): 7.5 t/s    — 二峰性修正 + deferred copy ← v4 時点
```

### v3 以降の主要な技術的困難と解決策

| 困難 | 根本原因 | 解決策 | 教訓 |
|------|---------|--------|------|
| 二峰性 tg 分布 | ホットパスの `fprintf(stderr)` が stdio バッファフラッシュでスパイク | always-on fprintf を debug-gated `RDMA_LOG_DBG` に置換 | **I/O をホットパスに入れない** |
| Deferred copy flush 競合 | バウンディングボックスの stale staging データが copy 先を上書き | flush → copy の実行順序を強制 | **状態更新の順序依存に注意** |
| Async compute e2e 改善なし | get_tensor の暗黙的同期で savings が吸収される | Two-phase loop 設計 (synchronize 実装) | **局所最適化では全体最適にならない** |
| Parallel dispatch e2e 改善なし | クライアント `ggml_backend_sched` が逐次 | RDMA バックエンド単独では解決不可 | **ボトルネックの所在を正確に特定してから最適化** |

### 最適化実験から得られた教訓

v3 以降、6件のネットワーク/メモリ最適化実験を実施。すべて e2e 改善なしという結果だが、重要な知見を得た:

| 実験 | 予想効果 | 実測効果 | 教訓 |
|------|---------|---------|------|
| Doorbell batching | ~10µs/token | 無視可能 | GPU compute (~17ms) が支配的な時、µs オーダーの NIC 最適化は無意味 |
| Pre-posted recv | RNR NAK 削減 | 変化なし | RNR NAK の主因はヘッダ↔データ間隔、ASYNC compute ではない |
| Combined send | Send/Recv 半減 | tg 低下 | recv バッファ拡張のオーバーヘッドが削減効果を上回る |
| Hotpath memory | malloc 削減 | 変化なし | malloc/free は GPU compute 時間の 0.01% 未満 |
| xdev cache | cudaAlloc 削減 | 変化なし | Layer split では xdev テンソルが稀 |
| Parallel dispatch | サーバー並列化 | サーバー正常/e2e なし | ボトルネックはクライアント側 (sched の逐次性) |

**総括**: P100 + ConnectX-4 環境では、**GPU 計算時間 (17ms/token)** と **クライアント側スケジューラの逐次性** が支配的ボトルネック。IB レベルのマイクロ最適化 (µs オーダー) は効果がない。今後の改善は **クライアント側アーキテクチャ変更** (two-phase loop, speculative decoding) に焦点を当てるべき。

---

## 全レポート一覧 (更新版)

v3 の 49 本 → v4 含め **66 本** (+ 補助ドキュメント 3 本)

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
| **50** | **02/09** | [2026-02-09_084919](2026-02-09_084919_github_deploy_key_setup.md) | **GitHub Deploy Key セットアップ** | **Infra** |
| **51** | **02/10** | [2026-02-10_214841](2026-02-10_214841_rdma_async_compute_optimization.md) | **Async compute 最適化** | **5** |
| **52** | **02/10** | [2026-02-10_222348](2026-02-10_222348_doorbell_batching_max_inline.md) | **Doorbell batching + max_inline** | **実験** |
| **53** | **02/11** | [2026-02-11_040659](2026-02-11_040659_prepost_recv_rnr_nak.md) | **Pre-posted recv (RNR NAK)** | **実験** |
| **54** | **02/11** | [2026-02-11_045619](2026-02-11_045619_combined_send_rnr_nak.md) | **Combined send (ヘッダ統合)** | **実験** |
| **55** | **02/11** | [2026-02-11_095500](2026-02-11_095500_hotpath_memory_optimization.md) | **Hotpath memory 最適化** | **実験** |
| **56** | **02/11** | [2026-02-11_133000](2026-02-11_133000_xdev_cache_optimization.md) | **Cross-device cache 最適化** | **実験** |
| **57** | **02/11** | [2026-02-11_214242](2026-02-11_214242_parallel_compute_dispatch.md) | **Parallel compute dispatch** | **5** |
| **58** | **02/12** | [2026-02-12_070600](2026-02-12_070600_hybrid_parallelism_research.md) | **Hybrid parallelism research** | **調査** |
| **59** | **02/12** | [2026-02-12_134623](2026-02-12_134623_ngram_speculative_decoding_test.md) | **Ngram speculative decoding test** | **実験** |
| **60** | **02/12** | [2026-02-12_160000](2026-02-12_160000_draft_speculative_decoding_test.md) | **Draft model speculative decoding test** | **実験** |
| **61** | **02/13** | [2026-02-13_044038](2026-02-13_044038_rdma_backend_comprehensive_review.md) | **RDMA バックエンド包括的レビュー** | **調査** |
| **62** | **02/13** | [2026-02-13_053056](2026-02-13_053056_deferred_copy_optimization.md) | **Deferred copy 最適化** | **5** |
| **63** | **02/13** | [2026-02-13_053658](2026-02-13_053658_rdma_synchronize_implementation.md) | **RDMA synchronize 実装** | **5** |
| **64** | **02/13** | [2026-02-13_090026](2026-02-13_090026_deferred_copy_statistical_benchmark.md) | **Deferred copy 統計的ベンチマーク** | **5** |
| **65** | **02/13** | [2026-02-13_153100](2026-02-13_153100_bimodal_tg_investigation.md) | **二峰性 tg 分布調査** | **バグ** |
| **66** | **02/13** | [2026-02-13_185255](2026-02-13_185255_rdma_backend_retrospective_v4.md) | **振り返りレポート v4 (本レポート)** | **—** |

### レポート数とファイル数

- report/ ディレクトリには通し番号付き .md レポート **66 本** + 補助 .md **1 本** + .html **2 本** = 計 **69 ファイル** が存在
- 通し番号は .md レポートのみを対象 (#1〜#66)
- .html は .md の Mermaid 図レンダリング版、補助 .md (`llama_cli_inference_pipeline.md`) はパイプライン解説ドキュメント

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
