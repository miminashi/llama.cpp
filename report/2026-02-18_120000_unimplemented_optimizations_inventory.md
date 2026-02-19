# 未実施最適化の棚卸しレポート

- **実施日時**: 2026年2月18日 12:00
- **ワークツリー**: `.worktree/rdma-backend`

## 前提・目的

過去のレポートおよび CLAUDE.md で計画・提案された最適化のうち、未実施のものを棚卸しする。各項目について現状の評価、実施の是非、過去レポートへの参照を記載する。

## 概要

| # | カテゴリ | 項目 | 評価 | 参照レポート |
|---|---------|------|------|-------------|
| 1 | 性能 | Two-phase loop (Plan A) | 効果なし (検証済み) | #61, #63, #67 |
| 2 | 性能 | Backend interface 拡張 (Plan B) | 未実施 (upstream 障壁) | #61 |
| 3 | 性能 | Per-device connection (Plan C) | 実装済み/無効 (HW制限) | #61, #46 |
| 4 | 性能 | Pipeline with lookahead (Plan D) | 未実施 (高複雑度) | #61 |
| 5 | 性能 | synchronize 実装 | 効果なし (検証済み) | #63, #67, #68 |
| 6 | 性能 | Speculative decoding | 効果限定 (検証済み) | #59, #60 |
| 7 | 品質 | alloc_host_staging エラー伝播 | 未実施 | #61 |
| 8 | 品質 | クライアント異常切断の接続回復 | 未実施 | CLAUDE.md |
| 9 | インフラ | テストインフラ拡充 | 未実施 | #61 |
| 10 | インフラ | スクリプトの16GPU対応 | 未実施 | #61 |

---

## 1. 性能最適化

### 1.1 Generation 速度改善 (RPC 比劣位の解消)

**背景**: RDMA は全デバイスが1つの QP を共有し graph_compute が逐次実行されるため、Generation 速度で RPC (デバイスごとの独立ソケット) に劣る。GLM-4.7 IQ2_M 11GPU で RDMA 6.8 t/s vs RPC 7.5 t/s (RPC が +10%)。

参照: [#61 セクション2: Generation 速度改善の設計調査](2026-02-13_044038_rdma_backend_comprehensive_review.md)

#### (A) Two-phase loop — 効果なし (検証済み)

スケジューラの split ループを2フェーズに分離し、Phase 1 で全デバイスに graph_compute を fire-and-forget、Phase 2 で結果回収する案。

**検証結果**: layer split ではスプリット間にデータ依存 (split N+1 の入力 = split N の出力) があるため、Phase 1 内で `ggml_backend_synchronize(input_backend)` が呼ばれ逐次化する。実測でも有意差なし (p=0.69〜1.00)。

| 参照レポート |
|-------------|
| [#61 案(A): Two-phase loop](2026-02-13_044038_rdma_backend_comprehensive_review.md) — 設計提案 |
| [#63 synchronize 実装 + Two-phase loop](2026-02-13_053658_rdma_synchronize_implementation.md) — 実装と理論分析 |
| [#68 synchronize 交絡排除再測定](2026-02-14_055914_synchronize_confound_free_remeasurement.md) — B vs C で効果なしを確定 |

**評価**: layer split では原理的に効果がない。row split を実装する場合のみ有効だが、P100 に NVLink がなく row split は性能が出ないことが確認済みのため、**実施不要**。

#### (B) Backend interface 拡張 — 未実施

`graph_compute_prepare()` / `graph_compute_await()` を upstream の ggml バックエンドインターフェースに追加する案。

| 参照レポート |
|-------------|
| [#61 案(B): Backend interface 拡張](2026-02-13_044038_rdma_backend_comprehensive_review.md) — 設計提案 |

**評価**: llama.cpp upstream の API 変更が必要でマージの障壁が極めて高い。また、layer split ではデータ依存の問題は解消されないため、Plan A と同じ制約を受ける。**実施不要** (upstream マージを目指す場合を除く)。

#### (C) Per-device connection — 実装済み、デフォルト無効

各デバイスが独立 QP を持つことで RPC と同等のパイプライン効果を得る案。`GGML_RDMA_PER_DEVICE_CONN=1` で有効化可能。

| 参照レポート |
|-------------|
| [#61 案(C): Per-device connection](2026-02-13_044038_rdma_backend_comprehensive_review.md) — 設計提案 |
| [#46 サーバー堅牢化 & Per-device connection](2026-02-08_020500_server_robustness_and_per_device_connections.md) — 実装 |

**評価**: ConnectX-4 の MTT キャッシュ制限 (~10-16GB) により、大規模モデルで RDMA Write がタイムアウトする。ConnectX-6+ では有効だが、現環境では使用不可。**HW アップグレード待ち**。

#### (D) Pipeline with lookahead — 未実施

D_i の get_tensor と D_{i+1} の graph_compute をオーバーラップさせる案。

| 参照レポート |
|-------------|
| [#61 案(D): Pipeline with lookahead](2026-02-13_044038_rdma_backend_comprehensive_review.md) — 設計提案 |

**評価**: `rdma_connection` の送受信をマルチスレッド化する必要があり、実装複雑度が高い。layer split ではレイヤー N の出力がレイヤー N+1 の入力になるため、compute と get_tensor の重畳が可能な区間は限定的。**費用対効果が低く、実施優先度は低い**。

### 1.2 synchronize (RDMA_CMD_SYNC) — 効果なし (検証済み)

`synchronize()` を NO-OP から `RDMA_CMD_SYNC` 送信に変更し、`compute_pending_` をドレインして get_tensor の RDMA Read 高速パスを有効化する案。

**検証結果**: 交絡因子を排除した A/B テスト (n=15) で有意差なし。

| 比較 | 指標 | 差 | p 値 | 判定 |
|------|------|---:|-----:|------|
| A' vs B | pp128 | -0.01% | 0.565 | n.s. |
| A' vs B | tg32 | -0.33% | 0.051 | n.s. |

| 参照レポート |
|-------------|
| [#63 synchronize 実装](2026-02-13_053658_rdma_synchronize_implementation.md) — 実装 |
| [#67 synchronize 二峰性修正マージ & 効果検証](2026-02-14_024402_synchronize_bimodal_fix_and_benchmark.md) — 初回測定 (交絡あり) |
| [#68 synchronize 交絡排除再測定](2026-02-14_055914_synchronize_confound_free_remeasurement.md) — 交絡排除測定 (効果なし確定) |

**評価**: SYNC コマンドは性能に影響しない。コードの正確性向上 (compute_pending_ のドレイン) の観点で有用だが、性能最適化としては**効果なし**。ワークツリー `rdma-synchronize` に実装が残っているが、**マージ不要**。

### 1.3 Speculative decoding — 効果限定 (検証済み)

ドラフトモデルまたは ngram による投機的デコーディング。

| 手法 | モデル | 結果 | 参照レポート |
|------|--------|------|-------------|
| Ngram | GLM-4.7 (MoE) | acceptance rate 4-11%, tg -7.8% | [#59](2026-02-12_134623_ngram_speculative_decoding_test.md) |
| Draft (GLM-4-9B) | GLM-4.7 | acceptance 56-60%, tg -2〜-6% | [#60](2026-02-12_160000_draft_speculative_decoding_test.md) |
| Draft (qwen2.5-0.5b) | GLM-4.7 | acceptance 50-52%, tg +3.9% | [#60](2026-02-12_160000_draft_speculative_decoding_test.md) |

**評価**: MoE モデル (GLM-4.7) では expert routing の確率的性質により投機の当たりが悪い。qwen2.5-0.5b ドラフトで +3.9% だが、ドラフトモデル用の VRAM 確保が16GPU 展開を複雑にする。**現時点では実施不要**。

### 1.4 過去に検証済みで効果なしだった最適化

以下は既にワークツリーで実装・テスト済みだが、e2e 改善が確認されず未マージのもの。

| 最適化 | 結果 | 参照レポート | ワークツリー |
|--------|------|-------------|-------------|
| Doorbell batching + max_inline | ~10µs/token 削減、e2e 効果なし | [#52](2026-02-10_222348_doorbell_batching_max_inline.md) | `opt-doorbell-batch` |
| Pre-posted recv (RNR NAK 削減) | RNR NAK 変化なし | [#53](2026-02-11_040659_prepost_recv_rnr_nak.md) | `opt-prepost-recv` |
| Combined send (ヘッダ+データ統合) | RNR NAK 6%減、tg 低下 | [#54](2026-02-11_045619_combined_send_rnr_nak.md) | `opt-combined-send` |
| Hotpath memory 最適化 | コード品質向上、e2e 効果なし | [#55](2026-02-11_095500_hotpath_memory_optimization.md) | `rdma-hotpath` |
| Cross-device cache | xdev 0.09→0.02ms、e2e 効果なし | [#56](2026-02-11_133000_xdev_cache_optimization.md) | `rdma-xdev-cache` |
| Parallel compute dispatch | サーバー側並列化OK、クライアント逐次で e2e 効果なし | [#57](2026-02-11_214242_parallel_compute_dispatch.md) | `rdma-parallel-compute` |
| Async compute | graph_compute 24.5x 高速化、e2e 効果なし | [#51](2026-02-10_214841_rdma_async_compute_optimization.md) | (rdma-backend にマージ済み) |

---

## 2. コード品質・ロバスト性

### 2.1 alloc_host_staging() エラー伝播 — 未実施

サーバー側 `alloc_host_staging()` が `void` 返り値で、メモリ確保失敗・MR 登録失敗時にエラーを呼び出し元に伝播できない。

| 参照レポート |
|-------------|
| [#61 セクション1.3: エラーハンドリング](2026-02-13_044038_rdma_backend_comprehensive_review.md) — 問題の特定と改善案 |

**改善案**: `bool` 返り値に変更し、失敗時は呼び出し元で Send/Recv フォールバックを選択。

**評価**: OOM 以外で発生しないパスのため、影響は低い。16GPU 展開で VRAM 制約が厳しくなる場合に重要度が上がる可能性あり。**優先度 Low**。

### 2.2 クライアント異常切断後の接続回復 — 未実施

クライアントが kill -9 等で異常切断した場合、サーバーの QP 状態が壊れ、再起動が必要になることがある。

| 参照 |
|------|
| [CLAUDE.md: クライアント異常切断後のサーバー復旧](../CLAUDE.md) — 残課題として記載 |
| [#46 サーバー堅牢化](2026-02-08_020500_server_robustness_and_per_device_connections.md) — Phase A (正常切断) は実装済み |

**現状**: 正常切断 (クライアントが rdma_disconnect を送信) からの復旧は Phase A で実装済み。異常切断 (RDMA CM DISCONNECTED イベントのみ) からの復旧は未実装。

**評価**: 開発時の利便性に影響するが、本番運用では手動再起動で対応可能。**優先度 Low**。

---

## 3. インフラ

### 3.1 テストインフラ拡充 — 未実施

既存テスト (`rdma-simple-test.cpp`, `rdma-test-client.cpp`) は単一デバイスの基本 set/get のみをカバー。

| 参照レポート |
|-------------|
| [#61 セクション4: テストインフラの現状と提案](2026-02-13_044038_rdma_backend_comprehensive_review.md) — ギャップ分析と提案 |

**不足しているテスト**:

| テスト | 優先度 | 関連する既知バグ |
|--------|--------|----------------|
| マルチデバイス同時アクセス | High | Multi-RDMA 出力破損 |
| 大バッファ (4GB 境界) | High | mmap + RDMA Write |
| GDR バジェット枯渇 | Medium | GPUDirect timeout |
| 接続回復 | Medium | 異常切断 |
| graph_compute (cache hit/miss) | Medium | graph_cache 整合性 |
| 自動テスト基盤 (CI/CD) | Low | — |

**評価**: 機能は安定しているため緊急度は低いが、16GPU 展開時の回帰テストに有用。**優先度 Medium**。

### 3.2 スクリプトの16GPU対応 — 未実施

`NODE2`, `REMOTE_DIR`, `PORT` がハードコードされており、3号機以降の追加に対応できない。

| 参照レポート |
|-------------|
| [#61 セクション3: ビルド・デプロイスクリプトの改善提案](2026-02-13_044038_rdma_backend_comprehensive_review.md) — 改善案4件 |

**改善案**:
1. 環境変数によるパラメータ上書き (優先度 Medium)
2. SSH 接続事前チェック (優先度 Low)
3. rsync dry-run オプション (優先度 Low)
4. サーバーログローテーション (優先度 Low)

**評価**: 16GPU (3ノード以上) に拡張する場合に必須。現在の2ノード構成では不要。**GPU 追加時に対応**。

---

## 4. 総括

### 実施すべき項目

**性能最適化は全案検証済みまたは原理的制約あり**で、layer split + ConnectX-4 の現環境で追加実施すべき項目はない。

| 項目 | 優先度 | 実施タイミング |
|------|--------|--------------|
| スクリプト16GPU対応 | Medium | GPU 追加時 |
| テストインフラ拡充 | Medium | 16GPU 展開前 |
| alloc_host_staging エラー伝播 | Low | コード整理時 |
| 接続回復 (異常切断) | Low | 必要性が発生した時 |

### 実施不要と確定した項目

| 項目 | 理由 |
|------|------|
| Two-phase loop (Plan A) | layer split でデータ依存により効果なし (検証済み) |
| Backend interface 拡張 (Plan B) | upstream 障壁 + layer split の制約 |
| Pipeline with lookahead (Plan D) | 高複雑度 + layer split での効果限定 |
| synchronize (RDMA_CMD_SYNC) | 効果なし (検証済み、p=0.051〜0.565) |
| Speculative decoding | MoE で効果限定 (+3.9% 最大)、VRAM 制約 |

### HW 依存で将来有効化可能な項目

| 項目 | 条件 |
|------|------|
| Per-device connection (Plan C) | ConnectX-6+ へのアップグレード |

### Generation 速度の RPC 比劣位について

現環境 (ConnectX-4 + P100 + layer split) では、RDMA の単一接続逐次実行は**アーキテクチャ的制約**であり、ソフトウェア最適化では解消できない。Plan A〜D のすべてが検証または分析済みで、いずれも layer split では効果がないことが確認された。

一方、Prompt 処理では RDMA Write ゼロコピーにより RPC 比 +19% の優位があり、RDMA バックエンドの価値は prompt-heavy なワークロード (長文入力) で発揮される。
