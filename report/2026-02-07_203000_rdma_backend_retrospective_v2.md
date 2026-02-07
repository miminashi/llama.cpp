# RDMA バックエンド 振り返りレポート v2

- **実施日時**: 2026年2月7日 20:30
- **前回レトロスペクティブ**: [2026-02-07_104703_rdma_backend_retrospective.md](2026-02-07_104703_rdma_backend_retrospective.md)

---

## 前回レトロスペクティブからの主要変更点

前回 (10:47) 以降の午後セッション (12:43〜20:27) で以下の大きな進展があった:

### 修正されたバグ (3件)

| # | バグ | 修正内容 | コミット |
|---|------|---------|---------|
| 1 | **マルチRDMAデバイス出力破損** | `cpy_tensor` を常に `false` 返却に変更。サーバー側GPU間コピーがグラフキャッシュと不整合を起こしていた | `e6cb22ef3` |
| 2 | **GLM-4.7 IQ2_M 出力品質** | 上記 `cpy_tensor` バグが原因だった。量子化の限界ではなく RDMA 固有の問題 | `e6cb22ef3` |
| 3 | **get_tensor stale data** | GDR無効時に RDMA Read パスが stale なステージングバッファを返していた。`no_gdr` チェック追加 | `e6cb22ef3` |

### 完了した調査 (1件)

| # | 調査 | 結論 |
|---|------|------|
| 1 | **cmake --build 無出力問題** | 体系的テスト (8パターン) で再現不可。実際のバグではなく、nvccパス間違い等による configure 失敗が原因と推定 |

### 新たに成功したテスト (2件)

| # | テスト | 結果 |
|---|--------|------|
| 1 | **GLM-4.7 IQ2_M 正常動作** | 7C+4R (11GPU) で4種類のプロンプトすべて正常出力。pp=6.6-9.2, tg=5.9-6.6 t/s |
| 2 | **gpt-oss-120b 11GPUフルクラスタ** | 7C+4R で初の正常動作。pp=36.6-48.4, tg=28.8-29.0 t/s |

### 作成されたレポート (7件)

| # | ファイル | タイトル |
|---|---------|---------|
| 33 | [2026-02-07_124327](2026-02-07_124327_glm47_iq2m_rdma_vs_rpc_output_quality.md) | GLM-4.7 IQ2_M RDMA vs RPC 出力品質比較 |
| 34 | [2026-02-07_173027](2026-02-07_173027_build_procedure_investigation.md) | ビルド手順の安定化調査 |
| 35 | [2026-02-07_175849](2026-02-07_175849_rdma_backend_technical_architecture.md) | RDMA バックエンド技術アーキテクチャ |
| 36 | [2026-02-07_192500](2026-02-07_192500_p100_pcie_p2p_transfer_spec.md) | P100 PCIe P2P転送仕様 |
| 37 | [2026-02-07_195500](2026-02-07_195500_multi_rdma_device_output_corruption_fix.md) | マルチRDMAデバイス出力破損バグ修正 |
| 38 | [2026-02-07_202700](2026-02-07_202700_cpy_tensor_fix_comprehensive_test.md) | cpy_tensor 修正の包括的テスト |
| 39 | [2026-02-07_203000](2026-02-07_203000_rdma_backend_retrospective_v2.md) | 振り返りレポート v2 (本レポート) |

---

## Step 別ステータス (更新版)

| Step | 内容 | 状態 | 前回からの変更 |
|------|------|:----:|:-------------:|
| Step 1 | 基本 RDMA 通信 | ✅ 完了 | なし |
| Step 2 | マルチノードクラスタ安定化 | ✅ 完了 | なし |
| Step 3 | RDMA 性能最適化 | ✅ 完了 | なし |
| Step 4 | GPUDirect RDMA 有効化 | ✅ 完了 | なし |
| Step 5 | GLM-4.7 Q4 on 16 P100s | 🔧 進行中 | 進捗あり (下記) |

### Step 5 の進捗更新

| マイルストーン | 前回 | 現在 |
|---------------|------|------|
| GLM-4.7 IQ2_M 動作 | 動作するがゴミ出力 | **正常動作** (cpy_tensor修正) |
| GLM-4.7 出力品質 | RDMA固有バグの疑い | **解決** (原因特定・修正済み) |
| gpt-oss-120b 11GPUクラスタ | 未テスト (cpy_tensor修正前) | **正常動作確認** |
| 16GPU ハードウェア | 待ち | 待ち (変更なし) |
| Q3_K_M/Q4_K_M テスト | 未着手 | 未着手 |

---

## 既知のバグと対応状況 (更新版)

| # | バグ | 前回状態 | 現在状態 | ワークアラウンド |
|---|------|---------|---------|-----------------|
| 1 | GPUDirect timeout on large models | 未修正 | **未修正** | `GGML_RDMA_NO_GDR=1` |
| 2 | GLM-4.7 IQ2_M 出力品質 | 要調査 | **修正済み** (cpy_tensorバグが原因) | — |
| 3 | RDMA デバイス逐次実行 | 設計上の制約 | **設計上の制約** (変更なし) | 将来の非同期化 |
| 4 | cmake --build 無出力 | ビルドシステム | **解決** (実際のバグではなかった) | — |
| 5 | mmap + RDMA Write overflow | 修正済み (未コミット) | **修正済み・コミット済み** (`3f3b3c67d`) | — |
| 6 | RNR NAK 2秒スパイク | 修正済み | 修正済み (変更なし) | — |
| 7 | supports_buft クロスデバイス | 修正済み | 修正済み (変更なし) | — |
| **8** | **マルチRDMAデバイス出力破損** | (午前時点で未発見) | **修正済み・コミット済み** (`e6cb22ef3`) | — |
| **9** | **get_tensor stale data** | (午前時点で未発見) | **修正済み・コミット済み** (`e6cb22ef3`) | — |

### バグ #8: マルチRDMAデバイス出力破損の詳細

- **症状**: 2台以上のRDMAデバイスを使用すると出力が破損
  - GLM-4.7 IQ2_M (7C+4R): ゴミ thinking トークン
  - gpt-oss-20b (1C+2R): 空出力 (EOSのみ)
- **根本原因**: `cpy_tensor` がサーバー側GPU間コピー (`cudaMemcpyPeer`) を実行するが、`graph_recompute` のキャッシュされたグラフが参照するメモリアドレスとコピー先が不一致。コピーされたデータが計算に使用されない
- **RPCとの違い**: RPC はデバイスごとに別ソケット → `cpy_tensor` は常に `false` → 問題が顕在化しない。RDMA は1接続を共有 → `cpy_tensor` が実行される
- **修正**: `cpy_tensor` を常に `false` を返すよう変更し、`get_tensor + set_tensor` フォールバックに統一
- **性能影響**: 約3%低下 (gpt-oss-20b tg: 47.1 → 46.3 t/s)
- **診断手法**: logits の argmax 分析でダイナミックレンジ低下を検出 (正常: ~26-49 vs 異常: ~13-20)

### バグ #9: get_tensor stale data の詳細

- **症状**: GDR無効 (`GGML_RDMA_NO_GDR=1`) + RDMA Read パス → stale なステージングバッファデータを読み出し
- **原因**: `get_tensor` の RDMA Read パスが `no_gdr` 環境変数をチェックしていなかった。CPU ステージング使用時はステージングバッファにコンピュート結果が反映されないため、RDMA Read は stale データを返す
- **修正**: `no_gdr` フラグチェックを追加し、GDR無効時は Send/Recv フォールバックを使用

---

## 性能サマリー (更新版)

### Step ごとのベストスコア

| Step | 構成 | モデル | pp128 (t/s) | tg32 (t/s) | ローカル比 (tg) |
|------|------|--------|:-----------:|:----------:|:--------------:|
| Step 1 完了 | RDMA 1+1 | qwen2.5-0.5b | — | 148.5 | 75.7% |
| Step 2 完了 | 11GPU (7C+4R) | gpt-oss-120b | 3〜11 | 0.7〜1.1 | 1.5〜2.3% |
| Step 3 完了 | RDMA 1+1 | gpt-oss-20b | 61.09 | 58.47 | **91.0%** |
| Step 4 完了 (GDR) | RDMA 1+1 | gpt-oss-20b | **403.69** | 59.52 | **92.6%** |
| Step 4 完了 (クラスタ) | 11GPU (7C+4R) | gpt-oss-120b | 201.02 | 36.92 | 84.7% |
| Step 5 事前検証 | 11GPU (7C+4R) | GLM-4.7 IQ2_M | 6.6〜9.2 | 5.9〜6.6 | — |
| **Step 5 事前検証** | **11GPU (7C+4R)** | **gpt-oss-120b** | **36.6〜48.4** | **28.8〜29.0** | — |

### 午後セッションで追加された性能データ

#### GLM-4.7 IQ2_M (7C+4R, 11GPU) — cpy_tensor修正後

| # | プロンプト | pp (t/s) | tg (t/s) |
|---|-----------|:--------:|:--------:|
| 1 | "The capital of France is" | 6.6 | 5.9 |
| 2 | "Explain quantum computing..." | 8.1 | 6.6 |
| 3 | 「こんにちは。日本の首都はどこですか？」 | 8.7 | 6.3 |
| 4 | "Write a Python function..." (コード生成) | 9.2 | 6.6 |

#### gpt-oss-120b Q4_K_M (7C+4R, 11GPU) — 初の正常動作

| # | プロンプト | pp (t/s) | tg (t/s) |
|---|-----------|:--------:|:--------:|
| 1 | "The capital of France is" | 36.6 | 28.8 |
| 2 | 「こんにちは」 | 48.4 | 29.0 |

### RDMA vs RPC 最終比較 (RNR修正後)

| モデル | 構成 | RDMA tg32 | RPC tg32 | RDMA 優位 |
|--------|------|:---------:|:--------:|:---------:|
| gpt-oss-20b | 1+1 | 52.39 | 50.66 | +3.3% |
| gpt-oss-20b | 2+2 | 56.92 | 36.79 | **+54.7%** |
| gpt-oss-120b | 7+2 | 41.23 | 29.46 | **+40.0%** |
| gpt-oss-120b | 6+2 | 40.23 | 28.11 | **+43.1%** |

---

## 残タスク (更新版)

| # | タスク | 前回状態 | 現在状態 | 優先度 | ブロッカー |
|---|--------|---------|---------|:------:|-----------|
| 1 | mmap バグ修正コミット | 未コミット | **完了** (`3f3b3c67d`) | — | — |
| 2 | GLM-4.7 出力品質 RDMA vs RPC 比較 | 計画済み | **完了** (実施済み、原因特定) | — | — |
| 3 | GPUDirect timeout 調査・修正 | 未着手 | **未着手** (優先度下げ可) | 低 | なし |
| 4 | 16GPU ハードウェア準備 | 待ち | **待ち** (変更なし) | 高 | 物理GPU追加 |
| 5 | Q3_K_M/Q4_K_M ダウンロード・テスト | 未着手 | **未着手** (16GPU待ち) | 高 | タスク4 |
| 6 | 16GPU 全体テスト | 未着手 | **未着手** (タスク4,5依存) | 中 | タスク4,5 |
| 7 | マルチ RDMA デバイス非同期化 | 未着手 | **未着手** (将来課題) | 低 | なし |

### タスク完了の詳細

- **タスク1** (mmap修正コミット): `3f3b3c67d` でコミット済み。4GB バッファサイズ制限 + Send/Recv フォールバック + 16MB チャンク RDMA Write
- **タスク2** (出力品質比較): [レポート](2026-02-07_124327_glm47_iq2m_rdma_vs_rpc_output_quality.md)で RPC=正常、RDMA=ゴミを確認。原因は `cpy_tensor` バグと特定し、[修正レポート](2026-02-07_195500_multi_rdma_device_output_corruption_fix.md)で解決

### タスク3 (GPUDirect timeout) の優先度下げの理由

- 現状 `GGML_RDMA_NO_GDR=1` ワークアラウンドで動作に支障なし
- GLM-4.7 IQ2_M で pp=6.6-9.2, tg=5.9-6.6 t/s が出ており、実用レベル
- GPUDirect 有効化は性能改善であり、16GPU拡張 (タスク4-6) の方が Step 5 完了に直結

### 量子化レベル選択 (変更なし)

| 量子化 | bpw | 推定サイズ | 16GPU (256GB) | 11GPU (176GB) |
|--------|----:|----------:|:-------------:|:-------------:|
| IQ2_M | 2.7 | 114 GB | 収容可能 | 収容可能 (品質△) |
| Q3_K_M | ~3.5 | ~150 GB | 収容可能 | 収容不可 |
| Q4_K_M | ~4.5 | ~195 GB | 収容可能 | 収容不可 |

Q3_K_M (150GB) が 16GPU での最有力候補。IQ2_M は 11GPU でも動作するが品質が低い (2.7 bpw)。

---

## 全レポート一覧 (更新版)

前回 32本 → 本レポート含め **39本** (+ HTML 1本, パイプラインドキュメント 2本)

| # | 日付 | ファイル | タイトル | Step |
|---|------|---------|---------|:----:|
| 1 | 01/31 | [2026-01-31_214933_gpudirect_rdma_feasibility.md](2026-01-31_214933_gpudirect_rdma_feasibility.md) | GPUDirect RDMA 実装可能性調査 | 1 |
| 2 | 01/31 | [2026-01-31_222100_gpudirect_rdma_backend_implementation.md](2026-01-31_222100_gpudirect_rdma_backend_implementation.md) | GPUDirect RDMA バックエンド実装 | 1 |
| 3 | 02/01 | [2026-02-01_104744_rdma_backend_test.md](2026-02-01_104744_rdma_backend_test.md) | RDMA バックエンド実行テスト・修正 | 1 |
| 4 | 02/03 | [2026-02-03_134920_pcie_topology_verification.md](2026-02-03_134920_pcie_topology_verification.md) | PCIeトポロジー検証 | 1 |
| 5 | 02/03 | [2026-02-03_165600_rdma_backend_test.md](2026-02-03_165600_rdma_backend_test.md) | RDMA バックエンド テスト・修正 (続) | 1 |
| 6 | 02/03 | [2026-02-03_183742_get_alloc_size_optimization.md](2026-02-03_183742_get_alloc_size_optimization.md) | get_alloc_size パフォーマンス最適化 | 1 |
| 7 | 02/03 | [2026-02-03_190500_rdma_timeout_fix.md](2026-02-03_190500_rdma_timeout_fix.md) | RDMAタイムアウト問題修正 | 1 |
| 8 | 02/04 | [2026-02-04_040100_rdma_graph_compute_buffer_fix.md](2026-02-04_040100_rdma_graph_compute_buffer_fix.md) | RDMAグラフ計算時バッファサイズ問題の修正 | 1 |
| 9 | 02/04 | [2026-02-04_052200_rdma_connection_lifetime_fix.md](2026-02-04_052200_rdma_connection_lifetime_fix.md) | RDMA接続ライフタイム修正 | 1 |
| 10 | 02/04 | [2026-02-04_060500_rdma_protocol_performance_optimization.md](2026-02-04_060500_rdma_protocol_performance_optimization.md) | RDMAプロトコル最適化 (0.3→2.7 t/s) | 1 |
| 11 | 02/04 | [2026-02-04_065100_rdma_graph_diff_update_optimization.md](2026-02-04_065100_rdma_graph_diff_update_optimization.md) | グラフ差分更新最適化 (2.7→47.2 t/s) | 1 |
| 12 | 02/04 | [2026-02-04_072500_rdma_adaptive_rsp_persistent_staging.md](2026-02-04_072500_rdma_adaptive_rsp_persistent_staging.md) | 適応的レスポンス + 永続ステージング | 1 |
| 13 | 02/04 | [2026-02-04_110225_rdma_benchmark_gpt_oss.md](2026-02-04_110225_rdma_benchmark_gpt_oss.md) | gpt-oss-20b/120b ベンチマーク | 2 |
| 14 | 02/04 | [2026-02-04_184824_rdma_120b_cluster_fix.md](2026-02-04_184824_rdma_120b_cluster_fix.md) | 120b クラスタ修正 (supports_buft) | 2 |
| 15 | 02/04 | [2026-02-04_214044_cpu_staged_vs_gpudirect_rdma_comparison.md](2026-02-04_214044_cpu_staged_vs_gpudirect_rdma_comparison.md) | CPU-staged vs GPUDirect 比較 | 3 |
| 16 | 02/04 | [2026-02-04_224611_rdma_batch_flush_optimization.md](2026-02-04_224611_rdma_batch_flush_optimization.md) | バッチフラッシュ最適化 | 3 |
| 17 | 02/05 | [2026-02-05_003452_step3_flush_compute_optimization.md](2026-02-05_003452_step3_flush_compute_optimization.md) | FLUSH+COMPUTE統合 & プロファイリング | 3 |
| 18 | 02/05 | [2026-02-05_101500_step4_gpudirect_rdma_enabled.md](2026-02-05_101500_step4_gpudirect_rdma_enabled.md) | GPUDirect RDMA 有効化 | 4 |
| 19 | 02/05 | [2026-02-05_121023_step4_gpudirect_120b_cluster_test.md](2026-02-05_121023_step4_gpudirect_120b_cluster_test.md) | GPUDirect 120b 11GPU クラスタテスト | 4 |
| 20 | 02/05 | [2026-02-05_133246_language_speed_comparison.md](2026-02-05_133246_language_speed_comparison.md) | 日本語/英語プロンプト速度差調査 | 4 |
| 21 | 02/05 | [2026-02-05_150903_language_speed_benchmark_v2.md](2026-02-05_150903_language_speed_benchmark_v2.md) | 言語速度比較ベンチマーク v2 | 4 |
| 22 | 02/05 | [2026-02-05_162500_rdma_vs_rpc_benchmark.md](2026-02-05_162500_rdma_vs_rpc_benchmark.md) | RDMA vs RPC 比較 | 4 |
| 23 | 02/05 | [2026-02-05_170441_gpu_combination_benchmark.md](2026-02-05_170441_gpu_combination_benchmark.md) | GPU組み合わせベンチマーク | 4 |
| 24 | 02/05 | [2026-02-05_175811_gpu_topology_benchmark.md](2026-02-05_175811_gpu_topology_benchmark.md) | GPUトポロジ考慮ベンチマーク | 4 |
| 25 | 02/05 | [2026-02-05_192000_gpu_combination_variance_analysis.md](2026-02-05_192000_gpu_combination_variance_analysis.md) | 性能ばらつき分析 | 4 |
| 26 | 02/05 | [2026-02-05_210745_rdma_variance_deep_investigation.md](2026-02-05_210745_rdma_variance_deep_investigation.md) | 性能ばらつき深掘り調査 | 4 |
| 27 | 02/05 | [2026-02-05_225700_rnr_nak_root_cause_fix.md](2026-02-05_225700_rnr_nak_root_cause_fix.md) | RNR NAK 根本原因特定・修正 | 4 |
| 28 | 02/06 | [2026-02-06_000500_rdma_vs_rpc_benchmark_v2.md](2026-02-06_000500_rdma_vs_rpc_benchmark_v2.md) | RDMA vs RPC v2 (RNR修正後) | 4 |
| 29 | 02/06 | [2026-02-06_202900_persistent_boot_config_guide.md](2026-02-06_202900_persistent_boot_config_guide.md) | init.sh 永続化マニュアル | 4 |
| 30 | 02/06 | [2026-02-06_213701_rdma_optimization_benchmark.md](2026-02-06_213701_rdma_optimization_benchmark.md) | 性能最適化ベンチマーク (5手法評価) | 4 |
| 31 | 02/07 | [2026-02-07_003000_glm47_iq2m_11gpu_test.md](2026-02-07_003000_glm47_iq2m_11gpu_test.md) | GLM-4.7 IQ2_M 11GPU テスト | 5 |
| 32 | 02/07 | [2026-02-07_095600_mmap_rdma_write_bugfix.md](2026-02-07_095600_mmap_rdma_write_bugfix.md) | mmap + RDMA Write バグ修正 | 4/5 |
| 33 | 02/07 | [2026-02-07_104703_rdma_backend_retrospective.md](2026-02-07_104703_rdma_backend_retrospective.md) | 振り返りレポート v1 | — |
| 34 | 02/07 | [2026-02-07_124327_glm47_iq2m_rdma_vs_rpc_output_quality.md](2026-02-07_124327_glm47_iq2m_rdma_vs_rpc_output_quality.md) | GLM-4.7 IQ2_M RDMA vs RPC 出力品質比較 | 5 |
| 35 | 02/07 | [2026-02-07_173027_build_procedure_investigation.md](2026-02-07_173027_build_procedure_investigation.md) | ビルド手順の安定化調査 | — |
| 36 | 02/07 | [2026-02-07_175849_rdma_backend_technical_architecture.md](2026-02-07_175849_rdma_backend_technical_architecture.md) | RDMA バックエンド技術アーキテクチャ | — |
| 37 | 02/07 | [2026-02-07_192500_p100_pcie_p2p_transfer_spec.md](2026-02-07_192500_p100_pcie_p2p_transfer_spec.md) | P100 PCIe P2P転送仕様 | — |
| 38 | 02/07 | [2026-02-07_195500_multi_rdma_device_output_corruption_fix.md](2026-02-07_195500_multi_rdma_device_output_corruption_fix.md) | マルチRDMAデバイス出力破損バグ修正 | 5 |
| 39 | 02/07 | [2026-02-07_202700_cpy_tensor_fix_comprehensive_test.md](2026-02-07_202700_cpy_tensor_fix_comprehensive_test.md) | cpy_tensor 修正の包括的テスト | 5 |
| 40 | 02/07 | [2026-02-07_203000_rdma_backend_retrospective_v2.md](2026-02-07_203000_rdma_backend_retrospective_v2.md) | 振り返りレポート v2 (本レポート) | — |

### 補助ドキュメント

| ファイル | 形式 | 説明 |
|---------|------|------|
| [2026-02-07_175849_rdma_backend_technical_architecture.html](2026-02-07_175849_rdma_backend_technical_architecture.html) | HTML | 技術アーキテクチャ (Mermaid図レンダリング版) |
| [2026-02-07_llama_cli_inference_pipeline.md](2026-02-07_llama_cli_inference_pipeline.md) | MD | llama-cli 推論パイプライン解説 |
| [2026-02-07_llama_cli_inference_pipeline.html](2026-02-07_llama_cli_inference_pipeline.html) | HTML | llama-cli 推論パイプライン (Mermaid図レンダリング版) |

---

## コード統計

### ファイル別行数

| ファイル | 行数 | 役割 |
|---------|-----:|------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 3,163 | メイン実装 (クライアント+サーバー+プロトコル) |
| `ggml/src/ggml-rdma/rdma-transport.cpp` | 869 | RDMA 接続管理 |
| `ggml/src/ggml-rdma/rdma-gdr.cpp` | 444 | GPUDirect RDMA |
| `ggml/src/ggml-rdma/rdma-memory.cpp` | 307 | ホストメモリ管理 |
| `ggml/src/ggml-rdma/rdma-transport.h` | 186 | トランスポート層ヘッダ |
| `ggml/src/ggml-rdma/rdma-gdr.h` | 116 | GPUDirect ヘッダ |
| `ggml/src/ggml-rdma/rdma-memory.h` | 109 | メモリ管理ヘッダ |
| `ggml/include/ggml-rdma.h` | 76 | 公開 C API |
| `tools/rdma/rdma-server.cpp` | 258 | サーバーエントリポイント |
| **合計** | **5,528** | **9ファイル** |

### コミット統計

| 指標 | 値 |
|------|-----|
| feature/rdma-backend のコミット数 (upstream含む) | 7,917 |
| RDMA 関連コミット数 | 21 |
| レポート数 | 40本 (+ 補助ドキュメント 3本) |

---

## まとめ

午後セッションの最大の成果は、**マルチRDMAデバイス出力破損バグの発見と修正** (`cpy_tensor` 問題) である。このバグは前回のレトロスペクティブ時点では「GLM-4.7 IQ2_M の出力品質問題」として認識されており、量子化の限界の可能性も考えられていた。RDMA vs RPC の比較実験 (タスク2) で RDMA 固有の問題と確定し、logits の argmax 分析によりデータ転送ではなく計算自体の破損と判明。最終的に `cpy_tensor` のグラフキャッシュ不整合が根本原因であることを特定した。

この修正により:
- GLM-4.7 IQ2_M が 11GPU クラスタで正常動作 (pp=6.6-9.2, tg=5.9-6.6 t/s)
- gpt-oss-120b が 11GPU フルクラスタで初の正常動作 (pp=36.6-48.4, tg=28.8-29.0 t/s)
- 性能影響は約3%低下のみ

Step 5 の残る主要タスクは **16GPU ハードウェア拡張** (物理GPU追加待ち) と **Q3_K_M/Q4_K_M でのテスト**。ソフトウェア面では、11GPU クラスタ上で GLM-4.7 と gpt-oss-120b の正常動作が確認できており、16GPU 拡張後のスムーズな移行が期待できる。
