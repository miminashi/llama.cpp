# RDMA バックエンド 振り返りレポート

- **実施日時**: 2026年2月7日 10:47

## プロジェクト概要

### 目標

2ノード16台の Tesla P100-PCIE-16GB で GLM-4.7 Q4 を GPUDirect RDMA による分散推論で動作させる。

### 段階的アプローチ

| Step | 内容 | 状態 |
|------|------|:----:|
| Step 1 | 基本 RDMA 通信 | ✅ 完了 |
| Step 2 | マルチノードクラスタ安定化 | ✅ 完了 |
| Step 3 | RDMA 性能最適化 | ✅ 完了 |
| Step 4 | GPUDirect RDMA 有効化 | ✅ 完了 |
| Step 5 | GLM-4.7 Q4 on 16 P100s | 🔧 進行中 |

### 環境

| 項目 | 値 |
|------|-----|
| 1号機 (192.168.100.1) | Tesla P100-PCIE-16GB × 7、合計 112 GB VRAM |
| 2号機 (192.168.100.2) | Tesla P100-PCIE-16GB × 4、合計 64 GB VRAM |
| ネットワーク | Mellanox ConnectX-4 100Gbps InfiniBand (RoCE v2) |
| GPU ドライバ | NVIDIA 535.288.01 |
| OS | Ubuntu 22.04, Linux 6.8.0-90-generic |

---

## Step 別の振り返り

### Step 1: 基本 RDMA 通信 (1/31〜2/4)

#### 期間・概要

llama.cpp に RDMA バックエンドをゼロから実装し、2ノード間の分散推論を実現した。実装可能性調査から始まり、プロトコル最適化を重ねて実用的な速度に到達。

#### 関連レポート (12本)

| # | ファイル | タイトル |
|---|---------|---------|
| 1 | [2026-01-31_214933_gpudirect_rdma_feasibility.md](2026-01-31_214933_gpudirect_rdma_feasibility.md) | GPUDirect RDMA 実装可能性調査レポート |
| 2 | [2026-01-31_222100_gpudirect_rdma_backend_implementation.md](2026-01-31_222100_gpudirect_rdma_backend_implementation.md) | GPUDirect RDMA バックエンド実装レポート |
| 3 | [2026-02-01_104744_rdma_backend_test.md](2026-02-01_104744_rdma_backend_test.md) | GPUDirect RDMA バックエンド実行テスト・修正レポート |
| 4 | [2026-02-03_134920_pcie_topology_verification.md](2026-02-03_134920_pcie_topology_verification.md) | PCIeトポロジー検証レポート |
| 5 | [2026-02-03_165600_rdma_backend_test.md](2026-02-03_165600_rdma_backend_test.md) | RDMA バックエンド テスト・修正レポート (続) |
| 6 | [2026-02-03_183742_get_alloc_size_optimization.md](2026-02-03_183742_get_alloc_size_optimization.md) | get_alloc_size パフォーマンス最適化レポート |
| 7 | [2026-02-03_190500_rdma_timeout_fix.md](2026-02-03_190500_rdma_timeout_fix.md) | RDMAタイムアウト問題 修正レポート |
| 8 | [2026-02-04_040100_rdma_graph_compute_buffer_fix.md](2026-02-04_040100_rdma_graph_compute_buffer_fix.md) | RDMAグラフ計算時バッファサイズ問題の修正 |
| 9 | [2026-02-04_052200_rdma_connection_lifetime_fix.md](2026-02-04_052200_rdma_connection_lifetime_fix.md) | RDMA接続ライフタイム修正 — 接続切断バグの解消 |
| 10 | [2026-02-04_060500_rdma_protocol_performance_optimization.md](2026-02-04_060500_rdma_protocol_performance_optimization.md) | RDMAプロトコル パフォーマンス最適化 — 0.3→2.7 t/s |
| 11 | [2026-02-04_065100_rdma_graph_diff_update_optimization.md](2026-02-04_065100_rdma_graph_diff_update_optimization.md) | RDMAグラフ差分更新 パフォーマンス最適化 — 2.7→47.2 t/s |
| 12 | [2026-02-04_072500_rdma_adaptive_rsp_persistent_staging.md](2026-02-04_072500_rdma_adaptive_rsp_persistent_staging.md) | RDMA Phase 3 — 適応的レスポンス送信 + 永続的ステージングバッファ |

#### 性能推移 (qwen2.5-0.5b, RDMA 1+1 GPU, tg)

| 最適化 | tg (t/s) | 改善倍率 | 主要変更 |
|--------|:--------:|:--------:|---------|
| 初期実装 | 0.3 | — | CQイベント駆動、プロトコル断片化 |
| Phase 1: CQ busy polling + メッセージ統合 | 2.7 | 9× | CQ busy polling、ヘッダ統合、バッファ拡大 |
| Phase 2: グラフ差分更新 | 47.2 | 17.5× | graph_cache 構造的比較、差分送信 |
| Phase 3: 適応的レスポンス + 永続ステージング | 148.5 | 3.1× | 小レスポンス1回send、MR alloc/free 排除 |
| **累積改善** | **148.5** | **495×** | |

#### 主要修正

- CQ completion channel → busy polling (`ibv_poll_cq` ループ)
- RDMA Send/Recv メッセージ統合 (8 completions/RT → 5)
- `graph_cache` の構造的比較 (不変フィールドのみ比較) + 差分更新コマンド
- 適応的レスポンス送信 (閾値 256B で 1回/2回 send を切り替え)
- 永続的ステージングバッファ (MR 登録/解除を排除)

---

### Step 2: マルチノードクラスタ安定化 (2/4)

#### 期間・概要

gpt-oss-120b (62.8 GB) を 11GPU クラスタ (7 CUDA + 4 RDMA) で安定動作させるために、クロスデバイスの不具合を修正。

#### 関連レポート (2本)

| # | ファイル | タイトル |
|---|---------|---------|
| 13 | [2026-02-04_110225_rdma_benchmark_gpt_oss.md](2026-02-04_110225_rdma_benchmark_gpt_oss.md) | ローカルGPU vs RDMA クラスタ ベンチマーク — gpt-oss-20b / gpt-oss-120b |
| 14 | [2026-02-04_184824_rdma_120b_cluster_fix.md](2026-02-04_184824_rdma_120b_cluster_fix.md) | RDMA 120b クラスタ修正 — supports_buft クロスデバイスバグ修正 |

#### 主要修正

- **`supports_buft` バグ**: `strstr` → `strcmp` でバックエンド名の部分一致を排除 (CUDA バッファが RDMA バックエンドで `true` を返す問題)
- **チャンク送受信**: 141MB の MoE エキスパート重みを 16MB チャンクで分割転送
- **クロスデバイスコピー安全ネット**: D2H + H2D による間接コピー

#### ベンチマーク

| 構成 | モデル | pp (t/s) | tg (t/s) |
|------|--------|:--------:|:--------:|
| ローカル 7GPU | gpt-oss-120b | 77.1 | 46.8 |
| 11GPU クラスタ (7C+4R) | gpt-oss-120b | 3〜11 | 0.7〜1.1 |
| ローカル 1GPU | gpt-oss-20b | 159.4 | 68.2 |
| RDMA 1+1 | gpt-oss-20b | 20.8 | 16.0 |

---

### Step 3: RDMA 性能最適化 (2/4〜2/5)

#### 期間・概要

RDMA Write のバッチフラッシュと FLUSH+COMPUTE 統合により、gpt-oss-20b の tg 性能損失を 77% → 9% に改善。プロファイリング機能を追加。

#### 関連レポート (3本)

| # | ファイル | タイトル |
|---|---------|---------|
| 15 | [2026-02-04_214044_cpu_staged_vs_gpudirect_rdma_comparison.md](2026-02-04_214044_cpu_staged_vs_gpudirect_rdma_comparison.md) | CPU-staged RDMA vs GPUDirect RDMA 実装難易度比較 |
| 16 | [2026-02-04_224611_rdma_batch_flush_optimization.md](2026-02-04_224611_rdma_batch_flush_optimization.md) | RDMA バッチフラッシュ最適化 |
| 17 | [2026-02-05_003452_step3_flush_compute_optimization.md](2026-02-05_003452_step3_flush_compute_optimization.md) | Step 3 FLUSH+COMPUTE統合 & プロファイリング |

#### 主要修正

- **バッチフラッシュ**: set_tensor 時は RDMA Write のみ、graph_compute 前に一括フラッシュ。dirty range をバウンディングボックスでマージし、バッファごとに 1 回の cudaMemcpy に集約。
- **FLUSH+COMPUTE 統合**: 2回のラウンドトリップ (FLUSH_ALL_STAGING → GRAPH_RECOMPUTE) を 1 コマンド (FLUSH_AND_RECOMPUTE) に統合。
- **プロファイリング**: サーバー側 (graph_recompute, flush, compute 時間) + クライアント側 (per-call タイミング)

#### ベンチマーク (Step 3 完了時)

| 構成 | モデル | pp128 (t/s) | ローカル比 | tg32 (t/s) | ローカル比 |
|------|--------|:-----------:|:----------:|:----------:|:----------:|
| ローカル 1GPU | qwen2.5-0.5b | 3,390 | 100% | 213.36 | 100% |
| RDMA 1+1 | qwen2.5-0.5b | 3,113 | 91.8% | 156.22 | 73.2% |
| ローカル 1GPU | gpt-oss-20b | 407 | 100% | 64.27 | 100% |
| RDMA 1+1 | gpt-oss-20b | 61.09 | 15.0% | **58.47** | **91.0%** |

**目標達成**: gpt-oss-20b tg32 = 58.47 t/s (目標 30+)、pp128 = 61.09 t/s (目標 40+)

---

### Step 4: GPUDirect RDMA 有効化 (2/5〜2/7)

#### 期間・概要

nvidia-peermem 経由の GPU メモリ直接 RDMA 登録を有効化。pp128 が CPU staging 比で 6.6 倍高速化。RNR NAK の根本原因特定・修正、RDMA vs RPC ベンチマーク比較、mmap + RDMA Write バグ修正を実施。

#### 関連レポート (14本)

| # | ファイル | タイトル |
|---|---------|---------|
| 18 | [2026-02-05_101500_step4_gpudirect_rdma_enabled.md](2026-02-05_101500_step4_gpudirect_rdma_enabled.md) | Step 4 GPUDirect RDMA 有効化 |
| 19 | [2026-02-05_121023_step4_gpudirect_120b_cluster_test.md](2026-02-05_121023_step4_gpudirect_120b_cluster_test.md) | Step 4 GPUDirect RDMA: 120b 11GPU クラスタテスト |
| 20 | [2026-02-05_133246_language_speed_comparison.md](2026-02-05_133246_language_speed_comparison.md) | 日本語/英語プロンプトの生成速度差調査 |
| 21 | [2026-02-05_150903_language_speed_benchmark_v2.md](2026-02-05_150903_language_speed_benchmark_v2.md) | 言語速度比較ベンチマーク v2 |
| 22 | [2026-02-05_162500_rdma_vs_rpc_benchmark.md](2026-02-05_162500_rdma_vs_rpc_benchmark.md) | RDMA vs RPC バックエンド パフォーマンス比較 |
| 23 | [2026-02-05_170441_gpu_combination_benchmark.md](2026-02-05_170441_gpu_combination_benchmark.md) | GPU組み合わせベンチマーク |
| 24 | [2026-02-05_175811_gpu_topology_benchmark.md](2026-02-05_175811_gpu_topology_benchmark.md) | GPUトポロジ考慮ベンチマーク |
| 25 | [2026-02-05_192000_gpu_combination_variance_analysis.md](2026-02-05_192000_gpu_combination_variance_analysis.md) | GPU組み合わせベンチマーク性能ばらつき分析 |
| 26 | [2026-02-05_210745_rdma_variance_deep_investigation.md](2026-02-05_210745_rdma_variance_deep_investigation.md) | RDMA性能ばらつき深掘り調査 |
| 27 | [2026-02-05_225700_rnr_nak_root_cause_fix.md](2026-02-05_225700_rnr_nak_root_cause_fix.md) | RDMA性能ばらつき根本原因の特定と修正 |
| 28 | [2026-02-06_000500_rdma_vs_rpc_benchmark_v2.md](2026-02-06_000500_rdma_vs_rpc_benchmark_v2.md) | RDMA vs RPC ベンチマーク v2 (RNR NAK修正後) |
| 29 | [2026-02-06_202900_persistent_boot_config_guide.md](2026-02-06_202900_persistent_boot_config_guide.md) | init.sh 設定のOS起動時永続化マニュアル |
| 30 | [2026-02-06_213701_rdma_optimization_benchmark.md](2026-02-06_213701_rdma_optimization_benchmark.md) | RDMA バックエンド性能最適化ベンチマーク |
| 31 | [2026-02-07_095600_mmap_rdma_write_bugfix.md](2026-02-07_095600_mmap_rdma_write_bugfix.md) | mmap + RDMA Write バグ修正 |

#### GPUDirect RDMA ベンチマーク (gpt-oss-20b, 1+1)

| モード | pp128 (t/s) | ローカル比 | tg32 (t/s) | ローカル比 |
|--------|:-----------:|:----------:|:----------:|:----------:|
| ローカル 1GPU | 407.68 | 100% | 64.27 | 100% |
| **GPUDirect RDMA** | **403.69** | **99.0%** | **59.52** | **92.6%** |
| CPU staging | 61.06 | 15.0% | 59.30 | 92.3% |

- **pp128**: GPUDirect は CPU staging の **6.6 倍**高速 (403.69 vs 61.06 t/s)
- **tg32**: GPUDirect と CPU staging はほぼ同等 (計算時間が支配的)

#### RNR NAK 修正効果

IB Send/Recv のデフォルト `min_rnr_timer=0` (655.36ms/リトライ) が 2 秒スパイクの根本原因。`min_rnr_timer=1` (0.01ms) に設定して解消。

| 指標 | 修正前 | 修正後 | 改善率 |
|------|:------:|:------:|:------:|
| tg32 分散 | ±20.48 (56.7%) | ±0.77 (1.5%) | 27倍安定化 |
| 1+4 tg32 速度 | 0.57 t/s | 48.67 t/s | 85倍高速化 |
| 2秒スパイク | 12回/10rep | 0回 | 完全解消 |

#### RDMA vs RPC 比較 (RNR修正後, gpt-oss-120b)

| 構成 | pp128 RDMA→RPC | tg32 RDMA→RPC |
|------|:--------------:|:-------------:|
| 7+2 | 214.52 → 193.90 (-9.6%) | 41.23 → 29.46 (-28.6%) |
| 6+2 | 211.99 → 192.56 (-9.2%) | 40.23 → 28.11 (-30.1%) |

RDMA は RPC に対して tg で **28〜30% 高速**、pp で **9〜10% 高速**。分散も RDMA が大幅に安定 (tg ±0.0〜1.5% vs RPC ±2.3〜3.6%)。

#### 最適化ベンチマーク (5手法の個別評価)

| 最適化手法 | tg 変化 | 判定 |
|-----------|:-------:|:----:|
| #1 prepost-recv (recv バッファ事前確保) | +2.1% | 微小 |
| #2 prealloc-buf (クライアントバッファ事前確保) | -0.2% | 微小 |
| #3 cq-backoff (CQ 適応的バックオフ) | **-27.5%** | 不採用 |
| #4 combined-send (ヘッダ+データ統合送信) | **-31.0%** | 不採用 |
| #5 all (#1+#2 統合) | +2.6% | 微小 |

CQ バックオフと combined-send は逆効果。現行のビジーポーリング + 分離送受信が最適。

#### mmap + RDMA Write バグ修正

- **原因**: RNIC ページテーブルキャッシュのオーバーフロー (4GB超 × 複数バッファ = 合計 40GB+ の MR 登録)
- **修正**: `RDMA_WRITE_MAX_BUFFER_SIZE = 4GB` で大バッファは Send/Recv フォールバック、16MB チャンク RDMA Write、`rdma_write_disabled` フラグ
- **結果**: `--no-mmap` ワークアラウンド不要に

---

### Step 5: GLM-4.7 事前検証 (2/7〜)

#### 期間・概要

Step 5 の最終目標に向けた事前検証。GLM-4.7 IQ2_M を 11GPU クラスタでテスト。

#### 関連レポート (2本)

| # | ファイル | タイトル |
|---|---------|---------|
| 31 | [2026-02-07_003000_glm47_iq2m_11gpu_test.md](2026-02-07_003000_glm47_iq2m_11gpu_test.md) | GLM-4.7 IQ2_M 11GPUクラスタ動作テスト |
| 32 | [2026-02-07_095600_mmap_rdma_write_bugfix.md](2026-02-07_095600_mmap_rdma_write_bugfix.md) | mmap + RDMA Write バグ修正 |

#### ベンチマーク

| 構成 | モデル | pp128 (t/s) | tg32 (t/s) |
|------|--------|:-----------:|:----------:|
| 11GPU (7C+4R) | GLM-4.7 IQ2_M (114 GiB, 358B params) | 23.48 | 7.66 |

#### 出力品質の問題

IQ2_M (2.7 bpw) では thinking トークン (`[Start thinking]` 〜 `[End thinking]`) がゴミ文字列になる。ただし、ユーザーが CPU+GPU 併用 (RDMA なし) で同モデルの正常な日本語生成を確認済みのため、量子化の限界だけでなく RDMA バックエンド固有の問題の可能性がある。RPC バックエンドとの出力品質比較実験で切り分けが必要。

---

## 性能推移サマリー

### Step ごとのベストスコア

| Step | 構成 | モデル | pp128 (t/s) | tg32 (t/s) | ローカル比 (tg) |
|------|------|--------|:-----------:|:----------:|:--------------:|
| Step 1 完了 | RDMA 1+1 | qwen2.5-0.5b | — | 148.5 | 75.7% |
| Step 2 完了 | 11GPU (7C+4R) | gpt-oss-120b | 3〜11 | 0.7〜1.1 | 1.5〜2.3% |
| Step 3 完了 | RDMA 1+1 | gpt-oss-20b | 61.09 | 58.47 | **91.0%** |
| Step 4 完了 (GDR) | RDMA 1+1 | gpt-oss-20b | **403.69** | 59.52 | **92.6%** |
| Step 4 完了 (クラスタ) | 11GPU (7C+4R) | gpt-oss-120b | 201.02 | 36.92 | 84.7% |
| Step 5 事前検証 | 11GPU (7C+4R) | GLM-4.7 IQ2_M | 23.48 | 7.66 | — |

### RDMA vs RPC 最終比較 (RNR修正後)

| モデル | 構成 | RDMA tg32 | RPC tg32 | RDMA 優位 |
|--------|------|:---------:|:--------:|:---------:|
| gpt-oss-20b | 1+1 | 52.39 | 50.66 | +3.3% |
| gpt-oss-20b | 2+2 | 56.92 | 36.79 | **+54.7%** |
| gpt-oss-120b | 7+2 | 41.23 | 29.46 | **+40.0%** |
| gpt-oss-120b | 6+2 | 40.23 | 28.11 | **+43.1%** |

リモート GPU 数が増えるほど RDMA の優位性が顕著になる。

---

## 既知のバグと対応状況

| # | バグ | 状態 | 影響 | ワークアラウンド |
|---|------|:----:|------|-----------------|
| 1 | GPUDirect RDMA timeout on large models | **未修正** | 大規模モデル (GLM-4.7) で GDR 有効時にタイムアウト。GDR 無効だと pp が大幅低下 | `GGML_RDMA_NO_GDR=1` |
| 2 | GLM-4.7 IQ2_M 出力品質 | **要調査** | thinking トークンがゴミ化。RDMA 固有の可能性あり | RDMA vs RPC 比較実験で原因切り分け予定 |
| 3 | RDMA デバイス逐次実行 | **設計上の制約** | 4 RDMA = 4× graph_compute/token | 将来の非同期化で改善可能 |
| 4 | cmake --build 無出力 | **ビルドシステム** | ビルドが実行されないことがある | `make -C build -j$(nproc)` を使用 |
| 5 | mmap + RDMA Write overflow | **修正済み (未コミット)** | 大規模モデルで remote access error (vendor_err=0x88) | 4GB バッファ制限 + Send/Recv フォールバック |
| 6 | RNR NAK 2秒スパイク | **修正済み** | tg 分散 1400%、2秒スパイク | min_rnr_timer=1 で完全解消 |
| 7 | supports_buft クロスデバイス | **修正済み** | gpt-oss-120b で CUDA illegal memory access | strstr → strcmp に変更 |

### バグ #2 の詳細 (出力品質問題)

- ユーザーが CPU+GPU 併用 (RDMA なし) で GLM-4.7 IQ2_M の正常な日本語生成を確認済み
- RDMA 経由では thinking トークンがゴミ化
- **原因仮説**: RDMA バックエンドでのテンソルデータ転送時にデータ化けの可能性
- **切り分け手順**: 同一モデル・同一 GPU 数で RDMA vs RPC の出力品質を比較

---

## 残タスク (Step 5 完了まで)

| # | タスク | 優先度 | ブロッカー | 状態 |
|---|--------|:------:|-----------|:----:|
| 1 | mmap バグ修正コミット | 高 | なし | コード完成、未コミット |
| 2 | GLM-4.7 出力品質の RDMA vs RPC 比較実験 | 高 | なし | 計画済み |
| 3 | GPUDirect timeout 調査・修正 | 高 | なし (調査可能) | 未着手 |
| 4 | 16GPU ハードウェア準備 | 高 | 物理 GPU 追加 | 待ち |
| 5 | Q3_K_M/Q4_K_M ダウンロード・テスト | 高 | 16GPU (11GPU で先行可能) | 未着手 |
| 6 | 16GPU 全体テスト | 中 | タスク 4, 5 | 未着手 |
| 7 | マルチ RDMA デバイス非同期化 | 低 | なし | 未着手 (将来課題) |

### 量子化レベル選択

| 量子化 | bpw | 推定サイズ | 16GPU (256GB) | 11GPU (176GB) |
|--------|----:|----------:|:-------------:|:-------------:|
| IQ2_M | 2.7 | 114 GB | 収容可能 | 収容可能 (**品質×**) |
| Q3_K_M | ~3.5 | ~150 GB | 収容可能 | 収容不可 |
| Q4_K_M | ~4.5 | ~195 GB | 収容可能 | 収容不可 |

Q3_K_M (150GB) が 16GPU での最有力候補。Q4_K_M (195GB) は 256GB でギリギリ。

---

## 全レポート一覧

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
