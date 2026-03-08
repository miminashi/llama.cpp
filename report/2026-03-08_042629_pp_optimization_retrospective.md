# PP性能改善施策 振り返りレポート

- **実施日時**: 2026年3月8日 04:26
- **ワークツリー**: N/A (レビュー・分析のみ)

## 前提・目的

プロジェクト開始 (2026-02-06) から現在 (2026-03-07) までに試行した Prompt Processing (PP) 性能改善施策を網羅的に振り返り、各施策の効果・位置づけを整理する。

- **対象モデル**: GLM-4.7 IQ2_M (11GPU, 7C+4R), Qwen3.5-35B-A3B Q4_K_M (4-8GPU), gpt-oss-20b Q4_K_M
- **対象期間**: 2026-02-06 〜 2026-03-07
- **目的**: 何が効いて何が効かなかったかの明確化、今後の施策選定への参照資料

## ベースライン変遷

### GLM-4.7 IQ2_M pp128 (11GPU, 7C+4R)

| 時点 | pp128 (t/s) | 備考 |
|------|:-----------:|------|
| Step 5 初期 (GDR無効) | 6.4 | `-fa` なし |
| Selective Signaling 有効 | ~30.6 | CQE DMA 削減 |
| Send always-signal 修正後 | ~24.3 | バッファ race 修正の副作用 |
| Ring buffer + `-fa 1` (現行) | ~24.2 | Ring buffer では回復せず |

### Qwen3.5-35B-A3B pp128 (4GPU, 2C+2R)

| 時点 | pp128 (t/s) | 備考 |
|------|:-----------:|------|
| 初期測定 | ~189 | feature/rdma-backend |
| Ring buffer 導入後 | ~200 | Send selective signaling 回復 |
| Pipeline+Per-device 組み合わせ | ~238 | サーバー側デバイス並列化 |

## 施策一覧

### カテゴリ A: RDMA通信最適化 (6施策)

#### A-1. RDMA通信パス最適化 (Pre-posted Recv 等)

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | なし (高分散、方向性不明) |
| **メカニズム** | サーバー側 recv WR の事前 post で RNR NAK を削減 |
| **結果** | RNR NAK は削減されたが、pp/tg の安定性が悪化。計測値の分散が大きく改善とは判断できず |
| **レポート** | [2026-02-11 Pre-posted Recv](2026-02-11_040659_prepost_recv_rnr_nak.md) |

#### A-2. Selective Signaling

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | **+26.1%** (23.35 → 29.44 t/s) |
| **メカニズム** | IB Send/RDMA Write の CQE 生成頻度を 1/64 に削減し、CQE DMA と poll 回数を減少 |
| **結果** | 非常に大きな効果。ただし Send バッファ race condition の修正 (A/B-12) で実質無効化され、効果は喪失 |
| **レポート** | [2026-02-19 Selective Signaling](2026-02-19_120000_selective_signaling.md) |
| **環境変数** | `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` で無効化 |

#### A-3. Server-Push (RDMA Write with IMM)

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | なし (+0.01%) |
| **TG効果** | +0.74% (統計的有意だが実用効果は小さい) |
| **メカニズム** | get_tensor を pull → push モデルに変更。graph_compute 完了後にサーバーが RDMA Write IMM で結果をプッシュ |
| **結果** | PP にはまったく効果なし。get_tensor の通信コストは PP ボトルネックではない |
| **レポート** | [2026-02-18 Server-Push](2026-02-18_200000_server_push_rdma_write_imm.md) |

#### A-4. Async Compute (Fire-and-Forget graph_compute)

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | なし |
| **TG効果** | +13% (6.0 → 6.8 t/s, GDR 有効時) |
| **メカニズム** | graph_compute のレスポンス待ちを廃止 (RPC 同等の fire-and-forget) |
| **結果** | TG のラウンドトリップ削減には有効だが、PP は元からバッチ処理のためレイテンシ隠蔽効果なし |
| **レポート** | [2026-02-10 Async Compute](2026-02-10_214841_rdma_async_compute_optimization.md) |

#### A-5. Deferred Copy

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU), gpt-oss-20b (5GPU) |
| **PP効果** | gpt-oss-20b: +1.6%, GLM-4.7: +1.45% |
| **TG効果** | gpt-oss-20b: +1.2%, GLM-4.7: +1.45% |
| **メカニズム** | `cpy_tensor` をサーバーローカル D2H+H2D に変換し、IB ネットワークラウンドトリップを排除 |
| **結果** | 小幅ながら統計的に有意な改善。Multi-RDMA デバイスの出力破損修正にも貢献 |
| **レポート** | [2026-02-13 Deferred Copy 統計ベンチマーク](2026-02-13_090026_deferred_copy_statistical_benchmark.md) |

#### A-6. サーバーサイドパイプラインバッファリング

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | **-0.21%** (微小悪化) |
| **TG効果** | +0.55% (有意だが小さい) |
| **メカニズム** | D0 結果を即送信、D1-D3 をバッチ化して send/recv オーバーヘッドを削減 |
| **結果** | PP は微小悪化。バッファリングのオーバーヘッドが send/recv 削減を相殺 |
| **レポート** | [2026-02-20 Server Pipeline Benchmark](2026-02-20_215838_server_pipeline_benchmark.md) |

---

### カテゴリ B: 並列化・パイプライン (5施策)

#### B-7. Per-device Connections (単独)

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 (11GPU), Qwen3.5 (4-8GPU) |
| **PP効果** | **悪化**: GLM-4.7 -1.35%, Qwen3.5 4GPU -1.6%, 8GPU -5.1% |
| **TG効果** | GLM-4.7 +1.76%, Qwen3.5 中立 |
| **メカニズム** | 各 RDMA デバイスに独立 QP を割り当て、並列送信を可能に |
| **結果** | 単独では PP 悪化。接続確立のオーバーヘッドが GPU 数に比例して増加。Pipeline との組み合わせで真価を発揮 |
| **環境変数** | `GGML_RDMA_PER_DEVICE_CONN=1` |

#### B-8. Pipeline Parallelism (単独)

| 項目 | 内容 |
|------|------|
| **対象モデル** | Qwen3.5 (4GPU), GLM-4.7 (11GPU) |
| **PP効果** | Qwen3.5: +0.4% (ns), GLM-4.7: +0.02% (ns) |
| **メカニズム** | Split-level pipeline dispatch: 複数 ubatch のグラフを事前構築し非同期ディスパッチ |
| **結果** | 単独では効果なし。Per-device connections なしでは単一接続の FIFO 処理がボトルネック |
| **環境変数** | `GGML_RDMA_PIPELINE=1` |
| **レポート** | [2026-03-03 Pipeline A/B Benchmark](2026-03-03_034054_pipeline_parallelism_ab_benchmark.md) |

#### B-9. Pipeline + Per-device 組み合わせ

| 項目 | 内容 |
|------|------|
| **対象モデル** | Qwen3.5 (4GPU), GLM-4.7 (11GPU) |
| **PP効果** | **Qwen3.5 4GPU: +25.9%** (189 → 238 t/s), GLM-4.7: -0.33% |
| **TG効果** | Qwen3.5: +1.8%, GLM-4.7: +1.17% |
| **メカニズム** | Pipeline が非同期ディスパッチ → Per-device の独立接続でサーバー側マルチデバイス並列計算を実現 |
| **結果** | Qwen3.5 (MoE) で劇的改善。GLM-4.7 では layer-split のデバイス依存関係が逐次実行を強制し PP 効果なし |
| **環境変数** | `GGML_RDMA_PIPELINE=1 GGML_RDMA_PER_DEVICE_CONN=1` |
| **レポート** | [2026-03-03 Pipeline+Per-device Combined](2026-03-03_145744_pipeline_perdevice_combined_benchmark.md) |

#### B-10. CUDA/RDMA Overlap

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU), Qwen3.5 (4-8GPU) |
| **PP効果** | **GLM-4.7 pp2048: +28.7%** (39.70 → 51.10 t/s), Qwen3.5 pp16384: +1.7-2.4% |
| **メカニズム** | 3フェーズ split 実行 — CUDA 計算中に RDMA コマンドを非同期ディスパッチし、CUDA/RDMA 処理をオーバーラップ |
| **結果** | 複数 ubatch 時に大幅改善。pp128 (1 ubatch) では非適用。Qwen3.5 は compute-bound (94.8%) のため効果小 |
| **制約** | `GGML_RDMA_PER_DEVICE_CONN=1` が必須。pp ≤ ubatch (1 ubatch) では効果なし |
| **環境変数** | `GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1` |
| **レポート** | [2026-03-05 CUDA/RDMA Overlap](2026-03-05_163449_cuda_rdma_overlap_benchmark.md) |

#### B-11. GDR + Parallel Dispatch (Expert Parallelism)

| 項目 | 内容 |
|------|------|
| **対象モデル** | Qwen3.5 (4-8GPU) |
| **PP効果** | なし (GDR との両立困難) |
| **メカニズム** | デバイス間並列ディスパッチ + GDR (GPU 直接 RDMA) の組み合わせ |
| **結果** | GDR 有効時に RNIC が GPU メモリに直接 RDMA Write → 非同期 CUDA カーネルと競合しクラッシュ。sync-before-recv で回避可能だがデバイス間並列性が失われる |
| **環境変数** | `GGML_RDMA_PARALLEL_DISPATCH=0` で無効化 |
| **レポート** | [2026-02-25 Expert Parallelism Phase 11](2026-02-25_043933_expert_parallelism_phase11_parallel_dispatch.md) |

---

### カテゴリ C: バッファ・シグナリング修正 (2施策)

#### C-12. Send Buffer Race Fix (always-signal)

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | **-20.6%** (30.6 → 24.3 t/s) — 正確性修正の副作用 |
| **メカニズム** | 再利用可能な send_buffer_ (16MB) を selective signaling 下で使用 → DMA 上書き race が発生。always-signal に戻して修正 |
| **結果** | バグ修正として必須だが、selective signaling の PP 効果を全て打ち消した。修正前の ~30.6 t/s は不正なデータに基づく数値だった |
| **レポート** | [2026-02-25 Send Buffer Race](2026-02-25_215905_send_buffer_race_condition_fix.md) |

#### C-13. Send Ring Buffer

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU), Qwen3.5 (4GPU) |
| **PP効果** | 中立 (GLM-4.7: 回復せず、Qwen3.5: -0.02%) |
| **メカニズム** | send_buffer_ をリングバッファとして使用し、selective signaling を安全に再有効化。interval=64 での CQ poll 削減 |
| **結果** | graph_compute コマンドは inline (<256B) で常に signaled のため、ring buffer では PP を回復できない。Qwen3.5 では ring buffer が Pipeline+Per-device の PP 効果 (+25.9%) を吸収 (ring buffer 単独で同等の効果) |
| **レポート** | [2026-03-04 Send Ring Buffer](2026-03-04_004316_send_ring_buffer_benchmark.md) |

---

### カテゴリ D: パラメータ・構成チューニング (4施策)

#### D-14. Flash Attention (`-fa 1`)

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | +0.6% (ns) |
| **TG効果** | **+10.4%** (7.72 → 8.52 t/s) |
| **メカニズム** | Flash Attention カーネルによるメモリ効率化 |
| **結果** | PP には効果なし。TG で大幅改善のため常時有効を推奨 |
| **レポート** | [2026-02-21 tg32 デグレ調査](2026-02-21_210816_tg32_degression_investigation.md) |

#### D-15. `-nkvo 1` 除去

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | 微小 (測定誤差範囲) |
| **TG効果** | **+31% 回復** (5.3 → 7.7 t/s) |
| **メカニズム** | `-nkvo 1` は KV キャッシュを CPU に配置し、tg で 78.5% の時間が HtoD memcpy に消費されていた |
| **結果** | ベンチマークアーティファクトの発見・除去。PP への影響は微小 |
| **レポート** | [2026-02-21 tg32 デグレ調査](2026-02-21_210816_tg32_degression_investigation.md) |

#### D-16. ubatch サイズチューニング

| 項目 | 内容 |
|------|------|
| **対象モデル** | Qwen3.5-35B-A3B (4-8GPU) |
| **PP効果** | **pp2048: +50-70%**, pp16384: +19% (ub=512 → ub=2048) |
| **メカニズム** | MoE の cuBLAS タイル充填率が ubatch サイズに依存。ub=2048 で飽和 (tokens/expert: 16 → 64) |
| **結果** | コード変更不要で最大の PP 改善。ub=2048 以降は +0.4% 未満で飽和。pp128 には効果なし (ub=128 < デフォルト 512) |
| **レポート** | [2026-03-02 ubatch チューニング](2026-03-02_205900_n_ubatch_tuning_benchmark.md), [2026-03-07 pp16384 全最適化](2026-03-07_074630_qwen35_pp16384_all_optimizations.md) |

#### D-17. PCIe トポロジ活用 (2C+2R vs 4C)

| 項目 | 内容 |
|------|------|
| **対象モデル** | Qwen3.5-35B-A3B (4GPU) |
| **PP効果** | pp2048: **+15.8%**, pp128: -0.6%, pp512: -0.5% |
| **TG効果** | -9.7% (RDMA コマンドレイテンシ) |
| **メカニズム** | Node 1 の GPU3-6 は同一 PCIe スイッチで帯域幅競合。2C+2R はノード間分散で PCIe 競合を解消 |
| **結果** | pp ≥ 2048 で RDMA 分散の利点が PCIe 競合解消として発現。pp ≤ 512 はローカル CUDA が有利 |
| **レポート** | [2026-03-06 RDMA サーバーオーバーヘッド分析](2026-03-06_091203_rdma_server_overhead_analysis.md) |

---

### カテゴリ E: 不採用・逆効果の施策 (3施策)

#### E-18. MMQ 強制有効化

| 項目 | 内容 |
|------|------|
| **対象モデル** | Qwen3.5-35B-A3B (4GPU) |
| **PP効果** | **-48%** (pp128: 216 → 112 t/s) |
| **メカニズム** | `GGML_CUDA_FORCE_MMQ` で P100 (cc 6.0) に MMQ パスを強制し、stream sync を排除 |
| **結果** | P100 は DP4A 命令非対応 (cc ≥ 6.1 必要)。ソフトウェアエミュレーションのコストが stream sync 排除の利得を大幅に上回る |
| **レポート** | [2026-03-07 MMQ Force P100](2026-03-07_211435_mmq_force_p100_benchmark.md) |

#### E-19. Gate+Up マージ GGUF

| 項目 | 内容 |
|------|------|
| **対象モデル** | Qwen3.5-35B-A3B |
| **PP効果** | 未検証 (期待 +2-5%) |
| **メカニズム** | `ffn_gate_exps` + `ffn_up_exps` → `ffn_gate_up_exps` に結合し `mul_mat_id` 呼び出しを 3→2 に削減 |
| **結果** | 計算量は不変。削減されるのはカーネルラウンチ + D2H/H2D sync 1 回分のみ。GGUF 再変換が必要で未実施 |
| **レポート** | [2026-03-07 PP最適化総合調査](2026-03-07_162952_qwen35_pp_optimization_comprehensive.md) |

#### E-20. 8GPU スケーリング (4GPU → 8GPU)

| 項目 | 内容 |
|------|------|
| **対象モデル** | Qwen3.5-35B-A3B |
| **PP効果** | **-0.4% 〜 -10.4%** (全 pp サイズで悪化) |
| **メカニズム** | GPU 数を 4 → 8 に倍増 |
| **結果** | MoE (3B active / 35B total) は GPU 追加でスケーリングしない。GPU あたりの計算量が減少し、通信・同期オーバーヘッドが支配的に |
| **レポート** | [2026-03-06 2C2R vs 4C4R](2026-03-06_205212_2c2r_vs_4c4r_gpu_scaling.md) |

---

### カテゴリ F: GLM-4.7 固有の検証 (2施策)

#### F-21. Pipeline + Per-device GLM-4.7 検証

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | pp128: -0.33%, pp128 (PIPELINE 単独): +0.02% (ns) |
| **TG効果** | +1.17% |
| **メカニズム** | B-9 と同一の Combined モードを GLM-4.7 で検証 |
| **結果** | Layer-split モデルではデバイス間に逐次依存関係があり、pipeline_waits がブロッキングを drain→send に移動するだけで合計時間は不変 |
| **レポート** | [2026-03-05 Pipeline PP 改善調査](2026-03-05_060119_pipeline_pp_improvement_investigation.md), [2026-03-04 GLM-4.7 Server-side Sync](2026-03-04_235405_glm47_server_side_sync_benchmark.md) |

#### F-22. GLM-4.7 PP128 ボトルネック分析

| 項目 | 内容 |
|------|------|
| **対象モデル** | GLM-4.7 IQ2_M (11GPU) |
| **PP効果** | (分析のみ、理論最適: +57%) |
| **メカニズム** | プロファイリングにより ubatch あたり CUDA 3366ms (64.6%) + RDMA 1846ms (35.4%) が逐次実行されていることを特定 |
| **結果** | CUDA/RDMA overlap (B-10) により pp2048 で +28.7% を実現。pp128 は 1 ubatch のためオーバーラップ不可 |
| **レポート** | [2026-03-05 Pipeline PP 改善調査](2026-03-05_060119_pipeline_pp_improvement_investigation.md) |

---

## 効果ランキング (PP改善率順)

| 順位 | 施策 | PP改善率 | 対象 | 条件 |
|:----:|------|:--------:|------|------|
| 1 | **D-16. ubatch チューニング** | **+50-70%** | Qwen3.5 pp2048-4096 | `-ub 2048` |
| 2 | **B-10. CUDA/RDMA Overlap** | **+28.7%** | GLM-4.7 pp2048 | 複数 ubatch 時 |
| 3 | **A-2. Selective Signaling** | **+26.1%** | GLM-4.7 pp128 | ※race fix で喪失 |
| 4 | **B-9. Pipeline+Per-device** | **+25.9%** | Qwen3.5 4GPU pp128 | 組み合わせが必須 |
| 5 | **D-16. ubatch チューニング** | **+19%** | Qwen3.5 pp16384 | `-ub 2048` |
| 6 | **D-17. PCIe トポロジ活用** | **+15.8%** | Qwen3.5 pp2048 | 2C+2R 構成 |
| 7 | **B-10. CUDA/RDMA Overlap** | **+2.4%** | Qwen3.5 pp16384 8GPU | compute-bound で効果小 |
| 8 | **A-5. Deferred Copy** | **+1.6%** | gpt-oss-20b pp128 | 統計的有意 |

※ Selective Signaling (+26.1%) は race condition 修正 (C-12) により実質的に利用不可。

## モデル別最適構成まとめ

### GLM-4.7 IQ2_M (Dense, 11GPU, communication-bound)

| パラメータ | 推奨値 | 理由 |
|-----------|--------|------|
| `-fa 1` | 有効 | TG +10% |
| `-nkvo` | **使わない** | TG -31% の劣化 |
| CUDA/RDMA Overlap | **有効** (`CUDA_OVERLAP=1 PER_DEVICE_CONN=1`) | pp2048 +28.7% |
| Pipeline | 不要 | layer-split 依存で効果なし |
| 最適 pp128 | ~24.2 t/s | 現行ベースライン |
| 最適 pp2048 | ~51.1 t/s | Overlap 有効時 |

### Qwen3.5-35B-A3B (MoE, 4GPU, compute-bound)

| パラメータ | 推奨値 | 理由 |
|-----------|--------|------|
| `-fa 1` | 有効 | TG 改善 |
| `-ub 2048` | pp ≥ 2048 時 | PP +50-70% |
| CUDA/RDMA Overlap | 有効 | pp16384 +1.7% (小幅) |
| GPU 数 | **4GPU (2C+2R) が最適** | 8GPU は負のスケーリング |
| Pipeline+Per-device | Ring buffer で吸収済み | Ring buffer 単独で同等 |
| 最適 pp128 | ~200-205 t/s | Ring buffer |
| 最適 pp16384 | ~418 t/s | ub=2048 + overlap |

## 未実装の有望施策

| 施策 | 期待効果 | 実装コスト | 備考 |
|------|---------|-----------|------|
| Send double-buffering | GLM-4.7 pp128 回復 (~30 t/s) | 中 | Selective signaling を安全に再有効化する別アプローチ |
| Gate+Up マージ GGUF | Qwen3.5 PP +2-5% | 低 (GGUF 再変換) | `mul_mat_id` 呼び出し 33% 削減 |
| CUDA/RDMA overlap (pp128 対応) | GLM-4.7 pp128 +57% (理論値) | 高 | ubatch を小さくして複数 ubatch 化が必要だが、per-ubatch overhead が相殺 |

## 教訓・知見

### 1. 施策の効果はモデルアーキテクチャに強く依存する

- **Dense モデル (GLM-4.7)**: communication-bound → RDMA 通信最適化が直接効く
- **MoE モデル (Qwen3.5)**: compute-bound (GPU 94.8%) → サーバー側並列化が効く
- 同じ施策でもモデルにより +25.9% (Qwen3.5) と -0.33% (GLM-4.7) のように正反対の結果になる

### 2. 組み合わせ効果の重要性

- Per-device connections 単独: PP -1.6% (悪化)
- Pipeline 単独: PP +0.4% (ns)
- **両方の組み合わせ: PP +25.9%**
- 単独で効果がない施策でも組み合わせにより劇的改善が生まれる場合がある

### 3. パラメータチューニングの費用対効果

- **コード変更ゼロの ubatch チューニング (+50-70%)** が最も費用対効果が高い
- 複雑な RDMA 最適化 (数週間の実装) よりも `-ub 2048` の 1 行変更が大きい改善をもたらすケースがある

### 4. 正確性修正と性能のトレードオフ

- Selective Signaling (+26.1%) は race condition の修正で打ち消された
- Ring buffer による回復を試みたが、ボトルネックが inline コマンドにあり回復不可
- **不正なデータ上の性能数値は参考にならない** — 正確性が最優先

### 5. P100 の構造的制約

- cc 6.0 は DP4A (MMQ), Tensor Core (MMA), CUDA Graph のいずれも非対応
- MoE の `mul_mat_id` が最遅フォールバックパスに落ちる (ubatch あたり 80-120 回の stream sync)
- これは RDMA 最適化では対処不可能な GPU アーキテクチャレベルの制約
