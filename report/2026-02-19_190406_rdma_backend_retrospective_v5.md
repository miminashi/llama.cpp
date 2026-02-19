# RDMA バックエンド 振り返りレポート v5

- **実施日時**: 2026年2月19日 19:04
- **ワークツリー**: `.worktree/rdma-backend`
- **前回レトロスペクティブ**: [2026-02-13_185255_rdma_backend_retrospective_v4.md](2026-02-13_185255_rdma_backend_retrospective_v4.md)

---

## 前回レトロスペクティブ (v4) からの主要変更点

v4 (02/13 18:52) 以降の約6日間で **15 本のレポート** (#67〜#81) が追加された。

- **実装**: 4件 (server-push, selective signaling, hugepage staging, row split fit_params)
- **調査**: 7件 (synchronize 再検証×2, 深堀り調査, 棚卸し, sched 分析, nvidia-peermem 調査, row vs layer)
- **インフラ**: 2件 (nvidia-peermem 復旧, GDR バジェット拡大テスト)
- **検証**: 2件 (deferred copy 再評価, PD 共有 per-device)

---

## v4→v5 統合サマリーテーブル (全15項目)

| # | 修正/実験内容 | レポート | ワークツリー | tg 改善 | マージ状況 |
|---|-------------|---------|------------|--------|-----------|
| 1 | synchronize 二峰性修正 & 効果検証 | [#67](2026-02-14_024402_synchronize_bimodal_fix_and_benchmark.md) | `rdma-synchronize` | なし (交絡因子あり) | 未マージ |
| 2 | synchronize 交絡因子排除再測定 | [#68](2026-02-14_055914_synchronize_confound_free_remeasurement.md) | `rdma-synchronize` | なし (p=0.051-0.565) | 未マージ |
| 3 | Deferred copy 再評価 (二峰性修正後) | [#69](2026-02-14_082220_deferred_copy_clean_ab_benchmark.md) | `rdma-backend` | **+1.45%** (p<10⁻¹², d=5.89) | - |
| 4 | Generation 最適化 深堀り調査 | [#70](2026-02-18_070912_generation_optimization_deep_investigation.md) | `rdma-backend` | (調査) | - |
| 5 | 未実装最適化の棚卸し | [#71](2026-02-18_120000_unimplemented_optimizations_inventory.md) | `rdma-backend` | (調査) | - |
| 6 | PD 共有 Per-device connection | [#72](2026-02-18_130000_shared_pd_per_device_connection.md) | `rdma-shared-pd` | **-2.96%** (p<10⁻¹¹) | 未マージ |
| 7 | Server-Push (RDMA Write IMM) | [#73](2026-02-18_200000_server_push_rdma_write_imm.md) | `rdma-server-push` | **+0.74%** (p<10⁻¹⁰, d=4.30) | 未マージ |
| 8 | ggml_backend_sched ボトルネック分析 | [#74](2026-02-19_090721_ggml_backend_sched_bottleneck_analysis.md) | `rdma-backend` | (調査) | - |
| 9 | Hugepage ステージングバッファ | [#75](2026-02-19_104648_hugepage_staging_benchmark.md) | `hugepage-staging` | +0.29% (非有意, p=0.11) | 未マージ |
| 10 | Selective signaling | [#76](2026-02-19_120000_selective_signaling.md) | `rdma-selective-signaling` | **+0.89%** (p=0.0009, d=1.08) | 未マージ |
| 11 | nvidia-peermem ロード失敗調査 | [#77](2026-02-19_125638_nvidia_peermem_load_failure.md) | `rdma-backend` | (調査) | - |
| 12 | nvidia-peermem 復旧 | [#78](2026-02-19_133220_nvidia_peermem_kernel_downgrade.md) | `rdma-backend` | (インフラ) | - |
| 13 | Hugepage + GDR バジェット拡大 | [#79](2026-02-19_160000_hugepage_gdr_benchmark.md) | `hugepage-staging` | なし (非有意) | 未マージ |
| 14 | Row vs Layer split ベンチマーク | [#80](2026-02-19_163700_row_vs_layer_split_benchmark.md) | `rdma-backend` | (検証) | - |
| 15 | Row split fit_params 修正 | [#81](2026-02-19_200000_row_split_fit_params_fix.md) | `rdma-row-split` | (機能追加) | 未マージ |

- **マージ状況の凡例**:
  - `-`: rdma-backend で直接作業、またはコード変更なし (マージ対象なし)
  - `マージ済み`: ワークツリーのコード変更が rdma-backend に取り込まれている
  - `未マージ`: ワークツリーにコード変更があるが rdma-backend には未取り込み
  - レポートは全て rdma-backend に作成されるため、マージ状況には含めない

---

## v4→v5 の重要な知見

### 知見1: ggml_backend_sched の逐次性が根本ボトルネック

v4 以降、Generation 速度改善のために多数の最適化を試みたが、ほぼすべてが同じ壁に衝突した。

**壁の正体**: llama.cpp のバックエンドスケジューラ (`ggml_backend_sched`) は、各デバイスに対して `graph_compute → get_tensor` を**逐次**実行する。つまり、デバイス D0 の計算完了を待ってから D1 に発行する。

**影響を受けた最適化の一覧と経緯**:

| 最適化 | 期待した効果 | 実際の結果 | なぜ効かなかったか |
|--------|------------|-----------|------------------|
| Async compute (v4 #51) | graph_compute 非同期化で並列発行 | e2e 改善なし | get_tensor で暗黙的同期が入り、savings が吸収される |
| Parallel dispatch (v4 #57) | サーバー側デバイス並列化 | サーバーは正常に並列化、e2e なし | クライアントが逐次発行するためサーバー並列化の恩恵なし |
| Per-device connection (#72) | 接続分離でコマンド並列化 | **-2.96%** (劣化) | 並列化効果なし + deferred copy が接続間で無効化 (-1.45%) |
| Server-Push (#73) | サーバー主導 RDMA Write で RTT 削減 | +0.74% のみ | クライアント逐次処理でバンド幅改善の効果が限定的 |
| Two-phase loop (v4 #63) | Phase1: 全デバイス非同期発行 → Phase2: 全 get_tensor 回収 | 効果なし | Layer split ではデバイス間にデータ依存があり並列化不可能 |
| SYNC command (#67-68) | synchronize で明示的同期 | 効果なし (p=0.051-0.565) | Deferred copy の効果と交絡しており、独立効果はゼロ |

**結論**: RDMA バックエンド側でできる改善は限界に達した。次のステップとして有効なのは:
1. クライアント側 (`ggml_backend_sched`) のアーキテクチャ変更 (server-side pipeline 等)
2. Layer split ではなく異なる並列化戦略 (Expert Parallelism 等)
3. ハードウェア更新 (ConnectX-6 で MTT 制限解消 → per-device connection + deferred copy 両立)

### 知見2: 有効だった最適化は「逐次性に依存しない」もの

逐次ループの壁にぶつからなかった最適化のみが効果を示した:

| 最適化 | tg 改善 | なぜ効いたか |
|--------|--------|------------|
| Deferred copy (#69) | **+1.45%** | IB ネットワーク RTT を排除 (サーバーローカル D2H+H2D)。デバイス並列化に依存しない |
| Server-Push (#73) | **+0.74%** | get_tensor の要求→応答 RTT を削減。小さいが並列化不要の改善 |
| Selective signaling (#76) | **+0.89%** | CQ ポーリング回数削減。NIC レベルの効率化で並列化に非依存 |

共通点: すべて**1回のリクエスト内の効率改善**であり、デバイス間並列化を必要としない。

### 知見3: GDR 環境の脆弱性 (nvidia-peermem 1週間停止)

**何が起きたか**: 02-12〜02-19 の約1週間、GPUDirect RDMA (GDR) が無効状態だった。

**原因の連鎖**:
1. Ubuntu の unattended-upgrades がカーネルを 6.8.0-90 → 6.8.0-100 に自動更新
2. NVIDIA DKMS は新カーネル用モジュールを自動ビルド → nvidia ドライバは動作継続
3. MLNX_OFED は 6.8.0-90 カーネル向けにのみビルド済み → 6.8.0-100 では `ib_core` 等がロード不可
4. `nvidia-peermem` は `ib_core` に依存 → ロード失敗 → GDR 無効化
5. RDMA 自体は libibverbs (ユーザー空間) で動作するため、カーネルモジュール不在でも通信は可能 → GDR 停止に気付きにくい

**復旧**: 2号機のカーネルを 6.8.0-90 にダウングレード + `/etc/apt/apt.conf.d/50unattended-upgrades` でカーネルパッケージを除外

**教訓**: GDR 環境は NVIDIA ドライバ + MLNX_OFED + カーネル の3つの整合性に依存しており、いずれかの自動更新で壊れる。`rdma-env-check.sh` に nvidia-peermem チェックを入れたことで早期検出可能になった。

### 知見4: Row split は P100 (NVLink なし) で全条件で劣位

gpt-oss-20b Q4_K_M での再検証 (#80):
- 2 GPU: layer split が row split 比 pp512 +17%, tg +14%
- 7 GPU: layer split が row split 比 pp512 +64%, tg +51%
- Row split は GPU 数増加で**負のスケーリング** (7GPU で 2GPU の 70.9%)
- P100 に NVLink がないため GPU 間通信が PCIe 経由 → row split のオールリデュースがボトルネック化

---

## 性能サマリー更新

### v5 で確認された有効な改善

| 最適化 | tg 改善 | 統計的有意性 | 備考 |
|--------|:------:|:----------:|------|
| Deferred copy | +1.45% | p<10⁻¹², d=5.89 | v4 でマージ済み、v5 で統計的に再確認 |
| Server-Push | +0.74% | p<10⁻¹⁰, d=4.30 | 未マージ |
| Selective signaling | +0.89% | p=0.0009, d=1.08 | 未マージ |
| **合計理論値** | **~+3.1%** | — | 独立効果仮定、実際は相互作用あり得る |

### v5 で否定された最適化

| 最適化 | 結果 | 理由 |
|--------|------|------|
| SYNC command | 0% | Deferred copy と交絡、独立効果なし |
| PD 共有 per-device connection | -2.96% | Deferred copy 無効化 + 並列化効果なし |
| Hugepage staging (GDR なし) | +0.29% (非有意) | MTT エントリ削減確認も速度効果なし |
| GDR バジェット 12→48GB 拡大 | 0% (非有意) | 1 GPU の GDR で pp 改善は十分、追加 GPU の GDR は tg に寄与しない |

### Step ごとのベストスコア (更新版)

| Step | 構成 | モデル | pp (t/s) | tg (t/s) | ローカル比 (tg) |
|------|------|--------|:--------:|:--------:|:--------------:|
| Step 1 完了 | RDMA 1+1 | qwen2.5-0.5b | — | 148.5 | 75.7% |
| Step 2 完了 | 11GPU (7C+4R) | gpt-oss-120b | 3〜11 | 0.7〜1.1 | 1.5〜2.3% |
| Step 3 完了 | RDMA 1+1 | gpt-oss-20b | 61.09 | 58.47 | **91.0%** |
| Step 4 完了 (GDR) | RDMA 1+1 | gpt-oss-20b | **403.69** | 59.52 | **92.6%** |
| Step 4 完了 (クラスタ) | 11GPU (7C+4R) | gpt-oss-120b | 201.02 | 36.92 | 84.7% |
| Step 5 (v4 時点) | 11GPU (7C+4R) | GLM-4.7 IQ2_M | 7.4 | 7.4〜7.5 | — |
| **Step 5 (v5 時点)** | **11GPU (7C+4R)** | **GLM-4.7 IQ2_M** | **7.4** | **7.65〜7.76** | **—** |

### GLM-4.7 RDMA vs RPC 比較 (更新版)

| バックエンド | Prompt (t/s) | Generation (t/s) | 備考 |
|-------------|:------------:|:----------------:|------|
| **RDMA (v5, server-push)** | **7.4** | **7.76** | Server-Push ON、deferred copy 有効 |
| RDMA (v4, fprintf 修正後) | 7.4 | 7.4〜7.5 | — |
| RDMA (v3, GDR budget 12GB) | 6.4 | 6.8 (ばらつき大) | 二峰性あり |
| RDMA (GDR 無効) | 6.4 | 6.0 | — |
| RPC (TCP) | 5.4 | 7.5 | — |

- **v5 時点**: RDMA は Prompt で RPC 比 **+37%**、Generation でも RPC を **+3.5% 上回る** (7.76 vs 7.5)
- v4 までは Generation で RPC と同等 (7.4-7.5) だったが、Server-Push + Selective signaling で**逆転**

---

## 新規ワークツリー一覧 (v4 以降: 5件)

| ワークツリー | ブランチ | 目的 | 備考 |
|-------------|---------|------|------|
| `hugepage-staging` | `feature/hugepage-staging` | Hugepage ステージングバッファ | 未マージ (速度効果なし) |
| `rdma-row-split` | `feature/rdma-row-split` | Row split fit_params 修正 | 未マージ (機能追加) |
| `rdma-selective-signaling` | `feature/rdma-selective-signaling` | Selective signaling | 未マージ (+0.89%) |
| `rdma-server-push` | `feature/rdma-server-push` | Server-Push (RDMA Write IMM) | 未マージ (+0.74%) |
| `rdma-shared-pd` | `feature/rdma-shared-pd` | PD 共有 per-device connection | 未マージ (-2.96%) |

### ワークツリー全体サマリー (更新版)

```
メイン:          1 ワークツリー (rdma-backend)
コード変更なし:  2 ワークツリー (調査のみ)
マージ済み:      1 ワークツリー (fix-bimodal-fprintf)
未マージ:       19 ワークツリー (実験/開発コード)
合計:           23 ワークツリー (master 除く)
```

---

## 新規レポート一覧 (#67-#81)

| # | 日付 | ファイル | タイトル | 区分 |
|---|------|---------|---------|:----:|
| 67 | 02/14 | [2026-02-14_024402](2026-02-14_024402_synchronize_bimodal_fix_and_benchmark.md) | synchronize 二峰性修正マージ & 効果検証 | 検証 |
| 68 | 02/14 | [2026-02-14_055914](2026-02-14_055914_synchronize_confound_free_remeasurement.md) | synchronize 交絡因子排除再測定 | 検証 |
| 69 | 02/14 | [2026-02-14_082220](2026-02-14_082220_deferred_copy_clean_ab_benchmark.md) | Deferred copy 再評価ベンチマーク (二峰性修正後) | 検証 |
| 70 | 02/18 | [2026-02-18_070912](2026-02-18_070912_generation_optimization_deep_investigation.md) | Generation 性能最適化 深堀り調査 | 調査 |
| 71 | 02/18 | [2026-02-18_120000](2026-02-18_120000_unimplemented_optimizations_inventory.md) | 未実施最適化の棚卸し | 調査 |
| 72 | 02/18 | [2026-02-18_130000](2026-02-18_130000_shared_pd_per_device_connection.md) | PD 共有 Per-device connection | 実装 |
| 73 | 02/18 | [2026-02-18_200000](2026-02-18_200000_server_push_rdma_write_imm.md) | Server-Push (RDMA Write with IMM) | 実装 |
| 74 | 02/19 | [2026-02-19_090721](2026-02-19_090721_ggml_backend_sched_bottleneck_analysis.md) | ggml_backend_sched 逐次処理ボトルネック分析 | 調査 |
| 75 | 02/19 | [2026-02-19_104648](2026-02-19_104648_hugepage_staging_benchmark.md) | Hugepage ステージングバッファ | 実装 |
| 76 | 02/19 | [2026-02-19_120000](2026-02-19_120000_selective_signaling.md) | Selective signaling | 実装 |
| 77 | 02/19 | [2026-02-19_125638](2026-02-19_125638_nvidia_peermem_load_failure.md) | nvidia-peermem ロード失敗調査 | 調査 |
| 78 | 02/19 | [2026-02-19_133220](2026-02-19_133220_nvidia_peermem_kernel_downgrade.md) | nvidia-peermem 復旧 (カーネルダウングレード) | Infra |
| 79 | 02/19 | [2026-02-19_160000](2026-02-19_160000_hugepage_gdr_benchmark.md) | Hugepage + GDR バジェット拡大 | 検証 |
| 80 | 02/19 | [2026-02-19_163700](2026-02-19_163700_row_vs_layer_split_benchmark.md) | Row vs Layer split ベンチマーク | 検証 |
| 81 | 02/19 | [2026-02-19_200000](2026-02-19_200000_row_split_fit_params_fix.md) | Row split fit_params 修正 | 実装 |

---

## v4 以降のコミット (rdma-backend ブランチ)

| コミット | 内容 |
|---------|------|
| `314686613` | docs: add v4 retrospective report with integrated summary table |
| `c576d3df0` | docs: fix notation consistency and add legend to v4 retrospective summary table |
| `b20672d37` | docs: add git -C rule and A/B benchmark statistics methodology to CLAUDE.md |
| `5d1f7e273` | docs: extract Step 1-4 history to HISTORY.md and reorganize CLAUDE.md |
| `2423992c7` | docs: add deferred copy and synchronize benchmark reports |
| `1aa0643b0` | docs: add permission settings rationale and update rules in CLAUDE.md |
| `b1a6f06b3` | feat: add GDB debug support for RDMA backend |
| `6878be7f6` | docs: update permission rules — SSH auto-approval fix, worktree and tool path rules |
| `db6b6033c` | docs: add optimization investigation and benchmark reports |
| `c33d1b86a` | feat: add multi-session workflow infrastructure (task board + GPU lock) |
| `1e256325d` | docs: add GPUDirect RDMA (GDR) setup guide to CLAUDE.md |
| `310682c72` | feat: add pre-experiment environment check script (rdma-env-check.sh) |
| `5dea2e773` | docs: clarify worktree merge policy — commit locally, no merge to rdma-backend |
| `cfd75866c` | docs: expand GPU lock scope and add GDR troubleshooting to CLAUDE.md |
| `184b99ba2` | feat: add process detection fallback to gpu-lock.sh and strengthen CLAUDE.md |
| `79bad6600` | feat: support inline env vars in gpu-lock.sh (KEY=VALUE before command) |
| `2b2c975ac` | docs: add inline env var example to gpu-lock.sh usage and update layer split benchmarks |
| `9fb5b7e64` | docs: fix CLAUDE.md contradictions with current code state |
| `dc66a55dc` | docs: extract procedural sections from CLAUDE.md into Claude Code skills |
| `f07146517` | docs: add benchmark reports and permissions reference |

---

## Step 別ステータス (更新版)

| Step | 内容 | 状態 | v4 からの変更 |
|------|------|:----:|:-------------:|
| Step 1 | 基本 RDMA 通信 | ✅ 完了 | なし |
| Step 2 | マルチノードクラスタ安定化 | ✅ 完了 | なし |
| Step 3 | RDMA 性能最適化 | ✅ 完了 | なし |
| Step 4 | GPUDirect RDMA 有効化 | ✅ 完了 | なし |
| Step 5 | GLM-4.7 Q4 on 16 P100s | 🔧 進行中 | 最適化限界到達、GDR 復旧 |

### Step 5 の進捗更新

| マイルストーン | v4 時点 | v5 時点 |
|---------------|--------|--------|
| GLM-4.7 IQ2_M 動作 | **安定動作** (7.4-7.5 t/s) | **安定動作** (7.65-7.76 t/s, server-push) |
| synchronize 検証 | 実装済み・未テスト | **検証完了** (効果なし、p=0.051-0.565) |
| Deferred copy 統計再確認 | 実装済み (+13% on 20b) | **GLM-4.7 で +1.45% 確認** (p<10⁻¹²) |
| Server-Push | 未実装 | **実装済み** (+0.74%, p<10⁻¹⁰) |
| Selective signaling | 未実装 | **実装済み** (+0.89%, p=0.0009) |
| PD 共有 per-device | 未実装 | **検証完了** (劣化 -2.96%、不採用) |
| Hugepage staging | 未実装 | **実装済み** (速度効果なし) |
| Row split fit_params | 未対応 | **修正済み** (CPU オフロード対応) |
| nvidia-peermem 復旧 | 動作中 | **復旧完了** (1週間停止→カーネル DG) |
| sched ボトルネック文書化 | 未文書化 | **完了** (#74) |
| 最適化棚卸し | 未整理 | **完了** (#71: 全項目に結論) |
| 16GPU ハードウェア | 待ち | **待ち** (変更なし) |
| Q3_K_M/Q4_K_M テスト | 未着手 | **未着手** (16GPU 待ち) |

---

## コード統計 (更新版)

### ファイル別行数

| ファイル | v4 行数 | v5 行数 | 差分 | 役割 |
|---------|-------:|-------:|----:|------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 3,735 | 3,735 | 0 | メイン実装 |
| `ggml/src/ggml-rdma/rdma-transport.cpp` | 968 | 968 | 0 | RDMA 接続管理 |
| `ggml/src/ggml-rdma/rdma-gdr.cpp` | 444 | 444 | 0 | GPUDirect RDMA |
| `ggml/src/ggml-rdma/rdma-memory.cpp` | 307 | 307 | 0 | ホストメモリ管理 |
| `ggml/src/ggml-rdma/rdma-transport.h` | 187 | 187 | 0 | トランスポート層ヘッダ |
| `ggml/src/ggml-rdma/rdma-gdr.h` | 116 | 116 | 0 | GPUDirect ヘッダ |
| `ggml/src/ggml-rdma/rdma-memory.h` | 109 | 109 | 0 | メモリ管理ヘッダ |
| `ggml/src/ggml-rdma/CMakeLists.txt` | 70 | 70 | 0 | ビルド設定 |
| `ggml/include/ggml-rdma.h` | 79 | 79 | 0 | 公開 C API |
| `tools/rdma/rdma-server.cpp` | 283 | 283 | 0 | サーバーエントリポイント |
| **RDMA コア合計** | **6,298** | **6,298** | **0** | **10 ファイル** |

v4→v5 では RDMA コアコードに変更なし。最適化実装は全てワークツリー上で行われ、rdma-backend にはマージされていない。

### ヘルパースクリプト

| スクリプト | v4 行数 | v5 行数 | 差分 | 備考 |
|-----------|-------:|-------:|----:|------|
| `scripts/rdma-build.sh` | 40 | 48 | +8 | — |
| `scripts/rdma-deploy.sh` | 21 | 27 | +6 | — |
| `scripts/rdma-server.sh` | 60 | 77 | +17 | — |
| `scripts/rdma-env-check.sh` | — | 360 | +360 | v5 で新規追加 |
| `scripts/gpu-lock.sh` | — | 157 | +157 | v5 で新規追加 |
| `scripts/task-board.sh` | — | 369 | +369 | v5 で新規追加 |
| **合計** | **121** | **1,038** | **+917** | — |

### コミット・レポート統計

| 指標 | v4 | v5 | 差分 |
|------|---:|---:|-----:|
| RDMA 関連コミット数 (ソースコード変更) | 18 | 18 | 0 |
| 全コミット数 (feature/rdma-backend) | 32 | 54 | +22 |
| レポート数 (MD ファイル) | 66 | 82 | +16 |
| git diff insertions (vs master) | 25,143 | 31,056 | +5,913 |
| 変更ファイル数 (vs master) | 94 | 122 | +28 |

---

## 残タスク (更新版)

| # | タスク | v4 状態 | v5 状態 | 優先度 | ブロッカー |
|---|--------|---------|---------|:------:|-----------|
| 1〜3 | (Step 1-3 タスク) | 完了 | 完了 | — | — |
| 4 | 16GPU ハードウェア準備 | 待ち | **待ち** (変更なし) | 高 | 物理 GPU 追加 |
| 5 | Q3_K_M/Q4_K_M ダウンロード・テスト | 未着手 | **未着手** (16GPU 待ち) | 高 | タスク 4 |
| 6 | 16GPU 全体テスト | 未着手 | **未着手** (タスク 4,5 依存) | 高 | タスク 4, 5 |
| 7 | RDMA Generation 速度改善 | 調査完了 | **限界到達** (sched 逐次性) | 低 | `ggml_backend_sched` 変更 |
| 8 | Server-Push + Selective signaling マージ | (未実装) | **実装完了・マージ待ち** | 中 | ユーザー判断 |

### タスク #7: Generation 速度改善の限界

v4 で計画されていた改善案は v5 で全て検証完了:

| 改善案 | v4 状態 | v5 結論 |
|--------|---------|---------|
| Two-phase loop | 設計完了 | **効果なし** (layer split のデータ依存) |
| synchronize (SYNC) | 実装済み・未テスト | **効果なし** (p=0.051-0.565) |
| Per-device connection (PD 共有) | 未実装 | **劣化** (-2.96%) |
| Server-Push | 未実装 | **+0.74%** (小効果) |
| Selective signaling | 未実装 | **+0.89%** (小効果) |

**結論**: RDMA バックエンド側での大幅な Generation 改善は達成不可能。残された有効手段は Server-Push (+0.74%) と Selective signaling (+0.89%) のマージのみ。

### タスク #8: マージ候補

| ワークツリー | 効果 | マージ推奨度 | 理由 |
|-------------|------|:----------:|------|
| `rdma-server-push` | +0.74% | **推奨** | 統計的に有意、副作用なし |
| `rdma-selective-signaling` | +0.89% | **推奨** | 統計的に有意、副作用なし |
| `rdma-row-split` | — | 中 | 機能追加 (fit_params 修正)、row split 自体は不使用 |
| `hugepage-staging` | 0% | 低 | 速度効果なし |
| `rdma-shared-pd` | -2.96% | **非推奨** | 劣化 |
| `rdma-synchronize` | 0% | 低 | 独立効果なし |

---

## プロジェクト全体の振り返り

### 20日間の成果総括

2026年1月31日の実装開始から2月19日まで、**20日間** で RDMA バックエンドを `ゼロから安定動作+最適化限界到達` まで進めた。

### 性能進化の軌跡

```
Day 1  (02/01): 0.3 t/s    — 初回動作 (naive protocol)
Day 4  (02/04): 148.5 t/s  — プロトコル最適化完了 (495×)
Day 5  (02/05): 59.5 t/s   — GPUDirect RDMA (20b, 92.6% of local)
Day 7  (02/07): 6.8 t/s    — GLM-4.7 47B on 11 GPUs
Day 10 (02/09): 7.2 t/s    — 最良値 (安定化後) ← v3 時点
Day 14 (02/13): 7.5 t/s    — 二峰性修正 + deferred copy ← v4 時点
Day 20 (02/19): 7.76 t/s   — server-push + selective signaling ← v5 時点
```

### v4→v5 の技術的困難と解決策

| 困難 | 根本原因 | 解決策/結論 | 教訓 |
|------|---------|------------|------|
| synchronize 効果測定の交絡 | Deferred copy 有無の差が SYNC 効果を隠蔽 | `GGML_RDMA_NO_DEFERRED_COPY=1` で deferred copy を無効化し再測定 → 独立効果ゼロ | **A/B 比較では交絡因子を厳密に排除すべき** |
| PD 共有 per-device で劣化 | Deferred copy が接続間で無効化 (-1.45%) + 並列化効果なし | 不採用 | **最適化の組み合わせ効果は非加法的。一方を有効にすると他方が無効化される場合がある** |
| nvidia-peermem 1週間停止 | カーネル自動更新で MLNX_OFED モジュール不整合 | カーネル DG + 自動更新除外 | **3層依存 (カーネル+OFED+NVIDIA) は自動更新で容易に壊れる** |
| Generation 改善の壁 | `ggml_backend_sched` の逐次処理モデル | 文書化 (#74)、限界を受容 | **ボトルネックがアーキテクチャレベルの場合、局所最適化の積み重ねでは解消不可能** |

---

## 全レポート一覧 (更新版)

v4 の 66 本 → v5 含め **82 本** (+ 補助ドキュメント 3 本)

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
| **67** | **02/14** | [2026-02-14_024402](2026-02-14_024402_synchronize_bimodal_fix_and_benchmark.md) | **synchronize 二峰性修正マージ & 効果検証** | **検証** |
| **68** | **02/14** | [2026-02-14_055914](2026-02-14_055914_synchronize_confound_free_remeasurement.md) | **synchronize 交絡因子排除再測定** | **検証** |
| **69** | **02/14** | [2026-02-14_082220](2026-02-14_082220_deferred_copy_clean_ab_benchmark.md) | **Deferred copy 再評価 (二峰性修正後)** | **検証** |
| **70** | **02/18** | [2026-02-18_070912](2026-02-18_070912_generation_optimization_deep_investigation.md) | **Generation 最適化 深堀り調査** | **調査** |
| **71** | **02/18** | [2026-02-18_120000](2026-02-18_120000_unimplemented_optimizations_inventory.md) | **未実施最適化の棚卸し** | **調査** |
| **72** | **02/18** | [2026-02-18_130000](2026-02-18_130000_shared_pd_per_device_connection.md) | **PD 共有 Per-device connection** | **実装** |
| **73** | **02/18** | [2026-02-18_200000](2026-02-18_200000_server_push_rdma_write_imm.md) | **Server-Push (RDMA Write with IMM)** | **実装** |
| **74** | **02/19** | [2026-02-19_090721](2026-02-19_090721_ggml_backend_sched_bottleneck_analysis.md) | **ggml_backend_sched ボトルネック分析** | **調査** |
| **75** | **02/19** | [2026-02-19_104648](2026-02-19_104648_hugepage_staging_benchmark.md) | **Hugepage ステージングバッファ** | **実装** |
| **76** | **02/19** | [2026-02-19_120000](2026-02-19_120000_selective_signaling.md) | **Selective signaling** | **実装** |
| **77** | **02/19** | [2026-02-19_125638](2026-02-19_125638_nvidia_peermem_load_failure.md) | **nvidia-peermem ロード失敗調査** | **調査** |
| **78** | **02/19** | [2026-02-19_133220](2026-02-19_133220_nvidia_peermem_kernel_downgrade.md) | **nvidia-peermem 復旧 (カーネルダウングレード)** | **Infra** |
| **79** | **02/19** | [2026-02-19_160000](2026-02-19_160000_hugepage_gdr_benchmark.md) | **Hugepage + GDR バジェット拡大** | **検証** |
| **80** | **02/19** | [2026-02-19_163700](2026-02-19_163700_row_vs_layer_split_benchmark.md) | **Row vs Layer split ベンチマーク** | **検証** |
| **81** | **02/19** | [2026-02-19_200000](2026-02-19_200000_row_split_fit_params_fix.md) | **Row split fit_params 修正** | **実装** |
| **82** | **02/19** | [2026-02-19_190406](2026-02-19_190406_rdma_backend_retrospective_v5.md) | **振り返りレポート v5 (本レポート)** | **—** |

### レポート数とファイル数

- report/ ディレクトリには通し番号付き .md レポート **82 本** + 補助 .md **1 本** + .html **2 本** = 計 **85 ファイル** が存在
- 通し番号は .md レポートのみを対象 (#1〜#82)
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
