# Memory → CLAUDE.md / スキル / ISSUES.md 反映レポート

- **実施日時**: 2026年3月2日 02:31
- **ワークツリー**: なし (`feature/rdma-backend` 本体で直接編集)

## 前提・目的

memory ファイル (`MEMORY.md`) に蓄積された知見が CLAUDE.md やスキルファイルに反映されておらず、以下の問題があった:

- CLAUDE.md の達成数値注記 (pp128 ≈ 30.6) が always-signal fix 後の実態 (≈ 24.3) と乖離
- bench スキルの期待値も同様に古い値 (30.6) のまま
- bench スキルに環境変数 4 件 (`GGML_RDMA_NO_SELECTIVE_SIGNAL`, `GGML_RDMA_PARALLEL_DISPATCH`, `GGML_RDMA_PARALLEL_COMPUTE`, `GGML_RDMA_NO_SERVER_PUSH`) が未記載
- memory に蓄積されたエラー対処・ベンチマーク tips がスキルに反映されていない
- CLAUDE.md の「残課題・懸念事項」に memory の性能課題 (Send selective signaling 回復、Expert parallelism GDR 制約) が統合されていない
- 課題管理情報が CLAUDE.md に埋め込まれており、メンテナンス性が低い

### 目的

1. 課題管理を `ISSUES.md` に分離し、memory の知見を統合
2. CLAUDE.md の達成数値を最新値に更新
3. bench / gdr スキルに memory の知見を反映

## 変更内容

### 1. ISSUES.md の新設

CLAUDE.md の「残タスク」「残課題・懸念事項」セクション (旧 L183-211) を `ISSUES.md` に移行し、以下の構成で整理:

| セクション | 内容 | 出典 |
|-----------|------|------|
| 残タスク | 16GPU 拡張、GLM-4.7 Q4 準備 | CLAUDE.md から移行 |
| 性能課題 | Send selective signaling 回復、Expert parallelism GDR 制約、RPC 比劣位、セッション間変動、GDR バジェット | CLAUDE.md + MEMORY.md |
| 既知の制約 | ggml_backend_sched 逐次処理、PCIe MaxPayload 256B、gpt-oss-20b bimodal tg | MEMORY.md + レポート群 |

memory の知見として新規追加した項目:

- **Send selective signaling の回復** — pp128: 30.6→24.3 の原因・改善案 (double-buffering)・関連ワークツリー
- **Expert parallelism の GDR 制約** — sync-before-recv の必要性・per-device write fencing の改善案
- **ggml_backend_sched 逐次処理制約** — 全デバイス逐次化の根本原因、参照レポートへのリンク
- **PCIe MaxPayload 256B** — P100 ハードウェア上限、帯域非対称性 (Write 9.8 vs Read 3.4 GB/s)
- **gpt-oss-20b bimodal tg 分布** — fprintf 修正後も残る二峰性、参照レポートへのリンク

### 2. CLAUDE.md の更新

| 箇所 | 変更前 | 変更後 |
|------|--------|--------|
| L173 注記 | pp128 ≈ 30.6, tg32 ≈ 8.5 | pp128 ≈ 24.3, tg32 ≈ 8.5 + always-signal 影響の説明 |
| L183-211 | 残タスク・残課題 (29行) | `ISSUES.md` へのリンク (3行) |

### 3. bench スキル更新

| 変更種別 | 詳細 |
|---------|------|
| 環境変数 +4 | `GGML_RDMA_NO_SELECTIVE_SIGNAL`, `GGML_RDMA_PARALLEL_DISPATCH`, `GGML_RDMA_PARALLEL_COMPUTE`, `GGML_RDMA_NO_SERVER_PUSH` |
| 期待値更新 | Prompt ≈ 30.6 → 24.3 t/s |
| エラー対処 +4 | RNR NAK スパイク、GDR+parallel dispatch クラッシュ、ゴミ文字出力、NaN 出力 |
| 新セクション | ベンチマークスクリプト Tips (CSV パース、サーバー再起動要件、set -e 対策、long flags) |
| 新セクション | RPC バックエンド使用法 (--rpc フラグ、rpc-server 起動例) |

### 4. gdr スキル更新

トラブルシューティングテーブルに 1 件追加:
- GDR + parallel dispatch でクラッシュ → sync-before-recv / `GGML_RDMA_PARALLEL_DISPATCH=0`

## 対象ファイル一覧

| ファイル | 操作 |
|---------|------|
| `ISSUES.md` | 新規作成 |
| `CLAUDE.md` | 編集 (注記更新 + セクション置換) |
| `.claude/skills/bench/SKILL.md` | 編集 (環境変数・期待値・エラー・新セクション2つ) |
| `.claude/skills/gdr/SKILL.md` | 編集 (トラブルシューティング1件追加) |

## 検証

- ISSUES.md が CLAUDE.md 旧セクション全項目 + memory 知見 6 件を網羅していることを確認
- bench スキルの環境変数テーブルが全 15 環境変数を含むことを確認
- CLAUDE.md の達成数値が最新値 (pp128 ≈ 24.3) と一致することを確認
- 計画で `trimodal` と記載されていた項目は、レポート調査の結果 `bimodal` (二峰性) が正確であったため修正して採用
- 計画で `PCIe MaxReadReq (256B)` と記載されていた項目は、実際には `MaxPayload` が 256B のハードウェア制約であり、`MaxReadReq` は 512B (変更可能) であったため正確に記載
