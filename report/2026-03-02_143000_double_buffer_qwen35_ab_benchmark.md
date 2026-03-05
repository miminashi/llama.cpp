# Double-Buffering A/B ベンチマーク: Qwen3.5-35B-A3B 6GPU マルチデバイス推論

- **実施日時**: 2026年3月2日 14:30
- **ワークツリー**: `.worktree/double-buffering`
- **ブランチ**: `feature/double-buffering`
- **コミット**: `c8ca4b1f6` (feat(rdma): skip is_last signaling for async fire-and-forget sends)

## 前提・目的

### 背景

前回のベンチマーク ([skip_last_signal レポート](2026-03-02_063140_skip_last_signal_benchmark.md)) では gpt-oss-20b (12GB, 単一 P100 に収容可能) を使用したため、llama-bench はデバイスごと個別ベンチマークを実行し、マルチデバイス統合推論のテストではなかった。

本テストでは、マルチ GPU 必須のモデル (Qwen3.5-35B-A3B, 22GB) を 6GPU (2C+4R) で使用し、マルチデバイス統合推論における selective signaling + double-buffering の効果を検証する。

### 目的

- double-buffering 実装 (3コミット) が安全に selective signaling を有効化できるか検証
- マルチデバイス統合推論で pp/tg 両方の性能影響を測定
- gpt-oss-20b の結果パターンがマルチデバイスでも再現するか確認

### テスト対象コミット

| コミット | 変更内容 |
|---------|---------|
| `ec2fb5a89` | feat(rdma): double-buffer staging and send buffers for selective signaling |
| `fa8f7c6a6` | fix(rdma): limit signal interval to 2 for double-buffered paths |
| `c8ca4b1f6` | feat(rdma): skip is_last signaling for async fire-and-forget sends |

## A/B ベンチマーク

### テスト設計

| 項目 | 設定 |
|------|------|
| モデル | Qwen3.5-35B-A3B (UD-Q4_K_M, 22GB MoE, 3B active) |
| GPU 構成 | CUDA5-6 + RDMA0-3 (6GPU: 2C+4R) |
| ツール | llama-cli (`--simple-io --single-turn --no-warmup`) |
| プロンプト | RDMA 技術説明文 (~50 tokens) |
| 生成数 | 32 tokens |
| A 条件 | `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` (全 WR シグナリング) |
| B 条件 | デフォルト (selective signaling + double-buffering + skip_last_signal) |
| パターン | ABAB 10ペア (ウォームアップ 1 回は破棄) |

### 交絡チェック

| チェック項目 | 結果 |
|-------------|------|
| 単一変数の分離 | **不完全**: `GGML_RDMA_NO_SELECTIVE_SIGNAL` が Send と RDMA Write の両方に影響 |
| 環境変数トグル | ✅ 同一バイナリ、クライアント側 `static const` |
| ホットパスのログ出力 | ✅ なし |
| 分布の単峰性 | ✅ (PP Pair 1 の外れ値を除く) |

**注意**: `GGML_RDMA_NO_SELECTIVE_SIGNAL` は以下の両方に影響する:
1. **`send()`** (rdma-transport.cpp): Send WR のシグナリング → skip_last_signal/double-buffer の効果
2. **`set_tensor`** (ggml-rdma.cpp L1004): RDMA Write WR のシグナリング → staging buffer の同期

A/B 比較は Send signaling と RDMA Write signaling の複合効果を測定している。

### 生データ

| Pair | PP A (t/s) | PP B (t/s) | PP diff | TG A (t/s) | TG B (t/s) | TG diff |
|:----:|:----------:|:----------:|:-------:|:----------:|:----------:|:-------:|
| 1 | 116.9 | **111.4** | -5.5 | 34.7 | 33.3 | -1.4 |
| 2 | 117.3 | 117.1 | -0.2 | 34.7 | 33.3 | -1.4 |
| 3 | 117.2 | 116.4 | -0.8 | 35.0 | 33.3 | -1.7 |
| 4 | 117.6 | 118.2 | +0.6 | 35.0 | 33.6 | -1.4 |
| 5 | 117.1 | 116.3 | -0.8 | 35.0 | 33.2 | -1.8 |
| 6 | 117.2 | 117.9 | +0.7 | 35.1 | 33.5 | -1.6 |
| 7 | 117.6 | 116.8 | -0.8 | 35.1 | 33.6 | -1.5 |
| 8 | 117.7 | 117.8 | +0.1 | 35.2 | 33.6 | -1.6 |
| 9 | 116.6 | 116.9 | +0.3 | 35.0 | 33.6 | -1.4 |
| 10 | 116.9 | 117.2 | +0.3 | 34.6 | 33.6 | -1.0 |

**Pair 1 PP B = 111.4** は cold start 外れ値と推定 (ウォームアップ後でもモデルロードの影響が残存)。

### 検定結果

#### PP (Prompt Processing) — 全10ペア

| 指標 | 値 |
|------|:---:|
| Mean A | 117.21 ± 0.35 t/s |
| Mean B | 116.60 ± 1.83 t/s |
| 差分平均 | -0.61 t/s (-0.52%) |
| t(9) | -1.065 |
| p 値 | 0.315 |
| Cohen's d | -0.34 (small) |
| 95% CI | [-1.91, +0.69] t/s |
| 全ペア B > A | 5/10 (50%) |

#### PP — Pair 1 外れ値除外 (9ペア)

| 指標 | 値 |
|------|:---:|
| Mean A | 117.24 ± 0.36 t/s |
| Mean B | 117.18 ± 0.63 t/s |
| 差分平均 | -0.07 t/s (-0.06%) |
| t(8) | -0.329 |
| p 値 | 0.751 |
| Cohen's d | -0.11 (negligible) |
| 95% CI | [-0.53, +0.40] t/s |
| 全ペア B > A | 5/9 (56%) |

**判定**: PP に有意差なし。selective signaling は pp 性能にニュートラル。

#### TG (Token Generation) — 全10ペア

| 指標 | 値 |
|------|:---:|
| Mean A | 34.94 ± 0.20 t/s |
| Mean B | 33.46 ± 0.16 t/s |
| 差分平均 | **-1.48 t/s (-4.24%)** |
| t(9) | -21.264 |
| p 値 | < 0.000001 |
| Cohen's d | -6.72 (extremely large) |
| 95% CI | [-1.64, -1.32] t/s |
| 全ペア B > A | **0/10 (0%)** |

**判定**: TG に高度に有意なレグレッション (-4.24%)。全10ペアで一貫して B < A。

## 分析

### PP が改善しない理由

| 要因 | 説明 |
|------|------|
| 短いプロンプト | ~50 tokens → graph_compute のイテレーション数が少なく、wait 省略効果が小さい |
| MoE モデル | active パラメータ 3B → 計算が非常に高速 (117 t/s) → RDMA 通信比率が小さい |
| 6 GPU 構成 | RDMA 4 デバイスのみ → iteration あたり 4 回の wait 省略 → 効果 < 1μs/iteration |

GLM-4.7 IQ2_M 11GPU では pp128 +26.1% の大幅改善が確認されている (MEMORY.md)。効果はモデルサイズ、GPU 数、プロンプト長に依存。

### TG レグレッションの原因: 分離テスト

初回テストでは `GGML_RDMA_NO_SELECTIVE_SIGNAL` が Send signaling と RDMA Write signaling の両方を制御していたため、効果を分離できなかった。追加の分離テスト (`27a48e9ae`) で set_tensor の RDMA Write signaling から `RDMA_NO_SELECTIVE_SIGNAL` を除去し、Send signaling のみの効果を測定した。

#### 分離テスト結果 (5 ABAB pairs)

| Pair | TG A (t/s) | TG B (t/s) | TG diff |
|:----:|:----------:|:----------:|:-------:|
| 1 | 33.6 | 32.7 | -0.9 |
| 2 | 33.7 | 33.3 | -0.4 |
| 3 | 34.0 | 33.0 | -1.0 |
| 4 | 33.7 | 33.4 | -0.3 |
| 5 | 34.2 | 33.1 | -1.1 |

**比較**:

| テスト | A_tg | B_tg | TG 差分 | 制御変数 |
|-------|:----:|:----:|:------:|---------|
| 初回 (複合) | 34.94 | 33.46 | **-4.24%** | Send + RDMA Write |
| 分離 (Send のみ) | 33.84 | 33.10 | **-2.19%** | Send のみ |

**分析**:

1. **Send signaling 寄与**: -2.19% — Send selective signaling + skip_last_signal による tg レグレッション
2. **RDMA Write signaling 寄与**: ~-2.0% — A_tg が 34.94→33.84 に低下 (RDMA Write all-signaling の除去)
3. **両者がほぼ等しく寄与**: tg レグレッションは単一の原因ではなく、Send と RDMA Write の複合効果

**注意**: tg 分析のチャンク・サイズ考察:
- tg 時の activation tensor は小さい (< 16MB) → set_tensor は常に単一チャンク (is_last=true) → signaling は A/B で同一のはず
- しかし実測では RDMA Write signaling の差異が tg に影響 → set_tensor 以外の RDMA Write パスまたはサーバー側の double-buffered send の挙動変更が関与する可能性
- 根本原因の解明には `GGML_RDMA_PROFILE=1` やハードウェアカウンタによる詳細プロファイリングが必要

### モデル依存性

| モデル/構成 | PP 効果 | TG 効果 |
|-----------|:-------:|:-------:|
| gpt-oss-20b (per-device, llama-bench) | -0.08% | -2.36% |
| Qwen3.5-35B-A3B (6GPU, llama-cli) | -0.06% | -4.24% |
| GLM-4.7 IQ2_M (11GPU, 過去データ) | **+26.1%** | +0.4% (非有意) |

GLM-4.7 では selective signaling の pp 改善が非常に大きく (+26.1%)、tg 影響は非有意。Qwen3.5 での tg レグレッションが GLM-4.7 でも再現するかは未検証。

## 結論

1. **Double-buffering は安全に動作**: Qwen3.5-35B-A3B 6GPU で正常な推論出力、PP 性能劣化なし
2. **PP 効果はモデル依存**: このモデル/構成では pp 効果がニュートラル (GLM-4.7 11GPU では +26.1%)
3. **TG レグレッションは Send + RDMA Write の複合効果**: 各々 ~2% ずつ寄与 (Qwen3.5 6GPU)
4. **RDMA Write signaling を decoupled** (`27a48e9ae`): `GGML_RDMA_NO_SELECTIVE_SIGNAL` は Send path のみに影響するよう変更
5. **GLM-4.7 検証が必要**: 過去データでは tg 影響なし — double-buffering 全体の効果は GLM-4.7 11GPU での実測が決定的

### 推奨次ステップ

1. **GLM-4.7 11GPU 検証** (最優先): pp128 が double-buffering で ~24.3 → ~30.6 に回復するか確認し、tg レグレッションの有無を検証
2. **TG レグレッション根本原因調査**: `GGML_RDMA_PROFILE=1` でタイミング内訳を分析
3. **サーバー側 double-buffered send の影響評価**: サーバーの rdma-transport.cpp も double-buffering コードを使用しており、レスポンス送信パターンが変化した可能性

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `27a48e9ae (feature/double-buffering)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 34°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| テスト GPU | CUDA5-6 (Node 1) + RDMA0-3 (Node 2) | |
| 同時実行 | llama-server (CUDA0-4, 別プロセス) | |
