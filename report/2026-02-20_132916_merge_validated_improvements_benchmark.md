# 検証済み改善ワークツリーのマージとベンチマーク

- **実施日時**: 2026年2月20日 13:29
- **ワークツリー**: `.worktree/merge-validated` (ブランチ: `merge/validated-improvements`)

## 前提・目的

未マージの改善ワークツリーのうち、過去のベンチマークで統計的に有意な改善が確認された2つの機能を `feature/rdma-backend` にまとめ、GLM-4.7 IQ2_M 11GPU 構成で再検証する。

### マージ対象

| 機能 | コミット | 過去のベンチマーク結果 |
|------|---------|---------------------|
| Server-Push (RDMA Write with IMM) | `bd2af16dc` | +0.74% tg (p=1.28e-10, d=4.30) ※per-device conn 有効時のみ |
| Selective Signaling | `789b88ace` | +0.89% tg (p=0.0009, d=1.08) |

### 参考レポート

- 過去のベンチマーク結果は各ワークツリー (`rdma-server-push`, `rdma-selective-signaling`) のコミットメッセージに記録

## 再現方法

### 1. ワークツリー作成と Cherry-pick

```bash
git worktree add -b merge/validated-improvements .worktree/merge-validated feature/rdma-backend
cd .worktree/merge-validated
git cherry-pick bd2af16dc  # server-push
git cherry-pick 789b88ace  # selective-signaling
```

両方ともコンフリクトなしでクリーンに適用。

### 2. ビルド・デプロイ

```bash
bash scripts/rdma-build.sh local
gpu-lock.sh run bash scripts/rdma-deploy.sh
bash scripts/rdma-server.sh start
```

### 3. Selective Signaling A/B テスト

同一バイナリ (merge-validated) で環境変数トグル:
- **A**: `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` (OFF)
- **B**: デフォルト (ON)

サーバーも各条件で再起動 (rdma-transport.cpp の `static const` が起動時に決定されるため)。

```bash
# A (OFF)
GGML_RDMA_NO_SELECTIVE_SIGNAL=1 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  GGML_RDMA_GDR_BUDGET_GB=12 llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0/CUDA1/.../RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -p 128 -n 32 -r 1

# B (ON)
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_GDR_BUDGET_GB=12 llama-bench \
  (同上)
```

### 4. Server-Push + Per-Device Connections A/B テスト

- **A**: デフォルト (shared connection, selective-signaling ON)
- **B**: `GGML_RDMA_PER_DEVICE_CONN=1` (per-device + server-push + selective-signaling ON)

### 5. 総合改善 A/B テスト

全機能 OFF vs 全機能 ON の直接比較:
- **A**: `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` (selective-signaling OFF, shared connection)
- **B**: `GGML_RDMA_PER_DEVICE_CONN=1` (selective-signaling ON, per-device + server-push)

各条件でサーバーを再起動してから計測。

```bash
# A (ベースライン: selective-signaling OFF, shared connection)
GGML_RDMA_NO_SELECTIVE_SIGNAL=1 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  GGML_RDMA_GDR_BUDGET_GB=12 llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0/CUDA1/.../RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -p 128 -n 32 -r 1

# B (全有効: selective-signaling ON, per-device + server-push)
GGML_RDMA_PER_DEVICE_CONN=1 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  GGML_RDMA_GDR_BUDGET_GB=12 llama-bench \
  (同上)
```

## 交絡因子チェック

- [x] **単一変数の分離**: 同一バイナリ + 環境変数トグルで条件切替
- [x] **ホットパスのログ出力**: ホットパスに fprintf なし (fix/bimodal-fprintf で修正済み)
- [x] **分布の単峰性**: ウォームアップ1回で初期化効果を排除、計測値に > 0.5 t/s のギャップなし
- [x] **ABAB 交互実行**: 全ペアで A→B 順に交互実行、各条件でサーバー再起動

## 結果

### テスト1: Selective Signaling (デフォルト設定)

**A = OFF, B = ON**

#### 生データ

| Pair | A pp128 | B pp128 | A tg32 | B tg32 |
|:----:|--------:|--------:|-------:|-------:|
| 1 | 23.370 | 29.367 | 7.784 | 7.782 |
| 2 | 23.428 | 29.456 | 7.779 | 7.861 |
| 3 | 23.358 | 29.730 | 7.687 | 7.572 |
| 4 | 23.359 | 29.269 | 7.671 | 7.861 |
| 5 | 23.228 | 29.400 | 7.781 | 7.781 |

#### 記述統計

| 指標 | A (OFF) | B (ON) |
|------|--------:|-------:|
| pp128 平均 ± SD | 23.349 ± 0.073 | 29.445 ± 0.165 |
| tg32 平均 ± SD | 7.740 ± 0.053 | 7.771 ± 0.113 |

#### 検定結果: pp128

| 指標 | 値 |
|------|:---:|
| 差分平均 | **+6.096 t/s (+26.11%)** |
| t(4) | 75.29 |
| p 値 | 1.87 × 10⁻⁷ |
| Cohen's d | 33.67 (極めて大きい) |
| 95% CI | [+5.871, +6.321] t/s |
| 全ペア正の効果 | 5/5 (100%) |

**判定: 有意かつ実質的** — p < 0.05、効果 +26.11% > 0.5%

#### 検定結果: tg32

| 指標 | 値 |
|------|:---:|
| 差分平均 | +0.031 t/s (+0.40%) |
| t(4) | 0.606 |
| p 値 | 0.577 |
| Cohen's d | 0.27 (小さい) |
| 95% CI | [-0.110, +0.171] t/s |
| 正の効果ペア | 2/5 (40%) |

**判定: 効果なし** — p = 0.577 ≥ 0.05

### テスト2: Per-Device Connections + Server-Push

**A = Shared connection (default), B = Per-device + Server-push (PER_DEVICE_CONN=1)**

#### 生データ

| Pair | A pp128 | B pp128 | A tg32 | B tg32 |
|:----:|--------:|--------:|-------:|-------:|
| 1 | 29.636 | 29.466 | 7.696 | 7.719 |
| 2 | 29.557 | 29.055 | 7.726 | 7.830 |
| 3 | 29.701 | 29.382 | 7.625 | 7.817 |
| 4 | 29.772 | 29.212 | 7.607 | 7.834 |
| 5 | 29.799 | 29.350 | 7.641 | 7.766 |

#### 記述統計

| 指標 | A (shared) | B (per-device+push) |
|------|----------:|-----------:|
| pp128 平均 ± SD | 29.693 ± 0.097 | 29.293 ± 0.149 |
| tg32 平均 ± SD | 7.659 ± 0.050 | 7.793 ± 0.048 |

#### 検定結果: pp128

| 指標 | 値 |
|------|:---:|
| 差分平均 | **-0.400 t/s (-1.35%)** |
| t(4) | -5.72 |
| p 値 | 0.0046 |
| Cohen's d | -2.56 (大きい) |
| 95% CI | [-0.594, -0.206] t/s |
| 退行ペア | 5/5 (100%) |

**判定: 有意な退行** — p < 0.05、退行 -1.35% > 0.5%

#### 検定結果: tg32

| 指標 | 値 |
|------|:---:|
| 差分平均 | **+0.134 t/s (+1.76%)** |
| t(4) | 3.79 |
| p 値 | 0.019 |
| Cohen's d | +1.69 (大きい) |
| 95% CI | [+0.036, +0.232] t/s |
| 全ペア正の効果 | 5/5 (100%) |

**判定: 有意かつ実質的** — p < 0.05、効果 +1.76% > 0.5%

### テスト3: 総合改善 (ベースライン vs 全機能有効)

テスト1・テスト2 で個別に評価した機能を、両方 OFF vs 両方 ON で直接比較した。

**A = ベースライン (selective-signaling OFF, shared connection)**
**B = 全有効 (selective-signaling ON, per-device + server-push)**

#### 生データ

| Pair | A pp128 | B pp128 | A tg32 | B tg32 |
|:----:|--------:|--------:|-------:|-------:|
| 1 | 23.417 | 29.183 | 7.672 | 7.820 |
| 2 | 23.258 | 29.221 | 7.639 | 7.822 |
| 3 | 23.310 | 29.158 | 7.773 | 7.831 |
| 4 | 23.218 | 29.254 | 7.695 | 7.841 |
| 5 | 23.311 | 29.220 | 7.735 | 7.839 |

#### 記述統計

| 指標 | A (ベースライン) | B (全有効) |
|------|----------:|-----------:|
| pp128 平均 ± SD | 23.303 ± 0.074 | 29.207 ± 0.037 |
| tg32 平均 ± SD | 7.703 ± 0.053 | 7.831 ± 0.009 |

#### 検定結果: pp128

| 指標 | 値 |
|------|:---:|
| 差分平均 | **+5.905 t/s (+25.34%)** |
| t(4) | 127.60 |
| p 値 | 2.26 × 10⁻⁸ |
| Cohen's d | 57.06 (極めて大きい) |
| 95% CI | [+5.776, +6.033] t/s |
| 全ペア正の効果 | 5/5 (100%) |

**判定: 有意かつ実質的** — p < 0.05、効果 +25.34% > 0.5%

#### 検定結果: tg32

| 指標 | 値 |
|------|:---:|
| 差分平均 | **+0.128 t/s (+1.66%)** |
| t(4) | 5.97 |
| p 値 | 0.0040 |
| Cohen's d | 2.67 (大きい) |
| 95% CI | [+0.068, +0.187] t/s |
| 全ペア正の効果 | 5/5 (100%) |

**判定: 有意かつ実質的** — p < 0.05、効果 +1.66% > 0.5%

## 分析

### Selective Signaling の効果

Selective signaling は **pp128 で +26.11% という劇的な改善** を示した。これは過去の gpt-oss-20b ベンチマーク (+0.89% tg) から予想されたよりも遥かに大きい。

**理由**: GLM-4.7 (355B MoE) は pp128 で大量のテンソルデータ (数百回の RDMA Write) をリモート GPU に転送する。各 Write を個別にシグナリングする場合、CQE (Completion Queue Event) のポーリングオーバーヘッドが累積する。Selective signaling は N 回に 1 回のみシグナルを送り、残りは unsignaled で送信するため、このオーバーヘッドを大幅に削減する。

pp128 への影響が tg32 より大きいのは:
- pp128: 128 トークン分のテンソルを一括転送 → 多数の RDMA Write が発生 → シグナリング削減の恩恵大
- tg32: 1 トークンずつ逐次処理 → 各ステップの RDMA Write 回数が少ない → シグナリング削減の恩恵小

### Per-Device Connections + Server-Push の効果

Per-device connections + Server-push は **pp128 で -1.35% の退行、tg32 で +1.76% の改善** というトレードオフを示した。

- **pp128 退行の原因**: Per-device connections は各リモート GPU ごとに独立した RDMA 接続を確立するため、接続数が 1→4 に増加。各接続の MTT (Memory Translation Table) エントリが個別に必要となり、ConnectX-4 の MTT キャッシュ圧迫が増大する。大量データ転送時にこのオーバーヘッドが顕在化。
- **tg32 改善の原因**: Server-push により、サーバーが計算結果を RDMA Write with IMM でクライアントに直接書き込む。従来のクライアント側 RDMA Read (get_tensor) のポーリングオーバーヘッドが排除され、tg のコマンドラウンドトリップが短縮。

### 総合改善と交互作用

テスト3 (全機能 OFF vs 全機能 ON) の結果と、テスト1・テスト2 の個別効果を加算した予測値を比較:

| 指標 | テスト1 | テスト2 | 加算予測 | テスト3 実測 | 乖離 |
|------|:------:|:------:|:-------:|:----------:|:----:|
| pp128 | +26.11% | -1.35% | +24.76% | +25.34% | +0.58% |
| tg32 | +0.40% (n.s.) | +1.76% | +2.16% | +1.66% | -0.50% |

両指標とも乖離が 2% 未満のため、**有意な交互作用はない**と判断。各機能の効果は独立して加算的に作用している。

feature/rdma-backend (マージ前ベースライン) からの総合改善:

| 指標 | ベースライン (テスト3 A) | 全有効 (テスト3 B) | 改善率 |
|------|:---:|:---:|:---:|
| pp128 | 23.30 t/s | 29.21 t/s | **+25.3%** |
| tg32 | 7.70 t/s | 7.83 t/s | **+1.7%** |

### 推奨構成

| 構成 | pp128 (t/s) | tg32 (t/s) | 用途 |
|------|:-----------:|:----------:|------|
| ベースライン (両方 OFF) | 23.30 | 7.70 | — |
| **Selective signaling のみ (デフォルト)** | **29.44** | **7.77** | **汎用 (推奨)** |
| 全有効 (+ per-device + server-push) | 29.21 (-0.8%) | 7.83 (+0.8%) | tg 重視時のみ |

- デフォルト設定 (selective-signaling ON, shared connection) が最もバランスの良い構成
- tg を最重視する場合のみ `GGML_RDMA_PER_DEVICE_CONN=1` を検討 (pp は 0.8% 犠牲)
- 全有効時の pp128 がデフォルトより低いのは per-device connections の MTT キャッシュオーバーヘッドによる退行 (-1.35%) が selective-signaling の恩恵を一部相殺するため

## 結論

1. **全機能有効でベースラインから pp128 +25.3%, tg32 +1.7% の総合改善** — 両指標で統計的に有意 (p < 0.005)、全5ペアで正の効果
2. **交互作用なし** — 加算予測との乖離は pp128 +0.58%、tg32 -0.50% と小さく、各機能の効果は独立して加算的
3. **Selective signaling は feature/rdma-backend にマージすべき** — pp128 で +26% の大幅改善、tg32 への悪影響なし
4. **Server-push は per-device connections とセットでオプション機能として維持** — tg +1.76% vs pp -1.35% のトレードオフ。デフォルト無効 (`GGML_RDMA_PER_DEVICE_CONN=1` で有効化) が適切
5. **推奨構成**: selective-signaling ON (デフォルト) + shared connection (デフォルト) で pp128=29.2 t/s, tg32=7.77 t/s。tg 最重視時のみ per-device を有効化で tg32=7.83 t/s (+0.8%, pp128 は同等)
6. **Cherry-pick はコンフリクトなし** — 両機能は独立しており、merge-validated ブランチで安定動作を確認

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `088f8c8d9 (merge/validated-improvements)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 35°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running |
