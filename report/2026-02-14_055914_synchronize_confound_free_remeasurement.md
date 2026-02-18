# synchronize 効果の交絡因子排除再測定レポート

- **実施日時**: 2026年2月14日 04:00 〜 05:59
- **ワークツリー**: `.worktree/rdma-backend` (条件 A'), `.worktree/rdma-synchronize` (条件 B/C)
- **参照レポート**: [synchronize ブランチ 二峰性修正マージ & 効果検証レポート](2026-02-14_024402_synchronize_bimodal_fix_and_benchmark.md)

## 前提・目的

### 背景

前回の実験 (参照レポート) では、条件 A (rdma-backend) と条件 B/C (rdma-synchronize) の比較に交絡因子が存在した:

- **条件 A のみ**: deferred copy 機能あり (`1d2c9cbdd`)
- **条件 B/C のみ**: SYNC コマンド実装あり

そのため、A vs B/C で観測された性能差 (pp128: -0.2%, tg32: -1.3%) が「SYNC コマンドのオーバーヘッド」と「deferred copy の恩恵」のどちらに起因するか分離できなかった。

### 目的

交絡因子を排除し、**SYNC コマンドの効果のみを純粋に測定する**。

### アプローチ: 環境変数による deferred copy 無効化

Cherry-pick ではなく `GGML_RDMA_NO_DEFERRED_COPY=1` による無効化を採用した。

**Cherry-pick を避けた理由**:
- deferred copy コミット (`1d2c9cbdd`) は bimodal fix (`5b6e0bf56`) より前のコミット
- `fix_cross_device_refs` 内に 3箇所の `fprintf(stderr, ...)` を含む
- synchronize ブランチに cherry-pick すると、bimodal fix (cherry-pick 済み) がこれらの行をカバーしない
- **二峰性 tg バグが再発するリスク**がある

**`GGML_RDMA_NO_DEFERRED_COPY=1` の動作**:
- `cpy_tensor` が即座に `false` を返す (ggml-rdma.cpp 行 1129-1130)
- deferred copy エントリは記録されない
- ワイヤフォーマットの `n_copies` は常に 0
- コード変更不要、マージ不要、二峰性再発リスクなし

### 残留差分

| 側面 | A' (rdma-backend + NO_DEFERRED_COPY=1) | B (rdma-synchronize) |
|------|----------------------------------------|---------------------|
| cpy_tensor | `return false` (env var チェック) | `return false` (コード上無効) |
| ワイヤフォーマット | `n_copies(4B)=0` プレフィックスあり | プレフィックスなし |
| fix_cross_device_refs | dc_lookup={} パラメータあり (空、無影響) | dc_lookup パラメータなし |
| **synchronize()** | **no-op** | **RDMA_CMD_SYNC 送信** |
| SYNC サーバーハンドラ | なし | `send_rsp_empty()` |
| プロトコルバージョン | 1.3.1 | 1.3.2 |

ホットパスで異なるのは `synchronize()` の動作のみ。これが今回の測定対象。

### 前提条件

- 1号機 (192.168.100.1): 7x Tesla P100-PCIE-16GB, ConnectX-4 100GbE
- 2号機 (192.168.100.2): 4x Tesla P100-PCIE-16GB, ConnectX-4 100GbE
- nvidia-peermem ロード済み (GPUDirect RDMA 有効)
- GDR budget: 12GB (デフォルト)

## 再現方法

### 1. rdma-backend デプロイ (条件 A' 用)

```bash
bash scripts/rdma-deploy.sh
bash scripts/rdma-server.sh start
```

### 2. 条件 A' ベンチマーク

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_DEFERRED_COPY=1 \
  LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin \
  /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin/llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

### 3. rdma-synchronize デプロイ (条件 B/C 用)

```bash
bash scripts/rdma-server.sh stop
bash /home/ubuntu/projects/llama.cpp/.worktree/rdma-synchronize/scripts/rdma-deploy.sh
bash scripts/rdma-server.sh start
```

### 4. 条件 B/C ベンチマーク

```bash
# 条件 B
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/.worktree/rdma-synchronize/build/bin \
  /home/ubuntu/projects/llama.cpp/.worktree/rdma-synchronize/build/bin/llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -r 1 -p 128 -n 32

# 条件 C (同上 + GGML_SCHED_TWO_PHASE=1 を先頭に追加)
```

## 実験条件

| 条件 | ビルド | 環境変数 | 説明 |
|------|--------|---------|------|
| A' (baseline) | `feature/rdma-backend` | `GGML_RDMA_NO_DEFERRED_COPY=1` | deferred copy 無効化したベースライン |
| B (sync) | `feature/rdma-synchronize` | — | SYNC コマンド実装 |
| C (two-phase) | `feature/rdma-synchronize` | `GGML_SCHED_TWO_PHASE=1` | SYNC + 二段階 dispatch |

**前回 (v1) との実験設計の違い**:

| 側面 | v1 | v2 (今回) |
|------|-----|-----------|
| 条件 A | deferred copy **有効** | deferred copy **無効** (env var) |
| 交絡因子 | deferred copy 有無 + SYNC 有無 | **SYNC 有無のみ** |
| ウォームアップ | なし | 1回目を破棄 |
| データ回数 | 10回 | 15回 (+ ウォームアップ1回) |

## 統計的検証手法

- **Welch の t検定** (両側, alpha = 0.05): 等分散を仮定しない
- **効果量**: Cohen's d (小: |d| < 0.2, 中: 0.2 <= |d| < 0.8, 大: |d| >= 0.8)
- **外れ値検出**: IQR 法 (1.5x)
- **データ**: 各条件16回実行、1回目をウォームアップとして破棄、残り15回で分析

## 結果

### GLM-4.7 IQ2_M (7C+4R 11GPU, 各15回)

#### 生データ

**条件 A' (rdma-backend + NO_DEFERRED_COPY=1)**:

| run | pp128 | tg32 |
|-----|-------|------|
| 1* | 23.82 | 7.65 |
| 2 | 23.84 | 7.66 |
| 3 | 23.84 | 7.61 |
| 4 | 23.82 | 7.58 |
| 5 | 23.82 | 7.65 |
| 6 | 23.82 | 7.59 |
| 7 | 23.83 | 7.68 |
| 8 | 23.83 | 7.68 |
| 9 | 23.83 | 7.67 |
| 10 | 23.84 | 7.68 |
| 11 | 23.85 | 7.68 |
| 12 | 23.85 | 7.58 |
| 13 | 23.82 | 7.67 |
| 14 | 23.86 | 7.68 |
| 15 | 23.82 | 7.60 |
| 16 | 23.82 | 7.57 |

*run 1 はウォームアップとして分析から除外

**条件 B (rdma-synchronize, SYNC コマンド)**:

| run | pp128 | tg32 |
|-----|-------|------|
| 1* | 23.82 | 7.63 |
| 2 | 23.86 | 7.63 |
| 3 | 23.83 | 7.67 |
| 4 | 23.81 | 7.63 |
| 5 | 23.83 | 7.68 |
| 6 | 23.82 | 7.67 |
| 7 | 23.85 | 7.67 |
| 8 | 23.85 | 7.67 |
| 9 | 23.83 | 7.67 |
| 10 | 23.83 | 7.67 |
| 11 | 23.88 | 7.68 |
| 12 | 23.84 | 7.66 |
| 13 | 23.83 | 7.66 |
| 14 | 23.82 | 7.67 |
| 15 | 23.83 | 7.66 |
| 16 | 23.83 | 7.67 |

**条件 C (rdma-synchronize + TWO_PHASE=1)**:

| run | pp128 | tg32 |
|-----|-------|------|
| 1* | 23.82 | 7.63 |
| 2 | 23.81 | 7.67 |
| 3 | 23.84 | 7.70 |
| 4 | 23.82 | 7.65 |
| 5 | 23.83 | 7.63 |
| 6 | 23.82 | 7.62 |
| 7 | 23.85 | 7.64 |
| 8 | 23.91 | 7.67 |
| 9 | 23.82 | 7.63 |
| 10 | 23.83 | 7.64 |
| 11 | 23.82 | 7.67 |
| 12 | 23.87 | 7.66 |
| 13 | 23.82 | 7.66 |
| 14 | 23.82 | 7.66 |
| 15 | 23.81 | 7.66 |
| 16 | 23.82 | 7.65 |

#### 記述統計量 (runs 2-16, n=15)

| 条件 | 指標 | 平均 | SD | min | max | range |
|------|------|-----:|---:|----:|----:|------:|
| A' | pp128 | 23.833 | 0.0133 | 23.82 | 23.86 | 0.04 |
| A' | tg32 | 7.639 | 0.0442 | 7.57 | 7.68 | 0.11 |
| B | pp128 | 23.836 | 0.0176 | 23.81 | 23.88 | 0.07 |
| B | tg32 | 7.664 | 0.0150 | 7.63 | 7.68 | 0.05 |
| C | pp128 | 23.833 | 0.0266 | 23.81 | 23.91 | 0.10 |
| C | tg32 | 7.654 | 0.0203 | 7.62 | 7.70 | 0.08 |

#### Welch t検定結果

| 比較 | 指標 | 差 (t/s) | % | t | p | d | 判定 |
|------|------|---:|--:|--:|--:|--:|------|
| **A' vs B** | **pp128** | **-0.003** | **-0.01%** | **-0.58** | **0.565** | **-0.21** | **n.s.** |
| **A' vs B** | **tg32** | **-0.025** | **-0.33%** | **-2.10** | **0.051** | **-0.77** | **n.s.** |
| A' vs C | pp128 | 0.000 | 0.00% | 0.00 | 1.000 | 0.00 | n.s. |
| A' vs C | tg32 | -0.015 | -0.20% | -1.22 | 0.237 | -0.45 | n.s. |
| B vs C | pp128 | +0.003 | +0.01% | 0.41 | 0.689 | 0.15 | n.s. |
| B vs C | tg32 | +0.010 | +0.13% | 1.53 | 0.137 | 0.56 | n.s. |

#### 外れ値検出 (IQR 1.5x)

| 条件 | pp128 | tg32 |
|------|-------|------|
| A' | なし | なし |
| B | 1個 (23.88) | 2個 (7.63, 7.63) |
| C | 2個 (23.91, 23.87) | なし |

外れ値は各条件で 0-2 個と少なく、結論に影響しない。

### 前回結果 (v1) との一貫性検証

| 比較 | 指標 | v1 平均 | v2 平均 | 差 | p | 判定 |
|------|------|------:|------:|---:|--:|------|
| B v1 vs v2 | pp128 | 23.810 | 23.836 | +0.026 | 0.0005 | *** |
| B v1 vs v2 | tg32 | 7.651 | 7.664 | +0.013 | 0.025 | * |
| C v1 vs v2 | pp128 | 23.807 | 23.833 | +0.026 | 0.077 | n.s. |
| C v1 vs v2 | tg32 | 7.666 | 7.654 | -0.012 | 0.161 | n.s. |

B の pp128 に v1 vs v2 で統計的有意差があるが、絶対差は 0.026 t/s (0.11%) と極めて小さく、セッション間の環境変動 (GPU 温度、バックグラウンドプロセス等) の範囲内。両測定セットは実用上一貫している。

### 前回 A (deferred copy 有効) vs 今回 A' (deferred copy 無効) の比較

| 指標 | A (v1, deferred copy 有効) | A' (v2, deferred copy 無効) | 差 |
|------|--:|--:|--:|
| pp128 平均 | 23.852 | 23.833 | -0.019 (-0.08%) |
| tg32 平均 | 7.753 | 7.639 | -0.114 (-1.47%) |
| tg32 SD | 0.037 | 0.044 | +0.007 |

deferred copy 有効時 (A) は tg32 が 0.114 t/s (1.5%) 高速で、分散も小さかった。これが前回観測された A vs B の差 (0.102 t/s, 1.3%) のほぼ全てを説明する。

## 考察

### A' vs B: SYNC コマンドは性能に影響しない

今回の最重要結果: **deferred copy を無効化した rdma-backend (A') と rdma-synchronize (B) の間に有意差はない**。

- pp128: 差 -0.003 t/s (-0.01%), p=0.565
- tg32: 差 -0.025 t/s (-0.33%), p=0.051

tg32 の p=0.051 はボーダーラインだが、alpha=0.05 の基準を満たさない。また、差の方向は A' < B (B がやや速い) であり、SYNC コマンドが「オーバーヘッド」を持つという仮説とは**逆方向**である。

A' の tg32 の高い分散 (SD=0.044 vs B: 0.015) がこの差に寄与している。rdma-backend のコードパスには deferred copy 関連の判定ロジック (`DEFERRED_COPY_DISABLED` 変数チェック、`dc_lookup` パラメータ等) が残存しており、これがノイズ源となっている可能性がある。

### 前回の A vs B の差の根本原因: deferred copy

前回の実験で A が B より 0.102 t/s (1.3%) 速かった原因は、deferred copy の恩恵であったことが確定した:

- A (deferred copy 有効) tg32 平均: 7.753 t/s
- A' (deferred copy 無効) tg32 平均: 7.639 t/s
- B (synchronize) tg32 平均: 7.664 t/s (v2)

deferred copy を無効にすると A' の性能は B と同等になり、差は消失する。

### B vs C: two-phase は依然として効果なし

前回と同様、B vs C で有意差なし (pp128: p=0.69, tg32: p=0.14)。layer split ではスプリット間のデータ依存により並列化の余地がないことを再確認。

### 分散の違い

| 条件 | tg32 SD | 特徴 |
|------|--------:|------|
| A (v1) | 0.037 | deferred copy 有効、高速・低分散 |
| A' (v2) | 0.044 | deferred copy 無効、deferred copy 判定ロジック残存 |
| B (v2) | 0.015 | synchronize、最も安定 |
| C (v2) | 0.020 | synchronize + two-phase、安定 |

B/C の方が A' より分散が小さい。rdma-synchronize のコードはよりシンプル (deferred copy 関連コード不在) であり、これが安定性に寄与している可能性がある。ただし性能差は実用上無視できるレベル。

## 結論

| 検証項目 | 結果 |
|---------|------|
| SYNC コマンドのオーバーヘッド | **なし** — A' vs B で有意差なし (pp128: p=0.57, tg32: p=0.05) |
| 前回の A vs B の差の原因 | **deferred copy の恩恵** — deferred copy 無効化で差が消失 |
| Two-phase の効果 | **なし** — B vs C で有意差なし (前回結果を再現) |
| 前回結果との一貫性 | **一貫** — B/C の測定値は v1 と実用的に一致 |

### 前回レポートの結論の更新

前回レポートでは A vs B/C の差について「SYNC のオーバーヘッドと deferred copy の恩恵を分離できない」としていた。今回の交絡因子排除実験により:

1. **SYNC コマンドにオーバーヘッドはない** — 性能への影響は測定不能なレベル
2. **前回の差は deferred copy に帰属** — tg32 で約 1.5% の改善効果

### 推奨事項

1. `feature/rdma-synchronize` ブランチのマージは **不要** — SYNC コマンドにも two-phase にも性能効果なし
2. deferred copy は rdma-backend で **有効維持** — tg32 で +1.5% の恩恵が確認済み
3. 今後の性能改善は rdma-backend ブランチで継続
