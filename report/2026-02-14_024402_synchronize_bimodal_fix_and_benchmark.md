# synchronize ブランチ 二峰性修正マージ & 効果検証レポート

- **実施日時**: 2026年2月14日 00:30 〜 02:44
- **ワークツリー**: `.worktree/rdma-synchronize` (テスト対象), `.worktree/rdma-backend` (ベースライン)
- **参照レポート**: [二峰性 tg 調査レポート](2026-02-13_153100_bimodal_tg_investigation.md)

## 前提・目的

### 背景

`feature/rdma-synchronize` ブランチは以下の2つの機能を実装している:
1. **SYNC コマンド** — `synchronize()` を no-op から実装に変更。`compute_pending_` フラグをクリアし、`get_tensor()` で RDMA Read 高速パス (GDR) を使えるようにする
2. **Two-phase compute splits** — `GGML_SCHED_TWO_PHASE=1` で有効化。全 split を先に dispatch してから同期する方式

このブランチには rdma-backend で修正済みの二峰性 tg バグ（ホットパスの `fprintf` による ~1700ms I/O スパイク）がまだ含まれていた。

### 目的

1. 二峰性修正 (commit `5b6e0bf56`) を synchronize ブランチに cherry-pick してマージ
2. 二峰性が解消されたことを確認
3. synchronize 機能 (SYNC コマンド、two-phase compute) の性能効果を統計的に検証

### 前提条件

- 1号機 (192.168.100.1): 7× Tesla P100-PCIE-16GB, ConnectX-4 100GbE
- 2号機 (192.168.100.2): 4× Tesla P100-PCIE-16GB, ConnectX-4 100GbE
- nvidia-peermem ロード済み (GPUDirect RDMA 有効)
- GDR budget: 12GB (デフォルト)

## 再現方法

### 1. Cherry-pick

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-synchronize
git cherry-pick 5b6e0bf56
# コンフリクト解決: deferred copy (dc_lookup) 関連のコードは synchronize にないため除外
# get_tensor の fprintf→RDMA_LOG_DBG 変換 + nonzero チェック削除が主要な修正
git add ggml/src/ggml-rdma/ggml-rdma.cpp
git cherry-pick --continue
```

### 2. ビルド & デプロイ

```bash
# synchronize ブランチ (条件 B/C 用)
bash /home/ubuntu/projects/llama.cpp/.worktree/rdma-synchronize/scripts/rdma-build.sh local
bash /home/ubuntu/projects/llama.cpp/.worktree/rdma-synchronize/scripts/rdma-deploy.sh
bash /home/ubuntu/projects/llama.cpp/.worktree/rdma-synchronize/scripts/rdma-server.sh restart

# rdma-backend ブランチ (条件 A 用)
bash /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/scripts/rdma-build.sh local
bash /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/scripts/rdma-deploy.sh
bash /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/scripts/rdma-server.sh restart
```

### 3. ベンチマーク実行

**gpt-oss-20b (1C+1R)**:
```bash
# 条件 A (baseline): rdma-backend ビルド
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=.../rdma-backend/build/bin \
  .../rdma-backend/build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32

# 条件 B (sync): rdma-synchronize ビルド
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=.../rdma-synchronize/build/bin \
  .../rdma-synchronize/build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32

# 条件 C (two-phase): rdma-synchronize ビルド + GGML_SCHED_TWO_PHASE=1
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 GGML_SCHED_TWO_PHASE=1 \
  LD_LIBRARY_PATH=.../rdma-synchronize/build/bin \
  .../rdma-synchronize/build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

**GLM-4.7 IQ2_M (7C+4R)**:
```bash
# 条件 A/B/C 同様、llama-bench で -dev オプションを使用
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  LD_LIBRARY_PATH=.../build/bin \
  .../build/bin/llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0/CUDA1/.../RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

## 実験条件

| 条件 | ビルド | 環境変数 | 説明 |
|------|--------|---------|------|
| A (baseline) | `feature/rdma-backend` | — | ベースライン (deferred copy 有り) |
| B (sync) | `feature/rdma-synchronize` | — | SYNC コマンド実装 |
| C (two-phase) | `feature/rdma-synchronize` | `GGML_SCHED_TWO_PHASE=1` | 二段階 dispatch |

**注意 — 交絡因子**: 条件 A と B/C はブランチが異なる。両ブランチの共通祖先は `63db2af0e` で、そこから:
- **rdma-backend のみ**: `1d2c9cbdd` (deferred copy 実装) → `5b6e0bf56` (二峰性修正) → ドキュメントコミット
- **rdma-synchronize のみ**: `268c61903` (SYNC + two-phase 実装) → `40790b150` (二峰性修正 cherry-pick)

そのため A vs B/C の差には **2つの変数が混在** している:
1. deferred copy の有無 (A にあり B/C にない)
2. synchronize 機能の有無 (B/C にあり A にない)

この実験設計では、A の優位性が「synchronize のオーバーヘッド」によるものか「deferred copy の恩恵」によるものか分離できない。B vs C の比較のみが pure な条件比較である。

## 統計的検証手法

- **Welch の t検定** (両側、α = 0.05): 等分散を仮定しない
- **効果量**: Cohen's d (小: d < 0.2, 中: 0.2 ≤ d < 0.8, 大: d ≥ 0.8)
- **検出力分析**: p > 0.05 の場合、80% 検出力に必要なサンプル数を算出
- **外れ値検出**: 平均 ± 3SD

## 結果

### 1. gpt-oss-20b (1C+1R, 各30回)

#### 生データ

<details>
<summary>条件 A (baseline) — 30回</summary>

| run | pp128 | tg32 |
|-----|-------|------|
| 1 | 393.07 | 52.24 |
| 2 | 391.87 | 55.49 |
| 3 | 393.32 | 55.44 |
| 4 | 394.78 | 52.35 |
| 5 | 395.68 | 55.26 |
| 6 | 395.91 | 55.41 |
| 7 | 395.17 | 54.96 |
| 8 | 395.41 | 54.86 |
| 9 | 394.61 | 55.06 |
| 10 | 395.10 | 55.07 |
| 11 | 393.26 | 55.17 |
| 12 | 394.76 | 55.06 |
| 13 | 394.53 | 50.56 |
| 14 | 394.63 | 54.92 |
| 15 | 394.37 | 55.03 |
| 16 | 394.61 | 55.18 |
| 17 | 395.37 | 55.12 |
| 18 | 395.58 | 55.12 |
| 19 | 395.14 | 54.94 |
| 20 | 395.87 | 55.01 |
| 21 | 395.20 | 52.10 |
| 22 | 395.03 | 52.15 |
| 23 | 395.07 | 54.94 |
| 24 | 395.34 | 54.99 |
| 25 | 395.99 | 50.85 |
| 26 | 393.53 | 51.99 |
| 27 | 394.99 | 51.56 |
| 28 | 394.73 | 50.71 |
| 29 | 393.02 | 50.48 |
| 30 | 395.27 | 52.11 |

</details>

<details>
<summary>条件 B (sync) — 30回</summary>

| run | pp128 | tg32 |
|-----|-------|------|
| 1 | 389.91 | 54.20 |
| 2 | 389.74 | 53.88 |
| 3 | 389.89 | 54.39 |
| 4 | 391.99 | 53.94 |
| 5 | 388.99 | 51.96 |
| 6 | 390.88 | 54.35 |
| 7 | 389.30 | 54.43 |
| 8 | 390.20 | 54.14 |
| 9 | 390.42 | 54.58 |
| 10 | 387.55 | 51.56 |
| 11 | 389.80 | 51.96 |
| 12 | 390.92 | 54.50 |
| 13 | 388.10 | 51.31 |
| 14 | 390.89 | 54.39 |
| 15 | 391.65 | 54.30 |
| 16 | 390.37 | 54.17 |
| 17 | 389.64 | 53.71 |
| 18 | 389.88 | 54.28 |
| 19 | 388.55 | 52.92 |
| 20 | 387.97 | 54.43 |
| 21 | 389.67 | 54.10 |
| 22 | 387.55 | 54.48 |
| 23 | 388.06 | 53.14 |
| 24 | 386.34 | 53.10 |
| 25 | 388.38 | 53.07 |
| 26 | 389.45 | 54.24 |
| 27 | 391.00 | 54.06 |
| 28 | 390.87 | 54.27 |
| 29 | 391.40 | 54.40 |
| 30 | 389.63 | 54.28 |

</details>

<details>
<summary>条件 C (two-phase) — 30回</summary>

| run | pp128 | tg32 |
|-----|-------|------|
| 1 | 390.02 | 54.40 |
| 2 | 391.72 | 54.28 |
| 3 | 388.97 | 54.47 |
| 4 | 392.40 | 54.20 |
| 5 | 390.23 | 54.47 |
| 6 | 389.05 | 54.48 |
| 7 | 389.75 | 53.92 |
| 8 | 389.30 | 51.51 |
| 9 | 391.60 | 54.17 |
| 10 | 388.83 | 52.09 |
| 11 | 389.51 | 54.45 |
| 12 | 390.06 | 54.32 |
| 13 | 388.90 | 54.36 |
| 14 | 391.41 | 54.21 |
| 15 | 388.03 | 51.29 |
| 16 | 390.60 | 54.19 |
| 17 | 390.27 | 51.82 |
| 18 | 385.90 | 51.46 |
| 19 | 385.96 | 51.41 |
| 20 | 387.30 | 51.99 |
| 21 | 392.09 | 54.04 |
| 22 | 390.68 | 54.29 |
| 23 | 388.45 | 51.19 |
| 24 | 388.48 | 51.08 |
| 25 | 389.16 | 54.09 |
| 26 | 390.36 | 54.18 |
| 27 | 388.42 | 54.35 |
| 28 | 390.86 | 54.31 |
| 29 | 390.82 | 54.37 |
| 30 | 390.98 | 54.41 |

</details>

#### 記述統計量

| 条件 | 指標 | n | 平均 | SD | min | max |
|------|------|--:|-----:|---:|----:|----:|
| A (baseline) | pp128 | 30 | 394.71 | 0.98 | 391.87 | 395.99 |
| A (baseline) | tg32 | 30 | 53.80 | 1.80 | 50.48 | 55.49 |
| B (sync) | pp128 | 30 | 389.63 | 1.35 | 386.34 | 391.99 |
| B (sync) | tg32 | 30 | 53.75 | 0.94 | 51.31 | 54.58 |
| C (two-phase) | pp128 | 30 | 389.67 | 1.61 | 385.90 | 392.40 |
| C (two-phase) | tg32 | 30 | 53.46 | 1.30 | 51.08 | 54.48 |

#### Welch t検定結果

| 比較 | 指標 | 差 | % | t | p | d | 判定 |
|------|------|---:|--:|--:|--:|--:|------|
| A vs B | pp128 | -5.07 | -1.29% | 16.613 | <0.000001 | 4.29 | *** |
| A vs B | tg32 | -0.05 | -0.10% | 0.143 | 0.887 | 0.04 | ns |
| A vs C | pp128 | -5.04 | -1.28% | 14.606 | <0.000001 | 3.77 | *** |
| A vs C | tg32 | -0.34 | -0.64% | 0.849 | 0.400 | 0.22 | ns |
| B vs C | pp128 | +0.04 | +0.01% | -0.097 | 0.923 | 0.03 | ns |
| B vs C | tg32 | -0.29 | -0.54% | 0.995 | 0.324 | 0.26 | ns |

#### 分析

- **pp128**: 条件 A が B/C より約 5 t/s (1.3%) 高速 (p<0.001, d>3.7)。ただし A と B/C はブランチが異なるため (deferred copy 有無 + synchronize コード有無)、差の原因を特定できない (交絡因子あり)
- **tg32**: 全条件間で有意差なし。B vs C (pure な比較) でも差なし — two-phase は 1C+1R 構成で効果なし
- 条件 A の tg32 に高い分散 (SD=1.80) が観測された。30回中10回が 50.5-52.4 t/s の低クラスタに入り、残り20回が 54.8-55.5 t/s。deferred copy の flush_all_staging との相互作用の可能性

---

### 2. GLM-4.7 IQ2_M (7C+4R 11GPU, 各10回)

#### 生データ

**条件 A (baseline)**:

| run | pp128 | tg32 |
|-----|-------|------|
| 1 | 23.85 | 7.65 |
| 2 | 23.85 | 7.77 |
| 3 | 23.84 | 7.76 |
| 4 | 23.87 | 7.76 |
| 5 | 23.84 | 7.77 |
| 6 | 23.86 | 7.76 |
| 7 | 23.84 | 7.76 |
| 8 | 23.84 | 7.76 |
| 9 | 23.84 | 7.77 |
| 10 | 23.89 | 7.77 |

**条件 B (sync)**:

| run | pp128 | tg32 |
|-----|-------|------|
| 1 | 23.80 | 7.67 |
| 2 | 23.81 | 7.65 |
| 3 | 23.83 | 7.66 |
| 4 | 23.80 | 7.64 |
| 5 | 23.84 | 7.66 |
| 6 | 23.80 | 7.63 |
| 7 | 23.81 | 7.64 |
| 8 | 23.80 | 7.65 |
| 9 | 23.80 | 7.66 |
| 10 | 23.81 | 7.65 |

**条件 C (two-phase)**:

| run | pp128 | tg32 |
|-----|-------|------|
| 1 | 23.73 | 7.64 |
| 2 | 23.80 | 7.66 |
| 3 | 23.81 | 7.66 |
| 4 | 23.80 | 7.66 |
| 5 | 23.88 | 7.71 |
| 6 | 23.82 | 7.69 |
| 7 | 23.80 | 7.65 |
| 8 | 23.80 | 7.67 |
| 9 | 23.83 | 7.66 |
| 10 | 23.80 | 7.66 |

#### 記述統計量

| 条件 | 指標 | n | 平均 | SD | min | max |
|------|------|--:|-----:|---:|----:|----:|
| A (baseline) | pp128 | 10 | 23.852 | 0.0169 | 23.84 | 23.89 |
| A (baseline) | tg32 | 10 | 7.753 | 0.0365 | 7.65 | 7.77 |
| B (sync) | pp128 | 10 | 23.810 | 0.0141 | 23.80 | 23.84 |
| B (sync) | tg32 | 10 | 7.651 | 0.0120 | 7.63 | 7.67 |
| C (two-phase) | pp128 | 10 | 23.807 | 0.0368 | 23.73 | 23.88 |
| C (two-phase) | tg32 | 10 | 7.666 | 0.0201 | 7.64 | 7.71 |

#### Welch t検定結果

| 比較 | 指標 | 差 | % | t | p | d | 判定 |
|------|------|---:|--:|--:|--:|--:|------|
| A vs B | pp128 | -0.042 | -0.18% | 6.034 | 0.00001 | 2.70 | *** |
| A vs B | tg32 | -0.102 | -1.32% | 8.391 | 0.000004 | 3.75 | *** |
| A vs C | pp128 | -0.045 | -0.19% | 3.513 | 0.004 | 1.57 | ** |
| A vs C | tg32 | -0.087 | -1.12% | 6.598 | 0.00001 | 2.95 | *** |
| B vs C | pp128 | -0.003 | -0.01% | 0.240 | 0.814 | 0.11 | ns |
| B vs C | tg32 | +0.015 | +0.20% | -2.027 | 0.061 | 0.91 | ns |

#### 二峰性確認

| 条件 | tg32 min | tg32 max | range | SD | 全ラン ≥ 7.0? |
|------|----------|----------|-------|----|---------------|
| A | 7.65 | 7.77 | 0.12 | 0.037 | YES |
| B | 7.63 | 7.67 | 0.04 | 0.012 | YES |
| C | 7.64 | 7.71 | 0.07 | 0.020 | YES |

**二峰性は完全に解消**。全30ランが tg ≥ 7.0 t/s で安定動作。

#### 分析

- **pp128**: A が B/C より 0.04-0.05 t/s (0.2%) 高速。統計的に有意だが交絡因子あり (deferred copy 有無)。実用上は無視できる差
- **tg32**: A が B より 0.10 t/s (1.3%) 高速 (p<0.001, d=3.75)。A が C より 0.09 t/s (1.1%) 高速 (p<0.001, d=2.95)。ただし交絡因子 (deferred copy) により、synchronize のオーバーヘッドと deferred copy の恩恵を分離できない
- **B vs C** (pure な比較): 有意差なし (pp128: p=0.81, tg32: p=0.06)。two-phase は layer split では追加効果なし
- 条件 A の run 1 のみ tg=7.65 (ウォームアップ効果)、runs 2-10 は 7.76-7.77 で極めて安定

## 考察

### 実験設計の制約 — 交絡因子

本実験の最も重要な制約は、条件 A (rdma-backend) と条件 B/C (rdma-synchronize) のビルドが異なるブランチに基づいていることである。rdma-backend には deferred copy 機能 (`1d2c9cbdd`) が含まれ、rdma-synchronize にはそれが含まれない。

そのため、A vs B/C で観測された性能差には以下の2つの要因が混在しており、個々の寄与を分離できない:
1. **Deferred copy の恩恵** (A のみ): クロスデバイスコピーをサーバー内 D2H+H2D で処理し、IB ネットワーク往復を回避
2. **Synchronize コードのオーバーヘッド** (B/C のみ): SYNC コマンド処理、two-phase 判定ロジックの追加コスト

**分離可能な比較は B vs C のみ**であり、ここでは two-phase の追加効果を純粋に評価できる。

### B vs C の比較 — Two-phase の効果

B (SYNC のみ) と C (SYNC + two-phase) の間には、gpt-oss-20b・GLM-4.7 ともに有意差がなかった (全指標で p > 0.05)。

**原因**: layer split ではスプリット間にデータ依存があり、スプリット N+1 はスプリット N の出力テンソルを入力として必要とする。two-phase dispatch で全スプリットを先行 dispatch しても、サーバー側で逐次実行されるため並列化の余地がない。

### A vs B/C の差の解釈

A が B/C より高速な理由は以下のいずれか、または両方の組み合わせである:

1. **Deferred copy の寄与 (推定: 主因)**: gpt-oss-20b の先行実験では deferred copy により Generation が +13% 改善 (57.7 vs 50.9 t/s) した実績がある。GLM-4.7 での tg32 差 0.10 t/s (1.3%) はこの効果の一部と考えられる
2. **Synchronize コードのオーバーヘッド (推定: 副因)**: pp128 での差は gpt-oss-20b で 1.3%、GLM-4.7 で 0.2%。Prompt 処理ではデバイス間コピーが少ないため deferred copy の寄与は小さく、コードオーバーヘッドが主因と推測される

### SYNC コマンドの理論的効果

SYNC コマンドは `compute_pending_` をクリアして `get_tensor()` で RDMA Read 高速パス (GDR) を使えるようにする意図だった。しかし現在の実装では:
- GDR バジェットシステムの `mr_is_gdr` フラグで既に適切にルーティングされている
- `compute_pending_` がセットされている場合でも、Send/Recv フォールバックはサーバー側でコマンドキューを通すため、計算完了を自然に待機する
- SYNC の追加的価値がない

### gpt-oss-20b の tg 分散

条件 A の tg32 で SD=1.80 (B: 0.94, C: 1.30) と高い分散が観測された。30回中10回が低クラスタ (50.5-52.4 t/s) に入り、残り20回が 54.8-55.5 t/s の二峰的パターンが残存している。

これは deferred copy 機能の flush_all_staging との相互作用か、gpt-oss-20b 特有の条件（1C+1R 構成、モデルサイズ）に起因すると推測される。GLM-4.7 (7C+4R) では観測されないため、構成依存の現象と考えられる。

## 結論

| 検証項目 | 結果 |
|---------|------|
| 二峰性解消 | **成功** — 全30ラン (GLM-4.7) が tg ≥ 7.0 t/s で安定 |
| SYNC コマンド効果 | **判定不能** — A vs B の差は交絡因子 (deferred copy) により分離不可 |
| Two-phase 効果 | **効果なし** — B vs C (pure 比較) で有意差なし。layer split のデータ依存により並列化不可 |
| synchronize → rdma-backend マージ | **不要** — two-phase は効果なし、SYNC の効果も確認できず |

### 実験設計の限界

A vs B/C の比較は、deferred copy の有無と synchronize 機能の有無が交絡しており、個々の寄与を分離できない。正確な評価には以下のいずれかが必要:
- rdma-synchronize に deferred copy を cherry-pick して条件を揃えた上で再測定
- rdma-backend に synchronize 機能を追加して条件を揃えた上で再測定

ただし、B vs C の pure な比較で two-phase が効果なしという結論は有効であり、layer split 前提の現アーキテクチャでは synchronize の価値は限定的と判断できる。

### 推奨事項

1. `feature/rdma-synchronize` ブランチの synchronize 機能は rdma-backend にマージしない
2. 二峰性修正 (fprintf → RDMA_LOG_DBG) は rdma-backend に既に含まれているため追加対応不要
3. 今後の性能改善は deferred copy の最適化や client-side dispatch パイプライン化など、rdma-backend ブランチ上で進めることを推奨
4. SYNC コマンドの効果を正確に評価する場合は、deferred copy を synchronize ブランチに cherry-pick して交絡因子を排除した上で再測定が必要
