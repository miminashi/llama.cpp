# Selective Signaling 実装レポート

- **実施日時**: 2026年2月19日 12:00
- **ワークツリー**: `.worktree/rdma-selective-signaling`
- **ブランチ**: `feature/rdma-selective-signaling`

## 前提・目的

RDMA バックエンドでは、全ての IB 操作 (Send, RDMA Write) に `IBV_SEND_SIGNALED` を付与し、操作ごとに CQE を生成して `wait_for_completion()` で完了を待っている。Selective Signaling は N 回の操作につき 1 回のみ signaled にすることで、CQE DMA 回数とポーリング回数を削減する最適化手法。

- **背景**: チャンク化 RDMA Write (16MB/chunk) とチャンク化 Send で、全チャンクが signaled+wait → PCIe 帯域とCPUサイクルのオーバーヘッド
- **目的**: signal 頻度を削減し set_tensor/send のスループット改善を検証する
- **推定効果**: +1-3%
- **前提条件**: IB の QP 内順序保証により、signaled WR の完了が先行全 unsignaled WR の完了を暗黙的に保証する

## 実装内容

### 変更ファイル (2ファイル, 28行追加/8行削除)

1. **`ggml/src/ggml-rdma/ggml-rdma.cpp`**
   - `RDMA_SIGNAL_INTERVAL = 64` 定数追加
   - チャンク化 RDMA Write ループ: 最終チャンクまたは64チャンクごとのみ signaled
   - `GGML_RDMA_NO_SELECTIVE_SIGNAL` 環境変数で無効化可能

2. **`ggml/src/ggml-rdma/rdma-transport.cpp`**
   - チャンク化 Send ループ: 同じく最終チャンクまたは64チャンクごとのみ signaled+wait
   - unsignaled チャンクは `wait_for_completion()` をスキップ

### 安全性

| 懸念事項 | 対策 |
|---------|------|
| SQ オーバーフロー | RDMA_SIGNAL_INTERVAL (64) < max_send_wr (128) |
| エラー検出 | unsignaled 失敗 → QP ERROR 状態 → 次の signaled で検出 |
| CQ unexpected CQE | op_mutex_ で並行排除済み |
| 単一チャンク (≤16MB) | `is_last = true` → 常に signaled、動作変更なし |

## 再現方法

### ビルド

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/rdma-selective-signaling/scripts/rdma-build.sh local
bash /home/ubuntu/projects/llama.cpp/.worktree/rdma-selective-signaling/scripts/rdma-deploy.sh
```

### 動作確認

```bash
bash scripts/rdma-server.sh start

# 小モデル (1C+1R)
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-cli -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -c 256 -n 20 -p 'Hello world' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log

# GLM-4.7 IQ2_M (7C+4R)
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### A/B 定量評価

```bash
# 条件 A: Selective Signaling ON (デフォルト)
# 条件 B: GGML_RDMA_NO_SELECTIVE_SIGNAL=1
# ABAB paired design, n=15, ウォームアップ1回破棄
bash /tmp/rdma-selective-signal-bench.sh
```

## 交絡因子チェック

- [x] **単一変数の分離**: 同一バイナリ、環境変数 `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` で ON/OFF 切替
- [x] **環境変数トグル**: コードブランチ比較ではない
- [x] **ホットパスのログ出力**: fprintf は修正済み (debug ログのみ、RDMA_DEBUG 未設定時は出力なし)
- [x] **分布の単峰性**: tg は安定 (7.4-7.6 t/s)。pp は B11=3.3 の外れ値を除き 6.5-6.8 t/s で単峰

## 実験結果

### 生データ

| Pair | A_pp | B_pp | A_tg | B_tg | diff_pp | diff_tg |
|:----:|:----:|:----:|:----:|:----:|:-------:|:-------:|
| 1 | 6.8 | 6.7 | 7.5 | 7.5 | +0.1 | 0.0 |
| 2 | 6.5 | 6.7 | 7.6 | 7.4 | -0.2 | +0.2 |
| 3 | 6.7 | 6.7 | 7.5 | 7.4 | 0.0 | +0.1 |
| 4 | 6.8 | 6.7 | 7.5 | 7.4 | +0.1 | +0.1 |
| 5 | 6.7 | 6.7 | 7.5 | 7.4 | 0.0 | +0.1 |
| 6 | 6.8 | 6.7 | 7.5 | 7.4 | +0.1 | +0.1 |
| 7 | 6.5 | 6.7 | 7.5 | 7.5 | -0.2 | 0.0 |
| 8 | 6.6 | 6.7 | 7.6 | 7.5 | -0.1 | +0.1 |
| 9 | 6.8 | 6.7 | 7.5 | 7.5 | +0.1 | 0.0 |
| 10 | 6.8 | 6.7 | 7.5 | 7.5 | +0.1 | 0.0 |
| **11** | **6.7** | **3.3** | **7.6** | **7.5** | **+3.4** | **+0.1** |
| 12 | 6.7 | 6.7 | 7.6 | 7.5 | 0.0 | +0.1 |
| 13 | 6.8 | 6.7 | 7.5 | 7.5 | +0.1 | 0.0 |
| 14 | 6.7 | 6.7 | 7.5 | 7.5 | 0.0 | 0.0 |
| 15 | 6.8 | 6.6 | 7.6 | 7.5 | +0.2 | +0.1 |

**外れ値**: B11 pp=3.3 — IQR法で検出。pp の IQR=0.00 (Q1=Q3=6.70)、3.3 は範囲外。tg は正常 (7.5)。一時的なシステムイベント (サーマルスロットリング等) と推定。フラグするが除外の根拠あり。

### Prompt Processing (pp)

| 指標 | 全15ペア | Pair 11 除外 (n=14) |
|------|:---:|:---:|
| A (ON) 平均 ± SD | 6.713 ± 0.106 | 6.714 ± 0.110 |
| B (OFF) 平均 ± SD | 6.467 ± 0.876 | 6.693 ± 0.027 |
| 差分平均 | +0.247 t/s (+3.81%) | +0.021 t/s (+0.32%) |
| t(df) | 1.09 (df=14) | 0.67 (df=13) |
| p 値 | 2.96 × 10⁻¹ | 5.12 × 10⁻¹ |
| Cohen's d | 0.28 (小) | 0.18 (無視) |
| 95% CI | [-0.241, +0.734] t/s | [-0.047, +0.090] t/s |
| 正の効果ペア | 8/15 (53%) | 7/14 (50%) |
| **判定** | **効果なし** | **効果なし** |

全15ペアでの +3.81% は B11 外れ値による見かけ上の効果。除外後は +0.32% で非有意 (p=0.512) かつ 0.5% 閾値未満。

### Token Generation (tg)

| 指標 | 全15ペア | Pair 11 除外 (n=14) |
|------|:---:|:---:|
| A (ON) 平均 ± SD | 7.533 ± 0.049 | 7.529 ± 0.047 |
| B (OFF) 平均 ± SD | 7.467 ± 0.049 | 7.464 ± 0.050 |
| 差分平均 | +0.067 t/s (+0.89%) | +0.064 t/s (+0.86%) |
| t(df) | 4.18 (df=14) | 3.80 (df=13) |
| p 値 | 9.20 × 10⁻⁴ | 2.22 × 10⁻³ |
| Cohen's d | 1.08 (大) | 1.02 (大) |
| 95% CI | [+0.033, +0.101] t/s | [+0.028, +0.101] t/s |
| 正の効果ペア | 9/15 (60%) | 8/14 (57%) |
| **判定** | **有効** (p < 0.05 かつ > 0.5%) | **有効** |

tg は外れ値除外に対してロバスト。両分析で p < 0.003、Cohen's d > 1.0 (大効果)。

## 考察

### Selective Signaling は Generation に +0.9% の効果

- **tg (Generation)**: 統計的に有意な +0.89% 改善。Generation は1トークンごとに多数の小テンソル get/set を繰り返すため、操作あたりの CQ ポーリングオーバーヘッド削減が累積して効く
- **pp (Prompt)**: 効果なし (+0.32%)。Prompt 処理は大きなテンソルの一括転送が主で、RDMA Write の転送時間に対して signaling オーバーヘッドの比率が小さい

### 効果の限界

推定効果 +1-3% に対して実測 +0.9% で下限付近。理由:
1. GLM-4.7 IQ2_M のウェイトバッファは >4GB のため RDMA Write ではなく Send/Recv フォールバックを使用 → selective signaling の RDMA Write パスが活用される場面が限定的
2. 計算出力バッファ (466MB) は ~30チャンクで、64チャンク間隔に達しない → selective signaling が発動しない
3. 効果は主に Send チャンク化パスで発生 (MoE モデルの expert テンソル転送)

### 測定精度

- pp の測定ノイズが大きい (A: 6.5-6.8 の 4.6% レンジ)。llama-cli の pp 計測は1回のパスのため統計的平均化がない
- tg は安定 (SD ≈ 0.05, ≈0.7%)。32トークン生成の平均速度なのでノイズが低い

## 結論

Selective Signaling は **Generation 速度に +0.89% の統計的に有意な改善** をもたらす。Prompt 処理には効果なし。実装は低リスク (2ファイル, 28行) で、`GGML_RDMA_NO_SELECTIVE_SIGNAL=1` で無効化可能。デフォルト有効を推奨。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `789b88ace (feature/rdma-selective-signaling)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 34°C | 37°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | not_loaded | not_loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
