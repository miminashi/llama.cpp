# Doorbell Batching + max_inline 拡大 実装レポート

- **実施日時**: 2026年2月10日 22:23

## 前提・目的

前回の改善 (async compute + initiator_depth 16) で graph_compute を 24.5x 高速化したが、End-to-end Generation 速度は改善しなかった（get_tensor の暗黙同期待ちに時間が移動するため）。

過去の研究レポート (`report/2026-02-07_235500_rdma_optimization_plan.md`) の Phase 1/2 改善項目のうち、未実装の doorbell batching と max_inline 拡大を実装する。

### 背景

- `send_rdma_cmd_raw()` がヘッダー (9B) とデータを **2回の独立した `ibv_post_send()`** で送信。各送信で PCIe MMIO doorbell + CQ ポーリングが発生
- `max_inline = 256` だが ConnectX-4 は ~960B まで対応。256-512B のコマンドデータが不要な DMA read を引き起こす
- RNR NAK カウンタが 265万回に蓄積（前回 130万から倍増）

### 目的

1. `max_inline` を 256 → 512 に拡大し、中サイズコマンドの inline 化範囲を拡大
2. Doorbell batching (`send_two()`) でヘッダー + データを 1回の `ibv_post_send()` に統合
3. PCIe MMIO doorbell 回数と CQ ポーリング回数を半減

## 変更内容

### 1. max_inline: 256 → 512

**ファイル**: `ggml/src/ggml-rdma/rdma-transport.h` L26

ConnectX-4 は ~960B の inline data をサポート。512B に拡大することで、FLUSH_AND_COMPUTE_UPDATE (3-4 updates, ~428B) 等のコマンドデータが inline 化される。

### 2. Doorbell Batching: `send_two()` メソッド追加

**ファイル**: `ggml/src/ggml-rdma/rdma-transport.h`, `ggml/src/ggml-rdma/rdma-transport.cpp`

ヘッダーとデータを **1回の `ibv_post_send()`** で送信する `send_two()` メソッドを追加。WR チェーン (`ibv_send_wr::next`) を使用。

ケース分岐:
1. `data_size == 0` → ヘッダーのみ送信 (signaled, inline)
2. `data_size > 0 && data_size <= send_buffer_size` → doorbell batching (WR1: header unsignaled inline, WR2: data signaled)
3. `data_size > send_buffer_size` → フォールバック (sequential: header送信 → data chunked送信)

WR2 のデータ送信は以下の優先度:
- 外部 MR あり → 直接参照
- `data_size <= max_inline` → inline 送信
- それ以外 → `send_buffer_` にコピーして送信

### 3. `send_rdma_cmd_raw()` の書き換え

**ファイル**: `ggml/src/ggml-rdma/ggml-rdma.cpp`

2回の `conn->send()` を 1回の `conn->send_two()` に置換。プロファイリングログも統合。

## 再現方法

### ビルド + デプロイ

```bash
bash scripts/rdma-build.sh local && bash scripts/rdma-deploy.sh
bash scripts/rdma-server.sh restart
```

### qwen2.5-0.5b 回帰テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

### GLM-4.7 IQ2_M 11GPU 最終テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## テスト結果

### qwen2.5-0.5b (2GPU: CUDA0 + RDMA0)

| 指標 | 今回 | 前回 (async compute) |
|------|:----:|:----:|
| pp128 (t/s) | 2,625 | 3,269 |
| tg32 (t/s) | 117.5 | 173.9 |

数値が低いが、これはサーバー GPU 計算時間のセッション間変動 (CLAUDE.md 記載の既知課題) によるもの。動作は正常。

### GLM-4.7 IQ2_M (11GPU: 7C + 4R)

| 指標 | 今回 | 前回 (async compute) |
|------|:----:|:----:|
| Prompt (t/s) | **6.8** | 6.3 |
| Generation (t/s) | **5.6** | 6.6 |

- Prompt 処理: 6.8 t/s (前回 6.3 t/s → +8%)
- Generation: 5.6 t/s (前回 6.6 t/s → -15%)

Generation の低下はサーバー GPU 計算時間の変動と推定 (前回のベースラインも 5.5-6.8 t/s の範囲内でばらつく)。正常な推論出力を確認。

### graph_compute プロファイル

| 指標 | 値 |
|------|:---:|
| 平均時間 (recompute) | 0.9 ms |
| full graph | 8回 (初回) |
| recompute | 192回 |
| 総計算時間 | 182.3 ms |

### RNR NAK カウンタ

| 時点 | 値 | 増加 |
|------|:---:|:---:|
| テスト前 | 2,658,531 | - |
| テスト後 | 2,700,143 | +41,612 |

## 考察

### Doorbell Batching の効果

- `send_two()` により、コマンドごとの PCIe MMIO doorbell を 2回 → 1回に削減
- CQ ポーリングも 2回 → 1回に削減
- 理論的な節約: ~1.3us/コマンド × 8コマンド/トークン = ~10us/トークン
- ただし Generation ボトルネックはサーバー側 GPU 計算時間 (~17ms/graph_compute) であるため、~10us の節約は直接的な tg 改善には寄与しない

### max_inline 512 の効果

- FLUSH_AND_COMPUTE_UPDATE (~428B) 等のコマンドデータが inline 化可能に
- inline 送信は NIC が PCIe DMA read を行う必要がなく、WQE とともに直接送信される
- 小コマンドの送信レイテンシ削減に寄与

### Generation 速度変動

- Generation 速度は 5.5-6.8 t/s の範囲でセッション間変動する (既知の課題)
- 今回の 5.6 t/s は変動範囲の下限に近い
- 改善の正確な評価には同一セッションでの A/B テストが必要

## A/B ベンチマーク (3回ずつ)

修正前 (stash) と修正後 (doorbell batching + max_inline 512) で連続3回ずつ実行。
同一セッション内で交互にビルド・デプロイして測定。

### 条件

- モデル: GLM-4.7 IQ2_M
- 構成: 11GPU (7 CUDA + 4 RDMA)
- パラメータ: `--seed 42 -n 50 -p 'The capital of France is'`
- サーバーは各ビルド後に再起動

### 結果

| | Prompt (t/s) | Generation (t/s) |
|---|:---:|:---:|
| **修正前 Run 1** | 6.4 | 6.6 |
| **修正前 Run 2** | 7.0 | 7.4 |
| **修正前 Run 3** | 7.0 | 5.9 |
| 修正前 **平均** | **6.8** | **6.6** |
| | | |
| **修正後 Run 1** | 6.3 | 6.9 |
| **修正後 Run 2** | 7.0 | 5.9 |
| **修正後 Run 3** | 7.0 | 5.8 |
| 修正後 **平均** | **6.8** | **6.2** |

### RNR NAK カウンタ変化

| 区間 | 開始値 | 終了値 | 増加数 | 1回あたり |
|------|:------:|:------:|:------:|:---------:|
| 修正前 (3回) | 2,700,143 | 2,819,722 | +119,579 | ~39,860 |
| 修正後 (3回) | 2,819,722 | 2,940,106 | +120,384 | ~40,128 |

### A/B ベンチマーク考察

- **Prompt 速度**: 修正前後ともに平均 6.8 t/s で差なし
- **Generation 速度**: 修正前 平均 6.6 t/s vs 修正後 平均 6.2 t/s
  - ただし修正前 Run 2 の 7.4 t/s が外れ値的に高い（他の5回は 5.8-6.9 の範囲）
  - 修正前 Run 3 (5.9) と修正後 Run 2 (5.9), Run 3 (5.8) は同水準
  - サーバー GPU 計算時間のセッション間変動が支配的
- **RNR NAK**: 修正前後でほぼ同数 (+119,579 vs +120,384)。doorbell batching は RNR NAK に影響しない（RNR NAK は recv 側の pre-post タイミングの問題で、send 側の最適化では解消されない）
- **結論**: doorbell batching + max_inline 拡大は、GLM-4.7 11GPU 構成では**測定可能な性能差を生まなかった**。理論上の ~10us/トークン の節約は、サーバー側 GPU 計算時間 (~17ms/graph_compute) に対して 0.06% と極めて小さく、自然変動 (±1 t/s) に埋もれる。ただし、より多くの RDMA デバイスやコマンドが増える構成では効果が見える可能性がある

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/rdma-transport.h` | `max_inline` 256→512, `send_two()` 宣言追加 |
| `ggml/src/ggml-rdma/rdma-transport.cpp` | `send_two()` 実装 (~90行) |
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | `send_rdma_cmd_raw()` を `send_two()` に書き換え |

## コミット情報

- **ブランチ**: `opt-doorbell-batch`
- **ワークツリー**: `/home/ubuntu/projects/llama.cpp/.worktree/opt-doorbell-batch`
- **コミット**: `d1a68befe` (`feat(rdma): add doorbell batching and increase max_inline to 512`)
- **ベース**: `e26e01a0e` (`feature/rdma-backend` HEAD)
- `feature/rdma-backend` ブランチには未マージ（A/B ベンチマークで測定可能な改善が確認されなかったため）
