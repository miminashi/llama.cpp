# Step 3 RDMA性能最適化: FLUSH+COMPUTE統合 & プロファイリングレポート

- **実施日時**: 2026年2月5日 00:34
- **関連レポート**: [バッチフラッシュ最適化](rdma_batch_flush_optimization_2026-02-04_224611.md), [CPU staged vs GPUDirect比較](cpu_staged_vs_gpudirect_rdma_comparison_2026-02-04_214044.md)

## 前提・目的

Step 3 (RDMA性能最適化) の一環として、以下の最適化を実装・検証する。

- **背景**: バッチフラッシュ最適化後、tg32 = 14.0 t/s (ローカル比 6.5%) でボトルネックは graph_compute のラウンドトリップ (~57ms/call) だった
- **目的**:
  1. サーバー側プロファイリングで graph_compute 内部の時間内訳を特定
  2. FLUSH_ALL_STAGING + GRAPH_RECOMPUTE/UPDATE の2ラウンドトリップを1回に統合
  3. クライアント側の詳細プロファイリングでボトルネックを特定
- **前提条件**:
  - 1号機 (192.168.100.1): 7× Tesla P100-PCIE-16GB, CUDA GPU 0 を使用
  - 2号機 (192.168.100.2): 4× Tesla P100-PCIE-16GB, rdma-server を実行
  - `GGML_RDMA_NO_GDR=1` (GPUDirect RDMA 無効)
  - テストモデル: qwen2.5-0.5b-instruct-q4_k_m.gguf, gpt-oss-20b-Q4_K_M.gguf
  - レイヤー分割 (`-sm layer`), `-ngl 999`

## 実装内容

### 1. サーバー側プロファイリング (`GGML_RDMA_PROFILE=1`)

サーバーコマンドディスパッチループに計測を追加:
- `graph_recompute`: fix_cross_device_refs + compute 時間
- `graph_compute_update`: apply_updates + fix_xdev + compute 時間
- `graph_compute` (full): deserialize + fix_xdev + compute 時間
- `flush_all_staging`: flush 時間 + entries/copies 数
- コマンドレベル: recv + processing + rsp_send 時間

### 2. FLUSH + COMPUTE コマンド統合

新コマンド2つを追加:
- `RDMA_CMD_FLUSH_AND_RECOMPUTE`: ワイヤーフォーマット `| n_flush(4B) | flush_entries(N*24B) | device(4B) |`
- `RDMA_CMD_FLUSH_AND_COMPUTE_UPDATE`: ワイヤーフォーマット `| n_flush(4B) | flush_entries(N*24B) | device(4B) | n_updates(4B) | updates(M*100B) |`

効果: graph_compute のたびに2回のラウンドトリップ (FLUSH → COMPUTE) を1回に削減。

### 3. クライアント側詳細プロファイリング

graph_compute 内の per-call タイミング:
- `pre_send`: flush entry 準備時間
- `send_cmd`: コマンド送信 → レスポンス受信 (ラウンドトリップ全体)
- `snapshot_update`: build_snapshot_map + gc.add 時間

## 再現方法

### サーバー起動 (2号機)

```bash
# 1 GPU のみ公開 (2GPU テスト用)
ssh 192.168.100.2
GGML_RDMA_NO_GDR=1 GGML_RDMA_PROFILE=1 CUDA_VISIBLE_DEVICES=0 \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server \
  --host 0.0.0.0 --port 50051 > /tmp/rdma-server.log 2>&1 &

# 4 GPU 全部公開 (5GPU テスト用)
GGML_RDMA_NO_GDR=1 GGML_RDMA_PROFILE=1 \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server \
  --host 0.0.0.0 --port 50051 > /tmp/rdma-server.log 2>&1 &
```

### ベンチマーク実行 (1号機)

```bash
# qwen2.5-0.5b (小規模テスト)
GGML_RDMA_NO_GDR=1 GGML_RDMA_PROFILE=1 CUDA_VISIBLE_DEVICES=0 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  timeout 120 build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32 2>/tmp/bench-profile.log

# gpt-oss-20b (中規模テスト)
GGML_RDMA_NO_GDR=1 GGML_RDMA_PROFILE=1 CUDA_VISIBLE_DEVICES=0 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  timeout 180 build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32 2>/tmp/bench-profile.log
```

## ベンチマーク結果

### qwen2.5-0.5b (0.5B params, 463 MiB)

| 構成 | pp128 (t/s) | tg32 (t/s) |
|------|:-----------:|:----------:|
| ローカル 1GPU | 3,390 | 213.36 |
| **RDMA 1+1 (FLUSH+COMPUTE統合)** | **3,113** | **156.22** |
| RDMA 1+1 (バッチフラッシュのみ) | 3,104 | 14.0 |
| Send/Recv ベースライン | 2,989 | 2.82 |

### gpt-oss-20b (20.9B params, 10.8 GiB)

| 構成 | pp128 (t/s) | tg32 (t/s) |
|------|:-----------:|:----------:|
| ローカル 1GPU | 407.12 | 64.27 |
| **RDMA 1+1 (1 remote GPU)** | **61.09** | **58.47** |
| RDMA 1+4 (4 remote GPUs) | 349.20 | 12.58 |

### ローカル比性能

| 構成 | モデル | pp128 比 | tg32 比 |
|------|--------|:--------:|:-------:|
| RDMA 1+1 | qwen2.5-0.5b | 91.8% | **73.2%** |
| RDMA 1+1 | gpt-oss-20b | 15.0% | **91.0%** |
| RDMA 1+4 | gpt-oss-20b | 85.8% | 19.6% |

### Step 3 目標達成状況

| 目標 | 基準 | 結果 | 達成 |
|------|------|------|:----:|
| gpt-oss-20b 2GPU tg32 | 30+ t/s | **58.47 t/s** | ✅ |
| gpt-oss-20b 2GPU pp128 | 40+ t/s | **61.09 t/s** | ✅ |

## プロファイル詳細

### サーバー側 (gpt-oss-20b, 1 remote GPU, tg32 定常状態)

```
[server profile] graph_recompute: total=3.21 ms (fix_xdev=0.01, compute=3.21)
[server profile] flush_and_recompute: total=3.31 ms (flush=0.09, n_flush=1)
[server profile] cmd FLUSH_AND_RECOMPUTE: total=3.42 ms (recv=0.10, rsp_send=0.00)
```

- **compute=3.21ms**: サーバー GPU での実際の計算時間
- **flush=0.09ms**: ステージング → GPU コピー
- **recv=0.10ms**: コマンド受信
- **total=3.42ms**: コマンド処理全体

### クライアント側 (gpt-oss-20b, 1 remote GPU, tg32 定常状態)

```
[client detail] recompute: pre_send=0.19 ms, send_cmd=9.26 ms
[client detail] snapshot_update: 0.30 ms
[client profile] graph_compute: 9.78 ms (reuse=1, n_flush=1, n_nodes=675)
```

- **pre_send=0.19ms**: flush entry 準備
- **send_cmd=9.26ms**: コマンド送信 + サーバー処理 + レスポンス受信
  - サーバー処理 3.42ms + ネットワーク RTT ~0.2ms × 2 + オーバーヘッド ~5.2ms
- **snapshot_update=0.30ms**: グラフスナップショット更新

### クライアント側 (qwen2.5-0.5b, 1 remote GPU, tg32 定常状態)

```
[client detail] recompute: pre_send=0.05 ms, send_cmd=0.95 ms
[client detail] snapshot_update: 0.05 ms
[client profile] graph_compute: 1.11 ms (reuse=1, n_flush=1, n_nodes=195)
```

- 小規模モデルでは graph_compute が **~1.1ms/call** と非常に高速

## 分析

### 成功ポイント

1. **tg32 の劇的改善**: qwen2.5-0.5b: 14.0 → 156.22 t/s (**11.2倍**)、gpt-oss-20b 2GPU: 58.47 t/s (ローカル比 91.0%)
2. **FLUSH+COMPUTE統合**: 2ラウンドトリップ → 1ラウンドトリップに削減
3. **Step 3 目標達成**: gpt-oss-20b 2GPU で tg32 58.47 t/s (目標 30+) と pp128 61.09 t/s (目標 40+)

### 発見事項

1. **マルチ RDMA デバイスのスケーリング問題**: サーバーが4 GPUを公開すると、スケジューラが5 splits (1 CUDA + 4 RDMA) を生成し、各 split がシリアルに実行される。graph_compute が ~4 calls/token になり、tg32 が 12.58 t/s に低下。1 remote GPU (2 splits) では 58.47 t/s と良好。
2. **pp128 の低さ (gpt-oss-20b)**: 1+1 構成で 61 t/s (ローカル比 15%)。初回の全グラフ送信 (~4.2秒) + 大量のモデルウェイト転送 (~5.8秒) がボトルネック。pp128 は warm-up 後は改善するが、llama-bench の計測には初回送信も含まれる。
3. **ラウンドトリップの構成**: gpt-oss-20b の send_cmd=9.26ms のうち、サーバー処理=3.42ms。残り ~5.8ms は send/recv のシリアライゼーション + カーネルオーバーヘッド。

### 残課題

1. **マルチ GPU スケーリング**: 4 remote GPU でも性能が出るよう、graph_compute の並列化が必要 (現在シリアル)
2. **pp128 改善**: warm-up キャッシュや非同期テンソル転送で初回のコスト削減
3. **send_cmd オーバーヘッド削減**: 9.26ms のうちサーバー処理は 3.42ms のみ。残りの ~5.8ms の削減余地あり

## まとめ

FLUSH+COMPUTE コマンド統合と詳細プロファイリングにより、RDMA バックエンドの性能が大幅に改善された:

- **qwen2.5-0.5b tg32**: 14.0 → **156.22 t/s** (ローカル比 73.2%)
- **gpt-oss-20b tg32**: → **58.47 t/s** (ローカル比 91.0%, **Step 3 目標達成**)
- **gpt-oss-20b pp128**: → **61.09 t/s** (Step 3 目標達成)

Step 3 の主要目標は達成。次のステップとして:
- マルチ GPU 並列化 (splits の並列 graph_compute)
- Step 4 (GPUDirect RDMA) への移行準備
