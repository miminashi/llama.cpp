# RDMA バッチフラッシュ最適化レポート

- **実施日時**: 2026年2月4日 22:46
- **関連レポート**: [ベンチマークレポート](rdma_benchmark_gpt_oss_2026-02-04_110225.md), [120bクラスタ修正](rdma_120b_cluster_fix_2026-02-04_184824.md)

## 前提・目的

Step 3 (RDMA性能最適化) の一環として、RDMA Write パスのバッチフラッシュ最適化を実装・検証する。

- **背景**: RDMA Write (one-sided) パスでは、set_tensor 後にサーバー側のホストステージングバッファから GPU メモリへの `cudaMemcpy` (FLUSH_STAGING) が必要。従来は各 set_tensor ごとに個別の FLUSH_STAGING コマンドを送信しており、ラウンドトリップオーバーヘッドにより Send/Recv パスより遅くなっていた。
- **目的**: set_tensor 時はステージングバッファへの RDMA Write のみ行い、graph_compute 前に一括でフラッシュすることで性能を改善する。
- **前提条件**:
  - 1号機 (192.168.100.1): 7x Tesla P100-PCIE-16GB, CUDA GPU 0 を使用
  - 2号機 (192.168.100.2): 4x Tesla P100-PCIE-16GB, rdma-server を実行
  - `GGML_RDMA_NO_GDR=1` (GPUDirect RDMA 無効)
  - テストモデル: qwen2.5-0.5b-instruct-q4_k_m.gguf (462.96 MiB)
  - レイヤー分割 (`-sm layer`), `-ngl 999` (全レイヤーをGPUにオフロード)

## 実装内容

### 1. 初期実装: Per-set_tensor RDMA Write + FLUSH_STAGING

各 set_tensor で:
1. RDMA Write でステージングバッファにデータ書き込み
2. FLUSH_STAGING コマンドを送信 (サーバーが cudaMemcpy 実行)
3. 応答を待つ

**問題**: 毎回のラウンドトリップ（Send/Recv + cudaMemcpy）のオーバーヘッドにより、per-call 平均で Send/Recv パスより遅い。

### 2. バッチフラッシュ (未マージ版)

set_tensor 時:
1. RDMA Write のみ実行 (FLUSH_STAGING は送らない)
2. dirty range を `g_pending_flushes` リストに記録

graph_compute 時:
1. 蓄積された全 dirty range を `FLUSH_ALL_STAGING` コマンドで一括送信
2. サーバーが各 range に対して cudaMemcpy を実行

**問題**: 278個のモデルウェイトテンソルに対して278回の個別 cudaMemcpy が graph_compute 時に発生。pp128 が 59.5 t/s まで低下。

### 3. バッチフラッシュ (マージ版) — 最終実装

クライアント側: 同一バッファの dirty range をバウンディングボックスでマージ
- `pending_flush_list` を `unordered_map<buffer_ptr, {min_offset, max_end}>` で管理
- 278個のテンソル → 数個のバッファ単位エントリにマージ

サーバー側: 同一バッファの range もマージ (二重マージ)
- バッファごとに1回の大きな `cudaMemcpy` のみ実行

## 再現方法

### 1. サーバー起動 (2号機)

```bash
ssh 192.168.100.2
cd /home/ubuntu/projects/llama.cpp
GGML_RDMA_NO_GDR=1 nohup build/bin/rdma-server --host 0.0.0.0 --port 50051 > /tmp/rdma-server.log 2>&1 &
```

### 2. RDMA Write + バッチフラッシュ ベンチマーク (1号機)

```bash
GGML_RDMA_NO_GDR=1 GGML_RDMA_PROFILE=1 CUDA_VISIBLE_DEVICES=0 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  timeout 120 build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32 2>/tmp/bench-merged-flush.log
```

### 3. Send/Recv ベースライン (2号機でサーバー再起動)

```bash
# 2号機: GGML_RDMA_NO_STAGING=1 でステージング無効化
GGML_RDMA_NO_GDR=1 GGML_RDMA_NO_STAGING=1 nohup build/bin/rdma-server --host 0.0.0.0 --port 50051 > /tmp/rdma-server.log 2>&1 &
```

```bash
# 1号機: 同じベンチマークコマンド
GGML_RDMA_NO_GDR=1 GGML_RDMA_PROFILE=1 CUDA_VISIBLE_DEVICES=0 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  timeout 120 build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32 2>/tmp/bench-sendrecv.log
```

### 4. ローカルのみベースライン

```bash
CUDA_VISIBLE_DEVICES=0 timeout 60 build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -r 1 -p 128 -n 32
```

## ベンチマーク結果

### スループット比較

| 構成 | pp128 (t/s) | tg32 (t/s) |
|------|:-----------:|:----------:|
| ローカルのみ (1 GPU) | 3390 | 214.1 |
| **RDMA Write + バッチフラッシュ (マージ)** | **3104** | **14.0** |
| Send/Recv | 2989 | 2.82 |
| RDMA Write + バッチフラッシュ (未マージ) | 59.5 | 3.71 |
| RDMA Write + Per-flush | 3060 | 0.54 |

### ローカル比性能

| 構成 | pp128 (対ローカル比) | tg32 (対ローカル比) |
|------|:-------------------:|:------------------:|
| **RDMA Write + バッチフラッシュ (マージ)** | **91.6%** | **6.5%** |
| Send/Recv | 88.2% | 1.3% |

### プロファイル詳細 (最終サマリー@140 graph_compute calls)

| 指標 | Send/Recv | Batch Flush (マージ) | 改善率 |
|------|-----------|---------------------|--------|
| set_tensor (total) | 69,535 ms | 184 ms | **378x** |
| set_tensor (avg/call) | 77.7 ms | 0.21 ms | **370x** |
| set_tensor スループット | 5.0 MB/s | 1,866 MB/s | **373x** |
| get_tensor (avg/call) | 441.7 us | 128.9 us | **3.4x** |
| get_tensor スループット | 1,376 MB/s | 4,713 MB/s | **3.4x** |
| graph_compute (total) | 10,437 ms | 7,979 ms | **1.3x** |
| graph_compute (avg) | 74.5 ms | 57.0 ms | **1.3x** |
| total RDMA time | 79,987 ms | 8,168 ms | **9.8x** |

## 分析

### 成功ポイント

1. **set_tensor の劇的高速化**: Send/Recv の 378 倍。RDMA Write (one-sided) はサーバー側の関与なしに完了するため、ラウンドトリップが不要。
2. **get_tensor も高速化**: RDMA Read (one-sided) により 3.4 倍の改善。
3. **graph_compute も改善**: フラッシュのマージにより、278個の個別 cudaMemcpy → 数個のバッファ単位コピーに削減。
4. **total RDMA time 9.8倍改善**: 全体的な RDMA 通信コストが大幅に削減。
5. **pp128 がローカルの 91.6%**: Prompt processing のオーバーヘッドがほぼ解消。

### 残課題

1. **tg32 の改善が限定的**: 14.0 t/s はローカル (214.1) の 6.5%。Token generation では各トークンで graph_compute (57ms) がボトルネック。RDMA Write+Flush 自体は高速だが、graph_compute のリモート実行コマンド送信→応答待ちのラウンドトリップ (~57ms) が支配的。
2. **graph_compute のラウンドトリップ**: tg32 改善には graph_compute の非同期化・パイプライン化が必要 (Step 3 タスク 3)。

### Thread Safety 修正

- `std::recursive_mutex op_mutex_` を `rdma_connection` に追加
- `send_rdma_cmd` / `send_rdma_cmd_with_rsp` の送受信シーケンスを保護
- `llama_params_fit` による並行RDMA操作でのクラッシュを解消

## まとめ

バッチフラッシュ最適化 (マージ版) により、RDMA Write パスの性能が大幅に改善された:
- **pp128**: Send/Recv比 **+3.8%** (2989 → 3104)、ローカル比 **91.6%**
- **tg32**: Send/Recv比 **5.0x** (2.82 → 14.0)
- **total RDMA time**: Send/Recv比 **9.8x 削減** (79987ms → 8168ms)

次のステップとして、tg32 のさらなる改善のため graph_compute の非同期化・パイプライン化 (タスク 3) を検討する。
