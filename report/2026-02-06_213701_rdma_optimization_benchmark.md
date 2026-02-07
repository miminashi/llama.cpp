# RDMA バックエンド性能最適化ベンチマーク

- **実施日時**: 2026年2月6日 21:37

## 前提・目的

Step 4 (GPUDirect RDMA) 完了後、5つの個別最適化手法の効果を定量的に評価する。

- **背景**: RDMAバックエンドの通信パスには複数の最適化余地（バッファ管理、CQポーリング、メッセージ統合等）がある
- **目的**: 各最適化を個別にベンチマークし、効果の有無と採用判断の根拠を得る
- **前提条件**:
  - 2ノード構成 (1号機 CUDA0 + 2号機 RDMA0)、GPUDirect RDMA 有効
  - Tesla P100-PCIE-16GB × 2
  - InfiniBand 100Gbps 接続
  - 各ワークツリーは `rdma-backend` ブランチからの差分として実装済み

### 参考レポート

- [Step 4 GPUDirect RDMA 有効化](2026-02-05_101500_step4_gpudirect_rdma_enabled.md)
- [RDMA vs RPC ベンチマーク v2](2026-02-06_000500_rdma_vs_rpc_benchmark_v2.md)

## 最適化手法一覧

| # | ワークツリー | 最適化内容 | 変更規模 |
|---|-------------|-----------|---------|
| 1 | `opt-prepost-recv` | サーバー側 recv バッファ事前確保 | 6行 |
| 2 | `opt-prealloc-buf` | クライアント側バッファ事前確保 | 24行 |
| 3 | `opt-cq-backoff` | CQポーリング適応的バックオフ | 14行 |
| 4 | `opt-combined-send` | ヘッダ+データ統合送信 (IB Send/Recv削減) | 101行 |
| 5 | `opt-all` | #1 + #2 のバッファ最適化統合 | 30行 |

## ベンチマーク条件

- **計測ツール**: `llama-bench`
- **パラメータ**: pp512, tg128, tg256 (各3回反復)
- **GPU構成**: CUDA0 + RDMA0 (1+1)、GPUDirect RDMA 有効 (`GGML_RDMA_NO_GDR` 未設定)
- **ビルド**: 全ワークツリーで同一ベースコミット `b2e86b6af` (7913)

## 結果

### gpt-oss-20b Q4_K_M (10.81 GiB)

| 最適化 | pp512 (t/s) | vs BL | tg128 (t/s) | vs BL | tg256 (t/s) | vs BL |
|--------|:-----------:|:-----:|:-----------:|:-----:|:-----------:|:-----:|
| **Baseline** | **651.51 ± 1.19** | — | **51.74 ± 2.24** | — | **50.12 ± 0.09** | — |
| #1 prepost-recv | 649.83 ± 3.12 | -0.3% | 52.51 ± 1.77 | +1.5% | 51.19 ± 0.02 | +2.1% |
| #2 prealloc-buf | 649.58 ± 0.85 | -0.3% | 52.39 ± 1.84 | +1.3% | 50.01 ± 0.02 | -0.2% |
| #3 cq-backoff | 641.13 ± 1.54 | **-1.6%** | 37.06 ± 0.76 | **-28.4%** | 36.34 ± 0.45 | **-27.5%** |
| #4 combined-send | 644.46 ± 1.56 | **-1.1%** | 35.41 ± 0.54 | **-31.6%** | 34.58 ± 0.05 | **-31.0%** |
| #5 all (1+2) | 649.26 ± 1.07 | -0.3% | 52.38 ± 1.67 | +1.2% | 51.42 ± 0.01 | +2.6% |

### qwen2.5-0.5b Q4_K_M (462.96 MiB)

| 最適化 | pp512 (t/s) | vs BL | tg128 (t/s) | vs BL | tg256 (t/s) | vs BL |
|--------|:-----------:|:-----:|:-----------:|:-----:|:-----------:|:-----:|
| **Baseline** | **5540.31 ± 58.11** | — | **144.70 ± 0.24** | — | **144.80 ± 0.05** | — |
| #1 prepost-recv | 5501.41 ± 20.60 | -0.7% | 136.83 ± 0.18 | **-5.4%** | 144.42 ± 0.19 | -0.3% |
| #2 prealloc-buf | 5551.52 ± 16.36 | +0.2% | 146.46 ± 0.31 | +1.2% | 144.10 ± 0.15 | -0.5% |
| #3 cq-backoff | 5078.90 ± 84.91 | **-8.3%** | 71.11 ± 0.97 | **-50.9%** | 73.01 ± 4.20 | **-49.6%** |
| #4 combined-send | 5130.77 ± 33.54 | **-7.4%** | 62.66 ± 0.02 | **-56.7%** | 62.86 ± 0.02 | **-56.6%** |
| #5 all (1+2) | 5542.21 ± 38.09 | +0.0% | 145.67 ± 0.29 | +0.7% | 146.12 ± 0.18 | +0.9% |

## 分析

### 効果なし / 微小改善 (#1 prepost-recv, #2 prealloc-buf, #5 all)

バッファ事前確保系の最適化は、gpt-oss-20b tg256で最大+2.6%の改善が見られるが、測定誤差の範囲内。GPUDirect RDMA有効時はバッファ確保/解放コストがボトルネックではなく、GPU計算時間とRDMA転送時間が支配的であるため。

### 大幅な性能低下 (#3 cq-backoff)

CQポーリングの適応的バックオフは全テストで性能低下。特に小モデル (qwen2.5-0.5b) では tg で **-50%** と壊滅的。原因:
- バックオフによるスリープがレイテンシクリティカルパスに挿入される
- RDMA通信はμsオーダーで完了するため、バックオフの最小スリープ時間でも過大な待ち時間となる
- CPU使用率削減のメリットよりレイテンシ増加のデメリットが遥かに大きい

### 大幅な性能低下 (#4 combined-send)

ヘッダ+データ統合送信も全テストで性能低下。特に tg で **-31%〜-57%**。原因:
- IB Send/Recv の recv バッファサイズを統合メッセージに合わせて拡大する必要があり、バッファ管理オーバーヘッドが増加
- CLAUDE.md にも記載の通り「combined send 最適化は recv バッファオーバーヘッドで逆効果」という過去の知見と一致
- Send/Recv の回数削減効果 < recv バッファ拡大のオーバーヘッド

## 結論

| 最適化 | 採用推奨 | 理由 |
|--------|:--------:|------|
| #1 prepost-recv | △ | 効果は誤差範囲だが、害もない。コード量が少ないため採用してもよい |
| #2 prealloc-buf | △ | 同上 |
| #3 cq-backoff | ✗ | 全シナリオで性能低下。RDMA環境ではビジーポーリングが最適 |
| #4 combined-send | ✗ | 全シナリオで性能低下。recvバッファオーバーヘッドが支配的 |
| #5 all (1+2) | △ | #1+#2の統合。微小改善の可能性あり (tg256で+2.6%) |

**総合評価**: 現時点のRDMAバックエンドは既にGPUDirect RDMAによりほぼ最適化されており、通信パス上のバッファ管理最適化による追加利得は限定的。ボトルネックはGPU計算時間とPCIe帯域幅にある。さらなる性能改善にはスケジューラレベルの最適化（複数RDMAデバイスの並列実行等）が必要。

## 再現方法

### 共通準備

```bash
# 各ワークツリーのバイナリは事前にビルド済みと仮定
# WT=rdma-backend | opt-prepost-recv | opt-prealloc-buf | opt-cq-backoff | opt-combined-send | opt-all
WT=rdma-backend
```

### 2号機へのデプロイ・ビルド

```bash
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' /home/ubuntu/projects/llama.cpp/.worktree/$WT/ 192.168.100.2:/home/ubuntu/projects/llama.cpp/
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && rm -rf build && \
  cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build -j\$(nproc)"
```

### rdma-server 起動 (GPUDirect RDMA 有効)

```bash
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
```

### ベンチマーク実行

```bash
# gpt-oss-20b: pp512 + tg128
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  /home/ubuntu/projects/llama.cpp/.worktree/$WT/build/bin/llama-bench \
  --model /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 3 -p 512 -n 128

# gpt-oss-20b: tg256
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  /home/ubuntu/projects/llama.cpp/.worktree/$WT/build/bin/llama-bench \
  --model /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 3 -p 0 -n 256

# qwen2.5-0.5b: pp512 + tg128
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  /home/ubuntu/projects/llama.cpp/.worktree/$WT/build/bin/llama-bench \
  --model /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 3 -p 512 -n 128

# qwen2.5-0.5b: tg256
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  /home/ubuntu/projects/llama.cpp/.worktree/$WT/build/bin/llama-bench \
  --model /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 3 -p 0 -n 256
```

### rdma-server 停止

```bash
ssh 192.168.100.2 "killall rdma-server"
```
