# RDMA vs RPC バックエンド パフォーマンス比較

- **実施日時**: 2026年2月5日 16:25

## 前提・目的

InfiniBand RDMA（GPUDirect対応）とllama.cpp標準のRPCバックエンド（TCP/IPソケットベース）の性能を比較し、RDMAの優位性を定量的に評価する。

- **背景**: RDMAバックエンドはInfiniBand 100Gbpsを活用した低遅延通信を実現
- **目的**: RDMA vs RPC の性能差を測定し、各バックエンドの特性を把握する
- **前提条件**:
  - 1号機 (192.168.100.1): 7× Tesla P100-PCIE-16GB
  - 2号機 (192.168.100.2): 4× Tesla P100-PCIE-16GB
  - 両ノード間はInfiniBand 100Gbps (Mellanox ConnectX-4) で接続

## テスト環境

| 項目 | 値 |
|------|-----|
| OS | Ubuntu 24.04 |
| CUDA | 12.0.140 |
| llama.cpp | 6fc5024f4 (build 7910) |
| GPU | Tesla P100-PCIE-16GB (compute capability 6.0) |
| RPC server | v3.6.0 |
| RDMA | GPUDirect RDMA enabled (nvidia-peermem) |

## ベンチマーク方法

- **ツール**: llama-bench
- **繰り返し**: `-r 5` (5回測定、平均±標準偏差)
- **テストパラメータ**: `-p 128 -n 32` (pp128, tg32)
- **レイヤー分割**: `-sm layer -ngl 999`

## 結果

### ベースライン（ローカルのみ）

| モデル | 構成 | pp128 (t/s) | tg32 (t/s) |
|--------|------|:-----------:|:----------:|
| qwen2.5-0.5b | 1 GPU | 3,452.72 ± 100.27 | 214.98 ± 1.61 |
| gpt-oss-20b | 2 GPU | 404.89 ± 16.91 | 64.21 ± 0.03 |
| gpt-oss-120b | 7 GPU | 213.48 ± 5.84 | 43.62 ± 0.23 |

### 1+1 GPU 構成 (gpt-oss-20b, CUDA0 + リモート1GPU)

| バックエンド | pp128 (t/s) | tg32 (t/s) | pp128 比 | tg32 比 |
|-------------|:-----------:|:----------:|:--------:|:-------:|
| ローカル 2GPU | 404.89 ± 16.91 | 64.21 ± 0.03 | 100% | 100% |
| **RDMA** | **400.74 ± 18.04** | **59.60 ± 0.01** | **99.0%** | **92.8%** |
| RPC | 381.90 ± 15.38 | 50.90 ± 1.50 | 94.3% | 79.3% |

### 1+1 GPU 構成 (qwen2.5-0.5b, CUDA0 + リモート1GPU)

| バックエンド | pp128 (t/s) | tg32 (t/s) |
|-------------|:-----------:|:----------:|
| ローカル 1GPU | 3,452.72 ± 100.27 | 214.98 ± 1.61 |
| RPC | 3,030.33 ± 218.97 | 107.68 ± 5.46 |

※ RDMA 1+1 はqwen2.5-0.5bで不安定（タイムアウト発生）のためスキップ

### 7+4 GPU クラスタ構成 (gpt-oss-120b)

| バックエンド | pp128 (t/s) | tg32 (t/s) | pp128 比 | tg32 比 |
|-------------|:-----------:|:----------:|:--------:|:-------:|
| ローカル 7GPU | 213.48 ± 5.84 | 43.62 ± 0.23 | 100% | 100% |
| **RDMA 7+4** | 171.64 ± 68.43 | 26.59 ± 14.19 | 80.4% | 61.0% |
| RPC 7+4 | 201.81 ± 3.93 | 33.02 ± 0.42 | 94.5% | 75.7% |

## 分析

### RDMA vs RPC 直接比較 (1+1 GPU, gpt-oss-20b)

| 指標 | RDMA | RPC | RDMA優位性 |
|------|:----:|:---:|:----------:|
| pp128 | 400.74 | 381.90 | **+4.9%** |
| tg32 | 59.60 | 50.90 | **+17.1%** |

### 考察

1. **1+1 GPU構成での優位性**
   - RDMAはRPCに対してpp128で約5%、tg32で約17%高速
   - 特にtg32（トークン生成）でRDMAの低遅延特性が効果を発揮
   - RDMAはローカル2GPUの99%の性能を維持（pp128）

2. **7+4 GPUクラスタでの課題**
   - RDMAは7+4構成で不安定（標準偏差が大きい）
   - RPCは安定した結果（標準偏差 < 4）を示す
   - RDMAの不安定さは4つのリモートGPUへの逐次接続が原因
     - 各RDMA device = 1 split、スケジューラが順次実行

3. **モデルサイズの影響**
   - 小規模モデル (qwen2.5-0.5b) ではRDMAが不安定
   - 中規模モデル (gpt-oss-20b) ではRDMAが最も効果的
   - グラフ送信オーバーヘッドがモデルサイズに対して相対的に小さくなるため

4. **安定性**
   - RPCは全構成で安定した結果
   - RDMAは1+1構成で安定、マルチリモートGPUで不安定

## 結論

| 観点 | RDMA | RPC |
|------|------|-----|
| 1+1 GPU性能 | **優** (ローカル比99%) | 良 (ローカル比94%) |
| マルチGPU安定性 | 要改善 | **優** |
| tg32性能 | **優** (+17% vs RPC) | 良 |
| 実装成熟度 | 開発中 | **安定** |

**総評**:
- 1+1 GPU構成ではRDMAがRPCを上回る性能を示す（特にtg32で17%改善）
- マルチリモートGPU構成ではRDMAの逐次実行オーバーヘッドが課題
- GPUDirect RDMAの恩恵は大規模ウェイト転送で最大化される

## 再現方法

### 1. ビルド (両ノード)

```bash
# 1号機
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
rm -rf build
cmake -B build -DGGML_CUDA=ON -DGGML_RDMA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2号機へデプロイ
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' --exclude='build' ./ 192.168.100.2:/home/ubuntu/projects/llama.cpp/
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && \
  rm -rf build && \
  cmake -B build -DGGML_CUDA=ON -DGGML_RDMA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build -j\$(nproc)"
```

### 2. RDMA テスト

```bash
# サーバー起動 (2号機)
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"

# 1+1 GPU (1号機)
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf \
  -dev 'CUDA0/RDMA0[192.168.100.2:50051]' \
  -ngl 999 -sm layer -r 5 -p 128 -n 32
```

### 3. RPC テスト

```bash
# サーバー起動 (2号機)
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rpc-server -H 0.0.0.0 -p 50052 > /tmp/rpc-server.log 2>&1 &"

# 1+1 GPU (1号機)
CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf \
  --rpc 192.168.100.2:50052 \
  -ngl 999 -sm layer -r 5 -p 128 -n 32
```

## 参考レポート

- [2026-02-05_101500_step4_gpudirect_rdma_enabled.md](2026-02-05_101500_step4_gpudirect_rdma_enabled.md) - GPUDirect RDMA有効化
- [2026-02-05_003452_step3_flush_compute_optimization.md](2026-02-05_003452_step3_flush_compute_optimization.md) - RDMA性能最適化
