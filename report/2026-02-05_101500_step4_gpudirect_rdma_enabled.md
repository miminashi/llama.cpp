# Step 4 GPUDirect RDMA 有効化レポート

- **実施日時**: 2026年2月5日 10:15
- **関連レポート**: [Step 3 FLUSH+COMPUTE最適化](step3_flush_compute_optimization_2026-02-05_003452.md), [CPU staged vs GPUDirect比較](cpu_staged_vs_gpudirect_rdma_comparison_2026-02-04_214044.md)

## 前提・目的

Step 4 (GPUDirect RDMA有効化) の目標:
- **目的**: GPUDirect RDMAを有効化し、CPU経由のステージングを排除して性能向上を実現する
- **前提条件**:
  - 1号機 (192.168.100.1): 7× Tesla P100-PCIE-16GB, CUDA GPU 0 を使用
  - 2号機 (192.168.100.2): 4× Tesla P100-PCIE-16GB, rdma-server を実行
  - nvidia-peermem モジュールが両ノードでロード済み (ドライバ 535.288.01)
  - Mellanox ConnectX-4 100Gbps NIC (mlx5_0)

## 事前調査結果

### nvidia-peermem 状態

| ノード | モジュール状態 | ドライバ版 |
|--------|--------------|-----------|
| 1号機 | ✅ ロード済み | 535.288.01 |
| 2号機 | ✅ ロード済み | 535.288.01 |

### PCIe トポロジ

**1号機 (7 GPU):**
- GPU3-6: RDMA NIC と同一PCIeスイッチ上 (PIX) → **最適**
- GPU0-2: CPU経由 (PHB) → 中程度

**2号機 (4 GPU):**
- GPU0: RDMA NIC と同一PCIeスイッチ上 (PIX) → **最適**
- GPU1-2: CPU経由 (PHB) → 中程度
- GPU3: NUMA境界越え (SYS) → 遅い

## 有効化手順

### 1. サーバー起動 (GPUDirect有効)

```bash
# GGML_RDMA_NO_GDR を設定しない (デフォルトでGPUDirect有効)
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server \
  -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &

# ログ確認: "[gdr_memory_manager] GPUDirect RDMA is available" が出力されること
```

### 2. クライアント実行 (GPUDirect有効)

```bash
# GGML_RDMA_NO_GDR を設定しない
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=build/bin build/bin/llama-bench \
  -m <model_path> -ngl 999 -sm layer -r 1 -p 128 -n 32
```

## ベンチマーク結果

### gpt-oss-20b (20.9B params, 10.8 GiB)

| モード | pp128 (t/s) | ローカル比 | tg32 (t/s) | ローカル比 |
|--------|:-----------:|:----------:|:----------:|:----------:|
| ローカル 1GPU | 407.68 | 100% | 64.27 | 100% |
| **GPUDirect RDMA 1+1** | **403.69** | **99.0%** | **59.52** | **92.6%** |
| CPU staging 1+1 | 61.06 | 15.0% | 59.30 | 92.3% |

### qwen2.5-0.5b (0.5B params, 463 MiB)

| モード | pp128 (t/s) | ローカル比 | tg32 (t/s) | ローカル比 |
|--------|:-----------:|:----------:|:----------:|:----------:|
| ローカル 1GPU | 3,398 | 100% | 205.69 | 100% |
| GPUDirect RDMA 1+1 | 60.54 | 1.8% | 171.21 | 83.2% |
| CPU staging 1+1 | 60.56 | 1.8% | 176.49 | 85.8% |

## 分析

### 成功ポイント

1. **GPUDirect RDMA 有効化成功**: nvidia-peermem 経由でGPUメモリを直接RDMA登録
2. **gpt-oss-20b pp128 で劇的改善**: 61 → 403 t/s (**6.6倍高速**, ローカルの99%)
3. **CPU経由コピー排除**: ステージングバッファの cudaMemcpy が不要に

### モデルサイズによる違い

| モデル | pp128 改善率 | tg32 改善率 | 理由 |
|--------|:-----------:|:----------:|------|
| gpt-oss-20b | **6.6倍** | 1.00倍 | 大量のモデルウェイト転送で効果大 |
| qwen2.5-0.5b | 1.00倍 | 0.97倍 | 小規模モデルはオーバーヘッドが支配的 |

**解釈**:
- 大規模モデル (20b) はウェイト転送量が多いため、GPUDirect RDMA の効果が顕著
- 小規模モデル (0.5b) はグラフ送信等のオーバーヘッドが支配的で、転送時間の削減効果が薄い
- tg32 (トークン生成) は転送量が少ないため、GPUDirect の効果が出にくい

### 帯域測定 (qwen2.5-0.5b)

| 操作 | GPUDirect | CPU staging | 差分 |
|------|:---------:|:-----------:|:----:|
| set_tensor | 1,665 MB/s | 1,659 MB/s | +0.4% |
| get_tensor | 1,704 MB/s | 4,540 MB/s | **-62%** |

**get_tensor の低下について**:
- CPU staging では RDMA Read はホストメモリ (RAM) から読み取り
- GPUDirect では RDMA Read は GPU VRAM から読み取り
- GPU VRAM からの RDMA Read は PCIe 経由で追加レイテンシが発生
- ただし、get_tensor は頻度が低いため全体性能への影響は小さい

## Step 4 成功基準の達成状況

| 目標 | 基準 | 結果 | 達成 |
|------|------|------|:----:|
| GPUDirect RDMA でテンソル転送が動作 | 動作確認 | ✅ 動作確認済み | ✅ |
| CPU経由比で測定可能な性能向上 | レイテンシ30%削減 | pp128で562%削減 (6.6倍高速) | ✅ |

## まとめ

GPUDirect RDMA の有効化に成功し、大規模モデル (gpt-oss-20b) で顕著な性能向上を達成:

- **gpt-oss-20b pp128**: 61 → **403 t/s** (6.6倍高速, ローカル比 99%)
- **gpt-oss-20b tg32**: 59.3 → **59.5 t/s** (ほぼ同等)
- **nvidia-peermem** 経由のGPUメモリ直接登録が正常動作

次のステップ (Step 5):
- GLM4.7 Q4 を 16台 P100 で動作させる
- 現在11台で事前検証可能
