# `-sm row` vs `-sm layer` ベンチマーク実験レポート

- **実施日時**: 2026年2月19日 16:37
- **ワークツリー**: `.worktree/rdma-backend`

## 前提・目的

### 背景

以前、別サーバ (P100×4) で `-sm row` (テンソル並列) は NVLink なしの環境では性能が出ないと結論づけた。
CLAUDE.md でも「レイヤー分割 (`-sm layer`) のみ — P100 には NVLink がなく、row split では性能が出ない」と記載されている。

### 目的

1号機 (P100 PCIe×7) で row split と layer split の性能差を実データで確認し、既存の結論を検証する。

### 前提条件

- **GPU**: 1号機 (192.168.100.1) の Tesla P100 PCIe 16GB × 7
- **P2P トポロジ**: GPU0-2 = PIX (同一 PCIe スイッチ)、GPU3-6 = PIX、GPU0-2 ↔ GPU3-6 = PHB
- **NVLink**: なし
- **RDMA**: 使用しない (ローカル CUDA のみ)
- **ビルド**: commit `1aa0643b0` (build 7939)

## 再現方法

### ビルド

```bash
bash scripts/rdma-build.sh local
```

### gpt-oss-20b ベンチマーク (2/4/7 GPU × row/layer)

```bash
# 例: 2GPU, row split
bash scripts/gpu-lock.sh run env CUDA_VISIBLE_DEVICES=0,1 \
  build/bin/llama-bench -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm row -p 128,512 -n 32 -r 3

# 例: 2GPU, layer split
bash scripts/gpu-lock.sh run env CUDA_VISIBLE_DEVICES=0,1 \
  build/bin/llama-bench -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -p 128,512 -n 32 -r 3
```

GPU 数の組み合わせ:
- 2 GPU: `CUDA_VISIBLE_DEVICES=0,1` (同一 PIX グループ)
- 4 GPU: `CUDA_VISIBLE_DEVICES=0,1,2,3` (PIX + PHB 混在)
- 7 GPU: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6` (全 GPU)

### GLM-4.7 IQ2_M ベンチマーク (7 GPU)

```bash
bash scripts/gpu-lock.sh run build/bin/llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm row -p 128 -n 32 -r 1
```

## 結果

### gpt-oss-20b Q4_K_M (Dense モデル, ~11 GB)

| GPU 数 | Split Mode | pp128 (t/s) | pp512 (t/s) | tg32 (t/s) |
|:------:|:----------:|------------:|------------:|-----------:|
| 2 | **row** | 375.81 ± 12.06 | 584.11 ± 4.23 | 56.15 ± 0.05 |
| 2 | **layer** | 394.63 ± 13.56 | 685.31 ± 4.92 | 64.06 ± 0.08 |
| 4 | **row** | 346.32 ± 10.23 | 520.09 ± 2.17 | 52.26 ± 0.03 |
| 4 | **layer** | 393.60 ± 13.31 | 682.93 ± 3.24 | 63.93 ± 0.02 |
| 7 | **row** | 297.98 ± 5.45 | 413.99 ± 1.54 | 42.17 ± 0.34 |
| 7 | **layer** | 389.25 ± 14.49 | 679.57 ± 3.44 | 63.50 ± 0.06 |

### layer 対比での row split の性能比率

| GPU 数 | pp128 | pp512 | tg32 |
|:------:|------:|------:|-----:|
| 2 | **95.2%** | **85.2%** | **87.7%** |
| 4 | **88.0%** | **76.2%** | **81.8%** |
| 7 | **76.6%** | **60.9%** | **66.4%** |

### GPU スケーリング (GPU 数増加に対する性能変化)

#### row split のスケーリング (2GPU 比)

| GPU 数 | pp128 | pp512 | tg32 |
|:------:|------:|------:|-----:|
| 2 | 100% | 100% | 100% |
| 4 | 92.2% | 89.0% | 93.1% |
| 7 | 79.3% | 70.9% | 75.1% |

#### layer split のスケーリング (2GPU 比)

| GPU 数 | pp128 | pp512 | tg32 |
|:------:|------:|------:|-----:|
| 2 | 100% | 100% | 100% |
| 4 | 99.7% | 99.7% | 99.8% |
| 7 | 98.6% | 99.2% | 99.1% |

### GLM-4.7 IQ2_M (MoE モデル, ~114 GB)

| GPU 数 | Split Mode | 結果 |
|:------:|:----------:|:----:|
| 7 | row | **ロード失敗** |
| 7 | layer | **ロード失敗** |

**原因**: モデルサイズ (~114 GB) が 7 GPU の合計 VRAM (7 × 16 GB = 112 GB) を超過。
row/layer の問題ではなく、純粋な VRAM 不足。11 GPU (7C+4R = 176 GB) では layer split で動作実績あり (pp=6.4, tg=6.8 t/s)。

## 分析

### 1. layer split は全条件で row split に勝る

gpt-oss-20b では、全 GPU 数・全テストタイプで **layer split が row split を上回った**。
差は GPU 数が増えるほど拡大し、7 GPU の pp512 では layer が row の **1.64 倍** の性能。

### 2. row split は GPU 数増加で性能が劣化する

- **row split**: 7 GPU で pp512 が 2 GPU 比 **70.9%** に低下 (負のスケーリング)
- **layer split**: 7 GPU で pp512 が 2 GPU 比 **99.2%** を維持 (ほぼ理想的)

row split の劣化原因:
- P100 には NVLink がなく、GPU 間通信は PCIe 経由 (`cudaMemcpyPeer`)
- row split は各レイヤーの計算後に AllReduce (全 GPU 間の結果集約) が必要
- PCIe 帯域幅がボトルネックとなり、GPU 数が増えるほど AllReduce コストが増大
- AllReduce の通信量はテンソルサイズに比例し、pp512 (長いシーケンス) で顕著

### 3. layer split は GPU 数によらず安定

layer split ではデバイス間通信は隣接レイヤー間のアクティベーション転送のみ:
- 通信量が少ない (1 レイヤー分のアクティベーションのみ)
- GPU 数を増やしても通信パターンが変わらない (パイプライン型)
- generation (tg) もほぼ一定 (64.06 → 63.50 t/s)

### 4. GLM-4.7 は 7 ローカル GPU では VRAM 不足

GLM-4.7 IQ2_M のモデルファイルサイズ:
- Part 1: 46.3 GB
- Part 2: 46.4 GB
- Part 3: 21.4 GB
- **合計: ~114 GB**

7 × 16 GB = 112 GB の VRAM では KV キャッシュ等のオーバーヘッドを加味すると明らかに不足。
このモデルの row/layer 比較は 8 GPU 以上の環境が必要。

## 結論

**「P100 (NVLink なし) では row split は性能が出ない」という既存の結論は、1号機 (P100×7) でも完全に成立する。**

- layer split は全条件で優位
- row split は GPU 数が増えるほど性能が劣化 (負のスケーリング)
- layer split は GPU 数に対してほぼ理想的なスケーリングを維持
- RDMA バックエンドで `-sm layer` のみをサポートする方針は妥当

### 理論予測との比較

| GPU 数 | 予測 | 実測結果 |
|:------:|------|---------|
| 2 (TP=2) | row が layer より速い可能性 | **layer が優位** (pp512: +17.3%, tg: +14.1%) |
| 4 (TP=4) | row ≈ layer | **layer が大幅に優位** (pp512: +31.3%, tg: +22.3%) |
| 7 (TP=7) | row < layer | **layer が圧倒的に優位** (pp512: +64.2%, tg: +50.6%) |

理論予測は AllReduce コストを過小評価していた。PCIe バスの実効帯域幅制約と `cudaMemcpyPeer` のオーバーヘッドにより、2 GPU でも row split は layer split に及ばない。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `184b99ba2 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 32°C | 33°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 17159) |
