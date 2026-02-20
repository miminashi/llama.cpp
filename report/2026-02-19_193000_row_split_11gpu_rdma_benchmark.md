# Row Split vs Layer Split 11GPU RDMA ベンチマーク

- **実施日時**: 2026年2月19日 19:30
- **ワークツリー**: `.worktree/rdma-row-split`
- **関連レポート**: [report/2026-02-19_163700_row_vs_layer_split_benchmark.md](2026-02-19_163700_row_vs_layer_split_benchmark.md) (7GPU ローカルベンチマーク)
- **関連レポート**: [report/2026-02-19_200000_row_split_fit_params_fix.md](2026-02-19_200000_row_split_fit_params_fix.md) (llama_params_fit 修正)

## 前提・目的

7GPU ローカルでの row split vs layer split ベンチマーク (前回レポート) に続き、11GPU (7 CUDA + 4 RDMA) 構成での同比較を実施する。

- **背景**: 前回レポートで 7 GPU ローカルでは row split が layer split に対して大幅に劣ることを確認 (pp512: -39%, tg: -34%)。11GPU マルチノード構成でも同様の傾向か検証する。
- **目的**: 11GPU (7 CUDA + 4 RDMA) 構成で row split と layer split の性能を比較し、マルチノードでの最適分割方式を確認する。
- **予測**: RDMA バックエンドは `split_buffer_type` 未実装のため、row split 指定時もリモートデバイス (RDMA0-3) は実質 layer split にフォールバックする。ローカル CUDA デバイス (0-6) のみが実際に row split される。結果として layer split より劣る。

### 事前知見: RPC/RDMA の split_buffer_type 未対応

`make_gpu_buft_list` は `ggml_backend_split_buffer_type` が NULL の場合、サイレントにデフォルトバッファにフォールバックする。つまり:
- `-sm row` 指定時、ローカル CUDA デバイスのみ row split
- リモート RDMA デバイスは通常バッファ (実質 layer split)
- エラーや警告は出ない

## 再現方法

### 1. rdma-row-split ワークツリーのビルド・デプロイ

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-row-split
rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DGGML_RPC=ON \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j $(nproc)

bash scripts/rdma-deploy.sh
```

### 2. RDMA サーバー起動

```bash
bash scripts/rdma-server.sh start
```

### 3. ベンチマーク実行

```bash
# Layer split
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  gpu-lock.sh run build/bin/llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm layer -p 128,512 -n 32 -r 3

# Row split
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  gpu-lock.sh run build/bin/llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm row -p 128,512 -n 32 -r 3
```

## 結果

### GLM-4.7 IQ2_M, 11 GPU (7 CUDA + 4 RDMA)

| 分割方式 | pp128 (t/s) | pp512 (t/s) | tg32 (t/s) |
|---------|:-----------:|:-----------:|:----------:|
| **layer split** | **23.70 ± 0.40** | **39.02 ± 0.31** | **7.70 ± 0.03** |
| row split | 21.75 ± 0.23 | 28.23 ± 0.07 | 6.70 ± 0.00 |
| **差分 (row - layer)** | **-8.2%** | **-27.7%** | **-13.0%** |

### 確認データ (2nd round)

| 分割方式 | pp128 (t/s) | tg32 (t/s) |
|---------|:-----------:|:----------:|
| **layer split** (2nd) | **23.64 ± 0.46** | **7.70 ± 0.03** |
| row split (2nd) | 21.70 ± 0.22 | 6.70 ± 0.00 |

結果は安定しており、2nd round でもほぼ同一の値を示した。

### 7GPU ローカル比較 (前回レポートより, gpt-oss-20b Q4_K_M)

| 分割方式 | pp512 差分 | tg 差分 |
|---------|:----------:|:------:|
| 7 GPU ローカル | -39% | -34% |
| **11 GPU RDMA** | **-28%** | **-13%** |

11GPU 構成では row split の劣位が 7GPU ローカルと比べて縮小している。これはリモート RDMA デバイスが `split_buffer_type` 未対応で layer split にフォールバックするため、「純粋な row split」ではなく「ハイブリッド」（ローカル: row, リモート: layer）になることが要因。

## 分析

### Row split が劣る理由

1. **ローカル GPU 間の PCIe AllReduce コスト**: P100 は NVLink なし。row split では各レイヤー計算後に GPU 間で All-Reduce が必要で、PCIe バスがボトルネックになる。
2. **リモートデバイスのフォールバック**: RDMA バックエンドは `split_buffer_type` を実装していないため、リモートデバイスは実質 layer split 動作。ローカル 7 GPU のみが row split の影響を受ける。
3. **特に pp512 で差が大きい** (-28%): バッチサイズが大きいほど、AllReduce のデータ量が増え、PCIe ボトルネックが顕著になる。

### 結論

**11GPU RDMA 構成でも layer split が row split に対して全条件で優位**:
- pp128: +8.2%
- pp512: +27.7%
- tg32: +13.0%

前回の 7GPU ローカルベンチマークと合わせて、P100 (NVLink なし) 環境では **layer split が最適な分割方式** であることが再確認された。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `6a096a20d (feature/rdma-row-split)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 36°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 40915) |
