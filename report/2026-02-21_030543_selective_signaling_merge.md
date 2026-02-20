# Selective Signaling マージ・最適化到達点レポート

- **実施日時**: 2026年2月21日 03:05
- **ブランチ**: `feature/rdma-backend` (コミット: `20e808adb`)
- **作業場所**: メインリポジトリ（ワークツリーなし、直接 cherry-pick）

## 前提・目的

Step 3-5 を通じて 15 以上の最適化施策を試行・検証してきた結果、通信レイヤーの最適化は実質的な上限に到達した。本レポートでは、最も効果が大きかった **Selective Signaling** を `feature/rdma-backend` にマージし、現時点での最適化到達点を総括する。

### 背景

- RDMA バックエンドの通信最適化として複数の施策を実装・ベンチマークしてきた
- GLM-4.7 IQ2_M 11GPU (7CUDA + 4RDMA) での tg32 ≈ 128ms/token のうち、GPU 計算が 93.8% (~120ms)、通信オーバーヘッドが 6.2% (~8ms)
- 通信層のさらなる最適化余地は限られており、次の大きな改善はハードウェア (16GPU) またはアーキテクチャ変更が必要

### 目的

1. Selective Signaling を `feature/rdma-backend` にマージし、ベンチマークで効果を確認する
2. これまでの最適化施策の結果を総括し、到達点を明確にする

### 参照レポート

- [Selective Signaling A/B ベンチマーク](2026-02-20_132916_merge_validated_improvements_benchmark.md)
- [Server Pipeline ベンチマーク](2026-02-20_215838_server_pipeline_benchmark.md)
- [RDMA Backend Retrospective v5](2026-02-19_163700_row_vs_layer_split_benchmark.md)

## 再現方法

### 1. Cherry-pick

```bash
cd /home/ubuntu/projects/llama.cpp
git checkout feature/rdma-backend
git cherry-pick 789b88ace
```

元コミット: `789b88ace` (feat: add selective signaling for chunked RDMA Write and Send)
- `merge/validated-improvements` ブランチで事前にコンフリクトなしを検証済み

### 2. ビルド・デプロイ

```bash
bash scripts/rdma-build.sh local
bash scripts/rdma-deploy.sh
bash scripts/rdma-server.sh restart
```

### 3. スモークテスト (1CUDA + 1RDMA)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

### 4. GLM-4.7 確認ベンチマーク (11GPU)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  build/bin/llama-bench -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm layer -r 3 -p 128 -n 32 -o csv
```

## 結果

### マージ結果

- Cherry-pick はコンフリクトなしで適用 (`20e808adb`)
- 変更ファイル: `ggml-rdma.cpp` (クライアント側 RDMA Write), `rdma-transport.cpp` (IB Send)
- スモークテスト成功 (qwen2.5-0.5b, 1CUDA+1RDMA): pp128 = 3085 t/s, tg32 = 137 t/s

### GLM-4.7 IQ2_M 11GPU ベンチマーク

| テスト | 結果 (t/s) | ±SD | 期待値 | 判定 |
|--------|:----------:|:---:|:------:|:----:|
| **pp128** | **30.15** | 0.49 | ≈ 29 | 期待値超え |
| **tg32** | **7.75** | 0.004 | ≈ 7.7 | 期待通り |

pp128 が前回の A/B ベンチマーク (29.44 t/s) をさらに +2.4% 上回っている。これはセッション間の GPU サーマル変動の範囲内と考えられる。

### Baseline 比較

| 構成 | pp128 (t/s) | vs Baseline | tg32 (t/s) | vs Baseline |
|------|:-----------:|:-----------:|:----------:|:-----------:|
| **マージ後 (今回)** | **30.15** | **+28.8%** | **7.75** | **+0.0%** |
| A/B ベンチマーク時 | 29.44 | +26.1% | 7.78 | +0.4% |
| Baseline (SS なし) | 23.41 | — | 7.75 | — |

### RPC (TCP) 比較

| バックエンド | pp128 (t/s) | tg32 (t/s) |
|-------------|:-----------:|:----------:|
| **RDMA (マージ後)** | **30.15** | **7.75** |
| RPC (TCP) | 5.4 | 7.5 |
| **RDMA / RPC** | **+458%** | **+3.3%** |

## 最適化施策の総括

### マージ済み施策

| 施策 | pp128 効果 | tg32 効果 | コミット |
|------|:----------:|:---------:|:--------:|
| **Selective signaling** | **+26-29%** | ≈ 0% | `20e808adb` |
| Deferred copy | ≈ 0% | +1.45% | (既存) |
| GDR budget | — | +13% (GDR有効化) | (既存) |

### Opt-in 維持の施策

| 施策 | pp128 効果 | tg32 効果 | 有効化方法 |
|------|:----------:|:---------:|:----------:|
| Per-device + server-push | -1.35% | +1.76% | `GGML_RDMA_PER_DEVICE_CONN=1` |

### マージしなかった施策

| 施策 | pp128 効果 | tg32 効果 | 理由 |
|------|:----------:|:---------:|:-----|
| Server-side pipeline | -0.21% (有意) | +0.55% (非有意) | コード複雑性に見合わない |

### ボトルネック分析 (tg32, ~128ms/token)

```
GPU 計算:              ~120ms (93.8%)  ← ハードウェア限界 (P100)
通信オーバーヘッド:     ~8ms   (6.2%)  ← 大幅に最適化済み
  うちプロトコル残余:   ~0.5-1ms       ← これ以上の削減は困難
```

通信最適化の理論的上限は全体の ~6% であり、そのうち実際に到達可能な改善は一部に限られる。現時点で実用的な通信最適化は完了したと判断できる。

## 結論

- **Selective Signaling のマージに成功**。pp128 +29% の改善を `feature/rdma-backend` に統合
- RDMA バックエンドは RPC (TCP) 比で pp128 +458%、tg32 +3.3% と両メトリクスで上回る
- **通信レイヤーの最適化は実質的な上限に到達**。次の改善は:
  1. **16GPU 拡張** — ハードウェア追加によるスケーリング
  2. **upstream llama.cpp 変更** — 計算グラフ自体の最適化
  3. **ConnectX-6 等の高性能 RNIC** — per-device connections のデフォルト有効化が可能に

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `20e808adb (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 37°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 94626) |

GGML_RDMA 環境変数: (none)
