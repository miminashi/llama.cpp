# PER_DEVICE_CONN 単体 8GPU A/B ベンチマーク

- **実施日時**: 2026年3月5日 03:32
- **ワークツリー**: `.worktree/pipeline-complete` (commit `34542217f`)

## 前提・目的

前回の 8GPU 再テスト ([pipeline_complete_8gpu_qwen35_retest](2026-03-05_031342_pipeline_complete_8gpu_qwen35_retest.md)) で `GGML_RDMA_PARALLEL=1` (= PIPELINE + PER_DEVICE_CONN 同時有効化) の結果:
- PP: pp128 中性、pp512 -0.93%、pp2048 **-6.35%** — pipeline dispatch overhead
- TG: tg32 **+2.65%**、tg128 **+2.49%** — per-device connections の効果と推定

**仮説**: PP 退行は PIPELINE のオーバーヘッド (ubatch ごとの alloc_graph + split_graph) が原因であり、PER_DEVICE_CONN 単体なら TG 改善のみで PP 退行なしになるはず。

## テスト構成

- **GPU**: Node 1 CUDA0-3 + Node 2 RDMA0-3 (8GPU)
- **モデル**: Qwen3.5-35B-A3B UD-Q4_K_M (MoE, 3B active params)
- **バイナリ**: `.worktree/pipeline-complete/build/bin/llama-bench` (commit `34542217f`)
- **サーバー**: `feature/rdma-backend` ビルドの rdma-server (PER_DEVICE_CONN はクライアント側のみ)
- **設計**: ABAB Paired Design × 5 pairs (ウォームアップ 1 回破棄)

### 条件

| 条件 | 環境変数 | 説明 |
|------|---------|------|
| A: Baseline | (デフォルト) | Ring buffer のみ |
| B: PER_DEVICE_CONN | `GGML_RDMA_PER_DEVICE_CONN=1` | Per-device connections のみ (PIPELINE なし) |

## 再現方法

### ベンチマークコマンド

条件 A:
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/ubuntu/projects/llama.cpp/.worktree/pipeline-complete/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -fa 1 -t 4 -p 128,512,2048 -n 32,128 -r 1 -o csv
```

条件 B: 先頭に `GGML_RDMA_PER_DEVICE_CONN=1` を追加。

## 結果

### 生データ (avg_ts, t/s)

| Pair | pp128 A | pp128 B | pp512 A | pp512 B | pp2048 A | pp2048 B | tg32 A | tg32 B | tg128 A | tg128 B |
|------|---------|---------|---------|---------|----------|----------|--------|--------|---------|---------|
| 1 | 200.13 | 196.73 | 324.91 | 308.58 | 334.83 | 319.71 | 33.40 | 34.31 | 34.52 | 34.39 |
| 2 | 199.21 | 196.68 | 323.97 | 306.48 | 330.82 | 319.24 | 34.23 | 34.15 | 34.45 | 34.40 |
| 3 | 199.75 | 196.28 | 324.07 | 309.95 | 334.11 | 320.64 | 34.38 | 34.22 | 34.51 | 34.43 |
| 4 | 200.18 | 196.43 | 323.11 | 305.72 | 333.89 | 317.22 | 34.30 | 34.11 | 34.50 | 34.26 |
| 5 | 199.85 | 196.58 | 325.22 | 307.94 | 332.40 | 316.64 | 34.14 | 33.95 | 34.47 | 34.36 |

### 統計解析 (対応ありt検定)

| Metric | A mean | B mean | Diff | t-stat | p-value | 判定 |
|--------|--------|--------|------|--------|---------|------|
| pp128 | 199.82 | 196.54 | **-1.64%** | 16.186 | 0.00009 | *** |
| pp512 | 324.25 | 307.74 | **-5.09%** | 26.068 | 0.00001 | *** |
| pp2048 | 333.21 | 318.69 | **-4.36%** | 16.097 | 0.00009 | *** |
| tg32 | 34.09 | 34.15 | +0.17% | -0.269 | 0.801 | ns |
| tg128 | 34.49 | 34.37 | -0.35% | 3.777 | 0.019 | * |

### 交絡チェック

- **Baseline 時間トレンド (pp512)**: r=-0.045, p=0.94 — トレンドなし
- **全データ時間トレンド (pp512)**: r=-0.191, p=0.60 — トレンドなし
- **Pair 1 tg32 A**: 33.40 (他は ~34.2-34.4) — コールドスタートによる外れ値の可能性あり

## 考察

### 仮説は棄却

PER_DEVICE_CONN 単体は **PP を大幅に退行させ (-1.6% ~ -5.1%)、TG 改善は得られなかった**。仮説「PP 退行は PIPELINE のみの原因」は棄却される。

### PP 退行の原因分析

PER_DEVICE_CONN は 1 サーバーに対して RDMA デバイスごとに独立した TCP 接続を確立する。これにより:

1. **接続確立オーバーヘッド**: 4 RDMA デバイス × 独立接続 = サーバー側で 4 スレッド管理
2. **コマンドシリアライゼーション消失**: 単一接続では graph_compute コマンドが直列化され、サーバーが順次処理。per-device では各接続が独立にコマンドを送信するが、サーバー側で同じ GPU リソースを争う
3. **PP のバッチ処理特性**: PP は大きなバッチを一括処理するため、デバイス並列化の恩恵が少なく、接続管理オーバーヘッドが支配的

**退行幅が pp サイズに比例** (pp128: -1.6%, pp512: -5.1%, pp2048: -4.4%) するのは、ubatch 分割数が増えるほど per-device 接続のオーバーヘッドが蓄積するため。

### 前回 PARALLEL=1 の TG 改善の再解釈

前回 `GGML_RDMA_PARALLEL=1` (PIPELINE + PER_DEVICE_CONN) で TG +2.65% の改善が見られたが、PER_DEVICE_CONN 単体では TG は中性 (+0.17%, ns)。したがって:

- **TG 改善は PIPELINE と PER_DEVICE_CONN の組み合わせ効果**であり、PER_DEVICE_CONN 単体の効果ではない
- PIPELINE の非同期ディスパッチ + per-device 独立接続の組み合わせがサーバー側デバイス並列化を実現
- どちらか一方では効果が出ない

### 過去結果との整合性

MEMORY に記録された過去の知見と一致:
- "Per-device alone (Qwen3.5 pp20000): -1.1% (4GPU) to -4.2% (8GPU) — overhead scales with GPU count"
- 今回の 8GPU 結果 (-1.6% ~ -5.1%) は、pp サイズ依存を新たに明らかにした

## 結論

| 構成 | PP | TG | 推奨 |
|------|----|----|------|
| Ring buffer のみ (デフォルト) | ベースライン | ベースライン | Qwen3.5 MoE に最適 |
| PER_DEVICE_CONN 単体 | **-1.6% ~ -5.1%** | 中性 | **非推奨** |
| PIPELINE + PER_DEVICE_CONN | -0.9% ~ -6.4% (PP) / +2.5% (TG) | +2.5% | TG 重視時のみ |

**Qwen3.5 MoE (compute-bound) では PER_DEVICE_CONN を有効化する理由がない。** Ring buffer のみがすべてのメトリクスで最善。PER_DEVICE_CONN は communication-bound モデル (GLM-4.7) での検証が別途必要。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c1ca6d5fd (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 39°C | 33°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1429397) |
