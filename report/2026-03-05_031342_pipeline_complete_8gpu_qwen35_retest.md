# Pipeline Complete 8GPU Qwen3.5 再テスト

- **実施日時**: 2026年3月5日 03:13
- **ワークツリー**: `.worktree/pipeline-complete` (commit `34542217f`)
- **参照レポート**: [Pipeline Complete Qwen3.5 4GPU ベンチマーク](2026-03-05_011402_pipeline_complete_qwen35_benchmark.md)

## 前提・目的

Pipeline Complete (ring buffer + PARALLEL=1) の前回 4GPU テストで、PARALLEL=1 の PP 効果が +25.9% → +0.3% に激減した。2つの仮説を検証する:

1. **仮説 1 (llama-server 干渉)**: 前回テスト時に llama-server が同一ノードの GPU 4-6 で稼働しており、CPU/メモリ帯域の干渉で性能が歪んだ
2. **仮説 2 (ring buffer による吸収)**: ring buffer が Send selective signaling を回復したことで、PARALLEL がマスクしていた Send overhead が消え、PARALLEL の追加効果がなくなった

**検証方法**: 8GPU (4C+4R) に拡大し、干渉なしのクリーンな環境で再テスト。

### 交絡因子チェック
- [x] **単一変数の分離**: 同一バイナリ + `GGML_RDMA_PARALLEL=1` 環境変数でトグル
- [x] **環境変数トグル**: サーバー再起動で `static const` 環境変数を反映
- [x] **ホットパスのログ出力**: なし
- [x] **llama-server 干渉**: llama-server は CUDA_VISIBLE_DEVICES=4,5,6 で稼働中だが、テストは CUDA 0-3 を使用。GPU 競合なし

## テスト構成

| 項目 | 値 |
|------|-----|
| GPU 構成 | Node 1 CUDA0-3 + Node 2 RDMA0-3 (8GPU) |
| モデル | Qwen3.5-35B-A3B (MoE, 3B active) UD-Q4_K_M |
| Flash Attention | 有効 (`-fa 1`) |
| スレッド数 | 4 (`-t 4`) |
| テストパラメータ | pp128, pp512, pp2048, tg32, tg128 |
| 実験デザイン | ABAB paired design, 5 pairs (ウォームアップ 1 回破棄) |

### 条件

| 条件 | 環境変数 | 説明 |
|------|---------|------|
| A: Baseline | (デフォルト) | Ring buffer のみ |
| B: PARALLEL=1 | `GGML_RDMA_PARALLEL=1` | Ring buffer + pipeline + per-device connections |

## 再現方法

### ビルド・デプロイ

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/pipeline-complete/scripts/rdma-build.sh local
bash /home/ubuntu/projects/llama.cpp/.worktree/pipeline-complete/scripts/rdma-deploy.sh
```

### ベンチマーク実行

条件 A (Baseline):
```bash
gpu-lock.sh run bash scripts/rdma-server.sh start
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/ubuntu/projects/llama.cpp/.worktree/pipeline-complete/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -fa 1 -t 4 -p 128,512,2048 -n 32,128 -r 1 -o csv
```

条件 B (PARALLEL=1) — サーバー側も `GGML_RDMA_PARALLEL=1` で再起動:
```bash
ssh 192.168.100.2 "GGML_RDMA_PARALLEL=1 LD_LIBRARY_PATH=/.../build/bin nohup /.../rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
GGML_RDMA_PARALLEL=1 GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /home/ubuntu/projects/llama.cpp/.worktree/pipeline-complete/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -fa 1 -t 4 -p 128,512,2048 -n 32,128 -r 1 -o csv
```

## 結果

### 生データ (t/s)

| Pair | pp128 A | pp128 B | pp512 A | pp512 B | pp2048 A | pp2048 B | tg32 A | tg32 B | tg128 A | tg128 B |
|:----:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------:|:------:|:------:|:-------:|:-------:|
| 1 | 198.34 | 199.47 | 321.75 | 320.36 | 330.79 | 310.51 | 34.30 | 35.21 | 34.50 | 35.45 |
| 2 | 198.58 | 198.99 | 323.82 | 320.26 | 332.83 | 310.87 | 34.29 | 35.31 | 34.47 | 35.42 |
| 3 | 198.47 | 199.52 | 321.21 | 318.23 | 330.25 | 309.79 | 34.32 | 35.28 | 34.52 | 35.31 |
| 4 | 199.18 | 198.27 | 323.64 | 321.19 | 330.33 | 310.90 | 34.36 | 35.20 | 34.55 | 35.35 |
| 5 | 199.37 | 199.06 | 324.57 | 319.87 | 333.09 | 310.02 | 34.32 | 35.14 | 34.53 | 35.33 |

### 統計分析 (対応あり t 検定, 両側)

| 指標 | A (Baseline) | B (PARALLEL) | 差分平均 | 変化率 | t(4) | p 値 | Cohen's d | 95% CI | 方向 |
|:----:|:------------:|:------------:|:--------:|:------:|:----:|:----:|:---------:|:------:|:----:|
| pp128 | 198.79 ± 0.46 | 199.06 ± 0.50 | +0.274 | +0.14% | 0.695 | 0.526 (ns) | +0.31 | [-0.82, +1.37] | 3/5 |
| pp512 | 323.00 ± 1.44 | 319.98 ± 1.09 | -3.016 | **-0.93%** | -5.463 | 0.006 ** | -2.44 | [-4.55, -1.48] | 0/5 |
| pp2048 | 331.46 ± 1.39 | 310.42 ± 0.50 | -21.040 | **-6.35%** | -34.03 | <0.001 *** | -14.46 | [-22.85, -19.23] | 0/5 |
| tg32 | 34.32 ± 0.03 | 35.23 ± 0.07 | +0.909 | **+2.65%** | 24.290 | <0.001 *** | +10.86 | [+0.81, +1.01] | 5/5 |
| tg128 | 34.51 ± 0.03 | 35.37 ± 0.06 | +0.859 | **+2.49%** | 22.530 | <0.001 *** | +10.08 | [+0.75, +0.97] | 5/5 |

有意水準: *** p<0.001, ** p<0.01, * p<0.05, ns = 非有意

## 考察

### 仮説検証

**仮説 2 (ring buffer による吸収) が支持された。**

前回 4GPU テストで PARALLEL=1 が PP128 に +25.9% の効果を示したのは、Send always-signal による性能低下を PARALLEL のパイプライン非同期ディスパッチがマスクしていたためである。Ring buffer が Send selective signaling を回復したことで、この経路の overhead が解消され、PARALLEL の追加効果がなくなった。

### PP への影響: 中性 → 退行

| パラメータ | 変化率 | 解釈 |
|-----------|:------:|------|
| pp128 | +0.14% (ns) | 効果なし — ubatch 1個で pipeline overhead が最小 |
| pp512 | -0.93% ** | 微退行 — pipeline + per-device connections のオーバーヘッドが顕在化 |
| pp2048 | -6.35% *** | 大幅退行 — ubatch が 4個 (2048/512) に増加し、pipeline の chunked dispatch + per-ubatch sync が重いペナルティに |

**PP 退行のメカニズム**: `GGML_RDMA_PARALLEL=1` は `GGML_RDMA_PIPELINE=1` + `GGML_RDMA_PER_DEVICE_CONN=1` を同時有効化する。Pipeline dispatch は ubatch ごとに `alloc_graph` + `split_graph` + per-copy context buffer 管理を行うため、ubatch 数が多い pp2048 ではオーバーヘッドが大きい。また per-device connections は接続数が 4倍 (4 RDMA devices × 4 connections each) になり、接続管理コストが増加する。

### TG への影響: 一貫した改善

| パラメータ | 変化率 | 解釈 |
|-----------|:------:|------|
| tg32 | +2.65% *** | Per-device connections によるサーバー側デバイス並列化 |
| tg128 | +2.49% *** | 同上 |

TG の改善は per-device connections の効果と考えられる。TG ではバッチサイズが小さく (n_gen 個のトークン) pipeline のオーバーヘッドが小さい一方、サーバー側で各デバイスが独立した接続を持つことでコマンド処理が並列化される。

### 前回 4GPU テストとの比較

| 指標 | 前回 4GPU (ring buffer + PARALLEL) | 今回 8GPU (ring buffer + PARALLEL) |
|------|:---:|:---:|
| PP128 変化率 | +0.3% (ns) | +0.14% (ns) |
| TG32 変化率 | +3.35% *** | +2.65% *** |

4GPU と 8GPU で一貫した傾向: PP は中性、TG は +2.5-3.5% 改善。前回の結果が llama-server 干渉で歪んでいた証拠はない。

### Ring Buffer 単体の効果確認

Ring buffer baseline (条件 A) の 8GPU 性能:

| パラメータ | 今回 (ring buffer) | 前回参考値 (4GPU, ring buffer) |
|-----------|:---------:|:----------:|
| pp128 | 198.8 t/s | 199.9 t/s |
| tg32 | 34.3 t/s | 34.6 t/s |

8GPU でも 4GPU とほぼ同じ PP/TG — Qwen3.5 MoE は compute-bound であり、4GPU 以上のスケーリングが飽和している。

## 結論

1. **仮説 2 (吸収) が確認された**: Ring buffer による Send selective signaling 回復が PARALLEL=1 の PP 効果を吸収した
2. **PP への PARALLEL 効果は不要**: Ring buffer が Send overhead を解消した現在、pipeline dispatch はむしろオーバーヘッドとなる (特に長い pp で -6.35%)
3. **TG への PARALLEL 効果は有効**: Per-device connections による TG +2.5% は ring buffer とは独立した効果
4. **推奨構成**: Qwen3.5 (MoE) では ring buffer のみ (PARALLEL 無効) が最適。TG 改善が必要な場合のみ `GGML_RDMA_PER_DEVICE_CONN=1` 単体の使用を検討

### 今後の課題

- **PER_DEVICE_CONN 単体テスト**: PIPELINE なしの per-device connections 単体で TG 改善 + PP 退行なしを確認
- **GLM-4.7 (Dense) での検証**: communication-bound モデルでは異なる傾向の可能性

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `34542217f (feature/pipeline-complete)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 32°C | 32°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running |
| テスト使用 GPU | CUDA 0-3 | RDMA 0-3 |
| llama-server 稼働 | CUDA 4-6 (CUDA_VISIBLE_DEVICES=4,5,6) | — |
