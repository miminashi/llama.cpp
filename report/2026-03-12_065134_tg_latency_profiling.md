# TG レイテンシプロファイリング (Phase 3-1)

- **実施日時**: 2026年3月12日 06:51
- **ワークツリー**: メインリポジトリ (`feature/rdma-backend`)
- **参照レポート**: [RDMA vs RPC ベンチマーク](2026-03-10_113211_rdma_vs_rpc_upstream_benchmark.md)

## 前提・目的

RDMA バックエンドは TG で RPC に対して -3.2% (122B) ～ -8.1% (35B) 劣位。
TG per-step のレイテンシを分解し、ボトルネックを特定する。

### TG ベースライン

| 構成 | RDMA (t/s) | RPC (t/s) | RDMA step (ms) | RPC step (ms) | 差分 (ms) |
|------|:----------:|:---------:|:--------------:|:-------------:|:---------:|
| 35B 4GPU (2C+2R) | 35.02 | 38.12 | 28.6 | 26.2 | +2.4 |
| 122B 11GPU (7C+4R) | 18.28 | 18.88 | 54.7 | 52.9 | +1.8 |

## 再現方法

### 35B 4GPU プロファイリング

```bash
gpu-lock.sh run env GGML_RDMA_PROFILE=1 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-bench \
  -m models/Qwen3.5-35B-A3B-Q4_K_M-fused.gguf \
  -ngl 999 -fa 1 -t 1 -p 0 -n 32 -r 1 \
  -dev 'CUDA0/CUDA1/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]'
```

### 122B 11GPU プロファイリング

```bash
gpu-lock.sh run env GGML_RDMA_PROFILE=1 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  build/bin/llama-bench \
  -m models/Qwen3.5-122B-A10B-Q4_K_M-fused.gguf \
  -ngl 999 -fa 1 -t 1 -p 0 -n 32 -r 1 \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]'
```

## TG Hot Path の構造

TG の 1 step は graph reuse パスで以下の順序で実行される:

```
set_tensor (KV cache + embeddings)  → RDMA Write (async)
  ↓
graph_compute (reuse=true)          → Send cmd (async) per RDMA device
  ↓                                   ※ 共有接続のため逐次ディスパッチ
get_tensor (output embeddings)      → Send/Recv (blocking)
```

`RDMA_ASYNC_COMPUTE` (デフォルト有効) により、graph_compute の Send は fire-and-forget。
レスポンスは次のコマンド送信時の drain、または get_tensor で回収される。

## プロファイリング結果

### 35B 4GPU (2C+2R) per-step 分解

TG-only インクリメンタルデータ (graph_compute calls 10→60, recompute 50回 = 25 steps):

| フェーズ | 時間 (ms/step) | 比率 | 内訳 |
|---------|:-------------:|:----:|------|
| set_tensor | **~0.0** | 0.1% | 260 calls, 0.8ms total, RDMA Write async |
| graph_compute (2 devices) | **5.96** | 20.8% | D0: 1.3ms + D1: 6.3ms (drain D0 待ち含む) |
| get_tensor | **8.86** | 31.0% | D1 compute 完了待ち + Send/Recv |
| **RDMA 小計** | **14.8** | **51.7%** | |
| CUDA ローカル (2 devices) | **13.8** | **48.3%** | |
| **合計** | **28.6** | 100% | |

#### graph_compute per-device 詳細 (recompute, async)

| デバイス | nodes | send (ms) | total (ms) | 備考 |
|---------|:-----:|:---------:|:----------:|------|
| RDMA0 (D0) | 1128 | 0.48 | 1.30 | 即座にディスパッチ |
| RDMA1 (D1) | 964 | **5.52** | **6.26** | D0 レスポンス drain で **~5ms ブロック** |

**重要発見**: D1 の send 5.52ms は IB 送信自体ではなく、**D0 のサーバーサイド compute レスポンスの drain 待ち**。
共有接続 (デフォルト) では全デバイスのコマンドが同一 QP を共有するため、前デバイスのレスポンスを回収してからでないと次のコマンドを送信できない。

### 122B 11GPU (7C+4R) per-step 分解

TG-only インクリメンタルデータ (graph_compute calls 20→130, recompute 110回 = 27.5 steps):

| フェーズ | 時間 (ms/step) | 比率 | 内訳 |
|---------|:-------------:|:----:|------|
| set_tensor | **~0.1** | 0.1% | 633 calls, 1.8ms total, RDMA Write async |
| graph_compute (4 devices) | **14.7** | 26.9% | 4 デバイス逐次ディスパッチ |
| get_tensor | **7.13** | 13.0% | D3 compute 完了待ち + Send/Recv |
| **RDMA 小計** | **21.8** | **39.9%** | |
| CUDA ローカル (7 devices) | **32.9** | **60.1%** | |
| **合計** | **54.7** | 100% | |

#### graph_compute per-device 詳細 (recompute, async)

| デバイス | nodes | send (ms) | total (ms) | 備考 |
|---------|:-----:|:---------:|:----------:|------|
| RDMA0 (D0) | 442 | 0.01 | 0.30 | 即座にディスパッチ |
| RDMA1 (D1) | 564 | **4.01** | **4.42** | D0 drain 待ち |
| RDMA2 (D2) | 442 | **5.02** | **5.31** | D1 drain 待ち |
| RDMA3 (D3) | 324 | **4.51** | **4.75** | D2 drain 待ち |

**graph_compute chain**: 0.3 + 4.4 + 5.3 + 4.8 = **14.8 ms** (RDMA デバイス数に比例してスケール)

### get_tensor 分析

| 構成 | avg (ms) | bytes/call | throughput (MB/s) |
|------|:--------:|:----------:|:-----------------:|
| 35B (2R) | 8.86 | 993 KB | 109 |
| 122B (4R) | 7.13 | 993 KB | 138 |

get_tensor のスループットが低い (IB 理論 12.5 GB/s に対して ~130 MB/s)。
これは同期的な Send/Recv ラウンドトリップ (リクエスト送信 → サーバー D2H コピー → レスポンス受信) のレイテンシが支配的であることを示す。

## RDMA vs RPC 差分の分析

### 35B 4GPU: RDMA が RPC より 2.4 ms/step 遅い

| 要因 | 推定影響 (ms) | 説明 |
|------|:-----------:|------|
| graph_compute drain 待ち | ~5.5 | D1 dispatch 時の D0 drain。RPC は独立ソケットで drain 不要 |
| get_tensor Send/Recv | ~8.9 | RPC は TCP socket で同等のオーバーヘッド |
| **差分の主因** | **~2.4** | drain 待ちの一部 (RPC の独立ソケットとの差) |

RPC は各デバイスに独立した TCP ソケットを使用するため、Device 0 の応答を待たずに Device 1 にコマンドを送信可能。
RDMA の共有接続モデルでは drain が必須。

### 122B 11GPU: RDMA が RPC より 1.8 ms/step 遅い

122B では CUDA ローカル compute (32.9 ms) が支配的なため、RDMA オーバーヘッドの影響が相対的に小さい。
しかし、4 RDMA デバイスの逐次 drain で ~13.5 ms を消費しており、16GPU (8R) 構成ではさらに増加する。

### drain 待ちの時間構造

```
35B (2R):  [D0: 0.5ms]---[D0 server compute ~5ms]---[drain]---[D1: 0.5ms]---[D1 compute]---[get_tensor: drain D1 + recv]
122B (4R): [D0: 0.3ms]---[drain D0: 4ms]---[drain D1: 5ms]---[drain D2: 4.5ms]---[get_tensor: 7ms]
                          ↑ D0 compute       ↑ D1 compute      ↑ D2 compute
```

各 drain は前デバイスのサーバーサイド compute 完了を待つ。サーバー compute 時間:
- 35B: D0 ~5ms (1128 nodes, MoE 含む)
- 122B: D0 ~4ms, D1 ~5ms, D2 ~4.5ms, D3 ~4.8ms (各デバイスの割当レイヤー数に依存)

## RDMA デバイス数のスケーリング

| 構成 | RDMA devices | graph_compute (ms/step) | get_tensor (ms/step) | RDMA total (ms/step) | RDMA 比率 |
|------|:-----------:|:----------------------:|:-------------------:|:-------------------:|:---------:|
| 35B 4GPU | 2 | 5.96 | 8.86 | 14.8 | 51.7% |
| 122B 11GPU | 4 | 14.7 | 7.13 | 21.8 | 39.9% |

graph_compute は RDMA デバイス数に比例して増加 (drain chain)。
get_tensor は 122B の方が若干速い (CUDA compute 中にサーバー compute が完了し、drain 待ちが短い)。

## ボトルネック分析

### 主因: graph_compute の逐次 drain (共有接続)

```
Priority 1: graph_compute drain chain  → 35B: 5.5ms, 122B: 14.5ms
Priority 2: get_tensor Send/Recv       → 35B: 8.9ms, 122B: 7.1ms
Priority 3: set_tensor                 → negligible
```

### graph_compute drain の根本原因

`RDMA_ASYNC_COMPUTE` モードでは、Send は fire-and-forget だが、**次のデバイスのコマンド送信前に前デバイスのレスポンスを drain する**必要がある。
これは共有接続 (単一 QP) で Send/Recv のシーケンス番号が混在するため。

**RPC との構造的差異**: RPC は各リモートデバイスに独立 TCP ソケットを持つため、drain なしで全デバイスに並行ディスパッチ可能。

## Phase 3-2 最適化候補 (優先度順)

### 1. Per-device connection の TG 最適化 (優先度: 高)

`GGML_RDMA_PER_DEVICE_CONN=1` で各 RDMA デバイスに独立接続を割当て、drain chain を解消。

- **期待効果**: graph_compute drain ~5ms (35B) / ~13.5ms (122B) → ~0ms
- **リスク**: per-device conn は PP で -1.6% ～ -5% の退行あり (接続管理オーバーヘッド)
- **対策**: TG のみ per-device、PP は共有接続にする動的切替は困難。
  PP 退行が許容範囲なら全体で per-device 有効化。
- **既存データ**: per-device alone で tg32 +1.76% (GLM-4.7)、Qwen3.5 8GPU で tg32 +0.17%(ns)

> 注: 既存の per-device 実装では TG でわずかな改善しか見られなかった (+0.17% ns)。
> これは per-device が drain を解消しても、get_tensor が同期的にレスポンス待ちするため、
> 節約した drain 時間が get_tensor の待ち時間に移動するだけの可能性がある。
> 効果を最大化するには、graph_compute の非同期ディスパッチ + get_tensor の遅延回収を組み合わせる必要がある。

### 2. get_tensor の RDMA Read 化 (優先度: 中)

現在 get_tensor は Send/Recv (リクエスト→レスポンス) で 7-9ms。
RDMA Read (one-sided) に変更すればサーバーサイドの処理待ちを排除可能。

- **期待効果**: get_tensor 7-9ms → 1-2ms (RDMA Read + polling)
- **前提**: サーバーサイドの compute が完了し、結果がホストメモリに配置済みであること
- **課題**: compute 完了通知が別途必要 (completion flag の RDMA polling)
- **リスク**: GDR が有効な場合、結果が GPU メモリ上 → RDMA Read は D2H + Read の 2 段階

### 3. graph_compute レスポンスの遅延回収 (優先度: 中)

現在: D0 dispatch → drain D0 → D1 dispatch → drain D1 → ... → get_tensor (drain 最終 D)

提案: D0 dispatch → D1 dispatch → ... → CUDA compute → drain ALL → get_tensor

共有接続でも、Send の火消し後にレスポンスを後でまとめて drain すれば、
drain 待ちを CUDA compute とオーバーラップ可能。

- **期待効果**: drain chain を CUDA compute 期間に隠蔽
- **課題**: 共有接続では Send/Recv のシーケンスが混在するため、interleave が複雑
- **リスク**: レスポンスの順序保証が必要

### 4. Pipelined set_tensor + graph_compute (優先度: 低)

set_tensor は既に negligible (0.1 ms 未満) なため、最適化の余地が少ない。

## 結論

1. **TG の RDMA ボトルネックは graph_compute の逐次 drain chain** (全 RDMA 時間の 35B: 40%, 122B: 67%)
2. **RPC との差分 (2.4ms / 1.8ms) は drain chain の一部** — RPC は独立ソケットで並行ディスパッチ
3. **RDMA デバイス数に比例して drain time が増加** — 16GPU (8R) では ~28ms の drain 予想
4. **get_tensor の Send/Recv オーバーヘッド (7-9ms) も改善余地あり** — RDMA Read 化で 1-2ms 目標
5. **set_tensor は完全に最適化済み** (RDMA Write async, negligible)

Phase 3-2 では per-device connection + 遅延 drain の組み合わせによる TG 改善を検討する。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `453f4c7c1 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 23°C | 30°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1979336) |

GGML_RDMA 環境変数: (none) ※ `GGML_RDMA_PROFILE=1` はプロファイリング時のみ指定
