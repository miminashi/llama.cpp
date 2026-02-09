# RDMA バックエンド性能改善調査レポート

- **実施日時**: 2026年2月7日 23:55
- **調査チーム**: 3エージェント並行調査 (protocol-researcher, transport-researcher, rpc-comparator)

## 前提・目的

P100 PCIe + ConnectX-4 (GPUDirect RDMA) 環境で、現在の RDMA バックエンド性能をさらに向上させる方法を調査する。

- **背景**: Step 5 (11GPU 事前検証) が完了し、GLM-4.7 IQ2_M で pp=6.4, tg=6.8 t/s を達成。ただし Generation 速度で RPC (TCP) 比 -10% の課題が残る
- **目的**: コード分析・HW設定調査・外部事例調査により、具体的な改善案と実装優先度を特定する
- **前提条件**: GPU は使用中のため実行テストは不可。コード分析とWeb調査のみ
- **参考レポート**: [RDMA vs RPC ベンチマーク v2](2026-02-06_000500_rdma_vs_rpc_benchmark_v2.md), [RNR NAK 修正](2026-02-05_225700_rnr_nak_root_cause_fix.md), [GPUDirect RDMA タイムアウト修正](2026-02-07_215841_gpudirect_rdma_timeout_fix.md)

### 現在の性能 (GLM-4.7 IQ2_M, 7 CUDA + 4 RDMA, 11GPU)

| バックエンド | Prompt (t/s) | Generation (t/s) |
|-------------|:------------:|:----------------:|
| **RDMA (GDR budget 12GB)** | **6.4** | **6.8** |
| RDMA (GDR 無効) | 6.4 | 6.0 |
| RPC (TCP) | 5.4 | 7.5 |

### 調査対象ファイル

| ファイル | 行数 | 内容 |
|---------|------|------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 3,292 | メインロジック (set_tensor, get_tensor, graph_compute) |
| `ggml/src/ggml-rdma/rdma-transport.cpp` | 900 | IB verbs 通信層 (QP, CQ, Send/Recv) |
| `ggml/src/ggml-rdma/rdma-transport.h` | 186 | 設定定数 |
| `ggml/src/ggml-rdma/rdma-gdr.cpp` | 444 | GPUDirect MR 管理 |
| `ggml/src/ggml-rdma/rdma-memory.cpp` | 307 | メモリ登録 |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | - | RPC バックエンド (比較対象) |

---

## 調査結果

### 1. Generation 速度で RPC 比 -10% の根本原因

3人の調査員が独立に分析した結果、以下の要因が一致して特定された。

#### 1.1 CQ ポーリングによる同期オーバーヘッド (最大要因)

**RPC (TCP)**: `send()` はカーネル TCP バッファにコピーして即座に返る。OS がバッファリング・フロー制御を管理。

**RDMA**: 各 Send/Recv で以下の同期処理が必須:
1. `ibv_post_send()` → `wait_for_completion()` で CQ ビジーポーリング（送信完了待ち）
2. `ibv_post_recv()` → `wait_for_completion()` で CQ ビジーポーリング（受信完了待ち）

1コマンドにつき最低 3 回の CQ ポーリング（header send 完了 + data send 完了 + response recv 完了）。Generation（1トークンずつ）ではサーバー計算時間に対するコマンド RTT の比率が大きく、このオーバーヘッドが顕著になる。

#### 1.2 graph_compute の同期レスポンス受信

| 項目 | RPC | RDMA |
|------|-----|------|
| graph_compute | **Fire-and-forget**（レスポンス受信なし） | **同期**（rsp_size=0 の 8B 受信待ち） |

RPC の `send_rpc_cmd()` は 3 引数版でレスポンスを受信しない。クライアントは即座にリターンし、サーバーの計算とクライアントの次コマンド準備がオーバーラップする。RDMA は毎回レスポンスの Send/Recv 往復が発生する。

#### 1.3 二次的要因

- **RDMA 独自の Flush 処理**: `FLUSH_AND_RECOMPUTE` は毎回フラッシュエントリをパックして送信。RPC には staging flush 概念がない
- **Delta Update**: `collect_updates` のスナップショット比較とシリアライゼーションコスト
- **op_mutex_**: `recursive_mutex` の取得/解放コスト（RPC にはなし）

### 2. 現在のシステム構成

transport-researcher がハードウェア設定を詳細に調査した結果:

| 項目 | 値 | 評価 |
|------|------|------|
| NIC | ConnectX-4 (mlx5_0), FW 12.21.1000 | - |
| リンク速度 | 100 Gb/sec (4X EDR), RoCE v2 | - |
| IB Active MTU | 4096 | ✅ 最適 |
| Ethernet MTU | 9000 (ジャンボフレーム) | ✅ 最適 |
| PCIe | Gen3 x16 (~15.75 GB/s) | - |
| **PCIe MaxReadReq** | **512B** | ⚠️ **推奨 4096B** |
| GPU | Tesla P100 PCIe 16GB × 7 | - |
| BAR1 | 16GB (full size) | ✅ 十分 |
| NUMA | 全 GPU + NIC が node 0 | ✅ 最適 |
| nvidia-peermem | loaded, live | ✅ 有効 |
| MLNX_OFED | 24.10-1.1.4.0 | ✅ 最新系 |
| Adaptive Coalescing | ON (rx-usecs=32, tx-usecs=8) | ✅ 適切 |
| **initiator_depth** | **1** | ⚠️ **推奨 16** |
| **max_inline** | **256** | ⚠️ **推奨 512** |
| **rnr_nak_retry_err** | **1,301,083** | ⚠️ **要対策** |

---

## 改善案一覧

### Phase 1: 即座に適用可能（コード変更なし/最小）

#### 1-1. PCIe MaxReadReq → 4KB

- **現状**: ConnectX-4 と P100 の両方で MaxReadReq = 512B
- **問題**: NVIDIA/Mellanox の推奨は 4096B。PCIe Read Request が小さいと、GPUDirect RDMA の大きな DMA Read が多数の小トランザクションに分割される
- **改善**: `setpci` で即時適用
- **難易度**: 低（コマンド実行のみ）
- **推定効果**: GPUDirect RDMA Read スループット 20-50% 改善

```bash
# 1号機: ConnectX-4
sudo setpci -s 0b:00.0 68.w=5920  # MaxReadReq=4096

# 1号機: P100 全7台 (各GPUのPCIeアドレスを確認して適用)
# ※ 元の値を確認してから bits [14:12] を 5 (=4096) に変更
sudo lspci -s 05:00.0 -vvv | grep MaxReadReq  # 現在値確認

# 2号機にも同様に適用
```

> **注意**: P100 の DevCap は MaxPayload=256B のため MaxPayload は変更不可。MaxReadReq のみ変更。永続化には udev ルールが必要。

#### 1-2. initiator_depth / responder_resources → 16

- **現状**: `initiator_depth = 1`, `responder_resources = 1` (rdma-transport.cpp:183-184, 317-318)
- **問題**: 同時に許可される RDMA Read/Atomic の outstanding 数が 1 に制限。RDMA Read のパイプライン化が不可能
- **改善**: 値を 16 に増加（ConnectX-4 の `max_qp_rd_atom = 16` に対応）
- **難易度**: 低（rdma-transport.cpp の 2 箇所を変更）
- **推定効果**: RDMA Read 多用時（get_tensor）のスループット大幅改善

```cpp
// rdma-transport.cpp:183-184, 317-318
conn_param.initiator_depth = 16;  // 変更前: 1
conn_param.responder_resources = 16;  // 変更前: 1
```

#### 1-3. max_inline → 512

- **現状**: `max_inline = 256` (rdma-transport.h:26)
- **問題**: コマンドヘッダ (9B) は inline だが、より大きなメタデータは NIC の DMA read が必要
- **改善**: ConnectX-4 は ~960B まで対応。512 に増やすことで Generation フェーズの小コマンドが全て inline 化
- **難易度**: 低（rdma-transport.h の 1 箇所を変更）
- **推定効果**: コマンドレイテンシ +1-3% 改善

```cpp
// rdma-transport.h:26
static constexpr int max_inline = 512;  // 変更前: 256
```

### Phase 2: 短期実装（中程度のコード変更）

#### 2-1. Unsignaled Send + Selective Signaling

- **現状**: `send()` で毎回 `IBV_SEND_SIGNALED` + `wait_for_completion()` を実行 (rdma-transport.cpp:430-442)
- **問題**: 各 Send で CQ ポーリングが必須。コマンド RTT の主要構成要素
- **改善**: N 回に 1 回のみ signaled にし、中間の send は unsignaled で即リターン。recv の前に CQ drain で未完了を回収
- **難易度**: 中（`rdma_connection::send()` に unsignaled オプション追加、`send_rdma_cmd_raw()` 改修）
- **推定効果**: +5-10%

参考: [RDMAmojo - Unsignaled Completions](https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/)

#### 2-2. Doorbell Batching / WR チェーン

- **現状**: header 送信と data 送信が別々の `ibv_post_send()` 呼び出し (ggml-rdma.cpp:619-653)。各呼び出しが PCIe MMIO doorbell を叩く (200-400ns/回)
- **改善**: `ibv_send_wr::next` でヘッダーとデータをチェーンし、1 回の `ibv_post_send()` で投入
- **難易度**: 中（`send_rdma_cmd_raw()` とトランスポート層の改修）
- **推定効果**: +3-5%

参考: [RDMAmojo - Tips and Tricks](https://www.rdmamojo.com/2013/06/08/tips-and-tricks-to-optimize-your-rdma-code/), [RDMA Doorbell Batching](https://sekwonlee.github.io/posts/2024/07/rdma-doorbell-batching/)

#### 2-3. Pre-post Recv（RNR NAK 根本排除）

- **現状**: `rnr_nak_retry_err = 1,301,083`（130 万回）。`min_rnr_timer=1` (0.01ms) で対処済みだが、累積 13 秒の待ち時間
- **問題**: 受信側が `post_recv` する前に送信側が `post_send` している
- **改善**: 接続確立直後に Recv バッファを複数プリポスト。受信完了後に即座に次の Recv を post（repost_recv パターン）
- **難易度**: 中（send/recv のパターン変更）
- **推定効果**: レイテンシジッター大幅改善

#### 2-4. graph_compute Fire-and-Forget 化

- **現状**: `FLUSH_AND_RECOMPUTE` コマンドで rsp_size=0 の同期レスポンスを毎回受信
- **問題**: RPC は fire-and-forget なのに、RDMA は不要なレスポンス受信で余分な RTT が発生
- **改善**: graph_compute/graph_recompute でレスポンス受信を省略。get_tensor 時に暗黙的に同期（RPC と同じモデル）
- **難易度**: 中（ggml-rdma.cpp のコマンド送信パス + サーバー側のレスポンス送信削除）
- **推定効果**: +5-10% (Generation)

> **注意**: graph_compute の完了を get_tensor で暗黙的に同期する設計が必要。RPC は同一ソケットの TCP ストリーム順序で保証しているが、RDMA では QP の Send 順序で同等の保証が得られる。

### Phase 3: 中長期（アーキテクチャ変更）

#### 3-1. コマンドパイプライン化

- **現状**: `ggml_backend_sched` が各バックエンドの graph_compute を逐次呼び出し
- **改善**: 4 RDMA デバイスの graph_compute を非同期送信し、サーバー計算をオーバーラップ。全デバイス送信後にまとめてレスポンス受信
- **難易度**: 高（`ggml_backend_sched` の改修が必要）
- **推定効果**: +15-25%

#### 3-2. RDMA Write for get_tensor Response

- **現状**: get_tensor レスポンスは IB Send/Recv (ggml-rdma.cpp:1001-1008)。logits (数百 KB〜数 MB) でチャンキングオーバーヘッド
- **改善**: サーバーがクライアントの事前登録 MR に RDMA Write で直接書き込み
- **難易度**: 高（プロトコル拡張）
- **推定効果**: +2-5%

#### 3-3. NCCL スタイルの制御/データ QP 分離

- **改善**: 1 接続に 2 つの QP（制御用 + データ用）で head-of-line blocking を回避
- **難易度**: 高
- **推定効果**: 将来のパイプライン化の基盤

---

## RPC との構造的差異まとめ

| 観点 | RPC (TCP) | RDMA | 影響 |
|------|-----------|------|------|
| Send 完了確認 | OS カーネル管理 (即返り) | **CQ ポーリング必須** | ← 最大ボトルネック |
| graph_compute | **Fire-and-forget** | 同期 (レスポンス受信) | Generation 速度差の主因 |
| バッファリング | カーネル TCP バッファ | なし (直接 NIC) | 小コマンドで TCP 有利 |
| 大データ転送 | TCP コピー | **RDMA Write ゼロコピー** | Prompt で RDMA 圧勝 |
| 接続モデル | 各 buffer に独立ソケット | 各デバイスに独立 QP | 同等 |
| コマンド構造 | 3 send (cmd+size+data) | 2 send (header+data) | RDMA やや有利 |

---

## 外部事例から得られた知見

### NCCL (NVIDIA)

- **2QP 分離**: データ用 QP + 制御用 QP（CTS）で制御メッセージの head-of-line blocking を回避
- **GDR dummy RDMA_READ**: GPUDirect RDMA 後に dummy RDMA_READ を発行し、PCIe 書き込みの到達を保証
- **Proxy thread**: CPU スレッドがネットワーク操作を管理

### eRPC / FaSST (CMU, USENIX ATC 2016)

- **Doorbell batching**: 複数の WR を 1 回の doorbell で投入して PCIe 往復を削減
- **Unsignaled completions**: N 回に 1 回のみ signaled で CQ ポーリング削減
- **推奨 QP 数**: ポートあたり 2-4 QP で最大スループット

### P100 固有の制約

| パラメータ | 値 |
|-----------|---|
| HBM2 帯域 | 732 GB/s |
| PCIe Gen3 x16 | ~16 GB/s (実効 ~12 GB/s) |
| Copy Engines | 2 (bidirectional) |
| NVLink | なし (PCIe モデル) |

- Generation 時のボトルネックは**計算**（PCIe 帯域ではない）
- Prompt 時のボトルネックは**PCIe 帯域**（RDMA Write の効果大）
- P100 は `asyncEngineCount=2` で compute/transfer overlap が可能だが、現在の RDMA バックエンドは未活用

---

## 推奨実装ロードマップ

```
Phase 1 (即時, ~1日):
  ├─ PCIe MaxReadReq → 4KB (setpci, コード変更なし)
  ├─ initiator_depth → 16 (2行変更)
  └─ max_inline → 512 (1行変更)

Phase 2 (短期, ~3-5日):
  ├─ Unsignaled Send (+5-10%)
  ├─ Doorbell Batching (+3-5%)
  ├─ Pre-post Recv (RNR NAK 排除)
  └─ graph_compute Fire-and-Forget (+5-10%)

Phase 3 (中長期, 要設計):
  ├─ コマンドパイプライン化 (+15-25%)
  ├─ RDMA Write for get_tensor Response (+2-5%)
  └─ QP 分離
```

### 期待される累積効果

| Phase | Generation 推定改善 | 達成後の推定 tg (t/s) |
|-------|:------------------:|:--------------------:|
| 現状 | - | 6.8 |
| Phase 1 | +5-10% | 7.1-7.5 |
| Phase 2 | +15-25% (累積) | 7.8-8.5 |
| Phase 3 | +30-45% (累積) | 8.8-9.9 |

> Phase 2 完了時点で RPC の Generation 速度 (7.5 t/s) に並ぶか上回る見込み。Prompt 速度は既に RDMA が +19% 優位なため、両フェーズで RPC を上回る可能性が高い。

---

## 参考資料

- [Design Guidelines for High Performance RDMA Systems (USENIX ATC 2016)](https://www.usenix.org/conference/atc16/technical-sessions/presentation/kalia)
- [RDMAmojo - Tips and Tricks to Optimize Your RDMA Code](https://www.rdmamojo.com/2013/06/08/tips-and-tricks-to-optimize-your-rdma-code/)
- [RDMAmojo - Working with Unsignaled Completions](https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/)
- [RDMA Doorbell Batching](https://sekwonlee.github.io/posts/2024/07/rdma-doorbell-batching/)
- [Optimized RDMA QP Communication (Cluster Computing 2025)](https://link.springer.com/article/10.1007/s10586-024-04796-7)
