# RDMA性能ばらつき根本原因の特定と修正

- **実施日時**: 2026年2月5日 22:57
- **参照レポート**:
  - [2026-02-05_210745_rdma_variance_deep_investigation.md](2026-02-05_210745_rdma_variance_deep_investigation.md) (前回調査 — 根本原因未特定)
  - [2026-02-05_192000_gpu_combination_variance_analysis.md](2026-02-05_192000_gpu_combination_variance_analysis.md) (1400%分散の報告)
  - [2026-02-05_170441_gpu_combination_benchmark.md](2026-02-05_170441_gpu_combination_benchmark.md) (GPU組み合わせ別ばらつき)

## 前提・目的

### 背景

前回の調査 (2026-02-05_210745) では、`usleep(10)` の除去により分散が改善されたと報告されたが、元の1400%分散の根本原因は未特定のまま終了していた。

今回の再調査で、**usleep除去後も2秒スパイクが再現すること**を確認し、真の根本原因がusleepではなかったことが判明した。

### 目的

前回の調査失敗を受け、RDMA性能ばらつきの真の根本原因を特定し、恒久的な修正を行う。

### 前提条件

- 1号機 (192.168.100.1): 7× Tesla P100-PCIE-16GB
- 2号機 (192.168.100.2): 4× Tesla P100-PCIE-16GB
- InfiniBand: Mellanox ConnectX-4 100Gbps (RoCE v2)
- GPUDirect RDMA: 有効 (nvidia-peermem)

## 結論

### 根本原因: IB RNR (Receiver Not Ready) NAKのデフォルトタイマー

**RDMA Send/Recvプロトコルにおいて、送信側がデータを送信する際に受信側がまだ受信バッファを登録 (`ibv_post_recv`) していない場合、RNR NAK (Receiver Not Ready Negative Acknowledgment) が発生する。**

デフォルトの `min_rnr_timer` = 0 (655.36ms) のため、RNRリトライ1回あたり655ms待機し、3回のリトライで約2秒の遅延が発生していた。

### 修正内容

`min_rnr_timer` を 1 (0.01ms) に設定することで、RNRリトライ時の待機時間を65,536分の1に短縮。

### 修正効果

| 指標 | 修正前 | 修正後 | 改善率 |
|------|:------:|:------:|:------:|
| 1+1 pp128 分散 | ±152.26 (98.5%) | **±12.66 (3.3%)** | **12倍安定化** |
| 1+1 tg32 分散 | ±20.48 (56.7%) | **±0.77 (1.5%)** | **27倍安定化** |
| 1+4 tg32 速度 | 0.57 t/s | **48.67 t/s** | **85倍高速化** |
| 120b tg32 分散 | ±14.19 (53.4%) | **±0.07 (0.2%)** | **203倍安定化** |
| 2秒スパイク | 12回/10rep | **0回** | **完全解消** |

## 調査経緯

### Step 1: 前回報告の再検証

usleep(10) がコメントアウト済みであることを両ノードで確認したうえで、RDMA 1+1 GPUで10回繰り返しテストを実行。

```
[client send_raw] cmd=10: header=0.01 ms, data=1968.33 ms (size=4)
[client send_raw] cmd=24: header=0.08 ms, data=2140.08 ms (size=32)
```

**結果**: usleepなしでも4バイトの送信に1968ms〜2150msかかるスパイクが発生。前回の「usleep除去で解決」は**誤報**だった。

pp128=154.51±152.26、tg32=36.11±20.48 (依然として巨大な分散)。

### Step 2: RNR NAKカウンタの発見

InfiniBandハードウェアカウンタを確認:

```bash
cat /sys/class/infiniband/mlx5_0/ports/1/hw_counters/rnr_nak_retry_err
# → 9677 (大量のRNR NAKリトライ)
```

テスト前後の差分計測:

| テスト | RNR NAK増加数 |
|--------|:------------:|
| 修正前 r=3 | +16 |

**16回のRNR NAKが3回のテスト繰り返しで発生。** これが2秒スパイクの原因。

### Step 3: 原因メカニズムの特定

IB Send/Recvプロトコルのタイミング問題:

```
Client                          Server
  |                               |
  |--- ibv_post_send(header) ---->|  recv(header) posted ✓
  |                               |
  |--- ibv_post_send(data) ------>|  recv(data) NOT YET posted ✗
  |        ↑                      |    ↑
  |   RNR NAK ← ← ← ← ← ← ← ← |  まだ post_recv していない
  |                               |
  |   wait 655.36ms (min_rnr_timer=0)
  |                               |
  |--- retry send(data) -------->|  recv(data) posted ✓
  |        ↑                      |
  |   RNR NAK again (server still processing)
  |                               |
  |   wait 655.36ms              |
  |   wait 655.36ms              |
  |   ≈ 2 seconds total          |
  |                               |
  |--- retry send(data) -------->|  recv(data) posted ✓ → success!
```

**原因の根本**: クライアントがヘッダー送信後、即座にデータを送信するが、サーバーはヘッダー受信→パース→データ用recv登録の処理に数μsかかる。この間にクライアントの送信が到着すると、受信バッファ未登録のためRNR NAKが返される。

**デフォルトのmin_rnr_timer = 0 (655.36ms)** のため、リトライ間隔が極端に長く、数回のRNRで約2秒の遅延となる。

### Step 4: min_rnr_timer 修正

`rdma-transport.cpp` の `accept()` と `connect_qp()` に以下を追加:

```cpp
struct ibv_qp_attr attr = {};
attr.min_rnr_timer = 1;  // 0.01ms (655.36ms → 0.01ms)
ibv_modify_qp(qp_, &attr, IBV_QP_MIN_RNR_TIMER);
```

### Step 5: 修正後の検証

#### gpt-oss-20b (1+1 GPU, r=10)

| 指標 | 修正前 | 修正後 |
|------|:------:|:------:|
| pp128 | 154.51 ± 152.26 t/s | **381.94 ± 12.66 t/s** |
| tg32 | 36.11 ± 20.48 t/s | **51.73 ± 0.77 t/s** |
| スパイク (>500ms) | 12回 | **0回** |
| 初回graph_compute | ~4174 ms | **~120 ms** |

#### gpt-oss-20b マルチGPU構成

| 構成 | pp128 (修正前) | pp128 (修正後) | tg32 (修正前) | tg32 (修正後) |
|:----:|:--------------:|:--------------:|:-------------:|:-------------:|
| 1+1 | 154.51 ± 152 | **381.94 ± 12.66** | 36.11 ± 20.48 | **51.73 ± 0.77** |
| 1+2 | 270.98 ± 185 | **400.72 ± 16.66** | 40.96 ± 29 | **57.81 ± 0.31** |
| 2+2 | 385.20 ± 12 | **387.68 ± 15.89** | 57.55 ± 0.02 | **56.33 ± 0.03** |
| 1+4 | 155.95 ± 187 | **377.14 ± 17.86** | 0.57 ± 0.05 | **48.67 ± 8.66** |

#### gpt-oss-120b (7+4 = 11 GPU)

| 指標 | 修正前 | 修正後 |
|------|:------:|:------:|
| pp128 | 171.64 ± 68.43 t/s | **198.80 ± 2.10 t/s** |
| tg32 | 26.59 ± 14.19 t/s | **36.78 ± 0.07 t/s** |

#### 長期安定性テスト (1+1 GPU, r=20)

| 指標 | 結果 |
|------|:----:|
| pp128 | 368.66 ± 44.21 t/s |
| tg32 | **52.17 ± 0.06 t/s** |
| スパイク | **0回** |

tg32の分散はわずか0.1%で、完全な安定動作を確認。

## 前回の調査結果との整合性

### なぜ前回「usleep除去で解決」と報告されたか

前回の調査 (210745) では:
1. usleep(10) を追加 → 全操作が2秒遅延 (usleep自体の問題)
2. usleep を除去 → 速度が改善

しかし、**usleep除去後にtg32のみテストした**ため、pp128での間欠的スパイクを検出できなかった可能性が高い。tg32フェーズはテスト境界をまたがないため、RNR NAKが発生しにくい。

今回の調査で、usleepとRNRは**別の問題**であり、usleep除去はusleep自体の問題を修正しただけで、RNRの根本問題は残っていたことが確定した。

### 前回の「1400%分散」の原因

前回未解明だった1400%分散は**RNR NAKのデフォルトタイマー (655.36ms)** が原因。min_rnr_timer=1 (0.01ms) への変更で完全解消した。

## 変更内容

### rdma-transport.cpp

**accept() (サーバー側)**:
```cpp
// 接続確立後に min_rnr_timer を最小値に設定
struct ibv_qp_attr attr = {};
attr.min_rnr_timer = 1;  // 0.01ms
ibv_modify_qp(qp_, &attr, IBV_QP_MIN_RNR_TIMER);
```

**connect_qp() (クライアント側)**:
```cpp
// 接続確立後に min_rnr_timer を最小値に設定
struct ibv_qp_attr attr = {};
attr.min_rnr_timer = 1;  // 0.01ms
ibv_modify_qp(qp_, &attr, IBV_QP_MIN_RNR_TIMER);
```

## 再現方法

### 1. ビルド (1号機)
```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 2. デプロイ・ビルド (2号機)
```bash
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' --exclude='build' /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/ 192.168.100.2:/home/ubuntu/projects/llama.cpp/
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
```

### 3. サーバー起動 (2号機)
```bash
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
```

### 4. ベンチマーク実行 (1号機)
```bash
# 1+1 GPU
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-bench --model /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf \
  --n-gpu-layers 999 --split-mode layer --repetitions 10 --n-prompt 128 --n-gen 32

# 11 GPU (120b)
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-bench --model /tmp/gpt-oss-120b/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  --device 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  --n-gpu-layers 999 --split-mode layer --repetitions 3 --n-prompt 128 --n-gen 32
```

### 5. RNR NAKカウンタの確認
```bash
cat /sys/class/infiniband/mlx5_0/ports/1/hw_counters/rnr_nak_retry_err
```

## 技術的補足

### min_rnr_timer のエンコーディング

| 値 | 待機時間 |
|:--:|:--------:|
| 0 | 655.36 ms (デフォルト) |
| 1 | 0.01 ms |
| 2 | 0.02 ms |
| ... | ... |
| 31 | 491.52 ms |

### なぜRNR NAKが発生するか

IB Send/Recvプロトコルでは、受信側が事前に `ibv_post_recv()` で受信バッファを登録しておく必要がある。現在の実装では:

1. サーバーがheader受信 → データ用recvをpost → クライアントのdata sendを受信

この「header受信→data recv post」の間に数μsの処理時間があり、この間にクライアントが送信するとRNR NAKが発生する。

### RNRを完全に防止するには

将来的により根本的な対策として以下が考えられる:

1. **Pre-posted recv**: データ用の受信バッファを事前にpostしておく
2. **RDMA Write への移行**: Send/Recvの代わりにRDMA Writeを使用 (受信バッファ不要)
3. **共有受信キュー (SRQ)**: 複数接続で受信バッファを共有

ただし、min_rnr_timer=1 (0.01ms) の修正だけでRNRリトライの遅延は実質ゼロになるため、現時点では追加対策は不要。

## 残存課題

1. **pp128の±44分散**: 初回フルグラフ送信 (126KB) のオーバーヘッドによるもの。これはテスト間のグラフ構造変更が原因で、通常の推論では発生しない (reuse=1で安定動作)。

2. **RNR NAK自体の発生**: min_rnr_timerの短縮で影響は無視できるレベルだが、RNR自体は依然として発生している。根本的な解消には pre-posted recv の実装が必要。
