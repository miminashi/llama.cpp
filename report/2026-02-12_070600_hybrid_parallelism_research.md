# ハイブリッド並列化リサーチレポート — GLM-4.7 on P100×16

- **実施日時**: 2026年2月12日 07:06
- **ワークツリー**: `.worktree/rdma-backend`
- **関連レポート**: [parallel_compute_dispatch](2026-02-11_214242_parallel_compute_dispatch.md)

## 用語解説

本レポートで頻出する並列化手法の略称を以下に整理する。

### PP (Pipeline Parallelism / パイプライン並列)

モデルを**レイヤー単位**で複数 GPU に分割する手法。llama.cpp では `-sm layer` に相当。

```
GPU0: Layer 0-22  →  GPU1: Layer 23-45  →  GPU2: Layer 46-68  →  GPU3: Layer 69-91
         ↓ hidden state     ↓ hidden state     ↓ hidden state
```

各 GPU は自分の担当レイヤーのみを計算し、隣の GPU に hidden state (中間表現) を渡す。
**利点**: 通信量が小さい (hidden state のみ)、実装がシンプル。
**欠点**: Generation (1トークンずつ生成) では全 GPU が直列に動作するため、GPU を増やしても**合計計算時間は減らない**。VRAM の分散にはなるが速度向上にはつながらない。

### TP (Tensor Parallelism / テンソル並列)

1つのレイヤー内の**行列演算**を複数 GPU に分割する手法。llama.cpp では `-sm row` に相当。

```
Layer N の行列乗算 W × x:
  GPU0: W の上半分 × x → 部分結果0 ─┐
                                     ├→ AllReduce → 完全な結果
  GPU1: W の下半分 × x → 部分結果1 ─┘
```

各 GPU は同じレイヤーの重み行列の一部を持ち、同時に計算する。部分結果を AllReduce (集約通信) で合算して完全な結果を得る。
**利点**: 各レイヤーの計算を GPU 数で分割できるため、Generation でも**計算時間が短縮**される。
**欠点**: 毎レイヤーで AllReduce 通信が必要。GPU 間の通信が遅い環境 (NVLink なし) ではレイテンシが支配的になる。

### EP (Expert Parallelism / エキスパート並列)

MoE (Mixture of Experts) モデル特有の手法。160 個の Expert を複数 GPU に**分散配置**する。

```
GPU0: Expert 0-19   GPU1: Expert 20-39   ...   GPU7: Expert 140-159
         ↑ トークンのルーティング結果に応じて該当 GPU で計算
```

各トークンは 160 Expert 中 8 個のみ使うため、大半の Expert は推論時に不要。
**利点**: VRAM 効率が極めて高い。通信量も小さい (hidden state の All-to-All)。
**欠点**: All-to-All 通信と動的ディスパッチの実装が複雑。

### ハイブリッド並列 (本レポートの主題)

上記の手法を**組み合わせ**て使う。例えば:
- **TP=2 + PP=8**: 同一ノード内の 2 GPU で TP、8 ペアを PP で接続 (計 16 GPU)
- **EP + PP**: Expert を分散配置しつつ、レイヤー群をパイプライン化

単一の手法では限界があるが、組み合わせることで各手法の弱点を補える。

---

## 前提・目的

GLM-4.7 (160×21B MoE, 93レイヤー) を P100 PCIe 16GPU (2ノード×8) で推論する際、
現在の `-sm layer` (パイプライン並列) では Generation ~7 t/s が限界。
`-sm row` (テンソル並列) は P100 PCIe の NVLink 不在により layer split より遅い。
**両者の中間的アプローチで限界を突破できないかリサーチし、定量的に分析する。**

### 背景

- **ハードウェア**: P100 PCIe 16GB × 16台 (2ノード × 8GPU)、100GbE RDMA (ConnectX-4)
- **現在の実績**: GLM-4.7 IQ2_M on 11GPU (7C+4R): pp=6.4 t/s, tg=6.8 t/s
- **制約**: NVLink なし (PCIe P2P は利用可能、NCCL_P2P_DISABLE=1)、ConnectX-4 MTT キャッシュ ~12GB

---

## 1. GLM-4.7 アーキテクチャ分析

### 1.1 モデルパラメータ (GGUF メタデータより)

| パラメータ | 値 |
|-----------|-----|
| アーキテクチャ | `glm4moe` (Mixture of Experts) |
| サイズラベル | 160×21B |
| 総レイヤー数 | 93 (うち leading dense = 3, NextN predict = 1) |
| **Transformer レイヤー数** | **92** (93 - 1 NextN) |
| Embedding 次元 | 5,120 |
| Attention heads | 96 (KV heads = 8, GQA) |
| Head 次元 | 128 (key/value 共通) |
| Dense FFN 次元 | 12,288 |
| Expert 数 | **160** |
| アクティブ Expert 数 | **8** (+ 共有 Expert 1) |
| Expert FFN 次元 | 1,536 |
| Expert グループ数 | 1 |
| Expert ゲーティング | softmax_weight (type=2) |
| Expert weights scale | 2.5 |
| コンテキスト長 | 202,752 |
| 語彙サイズ | 151,552 |

### 1.2 レイヤー構造の詳細

GLM-4.7 の 92 Transformer レイヤーは 2 種類:

**Dense レイヤー (il < 3): レイヤー 0, 1, 2**
```
Attention(Q,K,V → O) + Dense FFN(gate+up → down)
- Q: [5120, 12288] K: [5120, 1024] V: [5120, 1024] O: [12288, 5120]
- FFN up: [5120, 12288] gate: [5120, 12288] down: [12288, 5120]
```

**MoE レイヤー (il >= 3): レイヤー 3～91 (89レイヤー)**
```
Attention(Q,K,V → O) + MoE FFN(160 experts) + Shared Expert FFN
- Q: [5120, 12288] K: [5120, 1024] V: [5120, 1024] O: [12288, 5120]
- Gate: [5120, 160] (expert routing)
- Expert FFN: 160× {up: [5120, 1536], gate: [5120, 1536], down: [1536, 5120]}
- Shared Expert: up: [5120, 12288] gate: [5120, 12288] down: [12288, 5120]
```

### 1.3 パラメータサイズの推定 (IQ2_M ≈ 2.5 bit/param)

| コンポーネント | パラメータ数 | IQ2_M サイズ |
|--------------|:----------:|:----------:|
| Attention (per layer) | Q:63M + K:5.2M + V:5.2M + O:63M = **136M** | ~43MB |
| Dense FFN (per layer) | up:63M + gate:63M + down:63M = **189M** | ~59MB |
| Expert FFN (160 experts, per layer) | 160 × (up:7.9M + gate:7.9M + down:7.9M) = **3,789M** | ~1,184MB |
| Shared Expert (per layer) | up:63M + gate:63M + down:63M = **189M** | ~59MB |
| Gate (per layer) | 5120×160 = **0.8M** | ~0.3MB |
| **MoE レイヤー合計 (1層)** | **~4,115M** | **~1,286MB** |
| **Dense レイヤー合計 (1層)** | **~325M** | **~102MB** |
| 全 MoE レイヤー (89層) | ~366B | ~112GB |
| 全 Dense レイヤー (3層) | ~975M | ~306MB |
| Embedding + Output | ~777M × 2 = ~1.55B | ~485MB |
| **モデル合計 (推定)** | **~368B** | **~113GB** |

> **実測**: GGUF ファイル 3分割、合計約 40GB (IQ2_M)

### 1.4 MoE の VRAM 特性

**決定的な事実**: 各 MoE レイヤーの Expert 重みは全パラメータの 92% を占めるが、
推論時には 160 Expert 中 **8 個のみアクティブ** (5%)。

```
Expert 重み (160 experts):    ~1,184 MB/layer → 推論時は 8/160 = 5% のみ使用
Attention + Shared Expert:    ~102 MB/layer   → 100% 使用
```

これは Expert Parallelism が他のアーキテクチャより遥かに有効であることを意味する。
各 GPU が全 Expert の一部のみを保持すれば、VRAM 使用量を劇的に削減できる。

---

## 2. Layer Split (-sm layer) の限界分析

### 2.1 根本原因: Generation の直列実行

`ggml_backend_sched_compute_splits()` (`ggml/src/ggml-backend.cpp:1443`) のコード:

```cpp
for (int split_id = 0; split_id < sched->n_splits; split_id++) {
    // 1. 入力テンソルをコピー (前の split backend → 現在の split backend)
    for (int input_id = 0; input_id < split->n_inputs; input_id++) {
        ggml_backend_tensor_copy(input, input_cpy);  // 同期コピー
    }
    // 2. グラフ計算 (1 GPU ずつ)
    ggml_backend_graph_compute_async(split_backend, &split->graph);
    // 3. イベント記録 → 次の split へ
}
```

**Generation (batch=1) では各 GPU の計算が完全に直列**:
- GPU0 計算 → GPU0→GPU1 転送 → GPU1 計算 → GPU1→GPU2 転送 → ... → GPU(N-1) 計算
- **合計時間 = Σ(計算時間) + Σ(転送時間)** — GPU を追加しても合計計算時間は減らない

### 2.2 実測データからの逆算

**11GPU (7C+4R) 実績**: tg = 6.8 t/s → 1トークン ≈ 147ms

```
92 layers / 11 GPU ≈ 8.4 layers/GPU
147ms / 92 layers ≈ 1.6 ms/layer (計算 + 転送)
```

内訳推定:
- 計算: ~1.2 ms/layer (MoE layer: Attention + 8 Expert FFN + Shared FFN)
- 転送: ~0.4 ms/layer (hidden state 5120 × 2 bytes = 10KB、RDMA latency ≈ 50-100μs + overhead)

### 2.3 16GPU への拡張予測

```
16 GPU: 92 layers / 16 ≈ 5.75 layers/GPU
計算時間: 変わらない (92 × 1.2ms = 110ms)
転送回数: 15 回 (11GPU) → 15 回 (16GPU の場合も同数のレイヤー間転送)
転送時間: ノード間 RDMA のレイテンシ増加分を考慮

予測: 8-9 t/s (16GPU)
```

**Layer split の根本的限界**: GPU を何台追加しても、Generation 速度は各レイヤーの計算時間の合計に律速される。高速化は「レイヤーあたりの計算を複数 GPU で分担する」以外にない。

### 2.4 n_copies パイプライン (未活用)

`ggml_backend_sched` は `parallel=true` で `n_copies = 4` (GGML_SCHED_MAX_COPIES) を設定し、
連続バッチ間でパイプライン実行する機能を持つ (`ggml-backend.cpp:1653`)。

ただし、これは **Prompt 処理 (batch > 1) でのみ有効** で、Generation (batch=1) では効果がない。
RDMA バックエンドは `event_record` / `event_wait` が未実装 (`NULL`) のため、
現在はパイプラインが発動しない。

---

## 3. Row Split (-sm row) の限界分析

### 3.1 通信量の見積もり

Row split (テンソル並列) では各レイヤーの行列乗算を複数 GPU に分割し、
AllReduce で部分結果を集約する。

```
AllReduce データ量 (per layer):
  - Attention output: hidden_size × sizeof(f16) = 5120 × 2 = 10,240 bytes
  - FFN output: hidden_size × sizeof(f16) = 5120 × 2 = 10,240 bytes
  - 合計: ~20 KB/layer (非常に小さい)
```

### 3.2 問題はレイテンシ

通信量は小さいが、**レイテンシが支配的**:

| 通信パス | レイテンシ (片道) | 帯域幅 |
|---------|:-----------:|:-----:|
| NVLink (P100) | 存在しない | — |
| PCIe 3.0 (GPU↔CPU) | ~2-5 μs | 15.75 GB/s |
| PCIe P2P (GPU↔GPU, PIX) | ~2-10 μs | 10-11 GB/s |
| PCIe P2P (GPU↔GPU, PHB) | ~5-20 μs | 8-10 GB/s |
| RDMA (ノード間) | ~2-5 μs (RDMA Write) | 12.5 GB/s (100GbE) |

**P100 PCIe に NVLink はない**が、PCIe P2P (BAR1マッピング) による GPU-GPU 直接転送が可能
(1号機: 全7GPU間OK、2号機: GPU0-1-2間OK、GPU3はクロスソケットで不可)。
ただし llama.cpp の row split は `NCCL_P2P_DISABLE=1` 環境では NCCL の P2P を使用しない。
独自の memcpy ベース通信であり、`cudaMemcpyPeer` 経由で PCIe P2P が使われる。
AllReduce 1 回あたり 20-100μs (PCIe P2P、往復) とすると:

```
AllReduce cost (g GPU, PCIe P2P):
  - Ring AllReduce: 2 × (g-1)/g × 20KB, latency = 2 × (g-1) × 10-20μs (PIX/PHB)
  - g=2: ~20-40μs, g=4: ~60-120μs, g=8: ~140-280μs, g=16: ~300-600μs

Per-layer overhead with 2 AllReduce ops:
  - g=2: ~40-80μs, g=4: ~120-240μs, g=8: ~280-560μs, g=16: ~600-1200μs
```

### 3.3 llama.cpp の Row Split 実装

`ggml-cuda.cu:812+` の `ggml_backend_cuda_split_buffer` は:
- 各 GPU にテンソルの行方向スライスを配置
- **NCCL は不使用** — 独自の memcpy ベースの通信
- ローカル GPU 間のみ対応 (リモート GPU 非対応)
- PCIe P2P は P100 で利用可能 (PIX/PHB トポロジ)。2号機 GPU3 のみクロスソケット (SYS) で P2P 不可

### 3.4 Row Split 性能予測

```
計算時間 (per layer, g GPU):
  - Attention: ~0.8ms / g (行列乗算はリニアにスケール)
  - FFN (MoE): ~0.4ms / g (アクティブ Expert のみ)

AllReduce 時間 (per layer, 2 ops, PCIe P2P):
  - g=2: ~0.06ms, g=4: ~0.18ms, g=8: ~0.42ms

合計 (per layer):
  - g=1:  1.2ms (baseline)
  - g=2:  0.66ms (計算0.6ms + AllReduce 0.06ms)
  - g=4:  0.48ms (計算0.3ms + AllReduce 0.18ms)
  - g=8:  0.57ms (計算0.15ms + AllReduce 0.42ms) ← g=4 より悪化
```

**結論**: PCIe P2P が利用可能な P100 PCIe では、g=4 が最適 (0.48ms/layer)。g=2 も良好 (0.66ms)。g≥8 では AllReduce レイテンシ増加で悪化するが、NVLink なしでも g=4 まではスケールする。

---

## 4. ハイブリッドアプローチの定量分析

### 4.1 性能モデル

```
T_token = L × t_compute(g) + L × c × t_allreduce(g) + (L/g_local - 1) × t_transfer

L         = 92 (レイヤー数)
g         = TP グループサイズ (テンソル並列度)
c         = AllReduce 回数/layer (= 2: attention_out + ffn_out)
t_compute(g) = 1.2ms / g (per layer)
t_allreduce(g) = f(g) (上記の見積もり)
t_transfer = inter-stage transfer latency (hidden state, ~0.1ms)
N         = 総 GPU 数
PP stages = N / g
```

### 4.2 構成別予測

| 構成 | TP (g) | PP stages | 予測 t_token | 予測速度 | 確度 | 根拠 |
|------|:------:|:---------:|:----------:|:-------:|:----:|------|
| Pure PP (N=16) | 1 | 16 | 112ms | **8.9 t/s** | Very High | 現在の11GPU実績から外挿 |
| **TP=2 + PP=8** | **2** | **8** | **81ms** | **12.3 t/s** | **Medium-High** | AllReduce 2回×0.1ms/layer が追加 |
| TP=4 + PP=4 | 4 | 4 | 82ms | 12.2 t/s | Medium | AllReduce コスト増大で TP=2 と大差なし |
| TP=8 + PP=2 | 8 | 2 | 133ms | 7.5 t/s | Medium-Low | AllReduce が支配的、PP より悪い |
| Pure TP (N=16) | 16 | 1 | 290ms | 3.4 t/s | Low | AllReduce 15段が壊滅的 |

### 4.3 計算詳細 (TP=2 + PP=8)

```
TP group: 2 GPU (同一ノード内ペア)
PP stages: 8 (= 16 / 2)
Layers/stage: 92 / 8 ≈ 11.5 layers

Per-layer time:
  Attention compute: 0.8ms / 2 = 0.4ms
  MoE compute: 0.4ms / 2 = 0.2ms  (※ MoE は Expert 分割が自然)
  AllReduce ×2: 2 × 0.1ms = 0.2ms (g=2, ペア交換は高速)
  Total: ~0.8ms/layer

Total: 92 × 0.8ms + 7 × 0.1ms (inter-stage) ≈ 74ms → 13.5 t/s

保守的見積もり (overhead 10%): ~81ms → 12.3 t/s
```

**重要**: TP=2 の AllReduce はペア間の単純な交換で済むため、Ring AllReduce 不要。
`cudaMemcpyPeer` (同一ノード内) 1回で完了し、レイテンシは最小。

---

## 5. 各アプローチの実現可能性評価

### 5-A. Expert Parallelism (EP) — GLM-4.7 に最適

**概要**: 160 Expert を複数 GPU に分散配置し、トークンのルーティング結果に基づいて
該当 GPU でのみ Expert FFN を実行。結果を All-to-All 通信で集約。

**GLM-4.7 での利点**:
- 160 Expert 中 8 のみアクティブ → 各 GPU は 20 Expert のみ保持 (8GPU の場合)
- **VRAM 削減**: MoE 重みが 1/8 に → Q4_K_M でも 16GPU に収まる可能性
- 通信量: hidden_state (5120 × 2B = 10KB) の All-to-All — AllReduce より効率的
- Expert FFN が小さい (1536 次元) ため、計算は低レイテンシ

**性能予測**:
```
Per-layer time (8 GPU EP):
  Attention: 1.2ms (分割なし、各 GPU でフル計算)
  → TP=2 併用なら: 0.8ms
  Expert routing + All-to-All: ~0.2ms
  Expert FFN (8/20 experts on each GPU): ~0.05ms × (8/160 × 20) = ~0.05ms
  Shared FFN: ~0.15ms
  Total: ~1.2ms/layer (EP only) or ~0.85ms/layer (EP + TP=2)

予測: 14-20 t/s (EP + TP=2 + PP)
```

**実装複雑度**: 高
- Expert 重みのパーティショニングロジック
- All-to-All 通信 (ggml に未実装)
- Expert ルーティング結果に基づく動的ディスパッチ
- `build_moe_ffn()` (`llama-graph.cpp:1058`) のグラフ構築を大幅変更
- RDMA バックエンドで All-to-All の実装
- 上流の `ggml_op` に `MUL_MAT_ID` の分散版が必要

**実装期間**: 3-5 週

---

### 5-B. TP=2 (ノード内) + PP — 最もバランスの良い選択

**概要**: 同一ノード内の 2 GPU でテンソル並列、ノード間はパイプライン並列。

**利点**:
- AllReduce が 2 GPU ペアの単純交換 → 低レイテンシ
- NCCL 不要 (`cudaMemcpyPeer` または host-staging で十分)
- 上流 PR #19378 の設計と整合 (backend-agnostic TP)
- 既存の split buffer 機構 (`ggml-cuda.cu:812+`) を参考にできる

**性能予測**: 12-13 t/s (前述の計算)

**実装方法**:
1. RDMA バックエンドは変更不要 — TP は同一ノード内で完結
2. meta backend (PR #19378) を取り込み、ノード内 TP=2 + ノード間 RDMA PP
3. または、CUDA split buffer を 2 GPU ペア用にカスタマイズ

**懸念事項**:
- PR #19378 は MoE モデルを**明示的に未サポート** — Dense attention 部分のみ
  - GLM-4.7 では Attention は TP 可能だが、MoE FFN は EP が必要
  - Shared FFN は TP 可能
- PCIe P2P (BAR1) が利用可能なため `cudaMemcpyPeer` は GPU 直接転送。ただし NVLink 比で帯域は劣る (~10 GB/s vs ~35 GB/s)
- `NCCL_P2P_DISABLE=1` は NCCL の P2P を無効化するが、CUDA ネイティブの `cudaMemcpyPeer` は PCIe P2P を使用可能

**実装複雑度**: 中-高
**実装期間**: 2-4 週

---

### 5-C. Speculative Decoding — 直交的アプローチ

**概要**: 小さな Draft モデルで k トークンを予測し、大モデル (GLM-4.7) で一括検証。
検証パスは batch > 1 → パイプライン効率向上。

**llama.cpp での実装状況**:
- `common/speculative.cpp` に完全な実装あり
- 対応タイプ: draft model, EAGLE3, ngram (simple/map/mod/cache)
- `llama-cli` で `--draft-model` オプションで使用可能
- Vocabulary 互換性チェック済み (`common_speculative_are_compatible()`)

**GLM-4.7 への適用**:
- Draft モデル候補: GLM-4.7 自体の Dense 部分 (3 leading dense layers) — ただし不十分
- 外部 Draft モデル: GLM-4-9B (9B params, 同じトークナイザー `glm4`) が候補
- ngram ベースの speculative decoding は Draft モデル不要 — 即座に使用可能

**性能予測**:
```
Draft acceptance rate (typical): 60-80% for k=4
Speedup factor: ~1.5-2.5× (タスクとドメインに強く依存)
Base: 8.9 t/s (16GPU PP) → 予測: 13-22 t/s

ただし:
- MoE モデルは Expert routing が確定的でないため、acceptance rate が低い可能性
- GLM-4.7 の IQ2_M 量子化自体が精度に影響 → draft の質に影響
```

**利点**:
- **RDMA バックエンド変更不要** — 既存の llama.cpp 機能を使うだけ
- 実装は Draft モデルの選定と設定のみ
- ngram speculative は追加モデル不要

**懸念事項**:
- GLM-4.7 互換の Draft モデルの入手性
- MoE モデルでの acceptance rate の不確実性
- Draft モデル用の追加 VRAM が必要 (GPU メモリが逼迫している場合)

**実装複雑度**: 低
**実装期間**: 1-2 週 (ngram ベースなら数時間)

---

### 5-D. Client-side Parallel Dispatch — 低リスク

**概要**: 現在の `ggml_backend_sched_compute_splits()` は全 split を**逐次**処理する。
全デバイスの graph_compute を先に発行し、get_tensor を後でまとめて待つようにする。

**現状の問題** (parallel_compute_dispatch レポートで詳述):
```
現在: D0 async → D0 get_tensor (wait 12ms) → D1 async → D1 get_tensor (wait 12ms) → ...
理想: D0 async → D1 async → D2 async → D3 async → D0 get → D1 get → D2 get → D3 get
```

サーバー側の parallel compute dispatch は**実装済み**だが、クライアント側の
`ggml_backend_sched` が graph_compute → get_tensor を逐次発行するのがボトルネック。

**性能予測**:
```
現状 (4 RDMA devices, 逐次):
  4 × (async 0.7ms + get_tensor 12ms) = 50.8ms/round

理想 (4 RDMA devices, 並列):
  async: 4 × 0.7ms = 2.8ms (逐次送信)
  compute: max(12ms, 12ms, 12ms, 12ms) = 12ms (並列計算)
  get_tensor: 4 × 2ms = 8ms (逐次取得)
  Total: ~23ms/round

改善: 50.8ms → 23ms → Generation ~9-10 t/s (from ~6.8 t/s)
```

**実装方法**:
- `ggml_backend_sched_compute_splits()` を改修し、全 split の graph_compute を先に発行
- RDMA バックエンドの `graph_compute_async` が完了を待たない (既に async compute で実装済み)
- get_tensor で同期 (サーバーの compute_pending が解消されるまで待つ)

**懸念事項**:
- `ggml_backend_sched` は llama.cpp コアコードであり、変更の影響範囲が大きい
- 他のバックエンド (CUDA, Metal) との互換性を維持する必要がある
- split 間のデータ依存関係 (前の split の出力が次の split の入力) があるため、
  **同一パイプラインステージ内でのみ並列化が可能**
  - 実際には layer split では各 split は前段の出力に依存 → 並列化困難
  - **並列化が有効なのは「異なるデバイスが同一レイヤーの異なる部分を計算する」場合のみ**

**再評価**: Layer split 構成では split 間に依存関係があるため、dispatch の並列化は
同一レイヤー内の TP 的分割がない限り効果が薄い。
ただし、RDMA の async compute と組み合わせることで、compute と transfer の
オーバーラップは実現可能。

**実装複雑度**: 中
**実装期間**: 1-2 週

---

### 5-E. RDMA Event Support — Prompt 処理のパイプライン化

**概要**: RDMA バックエンドに `event_record` / `event_wait` を実装し、
`ggml_backend_sched` の `n_copies > 1` パイプラインを有効化する。

**現状**: RDMA バックエンドの event 関連は全て `NULL`:
```cpp
// ggml-rdma.cpp:1555
/* .event_record            = */ NULL,
/* .event_wait              = */ NULL,
// ggml-rdma.cpp:1669
/* .event_synchronize    = */ NULL,
```

`n_copies > 1` の場合、スケジューラは複数のテンソルコピーを作成し、
連続バッチの入力転送と前バッチの計算をオーバーラップさせる (`ggml-backend.cpp:1352-1372`)。

**効果の範囲**:
- **Prompt 処理 (batch > 1)**: バッチ間のパイプライン化で高速化の可能性
- **Generation (batch=1)**: パイプライン化する対象がないため**効果なし**

**性能予測**:
```
Prompt (pp128): 6.4 t/s → ~8-10 t/s (推定 30-50% 改善)
Generation: 変化なし
```

**実装複雑度**: 中
**実装期間**: 1 週

---

### 5-F. Sequence Parallelism — 非推奨

**概要**: 入力シーケンスを分割して複数デバイスで並列に Attention を計算。

**Generation (seq_len=1) には効果なし** — 分割するシーケンスが存在しない。
Prompt 処理では理論的に有効だが:
- Attention の ring 分割が必要 (Flash Attention の改修)
- KV キャッシュの分散管理
- 実装複雑度が極めて高い

**非推奨**: 効果が限定的で、実装コストが見合わない。

---

## 6. llama.cpp 上流の関連動向

### 6.1 PR #19378: Backend-agnostic Tensor Parallelism

| 項目 | 詳細 |
|------|------|
| **ステータス** | OPEN (2026-02-05 作成) |
| **著者** | JohannesGaessler |
| **概要** | "meta" バックエンドで複数 GPU のテンソル並列を統一的に扱う |
| **手法** | `--split-mode tensor` で有効化。重みを dim1 → dim0 で分散 |
| **性能** | 2× RTX 4090 (LLaMA 3 8B Q4_0): pp512=6306 t/s, tg128=102 t/s |
| **制限** | 1-8 GPU のみ、FlashAttention 必須、**MoE 未サポート** |
| **API 拡張** | `set_tensor_2d_async`, `get_tensor_2d_async`, `shfl_tensor_async`, `allreduce_tensor_async` (NCCL optional) |

**GLM-4.7 への影響**:
- Dense attention 部分は PR #19378 の TP が適用可能
- MoE FFN は明示的に未サポート → Expert Parallelism は別途実装が必要
- RDMA バックエンドが新 API (`allreduce_tensor_async` 等) を実装すれば、
  リモート GPU での TP も理論的に可能

### 6.2 Issue #13083: Tensor Parallelism over RPC

| 項目 | 詳細 |
|------|------|
| **ステータス** | CLOSED (2025-06-19) |
| **概要** | `--split-mode row` を RPC 経由で動作させる要望 |
| **開発者回答** | 「virtual backend」で複数バックエンドをラップする設計を推奨 |
| **コミュニティ** | LeaveNhA が RPC 実装に取り組んでいると報告 (2025-09) |

### 6.3 Issue #13314: Layer-to-device Assignment

| 項目 | 詳細 |
|------|------|
| **ステータス** | CLOSED |
| **概要** | CPU レイヤーの配置順序を制御するオプションの要望 |
| **関連性** | レイヤー割り当ての柔軟化は MoE の Expert 配置にも有用 |

### 6.4 MoE 関連の上流動向

| PR/Issue | 概要 | 関連性 |
|----------|------|--------|
| PR #11397 (merged) | `--override-tensor` — テンソル単位でバッファ指定 (`-ot exps=CPU`) | Expert のデバイス配置制御に有用 |
| Issue #11532 | MoE の動的 Expert ロード (アクティブ Expert のみ GPU) | Expert Parallelism の基礎 |
| Issue #11333 | NUMA-aware MoE Expert Allocation | マルチソケット環境での Expert 配置 |
| Discussion #18049 | GPU layers/tensor split の自動化 | MoE 向け Dense 優先配置 |
| Discussion #11784 | Expert は単一デバイスに配置 (デフォルト) | 現在の layer split の動作 |

---

## 7. 推奨アクション (優先度順)

### 優先度 1: Speculative Decoding (ngram ベース)

| 項目 | 詳細 |
|------|------|
| **予測効果** | 1.3-2.0× 高速化 (8.9 → 12-18 t/s) |
| **実装コスト** | 極小 (既存機能の設定のみ) |
| **リスク** | 低 (コード変更なし) |
| **期間** | 数時間 |
| **行動** | `llama-cli` に `--spec-type ngram_simple` を追加して即座にテスト可能 |

まず ngram ベースを試し、効果が確認できれば Draft モデル (GLM-4-9B 等) での speculative decoding に進む。

### 優先度 2: Client-side Compute/Transfer Overlap

| 項目 | 詳細 |
|------|------|
| **予測効果** | +15-30% (6.8 → 8-9 t/s) |
| **実装コスト** | 中 (ggml_backend_sched の改修) |
| **リスク** | 中 (コアコードの変更) |
| **期間** | 1-2 週 |
| **行動** | async compute の真のオーバーラップを実現。get_tensor の遅延呼び出し |

### 優先度 3: Expert Parallelism (EP)

| 項目 | 詳細 |
|------|------|
| **予測効果** | 14-20 t/s (最大の効果) |
| **実装コスト** | 高 (新しい並列化パラダイム) |
| **リスク** | 高 (大規模な設計変更) |
| **期間** | 3-5 週 |
| **行動** | Expert 重みのパーティション + All-to-All 通信 + 動的ディスパッチ |

GLM-4.7 は 160 Expert × 小さな FFN (1536) という構造が EP に最適。
ただし llama.cpp の既存アーキテクチャからの大幅な逸脱が必要。

### 優先度 4: TP=2 (ノード内) + PP

| 項目 | 詳細 |
|------|------|
| **予測効果** | 12-13 t/s |
| **実装コスト** | 中-高 |
| **リスク** | 中 (PR #19378 との整合) |
| **期間** | 2-4 週 |
| **行動** | PR #19378 の設計を参考に、2 GPU ペアの TP を実装 |

PR #19378 が MoE をサポートするまで待つか、Dense 部分のみ TP を適用。

### 優先度 5: RDMA Event Support

| 項目 | 詳細 |
|------|------|
| **予測効果** | Prompt +30-50%、Generation 変化なし |
| **実装コスト** | 中 |
| **リスク** | 低 |
| **期間** | 1 週 |
| **行動** | `event_record`/`event_wait` を RDMA バックエンドに実装 |

Generation 改善が主目的の場合は優先度低。

---

## 8. 総合的な見通し

### 単一アプローチでの限界

| アプローチ | 予測速度 (16GPU) | 現在比 |
|-----------|:---------------:|:------:|
| Pure PP (baseline) | 8-9 t/s | 1.0× |
| + Speculative (ngram) | 12-18 t/s | 1.3-2.0× |
| + Compute overlap | 9-10 t/s | 1.0-1.1× |
| TP=2 + PP | 12-13 t/s | 1.4-1.5× |
| EP + TP=2 + PP | 14-20 t/s | 1.6-2.2× |

### アプローチの組み合わせ

Speculative decoding は他の全アプローチと**直交的に組み合わせ可能**:

| 組み合わせ | 予測速度 | 確度 |
|-----------|:-------:|:----:|
| PP + Speculative (ngram) | 12-18 t/s | Medium |
| TP=2 + PP + Speculative | 16-26 t/s | Low-Medium |
| EP + PP + Speculative | 20-40 t/s | Low |

### 推奨ロードマップ

```
Week 0 (今すぐ): ngram speculative decoding テスト
  → 効果測定、MoE での acceptance rate 確認
  → 効果があれば Draft モデル探索

Week 1-2: Compute/Transfer overlap (async compute の真の活用)
  → get_tensor の遅延化による compute オーバーラップ
  → ggml_backend_sched の非侵襲的パッチ

Week 3+: EP または TP=2 の検討 (上記の結果を踏まえて判断)
  → Speculative + PP で目標速度に達するなら不要の可能性
  → 達しない場合は EP が最も効果的
```

---

## 9. 参考文献・参照コード

### ソースコード

| ファイル | 行 | 内容 |
|---------|:---:|------|
| `ggml/src/ggml-backend.cpp` | 1443 | `ggml_backend_sched_compute_splits()` — split の逐次実行ループ |
| `ggml/src/ggml-backend.cpp` | 1629-1665 | `ggml_backend_sched_new()` — `n_copies` / `parallel` の設定 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 812+ | CUDA split buffer (row split 実装) |
| `src/llama-model.cpp` | 2535 | `get_layer_buft_list()` — レイヤーのデバイス割り当て |
| `src/models/glm4-moe.cpp` | 1-170 | GLM-4.7 MoE forward pass (全体) |
| `src/llama-graph.cpp` | 1058-1192 | `build_moe_ffn()` — MoE FFN グラフ構築 |
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 1555-1556, 1669 | RDMA event (未実装 = NULL) |
| `common/speculative.cpp` | 1-80 | Speculative decoding 実装 |

### 上流 PR/Issue

| リンク | タイトル |
|--------|---------|
| [PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378) | ggml: backend-agnostic tensor parallelism |
| [Issue #13083](https://github.com/ggml-org/llama.cpp/issues/13083) | Tensor parallelism over RPC |
| [Issue #13314](https://github.com/ggml-org/llama.cpp/issues/13314) | Layer-to-device assignment |
| [PR #11397](https://github.com/ggml-org/llama.cpp/pull/11397) | `--override-tensor` (テンソルバッファ指定) |
| [Issue #11532](https://github.com/ggml-org/llama.cpp/issues/11532) | MoE 動的 Expert ロード |
| [Discussion #11784](https://github.com/ggml-org/llama.cpp/discussions/11784) | Expert のデバイス配置 |

### ハードウェア仕様

| コンポーネント | 仕様 |
|--------------|------|
| GPU | Tesla P100 PCIe 16GB × 16 (2ノード × 8) |
| NIC | ConnectX-4 100GbE (RDMA, GPUDirect 対応) |
| PCIe | 3.0 x16 (15.75 GB/s per direction) |
| NVLink | なし (P100 PCIe 版) |
| P2P | PCIe P2P 利用可能 (NCCL P2P は `NCCL_P2P_DISABLE=1` で無効) |
| MaxPayload | P100: 256B (hard limit), ConnectX-4: 256B |
