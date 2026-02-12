# サーバー側並列コンピュート・ディスパッチ実装レポート

- **実施日時**: 2026年2月11日 21:42
- **ワークツリー**: `.worktree/rdma-parallel-compute`

## 前提・目的

### 背景

過去11件の通信レベルのマイクロ最適化 (doorbell batching, max_inline, pre-post recv, combined send, hotpath memory, xdev cache) はいずれも e2e の Generation 速度に改善をもたらさなかった。システムは GPU 計算バウンドであり、唯一の構造的ボトルネックは RDMA サーバーが 4 デバイスの graph_compute を逐次処理していることだった。

### 目的

サーバー側で GPU 計算をデバイス別ワーカースレッドに並列ディスパッチし、4 デバイスの compute 時間を max(11ms) ≈ 11ms に短縮する。期待される Generation 速度改善は 20-35% (6.8 → 8-9 t/s)。

### 参考レポート

- `report/2026-02-10_222348_doorbell_batching_max_inline.md` — 通信最適化が e2e 効果なしの結論
- `report/2026-02-11_040659_prepost_recv_rnr_nak.md` — RNR NAK の修正
- `report/2026-02-11_095500_hotpath_memory_optimization.md` — ホットパスメモリ最適化
- `report/2026-02-11_133000_xdev_cache_optimization.md` — xdev キャッシュ最適化

## 実装内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | +371行 / -32行 |

### 主要コンポーネント

1. **`compute_worker` クラス**: デバイス別ワーカースレッド。mutex + condition variable でタスク通知・完了通知を行う。`cudaSetDevice()` でデバイスを固定。

2. **`compute_dispatcher` クラス**: 全デバイスのワーカーを管理。`dispatch(device, task)` で非ブロッキングディスパッチ、`wait_device(device)` でブロッキング完了待ち。

3. **`rdma_server` 新規メソッド**:
   - `flush_only_recompute()` — flush のみ実行、デバイス ID を返す
   - `flush_only_compute_update()` — flush のみ実行、compute_data を分離して返す
   - `execute_recompute(device)` — ワーカーから呼ばれる graph_recompute
   - `execute_compute_update(data)` — ワーカーから呼ばれる graph_compute_update
   - `execute_graph_compute(data)` — ワーカーから呼ばれる graph_compute
   - `get_buffer_device(buffer)` — バッファからデバイス ID を取得

4. **`handle_client` 修正**: ASYNC コマンドで flush → ワーカーにディスパッチ → 即座に次の recv。GET_TENSOR でデバイスの compute 完了を待機。

### 環境変数

| 環境変数 | 説明 | デフォルト |
|---------|------|----------|
| `GGML_RDMA_PARALLEL_COMPUTE` | `0` で並列ディスパッチを無効化 | 有効 |

### スレッド安全性

| リソース | アクセスパターン | 安全性 |
|---------|----------------|--------|
| `stored_graphs_[device]` | デバイス別ワーカーのみ | 安全 (別インデックス) |
| `backends_[device]` | デバイス別ワーカーのみ | 安全 (CUDA デバイスコンテキスト独立) |
| `buffer_device_map_` | 全スレッドから読み取り | 安全 (compute 中に変更なし) |
| RDMA QP (send/recv) | メインスレッドのみ | 安全 (ワーカーは QP 非接触) |

## 再現方法

### ビルド・デプロイ

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-parallel-compute
rm -rf build
cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=60
cmake --build build -- -j $(nproc)
```

### テスト (qwen2.5-0.5b 回帰テスト)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

### テスト (GLM-4.7 IQ2_M, 11GPU)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-bench -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

## 検証結果

### qwen2.5-0.5b (1 CUDA + 1 RDMA)

| 構成 | pp128 (t/s) | tg32 (t/s) |
|------|:-----------:|:----------:|
| PARALLEL_COMPUTE=0 | 2646 | 109 |
| PARALLEL_COMPUTE=1 | 2754 | 108 |
| 差分 | +4% | -1% |

1 リモートデバイスのため並列化効果なし。回帰なしを確認。

### GLM-4.7 IQ2_M (7 CUDA + 4 RDMA, llama-bench)

| 構成 | pp128 (t/s) | tg32 (t/s) |
|------|:-----------:|:----------:|
| PARALLEL_COMPUTE=0 | 23.25 | 7.72 |
| PARALLEL_COMPUTE=1 | 23.19 | 7.73 |
| PARALLEL_COMPUTE=1 + PROFILE | 23.27 | 7.75 |

**Generation 速度に有意な改善なし。**

### GLM-4.7 IQ2_M (7 CUDA + 4 RDMA, llama-cli × 3)

| テスト | pp (t/s) | tg (t/s) | 出力 |
|-------|:--------:|:--------:|:----:|
| 1. "The capital of France is" | 5.9 | 6.9 | 正常 |
| 2. "Explain what RDMA is" | 8.8 | 6.9 | 正常 |
| 3. "Write a haiku about parallel computing" | 8.1 | 5.9 | 正常 |

全テストで正常な推論出力を確認。安定性に問題なし。

## 分析: なぜ Generation 速度が改善しなかったか

### サーバー側プロファイリング結果

定常状態での典型的なサーバーログパターン:

```
FLUSH_AND_RECOMPUTE_ASYNC D0: total=91ms (recv=91ms) [parallel]
GET_TENSOR wait D0: 13.6ms
FLUSH_AND_RECOMPUTE_ASYNC D1: total=1.9ms (recv=0.3ms) [parallel]
GET_TENSOR wait D1: 12.2ms
FLUSH_AND_RECOMPUTE_ASYNC D2: total=1.9ms (recv=0.3ms) [parallel]
GET_TENSOR wait D2: 12.6ms
FLUSH_AND_RECOMPUTE_ASYNC D3: total=2.9ms (recv=0.2ms) [parallel]
GET_TENSOR wait D3: 11.1ms
```

### 根本原因: クライアント側スケジューラの逐次処理

llama.cpp の `ggml_backend_sched` が各デバイスの graph_compute → get_tensor を逐次呼び出す:

```
CUDA0 compute → CUDA1 compute → ... → CUDA6 compute  (← ~80ms)
RDMA0 async_send → RDMA0 get_tensor (wait 12ms)       (← D0 逐次)
RDMA1 async_send → RDMA1 get_tensor (wait 12ms)       (← D1 逐次)
RDMA2 async_send → RDMA2 get_tensor (wait 12ms)       (← D2 逐次)
RDMA3 async_send → RDMA3 get_tensor (wait 11ms)       (← D3 逐次)
```

D0 の async_send 直後に GET_TENSOR が来るため、D1-D3 の async_send はまだ送信されていない。各デバイスの compute は完全に逐次: **4 × ~12ms = 48ms**。

### サーバー側並列化は正しく機能している

- ASYNC コマンド処理: flush + dispatch で **1.9-2.9ms** (以前は flush + compute で **~13ms**)
- ワーカーへのディスパッチ自体は成功している
- しかし、クライアントが即座に GET_TENSOR を送信するため、ワーカーは `wait_device()` で待機

### 改善に必要な変更

サーバー側並列化の効果を引き出すには、**クライアント側の変更**が必要:

1. **全 RDMA デバイスの graph_compute を先に送信**: スケジューラが全 RDMA バックエンドの graph_compute を送信してから、get_tensor を処理する順番にする
2. **`ggml_backend_sched` の変更**: graph_compute と get_tensor の呼び出し順序を制御するフック

これは llama.cpp 上流コードの変更を伴うため、RDMA バックエンドだけでは完結しない。

## 結論

- **サーバー側並列コンピュート・ディスパッチは正しく実装・動作している**
- **回帰なし**: 全テストでベースラインと同等の性能
- **e2e 効果なし**: クライアント側スケジューラの逐次処理により、4 デバイスの compute が並列化されない
- **基盤としての価値**: クライアント側を改善すれば、サーバー側は既に並列化対応済み
- `GGML_RDMA_PARALLEL_COMPUTE=0` で従来動作にフォールバック可能
