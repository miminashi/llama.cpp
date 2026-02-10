# RDMA バックエンド性能改善: async compute + initiator_depth

- **実施日時**: 2026年2月10日 21:48

## 前提・目的

RDMA バックエンドは Prompt 処理で RPC 比 +19% の優位性を持つが、Generation 速度で -10% 劣位 (6.8 vs 7.5 t/s)。
主因は graph_compute の同期レスポンス待ちによるラウンドトリップオーバーヘッド。
RPC は graph_compute をレスポンス待ちなし（fire-and-forget）で実装しており、この差が Generation 速度に直結している。

### 参考レポート

- `report/2026-02-09_134640_gdr_budget_and_multi_rdma_fix.md` — GDR バジェット + Multi-RDMA 修正
- `report/2026-02-09_084919_github_deploy_key_setup.md`

## 実装内容

### 1. initiator_depth / responder_resources: 1 → 16

RDMA Read/Write の同時発行可能数を 1 から 16 に引き上げ。ConnectX-4 は `max_qp_rd_atom=16` をサポート。

**変更箇所**: `ggml/src/ggml-rdma/rdma-transport.cpp`
- L193-194 (accept): `initiator_depth=1, responder_resources=1` → `16, 16`
- L327-328 (connect_qp): 同上

### 2. Fire-and-forget graph_compute

RPC と同様に、graph_compute のレスポンス待ちを廃止。

**原理**: サーバーのコマンドループは逐次処理。fire-and-forget で送った graph_compute の後に get_tensor を送ると、サーバー側では compute 完了後に get_tensor を処理する。レスポンスを省略しても暗黙的に同期される。

**実装詳細**:

| 変更 | ファイル | 内容 |
|------|---------|------|
| ASYNC コマンド追加 | ggml-rdma.cpp | `RDMA_CMD_FLUSH_AND_RECOMPUTE_ASYNC`, `_COMPUTE_UPDATE_ASYNC`, `_GRAPH_COMPUTE_ASYNC` |
| `compute_pending` フラグ | rdma-transport.h | `std::atomic<bool> compute_pending_` を `rdma_connection` に追加 |
| `send_rdma_cmd_async` | ggml-rdma.cpp | send のみで recv を待たない。`compute_pending_` を set |
| graph_compute 変更 | ggml-rdma.cpp | 3パス (recompute/update/full) すべてで ASYNC コマンドを使用 |
| get_tensor 同期保証 | ggml-rdma.cpp | `compute_pending_` 時は RDMA Read をスキップし Send/Recv パスで暗黙同期 |
| send_rdma_cmd/with_rsp | ggml-rdma.cpp | recv 成功時に `compute_pending_` を clear |
| サーバー側ハンドラ | ggml-rdma.cpp | ASYNC コマンドは `send_rsp_empty()` を呼ばない |
| 環境変数 | ggml-rdma.cpp | `GGML_RDMA_ASYNC_COMPUTE=0` で無効化可能 (デフォルト有効) |

### 3. PCIe MaxReadReq 確認

| デバイス | MaxPayload (DevCap) | MaxPayload (DevCtl) | MaxReadReq |
|---------|:---:|:---:|:---:|
| ConnectX-4 (0b:00.0) | 512B | 256B | 512B |
| P100 GPU (05:00.0) | 256B | 256B | 512B |

P100 の MaxPayload が 256B 上限のため、PCIe 設定調整による大きな改善は見込めない。

## 検証結果

### qwen2.5-0.5b 回帰テスト (2 GPU: CUDA0 + RDMA0)

| モード | pp128 (t/s) | tg32 (t/s) |
|--------|:-:|:-:|
| async (デフォルト) | 2,741 | 147 |
| sync | 2,815 | 148 |

小規模モデルでは差なし (予想通り)。

### GLM-4.7 IQ2_M 11GPU (7C+4R) — 最終検証

| モード | Prompt (t/s) | Generation (t/s) | graph_compute avg |
|--------|:-:|:-:|:-:|
| async (デフォルト) | 6.3 | 6.6 | **0.7 ms** |
| sync | 6.3 | 6.6 | **17.2 ms** |

#### プロファイリング比較 (200 calls)

| メトリクス | async | sync |
|-----------|:-----:|:----:|
| graph_compute total | 141.2 ms | 3,437.4 ms |
| graph_compute avg | **0.7 ms** | **17.2 ms** |
| get_tensor RDMA Read | 50 | 100 |
| get_tensor Send/Recv | 299 | 249 |
| get_tensor total time | 3,527.5 ms | 364.4 ms |
| get_tensor throughput | 10.5 MB/s | 101.4 MB/s |

### 分析

graph_compute 自体は async で **24.5倍高速化** (0.7ms vs 17.2ms) されているが、Generation 速度は変わらなかった。

**原因**: async モードでは `compute_pending` フラグにより RDMA Read (one-sided, GPU 直接) がスキップされ、Send/Recv パス (two-sided, サーバー経由) にフォールスルーする。Send/Recv パスはサーバーコマンドキューで暗黙的に compute 完了を待つため、graph_compute で節約した時間がそのまま get_tensor の待ち時間に移動する。

つまり、**ラウンドトリップ削減の効果は get_tensor の待ち時間に吸収される**。

### 改善が効く条件

async compute が真に効果を発揮するのは以下の場合:
1. graph_compute → set_tensor (次のトークンのweight転送) が直接続く場合 — compute と weight 転送がオーバーラップ可能
2. 複数デバイスの graph_compute を並列発行する場合 — 現在は逐次 (shared connection)
3. get_tensor で RDMA Read が使える場合 — compute_pending が false なら GPU VRAM から直接 read

### initiator_depth 16 の効果

initiator_depth 変更は RDMA Read/Write の同時発行可能数を増やすが、現在の実装ではすべての RDMA 操作がシグナル付き (signaled=true) で完了を待つため、実質的に同時発行は 1 のまま。将来の最適化（バッチ RDMA Write 等）で効果が出る可能性がある。

## 再現方法

```bash
bash scripts/rdma-build.sh local && bash scripts/rdma-deploy.sh
bash scripts/rdma-server.sh restart

# async compute 有効 (デフォルト)
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin GGML_RDMA_PROFILE=1 \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log

# async compute 無効 (比較用)
GGML_RDMA_ASYNC_COMPUTE=0 GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin GGML_RDMA_PROFILE=1 \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## 結論

- **graph_compute の fire-and-forget 自体は正しく動作** — 24.5倍高速化
- **End-to-end の Generation 速度には改善なし** — get_tensor で暗黙同期待ちが発生するため
- **initiator_depth 16** — 将来の最適化基盤
- **PCIe MaxReadReq** — P100 の MaxPayload 256B 制約により改善余地なし
- 実装は正しく、デフォルト有効のまま維持。将来的に compute / weight 転送オーバーラップが実装されれば効果が出る
