# RDMAプロトコル パフォーマンス最適化 — 0.3 t/s → 2.7 t/s (9倍改善)

- **実施日時**: 2026年2月4日 06:05

## 前提・目的

RDMA経由の推論速度が 0.3 t/s（ローカルGPU 196 t/s の約650倍遅い）であった問題を、RDMA通信プロトコルのオーバーヘッド削減により改善する。

- **背景**: 接続ライフタイムの問題は解決済みだが、推論速度が依然として極端に遅い
- **目的**: RDMA通信のボトルネック（CQイベント駆動の遅延、プロトコルの断片化、MR登録/解除のオーバーヘッド）を排除し、推論速度を改善する
- **前提条件**: 1号機（192.168.100.1）と2号機（192.168.100.2）がRDMAネットワークで接続済み、接続ライフタイム修正適用済み
- **モデル**: qwen2.5-0.5b-instruct-q4_k_m.gguf

## 参考レポート

- [RDMA接続ライフタイム修正](rdma_connection_lifetime_fix_2026-02-04_052200.md)

## 根本原因分析

### ボトルネック1: CQイベント駆動モデルの遅延

`wait_for_completion()` (rdma-transport.cpp:601-660) が毎回:
1. `poll()` syscall でイベント待ち
2. `ibv_get_cq_event()` + `ibv_ack_cq_events()` + `ibv_req_notify_cq()`
3. `ibv_poll_cq()` リトライループ: `usleep(100)` × 最大100回 = 最悪10ms/completion

### ボトルネック2: プロトコルの断片化

各コマンドで4回の個別RDMA SEND/RECV + ブロッキング完了待ちが発生:
- `send_rdma_cmd_raw()`: 3回の個別send（cmd 1B, size 8B, data NB）
- サーバー側コマンドループ: 3回の個別recv + 2回の個別send

合計: 8 blocking completions / round-trip

### ボトルネック3: グラフデータのMR登録/解除

内部バッファ 64KB を超えるグラフデータ（0.5Bモデルで約90KB）に対して毎回 `ibv_reg_mr` / `ibv_dereg_mr` が発生。

## 修正内容

### 変更1: CQビジーポーリング化 (rdma-transport.cpp)

`wait_for_completion()` をイベント駆動モデル（poll + CQイベント + usleepリトライ）から `ibv_poll_cq()` のビジーポーリングに置き換えた。`setup_qp()` から completion channel の作成を削除し、CQをcompletion channelなしで生成。

- **効果**: completion あたり ~100μs-10ms → ~1-5μs

### 変更2: メッセージ統合 (ggml-rdma.cpp)

クライアント側:
- `send_rdma_cmd_raw()`: 3回send → 2回（ヘッダー `[cmd(1B)|size(8B)]` + データ `[NB]`）
- `recv_rdma_rsp()`: 2回recv → 1回（`[rsp_size(8B)|rsp_data(NB)]` を一括受信）

サーバー側:
- コマンド受信: 3回recv → 2回（ヘッダー + データ）
- レスポンス送信: 2回send → 1回（`[rsp_size(8B)|rsp_data(NB)]` を一括送信）

合計: 8 completions/round-trip → 5 completions/round-trip

### 変更3: 内部バッファ拡大 (rdma-transport.h) + MR登録削除

`recv_buf_size` を 64KB → 1MB に拡大。クライアント側・サーバー側の per-message MR登録/解除ロジックを完全に削除し、内部バッファ使用に統一。

### 変更4: プロトコルバージョン更新 (ggml-rdma.h)

`RDMA_PROTO_MINOR_VERSION` を 0 → 1 に更新。

## 修正対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/rdma-transport.cpp` | `wait_for_completion()` ビジーポーリング化、`setup_qp()` 簡略化 |
| `ggml/src/ggml-rdma/rdma-transport.h` | `recv_buf_size` 64KB → 1MB |
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | メッセージ統合（クライアント送受信 + サーバーコマンドループ）、MR登録削除 |
| `ggml/include/ggml-rdma.h` | プロトコルバージョン更新 |

## 再現方法

### 1. 1号機でビルド

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-rdma llama-cli rdma-server -j$(nproc)
```

### 2. 2号機にコピー＆リビルド

```bash
scp ggml/src/ggml-rdma/ggml-rdma.cpp ggml/src/ggml-rdma/rdma-transport.cpp \
    ggml/src/ggml-rdma/rdma-transport.h \
    ubuntu@192.168.100.2:/home/ubuntu/projects/llama.cpp/ggml/src/ggml-rdma/
scp ggml/include/ggml-rdma.h \
    ubuntu@192.168.100.2:/home/ubuntu/projects/llama.cpp/ggml/include/
ssh ubuntu@192.168.100.2 "cmake --build ~/projects/llama.cpp/build \
    --target ggml-rdma rdma-server -j\$(nproc)"
```

### 3. rdma-server再起動（2号機）

```bash
ssh ubuntu@192.168.100.2 "pkill -f rdma-server || true"
ssh ubuntu@192.168.100.2 'nohup env LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
    /home/ubuntu/projects/llama.cpp/build/bin/rdma-server --host 0.0.0.0 --port 50051 \
    > /tmp/rdma-server.log 2>&1 &'
```

### 4. ベースライン（ローカルGPU）

```bash
LD_LIBRARY_PATH=./build/bin ./build/bin/llama-cli \
    -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -dev CUDA0 -p 'こんにちは' -n 50 --no-warmup --single-turn --simple-io
```

### 5. RDMA推論テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 \
    LD_LIBRARY_PATH=./build/bin ./build/bin/llama-cli \
    -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -dev 'RDMA0[192.168.100.2:50051]' -p 'こんにちは' -n 50 \
    --no-warmup --single-turn --simple-io
```

## 結果

### 推論速度

| 構成 | Prompt速度 | Generation速度 | 改善比 |
|------|-----------|---------------|--------|
| ローカルGPU (CUDA0) | 328.5 t/s | 196.1 t/s | ベースライン |
| RDMA 修正前 | — | 0.3 t/s | — |
| **RDMA 修正後** | **21.7 t/s** | **2.7 t/s** | **9倍改善** |

### 推論出力

入力: `こんにちは`

出力: `こんにちは！(konnichiwa) こんにちは、こんにちは。何かお手伝いできることはありますか？`

出力に破綻はなく、正常な応答を確認。

## 考察

- CQビジーポーリング化が最も効果が大きく、イベント駆動モデルの `usleep(100)` リトライが主要なボトルネックだった
- メッセージ統合により completions/round-trip が 8→5 に削減され、さらなる改善が得られた
- 1MBバッファ拡大により、0.5Bモデルのグラフデータ（約90KB）が内部バッファに収まり、per-message MR登録/解除が不要になった
- ローカルGPU（196 t/s）との差は依然として大きい（約73倍）。残る差分は主にネットワークラウンドトリップ自体のレイテンシと、round-trip回数（トークンあたり複数のコマンドが必要）に起因すると考えられる
- 今後の改善方向: RDMA Write/Readによるワンサイド通信への移行、パイプライン化、グラフキャッシュの活用（recompute）などが考えられる
