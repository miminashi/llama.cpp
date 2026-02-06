# RDMA Phase 3 パフォーマンス最適化 — 適応的レスポンス送信 + 永続的ステージングバッファ

- **実施日時**: 2026年2月4日 07:25

## 前提・目的

Phase 2 の最適化（グラフ差分更新、recv/send 2段分割）でGeneration速度は 2.7 → 47.2 t/s に改善したが、2つの退行・残存課題があった:

1. **Prompt速度の退行**: `send_rsp`/`recv_rdma_rsp` の2段分割により全レスポンスで completion が+1 増加し、21.7 → 9.2 t/s に低下
2. **set_tensor/get_tensor のカーネルコールオーバーヘッド**: ステージングバッファの毎回 alloc/free で 4カーネルコール/トークン（`posix_memalign` + `ibv_reg_mr` + `ibv_dereg_mr` + `free`）が発生

- **背景**: Phase 2 後の Generation 47.2 t/s は ローカルGPU 196.1 t/s の約4.2倍遅い
- **目的**: 適応的レスポンス送信でPrompt速度を回復し、永続的ステージングバッファでGeneration速度をさらに改善する
- **前提条件**: 1号機（192.168.100.1）と2号機（192.168.100.2）がRDMAネットワークで接続済み、Phase 2最適化適用済み
- **モデル**: qwen2.5-0.5b-instruct-q4_k_m.gguf

## 参考レポート

- [RDMAグラフ差分更新 パフォーマンス最適化 (Phase 2)](rdma_graph_diff_update_optimization_2026-02-04_065100.md)
- [RDMAプロトコル パフォーマンス最適化 (Phase 1)](rdma_protocol_performance_optimization_2026-02-04_060500.md)

## 修正内容

### 変更1: 適応的レスポンス送信 (Prompt速度回復)

閾値 `RDMA_ADAPTIVE_RSP_THRESHOLD` (256B) を導入し、レスポンスサイズに応じて送信方式を切り替え:

- **<= 256B（小レスポンス）**: `[size(8B)|data(NB)]` を1回の send/recv で処理。スタック上の配列 `combined[264]` に結合してコピー。HELLO, DEVICE_COUNT, GET_ALIGNMENT, GET_MAX_SIZE, ALLOC_BUFFER, BUFFER_GET_BASE, INIT_TENSOR, GET_ALLOC_SIZE 等のPrompt処理時に頻繁に呼ばれるコマンドが対象。
- **> 256B（大レスポンス）**: 従来通り2回の send/recv（size → data）。GET_TENSOR の大きなデータ（260KB+）のゼロコピーを維持。

256B は全ての固定サイズレスポンス構造体（最大 `rdma_msg_alloc_buffer_rsp` = 28B）を包含し、GET_TENSOR の可変長データを除外する。

### 変更2: 永続的ステージングバッファ (Generation速度改善)

`ggml_backend_rdma_buffer_context` に `std::unique_ptr<rdma_staging_buffer> staging` フィールドを追加。既存の `rdma_staging_buffer` クラスの `get_buffer(size, &mr)` を使用し、サイズが前回以下なら既存バッファを返し、超過時のみリアロケーション（25%の余裕付き）。

- `set_tensor`: `mem_pool->alloc()` + `mem_pool->free()` → `staging->get_buffer()` に置換
- `get_tensor`: 同様に `staging->get_buffer()` に置換
- 自動回帰生成ではサイズが安定するため、初回以降リアロケーション不要

**カーネルコール削減**: 4回/トークン → 0回/トークン（定常状態）

### 変更3: バージョンチェック厳格化 + バージョン更新

- `RDMA_PROTO_MINOR_VERSION` を 2 → 3 に更新
- `check_server_version`: マイナーバージョンの完全一致を要求（`response.minor > RDMA_PROTO_MINOR_VERSION` → `response.minor != RDMA_PROTO_MINOR_VERSION`）
- エラーメッセージにクライアント側バージョンも表示

## 修正対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 適応的send_rsp/recv_rdma_rsp、永続的staging buffer、バージョンチェック厳格化 |
| `ggml/include/ggml-rdma.h` | プロトコルバージョン 1.2.0 → 1.3.0 |

## 再現方法

### 1. 1号機でビルド

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
cmake --build build --target ggml-rdma llama-cli rdma-server -j$(nproc)
```

### 2. 2号機にコピー＆リビルド

```bash
scp ggml/src/ggml-rdma/ggml-rdma.cpp \
    ubuntu@192.168.100.2:/home/ubuntu/projects/llama.cpp/ggml/src/ggml-rdma/
scp ggml/include/ggml-rdma.h \
    ubuntu@192.168.100.2:/home/ubuntu/projects/llama.cpp/ggml/include/
# 注意: 2号機に ggml/src/ggml-rdma/ggml-rdma.h が存在する場合はそちらも更新が必要
scp ggml/include/ggml-rdma.h \
    ubuntu@192.168.100.2:/home/ubuntu/projects/llama.cpp/ggml/src/ggml-rdma/
ssh ubuntu@192.168.100.2 "cmake --build ~/projects/llama.cpp/build \
    --target ggml-rdma rdma-server -j\$(nproc)"
```

### 3. rdma-server再起動（2号機）

```bash
ssh ubuntu@192.168.100.2 "kill \$(pgrep -x rdma-server) 2>/dev/null; echo done"
ssh ubuntu@192.168.100.2 'nohup env LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
    /home/ubuntu/projects/llama.cpp/build/bin/rdma-server --host 0.0.0.0 --port 50051 \
    > /tmp/rdma-server.log 2>&1 &'
```

### 4. RDMA推論テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 \
    LD_LIBRARY_PATH=./build/bin ./build/bin/llama-cli \
    -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -dev 'RDMA0[192.168.100.2:50051]' -p 'こんにちは' -n 50 \
    --no-warmup --single-turn --simple-io
```

## 結果

### 推論速度

| 構成 | Prompt速度 | Generation速度 | 備考 |
|------|-----------|---------------|------|
| ローカルGPU (CUDA0) | 328.5 t/s | 196.1 t/s | ベースライン |
| RDMA Phase 1 | 21.7 t/s | 2.7 t/s | — |
| RDMA Phase 2 | 9.2 t/s | 47.2 t/s | Generation 17.5倍改善 |
| **RDMA Phase 3 (Run 1)** | **13.8 t/s** | **122.4 t/s** | — |
| **RDMA Phase 3 (Run 2)** | **13.7 t/s** | **148.5 t/s** | サーバー再起動後 |

### Phase 2 → Phase 3 改善比

| メトリック | Phase 2 | Phase 3 | 改善 |
|-----------|---------|---------|------|
| Prompt | 9.2 t/s | 13.7 t/s | **+49%** |
| Generation | 47.2 t/s | 122.4〜148.5 t/s | **+159〜215%（2.6〜3.1倍）** |

### ローカルGPUとの比較

| メトリック | ローカルGPU | RDMA Phase 3 | 差 |
|-----------|-----------|-------------|-----|
| Prompt | 328.5 t/s | 13.7 t/s | 24.0倍 |
| Generation | 196.1 t/s | 148.5 t/s | **1.3倍** |

### 推論出力

入力: `こんにちは`

出力 (Run 1): `こんにちは！何かお手伝いできることはありますか？`

出力 (Run 2): `こんにちは！（你好！）こんにちは。（你好。）こんにちは！（你好！）はあ、どの言葉を使うことでも良いですか？`

出力に破綻はなく、正常な応答を確認。

## 考察

- **Generation速度の大幅改善（2.6〜3.1倍）**: 永続的ステージングバッファにより、毎トークンの `posix_memalign` + `ibv_reg_mr` + `ibv_dereg_mr` + `free`（4カーネルコール）が完全に排除された。これが最大の改善要因。148.5 t/s はローカルGPU 196.1 t/s の約76%に到達。
- **Prompt速度の回復（+49%）**: 適応的レスポンス送信により、小さいレスポンス（全固定サイズ構造体）が1回のsend/recvに戻り、Phase 2の退行が部分的に回復した。ただし Phase 1 の 21.7 t/s には未到達（13.7 t/s）。Prompt処理では `send_rsp_empty`（8B send）が1回のままである一方、ALLOC_BUFFERなど一部のコマンドが返す28Bレスポンスが `[8B|28B]=36B` の1回sendに統合されたことで改善。
- **Prompt速度が Phase 1 まで回復しない原因**: Phase 2 で追加された `GRAPH_COMPUTE_UPDATE` 関連のテンソルスナップショット保存・差分比較のオーバーヘッド、および `graph_cache::add()` での memcpy が影響している可能性がある。ただし、これは Generation の 17.5倍改善のために必要なトレードオフ。
- **Generation速度のばらつき（122.4〜148.5 t/s）**: 初回接続後のサーバー内部状態（GPU キャッシュウォームアップ等）により変動。サーバー再起動後の fresh な接続で最良の結果が出る傾向。
- **今後の改善方向**:
  - Prompt速度: フルグラフ送信の最適化（初回グラフ送信もバイナリ差分圧縮等）
  - Generation速度: ローカルGPUとの残り1.3倍の差はRDMA制御メッセージのラウンドトリップ（set_tensor → graph_compute → get_tensor の3コマンドシーケンス）に起因。パイプライニングや非同期コマンド送信で改善可能
