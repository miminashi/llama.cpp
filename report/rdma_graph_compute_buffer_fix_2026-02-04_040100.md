# RDMAグラフ計算時バッファサイズ問題の修正

- **実施日時**: 2026年2月4日 04:01

## 前提・目的

llama-cliでRDMAバックエンドを使用してグラフ計算（推論）を実行する際、バッファサイズ超過エラーが発生していた。本修正はこの問題を解消し、大きなグラフデータの送受信を可能にする。

- **背景**: 約1030個のテンソルを含むグラフ（約388KB）を送信しようとすると、RDMA送信バッファの制限（64KB）を超過しクラッシュする
- **目的**: グラフ送信時に一時Memory Region (MR)を登録し、大きなデータの送受信を可能にする
- **前提条件**: 1号機と2号機がRDMA over RoCEで接続済みであること
- **参照レポート**: [rdma_timeout_fix_2026-02-03_190500.md](rdma_timeout_fix_2026-02-03_190500.md)

## 問題の詳細

### エラーメッセージ（修正前）
```
[rdma_connection] Message too large for inline send without MR
   size=387900, max_inline=256, send_buffer_.size()=65536, send_mr_=...
ggml-rdma.cpp:841: Remote RDMA server crashed or returned malformed response
```

### 根本原因

3つの問題が存在した:

1. **クライアント側（送信）**: `send_rdma_cmd`でグラフデータ送信時にMRが指定されていなかった
   - `send()`の内部バッファ（64KB）を超えるデータを送信不可
2. **クライアント側（ヘッダ送信）**: `send_rdma_cmd_raw`がMRをヘッダ部分にも適用していた
   - ヘッダ（cmd 1byte, size 8bytes）はスタック上の変数で、データ用MRのアドレス範囲外
   - `local protection error`が発生
3. **サーバー側（受信）**: 大きなメッセージ受信時にもMRが未登録
   - 内部受信バッファ（64KB）を超えるデータを受信不可

## 修正内容

### 修正ファイル

`ggml/src/ggml-rdma/ggml-rdma.cpp` の3箇所を修正。

### 修正1: `send_rdma_cmd_raw` - ヘッダ送信時のMR除外（293-314行目）

MRをデータ部分の送信にのみ使用し、ヘッダ（cmd, size）はnullptrで送信するように修正。

```cpp
// 修正: ヘッダはMRなしで送信し、データ部分のみMRを使用
uint8_t cmd_byte = static_cast<uint8_t>(cmd);
if (!conn->send(&cmd_byte, 1, nullptr)) {    // ← nullptrに変更
    return false;
}
uint64_t size = input_size;
if (!conn->send(&size, sizeof(size), nullptr)) {    // ← nullptrに変更
    return false;
}
if (input_size > 0) {
    if (!conn->send(input, input_size, send_mr)) {  // ← データのみMR使用
        return false;
    }
}
```

### 修正2: `ggml_backend_rdma_graph_compute` - 送信時の一時MR登録（836-859行目）

グラフシリアル化データが64KBを超える場合、一時MRを登録して送信。

```cpp
serialize_graph(ctx->device, cgraph, input);

static const size_t RDMA_SEND_BUF_SIZE = 64 * 1024;
struct ibv_mr * send_mr = nullptr;
if (input.size() > RDMA_SEND_BUF_SIZE) {
    send_mr = ctx->conn->register_memory(
        input.data(), input.size(), IBV_ACCESS_LOCAL_WRITE);
    if (!send_mr) {
        GGML_LOG_ERROR("[rdma] Failed to register MR for graph data (size=%zu)\n", input.size());
        return GGML_STATUS_FAILED;
    }
}
bool status = send_rdma_cmd(ctx->conn.get(), RDMA_CMD_GRAPH_COMPUTE,
                             input.data(), input.size(), send_mr);
if (send_mr) {
    ctx->conn->deregister_memory(send_mr);
}
```

### 修正3: サーバー側受信ループ - 受信時の一時MR登録（1860-1878行目）

サーバーのメッセージ受信時、64KBを超えるデータに対して一時MRを登録。

```cpp
std::vector<uint8_t> msg_data(msg_size);
if (msg_size > 0) {
    static const size_t RDMA_RECV_BUF_SIZE = 64 * 1024;
    struct ibv_mr * recv_mr = nullptr;
    if (msg_size > RDMA_RECV_BUF_SIZE) {
        recv_mr = conn->register_memory(msg_data.data(), msg_size, IBV_ACCESS_LOCAL_WRITE);
        // ... エラーハンドリング ...
    }
    bool ok = conn->recv(msg_data.data(), msg_size, recv_mr);
    if (recv_mr) {
        conn->deregister_memory(recv_mr);
    }
    if (!ok) break;
}
```

## 検証結果

### ビルド
```bash
cmake --build build --target ggml-rdma llama-cli rdma-server -j$(nproc)
# 警告なし（既存の軽微な警告1件のみ）でビルド成功
```

### テスト実行
```bash
GGML_RDMA_DEBUG=1 GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 \
LD_LIBRARY_PATH=./build/bin ./build/bin/llama-cli \
-m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
-dev 'RDMA0[192.168.100.2:50051]' -p 'Hello' -n 1 --no-warmup
```

### 結果

- **修正前**: `Message too large for inline send without MR` でクラッシュ（exit code 134）
- 修正1のみ適用後: `local protection error` → `Send completion timeout` でクラッシュ（ヘッダにMRを適用してしまうバグ）
- 修正1+2のみ適用後: サーバー側で `No MR provided and message too large (387900 bytes) for internal buffer (65536 bytes)` でクラッシュ
- **修正1+2+3適用後**: グラフ計算が正常に開始。サーバー側ログに `ggml_cuda_graph_set_enabled: disabling CUDA graphs due to GPU architecture` が表示され、CUDA上でグラフ計算が実行されたことを確認。クラッシュなし。

## 再現方法

1. 1号機でビルド
   ```bash
   cmake --build build --target ggml-rdma llama-cli rdma-server -j$(nproc)
   ```

2. 2号機にファイルを転送しビルド
   ```bash
   scp ggml/src/ggml-rdma/ggml-rdma.cpp ubuntu@192.168.100.2:/home/ubuntu/projects/llama.cpp/ggml/src/ggml-rdma/
   ssh ubuntu@192.168.100.2 "cmake --build /home/ubuntu/projects/llama.cpp/build --target ggml-rdma rdma-server -j\$(nproc)"
   ```

3. 2号機でサーバー起動
   ```bash
   ssh ubuntu@192.168.100.2 "pkill -f rdma-server || true"
   ssh ubuntu@192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server --host 0.0.0.0 --port 50051 > /tmp/rdma-server.log 2>&1 &"
   ```

4. 1号機でテスト実行
   ```bash
   GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=./build/bin \
   ./build/bin/llama-cli -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
   -dev 'RDMA0[192.168.100.2:50051]' -p 'Hello' -n 1 --no-warmup
   ```
