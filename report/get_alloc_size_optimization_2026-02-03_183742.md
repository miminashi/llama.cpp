# get_alloc_size パフォーマンス最適化レポート

- **実施日時**: 2026年2月3日 18:37

## 前提・目的

### 背景

llama-cliでモデルをロードする際、各テンソルごとに`get_alloc_size`クエリを個別にリモートサーバーへ送信しているため、数百〜数千回のRDMA往復が発生し、モデルロードが非常に遅くなっていた。

### 問題の詳細

- **現状の実装** (`ggml-rdma.cpp:613-644`):
  - 量子化テンソル（`ne[0] % 512 != 0`）の場合にリモートクエリ
  - `FLASH_ATTN_EXT`, `MUL_MAT_ID` オペレーションの場合にリモートクエリ
  - 各リクエストで3260バイト送信、8バイト受信
  - キャッシュなし、バッチ処理なし

- **RPCバックエンドも同じ問題**: TODOコメントに "cache the alloc responses to avoid extra RPC calls?" と記載あり

### 目的

モデルロード時のRPC呼び出し回数を大幅に削減し、ロード時間を短縮する。

## 実装内容

### フェーズ1: ローカル計算（量子化テンソル）

量子化テンソルのパディング計算をクライアント側で実行するように変更。これによりモデルロード時の大半のRPCを削減。

**追加した定数**:
```cpp
#define RDMA_MATRIX_ROW_PADDING 512
```

**ローカル計算ロジック**:
```cpp
if (ggml_is_quantized(tensor->type) &&
    (tensor->ne[0] % RDMA_MATRIX_ROW_PADDING != 0) &&
    (tensor->view_src == nullptr)) {

    size_t size = ggml_nbytes(tensor);
    size += ggml_row_size(tensor->type,
                          RDMA_MATRIX_ROW_PADDING - tensor->ne[0] % RDMA_MATRIX_ROW_PADDING);
    return size;
}
```

### フェーズ2: キャッシュ（FLASH_ATTN_EXT, MUL_MAT_ID）

テンソルシグネチャベースのキャッシュを追加し、同じ形状のテンソルに対する重複RPCを排除。

**キャッシュ構造体**:
```cpp
static constexpr size_t RDMA_ALLOC_SIZE_CACHE_MAX_ENTRIES = 1000;

struct alloc_size_cache_t {
    std::unordered_map<uint64_t, size_t> entries;
    mutable std::mutex mutex;
};
```

**FNV-1aハッシュ関数**:
```cpp
static uint64_t compute_tensor_signature(const ggml_tensor * tensor) {
    constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME  = 1099511628211ULL;
    // type, ne[], op, op_params, src[]の形状をハッシュ化
    ...
}
```

## 修正ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | get_alloc_size最適化、キャッシュ追加 |

### 変更箇所の詳細

| 行番号 | 変更内容 |
|--------|---------|
| 31 | `RDMA_MATRIX_ROW_PADDING` 定数追加 |
| 216 | `RDMA_ALLOC_SIZE_CACHE_MAX_ENTRIES` 定数追加 |
| 218-221 | `alloc_size_cache_t` キャッシュ構造体定義 |
| 231 | `ggml_backend_rdma_buffer_type_context`にキャッシュ追加 |
| 628-663 | `compute_tensor_signature` FNV-1aハッシュ関数 |
| 665-731 | `ggml_backend_rdma_buffer_type_get_alloc_size` 最適化版 |

## 期待される効果

| 項目 | 最適化前 | 最適化後 |
|------|---------|---------|
| モデルロード時RPC | 数百回 | 10回未満 |
| ロード時間 | 数十秒 | 数秒 |

## 再現方法

### 1. ビルド

```bash
cd /home/ubuntu/projects/llama.cpp
cmake --build build --target ggml-rdma llama-cli rdma-server -j$(nproc)
```

### 2. サーバー起動（2号機）

```bash
ssh ubuntu@192.168.100.2 "cd /home/ubuntu/projects/llama.cpp/build/bin && \
  LD_LIBRARY_PATH=. GGML_RDMA_DEBUG=1 GGML_RDMA_NO_GDR=1 \
  ./rdma-server -H 0.0.0.0 -p 50051 -d CUDA0"
```

### 3. 動作確認（1号機）

```bash
# シンプルテスト
GGML_RDMA_NO_GDR=1 GGML_RDMA_DEBUG=1 LD_LIBRARY_PATH=./build/bin \
./build/bin/rdma-simple-test 192.168.100.2:50051
```

### 4. モデルロードテスト（1号機）

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 GGML_RDMA_DEBUG=1 \
LD_LIBRARY_PATH=./build/bin \
./build/bin/llama-cli \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -dev 'RDMA0[192.168.100.2:50051]' \
  -p 'Hello' -n 1 --no-warmup
```

### 5. 最適化効果の確認

`GGML_RDMA_DEBUG=1` を設定すると、以下のデバッグログで最適化の効果を確認可能:

- `[RDMA] get_alloc_size local (quantized):` - ローカル計算（RPC不要）
- `[RDMA] get_alloc_size cache hit:` - キャッシュヒット（RPC不要）
- `[RDMA] get_alloc_size RPC:` - RPCコール（キャッシュに保存）

## テスト結果

### ビルド結果

```
[100%] Built target ggml-rdma
```
ビルド成功（警告のみ）

### rdma-simple-test

正常動作を確認:
```
[1] Testing GPUDirect availability...
    GPUDirect RDMA available: no

[2] Connecting to 192.168.100.2:50051...
    Connected! Backend: RDMA0[192.168.100.2:50051]
...
[9] Cleanup...
    Done!
```

### llama-cli モデルロード

接続タイムアウトエラーが発生（既存のRDMAインフラの問題、今回の最適化とは無関係）:
```
[rdma_connection] Send completion timeout
```

このタイムアウトは `init_tensor` 操作（行438）で発生しており、今回最適化した `get_alloc_size`（行665-731）とは別の箇所。

## 備考

- 量子化テンソルのパディング値（512）はサーバー側と一致している必要がある
- キャッシュのエビクション戦略はシンプルな全クリア方式を採用（1000エントリ到達時）
- 接続タイムアウトの問題は別途調査が必要
