# GPUDirect RDMA 実装可能性調査レポート

- **実施日時**: 2026年1月31日 21:49

---

## 前提・目的

### 背景

llama.cppは大規模言語モデル(LLM)の推論を効率的に実行するためのC/C++ライブラリである。現在、分散推論機能としてTCP/IPベースのRPCフレームワークが実装されているが、より高速なノード間通信を実現するためGPUDirect RDMAの導入可能性を調査する。

### 目的

1. llama.cppの現在のGPU実装と分散処理アーキテクチャを分析する
2. GPUDirect RDMA統合の技術的実現可能性を評価する
3. 実装アプローチを提案する

### 前提条件

- llama.cpp リポジトリ（masterブランチ、コミット 41ea26144）を対象とする
- NVIDIA GPU環境（CUDA対応）を想定
- InfiniBand または RoCE v2 対応ネットワーク環境を想定

---

## 調査方法

### 調査対象ファイル

| ファイル | 内容 | 行数 |
|---------|------|------|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDAバックエンド実装 | 5,122行 |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | RPC分散推論実装 | 78,290行 |
| `ggml/include/ggml-backend.h` | バックエンドAPI定義 | 478行 |
| `ggml/src/ggml-backend-impl.h` | バックエンド内部インタフェース | 241行 |
| `ggml/src/ggml-backend-reg.cpp` | バックエンド登録システム | 513行 |
| `tools/rpc/rpc-server.cpp` | RPCサーバー実装 | 183行 |

### 調査観点

1. **GPU間通信機能**: P2P転送、ピン留めメモリ、非同期転送の有無
2. **分散処理基盤**: RPC実装、ノード間通信プロトコル
3. **バックエンド拡張性**: 新規バックエンド追加の容易さ
4. **メモリ管理**: GPUDirect RDMA要件との互換性

---

## 現状分析

### 1. CUDA バックエンド実装 (`ggml-cuda.cu`)

#### GPU間P2P転送

```cpp
// ggml-cuda.cu:1854-1871
static void ggml_cuda_cpy_tensor_2d(void * dst, const ggml_tensor * src, ...) {
    if (src->backend == GGML_BACKEND_TYPE_GPU && id != g_main_device) {
        // GPU間のPeer-to-Peer転送
        CUDA_CHECK(cudaMemcpy2DAsync(dst, nb0, src_ptr, nb1, ..., cudaMemcpyDeviceToDevice, stream));
    }
}
```

**特徴**:
- `cudaMemcpyPeerAsync` によるGPU間直接メモリ転送
- 複数CUDAデバイス間のP2Pアクセスサポート
- 非同期転送によるオーバーラップ処理

#### ピン留めメモリ（ホストメモリ）

```cpp
// ggml-cuda.cu:688-713
static void * ggml_cuda_host_malloc(size_t size) {
    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);  // ピン留めメモリ確保
    if (err != cudaSuccess) {
        GGML_LOG_WARN("cudaMallocHost failed, falling back to malloc\n");
        ptr = malloc(size);
    }
    return ptr;
}
```

**特徴**:
- `cudaMallocHost` によるページロックメモリ確保
- DMA転送の高速化サポート
- フォールバック機構あり

#### 仮想メモリ管理（VMM）

```cpp
// ggml-cuda.cu:414-477
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35;  // 32GB
    // 仮想アドレス空間の動的拡張
    CUmemGenericAllocationHandle handle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
};
```

**特徴**:
- 最大32GBの連続仮想アドレス空間
- 動的メモリ拡張
- RDMA登録に適した大きなメモリ領域管理

### 2. RPC 分散推論実装 (`ggml-rpc.cpp`)

#### アーキテクチャ

```cpp
// ggml-rpc.cpp:69-84
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    // ...
};
```

**特徴**:
- 明確なコマンドベースプロトコル
- バッファ管理、テンソル操作、グラフ計算の遠隔実行
- 最大16ノード対応

#### 通信レイヤー

```cpp
// ggml-rpc.cpp:164-200
class socket_t {
    int sockfd = -1;
    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
};
```

**現在の制限**:
- TCP/IPソケット通信のみ
- バイトストリームベースの通信
- RDMAネイティブ対応なし

#### 非同期転送API

```cpp
// ggml-rpc.cpp:1203-1232
static void rpc_set_tensor_async(ggml_backend_buffer_t buffer,
                                  ggml_tensor * tensor,
                                  const void * data,
                                  size_t offset, size_t size) {
    // 非同期テンソル設定
    rpc_msg_set_tensor_async_req request;
    request.tensor = rpc_tensor_from_ggml(tensor);
    // ...
}
```

**特徴**:
- 非同期テンソル転送サポート
- RDMA統合のための基盤が存在

### 3. バックエンド抽象化レイヤー

#### バックエンドインタフェース (`ggml-backend.h`)

```cpp
// ggml-backend.h:82-102
// Buffer interface
GGML_API void   ggml_backend_buffer_free(ggml_backend_buffer_t buffer);
GGML_API void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer);
GGML_API size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer);

// Tensor operations
GGML_API void ggml_backend_tensor_set(ggml_tensor * tensor, const void * data, size_t offset, size_t size);
GGML_API void ggml_backend_tensor_get(const ggml_tensor * tensor, void * data, size_t offset, size_t size);
GGML_API void ggml_backend_tensor_set_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
GGML_API void ggml_backend_tensor_get_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
```

#### バックエンド実装インタフェース (`ggml-backend-impl.h`)

```cpp
// ggml-backend-impl.h:20-58
struct ggml_backend_buffer_i {
    void (*free_buffer)(ggml_backend_buffer_t buffer);
    void * (*get_base)(ggml_backend_buffer_t buffer);
    void (*init_tensor)(ggml_backend_buffer_t buffer, ggml_tensor * tensor);
    void (*memset_tensor)(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
    void (*set_tensor)(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor)(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
    bool (*cpy_tensor)(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst);
    void (*clear)(ggml_backend_buffer_t buffer, uint8_t value);
    void (*reset)(ggml_backend_buffer_t buffer);
};
```

**特徴**:
- 明確に定義された関数ポインタインタフェース
- 新規バックエンド追加が容易な設計
- 非同期操作のサポート

---

## GPUDirect RDMA 概要

### 技術説明

GPUDirect RDMAは、NVIDIAが提供する技術で、ネットワークアダプタがGPUメモリに直接アクセスすることを可能にする。

```
従来の転送パス:
GPU Memory → CPU Memory → Network Buffer → Network

GPUDirect RDMA:
GPU Memory → Network (CPUバイパス)
```

### 主要コンポーネント

1. **nvidia-peermem カーネルモジュール**: GPUメモリのRDMA登録を可能にする
2. **libibverbs**: RDMA操作のためのユーザー空間ライブラリ
3. **librdmacm**: RDMA接続管理

### 要件

- NVIDIA Tesla/Quadro/RTX GPU（Kepler世代以降）
- Mellanox/NVIDIA ConnectXシリーズNIC
- InfiniBand または RoCE v2 ファブリック
- CUDA 11.0以降

---

## 実装可能性評価

### 肯定的要素

| 要素 | 詳細 | 評価 |
|-----|------|------|
| P2P転送基盤 | `cudaMemcpyPeerAsync`による GPU間直接転送が実装済み | 高 |
| ピン留めメモリ | `cudaMallocHost`, `cudaHostRegister`対応 | 高 |
| RPC分散推論 | 最大16ノード対応のフレームワーク | 高 |
| 非同期転送API | `set_tensor_async`, `cpy_tensor_async` | 高 |
| プラグイン可能バックエンド | 新規バックエンド追加が容易 | 高 |
| 仮想メモリ管理 | VMM対応（最大32GB連続アドレス空間） | 中〜高 |

### 課題・制限

| 課題 | 詳細 | 対策 |
|-----|------|------|
| TCP/IP通信 | 現在のRPCはソケットベース | RDMAトランスポート層を追加 |
| RDMA API統合 | libibverbs/librdmacm未使用 | 新規モジュールとして実装 |
| ハードウェア依存 | InfiniBand/RoCE必須 | 条件付きコンパイル |
| メモリ登録 | GPUメモリのRDMA登録が必要 | nvidia-peermem連携 |

### 総合評価

**実装可能性: 高い**

llama.cppは以下の点でGPUDirect RDMA統合に適している：

1. **既存の分散処理基盤**: RPCフレームワークが存在し、通信レイヤーを差し替え可能
2. **非同期転送サポート**: RDMAの非同期性質と親和性が高い
3. **メモリ管理の柔軟性**: ピン留めメモリ、VMMが実装済み
4. **モジュラー設計**: バックエンドを追加しやすいアーキテクチャ

---

## 実装アプローチ提案

### 推奨アプローチ: 新規RDMAバックエンド

既存のRPCフレームワークを拡張し、TCP/IPの代替としてRDMAトランスポートを追加する。

```
ggml/src/ggml-rdma/
├── ggml-rdma.h          # 公開API
├── ggml-rdma.cpp        # バックエンド実装
├── rdma-transport.h     # RDMAトランスポート抽象化
├── rdma-transport.cpp   # libibverbs/librdmacm統合
└── CMakeLists.txt       # ビルド設定
```

### 実装フェーズ

#### Phase 1: 基盤構築（短期）

1. **RDMAトランスポート層の実装**
   - libibverbs/librdmacmの統合
   - Queue Pair (QP) 管理
   - Memory Region (MR) 登録

2. **ホストメモリRDMA転送**
   - CPU-to-CPU RDMA（GPUDirect前の検証）
   - 接続確立とデータ転送の基本実装

#### Phase 2: GPUDirect統合（中期）

1. **GPUDirect RDMA有効化**
   - nvidia-peermem連携
   - CUDAメモリのRDMA登録
   - GPU-to-GPU直接転送

2. **RPCバックエンドへの統合**
   - 既存RPC APIとの互換性維持
   - トランスポート選択機能（TCP/RDMA）

#### Phase 3: 最適化（長期）

1. **パフォーマンス最適化**
   - パイプライン転送
   - 複数QP並列化
   - アダプティブルーティング

2. **高可用性機能**
   - 接続断からの復旧
   - フォールバック機構（RDMA→TCP）

### 変更対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/` (新規) | RDMAバックエンド実装 |
| `ggml/CMakeLists.txt` | RDMA関連ビルドオプション追加 |
| `ggml/include/ggml-backend.h` | RDMA固有API追加（必要に応じて） |
| `ggml/src/ggml-backend-reg.cpp` | RDMAバックエンド登録 |
| `tools/rpc/rpc-server.cpp` | RDMA対応オプション追加 |

### コード例: RDMAトランスポート抽象化

```cpp
// rdma-transport.h（概念設計）
class RDMATransport {
public:
    // 接続管理
    virtual bool connect(const char* host, int port) = 0;
    virtual void disconnect() = 0;

    // メモリ登録
    virtual void* register_memory(void* addr, size_t size, bool gpu_memory = false) = 0;
    virtual void deregister_memory(void* mr) = 0;

    // データ転送
    virtual bool rdma_write(void* remote_addr, void* local_addr, size_t size) = 0;
    virtual bool rdma_read(void* local_addr, void* remote_addr, size_t size) = 0;

    // 同期
    virtual bool poll_completion() = 0;
};
```

---

## 再現方法

### 調査の再現

1. llama.cppリポジトリをクローン
   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   git checkout 41ea26144
   ```

2. 関連ファイルの確認
   ```bash
   # CUDAバックエンド
   less ggml/src/ggml-cuda/ggml-cuda.cu

   # RPCフレームワーク
   less ggml/src/ggml-rpc/ggml-rpc.cpp

   # バックエンドAPI
   less ggml/include/ggml-backend.h
   less ggml/src/ggml-backend-impl.h
   ```

3. P2P転送コードの検索
   ```bash
   grep -n "cudaMemcpyPeer" ggml/src/ggml-cuda/*.cu
   grep -n "cudaMallocHost" ggml/src/ggml-cuda/*.cu
   ```

---

## 結論

llama.cppにおけるGPUDirect RDMAの実装は**技術的に実現可能**である。

### 主な根拠

1. **充実した分散処理基盤**: TCP/IPベースのRPCフレームワークが既に存在し、通信レイヤーの差し替えが可能な設計になっている

2. **GPU間通信の実装経験**: P2P転送、ピン留めメモリ、非同期転送など、GPUDirect RDMAに必要な要素技術が既に実装されている

3. **モジュラーアーキテクチャ**: バックエンドの追加が容易な設計により、既存コードへの影響を最小限に抑えられる

### 推奨事項

1. **段階的実装**: まずホストメモリRDMAで基盤を検証し、その後GPUDirectを追加する

2. **フォールバック機構**: RDMA未対応環境でもTCP/IPにフォールバックできる設計とする

3. **ベンチマーク**: 各フェーズでTCP/IP比較のベンチマークを実施し、効果を測定する

### 期待される効果

- ノード間テンソル転送のレイテンシ削減（数msec → 数十usec）
- CPU負荷の軽減（DMAオフロード）
- 大規模モデルの分散推論スループット向上

---

## 参考情報

- [NVIDIA GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [libibverbs API Reference](https://github.com/linux-rdma/rdma-core)
- llama.cpp Issue Tracker: 関連Issue検索推奨
