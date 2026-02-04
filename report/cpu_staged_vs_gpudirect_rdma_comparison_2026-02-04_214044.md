# CPU-staged RDMA vs GPUDirect RDMA 実装難易度比較レポート

- **実施日時**: 2026年2月4日 21:40

## 前提・目的

### 背景

本プロジェクトでは、llama.cppの分散推論バックエンドとしてRDMAを実装している。初期段階（Step 1）では「GPUDirectでないRDMA」（CPU-staged RDMA）を先に実装し、`GGML_RDMA_NO_GDR=1` で GPUDirect RDMA を無効化した状態で動作確認を行ってきた。

この判断は「GPUDirect RDMAは前提条件が多くハードルが高い」という認識に基づくものだった。

### 目的

以下の問いに答える:

> CUDAを利用する環境において、CPU-staged RDMAとGPUDirect RDMAのどちらが実装上簡単なのか？ CPU-stagedを先に実装した判断は合理的だったか？

### 参照した過去のレポート

- [`gpudirect_rdma_feasibility_2026-01-31_214933.md`](./gpudirect_rdma_feasibility_2026-01-31_214933.md) — GPUDirect RDMA 実装可能性調査
- [`gpudirect_rdma_backend_implementation_2026-01-31_222100.md`](./gpudirect_rdma_backend_implementation_2026-01-31_222100.md) — バックエンド実装レポート
- [`pcie_topology_verification_2026-02-03_134920.md`](./pcie_topology_verification_2026-02-03_134920.md) — PCIeトポロジー検証

---

## 両アプローチの技術概要

### CPU-staged RDMA（現在の実装）

```
[送信側]
GPU Memory
    ↓ cudaMemcpy (DeviceToHost)
CPU Pinned Buffer (cudaMallocHost)
    ↓ ibv_reg_mr (CPU memory)
    ↓ RDMA Write
    ↓ ~~~~ ネットワーク ~~~~
[受信側]
CPU Pinned Staging Buffer
    ↓ FLUSH_STAGING コマンド
    ↓ cudaMemcpy (HostToDevice)
GPU Memory
```

### GPUDirect RDMA

```
[送信側]
GPU Memory
    ↓ ibv_reg_mr (GPU pointer — nvidia-peermem が変換)
    ↓ RDMA Write
    ↓ ~~~~ ネットワーク ~~~~
[受信側]
GPU Memory（直接書き込み）
```

---

## 前提条件の比較

| 要素 | CPU-staged RDMA | GPUDirect RDMA |
|------|----------------|----------------|
| カーネルモジュール | 標準RDMA (ibverbs) のみ | nvidia-peermem 追加必要 |
| ドライバ | 標準CUDA + OFED | CUDA 11.4+/R470+ + OFED + nvidia-peermem |
| PCIeトポロジ | 制約なし | GPU-NIC間のトポロジが性能に直結（PIX/PHB推奨、SYS非推奨） |
| ハードウェア | 任意のGPU + RDMA NIC | Compute Capability 3.5以降 + ConnectX-3以降 |
| セットアップ手順 | NVIDIA driver + RDMA driver | MLNX_OFED → NVIDIA driver（順序重要）→ `modprobe nvidia_peermem` |
| 動作確認 | 接続確立のみ確認 | `nvidia-peermem` ロード確認 + PCIeトポロジ確認が必要 |

**ポイント**: GPUDirect RDMAの前提条件は明らかに多い。特にドライバのインストール順序やPCIeトポロジの制約は、動作確認までのハードルを大きく上げる。

---

## 実装難易度の比較

### CPU-staged RDMA に必要な追加コード

CPU-staged方式では、GPUDirect RDMAにない以下の機構が必要になる:

#### 1. ステージングバッファ管理

サーバ側でGPUバッファごとにCPU上のステージングバッファを確保・管理する必要がある:

```cpp
// ggml-rdma.cpp:1783-1788 — サーバ側バッファ割当時
} else if (!std::getenv("GGML_RDMA_NO_STAGING")) {
    // No GDR: allocate host staging buffer and register it for RDMA Write
    // Client will RDMA Write to this staging area, then send FLUSH_STAGING
    // to trigger cudaMemcpy from staging to GPU buffer
    alloc_host_staging(buffer, buffer->size, conn, response);
}
```

関連するデータ構造:
```cpp
// ggml-rdma.cpp:2452
std::unordered_map<ggml_backend_buffer_t, host_staging_info> staging_buffers_;
```

#### 2. FLUSH_STAGINGプロトコル

RDMA Writeの後にステージングバッファからGPUメモリへコピーするための専用コマンドが必要:

```cpp
// ggml-rdma.cpp:796-803 — クライアント側 set_tensor
// Tell server to flush staging buffer to GPU
rdma_msg_flush_staging_req flush_req;
flush_req.remote_ptr = ctx->remote_ptr;
flush_req.offset = buf_offset;
flush_req.size = size;
bool flush_ok = send_rdma_cmd(ctx->conn.get(), RDMA_CMD_FLUSH_STAGING,
                              &flush_req, sizeof(flush_req), nullptr);
```

```cpp
// ggml-rdma.cpp:2171-2213 — サーバ側 flush_staging 実装
bool flush_staging(const rdma_msg_flush_staging_req & request) {
    // ...staging buffer lookup...
    cudaMemcpy(dst, src, request.size, cudaMemcpyHostToDevice);
    // ...
}
```

#### 3. 二重メモリ確保

GPUバッファと同サイズのCPUステージングバッファを別途確保するため、メモリ消費が2倍になる。

#### 4. クライアント側ステージング

クライアント側でもGPUからCPUへのcudaMemcpyが必要:

```cpp
// rdma-memory.cpp の rdma_staging_buffer — クライアント側ステージング管理
// rdma-gdr.cpp:332-443 の pinned_memory_manager — ピン留めメモリ管理
```

### GPUDirect RDMA に必要な追加コード

#### 1. nvidia-peermem検出

```cpp
// rdma-gdr.cpp:23-44
bool gdr_memory_manager::is_peermem_loaded() {
    std::ifstream module_file("/sys/module/nvidia_peermem/initstate");
    // ...check "live"...
}
```

#### 2. GPUメモリの直接RDMA登録

**核心部分はわずか数行**:

```cpp
// rdma-gdr.cpp:182-186
int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
struct ibv_mr * mr = conn_->register_memory(gpu_addr, size, access_flags);
```

CPUメモリの登録と**全く同じAPIコール**。nvidia-peermemカーネルモジュールがGPUポインタの変換を透過的に処理する。

#### 3. デバイス初期化とCapabilityチェック

```cpp
// rdma-gdr.cpp:105-151 — Compute Capability 3.5以上の確認
```

### コード量の比較

| コンポーネント | CPU-staged固有 | GPUDirect固有 |
|---|---|---|
| ステージングバッファ管理 (alloc/free/lookup) | ~80行 | 不要 |
| FLUSH_STAGINGプロトコル (定義+クライアント+サーバ) | ~60行 | 不要 |
| pinned_memory_manager | ~110行 | 不要 |
| rdma_staging_buffer | ~40行 | 不要 |
| gdr_memory_manager (peermem検出+登録) | 不要 | ~200行 |
| GDR/非GDR分岐ロジック | ~30行 | 不要（分岐自体が不要に） |
| **合計** | **~320行** | **~200行** |

### 核心的発見

**CUDAコンテキストが既にある場合、GPUDirect RDMAのほうがコード量が少なくアーキテクチャがシンプルである。**

- CPU-staged: GPUバッファ → ステージングバッファ → RDMA登録 → 転送 → FLUSH → cudaMemcpy という多段パイプラインが必要
- GPUDirect: GPUバッファ → RDMA登録（`ibv_reg_mr` にGPUポインタを渡すだけ）→ 転送 で完結

特に `ibv_reg_mr` はCPUメモリでもGPUメモリでも同一APIであり、nvidia-peermemが透過的にアドレス変換を行う点が重要。つまり**RDMAライブラリの使い方自体は全く変わらない**。

---

## 現在の実装における GDR/非GDR パス

現在の実装は既にGDRと非GDRの両方のパスを持っている:

```cpp
// ggml-rdma.cpp:1773-1788 — サーバ側バッファ割当
if (gdr_memory_manager::is_available()) {
    // GDR available: register GPU buffer directly for RDMA access
    struct ibv_mr * mr = conn->register_memory(base, buffer->size, access_flags);
    // ...
} else if (!std::getenv("GGML_RDMA_NO_STAGING")) {
    // No GDR: allocate host staging buffer
    alloc_host_staging(buffer, buffer->size, conn, response);
}
```

```cpp
// ggml-rdma.cpp:2178-2183 — サーバ側 flush_staging
auto it = staging_buffers_.find(buffer);
if (it == staging_buffers_.end()) {
    // No staging buffer — GDR path wrote directly to GPU, no flush needed
    return true;
}
```

**GPUDirect RDMAの有効化は `GGML_RDMA_NO_GDR=1` 環境変数を外すだけ**で行える状態にある。rdma-gdr.cppのGPUDirect RDMAコード（nvidia-peermem検出、GPUメモリ登録、デバイス初期化）は完全に実装済みである。

---

## 当初の判断の妥当性評価

### CPU-stagedを先に実装した判断は合理的だった

以下の理由から、この段階的アプローチは正しかった:

#### 1. 環境の不確実性

- nvidia-peermemモジュールの動作確認が未完了だった
- ドライバのインストール順序（MLNX_OFED → NVIDIA driver）が正しいか未検証だった
- CPU-stagedなら環境要因を排除してプロトコル実装に集中できた

#### 2. PCIeトポロジーの問題

PCIeトポロジー検証（[2026-02-03のレポート](./pcie_topology_verification_2026-02-03_134920.md)）で判明した状況:

| ノード | GPU-NIC接続 | GDR適性 |
|---|---|---|
| 1号機 GPU3-6 | PIX（同一PCIeブリッジ） | 最適 |
| 1号機 GPU0-2 | PHB（同一NUMAノード） | 可能 |
| 2号機 GPU0 | PIX | 最適 |
| 2号機 GPU1-2 | PHB | 可能 |
| **2号機 GPU3** | **SYS（NUMAまたぎ）** | **困難** |

2号機GPU3のSYS接続では、GPUDirect RDMAの性能が著しく低下するか動作しない可能性がある。CPU-stagedであればこの制約を回避できる。

#### 3. デバッグの容易さ

- CPU-staged方式ではステージングバッファの内容をCPU側で検査可能
- GPUDirect方式ではGPUメモリを直接RDMAで読み書きするため、中間状態の確認が困難
- プロトコル設計やグラフシリアライズのバグ切り分けにはCPU-stagedが有利

#### 4. 本質的な課題はRDMA方式に依存しない

Step 1で解決した以下の課題は、CPU-staged/GPUDirectどちらでも同じ:

- RDMAコネクション確立（librdmacm）
- プロトコル設計（コマンド体系、テンソルシリアライズ）
- グラフの差分更新機構
- バッファ管理とメモリ登録
- マルチGPU対応

### GPUDirect RDMAを最初から実装した場合の問題

仮にGPUDirect RDMAから着手していた場合:

1. nvidia-peermemの環境構築でつまずいた場合、プロトコル実装自体が進まなかった
2. PCIeトポロジーの問題で2号機GPU3が使えず、GPU割当の最適化が先に必要になった
3. バグがRDMA通信なのかGPUDirect固有なのか切り分けが困難だった

---

## P100環境での具体的状況

### ハードウェア

- GPU: NVIDIA Tesla P100-PCIE-16GB（Compute Capability 6.0 ≥ 3.5 → GDR対応）
- NIC: Mellanox ConnectX-5（mlx5_0）— ConnectX-3以降 → GDR対応
- 接続: 100GbE RoCE v2

### PCIeトポロジーとGDR適性

```
1号機 (7 GPU):
  GPU3-6 ↔ NIC: PIX（同一PCIeスイッチ）→ GDR最適
  GPU0-2 ↔ NIC: PHB（同一NUMAノード内）→ GDR可能

2号機 (4 GPU):
  GPU0   ↔ NIC: PIX → GDR最適
  GPU1-2 ↔ NIC: PHB → GDR可能
  GPU3   ↔ NIC: SYS（NUMAまたぎ）→ GDR困難・性能低下
```

### 現在の性能状況（CPU-staged）

| 構成 | Prompt (t/s) | Generation (t/s) | ローカル比 |
|---|---|---|---|
| gpt-oss-20b ローカル2GPU | 328.5 | 70.3 | 100% |
| gpt-oss-20b RDMA 2GPU | 13.7 | 16.0 | **23%** |

性能損失77%の主因はCPUステージング経由のオーバーヘッド。GPUDirect RDMA有効化により大幅な改善が期待できる。

NVIDIA技術ブログのベンチマークによると:

| メトリック | CPU-staged | GPUDirect RDMA |
|---|---|---|
| 小メッセージレイテンシ | 8〜17 μs | 1.9〜2 μs |
| レイテンシ改善 | — | **4〜9倍** |

---

## 結論

### 実装難易度の回答

**CUDAアプリケーションにおいて、実装のコード量・アーキテクチャのシンプルさではGPUDirect RDMAのほうが簡単である。** ただし、前提条件のセットアップ（nvidia-peermem、ドライバ順序、PCIeトポロジー）はGPUDirect RDMAのほうが複雑である。

| 観点 | CPU-staged | GPUDirect |
|---|---|---|
| 前提条件のセットアップ | **簡単** | 複雑 |
| コード実装量 | ~320行 | **~200行** |
| アーキテクチャの複雑さ | 高（多段パイプライン） | **低（直接転送）** |
| デバッグの容易さ | **容易** | 困難 |
| 性能 | 低（CPU経由オーバーヘッド） | **高（直接転送）** |

### 判断の妥当性

CPU-stagedを先に実装した判断は**合理的だった**。理由:
1. 環境の不確実性を排除してプロトコル設計に集中できた
2. PCIeトポロジーの問題を事前に把握できた
3. デバッグが容易な状態でグラフシリアライズ等の複雑な機構を安定化できた

### 今後の方針

1. **Step 2（クラスタ安定化）を最優先で完了させる** — CPU-staged方式のまま
2. **Step 3-4でGPUDirect RDMAを有効化** — `GGML_RDMA_NO_GDR=1` を外すだけで有効化可能
3. **2号機GPU3の扱い** — SYS接続のためGPUDirect RDMA非対応。CPU-stagedのフォールバックを維持するか、GPU3をリモートバックエンドから除外する

---

## 再現方法

### 現在のGDR/非GDR実装の確認

```bash
# GDR関連コードの確認
grep -n "gdr_memory_manager" ggml/src/ggml-rdma/ggml-rdma.cpp
cat ggml/src/ggml-rdma/rdma-gdr.cpp

# ステージングバッファ関連コードの確認
grep -n "staging" ggml/src/ggml-rdma/ggml-rdma.cpp
grep -n "FLUSH_STAGING" ggml/src/ggml-rdma/ggml-rdma.cpp

# 現在のGDR無効化状態の確認
grep -rn "GGML_RDMA_NO_GDR" ggml/src/ggml-rdma/
```

### GPUDirect RDMA有効化テスト（Step 4で実施予定）

```bash
# nvidia-peermemの確認
cat /sys/module/nvidia_peermem/initstate

# GDR無効化フラグを外して起動
# サーバ側（2号機）
unset GGML_RDMA_NO_GDR
./build/bin/llama-rdma-server -H 0.0.0.0 -p 50051

# クライアント側（1号機）
unset GGML_RDMA_NO_GDR
./build/bin/llama-cli -m model.gguf --rdma 192.168.100.2:50051 --log-file /tmp/llama-cli.log
```

---

## 参考情報

- [NVIDIA GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Benchmarking GPUDirect RDMA on Modern Server Platforms — NVIDIA Technical Blog](https://developer.nvidia.com/blog/benchmarking-gpudirect-rdma-on-modern-server-platforms/)
- [The Evolution and Implementation of GPUDirect RDMA — DatenLord](https://medium.com/@datenlord/the-evolution-and-implementation-of-gpudirect-rdma-19751f7b9413)
- [ibv_reg_mr() Documentation — RDMAmojo](https://www.rdmamojo.com/2012/09/07/ibv_reg_mr/)
- [nvidia-peermem GitHub Repository](https://github.com/Mellanox/nv_peer_memory)
