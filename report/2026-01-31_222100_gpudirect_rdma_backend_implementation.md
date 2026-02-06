# GPUDirect RDMA バックエンド実装レポート

- **実施日時**: 2026年1月31日 22:21

## 前提・目的

### 背景
既存のRPCバックエンドはTCP/IPベースであり、大規模モデルの分散推論においてデータ転送がボトルネックとなる可能性がある。1号機と2号機は100GbEで接続されており、GPUDirect RDMAが利用可能な環境である。

### 目的
- libibverbs/librdmacmを使用したRDMAトランスポート層を実装
- nvidia-peermemと連携してGPUメモリへの直接転送（GPUDirect RDMA）を実現
- 既存のRPCバックエンドと互換性のあるプロトコルを使用

### 前提条件
- RDMAライブラリ（libibverbs, librdmacm）がインストールされていること
- 100GbE RDMA対応NICが使用可能であること
- GPUDirect RDMAにはnvidia-peermemモジュールが必要

## 実装概要

### ディレクトリ構成

```
ggml/src/ggml-rdma/
├── CMakeLists.txt          # ビルド設定（libibverbs/librdmacm リンク）
├── ggml-rdma.cpp           # メイン実装（バックエンドインタフェース）
├── rdma-transport.h        # RDMAトランスポート層ヘッダ
├── rdma-transport.cpp      # RDMAトランスポート層実装
├── rdma-memory.h           # RDMAメモリ管理ヘッダ
├── rdma-memory.cpp         # RDMAメモリ管理実装
├── rdma-gdr.h              # GPUDirect RDMA関連ヘッダ
└── rdma-gdr.cpp            # GPUDirect RDMA関連実装

ggml/include/
└── ggml-rdma.h             # 公開API
```

### 実装ファイル詳細

#### 1. CMakeLists.txt
- libibverbs/librdmacmの検出とリンク
- CUDAToolkitのオプション検出（GPUDirect RDMA用）
- nvidia-peermemの利用可能性チェック
- `ggml_add_backend_library`によるバックエンド登録

#### 2. ggml-rdma.h（公開API）
- `ggml_backend_rdma_init()` - バックエンド初期化
- `ggml_backend_rdma_buffer_type()` - バッファタイプ取得
- `ggml_backend_rdma_start_server()` - サーバー起動
- `ggml_backend_rdma_reg()` - バックエンド登録
- `ggml_backend_rdma_add_server()` - サーバー追加
- `ggml_backend_rdma_enable_gdr()` - GPUDirect有効化
- `ggml_backend_rdma_get_stats()` - 転送統計取得

#### 3. rdma-transport.h/cpp
- `rdma_connection` クラス
  - RDMA CM（rdma_cm）による接続確立（connect/accept）
  - 保護ドメイン（PD）、完了キュー（CQ）、キューペア（QP）の管理
  - RDMA Send/Receive操作（制御メッセージ用）
  - RDMA Write/Read操作（片方向データ転送）
  - メモリ登録（ibv_reg_mr）
  - 完了キューポーリングと通知
- `rdma_connection_manager` クラス
  - 接続プール管理
  - サーバーソケット管理

#### 4. rdma-memory.h/cpp
- `memory_region_info` 構造体 - メモリ領域情報
- `rdma_memory_pool` クラス
  - ホストメモリ割り当て/解放
  - ピン留めホストメモリ（cudaMallocHost）
  - メモリ領域のRDMA登録
- `rdma_staging_buffer` クラス
  - ステージングバッファ管理

#### 5. rdma-gdr.h/cpp
- `gdr_memory_manager` クラス
  - nvidia-peermem利用可能性チェック（`/sys/module/nvidia_peermem`）
  - GPUメモリのRDMA登録
  - GPUデバイスの初期化
- `pinned_memory_manager` クラス
  - ピン留めホストメモリ管理

#### 6. ggml-rdma.cpp
- **プロトコル**: RPCと互換性のあるコマンドセット
  - `RDMA_CMD_HELLO` - バージョンチェック
  - `RDMA_CMD_ALLOC_BUFFER` - バッファ割り当て（MR情報を含む応答）
  - `RDMA_CMD_SET_TENSOR` / `RDMA_CMD_GET_TENSOR` - テンソルデータ転送
  - `RDMA_CMD_GRAPH_COMPUTE` - グラフ計算
- **バッファインタフェース**
  - `set_tensor`: RDMA Writeを優先使用、フォールバックでSend
  - `get_tensor`: RDMA Readを優先使用、フォールバックでRecv
- **バックエンドインタフェース**
  - グラフキャッシュによる再計算の最適化
- **サーバー実装**
  - 接続受け入れとコマンドディスパッチ

## 再現方法

### 1. ビルド

```bash
cd /home/ubuntu/ogglm/llama.cpp

# RDMA バックエンドを有効にしてビルド
cmake -B build -DGGML_RDMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-rdma

# 全体をビルドする場合
cmake --build build -j$(nproc)
```

### 2. サーバー起動（2号機）

```bash
ssh ubuntu@192.168.100.2 "cd /home/ubuntu/ogglm/llama.cpp && \
    ./build/bin/llama-rdma-server -H 0.0.0.0 -p 50051"
```

※ `llama-rdma-server` は別途ツールとして実装が必要。現在の実装では `ggml_backend_rdma_start_server()` API経由で起動。

### 3. 使用例（コード）

```cpp
#include "ggml-rdma.h"

// バックエンド初期化
ggml_backend_t backend = ggml_backend_rdma_init("192.168.100.2:50051", 0);

// GPUDirect RDMAを有効化（利用可能な場合）
if (ggml_backend_rdma_gdr_available()) {
    ggml_backend_rdma_enable_gdr(backend);
}

// バッファタイプ取得
ggml_backend_buffer_type_t buft = ggml_backend_rdma_buffer_type("192.168.100.2:50051", 0);

// 以降は標準のggmlバックエンドAPIを使用
```

## ビルド結果

```
$ cmake -B build-rdma -DGGML_RDMA=ON
-- Using RDMA backend
--   ibverbs library: /usr/lib/x86_64-linux-gnu/libibverbs.so
--   rdmacm library: /usr/lib/x86_64-linux-gnu/librdmacm.so
-- Including RDMA backend

$ cmake --build build-rdma --target ggml-rdma
[100%] Built target ggml-rdma

$ ls -la build-rdma/bin/libggml-rdma.so*
libggml-rdma.so -> libggml-rdma.so.0
libggml-rdma.so.0 -> libggml-rdma.so.0.9.5
libggml-rdma.so.0.9.5 (実ファイル)
```

## 技術的なポイント

### RDMAプロトコル設計
- 制御メッセージはRDMA Send/Receiveで送受信
- 大規模データ（テンソル）はRDMA Write/Readで片方向転送
- バッファ割り当て時にメモリ領域（MR）情報を交換し、直接アクセスを可能に

### GPUDirect RDMA
- nvidia-peermemモジュールが必要
- GPUメモリを直接IBVに登録可能
- CPU経由のコピーを排除し、GPU-NIC間で直接転送

### 既存RPCとの互換性
- コマンドIDはRPCと同一の番号体系を使用
- テンソルシリアライズフォーマットは同一
- フォールバックとしてSend/Recv経由の転送も可能

## 今後の課題

1. **サーバーツールの実装**: `llama-rdma-server` バイナリの追加
2. **非同期転送**: 現在は同期的な実装、非同期転送のサポート
3. **マルチスレッドサーバー**: 複数クライアントの並列処理
4. **エラーハンドリング**: 接続断絶時の再接続処理
5. **ベンチマーク**: TCP RPCとの性能比較
