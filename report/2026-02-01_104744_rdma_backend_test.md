# GPUDirect RDMA バックエンド実行テスト・修正レポート

- **実施日時**: 2026年2月1日 10:47

## 前提・目的

GPUDirect RDMAバックエンドの実行テストを行い、未実装部分を修正する。

- **背景**: RDMAバックエンドのビルドは完了しているが、サーバー側のコマンドハンドラーが未実装であり、スタンドアロンサーバーバイナリも存在しなかった
- **目的**: サーバーハンドラーを完全実装し、rdma-serverツールを作成して動作確認を行う
- **前提条件**:
  - nvidia-peermemモジュールがロード済み
  - 1号機と2号機が100GbEで接続

## 実装内容

### Phase 1: サーバーコマンドハンドラーの実装

**ファイル**: `ggml/src/ggml-rdma/ggml-rdma.cpp`

以下のサーバーメソッドを`rdma_server`クラスに実装:

1. **set_tensor()** - テンソルへのデータ書き込み（行1233-1277）
2. **get_tensor()** - テンソルからのデータ読み出し（行1279-1307）
3. **copy_tensor()** - テンソル間コピー（行1309-1331）
4. **init_tensor()** - テンソル初期化（行1333-1355）
5. **get_alloc_size()** - 割り当てサイズ計算（行1357-1387）
6. **graph_compute()** - グラフ計算実行（行1389-1441）
7. **graph_recompute()** - キャッシュ済みグラフの再計算（行1443-1458）

ヘルパーメソッド:
- **deserialize_tensor()** - テンソルのデシリアライズ（行1461-1501）
- **create_node()** - グラフノード作成（行1503-1544）

サーバーメインループのswitch文に対応するcase文を追加（行1732-1815）

### Phase 2: サーバーツールの作成

**新規ファイル**:
- `tools/rdma/CMakeLists.txt` - ビルド設定
- `tools/rdma/rdma-server.cpp` - スタンドアロンサーバー実装

**変更ファイル**:
- `tools/CMakeLists.txt` - rdmaサブディレクトリの追加

### Phase 3: ビルド設定の修正

1. `ggml/src/ggml-backend-reg.cpp`:
   - `#include "ggml-rdma.h"` を追加
   - `register_backend(ggml_backend_rdma_reg())` を追加
   - `ggml_backend_load_best("rdma", silent, dir_path)` を追加

2. `ggml/src/ggml-rdma/ggml-rdma.cpp`:
   - `ggml_backend_rdma_reg_get_proc_address` に `ggml_backend_rdma_start_server` を追加

## 再現方法

### ビルド

```bash
cd /home/ubuntu/ogglm/llama.cpp

# クリーンビルド
rm -rf build
cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# ビルド
cmake --build build --target rdma-server -j$(nproc)
```

### 2号機でサーバー起動

```bash
# バイナリを2号機にコピー
rsync -avz build/bin/ ubuntu@192.168.100.2:/home/ubuntu/ogglm/llama.cpp/build/bin/

# 2号機でサーバー起動
ssh ubuntu@192.168.100.2 "cd /home/ubuntu/ogglm/llama.cpp/build/bin && \
  LD_LIBRARY_PATH=. GGML_RDMA_DEBUG=1 \
  ./rdma-server -H 0.0.0.0 -p 50051 -d CUDA0"
```

### 動作確認

```bash
# サーバーのプロセス確認
ssh ubuntu@192.168.100.2 "ps aux | grep rdma-server"

# ログ確認
ssh ubuntu@192.168.100.2 "cat /tmp/rdma-server.log"
```

## テスト結果

### ビルド結果

- `libggml-rdma.so.0.9.5` が正常にビルドされた
- `rdma-server` バイナリが生成された

### サーバー起動テスト

2号機でのサーバー起動に成功:

```
ggml_cuda_init: found 5 CUDA devices:
  Device 0: Tesla P100-PCIE-16GB, compute capability 6.0, VMM: yes
  Device 1: Tesla P100-PCIE-16GB, compute capability 6.0, VMM: yes
  Device 2: Tesla P100-PCIE-16GB, compute capability 6.0, VMM: yes
  Device 3: Tesla P100-PCIE-16GB, compute capability 6.0, VMM: yes
  Device 4: Tesla P100-PCIE-16GB, compute capability 6.0, VMM: yes

[gdr_memory_manager] GPUDirect RDMA is available
[rdma_connection_manager] Server listening on 0.0.0.0:50051
```

### 確認できた機能

- RDMAバックエンドの正常な初期化
- GPUDirect RDMAの利用可能性検出
- RDMA接続マネージャーによるサーバーリスニング

## 修正したファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | サーバーハンドラー実装、get_proc_address修正 |
| `ggml/src/ggml-backend-reg.cpp` | RDMAバックエンド登録 |
| `tools/rdma/CMakeLists.txt` | 新規作成 |
| `tools/rdma/rdma-server.cpp` | 新規作成 |
| `tools/CMakeLists.txt` | rdmaサブディレクトリ追加 |

## 次のステップ

1. クライアントからサーバーへの実際のRDMA接続テスト
2. テンソル転送の動作確認
3. 分散推論の実行テスト
