# RDMA接続ライフタイム修正 — 接続切断バグの解消

- **実施日時**: 2026年2月4日 05:22

## 前提・目的

RDMA推論で0.5Bモデル（qwen2.5-0.5b-instruct-q4_k_m.gguf）を使用した際、推論が10分以上かかる、またはモデルロード時にクラッシュするバグを修正する。

- **背景**: ローカルGPU（CUDA0）では192 t/sで推論可能なモデルが、RDMA経由では`GGML_ASSERT(buft) failed`でクラッシュするか、接続が切れたり再接続が繰り返されることで極端に遅くなる
- **目的**: 接続のライフタイム管理を修正し、RDMA推論を安定動作させる
- **前提条件**: 1号機（192.168.100.1）と2号機（192.168.100.2）がRDMAネットワークで接続済み

## 参考レポート

- [RDMA graph compute buffer fix](rdma_graph_compute_buffer_fix_2026-02-04_040100.md)
- [RDMA timeout fix](rdma_timeout_fix_2026-02-03_190500.md)
- [RDMA backend test](rdma_backend_test_2026-02-03_165600.md)

## 根本原因分析

`rdma_connection_manager`が接続を`std::weak_ptr`で保持していたため（`rdma-transport.h:168`）：

1. デバイス登録時（`ggml-rdma.cpp:1196`）に作成されるローカル`shared_ptr`がスコープを抜けると、参照カウントが0になり接続が破棄される
2. その後の`buffer_type`作成（`ggml-rdma.cpp:1090`）で`get_connection()`が再接続を試みるが、シングルスレッドのrdma-serverが前のセッションのクリーンアップ中でまだ`accept`に戻っておらず、接続失敗
3. 推論中も毎回`get_connection()`が新しい接続を作成→即破棄し、極端に遅くなる

修正前のエラーログ：
```
[rdma_connection] Disconnected from 192.168.100.2:50051
[rdma_connection] Failed to get CM event: Interrupted system call
[rdma] Failed to connect to 192.168.100.2:50051
Loading model... GGML_ASSERT(buft) failed
```

## 修正内容

### 修正方針

`connections_`マップの値の型を`std::weak_ptr`から`std::shared_ptr`に変更し、接続マネージャ（static singleton）が接続の所有権を持つようにする。これにより、一度確立した接続がプログラム終了まで維持される。

### 修正ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/rdma-transport.h` (168行目) | `connections_`の型を`weak_ptr`→`shared_ptr`に変更 |
| `ggml/src/ggml-rdma/rdma-transport.cpp` (690-701行目) | `get_connection()`メソッドを`shared_ptr`対応に修正 |

### 変更差分

**rdma-transport.h**
```cpp
// Before
std::unordered_map<std::string, std::weak_ptr<rdma_connection>> connections_;

// After
std::unordered_map<std::string, std::shared_ptr<rdma_connection>> connections_;
```

**rdma-transport.cpp**
```cpp
// Before
auto it = connections_.find(endpoint);
if (it != connections_.end()) {
    if (auto conn = it->second.lock()) {
        if (conn->is_connected()) {
            return conn;
        }
    }
}

// After
auto it = connections_.find(endpoint);
if (it != connections_.end()) {
    if (it->second && it->second->is_connected()) {
        return it->second;
    }
    connections_.erase(it);
}
```

## 再現方法

### 1. ビルド（両サーバー）

```bash
# 1号機
cmake --build build --target ggml-rdma llama-cli rdma-server -j$(nproc)

# 2号機にコピー＆リビルド
scp ggml/src/ggml-rdma/rdma-transport.{cpp,h} ubuntu@192.168.100.2:/home/ubuntu/projects/llama.cpp/ggml/src/ggml-rdma/
ssh ubuntu@192.168.100.2 "cmake --build /home/ubuntu/projects/llama.cpp/build --target ggml-rdma rdma-server -j\$(nproc)"
```

### 2. rdma-server再起動（2号機）

```bash
ssh ubuntu@192.168.100.2 "pkill -f rdma-server"
ssh ubuntu@192.168.100.2 "nohup bash -c 'LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin /home/ubuntu/projects/llama.cpp/build/bin/rdma-server --host 0.0.0.0 --port 50051' > /tmp/rdma-server.log 2>&1 &"
```

### 3. ベースライン測定（ローカルGPU）

```bash
LD_LIBRARY_PATH=./build/bin ./build/bin/llama-cli \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -dev CUDA0 -p 'こんにちは' -n 50 --no-warmup --single-turn --simple-io
```

### 4. RDMA推論テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=./build/bin \
./build/bin/llama-cli -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -dev 'RDMA0[192.168.100.2:50051]' -p 'こんにちは' -n 50 --no-warmup --single-turn --simple-io
```

## 検証結果

### ビルド

両サーバーでビルド成功（警告のみ、エラーなし）。

### 推論テスト

| 項目 | ローカルGPU (CUDA0) | RDMA (修正後) | RDMA (修正前) |
|------|-------------------|--------------|--------------|
| Prompt速度 | 326.0 t/s | 3.4 t/s | クラッシュ or >10分 |
| Generation速度 | 192.0 t/s | 0.3 t/s | クラッシュ or >10分 |
| 動作状態 | 正常 | 正常完走 | GGML_ASSERT失敗 |

### 考察

- **接続切断問題は解消**: 修正前に発生していた`GGML_ASSERT(buft) failed`クラッシュが解消され、推論が正常に最後まで完走するようになった
- **速度はまだ低い**: Generation 0.3 t/s（ローカル比約1/640）で、RDMA転送のオーバーヘッドが大きい。これは接続ライフタイムとは別の最適化課題（バッチ転送、非同期パイプラインなど）として今後対応が必要
- **接続の再利用が確認できる**: エラーログに接続切断・再接続のメッセージが出なくなり、接続がプログラム終了まで維持されていることが確認できた
