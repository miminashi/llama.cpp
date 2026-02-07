# マルチRDMAデバイス出力破損バグ修正レポート

- **実施日時**: 2026年2月7日 19:55
- **参照レポート**: [GLM-4.7 IQ2_M RDMA vs RPC 出力品質レポート](2026-02-07_124327_glm47_iq2m_rdma_vs_rpc_output_quality.md)

## 前提・目的

RDMAバックエンドで複数のリモートGPUを使用すると出力が破損するバグを修正する。

- **背景**: GLM-4.7 IQ2_Mを7C+4R (11GPU)で実行するとゴミ出力、gpt-oss-20bを1C+2R (3GPU)で実行すると空出力になる。同一モデルをRPCバックエンドやローカルで実行すると正常。
- **目的**: 根本原因を特定し、マルチRDMAデバイスでの正常な推論を実現する
- **前提条件**: 1号機(7GPU) + 2号機(4GPU)がIBで接続。`GGML_RDMA_NO_GDR=1`で動作。

## 診断過程

### 1. データ転送の正常性確認

サーバー側の `copy_tensor` に診断ログを追加し、GPU間コピー前後のデータを検証:

```
src_before=[-0.048012, 0.082197, 0.010867, 0.009743]
dst_after =[-0.048012, 0.082197, 0.010867, 0.009743]
```

結果: **データコピー自体は正しく動作** (`result=1`, src/dst のfloat値が一致)。

### 2. logits argmax分析

クライアント側 `get_tensor` にargmax計算を追加し、1R vs 2R の logits を比較:

| 構成 | argmax値 | max_val | 動作 |
|------|----------|---------|------|
| 1C+1R | 200005, 35644, 200008, 976... | 41.8, 49.4, 35.5, 26.3... | 正常 (thinking tokens) |
| 1C+2R | 1402, 7633, 899, 20... | 13.4, 19.8, 15.0, 17.4... | 異常 (ランダムtokens) |

結果: **2Rのlogitsはダイナミックレンジが~50%低い**。データ転送は正常だが計算結果が異なる → モデル計算自体が間違っている。

### 3. cpy_tensor 無効化テスト

`cpy_tensor` を常に `false` を返すように変更し、全デバイス間コピーを `get_tensor + set_tensor` フォールバックに強制:

```
1C+2R: "Paris" → 正常出力！
```

**根本原因が確定**: サーバー側 `copy_tensor` がバグの原因。

## 根本原因

### 問題のメカニズム

RDMAバックエンドは複数デバイスで1つの接続を共有する。そのため `cpy_tensor` が `conn mismatch` にならず、サーバー側 `copy_tensor` (GPU間 `cudaMemcpyPeer`) が実行される。

`copy_tensor` はデータを正しくコピーするが、**グラフキャッシュ (`graph_recompute`) が参照するテンソルのメモリアドレスと `copy_tensor` のコピー先が一致しない**ため、計算時に古いデータが使われる。

### RPCとの違い

RPC: デバイスごとに別プロセス・別ソケット → `cpy_tensor` は常に `sock != sock` で `false` → `get_tensor + set_tensor` フォールバック → 正常動作

RDMA: 全デバイスが1接続を共有 → `cpy_tensor` が `conn == conn` で実行 → サーバー側GPU間コピー → グラフキャッシュとの不整合

## 修正内容

### ファイル: `ggml/src/ggml-rdma/ggml-rdma.cpp`

`cpy_tensor` を常に `false` を返すように変更し、全デバイス間コピーを `get_tensor + set_tensor` フォールバックに強制:

```cpp
static bool ggml_backend_rdma_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
    const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}
```

加えて、`get_tensor` のRDMA Read パスをGDR無効時にスキップ (stale staging buffer問題の修正):

```cpp
static const bool no_gdr = (std::getenv("GGML_RDMA_NO_GDR") != nullptr);
if (ctx->mr_rkey != 0 && ctx->staging && ctx->base_ptr != nullptr && !no_gdr) {
```

## 検証結果

### gpt-oss-20b (1C+2R)

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| 出力 | 空 (empty) | "Paris" (正常) |
| pp (t/s) | 53.9 | 53.6 |
| tg (t/s) | 47.1 | 46.3 |

### GLM-4.7 IQ2_M (7C+4R, 11GPU)

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| 出力 | ゴミ文字列 | 正常 (thinking + Paris) |
| pp (t/s) | - | 6.6-8.1 |
| tg (t/s) | - | 5.9-6.6 |

### パフォーマンス影響

生成速度: ~3%低下 (46.3 vs 47.1 t/s for gpt-oss-20b)。デバイス間境界テンソル (norm, l_out) のみが影響を受けるため、影響は最小限。

## 再現方法

### 1. ビルド・デプロイ

```bash
# 1号機
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cd build && make -j$(nproc)

# 2号機
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/ 192.168.100.2:/home/ubuntu/projects/llama.cpp/
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cd build && make -j\$(nproc)"
```

### 2. サーバー起動

```bash
ssh 192.168.100.2 "GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
```

### 3. テスト実行

```bash
# gpt-oss-20b 1C+2R
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf \
  -dev 'CUDA0,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log

# GLM-4.7 IQ2_M 7C+4R
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 200 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```
