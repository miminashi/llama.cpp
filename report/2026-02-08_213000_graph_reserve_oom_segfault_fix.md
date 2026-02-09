# graph_reserve 失敗時の Segfault 修正レポート

- **実施日時**: 2026年2月8日 21:30

## 前提・目的

`-c` (コンテキストサイズ) を指定せずに GLM-4.7 を 11GPU (7C+4R) で実行すると、CUDA0 でコンピュートバッファの確保に失敗し (OOM)、Segfault が発生する問題を修正する。

- **背景**: `-c 2048` を指定すれば正常動作するが、未指定時にデフォルトコンテキストサイズが CUDA0 のフリーメモリ (~7GB) を超えてしまう
- **目的**: OOM 発生時にクラッシュではなくグレースフルなエラーメッセージで終了させる
- **前提条件**: 1号機 (7 CUDA GPU) と 2号機 (4 CUDA GPU, RDMA接続) の 11GPU クラスタ

## 症状

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 10956.01 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 11488208896
graph_reserve: failed to allocate compute buffers
Segmentation fault (core dumped)
```

`graph_reserve` のエラーログ後に即 Segfault。例外メッセージ (`"failed to allocate compute pp buffers"`) が出力されていない。

## 根本原因

GDB バックトレースにより特定:

```
#0  llama_context::n_ctx (this=0x0) at src/llama-context.cpp:587
#1  server_context_impl::load_model at tools/server/server-context.cpp:637
#2  server_context::load_model at tools/server/server-context.cpp:2856
#3  main at tools/cli/cli.cpp:234
```

`server-context.cpp:637` で `llama_n_ctx(ctx)` を呼ぶ際、`ctx` が **null ポインタ** だった。

### クラッシュフロー

1. `common_init_from_params()` → モデルロード成功、コンテキスト初期化失敗 (OOM)
2. `model` は非 null (モデルロード成功)、`ctx` は null (コンテキスト初期化失敗)
3. Line 630: `if (model == nullptr)` → 通過 (model は有効)
4. Line 637: `n_ctx = llama_n_ctx(ctx)` → **null ポインタデリファレンス → Segfault**

`model == nullptr` のチェックのみで `ctx == nullptr` のチェックが欠落していた。

### 補足: Debug ビルドでは再現しない

- Release (`-O3`): Segfault 再現
- RelWithDebInfo (`-O2 -g`): Segfault 再現
- Debug (`-g`): クラッシュせず exit code 1 で正常終了

Debug ビルドでは例外処理パスの最適化が異なるため、スタックアンワインディング中の挙動が変わる。

## 修正内容

### 1. `tools/server/server-context.cpp` — null チェック追加 (根本原因修正)

`model == nullptr` チェックの後に `ctx == nullptr` チェックを追加:

```cpp
if (model == nullptr) {
    SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
    return false;
}

if (ctx == nullptr) {
    SRV_ERR("failed to create context with model '%s'\n", params_base.model.path.c_str());
    return false;
}
```

### 2. `ggml/src/ggml-rdma/ggml-rdma.cpp` — buffer free の防御的改善

`ggml_backend_rdma_buffer_free_buffer()` で接続有効性を確認し、失敗時は `GGML_ABORT` の代わりにワーニングログを出力:

```cpp
static void ggml_backend_rdma_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rdma_buffer_context * ctx = (ggml_backend_rdma_buffer_context *)buffer->context;
    if (ctx->conn && ctx->conn->is_connected()) {
        rdma_msg_free_buffer_req request = {ctx->remote_ptr};
        bool status = send_rdma_cmd(ctx->conn.get(), RDMA_CMD_FREE_BUFFER, &request, sizeof(request), nullptr);
        if (!status) {
            GGML_LOG_WARN("[rdma] Failed to send FREE_BUFFER to server (connection may be closed)\n");
        }
    }
    delete ctx;
}
```

デストラクタ/クリーンアップパスでの `GGML_ABORT` はプロセス全体を異常終了させるため、ログ出力のみに変更。

## 再現方法

### クラッシュ再現 (修正前)

```bash
# 2号機: rdma-server 起動
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 \
  > /tmp/rdma-server.log 2>&1 &"

# 1号機: -c なしで実行 → Segfault (修正前) / グレースフル終了 (修正後)
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  ./build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 --seed 42 -p 'hello' -n 10 --single-turn --simple-io
```

### バックトレース取得方法

```bash
# core dump パターン設定
sudo sysctl -w kernel.core_pattern=/tmp/core.%e.%p

# core dump 有効化して実行
bash -c 'ulimit -c unlimited && GGML_RDMA_SERVERS=... ./build/bin/llama-cli ...'

# バックトレース取得
gdb ./build/bin/llama-cli /tmp/core.llama-cli.<PID> -batch -ex 'bt full'
```

## テスト結果

### Test 1: クラッシュ修正確認 (`-c` なし OOM)

Release ビルドで 3 回連続実行:

| Run | Exit Code | 結果 |
|-----|-----------|------|
| 1 | 1 | グレースフル終了 |
| 2 | 1 | グレースフル終了 |
| 3 | 1 | グレースフル終了 |

修正前は Exit Code 139 (Segfault) だったものが、エラーメッセージ表示後の正常終了に改善。

出力例:
```
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
srv    load_model: failed to create context with model '/tmp/GLM-4.7-IQ2_M/...'

Loading model...
Failed to load the model
```

### Test 2: 回帰テスト (GLM-4.7 IQ2_M, 11GPU, `-c 2048`)

| 指標 | 値 | 期待値 |
|------|------|------|
| Prompt | 6.4 t/s | ~6.4 t/s |
| Generation | 6.0 t/s | ~6.8 t/s |
| 推論出力 | 正常 (thinking 出力あり) | 正常 |

Generation がやや低い (6.0 vs 期待 6.8) が、サーバーGPU計算時間のセッション間変動 (5.5-7.1 t/s) の範囲内であり、回帰ではない。
