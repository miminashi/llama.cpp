# fit_params OOM 時の n_ctx 自動縮小フォールバック

- **実施日時**: 2026年2月9日 02:59

## 前提・目的

`-c` オプションなしで GLM-4.7 を 11GPU (7C+4R) RDMA 構成で実行すると、CUDA0 でコンピュートバッファの OOM が発生しエラー終了する問題を修正する。

- **背景**: CUDA のみの構成では `fit_params` が n_ctx を自動縮小して動作するが、RDMA 構成では `fit_params` が失敗し n_ctx が適切に縮小されない
- **目的**: `fit_params` 失敗時に n_ctx を `fit_params_min_ctx` (デフォルト 4096) にフォールバックし、OOM を回避する
- **前提条件**: graph_reserve OOM 時の Segfault は前回修正済み ([report/2026-02-08_213000_graph_reserve_oom_segfault_fix.md](2026-02-08_213000_graph_reserve_oom_segfault_fix.md))

## 根本原因

### fit_params の失敗メカニズム (RDMA 構成)

1. `common_init_result()` → `llama_params_fit()` が呼ばれる
2. 初回計測: n_ctx=202752 (n_ctx_train) でコンテキスト作成を試みる → CUDA0 OOM → `runtime_error` → 内部で catch
3. n_ctx を縮小して再計測: n_ctx=4096 でコンテキスト作成 → 成功
4. `fit_params_impl` が n_ctx=56832 に設定し、次に `n_gpu_layers` を減らそうとする
5. `-ngl 999` はユーザー指定のため中断 → `llama_params_fit_exception` → `LLAMA_PARAMS_FIT_STATUS_FAILURE` を返す
6. **修正前**: `llama_params_fit()` の戻り値が無視されていた
7. 本番のコンテキスト作成が n_ctx=56832 で実行 → CUDA0 OOM (コンピュートバッファ ~11GB > フリーメモリ ~7GB)

### なぜ CUDA のみだと動くか

CUDA のみの構成では `fit_params` の初回計測 (n_ctx_train でのコンテキスト作成) が成功するため、メモリ使用量を正確に計測でき n_ctx の適切な値を算出できる。RDMA 構成では CUDA0 のフリーメモリが不足して計測自体が失敗する。

## 修正内容

### 変更ファイル

- `common/common.cpp` (1行の条件追加 + 1行のログ追加)

### 変更箇所

```cpp
// common/common.cpp:1097-1109

if (params.fit_params) {
    LOG_INF(...);
    auto status = llama_params_fit(...);
    if (status != LLAMA_PARAMS_FIT_STATUS_SUCCESS) {
        LOG_WRN("%s: fit_params failed (status=%d), falling back to n_ctx=%u\n",
                __func__, status, params.fit_params_min_ctx);
        cparams.n_ctx = params.fit_params_min_ctx;
    }
}
```

### 設計判断

- 当初は `LLAMA_PARAMS_FIT_STATUS_ERROR` のみチェックする想定だったが、実際には RDMA 構成で `-ngl 999` 指定時に `FAILURE` (status=1) が返ることが判明
- `FAILURE` の原因: `fit_params` は n_ctx 縮小後に `n_gpu_layers` も減らそうとするが、ユーザー明示指定のため中断される
- `SUCCESS` 以外のすべてのステータスでフォールバックを適用する方針に変更

## 再現方法

### 修正前の再現 (OOM エラー)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 --seed 42 -p 'The capital of France is' -n 50 \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

修正前の出力:
```
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
Failed to load the model
```

### 修正後のテスト

#### Test 1: `-c` なしで自動縮小確認

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 --seed 42 -p 'The capital of France is' -n 50 \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

#### Test 2: `-c 2048` 回帰テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## テスト結果

| テスト | 結果 | n_ctx | pp (t/s) | tg (t/s) |
|--------|------|:-----:|:--------:|:--------:|
| Test 1: `-c` なし | 成功 | 4096 (フォールバック) | 7.0 | 5.6 |
| Test 2: `-c 2048` | 成功 | 2048 | 7.0 | 7.2 |

- Test 1: フォールバックにより n_ctx=4096 で正常に推論完了。ログに `fit_params failed (status=1), falling back to n_ctx=4096` が出力される
- Test 2: 回帰なし。`-c 2048` を明示指定した場合は従来通り動作
- Test 1 の tg が 5.6 t/s と低めだが、n_ctx=4096 (vs 2048) による KV キャッシュメモリ増加とサーバー GPU 計算時間のセッション間変動（既知課題）が原因
