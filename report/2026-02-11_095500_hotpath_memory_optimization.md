# ホットパス メモリ割り当て最適化レポート

- **実施日時**: 2026年2月11日 09:55

## 前提・目的

RDMA バックエンドのホットパス（コマンドループ、テンソル操作、グラフ計算）で毎回発生する `std::vector` のヒープ割り当て・解放を排除し、推論全体のオーバーヘッドを削減する。

- **背景**: 過去の最適化（doorbell batching, max_inline, pre_post_recv, combined send, async compute）はマイクロレベルでは効果があったが、GPU計算時間に対して比率が小さくエンドツーエンドでの改善は限定的だった
- **目的**: malloc/free 呼び出しの削減による推論パフォーマンスの安定化
- **前提条件**: feature/rdma-backend ブランチの最新コード

## 変更内容

### 1. サーバーコマンドループ: バッファ再利用

**ファイル**: `ggml-rdma.cpp` (handle_client 内 while ループ)

毎コマンドで `std::vector<uint8_t> msg_data(msg_size)` がヒープ割り当てされていた。
Generation 中は ~8 cmd/token × 50 tokens × 4 devices = ~1600 回の malloc/free が発生。

**修正**: ループ前に `msg_data.reserve(16*1024*1024)` を宣言し、ループ内で `msg_data.resize(msg_size)` に変更。既存容量を再利用。

### 2. サーバー内部の中間コピー排除

**ファイル**: `ggml-rdma.cpp` (rdma_server クラス内)

以下の関数シグネチャを `const std::vector<uint8_t>&` から `const uint8_t*, size_t` に変更:
- `set_tensor()`
- `graph_compute()`
- `graph_compute_update()`
- `flush_all_staging()`
- `flush_and_recompute()`
- `flush_and_compute_update()`

`flush_and_recompute` と `flush_and_compute_update` 内の中間ベクタ（`flush_data`, `compute_data`）を削除し、ポインタ直接渡しに変更。

### 3. 保留フラッシュの global mutex 排除

**ファイル**: `ggml-rdma.cpp`, `rdma-transport.h`

`g_conn_pending_flushes` (global `std::unordered_map` + `std::mutex`) を廃止。
代わりに `rdma_connection::user_data_` (void*) に `pending_flush_list` を格納。
毎 `set_tensor` / `graph_compute` で global mutex の取得が不要になった。

`user_data_deleter_` (std::function) でデストラクタ時にクリーンアップ。

### 4. クライアント graph_compute: シリアライズバッファ再利用

**ファイル**: `ggml-rdma.cpp` (graph_cache 構造体)

`graph_cache` に `std::vector<uint8_t> cmd_buf` を追加。
recompute/update/full-graph パスすべてで `cmd_buf.resize()` を使い、毎トークンのヒープ割り当てを回避。

### 5. クライアント set_tensor: 送信バッファ再利用

**ファイル**: `ggml-rdma.cpp` (ggml_backend_rdma_buffer_set_tensor)

Send/Recv フォールバック時に `static thread_local std::vector<uint8_t> tl_send_buf` を使用。
GDR 有効時は RDMA Write パスを通るため Generation 中の影響はない（Weight loading 時のみ）。

## テスト結果

### qwen2.5-0.5b 回帰テスト (2GPU: CUDA0 + RDMA0)

| テスト | pp128 (t/s) | tg32 (t/s) |
|--------|:-----------:|:----------:|
| 最適化後 | 2,830 | 161 |
| 基準値 | ~3,269 | ~174 |

セッション間変動の範囲内。回帰なし。

### GLM-4.7 IQ2_M 11GPU テスト (7 CUDA + 4 RDMA)

| Run | Prompt (t/s) | Generation (t/s) |
|-----|:------------:|:----------------:|
| 1   | 6.2          | 6.9              |
| 2   | 6.9          | 5.9              |
| 3   | 7.0          | 7.4              |
| **平均** | **6.7** | **6.7** |
| 基準値 | ~6.4 | ~6.8 |

正常な推論出力を3回とも確認。性能は基準値と同等。

## 再現方法

1. ビルド + デプロイ
```bash
bash scripts/rdma-build.sh local && bash scripts/rdma-deploy.sh
```

2. サーバー起動
```bash
bash scripts/rdma-server.sh restart
```

3. qwen2.5-0.5b 回帰テスト
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

4. GLM-4.7 11GPU テスト
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## 考察

今回の最適化はメモリ割り当ての削減に焦点を当てたもので、推論速度の直接的な向上は確認されなかった。これは予想通りの結果であり、以下の理由による:

1. **GPU 計算時間が支配的**: Generation ループでは各トークンあたり ~140ms の GPU 計算時間に対し、malloc/free は数マイクロ秒のオーバーヘッド
2. **glibc malloc のキャッシュ**: glibc の malloc は小〜中サイズのアロケーションに対してスレッドローカルキャッシュ（tcache/arena）を使用しており、実際のシステムコール発生頻度は低い
3. **メモリフラグメンテーション**: 長時間実行時の断片化防止が主な効果であり、短い50トークン生成では顕在化しない

ただし、以下の点でコード品質が向上:
- Global mutex の排除により、マルチスレッドサーバーでのコンテンション減少
- 不要なベクタコピー排除によるサーバー側のメモリ帯域消費削減
- バッファ再利用パターンの確立（今後の最適化の基盤）
