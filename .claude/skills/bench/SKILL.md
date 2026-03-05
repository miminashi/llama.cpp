---
name: bench
description: Benchmark execution and verification procedures for RDMA backend. Use when running llama-bench, llama-cli, testing with GLM-4.7, or checking environment variables and error troubleshooting.
user-invocable: true
---

# ベンチマーク実行・検証手順

## llama-bench 実行 (1号機)

**qwen2.5-0.5b 2GPU (CUDA0 + RDMA0)**
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -fa 1 -r 1 -p 128 -n 32
```

**gpt-oss-20b 2GPU (CUDA0 + RDMA0)**
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -fa 1 -r 1 -p 128 -n 32
```

### マルチファイルGGUFの使用

`llama-bench` は `-hf` フラグをサポートしていないため、HuggingFaceキャッシュ内のマルチファイルGGUF（gpt-oss-120bなど）を直接指定するとスプリットファイルの検出に失敗する。

**回避策**: 標準的なファイル名でシンボリックリンクを作成する

```bash
mkdir -p /tmp/gpt-oss-120b
ln -sf /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-120b-GGUF_Q4_K_M_gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
       /tmp/gpt-oss-120b/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf
ln -sf /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-120b-GGUF_Q4_K_M_gpt-oss-120b-Q4_K_M-00002-of-00002.gguf \
       /tmp/gpt-oss-120b/gpt-oss-120b-Q4_K_M-00002-of-00002.gguf
```

### `-dev` オプションのセパレータ

- **llama-cli**: カンマ区切り `,` を使用 (例: `CUDA0,CUDA1,RDMA0`)
- **llama-bench**: スラッシュ区切り `/` を使用 (例: `CUDA0/CUDA1/RDMA0`)

```bash
llama-bench -m /tmp/gpt-oss-120b/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -fa 1 -r 1 -p 128 -n 32
```

## llama-cli 実行 (1号機, 11GPU クラスタ)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -hf unsloth/gpt-oss-120b-GGUF:Q4_K_M \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -p 'こんにちは' -n 50 \
  --flash-attn on --no-warmup --single-turn --simple-io \
  --log-file /tmp/llama-cli.log
```

## 環境変数一覧

| 環境変数 | 説明 | デフォルト |
|---------|------|----------|
| `GGML_RDMA_SERVERS` | RDMA サーバーリスト (host:port) | 未設定 |
| `GGML_RDMA_NO_GDR` | `1` で GPUDirect RDMA を無効化 | 未設定 (GDR有効) |
| `GGML_RDMA_GDR_BUDGET_GB` | GPUDirect MR 登録の合計サイズ上限 (GB)。超過分はホストステージングにフォールバック | 12 |
| `GGML_RDMA_NO_STAGING` | `1` でホストステージングバッファを無効化 (Send/Recvフォールバック) | 未設定 |
| `GGML_RDMA_PROFILE` | `1` でクライアント側プロファイリング有効化 | 未設定 |
| `GGML_RDMA_DEBUG` | `1` でデバッグログ出力 | 未設定 |
| `GGML_RDMA_TIMEOUT_MS` | RDMA 操作 (Send/Recv/Write/Read) のタイムアウト (ms) | 30000 |
| `GGML_RDMA_COMPUTE_TIMEOUT_MS` | graph_compute 応答待ちのタイムアウト (ms) | 300000 |
| `GGML_RDMA_ASYNC_COMPUTE` | `0` で graph_compute の fire-and-forget を無効化 (デバッグ用) | 未設定 (有効) |
| `GGML_RDMA_NO_DEFERRED_COPY` | `1` で deferred copy を無効化 (`cpy_tensor` が常に false を返す) | 未設定 (有効) |
| `GGML_RDMA_VERIFY_COPY` | `1` で deferred copy の検証モード有効化 (デバッグ用) | 未設定 |
| `GGML_RDMA_PER_DEVICE_CONN` | `1` でデバイスごとに独立した RDMA 接続を使用 (ConnectX-6+ 向け) | 未設定 (共有接続) |
| `GGML_RDMA_NO_SELECTIVE_SIGNAL` | `1` で Send selective signaling を無効化 | 未設定 (有効) |
| `GGML_RDMA_PARALLEL_DISPATCH` | `0` で parallel dispatch を無効化 | 未設定 (有効) |
| `GGML_RDMA_PARALLEL_COMPUTE` | `0` で parallel compute を無効化 | 未設定 (有効) |
| `GGML_RDMA_NO_SERVER_PUSH` | `1` でサーバープッシュを無効化 (per-device conn 時のみ有効) | 未設定 |
| `GGML_RDMA_PIPELINE` | `1` で pipeline dispatch を有効化 (split-level pipeline parallelism) | 未設定 (無効) |
| `GGML_RDMA_PARALLEL` | `1` で `GGML_RDMA_PIPELINE` + `GGML_RDMA_PER_DEVICE_CONN` を同時有効化 | 未設定 (無効) |

## よくあるエラーと対処法

| エラー | 原因 | 対処法 |
|-------|------|--------|
| `CUDA illegal memory access` | `supports_buft` のバグ (修正済み) またはクロスデバイスアクセス | コードが最新か確認。2号機のバイナリが古い可能性 |
| `Connection refused` | rdma-server が起動していない | 2号機で `ps aux \| grep rdma` 確認、サーバー再起動 |
| `RDMA write completion timeout` | ステージングバッファの MR 情報不一致 | サーバーを再起動してバッファ再登録 |
| `chunk recv` ログ大量出力 | 大きなテンソル (>16MB) の転送 | 正常動作。MoEモデルで頻発 |
| `llama_params_fit` クラッシュ | スレッド安全性の問題 (修正済み) | `op_mutex_` による保護が有効か確認 |
| 2秒スパイク (RNR NAK) | Send/Recv タイミング不一致 | `min_rnr_timer=1` 確認。HW counter: `.../hw_counters/rnr_nak_retry_err` |
| GDR + parallel dispatch でクラッシュ | `conn->recv()` 中に RNIC GDR Write → CUDA 競合 | `GGML_RDMA_PARALLEL_DISPATCH=0` |
| ゴミ文字出力 (`>>9-A($%,...`) | Send buffer reuse race | always-signal fix 確認 (commit `9a5a4a911`) |
| NaN/`????` 出力 | RDMA Write staging buffer race | always-signal fix 確認 (commit `e5700e0f8`) |

## 検証方法

- 各ステップで `llama-bench` または `llama-cli` によるベンチマーク実行
- レポートは `report/` ディレクトリに `REPORT.md` のフォーマットに従って記録
- 性能数値は Prompt t/s と Generation t/s の両方を計測
- 実験前に `bash scripts/rdma-env-check.sh` を実行し、警告がないことを確認する
- レポート作成時は `bash scripts/rdma-env-check.sh --markdown` の出力を含める

### 11GPU クラスタテストの必須ルール

- **最終テストには必ず GLM-4.7 (IQ2_M) を使用すること**
- 作業中の動作確認や回帰テストで小さいモデル (qwen2.5-0.5b, gpt-oss-20b 等) を使うのは OK
- ただし、変更の最終検証は必ず GLM-4.7 を 11GPU (7C+4R) 構成で実行し、正常な推論出力と性能を確認すること
- GLM-4.7 テストコマンド:

```bash
# サーバー起動 (2号機)
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 \
  > /tmp/rdma-server.log 2>&1 &"

# 推論 (1号機)
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --flash-attn on --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

- 期待値: Prompt ≈ 24.3 t/s, Generation ≈ 8.5 t/s (GDR 有効, flash attention 有効時)

A/B 性能比較には ABAB Paired Design + 対応あり t 検定を使用。詳細: `/stats` スキル参照。

## ベンチマークスクリプト Tips

- CSV フィールドはクォート付き: `grep '"128","0","0"'` を使う (`grep ',128,0,'` は不可)
- avg_ts パース: `sed 's/.*,"\([0-9.]*\)","\([0-9.]*\)"$/\1/'` (末尾から2番目のフィールド)
- `static const` 環境変数 (`GGML_RDMA_NO_SELECTIVE_SIGNAL` 等) のトグル時はサーバー再起動が必要
- `set -e` はベンチスクリプトでサイレント失敗を引き起こす → bench コマンドには `|| true` を付ける
- 特殊文字を含むモデルパスでは long flags を推奨: `--model`, `--repetitions`, `--n-prompt`, `--n-gen`
- `| head` / `| tail` パイプは専用ツール強制のハードコード制限で Bash 自動承認不可。出力フィルタリングが必要な場合は Bash ツールのキャプチャ出力を確認する

## RPC バックエンド使用法

RPC バックエンドとの比較ベンチマーク時に使用:

- `--rpc host:port1,host:port2` フラグを使用 (`-dev RPC[...]` は不可)
- 各 `rpc-server` は 1 GPU 担当 (`CUDA_VISIBLE_DEVICES` で制限)

```bash
CUDA_VISIBLE_DEVICES=0 build/bin/rpc-server -H 0.0.0.0 -p 50052
CUDA_VISIBLE_DEVICES=1 build/bin/rpc-server -H 0.0.0.0 -p 50053

CUDA_VISIBLE_DEVICES=0 build/bin/llama-bench \
  --rpc 192.168.100.2:50052,192.168.100.2:50053 \
  -m model.gguf -ngl 999 -sm layer -fa 1
```
