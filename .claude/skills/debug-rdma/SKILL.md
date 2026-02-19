---
name: debug-rdma
description: GDB debugging procedures for RDMA backend including debug builds, gdb helper commands, breakpoint presets, and typical debug workflows.
user-invocable: true
---

# GDB デバッグ

printf デバッグの代替として gdb (v15.0.50) を使用できる。Debug ビルドと RDMA 固有の gdb ヘルパーを提供。

## Debug ビルド

```bash
bash scripts/rdma-build.sh local debug
bash scripts/rdma-deploy.sh debug
```

Debug ビルドは `-O0 -g3` で最適化を無効にし、全てのデバッグシンボルを含む。ステップ実行や変数検査に最適だが、実行速度は大幅に低下する。

> **重要**: Debug ビルドでは RDMA タイムアウトを延長すること。
> `GGML_RDMA_TIMEOUT_MS=120000` (2分) を推奨。デフォルト30秒ではブレークポイント停止中にタイムアウトする。

## クライアント側デバッグ (1号機)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_TIMEOUT_MS=120000 \
  bash scripts/rdma-debug.sh cli \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -c 256 -n 10 -p 'hello' \
  --no-warmup --single-turn --simple-io
```

## サーバー側デバッグ (2号機)

```bash
bash scripts/rdma-server.sh debug     # gdb 付きフォアグラウンド起動
bash scripts/rdma-server.sh attach    # 実行中プロセスにアタッチ
```

> **注意**: `debug` と `attach` は `ssh -t` (対話的ターミナル) を使用するため、Claude Code の自動承認は効かない。開発者がターミナルで直接実行する。

## gdb コマンド一覧

| コマンド | 説明 |
|---------|------|
| `rdma-conn <ptr>` | `rdma_connection` の状態表示 (endpoint, QP, stats) |
| `rdma-buf <ptr>` | バッファコンテキスト (remote_ptr, mr_rkey, GDR, size) |
| `rdma-ctx <ptr>` | バックエンドコンテキスト (endpoint, device, conn) |
| `rdma-wc <ptr>` | Work Completion エントリ (status, vendor_err) |
| `rdma-mr <ptr>` | Memory Region (addr, length, lkey, rkey) |
| `rdma-cmd-name <id>` | コマンドID→名前変換 (0=ALLOC_BUFFER, 9=GRAPH_COMPUTE, ...) |
| `rdma-threads` | スレッド一覧 + RDMA スレッドのヒント |

## ブレークポイントプリセット

| コマンド | ブレークポイント数 | 対象 |
|---------|:--:|------|
| `rdma-bp-errors` | 5 | Send/Recv/Write/Read タイムアウト + WC エラー |
| `rdma-bp-commands` | 1 | サーバーコマンドディスパッチ (トレース出力付き) |
| `rdma-bp-compute` | 4 | GRAPH_COMPUTE 関連コマンドのみ |

## 典型的なデバッグワークフロー

1. Debug ビルド: `bash scripts/rdma-build.sh local debug && bash scripts/rdma-deploy.sh debug`
2. サーバー gdb 起動: `bash scripts/rdma-server.sh debug` → gdb で `rdma-bp-errors` → `run`
3. クライアント gdb 起動: `bash scripts/rdma-debug.sh cli ...` → gdb で `rdma-bp-errors` → `run`
4. エラー発生時に停止 → `rdma-conn`, `rdma-wc` で状態確認
