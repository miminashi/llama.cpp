---
name: build-deploy
description: RDMA backend build, deploy, and server management procedures. Use when building llama.cpp, deploying to remote node, or managing rdma-server.
user-invocable: true
---

# ビルド・デプロイ・サーバー管理

> **注意**: SSH/rsync コマンドは `settings.local.json` に `Bash(ssh *)` 等を含めれば直接実行可能。
> ただし、ビルド・デプロイ・サーバー管理にはラッパースクリプト (`scripts/rdma-*.sh`) を推奨
> （エラーハンドリング・ログ管理が組み込まれているため）。
> 別ワークツリーから呼ぶ場合は絶対パスを使う: `bash /path/to/worktree/scripts/rdma-build.sh local`

## 1号機 (192.168.100.1) でビルド

```bash
bash scripts/rdma-build.sh local
```

## 2号機 (192.168.100.2) へのデプロイ (コード転送 + ビルド)

```bash
bash scripts/rdma-deploy.sh
```

## 両ノードでビルド (1号機ビルド + 2号機デプロイ)

```bash
bash scripts/rdma-build.sh local && bash scripts/rdma-deploy.sh
```

## rdma-server 管理 (2号機)

```bash
bash scripts/rdma-server.sh start    # 起動
bash scripts/rdma-server.sh stop     # 停止
bash scripts/rdma-server.sh restart  # 再起動
bash scripts/rdma-server.sh status   # 状態確認
bash scripts/rdma-server.sh log      # ログ表示
```

## ビルドコマンドに関する注意

- `--config Release` は Unix Makefiles ジェネレータでは無視されるため不要（`CMAKE_BUILD_TYPE=Release` は cmake configure 時に自動設定される）
- `cmake --build build` の代わりに `make -C build -j16` でも同等に動作する。**注意**: `-j$(nproc)` は `$()` コマンド置換のため自動承認されない。先に `nproc` で値を確認し `-j16` のようにリテラル指定すること
- ビルドログは `/tmp/cmake_configure.log` (configure) と `/tmp/build.log` (build) に出力される
- ビルド出力を `| tail` や `| head` で絞るパイプパターンは自動承認されないため避ける。出力が長い場合は Bash ツールのキャプチャ出力を確認する
