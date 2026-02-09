# サーバー側コマンド受信タイムアウト修正レポート

- **実施日時**: 2026年2月8日 16:30
- **関連レポート**: [RDMA Completion タイムアウト修正](2026-02-08_140512_rdma_completion_timeout_fix.md)

## 前提・目的

インタラクティブモード（`llama-cli` で `--single-turn` なし）使用時に、ユーザーの入力待ちの間にサーバー側のコマンド受信ループがタイムアウトして接続が切断される問題を修正する。

- **背景**: 前回の修正でクライアント側の `graph_compute` 応答待ちタイムアウトを 5 分に延長したが、サーバー側のコマンド受信ループ（`ggml-rdma.cpp:3034`）ではデフォルトの `RDMA_TIMEOUT_MS`（30秒）が適用されたままだった
- **問題の流れ**:
  1. クライアントがトークン生成完了 → `> ` プロンプト表示
  2. ユーザーが入力待ちの間、クライアントはサーバーに何も送信しない
  3. サーバーの `conn->recv(header_buf, ...)` が 30 秒後にタイムアウト
  4. サーバーがコマンドループを抜けて接続を切断
  5. ユーザーが次のプロンプトを入力 → クライアントが切断済みサーバーに送信 → 失敗
- **ログの証拠**: サーバーログに `timeout=30000 ms` → `Completion timeout after 30000 ms` が出力
- **目的**: サーバーのコマンド受信を無限待ちに変更し、インタラクティブモードでの接続維持を実現する

## 変更内容

### 変更ファイル

1. **`ggml/src/ggml-rdma/rdma-transport.cpp`** — `recv()` に `timeout_ms = 0`（無限待ち）サポート + `wait_for_completion()` に CPU 節約 sleep 追加
2. **`ggml/src/ggml-rdma/ggml-rdma.cpp`** — サーバーコマンドループで `timeout_ms = 0` を使用

### 詳細

#### rdma-transport.cpp: recv() のタイムアウト解釈変更

`timeout_ms` パラメータの意味を拡張:

| 値 | 意味 | 用途 |
|----|------|------|
| `-1`（デフォルト） | `RDMA_TIMEOUT_MS`（30秒）を使用 | 通常の RDMA 操作 |
| `0` | 無限待ち（タイムアウトなし） | サーバーのコマンド待ち |
| `> 0` | 指定値を使用 | `RDMA_COMPUTE_TIMEOUT_MS` 等 |

内部実装: `timeout_ms = 0` を `effective_timeout = -1` に変換し、`wait_for_completion()` に渡す。`wait_for_completion()` は `timeout_ms < 0` の場合タイムアウト判定をスキップ（既存ロジック）。

#### rdma-transport.cpp: wait_for_completion() の CPU 節約

無限待ち（`timeout_ms < 0`）の場合、1秒経過後に `usleep(1000)`（1ms sleep）を挿入。

- **通常の RDMA 操作**（`timeout >= 0`）: 常にビジーポーリング（最低レイテンシ優先）
- **サーバーのコマンド待ち**（`timeout < 0`）: 最初の1秒はビジーポーリング、以降は 1ms sleep で CPU 使用率を低減
- 1ms の sleep レイテンシはコマンド待ちでは許容範囲（次のコマンドは ms 単位で遅延しても問題ない）

#### ggml-rdma.cpp: サーバーコマンドループ

```cpp
// 変更前:
if (!conn->recv(header_buf, cmd_header_size, nullptr)) {

// 変更後:
if (!conn->recv(header_buf, cmd_header_size, nullptr, 0)) {  // 0 = no timeout
```

クライアント切断時は RDMA ハードウェアが QP エラーを通知するため、`ibv_poll_cq()` がエラーを返して `recv` は即座に失敗する（タイムアウト不要）。

## テスト結果

### 環境

- 1号機: 7x P100-PCIE-16GB (CUDA0-6)
- 2号機: 4x P100-PCIE-16GB (RDMA0-3)
- ネットワーク: InfiniBand (ConnectX-4)

### Test 1: 回帰テスト (GLM-4.7 IQ2_M, 11GPU, --single-turn)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

| 指標 | 結果 |
|------|------|
| Prompt | 6.3 t/s |
| Generation | 6.9 t/s |
| タイムアウト | なし |
| 判定 | **PASS** (期待: pp ~6.4, tg ~6.8) |

### Test 2: インタラクティブモード — 45秒アイドル待機テスト (qwen2.5-0.5b, 1C+1R)

FIFO を使って自動化。1回目のプロンプト送信 → 生成完了 → **45秒の無操作** → 2回目のプロンプト送信。

```bash
# FIFO経由でプロンプトを送信するスクリプトで実行
# 1回目: "Hello, what is your name?" → 30トークン生成
# 45秒待機（旧30秒タイムアウトを超過）
# 2回目: "What is 1+1?" → 30トークン生成
```

| 指標 | 1回目 | 2回目 |
|------|-------|-------|
| Prompt | 232.9 t/s | 488.0 t/s |
| Generation | 157.2 t/s | 153.0 t/s |
| タイムアウト | なし | なし |
| 判定 | **PASS** | **PASS** |

**サーバーログの確認**:

```
[rdma_connection] Still waiting for completion: 10002 ms elapsed (timeout=-1 ms)
[rdma_connection] Still waiting for completion: 20002 ms elapsed (timeout=-1 ms)
[rdma_connection] Still waiting for completion: 30002 ms elapsed (timeout=-1 ms)
[rdma_connection] Still waiting for completion: 40003 ms elapsed (timeout=-1 ms)
[rdma_connection] Still waiting for completion: 50003 ms elapsed (timeout=-1 ms)
[rdma_connection] Still waiting for completion: 60003 ms elapsed (timeout=-1 ms)
[rdma_connection] Still waiting for completion: 70003 ms elapsed (timeout=-1 ms)
```

- `timeout=-1 ms` = 無限待ちが正しく適用されている
- 30秒超過後も `Completion timeout` エラーなし
- 2回目の生成が正常に完了

### 結果サマリ

| テスト | モデル | 構成 | 待機時間 | 2回目生成 | 結果 |
|--------|--------|------|----------|-----------|------|
| Test 1 (回帰) | GLM-4.7 IQ2_M | 11GPU | - | - | PASS |
| Test 2 (45秒idle) | qwen2.5-0.5b | 1C+1R | 45秒 | 正常完了 | PASS |

## 結論

- サーバーのコマンド受信ループを無限タイムアウトに変更し、インタラクティブモードでの接続維持を実現
- 45秒のアイドル待機（旧30秒タイムアウト超過）後も2回目の推論が正常に完了
- 無限待ち時は 1ms sleep で CPU 使用率を低減（通常 RDMA 操作のレイテンシには影響なし）
- 回帰テスト (GLM-4.7 IQ2_M, 11GPU) で性能回帰なし (pp=6.3, tg=6.9 t/s)

## 再現方法

### 前提条件

- 1号機 (192.168.100.1): ソースコードとモデルが配置済み
- 2号機 (192.168.100.2): 1号機からデプロイ済み
- 前回の修正 ([RDMA Completion タイムアウト修正](2026-02-08_140512_rdma_completion_timeout_fix.md)) が適用済み

### 1. ビルド・デプロイ

CLAUDE.md の「ビルド・デプロイ・実行手順」に従う。

### 2. サーバー起動 (2号機)

```bash
ssh 192.168.100.2 "pkill -9 -f rdma-server" 2>/dev/null; sleep 3
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 \
  > /tmp/rdma-server.log 2>&1 &"
```

### 3. インタラクティブモードテスト (1号機)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -dev 'CUDA0,RDMA0[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 30 --seed 42 \
  --simple-io --log-file /tmp/llama-cli.log
```

1. プロンプト入力 → トークン生成完了を確認
2. **45秒以上待機**（旧タイムアウト30秒を超過）
3. 次のプロンプト入力 → 2回目の生成が正常に完了することを確認
4. 別ターミナルで `ssh 192.168.100.2 "tail -f /tmp/rdma-server.log"` を確認し、`Completion timeout` が出ないこと
