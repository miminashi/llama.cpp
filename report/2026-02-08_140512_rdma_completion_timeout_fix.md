# RDMA Completion タイムアウト修正レポート

- **実施日時**: 2026年2月8日 14:05

## 前提・目的

GLM-4.7 で長い出力トークンを生成する推論時に、`[rdma_connection] Completion timeout after 30000 ms` が発生して推論が中断される問題を修正する。

- **背景**: RDMA の全操作に一律 30 秒のハードコードされたタイムアウトが適用されており、`graph_compute` コマンドの応答待ち（サーバー GPU 計算完了を待つ recv）がこの制限を超えうる
- **根本原因**: `send_rdma_cmd()` が graph_compute コマンドを送信後、サーバーの応答を `conn->recv()` で待機。`recv()` 内部の `wait_for_completion(30000)` が 30 秒でタイムアウト。サーバーは GPU 計算完了後にしか応答を返さないため、長い推論では計算時間が 30 秒を超えうる
- **目的**: graph_compute 操作に専用の長いタイムアウト（デフォルト 5 分）を導入し、通常の RDMA 操作とは分離する

## 変更内容

### 変更ファイル

1. **`ggml/src/ggml-rdma/rdma-transport.h`** — `recv()` にタイムアウトパラメータ追加
2. **`ggml/src/ggml-rdma/rdma-transport.cpp`** — タイムアウトの環境変数化 + 進捗ログ追加
3. **`ggml/src/ggml-rdma/ggml-rdma.cpp`** — compute タイムアウトの導入 + 診断ログ追加
4. **`CLAUDE.md`** — 環境変数テーブルに追記

### 詳細

#### rdma-transport.cpp

- `GGML_RDMA_TIMEOUT_MS` 環境変数でデフォルト RDMA タイムアウトを設定可能に（デフォルト 30000ms）
- `send()`, `rdma_write()`, `rdma_read()` の 4 箇所のハードコード `30000` を `RDMA_TIMEOUT_MS` に置換
- `recv()` にタイムアウトパラメータ追加（`-1` でデフォルト値を使用）
- `wait_for_completion()` に 10 秒ごとの進捗ログを追加（`GGML_LOG_WARN` で出力）

#### ggml-rdma.cpp

- `GGML_RDMA_COMPUTE_TIMEOUT_MS` 環境変数で graph_compute 応答待ちタイムアウトを設定可能に（デフォルト 300000ms = 5 分）
- `send_rdma_cmd()` に `recv_timeout_ms` パラメータ追加
- graph_compute の 3 箇所（`FLUSH_AND_RECOMPUTE`, `FLUSH_AND_COMPUTE_UPDATE`, `GRAPH_COMPUTE`）に `RDMA_COMPUTE_TIMEOUT_MS` を渡す
- graph_compute 完了後に 10 秒超の場合は常時警告ログを出力（`RDMA_PROFILE` 不要）

### 新規環境変数

| 環境変数 | 説明 | デフォルト |
|---------|------|----------|
| `GGML_RDMA_TIMEOUT_MS` | RDMA 操作 (Send/Recv/Write/Read) のタイムアウト (ms) | 30000 |
| `GGML_RDMA_COMPUTE_TIMEOUT_MS` | graph_compute 応答待ちのタイムアウト (ms) | 300000 |

## テスト結果

### 環境

- 1号機: 7x P100-PCIE-16GB (CUDA0-6)
- 2号機: 4x P100-PCIE-16GB (RDMA0-3)
- ネットワーク: InfiniBand (ConnectX-4)
- モデル: GLM-4.7-UD-IQ2_M (~40GB)

### Test 1: 短い生成 (50 tokens) — 回帰テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
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

### Test 2: 中程度の生成 (200 tokens)

```bash
-p 'Write a detailed essay about the history of artificial intelligence' -n 200 --seed 42
```

| 指標 | 結果 |
|------|------|
| Prompt | 8.9 t/s |
| Generation | 7.0 t/s |
| タイムアウト | なし |
| 判定 | **PASS** |

### Test 3: 長い生成 (500 tokens)

```bash
-p 'Explain quantum computing in great detail, including its history, principles, and applications' -n 500 --seed 42
```

| 指標 | 結果 |
|------|------|
| Prompt | 10.6 t/s |
| Generation | 6.6 t/s |
| タイムアウト | なし |
| 判定 | **PASS** |

### Test 4: 日本語プロンプト (200 tokens)

```bash
-p '人工知能の歴史について詳しく説明してください' -n 200 --seed 42
```

| 指標 | 結果 |
|------|------|
| Prompt | 9.0 t/s |
| Generation | 7.0 t/s |
| タイムアウト | なし |
| 判定 | **PASS** |

### Test 5: 小モデル回帰テスト (qwen2.5-0.5b, 1C+1R)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

| 指標 | 結果 |
|------|------|
| pp128 | 3028.71 t/s |
| tg32 | 155.52 t/s |
| 判定 | **PASS** (回帰なし) |

### 結果サマリ

| テスト | トークン数 | pp (t/s) | tg (t/s) | タイムアウト | 結果 |
|--------|-----------|----------|----------|-------------|------|
| Test 1 (回帰) | 50 | 6.3 | 6.9 | なし | PASS |
| Test 2 (中) | 200 | 8.9 | 7.0 | なし | PASS |
| Test 3 (長) | 500 | 10.6 | 6.6 | なし | PASS |
| Test 4 (日本語) | 200 | 9.0 | 7.0 | なし | PASS |
| Test 5 (小モデル) | - | 3028.71 | 155.52 | なし | PASS |

## 結論

- 全テストがタイムアウトなしで完了し、回帰なし
- graph_compute の応答待ちに専用タイムアウト（5分）を導入することで、長い推論でもタイムアウトせずに完了可能
- 通常の RDMA 操作（Send/Recv/Write/Read）は従来通り 30 秒タイムアウト（環境変数で変更可能）
- 10 秒ごとの進捗ログにより、長時間待機中も状態が把握可能

## 再現方法

### 前提条件

- 1号機 (192.168.100.1): 7x P100-PCIE-16GB、ソースコードとモデルが配置済み
- 2号機 (192.168.100.2): 4x P100-PCIE-16GB、1号機からデプロイ済み
- モデル: `/tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf` (1号機に配置)

### 1. ビルド (1号機)

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
rm -rf build
cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="60"
cmake --build build -- -j $(nproc)
```

### 2. デプロイ (2号機)

```bash
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/ 192.168.100.2:/home/ubuntu/projects/llama.cpp/
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=60 && cmake --build build -- -j \$(nproc)"
```

### 3. サーバー起動 (2号機)

```bash
ssh 192.168.100.2 "pkill -f rdma-server" 2>/dev/null; sleep 1
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 \
  > /tmp/rdma-server.log 2>&1 &"
```

### 4. テスト実行 (1号機)

以下のコマンドで共通のフラグ部分:

```bash
COMMON_FLAGS="-m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 --seed 42 --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log"
```

#### Test 1: 短い生成 (50 tokens) — 回帰テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

期待: pp ~6.4 t/s, tg ~6.8 t/s, タイムアウトなし

#### Test 2: 中程度の生成 (200 tokens)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 200 --seed 42 \
  -p 'Write a detailed essay about the history of artificial intelligence' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

期待: タイムアウトなし、全トークン生成完了

#### Test 3: 長い生成 (500 tokens)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 500 --seed 42 \
  -p 'Explain quantum computing in great detail, including its history, principles, and applications' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

期待: タイムアウトなし、全トークン生成完了

#### Test 4: 日本語プロンプト (200 tokens)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 200 --seed 42 \
  -p '人工知能の歴史について詳しく説明してください' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

期待: タイムアウトなし、全トークン生成完了

#### Test 5: 小モデル回帰テスト (qwen2.5-0.5b)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

期待: pp128 ~3000 t/s, tg32 ~155 t/s

### 5. 進捗ログの確認方法

別ターミナルで `tail -f /tmp/llama-cli.log` を実行すると、10 秒超の待機時に以下のようなログが出力される:

```
[rdma_connection] Still waiting for completion: 10000 ms elapsed (timeout=300000 ms)
[rdma_connection] Still waiting for completion: 20000 ms elapsed (timeout=300000 ms)
```

graph_compute が 10 秒超かかった場合は以下のような警告も出力される:

```
[rdma] graph_compute took 12.3 s (device=0, n_nodes=624)
```

### 6. 環境変数によるタイムアウト調整

デフォルト値を変更して動作確認する場合:

```bash
# compute タイムアウトを 10 分 (600秒) に変更
GGML_RDMA_COMPUTE_TIMEOUT_MS=600000 GGML_RDMA_SERVERS=192.168.100.2:50051 ...

# 通常 RDMA タイムアウトを 60 秒に変更
GGML_RDMA_TIMEOUT_MS=60000 GGML_RDMA_SERVERS=192.168.100.2:50051 ...

# 意図的に短いタイムアウトでタイムアウト動作を確認 (例: compute 5秒)
GGML_RDMA_COMPUTE_TIMEOUT_MS=5000 GGML_RDMA_SERVERS=192.168.100.2:50051 ...
```
