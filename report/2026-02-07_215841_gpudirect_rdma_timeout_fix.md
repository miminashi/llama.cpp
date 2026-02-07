# GPUDirect RDMA タイムアウト修正レポート

- **実施日時**: 2026年2月7日 21:58

## 前提・目的

GPUDirect RDMA が有効な状態で、大規模モデル (GLM-4.7 IQ2_M: 4 RDMA デバイス × ~10GB = ~40GB の GPU MR) を使用すると RDMA Write が 30 秒でタイムアウトする問題を修正する。gpt-oss-20b (1 RDMA × ~11GB) では正常に動作していた。

- **背景**: Step 4 (GPUDirect RDMA) 完了後、GLM-4.7 での multi-RDMA デバイス構成で GDR 有効時にタイムアウトが発生
- **目的**: サーバー側で GPU MR 登録量にバジェット制限を設け、超過分を CPU ステージングにフォールバックさせることでタイムアウトを解消する
- **前提条件**:
  - 1号機 (192.168.100.1) と 2号機 (192.168.100.2) が InfiniBand (100GbE) で接続
  - 両ノードに nvidia-peermem モジュールがロード済み
  - ConnectX-4 RNIC の MTT キャッシュ上限が ~10-16GB と推定

### 参照レポート

- [Multi-RDMA デバイス出力破損修正](2026-02-07_195500_multi_rdma_device_output_corruption_fix.md)
- [cpy_tensor 修正テスト](2026-02-07_202700_cpy_tensor_fix_comprehensive_test.md)

## 根本原因

ConnectX-4 RNIC の MTT (Memory Translation Table) キャッシュオーバーフロー。サーバーの `alloc_buffer` が GDR 有効時に**全バッファの GPU VRAM を MR 登録**し、合計 ~40GB が RNIC の処理限界 (~10-16GB) を超過していた。

## 修正内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/include/ggml-rdma.h` | `RDMA_PROTO_PATCH_VERSION` 0→1 |
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | GDR バジェットシステム全体 (下記詳細) |
| `CLAUDE.md` | 環境変数テーブルに `GGML_RDMA_GDR_BUDGET_GB` 追加 |

### 変更詳細

#### 1. プロトコル拡張

`rdma_msg_alloc_buffer_rsp` に `mr_flags` フィールド (uint8_t) を追加。`RDMA_MR_FLAG_GDR` (0x01) フラグで、サーバーが GPUDirect MR を登録したか host staging MR を登録したかをクライアントに通知。

#### 2. クライアント: per-buffer GDR 追跡

`ggml_backend_rdma_buffer_context` に `mr_is_gdr` フラグを追加。`alloc_buffer` レスポンスの `mr_flags` から設定。

#### 3. サーバー: GDR バジェット管理

`rdma_server` クラスに `gdr_mr_total_bytes_` (現在の合計) と `gdr_mr_budget_bytes_` (上限) を追加。

- **環境変数**: `GGML_RDMA_GDR_BUDGET_GB` (デフォルト: 12)
- **ロジック**: `alloc_buffer` 時に `total + size <= budget` をチェック
  - 予算内: GPU VRAM を MR 登録 → `mr_flags = RDMA_MR_FLAG_GDR`
  - 予算超過: host staging にフォールバック → `mr_flags = 0`
  - MR 登録失敗: host staging にフォールバック
- **free_buffer**: GDR MR 解放時にバジェット減算

#### 4. クライアント: set_tensor の修正

- GDR MR は 4GB per-buffer サイズ制限をスキップ (RNIC バジェットで保護済み)
- GDR MR への RDMA Write 後は staging flush をスキップ (GPU VRAM に直接書き込み)

#### 5. クライアント: get_tensor の修正

`GGML_RDMA_NO_GDR` 環境変数の代わりに per-buffer `mr_is_gdr` フラグで RDMA Read の可否を判定。staging MR からの RDMA Read は stale データを返すため、GDR MR のみ許可。

### デフォルト値の根拠

**12GB**: gpt-oss-20b (~11GB) が単一デバイスで GDR を使えるよう十分大きく、かつ GLM-4.7 (4×10GB=40GB) のような多デバイス構成で MTT オーバーフローを防ぐ。

## 再現方法

### 1. ビルド・デプロイ

```bash
# 1号機
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
rm -rf build
cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=60
cmake --build build -- -j $(nproc)

# 2号機デプロイ
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' ./ 192.168.100.2:/home/ubuntu/projects/llama.cpp/
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && rm -rf build && \
  cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=60 && \
  cmake --build build -- -j \$(nproc)"
```

### 2. テスト A: gpt-oss-20b GDR 有効 (性能維持確認)

```bash
# サーバー (GDR 有効、デフォルトバジェット 12GB)
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 \
  > /tmp/rdma-server.log 2>&1 &"

# ベンチマーク
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

### 3. テスト B: GLM-4.7 IQ2_M GDR 有効 (タイムアウト解消確認)

```bash
# サーバー再起動 (テスト A の後は必ず再起動)
ssh 192.168.100.2 "killall rdma-server; sleep 2; \
  LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 \
  > /tmp/rdma-server.log 2>&1 &"

# 推論
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### 4. テスト C: GDR budget=0 (全 staging フォールバック)

```bash
ssh 192.168.100.2 "killall rdma-server; sleep 2; \
  GGML_RDMA_GDR_BUDGET_GB=0 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 \
  > /tmp/rdma-server.log 2>&1 &"
```

同じ llama-cli コマンドを実行。`GGML_RDMA_NO_GDR=1` と同等の動作になることを確認。

## テスト結果

### テスト A: gpt-oss-20b 1+1 GPU

| モード | pp128 (t/s) | tg32 (t/s) |
|--------|:-----------:|:----------:|
| **GDR 有効 (budget 12GB)** | **364.30** | **52.70** |
| GDR 無効 (リグレッション確認) | 377.63 | 42.11 |

サーバーログ (GDR 有効):
```
[server] GDR MR registered: size=1.6 GB, total=1.6/12.0 GB
[server] GDR MR registered: size=2.1 GB, total=3.7/12.0 GB
[server] GDR MR registered: size=2.1 GB, total=5.8/12.0 GB
[server] GDR MR registered: size=2.2 GB, total=8.0/12.0 GB
...
[server] GDR MR registered: size=0.1 GB, total=8.1/12.0 GB
```

合計 ~8GB、12GB バジェット以内で全バッファが GDR MR 登録成功。

### テスト B: GLM-4.7 IQ2_M 7C+4R (11 GPU)

| モード | pp (t/s) | tg (t/s) | 結果 |
|--------|:--------:|:--------:|------|
| **GDR 有効 (budget 12GB)** | **6.5** | **7.1** | **成功 (タイムアウトなし)** |
| GDR budget=0 (全 staging) | - | - | 成功 |
| GGML_RDMA_NO_GDR=1 | - | - | 成功 |
| 修正前 GDR 有効 | - | - | **30s タイムアウト (修正前)** |

サーバーログ (GDR 有効、budget 12GB):
```
[server] GDR MR registered: size=9.7 GB, total=9.7/12.0 GB        ← 1台目: GDR
[server] GDR budget exceeded (need 9.7 GB, used 9.7/12.0 GB), using staging  ← 2台目: staging
[server] GDR budget exceeded (need 11.2 GB, used 9.7/12.0 GB), using staging ← 3台目: staging
[server] GDR budget exceeded (need 8.6 GB, used 9.7/12.0 GB), using staging  ← 4台目: staging
[server] GDR MR registered: size=0.5 GB, total=10.4/12.0 GB       ← 小バッファ: GDR
...
[server] GDR MR registered: size=0.5 GB, total=11.8/12.0 GB
```

バジェットにより 1 台目の大バッファ (~9.7GB) のみ GDR、2-4 台目は staging にフォールバック。合計 ~11.8GB で 12GB 以内に収まり、MTT オーバーフロー回避。

## 動作シナリオ

| 構成 | バッファ | 予算 12GB | 結果 |
|------|---------|:---------:|------|
| gpt-oss-20b 1+1 | 1×~8GB | 8 < 12 | 全 GDR → pp128 ~364 t/s |
| GLM-4.7 7C+4R | 4×~10GB | 1台目 GDR (9.7<12), 2台目以降 staging | タイムアウトなし |
| gpt-oss-120b 7+4 | 4×~17GB | 1台目以降全て staging (>12) | タイムアウトなし |
| `GDR_BUDGET_GB=50` | 任意 | ユーザ設定 | 大容量 RNIC で全 GDR |
| `GDR_BUDGET_GB=0` | 任意 | 全て staging | GGML_RDMA_NO_GDR=1 と同等 |

## RPC バックエンドとの比較

### テスト条件

- **モデル**: GLM-4.7 IQ2_M (3分割 GGUF, 約40GB)
- **構成**: 7 ローカル CUDA + 4 リモート GPU (2ノード11GPU)
- **プロンプト**: `The capital of France is` (`-n 50 --seed 42`)
- **各モードで3回実行** (サーバーは毎回再起動)

### RPC サーバー起動方法

```bash
# 2号機で GPU ごとに 1 プロセス起動
CUDA_VISIBLE_DEVICES=0 nohup rpc-server -H 0.0.0.0 -p 50052 &
CUDA_VISIBLE_DEVICES=1 nohup rpc-server -H 0.0.0.0 -p 50053 &
CUDA_VISIBLE_DEVICES=2 nohup rpc-server -H 0.0.0.0 -p 50054 &
CUDA_VISIBLE_DEVICES=3 nohup rpc-server -H 0.0.0.0 -p 50055 &

# 1号機から実行
llama-cli -m ... \
  --rpc 192.168.100.2:50052,192.168.100.2:50053,192.168.100.2:50054,192.168.100.2:50055 \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### 全計測データ

| バックエンド | Run | Prompt (t/s) | Generation (t/s) |
|-------------|:---:|:------------:|:----------------:|
| RDMA (GDR budget 12GB) | 1 | 6.4 | 6.9 |
| RDMA (GDR budget 12GB) | 2 | 6.4 | 6.6 |
| RDMA (GDR budget 12GB) | 3 | 6.4 | 6.9 |
| RDMA (GDR 無効) | 1 | 6.4 | 5.6 |
| RDMA (GDR 無効) | 2 | 6.4 | 6.8 |
| RDMA (GDR 無効) | 3 | 6.3 | 5.5 |
| RPC (TCP) | 1 | 5.5 | 7.5 |
| RPC (TCP) | 2 | 5.3 | 7.5 |
| RPC (TCP) | 3 | 5.3 | 7.6 |

### 平均値比較

| バックエンド | Prompt (t/s) | Generation (t/s) |
|-------------|:------------:|:----------------:|
| **RDMA (GDR budget 12GB)** | **6.4** | **6.8** |
| RDMA (GDR 無効) | 6.4 | 6.0 |
| RPC (TCP) | 5.4 | 7.5 |

### 分析

#### Prompt 処理速度

- RDMA は RPC に対して **+19%** (6.4 vs 5.4 t/s)
- RDMA は GDR 有効/無効で差がない (6.4 vs 6.4 t/s)
- Prompt 処理は大量のウェイト転送 (set_tensor) が支配的であり、RDMA Write によるゼロコピー転送が TCP ソケットベースの RPC より効率的

#### Generation 速度

- RPC が RDMA (GDR) に対して **+10%** (7.5 vs 6.8 t/s)
- RDMA (GDR) は RDMA (no-GDR) に対して **+13%** (6.8 vs 6.0 t/s)
- Generation は get_tensor (出力取得) と graph_compute のラウンドトリップが支配的
- RPC は各デバイスが独立した TCP ソケットを持つため、4デバイスへのコマンド送信を並列化できる
- RDMA は単一接続を共有するため、4デバイスへのコマンドが逐次実行される (スケジューラ制約)

#### RPC との構造的差異

| 特性 | RDMA | RPC |
|------|------|-----|
| 接続 | 1接続で全デバイス共有 | デバイスごとに独立ソケット |
| set_tensor | RDMA Write (ゼロコピー) | TCP send |
| get_tensor | RDMA Read (GDR) / Send-Recv | TCP recv |
| graph_compute | 逐次 (接続共有) | 並列可能 (独立ソケット) |
| cpy_tensor | 無効 (get+set fallback) | 無効 (独立ソケットのため) |

## 考察

- GDR バジェットシステムにより、RNIC の MTT キャッシュ容量に応じた graceful degradation が実現できた
- 1 台目のバッファが GDR を使えることで、pp128 での GPUDirect 高速パスが一部維持される
- `GGML_RDMA_GDR_BUDGET_GB` でユーザがハードウェアに応じた設定が可能
- 今後より大容量の MTT を持つ RNIC (ConnectX-6 等) では、バジェットを増やすことで全デバイスの GDR が可能になる
- RPC との比較では、RDMA は Prompt 処理で +19% の優位性がある一方、Generation では RPC が +10% 高速
- RDMA の Generation 速度改善には、マルチデバイスへのコマンド並列送信 (接続分離またはパイプライン化) が今後の課題
