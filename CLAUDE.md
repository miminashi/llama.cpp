## 必読ドキュメント

- [AGENTS.md](AGENTS.md) — 作業開始前に必ず確認すること
- [REPORT.md](REPORT.md) — レポート作成ルール
- [GPU.md](GPU.md) — GPUサーバ情報

## ルール

- **ビルド**: llama.cppをビルドする際は、必ず事前に `build` ディレクトリを削除してからビルドすること (`rm -rf build && cmake -B build ...`)
- **コード転送**: 2号機 (192.168.100.2) にコードを転送する際は、2号機に既に存在するllama.cppのディレクトリを削除したうえで、1号機のコードをコピーすること
- **実行**: `llama-cli` を実行する際は、必ず `--log-file /tmp/llama-cli.log` オプションを付けること (ユーザが別ターミナルで `tail -f /tmp/llama-cli.log` によりリアルタイムにログを確認できるようにするため)
- **Bashコマンドに `#` コメント行を含めない**: コメント付きコマンドはパーミッション自動承認が効かないため、コメントは Bash ツールの description パラメータに記載すること
- **マルチラインコマンドを避ける**: 改行区切りの複数コマンドはパーミッション自動承認が効かない場合がある。`&&` や `;` で1行にまとめるか、複数の Bash 呼び出しに分割すること
- **SSH コマンドを直接使わない**: `ssh`, `rsync`, `scp` はパーミッション自動承認が効かない (Claude Code の既知の制限)。代わりに `scripts/rdma-*.sh` ラッパースクリプトを `bash scripts/rdma-*.sh` で呼び出すこと
- **レポート作成**: plan mode を使用してまとまった作業を行った場合は、完了時に `report/` ディレクトリにレポートを作成すること。フォーマットは [REPORT.md](REPORT.md) に従う

## プロジェクト目標と現在の状況

最終目標: GPUDirect RDMAを有効化し、2ノード16台のP100でGLM4.7 Q4を動作させること。

### 段階的アプローチ

上記の最終目標をいきなり達成するのは難しいため、以下のようなステップを設定して段階的に進める。

1. **Step 1** ✅: GPUDirectでないRDMA + P2P無効 + gpt-ossなどの中程度のパラメータ数のモデル
2. **Step 2** ✅: マルチノードクラスタの安定化 (gpt-oss-120b を11GPUクラスタで安定動作)
3. **Step 3** ✅: RDMA性能最適化 (性能損失50%以下、パイプライン化、非同期操作)
4. **Step 4** ✅: GPUDirect RDMA有効化 (CPU経由ステージング排除、GPU直接RDMA転送)
5. **Step 5 (最終Step)**: GPUDirect RDMA + 2ノード16台P100 + GLM4.7 Q4

### ステップ間の依存関係

```
Step 1 (完了) → Step 2 (完了) → Step 3 (完了) → Step 4 (完了) → Step 5 (11GPU事前検証完了, 16GPU待ち)
```

- Step 5 の 11GPU 事前検証は完了 (GLM-4.7 IQ2_M で pp=6.4, tg=6.8 t/s)
- 16GPU への拡張は GPU 追加待ち
- GLM-4.7 Q4 (IQ2_M より大きい) は 16GPU が必要な可能性

### モデル分割方式

- **レイヤー分割 (`-sm layer`) のみ** — P100にはNVLinkがなく、row splitでは性能が出ないことがシングルノード実験で確認済み
- RDMAバックエンドは既にレイヤー分割のみで動作している (各バックエンドインスタンス = 1リモートGPU)
- row split の実装は不要

### Step 5: 達成状況 (11GPU 事前検証)

#### 目標
GPUDirect RDMA + 2ノード16台P100で GLM4.7 Q4 を動作させる。
モデルはunslothの量子化モデル (Hugging Face) を使用予定。

#### 達成済み
- **GLM-4.7 IQ2_M が 11GPU (7C+4R) で安定動作** — 正常な推論出力を確認
- **GPUDirect RDMA タイムアウト解消** — GDR バジェットシステム (`GGML_RDMA_GDR_BUDGET_GB`) で RNIC MTT キャッシュオーバーフローを回避
- **Multi-RDMA デバイス出力破損修正** — `cpy_tensor` 無効化で `get_tensor+set_tensor` フォールバック
- **get_tensor stale data 修正** — per-buffer `mr_is_gdr` フラグで GDR MR のみ RDMA Read 許可
- **mmap + RDMA Write 修正** — 4GB per-buffer サイズ制限 + Send/Recv フォールバック

#### 達成数値 (GLM-4.7 IQ2_M, 7 CUDA + 4 RDMA, 11GPU)

| バックエンド | Prompt (t/s) | Generation (t/s) |
|-------------|:------------:|:----------------:|
| **RDMA (GDR budget 12GB)** | **6.4** | **6.8** |
| RDMA (GDR 無効) | 6.4 | 6.0 |
| RPC (TCP) | 5.4 | 7.5 |

- RDMA は Prompt 処理で RPC 比 **+19%** (RDMA Write ゼロコピーの効果)
- RPC は Generation で RDMA 比 **+10%** (デバイスごとの独立ソケットによるコマンド並列化)
- GDR 有効で Generation が GDR 無効比 **+13%** 改善

#### 前提条件
- Step 4 (GPUDirect RDMA) が動作していること ✅
- 16台のP100が利用可能であること (GPU追加後)

#### 成功基準
- GLM4.7 Q4 が16台P100で推論完了できること
- 実用的な推論速度が得られること (11GPU での IQ2_M 実績: pp=6.4, tg=6.8 t/s)

### 残タスク

1. **16GPU への拡張** (GPU 追加後)
   - 16GPU (8+8) への拡張と全 GPU での RDMA 接続確立テスト
   - GLM-4.7 Q4 (IQ2_M より大きい量子化) がVRAMに収まるか検証
   - `-sm layer` によるレイヤー分割で VRAM 分配計画を策定

2. **GLM-4.7 Q4 量子化モデルの準備**
   - unsloth GLM-4.7 Q4 のダウンロードとサイズ確認
   - IQ2_M (~40GB) では11GPUで動作したが、Q4 はより大きいため16GPU が必要な可能性

### 残課題・懸念事項

#### Generation 速度での RPC 比劣位
- RDMA は単一接続で全リモートデバイスを共有するため、graph_compute が逐次実行される
- RPC はデバイスごとに独立ソケットを持ち、コマンド送信を並列化可能
- **改善案**: RDMA 接続のデバイス分離またはパイプライン化 (未実装)

#### サーバーGPU 計算時間のセッション間変動
- RDMA (GDR 無効) の Generation 速度が 5.5-6.8 t/s と大きくばらつく
- GPU のサーマルスロットリングまたは CUDA コンテキスト初期化の影響と推定
- GDR 有効時は比較的安定 (6.6-6.9 t/s)

#### クライアント異常切断後のサーバー復旧
- クライアントがクラッシュした場合、サーバーの QP 状態が壊れることがある
- 次のテスト実行前にサーバーの再起動が必要
- シングルスレッドのサーバー設計に起因 (接続回復パスが未実装)

#### GDR バジェットのデフォルト値
- デフォルト 12GB は ConnectX-4 の MTT キャッシュ推定値 (~10-16GB) に基づく経験的な値
- より大容量の RNIC (ConnectX-6 等) では `GGML_RDMA_GDR_BUDGET_GB` を増やすことで全デバイス GDR が可能
- 現状では `GGML_RDMA_NO_GDR=1` も引き続き使用可能 (per-buffer フラグにフォールバック)

---

## ビルド・デプロイ・実行手順

> **重要**: SSH/rsync コマンドは Claude Code のパーミッション自動承認が効かないため、
> ラッパースクリプト (`scripts/rdma-*.sh`) 経由で実行すること。
> スクリプトは `bash scripts/rdma-*.sh` で呼び出せば `Bash(bash *)` にマッチして自動承認される。

### 1号機 (192.168.100.1) でビルド

```bash
bash scripts/rdma-build.sh local
```

### 2号機 (192.168.100.2) へのデプロイ (コード転送 + ビルド)

```bash
bash scripts/rdma-deploy.sh
```

### 両ノードでビルド (1号機ビルド + 2号機デプロイ)

```bash
bash scripts/rdma-build.sh local && bash scripts/rdma-deploy.sh
```

### rdma-server 管理 (2号機)

```bash
bash scripts/rdma-server.sh start    # 起動
bash scripts/rdma-server.sh stop     # 停止
bash scripts/rdma-server.sh restart  # 再起動
bash scripts/rdma-server.sh status   # 状態確認
bash scripts/rdma-server.sh log      # ログ表示
```

> **ビルドコマンドに関する注意:**
> - `--config Release` は Unix Makefiles ジェネレータでは無視されるため不要（`CMAKE_BUILD_TYPE=Release` は cmake configure 時に自動設定される）
> - `cmake --build build` の代わりに `make -C build -j$(nproc)` でも同等に動作する
> - ビルドログは `/tmp/cmake_configure.log` (configure) と `/tmp/build.log` (build) に出力される

### llama-bench 実行 (1号機)

**qwen2.5-0.5b 2GPU (CUDA0 + RDMA0)**
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

**gpt-oss-20b 2GPU (CUDA0 + RDMA0)**
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

### llama-cli 実行 (1号機, 11GPU クラスタ)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -hf unsloth/gpt-oss-120b-GGUF:Q4_K_M \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -p 'こんにちは' -n 50 \
  --no-warmup --single-turn --simple-io \
  --log-file /tmp/llama-cli.log
```

### 環境変数一覧

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

### よくあるエラーと対処法

| エラー | 原因 | 対処法 |
|-------|------|--------|
| `CUDA illegal memory access` | `supports_buft` のバグ (修正済み) またはクロスデバイスアクセス | コードが最新か確認。2号機のバイナリが古い可能性 |
| `Connection refused` | rdma-server が起動していない | 2号機で `ps aux \| grep rdma` 確認、サーバー再起動 |
| `RDMA write completion timeout` | ステージングバッファの MR 情報不一致 | サーバーを再起動してバッファ再登録 |
| `chunk recv` ログ大量出力 | 大きなテンソル (>16MB) の転送 | 正常動作。MoEモデルで頻発 |
| `llama_params_fit` クラッシュ | スレッド安全性の問題 (修正済み) | `op_mutex_` による保護が有効か確認 |

---

## 検証方法

- 各ステップで `llama-bench` または `llama-cli` によるベンチマーク実行
- レポートは `report/` ディレクトリに `REPORT.md` のフォーマットに従って記録
- 性能数値は Prompt t/s と Generation t/s の両方を計測

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
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

- 期待値: Prompt ≈ 6.4 t/s, Generation ≈ 6.8 t/s (GDR 有効時)

---

## llama-bench 実行時の注意点

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
# llama-bench での正しい指定方法
llama-bench -m /tmp/gpt-oss-120b/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

---

## 完了済みステップの記録

### Step 1: RDMA基本動作 ✅

#### 達成済み
- CPU経由RDMA (GPUDirectなし) + P2P無効での分散推論が動作
- プロトコル最適化で 0.3 → 148.5 t/s (小規模モデル生成速度)
- Graph diff更新、Adaptive response、Persistent staging buffer を実装
- gpt-oss-20b: 2GPU (ローカル+リモート) で動作確認済み (16.0 t/s生成)
- gpt-oss-120b: 7ローカルGPUで動作確認済み (46.8 t/s生成)

#### 当初の未解決課題 (後続Stepで解決済み)
- **gpt-oss-120b の11GPUクラスタテスト失敗** (CUDA illegal memory access) → Step 2 で修正
- RDMA経由の性能損失が大きい (20bモデルで77%低下) → Step 3 で改善
- Prompt処理速度がローカルの24分の1 (13.7 vs 328.5 t/s) → Step 3/4 で改善
- GPUDirect RDMA未有効化 (GGML_RDMA_NO_GDR=1で無効化中) → Step 4 で有効化
- 全操作が同期的 (async未実装) → Step 3 でバッチフラッシュ等を実装

### Step 2: マルチノードクラスタの安定化 ✅

#### 達成内容
- `supports_buft` バグ修正 (`strstr` → `strcmp`) でクロスデバイス問題を解決
- チャンク送受信実装 (141MB MoEエキスパート重み対応)
- クロスデバイスコピー安全ネット (D2H+H2D)
- gpt-oss-20b/120b が11GPUクラスタ (7 CUDA + 4 RDMA) で安定動作
- 120b: Prompt 3-11 t/s, Generation 0.7-1.1 t/s (複数プロンプトで再現確認済み)

### Step 3: RDMA性能最適化 ✅

#### 達成内容
- バッチフラッシュ: 複数テンソルの一括ステージング転送で Prompt 速度改善
- FLUSH+COMPUTE 統合: 2回のラウンドトリップを1回に削減 (FLUSH_ALL_STAGING + GRAPH_RECOMPUTE → 1コマンド)
- クライアント側 per-call プロファイリング追加 (GGML_RDMA_PROFILE=1)
- サーバー側プロファイリング追加 (graph_recompute, fix_xdev, compute 時間)

#### 達成数値
| 構成 | モデル | pp128 | tg32 | ローカル比 |
|------|--------|-------|------|-----------|
| Local 1GPU | qwen2.5-0.5b | 3,390 | 213.36 | 100% |
| RDMA 1+1 | qwen2.5-0.5b | 3,269 | 173.87 | 81.5% |
| Local 1GPU | gpt-oss-20b | 407 | 64.27 | 100% |
| RDMA 1+1 | gpt-oss-20b | 61 | 58.47 | 91.0% |

#### 成功基準の達成
- gpt-oss-20b tg32: 58.47 t/s ✅ (目標: 30+)
- gpt-oss-20b pp128: 61.09 t/s ✅ (目標: 40+)

#### 発見事項
- 複数RDMA デバイスはスケジューラの制約で逐次実行 (4デバイス = 4× graph_compute/token)
- サーバーGPU計算時間がセッション間で2-3倍変動 (原因不明)
- IB Send/Recv の combined send 最適化は recv バッファオーバーヘッドで逆効果

### Step 4: GPUDirect RDMA有効化 ✅

#### 達成内容
- nvidia-peermem 経由のGPUメモリ直接登録が正常動作
- `GGML_RDMA_NO_GDR=1` を外すだけで有効化 (コード変更不要)
- CPU経由のステージング (cudaMemcpy) を排除

#### 達成数値

| モード | pp128 (t/s) | ローカル比 | tg32 (t/s) | ローカル比 |
|--------|:-----------:|:----------:|:----------:|:----------:|
| ローカル 1GPU | 407.68 | 100% | 64.27 | 100% |
| **GPUDirect RDMA 1+1** | **403.69** | **99.0%** | **59.52** | **92.6%** |
| CPU staging 1+1 | 61.06 | 15.0% | 59.30 | 92.3% |

#### 成功基準の達成
- GPUDirect RDMA でテンソル転送が動作 ✅
- CPU経由比で測定可能な性能向上 (目標: 30%削減) → pp128で6.6倍高速 ✅

#### 発見事項
- 大規模モデル (20b) ではpp128が6.6倍高速 (ウェイト転送量が多い)
- 小規模モデル (0.5b) ではグラフ送信オーバーヘッドが支配的で効果薄
- get_tensor はGPU VRAM からのRDMA ReadでCPU stagingより遅い (PCIe経由)

#### 当初の計画
- 前提条件: Step 2 (クラスタ安定化) が完了していること / nvidia-peermem モジュールが両ノードでロード済み
- 成功基準: GPUDirect RDMA でテンソル転送が動作すること / CPU経由と比較して測定可能な性能向上 (目標: レイテンシ30%以上削減)
- 対象ファイル: `ggml/src/ggml-rdma/rdma-gdr.cpp` (GPUDirectメモリ管理) / `ggml/src/ggml-rdma/ggml-rdma.cpp` (GPUDirect パスの統合) / `ggml/src/ggml-rdma/rdma-memory.cpp` (メモリ登録)
