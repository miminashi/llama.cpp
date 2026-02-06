## 必読ドキュメント

- [AGENTS.md](AGENTS.md) — 作業開始前に必ず確認すること
- [REPORT.md](REPORT.md) — レポート作成ルール
- [GPU.md](GPU.md) — GPUサーバ情報

## ビルドルール

- llama.cppをビルドする際は、必ず事前に `build` ディレクトリを削除してからビルドすること (`rm -rf build && cmake -B build ...`)

## コード転送ルール

- 2号機 (192.168.100.2) にコードを転送する際は、2号機に既に存在するllama.cppのディレクトリを削除したうえで、1号機のコードをコピーすること

## 実行ルール

- `llama-cli` を実行する際は、必ず `--log-file /tmp/llama-cli.log` オプションを付けること (ユーザが別ターミナルで `tail -f /tmp/llama-cli.log` によりリアルタイムにログを確認できるようにするため)

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

## プロジェクト目標

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
Step 1 (完了) → Step 2 (完了) → Step 3 (完了) → Step 4 (完了) → Step 5 (GLM4.7) ★次
```

- Step 5 が次のステップ (GLM4.7 Q4 を 16台 P100 で動作させる)
- 16台のP100に依存するが、11台での事前検証は先行可能

### モデル分割方式

- **レイヤー分割 (`-sm layer`) のみ** — P100にはNVLinkがなく、row splitでは性能が出ないことがシングルノード実験で確認済み
- RDMAバックエンドは既にレイヤー分割のみで動作している (各バックエンドインスタンス = 1リモートGPU)
- row split の実装は不要

---

## Step 1 の達成状況

### 達成済み
- CPU経由RDMA (GPUDirectなし) + P2P無効での分散推論が動作
- プロトコル最適化で 0.3 → 148.5 t/s (小規模モデル生成速度)
- Graph diff更新、Adaptive response、Persistent staging buffer を実装
- gpt-oss-20b: 2GPU (ローカル+リモート) で動作確認済み (16.0 t/s生成)
- gpt-oss-120b: 7ローカルGPUで動作確認済み (46.8 t/s生成)

### 未解決の課題
- **gpt-oss-120b の11GPUクラスタテスト失敗** (CUDA illegal memory access)
- RDMA経由の性能損失が大きい (20bモデルで77%低下)
- Prompt処理速度がローカルの24分の1 (13.7 vs 328.5 t/s)
- GPUDirect RDMA未有効化 (GGML_RDMA_NO_GDR=1で無効化中)
- 全操作が同期的 (async未実装)

---

## Step 2: マルチノードクラスタの安定化 ✅

### 達成内容
- `supports_buft` バグ修正 (`strstr` → `strcmp`) でクロスデバイス問題を解決
- チャンク送受信実装 (141MB MoEエキスパート重み対応)
- クロスデバイスコピー安全ネット (D2H+H2D)
- gpt-oss-20b/120b が11GPUクラスタ (7 CUDA + 4 RDMA) で安定動作
- 120b: Prompt 3-11 t/s, Generation 0.7-1.1 t/s (複数プロンプトで再現確認済み)

---

## Step 3: RDMA性能最適化 ✅

### 達成内容
- バッチフラッシュ: 複数テンソルの一括ステージング転送で Prompt 速度改善
- FLUSH+COMPUTE 統合: 2回のラウンドトリップを1回に削減 (FLUSH_ALL_STAGING + GRAPH_RECOMPUTE → 1コマンド)
- クライアント側 per-call プロファイリング追加 (GGML_RDMA_PROFILE=1)
- サーバー側プロファイリング追加 (graph_recompute, fix_xdev, compute 時間)

### 達成数値
| 構成 | モデル | pp128 | tg32 | ローカル比 |
|------|--------|-------|------|-----------|
| Local 1GPU | qwen2.5-0.5b | 3,390 | 213.36 | 100% |
| RDMA 1+1 | qwen2.5-0.5b | 3,269 | 173.87 | 81.5% |
| Local 1GPU | gpt-oss-20b | 407 | 64.27 | 100% |
| RDMA 1+1 | gpt-oss-20b | 61 | 58.47 | 91.0% |

### 成功基準の達成
- gpt-oss-20b tg32: 58.47 t/s ✅ (目標: 30+)
- gpt-oss-20b pp128: 61.09 t/s ✅ (目標: 40+)

### 発見事項
- 複数RDMA デバイスはスケジューラの制約で逐次実行 (4デバイス = 4× graph_compute/token)
- サーバーGPU計算時間がセッション間で2-3倍変動 (原因不明)
- IB Send/Recv の combined send 最適化は recv バッファオーバーヘッドで逆効果

---

## Step 4: GPUDirect RDMA有効化 ✅

### 達成内容
- nvidia-peermem 経由のGPUメモリ直接登録が正常動作
- `GGML_RDMA_NO_GDR=1` を外すだけで有効化 (コード変更不要)
- CPU経由のステージング (cudaMemcpy) を排除

### 達成数値

| モード | pp128 (t/s) | ローカル比 | tg32 (t/s) | ローカル比 |
|--------|:-----------:|:----------:|:----------:|:----------:|
| ローカル 1GPU | 407.68 | 100% | 64.27 | 100% |
| **GPUDirect RDMA 1+1** | **403.69** | **99.0%** | **59.52** | **92.6%** |
| CPU staging 1+1 | 61.06 | 15.0% | 59.30 | 92.3% |

### 成功基準の達成
- GPUDirect RDMA でテンソル転送が動作 ✅
- CPU経由比で測定可能な性能向上 (目標: 30%削減) → pp128で6.6倍高速 ✅

### 発見事項
- 大規模モデル (20b) ではpp128が6.6倍高速 (ウェイト転送量が多い)
- 小規模モデル (0.5b) ではグラフ送信オーバーヘッドが支配的で効果薄
- get_tensor はGPU VRAM からのRDMA ReadでCPU stagingより遅い (PCIe経由)

### 前提条件
- Step 2 (クラスタ安定化) が完了していること
- nvidia-peermem モジュールが両ノードでロード済み

### 成功基準
- GPUDirect RDMA でテンソル転送が動作すること
- CPU経由と比較して測定可能な性能向上 (目標: レイテンシ30%以上削減)

### 対象ファイル
- `ggml/src/ggml-rdma/rdma-gdr.cpp` — GPUDirectメモリ管理
- `ggml/src/ggml-rdma/ggml-rdma.cpp` — GPUDirect パスの統合
- `ggml/src/ggml-rdma/rdma-memory.cpp` — メモリ登録

---

## Step 5 (最終Step): GLM4.7 Q4 on 16 P100s

### 目標
GPUDirect RDMA + 2ノード16台P100で GLM4.7 Q4 を動作させる。
モデルはunslothの量子化モデル (Hugging Face) を使用予定。

### タスク

1. **モデル準備と事前検証**
   - unsloth GLM4.7 Q4のダウンロードとサイズ確認
   - 11GPUでの動作確認 (GPU追加前に先行テスト可能)
   - `-sm layer` によるレイヤー分割でVRAM分配計画を策定 (P100 16GB × n台)
   - ※ row splitはP100 (NVLinkなし) では性能が出ないため不使用

2. **ハードウェアスケーリング** (GPU追加後)
   - 16GPU (8+8等) への拡張と接続確認
   - 全GPUでのRDMA接続確立テスト

3. **大規模分散推論の最適化**
   - 16GPUでのグラフスケジューリング最適化
   - ノード間通信パターンの最適化
   - メモリ使用量の最適化 (P100 16GBの制約下)

### 前提条件
- Step 4 (GPUDirect RDMA) が動作していること
- 16台のP100が利用可能であること (GPU追加後)

### 成功基準
- GLM4.7 Q4 が16台P100で推論完了できること
- 実用的な推論速度が得られること (具体値はStep 3/4の結果を踏まえて設定)

### 備考
- GPU追加前でも、11台でGLM4.7の事前検証は可能 (モデルがVRAMに収まる場合)
- 収まらない場合はGPU追加を待つ

---

## ビルド・デプロイ・実行手順

### 1号機 (192.168.100.1) でビルド

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 2号機 (192.168.100.2) へのデプロイ

```bash
# 2号機の既存ディレクトリを削除してから転送
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/ 192.168.100.2:/home/ubuntu/projects/llama.cpp/

# 2号機でビルド
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
```

### rdma-server 起動 (2号機)

```bash
ssh 192.168.100.2 "GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
```

### llama-bench 実行 (1号機)

**qwen2.5-0.5b 2GPU (CUDA0 + RDMA0)**
```bash
GGML_RDMA_NO_GDR=1 GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

**gpt-oss-20b 2GPU (CUDA0 + RDMA0)**
```bash
GGML_RDMA_NO_GDR=1 GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

### llama-cli 実行 (1号機, 11GPU クラスタ)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 \
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
| `GGML_RDMA_NO_STAGING` | `1` でホストステージングバッファを無効化 (Send/Recvフォールバック) | 未設定 |
| `GGML_RDMA_PROFILE` | `1` でクライアント側プロファイリング有効化 | 未設定 |
| `GGML_RDMA_DEBUG` | `1` でデバッグログ出力 | 未設定 |

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
