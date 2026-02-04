# ローカルGPU vs RDMA クラスタ ベンチマークレポート — gpt-oss-20b / gpt-oss-120b

- **実施日時**: 2026年2月4日 11:02

## 前提・目的

gpt-oss-20b と gpt-oss-120b の Q4_K_M 量子化モデルを使用し、**ローカルGPUのみ** vs **ローカル+リモートGPU (RDMA)** の推論速度を複数の日本語プロンプトで比較する。

- **背景**: 2ノード合計11GPUをRDMAクラスタとして活用することで、大規模モデルの推論が可能になるかを検証
- **目的**: ローカルGPUとRDMA経由リモートGPUの推論性能差を定量的に測定
- **前提条件**:
  - 1号機 (ローカル): 7 x Tesla P100-PCIE-16GB (CUDA0-6), 合計 112 GB VRAM
  - 2号機 (リモート): 4 x Tesla P100-PCIE-16GB (RDMA0-3), 合計 64 GB VRAM
  - ネットワーク: 100GbE RDMA (ibv_rc_pingpong で ~12 Gbps, 5.7 us/iter を確認済み)
  - ソフトウェア: llama.cpp feature/rdma-backend ブランチ (build b7902-23bbf6f13)

## テストマトリクス

### gpt-oss-20b (~11.6 GB, Q4_K_M, MoE 20.9B/3.6Bアクティブ, 32エキスパート Top-4)

| テスト | 構成 | GPU数 | `-dev` | `-c` |
|--------|------|-------|--------|------|
| 20b-Local | ローカル1GPU | 1 | `CUDA0` | 2048 |
| 20b-RDMA | ローカル1 + リモート1 (レイヤー分割) | 2 | `CUDA0,RDMA0[...]` | 2048 |

### gpt-oss-120b (~62.8 GB, Q4_K_M, MoE 116.8B/5.1Bアクティブ, 64エキスパート Top-8)

| テスト | 構成 | GPU数 | `-dev` | `-c` |
|--------|------|-------|--------|------|
| 120b-Local-7 | ローカル全GPU (レイヤー分割) | 7 | `CUDA0,...,CUDA6` | 2048 |
| 120b-Cluster-11 | ローカル7 + リモート4 (レイヤー分割) | 11 | `CUDA0,...,CUDA6,RDMA0,...,RDMA3` | 2048 |

### 日本語プロンプト (4種類)

| ID | カテゴリ | プロンプト | `-n` |
|----|---------|-----------|------|
| P1 | 挨拶（短文） | `こんにちは` | 100 (20b) / 50 (120b) |
| P2 | 一般知識 | `日本の四季について簡単に教えてください` | 同上 |
| P3 | 要約・創作 | `桃太郎の物語を100文字以内で要約してください` | 同上 |
| P4 | コード生成 | `Pythonでクイックソートを実装するコードを書いてください` | 同上 |

## 結果

### gpt-oss-20b ベンチマーク結果

| プロンプト | 構成 | Prompt (t/s) | Generation (t/s) |
|-----------|------|-------------|------------------|
| P1 こんにちは | 20b-Local (CUDA0) | **162.2** | **68.2** |
| P1 こんにちは | 20b-RDMA (CUDA0+RDMA0) | 26.6 | 12.7 |
| P2 日本の四季 | 20b-Local (CUDA0) | **159.1** | **68.1** |
| P2 日本の四季 | 20b-RDMA (CUDA0+RDMA0) | 28.8 | 14.0 |
| P3 桃太郎要約 | 20b-Local (CUDA0) | **159.0** | **68.1** |
| P3 桃太郎要約 | 20b-RDMA (CUDA0+RDMA0) | 9.3 | 19.9 |
| P4 クイックソート | 20b-Local (CUDA0) | **157.3** | **68.2** |
| P4 クイックソート | 20b-RDMA (CUDA0+RDMA0) | 18.6 | 17.2 |

**20b 平均性能比較:**

| 指標 | Local (1 GPU) | RDMA (2 GPU) | 比率 (RDMA/Local) |
|------|-------------|-------------|-------------------|
| Prompt 平均 | 159.4 t/s | 20.8 t/s | 0.13x (87%低下) |
| Generation 平均 | 68.2 t/s | 16.0 t/s | 0.23x (77%低下) |

### gpt-oss-120b ベンチマーク結果

| プロンプト | 構成 | Prompt (t/s) | Generation (t/s) |
|-----------|------|-------------|------------------|
| P1 こんにちは | 120b-Local-7 (CUDA0-6) | 74.7 | 46.4 |
| P2 日本の四季 | 120b-Local-7 (CUDA0-6) | 77.9 | 47.3 |
| P3 桃太郎要約 | 120b-Local-7 (CUDA0-6) | 76.6 | 46.5 |
| P4 クイックソート | 120b-Local-7 (CUDA0-6) | 79.0 | 47.1 |
| P1〜P4 | 120b-Cluster-11 | **失敗** | **失敗** |

**120b-Local-7 平均:**

| 指標 | Local (7 GPU) |
|------|-------------|
| Prompt 平均 | 77.1 t/s |
| Generation 平均 | 46.8 t/s |

### 120b クラスタテスト失敗の詳細

120b-Cluster-11 (7ローカル + 4リモート) テストは全てのプロンプトで失敗した。

**エラー内容**: リモートrdma-server上のCUDA計算で `illegal memory access` が発生し、クライアント側で `Receive completion timeout` → `Remote RDMA server crashed or returned malformed response` となり abort した。

```
ggml-cuda.cu:97: CUDA error
  current device: 0, in function ggml_backend_cuda_synchronize
  cudaStreamSynchronize(cuda_ctx->stream())
  an illegal memory access was encountered
```

**原因の推測**:
1. gpt-oss-120b は MoE (Mixture of Experts) アーキテクチャ (116.8Bパラメータ, 5.1Bアクティブ, 64エキスパート Top-8) であり、計算グラフが複雑
2. gpt-oss-20b も MoE (20.9Bパラメータ, 3.6Bアクティブ, 32エキスパート Top-4) だが、モデルサイズが小さいため問題が顕在化しなかった可能性
3. RDMAバックエンドのリモートCUDA実行が大規模MoEモデルの計算パターンに対応できていない可能性
3. クライアント (feature/rdma-backend ブランチ) とサーバー (master ブランチ) のバージョン不一致も一因の可能性

**追加試行**: RDMA-only 4GPU (RDMA0-3) での120b実行も試みたが、RDMA0のバッファサイズ (~17 GB) が単一GPU VRAM (16 GB) を超えるため、モデルロード段階で失敗。

### VRAM使用量

**20b-Local (CUDA0):**
| デバイス | 合計 | 空き | モデル | コンテキスト | 計算 |
|---------|------|------|--------|------------|------|
| CUDA0 | 16,276 MiB | 4,800 MiB | 10,694 MiB | 66 MiB | 398 MiB |
| Host | - | - | 379 MiB | 0 MiB | 16 MiB |

**20b-RDMA (CUDA0 + RDMA0):**
| デバイス | 合計 | 空き | モデル | コンテキスト | 計算 |
|---------|------|------|--------|------------|------|
| CUDA0 | 16,276 MiB | 10,182 MiB | 5,474 MiB | 34 MiB | 288 MiB |
| RDMA0 | 16,276 MiB | 10,328 MiB | 5,219 MiB | 31 MiB | 404 MiB |

**120b-Local-7 (CUDA0-6):**
| デバイス | 合計 | 空き | モデル | コンテキスト | 計算 |
|---------|------|------|--------|------------|------|
| CUDA0 | 16,276 MiB | 5,992 MiB | 9,814 MiB | 16 MiB | 134 MiB |
| CUDA1 | 16,276 MiB | 7,648 MiB | 8,177 MiB | 12 MiB | 120 MiB |
| CUDA2 | 16,276 MiB | 7,646 MiB | 8,177 MiB | 15 MiB | 120 MiB |
| CUDA3 | 16,276 MiB | 6,008 MiB | 9,813 MiB | 16 MiB | 120 MiB |
| CUDA4 | 16,276 MiB | 7,648 MiB | 8,177 MiB | 12 MiB | 120 MiB |
| CUDA5 | 16,276 MiB | 7,644 MiB | 8,178 MiB | 15 MiB | 120 MiB |
| CUDA6 | 16,276 MiB | 8,386 MiB | 7,130 MiB | 11 MiB | 431 MiB |

## 考察

### 20b: ローカルGPUが圧倒的に速い

- 20bモデルは1台の P100 (16GB) に収まるため、ローカルGPUのみで最高性能が得られる
- RDMA経由で2GPUにレイヤー分割しても、ネットワークレイテンシのオーバーヘッドにより **Prompt処理で約87%、Generation で約77% の性能低下**
- この結果は予想通りで、GPU間通信が不要な単一GPU実行がネットワーク越しの分散処理より有利

### 120b: ローカル7GPUで良好な性能

- 120b MoE モデル (~63 GB) は7台の P100 にレイヤー分割で収まり、平均 Prompt 77.1 t/s、Generation 46.8 t/s を達成
- MoE (5.1Bアクティブパラメータ) のためデコードが効率的で、20bの68.2 t/s に比べても遜色のない 46.8 t/s
- 各GPUのVRAM使用量は7-10 GB (16 GB中) と十分な余裕がある

### 120b クラスタ: RDMA バックエンドの MoE 対応に課題

- 120b のクラスタ (CUDA+RDMA混在) テストはリモートCUDA計算時の illegal memory access で全て失敗
- 20b (小規模MoE: 20.9B/3.6Bアクティブ) ではCUDA+RDMA混在が動作したため、大規模MoEモデル固有の問題と推測
- RDMAバックエンドは実験的機能であり、MoEモデルの複雑な計算グラフ（条件分岐的なエキスパート選択）への対応が不完全

### RDMAの意義

現時点では以下の結論:
1. **小さいモデル (GPU 1台に収まる)**: RDMAは不要。ローカルGPUが圧倒的に速い
2. **大きいモデル (ローカルGPUに収まる)**: 120bはローカル7GPUで十分な性能。クラスタ化の必要性は現時点では低い
3. **ローカルGPUに収まらないモデル**: RDMAクラスタの真の価値が発揮される場面だが、MoEモデルでの安定性に課題が残る

## 再現方法

### 環境構築

- llama.cpp feature/rdma-backend ブランチ (commit 23bbf6f13) をビルド
- RDMAバックエンド (`-DGGML_RDMA=ON`) を有効にしてCMakeビルド

### rdma-server 起動 (2号機)

```bash
ssh 192.168.100.2 'GGML_RDMA_NO_GDR=1 \
  LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server \
  --host 0.0.0.0 --port 50051 > /tmp/rdma-server.log 2>&1 &'
```

### 20b ローカルテスト

```bash
LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin \
  /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin/llama-cli \
  -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M \
  -dev CUDA0 -ngl 999 -c 2048 \
  -p 'こんにちは' -n 100 \
  --no-warmup --single-turn --simple-io
```

### 20b RDMAテスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 \
  LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin \
  /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin/llama-cli \
  -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M \
  -dev 'CUDA0,RDMA0[192.168.100.2:50051]' -sm layer -ngl 999 -c 2048 \
  -p 'こんにちは' -n 100 \
  --no-warmup --single-turn --simple-io
```

### 120b ローカルテスト

```bash
LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin \
  /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin/llama-cli \
  -hf unsloth/gpt-oss-120b-GGUF:Q4_K_M \
  -dev CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6 \
  -sm layer -ngl 999 -c 2048 \
  -p 'こんにちは' -n 50 \
  --no-warmup --single-turn --simple-io
```

### 120b クラスタテスト (失敗)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 \
  LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin \
  /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/build/bin/llama-cli \
  -hf unsloth/gpt-oss-120b-GGUF:Q4_K_M \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 \
  -p 'こんにちは' -n 50 \
  --no-warmup --single-turn --simple-io
```

## 備考

- 全テストでコンテキストサイズを `-c 2048` に統一（compute buffer が 16GB GPU VRAM に収まるよう制限）
- RDMAテスト前に毎回 rdma-server を再起動（前セッションのメモリ残留を防止）
- `GGML_RDMA_NO_GDR=1` はクライアント・サーバー双方に設定（GPUDirect RDMA の互換性問題回避）
- 20b RDMA テストは計画の「RDMA単体1GPU」から「ローカル1+RDMA1のレイヤー分割」に変更（RDMA単体ではcompute buffer が16GBを超えるため）
