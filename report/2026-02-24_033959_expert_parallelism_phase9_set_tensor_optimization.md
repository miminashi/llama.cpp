# Expert Parallelism Phase 9 — set_tensor ボトルネック診断と最適化

- **実施日時**: 2026年2月24日 03:39
- **ワークツリー**: `.worktree/expert-parallelism`
- **ブランチ**: `feature/expert-parallelism`
- **コミット**: `4324d315c`

## 前提・目的

Phase 8 で graph_cache の type 変更対応を完了したが、EP=11 の性能は pp=0.3, tg=0.8 t/s のまま (非EP: pp=30, tg=8.5)。RDMA プロファイル分析で set_tensor がランタイム RDMA 時間の約 30% を占めるボトルネックと判明。

### 仮説

`ggml_backend_rdma_buffer_set_tensor` で毎回 signaled RDMA Write + 同期 CQE ポーリングを実行しており、per-call オーバーヘッド (~330 us) が EP モードの多数の小サイズ転送 (~119 KB) に対して過大。

### 目的

1. 診断ログでボトルネックの内訳を特定
2. inter-call selective signaling (ring buffer) でCQEポーリングを削減
3. EP 性能改善の可能性を評価

## 実装内容

### Step 1: 診断機能 (GGML_RDMA_EP_DIAG)

- per-tensor 集計: テンソル名ごとの calls/bytes/time、上位 20 件表示
- sub-stage タイミング: memcpy (staging)、ibv_post_send (+poll)、CQE flush の内訳

### Step 2: Ring Buffer Inter-call Selective Signaling

- connection ごとに 16 MB の ring buffer (64 スロット × 256 KB) を mmap + ibv_reg_mr
- 小サイズ (≤ 256 KB) の single-chunk set_tensor は ring スロットを使用し unsignaled RDMA Write を post
- 64 回に 1 回 signaled write で CQE をポーリング、ring をリセット
- graph_compute 前に flush_unsignaled_set_tensor で未完了 writes を完了
- 大サイズ転送 (model loading) は従来の staging buffer + signaled path を維持
- 環境変数 `GGML_RDMA_NO_SET_TENSOR_SELECTIVE=1` で無効化可能

### 設計上の注意点

初期実装ではステージングバッファを共有する方式を試みたが、**unsignaled write が進行中にステージングバッファが再利用/再割り当てされ、MR が無効化される** ("local protection error") 問題が発生。ring buffer 方式に切り替えることで、各 unsignaled write が一意のメモリスロットを使用するようにして解決。

## 再現方法

### ビルド・デプロイ

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/scripts/rdma-build.sh local
gpu-lock.sh run bash /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/scripts/rdma-deploy.sh
bash /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/scripts/rdma-server.sh restart
```

### 診断テスト (ベースライン: 従来signaling)

```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 GGML_RDMA_EP_DIAG=1 \
  GGML_RDMA_NO_SET_TENSOR_SELECTIVE=1 \
  /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

### 最適化テスト (ring buffer有効)

```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 GGML_RDMA_EP_DIAG=1 \
  /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

## 結果

### 診断結果 (ベースライン)

Sub-stage 内訳 (set_tensor RDMA Write path):

| 段階 | 時間 | 割合 |
|------|------|------|
| memcpy (staging) | 2,347 ms | 5.2% |
| ibv_post_send + CQE poll | 42,797 ms | **94.8%** |
| CQE flush | 0 ms | 0% |

**CQE ポーリングが set_tensor RDMA 時間の 94.8% を占めることを確認。**

### EP=11 性能比較

| 指標 | ベースライン (常時signaled) | Ring Buffer 最適化 | 変化 |
|------|:---:|:---:|:---:|
| set_tensor total | 114,316 ms | 108,287 ms | **-5.3%** |
| set_tensor avg/call | 945.6 us | 895.7 us | -5.3% |
| per-tensor avg (runtime) | 378 us | 319 us | **-15.6%** |
| ibv_post_send | 42,797 ms | 36,523 ms | -14.7% |
| CQE flush | 0 ms | 6,007 ms | +6 sec |
| total RDMA time | 127,181 ms | 128,583 ms | +1.1% |
| **pp** | **0.3 t/s** | **0.3 t/s** | unchanged |
| **tg** | **0.9 t/s** | **0.8 t/s** | unchanged |

### 非EP 回帰テスト

| 指標 | 結果 |
|------|------|
| pp | 8.6 t/s |
| tg | 8.8 t/s |
| 回帰 | なし (通常範囲: pp≈9, tg≈8.5) |

## 分析

### ring buffer 最適化の効果が限定的な理由

1. **CQE flush のオーバーヘッド**: 64 回に 1 回の flush が 3.2 ms/回 (119K / 64 = 1,862 回の flush × 3.2 ms = 6 sec)。flush は signaled fence write で先行する 64 writes の完了を待つため、実質的に 64 writes 分のデータ転送待ちが生じる

2. **set_tensor 以外のオーバーヘッド**: set_tensor の呼び出し回数 (~6,000 calls/token) に比例した非 RDMA オーバーヘッド (関数呼び出し、ロック取得、スケジューラディスパッチ等) が支配的。set_tensor total 108 sec のうち、RDMA sub-stage (memcpy + post + flush) は 45 sec のみで、残り 63 sec は非 RDMA オーバーヘッド

3. **根本的なボトルネック**: EP モードは 1 token あたり ~6,000 回の set_tensor を発生させる (62 MoE 層 × 4 RDMA デバイス × ~24 テンソル/層)。per-call 300 us でも 1 token あたり 1.8 sec かかり、tg < 1 t/s は構造的な制約

### 今後の改善方向

EP 性能を桁違いに改善するには、**set_tensor の呼び出し回数自体を削減**する必要がある:

1. **バッチ set_tensor**: 同一 buffer への複数テンソルの書き込みを 1 回の RDMA Write にまとめる
2. **サーバーサイド expert routing**: expert weights の入力データをサーバー側で保持し、client → server の set_tensor を不要にする
3. **RDMA Write coalescing**: graph_compute の直前に同一 buffer への dirty regions を統合し、1 回の大きな RDMA Write で送信

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `5a7cd6bcb (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 34C | 36C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running |
