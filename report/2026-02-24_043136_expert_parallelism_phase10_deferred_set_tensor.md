# Phase 10: Deferred set_tensor 検証レポート

- **実施日時**: 2026年2月24日 04:31
- **ワークツリー**: `.worktree/expert-parallelism`
- **コミット**: `7db41ea43` (deferred set_tensor 実装)

## 前提・目的

Phase 10 の deferred set_tensor 最適化 (commit `7db41ea43`) の性能検証。

- **背景**: Expert Parallelism (EP=11) では各トークン生成時に多数の小さな set_tensor 呼び出しが発生し、個別の RDMA Write + CQE ポーリングがボトルネックとなる可能性がある。deferred set_tensor はこれらを graph_compute 時にバッチフラッシュすることで CQE オーバーヘッドを削減する。
- **目的**: (1) EP=11 での deferred ON/OFF 性能比較、(2) 非EP での回帰テスト
- **前提条件**: 11GPU クラスタ (7C+4R)、GLM-4.7 IQ2_M モデル

## テスト結果

### EP=11 性能比較

| 条件 | pp (t/s) | tg (t/s) |
|------|:--------:|:--------:|
| Deferred ON (デフォルト) | 0.3 | 0.8 |
| Deferred OFF (`GGML_RDMA_NO_DEFERRED_SET_TENSOR=1`) | 0.3 | 0.8 |

**結論**: deferred ON/OFF で測定可能な速度差なし。

### RDMA プロファイル比較 (最終サマリ)

| メトリクス | Deferred ON | Deferred OFF |
|-----------|:------------|:-------------|
| set_tensor calls | 120,896 | 121,132 |
| set_tensor total (ms) | 113,910 | 111,646 |
| set_tensor MB/s | 483.8 | 492.9 |
| get_tensor total (ms) | 6,718 | 6,474 |
| graph_compute calls | 8,100 | 8,080 |
| graph_compute total (ms) | 16,250 | 13,712 |
| total RDMA (ms) | 136,878 | 131,832 |

### set_tensor Sub-stage Breakdown

| Sub-stage | Deferred ON | Deferred OFF |
|-----------|:------------|:-------------|
| memcpy (staging) | 1,889 ms (4.8%) | 2,329 ms (5.1%) |
| ibv_post_send | 37,098 ms (95.2%) | 37,151 ms (81.1%) |
| CQE poll/wait | **0.0 ms (0.0%)** | **6,343 ms (13.8%)** |

### Deferred Flush 統計 (Deferred ON のみ)

| メトリクス | 値 |
|-----------|:---|
| フラッシュ回数 | 5,869 |
| エントリ総数 | 114,413 |
| フラッシュ時間 | 9,304 ms |
| 平均エントリ/フラッシュ | 19.5 |
| 平均フラッシュ時間 | 1.6 ms |

### 非EP 回帰テスト

| メトリクス | 測定値 | ベースライン | 判定 |
|-----------|:------:|:----------:|:----:|
| tg (t/s) | 8.9 | 8.5 | PASS (+4.7%) |
| pp (t/s) | 8.6 | N/A (*) | — |

(*) pp ベースライン (≈30 t/s) は llama-bench pp128 で測定。本テストは llama-cli で 7 トークンプロンプトのため直接比較不可。

## 分析

### Deferred set_tensor が速度改善に寄与しない理由

1. **CQE ポーリング削減は成功**: deferred ON では CQE poll/wait が 6,343 ms → 0 ms に完全削減されている
2. **しかし総 RDMA 時間は増加**: deferred ON (136,878 ms) > deferred OFF (131,832 ms) で +3.8%
3. **graph_compute オーバーヘッドの増加**: deferred ON (16,250 ms) > deferred OFF (13,712 ms) で +18.5%。flush 処理 (9,304 ms) が graph_compute 内で実行されるため
4. **set_tensor は tg のボトルネックではない**: generation phase の set_tensor は全体の ~276 ms (=113,910 - 113,634) に過ぎず、graph_compute (~11 秒) が支配的

### Prompt eval phase プロファイル比較

| メトリクス | Deferred ON | Deferred OFF |
|-----------|:------------|:-------------|
| set_tensor total (ms) | 113,634 | 111,323 |
| get_tensor total (ms) | 3,648 | 2,933 |
| graph_compute total (ms) | 5,172 | 4,596 |
| total RDMA (ms) | 122,453 | 118,852 |

Prompt eval phase でも deferred ON が遅い。set_tensor の大部分はモデルウェイト転送 (output.weight: 636MB で 2,758 ms) であり、deferred のバッチング対象ではない大きなテンソルが支配的。

### EP=11 の根本的ボトルネック

EP=11 で 20 トークン生成に約 25 秒 (0.8 t/s) かかる構造:
- **graph_compute**: ~11 秒 (1 トークンあたり 366 回の graph_compute 呼び出し)
- **get_tensor**: ~3 秒
- **ローカル CUDA 計算**: ~11 秒

1 トークンあたり 366 回の graph_compute (各 ~1.5 ms) が必要で、RDMA ラウンドトリップが蓄積。set_tensor の最適化では解決できない問題。

## 成功基準の判定

| テスト | 指標 | 合格条件 | 結果 | 判定 |
|--------|------|----------|------|:----:|
| EP=11 deferred ON | tg | ≥ 2 t/s | 0.8 t/s | **FAIL** |
| EP=11 deferred OFF | tg | ≈ 0.8 t/s | 0.8 t/s | PASS |
| 非EP 回帰 | tg | 8.5 ±5% | 8.9 t/s | PASS |
| プロファイル | deferred_flush | entries > 0 | 114,413 | PASS |

## 再現方法

### 1. デプロイ + サーバー再起動

```bash
gpu-lock.sh run bash /path/to/worktree/scripts/rdma-deploy.sh
bash /path/to/worktree/scripts/rdma-server.sh start
```

### 2. EP=11 deferred ON

```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 GGML_RDMA_EP_DIAG=1 \
  /path/to/worktree/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

### 3. EP=11 deferred OFF

上記コマンドに `GGML_RDMA_NO_DEFERRED_SET_TENSOR=1` を追加。

### 4. 非EP 回帰テスト

`--expert-parallel 11` と `GGML_RDMA_PROFILE=1 GGML_RDMA_EP_DIAG=1` を除外して実行。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `5a7cd6bcb (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 35°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 644882) |

## 次ステップへの提言

1. **graph_compute ラウンドトリップ削減が最重要**: EP=11 の tg ボトルネックは 1 トークンあたり 366 回の graph_compute 呼び出し。サーバーサイドでのエキスパートグラフ実行パイプライン化や、複数エキスパートの graph_compute バッチングが必要。
2. **Deferred set_tensor は非EP では無害**: tg 8.9 t/s で回帰なし。デフォルト有効のまま維持可能。
3. **GGML_RDMA_PROFILE=1 のオーバーヘッド**: 毎 graph_compute 後のプロファイル出力が性能に影響している可能性あり。プロファイルなしでの測定も検討。
