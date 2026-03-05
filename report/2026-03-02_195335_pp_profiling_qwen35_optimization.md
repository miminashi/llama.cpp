# PP 性能プロファイリング: Qwen3.5-35B-A3B 6GPU ボトルネック分析

- **実施日時**: 2026年3月2日 19:53
- **ワークツリー**: `.worktree/double-buffering`
- **ブランチ**: `feature/double-buffering`
- **コミット**: `f1f2cbe90` (プロファイリング強化)

## 前提・目的

前回の double-buffering + skip_last_signal 実装は Qwen3.5 6GPU で pp に改善効果がなかった (pp: -0.06%, tg: -4.24%)。pp ボトルネックの所在を特定するため、プロファイリング駆動の分析を実施する。

- **背景**: Selective signaling は GLM-4.7 11GPU で pp +26.1% だが、Qwen3.5 MoE (3B active, 6GPU) では効果なし
- **目的**: pp のどこに時間がかかっているかを定量的に特定し、最適化の余地があるか判断する
- **前提条件**: Node 1 CUDA5,6 + Node 2 RDMA0-3 の 6GPU 構成
- **参照レポート**: [double_buffer_qwen35_ab_benchmark](2026-03-02_143000_double_buffer_qwen35_ab_benchmark.md)

## 再現方法

### 1. ビルド・デプロイ

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/double-buffering/scripts/rdma-build.sh local
gpu-lock.sh run bash /home/ubuntu/projects/llama.cpp/.worktree/double-buffering/scripts/rdma-deploy.sh
bash /home/ubuntu/projects/llama.cpp/.worktree/double-buffering/scripts/rdma-server.sh start
```

### 2. プロファイリング実行

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=5,6 GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 \
  /home/ubuntu/projects/llama.cpp/.worktree/double-buffering/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -p "Explain RDMA in detail" -n 32 \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log 2>&1
```

## プロファイリング結果

### 全体サマリ (最終累積値, 130 graph_compute calls)

| 操作 | 呼出回数 | データ量 | 合計時間 (ms) | 平均時間 | スループット |
|------|---------|---------|-------------|---------|-------------|
| set_tensor | 1,223 | 12,201 MB | 15,054 | 12.3 ms/call | 850 MB/s |
| get_tensor | 32 | 30.3 MB | 380 | 11.9 ms/call | 84 MB/s |
| graph_compute | 130 | — | 426 | 3.3 ms/call | — |
| **合計 RDMA 時間** | — | — | **15,860** | — | — |

- set_tensor の 99.9% はモデルロード (533 calls, 12,200 MB, ~14,782 ms)
- graph_compute: 12 FULL_GRAPH (pp) + 118 recompute (tg)

### PP Phase 詳細: FULL_GRAPH graph_compute (8 calls)

| Call | flush (ms) | serialize (ms) | send (ms) | total (ms) | n_nodes | n_copies |
|------|-----------|----------------|-----------|------------|---------|----------|
| #1 (D0, iter1) | **0.07** | 1.57 | 0.66 | 3.47 | 1065 | 0 |
| #2 (D1, iter1) | **92.28** | 1.37 | 0.69 | 95.43 | 1065 | 2 |
| #3 (D2, iter1) | **43.14** | 1.25 | 0.64 | 46.45 | 1173 | 2 |
| #4 (D3, iter1) | **43.60** | 0.76 | 0.13 | 45.09 | 703 | 2 |
| #5 (D0, iter2) | **0.06** | 1.12 | 0.12 | 1.88 | 1065 | 0 |
| #6 (D1, iter2) | **55.17** | 0.61 | 0.60 | 56.97 | 1065 | 2 |
| #7 (D2, iter2) | **42.65** | 0.81 | 0.61 | 44.72 | 1173 | 2 |
| #8 (D3, iter2) | **44.67** | 0.28 | 0.07 | 45.37 | 703 | 2 |
| **合計** | **321.64** | **7.77** | **3.52** | **339.38** | — | — |

### PP Phase 時間内訳

```
graph_compute 合計: 339.38 ms
├── 同期 flush (サーバー計算待ち): 321.64 ms (94.8%)
├── グラフ serialize:               7.77 ms  (2.3%)
├── RDMA Send:                      3.52 ms  (1.0%)
└── その他 (drain, snapshot):       6.45 ms  (1.9%)
```

### TG Phase: 最初の FULL_GRAPH (calls 9-12)

| Call | flush (ms) | serialize (ms) | send (ms) | total (ms) |
|------|-----------|----------------|-----------|------------|
| #9 | 0.03 | 0.30 | 0.56 | 1.33 |
| #10 | 5.52 | 0.44 | 0.09 | 6.47 |
| #11 | 5.00 | 0.47 | 0.07 | 5.99 |
| #12 | 5.53 | 0.23 | 0.52 | 6.61 |

TG の recompute (calls 13+): **~0.5 ms/call** (flush+compute を 1 コマンドに統合済み)

### サーバー側 GPU 計算時間の推定

flush 待ち時間 ≈ 前デバイスのサーバー側 GPU 計算時間:

| デバイス | 推定計算時間 (iter1) | 推定計算時間 (iter2) | ノード数 |
|---------|-------------------|-------------------|---------|
| Device 0 | ~88 ms | ~53 ms | 1065 |
| Device 1 | ~41 ms | ~41 ms | 1065 |
| Device 2 | ~42 ms | ~43 ms | 1173 |
| Device 3 | (不明) | (不明) | 703 |

- Device 0 の iter1 が最も遅い (~88 ms) — 初回初期化オーバーヘッドの可能性
- Devices 1-3 は安定して ~42 ms

### set_tensor Per-Call 分析

**モデルロード中** (calls 1-480):
- 150 MB チャンク: ~35-45 ms (RDMA Write, ~850 MB/s)
- 小さなテンソル (8KB-9MB): ~0.01-3 ms
- 大チャンク (350 MB, call #386): 1,562 ms (コンテキスト初期化)

**推論中 (TG)** (calls 530+):
- 4-16 バイトのメタデータ: ~4.6 ms (signaled RDMA Write + CQ poll オーバーヘッド)
- 8-512 バイトのテンソル: < 0.01 ms (unsignaled)

### 性能結果

| 指標 | 値 |
|------|-----|
| Prompt throughput | 53.3 t/s |
| Generation throughput | 32.9 t/s |

## ボトルネック分析

### 結論: **Case C — サーバー側 GPU 計算がボトルネック**

PP phase における時間内訳:

| カテゴリ | 時間 (ms) | 割合 |
|---------|----------|------|
| サーバー側 GPU 計算 (flush 待ち) | 321.64 | **94.8%** |
| RDMA 通信 (serialize + send) | 11.29 | 3.3% |
| その他 (drain, snapshot) | 6.45 | 1.9% |
| **合計** | **339.38** | 100% |

**RDMA 通信は pp 時間の 3.3% に過ぎず、RDMA レベルの最適化で改善できる余地は事実上ない。**

### メカニズムの説明

FULL_GRAPH path での graph_compute の流れ:

```
1. Client: drain pending flushes → 即座に完了
2. Client: send RDMA_CMD_FLUSH_ALL_STAGING (同期) → サーバーが前デバイスの計算完了を待つ
3. Client: serialize graph → ローカル処理 (~1 ms)
4. Client: send RDMA_CMD_GRAPH_COMPUTE_ASYNC → 非同期送信 (~0.5 ms)
5. Client: build snapshot → ローカル処理 (~0.3 ms)
```

Step 2 の同期 flush が **律速** となる。サーバーは単一コネクションの単一スレッドでコマンドを順次処理するため、前デバイスの GPU 計算が完了するまで flush レスポンスを返さない。

### 理論的な改善上限

仮にフラッシュを非同期化 (recompute path と同様に flush + graph を 1 コマンドに統合) しても:
- クライアント側の待ち時間は削減される (339 ms → ~12 ms)
- しかしサーバー側の計算合計は変わらない (~300 ms)
- 結果収集 (get_tensor/synchronize) で同じ時間待つことになる
- **合計 pp 時間は変わらない** (サーバー側計算がクリティカルパス)

### 対比: GLM-4.7 11GPU との違い

| 要素 | Qwen3.5 6GPU | GLM-4.7 11GPU |
|------|-------------|--------------|
| Active params | 3B (MoE) | 9B (Dense) |
| GPU 数 | 6 | 11 |
| RDMA デバイス数 | 4 | 4 |
| 計算密度 | 低 (3B/6GPU) | 高 (9B/11GPU) |
| RDMA 通信比率 | 3.3% | 推定 > 20% |
| Selective signaling 効果 | なし (-0.06%) | +26.1% pp |

Qwen3.5 MoE は active パラメータが 3B と少なく、6GPU で分散すると各デバイスの計算負荷が軽い。結果として RDMA 通信がボトルネックにならない。一方 GLM-4.7 は 9B の dense モデルで 11GPU に分散するため、通信比率が相対的に高く、selective signaling の効果が出る。

## 推奨次ステップ

1. **Qwen3.5 での PP 最適化は終了** — RDMA レベルでは改善不可。計算バウンドであり、P100 の計算能力が律速
2. **GLM-4.7 11GPU での double-buffering 検証** — selective signaling が効果的な構成で double-buffering の効果を確認 (pp128 ≈ 30.6 の回復が見込める)
3. **FULL_GRAPH path の flush 非同期化** は将来的な改善候補 — クライアント側の応答性改善にはなるが、pp スループットには影響しない

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c1ca6d5fd (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 37°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 1203493) |

テスト構成:
- **Node 1**: CUDA5, CUDA6 (`CUDA_VISIBLE_DEVICES=5,6`)
- **Node 2**: RDMA0-3 (`GGML_RDMA_SERVERS=192.168.100.2:50051`)
- **モデル**: `unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M`
- **ワークツリーコミット**: `f1f2cbe90` (feature/double-buffering)
