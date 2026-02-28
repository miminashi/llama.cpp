# Expert Parallelism Phase 7: EP 重みバッファ USAGE_WEIGHTS 設定

- **実施日時**: 2026年2月23-24日
- **ワークツリー**: `.worktree/expert-parallelism`
- **ブランチ**: `feature/expert-parallelism`
- **コミット**: `6d9870b29`

## 前提・目的

Phase 5-6 のプロファイリングで、EP=11 推論の **97.5% の時間が set_tensor/get_tensor** に費やされることが判明した:

| 操作 | 時間 (ms) | 呼び出し回数 | データ量 | 比率 |
|:-----|:---------:|:----------:|:-------:|:----:|
| set_tensor | 609,651 | 15,794 | 547 GB | 46.4% |
| get_tensor | 672,656 | 14,384 | 496 GB | 51.2% |
| graph_compute | 32,252 | — | — | 2.5% |

Per graph_compute 平均 **730 MB** の set_tensor データは、tg 単一トークンのアクティベーション (~16 KB) の4.5万倍。明らかに重みデータが毎 forward pass で再転送されていた。

### 根本原因

通常の重みバッファ (`llama-model.cpp:7773`) は `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` が設定されるが、EP 重みバッファ (`llama-model.cpp:7889`) では未設定だった。

**影響**: スケジューラの `ggml_backend_sched_backend_id_from_cur` は `buffer->usage == WEIGHTS` を条件に重みテンソルからノードのバックエンドを決定する。EP 重みバッファは `usage=ANY` (デフォルト) のため不成立。結果:
1. **Pass 1**: MUL_MAT_ID ノードのバックエンドを重みの位置から決定できない
2. **Pass 2**: 隣接ノードからの伝播で CUDA に割り当て
3. **Pass 5**: EP 重み (RDMA) ≠ ノード (CUDA) → スプリット入力としてコピー作成
4. **Compute Splits**: 毎 forward pass で EP 重みを RDMA→CUDA コピー

### 参照レポート

- [Phase 5 性能レポート](2026-02-22_165358_expert_parallelism_phase5_performance.md)
- [Phase 6 グラフキャッシュレポート](2026-02-23_095006_expert_parallelism_phase6_graph_cache.md)

## 修正内容

### 1. EP 重みバッファに USAGE_WEIGHTS 設定 (llama-model.cpp)

```cpp
auto * buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_ep, buft);
ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);  // NEW
```

### 2. RDMA グループの REMAP/MASK をバックエンド強制 (llama-graph.cpp)

USAGE_WEIGHTS を設定すると MUL_MAT_ID がサーバー側 CUDA で実行されるが、スケジューラの "expand GPU down" パスが EXPERT_REMAP_IDS を前のグループのバックエンドに割り当て、RDMA-to-RDMA コピーで expert ID が破損する問題が発生した。

`ggml_backend_sched_set_tensor_backend` を使用して、RDMA グループのみ REMAP/MASK を EP 重みと同じバックエンドに強制:

```cpp
for (int g = 0; g < n_ep_groups; g++) {
    ggml_tensor * ep_weight = up_exps_ep[g];
    if (sched && ep_weight && ep_weight->buffer) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(ep_weight->buffer);
        const char * buft_name = ggml_backend_buft_name(buft);
        if (strncmp(buft_name, "RDMA", 4) != 0) continue;  // RDMA groups only
        // ... find matching backend and force assignment
    }
}
```

### 3. GGML_SCHED_MAX_SPLIT_INPUTS を 128 に増加 (ggml-backend.cpp)

EP=11 の MoE グラフは 11 デバイス間で大量の cross-backend 入力を生成し、デフォルトの 30 を超える。128 に増加。

### 4. デバッグコード削除 (ggml-rdma.cpp)

Phase 5-6 で追加した MMID 検証コード・DATA_VALIDATION・POST_FIX_VALIDATION を削除 (-133 行)。

## テスト結果

### EP=11 テスト (GLM-4.7 IQ2_M, 7C+4R)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 \
  llama-cli -m GLM-4.7-UD-IQ2_M.gguf \
  -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20
```

| 指標 | Phase 6 (修正前) | Phase 7 (今回) | 改善 |
|:-----|:----------------:|:-------------:|:----:|
| **pp (t/s)** | 0.1 | **0.3** | +200% |
| **tg (t/s)** | 0.02 | **0.8** | +3900% |
| get_tensor bytes | 496 GB | **198 MB** | -99.96% |
| set_tensor bytes | 547 GB | 52.6 GB* | -90.4% |
| graph_compute avg | — | 1.1 ms | — |
| graph_compute calls | — | 8100 | — |

\* set_tensor 52.6 GB の大部分はモデルロード時の重み転送

### RDMA プロファイルサマリー

```
set_tensor:  calls=120896, bytes=52560 MB, total=118010 ms, 467 MB/s
get_tensor:  calls=15979, bytes=198 MB, total=6986 ms, 30 MB/s
graph_compute: calls=8100, total=9244 ms, avg=1.1 ms
  type: full=3342, update=4758, recompute=0
```

### Compute バッファ改善

| デバイス | Phase 6 | Phase 7 | 削減率 |
|:---------|:-------:|:-------:|:------:|
| CUDA0-6 | 2649-2892 MB | **98 MB** | -96% |
| RDMA0-2 | 2637 MB | **86 MB** | -97% |
| RDMA3 | 3275 MB | 3275 MB | 0% |

RDMA3 の大きな compute バッファは ADD chain がスケジューラにより RDMA3 に伝播されるため。

### 非 EP 回帰テスト

| 指標 | ベースライン | 今回 | 判定 |
|:-----|:----------:|:----:|:----:|
| pp (t/s) | ~8.7 | 9.1 | 回帰なし |
| tg (t/s) | ~8.5 | 8.1 | 回帰なし |

## デバッグ過程で検証した代替アプローチ

### ADD chain のプライマリ CUDA 強制 (不採用)

ADD chain (Phase C) を gate_inp のデバイスに強制する方式をテスト。結果:
- get_tensor: 198 MB → **4,689 MB** (24倍増)
- graph_compute avg: 1.1 ms → **11.9 ms** (11倍増)
- send time per graph: 0.12 ms → **66 ms** (550倍増)

**不採用理由**: RDMA graph_compute の非同期パイプラインを破壊。サーバー側の partial 結果を同期取得するため、パイプライン効果がゼロになる。

### 全 EP グループの REMAP 強制 (不採用)

CUDA グループも含め全 11 グループの REMAP/MASK を強制。`GGML_SCHED_MAX_SPLIT_INPUTS` を 256 に引き上げる必要があり、RDMA-only 限定の方が変更範囲が小さい。性能差は微小 (tg 1.0 vs 0.8)。

## 残存ボトルネック分析

### graph_compute スプリット数の爆発

非 EP: ~30 graph_computes/forward pass (デバイスあたり 1-2 スプリット)
EP=11: **~405 graph_computes/forward pass** (62 MoE レイヤー × 11 デバイス)

各 graph_compute のシリアライズ・送信・実行レイテンシー (avg 1.1 ms) が積み重なり、総計 ~445 ms/forward pass。非 EP の ~30 ms/forward pass に対して **15 倍**。

### ADD chain の RDMA3 伝播

スケジューラの "expand GPU down" が MoE 出力の集約 (ADD chain) を最後の RDMA グループ (RDMA3) に割り当て。これにより:
- 全 partial 結果がネットワーク経由で RDMA3 に転送される
- RDMA3 の compute バッファが 3275 MB (他デバイスの 37 倍)
- 残差接続もサーバー側で実行される可能性

### 改善の方向性

1. **スプリット統合**: 複数 MoE レイヤーの graph_compute をバッチ化してスプリット数を削減
2. **ADD chain のスケジューリング**: Phase C の集約をクライアント側 CUDA で実行しつつパイプラインを維持する方法
3. **Per-device connections**: `GGML_RDMA_PER_DEVICE_CONN=1` で graph_compute の並列送信を検討 (ただし ConnectX-4 の MTT キャッシュ制限あり)

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `5a7cd6bcb (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 35°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 611739) |

## 再現方法

1. ワークツリー `expert-parallelism` をチェックアウト (コミット `6d9870b29`)
2. ビルド・デプロイ:
   ```bash
   bash scripts/rdma-build.sh local
   gpu-lock.sh run bash scripts/rdma-deploy.sh
   bash scripts/rdma-server.sh start
   ```
3. EP=11 テスト:
   ```bash
   gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 \
     build/bin/llama-cli -m GLM-4.7-UD-IQ2_M.gguf \
     -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
     -p "Hello, I am a large language model" -n 20 \
     --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
   ```
