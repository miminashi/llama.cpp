# Expert Parallelism Phase 8 — RDMA グラフキャッシュ type 変更対応

- **実施日時**: 2026年2月24日 01:12
- **ワークツリー**: `.worktree/expert-parallelism`
- **ブランチ**: `feature/expert-parallelism`
- **コミット**: `05ae04bbb`

## 前提・目的

Phase 7 で USAGE_WEIGHTS 修正により EP=11 の get_tensor データ量を 99.96% 削減したが、性能は pp=0.3, tg=0.8 t/s にとどまっている。

RDMA プロファイルデータ (修正前):
```
graph_compute: calls=8100, type: full=3342, update=4758
```

**42% が full send** — graph_cache の差分更新 (update) ではなく、毎回グラフ全体を再シリアライズしていた。

### 根本原因

GLM-4.7 IQ2_M は混合量子化 (IQ2_M, Q4_K_S, Q5_K_M 等)。EP=11 ではトークンごとにエキスパートルーティングが変わり、MUL_MAT_ID の src[0] (重みテンソル) の `type` フィールドが変化する。

グラフキャッシュの `is_cached()` と `compare_and_emit()` が type 変更をキャッシュ無効化として扱っていた。type 変更は構造的変化ではなくパラメータ変更として差分更新で送信可能。

### 参照レポート

- [Phase 7 USAGE_WEIGHTS レポート](2026-02-24_000458_expert_parallelism_phase7_usage_weights.md)
- [Phase 6 グラフキャッシュレポート](2026-02-23_095006_expert_parallelism_phase6_graph_cache.md)

## 変更内容

ファイル: `ggml/src/ggml-rdma/ggml-rdma.cpp` (28 insertions, 23 deletions)

1. **`rdma_indexed_update` 構造体**: `padding` フィールドを `type` に変更 (-1 = 変更なし, >=0 = ggml_type)
2. **`is_cached()`**: type チェックを除外 (op のみで構造判定)
3. **`build_remap_recursive()`**: positional remap の type チェックを除外
4. **`compare_and_emit()`**: type 変更を差分更新として emit (op 変更のみキャッシュ無効化)
5. **`collect_indexed_updates()`**: サイズ不一致時に `return false` (バグ修正、以前は `return true`)
6. **`graph_compute_indexed_update()`** (サーバー側): `upd.type >= 0` の場合にテンソル type を適用

## 再現方法

### ビルド・デプロイ

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/scripts/rdma-build.sh local
gpu-lock.sh run bash /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/scripts/rdma-deploy.sh
bash /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/scripts/rdma-server.sh start
```

### EP=11 テスト

```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 \
  /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

### 非 EP 回帰テスト

```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

## 結果

### EP=11 graph_compute プロファイル

| 指標 | 修正前 | 修正後 | 変化 |
|------|--------|--------|------|
| full send | 3342/8100 (41.3%) | 1333/8100 (16.5%) | **-60%** |
| update (差分) | 4758 | 6767 | +42% |
| avg graph_compute | — | 1.3 ms | — |

type 由来の full send (~2009回) は排除された。残りの 1333 回は **n_nodes 変更** (EP でトークンごとに異なるグラフ構造) が原因であり、これは正当なキャッシュ無効化。

### EP=11 性能

| 指標 | 修正前 | 修正後 |
|------|--------|--------|
| pp | 0.3 t/s | 0.3 t/s |
| tg | 0.8 t/s | 0.8 t/s |

性能は変化なし。ボトルネックは graph_compute ではなく **set_tensor** にある:
- set_tensor: 120,896 calls, 52,560 MB, 114,076 ms (RDMA 時間の 88%)
- 毎トークン、全エキスパート重みを RDMA 経由で転送しているため

### 非 EP 回帰テスト

| 指標 | 期待値 | 実測値 |
|------|--------|--------|
| pp | ~9 t/s | 8.8 t/s |
| tg | ~8 t/s | 8.8 t/s |

回帰なし。

### RDMA プロファイル詳細 (EP=11, 最終サマリ)

```
set_tensor:  calls=120896, bytes=55113153648 (52560.0 MB), total=114075.8 ms, avg=943.6 us, 483.1 MB/s
  path: RDMA Write=119828, Send/Recv=1068
get_tensor:  calls=15979, bytes=208013728 (198.4 MB), total=5426.2 ms, avg=339.6 us, 38.3 MB/s
  path: RDMA Read=14091, Send/Recv=1888
graph_compute: calls=8100, total=10348.1 ms, avg=1.3 ms
  type: full=1333, update=6767, recompute=0
total RDMA time: 129850.0 ms
```

## 分析・次のステップ

### グラフキャッシュ改善の限界

graph_compute の合計時間は 10.3 秒で、RDMA 時間全体 (129.9 秒) の 8% に過ぎない。full send を完全に排除しても最大 8% の改善にとどまる。

### 真のボトルネック: set_tensor (52GB)

EP=11 では毎トークン全エキスパート重みが set_tensor で転送されている (52GB/20トークン ≈ 2.6GB/トークン)。Phase 7 で get_tensor を 99.96% 削減したが、set_tensor は未対応。

次のステップとして検討すべき:
1. **エキスパート重みのサーバー側キャッシュ** — 重み変更がないエキスパートの再送を回避
2. **set_tensor の差分転送** — 変更されたテンソルのみ送信
3. **n_nodes 変更への対応** — EP でグラフ構造が変わる場合のキャッシュ戦略

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `5a7cd6bcb (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 36°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | running (PID 620090) | — |
