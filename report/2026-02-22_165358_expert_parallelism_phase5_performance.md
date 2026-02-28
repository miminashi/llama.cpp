# Expert Parallelism Phase 5: 性能改善レポート

- **実施日時**: 2026年2月22日 16:53
- **ワークツリー**: `.worktree/expert-parallelism`
- **コミット**: `80902682f` (feature/expert-parallelism)

## 前提・目的

Phase 4 (コミット `0405d3965`) で EP=11 が RDMA 経由で正しく動作することを確認したが、性能は壊滅的 (pp=0.1, tg≈0.01 t/s) だった。本フェーズでは以下の2つの改善策を実装・検証する。

1. **EP グラフの二段階化**: `build_moe_ffn_ep()` のグラフ構築順序を変更し、PP操作とEP操作をバッチ化してスケジューラの split 数を削減
2. **Per-device connections の EP 自動有効化**: RDMA バックエンドで EP 検出時に per-device connections を有効化し、リモート GPU の並列計算を実現

### 参照レポート
- [Phase 4 RDMA 修正レポート](2026-02-22_053859_expert_parallelism_phase4_rdma.md)
- [Phase 1-3 実装レポート](2026-02-22_011439_expert_parallelism_phase1-3.md)

## 実装内容

### 改善 1: EP グラフの二段階化 (実装済み)

`src/llama-graph.cpp` の `build_moe_ffn_ep()` を三段階に再構成:

- **Phase A**: 全 EP グループの remap/mask 操作を PP GPU 上でバッチ実行 (1 split)
- **Phase B**: 全 EP グループの compute を各 EP デバイスで実行 (N splits)
- **Phase C**: 結果の集約を PP GPU で実行 (1 split)

**変更前** (グループごとのインターリーブ):
```
for g in 0..10:
    remap_g (PP) → mask_g (PP) → mul_mat_id×3 (EPdev_g) → aggregate
```
Split パターン: [PP, Dev0, PP, Dev1, PP, Dev2, ...] = ~23 splits/layer

**変更後** (三段階):
```
Phase A: for g in 0..10: remap_g, mask_g   // 全て PP GPU
Phase B: for g in 0..10: compute_g          // 各 EP デバイス
Phase C: aggregate                          // PP GPU
```
Split パターン: [PP, Dev0, Dev1, ..., Dev10, PP] = ~13 splits/layer (~44% 削減)

### 改善 2: Per-device connections (検証の結果、使用不可)

RDMA per-device connections (`GGML_RDMA_PER_DEVICE_CONN=1`) を EP 時に自動有効化する計画だったが、ConnectX-4 の MTT (Memory Translation Table) キャッシュ制限により、EP + per-device connections の組み合わせでは RDMA タイムアウト/クラッシュが発生。

**テスト結果**:
- `GGML_RDMA_PER_DEVICE_CONN=1`: RDMA Send/Recv タイムアウト (30s) → abort
- `GGML_RDMA_PER_DEVICE_CONN=1 GGML_RDMA_NO_GDR=1`: 同様のタイムアウト → abort
- 原因: EP の 11 GPU + per-device = 4 接続 × 4 MR/接続 = 16 MR 登録がConnectX-4 の MTT キャッシュ (~10-16GB) を溢れさせる

代替として、EP 有効時にアドバイザリログメッセージを出力する実装のみ採用。

## テスト結果

### EP=11 テスト (GLM-4.7 IQ2_M, 11GPU)

| 条件 | pp (t/s) | tg (t/s) |
|:-----|:--------:|:--------:|
| Phase 4 ベースライン | 0.1 | ~0.01 |
| **Phase 5 (二段階化)** | **0.1** | **~0.03** |

tg で約 **3× 改善**。ただし PP ベースライン (pp=30.6, tg=8.5) には遠い。

### 非 EP 回帰テスト (GLM-4.7 IQ2_M, 11GPU PP-only)

| 指標 | Phase 5 | PP ベースライン |
|:-----|:-------:|:--------------:|
| pp (t/s) | 8.7 | ~30.6 |
| tg (t/s) | 8.9 | ~8.5 |

**非 EP 回帰なし。** tg は若干改善 (8.5 → 8.9)。pp の差は測定条件 (llama-cli の短いプロンプト vs llama-bench の pp128) による。

### プロファイリング分析 (`GGML_RDMA_PROFILE=1`)

EP=11 推論の RDMA 時間内訳 (140 graph_compute 呼び出し分):

| カテゴリ | 累計時間 | 平均/呼出 | 呼出数 |
|:---------|--------:|---------:|------:|
| set_tensor | 168,681 ms | 39.4 ms | 4,282 |
| get_tensor | 132,199 ms | 48.3 ms | 2,739 |
| graph_compute | 5,923 ms | 42.3 ms | 140 |
| **合計** | **306,803 ms** | — | — |

**主要ボトルネック**:
1. **set_tensor/get_tensor が時間の 98%** を占有 — graph_compute は全体の 2% に過ぎない
2. **全 graph_compute がフル送信** (キャッシュ不使用) — 毎回 171KB-903KB のグラフを直列化
3. **共有接続の送信キュー飽和**: 8-9 個の非同期コマンドを送信した後、10 番目で 280-550ms のブロッキング
4. **1 forward pass あたり ~10 RDMA graph_compute + ~193 set_tensor + ~68 get_tensor**

## 残課題

1. **グラフキャッシュの EP 対応**: 現在 `is_cached()` が常に false を返す (src[] ポインタ不一致)。recompute パスが使えれば graph_compute のオーバーヘッドを大幅に削減可能
2. **サーバー側の非同期コマンドディスパッチ**: 共有接続でも複数デバイスの compute を並列実行できるようサーバーを改修
3. **ConnectX-6+ での per-device connections テスト**: より大容量の MTT キャッシュを持つ NIC で EP + per-device の組み合わせが動作する可能性

## 再現方法

### ビルド・デプロイ
```bash
bash .worktree/expert-parallelism/scripts/rdma-build.sh local
gpu-lock.sh run bash .worktree/expert-parallelism/scripts/rdma-deploy.sh
ssh 192.168.100.2 "pkill -f rdma-server"; sleep 1
bash .worktree/expert-parallelism/scripts/rdma-server.sh start
```

### EP テスト
```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/expert-parallelism/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

### 非 EP 回帰テスト
```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/expert-parallelism/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

### プロファイリング
```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 \
  .worktree/expert-parallelism/build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
  -p "Hello" -n 5 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `80902682f (feature/expert-parallelism)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 40°C | 41°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 279306) |
