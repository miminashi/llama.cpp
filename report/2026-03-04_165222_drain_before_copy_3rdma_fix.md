# Combined モード 3+ RDMA デバイス ゴミ出力修正 (drain-before-copy)

- **実施日時**: 2026年3月4日 16:52
- **ワークツリー**: `.worktree/pipeline-splits`
- **ブランチ**: `feature/pipeline-splits`
- **コミット**: `01348a528`

## 前提・目的

`GGML_RDMA_PARALLEL=1` (pipeline + per-device connections combined) で 3+ RDMA デバイス使用時にゴミ出力が発生するバグを修正する。

- **背景**: Bug 1 (pp20000 クラッシュ) は `a423ade64` で修正済みだが、Bug 2 (3+ RDMA combined でのゴミ出力) は未解決
- **発現条件**: Per-device alone + 3 RDMA = OK, Pipeline alone + 3 RDMA = OK, Combined + 2 RDMA = OK, **Combined + 3 RDMA = FAIL**
- **根本原因**: `cpy_tensor_async` の cross-connection パスで `pipeline_wait` (サーバー側待機) を使って依存関係を管理しているが、3+ デバイスのチェーン (R0→R1→R2) で実際にデータ破損が発生
- **参照**: [pipeline + per-device combined ベンチマーク](report/2026-03-03_145744_pipeline_perdevice_combined_benchmark.md), [pipeline pp20000 bugfix](report/2026-03-04_134239_pipeline_bugfix_per_ubatch_sync.md)

## 修正内容

`ggml_backend_rdma_cpy_tensor_async` の cross-connection パスで、`pipeline_wait` (サーバー側待機) を `drain_pending_compute` (クライアント側同期) に置き換え。

**変更前**: cross-connection の deferred copy 時に `get_pipeline_state(dst_conn)->add_wait(src_device, src_seq)` でサーバー側に待機を依頼。3+ デバイスでタイミング問題によりデータ破損。

**変更後**: deferred copy 登録前に `drain_pending_compute(src_conn)` でソースデバイスの非同期計算完了をクライアント側で待機。サーバー側の pipeline_wait は不要。

### 保持される最適化
- Deferred copy (サーバー側 D2H+H2D): クライアント経由のラウンドトリップ回避
- 非同期コマンドディスパッチ: graph_compute は fire-and-forget
- Pipeline-only モード: 同一接続の場合は変更なし

### 失われる最適化
- サーバー側デバイス間並列性: クライアントが dispatch を直列化するため、R0→R1→R2 が順次実行に

## 再現方法

### ビルド・デプロイ
```bash
bash .worktree/pipeline-splits/scripts/rdma-build.sh local
bash scripts/gpu-lock.sh run bash .worktree/pipeline-splits/scripts/rdma-deploy.sh
bash scripts/gpu-lock.sh run bash .worktree/pipeline-splits/scripts/rdma-server.sh restart
```

### Bug 2 正確性テスト (3 RDMA combined)
```bash
bash scripts/gpu-lock.sh run GGML_RDMA_PARALLEL=1 CUDA_VISIBLE_DEVICES=4,5,6 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/pipeline-splits/build/bin/llama-cli \
  --model /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051]' \
  -ngl 999 -sm layer -fa 1 -p '日本語で美しい詩を書いてください。' -n 128 \
  --log-file /tmp/llama-cli.log
```

## 検証結果

### Step 1: Bug 2 正確性テスト (3 RDMA combined)
- **結果: PASS** — 正常な日本語出力を確認、ゴミ出力なし
- PP 54.9 t/s, TG 34.9 t/s (6GPU: 3 CUDA + 3 RDMA)

### Step 2: Bug 1 回帰テスト (pp20000 pipeline)
- **結果: PASS** — クラッシュなし
- pp20000 = 281.7 t/s, tg32 = 36.2 t/s (4GPU: 2 CUDA + 2 RDMA)

### Step 3: Baseline 回帰テスト (PARALLEL なし)
- **結果: PASS** — 回帰なし
- pp128 = 201.9 t/s, tg32 = 36.0 t/s (4GPU: 2 CUDA + 2 RDMA)

### Step 4: 2 RDMA combined 回帰テスト
- **結果: PASS** — 正常な日本語出力を確認
- PP 62.7 t/s, TG 35.8 t/s (4GPU: 2 CUDA + 2 RDMA)

## 結果サマリ

| テスト | 条件 | 結果 | 備考 |
|--------|------|------|------|
| Bug 2 (3 RDMA combined) | PARALLEL=1, 3C+3R | **PASS** | ゴミ出力解消 |
| Bug 1 (pp20000 pipeline) | PIPELINE=1, 2C+2R | **PASS** | クラッシュなし |
| Baseline (no flags) | 2C+2R | **PASS** | 回帰なし |
| 2 RDMA combined | PARALLEL=1, 2C+2R | **PASS** | 回帰なし |

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c1ca6d5fd (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 35°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 1344608) |
