# CUDA/RDMA Overlap マージ + synchronize 退行修正レポート

- **実施日時**: 2026年3月12日 06:12
- **ワークツリー**: `.worktree/merge-overlap` (ブランチ: `merge/cuda-rdma-overlap`)

## 前提・目的

PP/TG 性能向上ロードマップの Phase 1-1 として、`.worktree/cuda-rdma-overlap` (commit `64a67bf25`) から 16 コミットを `feature/rdma-backend` に cherry-pick し、Pipeline parallelism + Send ring buffer + CUDA/RDMA Overlap 機能をマージする。全機能は環境変数ゲート付き (デフォルト無効)。

### 参照レポート
- [PP/TG 性能向上ロードマップ](2026-03-12_004808_pp_tg_optimization_roadmap.md)

## Cherry-pick 内容

16 コミットを `feature/rdma-backend` (453f4c7c1) 上に cherry-pick:

1. `a64327cd1` - inter-ubatch pipeline parallelism (llama-context.cpp)
2. `4b74e4f07` - no-op event caps for RDMA backend
3. `a37da4548` - functional RDMA events and synchronize
4. `84871fc37` - eliminate sync flush + pipeline profiling
5. `abd9a0b46` - multi-slot graph_cache
6. `6f04a013e` - split-level pipeline parallelism (Phase 2-4)
7. `7cb1f461b` - per-copy split snapshots
8. `5c6b5ac68` - per-copy context buffers fix
9. `73af4238a` - GGML_RDMA_PARALLEL env var
10. `ad9177a77` - remove chunked pipeline dispatch
11. `d09174731` - drain-before-copy for cross-device
12. `e93bc1319` - per-device server-side sync
13. `921676339` - send ring buffer
14. `7969d905e` - CUDA/RDMA overlap
15. `73cc8e2e2` - 3-phase CUDA/RDMA overlap fix
16. `c752dab4f` - no_wait send + drain_send_cq

1 件のコンフリクト (commit 4, llama-context.cpp) を手動解決。残りは全てクリーンに適用。

## 退行の発見とバイナリサーチ

### 初回テスト結果 (退行あり)

| 指標 | ベースライン | merge-overlap | 差分 |
|------|:-----------:|:------------:|:----:|
| 122B pp128 | 116.7 ± 1.5 | 115.0 ± 1.1 | -1.5% |
| 122B pp16384 | 169.6 ± 0.6 | 167.6 ± 0.5 | -1.2% |
| 122B tg32 | 18.29 ± 0.02 | 17.73 ± 0.01 | **-3.1%** |
| 35B tg32 | 35.0 ± 0.1 | 35.7 ± 0.1 | +1.9% |

11GPU tg32 の -3.1% 退行は再テストでも再現 (17.65 ± 0.01)。

### バイナリサーチ結果

| テスト構成 | tg32 | 判定 |
|-----------|:----:|:---:|
| Baseline client + baseline server | 18.29 | ✅ |
| Commit 1 client + baseline server | 18.29 | ✅ (llama-context.cpp のみ、影響なし) |
| Commit 1 client + commit 3 server | 18.30 | ✅ (サーバー側変更は影響なし) |
| Commit 3 client + commit 3 server | 17.57 | ❌ (-3.9%) |
| Commit 3 client (async=false fix) + commit 3 server | 17.67 | ❌ (async flag は原因でない) |
| Commit 3 client (synchronize no-op) + commit 3 server | **18.30** | ✅ **退行解消** |
| Commit 6 client + commit 6 server | 17.71 | ❌ (-3.2%) |
| Commit 12 client + commit 12 server | 17.53 | ❌ (-4.2%) |
| Full (16 commits) client + server | 17.65 | ❌ (-3.5%) |

### 根本原因

**Commit 3** (`a37da4548`) が `ggml_backend_rdma_synchronize()` を no-op から `drain_pending_compute()` に変更。この関数は `ggml_backend_sched_compute_splits()` 内で各 split 境界で複数回呼ばれる (lines 1459, 1492, 1569, 1582)。

`drain_pending_compute()` は `RDMA_CMD_FLUSH_ALL_STAGING` (0エントリ) を送信し応答を待つ IB ラウンドトリップ。11GPU (4 RDMA デバイス) では、TG の各ステップで数十回の不要なラウンドトリップが発生し、~2ms/step の追加レイテンシとなっていた。

### 追加原因

**Commit 2** (`4b74e4f07`) が `async=true, events=true` を無条件に設定。これにより `pipeline_parallel=true` が自動有効化され、`n_copies=4` + イベント同期が全操作に適用された。

### 修正内容 (commit `f38528b20`)

1. `synchronize()` を `RDMA_PIPELINE` 有効時のみ drain 実行するよう条件化
2. `async`/`events` caps を `RDMA_PIPELINE` に連動させ、デフォルトでは無効化

## 修正後の最終結果

### 11GPU (122B fused, 7C+4R)

| 指標 | ベースライン | 修正版 merge-overlap | 差分 |
|------|:-----------:|:-------------------:|:----:|
| pp128 | 116.7 ± 1.5 | 116.9 ± 1.3 | +0.1% |
| pp16384 | 169.6 ± 0.6 | 169.6 ± 0.7 | 0.0% |
| tg32 | 18.29 ± 0.02 | 18.31 ± 0.01 | +0.1% |

### 4GPU (35B fused, 2C+2R)

| 指標 | ベースライン | 修正版 merge-overlap | 差分 |
|------|:-----------:|:-------------------:|:----:|
| pp128 | 273.4 ± 2.6 | 274.2 ± 2.9 | +0.3% |
| pp512 | 454.9 ± 2.4 | 457.3 ± 2.2 | +0.5% |
| pp2048 | 515.9 ± 2.7 | 577.5 ± 3.6 | **+11.9%** |
| tg32 | 35.0 ± 0.1 | 36.3 ± 0.1 | **+3.5%** |

**pp2048 +11.9%** は `RDMA_CMD_FLUSH_AND_GRAPH_COMPUTE_ASYNC` 統合コマンドにより flush + copies + graph_compute を 1 コマンドに統合し、IB ラウンドトリップが削減されたことによる。

### 正確性テスト
- 35B 4GPU (2C+2R) で `llama-cli` による 256 token 生成: ✅ 正常な推論出力

## 再現方法

```bash
# 1. ワークツリー作成 + cherry-pick
git worktree add .worktree/merge-overlap -b merge/cuda-rdma-overlap feature/rdma-backend
# (16 commits cherry-pick + fix commit)

# 2. ビルド・デプロイ
bash .worktree/merge-overlap/scripts/rdma-build.sh local
bash .worktree/merge-overlap/scripts/rdma-deploy.sh
bash .worktree/merge-overlap/scripts/rdma-server.sh restart

# 3. 11GPU テスト
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/merge-overlap/build/bin/llama-bench \
  -m models/Qwen3.5-122B-A10B-Q4_K_M-fused.gguf \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -t 1 -r 5 -p 128,16384 -n 0,32
```

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `f38528b20 (merge/cuda-rdma-overlap)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 25°C | 32°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1979336) |

GGML_RDMA 環境変数: (none)

## 結論

- Phase 1-1 (CUDA/RDMA Overlap マージ) は修正込みで完了
- 16 コミット cherry-pick + 退行修正 1 コミット = 計 17 コミット
- デフォルトパスで退行なし、pp2048 で +11.9% 改善を確認
- Pipeline/Overlap 機能は `GGML_RDMA_PIPELINE=1` / `GGML_RDMA_CUDA_OVERLAP=1` で有効化可能
- `feature/rdma-backend` へのマージはユーザー判断
