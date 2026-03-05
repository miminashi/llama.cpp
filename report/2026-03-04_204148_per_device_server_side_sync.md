# Per-Device サーバー側同期によるクロスデバイス Deferred Copy 改善

- **実施日時**: 2026年3月4日 20:41
- **ワークツリー**: `.worktree/pipeline-splits`
- **コミット**: `d0d1c678a`

## 前提・目的

### 背景

`cpy_tensor_async` でクロスデバイスコピー時に `drain_pending_compute(src_conn)` を呼び、クライアント側で `FLUSH_ALL_STAGING` ラウンドトリップをブロッキング待ちしていた（commit `01348a528`）。これによりクライアントのディスパッチループが止まり、サーバー側デバイス間並列性が失われていた。

- 参照: [drain-before-copy 修正](report/2026-03-04_165222_drain_before_copy_3rdma_fix.md)
- 参照: [pipeline + per-device combined ベンチマーク](report/2026-03-03_145744_pipeline_perdevice_combined_benchmark.md)

### 目的

クライアント側ブロッキング drain をサーバー側 per-device 待ち合わせに置き換え、デバイス間並列性を回復する。

### 前提条件

- `feature/pipeline-splits` ブランチの `01348a528` をベース
- Qwen3.5-35B-A3B (MoE, 3B active) を 4C+4R (CUDA0-3 + RDMA0-3) で使用
- 1号機と2号機が 100GbE RDMA で接続されていること

## 実装

### 変更内容

1. **`deferred_copy_entry`**: `padding[2]` → `wait_device` + `wait_expected_seq` (48B 維持)
2. **`deferred_copy_list::add`**: オプショナル `wait_dev`/`wait_seq` パラメータ追加
3. **`cpy_tensor_async`**: `drain_pending_compute()` を削除し、`g_device_compute_seq[src_device]` を deferred copy entry に埋め込み
4. **`execute_deferred_copies`**: バッファ検証後、`cudaSetDevice` の前に `pipeline_compute_cv_[wait_device].wait()` 挿入（session base offset 付き）
5. **`ggml-rdma.h`**: `RDMA_PROTO_PATCH_VERSION` 1→2
6. **`rdma_server`**: `session_seq_base_` + `active_connections_` によるセッション管理

### セッションベース問題の発見と修正

初期実装では `pipeline_compute_seq_` の絶対値を wait に使用していたが、サーバーの `pipeline_compute_seq_` は累積カウンターで前回セッションの値を保持する一方、クライアントの `g_device_compute_seq` は新プロセスごとに 0 にリセットされる。

**症状**: 2回目以降の接続で、サーバーの seq が既に高い値を保持しているため wait 条件が即座に通過し、ソースデバイスの compute 完了前にコピーが実行されゴミ出力となる。

**修正**: `init_session_base()` で最初の接続時に `pipeline_compute_seq_` のスナップショットを `session_seq_base_` として記録。`execute_deferred_copies` での wait 条件を `pipeline_compute_seq_[dev] >= (wait_expected_seq + session_seq_base_[dev])` に変更。`end_connection()` で最後の接続切断時にフラグリセット。

## 正確性テスト結果

| 構成 | PARALLEL=1 | 結果 |
|------|:----------:|:----:|
| 2C+2R | ✓ | 正常出力 |
| 3C+3R (1回目) | ✓ | 正常出力 |
| 3C+3R (2回目, warm server) | ✓ | 正常出力 |
| 4C+4R | ✓ | 正常出力 |

全構成で連続実行しても正常な推論出力を確認。

## 性能テスト

### 実験設計

- **方式**: ABAB Paired Design (10ペア)
- **条件A**: Baseline (PARALLEL なし、per-device connections のみ)
- **条件B**: `GGML_RDMA_PARALLEL=1` (pipeline + per-device connections)
- **モデル**: Qwen3.5-35B-A3B UD-Q4_K_M
- **構成**: 4C+4R (CUDA0-3 + RDMA0-3)
- **交絡チェック**: 同一バイナリ + 環境変数切替、ウォームアップ 1回破棄済み

### 生データ

| Pair | A pp128 | B pp128 | A tg32 | B tg32 |
|:----:|:-------:|:-------:|:------:|:------:|
| 1 | 199.13 | 199.38 | 34.17 | 34.84 |
| 2 | 199.14 | 199.77 | 34.10 | 35.07 |
| 3 | 199.32 | 200.05 | 34.23 | 35.06 |
| 4 | 198.29 | 199.60 | 34.10 | 35.06 |
| 5 | 199.18 | 199.85 | 34.43 | 35.15 |
| 6 | 199.29 | 199.41 | 34.29 | 35.15 |
| 7 | 199.46 | 200.01 | 34.24 | 35.14 |
| 8 | 199.07 | 199.04 | 34.29 | 35.12 |
| 9 | 199.61 | 199.84 | 34.08 | 35.10 |
| 10 | 199.61 | 199.91 | 34.29 | 35.02 |

### PP128 結果

| 指標 | 値 |
|------|:---:|
| Baseline 平均 | 199.21 ± 0.37 t/s |
| PARALLEL 平均 | 199.69 ± 0.31 t/s |
| 差分平均 | +0.47 t/s (+0.24%) |
| p 値 | 0.0037 |
| Cohen's d | 1.35 (大) |
| 95% CI | [+0.20, +0.75] t/s |
| 改善ペア | 9/10 (90%) |
| **判定** | **統計的に有意だが効果 < 0.5%** |

Qwen3.5 は compute-bound のため PP での改善は限定的。

### TG32 結果

| 指標 | 値 |
|------|:---:|
| Baseline 平均 | 34.22 ± 0.10 t/s |
| PARALLEL 平均 | 35.07 ± 0.09 t/s |
| 差分平均 | +0.85 t/s (+2.48%) |
| p 値 | < 0.001 |
| Cohen's d | 8.40 (大) |
| 95% CI | [+0.77, +0.93] t/s |
| 改善ペア | 10/10 (100%) |
| **判定** | **有効 (p < 0.05 かつ > 0.5%)** |

TG32 で **+2.48%** の改善。10/10 ペアで一貫して改善しており、サーバー側デバイス並列性の回復効果が確認できた。

## 再現方法

```bash
bash .worktree/pipeline-splits/scripts/rdma-build.sh local
bash scripts/gpu-lock.sh run bash .worktree/pipeline-splits/scripts/rdma-deploy.sh
ssh 192.168.100.2 "pkill -f rdma-server"; sleep 1
bash .worktree/pipeline-splits/scripts/rdma-server.sh start

GGML_RDMA_PARALLEL=1 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/gpu-lock.sh run \
  .worktree/pipeline-splits/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -fa 1 -r 5 -p 128 -n 32 -o csv
```

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `d0d1c678a (feature/pipeline-splits)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 34°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1389309) |

## 結論

- クライアント側 `drain_pending_compute` をサーバー側 per-device `pipeline_compute_cv_` 待ちに置き換え成功
- セッション間の `pipeline_compute_seq_` 累積問題を `session_seq_base_` で解決
- 3+ RDMA デバイスでの正確性を確認（2C+2R, 3C+3R, 4C+4R）
- TG32 で **+2.48%** の改善（Qwen3.5 4C+4R、p < 0.001）
- PP128 は compute-bound のため +0.24% で実質ニュートラル
- GLM-4.7 (communication-bound) ではより大きな効果が期待される
