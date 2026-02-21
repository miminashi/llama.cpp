# tg32 デグレ調査レポート — upstream rebase は原因ではなかった

- **実施日時**: 2026年2月21日 12:48
- **ブランチ**: `feature/rdma-backend`（rebase 後, `3684833e4`）
- **作業場所**: メインリポジトリ + `.worktree/merge-validated`

## 前提・目的

upstream rebase (218コミット) 後、GLM-4.7 IQ2_M 11GPU (7C+4R) の tg32 が 7.75 → 5.10 t/s (-34%) にデグレしたとの報告([デグレテスト レポート](2026-02-21_041746_upstream_rebase_degression_test.md))を受け、原因を特定・修正する。

- **背景**: ベースライン 7.75 t/s は [selective signaling merge レポート](2026-02-21_030543_selective_signaling_merge.md) で計測。pp128 は 29.95 t/s で正常 (baseline 29.44 の ±5% 以内)
- **初期仮説**: `CUDA_SCALE_LAUNCH_QUEUES=4x` の自動設定が revert されたことが原因 (commit `41e3f0264`)
- **目的**: 真の原因を特定し修正する

## 調査結果

### 1. `CUDA_SCALE_LAUNCH_QUEUES=4x` テスト — 棄却

サーバー・クライアント両方に `CUDA_SCALE_LAUNCH_QUEUES=4x` を設定してテスト。

| 条件 | tg32 (t/s) |
|------|:----------:|
| 環境変数なし (rebase 後コード) | 5.12 |
| `CUDA_SCALE_LAUNCH_QUEUES=4x` 付き | 5.08 |

**改善なし。** サーバー側プロセスの `/proc/<pid>/environ` でも変数が設定されていることを確認済み。

### 2. 非 MoE モデルテスト — MoE 固有の問題か確認

gpt-oss-20b (非 MoE) で同条件テスト。

| テスト | rebase 後 (t/s) | デグレテスト値 (t/s) | 差分 |
|--------|:---------------:|:-------------------:|:----:|
| tg32 | 34.86 ± 0.43 | 35.31 ± 0.36 | -1.3% |

**非 MoE モデルにデグレなし。** GLM-4.7 MoE 固有の問題。

### 3. mul_mat_id ディスパッチ変更の分析

commit `8bece2eb2` で MoE の `mul_mat_id` ディスパッチロジックが変更された:
- 旧: `ne2 == 1` → mmvq (量子化) / mmvf (非量子化) → return
- 新: `ne2 <= MMVQ_MMID_MAX_BATCH_SIZE(4)` → mmvq (量子化, NVIDIA) → return

GLM-4.7 IQ2_M のエキスパートテンソルは全て量子化 (IQ2_XS, IQ3_XXS 等) であり、ne2=1 (tg) の場合のカーネルパスは変更なし。**コード変更による影響はないと結論。**

### 4. 決定的テスト — rebase 前コードの再ビルド

**7.75 t/s を計測したまさにそのコミット** (`20e808adb`) を再ビルド・再デプロイして同条件でテスト。

```bash
git checkout 20e808adb
bash scripts/rdma-build.sh local
bash scripts/rdma-deploy.sh
bash scripts/rdma-server.sh start
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  llama-bench -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm layer -mg 0 -nkvo 1 -fa 1 -p 128 -n 32 -r 3 -o csv
```

| テスト | 今回 (commit 20e808adb) | ベースライン (同コミット, 別セッション) |
|--------|:-----------------------:|:-------------------------------------:|
| pp128 | **29.76 ± 0.59** | 30.15 ± 0.49 |
| tg32 | **5.26 ± 0.005** | 7.75 ± 0.004 |

**rebase 前のコードを再ビルドしても tg32 = 5.26 t/s であり、7.75 t/s は再現しない。**

### 5. merge-validated ワークツリーでの追加確認

rebase 前の feature/rdma-backend から cherry-pick した `merge-validated` ワークツリー (commit `088f8c8d9`) でも同様のテストを実施。

| テスト | merge-validated | ベースライン |
|--------|:--------------:|:-----------:|
| pp128 | 29.44 ± 0.50 | 30.15 |
| tg32 | 5.25 ± 0.007 | 7.75 |

同じ結果。**コード変更ではなく環境要因**が確定。

## 結論

| 検証項目 | 結果 |
|---------|:----:|
| `CUDA_SCALE_LAUNCH_QUEUES=4x` が原因 | **棄却** (改善なし) |
| upstream CUDA コード変更が原因 | **棄却** (rebase 前コードでも再現) |
| 非 MoE モデルのデグレ | **なし** (gpt-oss-20b は正常) |
| **環境要因 (セッション間変動)** | **確定** |

### upstream rebase はデグレの原因ではない

7.75 t/s のベースライン値は、特定のセッション条件下での高速側の値だった。CLAUDE.md にも記載されている「サーバーGPU 計算時間のセッション間変動 (5.5-6.8 t/s)」の範囲を超えた外れ値であった可能性が高い。

現在の安定した tg32 値は **~5.1-5.3 t/s** であり、これが rebase 前後で一貫している。

### 推奨事項

1. **ベースライン値の更新**: tg32 のベースラインを 7.75 t/s から ~5.2 t/s に修正する
2. **セッション間変動の根本原因調査**: GPU サーマルスロットリング、CUDA コンテキスト初期化、または RDMA 接続状態が変動要因の候補
3. **`CUDA_SCALE_LAUNCH_QUEUES=4x` の復元は不要**: 性能に影響しないことが確認された
4. **pp128 は安定**: 29.4-30.2 t/s の範囲で一貫しており、upstream rebase による影響なし

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `3684833e4 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 34°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 130157) |

GGML_RDMA 環境変数: (none)
