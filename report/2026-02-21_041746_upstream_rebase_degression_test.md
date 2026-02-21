# Upstream Rebase + デグレテスト レポート

- **実施日時**: 2026年2月21日 04:17
- **ブランチ**: `feature/rdma-backend`（rebase 後）

## 前提・目的

`feature/rdma-backend` ブランチを upstream (`ggml-org/llama.cpp` master) の最新に追従させ、RDMA バックエンドにデグレッションがないことを検証する。

- **背景**: upstream/master に 218 コミットの新規変更があり、feature ブランチが古くなっていた
- **目的**: rebase により最新の upstream に追従し、ビルドとベンチマークでデグレを検出する
- **前提条件**: feature/rdma-backend に 60 コミット、RDMA バックエンドは自己完結型実装

### ベースライン数値（rebase 前、selective signaling merge レポートより）

| テスト | ベースライン (t/s) | モデル |
|--------|:------------------:|--------|
| pp128 | 29.44 - 30.15 | GLM-4.7 IQ2_M, 11GPU (7C+4R) |
| tg32 | 7.75 | GLM-4.7 IQ2_M, 11GPU (7C+4R) |

参照: [selective_signaling_merge.md](2026-02-21_030543_selective_signaling_merge.md)

## 実施内容

### 1. Rebase

```bash
git fetch upstream
git checkout master
git merge --ff-only upstream/master    # 218コミット fast-forward
git checkout feature/rdma-backend
git rebase master                      # 60コミット
```

**結果: 衝突ゼロで全 60 コミットが正常に rebase 完了。**

| 項目 | 値 |
|------|:--:|
| upstream 新規コミット | 218 |
| feature ブランチコミット | 60 |
| 衝突ファイル数 | **0** |
| 旧 master HEAD | `41ea26144` |
| 新 master HEAD | `b908baf18` |
| 新 feature HEAD | `3684833e4` |

### 2. ビルド検証

```bash
bash scripts/rdma-build.sh local    # 1号機ビルド
bash scripts/rdma-deploy.sh         # 2号機デプロイ
```

**結果: 両ノードともビルド成功。コンパイルエラーなし。**

### 3. 環境チェック

```bash
bash scripts/rdma-env-check.sh
```

**結果: All checks passed. 警告なし。**

### 4. デグレテスト

#### 4a. GLM-4.7 UD-IQ2_M 11GPU (7C+4R)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  llama-bench -m .../GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm layer -mg 0 -nkvo 1 -fa 1 -r 5 -o csv
```

| テスト | Rebase 後 (t/s) | ±SD | ベースライン (t/s) | 差分 | 判定 |
|--------|:---------------:|:---:|:------------------:|:----:|:----:|
| **pp128** | **29.95** | 0.36 | 29.44 - 30.15 | **+1.7%** | **PASS** |
| **tg32** | **5.10** | 0.01 | 7.75 | **-34.2%** | **FAIL** |

- pp128 は正常範囲内（ベースラインの±5%以内）
- **tg32 に -34% の大幅デグレを検出**

#### 4b. tg32 デグレ原因の切り分け

| 試行 | 条件 | tg32 (t/s) | 結果 |
|------|------|:----------:|:----:|
| 1回目 | デフォルト | 5.14 | 低速 |
| 2回目 | サーバー再起動後 | 5.10 | 再現 |
| 3回目 | `-nopo 1` (op offload 無効) | 5.04 | 改善なし |

- サーバー再起動しても再現 → セッション変動ではない
- `-nopo 1`（upstream 新機能の無効化）でも改善なし → op offload は原因ではない
- pp128 は正常 → RDMA 転送パスは問題なし
- サーバーログに `ggml_cuda_graph_set_enabled: disabling CUDA graphs due to GPU architecture` が大量出力（upstream で追加された CUDA Graph 機能、P100 未対応のため毎回無効化）

#### 4c. gpt-oss-20b 11GPU（参考）

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  llama-bench -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -mg 0 -nkvo 1 -fa 1 -r 3 -o csv
```

| テスト | 結果 (t/s) | ±SD |
|--------|:----------:|:---:|
| pp128 | 383.10 | 7.48 |
| tg32 | 35.31 | 0.36 |

gpt-oss-20b は RDMA 動作正常。ただし rebase 前のベースライン数値がないため直接比較はできない。

### 注意事項: llama-bench 引数変更

upstream の変更により、以下のフラグが値引数を必要とするようになった:
- `-nkvo` → `-nkvo 1`（以前は単独フラグとして使用可能だった）
- `-fa` → `-fa 1`（変更なし、元から値引数）
- `-nopo <0|1>` — 新規追加オプション

## 分析

### tg32 デグレの推定原因

tg32（1トークンずつの逐次生成）のみが -34% 低下し、pp128（バッチ処理）は正常であることから、upstream 218コミットの中に **MoE モデルの逐次生成パスに影響する変更** が含まれていると推定される。

候補:
1. **CUDA Graph 試行オーバーヘッド** — 毎回の `ggml_cuda_graph_set_enabled` 呼び出しがログ出力を含む（P100 では常に無効化されるが、チェック自体のコストが蓄積）
2. **MoE ルーティング/エキスパートディスパッチの変更** — GLM-4.7 は 355B MoE (32B active) であり、MoE 固有のコード変更に敏感
3. **CUDA バックエンドの計算グラフ構造変更** — tg パスの per-token オーバーヘッドに影響

**gpt-oss-20b（非MoE）では問題が見えないため、MoE 固有の変更が原因である可能性が高い。**

### 次のアクション

1. **upstream の MoE 関連変更を特定** — `git log --oneline master~218..master` で MoE/expert 関連のコミットを検索
2. **bisect** — upstream の 218 コミットを二分探索して原因コミットを特定
3. **CUDA Graph ログ抑制の検討** — P100 で毎回出力されるログを抑制してオーバーヘッドを確認

## 結論

| 検証項目 | 結果 |
|---------|:----:|
| Rebase 完了（衝突なし） | **PASS** |
| ビルド成功（両ノード） | **PASS** |
| 環境チェック（警告なし） | **PASS** |
| pp128 ベースライン±5%以内 | **PASS** (+1.7%) |
| tg32 ベースライン±5%以内 | **FAIL** (-34.2%) |

**rebase 自体は問題なく完了したが、upstream の変更により GLM-4.7 MoE の tg32 に -34% のデグレが発生している。** RDMA バックエンド固有の問題ではなく、upstream の CUDA/ggml 変更に起因すると推定。原因特定のために bisect が推奨される。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `3684833e4 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 35°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 104722) |

GGML_RDMA 環境変数:
- (none)
