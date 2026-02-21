# tg32 デグレ深層調査レポート — `-nkvo 1` フラグが原因

- **実施日時**: 2026年2月21日 21:08
- **ワークツリー**: `.worktree/tg32-debug`
- **ブランチ**: `feature/rdma-backend` (コミット: `3684833e4`)

## 前提・目的

upstream rebase (218コミット) 後、GLM-4.7 IQ2_M 11GPU (7C+4R) の **tg32 が 7.75 → 5.10 t/s (-34%)** にデグレした。pp128 は正常 (29.95 vs baseline 29.44)。

- **背景**: upstream rebase degression test ([report/2026-02-21_041746_upstream_rebase_degression_test.md](2026-02-21_041746_upstream_rebase_degression_test.md)) で tg32 デグレを検出
- **目的**: 根本原因を特定し、tg32 ≧ 7.0 t/s への回復を目指す
- **前提条件**: 11GPU (7CUDA + 4RDMA) 構成、GLM-4.7 IQ2_M モデル

### 参照レポート

- [Selective Signaling マージレポート](2026-02-21_030543_selective_signaling_merge.md) — tg32 = 7.75 のベースライン
- [Upstream Rebase デグレテスト](2026-02-21_041746_upstream_rebase_degression_test.md) — デグレ検出 (tg32 = 5.10)
- [二峰性 tg 調査](2026-02-13_153100_bimodal_tg_investigation.md) — 過去の tg 変動調査
- [A/B ベンチマーク](2026-02-20_132916_merge_validated_improvements_benchmark.md) — tg32 = 7.70-7.78 のベースライン

## 結論 (先出し)

**デグレの根本原因は upstream rebase 自体ではなく、degression test で新たに追加された `-nkvo 1` ベンチマークフラグ**。

`-nkvo 1` (no KV offload) は KV キャッシュを GPU ではなく CPU メモリに配置するため、generation の各トークンで大量の Host→Device memcpy が発生し、tg32 が -31% 劣化していた。

7.75 t/s を記録した selective signaling merge テストでは `-nkvo` フラグは使用されていなかった（KV キャッシュは GPU 上）。

| フラグ | pp128 (t/s) | tg32 (t/s) | tg32 vs baseline |
|--------|:-----------:|:----------:|:----------------:|
| フラグなし (ベースライン再現) | 30.44 | **7.72** | -0.4% |
| `-fa 1` のみ | 30.62 | **8.52** | **+9.9%** |
| `-nkvo 1 -fa 1` (デグレ条件) | 29.75 | **5.31** | **-31.5%** |

## 調査過程

### セッション 1 (前回コンテキスト): コード監査と初期テスト

前回セッションで以下の仮説を検証・棄却:

| 仮説 | 結果 |
|------|------|
| CUDA_SCALE_LAUNCH_QUEUES=4x 削除 | 棄却 — 環境変数追加しても改善なし |
| upstream CUDA コード変更 | 棄却 — rebase 前コード (20e808adb) でも tg32 = 5.26 |
| `mul_mat_id` ディスパッチ変更 | 棄却 — ne2=1 のカーネルパスは変更なし |
| 非 MoE モデルのデグレ | なし — gpt-oss-20b は正常 |
| send buffer DMA race (正当性バグ) | 修正済み — ただし性能には影響なし |
| CUDA graph ログスパム | 修正済み — ただし性能には影響なし |

### セッション 2 (今回): 系統的排除法

全テストで一貫して tg32 ≈ 5.30 ± 0.05 t/s:

| テスト | pp128 | tg32 | 効果 |
|--------|:-----:|:----:|:----:|
| ベースライン (-nkvo 1 -fa 1) | 29.75 | 5.31 | — |
| stderr → /dev/null (両ノード) | 30.11 | 5.31 | なし |
| SS 無効 (GGML_RDMA_NO_SELECTIVE_SIGNAL=1) | 23.58 | 5.32 | なし |
| 1328 MHz + Persistence Mode ON | 30.05 | 5.29 | なし |
| Pre-rebase コード (merge-validated, 088f8c8d9) | 29.75 | 5.31 | なし |
| CUDA graphs OFF (-DGGML_CUDA_GRAPHS=OFF) | 30.03 | 5.32 | なし |
| GDR 無効 (GGML_RDMA_NO_GDR=1) | 29.81 | 5.33 | なし |
| CPU governor=performance (3.2-3.5 GHz) | 29.82 | 5.34 | なし |
| Page cache + swap フラッシュ | 29.66 | 5.35 | なし |
| tg128 | — | 5.29 | 一定 (トークン数依存なし) |

### 転機: nvprof プロファイリング

nvprof で CUDA 操作の内訳を取得:

**GPU activities (tg32, 32トークン):**
```
CUDA memcpy HtoD:  78.50%  8.998秒  79,825回  avg 112.72us
compute kernels:   21.50%  2.462秒
```

n=2 vs n=32 の差分で推論中の HtoD を切り分け:

| メトリック | n=2 | n=32 | 差分 (30トークン) | per token |
|-----------|:---:|:----:|:-----------------:|:---------:|
| HtoD calls | 75,535 | 79,825 | +4,290 | 143/tok |
| HtoD time | 8.69s | 9.00s | +310ms | ~10ms/tok |
| StreamSync calls | 2,001 | 21,801 | +19,800 | 660/tok |
| StreamSync time | 205ms | 2,287ms | +2,082ms | ~69ms/tok |

HtoD の 99% はモデルロード時 (75,535回)。推論中は 143回/token (10ms/tok) — 正常。

**cudaStreamSynchronize が 69ms/token** — これが per-token overhead の主体。KV キャッシュが CPU にあるため、各 attention 計算で HtoD/DtoH 同期が大量発生。

### 根本原因の特定

過去のベンチマークコマンドを比較:

**7.75 t/s の selective signaling test (2/21 03:05):**
```bash
llama-bench ... -ngl 999 -sm layer -r 3 -p 128 -n 32 -o csv
```

**5.10 t/s の degression test (2/21 04:17):**
```bash
llama-bench ... -ngl 999 -sm layer -mg 0 -nkvo 1 -fa 1 -r 5 -o csv
```

**差分: `-nkvo 1 -fa 1 -mg 0` の追加**。

degression test のレポートに以下の注記あり:
> `-nkvo` → `-nkvo 1`（以前は単独フラグとして使用可能だった）

upstream rebase で CLI 引数の構文が変更されたことを受けて `-nkvo 1` が追加されたが、ベースライン (7.75) では **`-nkvo` は使用されていなかった** (KV キャッシュは GPU 上)。

### 検証: `-nkvo 1` 除去で回復確認

| フラグ | pp128 (t/s) | tg32 (t/s) | 判定 |
|--------|:-----------:|:----------:|:----:|
| フラグなし (ベースライン条件を再現) | 30.44 | **7.72** ± 0.004 | PASS (≧ 7.0) |
| `-fa 1` のみ (flash attention) | 30.62 | **8.52** ± 0.003 | PASS (+10%) |
| `-nkvo 1 -fa 1` (デグレ条件) | 29.75 | **5.31** ± 0.004 | FAIL |

## `-nkvo 1` の影響メカニズム

```
┌─ KV キャッシュ on GPU (default, no -nkvo 1) ─┐
│  GPU 内メモリアクセス → 高帯域 (~700 GB/s)      │
│  Attention 計算が GPU 上で完結                    │
│  tg32 = 7.72-8.52 t/s                            │
└──────────────────────────────────────────────────┘

┌─ KV キャッシュ on CPU (-nkvo 1) ──────────────┐
│  PCIe 経由 HtoD/DtoH (~12 GB/s)                  │
│  各トークンの Attention で KV 転送が発生          │
│  143 HtoD calls/token × 11GPU = ~1573 calls/tok  │
│  + 660 cudaStreamSynchronize/token                │
│  tg32 = 5.31 t/s (-31%)                          │
└──────────────────────────────────────────────────┘
```

## 副次的発見

### 1. Flash Attention の tg 改善効果

`-fa 1` (flash attention) は tg32 を +10% 改善 (7.72 → 8.52 t/s)。pp128 への影響はほぼなし (+0.6%)。

### 2. CUDA Graph 管理コードの影響なし

P100 (cc 6.0) では CUDA graphs が無効だが、管理コード (`USE_CUDA_GRAPH`) は常にコンパイルされている。`-DGGML_CUDA_GRAPHS=OFF` でビルドしても tg32 に変化なし。

### 3. 棄却済み環境要因

以下の全てが tg32 に影響なしと確認:
- GPU クロック (1189 vs 1328 MHz)
- GPU Persistence Mode (ON/OFF)
- CPU governor (schedutil vs performance)
- GDR (GPUDirect RDMA) の有無
- Selective signaling の有無
- stderr リダイレクト先 (/dev/null vs ファイル)
- ページキャッシュ・swap 状態
- RDMA プロファイリングオーバーヘッド

### 4. OOM Killer イベント

調査中に node 1 で python3 プロセス (91GB RSS) が OOM kill された。これは調査結果に直接影響しなかったが、dmesg に `vmstat_update hogged CPU for >10000us` の警告が記録されていた。

## 推奨事項

1. **ベンチマーク手順書の更新**: GLM-4.7 のような大規模モデルでは `-nkvo 1` を使用しないこと。11GPU × 16GB = 176GB の VRAM に 114GB のモデルと KV キャッシュの両方が収まる
2. **Flash Attention の採用**: `-fa 1` を標準ベンチマークフラグに含める (+10% tg 改善)
3. **ベースラインの更新**: `-fa 1` を含む条件での新しいベースライン — pp128 ≈ 30.6, tg32 ≈ 8.5

## 再現方法

### デグレの再現 (tg32 ≈ 5.3 t/s)

```bash
bash scripts/rdma-server.sh start
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  llama-bench -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm layer -mg 0 -nkvo 1 -fa 1 -p 128 -n 32 -r 3 -o csv
```

### 回復の再現 (tg32 ≈ 7.7 t/s)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  llama-bench -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm layer -mg 0 -p 128 -n 32 -r 3 -o csv
```

### Flash Attention 込み (tg32 ≈ 8.5 t/s)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  llama-bench -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm layer -mg 0 -fa 1 -p 128 -n 32 -r 3 -o csv
```

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `3684833e4 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 39°C | 40°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 189323) |

GGML_RDMA 環境変数: (none)

**注**: 調査中に以下の GPU 設定を変更しました (調査完了後も維持):
- SM application clocks: 1189 → 1328 MHz (両ノード)
- Persistence Mode: Disabled → Enabled (両ノード)
- CPU governor: schedutil → performance → schedutil (元に戻し済み)
