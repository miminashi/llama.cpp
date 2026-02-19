# Hugepage + GDR バジェット拡大ベンチマーク

- **実施日時**: 2026年2月19日 16:00
- **ワークツリー**: `.worktree/hugepage-staging`
- **参照レポート**: [hugepage_staging_benchmark (GDR 未使用)](report/2026-02-19_104648_hugepage_staging_benchmark.md)

## 前提・目的

### 背景

前回の実験 (`2026-02-19_104648`) では `nvidia-peermem` がロードできず GDR が使えない状態で hugepage 単体の効果を検証し、有意な効果はなかった (tg: +0.29%, p=0.11)。

本実験では `nvidia-peermem` の問題を解決し (カーネルを 6.8.0-90-generic に戻してリブート)、GDR が動作する状態で hugepage + GDR バジェット拡大の効果を検証する。

### 仮説

- 4KB ページのステージングバッファ (~40GB) は ~1000万 MTT エントリを消費
- 2MB hugepage なら ~2万 MTT エントリに削減 (512分の1)
- MTT キャッシュの空きが増え、GDR バジェットを 12GB → 48GB に拡大可能
- より多くの GPU バッファが GDR パス (ゼロコピー RDMA) を使える

### 条件

| 条件 | Hugepage | GDR Budget | GDR GPU 数 | ステージング |
|:----:|:--------:|:----------:|:----------:|:----------:|
| **A** | ON (2MB) | 48 GB | 4/4 GPU | なし (全 GDR) |
| **B** | OFF (4KB) | 12 GB | 1/4 GPU | 3 GPU 分 (~30 GB) |

### 交絡因子チェック

- [x] **単一変数の分離**: 条件 A/B は環境変数 (`GGML_RDMA_NO_HUGEPAGE`, `GGML_RDMA_GDR_BUDGET_GB`) で切替。ただし 2 変数を同時に変更しているため、hugepage 単体の効果と GDR budget の効果は分離できない
- [x] **環境変数トグル**: 同一バイナリ (commit `c06f0d957`) + 環境変数で条件切替
- [x] **ホットパスのログ出力**: ホットパスに fprintf なし (前回の修正済みビルド)
- [x] **分布の単峰性**: パイロットラン (5回×2条件) で確認済み。pp: SD ≈ 0.35-0.38, tg: SD ≈ 0.16-0.20。> 0.5 t/s のギャップなし

## 再現方法

### ビルド・デプロイ

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/hugepage-staging/scripts/rdma-build.sh local
bash /home/ubuntu/projects/llama.cpp/.worktree/hugepage-staging/scripts/rdma-deploy.sh
```

### Hugepage 確保 (2号機)

```bash
ssh 192.168.100.2 "sudo sh -c 'echo 20480 > /proc/sys/vm/nr_hugepages'"
```

### サーバー起動

条件 A:
```bash
ssh 192.168.100.2 "GGML_RDMA_GDR_BUDGET_GB=48 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
```

条件 B:
```bash
ssh 192.168.100.2 "GGML_RDMA_NO_HUGEPAGE=1 GGML_RDMA_GDR_BUDGET_GB=12 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
```

### ベンチマーク (ABAB 交互実行, n=16 ペア, ペア 0 はウォームアップ破棄)

スクリプト: `/tmp/hugepage_gdr_abab_bench.sh`

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_GDR_BUDGET_GB=48 \
  LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/.worktree/hugepage-staging/build/bin \
  /home/ubuntu/projects/llama.cpp/.worktree/hugepage-staging/build/bin/llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

## 結果

### 重要な発見: GDR 有効化による大幅な pp 改善

GDR が初めて有効化された結果、**pp (Prompt Processing) が約 3.6 倍に改善**した:

| 状態 | pp128 (t/s) | tg32 (t/s) |
|------|:-----------:|:----------:|
| GDR 無効 (前回実験) | ~6.4 | ~7.7 |
| GDR 有効 budget=12 (1 GPU GDR) | ~23.3 | ~7.7 |
| GDR 有効 budget=48 (4 GPU GDR) | ~23.2 | ~7.8 |

この pp 改善は hugepage/budget の条件とは無関係に発生している。GDR が 1 GPU でも有効であれば pp が大幅に改善される。

### ABAB ベンチマーク生データ (ペア 1-15, ウォームアップ除外)

| Pair | A pp128 | B pp128 | diff pp | A tg32 | B tg32 | diff tg |
|:----:|:-------:|:-------:|:-------:|:------:|:------:|:-------:|
| 1 | 23.29 | 23.18 | +0.11 | 7.69 | 7.69 | +0.00 |
| 2 | 23.27 | 23.28 | -0.01 | 7.78 | 7.77 | +0.01 |
| 3 | 23.32 | 23.31 | +0.01 | 7.72 | 7.67 | +0.05 |
| 4 | 23.31 | 23.34 | -0.03 | 7.72 | 7.71 | +0.01 |
| 5 | 23.09 | 23.20 | -0.11 | 7.78 | 7.78 | +0.00 |
| 6 | 23.32 | 23.33 | -0.01 | 7.77 | 7.72 | +0.05 |
| 7 | 23.22 | 23.24 | -0.02 | 7.77 | 7.70 | +0.07 |
| 8 | 23.19 | 23.34 | -0.15 | 7.80 | 7.78 | +0.02 |
| 9 | 23.22 | 23.29 | -0.07 | 7.76 | 7.79 | -0.03 |
| 10 | 23.28 | 23.30 | -0.02 | 7.68 | 7.77 | -0.09 |
| 11 | 23.11 | 23.31 | -0.20 | 7.80 | 7.70 | +0.10 |
| 12 | 23.21 | 23.31 | -0.10 | 7.72 | 7.71 | +0.01 |
| 13 | 23.13 | 23.27 | -0.14 | 7.77 | 7.72 | +0.05 |
| 14 | 23.25 | 23.34 | -0.09 | 7.71 | 7.72 | -0.01 |
| 15 | 23.20 | 23.32 | -0.12 | 7.80 | 7.69 | +0.11 |

### 記述統計

| 指標 | 条件 A (hugepage+budget48) | 条件 B (no-hugepage+budget12) |
|------|:-:|:-:|
| pp128 平均 ± SD | 23.227 ± 0.074 | 23.291 ± 0.050 |
| tg32 平均 ± SD | 7.751 ± 0.041 | 7.728 ± 0.039 |

### 検定結果: pp128 (Prompt Processing)

| 指標 | 値 |
|------|:---:|
| 差分平均 | -0.063 t/s (-0.27%) |
| t(14) | -3.138 |
| p 値 | 0.0017 |
| Cohen's d | -0.810 (large) |
| 95% CI | [-0.107, -0.020] t/s |
| A > B のペア数 | 2/15 (13%) |

**判定: NOT SIGNIFICANT** — p < 0.05 だが効果 < 0.5% (判定基準は p < 0.05 **かつ** 効果 > 0.5%)。条件 A がわずかに **遅い** 方向であり、仮説と逆の結果。

### 検定結果: tg32 (Token Generation)

| 指標 | 値 |
|------|:---:|
| 差分平均 | +0.023 t/s (+0.30%) |
| t(14) | 1.779 |
| p 値 | 0.075 |
| Cohen's d | 0.459 (small) |
| 95% CI | [-0.005, +0.051] t/s |
| A > B のペア数 | 10/15 (67%) |

**判定: NOT SIGNIFICANT** — p ≥ 0.05 かつ効果 < 0.5%。

### 外れ値

- pp128: IQR=0.110, フェンス [-0.285, +0.155] → 外れ値 0 件
- tg32: IQR=0.050, フェンス [-0.075, +0.125] → 外れ値 1 件 (pair 10: -0.09)、除外せず

## 考察

### 仮説は棄却: hugepage + GDR バジェット拡大の効果は認められない

- **pp128**: 条件 A (全 4 GPU GDR) が条件 B (1 GPU GDR) より -0.27% 遅いという逆方向の結果。budget=48 で 40.4 GB の GPU MR を登録する overhead (RNIC の MTT エントリ追加) が、ステージング不要のメリットを上回った可能性がある
- **tg32**: +0.30% の微小改善だが統計的に有意でない

### GDR 有効化自体は大きな効果

本実験の副産物として、GDR 有効化 (nvidia-peermem ロード) が pp を 6.4 → 23.3 t/s に改善することを確認した。これは前回の GDR 無効実験 (report/2026-02-19_104648) との比較から明らか。ただし:

- GDR budget=12 (1 GPU のみ GDR) でも budget=48 (4 GPU 全 GDR) でも pp はほぼ同じ (~23.3)
- つまり、**GDR は最初の 1 GPU でほぼ全ての効果を発揮**し、追加 GPU の GDR 化は性能に寄与しない
- 推定原因: pp (バッチ処理) のボトルネックは GPU 計算であり、weight 転送パスの GDR/非 GDR の差は無視できる

### 条件 A で pp がわずかに遅い理由の推測

Budget=48 では 40.4 GB の GPU VRAM を RDMA MR として登録する。これは ConnectX-4 の MTT (Memory Translation Table) に ~1000万エントリを追加し、MTT キャッシュミスが増える可能性がある。一方 budget=12 では 10.9 GB (1 GPU 分) のみで MTT 負荷が小さい。ステージングバッファ (~30 GB) は hugepage なしでも連続メモリ領域なのでアドレス変換は比較的効率的。結果として、budget=48 の方が RNIC の MTT キャッシュ効率が悪く、微小な性能低下が発生した可能性がある。

### 前回実験との比較

| 実験 | GDR 状態 | pp128 (t/s) | tg32 (t/s) |
|------|---------|:-----------:|:----------:|
| 前回 (2/19 AM) | GDR 無効 (peermem 未ロード) | ~6.3 | ~7.7 |
| 今回 条件 B | GDR 有効 (budget=12, 1 GPU) | 23.29 | 7.73 |
| 今回 条件 A | GDR 有効 (budget=48, 4 GPU) | 23.23 | 7.75 |

## 結論

1. **Hugepage + GDR バジェット拡大 (12→48GB) の効果はない** (pp: -0.27% 逆方向, tg: +0.30% 有意差なし)
2. **GDR 有効化自体は pp を約 3.6 倍に改善** (6.4 → 23.3 t/s)
3. **GDR は 1 GPU (budget=12) で十分** — 追加 GPU の GDR 化による性能改善は認められない
4. Hugepage はステージングバッファの MTT エントリ削減には有効だが、性能改善には繋がらない (ConnectX-4 環境)

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c06f0d957 (feature/hugepage-staging)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 36°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| カーネル | 6.8.0-90-generic | 6.8.0-90-generic |
| Hugepage (2号機) | — | 20480 × 2MB = 40 GB |

### GDR MR 割り当て (サーバーログ)

条件 A (budget=48):
```
GDR MR registered: size=10.9 GB, total=10.9/48.0 GB
GDR MR registered: size=9.7 GB, total=20.6/48.0 GB
GDR MR registered: size=11.2 GB, total=31.8/48.0 GB
GDR MR registered: size=8.6 GB, total=40.4/48.0 GB
```
→ 全 4 GPU が GDR パス (40.4/48.0 GB)

条件 B (budget=12):
```
GDR MR registered: size=10.9 GB, total=10.9/12.0 GB
GDR budget exceeded (need 9.7 GB, used 10.9/12.0 GB), using staging
GDR budget exceeded (need 11.2 GB, used 10.9/12.0 GB), using staging
GDR budget exceeded (need 8.6 GB, used 10.9/12.0 GB), using staging
```
→ 1 GPU のみ GDR、3 GPU はホストステージング
