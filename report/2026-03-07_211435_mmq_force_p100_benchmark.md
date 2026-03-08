# MMQ 強制有効化による Qwen3.5 PP 性能評価

- **実施日時**: 2026年3月7日 21:14
- **ワークツリー**: `.worktree/mmq-force` (branch: `feature/mmq-force`)

## 前提・目的

Qwen3.5-35B-A3B (MoE) の PP 性能が P100 (cc 6.0) の `mul_mat_id` フォールバックパスで構造的に制限されている可能性を検証する。このフォールバックは ubatch あたり 80-120 回の `cudaStreamSynchronize` を伴い、GPU パイプラインを破壊する。MMQ パスが有効なら GPU 側で expert sort を完結させ、stream sync を排除できる。

- **問題**: `mmq.cu:304-306` の DP4A チェックが `GGML_CUDA_FORCE_MMQ` チェックより先に評価されるため、P100 では FORCE_MMQ が効かない
- **仮説**: MMQ 強制有効化により stream sync 排除 → PP 性能改善
- **リスク**: P100 の dp4a ソフトウェアエミュレーション（4 × int8 乗加算）のコストが stream sync 排除の利得を相殺する可能性

## 変更内容

`ggml/src/ggml-cuda/mmq.cu` の `ggml_cuda_should_use_mmq` 関数で `GGML_CUDA_FORCE_MMQ` チェックを DP4A アーキテクチャチェックより前に移動:

```cpp
// Before: DP4A check blocks FORCE_MMQ on P100
if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
    return false;  // P100 (cc 6.0) はここで return
}
#ifdef GGML_CUDA_FORCE_MMQ
    return true;  // 到達不可能
#endif

// After: FORCE_MMQ が DP4A check を迂回
#ifdef GGML_CUDA_FORCE_MMQ
    return true;  // P100 でも MMQ 有効
#endif
if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
    return false;
}
```

ビルドオプションに `-DGGML_CUDA_FORCE_MMQ=ON` を追加。

## 再現方法

1. ワークツリー作成
   ```bash
   git worktree add -b feature/mmq-force .worktree/mmq-force feature/rdma-backend
   ```

2. `mmq.cu` の修正（上記参照）

3. `scripts/rdma-build.sh` の `CMAKE_OPTS` に `-DGGML_CUDA_FORCE_MMQ=ON` を追加

4. ビルド + デプロイ
   ```bash
   bash .worktree/mmq-force/scripts/rdma-build.sh local
   bash .worktree/mmq-force/scripts/rdma-deploy.sh
   ```

5. ABAB ベンチマーク実行（スクリプト: `/tmp/mmq-abab-bench.sh`）
   - A: ベースライン（FORCE_MMQ なし）、B: MMQ 強制有効
   - Node 2 に両方のサーバーバイナリを事前配置し、切り替え時にサーバー再起動

## 実験条件

- **モデル**: Qwen3.5-35B-A3B-Q4_K_M (MoE, 3B active params)
- **GPU 構成**: Node 1 CUDA4,5 + Node 2 RDMA0,1 (4GPU)
- **テスト条件**: pp128, pp512, pp2048, tg32
- **反復**: ABAB × 8 pairs (ウォームアップ 1 回破棄)
- **Flash attention**: 有効 (`-fa 1`)

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `e674c4209 (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 31°C | 30°C |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |

## 結果

### 生データ

| Pair | pp128 A | pp128 B | pp512 A | pp512 B | pp2048 A | pp2048 B | tg32 A | tg32 B |
|:----:|--------:|--------:|--------:|--------:|---------:|---------:|-------:|-------:|
| 1 | 215.64 | 112.28 | 337.87 | 225.35 | 352.42 | 237.32 | 34.71 | 34.50 |
| 2 | 215.51 | 112.30 | 338.95 | 225.04 | 350.63 | 237.24 | 33.92 | 33.66 |
| 3 | 216.21 | 112.59 | 339.47 | 225.24 | 354.03 | 236.08 | 34.68 | 34.67 |
| 4 | 215.65 | 112.62 | 338.37 | 224.88 | 351.96 | 237.21 | 34.21 | 34.53 |
| 5 | 216.52 | 111.92 | 340.33 | 224.12 | 354.97 | 236.07 | 33.70 | 34.08 |
| 6 | 216.71 | 112.49 | 339.39 | 225.19 | 355.54 | 237.28 | 34.66 | 34.61 |
| 7 | 215.80 | 112.53 | 339.70 | 225.23 | 355.11 | 237.02 | 33.49 | 34.68 |
| 8 | 216.17 | 112.36 | 341.27 | 224.28 | 355.13 | 235.24 | 34.62 | 34.78 |

### 統計分析

#### pp128

| 指標 | 値 |
|------|:---:|
| A (baseline) 平均 ± SD | 216.026 ± 0.443 t/s |
| B (MMQ) 平均 ± SD | 112.385 ± 0.229 t/s |
| 差分平均 | **-103.641 t/s (-47.98%)** |
| t(7) | -540.46 |
| p 値 | 1.96 × 10⁻¹⁷ |
| Cohen's d | -191.08 (極大) |
| 95% CI | [-104.095, -103.188] t/s |
| 全ペア負の効果 | 8/8 (100%) |

#### pp512

| 指標 | 値 |
|------|:---:|
| A (baseline) 平均 ± SD | 339.419 ± 1.074 t/s |
| B (MMQ) 平均 ± SD | 224.918 ± 0.465 t/s |
| 差分平均 | **-114.501 t/s (-33.73%)** |
| t(7) | -224.29 |
| p 値 | 9.25 × 10⁻¹⁵ |
| Cohen's d | -79.30 (極大) |
| 95% CI | [-115.708, -113.294] t/s |
| 全ペア負の効果 | 8/8 (100%) |

#### pp2048

| 指標 | 値 |
|------|:---:|
| A (baseline) 平均 ± SD | 353.724 ± 1.821 t/s |
| B (MMQ) 平均 ± SD | 236.684 ± 0.781 t/s |
| 差分平均 | **-117.040 t/s (-33.09%)** |
| t(7) | -143.45 |
| p 値 | 2.11 × 10⁻¹³ |
| Cohen's d | -50.72 (極大) |
| 95% CI | [-118.969, -115.110] t/s |
| 全ペア負の効果 | 8/8 (100%) |

#### tg32

| 指標 | 値 |
|------|:---:|
| A (baseline) 平均 ± SD | 34.249 ± 0.493 t/s |
| B (MMQ) 平均 ± SD | 34.437 ± 0.379 t/s |
| 差分平均 | +0.189 t/s (+0.55%) |
| t(7) | 1.14 |
| p 値 | 0.291 |
| Cohen's d | 0.40 (小～中) |
| 95% CI | [-0.202, +0.579] t/s |
| 正の効果ペア | 4/8 (50%) |

### サマリー

| 条件 | A (baseline) | B (MMQ) | 差分 | p 値 | 判定 |
|------|:------------:|:-------:|:----:|:----:|:----:|
| pp128 | 216.0 | 112.4 | **-48.0%** | < 0.001 | **大幅退行** |
| pp512 | 339.4 | 224.9 | **-33.7%** | < 0.001 | **大幅退行** |
| pp2048 | 353.7 | 236.7 | **-33.1%** | < 0.001 | **大幅退行** |
| tg32 | 34.2 | 34.4 | +0.6% | 0.291 | 変化なし |

## 分析・考察

### MMQ が PP を大幅に悪化させる原因

仮説で期待した「stream sync 排除による PP 改善」は実現しなかった。代わりに **-33% ～ -48% の大幅退行** が発生した。

**根本原因**: P100 (cc 6.0) には DP4A 命令（`__dp4a`、4×int8 乗加算）のハードウェアサポートがない。MMQ カーネルは DP4A を前提に設計されており、P100 ではソフトウェアエミュレーション（通常の int32 乗加算 4 回に展開）で実行される。このエミュレーション・オーバーヘッドが stream sync 排除の利得を大幅に上回る。

**退行幅が pp サイズで変わる理由**: pp128 では -48% だが pp512/pp2048 では -33%。これは pp128 で batch サイズが小さく、MMQ の 64×64 タイルの稼働率が低い（SM 稼働率の非効率性がより顕在化する）ためと推測される。pp が大きくなるとタイル稼働率が改善し、退行幅が縮小する。

**TG が影響を受けない理由**: tg32 は batch_size=1 の逐次生成であり、`mul_mat_id` フォールバックの expert sort + stream sync がボトルネックではない。また、llama.cpp は tg 時に graph reuse を行うため、mul_mat パスの違いが影響しにくい。

### 交絡チェック

- [x] **単一変数の分離**: 同一ソースコード、差異は `GGML_CUDA_FORCE_MMQ=ON` のみ（+ mmq.cu のチェック順序入れ替え）
- [x] **バイナリ比較の正当性**: コンパイル時フラグのため環境変数トグルは不可能。バイナリ差異は FORCE_MMQ の有無のみ
- [x] **ホットパスのログ出力**: なし
- [x] **分布の単峰性**: 各条件の SD が小さく安定（pp128 SD < 0.5 t/s）

## 結論

**MMQ 強制有効化は P100 (cc 6.0) では採用不可**。DP4A ソフトウェアエミュレーションのコストが stream sync 排除の利得を大幅に上回り、PP が -33% ～ -48% 退行する。TG は影響なし。

P100 での MoE PP 最適化は、MMQ ではなく別のアプローチ（例: `mul_mat_id` フォールバックパスの stream sync 削減、expert batching の改善）を検討する必要がある。
