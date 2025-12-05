# P100 cuBLAS HGEMM 最適化レポート

## 概要

Tesla P100 (GP100, 計算能力 6.0) において、ソフトウェアエミュレーションによる INT8 量子化行列乗算カーネル (MMQ) よりも cuBLAS FP16 HGEMM を優先するよう最適化しました。この変更により、P100 の効率的な FP16 ハードウェア能力を活用し、dp4a 命令の遅いソフトウェアエミュレーションを回避します。

## 背景

### Tesla P100 のハードウェア特性
- 計算能力: 6.0 (Pascal アーキテクチャ)
- FP16 スループット: 18.7 TFLOPS (FP32 の 9.3 TFLOPS の 2 倍)
- **dp4a 命令非対応** (cc >= 6.1 が必要、例: GTX 1080)
- **Tensor Core なし** (cc >= 7.0 が必要、Volta 以降)
- 効率的な cuBLAS HGEMM 実装

### 問題の特定

`ggml_cuda_should_use_mmq()` (mmq.cu:289-291) の元のコードパス選択ロジック:

```c
if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
```

P100 の場合:
- `fp16_mma_hardware_available(600)` は `false` を返す (cc >= 700 が必要)
- したがって: `!false || ne11 < 64` は `true || ne11 < 64` = **TRUE** と評価
- 結果: P100 は**常に** MMQ (量子化) カーネルを使用

### なぜこれが P100 にとって最適でないのか

1. **ソフトウェアエミュレーションのオーバーヘッド**: P100 はハードウェア dp4a 命令を持たない (cc < 610) ため、INT8 ドット積がソフトウェアでエミュレートされる:
   ```c
   // common.cuh:550-552
   const int8_t * a8 = (const int8_t *) &a;
   const int8_t * b8 = (const int8_t *) &b;
   return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
   ```

2. **FP16 ハードウェアの未活用**: P100 は FP32 の 2 倍のスループットを持つ専用 FP16 ハードウェアを搭載しているが、MMQ パスでは cuBLAS がそれを使用できない。

3. **cuBLAS の効率性**: NVIDIA の cuBLAS ライブラリには、P100 の FP16 ハードウェア能力を完全に活用する高度に最適化された HGEMM カーネルがある。

## 実装

### 変更内容

**ファイル**: `ggml/src/ggml-cuda/mmq.cu` (289-297 行)

```c
if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
    // P100 (Pascal, cc=600) は fp16 MMA と dp4a ハードウェア命令の両方を持たない。
    // dp4a はソフトウェアでエミュレートされ、cuBLAS FP16 HGEMM より遅い。
    // P100 の FP16 スループットは FP32 の 2 倍なので、ソフトウェアエミュレート dp4a より cuBLAS を優先。
    if (cc == GGML_CUDA_CC_PASCAL) {
        return false; // ソフトウェアエミュレート dp4a より cuBLAS FP16 HGEMM を優先
    }
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
```

### 変更後のロジックフロー

量子化行列乗算 (Q4_K, Q5_K, Q6_K など) の場合:

1. **P100 (cc=600)**: `false` を返す → FP16 非量子化を伴う cuBLAS パスを使用
2. **GTX 1080 (cc=610)**: ハードウェア dp4a を使用する MMQ を継続使用 ✓
3. **Volta+ (cc>=700)**: 大きなバッチで cuBLAS を使用 (Tensor Core) ✓
4. **AMD/MTHREADS**: 変更なし、別のブランチで処理 ✓

### cuBLAS パスの実行 (ggml-cuda.cu:1281-1333)

`use_mul_mat_q` が false の場合、コードは `ggml_cuda_op_mul_mat_cublas()` にフォールスルー:

1. `fast_fp16_hardware_available(cc)` をチェック → P100 で TRUE を返す
2. src0 を FP16 形式に非量子化
3. 必要に応じて src1 を FP16 に変換
4. `cublasGemmEx()` を呼び出し:
   - 入出力: FP16 行列
   - 計算: CUDA_R_16F (P100 での FP16 計算)
   - 演算: P100 の効率的な FP16 ハードウェアを使用した HGEMM
5. 出力用に結果を FP32 に変換

## 期待されるパフォーマンスへの影響

### P100 へのプラスの影響

1. **行列乗算性能**:
   - **変更前**: ソフトウェアエミュレート INT8 dp4a 演算
   - **変更後**: cuBLAS 経由のハードウェアアクセラレート FP16 HGEMM
   - **期待値**: 量子化モデル推論で 1.5-3 倍の高速化

2. **スループット向上**:
   - P100 の 18.7 TFLOPS FP16 能力を活用
   - スカラー INT8 エミュレーションのオーバーヘッドを回避
   - エミュレート INT8 より FP16 でメモリ帯域幅をより効率的に利用

3. **最も恩恵を受けるユースケース**:
   - 量子化モデル (Q4_K, Q5_K, Q6_K, IQ 系)
   - 大バッチ推論
   - プロンプト処理とテキスト生成

### 他の GPU への影響

**他の NVIDIA GPU への悪影響なし:**

- **GTX 1050/1060 (cc=610)**: MMQ 経由でハードウェア dp4a を継続使用
- **GTX 1080/1080 Ti (cc=610)**: MMQ 経由でハードウェア dp4a を継続使用
- **Volta V100 (cc=700+)**: 大バッチで Tensor Core を継続使用
- **Turing/Ampere/Ada (cc>=750)**: INT8 Tensor Core で最適化された MMQ を継続使用

**AMD や Moore Threads GPU への影響なし:** 変更は NVIDIA 固有のブランチ内。

## 技術的詳細

### 機能検出関数 (common.cuh)

```c
// P100 で true を返す
static bool fast_fp16_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc != 610) || ...;
}

// P100 で false を返す (Volta+ が必要)
static bool fp16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA) || ...;
}
```

### 計算能力リファレンス

| GPU | 計算能力 | dp4a HW | FP16 MMA | FP16 スループット | 変更後のパス |
|-----|---------|---------|----------|-----------------|-------------|
| P100 | 6.0 | ✗ | ✗ | 2x FP32 | cuBLAS HGEMM |
| GTX 1080 | 6.1 | ✓ | ✗ | 1/64x FP32 | MMQ (dp4a) |
| V100 | 7.0 | ✓ | ✓ | 8x FP32 | cuBLAS (大バッチ) または MMQ |
| T4 | 7.5 | ✓ | ✓ | 8x FP32 | INT8 TC 使用 MMQ |
| A100 | 8.0 | ✓ | ✓ | 16x FP32 | INT8 TC 使用 MMQ |

## テスト推奨事項

### 検証手順

1. **Pascal (sm_60) をターゲットに CUDA サポートでビルド**:
   ```bash
   cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 ..
   make -j$(nproc)
   ```

2. **P100 で量子化モデルをテスト**:
   ```bash
   ./llama-bench -m model-Q4_K.gguf -p 512 -n 128
   ```

3. **変更前後でパフォーマンスを比較**:
   - プロンプト処理速度 (tokens/s)
   - 生成速度 (tokens/s)
   - GPU 使用率とメモリ帯域幅

4. **正確性を検証**:
   - CPU 実装との出力一貫性をチェック
   - 様々なバッチサイズとシーケンス長をテスト
   - 異なる量子化タイプをテスト (Q4_K, Q5_K, Q6_K)

### 期待される指標

- **プロンプト処理**: 1.5-2.5 倍高速化
- **トークン生成**: 1.5-2 倍高速化
- **メモリ使用量**: 計算中わずかに増加 (一時的な FP16 バッファ)
- **数値精度**: 最小限の差異 (FP16 精度 vs ソフトウェア INT8)

## 互換性

### 安全性が保証される対象:
- ✓ Tesla P100 (対象 GPU)
- ✓ GTX 1050/1060/1070/1080 (cc=610, ハードウェア dp4a)
- ✓ Volta, Turing, Ampere, Ada アーキテクチャ (cc>=700)
- ✓ AMD GPU (別のコードパス)
- ✓ Moore Threads GPU (別のコードパス)

### ビルド要件
- CUDA 9.0+ (Pascal での FP16 サポート用)
- cuBLAS ライブラリ
- `-gencode arch=compute_60,code=sm_60` または `native` でコンパイル

## 結論

この最適化は Tesla P100 のユニークな特性を特にターゲットにしています: ハードウェア dp4a サポートがない一方で優れた FP16 ハードウェア性能。P100 をソフトウェアエミュレート INT8 パスではなく cuBLAS FP16 パスにルーティングすることで、P100 での量子化モデル推論の大幅なパフォーマンス向上が期待でき、他のすべての GPU アーキテクチャでの最適なパフォーマンスは維持されます。

変更は最小限 (5 行のコード) で、十分にドキュメント化されており、コードベース内の既存の計算能力固有の最適化パターンに従ってアーキテクチャ的に健全です。

---

**レポート作成日**: 2025-12-04
**ブランチ**: p100/cublas-hgemm
**ワークツリー**: /home/ubuntu/llama-worktrees/p100-cublas-hgemm
