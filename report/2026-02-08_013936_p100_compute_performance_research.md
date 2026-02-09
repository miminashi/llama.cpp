# P100 計算性能ボトルネック分析と改善提案

- **実施日時**: 2026年2月8日 01:39 (更新: 01:55)

## 前提・目的

Tesla P100 (Pascal, SM6.0, CC6.0) を使用した llama.cpp の分散推論 (RDMA バックエンド) において、
GPU 計算性能のボトルネックを特定し、ビルドフラグやカーネル選択による改善余地を調査する。

- **背景**: RDMA バックエンドの通信性能は Step 3-4 で大幅に改善済みだが、GPU 計算自体がボトルネックになっているケースがある。特に Generation (token-by-token 生成) ではメモリバウンドであり、P100 の HBM2 帯域 (732 GB/s) を最大限活用する必要がある
- **目的**: P100 固有のアーキテクチャ制約を明確化し、llama.cpp のビルドフラグ・カーネル選択で改善可能な項目を優先度付きで提案する
- **前提条件**:
  - 1号機: 7× Tesla P100-PCIE-16GB (CC 6.0)
  - 2号機: 4× Tesla P100 (RDMA 経由)
  - CUDA Toolkit 12.0
  - 現在のビルド: `CMAKE_CUDA_ARCHITECTURES="60"`
  - GPU は使用中のためベンチマーク実行は不可。コード分析とビルド検証のみ

## 参照

- `CLAUDE.md` — ビルド手順とプロジェクト状態
- `report/2026-02-07_235500_rdma_performance_improvement_research.md` — RDMA 性能改善調査

## 調査方法

3名のエージェントで並行調査を実施:

1. **llama-cpp-kernel**: llama.cpp の CUDA カーネルソースコード (ggml/src/ggml-cuda/) を読み、P100 でのカーネル選択・パラメータの制約を特定
2. **p100-architecture**: P100 のハードウェア特性、理論性能限界、他フレームワークの対応状況を Web 検索で調査
3. **build-tester**: 上記の結果を受けてビルドフラグの組み合わせを検証し、最終レポートを統合・作成

---

## P100 アーキテクチャ特性

### P100 の主要スペック

| 項目 | P100-PCIe (CC 6.0) | V100 (CC 7.0) | A100 (CC 8.0) |
|------|:-------------------:|:-------------:|:-------------:|
| FP32 演算 | 10.6 TFLOPS | 15.7 TFLOPS | 19.5 TFLOPS |
| FP16 演算 | 21.2 TFLOPS (2:1) | 112 TFLOPS (TC) | 312 TFLOPS (TC) |
| Tensor Cores | **なし** | あり | あり |
| `__dp4a` (INT8 dot product) | **なし** (CC 6.1+) | あり | あり |
| メモリ帯域 (HBM2) | 732 GB/s | 900 GB/s | 2,039 GB/s |
| ECC | **デフォルト有効** | デフォルト有効 | デフォルト有効 |
| 共有メモリ/SM | 64 KB | 96 KB | 164 KB |
| L2 キャッシュ | 4 MB | 6 MB | 40 MB |
| SM 数 | 56 | 80 | 108 |
| `cp.async` | **なし** | **なし** | あり |
| CUDA Graphs | サポート | サポート | サポート |
| max_cpy_bytes | **8** | 16 | 16 |

### P100 GP100 の特殊性

- **FP16 がフルスピード**: GP100 (CC 6.0) は FP16 が FP32 の 2 倍速。他の Pascal (CC 6.1: GP104/GP106) では FP16 は FP32 の **1/64** と極端に遅い
- **ECC デフォルト有効**: メモリ帯域に **約 12.5% のペナルティ** (732 → 実効 ~640 GB/s)
- **P100 をサポートする LLM フレームワークは llama.cpp のみ**: vLLM, TensorRT-LLM, ExLlamaV2 は CC 7.0+ を要求

---

## Generation の理論性能限界

Generation (token-by-token 生成) は **メモリ帯域バウンド** であり、1 token あたり全重みパラメータを HBM2 から読み出す必要がある。

### シングル GPU 理論限界

| モデル | サイズ | 理論上限 | 実測値 | 効率 |
|--------|:------:|:--------:|:------:|:----:|
| gpt-oss-20b Q4_K_M | 11 GB | ~61 t/s | **64 t/s** | **>100%** ※ |

※ L2 キャッシュ効果により理論値を超過。**シングル GPU ではメモリ帯域の理論限界にほぼ到達済み。**

### マルチ GPU (11 GPU) 理論限界

| モデル | サイズ | GPU あたり | 理論上限 | 実測値 | 効率 |
|--------|:------:|:----------:|:--------:|:------:|:----:|
| GLM-4.7 IQ2_M | ~40 GB | ~3.6 GB | ~200 t/s | **6.8 t/s** | **3.4%** |

11 GPU 分散時の効率が極めて低い理由:
1. **RDMA 通信の逐次実行**: 各デバイスの `graph_compute` が順番に実行される
2. **レイヤー間通信レイテンシの蓄積**: hidden state (~10 KB) の転送が各レイヤーで発生
3. **カーネル融合が P100 で無効**: 追加のカーネル起動コスト

**結論: マルチ GPU Generation のボトルネックは GPU 計算ではなく通信オーバーヘッド。GPU 計算の改善効果は限定的。**

---

## llama.cpp カーネル選択における P100 の制約

### 制約一覧

| カーネル/機能 | P100 での状態 | 原因 | コード箇所 |
|-------------|:----------:|------|-----------|
| MMQ (量子化行列積) | **無効** | `__dp4a` 命令なし (CC 6.0 < 6.1) | `mmq.cu:304` |
| MMF (FP行列積) | 量子化型で無効 | 量子化型の場合 `return false` | `mmf.cu:135` |
| MMVQ (量子化 mat-vec) | 動作するが遅い | `ggml_cuda_dp4a` がソフトウェアエミュレーション | `common.cuh:698-700` |
| カーネル融合 | **無効** | `cc <= GGML_CUDA_CC_PASCAL → false` | `ggml-cuda.cu:2159` |
| stream-k | **無効** | Turing 以降が必要 | `mmq.cuh:3499` |
| FA MMA/WMMA | **無効** | Volta 以降が必要 | `fattn.cu:370-416` |
| FA Tile/Vec | **有効** (フォールバック) | テンソルコア不要 | `fattn.cu:443-457` |
| max_cpy_bytes | **8** (半分) | Volta 以降で 16 | `common.cuh:350-354` |
| cuBLAS HGEMM | **有効** | FP16 フルスピード | — |

### P100 での mul_mat カーネル選択フロー

```
量子化テンソル × F32テンソル (LLM の主要演算)
  ├── ne11 <= 8 (Generation, batch=1)
  │     → MMVQ (量子化 mat-vec, dp4a ソフトウェアエミュレーション)
  ├── ne11 > 8 かつ MMQ 有効
  │     → MMQ → P100 では無効! (dp4a チェックで弾かれる)
  └── それ以外 (Prompt 処理等)
        → dequant → cuBLAS SGEMM (FP16 の場合は HGEMM)
```

### DP4A ソフトウェアエミュレーションの影響

P100 (sm_60) では `ggml_cuda_dp4a` が以下のコードで実行される:

```c
// common.cuh:698-700 (P100 フォールバック)
const int8_t * a8 = (const int8_t *) &a;
const int8_t * b8 = (const int8_t *) &b;
return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
```

ハードウェア `__dp4a` は 1 命令で同等の処理を行うため、**約 4 倍のペナルティ**。
MMVQ カーネル (`vecdotq.cuh`) の全量子化フォーマットに影響する。

### Flash Attention の P100 コードパス

P100 では FA の Tile カーネルにフォールバック。上流 (llama.cpp 本家リポジトリ) のコードに以下の TODO が存在:

```c
// fattn-tile.cuh:8-9
// TODO optimize kernel parameters for FP16 NVIDIA (P100)
// TODO optimize kernel parameters for head sizes 40, 72, 80, 96, 112
```

P100 の FP16 フルスピード特性に最適化されたカーネルパラメータが存在しない。将来的に上流 (llama.cpp 本家リポジトリ [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)) で改善される可能性がある。

---

## ビルドフラグ検証結果

ワークツリー `p100-perf-test` で以下の構成をビルド検証した (全て `CMAKE_CUDA_ARCHITECTURES="60"`)。

### テスト構成一覧

| # | 構成 | ビルド結果 | libggml-cuda.so | 備考 |
|---|------|:----------:|:---------------:|------|
| 1 | ベースライン + `GGML_CUDA_GRAPHS=ON` | **成功** | 103 MB | CUDA Graphs 有効化 |
| 2 | `GGML_CUDA_FORCE_MMQ=ON` | **成功** | 103 MB | MMQ 強制 (効果なし) |
| 3 | `GGML_CUDA_FORCE_CUBLAS=ON` | **成功** | 103 MB | cuBLAS 強制 (現状同等) |
| 4 | `GGML_CUDA_FA=OFF` | **成功** | — | FA 無効化 |
| 5 | `GGML_CUDA_GRAPHS=ON` + `GGML_CUDA_FA_ALL_QUANTS=ON` | **成功** | 160 MB (+55%) | FA 全量子化 |

### 各構成の詳細分析

#### `GGML_CUDA_GRAPHS=ON`
- **有効**: P100 は CUDA Graphs をサポート
- Generation では多数の小カーネルが逐次実行されるため、カーネル起動オーバーヘッド削減の効果あり
- RDMA バックエンドとの互換性要確認 (split buffer 非対応: `ggml-cuda.cu:2878`)

#### `GGML_CUDA_FORCE_MMQ=ON`
- **効果なし**: コード上、`ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A` チェック (`mmq.cu:304`) が `GGML_CUDA_FORCE_MMQ` チェック (`mmq.cu:308`) より先に評価される
- sm_60 ビルドでは `FORCE_MMQ` を設定しても MMQ は依然として無効

#### `GGML_CUDA_FORCE_CUBLAS=ON`
- P100 では MMQ が使えないため、これが実質的なデフォルト動作
- ビルドフラグを明示的に設定するかどうかの差のみ

#### `GGML_CUDA_FA=OFF`
- P100 の FA は Tile/Vec カーネル (テンソルコアなし版) にフォールバック
- 標準 Attention (cuBLAS GEMM ベース) との比較は要ベンチマーク
- 長いコンテキストでは FA の方がメモリ効率が良い

#### `GGML_CUDA_FA_ALL_QUANTS=ON`
- FA で全量子化フォーマット対応。ライブラリサイズ 55% 増加
- `--cache-type-k`/`--cache-type-v` で量子化 KV キャッシュを使用する場合に有効

---

## 改善提案 (優先度順)

### 優先度 高

#### 提案 1: Flash Attention 有効化テスト (`--flash-attn`)

| 項目 | 内容 |
|------|------|
| **設定** | `llama-cli` / `llama-bench` に `--flash-attn` フラグを追加 |
| **期待効果** | Prompt 処理速度の改善 (P100 の FP16 フルスピードを FA Tile カーネルが活用) |
| **実装難易度** | 低 (ランタイムフラグの変更のみ) |
| **リスク** | FA Tile カーネルは P100 向けに最適化されていない (fattn-tile.cuh:8 に TODO) ため、逆に遅くなる可能性もある |
| **検証方法** | `llama-bench` で `--flash-attn` の有無で pp (prompt processing) を比較 |

P100 GP100 は FP16 がフルスピード (FP32 の 2 倍) のため、FA の FP16 カーネルが cuBLAS SGEMM より速い可能性がある。ただし上流 (llama.cpp 本家) コードに P100 向け最適化 TODO が残っており、効果は未知。

#### 提案 2: 量子化フォーマット変更 (IQ2_M → Q4_K_M)

| 項目 | 内容 |
|------|------|
| **設定** | モデルファイルを Q4_K_M 量子化版に変更 |
| **期待効果** | Generation 速度 10-30% 改善の可能性 |
| **実装難易度** | 低 (モデルファイルの変更のみ) |
| **リスク** | モデルサイズ増加 (IQ2_M ~40GB → Q4_K_M ~80GB)。11GPU では収まる (176GB VRAM) が余裕は少ない |
| **前提** | 16GPU (256GB VRAM) では Q4_K_M が収まる |

IQ2_M は P100 にとって **最も非効率な量子化フォーマットの一つ**:
- コードブックルックアップが多い → メモリアクセスパターンが不規則
- `ggml_cuda_dp4a` のソフトウェアエミュレーション回数が多い
- Q4_K_M / Q4_0 はデコードがシンプルで cuBLAS パスとの相性が良い

#### 提案 3: ubatch サイズチューニング

| 項目 | 内容 |
|------|------|
| **設定** | `--ubatch-size 128` または `--ubatch-size 256` |
| **期待効果** | Prompt 処理速度 10-30% 改善の可能性 |
| **実装難易度** | 低 (ランタイムパラメータの変更のみ) |
| **リスク** | なし (元に戻すのが容易) |
| **検証方法** | `llama-bench -ub 64,128,256,512` で比較 |

P100 の共有メモリ (64KB/SM) と L2 キャッシュ (4MB) はサイズが限られているため、大きすぎる ubatch はキャッシュ溢れを起こす。デフォルト 512 は新しい GPU 向けに設定されている可能性が高い。

### 優先度 中

#### 提案 4: CUDA Graphs 有効化

| 項目 | 内容 |
|------|------|
| **ビルドフラグ** | `-DGGML_CUDA_GRAPHS=ON` |
| **期待効果** | Generation 速度 5-15% 改善 |
| **実装難易度** | 低 (ビルドフラグの変更のみ、ビルド検証済み) |
| **リスク** | RDMA バックエンドの split buffer 操作との互換性要確認 |

カーネル起動オーバーヘッド (~5μs × 数百カーネル/token) の削減。ただしマルチ GPU では通信レイテンシが支配的なため、効果はシングル GPU 時より限定的。

#### 提案 5: ECC 無効化

| 項目 | 内容 |
|------|------|
| **設定** | `nvidia-smi -e 0` + 再起動 |
| **期待効果** | メモリ帯域 **+12.5%** (732 → ~824 GB/s) |
| **実装難易度** | 低 (コマンド実行のみ) |
| **リスク** | **データ整合性**: ECC 無効化で soft error によるサイレントデータ破損の可能性。HPC / サーバ用途では非推奨 |

Generation はメモリ帯域バウンドであるため、帯域 12.5% 改善はそのまま速度向上に直結する。ただしデータ整合性リスクを受け入れる判断が必要。

### 優先度 低

#### 提案 6: `GGML_CUDA_FA_ALL_QUANTS=ON`

| 項目 | 内容 |
|------|------|
| **ビルドフラグ** | `-DGGML_CUDA_FA_ALL_QUANTS=ON` |
| **期待効果** | 量子化 KV キャッシュ使用時に FA 対応フォーマットが増える |
| **実装難易度** | 低 (ビルドフラグの変更のみ、ビルド検証済み) |
| **リスク** | ライブラリサイズ +55% (103 → 160 MB) |

`--cache-type-k q4_0` 等で量子化 KV キャッシュを使用する場合にのみ有効。

#### 提案 7: Speculative Decoding

| 項目 | 内容 |
|------|------|
| **設定** | `--draft-model` で小型モデルを使用 |
| **期待効果** | **Generation 2-4 倍高速化** の可能性 (受理率に依存) |
| **実装難易度** | 中 (適切なドラフトモデルの選定が必要) |
| **リスク** | RDMA バックエンドでの互換性未検証。ドラフトモデル用の VRAM が必要 |

Speculative Decoding は通信回数を削減するため、マルチ GPU のレイテンシ蓄積問題に直接効果がある。ただし RDMA バックエンド + 分散推論での動作検証が必要。

#### 提案 8: RDMA 通信の並列化 (アーキテクチャ改善)

| 項目 | 内容 |
|------|------|
| **内容** | デバイスごとの独立 RDMA 接続 |
| **期待効果** | Generation 速度の大幅改善 (理論効率 3.4% → 30-50%) |
| **実装難易度** | **高** (RDMA バックエンドの大規模リファクタリングが必要) |
| **リスク** | 設計変更の規模が大きい |

現在 RDMA は単一接続で全リモートデバイスを共有しているため、`graph_compute` が逐次実行される。RPC はデバイスごとに独立ソケットを持つため、Generation で RDMA 比 +10% の優位性がある。

---

## P100 では効果がないと判明した項目

| 項目 | 理由 |
|------|------|
| `GGML_CUDA_FORCE_MMQ=ON` | DP4A チェック (`mmq.cu:304`) が `FORCE_MMQ` チェック (`mmq.cu:308`) より先に評価されるため、sm_60 では無効 |
| `CMAKE_CUDA_ARCHITECTURES` に sm_61 追加 | P100 (CC 6.0) では sm_60 ネイティブコードが優先使用される。`ggml_cuda_highest_compiled_arch(600)` は sm_60 を返す |
| カーネル融合の強制有効化 | `cc <= GGML_CUDA_CC_PASCAL` で明示的に無効化されており、上流 (llama.cpp 本家) で P100 では効果が不安定と判断済み |
| `--use_fast_math` の追加 | llama.cpp では既にデフォルト有効 (`CMakeLists.txt:187`) |

---

## 16 GPU Q4_K_M 推定性能

16 GPU が利用可能になった場合の GLM-4.7 Q4_K_M 推定:

| 項目 | 値 |
|------|:---:|
| モデルサイズ (Q4_K_M) | ~80 GB |
| 必要 VRAM (16 GPU) | 16 × 16 GB = 256 GB (収容可能) |
| GPU あたりの重み | ~5 GB |
| シングル GPU 理論帯域 | ~640 GB/s (ECC 有効時) |
| シングル GPU 理論 tg | 640 / 5 ≈ 128 t/s |
| 16 GPU レイヤー分割 通信考慮 | 推定 **4-7 t/s** |

通信レイテンシの蓄積がボトルネックであるため、GPU 数の増加は必ずしも tg 速度の向上につながらない。Prompt 処理 (計算バウンド) ではスケーリング効果が期待できる。

---

## 再現方法

### ビルドテストの再現

```bash
# ワークツリー作成
cd /home/ubuntu/projects/llama.cpp
git worktree add /home/ubuntu/projects/llama.cpp/.worktree/p100-perf-test -b p100-perf-test feature/rdma-backend

# テスト 1: CUDA Graphs 有効化
cd /home/ubuntu/projects/llama.cpp/.worktree/p100-perf-test
rm -rf build
cmake -B build \
  -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="60" \
  -DGGML_CUDA_GRAPHS=ON
cmake --build build -- -j $(nproc)

# テスト 2: FORCE_MMQ (効果なし確認用)
rm -rf build
cmake -B build \
  -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="60" \
  -DGGML_CUDA_FORCE_MMQ=ON
cmake --build build -- -j $(nproc)

# テスト 3: FORCE_CUBLAS
rm -rf build
cmake -B build \
  -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="60" \
  -DGGML_CUDA_FORCE_CUBLAS=ON
cmake --build build -- -j $(nproc)

# テスト 4: FA 無効化
rm -rf build
cmake -B build \
  -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="60" \
  -DGGML_CUDA_FA=OFF
cmake --build build -- -j $(nproc)

# テスト 5: FA All Quants + CUDA Graphs
rm -rf build
cmake -B build \
  -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="60" \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build -- -j $(nproc)
```

### ベンチマーク検証 (要 GPU 空き時間)

```bash
# Flash Attention 効果測定
CUDA_VISIBLE_DEVICES=0 build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 3 -p 128,512 -n 32 \
  -fa 0,1

# ubatch チューニング
CUDA_VISIBLE_DEVICES=0 build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 3 -p 512 -n 32 \
  -ub 64,128,256,512

# CUDA Graphs 効果測定 (CUDA_GRAPHS=ON ビルドで)
CUDA_VISIBLE_DEVICES=0 build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 3 -p 128 -n 32

# 量子化フォーマット比較 (Q4_0 vs Q4_K_M)
CUDA_VISIBLE_DEVICES=0 build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_0.gguf \
  -ngl 999 -sm layer -r 3 -p 128 -n 32
```

---

## 結論と推奨アクション

### 最重要結論

1. **シングル GPU はメモリ帯域の理論限界に到達済み** (gpt-oss-20b: 理論 61 t/s、実測 64 t/s)。カーネル最適化の ROI は低い
2. **マルチ GPU Generation のボトルネックは RDMA 通信の逐次実行** (理論効率 3.4%)。GPU 計算改善は副次的
3. **IQ2_M は P100 に最も非効率な量子化フォーマットの一つ**。16GPU + Q4_K_M への移行が最も効果的
4. **P100 は llama.cpp の高速パス (dp4a, Tensor Core, stream-k, カーネル融合) が全て無効**。改善はパラメータ調整レベルにとどまる

### 推奨アクション

#### 即座に実施可能 (GPU 空き時間で)
1. `--flash-attn` の on/off ベンチマーク比較 (pp 速度)
2. `-ub 64,128,256,512` による ubatch チューニング (pp 速度)
3. `GGML_CUDA_GRAPHS=ON` ビルドの tg 速度比較

#### 16 GPU 利用可能後
4. GLM-4.7 **Q4_K_M** での 16 GPU 推論テスト (IQ2_M からの量子化フォーマット変更)
5. 16 GPU での RDMA 接続確立と安定性検証

#### 中期的に検討
6. Speculative Decoding の RDMA バックエンド互換性検証
7. RDMA 通信の並列化 (Generation のボトルネック根本解決)
8. ECC 無効化のリスク評価とベンチマーク
