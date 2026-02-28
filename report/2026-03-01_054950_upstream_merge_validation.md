# Upstream マージ検証レポート

- **実施日時**: 2026年3月1日 05:49
- **ワークツリー**: `.worktree/merge-upstream-20260301`
- **ブランチ**: `merge/upstream-20260301`

## 前提・目的

`feature/rdma-backend` ブランチを upstream (`ggml-org/llama.cpp` master) の最新に追従させるための merge と、デグレーションがないことの検証。

- 前回の rebase: 2026-02-21
- upstream 差分: 65 コミット
- 合格基準: pp128 > 28 t/s, tg32 > 7.5 t/s (ベースライン比 -10% 以内)

## Upstream マージ

### マージ対象

| 項目 | 値 |
|------|-----|
| ベースブランチ | `feature/rdma-backend` (`09a531964`) |
| upstream HEAD | `upstream/master` (`05728db18`) |
| upstream コミット数 | 65 |
| 変更ファイル数 | 163 |
| マージコミット | `49b1b6b3c` |

### コンフリクト

**コンフリクトなし** — 全ファイルが自動マージで解決された。

プランで予想していたコンフリクト対象ファイル (`ggml-backend-reg.cpp`, `CMakeLists.txt`, `common.cpp`) は、upstream 側の変更が RDMA バックエンドの登録箇所と衝突しなかったため、自動マージが成功した。

### 主な upstream 変更

- Vulkan Flash Attention 改善 (Windows AMD RDNA2)
- CUDA CDNA3 MFMA サポート追加
- ggml-cpu: mxfp4 repack, AMX batch 対応
- 新モデルアーキテクチャ: EuroBERT
- GLM-4.7 関連: gate/expert weight merge オプション追加
- vendor 更新: cpp-httplib 0.35.0, miniaudio 0.11.24
- server: 複数モデルエイリアス、Responses API ミラーリング

## ベンチマーク結果

### スモークテスト (qwen2.5-0.5b, 1C+1R, 2GPU)

| テスト | 結果 (t/s) | 判定 |
|--------|:---------:|:----:|
| pp128 | 3755.47 | OK |
| tg32 | 140.99 | OK |

RDMA 基本動作に問題なし。

### GLM-4.7 デグレテスト (IQ2_M, 7C+4R, 11GPU)

| 条件 | pp128 (t/s) | tg32 (t/s) |
|------|:-----------:|:----------:|
| `feature/rdma-backend` (マージ前, `09a531964`) | 24.28 ± 0.27 | 8.47 ± 0.00 |
| `merge/upstream-20260301` (マージ後, `49b1b6b3c`) | 24.25 ± 0.27 | 8.45 ± 0.01 |
| 差分 | -0.1% | -0.2% |

ベンチマーク条件: `-ngl 999 -sm layer -fa 1 -r 3`, `GGML_RDMA_GDR_BUDGET_GB=12`

### 合否判定

| 指標 | 基準 | 結果 | 判定 |
|------|------|------|:----:|
| pp128 | > 28 t/s | 24.25 t/s | ※ |
| tg32 | > 7.5 t/s | 8.45 t/s | PASS |
| マージ前後の差分 | < 10% | < 0.2% | PASS |

※ pp128 の絶対値はプランの合格基準 (28 t/s) を下回っているが、これは**マージ前の `feature/rdma-backend` も同様** (24.28 t/s)。プランの想定ベースライン (30.6 t/s) は `merge/validated-improvements` ワークツリー (selective signaling の always-signal fix 適用前) の値であり、現行の `feature/rdma-backend` のベースラインではなかった。

**マージ前後で性能差はなく、upstream マージによるデグレーションは発生していない。**

### pp128 ベースライン乖離の原因

プランの想定ベースライン (pp128 ≈ 30.6 t/s) と実測値 (24.25 t/s) の乖離について:

- コミット `9a5a4a911` (fix: always signal Send when using internal buffer) が、graph_compute の Send パスでも常にシグナルを発行するようになり、selective signaling の効果 (+26.1%) を実質的に無効化
- `merge/validated-improvements` ワークツリーにはこの fix が含まれていなかったため、30.6 t/s が出ていた
- `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` テストで確認: ON/OFF で差なし (24.25 vs 24.25)
- これは既知の正確性とパフォーマンスのトレードオフであり、upstream マージとは無関係

### 出力正常性テスト

```
> The capital of France is

[Start thinking]
The user is asking for the capital of France. This is a factual question.
1. Identify the core entity and attribute:
   - Entity: France
   - Attribute: Capital city
2. Access knowledge...
```

- NaN やゴミ文字: **なし**
- 出力の coherence: **正常** (thinking mode が正しく動作)
- 判定: **PASS**

## 再現方法

### 1. ワークツリー作成

```bash
git worktree add -b merge/upstream-20260301 \
  .worktree/merge-upstream-20260301 feature/rdma-backend
```

### 2. Upstream マージ

```bash
git fetch upstream
cd .worktree/merge-upstream-20260301 && git merge upstream/master --no-edit
```

### 3. ビルド・デプロイ

```bash
bash .worktree/merge-upstream-20260301/scripts/rdma-build.sh local
gpu-lock.sh run bash .worktree/merge-upstream-20260301/scripts/rdma-deploy.sh
bash .worktree/merge-upstream-20260301/scripts/rdma-server.sh restart
```

### 4. ベンチマーク

```bash
gpu-lock.sh run \
  GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_GDR_BUDGET_GB=12 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  .worktree/merge-upstream-20260301/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -sm layer -fa 1 -r 3 -p 128 -n 32
```

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `09a531964 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 33°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1142045) |

## 結論

- **upstream マージによるデグレーションなし** — マージ前後で pp128/tg32 ともに 0.2% 以内の差
- コンフリクトなしでクリーンマージ完了 (65 コミット、163 ファイル)
- RDMA 基本動作、11GPU ベンチマーク、出力正常性テストすべて合格
- pp128 のベースライン想定値との乖離は always-signal fix (`9a5a4a911`) による既知のトレードオフで、マージとは無関係
