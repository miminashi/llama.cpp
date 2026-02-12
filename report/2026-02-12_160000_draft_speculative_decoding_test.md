# ドラフトモデルベース Speculative Decoding テスト結果

- **実施日時**: 2026年2月12日 16:00
- **ワークツリー**: `.worktree/draft-spec-test`

## 前提・目的

前回の ngram speculative decoding テストで、GLM-4.7 (MoE) モデルでは ngram ベースの acceptance rate が最大 10.6% にとどまり、Generation 速度が改善しないことが判明した。

本テストでは、より高い acceptance rate が期待できるドラフトモデルベースの speculative decoding を検証する。

- **背景**: ngram speculative decoding は MoE モデルの確率的 Expert routing と高エントロピー出力で効果なし
- **目的**: ドラフトモデル (Dense) を用いた speculative decoding の GLM-4.7 11GPU クラスタでの実効性を定量評価
- **前提条件**: GLM-4.7 IQ2_M が 11GPU (7C+4R) で正常動作していること

### 参照レポート
- [Ngram Speculative Decoding テスト](2026-02-12_134623_ngram_speculative_decoding_test.md)
- [ハイブリッド並列化リサーチ](2026-02-12_070600_hybrid_parallelism_research.md)

## ドラフトモデル候補の調査

### VRAM 空き状況 (GLM-4.7 IQ2_M ロード後)

- CUDA0: **6,562 MiB** free (最大、ドラフトモデル配置先)
- CUDA1-6: 3,500-4,700 MiB free

### 候補モデル

| モデル | パラメータ | 種別 | GGUF サイズ | CUDA0 に収まるか |
|--------|-----------|------|:-----------:|:---------------:|
| GLM-4-9B-0414 | 9B | Dense | 3.7 GB (IQ2_M) | o |
| qwen2.5-0.5b | 0.5B | Dense | 0.47 GB (Q4_K_M) | o |
| GLM-4.7-Flash | 30B (A3B) | MoE | 11.9 GB (Q2_K_XL) | x |

GLM-4.7-Flash は MoE かつ 11.9 GB で CUDA0 に収まらないため除外。

### トークナイザー互換性

| 項目 | GLM-4.7 (target) | GLM-4-9B-0414 (draft) | qwen2.5-0.5b (draft) |
|------|:-----------------:|:---------------------:|:--------------------:|
| model | gpt2 | gpt2 | gpt2 |
| pre | glm4 | glm4 | qwen2 |
| vocab size | 151,552 | 151,552 | 151,936 |
| BOS | 151331 | 151329 | - |
| EOS | 151329 | 151336 | 151645 |

- **GLM-4-9B-0414**: 同じ vocab サイズ・tokenizer だが BOS/EOS が異なる → 自動 retokenization 発動
- **qwen2.5-0.5b**: 異なる tokenizer family → 自動 retokenization 発動

## コード変更

前回の ngram テストと同じ変更を cherry-pick:
- `common/arg.cpp` — spec フラグの CLI 公開
- `tools/cli/cli.cpp` — ドラフト統計のタイミング表示追加
- `scripts/rdma-build.sh`, `scripts/rdma-deploy.sh` — `-DGGML_RPC=ON` 追加
- `scripts/rpc-server.sh` — RPC サーバー管理スクリプト

## 再現方法

### ビルド・デプロイ

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/draft-spec-test
bash scripts/rdma-build.sh local
bash scripts/rdma-deploy.sh
```

### ドラフトモデルのダウンロード

```python
from huggingface_hub import hf_hub_download
hf_hub_download('bartowski/THUDM_GLM-4-9B-0414-GGUF',
                'THUDM_GLM-4-9B-0414-IQ2_M.gguf',
                local_dir='/tmp/GLM-4-9B-0414-IQ2_M')
```

### RDMA ベースラインテスト

```bash
bash scripts/rdma-server.sh start
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -hf unsloth/GLM-4.7-GGUF:UD-IQ2_M \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 4096 -n 2048 --seed 42 \
  -p 'Explain quantum computing in great detail, covering its complete history from theoretical foundations to modern developments, the fundamental principles including superposition and entanglement, gate-based and annealing approaches, error correction challenges, current hardware implementations by major companies, and potential future applications in cryptography, drug discovery, optimization, and machine learning.' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### RDMA + ドラフトモデルテスト (例: GLM-4-9B, draft=8)

上記コマンドに以下を追加:
```
-md /tmp/GLM-4-9B-0414-IQ2_M/THUDM_GLM-4-9B-0414-IQ2_M.gguf --draft 8 -ngld 999 -devd CUDA0 -v
```

### RPC ベースラインテスト

```bash
bash scripts/rpc-server.sh start
LD_LIBRARY_PATH=build/bin build/bin/llama-cli \
  -hf unsloth/GLM-4.7-GGUF:UD-IQ2_M \
  --rpc 192.168.100.2:50052,192.168.100.2:50053,192.168.100.2:50054,192.168.100.2:50055 \
  -sm layer -ngl 999 -c 4096 -n 2048 --seed 42 \
  -p '(同上)' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## テスト結果

### テスト条件

- **ターゲットモデル**: GLM-4.7 IQ2_M (unsloth/GLM-4.7-GGUF:UD-IQ2_M)
- **構成**: 11GPU (7 CUDA + 4 リモート)
- **生成長**: 2048 トークン
- **コンテキスト**: 4096
- **Seed**: 42

### ベースライン

| # | バックエンド | Prompt (t/s) | Generation (t/s) |
|---|------------|:------------:|:----------------:|
| 1a | RDMA (7C+4R) | 12.2 | 5.1 |
| 1b | RPC (7C+4R) | 14.9 | 7.4 |

### GLM-4-9B-0414 IQ2_M (3.7 GB) をドラフトモデルとして使用

| # | バックエンド | Draft数 | pp (t/s) | tg (t/s) | Accepted/Total | Accept Rate | tg 変化 |
|---|------------|:-------:|:--------:|:--------:|:--------------:|:-----------:|:-------:|
| 2a | RDMA | 16 | 15.9 | 4.8 | 1169/2086 | 56.0% | -5.9% |
| 2b | RDMA | 8 | 15.7 | 5.0 | 1221/2041 | 59.8% | -2.0% |
| 2c | RDMA | 4 | 15.7 | 4.8 | 1126/1914 | 58.8% | -5.9% |

### qwen2.5-0.5b Q4_K_M (0.47 GB) をドラフトモデルとして使用

| # | バックエンド | Draft数 | pp (t/s) | tg (t/s) | Accepted/Total | Accept Rate | tg 変化 |
|---|------------|:-------:|:--------:|:--------:|:--------------:|:-----------:|:-------:|
| 3a | RDMA | 16 | 15.7 | **5.3** | 1009/1942 | 52.0% | **+3.9%** |
| 3b | RDMA | 8 | 15.7 | 5.2 | 972/1914 | 50.8% | +2.0% |
| 3c | RDMA | 4 | 15.7 | 5.1 | 968/1855 | 52.2% | ±0% |

### RPC + qwen2.5-0.5b (最良構成)

| # | バックエンド | Draft数 | pp (t/s) | tg (t/s) | Accepted/Total | Accept Rate | tg 変化 |
|---|------------|:-------:|:--------:|:--------:|:--------------:|:-----------:|:-------:|
| 4a | RPC | 16 | 16.0 | 5.6 | 994/1963 | 50.6% | **-24.3%** |

## 分析

### 1. ドラフトモデル vs ngram: Acceptance Rate の劇的改善

| 手法 | 最高 Accept Rate |
|------|:----------------:|
| ngram-simple (n=4, m=8) | 10.6% |
| GLM-4-9B-0414 draft=8 | **59.8%** |
| qwen2.5-0.5b draft=16 | **52.0%** |

ドラフトモデルベースは ngram 比で acceptance rate が **5-6 倍** に向上。GLM-4-9B は同じ GLM 系列の tokenizer を持つため、qwen2.5-0.5b より acceptance rate が高い。

### 2. 大きいドラフトモデル (9B) vs 小さいドラフトモデル (0.5B)

GLM-4-9B (3.7 GB) は acceptance rate が高い (56-60%) が Generation 速度は低下 (-2% ~ -6%)。一方 qwen2.5-0.5b (0.47 GB) は acceptance rate がやや低い (50-52%) が Generation 速度は微増 (+0% ~ +4%)。

**原因**: ドラフトモデルの推論時間の差。
- GLM-4-9B IQ2_M: CUDA0 上で ~28.7 ms/token (eval time per run)
- qwen2.5-0.5b Q4_K_M: CUDA0 上で ~4.8 ms/token

qwen2.5-0.5b はドラフト推論が **6 倍高速** なため、バッチ検証のオーバーヘッドをより多く相殺できる。

### 3. RDMA vs RPC: Speculative Decoding の影響の非対称性

- **RDMA**: ベースライン 5.1 → speculative 5.3 t/s (**+3.9%**、微改善)
- **RPC**: ベースライン 7.4 → speculative 5.6 t/s (**-24.3%**、大幅劣化)

RPC は per-device 独立ソケットにより、**各デバイスの graph_compute を並列発行** できる。Speculative decoding のバッチ検証はこの並列性を破壊する (バッチ検証は全デバイスを逐次走査する必要がある)。

RDMA は元から単一接続で逐次実行のため、speculative decoding による追加的な逐次化ペナルティがない。

### 4. Prompt 速度の変動

Speculative decoding テスト全体で Prompt 速度は 15.7-16.0 t/s とベースライン (12.2-14.9 t/s) より高い傾向。これは speculative decoding の効果ではなく、サーバー GPU の計算時間のセッション間変動 (既知課題) と推定される。

### 5. 最適ドラフトトークン数

| ドラフトモデル | 最適 Draft数 | Generation (t/s) |
|--------------|:-----------:|:----------------:|
| GLM-4-9B | 8 | 5.0 |
| qwen2.5-0.5b | 16 | 5.3 |

軽量ドラフトモデルほどドラフトトークン数を増やしても速度ペナルティが小さい。qwen2.5-0.5b は 16 トークンまでドラフトしても 5ms/token のため、バッチ検証 1 回あたりのドラフト生成コストは ~80ms に収まる。

## 結論

**ドラフトモデルベース speculative decoding は ngram 比で大幅に改善するが、11GPU 分散環境では Generation 速度の実質的な改善は限定的 (+3.9% at best)。**

### 主要な発見

1. **Acceptance rate は大幅改善**: ngram の 10.6% → ドラフトモデルの **52-60%**
2. **軽量ドラフトが有利**: qwen2.5-0.5b (0.5B) は GLM-4-9B (9B) よりドラフト推論が 6 倍速く、トータルで優位
3. **RDMA で微改善、RPC で大幅劣化**: 単一接続の RDMA はバッチ検証のペナルティが小さいが、並列ソケットの RPC では並列性が破壊される
4. **分散環境のバッチ検証コストが律速**: 11GPU にまたがるバッチ forward pass は、単一トークンの 11GPU forward pass より通信オーバーヘッドが大きい

### RDMA vs RPC の speculative decoding 適性

| バックエンド | Speculative 適性 | 理由 |
|------------|:----------------:|------|
| RDMA | △ (微改善) | 元から逐次実行のため追加ペナルティなし |
| RPC | x (大幅劣化) | 並列ソケットの恩恵がバッチ検証で失われる |

### Ngram vs ドラフトモデル 総合比較

| 手法 | Accept Rate | RDMA tg 変化 | RPC tg 変化 | 追加リソース |
|------|:-----------:|:-----------:|:-----------:|:----------:|
| Ngram (best) | 10.6% | -2.0% | -6.8% | なし |
| ドラフト GLM-4-9B | 59.8% | -2.0% | N/A | 3.7 GB VRAM |
| **ドラフト qwen2.5-0.5b** | **52.0%** | **+3.9%** | -24.3% | 0.47 GB VRAM |

### 推奨

- **RDMA バックエンド**: qwen2.5-0.5b draft=16 で微改善 (+3.9%) が期待できるが、サーバーGPU の計算時間変動 (±10%) と比較して有意差とは言い難い
- **RPC バックエンド**: speculative decoding は使用しない (大幅劣化)
- **根本的な改善**: speculative decoding ではなく、RDMA 接続のデバイス分離 (per-device connection) やパイプライン化による graph_compute 並列化が必要
