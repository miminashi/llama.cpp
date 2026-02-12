# Ngram Speculative Decoding テスト結果

- **実施日時**: 2026年2月12日 13:46
- **ワークツリー**: `.worktree/ngram-spec-test`

## 前提・目的

ハイブリッド並列化リサーチレポート (`report/2026-02-12_070600_hybrid_parallelism_research.md`) で「優先度1: 実装コスト極小」と推奨された ngram speculative decoding を、RDMA / RPC 両バックエンドで実測し、GLM-4.7 MoE モデルでの実効性を検証する。

- **背景**: ngram speculative decoding はトークン履歴から繰り返しパターンを検索し、ドラフトトークンをバッチ生成 → 検証する手法。追加モデル不要で即座に使用可能
- **目的**: MoE モデル (Expert routing が確率的) での acceptance rate と Generation 速度への影響を定量評価する
- **前提条件**: GLM-4.7 IQ2_M が 11GPU (7C+4R) で正常動作していること

### 参照レポート
- [ハイブリッド並列化リサーチ](2026-02-12_070600_hybrid_parallelism_research.md)

## コード変更

### 1. `common/arg.cpp` — ngram spec フラグを CLI に公開
`--spec-type`, `--spec-ngram-size-n`, `--spec-ngram-size-m`, `--spec-ngram-check-rate`, `--spec-ngram-min-hits` の5つのフラグが `LLAMA_EXAMPLE_SERVER` のみだったのを `{LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}` に変更。

### 2. `tools/cli/cli.cpp` — ドラフト統計のタイミング表示追加
`result_timings.draft_n > 0` の場合に acceptance rate を表示するように変更。

### 3. `scripts/rdma-build.sh`, `scripts/rdma-deploy.sh` — RPC ビルド有効化
`CMAKE_OPTS` に `-DGGML_RPC=ON` を追加。

### 4. `scripts/rpc-server.sh` — RPC サーバー管理スクリプト新規作成
rdma-server.sh と同構造。4 GPU × 個別ポート (50052-50055)。

## 発見事項: CLI のログレベル問題

speculative decoding の統計は `LOG_INF` (INFO レベル) で出力されるが、CLI はデフォルトで `LOG_LEVEL_ERROR` に設定される (`cli.cpp:192`)。そのため、`-v` フラグなしでは統計が表示されない。

さらに、`result_timings` 構造体には `draft_n` / `draft_n_accepted` フィールドがあるが、CLI のタイミング表示 (`cli.cpp:401`) には含まれていなかった。本テストでは表示を追加して対応。

## 再現方法

### ビルド・デプロイ

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/ngram-spec-test
bash scripts/rdma-build.sh local
bash scripts/rdma-deploy.sh
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

### RDMA + ngram-simple テスト

上記コマンドに `--spec-type ngram-simple --spec-ngram-size-n 4 --spec-ngram-size-m 8 -v` を追加。

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

### RPC + ngram テスト

上記コマンドに `--spec-type ngram-simple --spec-ngram-size-n 4 --spec-ngram-size-m 8 -v` を追加。

## テスト結果

### テスト条件
- **モデル**: GLM-4.7 IQ2_M (unsloth/GLM-4.7-GGUF:UD-IQ2_M)
- **構成**: 11GPU (7 CUDA + 4 リモート)
- **生成長**: 2048 トークン
- **コンテキスト**: 4096
- **プロンプト**: 量子コンピューティングの詳細な説明 (英語、~60 トークン)
- **Seed**: 42
- **ngram パラメータ**: n=4 (lookup gram), m=8 (draft gram) — デフォルト (n=12, m=48) ではドラフトが一切生成されなかったため縮小

### ベースライン

| # | バックエンド | Prompt (t/s) | Generation (t/s) |
|---|------------|:------------:|:----------------:|
| 3a | RDMA (7C+4R) | 13.5 | 5.1 |
| 3b | RPC (7C+4R) | 14.9 | 7.3 |

### Ngram Speculative テスト

| # | バックエンド | Spec Type | pp (t/s) | tg (t/s) | Accepted/Total | Accept Rate |
|---|------------|-----------|:--------:|:--------:|:--------------:|:-----------:|
| 4a | RDMA | ngram-simple | 16.4 | 4.7 | 58/592 | 9.8% |
| 4b | RDMA | ngram-map-k4v | 16.4 | 5.0 | 30/284 | 10.6% |
| 4c | RDMA | ngram-mod | 13.0 | 4.7 | 29/698 | 4.2% |
| 4d | RPC | ngram-simple | 14.9 | 6.1 | 50/629 | 7.9% |
| 4e | RPC | ngram-map-k4v | 16.1 | 6.8 | 23/312 | 7.4% |
| 4f | RPC | ngram-mod | 16.0 | 6.5 | 28/615 | 4.6% |

### Generation 速度変化 (vs ベースライン)

| バックエンド | Spec Type | tg 変化 |
|------------|-----------|:-------:|
| RDMA | ngram-simple | -7.8% |
| RDMA | ngram-map-k4v | -2.0% |
| RDMA | ngram-mod | -7.8% |
| RPC | ngram-simple | -16.4% |
| RPC | ngram-map-k4v | -6.8% |
| RPC | ngram-mod | -11.0% |

## 分析

### 1. Acceptance Rate が低い原因

**デフォルトパラメータ (n=12, m=48) ではドラフト生成ゼロ**。12トークンの完全一致が2048トークンの技術文書では発生しない。n=4 に縮小してもacceptance rate は 4-11% にとどまった。

GLM-4.7 は MoE (Mixture of Experts) モデルであり、以下の特性がngram speculative decodingに不利:
- **Expert routing の確率的性質**: 同じトークン列でも異なるエキスパートがルーティングされ、出力分布が変わる
- **Thinking/reasoning トークン**: GLM-4.7 は `[Start thinking]` タグで構造化推論を行い、各ステップの内容が異なるため繰り返しパターンが少ない
- **高エントロピー出力**: 技術的説明テキストは自然言語の中でも反復が少ない部類

### 2. Generation 速度が低下する理由

Acceptance rate が低い場合、speculative decoding のオーバーヘッドが恩恵を上回る:
- **ドラフトバッチの検証コスト**: 8トークン (m=8) のドラフトバッチをフル forward pass で検証する必要がある
- **リジェクト時の無駄**: acceptance rate 10% では、ドラフトの90%が無駄な計算
- **分散環境でのバッチ検証コスト**: 11GPU にまたがる batch forward pass は、単一トークン forward pass より通信オーバーヘッドが大きい

### 3. Prompt 速度の変動

Speculative decoding は prompt processing に影響しないはずだが、13.0-16.4 t/s と変動が見られる。これはサーバーGPU の計算時間のセッション間変動 (CLAUDE.md に記載の既知課題) と考えられる。

### 4. RDMA vs RPC の傾向

- RPC ベースラインの方が Generation が速い (7.3 vs 5.1 t/s) — デバイスごとの独立ソケットによるコマンド並列化の効果
- Speculative decoding による速度低下は RPC の方が大きい (-6.8% ~ -16.4% vs -2.0% ~ -7.8%) — バッチ検証時の通信コストが RPC の方が高い可能性

## 結論

**Ngram speculative decoding は GLM-4.7 (MoE) モデルでは効果なし。Generation 速度を改善せず、むしろ低下させる。**

- デフォルトパラメータ (n=12, m=48) ではドラフトが一切生成されない
- パラメータ縮小 (n=4, m=8) でもacceptance rate は最大 10.6% にとどまり、速度改善に転じない
- MoE モデルの確率的 Expert routing と高エントロピー出力が根本原因
- **推奨**: MoE モデルでの ngram speculative decoding は使用しない。ドラフトモデルベースの speculative decoding (小規模な Dense モデルをドラフターに使用) の方が有望だが、追加VRAM が必要

### パラメータチューニングの判断

計画では「acceptance rate > 10% の場合にチューニングを実施」としていたが、最高値 10.6% でも Generation 速度の改善が見られないため、チューニングは不要と判断した。Acceptance rate が低すぎるため、パラメータ調整では根本的な改善は望めない。
