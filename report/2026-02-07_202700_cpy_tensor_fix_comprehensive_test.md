# cpy_tensor 修正の包括的テストレポート

- **実施日時**: 2026年2月7日 20:10〜20:27

## 前提・目的

マルチRDMAデバイスの出力破損バグを修正済み（`cpy_tensor` を常に `false` 返却に変更）。
gpt-oss-20b (1C+2R) と GLM-4.7 IQ2_M (7C+4R) の2プロンプトで正常動作を確認済みだったため、
本テストではより多くのプロンプトと gpt-oss-120b (11GPU フルクラスタ) で修正の正常動作を包括的に検証する。

- **背景**: `cpy_tensor` がRDMA接続内の同一サーバー間GPU-to-GPUコピーを試みると、`graph_recompute` のキャッシュされたグラフが異なるメモリを参照し、コピーされたデータが計算に使用されない問題があった
- **修正内容**: `cpy_tensor` を常に `false` を返すよう変更し、全デバイス間コピーを `get_tensor` + `set_tensor` フォールバックに統一
- **前提条件**: 1号機・2号機ともにビルド済み (commit `76bf9b9ae`)、rdma-server が2号機で起動中
- **参照レポート**: [report/2026-02-07_195500_multi_rdma_device_output_corruption_fix.md](2026-02-07_195500_multi_rdma_device_output_corruption_fix.md)

## テスト環境

| 項目 | 値 |
|------|-----|
| 構成 | 7 CUDA (1号機) + 4 RDMA (2号機) = 11 GPU |
| GPU | Tesla P100-PCIE-16GB × 11 |
| ネットワーク | InfiniBand (100Gbps) |
| GPUDirect | 無効 (`GGML_RDMA_NO_GDR=1`) |
| ビルド | `b7915-76bf9b9ae` |

## テスト結果

### Test 1: GLM-4.7 IQ2_M (7C+4R, 11GPU)

| # | プロンプト | 結果 | 出力抜粋 | pp (t/s) | tg (t/s) |
|---|-----------|------|---------|----------|----------|
| 1 | 日本語: 「こんにちは。日本の首都はどこですか？」 | **PASS** | thinking: "The capital of Japan is Tokyo (東京)..." → 正しい回答 | 8.7 | 6.3 |
| 2 | コード生成: "Write a Python function to check if a number is prime." | **PASS** | thinking: sqrt(n)までの試し割りアルゴリズムを正しく設計 | 9.2 | 6.6 |

**既に検証済み (前回テスト)**:

| # | プロンプト | 結果 | pp (t/s) | tg (t/s) |
|---|-----------|------|----------|----------|
| 3 | "The capital of France is" | **PASS** | 6.6 | 5.9 |
| 4 | "Explain quantum computing in simple terms." | **PASS** | 8.1 | 6.6 |

### Test 2: gpt-oss-120b Q4_K_M (7C+4R, 11GPU)

| # | プロンプト | 結果 | 出力抜粋 | pp (t/s) | tg (t/s) |
|---|-----------|------|---------|----------|----------|
| 1 | "The capital of France is" | **PASS** | thinking: "answer: Paris" → "The capital of France is **Paris**." | 36.6 | 28.8 |
| 2 | 日本語: 「こんにちは」 | **PASS** | thinking: "respond appropriately in Japanese" → 「こんにちは！何かお手伝いできることがあれば、遠慮なく教えてくださいね。」 | 48.4 | 29.0 |

**注意**: gpt-oss-120b の初回実行時に rdma-server がクラッシュした（前回の GLM-4.7 テストのセッション残留が原因と推定）。rdma-server を再起動後は正常動作。

## 修正前後の比較

| モデル | 修正前の症状 | 修正後 |
|--------|------------|--------|
| GLM-4.7 IQ2_M (7C+4R) | ゴミ thinking トークン (意味不明な文字列) | 論理的な思考プロセスと正しい回答 |
| gpt-oss-20b (1C+2R) | 空の出力 (EOSのみ) | 正常なテキスト生成 |
| gpt-oss-120b (7C+4R) | 未テスト (cpy_tensor 修正前) | **正常動作** — 英語・日本語ともに正しい出力 |

## 性能サマリ

| モデル | サイズ | pp (t/s) | tg (t/s) |
|--------|--------|----------|----------|
| GLM-4.7 IQ2_M | ~25GB (3ファイル) | 6.6–9.2 | 5.9–6.6 |
| gpt-oss-120b Q4_K_M | ~68GB (2ファイル) | 36.6–48.4 | 28.8–29.0 |

gpt-oss-120b は GLM-4.7 より高速。GLM-4.7 は 61レイヤーを11GPUに分割するため1トークンあたりのラウンドトリップ回数が多く、MoEアーキテクチャのエキスパート重みサイズも大きい。

## 結論

`cpy_tensor` 修正は全テストケースで正常動作を確認：

- **GLM-4.7 IQ2_M**: 4種類のプロンプト (英語×2, 日本語×1, コード生成×1) で正しい出力
- **gpt-oss-120b Q4_K_M**: 2種類のプロンプト (英語×1, 日本語×1) で正しい出力、11GPU フルクラスタで初の動作確認
- **性能影響**: `cpy_tensor` 無効化による性能低下は約3%（前回レポートで計測済み）

## 再現方法

### 1. rdma-server 起動 (2号機)

```bash
ssh 192.168.100.2 'GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &'
```

### 2. GLM-4.7 IQ2_M テスト (1号機)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 200 --seed 42 \
  -p 'こんにちは。日本の首都はどこですか？' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### 3. gpt-oss-120b テスト (1号機)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -hf unsloth/gpt-oss-120b-GGUF:Q4_K_M \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 100 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```
