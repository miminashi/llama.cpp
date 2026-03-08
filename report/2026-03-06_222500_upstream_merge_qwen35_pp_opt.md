# Upstream マージ デグレテスト: feature/qwen35-pp-optimization

- 日付: 2026-03-06
- ブランチ: `feature/qwen35-pp-optimization`
- ワークツリー: `.worktree/qwen35-pp-opt`
- コミット: `6fbec8dc0`

## 目的

`feature/qwen35-pp-optimization` ブランチに upstream/master の最新 36 コミット (`05728db18..f5ddcd169`) をマージし、性能デグレがないことを検証する。

注目すべき upstream 変更:
- `ggml-backend.cpp`: sync 削減、`tensor_copy_async` 導入
- `ggml-cuda.cu`: CPU→CUDA async copy 対応

## マージ

```
git merge upstream/master -m "Merge upstream/master (f5ddcd169) into feature/qwen35-pp-optimization"
```

- コンフリクト: `README.md` のみ (Hexagon パス変更 + RDMA 行保持で解消)
- `tools/cli/cli.cpp`, `tools/server/server-context.cpp` は自動マージ成功

## テスト結果

### テスト A: Qwen3.5-35B-A3B (4GPU: 2C+2R)

構成: Node 1 CUDA4,5 + Node 2 RDMA0,1

| メトリクス | 今回 (t/s) | 期待値 (t/s) | 差分 |
|-----------|:----------:|:----------:|:----:|
| pp128 | 206.9 | ~204 | **+1.4%** |
| pp512 | 330.3 | ~332 | -0.5% |
| pp2048 | 371.1 | ~376 | -1.3% |
| tg32 | 36.3 | ~36 | +0.9% |
| tg128 | 36.4 | ~36 | +1.1% |

### テスト B: GLM-4.7 IQ2_M (11GPU: 7C+4R)

構成: Node 1 CUDA0-6 + Node 2 RDMA0-3

| メトリクス | 今回 (t/s) | 期待値 (t/s) | 差分 |
|-----------|:----------:|:----------:|:----:|
| pp128 | 24.18 | ~24 | +0.7% |
| pp20000 | 31.12 | ~31 | +0.4% |
| tg32 | 8.51 | ~8.5 | +0.1% |

### 正確性テスト (llama-cli)

プロンプト: `"Explain RDMA in three sentences."`

| モデル | 構成 | 結果 |
|--------|------|------|
| Qwen3.5 (4GPU) | 2C+2R | 正常 (thinking + 意味のある応答, pp=63.0, tg=36.0) |
| GLM-4.7 (11GPU) | 7C+4R | 正常 (thinking + 意味のある応答, pp=5.0, tg=8.6) |

ゴミ文字、NaN、出力破損なし。

## 判定

全メトリクスで -2% 以上の劣化なし、正確性テストも正常。**合格**。

upstream の sync 削減 (`2cd20b72e`) や async copy 対応は RDMA バックエンドの性能・正確性に悪影響を与えていない。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `6fbec8dc0 (dirty) (feature/qwen35-pp-optimization)` | -- |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 34C | 35C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | -- | running (PID 1518167) |

GGML_RDMA 環境変数: (none)

## 再現手順

```bash
bash scripts/rdma-build.sh local
gpu-lock.sh run bash scripts/rdma-deploy.sh
gpu-lock.sh run bash scripts/rdma-server.sh restart

# Qwen3.5 PP
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -fa 1 -p 128,512,2048 -n 0 -r 5 -o csv \
  -dev 'CUDA0/CUDA1/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]'

# Qwen3.5 TG
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_Qwen3.5-35B-A3B-GGUF_Qwen3.5-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -fa 1 -p 0 -n 32,128 -r 5 -o csv \
  -dev 'CUDA0/CUDA1/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]'

# GLM-4.7 PP
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -fa 1 -p 128,20000 -n 0 -r 5 -o csv

# GLM-4.7 TG
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -fa 1 -p 0 -n 32 -r 5 -o csv
```
