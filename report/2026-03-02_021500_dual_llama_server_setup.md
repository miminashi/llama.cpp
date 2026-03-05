# 2プロセス構成 llama-server 起動レポート

- **実施日時**: 2026年3月2日 02:15
- 作業者: Claude
- ブランチ: feature/rdma-backend

## 概要

11GPU (7 CUDA + 4 RDMA) を2つの独立した llama-server プロセスに分割し、同一モデル (Qwen3.5-35B-A3B UD-Q4_K_M) を2ポートで提供する構成を起動した。

## 構成

| | Process 1 (CUDA only) | Process 2 (CUDA + RDMA) |
|---|---|---|
| GPU | Node 1: CUDA0-4 (5基) | Node 1: CUDA5-6 (2基) + Node 2: RDMA0-3 (4基) = 6基 |
| Port | 8080 | 8081 |
| CUDA_VISIBLE_DEVICES | 0,1,2,3,4 | 5,6 |
| RDMA | なし | 192.168.100.2:50051 |
| モデルバッファ合計 | ~18.9 GiB | ~18.9 GiB |
| KV キャッシュ | 2560 MiB (5 GPU分) | 2560 MiB (6 GPU分) |
| n_parallel | 4 (auto) | 4 (auto) |
| ctx_size | 131072 | 131072 |
| flash_attn | enabled | enabled |

## 起動コマンド

### Process 1 (port 8080)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  ./build/bin/llama-server \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4' \
  -sm layer -ngl 999 -fa 1 \
  --ctx-size 131072 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --host 0.0.0.0 --port 8080 --jinja
```

### Process 2 (port 8081)

```bash
CUDA_VISIBLE_DEVICES=5,6 \
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  ./build/bin/llama-server \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 \
  --ctx-size 131072 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --host 0.0.0.0 --port 8081 --jinja
```

## VRAM 使用量

### Process 1 (5 GPU)

| GPU | モデル | KV | RS | Compute | 合計 |
|-----|--------|-----|-----|---------|------|
| CUDA0 | 4108 MiB | 512 MiB | 59 MiB | 1289 MiB | ~5968 MiB |
| CUDA1 | 3652 MiB | 512 MiB | 50 MiB | 1671 MiB | ~5885 MiB |
| CUDA2 | 3652 MiB | 512 MiB | 50 MiB | 1671 MiB | ~5885 MiB |
| CUDA3 | 3652 MiB | 512 MiB | 50 MiB | 1671 MiB | ~5885 MiB |
| CUDA4 | 3528 MiB | 512 MiB | 42 MiB | 2043 MiB | ~6125 MiB |

### Process 2 (2 CUDA + 4 RDMA)

| GPU | モデル | KV | RS | Compute | 合計 |
|-----|--------|-----|-----|---------|------|
| CUDA0 (物理5) | 3196 MiB | 256 MiB | 50 MiB | 420 MiB | ~3922 MiB |
| CUDA1 (物理6) | 3195 MiB | 512 MiB | 42 MiB | 233 MiB | ~3982 MiB |
| RDMA0 | 3195 MiB | 512 MiB | 42 MiB | 260 MiB | ~4009 MiB |
| RDMA1 | 3195 MiB | 512 MiB | 42 MiB | 260 MiB | ~4009 MiB |
| RDMA2 | 3196 MiB | 256 MiB | 50 MiB | 260 MiB | ~3762 MiB |
| RDMA3 | 2615 MiB | 512 MiB | 25 MiB | 497 MiB | ~3649 MiB |

## 結果

- 両プロセスとも正常に起動完了
- Process 1: `http://0.0.0.0:8080` でリスニング中
- Process 2: `http://0.0.0.0:8081` でリスニング中
- 両プロセスとも `all slots are idle` を表示し、リクエスト受付可能状態
- マルチモーダルモデル (mmproj-BF16.gguf) も両プロセスでロード済み
- Thinking mode 有効 (`thinking = 1`)

## 停止方法

```bash
pkill -f 'llama-server.*--port 8080'
pkill -f 'llama-server.*--port 8081'
```

## 備考

- `CUDA_VISIBLE_DEVICES` でGPUを完全分離しているため、gpu-lock.sh による排他制御は不要
- Node 2 の RDMA サーバーは Process 2 のみが使用
- 両プロセスで同じモデルファイルをロードしているが、mmap により物理メモリは共有される (CPU_Mapped 333 MiB 部分)
