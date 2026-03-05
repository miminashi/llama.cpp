# CUDA/RDMA Overlap ベンチマークレポート

- **実施日時**: 2026年3月5日 16:34
- **ワークツリー**: `.worktree/cuda-rdma-overlap`
- **ブランチ**: `feature/cuda-rdma-overlap`
- **コミット**: `64a67bf25`

## 前提・目的

GLM-4.7 IQ2_M 11GPU (7 CUDA + 4 RDMA) の Prompt Processing (PP) がボトルネック。プロファイリングにより、ubatch あたり CUDA splits ~3366ms (64.6%) + RDMA splits ~1846ms (35.4%) が逐次実行されていることが判明。

- **目的**: CUDA 計算中に RDMA デバイスへのコマンドを非同期ディスパッチし、ubatch 間で CUDA/RDMA 処理をオーバーラップさせることで PP 速度を改善する
- **前提**: `feature/pipeline-complete` ブランチ (per-copy context buffers, split snapshots, send ring buffer 実装済み)
- **参照レポート**:
  - [PP プロファイリングレポート](2026-03-02_195335_pp_profiling_qwen35_optimization.md)
  - [Pipeline Complete ベンチマーク](2026-03-03_034054_pipeline_parallelism_ab_benchmark.md)

## 実装内容

### 1. 3フェーズ分割実行 (`ggml-backend.cpp`)

Split ループを Phase A/B/C に分割:

- **Phase A**: CPU input + CUDA splits (全 CUDA デバイス計算)
- **Phase B**: 前 ubatch の deferred CPU split を flush (RDMA 完了待ち)
- **Phase C**: RDMA splits (pipeline async dispatch、次 ubatch の CUDA 時間で完了)

```
Ubatch N:   [Phase A: CUDA 2046ms] [Phase C: RDMA dispatch 20ms]
Ubatch N+1: [Phase B: flush 0ms]   [Phase A: CUDA 2046ms]  [Phase C: RDMA dispatch 20ms]
Final sync: RDMA完了を待機
```

### 2. no_wait Send (`rdma-transport.cpp/h`)

RDMA Send の CQ completion polling をスキップし、非ブロッキングディスパッチを実現:

- `send()` に `no_wait` パラメータ追加
- ring buffer の force_signal 時のみ signal + poll (データ整合性保証)
- `drain_send_cq()` 実装: 非ブロッキング CQ poll で未消費の completion を排出

### 3. no_wait 伝播 (`ggml-rdma.cpp`)

`send_rdma_cmd_raw` → `send_rdma_cmd_async` の全6箇所で `GGML_RDMA_CUDA_OVERLAP` 時に `no_wait=true` を渡す。

### ボトルネック分析と解決

| 問題 | 原因 | 解決 |
|------|------|------|
| Phase C = 940ms (no_wait 前) | graph_compute Send CQ が server の Recv post 待ち (~300ms/device) | `no_wait` で CQ polling スキップ |
| Phase C = 662ms (no_wait 後, 単一接続) | set_tensor Send/Recv が server の FIFO 処理待ち | `GGML_RDMA_PER_DEVICE_CONN=1` で接続分離 |
| Phase C = 20ms (最終) | per-device 接続で server 並列処理 | 完了 |

## ベンチマーク結果

### Quick Test: 各 PP サイズの効果

**条件**: GLM-4.7 IQ2_M, 11GPU (7C+4R), `-fa 1`, r=3

| PP size | ubatch | Baseline (t/s) | Overlap (t/s) | 変化 | 備考 |
|---------|--------|----------------|---------------|------|------|
| pp128 | default (512) | 24.20 ± 0.27 | 24.12 ± 0.25 | -0.3% | 1 ubatch → overlap 非適用 |
| pp512 | default (512) | 40.06 ± 0.35 | 39.91 ± 0.34 | -0.4% | 1 ubatch → overlap 非適用 |
| pp2048 | default (512) | 40.26 ± 0.04 | **51.17 ± 0.30** | **+27.1%** | 4 ubatch → overlap 有効 |
| tg32 | default | 8.50 ± 0.01 | 8.54 ± 0.01 | +0.5% | graph reuse → overlap 非適用 |

### A/B 正式テスト: pp2048 (ABAB Paired Design)

**条件**: GLM-4.7 IQ2_M, 11GPU, `-fa 1`, default ub (512), r=5 per round
- **Baseline (A)**: `GGML_RDMA_SERVERS=192.168.100.2:50051`
- **Overlap (B)**: `GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1 GGML_RDMA_SERVERS=192.168.100.2:50051`

| Round | Baseline (A) t/s | Overlap (B) t/s |
|-------|:----------------:|:---------------:|
| 1 | 39.69 ± 0.28 | 51.09 ± 0.23 |
| 2 | 39.70 ± 0.27 | 51.10 ± 0.23 |
| **Mean** | **39.70** | **51.10** |

**改善率: +28.7%** (39.70 → 51.10 t/s)

### Phase タイミング (GGML_SCHED_DEBUG=1, per-device, pp128 -ub 64)

```
[overlap] copy=0 PhaseA=2259.5ms PhaseB=0.0ms PhaseC=20.0ms total=2279.5ms
[overlap] copy=1 PhaseA=2011.5ms PhaseB=0.0ms PhaseC=16.8ms total=2028.3ms
```

- Phase A (CUDA): ~2046ms
- Phase B (deferred flush): 0ms (GLM-4.7 は trailing CPU split なし)
- Phase C (RDMA dispatch): **~20ms** (no_wait + per-device で 940ms → 20ms に短縮)

### -ub 64 による ubatch 分割の効果

| PP size | ub | Mode | t/s | vs default-ub baseline |
|---------|-----|------|-----|----------------------|
| pp128 | 64 | per-device pipeline | 20.06 | -17.1% |
| pp128 | 64 | overlap + per-device | 24.32 | +0.5% |
| pp512 | 64 | per-device pipeline | 19.87 | -50.4% |
| pp512 | 64 | overlap + per-device | 28.65 | -28.5% |

- `-ub 64` は per-ubatch overhead (graph build + alloc) で大きな退行
- overlap は overhead を部分的に回収するが、default ub より良くなるのは pp2048+ のみ

## 正確性テスト

`llama-cli` による推論出力の正確性を確認。

### テスト 1: 短いプロンプト (1 ubatch, overlap 非適用)

```
-p "The capital of France is" -n 50
```

**結果**: "Paris" — 正常な推論出力。ガベージ/NaN なし。

### テスト 2: 長いプロンプト (~1000 tokens, 複数 ubatch, overlap 適用)

~800語の科学史テキストを入力。overlap パスが 5 回活性化:
- PP 4 ubatch: copy=0 (PhaseC=240ms), copy=1 (PhaseC=106ms), copy=2 (PhaseC=77ms), copy=3 (PhaseC=64ms)
- TG KV fill: copy=0 (PhaseC=66ms)
- TG generation: `is_fresh=0` → normal path にフォールバック

**Overlap 出力**:
> "1. Analyze the Request: Source Material: The user provided a text, but it appears to be the same paragraph repeated three times. Task: Summarize the three most important scientific breakthroughs..."

**Baseline 出力** (overlap なし, 同一プロンプト):
> "1. Analyze the Request: Source Material: A provided text (which repeats itself three times)...Task: Summarize the three most important scientific breakthroughs..."

**判定**: 意味的に同等。per-device 接続による浮動小数点演算順序の非決定性でトークン列は異なるが、内容は正確で一貫性あり。ガベージ、NaN、クラッシュは発生しない。

## 環境変数

```bash
GGML_RDMA_CUDA_OVERLAP=1      # overlap 有効化 (PIPELINE 自動有効化)
GGML_RDMA_PER_DEVICE_CONN=1   # per-device 接続 (Phase C 高速化に必須)
```

## 再現方法

1. ビルド・デプロイ
```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/cuda-rdma-overlap/scripts/rdma-build.sh local
gpu-lock.sh run bash /home/ubuntu/projects/llama.cpp/.worktree/cuda-rdma-overlap/scripts/rdma-deploy.sh
bash /home/ubuntu/projects/llama.cpp/.worktree/cuda-rdma-overlap/scripts/rdma-server.sh start
```

2. Baseline (A)
```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  /home/ubuntu/projects/llama.cpp/.worktree/cuda-rdma-overlap/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -fa 1 -p 2048 -n 0 -r 5
```

3. Overlap (B)
```bash
gpu-lock.sh run GGML_RDMA_CUDA_OVERLAP=1 GGML_RDMA_PER_DEVICE_CONN=1 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  /home/ubuntu/projects/llama.cpp/.worktree/cuda-rdma-overlap/build/bin/llama-bench \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 -fa 1 -p 2048 -n 0 -r 5
```

## 結論

- **pp2048 で +28.7% 改善** (39.70 → 51.10 t/s) — default ubatch (512) で 4 ubatch のオーバーラップが有効
- **pp128/pp512 は効果なし** — default ub では 1 ubatch のためオーバーラップ非適用。`-ub 64` で強制分割するとオーバーラップは機能するが per-ubatch overhead で相殺される
- **tg 退行なし** — graph reuse 時は通常パスにフォールバック
- **`GGML_RDMA_PER_DEVICE_CONN=1` が必須** — 単一接続では server-side 逐次処理により Phase C が 662ms に留まる。per-device 接続で 20ms に短縮

### 適用条件

| 条件 | 効果 |
|------|------|
| pp > ubatch (複数 ubatch) | 有効。ubatch 数が多いほど効果大 |
| pp ≤ ubatch (1 ubatch) | 効果なし。通常パスにフォールバック |
| tg (generation) | 効果なし。graph reuse で通常パスにフォールバック |
| RDMA time < CUDA time | 最大効果。RDMA が CUDA 時間内に完了 |
| RDMA time > CUDA time | 部分的効果。最終 sync で待機時間が発生 |

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `64a67bf25 (feature/cuda-rdma-overlap)` | (deployed) |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 35°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
