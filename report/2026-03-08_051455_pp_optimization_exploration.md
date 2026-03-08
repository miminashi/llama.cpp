# Qwen3.5 PP性能改善 探索レポート

- **実施日時**: 2026年3月8日 05:14
- **ワークツリー**: N/A (調査・分析のみ)
- **対象モデル**: Qwen3.5-35B-A3B Q4_K_M (21GB), Qwen3.5-122B-A10B Q4_K_M (72GB)
- **リソース**: P100 x11 (7C+4R), 176GB VRAM, 100GbE RDMA

## 目的

これまでの施策 ([振り返りレポート](2026-03-08_042629_pp_optimization_retrospective.md) 参照) を踏まえ、11GPU / 176GB VRAMという潤沢なリソースと、通信がボトルネックでない状況を活かした新たなPP改善策を探索する。

## 調査チーム構成

3エージェントによる並行調査:

| エージェント | 担当 | 主な調査対象 |
|---|---|---|
| cuda-compute-path | CUDA計算パス分析 | `mul_mat_id` 実装、stream sync、P100固有パス |
| gpu-scaling-strategy | GPUスケーリング・メモリ戦略 | 122Bモデル構成、余剰VRAM活用、Data/Expert Parallelism |
| upstream-literature | 上流・文献調査 | upstream MoE最適化、Qwen3.5アーキテクチャ |

## 重要な発見

### 1. Qwen3.5 MoEはdelta-net hybridモデル (75%がrecurrent層)

`src/llama-model.cpp:2556-2561` より、`full_attn_interval = 4` で全層の75%がrecurrent (delta-net/linear attention) 層:

| モデル | 総レイヤー数 | Recurrent層 | Full Attention層 | Recurrent比率 |
|--------|:-----------:|:-----------:|:----------------:|:------------:|
| Qwen3.5-35B-A3B | 40 | 30 | 10 | 75% |
| Qwen3.5-122B-A10B | 48 | 36 | 12 | 75% |

- **Recurrent層**: delta-net linear attention + SSM conv (`build_layer_attn_linear`)
- **Full Attention層**: 標準的なQKV attention + RoPE (`build_layer_attn`)
- **MoE FFN**: 全層で使用 (recurrent/attention共通)

`src/models/qwen35moe.cpp:35-41`:
```cpp
if (hparams.is_recurrent(il)) {
    cur = build_layer_attn_linear(inp->get_recr(), cur, il);
} else {
    cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
}
```

### 2. P100での `mul_mat_id` パス選択: 常にCPUソートfallback

`ggml/src/ggml-cuda/ggml-cuda.cu:2268-2423` の分岐解析:

| パス | 条件 | P100 (cc 6.0) + Q4_K_M |
|------|------|:-:|
| **MMVQ** (GPU native) | quantized && ne2 <= 4 | tgのみ (pp128では ne2=128 > 4) |
| **MMQ** (DP4A) | cc >= 6.1 (`GGML_CUDA_CC_DP4A=610`) | **不可** (cc 6.0 < 6.1) |
| **MMF** (FP native) | !quantized | **不可** (Q4_K_Mはquantized) |
| **cuBLAS fallback** | 上記すべて不成立 | **常にこのパス** |

cuBLAS fallbackパス (`ggml-cuda.cu:2309-2423`) の処理:
1. IDs D→H コピー + **cudaStreamSynchronize** (line 2337-2338)
2. CPU上でexpert別トークンソート (line 2340-2353)
3. ソート結果 H→D コピー + **cudaStreamSynchronize** (line 2358-2359)
4. Expert別ループ: 各expertごとに個別 `ggml_cuda_mul_mat` 呼び出し (line 2372-2417)

Qwen3.5-35B-A3B の expert数は64 (top-2 routing) → 各 `mul_mat_id` 呼び出しで2回のstream sync + 最大64回のcuBLAS GEMM。

### 3. 122Bモデルのスケーリング予測

| 項目 | 35B-A3B | 122B-A10B |
|------|:-------:|:---------:|
| Active params | 3B | 10B |
| Total params | 35B | 122B |
| Model size (Q4_K_M) | 21GB | 72GB |
| 最小GPU数 | 2 (32GB) | 5 (80GB) |
| Active/GPU (4GPU) | 0.75B | N/A |
| Active/GPU (11GPU) | 0.27B | 0.91B |

35Bが4GPU→8GPUで負のスケーリングだった理由: GPU追加でactive params/GPUが減少 → 通信・同期オーバーヘッドが支配的に。

**122Bの見込み**: Active paramsが3.3倍 → GPU当たり計算量が多い → 8-11GPUでもスケーリングが改善する可能性が高い。

### 4. 上流エージェント結果の検証

以下のコミット/機能は**存在しない** (エージェントの幻覚):
- Gate+Up Expert融合 (`--fuse-gate-up-exps`) — 存在せず
- Expert Remap IDs (`GGML_OP_EXPERT_REMAP_IDS`) — 存在せず
- CUDA Graph Warmup遅延 — P100はCUDA Graph非対応 (cc < 7.0)

以下は**実在**するが適用範囲に注意:
- SSM Conv Shared Memory (`1e38a7a6f`): merge/upstream-20260306にマージ済み。Qwen3.5のrecurrent層 (75%) に関連するが、PPボトルネックはSSM convではなくMoE FFN (mul_mat_id)
- MMVQ multi-token template: ne2 <= 4の場合のみ → PP (ne2 >> 4) には無関係

---

## 改善策一覧 (優先度順)

### Tier 1: すぐに実験可能 (コード変更なし)

#### 1-A. ubatch超大サイズスイープ (Qwen3.5-35B, 4GPU)

| 項目 | 内容 |
|------|------|
| **概要** | 余剰VRAM (43GB) を活用して -ub 4096, 8192 を試行 |
| **根拠** | -ub 2048で+50-70%の実績。cuBLASタイル充填率はubatch増加で継続改善の可能性 |
| **期待効果** | pp2048+: さらに+5-20% (飽和曲線の確認) |
| **リスク** | 低。VRAM不足ならエラーで停止するだけ |
| **既知の参考値** | Qwen3.5-35B 4GPU: pp128=200 t/s, pp16384(ub=2048)=418 t/s |

> **注: 実施済み** — [pp16384全最適化ベンチマーク](2026-03-07_074630_qwen35_pp16384_all_optimizations.md) にて同一モデル・同一4GPU構成で ub=512/2048/4096/8192/16384 をスイープ済み。結果: ub=2048 で飽和 (+18.9%)、ub=4096 (+19.1%)、ub=8192 (+19.4%) と 2048 以降の追加効果は <0.5%。ただし当該実験は pp16384 固定であり、本施策の pp2048 での ub=4096/8192 は未検証のため、念の為再度実験する価値はある。

**前提条件**:
- 1号機でビルド済み: `bash scripts/rdma-build.sh local`
- 2号機にデプロイ済み: `gpu-lock.sh run bash scripts/rdma-deploy.sh`
- rdma-server 起動済み: `gpu-lock.sh run bash scripts/rdma-server.sh restart`

**実行手順**:

```bash
MODEL=/home/ubuntu/.cache/huggingface/hub/models--unsloth--Qwen3.5-35B-A3B-GGUF/snapshots/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-Q4_K_M.gguf

DEV='CUDA4/CUDA5/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]'

gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-bench \
  -m $MODEL -dev "$DEV" \
  -ngl 999 -sm layer -fa 1 \
  -ub 512 -p 2048 -n 0 -r 3

gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-bench \
  -m $MODEL -dev "$DEV" \
  -ngl 999 -sm layer -fa 1 \
  -ub 1024 -p 2048 -n 0 -r 3

gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-bench \
  -m $MODEL -dev "$DEV" \
  -ngl 999 -sm layer -fa 1 \
  -ub 2048 -p 2048 -n 0 -r 3

gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-bench \
  -m $MODEL -dev "$DEV" \
  -ngl 999 -sm layer -fa 1 \
  -ub 4096 -p 4096 -n 0 -r 3

gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-bench \
  -m $MODEL -dev "$DEV" \
  -ngl 999 -sm layer -fa 1 \
  -ub 8192 -p 8192 -n 0 -r 3
```

**注意**: pp サイズ >= ub サイズでないと複数ubatchにならない。`-p 2048 -ub 4096` は意味がないので `-p 4096 -ub 4096` とする。

**成功基準**: ub=2048以降も改善が続くか、飽和するかを確認。飽和曲線をプロットする。

#### 1-B. Qwen3.5-122B-A10B ベースライン (11GPU)

| 項目 | 内容 |
|------|------|
| **概要** | 122Bモデルを11GPUで初めてベンチマーク。動作確認とベースライン取得 |
| **根拠** | Active params 10B → GPU追加のスケーリングが35B (3B active) より改善するはず |
| **期待効果** | 35Bの負のスケーリングが解消 → 11GPU活用の正当化 |
| **前提** | 72GB model → 最低5GPU (80GB VRAM)、KVキャッシュ考慮で9GPU以上推奨 |

**前提条件**:
- 122Bモデルがダウンロード済み (3ファイル分割)
- マルチファイルGGUFのシンボリックリンク作成

**シンボリックリンク作成** (未作成の場合):
```bash
mkdir -p /tmp/Qwen3.5-122B-A10B-Q4_K_M/Q4_K_M
SNAP=/home/ubuntu/.cache/huggingface/hub/models--unsloth--Qwen3.5-122B-A10B-GGUF/snapshots/51eab4d59d53f573fb9206cb3ce613f1d0aa392b/Q4_K_M
ln -sf $SNAP/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf /tmp/Qwen3.5-122B-A10B-Q4_K_M/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf
ln -sf $SNAP/Qwen3.5-122B-A10B-Q4_K_M-00002-of-00003.gguf /tmp/Qwen3.5-122B-A10B-Q4_K_M/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00002-of-00003.gguf
ln -sf $SNAP/Qwen3.5-122B-A10B-Q4_K_M-00003-of-00003.gguf /tmp/Qwen3.5-122B-A10B-Q4_K_M/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00003-of-00003.gguf
```

**実行手順 (11GPU: 7C+4R)**:
```bash
MODEL122=/tmp/Qwen3.5-122B-A10B-Q4_K_M/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf

DEV11='CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]'

gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 \
  build/bin/llama-bench \
  -m $MODEL122 -dev "$DEV11" \
  -ngl 999 -sm layer -fa 1 \
  -p 128,512,2048 -n 32 -r 3
```

**動作確認 (llama-cli で推論テスト)**:
```bash
gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m $MODEL122 \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --flash-attn on --no-warmup --single-turn --simple-io \
  --log-file /tmp/llama-cli.log
```

**成功基準**: まず正常に推論が完了すること。次にpp/tg数値のベースライン取得。

#### 1-C. Qwen3.5-122B GPU数スイープ

| 項目 | 内容 |
|------|------|
| **概要** | 122Bモデルを異なるGPU数で比較し、スケーリング特性を確認 |
| **テスト構成** | 6GPU (6C), 7GPU (7C), 8GPU (4C+4R), 9GPU (5C+4R), 11GPU (7C+4R) |
| **依存** | 1-B の結果 (11GPUで動作することを確認済み) |

72GBモデルなので最低5GPU (80GB) が必要。KVキャッシュ分を考慮し6GPUから開始:

```bash
MODEL122=/tmp/Qwen3.5-122B-A10B-Q4_K_M/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf

DEV6='CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5'
gpu-lock.sh run env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  build/bin/llama-bench \
  -m $MODEL122 -dev "$DEV6" \
  -ngl 999 -sm layer -fa 1 \
  -p 128,512 -n 32 -r 3

DEV7='CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6'
gpu-lock.sh run env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  build/bin/llama-bench \
  -m $MODEL122 -dev "$DEV7" \
  -ngl 999 -sm layer -fa 1 \
  -p 128,512 -n 32 -r 3

DEV8='CUDA0/CUDA1/CUDA2/CUDA3/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]'
gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  build/bin/llama-bench \
  -m $MODEL122 -dev "$DEV8" \
  -ngl 999 -sm layer -fa 1 \
  -p 128,512 -n 32 -r 3

DEV11='CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]'
gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 \
  build/bin/llama-bench \
  -m $MODEL122 -dev "$DEV11" \
  -ngl 999 -sm layer -fa 1 \
  -p 128,512 -n 32 -r 3
```

**注意**: RDMAデバイスを使う構成では `GGML_RDMA_SERVERS` が必要。CUDAのみの構成では不要。

**成功基準**: 35Bでは4→8GPUで-0.4%〜-10.4%劣化したが、122Bでは改善が見られるか。

#### 1-D. Qwen3.5-122B ubatchスイープ

| 項目 | 内容 |
|------|------|
| **概要** | 122Bモデルでの最適ubatchサイズ特定 |
| **依存** | 1-B の結果 (11GPUベースライン) |
| **テスト範囲** | -ub 256, 512, 1024, 2048 |

```bash
MODEL122=/tmp/Qwen3.5-122B-A10B-Q4_K_M/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf

DEV11='CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]'

for UB in 256 512 1024 2048; do
  gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 \
    build/bin/llama-bench \
    -m $MODEL122 -dev "$DEV11" \
    -ngl 999 -sm layer -fa 1 \
    -ub $UB -p 2048 -n 0 -r 3
done
```

**注意**: ubatch は pp サイズ以下にすること。122Bは11GPUでもVRAM余裕が少ない (~104GB free - KV cache) ため、超大ubatch (4096+) はOOMの可能性がある。まず2048まで試す。

**成功基準**: 35Bと同様にub=2048で飽和するか、より早く飽和するかを確認。

---

### Tier 2: 上流マージ検証 (低リスク)

#### 2-A. merge/upstream-20260306 での Qwen3.5 ベンチマーク

| 項目 | 内容 |
|------|------|
| **概要** | upstream マージ済みブランチに含まれるSSM conv shared mem等の効果を測定 |
| **対象ブランチ** | `merge/upstream-20260306` (commit `e674c4209`) |
| **比較ベースライン** | `feature/rdma-backend` での同一構成ベンチマーク |
| **含まれる上流改善** | SSM Conv Shared Memory (`1e38a7a6f`) 他 |

**手順**:
1. `merge/upstream-20260306` ブランチのワークツリーを作成
2. ビルド → デプロイ → サーバー再起動
3. Qwen3.5-35B 4GPU (2C+2R) で llama-bench 実行
4. `feature/rdma-backend` の同一条件結果と比較

```bash
git worktree add .worktree/merge-upstream merge/upstream-20260306

bash /home/ubuntu/projects/llama.cpp/.worktree/merge-upstream/scripts/rdma-build.sh local

gpu-lock.sh run bash /home/ubuntu/projects/llama.cpp/.worktree/merge-upstream/scripts/rdma-deploy.sh
gpu-lock.sh run bash /home/ubuntu/projects/llama.cpp/.worktree/merge-upstream/scripts/rdma-server.sh restart

MODEL=/home/ubuntu/.cache/huggingface/hub/models--unsloth--Qwen3.5-35B-A3B-GGUF/snapshots/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-Q4_K_M.gguf
DEV='CUDA4/CUDA5/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]'

gpu-lock.sh run env GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  /home/ubuntu/projects/llama.cpp/.worktree/merge-upstream/build/bin/llama-bench \
  -m $MODEL -dev "$DEV" \
  -ngl 999 -sm layer -fa 1 \
  -p 128,512,2048 -n 32 -r 5
```

**成功基準**: feature/rdma-backend 比で有意な改善があるか (ABAB paired design + 対応ありt検定)。

> **注: 実施済み** — [upstream マージ デグレテスト](2026-03-06_222500_upstream_merge_qwen35_pp_opt.md) にて `feature/qwen35-pp-optimization` ブランチに upstream 36 コミットをマージし、Qwen3.5-35B 4GPU (2C+2R) および GLM-4.7 11GPU (7C+4R) で検証済み。結果: Qwen3.5 pp128 +1.4%, pp512 -0.5%, pp2048 -1.3% (デグレなし判定)。ただし当該実験は `feature/qwen35-pp-optimization` ブランチでの検証であり、`merge/upstream-20260306` ブランチとはベースが異なるため、念の為再度実験してみる。

---

### Tier 3: コード変更 (中規模)

#### 3-A. mul_mat_id のGPUネイティブIDs reordering

| 項目 | 内容 |
|------|------|
| **概要** | CPUソート (line 2340-2353) + 2回のstream syncをGPUカーネルで置換 |
| **対象ファイル** | `ggml/src/ggml-cuda/ggml-cuda.cu:2309-2423` |
| **既存ヘルパー** | `ggml/src/ggml-cuda/mmid.cu` にGPUヘルパーカーネルが存在 |
| **ボトルネック** | 2回の `cudaStreamSynchronize` (line 2338, 2359) がubatch当たり数百us |
| **期待効果** | mul_mat_id当たり2回のstream sync削除。全MoE層で累積効果 |
| **実装コスト** | 中。既存ヘルパーの活用、ただしexpertループのCPU制御をGPU制御に変換が必要 |
| **リスク** | cuBLAS呼び出し部分はGPU制御困難 → ソートのみGPU化が現実的 |

**現在のコード構造** (`ggml-cuda.cu:2336-2359`):
```cpp
// Stream sync #1: IDs をGPU→CPUコピー
cudaMemcpyAsync(ids_host.data(), ids->data, ..., cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);  // ← 削除対象

// CPU上でexpert別にトークンをソート
for (i02 = 0; i02 < ne02; ++i02) {   // ne02 = 64 experts
    for (i12 = 0; i12 < ne12; ++i12) { // ne12 = n_tokens
        // ...ソートロジック...
    }
}

// Stream sync #2: ソート結果をCPU→GPUコピー
cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), ..., cudaMemcpyHostToDevice, stream);
cudaStreamSynchronize(stream);  // ← 削除対象
```

**改善方針**: CPUソートループをCUDAカーネルに置換し、`ids->data` から直接GPU上で `ids_to_sorted` / `ids_from_sorted` を計算する。`mmid.cu` の `mmid_get_rows` カーネルを参考に、prefix sumベースのexpert offset計算をGPUカーネル化する。

#### 3-B. cuBLAS Batched GEMMによるExpertバッチ化

| 項目 | 内容 |
|------|------|
| **概要** | Expert別ループ (line 2372-2417) を `cublasSgemmBatched` に置換 |
| **対象ファイル** | `ggml/src/ggml-cuda/ggml-cuda.cu:2372-2417` |
| **現状** | 最大64 expert × 個別cuBLAS GEMM → カーネルラウンチオーバーヘッド |
| **期待効果** | カーネルラウンチ回数を最大64→1に削減 |
| **実装コスト** | 中。各expertのトークン数が不均一なためパディングが必要 |
| **リスク** | パディングによる無駄計算 vs ラウンチ削減のトレードオフ |

**現在のコード構造** (`ggml-cuda.cu:2372-2417`):
```cpp
for (int64_t i02 = 0; i02 < ne02; ++i02) {  // ne02 = 最大64 experts
    if (tokens_per_expert[i02] == 0) continue;
    // expertごとにスライスを作成
    ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);  // 個別GEMM
}
```

**改善方針**: `cublasSgemmBatched(handle, ..., A_array, B_array, C_array, batchCount)` で全expertのGEMMを一括実行。各expertの行列ポインタと次元を配列化する。ただしexpertごとにトークン数が異なるため、最大トークン数にパディングが必要。

#### 3-C. F16 dequant + cuBLAS F16 GEMM パス

| 項目 | 内容 |
|------|------|
| **概要** | Q4_K_M → F16にdequantしてcuBLAS HGEMM (FP16) を使用 |
| **根拠** | P100のFP16スループットはFP32の2倍 (21.2 TFLOPS vs 10.6 TFLOPS) |
| **現状** | cuBLAS fallbackはF32で演算。P100のFP16ユニットが未活用 |
| **期待効果** | GEMM部分が最大2倍高速化 (dequant+変換コスト次第) |
| **実装コスト** | 高。dequantカーネル + F16バッファ管理 + cuBLAS HGEMM呼び出し |
| **リスク** | dequant + F16→F32変換オーバーヘッドが利得を相殺する可能性 |

---

### Tier 4: アーキテクチャレベル (大規模・非推奨)

#### 4-A. 余剰VRAMによるモデルレプリカ + Batch分割

**結論: 非推奨** — llama.cppのシングルコンテキスト設計 (`llama_context` = 1スケジューラ + 1メモリ管理) の変更が必要。実装コストが利得を大幅に上回る。

#### 4-B. Expert Parallelism (Expert分散配置)

**結論: 非推奨** — P100 (NVLink無し) の環境ではRDMAレイテンシが相殺。またexpert活性化は動的 (データ依存) のため、コンパイル時の分散配置が困難。`ggml_backend_sched` の再設計が必要。

---

## 「通信が余っている」ことの活用可能性

現在、Qwen3.5 MoEはcompute-bound (GPU計算 94.8%) で通信はボトルネックではない。この「通信の余裕」をどう活かすか:

### 活用できるケース

1. **122Bモデルでの11GPU活用**: 72GBモデルを11GPUに分散 → GPU当たり計算量が十分に大きく、通信余裕を使い切る前にGPU計算がボトルネック化。35Bの「GPU追加で劣化」が122Bでは起きない可能性。

2. **CUDA/RDMA Overlap (既存)**: 複数ubatch時にCUDA計算とRDMAデータ転送をオーバーラップ。通信が余っているため、転送を計算と完全に並列化でき、転送コストがゼロに近づく。

3. **大ubatchでの通信吸収**: ubatch=4096+ではデータ転送量が増加するが、通信帯域に余裕があるため、ubatch大型化の追加通信コストがほぼゼロ。

### 活用が困難なケース

- **モデルレプリカ**: 通信が余っていても、llama.cppのシングルコンテキスト設計がボトルネック
- **Expert Parallelism**: 通信レイテンシ (RTT) は帯域幅とは独立の制約

### 結論

通信余裕の最大の活用法は**GPU数の増加** (122Bモデルで11GPU活用) と**ubatch大型化** (追加通信コストの吸収)。どちらもコード変更なしで実験可能。

---

## 共通の前提手順

すべてのベンチマークに共通する前提:

### ビルド・デプロイ

```bash
bash scripts/rdma-build.sh local
gpu-lock.sh run bash scripts/rdma-deploy.sh
gpu-lock.sh run bash scripts/rdma-server.sh restart
```

### 環境チェック

```bash
bash scripts/rdma-env-check.sh
```

警告がないことを確認する。

### gpu-lock

すべてのGPU使用コマンドは `gpu-lock.sh run` または `gpu-lock.sh wait` で包むこと。直接実行は禁止。

### モデルパス

| モデル | パス |
|--------|------|
| Qwen3.5-35B-A3B Q4_K_M | `/home/ubuntu/.cache/huggingface/hub/models--unsloth--Qwen3.5-35B-A3B-GGUF/snapshots/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-Q4_K_M.gguf` |
| Qwen3.5-122B-A10B Q4_K_M (分割) | `/tmp/Qwen3.5-122B-A10B-Q4_K_M/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf` |

122Bモデルはマルチファイルのため、シンボリックリンク経由で使用する (手順は1-Bを参照)。

### GPU構成テンプレート

| 構成 | -dev (llama-bench, `/`区切り) | 環境変数 |
|------|-------------------------------|----------|
| 4GPU (2C+2R) | `CUDA4/CUDA5/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]` | `GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5` |
| 6GPU (6C) | `CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5` | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` |
| 7GPU (7C) | `CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6` | なし |
| 8GPU (4C+4R) | `CUDA0/CUDA1/CUDA2/CUDA3/RDMA0[...]/RDMA1[...]/RDMA2[...]/RDMA3[...]` | `GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0,1,2,3` |
| 11GPU (7C+4R) | `CUDA0/.../CUDA6/RDMA0[...]/RDMA1[...]/RDMA2[...]/RDMA3[...]` | `GGML_RDMA_SERVERS=192.168.100.2:50051` |

**注意**: llama-cli は `,` 区切り、llama-bench は `/` 区切り。

---

## まとめ

| 優先度 | 施策 | 期待効果 | コスト | 対象モデル |
|:------:|------|:--------:|:------:|:----------:|
| **1** | ubatch > 2048 スイープ (1-A) | +5-20% (pp2048+) | ゼロ | 35B |
| **2** | 122B 11GPUベースライン (1-B) | 初回測定 | ゼロ | 122B |
| **3** | 122B GPU数スイープ (1-C) | スケーリング確認 | ゼロ | 122B |
| **4** | 122B ubatchスイープ (1-D) | 最適ubatch特定 | ゼロ | 122B |
| **5** | 上流マージ検証 (2-A) | +1-5% | 低 | 35B, 122B |
| **6** | GPUネイティブIDs reorder (3-A) | +5-15% (推定) | 中 | 35B, 122B |
| **7** | cuBLAS Batched GEMM (3-B) | +3-8% (推定) | 中 | 35B, 122B |
| **8** | F16 HGEMM パス (3-C) | 最大+50% (理論) | 高 | 35B, 122B |

最も費用対効果が高いのは Tier 1 のベンチマーク群 (コード変更ゼロ)。特に122Bモデルでの11GPU活用は、35Bでは活かしきれなかったGPUリソースを有効化できる可能性がある。
