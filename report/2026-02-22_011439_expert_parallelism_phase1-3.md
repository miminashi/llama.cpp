# Expert Parallelism (EP) Phase 1-3 実装レポート

- **実施日時**: 2026年2月22日 01:14
- **ワークツリー**: `.worktree/expert-parallelism`
- **ブランチ**: `feature/expert-parallelism`
- **コミット**: `a865a29ed`

## 前提・目的

GLM-4.7 は 160 Expert の MoE モデルだが、トークンあたり 8 Expert のみ使用 (5%)。
現在の Pipeline Parallelism (PP) では各 GPU が担当レイヤーの**全 160 Expert** を保持しており、
VRAM の大半が使われない Expert 重みに消費されている。

Expert Parallelism (EP) により Expert 重みを複数 GPU に分散配置することで:
1. **VRAM 削減**: 各 GPU の Expert 重みが 1/N に → Q4_K_M 等の高品質量子化が 16GPU に収容可能に
2. **速度向上の可能性**: Expert 計算の並列化で tg 改善の余地

**スコープ**: Phase 1-3 (新規 Op + モデルロード + グラフ構築) を実装し、ビルドと基本動作を検証する。
RDMA 対応 (Phase 4) は後続。

## 設計概要

### EP + PP ハイブリッド構成

```
GPU 0: PP layers 0-8 (Attention + Shared Expert) + EP experts 0-14 (全89 MoE層)
GPU 1: PP layers 9-17 (Attention + Shared Expert) + EP experts 15-29 (全89 MoE層)
...
GPU 10: PP layers 83-91 (Attention + Shared Expert) + EP experts 145-159 (全89 MoE層)
```

- **Attention + Shared Expert**: PP (レイヤー分割) — 従来通り
- **Routed Experts**: EP (エキスパート分割) — 全 GPU に分散
- Gate weights: 全 GPU に複製 (小さい: 89層 × 0.3MB = 27MB)

### 新規 ggml Op

| Op | 入力 | 出力 | 用途 |
|:---|:---|:---|:---|
| `GGML_OP_EXPERT_REMAP_IDS` | `ids [n_eu, n_tok]` (i32) | 範囲内: `id - offset`, 範囲外: `0` | グローバル→ローカル ID 変換 |
| `GGML_OP_EXPERT_MASK_WEIGHTS` | `ids` (i32) + `weights` (f32) | 範囲内: そのまま, 範囲外: `0.0` | 非ローカル Expert のゼロマスク |

### 1 MoE レイヤーの EP 実行フロー

```
PP GPU (レイヤー担当):
  1. Attention 計算
  2. ルーティング: gate × hidden → softmax → top-8 expert 選択

EP GPU 0..N-1 (ggml_backend_sched が自動ディスパッチ):
  3. expert_remap_ids → ローカル Expert ID
  4. expert_mask_weights → 非ローカル重みをゼロに
  5. MUL_MAT_ID × 3 (up, gate, down) + activation
  6. weighted sum → partial result

PP GPU:
  7. 全 EP グループの partial result を加算
  8. Shared Expert 計算 + 合算
```

## 実装内容

### Phase 1: 新規 ggml Op (2 ops, CPU + CUDA)

**変更ファイル**:
- `ggml/include/ggml.h` — Op 列挙型 + 関数宣言 (+25 行)
- `ggml/src/ggml.c` — Op ファクトリ関数 (+56 行)
- `ggml/src/ggml-cpu/ggml-cpu.c` — CPU 実装 + ディスパッチャ (+76 行)
- `ggml/src/ggml-cuda/expert-remap.cu` — CUDA カーネル (新規, 61 行)
- `ggml/src/ggml-cuda/expert-remap.cuh` — CUDA ヘッダ (新規, 4 行)
- `ggml/src/ggml-cuda/ggml-cuda.cu` — CUDA ディスパッチャ + supports_op (+11 行)

### Phase 2: Expert テンソルシャーディング (モデルロード)

**変更ファイル**:
- `common/arg.cpp` — `--expert-parallel N` CLI パラメータ (+7 行)
- `common/common.h` — `n_expert_parallel` フィールド (+2 行)
- `common/common.cpp` — model params マッピング (+1 行)
- `include/llama.h` — `llama_model_params.n_expert_parallel` (+4 行)
- `src/llama-hparams.h` — `n_expert_parallel` hparams (+1 行)
- `src/llama-model.h` — `ffn_{up,gate,down}_exps_ep` vectors (+5 行)
- `src/llama-model.cpp` — EP テンソル作成 + CPU→GPU シャーディング (+116 行)
- `src/llama.cpp` — hparams 設定 (+5 行)

**シャーディング方式**:
1. 全 Expert テンソルを CPU バッファにロード (GGUF データ充填)
2. ポストプロセスで EP パーティションテンソルを各 GPU に作成
3. CPU テンソルからスライスコピー (Expert dim は最外次元 → 連続メモリ)

### Phase 3: EP 対応グラフ構築

**変更ファイル**:
- `src/llama-graph.h` — `build_moe_ffn_ep()` 宣言 (+18 行)
- `src/llama-graph.cpp` — `build_moe_ffn_ep()` 実装 (+170 行)
- `src/llama-context.cpp` — `graph_max_nodes()` EP 対応 (+5 行)
- `src/models/glm4-moe.cpp` — EP/標準パス分岐 (+42 行, -16 行)

**graph_max_nodes 修正**:
EP=7 では MoE 層あたり ~166 テンソルが生成される (標準の ~33 テンソルの 5 倍)。
`graph_max_nodes()` に EP 用の追加ノード数計算を追加:
```cpp
if (model.hparams.n_expert_parallel > 0) {
    const uint32_t n_moe_layers = model.hparams.n_layer - model.hparams.n_layer_dense_lead;
    const uint32_t nodes_per_ep_group = 10 + 3 * model.hparams.n_expert_used;
    res += n_moe_layers * model.hparams.n_expert_parallel * nodes_per_ep_group;
}
```

## 検証結果

### ビルド

ワークツリーで `bash scripts/rdma-build.sh local` → 成功 (warnings なし)

### 回帰テスト (非 MoE モデル)

qwen2.5-0.5b-instruct-q4_k_m.gguf を EP コード変更後のバイナリで実行:
- **結果: 正常動作** — Prompt: 166.2 t/s, Generation: 207.7 t/s
- EP コードは `n_expert_parallel == 0` (デフォルト) で完全にバイパスされる

### EP=7 テスト (GLM-4.7 IQ2_M, 7×P100)

| テスト | 結果 | 詳細 |
|:---|:---|:---|
| モデルロード | 成功 | Expert シャーディングログ正常出力 |
| グラフ構築 | 成功 | max_nodes=35,308, pool_size=15.1 MB |
| 推論実行 | **VRAM OOM** | CUDA0 compute buffer 1430 MiB 確保失敗 |

**VRAM OOM の原因分析**:
- EP=7 パーティション: 160/7 ≈ 23 experts × 89 MoE 層 × ~7.4 MB = **~15.2 GB/GPU**
- Compute buffer: ~1.4 GB
- 合計: ~16.6 GB > P100 の 16 GB

**結論**: EP=7 on 7×P100 は VRAM 容量の制約で動作不可。
設計通り EP=11 on 11 GPUs で使用する必要がある (~9.9 GB/GPU, 余裕あり)。

### graph_max_nodes 修正の経緯

初期テストで `GGML_ASSERT(obj_new) failed` が発生:
- 原因: EP=7 で MoE 層あたりのテンソル数が 5 倍に増加し、デフォルトの `8*n_tensors` ノード上限を超過
- 修正 1: `(n_expert_parallel - 1) * (8 + 2*n_eu)` → 368 バイト不足
- 修正 2: `n_expert_parallel * (10 + 3*n_eu)` → 十分なマージンで成功

## VRAM 見積もり

### EP=11, IQ2_M, 11 GPUs (設計ターゲット)

| コンポーネント | per GPU | 配置方式 |
|:---|---:|:---|
| Expert パーティション (89層 × ~15 experts × 7.4MB) | ~9.9 GB | EP |
| Attention 重み (~8層 × 43MB) | ~344 MB | PP |
| Shared Expert (~8層 × 59MB) | ~472 MB | PP |
| Gate 重み (89層 × 0.3MB, 複製) | ~27 MB | 全 GPU |
| KV cache + Embedding + その他 | ~500 MB | PP |
| **合計** | **~11.2 GB** | — |

→ P100 16GB に収容可能

### EP=16, Q4_K_M, 16 GPUs (将来目標)

| コンポーネント | per GPU |
|:---|---:|
| Expert パーティション (89層 × 10 experts × 13.3MB) | ~11.8 GB |
| Attention + Shared Expert + その他 | ~1.7 GB |
| **合計** | **~13.5 GB** |

→ P100 16GB に収容可能 (**Q4_K_M が 16GPU で動作可能**)

## 変更規模サマリ

```
18 files changed, 593 insertions(+), 16 deletions(-)
新規ファイル: 2 (expert-remap.cu, expert-remap.cuh)
```

## 後続作業

1. **Phase 4: RDMA バックエンド対応** — 新 Op のシリアライズ対応、サーバー側 EP 実行
2. **EP=11 テスト** — 11 GPU (7C+4R) での動作検証 (RDMA 対応後)
3. **正確性テスト** — EP 有無での出力一致検証
4. **ベンチマーク** — pp128/tg32 での性能測定
5. **ダミー Expert 計算の最適化** — 条件付き MUL_MAT_ID でスキップ

## 再現方法

```bash
git worktree add -b feature/expert-parallelism .worktree/expert-parallelism feature/rdma-backend
cd .worktree/expert-parallelism
bash scripts/rdma-build.sh local

# 非 MoE 回帰テスト
gpu-lock.sh run timeout 30 build/bin/llama-cli \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 99 -p "Hello" -n 5 --log-file /tmp/llama-cli.log -fit off --no-warmup

# EP テスト (11 GPUs 必要)
gpu-lock.sh run build/bin/llama-cli \
  -m /home/ubuntu/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 999 --expert-parallel 11 -p "Hello" -n 20 --log-file /tmp/llama-cli.log -fa 1
```
