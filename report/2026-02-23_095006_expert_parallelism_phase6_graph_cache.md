# Expert Parallelism Phase 6 — グラフキャッシュの EP 対応

- **実施日時**: 2026年2月22日〜23日
- **ワークツリー**: `.worktree/expert-parallelism`
- **ブランチ**: `feature/expert-parallelism`
- **コミット**: `7a6e51c21`

## 前提・目的

### 背景

Phase 5 のプロファイリングで、EP=11 推論の **98% の時間が set_tensor/get_tensor** (テンソルデータ転送) に費やされ、graph_compute (グラフ直列化+計算) は 2% に過ぎないことが判明。根本原因は **グラフキャッシュが EP モードで常にミスし、毎回フルグラフ (171KB-903KB) を直列化・送信** していたこと。

### 目的

1. EP モードでグラフキャッシュを有効化し、graph_compute のコストを削減する
2. キャッシュヒット時はインデックス付きパラメータ更新 (~80KB) のみ送信する
3. 非 EP モードに回帰を起こさない

### 前提条件

- Phase 5 (EP=11, GLM-4.7 IQ2_M) が動作していること
- `.worktree/expert-parallelism` ブランチ上で作業

### 参照レポート

- [Expert Parallelism Phase 1-3](report/2026-02-22_011439_expert_parallelism_phase1-3.md)
- [Expert Parallelism Phase 4 (RDMA)](report/2026-02-22_053859_expert_parallelism_phase4_rdma.md)
- [Expert Parallelism Phase 5 (Performance)](report/2026-02-22_165358_expert_parallelism_phase5_performance.md)

## 技術的課題

### グラフキャッシュのミス原因

`is_cached()` は `n_nodes`, `type`, `op`, `ne[]` を比較するが、EP グラフでは `src[]` ポインタが forward pass 間で変化する（MoE の expert routing により異なるエキスパート重みが選択されるため）。

### MoE 混合量子化の問題

GLM-4.7 IQ2_M は混合量子化を使用しており、同じ (node, src) 位置で異なる量子化タイプが出現する:
- IQ2_M (type=17, nb[0]=74)
- IQ2_S (type=14, nb[0]=82)
- IQ3_S (type=23)
- Q5_1 (type=22)

これにより、キャッシュヒット時に型が変わるとサーバー側の `nb[0]` が不整合になり、CUDA カーネルがクラッシュする。

### 三段階バグのメカニズム

1. **Pass N**: src が type_A → スナップショット type_A を保存
2. **Pass N+1**: src が type_B → type ガードがスキップ (B≠A) → **スナップショットが type_B に更新される**
3. **Pass N+2**: src が type_B → スナップショットと一致 → type_B の nb[] をサーバーに送信 → **サーバーのテンソルはまだ type_A** → `GGML_ASSERT(nb00 == ts_src0) failed`

## 実装

### `rdma_indexed_update` 構造体 (128B)

```cpp
struct rdma_indexed_update {
    uint16_t node_index;     // 2B  — ノード位置 (0..n_nodes-1)
    int8_t   src_index;      // 1B  — -1=ノード自身, 0..9=src[j]
    int8_t   padding;        // 1B
    int32_t  op_params[16];  // 64B
    uint64_t data;           // 8B  (0 = 更新不要)
    int64_t  ne[4];          // 32B
    uint32_t nb[4];          // 16B
    int32_t  flags;          // 4B
};
```

### 主要な変更

1. **`collect_indexed_updates()`**: ノードスナップショットと現在のグラフを比較し、変更されたパラメータのみ `rdma_indexed_update` として収集
2. **`compare_and_emit()`**: 個々のテンソルを比較。**type/op が変わった場合は `false` を返してキャッシュを無効化**
3. **キャッシュ無効化フォールバック**: `collect_indexed_updates()` が `false` を返した場合、フルグラフ送信にフォールバック
4. **`RDMA_CMD_GRAPH_COMPUTE_INDEXED_UPDATE`**: 新コマンド (cmd=30)。サーバー側で既存グラフに差分適用後に計算実行
5. **`RDMA_CMD_FLUSH_AND_INDEXED_UPDATE`**: 新コマンド (cmd=31)。バッファフラッシュ付きインデックス更新

### サーバー側

- `graph_compute_indexed_update()`: インデックス付き更新を受信し、(node_index, src_index) を使ってサーバー側のテンソルを更新
- 更新後に `fix_cross_device_refs()` を呼び出してデバイス間参照を修正
- `ggml_backend_graph_compute()` で計算実行

## テスト結果

### EP=11 テスト (GLM-4.7 IQ2_M, 7 CUDA + 4 RDMA)

```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 \
  .worktree/expert-parallelism/build/bin/llama-cli \
  -m GLM-4.7-UD-IQ2_M.gguf \
  -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

**結果**: クラッシュなしで 20 トークン生成完了

| 指標 | 値 |
|:-----|:---|
| Prompt | 0.1 t/s |
| Generation | ~0.02 t/s |
| graph_compute 回数 | 750 |
| full (フルグラフ送信) | 415 (55.3%) |
| update (インデックス更新) | 335 (44.7%) |
| graph_compute 合計時間 | 32,252 ms (2.5%) |
| set_tensor 合計時間 | 609,651 ms (46.4%) |
| get_tensor 合計時間 | 672,656 ms (51.2%) |
| 総 RDMA 時間 | 1,314,558 ms |

### キャッシュヒット率の分析

44.7% のキャッシュヒット率は、MoE の expert routing による型変化が原因で制限されている。各 tg ステップ (36 graph_compute) の典型的パターン:

| デバイス | パターン | 理由 |
|:---------|:---------|:-----|
| RDMA0 | full→update×4→full(type mismatch)→full(type mismatch)→... | 型ローテーション |
| RDMA1 | full→update×4→full(type mismatch)→... | 同上 |
| RDMA2 | full→update×4→full(type mismatch)→... | 同上 |
| RDMA3 | full→full→...→full | 常にフルグラフ (n_nodes が異なるパターン) |

型ミスマッチの検出パターン:
- `node 151 src 0: type 14 vs 23` — IQ2_S ↔ IQ3_S
- `node 185 src 0: type 22 vs 17` — Q5_1 ↔ IQ2_M
- `node 0 src 0: type 17 vs 22` — IQ2_M ↔ Q5_1
- `node 139 src 0: type 13 vs 23` — Q4_1 ↔ IQ3_S

### 非 EP 回帰テスト (GLM-4.7 IQ2_M, 7 CUDA + 4 RDMA)

```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/expert-parallelism/build/bin/llama-cli \
  -m GLM-4.7-UD-IQ2_M.gguf \
  -ngl 999 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

| 指標 | 値 |
|:-----|:---|
| Prompt | **8.7 t/s** |
| Generation | **8.8 t/s** |

**回帰なし** — 非 EP モードのパフォーマンスは正常範囲内。

## パフォーマンス分析

### グラフキャッシュの効果

| 指標 | フルグラフ送信 | インデックス更新 | 削減率 |
|:-----|:-------------:|:---------------:|:------:|
| graph_compute 時間 | ~2-3 ms | ~1 ms | ~50% |
| 送信データ量 | 171KB-903KB | ~80KB | 55-91% |

graph_compute 自体のコスト削減は明確だが、総 RDMA 時間の 2.5% にしか寄与しないため、全体的なスピードアップは限定的。

### ボトルネック

EP=11 での推論速度が極端に遅い根本原因は、graph_compute ではなく **set_tensor/get_tensor** (テンソルデータ転送) にある:

- `set_tensor`: 609,651 ms — EP 用のエキスパート重み転送
- `get_tensor`: 672,656 ms — 計算結果の取得
- `graph_compute`: 32,252 ms — グラフ構造の送信+計算コマンド

グラフキャッシュはグラフ構造の送信を最適化するが、テンソルデータ転送（重みの set_tensor と結果の get_tensor）は削減できない。EP モードでは各 forward pass で選択されたエキスパートの重みを転送する必要があり、これが支配的なコストとなっている。

### 今後の改善方向性

1. **エキスパート重みのサーバー側キャッシュ**: サーバーが頻繁にアクセスされるエキスパート重みを保持し、転送を省略
2. **重み転送のパイプライン化**: expert routing 結果が確定した時点で重み転送を開始し、計算と重なるようにする
3. **型安定化**: type mismatch によるキャッシュ無効化を減らすため、型ごとにグラフテンプレートを保持する

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `5a7cd6bcb (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 37°C | 39°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 524997) |

## まとめ

EP モードのグラフキャッシュをインデックス付きパラメータ更新で実装した。MoE 混合量子化による型ミスマッチをキャッシュ無効化フォールバックで安全に処理し、クラッシュを防止する。

キャッシュヒット率 44.7% を達成し、graph_compute のコストを約 50% 削減。ただし、graph_compute は総 RDMA 時間の 2.5% にしか寄与しないため、EP=11 の全体的な推論速度への影響は限定的。真のボトルネックはテンソルデータ転送 (set_tensor/get_tensor) であり、これが総時間の 97.5% を占める。

非 EP モードへの回帰はなし (pp=8.7, tg=8.8 t/s)。
