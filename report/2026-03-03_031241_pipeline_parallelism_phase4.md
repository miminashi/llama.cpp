# パイプライン並列化 Phase 4: graph_cache マルチスロット化

- **実施日時**: 2026年3月3日 03:12
- **ワークツリー**: `.worktree/pipeline-parallelism`
- **ブランチ**: `feature/pipeline-parallelism`
- **コミット**: `f97fa34f2`

## 前提・目的

Phase 3 (`1a4c7225c`) で FULL_GRAPH パスの同期 flush を排除したが、プロファイリングで **パイプラインモード時に毎 ubatch FULL_GRAPH が実行される** 問題を確認していた。

### 原因

`ggml_backend_sched_alloc_graph` が copy slot をローテーション（n_copies=4 で slot 0→1→2→3→0→...）する。各 slot で `node->src[j]` ポインタが異なるコピーを指すため、`graph_cache::is_cached()` の `src` ポインタ比較が常に失敗し FULL_GRAPH となる。

### 目的

graph_cache を単一エントリからマルチスロット（最大 8 エントリ）に拡張し、各 copy slot 毎にキャッシュエントリを持たせることで、同じ slot の再訪時に RECOMPUTE パスを使用可能にする。

### 参照レポート

- [Phase 1: パイプライン並列化基盤](report/2026-03-02_233154_pipeline_parallelism_phase1.md)
- [Phase 1.5: イベントキャップ実装](report/2026-03-03_004144_pipeline_parallelism_phase1.5_event_caps.md)
- [Phase 2: 機能的イベント実装](report/2026-03-03_013808_pipeline_parallelism_phase2.md)
- [Phase 3: FULL_GRAPH 同期 flush 排除](report/2026-03-03_022921_pipeline_parallelism_phase3.md)

## 変更内容

### graph_cache 構造体のマルチスロット化

**変更前（単一エントリ）**:
```cpp
struct graph_cache {
    std::vector<ggml_tensor> last_graph;
    std::unordered_map<uint64_t, ggml_tensor> snapshot_all_;
    std::unordered_map<uint64_t, const ggml_tensor*> snapshot_map_;
};
```

**変更後（マルチスロット）**:
```cpp
struct graph_cache {
    static constexpr int MAX_SLOTS = 8;
    struct cache_slot {
        std::vector<ggml_tensor> last_graph;
        std::unordered_map<uint64_t, ggml_tensor> snapshot_all;
        std::unordered_map<uint64_t, const ggml_tensor*> snapshot_map;
    };
    std::vector<cache_slot> slots_;
    int active_ = -1;
};
```

### 主な変更点

1. **`is_cached()`**: 全スロットを順に試行し、マッチしたスロットインデックスを `active_` に保存
2. **`add()`**: `active_ < 0`（キャッシュミス）なら新規スロット作成。MAX_SLOTS 超過時は全クリア
3. **`build_snapshot_map()` / `collect_updates()`**: `active_` スロットを操作するよう変更
4. **呼び出し順序変更**: `build_snapshot_map()` → `add()` から `add()` → `build_snapshot_map()` へ（スロット未作成での assertion failure 回避）
5. **プロファイリング**: `slot=X/N` 形式でスロット情報をログに追加

## 再現方法

### 1. ビルド・デプロイ

```bash
bash .worktree/pipeline-parallelism/scripts/rdma-build.sh local
bash .worktree/pipeline-parallelism/scripts/rdma-deploy.sh
gpu-lock.sh run bash .worktree/pipeline-parallelism/scripts/rdma-server.sh restart
```

### 2. プロファイリング付き正確性テスト

```bash
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  GGML_RDMA_PROFILE=1 \
  .worktree/pipeline-parallelism/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "Explain RDMA in detail" -n 64 \
  --single-turn --simple-io --log-file /tmp/llama-cli.log 2>&1
```

## 検証結果

### キャッシュ動作

**PP フェーズ（初回、cold start）**: 8 FULL_GRAPH（4 ubatch × 2 RDMA デバイス）

| Call | Node数 | Slot | 動作 |
|------|--------|------|------|
| #1 | 1614 | new → 0 | FULL_GRAPH（RDMA0, ubatch 1） |
| #2 | 1327 | new → 0 | FULL_GRAPH（RDMA1, ubatch 1） |
| #3 | 1614 | new → 1 | FULL_GRAPH（RDMA0, ubatch 2） |
| #4 | 1327 | new → 1 | FULL_GRAPH（RDMA1, ubatch 2） |
| #5 | 1614 | new → 2 | FULL_GRAPH（RDMA0, ubatch 3） |
| #6 | 1327 | new → 2 | FULL_GRAPH（RDMA1, ubatch 3） |
| #7 | 1118 | new → 3 | FULL_GRAPH（RDMA0, ubatch 4） |
| #8 | 955 | new → 3 | FULL_GRAPH（RDMA1, ubatch 4） |

**TG フェーズ**: 全て RECOMPUTE でキャッシュヒット（slot=3/4）

### Profile Summary（最終）

```
graph_compute: calls=130, total=115.2 ms, avg=0.9 ms
  type: full=8, update=0, recompute=122
```

- 初回 8 FULL_GRAPH 以降、全 122 回が RECOMPUTE
- RECOMPUTE 1回あたり: pre_send=0.28ms, send=0.01ms（serialize_graph の ~1.5ms を排除）
- 推論出力: 正常（"RDMA (Remote Direct Memory Access)..." の説明文を正しく生成）

### 性能数値

| 指標 | 値 |
|------|-----|
| Prompt | 56.3 t/s |
| Generation | 35.9 t/s |

### 期待される改善効果

2回目以降のプロンプト（llama-bench 等）では全 ubatch がキャッシュヒットし、ubatch あたり ~1.5ms の serialize_graph コストが排除される:

- **pp128 -ub 32**: 4 ubatch × 2 RDMA デバイス × ~1.5ms = **~12ms/prompt 削減**

### 非パイプラインモード（tg）への影響

- copy slot 変化なし → 1 スロットのみ使用 → 現状と同一動作
- デグレなし
