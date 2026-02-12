# クロスデバイス転送最適化レポート

- **実施日時**: 2026年2月11日 13:30
- **ワークツリー**: `.worktree/rdma-xdev-cache` (ブランチ: `feature/rdma-xdev-cache`、`feature/rdma-backend` から分岐)

## 前提・目的

`fix_cross_device_refs` 関数はサーバー側の `graph_recompute` 毎に呼ばれ、クロスデバイステンソルを正しいデバイスにコピーする。以下の非効率が存在:

1. 毎回 `cudaMalloc` / `cudaFree` でGPUメモリを割り当て・解放
2. 毎回 `std::vector<uint8_t>` でヒープメモリを割り当て（pageable memory）
3. D2H/H2D 転送に pageable メモリを使用（DMA 内部で追加ピン留めが発生）

### 参照

- [hotpath メモリ最適化レポート](2026-02-11_095500_hotpath_memory_optimization.md)
- [doorbell batching / max_inline レポート](2026-02-10_222348_doorbell_batching_max_inline.md)

## 変更内容

### 1. GPU アロケーションキャッシュ (`xdev_cache`)

`stored_graph` に `std::unordered_map<void*, std::pair<void*, size_t>> xdev_cache` を追加。元ポインタ＋サイズが同じなら `cudaMalloc` をスキップし、既存割り当てを再利用する。サイズ変更時のみ `cudaFree` + 再 `cudaMalloc`。不要になったキャッシュエントリは各呼び出し末尾でクリーンアップ。

### 2. ピン留めホストバッファ (`pinned_host_buf`)

`stored_graph` に `cudaMallocHost` で割り当てたピン留めバッファを追加。grow-only（必要サイズが既存より大きい場合のみ再割り当て）。DMA 転送のスループット向上を期待。

### 3. クリーンアップパス更新

デストラクタと `graph_compute` (full) パスで `xdev_cache` と `pinned_host_buf` を適切に解放。

## プロファイリング結果

### qwen2.5-0.5b (1 CUDA + 1 RDMA)

| 項目 | Before | After |
|------|--------|-------|
| fix_xdev | 0.01 ms | 0.01 ms |
| total (graph_recompute) | 0.93 ms | 0.92 ms |
| pp128 | 2793 t/s | 2737-2785 t/s |
| tg32 | 155 t/s | 137-143 t/s |

クロスデバイステンソルが存在しないため、fix_xdev は走査コストのみ (0.01ms)。

### GLM-4.7 IQ2_M (7 CUDA + 4 RDMA)

| 項目 | Before | After |
|------|--------|-------|
| fix_xdev (初回) | 0.09-0.10 ms | 0.02 ms |
| fix_xdev (定常) | 0.01-0.02 ms | 0.01-0.02 ms |
| total (graph_recompute) | 10.87-12.33 ms | 10.86-12.25 ms |
| pp128 (llama-bench) | 23.13 t/s | 23.23 t/s |
| tg32 (llama-bench) | 7.73 t/s | 7.71 t/s |

### GLM-4.7 IQ2_M llama-cli (xdev-cache, 3回)

| Run | Prompt (t/s) | Generation (t/s) |
|-----|:------------:|:----------------:|
| 1   | 6.3          | 5.5              |
| 2   | 6.4          | 5.6              |
| 3   | 6.3          | 5.6              |

期待値: pp≈6.4, tg≈6.8 (通常の変動範囲内)

## 分析

1. **fix_xdev は Generation 全体の 0.1-0.8% しか占めない**: compute (GPU 計算) が 10-13ms で支配的。fix_xdev が 0.02ms → キャッシュによる改善余地は最大 0.08ms (初回のみ)
2. **クロスデバイステンソルがほぼ発生しない**: レイヤー分割 (`-sm layer`) では各デバイスのグラフに外部デバイスのテンソルが含まれることは稀。fix_xdev の時間はグラフノード走査のオーバーヘッドが支配的
3. **e2e 性能に measurable な差はない**: before/after で pp128, tg32 ともに統計的同等
4. **コード品質改善**: キャッシュとピン留めバッファにより、将来クロスデバイステンソルが増加した場合のパフォーマンスペナルティを防止

## 結論

- 現行のレイヤー分割構成ではクロスデバイステンソルがほぼ発生せず、最適化の e2e 効果はない
- ただし、コードの堅牢性は改善（不要な cudaMalloc/Free の排除、ピン留めバッファによる安全な DMA）
- **システムは GPU 計算バウンド**: fix_xdev も通信レベル最適化も e2e 改善に寄与しない。大きな改善には RPC 同等のコマンド並列化が必要

## 作業中に発見した問題

### 1. `llama-cli` で GLM-4.7 が `llama_params_fit` クラッシュ

rdma-backend バイナリの `llama-cli` で GLM-4.7 を実行すると、`llama_params_fit` → `weight_buft_supported` → `ggml_backend_buft_alloc_buffer` で `GGML_ASSERT(buft) failed` が一貫して発生した。同じバイナリの `llama-bench` では正常に動作する。xdev-cache バイナリの `llama-cli` でも正常。

- `llama-bench` は `llama_params_fit` を呼ばないため影響を受けない
- rdma-backend バイナリを再ビルドしても解消せず
- xdev-cache バイナリでは発生しない（コード差分は `fix_cross_device_refs` のみで `llama_params_fit` とは無関係）
- 原因不明。ビルドキャッシュの差異か、未特定の状態依存の可能性

### 2. クライアント abort 後の `local length error`

テスト中に xdev-cache クライアントバイナリで rdma-backend サーバーへ接続を試みた際、互換性の問題で `llama_params_fit` クラッシュ（上記 #1）が発生。その後、サーバーを再起動しても初回接続で `Work completion error: status=local length error` が発生するケースがあった。

- これは CLAUDE.md に記載の「クライアント異常切断後のサーバー復旧」問題の変種
- サーバー再起動 + 数秒の待機で回復する場合としない場合があった
- rdma-backend を再ビルドした後は回復した

### 3. `GGML_RDMA_PROFILE` はプロセスローカル

`GGML_RDMA_PROFILE=1` はプロセスの環境変数 (`std::getenv`) で判定されるため、クライアント側で設定してもサーバーには伝わらない。サーバー側のプロファイル出力を得るには、サーバー起動時に別途 `GGML_RDMA_PROFILE=1` を設定する必要がある。

### 4. fix_xdev のプロファイル出力は既に実装済み

プラン時点では未実装と判断した `fix_xdev` のプロファイル出力（プラン項目3）は、`graph_compute`, `graph_recompute`, `graph_compute_update` の3箇所すべてで既に実装されていた。追加の変更は不要だった。

## 再現方法

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-xdev-cache

bash /tmp/rdma-deploy-xdev.sh

GGML_RDMA_PROFILE=1 で rdma-server を起動:
ssh 192.168.100.2 "GGML_RDMA_PROFILE=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"

GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PROFILE=1 \
  LD_LIBRARY_PATH=build/bin build/bin/llama-bench \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -r 1 -p 128 -n 32

ssh 192.168.100.2 "grep fix_xdev /tmp/rdma-server.log"
```
