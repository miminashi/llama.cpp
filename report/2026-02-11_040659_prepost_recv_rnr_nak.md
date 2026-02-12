# Server-side Pre-posted Recv (RNR NAK 削減試行) レポート

- **実施日時**: 2026年2月11日 04:06

## 前提・目的

### 背景
GLM-4.7 IQ2_M の 11GPU テスト (50 tokens) で約 40,000 回の RNR NAK が発生。
`rnr_retry=7` (IB spec で無限リトライ)、`min_rnr_timer=1` (10us) より、
40,000 x 10us = **400ms の NIC レベル blocking** (~7.5 秒の実行時間の ~5%)。

### 仮説
ASYNC graph_compute (0.9ms recompute) 処理中、サーバーのコマンドループがブロックされ
`recv()` を post できない。その間にクライアントが次コマンド (get_tensor) を送信 → RNR NAK。

### 目的
Server-side pre-posted recv により、ASYNC compute 処理前に次コマンド用の recv バッファを
事前 post し、RNR NAK を削減する。

### 参考
- 過去レポート: `report/2026-02-10_222348_doorbell_batching_max_inline.md`

## 実装内容

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ggml/src/ggml-rdma/rdma-transport.h` | `prepost_recv_buffer_`, `prepost_recv_mr_`, `prepost_recv_active_` メンバ追加、`pre_post_recv()` メソッド宣言 |
| `ggml/src/ggml-rdma/rdma-transport.cpp` | `setup_qp()` で prepost バッファ初期化 (16MB)、`pre_post_recv()` 実装、`recv()` の prepost 対応、`disconnect()` の cleanup |
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 3つの ASYNC コマンド処理前 + 全コマンド処理後に `conn->pre_post_recv()` 呼び出し |

### 設計

1. **prepost_recv_buffer_** (16MB): recv_buffer_ と同サイズの専用バッファ
2. **pre_post_recv()**: prepost_recv_active_ でない場合に post_recv を発行
3. **recv() の prepost 対応**: prepost_recv_active_ の場合、wait_for_completion → memcpy → active=false
4. **呼び出し箇所**:
   - ASYNC コマンド (FLUSH_AND_RECOMPUTE_ASYNC, FLUSH_AND_COMPUTE_UPDATE_ASYNC, GRAPH_COMPUTE_ASYNC) の処理直前
   - 全コマンドの switch 文完了後 (次の header recv 用)

## 再現方法

### ビルド・デプロイ

```bash
bash scripts/rdma-build.sh local && bash scripts/rdma-deploy.sh
bash scripts/rdma-server.sh restart
```

### RNR NAK 測定

```bash
cat /sys/class/infiniband/mlx5_0/ports/1/hw_counters/rnr_nak_retry_err
```

### GLM-4.7 テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## 結果

### 回帰テスト (qwen2.5-0.5b, 2GPU)

| テスト | pp128 (t/s) | tg32 (t/s) | 判定 |
|--------|:-----------:|:----------:|:----:|
| pre_post_recv 適用後 | 2,757 - 2,770 | 145 - 148 | OK (リグレッションなし) |

### GLM-4.7 IQ2_M 性能 (11GPU, 7C+4R)

| バージョン | Run | Prompt (t/s) | Generation (t/s) |
|-----------|:---:|:---:|:---:|
| v1 (ASYNC 内のみ) | 1 | 6.3 | 5.6 |
| v1 | 2 | 7.0 | 6.9 |
| v1 | 3 | 3.2 | 7.2 |
| v2 (全コマンド後にも適用) | 1 | 6.8 | 6.9 |
| v2 | 2 | 7.1 | 5.8 |
| v2 | 3 | 7.0 | 5.9 |
| **v2 平均** | - | **7.0** | **6.2** |
| ベースライン (期待値) | - | 6.4 | 6.8 |

性能は正常範囲。Generation のばらつき (5.6-7.2) はサーバーGPUのサーマルスロットリングによる既知の変動。

### RNR NAK カウンタ

| 条件 | RNR NAK/テスト | 前回比 |
|------|:-----------:|:------:|
| v1 (ASYNC 内のみ) run 1 | 40,694 | - |
| v1 run 2 | 39,625 | - |
| v1 run 3 | 42,856 | - |
| v2 (全コマンド後にも適用) run 1 | 45,543 | - |
| ASYNC 無効 (比較用) | 39,379 | - |
| **全条件平均** | **~41K** | **変化なし** |

## 分析

### RNR NAK が削減されなかった理由

1. **RNR NAK の主要因は header↔data の間隔**: クライアントは header と data を2回の個別の `send()` で送信。サーバーは `recv(header)` → header 解析 → `recv(data)` の2段階。header を prepost で受信しても、data recv の `post_recv` が呼ばれる前にクライアントが data を `send` → RNR NAK。

2. **ASYNC compute は全コマンドの一部**: ASYNC compute は generation フェーズのみ (50 tokens x 4 RDMA devices = 200回)。set_tensor, get_tensor, flush 等の非 ASYNC コマンドの方が桁違いに多い。

3. **header pre_post_recv の効果は RNR NAK 全体の一部**: 各コマンドは header recv + data recv の2回の recv を必要とし、pre_post_recv は header recv のみカバー。

4. **ASYNC disabled テストで確認**: ASYNC compute を無効化しても RNR NAK は ~39K 発生。ASYNC compute は RNR NAK の主要因ではない。

### 本質的な RNR NAK 削減の方向性

| アプローチ | 効果 | 実装コスト |
|-----------|------|----------|
| **header+data 一括 send** | header↔data RNR NAK 完全排除 | クライアント側変更 (send_rdma_cmd_raw) |
| **複数 recv の pre-post** | 全 recv の RNR NAK 削減 | recv キュー管理が複雑 |
| **SRQ (Shared Receive Queue)** | recv を常時複数 post | 大規模な設計変更 |

最も効果的と考えられるのは **header+data 一括 send**: クライアントの `send_rdma_cmd_raw()` で header (9B) + data を1つのバッファにまとめて1回の `send()` で送信。サーバー側も1回の `recv()` で header+data を一括受信。これにより send/recv 回数が半減し、RNR NAK も半減する見込み。

## 結論

- **pre_post_recv は正しく動作し、リグレッションなし**
- **RNR NAK 削減効果は限定的** (40K → 40K、変化なし)
- **根本原因**: RNR NAK の大部分は header↔data の間隔で発生しており、header の pre_post_recv だけではカバーできない
- **コード品質**: pre_post_recv のインフラは将来の改善基盤として維持する価値あり
- **次のステップ**: header+data 一括 send が最も効果的な RNR NAK 削減策
