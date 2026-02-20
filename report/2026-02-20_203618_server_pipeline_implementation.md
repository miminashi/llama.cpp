# サーバーサイドパイプライン実装レポート

- **実施日時**: 2026年2月20日 20:36
- **ワークツリー**: `.worktree/server-pipeline`
- **ブランチ**: `feature/server-pipeline`
- **コミット**: `09549aca0`

## 前提・目的

RDMA バックエンドの Generation 速度最適化として、graph_compute の ASYNC 送信をパイプライン化する。

- **背景**: デフォルト構成では全リモートデバイス (D0-D3) が1本の共有接続を使用し、`ggml_backend_sched` が各デバイスの graph_compute を逐次呼び出す。D0 の送信は即座に行われるが、D1-D3 は個別に IB send + サーバー recv+parse のオーバーヘッドが発生する (3 × 0.35ms + 3 × 0.2ms = ~1.6ms)
- **目的**: D0 は即送信を維持しつつ、D1-D3 のコマンドをバッファに蓄積し、最終デバイスで一括送信することで IB send/recv オーバーヘッドを削減する
- **期待効果**: tg ~+1% (129ms/token 中 ~1.3ms 削減)、pp128 は中立〜小幅改善

## 設計: "First-immediate, rest-batched" パイプライン

### フロー変更

**変更前**:
```
Client: ASYNC(D0) →[0.35ms]→ ASYNC(D1) →[0.35ms]→ ASYNC(D2) →[0.35ms]→ ASYNC(D3)
Server: recv+parse D0 → compute(D0) → recv+parse D1 → compute(D1) → ...
```

**変更後**:
```
Client: ASYNC(D0) →[0.35ms]→ buffer(D1) →[~0]→ buffer(D2) →[~0]→ buffer(D3)+PIPELINE(D1,D2,D3) →[0.5ms]
Server: recv+parse D0 → compute(D0) → recv+parse PIPELINE → compute(D1) → compute(D2) → compute(D3)
```

### 主な変更内容

| 変更 | 内容 |
|------|------|
| 環境変数 | `GGML_RDMA_NO_PIPELINE=1` で無効化 (デフォルトON) |
| 新コマンド | `RDMA_CMD_PIPELINE_COMPUTE_ASYNC` — 複数 ASYNC ステージをバッチ化 |
| クライアント | `pipeline_send_async()` — D0 即送信、D1+ バッファ蓄積、最終デバイスでフラッシュ |
| サーバー | PIPELINE ハンドラ — n_stages をデシリアライズし各ステージを逐次実行 |
| 安全弁 | `get_tensor` の先頭で `flush_pipeline()` を呼び、未フラッシュのステージを確実に送信 |

### ワイヤーフォーマット

```
PIPELINE_COMPUTE_ASYNC:
| n_stages(4B) | { cmd_type(1B) | stage_size(4B) | stage_data(stage_size B) } × n_stages |
```

### エッジケース

- **RDMA デバイス 1台のみ**: D0 が即送信。パイプラインバッファは空のまま
- **PER_DEVICE_CONN=1**: 接続が分離されるため各接続のパイプラインバッファに1デバイスのみ。D0 即送信で従来動作にフォールバック
- **全デバイスが使われない**: `get_tensor` の安全弁フラッシュで対処

## 変更ファイル

- `ggml/src/ggml-rdma/ggml-rdma.cpp` (+149, -3)

## ビルド結果

```
=== Building on node 1 (local) [Release] ===
CMAKE CONFIGURE OK
BUILD OK
```

## 検証方法 (A/B ベンチマーク)

### 条件

- 条件A (OFF): `GGML_RDMA_NO_PIPELINE=1`
- 条件B (ON): パイプラインデフォルト ON

サーバー再起動不要 (パイプラインはクライアント側バッファリングのみ、サーバーは新コマンドを受信するだけで `static const` env var は関与しない)。

### コマンド例

```bash
gpu-lock.sh run bash scripts/rdma-deploy.sh
gpu-lock.sh run bash scripts/rdma-server.sh restart

# 条件A (OFF)
GGML_RDMA_NO_PIPELINE=1 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  gpu-lock.sh run build/bin/llama-bench \
    -m /storage/models/glm-4-9b-chat-iq2_m-imat.gguf \
    -ngl 999 -pp 128 -tg 32 -r 5

# 条件B (ON)
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  gpu-lock.sh run build/bin/llama-bench \
    -m /storage/models/glm-4-9b-chat-iq2_m-imat.gguf \
    -ngl 999 -pp 128 -tg 32 -r 5
```

### 期待値

- tg32: +1% (~1.3ms 削減)
- pp128: 0% 〜 小幅改善
- 有意性: Cohen's d >= 0.5 なら n=5 paired で p < 0.05
