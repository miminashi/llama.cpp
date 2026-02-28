# RDMA Write ステージングバッファ競合の調査と修正

- **実施日時**: 2026年2月25日 21:18
- **ワークツリー**: `.worktree/fix-rdma-write-signal` (ブランチ: `fix/rdma-write-always-signal`)
- **調査ワークツリー**: `.worktree/debug-get-tensor` (ブランチ: `debug/get-tensor-inspection`)

## 前提・目的

`feature/rdma-backend` ブランチで `llama-cli` を使い GLM-4.7 IQ2_M (11GPU) や qwen2.5-0.5b (2GPU) を推論すると、出力がすべて `????` になる現象が発生。モデルロード・推論速度は正常であり、ローカル (RDMA なし) 実行では正常出力を確認済み。

- **背景**: Feb 20 のテストでは正常動作。最後のコード変更は `782b710f3 feat: add selective signaling` (Feb 20)
- **目的**: RDMA バックエンド経由の推論で出力が破損する根本原因を特定し修正する

## 参照レポート

- [Feb 20 llama-cli A/B テスト](report/2026-02-20_212013_llama_cli_abab_replication.md)

## 調査プロセス

### Phase 1: 症状の絞り込み

1. **サーバー側 GPU 計算で NaN 発生を確認**: `get_tensor` デバッグログを追加し、`result_output` (logits) テンソルがサーバー側で NaN (`0x7FFFFFFF`) であることを確認。中間テンソル (norm-*, l_out-*) は正常値
2. **全モデル・全 GPU で再現**: qwen2.5-0.5b (2GPU) でも同じ NaN パターン
3. **2 つの障害モード**:
   - Staging パス: `result_output` に NaN
   - NO_STAGING パス: `set_tensor` デシリアライゼーション失敗 + ガベージ出力

### Phase 2: Selective Signaling の特定

**テスト結果**:

| Client | Server | 結果 |
|--------|--------|------|
| `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` | `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` | OK |
| `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` | (通常) | OK |
| (通常) | `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` | NG (`????`) |

**結論**: バグは **client 側** の selective signaling にある。

### Phase 3: RDMA Write vs Send の切り分け

`set_tensor` の RDMA Write パスのみ always-signal に変更し、Send の selective signaling はそのまま:

| 修正内容 | 結果 |
|---------|------|
| RDMA Write always-signal + Send selective | OK (`Hello! How can I assist you today?`) |

**結論**: **RDMA Write のステージングバッファ再利用競合** が根本原因。Send selective signaling は問題なし。

## 根本原因

### DMA バッファ再利用競合

`set_tensor` の RDMA Write パスでは、16MB チャンクごとにリモートステージングバッファへ書き込む:

```
while (remaining > 0) {
    buf = staging->get_buffer(chunk, &mr);  // 常に同じポインタを返す
    memcpy(buf, src, chunk);                // ← DMA ソースバッファに書き込み
    rdma_write(buf, chunk, mr, remote, do_signal);
    // do_signal=false の場合、DMA 完了を待たずに戻る
    // → 次の memcpy が DMA 読み取り中のバッファを上書き!
}
```

`rdma_staging_buffer::get_buffer()` は **単一の再利用バッファ** を返す。Selective signaling (interval=64) では最大 63 回の unsignaled RDMA Write が DMA 完了なしに投稿され、次のイテレーションの `memcpy` が DMA ソースバッファを上書きする。

**結果**: サーバーのステージングバッファに破損データが書き込まれ、GPU へのフラッシュ後に NaN を含む重みデータとなり、計算結果が NaN になる。

### なぜ Send は影響を受けないか

`rdma_connection::send()` も同じ `send_buffer_` 再利用パターンを持つが、graph_compute コマンドのシリアライズデータは常に 16MB 以下 (単一チャンク、`is_last=true` で必ず signaled) のため影響しない。

## 修正内容

`set_tensor` の RDMA Write を常に signaled にする (`do_signal = true`)。

```diff
-            static constexpr size_t RDMA_SIGNAL_INTERVAL = 64;
-            size_t unsignaled_count = 0;
             while (remaining > 0) {
                 ...
-                bool do_signal = RDMA_NO_SELECTIVE_SIGNAL || is_last
-                                 || (++unsignaled_count >= RDMA_SIGNAL_INTERVAL);
-                if (!ctx->conn->rdma_write(buf, chunk, mr, remote, do_signal)) {
+                // Always signal: staging buffer is single-allocation, must wait
+                // for DMA completion before next memcpy overwrites it.
+                if (!ctx->conn->rdma_write(buf, chunk, mr, remote, true)) {
```

### 性能への影響

- **pp/tg 性能: 影響なし** — selective signaling の +26.1% pp128 改善は `rdma_connection::send()` (graph_compute コマンド送信) から得られるもので、`set_tensor` の RDMA Write からではない
- **モデルロード速度**: チャンク毎の CQ ポーリングにより若干低下するが、一回限りのコスト

## 検証結果

### qwen2.5-0.5b (2GPU: CUDA0 + RDMA0)

| 条件 | 出力 | pp (t/s) | tg (t/s) |
|------|------|----------|----------|
| 修正前 (selective signaling) | `??????????` (NaN) | 219 | 128 |
| 修正後 (always signal) | `Hello! How can I assist you today?` | 221 | 130 |

### GLM-4.7 IQ2_M (11GPU: 7C+4R)

| 条件 | 出力 | pp (t/s) | tg (t/s) |
|------|------|----------|----------|
| 修正後 | 正常トークン生成 (中国語 thinking) | 8.3 | 8.6 |

## 再現方法

### 修正前 (バグ再現)

```bash
# feature/rdma-backend ブランチのビルドをデプロイ
bash scripts/rdma-deploy.sh

# サーバー起動
ssh 192.168.100.2 "LD_LIBRARY_PATH=.../build/bin nohup .../rdma-server -H 0.0.0.0 -p 50051 &"

# クライアント実行 → ???? 出力
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-cli -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -c 256 -n 32 -p 'hello' --simple-io --log-file /tmp/llama-cli.log
```

### 修正後

```bash
# fix/rdma-write-always-signal ブランチのビルドをデプロイ
bash .worktree/fix-rdma-write-signal/scripts/rdma-deploy.sh

# 同じコマンドで正常出力
```

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `e5700e0f8 (fix/rdma-write-always-signal)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 31°C | 32°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 1092744) |

## 将来の改善案

1. **ダブルバッファリング**: ステージングバッファを 2 つ使い交互に使用すれば、RDMA Write の selective signaling を interval=2 で復活可能。モデルロード速度が CQ ポーリング 50% 削減で改善
2. **Send パスの監視**: 現在は graph データ <16MB で安全だが、将来テンソル数が大幅に増える場合は `rdma_connection::send()` のバッファ再利用にも同様の対策が必要
