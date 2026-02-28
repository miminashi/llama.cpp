# Send パス バッファ再利用競合の修正

- **実施日時**: 2026年2月25日 21:59
- **ワークツリー**: `.worktree/fix-rdma-write-signal`
- **ブランチ**: `fix/rdma-write-always-signal`
- **コミット**: `9a5a4a911`

## 前提・目的

RDMA Write の selective signaling バグ修正後、qwen2.5-0.5b は正常出力を確認したが、GLM-4.7 IQ2_M (11GPU) では依然としてガベージ ASCII 出力 (`>>9-A($%,->,923<7,...`) が発生していた。速度は正常 (pp 7.9, tg 8.6 t/s)。

- **背景**: 先行修正 (コミット `e5700e0f8`) で RDMA Write パスの staging buffer 競合は修正済み
- **目的**: Send パスにも存在する同一の競合を修正し、GLM-4.7 のガベージ出力を解消する
- **関連レポート**: [report/2026-02-25_211814_rdma_write_staging_buffer_race_condition.md](2026-02-25_211814_rdma_write_staging_buffer_race_condition.md)

## 根本原因

`rdma_connection::send()` にも RDMA Write と同一のバッファ再利用競合が存在した。

### GLM-4.7 で発現するメカニズム

1. GLM-4.7 の RDMA バッファは ~10GB/デバイス → `ctx->size > 4GB` → `size_ok = false`
2. RDMA Write パスがスキップされ、`set_tensor` が Send/Recv フォールバック (line 1046) を使用
3. `send_rdma_cmd` → `conn->send()` で `send_buffer_` (16MB) を再利用しながら selective signaling
4. unsignaled Send の DMA 完了前に次の `memcpy` がバッファを上書き → データ破損

### qwen2.5-0.5b で発現しない理由

- バッファサイズ < 4GB → `size_ok = true` → RDMA Write パスを使用（先行修正で対応済み）

## 修正内容

### `rdma-transport.cpp`: `send()` の内部バッファ使用時に always-signal

`send_buffer_` を使用するブランチでのみ常に signaled にする変更。`mr` 提供時と inline 時は呼び出し元バッファを使うため影響なし。

```diff
     static constexpr size_t RDMA_SIGNAL_INTERVAL = 64;
     size_t unsignaled_count = 0;
+    bool uses_internal_buffer = false;

     while (remaining > 0) {
+        uses_internal_buffer = false;
         // ...
         } else if (chunk_size > 0 && send_mr_) {
+            uses_internal_buffer = true;
             // Use internal send buffer, chunk if needed
         // ...
-        bool do_signal = RDMA_NO_SELECTIVE_SIGNAL || is_last || (++unsignaled_count >= RDMA_SIGNAL_INTERVAL);
+        bool do_signal = RDMA_NO_SELECTIVE_SIGNAL || is_last
+                         || uses_internal_buffer
+                         || (++unsignaled_count >= RDMA_SIGNAL_INTERVAL);
```

### `ggml-rdma.cpp`: コメント修正

RDMA Write の修正コメント内にあった「Send selective signaling は safe」という誤った記述を修正。

## 性能への影響

- **pp/tg 性能**: 影響なし — `graph_compute` の Send は < 16MB（単一チャンク、`is_last=true` で常に signaled）
- **モデルロード速度**: 大きいバッファ (> 4GB) の Send/Recv フォールバック時にチャンク毎の CQ ポーリングで若干低下。一回限りのコストで許容範囲

## 検証結果

### 回帰テスト: qwen2.5-0.5b (2GPU: 1 CUDA + 1 RDMA)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  llama-cli -m qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -c 256 -n 32 -p 'hello' --simple-io --single-turn
```

- **結果**: 正常な英語出力 "Hello! How can I assist you today?"
- **速度**: pp 495.0 t/s, tg 131.0 t/s

### 最終テスト: GLM-4.7 IQ2_M (11GPU: 7 CUDA + 4 RDMA)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  llama-cli -m GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,...,CUDA6,RDMA0[192.168.100.2:50051],...,RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' --flash-attn on --no-warmup --single-turn --simple-io
```

- **結果**: 正常なテキスト出力（thinking mode で "The user is asking for the capital of France..." と論理的な推論を開始）
- **ガベージ ASCII 出力なし**
- **速度**: pp 8.2 t/s, tg 8.6 t/s

## まとめ

RDMA Write (staging buffer) と Send (send_buffer_) の両方でバッファ再利用競合を修正。これにより全モデルサイズで正常出力が確認された:

| モデル | バッファサイズ | 使用パス | 修正 | 結果 |
|-------|:----------:|---------|------|------|
| qwen2.5-0.5b | < 4GB | RDMA Write | `e5700e0f8` | 正常出力 |
| GLM-4.7 IQ2_M | > 4GB | Send/Recv fallback | `9a5a4a911` (本修正) | 正常出力 |
