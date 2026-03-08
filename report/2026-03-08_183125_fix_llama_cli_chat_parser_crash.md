# llama-cli チャットパーサークラッシュ修正

- **実施日時**: 2026年3月8日 18:31
- **ワークツリー**: `.worktree/merge-gpu-native-sort`
- **ブランチ**: `merge/gpu-native-sort`
- **コミット**: `badecee1d`

## 前提・目的

`merge/gpu-native-sort` ブランチに upstream/master (`213c4a0b8`, 14 commits) をマージした結果、llama-cli が Qwen3.5 の thinking 出力を jinja テンプレート有効時に解析できずクラッシュする問題が発生。llama-cli は実ユースケース (コーディングエージェント) で必須のため修正。

## クラッシュの根本原因

### クラッシュフロー

```
main() → cli_context::generate_completion()
  → server_response_reader::next()
    → result->update(states[idx])
      → state.update_chat_msg(content, false)      // is_partial=false (最終パース)
        → common_chat_parse(generated_text, false)
          → common_chat_peg_parse()                 // chat.cpp:1548 — throw
```

### 原因の詳細

1. `cli.cpp:195` で `reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK` をハードコード
2. `cli.cpp:196` で `enable_thinking = true` (Qwen3.5 テンプレートに `<think>` タグあり)
3. autoparser が `FORCED_OPEN` モードの PEG 文法を生成: thinking content → `</think>` → actual content を期待
4. `-n 50` ではトークン数が不足し、モデルが `</think>` を生成しないまま生成終了
5. 最終パース (`is_partial=false`) で `</think>` が見つからず文法不一致 → `std::runtime_error` throw
6. `server_response_reader::next()` に try-catch がなく、例外が `main()` まで伝搬 → abort

部分パース (`is_partial=true`) は `chat.cpp:1535` で失敗時にも部分結果を返すため成功する。最終パースのみ throw する設計。

## 修正内容

**ファイル**: `common/chat.cpp` (1箇所)

`common_chat_peg_parse()` の最終パース失敗時の `throw` を `LOG_WRN` + raw content フォールバックに変更:

```cpp
// 変更前:
throw std::runtime_error(std::string("Failed to parse input at pos ") +
    std::to_string(result.end) + ": " + input.substr(result.end));

// 変更後:
LOG_WRN("Failed to parse chat output at pos %zu, treating as raw content\n",
    (size_t)result.end);
common_chat_msg msg;
msg.role = "assistant";
msg.content = input;
return msg;
```

### 影響範囲

- 正常パース時のコードパスに影響なし
- `-n` が十分大きく `</think>` が生成される場合はフォールバック不発動
- cli の表示出力は streaming 中の `diff.content_delta` から構築されるため、最終パースの結果は表示に影響しない
- upstream にも同じ問題あり (autoparser refactoring PR #18675)

## 検証結果

### `-n 50` (thinking 未完了、フォールバック発動)

```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  .../build/bin/llama-cli \
  -m /tmp/qwen35-35b-a3b-q4km-fused.gguf \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -p 'The capital of France is' -n 50 --seed 42 -fa 1 \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

- **結果**: クラッシュせず正常終了
- thinking content が途中まで表示された (50 トークンで `</think>` 未到達)
- Prompt: 69.7 t/s, Generation: 36.8 t/s

### `-n 200` (thinking 完了、通常パース)

- **結果**: クラッシュせず正常終了
- `[Start thinking]` → thinking content → `[End thinking]` → `The capital of France is **Paris**.`
- thinking block が正しくパースされ、actual content も正常に表示
- Prompt: 70.4 t/s, Generation: 36.5 t/s

### llama-bench 退行確認

| テスト | t/s | 期待値 | 差分 |
|--------|----:|-------:|-----:|
| pp128 | 271.75 ± 2.81 | ≈272 | -0.1% |
| pp512 | 458.11 ± 1.56 | ≈458 | +0.0% |
| tg32 | 36.31 ± 0.08 | ≈36.3 | +0.0% |

性能退行なし。

## 次の課題: pp2048 絶対値退行

upstream マージにより pp2048 が 518.91 → 467.76 (-9.8%) 退行。

- **原因**: upstream の `ggml_cuda_check_fusion_memory_ranges()` (PR #19916) が topk_moe fusion を correctness fix として一部無効化
- upstream/master 自体も同じ退行 (merge 固有ではない)
- 対策案: upstream に issue/PR で報告、または P100 での topk_moe fusion のメモリ範囲チェックを調査

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `badecee1d (merge/gpu-native-sort)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 28°C | 33°C |
| 電力制限 | 180.00W | 180.00W |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running (PID 1692147) |
