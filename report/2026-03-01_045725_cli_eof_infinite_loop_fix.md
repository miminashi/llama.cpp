# llama-cli 会話モードの stdin EOF 無限ループ修正

- **実施日時**: 2026年3月1日 04:57
- **ワークツリー**: `.worktree/fix-cli-eof` (ブランチ: `fix/cli-eof-infinite-loop`)

## 前提・目的

llama-cli をバックグラウンドプロセスとして実行すると（stdin=/dev/null）、会話モードで `> ` プロンプトが無限に出力され、stdout に GByte 単位のデータが書き込まれる問題を修正する。

- **背景**: ターミナルからの手動実行では再現しない。バックグラウンドプロセスや `echo "" | llama-cli ...` のようにパイプで実行した場合にのみ発生する
- **原因**: `tools/cli/cli.cpp:303-306` の `do-while` ループで `readline()` が EOF（`false`）を返した後、`buffer` が空のまま 340-341 行目の `if (buffer.empty()) { continue; }` でメインループに戻り、再び `> ` を出力する無限ループが発生
- **目的**: stdin が EOF になった場合に会話ループを正常に終了させる

## 修正内容

### 対象ファイル

- `tools/cli/cli.cpp` (307-309行)

### 変更差分

```diff
             do {
                 another_line = console::readline(line, params.multiline_input);
                 buffer += line;
             } while (another_line);
+            if (!another_line && buffer.empty()) {
+                break;  // stdin EOF
+            }
         } else {
```

`readline()` が最初の呼び出しで EOF を返し（`another_line == false`）、かつ buffer が空の場合、`break` で会話ループを抜ける。

340-341行目の `if (buffer.empty()) { continue; }` は、ユーザーが空行を入力した場合（Enter のみ押した場合、`readline()` は `true` を返す）のために残す。

## 再現方法

### ビルド

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/fix-cli-eof/scripts/rdma-build.sh local
```

### テスト1: EOF 挙動（修正の検証）

```bash
echo "" | build/bin/llama-cli -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf -ngl 0 -c 256 --log-file /tmp/llama-cli.log
```

- **修正前**: 無限ループ（`timeout` で強制終了、exit code 124）
- **修正後**: `> ` が1回表示され、`Exiting...` で正常終了

### テスト2: 通常動作（回帰なし）

ターミナルから手動で llama-cli を起動し、以下を確認:
- 空行入力（Enter のみ）→ `continue` で再プロンプト（既存動作を維持）
- Ctrl+D で終了 → 正常終了

## 検証結果

| テスト | 修正前 | 修正後 |
|--------|--------|--------|
| `echo "" \| llama-cli ...` | 無限ループ（exit 124） | 正常終了 |
| ターミナル手動操作 | 正常 | 正常（回帰なし） |
