---
name: models
description: ダウンロード済みモデルの一覧表示。HF キャッシュ内のモデルを確認する。
user-invocable: true
---

# モデル一覧

## 全モデル一覧

```bash
bash scripts/hf-models.sh
```

リポジトリ名、サイズ、GGUF ファイル、スナップショットパスを表示する。

## GGUF パスのみ出力 (`-m` 用)

```bash
bash scripts/hf-models.sh --gguf
```

`-m` フラグに直接渡せる GGUF ファイルの絶対パスを出力する。
