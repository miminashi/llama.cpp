# `GGML_RDMA_PARALLEL` 統合環境変数の実装

- **実施日時**: 2026年3月3日 19:03
- **ワークツリー**: `.worktree/pipeline-splits`
- **ブランチ**: `feature/pipeline-splits`
- **コミット**: `d7e8c45da`

## 前提・目的

pipeline parallelism と per-device connections を有効化するには、従来 2つの環境変数を個別に設定する必要があった:

```bash
GGML_RDMA_PIPELINE=1 GGML_RDMA_PER_DEVICE_CONN=1 llama-bench ...
```

これらは組み合わせて使用することが多く (単独では効果が限定的)、設定の手間とミスを減らすために 1つの環境変数 `GGML_RDMA_PARALLEL=1` で両方を同時に有効化するショートカットを実装した。

- 参照: [pipeline + per-device 組み合わせベンチマーク](2026-03-03_145744_pipeline_perdevice_combined_benchmark.md)

## 変更内容

### コード変更 (4ファイル4箇所)

1. **`ggml/src/ggml-rdma/ggml-rdma.cpp`**: `GGML_RDMA_PARALLEL` 環境変数の読み取りロジックを追加。`1` が設定されている場合、`GGML_RDMA_PIPELINE` と `GGML_RDMA_PER_DEVICE_CONN` の両方を有効化する。
2. **`ggml/src/ggml-rdma/rdma-server.cpp`**: サーバー側でも同様に `GGML_RDMA_PARALLEL` を読み取り、`GGML_RDMA_PER_DEVICE_CONN` を有効化する (pipeline はクライアント側のみの機能)。
3. **`.claude/skills/bench/SKILL.md`**: 環境変数テーブルに `GGML_RDMA_PIPELINE` と `GGML_RDMA_PARALLEL` の2行を追加。
4. **`README.md`**: 環境変数テーブルに `GGML_RDMA_PIPELINE` と `GGML_RDMA_PARALLEL` の2行を追加。

### 使用方法

```bash
GGML_RDMA_PARALLEL=1 llama-bench -m model.gguf ...
```

上記は以下と等価:

```bash
GGML_RDMA_PIPELINE=1 GGML_RDMA_PER_DEVICE_CONN=1 llama-bench -m model.gguf ...
```

個別の環境変数は引き続き使用可能。`GGML_RDMA_PARALLEL=1` と個別変数を同時に設定した場合、個別変数の値が優先される。

## 検証

- ワークツリー `.worktree/pipeline-splits` でビルド成功を確認
- ドキュメントの markdown テーブルが正しくレンダリングされることを確認
