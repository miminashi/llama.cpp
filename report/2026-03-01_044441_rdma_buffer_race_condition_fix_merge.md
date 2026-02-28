# RDMA バッファ再利用レースコンディション修正 — 検証・マージレポート

- **実施日時**: 2026年3月1日 04:44
- **ワークツリー**: `.worktree/fix-rdma-write-signal`
- **ブランチ**: `fix/rdma-write-always-signal` → `feature/rdma-backend` にマージ

## 前提・目的

selective signaling 最適化（`782b710f3`）導入後、RDMA バックエンドのモデルロード時にバッファ再利用によるレースコンディションが2箇所で発生していた。本レポートでは修正内容の検証と `feature/rdma-backend` へのマージを記録する。

### 背景

- selective signaling は Send 操作の完了通知を間引くことで pp128 を **+26.1%** 改善する最適化
- しかし、再利用される内部バッファに対して unsignaled な DMA 操作を行うと、次の反復で `memcpy` がバッファを上書きし、RNIC がまだ前回の DMA を完了していない場合にデータ破損が発生する
- 影響はモデルロード時のみ（推論性能 pp/tg には影響なし）

### 参照レポート

- [RDMA Write ステージングバッファ レースコンディション](report/2026-02-25_211814_rdma_write_staging_buffer_race_condition.md)
- [Send バッファ レースコンディション修正](report/2026-02-25_215905_send_buffer_race_condition_fix.md)

## 修正内容

### 修正1: RDMA Write ステージングバッファ (`e5700e0f8`)

- **ファイル**: `ggml/src/ggml-rdma/ggml-rdma.cpp`
- **原因**: `rdma_staging_buffer::get_buffer()` は毎回同じポインタを返す。selective signaling (interval=64) により最大63回の Write が完了通知なしで投入され、`memcpy` が DMA ソースバッファを上書き
- **症状**: サーバー GPU 上の重みデータ破損 → 計算結果に NaN 出現
- **修正**: `set_tensor` 内の RDMA Write を常に signaled にする
- **性能影響**: なし（モデルロードの一時的コストのみ。Send の selective signaling は維持）

### 修正2: Send 内部バッファ (`9a5a4a911`)

- **ファイル**: `ggml/src/ggml-rdma/ggml-rdma.cpp`, `ggml/src/ggml-rdma/rdma-transport.cpp`
- **原因**: `send_buffer_`（16MB）が毎反復で再利用される。unsignaled Send では次の `memcpy` が DMA 完了前にバッファを上書き
- **症状**: GLM-4.7 固有（バッファ > 4GB → `size_ok=false` → Send/Recv フォールバック）でゴミ文字（`>>9-A($%,...`）出力
- **修正**: `uses_internal_buffer` の場合は常に signaled にする
- **性能影響**: なし（モデルロードの一時的コストのみ）

## 検証手順

### 1. ビルド・デプロイ確認

- ワークツリー `.worktree/fix-rdma-write-signal` のバイナリが最新であることを確認
- 2号機 (192.168.100.2) にデプロイ済みであることを確認

### 2. GLM-4.7 日本語プロンプトテスト

RDMA サーバーを再起動後、11GPU (7 CUDA + 4 RDMA) で GLM-4.7 IQ2_M を起動し、日本語プロンプトで推論テストを実施。

```bash
bash scripts/rdma-server.sh restart
bash scripts/gpu-lock.sh run \
  GGML_RDMA_SERVERS=192.168.100.2:50051 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  .worktree/fix-rdma-write-signal/build/bin/llama-cli \
    -m ~/.cache/llama.cpp/unsloth_GLM-4.7-GGUF_UD-IQ2_M_GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
    -ngl 999 -sm layer -fa 1 \
    -p "日本の四季について教えてください" -n 256 \
    --log-file /tmp/llama-cli.log
```

### 3. テスト結果

| 項目 | 結果 |
|------|------|
| モデルロード | 成功 (7 CUDA + 4 RDMA) |
| ビルドバージョン | `b8183-9a5a4a911` |
| NaN | なし |
| ゴミ文字 | なし |
| Thinking出力 | 正常（日本語プロンプトに対する構造化された思考プロセス） |
| Prompt速度 | 6.4 t/s |
| Generation速度 | 8.3 t/s |

モデルは日本語プロンプト「日本の四季について教えてください」に対して、四季の特徴を分析する構造化された Thinking 出力を正常に生成した。出力にデータ破損の兆候はなく、テスト合格とした。

## マージ

テスト合格後、`feature/rdma-backend` ブランチで以下を実行:

```bash
git merge fix/rdma-write-always-signal
```

結果: **Fast-forward マージ**（コンフリクトなし）

```
Updating 0e9eea427..9a5a4a911
Fast-forward
 ggml/src/ggml-rdma/ggml-rdma.cpp      | 20 ++++++++------------
 ggml/src/ggml-rdma/rdma-transport.cpp | 11 ++++++++++-
 2 files changed, 18 insertions(+), 13 deletions(-)
```

## まとめ

- 2つのバッファ再利用レースコンディション修正を `feature/rdma-backend` にマージ完了
- GLM-4.7 IQ2_M 11GPU での日本語推論テストで NaN・ゴミ文字なしを確認
- 推論性能への影響なし（修正はモデルロード時の一時的コストのみ）
- Prompt 6.4 t/s, Generation 8.3 t/s は期待値の範囲内
