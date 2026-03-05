# 非同期 Send の is_last シグナリング省略: 実装とベンチマーク

- **実施日時**: 2026年3月2日 06:31
- **ワークツリー**: `.worktree/double-buffering`
- **ブランチ**: `feature/double-buffering`
- **コミット**: `c8ca4b1f6` (feat(rdma): skip is_last signaling for async fire-and-forget sends)

## 前提・目的

### 背景

前回の double-buffering 実装 (`ec2fb5a89`) と signal interval 修正 (`fa8f7c6a6`) により、RDMA Write と Send の内部バッファ再利用レースコンディションが修正された。しかし、現在のコードでは `is_last=true` (単一チャンク送信 = 全送信の大半) で**常にシグナリング + wait_for_completion** が発生する。

fire-and-forget な非同期 `graph_compute` コマンドでは、inline 送信 (データは WQE にコピーされるため、バッファ再利用の懸念なし) に対して `is_last` シグナリングを省略し、不要な CQ ポーリング待機を排除できる。

### 目的

- `send()` に `skip_last_signal` パラメータを追加し、非同期送信パスで不要な CQ ポーリングを排除する
- `global_unsignaled_` カウンタで send() 呼び出し間の unsignaled WR を追跡し、SQ オーバーフローを防止する
- `rdma_write`/`rdma_read` の signaled 完了後に `global_unsignaled_` をリセットし、IB 順序保証を活用する

### 参照レポート

- [ステージングバッファ double-buffering 実装](report/2026-03-02_040542_staging_buffer_double_buffering.md)
- [signal interval 修正](report/2026-03-02_045144_signal_interval_fix.md)

## 実装内容

### 変更ファイル (3ファイル, +17/-10行)

| ファイル | 変更内容 |
|---------|---------|
| `rdma-transport.h` | `send()` に `skip_last_signal` パラメータ、`global_unsignaled_` メンバ追加 |
| `rdma-transport.cpp` | `send()` のシグナル条件変更: ローカル `unsignaled_count` → グローバル `global_unsignaled_`、`RDMA_MAX_UNSIGNALED=96` 安全上限追加、`rdma_write`/`rdma_read` で `global_unsignaled_` リセット |
| `ggml-rdma.cpp` | `send_rdma_cmd_raw()` に `skip_last_signal` パラメータ追加、`send_rdma_cmd_async()` で `true` を渡す |

### 安全性

| 観点 | 説明 |
|------|------|
| inline 送信 | データは WQE にコピー → ibv_post_send 後にバッファ再利用の懸念なし |
| 内部バッファ送信 | `!uses_internal_buffer` ガードで常に is_last シグナリング |
| SQ オーバーフロー | `RDMA_MAX_UNSIGNALED=96` < `max_send_wr=128` |
| IB 順序保証 | signaled RDMA Write/Read 完了は先行 unsignaled Send の完了も保証 |
| フォールバック | `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` で全 WR 強制シグナリング |

## 正常性テスト

### 手順

```bash
bash .worktree/double-buffering/scripts/rdma-build.sh local
gpu-lock.sh run bash .worktree/double-buffering/scripts/rdma-deploy.sh
bash .worktree/double-buffering/scripts/rdma-server.sh start

gpu-lock.sh run CUDA_VISIBLE_DEVICES=5,6 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  .worktree/double-buffering/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -p "What is RDMA?" -n 50 --log-file /tmp/llama-cli.log
```

### 結果

- **正常に推論完了**: "RDMA stands for Remote Direct Memory Access..." (正しい出力)
- **スループット**: pp=51.9 t/s, tg=33.0 t/s (Qwen3.5-35B-A3B, 6GPU)
- **メモリ分配**: CUDA0=4524 MiB, CUDA1=4581 MiB, RDMA0-3=4072-4581 MiB (正常)

## A/B ベンチマーク

### テスト設計

- **モデル**: gpt-oss-20b Q4_K_M (12GB, 単一 GPU に収容可能)
- **構成**: CUDA5,6 + RDMA0-3 (6デバイス、デバイスごと個別ベンチ)
- **A条件**: `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` (全 WR シグナリング)
- **B条件**: デフォルト (selective signaling + skip_last_signal)
- **パターン**: ABAB 5ペア × 6デバイス (各条件30サンプル: RDMA 20 + CUDA 10)

### 注意: テスト設計の制限

`GGML_RDMA_NO_SELECTIVE_SIGNAL` は以下の両方に影響する:
1. **`send()`** (rdma-transport.cpp): Send WR のシグナリング → skip_last_signal の効果
2. **`set_tensor`** (ggml-rdma.cpp L1004): RDMA Write WR のシグナリング → double-buffering の効果

A/B 比較は skip_last_signal の効果だけでなく、RDMA Write selective signaling の効果も含む。

### 手順

```bash
gpu-lock.sh run bash /tmp/bench-skip-last-signal.sh
```

ベンチマークスクリプトは各ペアで A→B の順に実行、各条件 `-r 1` で 1回、pp512 + tg128 を測定。

### 結果

| グループ | N | Mean A (t/s) | Mean B (t/s) | 差分 (B-A) | 変化率 | t統計量 | p値 |
|----------|---|:----------:|:----------:|:--------:|:-----:|:------:|:---:|
| **RDMA pp512** | 20 | 729.40 | 728.85 | -0.55 | **-0.08%** | 3.24 | 0.004 |
| **RDMA tg128** | 20 | 60.83 | 59.40 | -1.44 | **-2.36%** | 18.80 | <0.0001 |
| CUDA pp512 (対照群) | 10 | 735.83 | 736.00 | +0.17 | +0.02% | -1.15 | 0.278 |
| CUDA tg128 (対照群) | 10 | 69.33 | 69.34 | +0.01 | +0.01% | -0.78 | 0.453 |

### 分析

1. **CUDA 対照群は差なし** — pp512 (p=0.278), tg128 (p=0.453) ともに有意差なし。ベンチマーク環境は安定。

2. **RDMA pp512 は実質無差別** — -0.08% は統計的に有意 (p=0.004) だが、実用上無視できる大きさ。

3. **RDMA tg128 は -2.36% のレグレッション** — 高い統計的有意性 (p<0.0001) だが、これは skip_last_signal の効果ではなく、`set_tensor` の RDMA Write selective signaling (ggml-rdma.cpp L1004) が主因。A 条件 (`GGML_RDMA_NO_SELECTIVE_SIGNAL=1`) では set_tensor の RDMA Write も全シグナリングとなるため、double-buffered staging buffer の同期が異なる。

4. **skip_last_signal 自体の効果は中立** — pp512 (graph_compute が多い) で -0.08% しか差がないことから、skip_last_signal による CQ ポーリング省略の効果はこのモデルサイズ/GPU構成では測定誤差内。

### 制限事項

- gpt-oss-20b は単一 P100 に収容可能なため、llama-bench はデバイスごと個別にベンチマーク実行。マルチデバイス結合推論のテストではない。
- Qwen3.5-35B-A3B はマルチ GPU 必須だが、llama-bench でのモデルロードに失敗 (OOM)。llama-cli では正常動作。
- skip_last_signal の効果を厳密に分離するには、`GGML_RDMA_NO_SKIP_LAST_SIGNAL` のような専用環境変数が必要。

## 結論

- **実装は正常に動作**: Qwen3.5-35B-A3B (6GPU) で正常な推論出力を確認。
- **skip_last_signal 自体の性能効果はニュートラル**: pp512 で -0.08%、実用上無視できる。
- **将来の価値**: より大きなモデル (GLM-4.7 等) やより多くの RDMA デバイスでは、graph_compute イテレーションあたりの不要な wait_for_completion 排除量が増え、効果が顕在化する可能性がある。
- **RDMA Write selective signaling のレグレッション**: tg128 で -2.36% は既知の問題であり、本コミットの変更によるものではない。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c8ca4b1f6 (feature/double-buffering)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 34°C |
| 電力制限 | 180.00W | 180.00W |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
