# Server-Push (RDMA Write with IMM) 実装レポート

## 概要

get_tensor を pull モデル (クライアント Send/Recv) から push モデル (サーバーが graph_compute 完了後に RDMA Write with IMM で結果をプッシュ) に変更。per-device connection と組み合わせて実装。

## 結果サマリー

| 指標 | Push ON | Push OFF | 差分 |
|------|:-------:|:--------:|:----:|
| tg32 (t/s) | 7.763 ± 0.012 | 7.705 ± 0.007 | **+0.057 (+0.74%)** |
| pp128 (t/s) | 23.778 ± 0.013 | 23.777 ± 0.016 | +0.001 (+0.01%) |

- GLM-4.7 IQ2_M, 7 CUDA + 4 RDMA (11 GPU)
- p = 1.28 × 10⁻¹⁰, Cohen's d = 4.30, 15/15 ペア正の効果
- **判定**: 統計的に有意 (p < 0.05) かつ効果 0.74% > 0.5% 閾値。ただし実用的効果は小さい。

## 実装内容

### アーキテクチャ

```
[従来] get_tensor:
  Client --[Send GET_TENSOR]--> Server --[compute result]--> Server --[Send response]--> Client

[Server-Push] get_tensor:
  Client(graph_compute): push request を ASYNC コマンドに埋め込み送信
  Server: compute 完了後、結果を RDMA Write with IMM でクライアント push_recv_buf に書き込み
  Client(get_tensor): recv_cq_ の IMM CQE をポーリング → push_recv_buf からコピー
```

### 主要コンポーネント

#### 1. Transport 層 (rdma-transport.h/cpp)
- **CQ 分離**: 単一 `cq_` を `send_cq_` と `recv_cq_` に分離
- **`rdma_write_imm()`**: `IBV_WR_RDMA_WRITE_WITH_IMM` による RDMA Write
- **`poll_for_imm()`**: `recv_cq_` で IMM CQE をポーリング
- **`post_imm_recv()`**: IMM 受信用 recv WR の事前 post

#### 2. プロトコル拡張 (ggml-rdma.cpp)
- **`RDMA_CMD_REGISTER_PUSH_BUFFER`**: クライアントが push 受信バッファ情報をサーバーに登録
- **Push payload trailer**: ASYNC コマンドに `| n_push(4B) | entries(P*32B) | total_size(4B) |` を付加
- **IMM データ**: `buffer_index` をエンコード (クライアントがどのバッファの結果か識別)

#### 3. クライアント側
- **push_recv_buf**: バッファ割り当て時にホスト pinned memory を確保 + MR 登録
- **push_request_tracker**: get_tensor 呼び出しパターンをキャッシュし、次回 graph_compute で push request として送信
- **push_remaining_**: atomic カウンタで push 受信待ちの IMM 数を管理
- **サイズ制限**: `RDMA_PUSH_MAX_BUFFER_SIZE = 1 GB` — ウェイトバッファ (>8 GB) は push 対象外

#### 4. サーバー側
- **push request 受信**: ASYNC コマンドから push request を抽出
- **compute 完了後の push**: GPU 結果を読み出し → RDMA Write with IMM でクライアント push_recv_buf に書き込み
- **per-device 独立 push**: 各デバイスの接続から独立に push 実行

### 環境変数

| 変数 | 説明 | デフォルト |
|------|------|----------|
| `GGML_RDMA_NO_SERVER_PUSH` | `1` で push 無効化 (Send/Recv フォールバック) | 未設定 (有効) |
| `GGML_RDMA_PER_DEVICE_CONN` | `1` で per-device 接続有効化 (push の前提) | 未設定 |

## デバッグ経緯

### Bug 1: n_push > 1 でクラッシュ
- **症状**: 2番目の push IMM 受信時にフォールバックの Send/Recv がストールの CQE を拾いクラッシュ
- **原因**: boolean `compute_pending_` で push 完了を管理 → 最初の push で false に設定 → 2番目は push パスに入れずフォールバック
- **修正**: `push_remaining_` atomic カウンタに変更、各 push 受信で decrement

### Bug 2: バッファインデックス不一致
- **症状**: `poll_for_imm` 成功するが `lookup_push_buffer` が nullptr を返す
- **原因**: サーバーが独自カウンタで IMM をエンコード、クライアントは別カウンタでルックアップ
- **修正**: `rdma_msg_register_push_buffer_req` にクライアント割り当ての `push_buffer_index` を追加、サーバーがそのまま使用

### Bug 3: GLM-4.7 Prompt 速度低下 (6.4 → 2.9 t/s)
- **症状**: push_recv_buf を全バッファに割り当て → ~40 GB の余分な pinned memory + MR 登録
- **原因**: ウェイトバッファ (~10 GB × 4) にも push_recv_buf を割り当て → ConnectX-4 MTT キャッシュ圧迫
- **修正**: `RDMA_PUSH_MAX_BUFFER_SIZE = 256 MB` サイズ制限追加

### Bug 4: GLM-4.7 で Push=0 (push 未有効化)
- **症状**: compute バッファ (48-72 MB) は push_eligible だが、get_tensor 対象テンソルは 466 MB バッファに存在
- **原因**: 256 MB 制限が厳しすぎ、466 MB の compute 出力バッファが push 対象外
- **修正**: サイズ制限を 1 GB に引き上げ (ウェイトバッファ >8 GB のみ除外)

## 定量評価

### 交絡チェック

- [x] 単一変数分離: 同一バイナリ + `GGML_RDMA_NO_SERVER_PUSH=1` で条件切替
- [x] ホットパスログ出力なし: 診断 fprintf は評価前に全削除済み
- [x] 分布の単峰性: パイロットラン確認済み

### 実験設計

- ABAB Paired Design、n=15 ペア
- llama-bench (小数第2位の精度)
- GLM-4.7 IQ2_M, 7C+4R, pp128/tg32

### 生データ

| Pair | Push ON (tg) | Push OFF (tg) | Diff |
|------|:------------:|:-------------:|:----:|
| 1 | 7.75 | 7.71 | +0.04 |
| 2 | 7.74 | 7.70 | +0.04 |
| 3 | 7.76 | 7.69 | +0.07 |
| 4 | 7.77 | 7.71 | +0.06 |
| 5 | 7.75 | 7.70 | +0.05 |
| 6 | 7.78 | 7.70 | +0.08 |
| 7 | 7.76 | 7.71 | +0.05 |
| 8 | 7.76 | 7.70 | +0.06 |
| 9 | 7.78 | 7.71 | +0.07 |
| 10 | 7.75 | 7.71 | +0.04 |
| 11 | 7.76 | 7.72 | +0.04 |
| 12 | 7.77 | 7.70 | +0.07 |
| 13 | 7.77 | 7.71 | +0.06 |
| 14 | 7.77 | 7.71 | +0.06 |
| 15 | 7.77 | 7.70 | +0.07 |

### 検定結果

| 指標 | 値 |
|------|:---:|
| 差分平均 | +0.057 t/s (+0.74%) |
| t(14) | 16.64 |
| p 値 | 1.28 × 10⁻¹⁰ |
| Cohen's d | 4.30 (非常に大きい) |
| 95% CI | [+0.050, +0.065] t/s |
| 全ペア正の効果 | 15/15 (100%) |

### 判定

**有効**: p < 0.05 かつ効果 +0.74% > 0.5% 閾値を満たす。

ただし実用的効果は小さい (+0.057 t/s)。効果が限定的な理由は以下の通り:

1. **クライアントの逐次処理**: `ggml_backend_sched` が各デバイスに対して `graph_compute → get_tensor` を逐次呼び出すため、per-device 並列 push の恩恵が限定的
2. **GET_TENSOR round-trip は元々小さい**: 12ms の GPU compute に対し、Send/Recv round-trip は ~0.5ms (4%)
3. **push の実際の節約**: 4 デバイス × 0.5ms = 2ms per token ≈ +0.7% — 実測値と一致

## ワークツリーとファイル

- ワークツリー: `/home/ubuntu/projects/llama.cpp/.worktree/rdma-server-push`
- 変更ファイル:
  - `ggml/src/ggml-rdma/rdma-transport.h` — CQ 分離、push_remaining_、push_active_
  - `ggml/src/ggml-rdma/rdma-transport.cpp` — CQ 分離実装、rdma_write_imm、poll_for_imm、post_imm_recv
  - `ggml/src/ggml-rdma/ggml-rdma.cpp` — Push 受信バッファ、get_tensor push 対応、graph_compute push request、サーバー push 実行
  - `ggml/include/ggml-rdma.h` — プロトコルバージョン

## 今後の展望

Server-Push の効果を最大化するには、クライアント側で全デバイスの graph_compute を先に送信してから get_tensor を一括で待つ実装が必要。これは `ggml_backend_sched` の変更を伴うため、RDMA バックエンド単体では実現困難。
