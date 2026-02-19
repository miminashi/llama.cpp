# PD 共有 Per-device Connection 実装・検証レポート

- **実施日時**: 2026年2月18日 13:00
- **ワークツリー**: `.worktree/rdma-shared-pd`

## 前提・目的

### 背景

RDMA バックエンドの Generation 速度は RPC 比 -10% (6.8 vs 7.5 t/s, GLM-4.7 IQ2_M 11GPU)。主因は全デバイスが 1 つの QP を共有しコマンドが逐次実行されること。

Per-device connection は既に実装済み (`GGML_RDMA_PER_DEVICE_CONN=1`) だったが、ConnectX-4 の MTT キャッシュ制限で無効化されていた。各接続が独自 PD を作成 → 同じバッファを各 PD で独立に MR 登録 → MTT エントリが接続数倍に増加しオーバーフロー。

### 目的

PD を全接続で共有し、MR の重複登録を排除しつつ、各デバイスが独立 QP を持つことで RPC 同等の並列性を実現する。IB 仕様上、同一 PD 内の全 QP は共有 MR にアクセス可能。

### 前提条件

- Step 4 (GPUDirect RDMA) が動作していること
- GLM-4.7 IQ2_M が 11GPU (7C+4R) で安定動作していること

## 実装内容

### 変更ファイル

1. `ggml/src/ggml-rdma/rdma-transport.h` — インターフェース変更
2. `ggml/src/ggml-rdma/rdma-transport.cpp` — PD 共有ロジック
3. `CLAUDE.md` — 環境変数テーブル更新

### 主要な変更点

#### 1. PD 共有メカニズム

- `connect()`, `accept()`, `setup_qp()` に `external_pd` パラメータを追加
- 外部 PD が渡された場合は `ibv_alloc_pd()` をスキップし、共有 PD を使用
- `owns_pd_` フラグで PD 所有権を管理 (外部 PD 使用時は `false`)

#### 2. クライアント側: `get_connection()` の PD 共有

- 同一エンドポイントへの最初の接続が PD を作成
- `shared_pds_` マップに PD を保存し、所有権を移転 (`release_pd_ownership()`)
- 後続の接続は `shared_pds_` から PD を取得

#### 3. サーバー側: `accept_connection()` の PD 共有

- 最初に accept した接続の PD を `server_shared_pd_` に保存
- 後続の接続は `server_shared_pd_` を共有

#### 4. デフォルト動作の決定

ベンチマーク結果に基づき:
- **デフォルト**: 単一共有接続 (旧動作、変更なし)
- **opt-in**: `GGML_RDMA_PER_DEVICE_CONN=1` で PD 共有 per-device 接続を有効化

### レビューで発見・修正したバグ

| # | バグ | 重要度 | 修正 |
|---|------|--------|------|
| 1 | PD ダブルフリー (クライアント) | Critical | `release_pd_ownership()` で所有権を `shared_pds_` に移転 |
| 2 | PD ダブルフリー (サーバー) | Critical | `release_pd_ownership()` で所有権を `server_shared_pd_` に移転 |
| 3 | CM ID ダブルフリー (既存バグ) | High | `accept_connection()` の `rdma_destroy_id(client_id)` 削除 (デストラクタが処理) |

## 機能テスト結果

| テスト | 条件 | pp (t/s) | tg (t/s) | 結果 |
|--------|------|:--------:|:--------:|:----:|
| qwen2.5-0.5b | 1C+1R, per-device | 3121 | 145 | OK |
| gpt-oss-20b | 1C+2R, per-device | 397 | 56 | OK |
| GLM-4.7 IQ2_M | 7C+4R, per-device | 6.1 | 7.2 | OK |
| GLM-4.7 IQ2_M | 7C+4R, shared (fallback) | 6.7 | 7.4 | OK |
| GLM-4.7 IQ2_M | 7C+4R, per-device + GDR 12GB | 6.4 | 7.2 | OK |
| GLM-4.7 IQ2_M | 7C+4R, shared (最終デフォルト) | 6.7 | 7.5 | OK |
| GLM-4.7 IQ2_M | 7C+4R, per-device (opt-in) | 6.2 | 7.2 | OK |

## 定量評価

### 交絡因子チェック

- [x] 単一変数の分離: 環境変数 (`GGML_RDMA_PER_DEVICE_CONN`) のみで条件切替 (同一バイナリ)
- [x] ホットパスのログ出力なし (fprintf 二峰性問題は修正済み)
- [x] パイロットランで分布の単峰性を確認 (tg は全ペア一致)

### 実験設計

- **条件 A**: PD 共有 per-device connection (`GGML_RDMA_PER_DEVICE_CONN=1`)
- **条件 B**: 単一共有接続 (デフォルト、環境変数なし)
- **モデル**: GLM-4.7 IQ2_M, 11GPU (7 CUDA + 4 RDMA)
- **交互実行**: A→B→A→B... (15ペア)
- **ウォームアップ**: 1回 (破棄)

### 生データ

| Pair | A_pp | A_tg | B_pp | B_tg | d_pp | d_tg |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 1 | 6.3 | 7.2 | 6.7 | 7.5 | -0.40 | -0.30 |
| 2 | 6.3 | 7.2 | 6.7 | 7.4 | -0.40 | -0.20 |
| 3 | 6.4 | 7.2 | 6.7 | 7.4 | -0.30 | -0.20 |
| 4 | 6.2 | 7.2 | 6.7 | 7.5 | -0.50 | -0.30 |
| 5 | 6.3 | 7.2 | 6.7 | 7.4 | -0.40 | -0.20 |
| 6 | 6.3 | 7.2 | 6.7 | 7.4 | -0.40 | -0.20 |
| 7 | 6.5 | 7.2 | 6.7 | 7.4 | -0.20 | -0.20 |
| 8 | 6.2 | 7.2 | **4.5** | 7.4 | +1.70 | -0.20 |
| 9 | 6.3 | 7.2 | 6.8 | 7.4 | -0.50 | -0.20 |
| 10 | 6.2 | 7.2 | 6.7 | 7.4 | -0.50 | -0.20 |
| 11 | 6.4 | 7.2 | 6.6 | 7.4 | -0.20 | -0.20 |
| 12 | 6.2 | 7.2 | 6.6 | 7.4 | -0.40 | -0.20 |
| 13 | 6.1 | 7.2 | 6.6 | 7.4 | -0.50 | -0.20 |
| 14 | 6.4 | 7.2 | 6.7 | 7.4 | -0.30 | -0.20 |
| 15 | 6.4 | 7.2 | 6.7 | 7.5 | -0.30 | -0.30 |

**外れ値**: Pair 8 B の pp=4.5 (IQR法で検出)。tg は正常のため pp のみの一時的な異常。

### 記述統計

| 条件 | pp (t/s) | tg (t/s) |
|------|:--------:|:--------:|
| A (per-device) | 6.300 ± 0.107 | 7.200 ± 0.000 |
| B (shared) | 6.540 ± 0.567 | 7.420 ± 0.041 |

### 検定結果: Generation (tg)

| 指標 | 値 |
|------|:---:|
| 差分平均 | -0.220 t/s (-2.96%) |
| t(14) | -20.58 |
| p 値 | 7.30 × 10⁻¹² |
| Cohen's d | -5.31 (very large) |
| 95% CI | [-0.243, -0.197] t/s |
| 全ペア負の効果 | 15/15 (100%) |

**判定: 有意な性能低下** (p < 0.05 かつ |効果| > 0.5%)

### 検定結果: Prompt Processing (pp)

| 指標 | 値 |
|------|:---:|
| 差分平均 | -0.240 t/s (-3.67%) |
| t(14) | -1.70 |
| p 値 | 1.11 × 10⁻¹ |
| Cohen's d | -0.44 (small-medium) |
| 95% CI | [-0.542, +0.062] t/s |
| A > B pairs | 1/15 (7%) |

**判定: 統計的に有意でない** (p > 0.05、Pair 8 外れ値の影響で分散が大きい)

ただし、外れ値 (Pair 8) を除外すると 14/14 ペアで B > A であり、実質的に per-device は pp でも遅い。

## 原因分析

Per-device 接続が遅い理由:

1. **クライアント側の逐次実行**: `ggml_backend_sched` が各デバイスに対して graph_compute → get_tensor を順番に呼び出すため、独立 QP による並列化の恩恵がない
2. **Deferred copy の無効化**: per-device モードでは `src_ctx->conn != dst_ctx->conn` となるため `cpy_tensor` が false を返し、deferred copy (-1.45%) が無効化される
3. **接続オーバーヘッド**: 4 接続 × (QP + CQ + recv_buffer + send_buffer) のリソース消費増
4. **サーバースレッド競合**: 4 スレッドが GPU リソースと CUDA コンテキストで競合

### per-device が有効になる条件

`ggml_backend_sched` がデバイス間でコマンドを並列送信するように変更されれば、per-device の恩恵が実現する。具体的には:
- 全デバイスの graph_compute をまとめて送信 (fire-and-forget)
- 全デバイスの get_tensor をまとめて要求
- これにより各デバイスの QP が並列にコマンドを処理可能

## 結論

1. **PD 共有 per-device 接続の実装は成功** — ConnectX-4 での MTT キャッシュオーバーフローなく動作
2. **しかし、現状では性能低下** — tg で -2.96% (p < 10⁻¹¹, d=-5.31)
3. **デフォルトは共有接続のまま** — `GGML_RDMA_PER_DEVICE_CONN=1` で opt-in 有効化
4. **将来のクライアント並列化実装後に再評価が必要**

## 成果物

- PD 共有メカニズム (rdma-transport.h/cpp)
- PD 所有権管理 (`owns_pd_`, `release_pd_ownership()`)
- CM ID ダブルフリーバグ修正 (既存バグ)
- サーバー側 PD 共有 (`server_shared_pd_`)
- 環境変数 `GGML_RDMA_PER_DEVICE_CONN` (opt-in)

## 再現方法

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-shared-pd

# ビルド・デプロイ
bash scripts/rdma-build.sh local && bash scripts/rdma-deploy.sh

# サーバー起動
bash scripts/rdma-server.sh restart

# 条件 A (per-device, opt-in)
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_PER_DEVICE_CONN=1 \
  LD_LIBRARY_PATH=build/bin build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log

# 条件 B (shared, デフォルト)
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  LD_LIBRARY_PATH=build/bin build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```
