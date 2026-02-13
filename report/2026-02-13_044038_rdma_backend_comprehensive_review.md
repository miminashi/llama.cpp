# RDMA バックエンド総合調査レポート

- **実施日時**: 2026年2月13日 04:40
- **ワークツリー**: `.worktree/rdma-backend`

## 前提・目的

GPU テスト実行中の空き時間を活用し、RDMA バックエンドの現状を以下5つの観点から総合的に調査・文書化する。

1. コード品質・静的解析
2. Generation 速度改善の設計調査
3. ビルド・デプロイスクリプトの改善提案
4. テストインフラの現状と提案
5. 16GPU 拡張の準備状況

### 対象ファイル

| ファイル | 行数 | 役割 |
|---------|------|------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | 3,421 | クライアント/サーバー本体 |
| `ggml/src/ggml-rdma/rdma-transport.cpp` | 968 | RDMA 接続・通信レイヤー |
| `ggml/src/ggml-rdma/rdma-transport.h` | ~200 | 接続構成定義 |
| `ggml/src/ggml-backend.cpp` | ~1,800 | ggml バックエンドスケジューラ |
| `scripts/rdma-{build,deploy,server}.sh` | 41+22+61 | ビルド・デプロイ・サーバー管理 |
| `tools/rdma/rdma-{simple-test,test-client}.cpp` | 82+170 | テストプログラム |

---

## セクション1: コード品質・静的解析

### 1.1 メモリリークリスク

#### (a) レジストリ/デバイスコンテキストの永続確保 — Severity: **Info** (意図的)

`ggml_backend_rdma_reg()` (L1855-1915) と `ggml_backend_rdma_add_server()` (L1917-1970) で `new` 確保されるオブジェクトにはプロセス終了まで対応する `delete` がない。

```
L1863: new ggml_backend_rdma_reg_context   — delete なし
L1894: new ggml_backend_rdma_device_context — delete なし
L1900: new ggml_backend_device              — delete なし
L1942: new ggml_backend_rdma_reg_context   — delete なし (add_server)
L1947: new ggml_backend_reg                — delete なし (add_server)
L1954: new ggml_backend_rdma_device_context — delete なし (add_server)
L1960: new ggml_backend_device              — delete なし (add_server)
```

これらは llama.cpp のバックエンドレジストリの設計慣習に従ったもので、`static` 変数やグローバルマップ (`reg_map`, L1921) に格納されプロセス寿命で管理される。他のバックエンド (CUDA, Metal 等) も同様のパターンを使用しており、実質的な問題ではない。

#### (b) バッファコンテキストの確保/解放 — Severity: **Low**

`ggml_backend_rdma_buffer_type_alloc_buffer()` (L1124-1167):

```cpp
// L1146: new で確保
auto * ctx = new ggml_backend_rdma_buffer_context { ... };

// L1159: ggml_backend_buffer_init に渡す
ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ..., ctx, ...);
```

対応する `delete` は `ggml_backend_rdma_buffer_free_buffer()` (L866-876) に存在する:

```cpp
// L875: delete ctx;
```

**潜在リスク**: L1159 の `ggml_backend_buffer_init()` が null を返す場合、`ctx` がリークする。ただし現在の `ggml_backend_buffer_init()` は `GGML_ASSERT` で失敗時にプロセスを終了するため、実際のリークパスは存在しない。

#### (c) バックエンドコンテキストの確保/解放 — Severity: **Info** (正常)

`ggml_backend_rdma_init()` (L1710-1749):

```cpp
// L1730: new ggml_backend_rdma_context
// L1741: new ggml_backend
```

対応する `delete` は `ggml_backend_rdma_free()` (L1302-1306):

```cpp
// L1304: delete ctx;
// L1305: delete backend;
```

これは正常なライフサイクル管理。問題なし。

#### (d) `setup_qp()` の部分失敗 — Severity: **Low**

`rdma-transport.cpp:setup_qp()` (L271-323) で、CQ 作成成功 (L280) 後に QP 作成 (L297) が失敗した場合、CQ と PD がこの関数内では解放されない:

```cpp
// L280: cq_ = ibv_create_cq(...);  // 成功
// L297: if (rdma_create_qp(...) != 0) {
//           return false;  // ← cq_ と pd_ が未解放
//       }
```

ただし、呼び出し元が `false` を受けた場合、`rdma_connection` デストラクタ (L222-269) で `disconnect()` が呼ばれ、`cq_` と `pd_` は正しくクリーンアップされる:

```cpp
// L243: ibv_destroy_cq(cq_);
// L253: ibv_dealloc_pd(pd_);
```

**結論**: リソースリークはデストラクタで防がれているが、`setup_qp()` 内で明示的にクリーンアップするとより堅牢。

### 1.2 スレッド安全性

#### (a) `graph_cache` クラス — Severity: **Low** (現状問題なし、将来のリスク)

`graph_cache` (L460-611) のメンバ変数:

```cpp
// L506: std::vector<ggml_tensor> last_graph;
// L609: std::unordered_map<uint64_t, ggml_tensor> snapshot_all_;
// L610: std::unordered_map<uint64_t, const ggml_tensor*> snapshot_map_;
```

これらは mutex なしでアクセスされる。`graph_cache` は `ggml_backend_rdma_context` (L613) のメンバとして各バックエンドインスタンスに1つずつ存在し、`graph_compute()` から呼ばれる。

**現状**: `ggml_backend_sched_compute_splits()` (L1443-1627) の逐次ループにより、同一バックエンドの `graph_compute` が同時に呼ばれることはない。そのため現在は問題なし。

**将来のリスク**: Two-phase loop (セクション2参照) を実装して複数デバイスの `graph_compute` を並列化する場合、各デバイスは別々の `graph_cache` インスタンスを持つため依然として安全。ただし、同一接続に対する `send_rdma_cmd` の並列呼び出しには `rdma_connection` 側の mutex が必要になる。

#### (b) `pending_flush_list` — Severity: **None** (安全)

`pending_flush_list` (L370-412) は内部に `std::mutex` を持ち、`add()` と `drain()` の両方で `lock_guard` を使用している:

```cpp
// L372: std::lock_guard<std::mutex> lock(mutex);  // add()
// L382: std::lock_guard<std::mutex> lock(mutex);  // drain()
```

グローバルマップ `g_conn_pending_flushes` (L399) も専用 mutex (`g_conn_pending_flushes_mutex`, L400) で保護されている。スレッドセーフ。

### 1.3 エラーハンドリング

#### (a) `alloc_host_staging()` の返り値なし — Severity: **Medium**

`ggml-rdma.cpp` のサーバー側関数 `alloc_host_staging()` は `void` 返り値で、内部のエラー (メモリ確保失敗、MR 登録失敗) は `GGML_LOG_ERROR` でログ出力した後 `return` するだけ:

```cpp
// 呼び出し元 (L2124, L2134, L2141) はエラーを検知できない
alloc_host_staging(buffer, size, conn, response);
// → response の mr_addr, mr_rkey が未設定のまま使われる可能性
```

**影響**: ホストステージング確保が失敗した場合、レスポンスの MR 情報が不正な値となり、クライアントが RDMA Write を試行して `remote access error` が発生する。ただし、OOM 以外でこのパスに到達することは稀。

#### (b) `n_flush` の型 — Severity: **None** (問題なし)

`n_flush` は `uint32_t` (L1363) で、`pending.size()` の戻り値 (`size_t`) から暗黙変換される:

```cpp
// L1363: uint32_t n_flush = pending.size();
```

`dirty_buffers` は `unordered_map<uint64_t, pair>` であり、キーはバッファの `remote_ptr`。一接続あたりのバッファ数は数十〜数百程度であるため、`uint32_t` (最大 42 億) のオーバーフローリスクは実質ゼロ。

### 1.4 コード品質

#### (a) `cpy_tensor` 無効化のワークアラウンド — 意図的

`ggml_backend_rdma_buffer_cpy_tensor()` (L1083-1096) は常に `false` を返す。詳細なコメント (10行) で理由を説明しており、意図的な設計判断:

```cpp
// L1083-1096: server-side copy_tensor が graph_cache と不整合を起こすため無効化
// パフォーマンス影響: ~3% (inter-layer boundary tensors のみ)
return false;
```

#### (b) `new/delete` と `unique_ptr` の混在

バッファコンテキスト内では `unique_ptr` を使用 (`mem_pool`, `staging`, L448-451) する一方、コンテキスト自体は raw `new/delete` で管理する。これは ggml のバックエンドインターフェースが `void * context` を期待するためで、他のバックエンドも同様のパターン。改善可能だが優先度は低い。

### 推奨アクション

| 項目 | 優先度 | アクション |
|------|--------|-----------|
| `alloc_host_staging` 返り値 | Medium | `bool` 返り値に変更し、呼び出し元でエラーハンドリング |
| `setup_qp` 部分失敗 | Low | CQ 作成後の失敗パスで `ibv_destroy_cq`/`ibv_dealloc_pd` を追加 |
| `graph_cache` スレッド安全性 | Low | 将来の並列化時に再評価。現状は不要 |
| レジストリ永続確保 | Info | llama.cpp の慣習に従い、変更不要 |

---

## セクション2: Generation 速度改善の設計調査

### 2.1 現状のボトルネック

`ggml_backend_sched_compute_splits()` (`ggml-backend.cpp:1443-1627`) がすべてのデバイス split を**逐次実行**する:

```
for split_id in [0..n_splits):     // L1451
    input_copy(split)               // L1457-1577
    graph_compute_async(split)      // L1580
    event_record(split)             // L1621
```

この構造では、各 split の `graph_compute_async` は前の split の `input_copy` が完了してから呼ばれる。RDMA バックエンドでは `input_copy` が `get_tensor` (前デバイスの結果取得) を含むため、全デバイスが直列化される:

```
D0: compute(12ms) → D0: get_tensor(wait) → D1: set_tensor → D1: compute(12ms) → ...
```

### 2.2 RPC との差異

RPC バックエンドはデバイスごとに独立 TCP ソケットを持つ。スケジューラの逐次ループでも、各ソケットの送信バッファが独立にデータを保持するため、実質的にコマンド送信がパイプライン化される。

RDMA バックエンドは全デバイスが1つの RDMA 接続 (QP) を共有するため、コマンドの送受信が完全に直列化される。

### 2.3 改善案

#### 案 (A): Two-phase loop — **推奨**

スケジューラの split ループを2フェーズに分離:

```
Phase 1: for split: input_copy + graph_compute_async  (fire-and-forget)
Phase 2: for split: event_wait + output_retrieval
```

**実装箇所**: `ggml-backend.cpp:1451-1624` のループ構造変更

**利点**: バックエンド API の変更不要。既存の `graph_compute_async` (RDMA の fire-and-forget) と組み合わせ可能

**課題**: split 間に入力依存がある場合 (split N+1 の入力が split N の出力)、Phase 1 内でも同期が必要。`input_copy` の L1566 で `cpy_tensor_async` が失敗すると同期コピーにフォールバックし、前の split の完了を待つ必要がある

**期待される改善**: 4 RDMA デバイスの場合、逐次 48ms → 並列 12ms + 同期オーバーヘッド ≈ 15-20ms

#### 案 (B): Backend interface 拡張 — 中期的

新しいインターフェース関数を追加:

```cpp
// バックエンドインターフェースに追加
graph_compute_prepare(backend, graph)  // グラフ送信のみ (応答不要)
graph_compute_await(backend)           // 結果待ち
```

**利点**: バックエンド側で最適なパイプライン化が可能

**課題**: llama.cpp upstream の API 変更が必要。マージの障壁が高い

#### 案 (C): Per-device connection — **既に実装済み**

`GGML_RDMA_PER_DEVICE_CONN=1` で有効化。各デバイスが独立 QP を持つことで、RPC と同様のパイプライン効果が得られる。

**現状の制約**: ConnectX-4 の MTT キャッシュ制限 (~10-16GB) により、大規模モデルでは RDMA Write がタイムアウトする。ConnectX-6+ では問題なし。

#### 案 (D): Pipeline with lookahead — 長期的

D_i の `get_tensor` と D_{i+1} の `graph_compute` を重畳:

```
D0: compute → [D0: get_tensor | D1: compute] → [D1: get_tensor | D2: compute] → ...
```

**利点**: 理論上の最適解。通信と計算の完全なオーバーラップ

**課題**: 実装複雑度が高い。`rdma_connection` の送受信をマルチスレッド化する必要がある

### 推奨アクション

| 案 | 優先度 | 実現可能性 | 期待効果 |
|----|--------|-----------|---------|
| (A) Two-phase loop | **High** | Medium | Generation +50-100% |
| (C) Per-device conn | **High** | 実装済み | ConnectX-6+ で有効 |
| (B) Interface 拡張 | Medium | Low (upstream) | Generation +100-200% |
| (D) Pipeline | Low | Low | 理論上最適 |

---

## セクション3: ビルド・デプロイスクリプトの改善提案

### 3.1 現状の評価

3スクリプト (`rdma-build.sh`, `rdma-deploy.sh`, `rdma-server.sh`) はいずれも良い実践に基づいている:

- `set -euo pipefail` による安全なエラーハンドリング
- 明確なエラーメッセージとステップ番号
- ビルドログの `/tmp` 出力
- Claude Code の SSH 自動承認問題を回避するラッパー設計

### 3.2 改善提案

#### (a) 環境変数によるパラメータ上書き — 優先度: **Medium**

3スクリプトとも `NODE2`, `REMOTE_DIR`, `PORT` がハードコードされている:

```bash
# rdma-build.sh L4, rdma-deploy.sh L4, rdma-server.sh L4
NODE2="192.168.100.2"
```

16GPU 拡張時に3号機以降が追加される場合、各スクリプトの修正が必要になる。

**改善案**:

```bash
NODE2="${RDMA_NODE2:-192.168.100.2}"
REMOTE_DIR="${RDMA_REMOTE_DIR:-/home/ubuntu/projects/llama.cpp}"
PORT="${RDMA_SERVER_PORT:-50051}"
```

#### (b) SSH 接続事前チェック — 優先度: **Low**

`rdma-deploy.sh` は SSH 接続が失敗すると `rm -rf` 後の rsync でエラーになり、2号機のコードが消失した状態で停止する。

**改善案**: スクリプト冒頭で SSH 接続テスト:

```bash
ssh -o ConnectTimeout=5 "$NODE2" "true" || { echo "ERROR: Cannot connect to $NODE2"; exit 1; }
```

#### (c) rsync dry-run オプション — 優先度: **Low**

`rdma-deploy.sh` の `rsync -a` (L16) に `--delete` がないため、2号機に古いファイルが残る可能性がある。ただし L13 で `rm -rf` しているため実質問題なし。

`--dry-run` モードの追加は情報提供用に有用:

```bash
if [ "${DRY_RUN:-}" = "1" ]; then
    rsync -an --exclude='.git' "$WORKTREE_DIR/" "$NODE2:${REMOTE_DIR}/"
    exit 0
fi
```

#### (d) サーバーログのローテーション — 優先度: **Low**

`rdma-server.sh` の `server_start()` (L21) は `/tmp/rdma-server.log` を上書きする。長時間運用時に過去のログが失われる。

**改善案**: タイムスタンプ付きログ + シンボリックリンク:

```bash
LOG="/tmp/rdma-server-$(date +%Y%m%d_%H%M%S).log"
ssh "$NODE2" "... > $LOG 2>&1 &"
ssh "$NODE2" "ln -sf $LOG /tmp/rdma-server.log"
```

### 推奨アクション

| 項目 | 優先度 | 影響範囲 |
|------|--------|---------|
| 環境変数パラメータ化 | Medium | 16GPU 拡張時に必須 |
| SSH 事前チェック | Low | デプロイ安全性向上 |
| rsync dry-run | Low | デバッグ用 |
| ログローテーション | Low | 長時間運用時 |

---

## セクション4: テストインフラの現状と提案

### 4.1 既存テスト

#### `rdma-simple-test.cpp` (82行)

最小限の接続テスト。9ステップで基本機能を検証:

1. GPUDirect 可用性チェック
2. サーバー接続
3. デバイスメモリ照会
4. バッファタイプ取得
5. テンソル作成 (F32 × 16)
6. バッファ確保
7. `set_tensor` (64バイト)
8. `get_tensor` (64バイト)
9. クリーンアップ

**カバレッジ**: 基本接続、小テンソルの set/get のみ。データ検証は目視 (値の印刷)。

#### `rdma-test-client.cpp` (170行)

より詳細なテスト。引数でデバイス番号とテンソルサイズを指定可能:

1. 接続 + デバイスメモリ照会
2. 2テンソル (a, b) の作成・確保
3. `set_tensor` (configurable サイズ)
4. `get_tensor` + 浮動小数点比較検証 (1e-6 tolerance)
5. RDMA 統計表示
6. 終了コードによる合否判定 (`return (match_a && match_b) ? 0 : 1`)

**カバレッジ**: 基本接続、カスタムサイズテンソルの round-trip 検証、RDMA 統計。

### 4.2 不足しているテスト

#### (a) マルチデバイステスト — 優先度: **High**

現在のテストは単一デバイス (device=0) のみ。RDMA バックエンドの主要な使用パターンである複数リモートデバイスへの同時アクセスがテストされていない。

**必要なテスト**:
- 複数デバイスへの同時接続・テンソル操作
- デバイス間でのテンソルコピー (set_tensor → get_tensor のデバイス間パス)
- 接続共有モード vs per-device 接続モード

#### (b) 大バッファテスト — 優先度: **High**

既知のバグ (mmap + RDMA Write, 4GB 制限) に関連するサイズ境界のテストが不足:

**必要なテスト**:
- 4GB 超のバッファ確保 (Send/Recv フォールバックの検証)
- 16MB 超のテンソル (チャンク送受信の検証)
- バッファサイズ境界値 (4GB ± 1バイト)

#### (c) GDR バジェット枯渇テスト — 優先度: **Medium**

`GGML_RDMA_GDR_BUDGET_GB` の挙動をテストする:

**必要なテスト**:
- バジェット超過時のホストステージングフォールバック
- `mr_is_gdr` フラグの正しい設定
- GDR と非 GDR バッファの混在時の `get_tensor` パス選択

#### (d) 接続回復テスト — 優先度: **Medium**

**必要なテスト**:
- クライアント正常切断後のサーバー再接続
- クライアント異常切断 (kill -9) 後のサーバー状態
- サーバー再起動後のクライアント再接続

#### (e) graph_compute テスト — 優先度: **Medium**

**必要なテスト**:
- 簡単なグラフ (add, mul_mat) のリモート実行
- graph_cache の hit/miss 検証
- async compute の正常動作

#### (f) 自動テスト実行基盤 — 優先度: **Low**

現在のテストは手動実行のみ。CI/CD 統合のためには:
- テスト用の軽量サーバーモック (実際の RDMA ハードウェアなしで実行可能)
- テストスクリプト (サーバー起動 → テスト実行 → サーバー停止)
- 合否判定の統一 (終了コード)

### 推奨アクション

| テスト種別 | 優先度 | 実装難易度 | 既知バグとの関連 |
|-----------|--------|-----------|----------------|
| マルチデバイス | High | Medium | Multi-RDMA 出力破損 |
| 大バッファ | High | Low | mmap + RDMA Write |
| GDR バジェット | Medium | Medium | GPUDirect timeout |
| 接続回復 | Medium | Medium | クライアント異常切断 |
| graph_compute | Medium | High | graph_cache 整合性 |
| 自動テスト基盤 | Low | High | — |

---

## セクション5: 16GPU 拡張の準備状況

### 5.1 ハードコードされた制限の有無

RDMA バックエンドにデバイス数やノード数のハードコード制限は**存在しない**:

- **接続管理**: `rdma_connection_manager` クラス (`rdma-transport.h:152`) が接続を管理。`get_connection()` でキー `"host:port#device"` により動的に接続を作成・取得する
- **デバイス ID**: `uint32_t device` で動的に割り当て。サーバーの `RDMA_CMD_DEVICE_COUNT` レスポンスに基づく
- **レジストリ**: `reg_ctx->devices` は `vector<ggml_backend_device*>` で動的追加

### 5.2 要調整パラメータ

#### (a) WR depth (Work Request depth) — **要注意**

`rdma_config` (`rdma-transport.h:22-23`):

```cpp
uint32_t max_send_wr = 128;
uint32_t max_recv_wr = 128;
uint32_t cq_size     = 256;
```

16GPU (8+8) 構成では、1接続あたり最大8デバイスのコマンドが送受信される。現在の 128 WR は十分余裕がある (1デバイスあたり ~16 WR で十分)。

**判定**: 変更不要。

#### (b) GDR バジェット — **要調整**

デフォルト 12GB (`GGML_RDMA_GDR_BUDGET_GB`)。16GPU 構成では8台のリモート GPU に分散されるため、per-GPU のバッファサイズは現在の 11GPU 構成より小さくなる可能性がある。

GLM-4.7 Q4 のモデルサイズを仮定すると:
- Q4_K_M: ~50GB → 16GPU で ~3.1GB/GPU
- IQ2_M: ~40GB → 16GPU で ~2.5GB/GPU (11GPU の ~3.6GB/GPU より小さい)

12GB バジェットで8デバイス × 3.1GB = 24.8GB をカバーするのは不可能。ただし、GDR バジェットは ConnectX-4 の MTT キャッシュ制限を考慮した値であり、16GPU でも物理的な制約は同じ。

**判定**: デフォルト値の変更は不要。ただし、16GPU でのテスト時にバジェットを 8GB に下げる必要がある可能性あり (MTT キャッシュが MR 数にも依存するため)。

#### (c) ステージングバッファサイズ — **監視が必要**

ステージングバッファはバッファ確保時にサーバー側で作成される。16GPU では8つのステージングバッファが同時に存在し、合計サイズが ConnectX-4 の MTT キャッシュを圧迫する可能性がある。

現在の 4GB per-buffer サイズ制限 (`RDMA_WRITE_MAX_BUFFER_SIZE`) により、大バッファは Send/Recv フォールバックするため、ステージングの MR 登録は制限される。

**判定**: 4GB 制限が機能している限り問題なし。16GPU 初期テストで MTT エラーが出た場合は `GGML_RDMA_GDR_BUDGET_GB` を下げて対応。

### 5.3 スケーラビリティ懸念

#### (a) サーバースレッドモデル

現在のサーバーは接続ごとに1スレッド。16GPU (8リモート) では1接続 × 1スレッドで8デバイスを処理する (共有接続モード)。

`compute_dispatcher` (並列 compute dispatch) は実装済みだが、e2e の改善効果はクライアント側スケジューラの逐次性により制限される (MEMORY.md 参照)。

**判定**: 現状の設計で 16GPU に対応可能。ボトルネックはサーバーではなくクライアント側。

#### (b) MTT キャッシュ — **最大リスク**

ConnectX-4 の MTT キャッシュは ~10-16GB。16GPU 構成で GDR MR + ステージング MR の合計が増加すると、RDMA Write タイムアウトのリスクが高まる。

現在の緩和策:
- GDR バジェット (12GB) で GDR MR の合計を制限
- 4GB per-buffer 制限で大ステージング MR を回避
- Send/Recv フォールバックでステージング不要のパスを提供

**判定**: 既存の緩和策で対応可能だが、16GPU 初期テストでの慎重な検証が必要。

#### (c) コマンドスループット

単一 QP で 8 デバイスのコマンドを処理する場合、1トークンあたり 8× の graph_compute + 8× の get_tensor + 8× の set_tensor = 最大 24 round-trip/token。

現在の 11GPU (4 RDMA) では ~12 round-trip/token で Generation 6.8 t/s。16GPU (8 RDMA) では ~24 round-trip/token となり、Generation 速度が約半分 (~3.4 t/s) に低下する可能性がある。

**判定**: 許容可能かどうかはモデルの計算量に依存。GLM-4.7 Q4 は IQ2_M より計算量が多いため、compute time が dominant なら影響は小さい。

### 5.4 16GPU 拡張チェックリスト

| 項目 | 状態 | 備考 |
|------|------|------|
| デバイス数制限なし | ✅ | 動的管理 |
| WR depth 十分 | ✅ | 128 WR / 256 CQ |
| GDR バジェット | ⚠️ | テスト時に調整の可能性 |
| MTT キャッシュ | ⚠️ | 既存緩和策あり、要検証 |
| サーバースレッド | ✅ | 1接続1スレッドで十分 |
| コマンドスループット | ⚠️ | 24 round-trip/token、要計測 |
| スクリプト対応 | ❌ | 3号機以降のサポート未対応 |

### 推奨アクション

| 項目 | 優先度 | アクション |
|------|--------|-----------|
| 16GPU 初期テスト | **High** | GPU 追加後、まず 2デバイス接続テストから段階的に拡張 |
| GDR バジェット調整 | Medium | 8 RDMA デバイスでの MTT エラー有無を監視 |
| スクリプト拡張 | Medium | 環境変数パラメータ化 (セクション3参照) |
| コマンドスループット計測 | Medium | `GGML_RDMA_PROFILE=1` で 8 RDMA のレイテンシ分布を収集 |

---

## 総括

### 全体評価

RDMA バックエンドは 11GPU (7C+4R) での GLM-4.7 IQ2_M 安定動作を達成しており、コード品質は実用レベルに達している。16GPU 拡張に向けたハードコード制限はなく、既存の緩和策 (GDR バジェット、4GB バッファ制限) により ConnectX-4 の物理的制約に対応している。

### 最優先改善項目

1. **Generation 速度の RPC 比劣位** — Two-phase loop (案A) の実装がもっとも効果的
2. **テストカバレッジ** — マルチデバイステスト、大バッファテストの追加
3. **16GPU 拡張準備** — スクリプトの環境変数パラメータ化

### リスク要因

- ConnectX-4 の MTT キャッシュが 8 RDMA デバイスでオーバーフローする可能性
- コマンドスループットの線形劣化 (4→8 RDMA で ~2× 低下)
- `alloc_host_staging` のエラーハンドリング欠如 (OOM 時のみだが)
