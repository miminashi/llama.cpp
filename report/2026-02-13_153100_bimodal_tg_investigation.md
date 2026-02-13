# GLM-4.7 Generation速度 二峰性分布の原因調査レポート

- **実施日時**: 2026年2月13日 04:40 - 15:31
- **ワークツリー**: `.worktree/rdma-backend` (調査用: `.worktree/bimodal-investigation`)
- **参照レポート**: [Deferred Copy 統計ベンチマーク](2026-02-13_090026_deferred_copy_statistical_benchmark.md)

## 前提・目的

### 背景

Deferred Copy 統計ベンチマーク (20ペア, 41ラン) で GLM-4.7 IQ2_M (7C+4R, 11GPU) の Generation速度に明確な二峰性分布が観測された:

- **低速モード**: ~5.5 t/s (19/41ラン, 46%)
- **高速モード**: ~6.7 t/s (22/41ラン, 54%)
- **ギャップ**: 5.7以下と6.5以上の間に中間値なし (18%の性能差)

各ラン内では50トークン全てが同じモードで一貫し、ラン間でランダムに切り替わるという特徴があった。

### 目的

二峰性分布の根本原因を特定し、修正を適用して安定した Generation速度を実現する。

### 前提条件

- 1号機: 7x P100 (CUDA0-6), 2号機: 4x P100 (RDMA0-3)
- モデル: GLM-4.7 IQ2_M (約40GB)
- GDR budget: 12GB (デフォルト)
- Deferred copy: 有効
- Async compute: 有効
- rdma-server は `/tmp/rdma-server.log` にログ出力

## 調査過程

### Phase 1: GPU クロック/P-state 調査 (仮説1)

**仮説**: P100の GPU Boost によるクロック変動が二峰性の原因。

**Step 1a: クロック情報収集**
- 1号機: アイドル時 405 MHz, アプリケーションクロック 1189 MHz, 最大ブースト 1328 MHz
- 2号機: アクティブ時 1189 MHz (rdma-server 実行中), 最大ブースト 1328 MHz
- 両ノードとも温度 27-34°C でサーマルスロットリングなし
- Persistence mode: 両ノードとも無効 (調査中に有効化)

**Step 1b: ベンチマーク中のクロックモニタリング**

`nvidia-smi dmon -s pc -d 1` で1秒間隔のクロック監視を実行しながら16回ベンチマーク:

| Run | tg (t/s) | モード | 備考 |
|-----|----------|--------|------|
| 1-6 | 6.7-7.0 | HIGH | 初期6回は全て HIGH |
| 7 | 5.9 | LOW | |
| 8 | 7.0 | HIGH | |
| 9 | 5.7 | LOW | **全GPU 1328 MHz** |
| 10 | 7.0 | HIGH | **全GPU 1328 MHz** |
| 11 | 6.2 | LOW | |
| 12 | 6.9 | HIGH | |
| 13 | 6.7 | HIGH | |
| 14 | 5.8 | LOW | |
| 15-16 | 7.0 | HIGH | |

**Step 1c: 判定**

Run 9 (5.7 t/s, LOW) と Run 10 (7.0 t/s, HIGH) で全GPU が同一クロック (1328 MHz) であったことから、**GPU クロックは二峰性の原因ではない**と判定。

### Phase 2: GPU クロックロックによる検証

P100 は `nvidia-smi -lgc` (locked GPU clocks) を非サポートのため、代わりに `-ac 715,1328` (application clocks) で最大クロックに固定。

```bash
sudo nvidia-smi -pm 1               # persistence mode 有効化
sudo nvidia-smi -ac 715,1328         # 両ノードで実行
```

クロック固定 (1328 MHz) で10回実行:

| Run | pp (t/s) | tg (t/s) | モード |
|-----|----------|----------|--------|
| 1 | 7.4 | 7.1 | HIGH |
| 2 | 7.4 | 7.1 | HIGH |
| 3 | 7.4 | 7.4 | HIGH |
| 4 | 7.4 | 7.4 | HIGH |
| 5 | 7.1 | 5.9 | LOW |
| 6 | 7.4 | 5.9 | LOW |
| 7 | 7.4 | 5.9 | LOW |
| 8 | 7.1 | 6.0 | LOW |
| 9 | 7.4 | 7.5 | HIGH |
| 10 | 7.4 | 7.5 | HIGH |

**結果**: クロック固定でも二峰性は消えず。連続LOW (Run 5-8) の「スティッキー」挙動を確認。
**副発見**: アプリケーションクロック 1189→1328 MHz で pp が 6.9→7.4 t/s に改善 (+7%)。

### Phase 3: サーバー側プロファイリング

`.worktree/bimodal-investigation` ワークツリーで `GGML_RDMA_PROFILE=1` を有効化してサーバーを起動し、6回ベンチマーク。

| Run | tg (t/s) | モード |
|-----|----------|--------|
| 1 | 7.1 | HIGH |
| 2 | 5.9 | LOW |
| 3 | 5.9 | LOW |
| 4 | 5.9 | LOW |
| 5 | 7.4 | HIGH |
| 6 | 7.0 | HIGH |

**サーバープロファイルの比較分析:**

| メトリクス | LOW (Run 2) | HIGH (Run 5) | 差 |
|-----------|-------------|--------------|-----|
| graph_recompute 平均 | 11.55 ms | 11.55 ms | **同一** |
| graph_recompute 合計 | 2217 ms | 2217 ms | **同一** |
| ASYNC command 合計 | 8199 ms | 6558 ms | +1641 ms |
| flush_and_recompute 合計 | 4087 ms | 2392 ms | +1695 ms |
| flush 時間 (outer) | 2045 ms | 351 ms | **+1694 ms** |
| flush 時間 (inner) | ~15 ms | ~15 ms | **同一** |

**決定的な発見**: `flush_and_recompute` の外側計測 (呼び出し元) と内側計測 (関数内) に巨大な乖離があった。LOW ランでは ~1700ms のスパイクが1回発生し、それが全体のtg低下を説明していた。

具体的なスパイク:
- Run 2: 位置64/192 で 1707.69 ms (inner flush=1695.99ms, しかし `flush_all_staging` 内部測定は 1.63ms)
- Run 3: 位置65/192 で 1685.35 ms (inner flush=1.63ms, compute=11.17ms, **1672ms が説明不能**)
- Run 4: 位置65/192 で 1731.29 ms

**乖離の原因**: `flush_all_staging()` と `graph_recompute()` の内部計測は `fprintf` 呼び出し前に完了するが、外側の `flush_and_recompute()` 計測はこれらの `fprintf` 呼び出し時間を含んでいた。つまり **`fprintf(stderr, ...)` の I/O ブロッキングが 1700ms のスパイクの原因**。

### 決定的検証: stderr → /dev/null テスト

サーバーの stderr を `/dev/null` にリダイレクトして起動し、10回ベンチマーク:

```bash
ssh 192.168.100.2 "LD_LIBRARY_PATH=... nohup rdma-server -H 0.0.0.0 -p 50051 > /dev/null 2>&1 &"
```

**結果: 10/10 ラン全てが 7.4-7.5 t/s。二峰性は完全に消失。**

## 根本原因

### 原因

rdma-server の `get_tensor()` 関数に含まれる **always-on fprintf(stderr, ...)** がファイル I/O のブロッキングにより ~1700ms のスパイクを引き起こしていた。

GLM-4.7 (11GPU, 4 RDMA デバイス) では、Generation の各トークン生成時に `get_tensor()` が192回呼ばれる（4デバイス × 48呼び出し）。毎回 fprintf が実行されるため、ファイルI/Oのバッファフラッシュタイミングによって:

- **バッファフラッシュが発生しないラン**: 7.4-7.5 t/s (HIGH)
- **バッファフラッシュが発生するラン**: ~1700ms のスパイクにより 5.5-6.0 t/s (LOW)

スパイクは各ラン内で最大1回発生し、発生位置はラン間で微妙に異なる (位置 64-65/192)。これは stdio バッファ (通常 8KB) がファイルリダイレクト時にフルバッファリングモードになり、バッファが満杯になった時点で一括 write() するためと推定される。

### 影響を受けた fprintf 呼び出し

1. `get_tensor()` のバッファ/オフセット情報ログ (毎回実行)
2. `get_tensor()` の応答データ非ゼロチェックログ (毎回実行)
3. `fix_cross_device_refs()` の deferred copy ルックアップログ (毎 graph_recompute で実行)

## 修正内容

### 変更ファイル

`ggml/src/ggml-rdma/ggml-rdma.cpp`:

1. **`get_tensor()` のバッファ情報ログ**: `fprintf(stderr, ...)` → `RDMA_LOG_DBG(...)` に変更。`GGML_RDMA_DEBUG=1` 時のみ出力。

2. **`get_tensor()` の応答データ非ゼロチェック**: コードブロック全体を削除。デバッグ用コードがホットパスに残っていた。

3. **`fix_cross_device_refs()` のログ3箇所**: `fprintf(stderr, ...)` → `RDMA_LOG_DBG(...)` に変更。

### 修正方針

- ホットパス (トークン生成ごとに呼ばれるコード) の always-on fprintf を全て排除
- デバッグに必要なログは `RDMA_LOG_DBG` マクロ (`GGML_RDMA_DEBUG=1` でのみ有効) に移行
- サーバー起動時やエラーパスの fprintf は変更不要 (1回のみ実行)
- `RDMA_PROFILE` でゲートされたプロファイリング fprintf も変更不要 (デフォルト無効)

## 検証結果

### 修正後ベンチマーク (サーバー stderr → ファイルログ出力)

```
=== Verification test (fprintf fix, server logging to file): 10 runs ===
Run 1:  [ Prompt: 6.7 t/s | Generation: 7.4 t/s ]
Run 2:  [ Prompt: 7.4 t/s | Generation: 7.5 t/s ]
Run 3:  [ Prompt: 7.4 t/s | Generation: 7.5 t/s ]
Run 4:  [ Prompt: 7.4 t/s | Generation: 7.5 t/s ]
Run 5:  [ Prompt: 7.4 t/s | Generation: 7.5 t/s ]
Run 6:  [ Prompt: 7.4 t/s | Generation: 7.5 t/s ]
Run 7:  [ Prompt: 7.4 t/s | Generation: 7.5 t/s ]
Run 8:  [ Prompt: 7.4 t/s | Generation: 7.5 t/s ]
Run 9:  [ Prompt: 7.4 t/s | Generation: 7.5 t/s ]
Run 10: [ Prompt: 7.4 t/s | Generation: 7.5 t/s ]
```

**10/10 ラン全てが tg=7.4-7.5 t/s。二峰性は完全に解消。**

Run 1 の pp=6.7 は初回起動時の warm-up 効果 (以降は安定して 7.4)。

### 修正前後の比較

| 条件 | pp (t/s) | tg (t/s) | 二峰性 |
|------|----------|----------|--------|
| 修正前 (fprintf あり, ログファイル) | 6.9-7.4 | 5.5-6.7 | **あり** (LOW 46%, HIGH 54%) |
| 修正前 (fprintf あり, /dev/null) | 7.4 | 7.4-7.5 | なし |
| **修正後 (fprintf 除去, ログファイル)** | **7.4** | **7.4-7.5** | **なし** |

### 副次的改善

- アプリケーションクロック 1189→1328 MHz で pp が約7%改善 (6.9→7.4 t/s)
- Persistence mode を両ノードで有効化 (GPU アイドル時のクロックダウンを防止)

## 再現方法

### 修正前の二峰性を再現する場合

1. `get_tensor()` に always-on fprintf を追加 (例: バッファ情報ログ)
2. rdma-server を `/tmp/rdma-server.log` にログ出力して起動
3. GLM-4.7 IQ2_M を 7C+4R で10回以上実行
4. tg が 5.5-6.0 と 6.7-7.5 の二峰分布になることを確認

### 修正後の安定動作を確認する場合

```bash
bash scripts/rdma-server.sh restart

bash /tmp/bench-bimodal-verify.sh 10
```

## まとめ

- **根本原因**: rdma-server の `get_tensor()` 内の always-on `fprintf(stderr, ...)` がファイル I/O ブロッキングで ~1700ms のスパイクを引き起こし、GLM-4.7 の Generation速度に二峰性分布を生じさせていた
- **修正**: ホットパスの fprintf を `RDMA_LOG_DBG` (デバッグ専用マクロ) に移行、不要なデバッグコードを削除
- **結果**: 二峰性が完全に解消され、Generation速度が 7.4-7.5 t/s で安定
- **教訓**: 高頻度で呼ばれるサーバーコードパスに `fprintf(stderr, ...)` を残すと、stderr がファイルにリダイレクトされた場合にバッファフラッシュによる予測不能なスパイクが発生する。ログ出力は必ずデバッグフラグでゲートするか、非同期ログライブラリを使用すべき

## ワークツリー管理

- **調査**: `.worktree/bimodal-investigation` (branch: `bimodal-investigation`) — Phase 3 のプロファイリングコード追加
- **修正**: `.worktree/fix-bimodal-fprintf` (branch: `fix/bimodal-fprintf`) — fprintf 除去の本修正
- 修正は `feature/rdma-backend` にマージ済み
