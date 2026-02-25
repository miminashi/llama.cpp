# Phase 11: サーバーサイド Multi-Device 並列ディスパッチ — GDR レースコンディション調査

- **実施日時**: 2026年2月25日 04:39
- **ワークツリー**: `.worktree/expert-parallelism`
- **ブランチ**: `feature/expert-parallelism`
- **コミット**: `86d4af52e`

## 前提・目的

EP=11 (7 CUDA + 4 RDMA) 構成で GLM-4.7 IQ2_M の tg 性能を改善するため、サーバー側で graph_compute を並列ディスパッチする仕組みを実装する。

Phase 10 の分析から、tg ボトルネックの 38% が graph_compute のラウンドトリップ蓄積（1トークンあたり 366 回の逐次呼び出し）にあることが判明していた。4 つのリモート RDMA デバイスへの compute は同一レイヤー内で独立であるため、並列実行により理論上 +29% の tg 改善が期待された。

## 設計

### 当初の計画: per-device ワーカースレッド

Phase 10 の分析に基づき、4 RDMA デバイスへの graph_compute を並列実行するため per-device ワーカースレッドの導入を計画した。理論上、62 MoE レイヤー × 4 デバイスの逐次 compute (372ms) を並列化で ~93ms に短縮し、**tg +29%** を見込んでいた。

しかし実装過程で GDR レースコンディション（後述）が発見され、inter-device 並列化は安全に実現できないことが判明。ワーカースレッドは断念し、main-thread async モードに切り替えた。`device_worker` 構造体・`worker_loop`・dispatch/wait API はコードに残存するが、`init_workers()` は no-op（`workers_` を populate しない）であり、実際にはワーカースレッドは起動されない。

### 実装: prepare_only + main-thread async

compute 関数群（`graph_compute`, `graph_recompute`, `graph_compute_update`, `flush_and_*`）に `prepare_only` パラメータを追加し、prepare フェーズ（deferred copies, cross-device refs 修正）と compute フェーズ（CUDA カーネル起動）を分離。

```cpp
// Prepare のみ
server->flush_and_indexed_update(data, /*prepare_only=*/true);
// 非同期カーネル起動 (main thread, non-blocking)
server->compute_stored_graph_async(device);
// 後で同期
server->sync_device(device);
```

### 重要な発見: `ggml_backend_rdma_synchronize` は NO-OP

`compute_stored_graph_async` は `ggml_backend_graph_compute_async(backends_[device], sg.graph)` を呼ぶ。これは内部で `ggml_backend_graph_compute` を呼び、その末尾で `ggml_backend_synchronize(backend)` を実行する。

しかし、ここで渡される `backend` は **RDMA バックエンド** であり、その `synchronize` 実装は:

```cpp
static void ggml_backend_rdma_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);  // NO-OP
}
```

つまり `ggml_backend_graph_compute_async` は CUDA カーネルを起動するが **CUDA ストリームの完了を待たない**。実際の同期は `sync_device` が `ggml_backend_synchronize(backends_[device])` を呼ぶことで、裏側の CUDA バックエンドの `cudaStreamSynchronize` を通じて実行される。

この NO-OP 設計が prepare_only + async パターンを可能にしている。

### パイプライン ラムダ

```cpp
auto pipeline_prepare_and_dispatch = [&](cmd_type, device, data) {
    sync_async_computes();           // 全 async デバイスを同期
    // ... prepare_only で prepare ...
    server->compute_stored_graph_async(device);  // カーネル起動（非ブロック）
    async_devices.insert(device);
};
```

## GDR レースコンディション調査

### 発見された問題

deferred sync パターン（次の pipeline 呼び出し時に前回の async compute を sync）が GDR 有効時にクラッシュする:

```
CUDA error: an illegal memory access was encountered
  current device: 3, in function launch_fattn
```

### 系統的テスト結果

| テスト | sync 位置 | GDR | 結果 |
|--------|----------|-----|------|
| PARALLEL_DISPATCH=0 (baseline) | compute 内 (full sync) | 有効 | **PASS** |
| async + 即時 sync_device | launch 直後 | 有効 | **PASS** |
| async + 即時 sync_async_computes | insert 直後 | 有効 | **PASS** |
| async + switch 末尾で sync | recv() の前 | 有効 | **PASS** |
| async + recv() 後に sync | recv() の後、switch の前 | 有効 | **CRASH** |
| async + deferred sync (pipeline先頭) | 次の pipeline call 時 | 有効 | **CRASH** |
| async + deferred + cudaDeviceSynchronize ALL | recv() 後 | 有効 | **CRASH** |
| async + deferred sync | pipeline先頭 | **無効** (budget=0) | **PASS** |
| async + device-specific sync | target device のみ | 有効 | **CRASH** |

### 根本原因

#### IB 順序保証と RDMA Write の到達タイミング

InfiniBand の順序保証により、RDMA Write は後続の Send completion がポーリングされる前に完了する。クライアント側の処理フロー:

1. `flush_deferred_writes()` — 蓄積された RDMA Write を一括発行（GPU メモリに直接書き込み）
2. `conn->send()` — 次のコマンドを Send で送信

サーバーの `conn->recv()` が返る（= Send completion をポーリングする）時点で、先行する RDMA Write は **IB ハードウェアレベルで既に完了** している。これはサーバーのコードとは無関係に RNIC が実行する。

#### レースコンディションのメカニズム

非同期 CUDA カーネルが GPU メモリを読み込んでいる間に、`conn->recv()` でブロック中のサーバーに対して、クライアントの RNIC が GDR 経由で同じ GPU メモリに RDMA Write を実行する。結果としてメモリ破損が発生し `illegal memory access` エラーとなる。

#### テスト7 (`cudaDeviceSynchronize ALL` 後 recv) が失敗する理由

直感的には「recv() 後に全 GPU を同期すれば安全」と思えるが、実際にはダメージは recv() の **ブロック中** に発生する。recv() が返った時点では RDMA Write による GPU メモリ上書きは既に完了しており、事後の CUDA 同期では手遅れである。

#### GDR 無効化で確認

`GGML_RDMA_GDR_BUDGET_GB=0` を設定すると、RDMA Write はホストのステージングバッファに書き込まれ、GPU メモリに直接書き込まれないため、deferred sync パターンでもクラッシュしない。これにより問題が GDR 固有であることを実証した。

#### device-specific sync が失敗する理由

クライアントの `flush_deferred_writes` はコネクション単位で全 pending writes を一括フラッシュする（デバイス単位ではない）。device N+1 のコマンド送信時に `flush_deferred_writes(conn)` が呼ばれると、device N を含む **全デバイス** のバッファへの RDMA Write が発行される。そのため、device N+1 のみを sync しても device N の GPU メモリへの書き込みは防げない。

### 解決策: sync-before-recv

```cpp
while (true) {
    if (server->parallel_dispatch_enabled() && !async_devices.empty()) {
        sync_async_computes();  // 全 async デバイスを同期してから recv()
    }
    conn->recv(...);  // GPU はアイドル状態 → RDMA Write は安全
    switch (cmd) { ... }
}
```

この方式では各コマンド受信前に全 async デバイスを同期するため、事実上は逐次実行と同等となり、inter-device 並列化の性能効果は得られない。

### 期待 +29% vs 実測 0%: ギャップの分析

当初の計画では graph_compute の parallel dispatch で **tg +29%** を見込んでいた（372ms → 93ms）。しかし sync-before-recv 制約により、各コマンドは依然として逐次処理される。dispatch ON/OFF で同一速度 (0.8 t/s) という結果はこの制約の直接的な帰結である。

本フェーズの成果は性能改善ではなく、GDR レースコンディションの発見と体系的な分析、および将来の真の parallel dispatch に必要な前提条件の明確化にある。prepare_only パラメータ、compute_stored_graph_async、sync_device などのインフラは、将来のプロトコル変更時にそのまま活用可能。

## テスト結果

### 機能テスト (GLM-4.7 IQ2_M, EP=11, 7C+4R)

| 条件 | Prompt (t/s) | Generation (t/s) | 結果 |
|------|:------------:|:----------------:|:----:|
| dispatch ON (デフォルト) | 0.3 | 0.8 | PASS |
| dispatch OFF (fallback) | 0.3 | 0.8 | PASS |
| 非EP RDMA | 8.7 | 8.8 | PASS (回帰なし) |

### 正当性

- dispatch ON/OFF で同一の推論速度
- 非EP モードで性能回帰なし
- 50+ トークン生成でクラッシュ・ハングなし

## 再現方法

### ビルド・デプロイ

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/scripts/rdma-build.sh local
bash /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/scripts/rdma-deploy.sh
bash /home/ubuntu/projects/llama.cpp/.worktree/expert-parallelism/scripts/rdma-server.sh start
```

### EP=11 テスト

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  llama-cli -m /path/to/GLM-4.7-IQ2_M.gguf \
  -ngl 999 --expert-parallel 11 -fa 1 -c 256 \
  -p "Hello, I am a large language model" -n 20 \
  --log-file /tmp/llama-cli.log --no-warmup --single-turn --simple-io
```

### GDR 無効化テスト (レースコンディション検証)

```bash
# サーバー側
ssh 192.168.100.2 "LD_LIBRARY_PATH=... GGML_RDMA_GDR_BUDGET_GB=0 nohup rdma-server ... > /tmp/rdma-server.log 2>&1 &"
# クライアント側 (同じコマンド)
```

## 成果物

- **prepare_only パラメータ**: 6 関数に追加（graph_compute, graph_recompute, graph_compute_update, flush_and_recompute, flush_and_compute_update, flush_and_indexed_update）
- **compute_stored_graph_async / sync_device**: 非同期カーネル起動と同期 API
- **sync-before-recv パターン**: GDR レースコンディション回避
- **GGML_RDMA_PARALLEL_DISPATCH**: 環境変数による有効/無効切替（デフォルト有効）

## 将来の最適化に向けた知見

inter-device 並列化を実現するには、**GDR + RDMA Write が recv() ブロック中に GPU メモリを上書きする**問題を解決する必要がある。具体的には以下のいずれかのアプローチ:

1. **クライアント側デバイス別 write fencing**: 現在 `flush_deferred_writes(conn)` はコネクション上の全 pending writes を一括フラッシュする。これをデバイス単位 (`flush_deferred_writes(conn, target_device)`) に分離し、ターゲットデバイスの writes のみフラッシュする。これにより device-specific sync が安全になる
2. **非ブロッキング recv + バッチ処理**: `conn->recv()` のブロックが問題の根本であるため、IB completion キューの non-blocking poll (`ibv_poll_cq`) で即座に利用可能なコマンドをバッチ処理し、ブロック前にのみ sync する。recv() でブロックしなければ RDMA Write の到達タイミングは制御可能
3. **双方向同期プロトコル**: サーバーが async compute 完了をクライアントに通知し、クライアントは通知を受けてから RDMA Write を送信。これにより RDMA Write と CUDA カーネルの時間的重複を排除

いずれもプロトコル変更またはトランスポート層の改修が必要。アプローチ 2 が最も有望（クライアント側変更なし、サーバーの recv ループのみ変更）だが、コマンドのフレーミングとエラーハンドリングの再設計が伴う。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `86d4af52e (feature/expert-parallelism)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 35°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| rdma-server | — | running |
