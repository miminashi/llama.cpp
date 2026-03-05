# 課題管理

## 残タスク

### 1. 16GPU への拡張 (GPU 追加後)

- 16GPU (8+8) への拡張と全 GPU での RDMA 接続確立テスト
- GLM-4.7 Q4 (IQ2_M より大きい量子化) が VRAM に収まるか検証
- `-sm layer` によるレイヤー分割で VRAM 分配計画を策定

### 2. GLM-4.7 Q4 量子化モデルの準備

- unsloth GLM-4.7 Q4 のダウンロードとサイズ確認
- IQ2_M (~40GB) では 11GPU で動作したが、Q4 はより大きいため 16GPU が必要な可能性

## 性能課題

### Send selective signaling の回復 (pp128: 30.6 → 24.3)

- **現状**: バッファ再利用レースコンディション修正 (Send always-signal, `9a5a4a911`) により、graph_compute の Send パスで selective signaling が実質無効化
- **影響**: pp128 が ~30.6 → ~24.3 t/s に低下 (-20.6%)
- **根本原因**: `send_buffer_` (16MB) が各イテレーションで再利用される。selective signaling では前回の Send 完了前に次の Send がバッファを上書きする可能性
- **改善案**: Send double-buffering (2つの send_buffer_ を交互に使用し、前回完了を次の次の Send で確認)
- **ワークツリー**: `.worktree/fix-rdma-write-signal`, branch `fix/rdma-write-always-signal`

### Expert parallelism の GDR 制約

- **現状**: GDR 有効時、parallel dispatch で `conn->recv()` 中に RNIC が GDR Write で直接サーバー GPU メモリに書き込み、非同期 CUDA カーネルと競合してクラッシュ
- **対策**: sync-before-recv (全非同期デバイスを `conn->recv()` 前に同期) で回避。ただしデバイス間並列性が失われる
- **改善案**: クライアント側 per-device write fencing の実装
- **ワークツリー**: `.worktree/expert-parallelism`, branch `feature/expert-parallelism`
- **トグル**: `GGML_RDMA_PARALLEL_DISPATCH=0` で無効化

### Generation 速度での RPC 比劣位

- RDMA は単一接続で全リモートデバイスを共有するため、`graph_compute` が逐次実行される
- RPC はデバイスごとに独立ソケットを持ち、コマンド送信を並列化可能
- **改善案**: RDMA 接続のデバイス分離またはパイプライン化 (per-device connections は実装済み `GGML_RDMA_PER_DEVICE_CONN=1`、ConnectX-4 では MTT キャッシュ制限によりデフォルト無効)
- **注記**: selective signaling + flash attention 適用後のベースラインでは RPC 比劣位は解消済み

### サーバー GPU 計算時間のセッション間変動

- RDMA (GDR 無効) の Generation 速度が 5.5-6.8 t/s と大きくばらつく
- GPU のサーマルスロットリングまたは CUDA コンテキスト初期化の影響と推定
- GDR 有効時は比較的安定 (6.6-6.9 t/s)

### GDR バジェットのデフォルト値

- デフォルト 12GB は ConnectX-4 の MTT キャッシュ推定値 (~10-16GB) に基づく経験的な値
- より大容量の RNIC (ConnectX-6 等) では `GGML_RDMA_GDR_BUDGET_GB` を増やすことで全デバイス GDR が可能
- 現状では `GGML_RDMA_NO_GDR=1` も引き続き使用可能 (per-buffer フラグにフォールバック)

## 既知の制約

### ggml_backend_sched の逐次処理制約

- `ggml_backend_sched_compute_splits()` がデバイスを逐次処理 (graph_compute → get_tensor をペアで呼ぶ)
- RDMA では get_tensor の応答が compute 完了後にしか返らないため、全デバイスが完全に逐次化される
- RPC でも真の並列性は活用されていないが、TCP バッファリングによる暗黙的パイプラインが寄与
- **根本制約**: スケジューラの設計変更なしには解消不可。詳細: [ggml_backend_sched 逐次処理ボトルネック分析](report/2026-02-19_090721_ggml_backend_sched_bottleneck_analysis.md)

### PCIe MaxPayload (P100: 256B hard limit)

- P100 の DevCap MaxPayload は 256B がハードウェア上限 (ConnectX-4 DevCap は 512B)
- MaxReadReq は 512B → 4096B に変更可能だが、MaxPayload 256B が DMA 転送のボトルネック
- PCIe Read/Write 方向の帯域非対称性: RNIC→GPU Write 9.8 GB/s vs GPU→RNIC Read 3.4 GB/s
- 詳細: [Generation 最適化 深堀り調査](report/2026-02-18_070912_generation_optimization_deep_investigation.md)

### gpt-oss-20b の bimodal tg 分布

- gpt-oss-20b 1C+2R 構成で tg 速度にバイモーダル分布が発生
- fprintf 二峰性 (サーバー側 stderr 出力のブロッキング) は修正済み (`fix-bimodal-fprintf` ワークツリーでマージ)
- fprintf 修正後も gpt-oss-20b で新たな二峰性が観測されたが、deferred copy とは無関係
- 原因は GPU 計算時間のセッション間変動と推定
- 詳細: [二峰性 tg 分布調査](report/2026-02-13_153100_bimodal_tg_investigation.md)、[deferred copy 再評価](report/2026-02-14_082220_deferred_copy_clean_ab_benchmark.md)

GDR は `nvidia_peermem` モジュールがロードされていれば自動有効化。詳細: `/gdr` スキル参照。
