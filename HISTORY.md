# 完了済みステップの記録

## Step 1: RDMA基本動作 ✅

### 達成済み
- CPU経由RDMA (GPUDirectなし) + P2P無効での分散推論が動作
- プロトコル最適化で 0.3 → 148.5 t/s (小規模モデル生成速度)
- Graph diff更新、Adaptive response、Persistent staging buffer を実装
- gpt-oss-20b: 2GPU (ローカル+リモート) で動作確認済み (16.0 t/s生成)
- gpt-oss-120b: 7ローカルGPUで動作確認済み (46.8 t/s生成)

### 当初の未解決課題 (後続Stepで解決済み)
- **gpt-oss-120b の11GPUクラスタテスト失敗** (CUDA illegal memory access) → Step 2 で修正
- RDMA経由の性能損失が大きい (20bモデルで77%低下) → Step 3 で改善
- Prompt処理速度がローカルの24分の1 (13.7 vs 328.5 t/s) → Step 3/4 で改善
- GPUDirect RDMA未有効化 (GGML_RDMA_NO_GDR=1で無効化中) → Step 4 で有効化
- 全操作が同期的 (async未実装) → Step 3 でバッチフラッシュ等を実装

## Step 2: マルチノードクラスタの安定化 ✅

### 達成内容
- `supports_buft` バグ修正 (`strstr` → `strcmp`) でクロスデバイス問題を解決
- チャンク送受信実装 (141MB MoEエキスパート重み対応)
- クロスデバイスコピー安全ネット (D2H+H2D)
- gpt-oss-20b/120b が11GPUクラスタ (7 CUDA + 4 RDMA) で安定動作
- 120b: Prompt 3-11 t/s, Generation 0.7-1.1 t/s (複数プロンプトで再現確認済み)

## Step 3: RDMA性能最適化 ✅

### 達成内容
- バッチフラッシュ: 複数テンソルの一括ステージング転送で Prompt 速度改善
- FLUSH+COMPUTE 統合: 2回のラウンドトリップを1回に削減 (FLUSH_ALL_STAGING + GRAPH_RECOMPUTE → 1コマンド)
- クライアント側 per-call プロファイリング追加 (GGML_RDMA_PROFILE=1)
- サーバー側プロファイリング追加 (graph_recompute, fix_xdev, compute 時間)

### 達成数値
| 構成 | モデル | pp128 | tg32 | ローカル比 |
|------|--------|-------|------|-----------|
| Local 1GPU | qwen2.5-0.5b | 3,390 | 213.36 | 100% |
| RDMA 1+1 | qwen2.5-0.5b | 3,269 | 173.87 | 81.5% |
| Local 1GPU | gpt-oss-20b | 407 | 64.27 | 100% |
| RDMA 1+1 | gpt-oss-20b | 61 | 58.47 | 91.0% |

### 成功基準の達成
- gpt-oss-20b tg32: 58.47 t/s ✅ (目標: 30+)
- gpt-oss-20b pp128: 61.09 t/s ✅ (目標: 40+)

### 発見事項
- 複数RDMA デバイスはスケジューラの制約で逐次実行 (4デバイス = 4× graph_compute/token)
- サーバーGPU計算時間がセッション間で2-3倍変動 (原因不明)
- IB Send/Recv の combined send 最適化は recv バッファオーバーヘッドで逆効果

## Step 4: GPUDirect RDMA有効化 ✅

### 達成内容
- nvidia-peermem 経由のGPUメモリ直接登録が正常動作
- `GGML_RDMA_NO_GDR=1` を外すだけで有効化 (コード変更不要)
- CPU経由のステージング (cudaMemcpy) を排除

### 達成数値

| モード | pp128 (t/s) | ローカル比 | tg32 (t/s) | ローカル比 |
|--------|:-----------:|:----------:|:----------:|:----------:|
| ローカル 1GPU | 407.68 | 100% | 64.27 | 100% |
| **GPUDirect RDMA 1+1** | **403.69** | **99.0%** | **59.52** | **92.6%** |
| CPU staging 1+1 | 61.06 | 15.0% | 59.30 | 92.3% |

### 成功基準の達成
- GPUDirect RDMA でテンソル転送が動作 ✅
- CPU経由比で測定可能な性能向上 (目標: 30%削減) → pp128で6.6倍高速 ✅

### 発見事項
- 大規模モデル (20b) ではpp128が6.6倍高速 (ウェイト転送量が多い)
- 小規模モデル (0.5b) ではグラフ送信オーバーヘッドが支配的で効果薄
- get_tensor はGPU VRAM からのRDMA ReadでCPU stagingより遅い (PCIe経由)

### 当初の計画
- 前提条件: Step 2 (クラスタ安定化) が完了していること / nvidia-peermem モジュールが両ノードでロード済み
- 成功基準: GPUDirect RDMA でテンソル転送が動作すること / CPU経由と比較して測定可能な性能向上 (目標: レイテンシ30%以上削減)
- 対象ファイル: `ggml/src/ggml-rdma/rdma-gdr.cpp` (GPUDirectメモリ管理) / `ggml/src/ggml-rdma/ggml-rdma.cpp` (GPUDirect パスの統合) / `ggml/src/ggml-rdma/rdma-memory.cpp` (メモリ登録)
