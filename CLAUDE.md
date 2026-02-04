## 必読ドキュメント

- [AGENTS.md](AGENTS.md) — 作業開始前に必ず確認すること
- [REPORT.md](REPORT.md) — レポート作成ルール
- [GPU.md](GPU.md) — GPUサーバ情報

## ビルドルール

- llama.cppをビルドする際は、必ず事前に `build` ディレクトリを削除してからビルドすること (`rm -rf build && cmake -B build ...`)

## 実行ルール

- `llama-cli` を実行する際は、必ず `--log-file /tmp/llama-cli.log` オプションを付けること (ユーザが別ターミナルで `tail -f /tmp/llama-cli.log` によりリアルタイムにログを確認できるようにするため)

## プロジェクト目標

最終目標: GPUDirect RDMAを有効化し、2ノード16台のP100でGLM4.7 Q4を動作させること。

### 段階的アプローチ

上記の最終目標をいきなり達成するのは難しいため、以下のようなステップを設定して段階的に進める。

1. **Step 1** ✅: GPUDirectでないRDMA + P2P無効 + gpt-ossなどの中程度のパラメータ数のモデル
2. **Step 2**: マルチノードクラスタの安定化 (gpt-oss-120b を11GPUクラスタで安定動作)
3. **Step 3**: RDMA性能最適化 (性能損失50%以下、パイプライン化、非同期操作)
4. **Step 4**: GPUDirect RDMA有効化 (CPU経由ステージング排除、GPU直接RDMA転送)
5. **Step 5 (最終Step)**: GPUDirect RDMA + 2ノード16台P100 + GLM4.7 Q4

### ステップ間の依存関係

```
Step 1 (完了) → Step 2 (クラスタ安定化) ★最優先
                    ↓
               Step 3 (性能最適化)
                    ↓
               Step 4 (GPUDirect RDMA)
                    ↓
               Step 5 (GLM4.7 on 16 P100s)
```

- **安定化優先**: Step 2を最優先で進める
- Step 3は Step 2完了後に着手 (安定した環境で性能測定するため)
- Step 4は Step 2の完了が前提 (不安定な状態でGPUDirectを追加すると問題切り分けが困難)
- Step 5はハードウェア要件 (16台のP100) に依存するが、11台での事前検証は先行可能

### モデル分割方式

- **レイヤー分割 (`-sm layer`) のみ** — P100にはNVLinkがなく、row splitでは性能が出ないことがシングルノード実験で確認済み
- RDMAバックエンドは既にレイヤー分割のみで動作している (各バックエンドインスタンス = 1リモートGPU)
- row split の実装は不要

---

## Step 1 の達成状況

### 達成済み
- CPU経由RDMA (GPUDirectなし) + P2P無効での分散推論が動作
- プロトコル最適化で 0.3 → 148.5 t/s (小規模モデル生成速度)
- Graph diff更新、Adaptive response、Persistent staging buffer を実装
- gpt-oss-20b: 2GPU (ローカル+リモート) で動作確認済み (16.0 t/s生成)
- gpt-oss-120b: 7ローカルGPUで動作確認済み (46.8 t/s生成)

### 未解決の課題
- **gpt-oss-120b の11GPUクラスタテスト失敗** (CUDA illegal memory access)
- RDMA経由の性能損失が大きい (20bモデルで77%低下)
- Prompt処理速度がローカルの24分の1 (13.7 vs 328.5 t/s)
- GPUDirect RDMA未有効化 (GGML_RDMA_NO_GDR=1で無効化中)
- 全操作が同期的 (async未実装)

---

## Step 2: マルチノードクラスタの安定化

### 目標
gpt-oss-120b を11GPUクラスタ (7ローカル + 4リモート) で安定動作させる。

### タスク

1. **120bクラスタ障害の原因調査と修正**
   - CUDA illegal memory access エラーのデバッグ
   - クライアント・サーバー間のバージョン整合性確認 (クライアントはfeature/rdma-backend、サーバーがmasterの可能性)
   - 大規模MoEモデル (64 experts, Top-8) のグラフシリアライズ検証
   - リモートサーバー側でのバッファ管理・メモリアクセス境界チェック

2. **プロトコルの堅牢性向上**
   - エラーハンドリングの改善 (障害時の詳細なエラー情報)
   - 大規模グラフのシリアライズ/デシリアライズの検証

### 成功基準
- gpt-oss-120b が11GPUクラスタで安定的に推論完了できること
- 複数プロンプトで再現性のある結果が得られること

### 対象ファイル
- `ggml/src/ggml-rdma/ggml-rdma.cpp` — グラフシリアライズ、バッファ管理
- `ggml/src/ggml-rdma/rdma-transport.cpp` — エラーハンドリング

---

## Step 3: RDMA性能最適化

### 目標
中規模モデル (gpt-oss-20b) でのRDMA性能損失を77% → 50%以下に削減する。

### タスク

1. **Prompt処理速度の改善**
   - 初回グラフ送信の差分圧縮 (現在90KB全量送信)
   - set_tensor → graph_compute → get_tensor のパイプライン化
   - 複数テンソルのバッチ転送

2. **RDMA one-sided操作の活用拡大**
   - 現在Send/Recvベースの制御メッセージをRDMA Write/Readに移行
   - ラウンドトリップ削減

3. **非同期操作の導入**
   - set_tensor_async / get_tensor_async の実装
   - 計算と通信のオーバーラップ

### 成功基準
- gpt-oss-20b 2GPU RDMA: 生成速度 30 t/s以上 (現在16.0 t/s)
- gpt-oss-20b 2GPU RDMA: プロンプト速度 40 t/s以上 (現在20.8 t/s)
- gpt-oss-120b 11GPUクラスタ: ローカル7GPUの70%以上の性能

### 対象ファイル
- `ggml/src/ggml-rdma/ggml-rdma.cpp` — パイプライン化、非同期操作
- `ggml/src/ggml-rdma/rdma-transport.cpp` — one-sided操作の拡充

---

## Step 4: GPUDirect RDMA有効化

### 目標
GPUDirect RDMAを有効化し、CPU経由のステージングを排除して性能向上を実現する。

### タスク

1. **GPUDirect RDMA の有効化テスト**
   - `GGML_RDMA_NO_GDR=1` を外して動作確認
   - `nvidia-peermem` モジュールの動作検証
   - `rdma-gdr.cpp` のGPUメモリ登録パスのテスト

2. **GPUメモリの直接RDMA登録**
   - CUDAバッファをibv_reg_mrで直接登録
   - GPU→RDMA NIC→リモートGPUの直接データパス確立
   - ステージングバッファ経由のフォールバック維持

3. **性能測定と最適化**
   - GPUDirect有効/無効での性能比較
   - ボトルネック分析 (PCIeバス帯域、NIC帯域)
   - P100のPCIeトポロジに応じたGPU割り当て最適化

### 前提条件
- Step 2 (クラスタ安定化) が完了していること
- nvidia-peermem モジュールが両ノードでロード済み

### 成功基準
- GPUDirect RDMA でテンソル転送が動作すること
- CPU経由と比較して測定可能な性能向上 (目標: レイテンシ30%以上削減)

### 対象ファイル
- `ggml/src/ggml-rdma/rdma-gdr.cpp` — GPUDirectメモリ管理
- `ggml/src/ggml-rdma/ggml-rdma.cpp` — GPUDirect パスの統合
- `ggml/src/ggml-rdma/rdma-memory.cpp` — メモリ登録

---

## Step 5 (最終Step): GLM4.7 Q4 on 16 P100s

### 目標
GPUDirect RDMA + 2ノード16台P100で GLM4.7 Q4 を動作させる。
モデルはunslothの量子化モデル (Hugging Face) を使用予定。

### タスク

1. **モデル準備と事前検証**
   - unsloth GLM4.7 Q4のダウンロードとサイズ確認
   - 11GPUでの動作確認 (GPU追加前に先行テスト可能)
   - `-sm layer` によるレイヤー分割でVRAM分配計画を策定 (P100 16GB × n台)
   - ※ row splitはP100 (NVLinkなし) では性能が出ないため不使用

2. **ハードウェアスケーリング** (GPU追加後)
   - 16GPU (8+8等) への拡張と接続確認
   - 全GPUでのRDMA接続確立テスト

3. **大規模分散推論の最適化**
   - 16GPUでのグラフスケジューリング最適化
   - ノード間通信パターンの最適化
   - メモリ使用量の最適化 (P100 16GBの制約下)

### 前提条件
- Step 4 (GPUDirect RDMA) が動作していること
- 16台のP100が利用可能であること (GPU追加後)

### 成功基準
- GLM4.7 Q4 が16台P100で推論完了できること
- 実用的な推論速度が得られること (具体値はStep 3/4の結果を踏まえて設定)

### 備考
- GPU追加前でも、11台でGLM4.7の事前検証は可能 (モデルがVRAMに収まる場合)
- 収まらない場合はGPU追加を待つ

---

## 検証方法

- 各ステップで `llama-bench` または `llama-cli` によるベンチマーク実行
- レポートは `report/` ディレクトリに `REPORT.md` のフォーマットに従って記録
- 性能数値は Prompt t/s と Generation t/s の両方を計測
