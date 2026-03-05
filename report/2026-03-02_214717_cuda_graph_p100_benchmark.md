# P100 CUDA Graph 有効化ベンチマーク

- **実施日時**: 2026年3月2日 21:47
- **ワークツリー**: `.worktree/pp-optimization` (branch: `feature/pp-optimization`)
- **参照レポート**: [非RDMA PP最適化調査](2026-03-02_200704_non_rdma_pp_optimization_survey.md)

## 前提・目的

P100 (Compute Capability 6.0) では CUDA Graph がアーキテクチャチェック (`CC < AMPERE`) で無効化されている。しかし CUDA API レベルでは CC 3.5+ で CUDA Graph をサポートしているため、チェックを緩和して P100 で有効化し、Dense モデルの推論性能への影響を測定する。

- **背景**: CUDA Graph はカーネル起動オーバーヘッドを削減し、多数の小さなカーネルで構成されるグラフの実行を高速化できる可能性がある
- **仮説**: P100 でも CUDA Graph の replay が正常に動作すれば、特に PP (多バッチ処理) で改善が見られる
- **リスク**: `cudaGraphExecUpdate()` が P100 で毎回 recreate になる場合、オーバーヘッドが増加する可能性

## 変更内容

### 変更1: CC チェック緩和 (`ggml-cuda.cu:3906`)

```diff
- if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
+ if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_PASCAL) {
```

### 変更2: ビルドオプション追加 (`scripts/rdma-build.sh:14`)

```diff
- CMAKE_OPTS="... -DGGML_CUDA=ON ..."
+ CMAKE_OPTS="... -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON ..."
```

### 変更3: ランタイム無効化環境変数追加

`GGML_CUDA_DISABLE_GRAPHS=1` 環境変数で実行時に CUDA Graph を無効化可能にした（A/B テスト用）。

## テスト構成

| 項目 | 値 |
|------|-----|
| モデル | gpt-oss-20b Q4_K_M (Dense, ~11GB) |
| GPU構成 | Node 1 CUDA5,6 + Node 2 RDMA0-3 (6GPU) |
| ツール | llama-cli (`-f /tmp/long_prompt.txt -n 32 --single-turn --simple-io`) |
| 固定パラメータ | `-sm layer -ngl 999 -fa 1` |
| プロンプト | 8,433 bytes (~2,000 tokens) |

### A/B 条件

| 条件 | 内容 |
|------|------|
| A (ベースライン) | `GGML_CUDA_DISABLE_GRAPHS=1` (CUDA Graph 無効) |
| B (CUDA Graph) | 環境変数なし (CUDA Graph 有効) |

両条件とも同一ビルド (`GGML_CUDA_GRAPHS=ON` + CC 緩和) を使用。サーバー側も各条件で再起動して環境変数を反映。

## 再現方法

1. ワークツリー作成・ビルド
   ```bash
   git worktree add -b feature/pp-optimization .worktree/pp-optimization feature/rdma-backend
   # ggml-cuda.cu と rdma-build.sh を変更 (上記 diff 参照)
   bash .worktree/pp-optimization/scripts/rdma-build.sh local
   bash .worktree/pp-optimization/scripts/rdma-deploy.sh
   ```

2. ベンチマーク実行 (ABAB パターン、各条件3回)
   ```bash
   # 条件A: サーバー起動時に GGML_CUDA_DISABLE_GRAPHS=1 を設定
   # 条件B: 環境変数なしでサーバー起動
   # 各条件切り替え時にサーバー再起動
   gpu-lock.sh run bash /tmp/bench_cuda_graph.sh
   ```

3. クライアント実行コマンド
   ```bash
   # 条件A
   GGML_CUDA_DISABLE_GRAPHS=1 GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=5,6 \
     llama-cli -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M -sm layer -ngl 999 -fa 1 \
     -f /tmp/long_prompt.txt -n 32 --single-turn --simple-io

   # 条件B
   GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=5,6 \
     llama-cli -hf unsloth/gpt-oss-20b-GGUF:Q4_K_M -sm layer -ngl 999 -fa 1 \
     -f /tmp/long_prompt.txt -n 32 --single-turn --simple-io
   ```

## 結果

### CUDA Graph 動作確認

CUDA Graph は P100 で正常に動作した（クラッシュなし）:
- クライアント側: `ggml_backend_cuda_graph_compute: CUDA graph warmup complete` (2デバイス分)
- サーバー側: `ggml_backend_cuda_graph_compute: CUDA graph warmup complete` (4デバイス分)

### 性能測定結果

| 条件 | Iter | PP (t/s) | TG (t/s) |
|------|------|----------|----------|
| A (無効) | 1 | 541.4 | 57.9 |
| B (有効) | 1 | 537.4 | 56.7 |
| A (無効) | 2 | 540.9 | 58.1 |
| B (有効) | 2 | 541.1 | 52.9 |
| A (無効) | 3 | 545.9 | 54.3 |
| B (有効) | 3 | 538.2 | 56.5 |

### 統計分析 (対応あり t 検定)

| 指標 | A 平均 | B 平均 | 変化率 | t 統計量 | p 値 |
|------|--------|--------|--------|----------|------|
| PP (t/s) | 542.73 | 538.90 | **-0.71%** | 1.680 | 0.235 |
| TG (t/s) | 56.77 | 55.37 | **-2.47%** | 0.655 | 0.580 |

**いずれも統計的に有意ではない** (p > 0.05)。

## 考察

### CUDA Graph が P100 で改善をもたらさない理由

1. **ハードウェアサポートの制限**: CUDA Graph の replay 最適化は Ampere 以降のアーキテクチャ（HW-accelerated graph launch）で効果が大きい。P100 (Pascal) では SW ベースの replay となり、カーネル起動オーバーヘッド削減効果が限定的

2. **Graph キャプチャ・更新コスト**: P100 では `cudaGraphExecUpdate()` の効率が低く、グラフのキャプチャや更新にかかるコストが replay による節約を相殺している可能性

3. **モデルサイズの影響**: gpt-oss-20b は 6GPU に対して比較的小さく (GPU あたり ~1.8GB)、各 graph_compute の実行時間が短い。ただし、これは逆に CUDA Graph が効果を発揮しやすい条件（カーネル起動比率が高い）であるにもかかわらず改善が見られなかったことを意味する

4. **上流の設計判断は正しい**: `GGML_CUDA_CC_AMPERE` チェックは妥当な設計判断であり、P100 での CUDA Graph 有効化は推奨されない

## 結論

**P100 での CUDA Graph 有効化は PP 最適化に寄与しない。** CUDA Graph は P100 で正常に動作するものの、性能改善は見られず、わずかな性能低下傾向 (PP -0.7%, TG -2.5%、いずれも有意差なし) が観測された。上流コードの `CC < AMPERE` チェックは適切であり、この最適化パスは棄却する。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `d7367565b (feature/pp-optimization)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1214750) |
