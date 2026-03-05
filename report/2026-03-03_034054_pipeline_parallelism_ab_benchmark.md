# Pipeline Parallelism Phase 1-4 A/B ベンチマーク

- **実施日時**: 2026年3月3日 03:30-04:40
- **ワークツリー**: `.worktree/pipeline-parallelism`

## 前提・目的

Pipeline Parallelism Phase 1-4 の実装が完了したため（`feature/pipeline-parallelism` ブランチ, commit `f97fa34f2`）、`feature/rdma-backend` ベースライン（commit `c1ca6d5fd`）と定量比較し、ubatch 間パイプライン化によるスループット改善を確認する。

- **背景**: Phase 1-4 では ubatch 間の RDMA 通信とサーバー GPU 計算をパイプライン化し、通信レイテンシを隠蔽する改善を実装
- **変更範囲**: `ggml-rdma.cpp`（381行）と `llama-context.cpp`（39行）の 2 ファイルのみ。`rdma-server.cpp` は差分なし
- **仮説**: PP (Prompt) で改善が見込まれる。TG (Generation) は n_tokens=1 のためパイプライン効果なしと予想
- **参照レポート**: [Pipeline Parallelism Phase 4 レポート](2026-03-03_031241_pipeline_parallelism_phase4.md)

## 交絡因子チェック

- [x] **単一変数の分離**: 比較条件はクライアントバイナリのみ。`rdma-server.cpp` に差分なし → サーバー共有
- [ ] **環境変数トグル**: コードブランチ比較のため環境変数切替は不可。ただしサーバー側コードは同一であり、差分は `ggml-rdma.cpp` と `llama-context.cpp` のみ
- [x] **ホットパスのログ出力**: `GGML_RDMA_PROFILE` 未設定。fprintf なし
- [x] **分布の単峰性**: 全計測値が単峰分布（PP: 109.4-113.7 連続、TG: 33.6-36.1 連続）

## テスト構成

| 項目 | 値 |
|------|-----|
| GPU 構成 | Node 1 CUDA4,5 + Node 2 RDMA0,1（4GPU） |
| モデル | `unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M` |
| パラメータ | `-sm layer -ngl 999 -fa 1 -ub 32` |
| プロンプト | `"hello " × 100`（~100 tokens） |
| 生成トークン | `-n 64` |

### 比較条件

| 条件 | ブランチ | コミット | バイナリパス |
|------|---------|---------|------------|
| A (Baseline) | `feature/rdma-backend` | `c1ca6d5fd` | `/home/ubuntu/projects/llama.cpp/build/bin/llama-cli` |
| B (Pipeline) | `feature/pipeline-parallelism` | `f97fa34f2` | `.worktree/pipeline-parallelism/build/bin/llama-cli` |

## 再現方法

### 1. サーバー起動（1回のみ）

```bash
bash scripts/rdma-server.sh restart
```

`rdma-server.cpp` に差分がないため、どちらのブランチのサーバーでも可。

### 2. ベンチマーク実行

```bash
# Condition A (Baseline)
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  /home/ubuntu/projects/llama.cpp/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "$(python3 -c "print('hello ' * 100)")" -n 64 \
  --simple-io --log-file /tmp/llama-cli.log 2>&1

# Condition B (Pipeline) - バイナリパスのみ差し替え
gpu-lock.sh run CUDA_VISIBLE_DEVICES=4,5 GGML_RDMA_SERVERS=192.168.100.2:50051 \
  /home/ubuntu/projects/llama.cpp/.worktree/pipeline-parallelism/build/bin/llama-cli \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_M \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -sm layer -ngl 999 -fa 1 -ub 32 \
  -p "$(python3 -c "print('hello ' * 100)")" -n 64 \
  --simple-io --log-file /tmp/llama-cli.log 2>&1
```

### 3. ABAB 交互実行

Warmup 各1回（破棄）後、A→B→A→B... で 15 ペア実行。各実行間に 2 秒のスリープ。GPU 温度 70°C 超過時は冷却待機。

## 結果

### 生データ

| Pair | A_pp | B_pp | d_pp | A_tg | B_tg | d_tg |
|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 1 | 109.9 | 112.8 | +2.9 | 34.3 | 35.5 | +1.2 |
| 2 | 111.8 | 112.6 | +0.8 | 34.0 | 35.7 | +1.7 |
| 3 | 112.1 | 112.0 | -0.1 | 34.1 | 35.6 | +1.5 |
| 4 | 109.9 | 113.5 | +3.6 | 33.6 | 36.1 | +2.5 |
| 5 | 111.7 | 113.7 | +2.0 | 34.1 | 35.3 | +1.2 |
| 6 | 110.2 | 113.5 | +3.3 | 33.9 | 36.1 | +2.2 |
| 7 | 109.6 | 113.0 | +3.4 | 34.1 | 36.0 | +1.9 |
| 8 | 110.3 | 112.4 | +2.1 | 34.4 | 35.7 | +1.3 |
| 9 | 112.0 | 113.3 | +1.3 | 34.2 | 35.2 | +1.0 |
| 10 | 109.4 | 113.4 | +4.0 | 34.0 | 36.0 | +2.0 |
| 11 | 109.6 | 112.6 | +3.0 | 34.2 | 35.3 | +1.1 |
| 12 | 110.4 | 112.3 | +1.9 | 34.0 | 35.3 | +1.3 |
| 13 | 110.0 | 113.1 | +3.1 | 34.2 | 35.3 | +1.1 |
| 14 | 110.2 | 113.4 | +3.2 | 34.1 | 36.0 | +1.9 |
| 15 | 111.6 | 113.3 | +1.7 | 34.0 | 36.0 | +2.0 |

### Prompt (PP) 統計

| 指標 | 値 |
|------|:---:|
| Baseline (A) | 110.580 +/- 0.967 t/s |
| Pipeline (B) | 112.993 +/- 0.515 t/s |
| 差分平均 | **+2.413 t/s (+2.18%)** |
| t(14) | 8.1184 |
| p 値 | 1.15 x 10⁻⁶ |
| Cohen's d | 2.0962 (large) |
| 95% CI | [+1.776, +3.051] t/s |
| 全ペア正の効果 | 14/15 (93%) |
| **判定** | **有効** |

### Generation (TG) 統計

| 指標 | 値 |
|------|:---:|
| Baseline (A) | 34.080 +/- 0.186 t/s |
| Pipeline (B) | 35.673 +/- 0.339 t/s |
| 差分平均 | **+1.593 t/s (+4.68%)** |
| t(14) | 13.2225 |
| p 値 | 2.67 x 10⁻⁹ |
| Cohen's d | 3.4140 (large) |
| 95% CI | [+1.335, +1.852] t/s |
| 全ペア正の効果 | 15/15 (100%) |
| **判定** | **有効** |

## 考察

### PP 改善 (+2.18%)

PP の改善は予想通り。ubatch 間のパイプライン化により、前の ubatch のサーバー計算と次の ubatch の RDMA 通信が重複実行される。ただし Qwen3.5 MoE (4GPU) は[過去の分析](2026-03-02_195335_pp_profiling_qwen35_optimization.md)でサーバー計算がボトルネック（PP 時間の 94.8%）であるため、通信隠蔽の効果は限定的。改善幅は比較的小さい（+2.18%）。

### TG 改善 (+4.68%) — 予想外

TG の改善は予想外だった。n_tokens=1 の場合、ubatch 間パイプラインは無効のはずである。しかし Pipeline 版には Phase 1-4 全体の変更（381行）が含まれており、以下の可能性が考えられる：

1. **graph_cache の最適化**: Phase 4 で導入された multi-slot graph_cache により、graph 再構築のオーバーヘッドが削減された可能性
2. **コード変更の副次効果**: Phase 1-4 の実装変更（send/recv パターンの改善、pipeline manager の導入）がTG にも影響した可能性
3. **llama-context.cpp の変更**: ubatch ループ構造の変更（39行）が TG パスにも影響

TG 改善の根本原因の特定には、Phase 1-4 を個別に分離してテストする必要がある。

### 分散の変化

Pipeline 版の PP 分散が顕著に低下している（SD: 0.967 → 0.515 t/s）。パイプライン化により実行パスがより決定的になった可能性がある。

## 結論

Pipeline Parallelism Phase 1-4 は Qwen3.5-35B-A3B (4GPU, MoE) で：

- **PP**: +2.18% 改善（統計的に有効、p = 1.15 × 10⁻⁶）
- **TG**: +4.68% 改善（統計的に有効、p = 2.67 × 10⁻⁹）

両指標とも大きな効果量（Cohen's d > 2.0）で、一貫した改善を示す。Qwen3.5 は compute-bound モデルのため PP 改善幅は控えめだが、communication-bound モデル（GLM-4.7 等）ではより大きな PP 改善が期待される。

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c1ca6d5fd (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 35°C | 35°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1255484) |

GGML_RDMA 環境変数: (none)
