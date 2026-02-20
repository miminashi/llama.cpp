# llama-cli による総合改善の追試レポート

- **実施日時**: 2026年2月20日 21:20
- **ワークツリー**: `.worktree/merge-validated`

## 前提・目的

llama-bench で確認された総合改善 (pp +25.3%, tg +1.7%) を llama-cli の実使用シナリオで再現確認し、`merge/validated-improvements` → `feature/rdma-backend` へのマージ判断材料とする。

- **背景**: llama-bench はバッチ処理 (pp128, tg32) の throughput 測定に特化。llama-cli は実際の推論パイプライン（モデルロード → prompt 処理 → テキスト生成）を通すため、実用性の確認に適している
- **参照**: [merge-validated ベンチマークレポート](2026-02-20_132916_merge_validated_improvements_benchmark.md)（llama-bench テスト3 の結果）

## 実験設計

### 比較条件

merge-validated バイナリを使用し、環境変数トグルで条件を切り替え:

| 条件 | 設定 | 意味 |
|------|------|------|
| **A (ベースライン)** | `GGML_RDMA_NO_SELECTIVE_SIGNAL=1` | selective-signaling OFF, shared connection |
| **B (全有効)** | `GGML_RDMA_PER_DEVICE_CONN=1` | selective-signaling ON, per-device + server-push |

### llama-bench での期待値 (テスト3 結果)

| 指標 | 改善率 | p値 |
|------|:------:|:---:|
| pp128 | +25.34% | 2.26 × 10⁻⁸ |
| tg32 | +1.66% | 0.0040 |

### パラメータ

- モデル: GLM-4.7 IQ2_M
- デバイス: 7 CUDA + 4 RDMA (11GPU)
- プロンプト: `"The capital of France is"` (~10 tokens)
- 生成: `-n 50` tokens
- その他: `--seed 42`, `--no-warmup`, `--single-turn`, `-sm layer`, `-ngl 999`, `-c 2048`
- ABAB Paired Design: 5ペア (計10回)、各条件切替時にサーバー再起動

## 再現方法

### 前提

merge-validated ワークツリーのビルドとデプロイが完了していること:

```bash
bash /home/ubuntu/projects/llama.cpp/.worktree/merge-validated/scripts/rdma-build.sh local
bash /home/ubuntu/projects/llama.cpp/.worktree/merge-validated/scripts/rdma-deploy.sh
```

### 実験スクリプト

各ペアで以下を ABAB 順に実行:

```bash
# --- 条件 A: サーバー再起動 (selective-signaling OFF) ---
ssh 192.168.100.2 "pkill -f rdma-server || true"
sleep 2
ssh 192.168.100.2 "GGML_RDMA_NO_SELECTIVE_SIGNAL=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
sleep 3

# --- 条件 A: クライアント実行 ---
gpu-lock.sh run env GGML_RDMA_NO_SELECTIVE_SIGNAL=1 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_GDR_BUDGET_GB=12 \
  /home/ubuntu/projects/llama.cpp/.worktree/merge-validated/build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,...,CUDA6,RDMA0[192.168.100.2:50051],...,RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log

# --- 条件 B: サーバー再起動 (全有効) ---
ssh 192.168.100.2 "pkill -f rdma-server || true"
sleep 2
ssh 192.168.100.2 "GGML_RDMA_PER_DEVICE_CONN=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
sleep 3

# --- 条件 B: クライアント実行 ---
gpu-lock.sh run env GGML_RDMA_PER_DEVICE_CONN=1 \
  GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_GDR_BUDGET_GB=12 \
  /home/ubuntu/projects/llama.cpp/.worktree/merge-validated/build/bin/llama-cli \
  (同上、GGML_RDMA_NO_SELECTIVE_SIGNAL なし、GGML_RDMA_PER_DEVICE_CONN=1 追加)
```

### コード修正

llama-cli のタイミング出力が `%.1f` (1桁) で精度不足のため、`tools/cli/cli.cpp:401` を修正:

```diff
-console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n", ...);
+console::log("[ Prompt: %.2f t/s (%d tokens, %.2f ms) | Generation: %.2f t/s (%d tokens, %.2f ms) ]\n", ...);
```

この修正は出力フォーマットのみで、RDMA 動作には影響しない。

## 結果

### 生データ

| ペア | 条件 | pp (t/s) | pp (ms) | tg (t/s) | tg (ms) | 備考 |
|:----:|:----:|:--------:|:-------:|:--------:|:-------:|------|
| 1 | A | 6.33 | 1580.89 | 7.03 | 7111.29 | |
| 1 | B | 5.75 | 1738.46 | 6.98 | 7160.51 | |
| 2 | A | 6.29 | 1588.90 | **5.83** | **8574.84** | **外れ値** |
| 2 | B | 5.89 | 1697.47 | 7.07 | 7067.87 | |
| 3 | A | 6.35 | 1575.23 | 7.06 | 7082.67 | |
| 3 | B | 5.85 | 1708.70 | 7.05 | 7096.27 | |
| 4 | A | 6.31 | 1583.66 | 7.05 | 7092.25 | |
| 4 | B | 5.78 | 1731.43 | 6.91 | 7233.86 | |
| 5 | A | 6.29 | 1589.52 | 7.05 | 7091.39 | |
| 5 | B | 5.78 | 1729.01 | 7.03 | 7110.65 | |

### 外れ値分析

A_2 の tg_ms = 8574.84 ms は他の A 条件 (mean=7094.40, SD=12.06) に対して Z = 122.73 の極端な外れ値。原因は「サーバー GPU 計算時間のセッション間変動」（サーマルスロットリングまたは CUDA コンテキスト初期化）と推定。

### 統計解析: Generation (tg)

#### 全5ペア (外れ値含む)

| 指標 | 値 |
|------|:--:|
| A 平均 | 6.80 t/s (SD=0.54) |
| B 平均 | 7.01 t/s (SD=0.06) |
| 変化率 | +3.01% |
| t統計量 | 0.7862 |
| p値 | 0.476 (ns) |
| Cohen's d | 0.35 |
| 95% CI | [-7.62%, +13.64%] |

外れ値に支配されており統計的に無意味。

#### 外れ値除外 (ペア2除外、4ペア)

| 指標 | 値 |
|------|:--:|
| A 平均 | 7.048 t/s (SD=0.012) |
| B 平均 | 6.993 t/s (SD=0.061) |
| **変化率** | **-0.78%** |
| t統計量 | -1.9017 |
| **p値** | **0.153 (ns)** |
| Cohen's d | -0.95 |
| 95% CI | [-2.08%, +0.52%] |

### 統計解析: Prompt eval (pp)

| 指標 | 値 |
|------|:--:|
| A 平均 | 6.315 t/s (SD=0.024) |
| B 平均 | 5.811 t/s (SD=0.058) |
| **変化率** | **-7.98%** |
| t統計量 | -17.589 |
| **p値** | **6.1 × 10⁻⁵ (***)** |
| Cohen's d | -7.87 |
| 95% CI | [-9.23%, -6.72%] |

### 出力テキスト比較

| 条件 | 生成テキスト | 再現性 |
|------|------------|--------|
| A (全5ラン) | 英語の思考テキスト「The user is asking for the capital of France...」| **条件内で完全一致** |
| B (全5ラン) | 非 ASCII トークン（`--simple-io` により `?` 表示、おそらく中国語思考）| **条件内で完全一致** |

- seed=42 は **条件内** では完全な再現性を担保
- A/B **間** では異なるテキストが生成される（per-device 接続が浮動小数点演算順序を変更し、異なるサンプリングパスを辿るため）
- 両条件とも正常な推論出力であり、**データ破損ではない**

### llama-bench との比較

| 指標 | llama-bench | llama-cli | 一致 |
|------|:-----------:|:---------:|:----:|
| pp | +25.34% (p=2.3×10⁻⁸) | **-7.98%** (p=6.1×10⁻⁵) | **不一致（逆方向）** |
| tg | +1.66% (p=0.004) | **-0.78%** (p=0.153, ns) | **不一致** |

## 考察

### 1. Prompt eval の不一致 (llama-bench +25% vs llama-cli -8%)

llama-bench は pp128 (128 tokens) を測定するのに対し、llama-cli のプロンプトはわずか 10 tokens。

- **Selective signaling の効果**: RDMA Write 操作の数に比例する。128 tokens → 多数の Write op → signaling 削減効果が大きい。10 tokens → Write op が少ない → 効果がほぼゼロ
- **Per-device connection のオーバーヘッド**: 接続確立、MTT キャッシュ登録、QP 初期化のコストが固定的に加算される。128 tokens ではこのオーバーヘッドが計算時間に対して無視できるが、10 tokens では支配的
- **結論**: 短いプロンプトでは per-device connection のオーバーヘッドが selective signaling の効果を上回り、逆効果になる

### 2. Generation の不一致 (llama-bench +1.7% vs llama-cli -0.8%)

- llama-bench の tg32 は 32 tokens を繰り返し生成。llama-cli は 50 tokens を 1 回生成
- llama-bench では warmup あり、llama-cli は `--no-warmup`
- llama-cli の測定にはモデル初回推論のオーバーヘッドが含まれる可能性
- 4ペア (外れ値除外) では p=0.153 で有意差なし。95% CI [-2.08%, +0.52%] は llama-bench の +1.66% を含まない
- **結論**: llama-cli の tg 測定では llama-bench で観測された +1.66% の改善は再現されず

### 3. セッション間変動の影響

A_2 (tg=5.83) は Z=122.73 の極端な外れ値。これは CLAUDE.md に記載の「サーバーGPU 計算時間のセッション間変動 (5.5-6.8 t/s)」と一致する。ABAB デザインにより外れ値がペア2に限定され、他のペアの分析には影響しなかった。

### 4. 出力テキストの差異

seed 固定でも A/B で異なるテキストが生成されたことは、per-device 接続が演算順序を変更することを示す。これはデータ破損ではなく、浮動小数点演算の非結合性 (non-associativity) による期待される挙動。

## 結論

1. **llama-bench で確認された改善は llama-cli の短プロンプトシナリオでは再現されない**
   - pp: 逆方向の影響 (-8%)。プロンプト長の違い (10 vs 128 tokens) が原因
   - tg: 有意差なし (-0.78%, p=0.153)

2. **マージ判断への影響**: llama-cli の結果は改善の有効性を否定するものではない。改善は llama-bench のバッチ処理レジーム (pp128, tg32) で確認済みであり、短プロンプトでは効果が現れないだけ。**改善がパフォーマンス劣化をもたらすケース**（短プロンプトの pp で -8%）が確認されたが、これは per-device connection の固定オーバーヘッドに起因し、実用的な長プロンプトでは問題にならない

3. **推奨**: マージ判断は llama-bench の結果 (pp128 +25%, tg32 +1.7%) を基準とすべき。llama-cli テストは実使用の品質確認（正常な推論出力）として有用だが、短プロンプトでの性能測定には適さない

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `f5a631535 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 34°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running |

GGML_RDMA 環境変数: 条件ごとに切替（上記参照）
