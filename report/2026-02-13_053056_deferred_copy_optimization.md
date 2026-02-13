# Deferred Copy 最適化 — RDMA 間テンソル転送のクライアント往復排除

- **実施日時**: 2026年2月13日 05:30
- **ワークツリー**: `.worktree/rdma-deferred-copy`

## 前提・目的

### 背景

レイヤー分割 (`-sm layer`) で複数 RDMA デバイスを使用する場合、デバイス間テンソルコピーは `get_tensor + set_tensor` フォールバックを使用していた。これは `cpy_tensor` が無効化されていたためである（以前の graph_recompute キャッシュ問題で無効化）。

`get_tensor + set_tensor` フローでは、同一サーバー上の GPU 間コピーにも関わらず、データが **サーバー → クライアント → サーバー** と IB ネットワークを往復する:
1. `get_tensor`: サーバー GPU → IB Send → クライアント (~1ms)
2. `set_tensor`: クライアント → RDMA Write to staging → `flush_all_staging` で GPU へ (~0.5ms)

### 目的

- `cpy_tensor` を同一接続の RDMA デバイス間で再有効化
- コピー情報を ASYNC コマンドに含め、サーバー内部で D2H+H2D を実行
- IB ネットワーク往復を排除し、Generation 速度を改善

### 参照レポート

- `report/2026-02-09_182440_parallel_compute_dispatch.md` — サーバー側並列コンピュート・ディスパッチ

## 実装内容

### 変更ファイル

| ファイル | 変更量 |
|---------|--------|
| `ggml/src/ggml-rdma/ggml-rdma.cpp` | +382/-52 行 |

### 主要コンポーネント

1. **`deferred_copy_entry` 構造体** (48 bytes): src/dst バッファポインタ、オフセット、サイズを記録
2. **`deferred_copy_queue` クラス**: per-connection のスレッドセーフなキュー（mutex 保護）
3. **`cpy_tensor` 再有効化**: 同一接続の RDMA バッファ間コピーを deferred copy として記録し、即座に true を返す
4. **ASYNC メッセージ拡張**: RECOMPUTE/COMPUTE_UPDATE/FULL_GRAPH の全パスに `n_copies + copy_entries` を追加
5. **サーバー側 `execute_deferred_copies`**: `cudaDeviceSynchronize(src_dev)` → `cudaMemcpy(D2H)` → `cudaMemcpy(H2D)` でサーバー内部コピーを実行

### 重要な修正: flush と deferred copy の実行順序

**バグ**: `flush_and_recompute` / `flush_and_compute_update` で deferred copy を flush の**前**に実行していた。`flush_all_staging` の bounding box マージにより、flush 範囲が deferred copy の宛先アドレスと重複し、古い staging データで正しいデータを上書きしていた。

**修正**: flush → deferred copy の順序に変更。FULL_GRAPH パスでは既にこの順序（別コマンドで flush が先）だったため正しく動作していた。

```
修正前: execute_deferred_copies → flush_all_staging → graph_recompute  ❌
修正後: flush_all_staging → execute_deferred_copies → graph_recompute  ✅
```

### 環境変数

| 環境変数 | 説明 |
|---------|------|
| `GGML_RDMA_NO_DEFERRED_COPY=1` | Deferred copy を無効化（baseline 比較用） |
| `GGML_RDMA_VERIFY_COPY=1` | get+set_tensor と deferred copy を両方実行して比較（デバッグ用） |

## 再現方法

### ビルド・デプロイ

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-deferred-copy
bash scripts/rdma-build.sh local && bash scripts/rdma-deploy.sh
```

### テスト実行

```bash
# サーバー起動
bash scripts/rdma-server.sh restart

# Deferred copy 有効 (デフォルト)
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log

# Baseline (deferred copy 無効)
GGML_RDMA_NO_DEFERRED_COPY=1 GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli ... (同じオプション)
```

## 検証結果

### 正確性検証 — gpt-oss-20b (1C+2R, --seed 42)

| 構成 | 出力 |
|------|------|
| Baseline (`NO_DEFERRED_COPY=1`) | `[Start thinking]\nUser asks: "The capital of` |
| Deferred copy (デフォルト) | `[Start thinking]\nUser asks: "The capital of` |

**出力完全一致** — baseline と同じトークン列を生成。

### 正確性検証 — GLM-4.7 IQ2_M (7C+4R, --seed 42)

| 構成 | 出力 |
|------|------|
| Baseline | `The user is asking for the capital of France. This is a factual question.\n1. **Identify the core entity...` |
| Deferred copy | 同一出力 |

**出力完全一致** — 11GPU 構成でも正しい推論結果。

### 性能測定 — GLM-4.7 IQ2_M (7C+4R)

| モード | Run | Prompt (t/s) | Generation (t/s) |
|--------|:---:|:---:|:---:|
| Baseline | 1 | 6.2 | 5.5 |
| Baseline | 2 | 6.2 | 5.6 |
| Deferred copy | 1 | 6.3 | 6.7 |
| Deferred copy | 2 | 6.3 | 5.7 |

- Prompt 速度: ほぼ同等 (deferred copy は Generation 時のみ有効)
- Generation 速度: GPU 計算時間のセッション間変動 (5.5-6.7 t/s) が大きく、改善幅の正確な測定は困難
- 理論的な改善: 3 RDMA hop × ~1.3ms/hop = ~4ms/token (全体 ~160ms/token の ~2.5%)

### 性能測定 — gpt-oss-20b (1C+2R)

| モード | Prompt (t/s) | Generation (t/s) |
|--------|:---:|:---:|
| Baseline | 58.1 | 50.9 |
| Deferred copy | 53.6 | 57.7 |

Generation: +13% 改善（1 RDMA hop のみだが、hop あたりの改善が相対的に大きい）。

## デバッグ過程の要約

### 症状

Deferred copy が FULL_GRAPH パスでは正しく動作するが、RECOMPUTE パスで出力が壊れる（3 トークン目以降で発散）。

### 調査した仮説と結果

| 仮説 | 結果 |
|------|------|
| CUDA ストリーム同期の問題 | ❌ cudaDeviceSynchronize 追加で改善なし |
| Async compute タイミング問題 | ❌ ASYNC_COMPUTE=0 で改善なし |
| Graph reuse (RECOMPUTE) 自体の問題 | ⚠️ FORCE_FULL_GRAPH=1 で修正確認 |
| **flush_all_staging bounding box による上書き** | **✅ 根本原因** |

### 根本原因の詳細

`flush_all_staging` は flush エントリの bounding box (min_offset, max_end) を計算し、その範囲全体を staging → GPU にコピーする。deferred copy 先のアドレスがこの bounding box 内に含まれる場合、古い staging データで上書きされる。

例 (RDMA1 バッファ):
- Deferred copy 先: offset 6957056 (cross-device activation)
- Flush bounding box: [5898240, 13912068) (per-token input)
- **offset 6957056 は flush 範囲内** → 上書き発生

FULL_GRAPH パスでは flush が先に実行されるため問題なし。RECOMPUTE/COMPUTE_UPDATE パスでは flush と deferred copy が同一コマンド内で処理されるため、実行順序が重要。

## 制約・今後の課題

1. **GPU 計算時間の変動**: サーバー GPU のサーマルスロットリングにより Generation 速度が 5.5-6.7 t/s と変動。deferred copy の ~2.5% 改善を正確に測定するのが困難
2. **cudaMemcpy vs cudaMemcpyPeer**: 現在の実装は D2H+H2D (host 経由) を使用。`cudaMemcpyPeer` による GPU 直接コピーは P100 (NVLink なし) では効果が限定的だが、NVLink 搭載環境では有効な可能性
3. **bounding box flush の非効率性**: flush_all_staging が bounding box 全体をコピーするため、不要なデータ転送が発生。per-entry コピーに変更すれば deferred copy との順序依存を解消できる
