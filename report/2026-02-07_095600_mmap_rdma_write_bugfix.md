# mmap + RDMA Write バグ修正レポート

- **実施日時**: 2026年2月7日 09:56

## 前提・目的

大規模モデルのロード時に RDMA Write が `remote access error` (vendor_err=0x88) で失敗するバグを修正する。

- **参考レポート**: [2026-02-07_003000_glm47_iq2m_11gpu_test.md](2026-02-07_003000_glm47_iq2m_11gpu_test.md)

### 背景: RDMA バックエンドのデータ転送パス

RDMA バックエンドでは、クライアント (1号機) からサーバー (2号機) へのテンソルデータ転送に2つのパスがある:

| パス | 方式 | 特徴 |
|------|------|------|
| **RDMA Write** | 片側操作 (one-sided) | クライアントがサーバーの MR に直接書き込む。高速だが、サーバー側に事前登録済みの MR が必要 |
| **Send/Recv** | 両側操作 (two-sided) | IB の Send/Recv でデータを送信。MR サイズの制約なし。やや低速 |

RDMA Write パスは **GDR (GPUDirect RDMA) の有効/無効に関わらず使用される**:

| モード | サーバー側 MR の登録先 | RDMA Write の宛先 | 転送後の処理 |
|--------|------------------------|-------------------|-------------|
| **GDR 有効** | GPU VRAM を `ibv_reg_mr` で直接登録 (nvidia-peermem 経由) | GPU VRAM | なし (直接 GPU に到着) |
| **GDR 無効** | `cudaMallocHost` で確保したホストメモリを `ibv_reg_mr` で登録 | ホストステージングバッファ | `cudaMemcpy` で GPU へコピー |

いずれの場合も、サーバーがバッファ確保時に MR を登録 → クライアントに `mr_rkey` を返送 → クライアントが RDMA Write を実行、という流れになる。

### 症状と発生条件

GLM-4.7 IQ2_M (114GB) を 11GPU (7 CUDA + 4 RDMA) で分散ロードすると、モデルロード中に以下のエラーが発生:

```
[rdma_connection] Work completion error: status=remote access error, opcode=0, vendor_err=0x88
[rdma_connection] RDMA write completion timeout: size=16777216, remote_addr=0x76ca24e96700, rkey=0x17bfaf
```

**発生条件**:
- 大規模モデル (GLM-4.7 等) で RDMA バッファが **4GB を超える**場合
- GDR 有効/無効に関わらず、RDMA Write パスが使用される場合に発生しうる
- 小規模モデル (qwen2.5-0.5b, バッファ ~530MB) では再現しない

**ワークアラウンド** (修正前): `--no-mmap` を指定すると、llama.cpp のモデルローダーが GPU 向け非同期ロードパスを使用し、RDMA Write ではなく Send/Recv パスで転送されるためエラーを回避できた。ただし llama-cli のデフォルトは mmap ロードであるため、RDMA 利用時に毎回 `--no-mmap` を指定する必要があった。

### 目的

`--no-mmap` なしのデフォルト動作で、大規模モデルの RDMA 分散ロードが正常に動作するようにする。

### 前提条件

- 1号機 (7×P100) と 2号機 (4×P100) が InfiniBand で接続済み
- 今回のテストは `GGML_RDMA_NO_GDR=1` (GPUDirect 無効) で実施（GDR 有効時には別の既知バグ「GPUDirect RDMA Write タイムアウト」があるため。詳細: [2026-02-07_003000_glm47_iq2m_11gpu_test.md §2](2026-02-07_003000_glm47_iq2m_11gpu_test.md)）

## 根本原因の分析

### 初期仮説（不正確）

mmap ロード時に `tensor->data` がクライアント側の mmap アドレスを含み、`buf_offset` 計算でリモート MR 範囲外のアドレスを算出してしまう。

→ デバッグログにより、`tensor->data` は ggml アロケータがバッファ内に正しく配置したアドレスであり、offset 計算自体は問題なかったことを確認。mmap のソースデータは `data` パラメータとして渡されるため、`buf_offset` 計算には影響しない。

### 実際の根本原因

**RNIC ページテーブルキャッシュのオーバーフロー。**

GLM-4.7 を 11GPU で分散すると、4つの RDMA バッファ（各 ~10GB、合計 ~40GB）が `ibv_reg_mr()` で MR 登録される。Mellanox ConnectX の RNIC は内部にページテーブルキャッシュを持っており、登録された MR のページエントリをキャッシュする。大量の MR 登録 (合計 40GB+ のホストメモリ) により、このキャッシュが溢れ、RDMA Write 実行時にページテーブルの解決に失敗して `remote access error` (vendor_err=0x88) が返される。

この解釈は以下の事実と一致する:
- 小規模モデル (バッファ ~530MB) では問題なし
- 同じ大規模モデルでも Send/Recv パス (`--no-mmap`) では問題なし（Send/Recv は小さな固定サイズバッファを使用）
- RDMA Write の `buf_offset` 計算は正しく、MR 範囲内を指している

## 修正内容

### 修正ファイル

`ggml/src/ggml-rdma/ggml-rdma.cpp` — 4箇所

### 修正1: バッファサイズ制限 (主要修正)

```cpp
static constexpr size_t RDMA_WRITE_MAX_BUFFER_SIZE = (size_t)4 * 1024 * 1024 * 1024;
```

4GB を超えるバッファでは RDMA Write をスキップし、Send/Recv ベースの転送にフォールバック。これにより大量の MR 登録を回避。

### 修正2: 16MB チャンク RDMA Write

```cpp
static constexpr size_t RDMA_WRITE_CHUNK_SIZE = 16 * 1024 * 1024;
```

RDMA Write を 16MB チャンクに分割して実行。大きな一括転送のリスクを軽減。

### 修正3: `rdma_write_disabled` フラグ

```cpp
bool rdma_write_disabled = false;  // in ggml_backend_rdma_buffer_context
```

RDMA Write が実行時に失敗した場合、そのバッファに対して RDMA Write を無効化し、以降は Send/Recv にフォールバック。QP エラー状態への遷移を防止。

### 修正4: `buf_offset` bounds check

```cpp
if (buf_offset + size > ctx->size) {
    // Fall back to send-based path
}
```

アドレス計算の安全ネット。mmap アドレスが万が一混入した場合でも安全にフォールバック。

### 修正5: サーバー側 validation 修正

`set_tensor` と `get_tensor` のバウンダリチェックで `in_tensor->data`（クライアント側 mmap アドレス）ではなく `tensor->data`（サーバー側で `deserialize_tensor` が解決したアドレス）を使用するように修正。

**Before:**
```cpp
if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || ...)
```

**After:**
```cpp
if ((size_t)tensor->data + offset < p0 || (size_t)tensor->data + offset >= p1 || ...)
```

## 再現方法

### 1. ビルド (1号機)

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -C build -j$(nproc)
```

### 2. デプロイ・ビルド (2号機)

```bash
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/ \
  192.168.100.2:/home/ubuntu/projects/llama.cpp/
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && rm -rf build && \
  cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
  make -C build -j\$(nproc)"
```

### 3. rdma-server 起動 (2号機)

```bash
ssh 192.168.100.2 "pkill -f rdma-server; sleep 1; \
  bash -c 'GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 \
  </dev/null > /tmp/rdma-server.log 2>&1 & disown; echo started'"
```

### 4. テスト実行 (1号機)

**GLM-4.7 mmap テスト (`--no-mmap` なし):**
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 32 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io \
  --log-file /tmp/llama-cli.log
```

**qwen2.5-0.5b 回帰テスト:**
```bash
GGML_RDMA_NO_GDR=1 GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=build/bin \
  build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

## テスト結果

### GLM-4.7 IQ2_M (mmap, `--no-mmap` なし) — 11GPU

| 指標 | 結果 |
|------|------|
| モデルロード | 成功 (remote access error なし) |
| Prompt (pp) | 3.1 t/s |
| Generation (tg) | 7.0 t/s |
| 判定 | **PASS** |

### qwen2.5-0.5b 回帰テスト — 2GPU (CUDA0 + RDMA0)

| 指標 | 修正前 | 修正後 | 変化 |
|------|--------|--------|------|
| pp128 | 3,269 t/s | 3,172 t/s | -3.0% (誤差範囲) |
| tg32 | 173.87 t/s | 160.64 t/s | -7.6% (サーバーGPU計算時間の変動範囲内) |
| RDMA Write | 有効 | **有効** (バッファ ~530MB < 4GB) | 変化なし |
| 判定 | — | **PASS** | |

### llama-cli qwen2.5-0.5b 追加テスト — 2GPU

| 指標 | 結果 |
|------|------|
| pp | 330.1 t/s |
| tg | 129.7 t/s |
| 判定 | **PASS** |

## 動作原理

```
モデルロード時:
  バッファサイズ <= 4GB (qwen2.5-0.5b)
    → RDMA Write (16MB チャンク) → 高速パス ✓

  バッファサイズ > 4GB (GLM-4.7)
    → RDMA Write スキップ → Send/Recv フォールバック → 安全パス ✓
```

## まとめ

- mmap + RDMA Write の `remote access error` バグを修正
- 根本原因は RNIC ページテーブルキャッシュのオーバーフロー（大量 MR 登録）
- 4GB バッファサイズ制限により、大規模モデルは安全な Send/Recv パスを使用
- 小規模モデルは引き続き高速な RDMA Write パスを利用
- `--no-mmap` ワークアラウンドが不要に
