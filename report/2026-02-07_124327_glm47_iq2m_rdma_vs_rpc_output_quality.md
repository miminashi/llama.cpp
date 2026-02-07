# GLM-4.7 IQ2_M 出力品質 RDMA vs RPC 比較実験レポート

- **実施日時**: 2026年2月7日 11:08〜12:43
- **関連レポート**: [GLM-4.7 IQ2_M 11GPU テスト](2026-02-07_003000_glm47_iq2m_11gpu_test.md)

## 前提・目的

GLM-4.7 IQ2_M を RDMA バックエンドで 11GPU クラスタ実行した際、thinking トークン (`[Start thinking]`〜`[End thinking]`) がゴミ文字列になる問題が発生していた。ユーザーは CPU+GPU 併用 (RDMA なし) で同モデルの正常な日本語生成を確認済みのため、RDMA バックエンド固有の問題の可能性がある。

**目的**: RPC バックエンドおよびローカル実行との比較により、ゴミ出力の原因が RDMA バックエンド固有なのか、IQ2_M 量子化の限界なのかを切り分ける。

## 実験設計

### 比較構成

| # | 構成 | GPU数 | 備考 |
|---|------|:-----:|------|
| A | **RDMA** 7C+4R | 11 | 問題再現 |
| B | **RPC** 7C+2R | 9 | RDMA固有か判定 |
| C | **ローカル** 7C + CPU offload | 7+CPU | ベースライン |

### テストプロンプト

| ID | プロンプト | `-n` |
|----|-----------|:----:|
| P1 | `The capital of France is` | 200 |
| P2 | `こんにちは。日本の首都はどこですか？` | 200 |
| P3 | `1+1=` | 100 |

### 共通パラメータ

- モデル: GLM-4.7 IQ2_M (114.02 GiB, 358.34B params, 3ファイル split)
- `--seed 42`, `--simple-io`, `--single-turn`, `--no-warmup`
- `-c 2048`, `-sm layer`, `-ngl 999` (構成C以外)
- 構成C: `-ngl 50` (71レイヤー中50をGPU、残りCPU)

## 再現方法

### ビルド (両ノード)

```bash
# 1号機
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
make -C build llama-cli rdma-server rpc-server -j$(nproc)

# 2号機
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' ./ 192.168.100.2:/home/ubuntu/projects/llama.cpp/
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && rm -rf build && \
  cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release && \
  make -C build llama-cli rdma-server rpc-server -j\$(nproc)"
```

### サーバー起動 (2号機)

```bash
# RDMA サーバー
ssh 192.168.100.2 "GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"

# RPC サーバー (GPU0=50052, GPU1=50053)
ssh 192.168.100.2 "CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rpc-server -H 0.0.0.0 -p 50052 > /tmp/rpc-server-0.log 2>&1 &"
ssh 192.168.100.2 "CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rpc-server -H 0.0.0.0 -p 50053 > /tmp/rpc-server-1.log 2>&1 &"
```

### 構成 A: RDMA (7C+4R)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_NO_GDR=1 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 200 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### 構成 B: RPC (7C+2R)

```bash
LD_LIBRARY_PATH=build/bin build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  --rpc 192.168.100.2:50052,192.168.100.2:50053 \
  -sm layer -ngl 999 -c 2048 -n 200 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

### 構成 C: ローカル (7C + CPU offload)

```bash
LD_LIBRARY_PATH=build/bin build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -ngl 50 -c 2048 -n 200 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## 実験結果

### 出力品質まとめ

| 構成 | P1 (France) | P2 (日本語) | P3 (1+1) | 判定 |
|------|:-----------:|:-----------:|:---------:|:----:|
| **A: RDMA 7C+4R** | ゴミ | ゴミ | ゴミ | **NG** |
| **B: RPC 7C+2R** | 正常 | 正常 | 正常 | **OK** |
| **C: ローカル 7C+CPU** | 正常 | 正常 | 正常 | **OK** |

### 性能比較

| 構成 | pp (t/s) | tg (t/s) |
|------|:--------:|:--------:|
| A: RDMA 7C+4R | 6.4〜8.8 | 6.8〜6.9 |
| B: RPC 7C+2R | 6.0〜8.6 | 7.8〜7.9 |
| C: ローカル 7C+CPU | 2.8〜3.6 | 1.2〜1.3 |

### 構成 A: RDMA — 出力詳細

**全プロンプトで同一のゴミパターン:**

```
[Start thinking]
2%D<6C6E3@+B#G/.<GG,H7'-C.)/"$>6G#:HG--><F5H%(*9&7F6H#B1&,'F(:&'<A=5G*H=G%0D<#@!$6C)>@<A:)'$B--70;6%4":03<:%#<04.=;69*41;E!A@GE=5,"0A-59:H+--003/A/*7E.>,6:<C)-2"/H22380$1902EC#H,F>B9:095/,A/':4%A0$$H4
```

注目点:
- **P1/P2/P3 で全く同じゴミ文字列** — プロンプト入力に関係なく同じ出力
- `[Start thinking]` トークン自体は正しく出力されている
- その後のトークンが全て壊れている
- `--seed 42` が設定されているが、異なるプロンプトで同じ出力 → サンプリングが入力を反映していない

### 構成 B: RPC — 出力詳細

**P1: The capital of France is**
```
[Start thinking]
The user is asking for the capital of France. This is a factual question.
1. **Identify the core entity and attribute:** Entity: France, Attribute: Capital city
2. **Access knowledge base:** Result: Paris.
...
[End thinking]
Paris.
```

**P2: こんにちは。日本の首都はどこですか？**
```
[Start thinking]
The user is asking for the capital of Japan in Japanese...
[End thinking]
こんにちは。日本の首都は**東京**です。
```

**P3: 1+1=**
```
[Start thinking]
The user has provided a simple arithmetic expression: "1+1=". This is a very common and fundamental question...
[End thinking]
2
```

### 構成 C: ローカル — 出力詳細

構成 B と同等の品質。全プロンプトで正しいthinkingと回答を生成。

## 分析

### 判定結果

**RDMA のみゴミ、RPC/Local 正常** → **RDMA バックエンド固有のデータ転送バグ**

### 症状の特徴

1. **プロンプト非依存のゴミ出力**: 3つの異なるプロンプトで完全に同一のゴミ文字列が出力される。これはサンプリングに使用される logits テンソルが壊れていることを強く示唆する。

2. **`[Start thinking]` は正常**: チャットテンプレートの特殊トークンは正しく出力されており、トークナイザやチャットテンプレート処理は正常。問題はその後の通常トークン生成で発生。

3. **ゴミパターンの再現性**: seed=42 で同一のゴミが再現されるため、ランダムな破損ではなく、決定論的に壊れた logits から一貫してサンプリングされている。

### 推定原因

RDMA バックエンドの `get_tensor` (サーバー → クライアントへの logits テンソル読み出し) でデータ破損が発生している可能性が高い。具体的には:

- サーバー側で計算された logits の読み出し時にアドレス計算ミスまたはバッファ不整合
- mmap ロードとの組み合わせで、大規模モデル (114GB) のバッファオフセットが不正になる
- RDMA Write / Send/Recv フォールバック時のデータ転送エラー

### RPC と RDMA の違い

- RPC: llama.cpp のアップストリーム実装。各 RPC サーバーが独立した GPU を管理
- RDMA: カスタム実装。1つの rdma-server が複数 GPU を管理し、クライアントが RDMA デバイスとして認識

RPC で正常に動作するため、モデル自体 (IQ2_M 量子化)、レイヤー分割ロジック、スケジューラには問題がない。**RDMA バックエンドのテンソルデータ転送パスにバグがある**。

## 次のアクション

1. **RDMA バックエンドの `get_tensor` / logits 転送パスをデバッグ**
   - サーバー側で計算された logits と、クライアントが受信した logits を比較
   - 特に大規模バッファ (>4GB) でのオフセット計算を確認

2. **小規模モデルでの RDMA 出力品質を確認**
   - gpt-oss-20b や qwen2.5-0.5b で同じテストを実行
   - 問題が大規模モデル固有か、RDMA 全般の問題かを切り分け
