# ビルド手順の安定化調査レポート

- **実施日時**: 2026年2月7日 15:22 - 17:30

## 前提・目的

llama.cpp RDMAバックエンドのビルドで「`cmake --build` が出力なしで実際にビルドされない」問題が過去のセッションで繰り返し発生していた。MEMORY.md にワークアラウンド (`make -C build -j$(nproc)`) は記録済みだったが、根本原因の特定と信頼できるビルド手順の確立ができていなかった。

### 仮説

1. シェルのワンライナー構成 (`&&`チェーン、`$(nproc)`展開、SSH内でのエスケープ) が原因
2. `cmake --build` の `--config Release` フラグ (Unix Makefiles では無意味) が問題
3. Claude Code の CWD リセット挙動が原因

### 調査環境

- **1号機** (192.168.100.1): 48コア、CUDA 12.0、nvcc: `/usr/bin/nvcc`
- **2号機** (192.168.100.2): 40コア、CUDA 12.0、nvcc: `/usr/bin/nvcc`
- テスト用ワークツリー: `/home/ubuntu/projects/llama.cpp/.worktree/build-test` (detached HEAD @ 76bf9b9ae)

## 再現方法

### テスト用ワークツリー作成

```bash
git -C /home/ubuntu/projects/llama.cpp worktree add --detach \
  /home/ubuntu/projects/llama.cpp/.worktree/build-test feature/rdma-backend
```

### cmake configure (共通)

```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/build-test
rm -rf build
cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=60
```

### ビルドコマンド体系テスト

各パターンで `rm -rf build && cmake -B build ...` からやり直して実行:

| パターン | コマンド |
|---------|---------|
| C | `cmake --build build --config Release -- -j $(nproc)` |
| D | `cmake --build build -- -j$(nproc)` |
| E | `make -C build -j$(nproc)` |
| F | `make -C build -j $(nproc)` |

### SSH リモートビルドテスト

```bash
# テスト用コード転送
rsync -a --exclude='.git' --exclude='build' \
  /home/ubuntu/projects/llama.cpp/.worktree/build-test/ \
  192.168.100.2:/tmp/build-test/

# パターン1: ダブルクォート + cmake --build
ssh 192.168.100.2 "cd /tmp/build-test && rm -rf build && \
  cmake -B build ... && cmake --build build -- -j \$(nproc)"

# パターン2: ダブルクォート + make
ssh 192.168.100.2 "cd /tmp/build-test && rm -rf build && \
  cmake -B build ... && make -C build -j\$(nproc)"

# パターン3: シングルクォート + make
ssh 192.168.100.2 'cd /tmp/build-test && rm -rf build && \
  cmake -B build ... && make -C build -j$(nproc)'
```

## 結果

### 1. cmake configure

- 終了コード: 0 (成功)
- `build/Makefile` 生成: OK
- RDMA 検出: OK (`ibverbs`, `rdmacm`, GPUDirect RDMA available)
- CUDA 検出: OK (CUDA 12.0.140)
- **発見**: CLAUDE.md に記載の `/usr/local/cuda-12.9/bin/nvcc` は存在しない。正しくは `/usr/bin/nvcc` (CUDA 12.0)

### 2. ビルドコマンド体系テスト結果

| パターン | コマンド | Exit | 出力行数 | ビルド時間 | rdma-server | llama-bench |
|---------|---------|:----:|:-------:|:---------:|:-----------:|:-----------:|
| C | `cmake --build build --config Release -- -j $(nproc)` | 0 | 715 | 396s | EXISTS | EXISTS |
| D | `cmake --build build -- -j$(nproc)` | 0 | 715 | 214s | EXISTS | EXISTS |
| E | `make -C build -j$(nproc)` | 0 | 1107 | 220s | EXISTS | EXISTS |
| F | `make -C build -j $(nproc)` | 0 | 1107 | 221s | EXISTS | EXISTS |

- **全パターンで正常にビルド成功。「無出力」問題は再現しなかった。**
- パターンCの396秒は初回ビルド (ccacheなし) によるもの。D/E/Fはccacheが効いて約半分。
- `--config Release` は Unix Makefiles では無視される (`CMAKE_BUILD_TYPE=Release` は cmake configure 時に設定済み)
- `make` は `cmake --build` より出力が多い (Entering/Leaving directory メッセージが加算)

### 3. 「無出力」問題の再現テスト結果

| テスト | 結果 |
|-------|------|
| リビルド時 (変更なし) | 出力あり (97行の "Built target" メッセージ) |
| `&&` チェーン ワンライナー | 正常 (715行) |
| `bash -c` サブシェル | 正常 (715行) |
| build dir なし | エラー出力あり (`Error: ... is not a directory`, exit=1) |
| build dir あるが Makefile なし | エラー出力あり (`Error: could not load cache`, exit=1) |

**「完全に無出力」のケースは一切再現しなかった。**

### 4. SSH リモートビルドテスト結果

| パターン | 結果 | `$(nproc)` 展開 |
|---------|------|----------------|
| ダブルクォート + `\$(nproc)` | 成功 (715行) | リモート (40) |
| ダブルクォート + `$(nproc)` (エスケープなし) | 成功だがリモート値ではない | ローカル (48) |
| シングルクォート | 成功 (1107行) | リモート (40) |

**全SSH パターンでビルド成功。**

### 5. `$(nproc)` 展開の挙動

| SSH クォート | 展開場所 | 値 | 正しさ |
|-------------|---------|:--:|:-----:|
| `"echo \$(nproc)"` | リモート | 40 | 正しい |
| `'echo $(nproc)'` | リモート | 40 | 正しい |
| `"echo $(nproc)"` (エスケープなし) | ローカル | 48 | 間違い |

## 分析・結論

### 「無出力」問題の根本原因

今回の体系的テストでは問題を再現できなかった。過去のセッションで報告された「無出力」の推定原因:

1. **CWD リセット (最有力仮説)**: Claude Code は各 Bash ツール呼び出しの間で作業ディレクトリをプロジェクトルートにリセットする。`cd /path && cmake --build build` を別々の Bash 呼び出しに分割した場合、2回目の呼び出しはプロジェクトルートで実行され、`build` ディレクトリが見つからずエラーになる。ただし、この場合はエラーメッセージが出力されるため「完全な無出力」にはならない。

2. **configure 失敗の見落とし**: `&&` チェーンで `cmake -B build` が失敗した場合、後続の `cmake --build` は実行されない。チェーン全体の出力がなかったように見える可能性がある。

3. **nvcc パスの問題**: CLAUDE.md に記載されていた `/usr/local/cuda-12.9/bin/nvcc` は実際には存在しない。これにより configure が失敗し、build ステップに到達しなかった可能性が高い。

### `--config Release` について

- Unix Makefiles ジェネレータでは `--config` フラグは無視される
- `CMAKE_BUILD_TYPE=Release` は cmake configure 時に自動設定される (llama.cpp の CMakeLists.txt による)
- 害はないが不要なので除去を推奨

### CLAUDE.md の更新内容

1. `CMAKE_CUDA_COMPILER` を `/usr/local/cuda-12.9/bin/nvcc` → `/usr/bin/nvcc` に修正
2. `--config Release` を除去
3. `-DLLAMA_CURL=ON` を除去 (非推奨で無視される)
4. SSH でのビルドコマンドに関する注意事項を追加

### 推奨ビルドコマンド

**ローカル (1号機)**:
```bash
rm -rf build
cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="60"
cmake --build build -- -j $(nproc)
```

**リモート (2号機、SSH 経由)**:
```bash
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && \
  rm -rf build && \
  cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=60 && \
  cmake --build build -- -j \$(nproc)"
```

`cmake --build` と `make` は同等に動作する。`make -C build -j$(nproc)` でも可。
