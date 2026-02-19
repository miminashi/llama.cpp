## 必読ドキュメント

- [AGENTS.md](AGENTS.md) — 作業開始前に必ず確認すること
- [REPORT.md](REPORT.md) — レポート作成ルール
- [GPU.md](GPU.md) — GPUサーバ情報
- [HISTORY.md](HISTORY.md) — 完了済みステップ (Step 1-4) の記録

## ルール

- **ビルド**: llama.cppをビルドする際は、必ず事前に `build` ディレクトリを削除してからビルドすること (`rm -rf build && cmake -B build ...`)
- **コード転送**: 2号機 (192.168.100.2) にコードを転送する際は、2号機に既に存在するllama.cppのディレクトリを削除したうえで、1号機のコードをコピーすること
- **実行**: `llama-cli` を実行する際は、必ず `--log-file /tmp/llama-cli.log` オプションを付けること (ユーザが別ターミナルで `tail -f /tmp/llama-cli.log` によりリアルタイムにログを確認できるようにするため)
- **Bashコマンドに `#` コメント行を含めない**: コメント付きコマンドはパーミッション自動承認が効かないため、コメントは Bash ツールの description パラメータに記載すること
- **マルチラインコマンドを避ける**: 改行区切りの複数コマンドはパーミッション自動承認が効かない。`&&` や `;` で1行にまとめるか、複数の Bash 呼び出しに分割すること。ただし `&&`/`;` チェインもビルトイン安全コマンド（`echo`, `true` 等）以外の異種コマンドの組み合わせでは自動承認されないため、複数 Bash 呼び出しへの分割が最も確実
- **ファイルへのリダイレクトを使わない**: `2>/tmp/file.log` や `>/tmp/file.log` 等のファイルリダイレクトはパーミッション自動承認が効かない（`2>/dev/null` と `2>&1` のみ許可）。stderr をファイルに保存したい場合は、Bash ツールの出力を直接利用すること
- **パーミッション設定の優先順位**: プロジェクト `settings.local.json` に `permissions.allow` がある場合、グローバル `settings.local.json` の `permissions.allow` は置換される（マージされない）。プロジェクト設定には必要なグローバルパターンも含めること。ワークツリーの場合、メインリポジトリの `.claude/settings.local.json` が読まれるため、設定変更はメインリポジトリ側で行うこと
- **SSH コマンド**: `ssh`, `rsync`, `scp` は `settings.local.json` に `Bash(ssh *)` 等を含めれば自動承認される。ただし、2号機でのビルド・デプロイ・サーバー管理には `scripts/rdma-*.sh` ラッパースクリプトを推奨（エラーハンドリング・ログ管理が組み込まれているため）
- **`git -C` を使わない**: `Bash(git -C *)` は push 等の破壊的コマンドも許可するため安全でない。代替手段: (1) 対象ディレクトリに `cd` してから `git` を実行 (`cd /path && git status`)、(2) 現在のワークツリーで作業中なら `-C` は不要
- **別ワークツリーでのコマンド実行**: `cd /path/to/worktree && bash scripts/...` パターンは使わない（`cd` は非ビルトインのため `&&` チェインが自動承認されない）。代わりに**絶対パス**でスクリプトやバイナリを呼ぶ: `bash /absolute/path/to/worktree/scripts/rdma-build.sh local`。`rdma-*.sh` スクリプトは `$(dirname "$0")` でパス解決するため任意のディレクトリから呼べる。バイナリ実行も絶対パスを使う: `/absolute/path/to/worktree/build/bin/llama-cli ...`
- **ツールのパスに `~` を使わない**: Read, Glob, Grep 等のツールは `~` をシェル展開しない。`~/projects/...` ではなく `/home/ubuntu/projects/...` のように絶対パスを使うこと
- **ワークツリー**: 改善策を実装する際は、`feature/rdma-backend` ブランチから新しいワークツリーを作成して作業すること。ワークツリーは `/home/ubuntu/projects/llama.cpp/.worktree/` 配下に作成する。実装が完了したらワークツリー上でコミットするが、`feature/rdma-backend` へのマージは行わないこと（マージはユーザーが判断する）
- **レポート作成**: plan mode を使用してまとまった作業を行った場合は、完了時にレポートを作成すること。フォーマットは [REPORT.md](REPORT.md) に従う。レポートは作業ワークツリーに関わらず、常に `/home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/report/` に作成する

## マルチセッション ワークフロー

### タスクボード

タスク管理: `bash scripts/task-board.sh <command>`

```bash
task-board.sh list                           # 全タスク一覧
task-board.sh add "タスク名" [ワークツリー名]  # 新規タスク追加 (state=impl)
task-board.sh test <ID> [メモ]               # impl → test に遷移
task-board.sh done <ID> [結果メモ]            # test → done に遷移
task-board.sh delete <ID>                    # タスク削除
task-board.sh show <ID>                      # タスク詳細表示
```

状態遷移: `impl`(実装中) → `test`(実験待ち) → `done`(完了)

タスクデータは `/tmp/rdma-workflow/tasks.yaml` に保存。更新操作のたびに `TASK.md` が自動生成される。

### GPU ロック

GPU やリモートノードの状態に影響する操作は排他制御を通して実行すること:

```bash
gpu-lock.sh status                           # ロック状態確認
gpu-lock.sh run <command...>                 # 即座に実行 (ロック中ならエラー)
gpu-lock.sh wait [--timeout N] <command...>  # ロック待ち→実行
```

#### ロック必須の操作

| カテゴリ | 操作 | 理由 |
|---------|------|------|
| **GPU 推論** | `llama-bench`, `llama-cli`, その他 GPU を使う CUDA プログラム | GPU リソース競合 |
| **サーバー管理** | `rdma-server.sh stop`, `rdma-server.sh restart` | 実行中クライアントの RDMA 接続が切断される |
| **デプロイ** | `rdma-deploy.sh` | 2号機のビルドディレクトリ削除・再構築。テスト中に実行すると整合性が崩れる |
| **カーネルモジュール** | `modprobe nvidia-peermem`, `rmmod nvidia-peermem` | GDR モジュール操作は実行中の RDMA 転送をクラッシュさせる |
| **GPU 設定変更** | `nvidia-smi -pl`, `nvidia-smi -ac`, persistence mode 変更 | ベンチマーク計測値に影響 |

#### ロック不要の操作

| 操作 | 理由 |
|------|------|
| `nvidia-smi`（引数なし / 読み取りクエリ） | 状態確認のみ |
| `rdma-server.sh status`, `rdma-server.sh log` | 読み取り専用 |
| `rdma-server.sh start`（新規起動） | 既存プロセスに影響しない |
| `rdma-build.sh local` | 1号機ビルドのみ、GPU 不使用 |
| `rdma-env-check.sh` | 読み取り専用の環境チェック |

- GPU がロック中なら別タスクの実装を継続すること
- バックグラウンド実行パターン: `Bash(run_in_background=true)` + `gpu-lock.sh wait` で GPU 待ちの間に別作業を進められる

### セッションの進め方

1. `task-board.sh list` でタスク一覧確認
2. タスクを選んで実装開始
3. 実装完了 → `task-board.sh test <ID>`
4. GPU 空き確認 → `gpu-lock.sh run <test-command>`
5. 実験完了 → `task-board.sh done <ID> "結果"`

## プロジェクト目標と現在の状況

最終目標: GPUDirect RDMAを有効化し、2ノード16台のP100でGLM4.7 Q4を動作させること。

### 段階的アプローチ

上記の最終目標をいきなり達成するのは難しいため、以下のようなステップを設定して段階的に進める。

1. **Step 1** ✅: GPUDirectでないRDMA + P2P無効 + gpt-ossなどの中程度のパラメータ数のモデル
2. **Step 2** ✅: マルチノードクラスタの安定化 (gpt-oss-120b を11GPUクラスタで安定動作)
3. **Step 3** ✅: RDMA性能最適化 (性能損失50%以下、パイプライン化、非同期操作)
4. **Step 4** ✅: GPUDirect RDMA有効化 (CPU経由ステージング排除、GPU直接RDMA転送)
5. **Step 5 (最終Step)**: GPUDirect RDMA + 2ノード16台P100 + GLM4.7 Q4

### ステップ間の依存関係

```
Step 1 (完了) → Step 2 (完了) → Step 3 (完了) → Step 4 (完了) → Step 5 (11GPU事前検証完了, 16GPU待ち)
```

- Step 5 の 11GPU 事前検証は完了 (GLM-4.7 IQ2_M で pp=6.4, tg=6.8 t/s)
- 16GPU への拡張は GPU 追加待ち
- GLM-4.7 Q4 (IQ2_M より大きい) は 16GPU が必要な可能性

### モデル分割方式

- **レイヤー分割 (`-sm layer`) のみ** — P100にはNVLinkがなく、row splitでは性能が出ないことがシングルノード実験で確認済み
- RDMAバックエンドは既にレイヤー分割のみで動作している (各バックエンドインスタンス = 1リモートGPU)
- row split の実装は不要

### Step 5: 達成状況 (11GPU 事前検証)

#### 目標
GPUDirect RDMA + 2ノード16台P100で GLM4.7 Q4 を動作させる。
モデルはunslothの量子化モデル (Hugging Face) を使用予定。

#### 達成済み
- **GLM-4.7 IQ2_M が 11GPU (7C+4R) で安定動作** — 正常な推論出力を確認
- **GPUDirect RDMA タイムアウト解消** — GDR バジェットシステム (`GGML_RDMA_GDR_BUDGET_GB`) で RNIC MTT キャッシュオーバーフローを回避
- **Multi-RDMA デバイス出力破損修正** — `cpy_tensor` 無効化で `get_tensor+set_tensor` フォールバック
- **get_tensor stale data 修正** — per-buffer `mr_is_gdr` フラグで GDR MR のみ RDMA Read 許可
- **mmap + RDMA Write 修正** — 4GB per-buffer サイズ制限 + Send/Recv フォールバック

#### 達成数値 (GLM-4.7 IQ2_M, 7 CUDA + 4 RDMA, 11GPU)

| バックエンド | Prompt (t/s) | Generation (t/s) |
|-------------|:------------:|:----------------:|
| **RDMA (GDR budget 12GB)** | **6.4** | **6.8** |
| RDMA (GDR 無効) | 6.4 | 6.0 |
| RPC (TCP) | 5.4 | 7.5 |

- RDMA は Prompt 処理で RPC 比 **+19%** (RDMA Write ゼロコピーの効果)
- RPC は Generation で RDMA 比 **+10%** (デバイスごとの独立ソケットによるコマンド並列化)
- GDR 有効で Generation が GDR 無効比 **+13%** 改善

#### 前提条件
- Step 4 (GPUDirect RDMA) が動作していること ✅
- 16台のP100が利用可能であること (GPU追加後)

#### 成功基準
- GLM4.7 Q4 が16台P100で推論完了できること
- 実用的な推論速度が得られること (11GPU での IQ2_M 実績: pp=6.4, tg=6.8 t/s)

### 残タスク

1. **16GPU への拡張** (GPU 追加後)
   - 16GPU (8+8) への拡張と全 GPU での RDMA 接続確立テスト
   - GLM-4.7 Q4 (IQ2_M より大きい量子化) がVRAMに収まるか検証
   - `-sm layer` によるレイヤー分割で VRAM 分配計画を策定

2. **GLM-4.7 Q4 量子化モデルの準備**
   - unsloth GLM-4.7 Q4 のダウンロードとサイズ確認
   - IQ2_M (~40GB) では11GPUで動作したが、Q4 はより大きいため16GPU が必要な可能性

### 残課題・懸念事項

#### Generation 速度での RPC 比劣位
- RDMA は単一接続で全リモートデバイスを共有するため、graph_compute が逐次実行される
- RPC はデバイスごとに独立ソケットを持ち、コマンド送信を並列化可能
- **改善案**: RDMA 接続のデバイス分離またはパイプライン化 (未実装)

#### サーバーGPU 計算時間のセッション間変動
- RDMA (GDR 無効) の Generation 速度が 5.5-6.8 t/s と大きくばらつく
- GPU のサーマルスロットリングまたは CUDA コンテキスト初期化の影響と推定
- GDR 有効時は比較的安定 (6.6-6.9 t/s)

#### クライアント異常切断後のサーバー復旧
- クライアントがクラッシュした場合、サーバーの QP 状態が壊れることがある
- 次のテスト実行前にサーバーの再起動が必要
- シングルスレッドのサーバー設計に起因 (接続回復パスが未実装)

#### GDR バジェットのデフォルト値
- デフォルト 12GB は ConnectX-4 の MTT キャッシュ推定値 (~10-16GB) に基づく経験的な値
- より大容量の RNIC (ConnectX-6 等) では `GGML_RDMA_GDR_BUDGET_GB` を増やすことで全デバイス GDR が可能
- 現状では `GGML_RDMA_NO_GDR=1` も引き続き使用可能 (per-buffer フラグにフォールバック)

---

## GPUDirect RDMA (GDR) の有効化

### 前提条件

| 要件 | 現環境 | 備考 |
|------|--------|------|
| NVIDIA Driver R470+ | 535.288.01 | `nvidia-peermem` モジュール同梱 |
| MLNX_OFED | 24.10-1.1.4.0 | InfiniBand/RoCE ドライバ。**カーネルモジュールは 6.8.0-90-generic 向けプリコンパイル** — カーネル更新時は再インストール要 |
| ConnectX-4+ RNIC | mlx5_0 (CX-4) | FW 12.21.1000 |
| Compute Capability 3.5+ | P100 (6.0) | GPUDirect RDMA の最低要件 |
| カーネルバージョン | **6.8.0-90-generic** (固定) | GRUB で固定済み。変更禁止 (下記参照) |

> **カーネルバージョン固定 (重要)**: MLNX_OFED カーネルモジュールは `6.8.0-90-generic` 向けプリコンパイルのため、カーネルが変わると `nvidia-peermem` がロードできなくなり GDR が動作しない。2026-02-10 に `unattended-upgrade` がカーネルを `6.8.0-100` に自動更新し、2026-02-12 の再起動後に GDR が壊れた。対策:
> - 両ノードの GRUB を `6.8.0-90-generic` に固定済み
> - `unattended-upgrade` のカーネル blacklist を設定済み (`/etc/apt/apt.conf.d/50unattended-upgrades`)
> - **`apt upgrade` や `unattended-upgrade` でカーネルを更新しないこと**。更新する場合は MLNX_OFED の再インストール (`mlnxofedinstall --add-kernel-support`) が必要

### カーネルモジュール: `nvidia-peermem`

> **よくある間違い**: `nv_peer_mem` は旧名 (Driver R470 未満 + Mellanox 提供の別パッケージ)。
> Driver R470 以降は `nvidia-peermem` がドライバに同梱されており、`nv_peer_mem` は存在しない。

| 名前の使い分け | 記法 | 用途 |
|---------------|------|------|
| ハイフン区切り | `nvidia-peermem` | `modprobe`, `modinfo`, 設定ファイル |
| アンダースコア区切り | `nvidia_peermem` | `lsmod` 出力, `/sys/module/`, `/proc/modules` |

```bash
# モジュールの手動ロード
sudo modprobe nvidia-peermem

# 起動時の自動ロード設定 (両ノード)
echo 'nvidia-peermem' | sudo tee /etc/modules-load.d/nvidia-peermem.conf
```

### GDR 状態確認コマンド

```bash
# モジュールがロードされているか
lsmod | grep nvidia_peermem

# モジュールの詳細情報 (バージョン、依存関係)
modinfo nvidia-peermem

# sysfs での状態確認 (live = 正常)
cat /sys/module/nvidia_peermem/initstate

# 2号機も同様に確認
ssh 192.168.100.2 "lsmod | grep nvidia_peermem"
```

### トラブルシューティング

| 症状 | 原因 | 対処法 |
|------|------|--------|
| `modprobe nv_peer_mem` → module not found | 旧モジュール名を指定している | `modprobe nvidia-peermem` を使う |
| `modprobe nvidia-peermem` → module not found | NVIDIA ドライバが R470 未満 or DKMS 再ビルドが必要 | `nvidia-smi` でバージョン確認、`dkms status` で確認 |
| `lsmod` に `nvidia_peermem` がない | モジュール未ロード | `sudo modprobe nvidia-peermem` |
| RDMA バックエンドログに `nvidia-peermem module not loaded` | サーバー側でモジュール未ロード | 2号機でも `modprobe` 実行 |
| `ibv_reg_mr` 失敗 (GPU アドレス) | peermem 未ロード or ドライバ不整合 | `modinfo nvidia-peermem` でバージョンが `nvidia-smi` と一致するか確認 |
| RDMA Write タイムアウト (大モデル) | ConnectX-4 MTT キャッシュ溢れ | `GGML_RDMA_GDR_BUDGET_GB=12` (デフォルト) で制限 |
| `modprobe nvidia-peermem` → `Unknown symbol ib_register_peer_memory_client` | MLNX_OFED カーネルモジュールが現カーネル向けにビルドされていない。inbox `ib_uverbs` には peer memory API がない | MLNX_OFED を現カーネル向けに再インストール (`mlnxofedinstall --add-kernel-support`)、または MLNX_OFED モジュールがビルドされたカーネルで起動 |
| `modules-load.d` に設定済みだが起動後に未ロード | `systemd-modules-load` が nvidia/ib_uverbs ドライバより先に実行される | 起動後に `lsmod \| grep nvidia_peermem` で確認。未ロードなら手動で `sudo modprobe nvidia-peermem` |

> **注意**: カーネルアップデート後は必ず `lsmod | grep nvidia_peermem` で GDR モジュールが実際にロードされているか確認すること。MLNX_OFED カーネルモジュールは DKMS ではなくプリコンパイルバイナリのため、カーネルバージョンが変わると自動再ビルドされない。`nvidia-peermem` 自体は NVIDIA DKMS により再ビルドされるが、依存先の `ib_uverbs` (MLNX_OFED 版) が不在だとシンボル解決に失敗する。

### RDMA バックエンドでの GDR 設定

GDR は `nvidia_peermem` モジュールがロードされていれば**自動的に有効化**される。
コード側の検出ロジック (`ggml/src/ggml-rdma/rdma-gdr.cpp`):
1. `/sys/module/nvidia_peermem/initstate` が `live` であることを確認
2. フォールバック: `/proc/modules` に `nvidia_peermem` が含まれるか確認

手動制御:
- **無効化**: `GGML_RDMA_NO_GDR=1` (全 GPU をホストステージング経由に)
- **バジェット調整**: `GGML_RDMA_GDR_BUDGET_GB=12` (デフォルト、ConnectX-4 向け)
  - ConnectX-6+ では `50` 等に増やすことで全デバイス GDR 化が可能

---

## ビルド・デプロイ・実行手順

> **注意**: SSH/rsync コマンドは `settings.local.json` に `Bash(ssh *)` 等を含めれば直接実行可能。
> ただし、ビルド・デプロイ・サーバー管理にはラッパースクリプト (`scripts/rdma-*.sh`) を推奨
> （エラーハンドリング・ログ管理が組み込まれているため）。
> 別ワークツリーから呼ぶ場合は絶対パスを使う: `bash /path/to/worktree/scripts/rdma-build.sh local`

### 1号機 (192.168.100.1) でビルド

```bash
bash scripts/rdma-build.sh local
```

### 2号機 (192.168.100.2) へのデプロイ (コード転送 + ビルド)

```bash
bash scripts/rdma-deploy.sh
```

### 両ノードでビルド (1号機ビルド + 2号機デプロイ)

```bash
bash scripts/rdma-build.sh local && bash scripts/rdma-deploy.sh
```

### rdma-server 管理 (2号機)

```bash
bash scripts/rdma-server.sh start    # 起動
bash scripts/rdma-server.sh stop     # 停止
bash scripts/rdma-server.sh restart  # 再起動
bash scripts/rdma-server.sh status   # 状態確認
bash scripts/rdma-server.sh log      # ログ表示
```

> **ビルドコマンドに関する注意:**
> - `--config Release` は Unix Makefiles ジェネレータでは無視されるため不要（`CMAKE_BUILD_TYPE=Release` は cmake configure 時に自動設定される）
> - `cmake --build build` の代わりに `make -C build -j$(nproc)` でも同等に動作する
> - ビルドログは `/tmp/cmake_configure.log` (configure) と `/tmp/build.log` (build) に出力される

### llama-bench 実行 (1号機)

**qwen2.5-0.5b 2GPU (CUDA0 + RDMA0)**
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

**gpt-oss-20b 2GPU (CUDA0 + RDMA0)**
```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
  build/bin/llama-bench \
  -m /home/ubuntu/models/gpt-oss-20b-Q4_K_M.gguf \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

#### マルチファイルGGUFの使用

`llama-bench` は `-hf` フラグをサポートしていないため、HuggingFaceキャッシュ内のマルチファイルGGUF（gpt-oss-120bなど）を直接指定するとスプリットファイルの検出に失敗する。

**回避策**: 標準的なファイル名でシンボリックリンクを作成する

```bash
mkdir -p /tmp/gpt-oss-120b
ln -sf /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-120b-GGUF_Q4_K_M_gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
       /tmp/gpt-oss-120b/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf
ln -sf /home/ubuntu/.cache/llama.cpp/unsloth_gpt-oss-120b-GGUF_Q4_K_M_gpt-oss-120b-Q4_K_M-00002-of-00002.gguf \
       /tmp/gpt-oss-120b/gpt-oss-120b-Q4_K_M-00002-of-00002.gguf
```

#### `-dev` オプションのセパレータ

- **llama-cli**: カンマ区切り `,` を使用 (例: `CUDA0,CUDA1,RDMA0`)
- **llama-bench**: スラッシュ区切り `/` を使用 (例: `CUDA0/CUDA1/RDMA0`)

```bash
llama-bench -m /tmp/gpt-oss-120b/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  -dev 'CUDA0/CUDA1/CUDA2/CUDA3/CUDA4/CUDA5/CUDA6/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]/RDMA2[192.168.100.2:50051]/RDMA3[192.168.100.2:50051]' \
  -ngl 999 -sm layer -r 1 -p 128 -n 32
```

### llama-cli 実行 (1号機, 11GPU クラスタ)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 \
  LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -hf unsloth/gpt-oss-120b-GGUF:Q4_K_M \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -p 'こんにちは' -n 50 \
  --no-warmup --single-turn --simple-io \
  --log-file /tmp/llama-cli.log
```

### 環境変数一覧

| 環境変数 | 説明 | デフォルト |
|---------|------|----------|
| `GGML_RDMA_SERVERS` | RDMA サーバーリスト (host:port) | 未設定 |
| `GGML_RDMA_NO_GDR` | `1` で GPUDirect RDMA を無効化 | 未設定 (GDR有効) |
| `GGML_RDMA_GDR_BUDGET_GB` | GPUDirect MR 登録の合計サイズ上限 (GB)。超過分はホストステージングにフォールバック | 12 |
| `GGML_RDMA_NO_STAGING` | `1` でホストステージングバッファを無効化 (Send/Recvフォールバック) | 未設定 |
| `GGML_RDMA_PROFILE` | `1` でクライアント側プロファイリング有効化 | 未設定 |
| `GGML_RDMA_DEBUG` | `1` でデバッグログ出力 | 未設定 |
| `GGML_RDMA_TIMEOUT_MS` | RDMA 操作 (Send/Recv/Write/Read) のタイムアウト (ms) | 30000 |
| `GGML_RDMA_COMPUTE_TIMEOUT_MS` | graph_compute 応答待ちのタイムアウト (ms) | 300000 |
| `GGML_RDMA_ASYNC_COMPUTE` | `0` で graph_compute の fire-and-forget を無効化 (デバッグ用) | 未設定 (有効) |

### よくあるエラーと対処法

| エラー | 原因 | 対処法 |
|-------|------|--------|
| `CUDA illegal memory access` | `supports_buft` のバグ (修正済み) またはクロスデバイスアクセス | コードが最新か確認。2号機のバイナリが古い可能性 |
| `Connection refused` | rdma-server が起動していない | 2号機で `ps aux \| grep rdma` 確認、サーバー再起動 |
| `RDMA write completion timeout` | ステージングバッファの MR 情報不一致 | サーバーを再起動してバッファ再登録 |
| `chunk recv` ログ大量出力 | 大きなテンソル (>16MB) の転送 | 正常動作。MoEモデルで頻発 |
| `llama_params_fit` クラッシュ | スレッド安全性の問題 (修正済み) | `op_mutex_` による保護が有効か確認 |

### GDB デバッグ

printf デバッグの代替として gdb (v15.0.50) を使用できる。Debug ビルドと RDMA 固有の gdb ヘルパーを提供。

#### Debug ビルド

```bash
bash scripts/rdma-build.sh local debug
bash scripts/rdma-deploy.sh debug
```

Debug ビルドは `-O0 -g3` で最適化を無効にし、全てのデバッグシンボルを含む。ステップ実行や変数検査に最適だが、実行速度は大幅に低下する。

> **重要**: Debug ビルドでは RDMA タイムアウトを延長すること。
> `GGML_RDMA_TIMEOUT_MS=120000` (2分) を推奨。デフォルト30秒ではブレークポイント停止中にタイムアウトする。

#### クライアント側デバッグ (1号機)

```bash
GGML_RDMA_SERVERS=192.168.100.2:50051 GGML_RDMA_TIMEOUT_MS=120000 \
  bash scripts/rdma-debug.sh cli \
  -m /home/ubuntu/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -ngl 999 -sm layer -c 256 -n 10 -p 'hello' \
  --no-warmup --single-turn --simple-io
```

#### サーバー側デバッグ (2号機)

```bash
bash scripts/rdma-server.sh debug     # gdb 付きフォアグラウンド起動
bash scripts/rdma-server.sh attach    # 実行中プロセスにアタッチ
```

> **注意**: `debug` と `attach` は `ssh -t` (対話的ターミナル) を使用するため、Claude Code の自動承認は効かない。開発者がターミナルで直接実行する。

#### gdb コマンド一覧

| コマンド | 説明 |
|---------|------|
| `rdma-conn <ptr>` | `rdma_connection` の状態表示 (endpoint, QP, stats) |
| `rdma-buf <ptr>` | バッファコンテキスト (remote_ptr, mr_rkey, GDR, size) |
| `rdma-ctx <ptr>` | バックエンドコンテキスト (endpoint, device, conn) |
| `rdma-wc <ptr>` | Work Completion エントリ (status, vendor_err) |
| `rdma-mr <ptr>` | Memory Region (addr, length, lkey, rkey) |
| `rdma-cmd-name <id>` | コマンドID→名前変換 (0=ALLOC_BUFFER, 9=GRAPH_COMPUTE, ...) |
| `rdma-threads` | スレッド一覧 + RDMA スレッドのヒント |

#### ブレークポイントプリセット

| コマンド | ブレークポイント数 | 対象 |
|---------|:--:|------|
| `rdma-bp-errors` | 5 | Send/Recv/Write/Read タイムアウト + WC エラー |
| `rdma-bp-commands` | 1 | サーバーコマンドディスパッチ (トレース出力付き) |
| `rdma-bp-compute` | 4 | GRAPH_COMPUTE 関連コマンドのみ |

#### 典型的なデバッグワークフロー

1. Debug ビルド: `bash scripts/rdma-build.sh local debug && bash scripts/rdma-deploy.sh debug`
2. サーバー gdb 起動: `bash scripts/rdma-server.sh debug` → gdb で `rdma-bp-errors` → `run`
3. クライアント gdb 起動: `bash scripts/rdma-debug.sh cli ...` → gdb で `rdma-bp-errors` → `run`
4. エラー発生時に停止 → `rdma-conn`, `rdma-wc` で状態確認

---

## 検証方法

- 各ステップで `llama-bench` または `llama-cli` によるベンチマーク実行
- レポートは `report/` ディレクトリに `REPORT.md` のフォーマットに従って記録
- 性能数値は Prompt t/s と Generation t/s の両方を計測
- 実験前に `bash scripts/rdma-env-check.sh` を実行し、警告がないことを確認する
- レポート作成時は `bash scripts/rdma-env-check.sh --markdown` の出力を含める

### 11GPU クラスタテストの必須ルール

- **最終テストには必ず GLM-4.7 (IQ2_M) を使用すること**
- 作業中の動作確認や回帰テストで小さいモデル (qwen2.5-0.5b, gpt-oss-20b 等) を使うのは OK
- ただし、変更の最終検証は必ず GLM-4.7 を 11GPU (7C+4R) 構成で実行し、正常な推論出力と性能を確認すること
- GLM-4.7 テストコマンド:

```bash
# サーバー起動 (2号機)
ssh 192.168.100.2 "LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin \
  nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 \
  > /tmp/rdma-server.log 2>&1 &"

# 推論 (1号機)
GGML_RDMA_SERVERS=192.168.100.2:50051 LD_LIBRARY_PATH=build/bin \
  build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -dev 'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051],RDMA2[192.168.100.2:50051],RDMA3[192.168.100.2:50051]' \
  -sm layer -ngl 999 -c 2048 -n 50 --seed 42 \
  -p 'The capital of France is' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

- 期待値: Prompt ≈ 6.4 t/s, Generation ≈ 6.8 t/s (GDR 有効時)

### 定量評価の統計手法

A/B 性能比較（例: 最適化 ON/OFF、バックエンド間比較）を行う際は、以下の手法に従う。
単一構成のスポット計測（動作確認、回帰テスト等）には適用不要。

#### 1. 交絡因子の事前チェック

定量評価の前に、比較条件間で評価対象以外の差異がないことを確認する。

**チェックリスト**:
- [ ] **単一変数の分離**: 比較する2条件の差異が評価対象の機能のみであること
- [ ] **環境変数トグル**: コードブランチ比較ではなく、同一バイナリ + 環境変数で条件切替（ブランチ比較は複数のコード差分が交絡する）
- [ ] **ホットパスのログ出力**: `fprintf(stderr, ...)` 等の常時出力がホットパスにないこと（stdio バッファフラッシュが二峰性分布を引き起こす）
- [ ] **分布の単峰性**: パイロットラン（5-10回）で計測値をソートし、> 0.5 t/s のギャップがないことを確認（多峰性 → 根本原因調査が先）

**過去の交絡事例**:

| 事例 | 交絡因子 | 影響 | 検出方法 |
|------|---------|------|---------|
| ブランチ間比較 | deferred copy + SYNC の同時差異 | 効果の帰属が不明に | コミット差分の精査 |
| fprintf 二峰性 | ホットパスの stderr 出力 | ±18% ノイズで 1.5% の効果が隠蔽 | プロファイリング + stderr リダイレクトテスト |
| モデル依存効果 | MoE vs Dense でコピー頻度が異なる | 効果が特定モデルでのみ発現 | 複数モデルでの再現テスト |

#### 2. 実験設計: ABAB Paired Design

1. **条件トグル**: 環境変数で切り替える（ブランチ比較は避ける）
2. **交互実行**: A→B→A→B... の順で時間的ドリフト（温度、CUDA 状態）を制御
3. **ウォームアップ**: 1回目は破棄（キャッシュ・初期化の影響を排除）
4. **温度管理**: GPU 温度 70°C 超で冷却待機

#### 3. サンプル数

- 保守的 SD_diff 推定（過去計測に基づく）から必要ペア数を算出
- 最小検出効果: ベースライン性能の **0.5%**
- 目安: **n = 15 ペア**（GLM-4.7 11GPU）、**n = 20 ペア**（小規模構成）

#### 4. 統計検定

| 項目 | 手法 |
|------|------|
| **主検定** | 対応あり t 検定（paired t-test）、α = 0.05、両側 |
| **効果量** | Cohen's d（\|d\| < 0.2: 無視、0.2-0.8: 中、≥ 0.8: 大） |
| **信頼区間** | 差分の 95% CI |
| **一貫性** | 期待方向のペア数 / 全ペア数（例: 15/15 = 100%） |

#### 5. 判定基準

- **有効**: p < 0.05 **かつ** 効果 > 0.5%（両方満たす場合のみ）
- **効果なし**: p ≥ 0.05 **または** 効果 < 0.5%

#### 6. 外れ値

- IQR 法（1.5倍）で検出。フラグするが正当な理由なく除外しない

#### 7. レポート記載形式

ベンチマークレポートには以下を必ず含める:

1. **交絡チェック結果**: 上記チェックリストの確認結果
2. **生データ表**: 全ペアの A/B 計測値
3. **記述統計**: `平均 ± SD`（例: `7.763 ± 0.015`）
4. **検定結果表**:

| 指標 | 値 |
|------|:---:|
| 差分平均 | +X.XXX t/s (+X.XX%) |
| t(df) | XX.XX |
| p 値 | X.XX × 10⁻ⁿ |
| Cohen's d | X.XX (効果量の解釈) |
| 95% CI | [+X.XXX, +X.XXX] t/s |
| 全ペア正の効果 | XX/XX (XX%) |

5. **判定**: p < 0.05 かつ > 0.5% を満たすかの明示的結論

---

## Claude Code パーミッション設定の経緯

2026-02-18 の調査で判明した Claude Code のパーミッション自動承認の挙動をまとめる。
CLAUDE.md のルールセクション（マルチラインコマンド、リダイレクト、SSH 等）の根拠となる知見。

### 基本ルール — パターンマッチ仕様

- `:*` 構文は非推奨。` *`（スペース+アスタリスク）が推奨構文（両方とも動作する）
- `Bash(cmd *)` はワード境界を強制: `cmd -la` にマッチ、`cmdother` にマッチしない
- `Bash(cmd*)` はプレフィックスマッチ: `cmd -la` にも `cmdother` にもマッチ
- **`Bash(cmd *)` は引数なしの `cmd` にマッチしない**: `nvidia-smi --args` ✓、`nvidia-smi` 単体 ✗
- 環境変数付きコマンドは `Bash(GGML_RDMA_*)` のような prefix で広域許可可能

### 設定ファイルの優先順位 — 置換挙動とワークツリーの罠

- プロジェクト `settings.local.json` に `permissions.allow` があると、グローバル `settings.local.json` の `permissions.allow` は**置換される（マージされない）**
- ワークツリーでは**メインリポジトリの** `.claude/settings.local.json` が読まれる（ワークツリー側の設定ファイルは無視）
- **対策**: 必要なパターンはすべてプロジェクト設定に含めるか、グローバルにのみ記載してプロジェクトでは `permissions.allow` キーを省略する

### ファイルリダイレクト — パターンとは独立した安全機構

- `2>/dev/null`, `>/dev/null`, `2>&1` → 許可
- `2>/tmp/file.log`, `>/tmp/file.log`, `>>/tmp/file.log` → **ブロック**（パターンマッチとは独立した判定）
- **回避策**: Bash ツールの出力を直接利用する（stdout/stderr 両方キャプチャされる）

### シェルオペレータ — `&&`/`;` チェインの制限

- ビルトイン安全コマンド同士のチェインは通る: `echo "a" && echo "b"` ✓
- 非ビルトインが含まれると通らない: `cd /tmp && echo` ✗、`ls; echo` ✗
- **回避策**: チェインを避け、複数の Bash 呼び出しに分割するのが最も確実

### 専用ツール対応コマンドの制限 — ハードコード制限

- `ls`, `find`, `cat`, `head`, `tail` は `settings.local.json` に含めても **Bash で自動承認されない**
- `grep` は例外的に自動承認される
- Claude Code が専用ツール (Glob, Read, Grep, Edit) の使用を強制するハードコード制限と推定
- 実用上の影響は小さい — 専用ツールで代替可能

### その他の知見

- ユーザーがコマンド承認すると、プロジェクト設定に完全一致エントリが自動追加される
- **SSH/rsync/scp**: `Bash(ssh *)` 等をプロジェクト `settings.local.json` に含めれば自動承認される。以前「ハードコード制限」と誤認していたが、実際にはプロジェクト設定がグローバル設定を置換し、SSH パターンが欠落していたことが原因だった
- **`git -C` を使わない理由**: `Bash(git -C *)` は push 等の破壊的コマンドも許可してしまうため安全でない
- `echo`, `true`, `date`, `hostname`, `which` 等はビルトイン安全コマンドとして設定不要で自動承認

---

## 完了済みステップの記録

Step 1-4 の詳細は [HISTORY.md](HISTORY.md) を参照。
