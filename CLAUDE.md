## 必読ドキュメント

- [AGENTS.md](AGENTS.md) — 作業開始前に必ず確認すること
- [REPORT.md](REPORT.md) — レポート作成ルール
- [GPU.md](GPU.md) — GPUサーバ情報
- [HISTORY.md](HISTORY.md) — 完了済みステップ (Step 1-4) の記録

### 参照スキル（必要時にオンデマンド読み込み）

- `/build-deploy` — ビルド・デプロイ・サーバー管理手順
- `/bench` — ベンチマーク実行・検証手順・環境変数一覧・エラー対処
- `/stats` — A/B ベンチマーク統計手法
- `/gdr` — GPUDirect RDMA セットアップ・トラブルシューティング
- `/debug-rdma` — GDB デバッグ手順・コマンド一覧

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
- **レポート作成**: plan mode を使用してまとまった作業を行った場合は、完了時にレポートを作成すること。フォーマットは [REPORT.md](REPORT.md) に従う。レポートは作業ワークツリーに関わらず、常に `/home/ubuntu/projects/llama.cpp/report/` に作成する

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

> **重要**: `llama-bench`, `llama-cli` 等の GPU コマンドは**絶対に直接実行しないこと**。
> 必ず `gpu-lock.sh run` または `gpu-lock.sh wait` で包んで実行する。
> 直接実行すると他セッションのテストと競合し、両方の結果が信頼できなくなる。

**正しい例**:
```bash
gpu-lock.sh run llama-bench -m model.gguf -ngl 999 ...
gpu-lock.sh run bash scripts/rdma-server.sh restart
gpu-lock.sh wait --timeout 300 llama-bench ...
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 llama-bench -m model.gguf
```

**誤った例（禁止）**:
```bash
llama-bench -m model.gguf -ngl 999 ...          # ← gpu-lock なし
build/bin/llama-cli -m model.gguf ...            # ← gpu-lock なし
GGML_RDMA_SERVERS=... llama-bench ...            # ← 環境変数付きでも同様
```

#### ロック必須の操作

| カテゴリ | 実行方法 | 理由 |
|---------|---------|------|
| **GPU 推論** | `gpu-lock.sh run llama-bench ...` / `gpu-lock.sh run llama-cli ...` | GPU リソース競合 |
| **サーバー管理** | `gpu-lock.sh run bash scripts/rdma-server.sh stop` | 実行中クライアントの RDMA 接続が切断される |
| **デプロイ** | `gpu-lock.sh run bash scripts/rdma-deploy.sh` | 2号機のビルドディレクトリ削除・再構築。テスト中に実行すると整合性が崩れる |
| **カーネルモジュール** | `gpu-lock.sh run sudo modprobe nvidia-peermem` | GDR モジュール操作は実行中の RDMA 転送をクラッシュさせる |
| **GPU 設定変更** | `gpu-lock.sh run nvidia-smi -pl 250` | ベンチマーク計測値に影響 |

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

- **レイヤー分割 (`-sm layer`) のみ** — P100 (NVLink なし) では row split は全条件で layer split に劣る
- 1号機 (P100×7) でのベンチマーク実測値 (gpt-oss-20b Q4_K_M):
  - 2 GPU: layer が row 比 pp512 +17%, tg +14% 優位
  - 7 GPU: layer が row 比 pp512 +64%, tg +51% 優位
  - row split は GPU 数増加で性能が劣化 (負のスケーリング)、layer split は 99% 維持
- 詳細: [report/2026-02-19_163700_row_vs_layer_split_benchmark.md](report/2026-02-19_163700_row_vs_layer_split_benchmark.md)
- RDMAバックエンドは既にレイヤー分割のみで動作している (各バックエンドインスタンス = 1リモートGPU)
- row split の実装は不要

### Step 5: 達成状況 (11GPU 事前検証)

#### 目標
GPUDirect RDMA + 2ノード16台P100で GLM4.7 Q4 を動作させる。
モデルはunslothの量子化モデル (Hugging Face) を使用予定。

#### 達成済み
- **GLM-4.7 IQ2_M が 11GPU (7C+4R) で安定動作** — 正常な推論出力を確認
- **GPUDirect RDMA タイムアウト解消** — GDR バジェットシステム (`GGML_RDMA_GDR_BUDGET_GB`) で RNIC MTT キャッシュオーバーフローを回避
- **Multi-RDMA デバイス出力破損修正** — deferred copy によるサーバーローカル D2H+H2D コピーに改善 (`cpy_tensor` 再有効化済み)
- **Deferred copy 実装** — `cpy_tensor` をサーバーローカル実行に変更し、IB ネットワークラウンドトリップを排除。GLM-4.7 で tg +1.45% 改善
- **サーバー堅牢化** — マルチスレッド化により、クライアント切断後も再起動不要で次の接続を受付
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

> **注記**: 上記数値は Step 5 初期検証時点 (deferred copy マージ前) の測定値。Deferred copy マージ後のベンチマークでは tg ≈ 7.65-7.76 t/s を記録しており、Generation での RPC 比劣位は縮小している可能性がある。

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
- **改善案**: RDMA 接続のデバイス分離またはパイプライン化 (per-device connections は実装済み `GGML_RDMA_PER_DEVICE_CONN=1`、ConnectX-4 では MTT キャッシュ制限によりデフォルト無効)

#### サーバーGPU 計算時間のセッション間変動
- RDMA (GDR 無効) の Generation 速度が 5.5-6.8 t/s と大きくばらつく
- GPU のサーマルスロットリングまたは CUDA コンテキスト初期化の影響と推定
- GDR 有効時は比較的安定 (6.6-6.9 t/s)

#### GDR バジェットのデフォルト値
- デフォルト 12GB は ConnectX-4 の MTT キャッシュ推定値 (~10-16GB) に基づく経験的な値
- より大容量の RNIC (ConnectX-6 等) では `GGML_RDMA_GDR_BUDGET_GB` を増やすことで全デバイス GDR が可能
- 現状では `GGML_RDMA_NO_GDR=1` も引き続き使用可能 (per-buffer フラグにフォールバック)

GDR は `nvidia_peermem` モジュールがロードされていれば自動有効化。詳細: `/gdr` スキル参照。

---

## ビルド・デプロイ・実行手順

詳細は `/build-deploy` スキルを参照。基本コマンド:
- ビルド: `bash scripts/rdma-build.sh local`
- デプロイ: `bash scripts/rdma-deploy.sh`
- サーバー管理: `bash scripts/rdma-server.sh {start|stop|restart|status|log}`

ベンチマーク実行、環境変数一覧、エラー対処法は `/bench` スキルを参照。
Debug ビルドと gdb ヘルパーは `/debug-rdma` スキルを参照。

---

## 検証方法

- 各ステップで `llama-bench` または `llama-cli` によるベンチマーク実行
- レポートは `report/` ディレクトリに `REPORT.md` のフォーマットに従って記録
- 性能数値は Prompt t/s と Generation t/s の両方を計測
- 実験前に `bash scripts/rdma-env-check.sh` を実行し、警告がないことを確認する
- レポート作成時は `bash scripts/rdma-env-check.sh --markdown` の出力を含める
- **最終テストには必ず GLM-4.7 (IQ2_M) を 11GPU (7C+4R) で実行すること**

コマンド例・11GPU テスト手順の詳細は `/bench` スキルを参照。
A/B 性能比較には ABAB Paired Design + 対応あり t 検定を使用。詳細: `/stats` スキル参照。

---

## 完了済みステップの記録

Step 1-4 の詳細は [HISTORY.md](HISTORY.md) を参照。
