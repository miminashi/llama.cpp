# GitHub Push 環境セットアップ (Deploy Key)

- **実施日時**: 2026年2月9日 08:49

## 前提・目的

feature/rdma-backend ブランチの成果を GitHub にプッシュするための環境を構築する。

- **背景**: 現在 origin は `ggml-org/llama.cpp` (upstream) を指しており push 権限がない
- **目的**: ユーザーのフォークリポジトリに対して Deploy Key 方式で push できるようにする
- **前提条件**:
  - SSH鍵 `~/.ssh/id_ed25519` が存在する
  - 公開鍵: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIZBPHuQXFi1gzeduWz705eVIq4G2X6bDRjXRanjDuZ7 ubuntu@chungpu`
  - GitHub にフォーク済みのリポジトリがある
  - `gh` CLI はインストール済みだが未認証

## 手順

### Step 1: GitHub に Deploy Key を登録 (ユーザー手動操作)

1. フォークしたリポジトリの GitHub ページを開く
2. **Settings** → **Deploy keys** → **Add deploy key**
3. Title: `gpu-server` (任意)
4. Key: 上記の公開鍵を貼り付け
5. **Allow write access** にチェックを入れる
6. **Add key** をクリック

> Deploy Key は登録したリポジトリにのみスコープされるため、他のリポジトリには一切アクセスできない。

### Step 2: SSH 接続テスト

```bash
ssh -T git@github.com
```

`Hi <username>/<repo>! ...` のようなメッセージが出れば成功。

### Step 3: フォークをリモートに追加

```bash
git remote add fork git@github.com:<username>/llama.cpp.git
git remote -v
```

出力例:
```
fork    git@github.com:<username>/llama.cpp.git (fetch)
fork    git@github.com:<username>/llama.cpp.git (push)
origin  https://github.com/ggml-org/llama.cpp (fetch)
origin  https://github.com/ggml-org/llama.cpp (push)
```

### Step 4: ブランチを push

```bash
git push fork feature/rdma-backend
```

## 補足: Deploy Key が既に他のリポジトリに登録されている場合

GitHub は同じ SSH 鍵を複数リポジトリの Deploy Key として登録できない。
その場合は専用鍵を生成し、`~/.ssh/config` で使い分ける:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_github_llama -C "llama.cpp deploy key" -N ""
```

`~/.ssh/config` に追加:
```
Host github-llama
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_github_llama
  IdentitiesOnly yes
```

リモート追加時に `github-llama` を使う:
```bash
git remote add fork git@github-llama:<username>/llama.cpp.git
```

## テスト手順

```bash
ssh -T git@github.com                         # 接続確認
git push fork feature/rdma-backend --dry-run   # dry-run で確認
git push fork feature/rdma-backend             # 本番 push
```
