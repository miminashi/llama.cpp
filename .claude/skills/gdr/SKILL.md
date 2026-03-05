---
name: gdr
description: GPUDirect RDMA (GDR) setup, nvidia-peermem module management, and troubleshooting. Use when diagnosing GDR issues, checking peermem module status, or configuring GDR budget.
user-invocable: true
---

# GPUDirect RDMA (GDR) の有効化

## 前提条件

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

## カーネルモジュール: `nvidia-peermem`

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

## GDR 状態確認コマンド

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

## トラブルシューティング

| 症状 | 原因 | 対処法 |
|------|------|--------|
| `modprobe nv_peer_mem` → module not found | 旧モジュール名を指定している | `modprobe nvidia-peermem` を使う |
| `modprobe nvidia-peermem` → module not found | NVIDIA ドライバが R470 未満 or DKMS 再ビルドが必要 | `nvidia-smi` でバージョン確認、`dkms status` で確認 |
| `lsmod` に `nvidia_peermem` がない | モジュール未ロード | `sudo modprobe nvidia-peermem` |
| RDMA バックエンドログに `nvidia-peermem module not loaded` | サーバー側でモジュール未ロード | 2号機でも `modprobe` 実行 |
| `ibv_reg_mr` 失敗 (GPU アドレス) | peermem 未ロード or ドライバ不整合 | `modinfo nvidia-peermem` でバージョンが `nvidia-smi` と一致するか確認 |
| RDMA Write タイムアウト (大モデル) | ConnectX-4 MTT キャッシュ溢れ | `GGML_RDMA_GDR_BUDGET_GB=12` (デフォルト) で制限 |
| GDR + parallel dispatch でクラッシュ | `conn->recv()` 中に RNIC GDR Write → サーバー GPU CUDA カーネルと競合 | sync-before-recv、または `GGML_RDMA_PARALLEL_DISPATCH=0` |
| `modprobe nvidia-peermem` → `Unknown symbol ib_register_peer_memory_client` | MLNX_OFED カーネルモジュールが現カーネル向けにビルドされていない。inbox `ib_uverbs` には peer memory API がない | MLNX_OFED を現カーネル向けに再インストール (`mlnxofedinstall --add-kernel-support`)、または MLNX_OFED モジュールがビルドされたカーネルで起動 |
| `modules-load.d` に設定済みだが起動後に未ロード | `systemd-modules-load` が nvidia/ib_uverbs ドライバより先に実行される | 起動後に `lsmod \| grep nvidia_peermem` で確認。未ロードなら手動で `sudo modprobe nvidia-peermem` |

> **注意**: カーネルアップデート後は必ず `lsmod | grep nvidia_peermem` で GDR モジュールが実際にロードされているか確認すること。MLNX_OFED カーネルモジュールは DKMS ではなくプリコンパイルバイナリのため、カーネルバージョンが変わると自動再ビルドされない。`nvidia-peermem` 自体は NVIDIA DKMS により再ビルドされるが、依存先の `ib_uverbs` (MLNX_OFED 版) が不在だとシンボル解決に失敗する。

## RDMA バックエンドでの GDR 設定

GDR は `nvidia_peermem` モジュールがロードされていれば**自動的に有効化**される。
コード側の検出ロジック (`ggml/src/ggml-rdma/rdma-gdr.cpp`):
1. `/sys/module/nvidia_peermem/initstate` が `live` であることを確認
2. フォールバック: `/proc/modules` に `nvidia_peermem` が含まれるか確認

手動制御:
- **無効化**: `GGML_RDMA_NO_GDR=1` (全 GPU をホストステージング経由に)
- **バジェット調整**: `GGML_RDMA_GDR_BUDGET_GB=12` (デフォルト、ConnectX-4 向け)
  - ConnectX-6+ では `50` 等に増やすことで全デバイス GDR 化が可能
