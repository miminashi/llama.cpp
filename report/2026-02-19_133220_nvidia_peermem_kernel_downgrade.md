# nvidia-peermem 復旧: カーネルダウングレード + 再発防止

- **実施日時**: 2026年2月19日 13:32
- **ワークツリー**: `.worktree/rdma-backend`

## 前提・目的

GPUDirect RDMA (GDR) が動作しなくなった原因を特定し、復旧する。

- **背景**: 2026-02-12 以降、`nvidia-peermem` モジュールがロードできず GDR が無効化されていた
- **目的**: カーネルダウングレードによる GDR 復旧、および再発防止策の適用
- **前提条件**: 両ノードで MLNX_OFED 24.10-1.1.4.0 が `6.8.0-90-generic` 向けにインストール済み

### 参照レポート

- [nvidia_peermem ロード失敗調査](2026-02-19_125638_nvidia_peermem_load_failure.md)

## 原因のタイムライン

| 日付 | イベント |
|------|---------|
| 2026-02-05 | GDR 正常動作確認 (カーネル: `6.8.0-90-generic`) |
| 2026-02-10 | `unattended-upgrade` がカーネルを `6.8.0-90` → `6.8.0-100` に自動更新 |
| 2026-02-12 | サーバー再起動 → 新カーネル `6.8.0-100` で起動 |
| 2026-02-12〜 | `nvidia-peermem` ロード不可 → GDR 無効 |
| 2026-02-19 | 原因特定 + カーネルダウングレード実施 |

### 根本原因

MLNX_OFED カーネルモジュール (`ib_uverbs`, `mlx5_ib` 等) は DKMS ではなく**プリコンパイルバイナリ**として `6.8.0-90-generic` 向けにのみ提供される。カーネルが `6.8.0-100` に変わると:

1. MLNX_OFED 版 `ib_uverbs` がロードされず、inbox (Ubuntu 標準) 版 `ib_uverbs` がロードされる
2. inbox `ib_uverbs` には `ib_register_peer_memory_client` シンボルがない
3. `nvidia-peermem` の `modprobe` が `Unknown symbol` エラーで失敗

## 復旧手順

### 1. GRUB デフォルトカーネルの変更

**2号機** (UUID: `49d668ea-d28d-4fef-877c-82859ee9ad68`):
```bash
sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT="gnulinux-advanced-49d668ea-d28d-4fef-877c-82859ee9ad68>gnulinux-6.8.0-90-generic-advanced-49d668ea-d28d-4fef-877c-82859ee9ad68"/' /etc/default/grub
sudo update-grub
sudo reboot
```

**1号機** (UUID: `53c30741-2819-4592-9cc4-79fe13875f29`):
```bash
sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT="gnulinux-advanced-53c30741-2819-4592-9cc4-79fe13875f29>gnulinux-6.8.0-90-generic-advanced-53c30741-2819-4592-9cc4-79fe13875f29"/' /etc/default/grub
sudo update-grub
sudo reboot
```

### 2. 復旧確認

```bash
uname -r                                     # → 6.8.0-90-generic
lsmod | grep nvidia_peermem                  # → nvidia_peermem ロード済み
cat /sys/module/nvidia_peermem/initstate     # → live
```

### 3. 再発防止: カーネル自動更新の除外

両ノードの `/etc/apt/apt.conf.d/50unattended-upgrades` の `Package-Blacklist` セクションに追加:

```
    // Prevent kernel updates from breaking MLNX_OFED / nvidia-peermem (GDR)
    "linux-image-.*";
    "linux-headers-.*";
    "linux-modules-.*";
```

## 実施結果

| 項目 | 2号機 | 1号機 |
|------|:-----:|:-----:|
| GRUB 変更 | ✅ | ✅ |
| reboot | ✅ 実施済み | ⏳ ユーザーが手動実施 |
| `uname -r` = `6.8.0-90-generic` | ✅ | ⏳ reboot 後 |
| `nvidia_peermem` ロード | ✅ | ⏳ reboot 後 |
| `initstate` = `live` | ✅ | ⏳ reboot 後 |
| カーネル blacklist 設定 | ✅ | ✅ |
| CLAUDE.md 更新 | ✅ | — |

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `5dea2e773 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 33°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Disabled |
| nvidia-peermem | not_loaded (reboot 前) | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| カーネル | 6.8.0-100 (reboot 前) | 6.8.0-90 (復旧済み) |

> 1号機は reboot 前のため `nvidia-peermem` が `not_loaded`。reboot 後に `loaded` になる見込み。
