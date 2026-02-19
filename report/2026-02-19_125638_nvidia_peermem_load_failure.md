# nvidia-peermem モジュールロード失敗の調査

- **実施日時**: 2026年2月19日 12:56
- **ワークツリー**: `.worktree/rdma-backend`

## 前提・目的

両ノードで `nvidia-peermem` モジュールがロードされていないことが判明し、`sudo modprobe nvidia-peermem` によるロードを試行する。GDR (GPUDirect RDMA) を有効化するには、このモジュールがロードされている必要がある。

- **背景**: `rdma-env-check.sh` が両ノードで `nvidia-peermem: not_loaded` を報告
- **目的**: モジュールのロードを試行し、失敗時は根本原因を特定する
- **前提条件**: `/etc/modules-load.d/nvidia-peermem.conf` は設定済み

## 調査結果

### modprobe の実行結果

```
$ sudo modprobe nvidia-peermem
modprobe: ERROR: could not insert 'nvidia_peermem': Unknown symbol in module, or unknown parameter (see dmesg)
```

### dmesg のエラーログ

```
nvidia_peermem: Unknown symbol ib_register_peer_memory_client (err -2)
nvidia_peermem: Unknown symbol ib_unregister_peer_memory_client (err -2)
```

### 根本原因: カーネルバージョンと MLNX_OFED モジュールの不一致

| 項目 | 値 |
|------|-----|
| 実行中カーネル | `6.8.0-100-generic` |
| MLNX_OFED カーネルモジュール | `6.8.0-90-generic` 向けにプリコンパイル |
| NVIDIA DKMS | `6.8.0-100-generic` 向けに再ビルド済み |

**詳細**: カーネルが `6.8.0-90` → `6.8.0-100` にアップデートされた際に:

1. **NVIDIA DKMS** (`nvidia-peermem` 含む): DKMS により自動再ビルド → `6.8.0-100` 向けモジュールあり
2. **MLNX_OFED カーネルモジュール** (`ib_core`, `ib_uverbs` 等): プリコンパイルバイナリ (DKMS ではない) → `6.8.0-90` 向けのみ、`6.8.0-100` 向けなし

結果、`6.8.0-100` カーネルでは inbox (Ubuntu 標準) の `ib_core`/`ib_uverbs` が使われるが、これらには MLNX_OFED 固有の `ib_register_peer_memory_client` / `ib_unregister_peer_memory_client` シンボルが含まれない。`nvidia-peermem` はこれらのシンボルに依存するため、ロードに失敗する。

### モジュール存在状況

| モジュール | 6.8.0-100 (現カーネル) | 6.8.0-90 (旧カーネル) |
|-----------|:---:|:---:|
| `ib_core` (inbox) | あり | あり |
| `ib_core` (MLNX_OFED) | **なし** | あり (`updates/dkms/ib_core.ko`) |
| `ib_uverbs` (inbox) | あり | あり |
| `ib_uverbs` (MLNX_OFED) | **なし** | あり (`updates/dkms/ib_uverbs.ko`) |
| `nvidia-peermem` (DKMS) | あり | あり |
| `mlx5_core` (inbox) | あり | あり |
| `mlx5_core` (MLNX_OFED) | **なし** | あり |

### シンボル提供元の確認

```
$ grep 'ib_register_peer_memory_client' /lib/modules/6.8.0-90-generic/modules.symbols
alias symbol:ib_register_peer_memory_client ib_uverbs

$ grep 'ib_register_peer_memory_client' /lib/modules/6.8.0-100-generic/modules.symbols
(出力なし)
```

MLNX_OFED 版 `ib_uverbs` がシンボルを提供しており、inbox 版には存在しない。

### 2号機の状態

1号機と同一。カーネル `6.8.0-100-generic`、MLNX_OFED モジュールは `6.8.0-90` 向けのみ。

## 解決策

### 方法 1: MLNX_OFED 再インストール (推奨)

```bash
./mlnxofedinstall --add-kernel-support --without-dkms
```

MLNX_OFED インストーラ (ISO/tar) を取得して、現カーネル向けにカーネルモジュールを再ビルドする。ただし、**インストーラがシステム上に残っていない**ため、Mellanox のサイトからダウンロードが必要。

### 方法 2: 旧カーネルで起動

`6.8.0-90-generic` カーネルはまだインストールされているため、GRUB で旧カーネルを選択して起動すれば MLNX_OFED モジュールが使われる。

```bash
# GRUB でカーネル選択 (手動操作)
# Advanced options for Ubuntu → Ubuntu, with Linux 6.8.0-90-generic
```

**注意**: カーネルダウングレードはセキュリティパッチの喪失を伴う。

### 方法 3: MLNX_OFED DKMS パッケージの導入

MLNX_OFED を DKMS モードでインストールし直せば、将来のカーネルアップデートでも自動再ビルドが行われる。

```bash
./mlnxofedinstall --add-kernel-support --dkms
```

## CLAUDE.md への修正

以下を追記:

1. **前提条件テーブル**: MLNX_OFED がプリコンパイルバイナリであり、カーネル更新時は再インストールが必要な旨を追記
2. **トラブルシューティングテーブル**: `Unknown symbol ib_register_peer_memory_client` エラーの行を追加
3. **注意事項**: `modules-load.d` 設定があってもロードされないケース、カーネルアップデート後の確認手順を追記

## 再現方法

```bash
# 1. 1号機でモジュールロードを試行
sudo modprobe nvidia-peermem
# → ERROR: Unknown symbol ...

# 2. dmesg で詳細確認
sudo dmesg | tail -10
# → nvidia_peermem: Unknown symbol ib_register_peer_memory_client (err -2)

# 3. カーネルバージョン確認
uname -r
# → 6.8.0-100-generic

# 4. MLNX_OFED パッケージのカーネルバージョン確認
dpkg -l mlnx-ofed-kernel-modules | tail -1
# → 24.10.OFED.24.10.1.1.4.1-1.kver.6.8.0-90-generic

# 5. 現カーネルに MLNX_OFED ib_uverbs が存在しないことを確認
find /lib/modules/$(uname -r)/updates -name 'ib_uverbs*'
# → (出力なし)
```

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `5dea2e773 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 33°C | 36°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | not_loaded | not_loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 444838) |

GGML_RDMA 環境変数:
- (none)

**警告:**
- 1号機 nvidia-peermem が未ロード
- 2号機 nvidia-peermem が未ロード
