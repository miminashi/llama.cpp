# init.sh 設定のOS起動時永続化マニュアル

- **実施日時**: 2026年2月6日 20:29

## 前提・目的

### 背景

現在、両ノード (1号機: 192.168.100.1 / 2号機: 192.168.100.2) ではOS起動後に `~/init.sh` を手動実行してRDMA/GPU環境を初期化している。rebootのたびに手動実行が必要であり、忘れると RDMA クラスタが機能しない。

### 目的

`init.sh` の各設定を Linux の正規メカニズム (systemd / netplan) で永続化し、OS起動時に自動適用されるようにする。

### 前提条件

- OS: Ubuntu 24.04.3 LTS (両ノード共通)
- MFT: 4.30.1-8 (`/etc/init.d/mst` 存在、systemd SysV ラッパー利用可能)
- mlxlink: `/usr/bin/mlxlink`
- MST デバイスパス: `/dev/mst/mt4115_pciconf0` (両ノード共通)
- NVIDIA ドライバ: 535.288.01, `nvidia-persistenced` は `static` で有効化済み
- 1号機: GPU 7台, IB デバイス `enp11s0np0`
- 2号機: GPU 4台, IB デバイス `enp4s0np0`

### init.sh の現状

**1号機** (`/home/ubuntu/init.sh`):
```bash
#!/bin/sh
sudo mst start
sleep 1
sudo mlxlink -d mlx5_0 --speeds 100G --link_mode_force
sudo mlxlink -d mlx5_0 --fec RS --fec_speed 100G
sudo mlxlink -d mlx5_0 --port_state UP
sleep 5
dev="$(ls /sys/class/infiniband/mlx5_0/device/net/)"
sudo ip addr add 192.168.100.1/24 dev "${dev}"
sudo ip link set "${dev}" up mtu 9000
sudo nvidia-smi -pl 180
```

**2号機** (`/home/ubuntu/init.sh`):
差分は IP アドレス (`192.168.100.2/24`) のみ。IB デバイス名は `mlx5_0` で共通だが、ネットワークインタフェース名が異なる (`enp4s0np0`)。

---

## 現状分析

### init.sh の各行の役割

| # | コマンド | 役割 | 備考 |
|---|---------|------|------|
| 1 | `sudo mst start` | Mellanox Software Tools のデバイスファイル作成 (`/dev/mst/*`) | mlxlink が依存 |
| 2 | `sudo mlxlink -d mlx5_0 --speeds 100G --link_mode_force` | リンク速度を 100Gbps に強制設定 | Auto Negotiation を無効化 |
| 3 | `sudo mlxlink -d mlx5_0 --fec RS --fec_speed 100G` | Forward Error Correction を RS(528,514) に設定 | 100G ケーブルで必須 |
| 4 | `sudo mlxlink -d mlx5_0 --port_state UP` | ポートを LinkUp 状態にする | |
| 5 | `sudo ip addr add 192.168.100.X/24 dev <dev>` | IB インタフェースに IP アドレスを付与 | |
| 6 | `sudo ip link set <dev> up mtu 9000` | インタフェースを UP にし MTU 9000 (Jumbo Frame) を設定 | |
| 7 | `sudo nvidia-smi -pl 180` | 全 GPU のパワーリミットを 180W に設定 (デフォルト 250W) | 電源容量制約のため |

### 現在の systemd 状態

| サービス | 1号機 | 2号機 |
|---------|-------|-------|
| `mst.service` | disabled | disabled |
| `nvidia-persistenced` | static (有効) | static (有効) |

### 現在の netplan 構成

両ノードとも `/etc/netplan/50-cloud-init.yaml` のみ存在。IB インタフェースの設定は含まれていない。

---

## 設定手順 (1号機: 192.168.100.1)

### 設定 1: MST サービスの有効化

`/etc/init.d/mst` が既に存在しており、systemd の SysV 互換ラッパーが利用可能。

```bash
sudo systemctl enable mst
```

確認:
```bash
systemctl is-enabled mst
# → enabled
```

### 設定 2: mlxlink 設定 (カスタム systemd service)

MST 起動後にリンク設定を適用する oneshot サービスを作成する。

```bash
sudo tee /etc/systemd/system/mlxlink-config.service << 'EOF'
[Unit]
Description=Configure Mellanox ConnectX-4 link (100G / RS-FEC)
After=mst.service
Requires=mst.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/mlxlink -d mlx5_0 --speeds 100G --link_mode_force
ExecStart=/usr/bin/mlxlink -d mlx5_0 --fec RS --fec_speed 100G
ExecStart=/usr/bin/mlxlink -d mlx5_0 --port_state UP

[Install]
WantedBy=multi-user.target
EOF
```

有効化:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mlxlink-config.service
```

### 設定 3: IB ネットワーク (netplan)

cloud-init の `50-cloud-init.yaml` には触れず、別ファイルで IB インタフェースを設定する。

```bash
sudo tee /etc/netplan/60-ib-rdma.yaml << 'EOF'
network:
  version: 2
  ethernets:
    enp11s0np0:
      addresses:
        - 192.168.100.1/24
      mtu: 9000
EOF
```

適用テスト (reboot 不要で即時確認可能):
```bash
sudo netplan apply
ip addr show enp11s0np0
```

### 設定 4: NVIDIA パワーリミット (カスタム systemd service)

```bash
sudo tee /etc/systemd/system/nvidia-power-limit.service << 'EOF'
[Unit]
Description=Set NVIDIA GPU power limit to 180W
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nvidia-smi -pl 180

[Install]
WantedBy=multi-user.target
EOF
```

有効化:
```bash
sudo systemctl daemon-reload
sudo systemctl enable nvidia-power-limit.service
```

### 1号機の全コマンドまとめ

以下を順に実行する:

```bash
# 1. MST サービス有効化
sudo systemctl enable mst

# 2. mlxlink 設定サービス作成・有効化
sudo tee /etc/systemd/system/mlxlink-config.service << 'EOF'
[Unit]
Description=Configure Mellanox ConnectX-4 link (100G / RS-FEC)
After=mst.service
Requires=mst.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/mlxlink -d mlx5_0 --speeds 100G --link_mode_force
ExecStart=/usr/bin/mlxlink -d mlx5_0 --fec RS --fec_speed 100G
ExecStart=/usr/bin/mlxlink -d mlx5_0 --port_state UP

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlxlink-config.service

# 3. IB ネットワーク (netplan)
sudo tee /etc/netplan/60-ib-rdma.yaml << 'EOF'
network:
  version: 2
  ethernets:
    enp11s0np0:
      addresses:
        - 192.168.100.1/24
      mtu: 9000
EOF

sudo netplan apply

# 4. NVIDIA パワーリミットサービス作成・有効化
sudo tee /etc/systemd/system/nvidia-power-limit.service << 'EOF'
[Unit]
Description=Set NVIDIA GPU power limit to 180W
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nvidia-smi -pl 180

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nvidia-power-limit.service
```

---

## 設定手順 (2号機: 192.168.100.2)

1号機との差分は **netplan の IB インタフェース名と IP アドレスのみ**。

### 差分一覧

| 設定 | 1号機 | 2号機 |
|------|-------|-------|
| MST | 同一 | 同一 |
| mlxlink | 同一 | 同一 |
| netplan インタフェース名 | `enp11s0np0` | `enp4s0np0` |
| netplan IP アドレス | `192.168.100.1/24` | `192.168.100.2/24` |
| NVIDIA パワーリミット | 同一 | 同一 |

### 2号機の全コマンドまとめ

```bash
ssh 192.168.100.2
```

```bash
# 1. MST サービス有効化
sudo systemctl enable mst

# 2. mlxlink 設定サービス作成・有効化
sudo tee /etc/systemd/system/mlxlink-config.service << 'EOF'
[Unit]
Description=Configure Mellanox ConnectX-4 link (100G / RS-FEC)
After=mst.service
Requires=mst.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/mlxlink -d mlx5_0 --speeds 100G --link_mode_force
ExecStart=/usr/bin/mlxlink -d mlx5_0 --fec RS --fec_speed 100G
ExecStart=/usr/bin/mlxlink -d mlx5_0 --port_state UP

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlxlink-config.service

# 3. IB ネットワーク (netplan) ★ IP とインタフェース名が異なる
sudo tee /etc/netplan/60-ib-rdma.yaml << 'EOF'
network:
  version: 2
  ethernets:
    enp4s0np0:
      addresses:
        - 192.168.100.2/24
      mtu: 9000
EOF

sudo netplan apply

# 4. NVIDIA パワーリミットサービス作成・有効化
sudo tee /etc/systemd/system/nvidia-power-limit.service << 'EOF'
[Unit]
Description=Set NVIDIA GPU power limit to 180W
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nvidia-smi -pl 180

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nvidia-power-limit.service
```

---

## 起動順序の依存関係

```
                  sysinit.target
                       │
                       ▼
              ┌─── mst.service ───┐
              │    (mst start)    │
              │                   │
              ▼                   │
    mlxlink-config.service        │
    (100G / RS-FEC / UP)          │
              │                   │
              ▼                   │
     netplan (networkd)           │
    (IP / MTU 9000)               │
                                  │
                                  │
       nvidia-persistenced ◄──────┘ (独立、既に有効)
              │
              ▼
    nvidia-power-limit.service
        (nvidia-smi -pl 180)
```

### 依存関係の説明

- `mst.service` → `mlxlink-config.service`: mlxlink は `/dev/mst/*` デバイスに依存
- `mlxlink-config.service` → netplan: リンクが UP になっていないと IP 設定が意味をなさない (ただし netplan 自体は networkd 経由で独立動作)
- `nvidia-persistenced` → `nvidia-power-limit.service`: GPU デバイスが初期化済みである必要がある
- mlxlink 系と NVIDIA 系は互いに独立 (並列起動可能)

---

## 検証手順

reboot 後に以下のコマンドで各設定が適用されていることを確認する。

### 両ノード共通

```bash
# 1. MST デバイスの存在確認
sudo mst status
# 期待: /dev/mst/mt4115_pciconf0 が表示される

# 2. リンク状態の確認
sudo mlxlink -d mlx5_0
# 期待: State=Active, Speed=100GbE, FEC=Standard RS-FEC, Physical state=LinkUp

# 3. IP アドレスと MTU の確認
ip addr show <dev>
# <dev> = enp11s0np0 (1号機) / enp4s0np0 (2号機)
# 期待: inet 192.168.100.X/24, mtu 9000

# 4. GPU パワーリミットの確認
nvidia-smi -q -d POWER | grep "Current Power Limit"
# 期待: 180.00 W

# 5. ノード間疎通確認
ping -c 3 192.168.100.1  # 2号機から
ping -c 3 192.168.100.2  # 1号機から

# 6. systemd サービス状態の確認
systemctl status mst.service
systemctl status mlxlink-config.service
systemctl status nvidia-power-limit.service
# 期待: 全て active (exited) — oneshot の正常終了状態
```

### 全設定の一括確認スクリプト

```bash
#!/bin/bash
echo "=== 永続化設定の検証 ==="
echo ""

echo "[1/5] MST デバイス"
if [ -e /dev/mst/mt4115_pciconf0 ]; then
    echo "  OK: /dev/mst/mt4115_pciconf0 存在"
else
    echo "  NG: MST デバイスが見つかりません"
fi
echo ""

echo "[2/5] リンク状態"
speed=$(sudo mlxlink -d mlx5_0 2>/dev/null | grep "^Speed" | awk '{print $NF}')
if [ "$speed" = "100GbE" ]; then
    echo "  OK: Speed=$speed"
else
    echo "  NG: Speed=$speed (期待: 100GbE)"
fi
echo ""

echo "[3/5] IP / MTU"
dev=$(ls /sys/class/infiniband/mlx5_0/device/net/)
ip_mtu=$(ip addr show "$dev" 2>/dev/null)
echo "$ip_mtu" | grep -q "192.168.100" && echo "  OK: IP 設定済み" || echo "  NG: IP 未設定"
echo "$ip_mtu" | grep -q "mtu 9000" && echo "  OK: MTU 9000" || echo "  NG: MTU が 9000 ではない"
echo ""

echo "[4/5] GPU パワーリミット"
pl=$(nvidia-smi -q -d POWER 2>/dev/null | grep "Current Power Limit" | head -1 | awk '{print $5}')
if [ "$pl" = "180.00" ]; then
    echo "  OK: Power Limit=$pl W"
else
    echo "  NG: Power Limit=$pl W (期待: 180.00)"
fi
echo ""

echo "[5/5] systemd サービス"
for svc in mst mlxlink-config nvidia-power-limit; do
    status=$(systemctl is-active "$svc" 2>/dev/null)
    enabled=$(systemctl is-enabled "$svc" 2>/dev/null)
    if [ "$enabled" = "enabled" ]; then
        echo "  OK: $svc.service — enabled ($status)"
    else
        echo "  NG: $svc.service — $enabled ($status)"
    fi
done
```

---

## ロールバック手順

設定を元に戻す (init.sh による手動実行に戻す) 場合の手順。

### 1. systemd サービスの無効化と削除

```bash
# 両ノードで実行
sudo systemctl disable mlxlink-config.service
sudo systemctl disable nvidia-power-limit.service
sudo systemctl disable mst

sudo rm /etc/systemd/system/mlxlink-config.service
sudo rm /etc/systemd/system/nvidia-power-limit.service
sudo systemctl daemon-reload
```

### 2. netplan 設定の削除

```bash
# 両ノードで実行
sudo rm /etc/netplan/60-ib-rdma.yaml
sudo netplan apply
```

### 3. 確認

```bash
# サービスが存在しないことを確認
systemctl status mlxlink-config.service
# → Unit mlxlink-config.service could not be found.

systemctl status nvidia-power-limit.service
# → Unit nvidia-power-limit.service could not be found.

systemctl is-enabled mst
# → disabled
```

以降は従来通り `~/init.sh` を手動実行して環境を初期化する。

---

## 補足事項

### nvidia-peermem について

GPUDirect RDMA に必要な `nvidia-peermem` カーネルモジュールは、NVIDIA ドライバのインストール時に自動ロードされるよう設定されており、本マニュアルでの追加設定は不要。

```bash
lsmod | grep nvidia_peermem
# nvidia_peermem  16384  0
```

### init.sh の扱い

永続化設定の適用・検証後も `~/init.sh` は削除せず保持しておくことを推奨。ロールバック時や設定内容の参照に使用できる。
