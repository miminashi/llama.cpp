# P100 PCIe P2P転送仕様レポート

- **実施日時**: 2026年2月7日 19:25

## 前提・目的

RDMAバックエンドプロジェクトにおいて、1号機・2号機に搭載されているTesla P100 PCIe GPUのP2P（Peer-to-Peer）転送の仕様を調査し、実機検証結果とあわせて文書化する。

- **背景**: RDMAバックエンドでは、ノード間はRDMA/GPUDirect RDMAで転送し、ノード内はCUDA P2Pで転送する。P100 PCIeはNVLinkを搭載していないが、PCIe経由のP2P転送には対応しており、その制約と可用性を把握することがGPU配置やスケジューリングの設計に重要
- **目的**: P100 PCIeのP2P転送仕様の整理、実機でのトポロジとP2P可用性の検証、RDMAバックエンドへの示唆の文書化
- **前提条件**: 1号機（7 GPU）と2号機（4 GPU）のTesla P100-PCIE-16GBクラスタが稼働中

## 再現方法

```bash
# トポロジマトリクス
nvidia-smi topo --matrix

# P2P Read/Write/NVLink/Atomic サポート状況
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -p2p n
nvidia-smi topo -p2p a

# BAR1メモリ
nvidia-smi -q | grep -A3 "BAR1 Memory"

# PCIeリンク速度
sudo lspci -vvs <GPU BUS ID> | grep -i lnk

# PCIeトポロジツリー
lspci -tv
```

## P100 PCIe P2P転送の仕様

### P2Pメカニズム

P100 PCIeのP2P転送は、**BAR1（Base Address Register 1）マッピング**を利用する。

1. **BAR1マッピング**: 一方のGPUのVRAMをPCIeアドレス空間にマップし、他方のGPUからPCIe経由でリード/ライトする
2. **有効化API**: `cudaDeviceEnablePeerAccess()` でGPUペアごとにP2Pアクセスを有効化する必要がある
3. **可用性チェック**: `cudaDeviceCanAccessPeer()` で事前に確認可能
4. **メモリ登録**: GPUメモリは64KBチャンク単位でBAR空間にピン留めされる

### PCIe P2Pの帯域

| 接続 | 理論帯域（片方向） | 備考 |
|------|:------------------:|------|
| PCIe 3.0 x16 | ~15.75 GB/s | 実効 ~12-13 GB/s（プロトコルオーバーヘッド含む） |
| PCIe P2P (PIX, 同一スイッチ) | ~10-11 GB/s | 実測値（公開ベンチマーク） |
| PCIe P2P (PHB, ホストブリッジ経由) | ~8-10 GB/s | CPUのPCIeルートコンプレックスを経由 |
| NVLink (P100 SXM2版のみ) | ~35 GB/s | PCIe版には非搭載 |

### P2P Atomicの制限

> "For GP100 atomic operations may target the memories of peer GPUs connected through NVLink. GPUs connected via PCIE do not support this feature."
> — NVIDIA Pascal Tuning Guide

P100 PCIe版ではP2P Atomic操作は使用不可。NVLink経由でのみサポートされる。

### BAR1サイズ

全GPU共通: **BAR1 = 16384 MiB (16 GB)**

Large BAR (16GB) が有効なため、VRAM全体をBAR空間にマッピング可能。これはGPUDirect RDMAにも必要な条件であり、本クラスタではBIOS設定済み。

## 実機構成と検証結果

### 1号機 (192.168.100.1) — 7 GPU

**基本情報:**
- GPU: Tesla P100-PCIE-16GB × 7
- アーキテクチャ: Pascal (GP100)
- ドライバ: 535.288.01 / CUDA 12.2
- PCIeリンク: Gen3 x16 (8GT/s)
- 全GPUが**NUMA 0**（同一ソケット）に接続

**PCIeトポロジ:**

```
CPU0 (NUMA 0)
├── PCIe Root Port 02 → PLX Switch
│   ├── GPU0 (05:00.0) ──┐
│   ├── GPU1 (07:00.0) ──┤ PIX (同一スイッチ)
│   └── GPU2 (08:00.0) ──┘
│
└── PCIe Root Port 03 → PLX Switch
    ├── GPU3 (0c:00.0) ──┐
    ├── GPU4 (0d:00.0) ──┤ PIX (同一スイッチ)
    ├── GPU5 (0e:00.0) ──┤
    ├── GPU6 (0f:00.0) ──┘
    └── NIC0 (mlx5_0)
```

- GPU0-1-2: 同一PLXスイッチ（PIX）
- GPU3-4-5-6: 同一PLXスイッチ（PIX）、NIC0も同じスイッチ
- 異なるスイッチ間: PHB（PCIe Host Bridge経由）

**トポロジマトリクス (`nvidia-smi topo --matrix`):**

```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  NIC0
GPU0     X    PIX   PIX   PHB   PHB   PHB   PHB   PHB
GPU1    PIX    X    PIX   PHB   PHB   PHB   PHB   PHB
GPU2    PIX   PIX    X    PHB   PHB   PHB   PHB   PHB
GPU3    PHB   PHB   PHB    X    PIX   PIX   PIX   PIX
GPU4    PHB   PHB   PHB   PIX    X    PIX   PIX   PIX
GPU5    PHB   PHB   PHB   PIX   PIX    X    PIX   PIX
GPU6    PHB   PHB   PHB   PIX   PIX   PIX    X    PIX
NIC0    PHB   PHB   PHB   PIX   PIX   PIX   PIX    X
```

**P2P Read/Write/NVLink/Atomic サポートマトリクス:**

| | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 |
|------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| GPU0 | X | OK | OK | OK | OK | OK | OK |
| GPU1 | OK | X | OK | OK | OK | OK | OK |
| GPU2 | OK | OK | X | OK | OK | OK | OK |
| GPU3 | OK | OK | OK | X | OK | OK | OK |
| GPU4 | OK | OK | OK | OK | X | OK | OK |
| GPU5 | OK | OK | OK | OK | OK | X | OK |
| GPU6 | OK | OK | OK | OK | OK | OK | X |

- **P2P Read**: 全ペアOK
- **P2P Write**: 全ペアOK
- **NVLink**: 全ペアNS（Not Supported）
- **P2P Atomic**: 全ペアNS（Not Supported）

**結論**: 1号機は全7GPU間でPCIe P2P転送が可能。全GPUが同一NUMAノードにあるため、PHB接続でもP2P転送がサポートされる。

### 2号機 (192.168.100.2) — 4 GPU

**基本情報:**
- GPU: Tesla P100-PCIE-16GB × 4
- アーキテクチャ: Pascal (GP100)
- ドライバ: 535.288.01 / CUDA 12.2
- PCIeリンク: Gen3 x16 (8GT/s)
- GPU0-1-2: **NUMA 0**、GPU3: **NUMA 1**（異なるソケット）

**PCIeトポロジ:**

```
CPU0 (NUMA 0)
├── PCIe Root Port 02 → PLX Switch
│   ├── GPU0 (05:00.0) ──┐ PIX
│   └── NIC0 (mlx5_0)    │
│                         │
└── PCIe Root Port 03 → PLX Switch
    ├── GPU1 (08:00.0) ──┤ PIX
    └── GPU2 (09:00.0) ──┘

        ~~~ QPI/UPI ~~~

CPU1 (NUMA 1)
└── PCIe Root Port 03 → PLX Switch
    └── GPU3 (88:00.0) ← 別ソケット (SYS)
```

- GPU0: NIC0と同一スイッチ（PIX）
- GPU1-2: 同一PLXスイッチ（PIX）
- GPU0 ↔ GPU1/2: PHB（異なるPCIeルートポートだがCPU0内）
- GPU3: CPU1配下（SYS = QPI経由）

**トポロジマトリクス (`nvidia-smi topo --matrix`):**

```
        GPU0  GPU1  GPU2  GPU3  NIC0
GPU0     X    PHB   PHB   SYS   PIX
GPU1    PHB    X    PIX   SYS   PHB
GPU2    PHB   PIX    X    SYS   PHB
GPU3    SYS   SYS   SYS    X    SYS
NIC0    PIX   PHB   PHB   SYS    X
```

**P2P Read/Write サポートマトリクス:**

| | GPU0 | GPU1 | GPU2 | GPU3 |
|------|:----:|:----:|:----:|:----:|
| GPU0 | X | OK | OK | **TNS** |
| GPU1 | OK | X | OK | **TNS** |
| GPU2 | OK | OK | X | **TNS** |
| GPU3 | **TNS** | **TNS** | **TNS** | X |

- **GPU0-1-2間**: P2P Read/Write OK
- **GPU3 ↔ 他GPU**: TNS (Topology Not Supported) — QPIをまたぐP2Pは不可
- **NVLink**: 全ペアNS
- **P2P Atomic**: 全ペアNS

**結論**: 2号機はGPU0-1-2間でPCIe P2P転送が可能だが、GPU3は別ソケット（NUMA 1）にあるためP2P不可。GPU3とのデータ転送はCPUメモリ経由のステージング（`cudaMemcpy` D2H→H2D）が必要。

### PCIeリンク速度

```
LnkCap: Port #8, Speed 8GT/s, Width x16, ASPM not supported
LnkSta: Speed 8GT/s, Width x16
```

- 全GPUがPCIe Gen3 x16でネゴシエーション済み
- 理論帯域: 8GT/s × 16レーン × 128/130b符号化 ≈ 15.75 GB/s（片方向）

### PCIe BAR領域 (GPU0, 1号機)

```
Region 0: Memory at c4000000 (32-bit, non-prefetchable) [size=16M]    ← 制御レジスタ
Region 1: Memory at 27800000000 (64-bit, prefetchable) [size=16G]     ← BAR1 (VRAM)
Region 3: Memory at 27c00000000 (64-bit, prefetchable) [size=32M]     ← BAR3
```

Region 1の16GBがBAR1であり、VRAM全体をPCIeアドレス空間にマッピング可能。

## トポロジとP2P可用性の関係

| トポロジ | 略称 | 意味 | P2P可否 | 性能 |
|---------|------|------|:------:|------|
| **PIX** | Peer via switch | 同一PCIeスイッチ配下 | OK | 最良（スイッチ内ルーティング） |
| **PHB** | PCIe Host Bridge | 同一CPUの異なるルートポート | OK | 良好（CPUのPCIeルートコンプレックス経由） |
| **PXB** | Multiple bridges | 複数PCIeブリッジ経由 | OK | 中間 |
| **NODE** | NUMA internal | NUMA内のホストブリッジ間 | 条件付き | 中間 |
| **SYS** | Cross-socket | QPI/UPI経由のソケット間 | **TNS** | 不可（PCIe P2Pルーティング不可） |

**重要な知見**: PCIe P2Pは同一PCIeルートコンプレックス（同一CPU）配下のGPU間でのみ動作する。QPI/UPIをまたぐソケット間ではPCIeのP2Pルーティングがサポートされない（チップセット依存だが、Broadwell-EP世代ではSYSトポロジでTNS）。

## RDMAバックエンドへの示唆

### ノード内P2P vs ノード間RDMA

| 転送パス | 方式 | 帯域 |
|---------|------|------|
| 同一ノード内、同一スイッチ (PIX) | CUDA P2P | ~10-11 GB/s |
| 同一ノード内、異なるルートポート (PHB) | CUDA P2P | ~8-10 GB/s |
| 同一ノード内、異なるソケット (SYS) | D2H→H2D | ~6-8 GB/s（CPUメモリ経由） |
| ノード間 | RDMA (IB) | ~12 GB/s（ConnectX-4 100GbE） |
| ノード間 (GPUDirect) | GPUDirect RDMA | ~10-12 GB/s（GPU直接） |

### GPUDirect RDMAとNIC配置の関係

- **1号機**: NIC0はGPU3-6と同一PLXスイッチ（PIX）。GPU3-6からのGPUDirect RDMAが最適パス。GPU0-2はPHB経由。
- **2号機**: NIC0はGPU0と同一PLXスイッチ（PIX）。GPU0からのGPUDirect RDMAが最適パス。GPU1-2はPHB経由。GPU3はSYSのためGPUDirect RDMAは性能が低い（QPI経由）。

### 16 GPU構成時の考慮事項

Step 5（GLM4.7 Q4 on 16 P100s）に向けて:

1. **2号機GPU3の制約**: 他のGPUとP2P不可、NICともSYS接続。RDMA転送時にCPUステージングが必要となり、レイテンシが増加する可能性がある
2. **レイヤー分割の配置最適化**: 連続するレイヤーを同一PLXスイッチ内のGPUに配置することで、レイヤー間の中間データ転送がPIXパスを使用でき最適
3. **NIC近接GPU優先**: ノード間転送が頻繁なレイヤー（最初・最後のレイヤーや、結合点）はNIC0に近いGPUに配置するのが望ましい
