# PCIeトポロジー検証レポート

- **実施日時**: 2026年2月3日 13:49

## 前提・目的

### 背景
前回の調査（[`gpudirect_multinode_test_2026-01-31_204135.md`](./gpudirect_multinode_test_2026-01-31_204135.md)）で、2号機のPCIeトポロジーに問題があることが判明した：
- GPU0-3とNICがNUMA間接続（SYS）
- GPUDirectRDMAが無効化される原因

### 目的
2号機再起動後のPCIeトポロジーが改善されたかを確認する。

## 検証結果

### 1号機 (192.168.100.1) - 現在のトポロジー

```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  NIC0  NUMA
GPU0     X    PIX   PIX   PHB   PHB   PHB   PHB   PHB    0
GPU1    PIX    X    PIX   PHB   PHB   PHB   PHB   PHB    0
GPU2    PIX   PIX    X    PHB   PHB   PHB   PHB   PHB    0
GPU3    PHB   PHB   PHB    X    PIX   PIX   PIX   PIX    0
GPU4    PHB   PHB   PHB   PIX    X    PIX   PIX   PIX    0
GPU5    PHB   PHB   PHB   PIX   PIX    X    PIX   PIX    0
GPU6    PHB   PHB   PHB   PIX   PIX   PIX    X    PIX    0
NIC0    PHB   PHB   PHB   PIX   PIX   PIX   PIX    X     -
```

**特徴**:
- 7台のGPU（前回5台から増加）
- GPU3-6とNIC（mlx5_0）がPIX接続（同一PCIeブリッジ下）
- GPU0-2とNICはPHB接続
- 全GPUがNUMA 0に配置

### 2号機 (192.168.100.2) - 現在のトポロジー

```
        GPU0  GPU1  GPU2  GPU3  NIC0  NUMA
GPU0     X    PHB   PHB   SYS   PIX    0
GPU1    PHB    X    PIX   SYS   PHB    0
GPU2    PHB   PIX    X    SYS   PHB    0
GPU3    SYS   SYS   SYS    X    SYS    1
NIC0    PIX   PHB   PHB   SYS    X     -
```

**特徴**:
- 4台のGPU（前回5台から1台減少、GPU4が消失）
- **GPU0とNIC（mlx5_0）がPIX接続**（前回はSYS → **改善**）
- GPU1-2とNICはPHB接続（前回はSYS → **改善**）
- GPU3とNICはSYS接続（NUMA間、変化なし）
- GPU0-2はNUMA 0、GPU3はNUMA 1に配置

### 前回との比較（2号機）

| GPU | 前回(2026/1/31) | 現在(2026/2/3) | 変化 |
|-----|-----------------|----------------|------|
| GPU0-NIC | SYS | **PIX** | ✅ 大幅改善 |
| GPU1-NIC | SYS | **PHB** | ✅ 改善 |
| GPU2-NIC | SYS | **PHB** | ✅ 改善 |
| GPU3-NIC | SYS | SYS | − 変化なし |
| GPU4-NIC | PHB | (消失) | GPU4が認識されず |

## 結論

### 改善点
1. **2号機のGPU0-NIC接続がPIXに改善** - GPUDirectRDMAが有効になる可能性
2. GPU1-2のNIC接続もPHBに改善 - CPUホストブリッジ経由だが同一NUMA内

### 課題
1. 2号機のGPU4が認識されなくなった（要確認）
2. 2号機GPU3はNUMA 1に配置されたままでNICとSYS接続

### 推奨事項
1. **GPUDirectRDMAのテスト**: GPU0を使用した構成で`NCCL_NET_GDR_LEVEL`を有効にしてテスト
2. **GPU4の調査**: `lspci`でGPU4が物理的に認識されているか確認
3. **最適な構成**: 2号機ではGPU0を優先的に使用することでGDR有効化の可能性

## 再現方法

### トポロジー確認コマンド

```bash
# 1号機
nvidia-smi topo -m

# 2号機
ssh ubuntu@192.168.100.2 "nvidia-smi topo -m"
```
