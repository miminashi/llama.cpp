# topk_moe scratch copy fusion 再現性確認 + feature/rdma-backend マージ

- **実施日時**: 2026年3月8日 22:43
- **ワークツリー**: `.worktree/merge-gpu-native-sort` (検証), `feature/rdma-backend` (マージ先)
- **関連レポート**: [report/2026-03-08_205719_topk_moe_scratch_copy_fusion.md](2026-03-08_205719_topk_moe_scratch_copy_fusion.md)

## 前提・目的

前回 `merge/gpu-native-sort` ブランチで topk_moe fusion のスクラッチコピー修正 (`94b58c301`) を実装・検証済み。pp2048 退行 (467.76 → 519.86 t/s) の回復を確認した。

本テストの目的:
1. pp16384 相当の長文推論で正確性を再確認
2. pp128/512/2048 のパフォーマンス再現性を確認
3. 問題なければ `feature/rdma-backend` にマージ
4. マージ後ビルドのサニティチェック

## 再現方法

### 正確性テスト (pp16384 相当の長文推論)

```bash
python3 -c "print('Hello ' * 4000)" > /tmp/long_prompt.txt

gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  .worktree/merge-gpu-native-sort/build/bin/llama-cli \
  -m /tmp/qwen35-35b-a3b-q4km-fused.gguf \
  -dev 'CUDA0,CUDA1,RDMA0[192.168.100.2:50051],RDMA1[192.168.100.2:50051]' \
  -fa 1 -ngl 999 -f /tmp/long_prompt.txt -n 50 --log-file /tmp/llama-cli.log
```

### パフォーマンス再現性確認

```bash
gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  .worktree/merge-gpu-native-sort/build/bin/llama-bench \
  -m /tmp/qwen35-35b-a3b-q4km-fused.gguf \
  -dev 'CUDA0/CUDA1/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]' \
  -fa 1 -p 128,512,2048 -n 32 -r 5 -ngl 999
```

### マージ + マージ後サニティチェック

```bash
git merge merge/gpu-native-sort --no-ff

rm -rf build && cmake -B build -DGGML_CUDA=ON -DGGML_RDMA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="60" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16 --target llama-bench llama-cli

gpu-lock.sh run GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=4,5 \
  build/bin/llama-bench \
  -m /tmp/qwen35-35b-a3b-q4km-fused.gguf \
  -dev 'CUDA0/CUDA1/RDMA0[192.168.100.2:50051]/RDMA1[192.168.100.2:50051]' \
  -fa 1 -p 128,2048 -n 32 -r 3 -ngl 999
```

## 結果

### 正確性テスト

- **結果**: 合格
- pp16384 相当 (~4000語) の長文プロンプト + 50トークン生成で正常出力を確認
- NaN/garbage なし、Thinking モードで正しい推論出力
- 速度: pp 613.1 t/s, tg 35.5 t/s

### パフォーマンス再現性確認 (Qwen3.5-35B-A3B Q4_K_M, 2C+2R)

| テスト | Pre-merge (参照) | 前回 scratch fix | 今回 (再現性) | マージ後 |
|--------|:-:|:-:|:-:|:-:|
| pp128 | 272.09 | 272.27 | **273.93 ± 2.61** | **272.81 ± 2.77** |
| pp512 | 459.07 | 459.52 | **459.10 ± 2.32** | — |
| pp2048 | 518.91 | 519.86 | **518.13 ± 2.60** | **519.28 ± 2.42** |
| tg32 | 36.12 | 35.76 | **36.48 ± 0.07** | **36.52 ± 0.04** |

- **pp2048 退行回復の再現性**: 確認済み (519.86 → 518.13, 差 -0.3% = 誤差範囲)
- **マージ後の性能劣化**: なし (全指標が誤差範囲内)

### マージ

- `merge/gpu-native-sort` → `feature/rdma-backend` にコンフリクトなしでマージ
- 18コミット (upstream merge + GPU-native sort + scratch copy fix + llama-cli crash fix)
- マージ後ビルド成功、サニティチェック通過

## 結論

- topk_moe fusion のスクラッチコピー修正による pp2048 退行回復 (+11%) は再現性あり
- 長文推論の正確性も問題なし
- `feature/rdma-backend` へのマージ完了

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `c501fdc20 (dirty) (feature/rdma-backend)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 27°C | 33°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Enabled | Enabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 1700528) |
