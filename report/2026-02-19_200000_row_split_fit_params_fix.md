# GLM-4.7 row split ロード失敗の調査と修正

- **実施日時**: 2026年2月19日 20:00
- **ワークツリー**: `.worktree/rdma-row-split`
- **参照レポート**: [report/2026-02-19_163700_row_vs_layer_split_benchmark.md](2026-02-19_163700_row_vs_layer_split_benchmark.md)

## 前提・目的

### 背景

[row vs layer split ベンチマーク](2026-02-19_163700_row_vs_layer_split_benchmark.md)で、GLM-4.7 IQ2_M を 7 ローカル GPU で `-sm row` 実行したところ「ロード失敗」となった。layer split でも同様に失敗しており、VRAM 不足 (114GB > 112GB) が原因。

### 目的

1. `llama_params_fit` が row split に対応していない問題を調査・修正する
2. 修正により、VRAM がわずかに不足する場合でも自動的にレイヤーを CPU オフロードして row split で動作させる

### 前提条件

- **GPU**: 1号機 (192.168.100.1) の Tesla P100 PCIe 16GB × 7
- **モデル**: GLM-4.7 IQ2_M (~114 GB), gpt-oss-20b Q4_K_M (~11 GB)
- **合計 VRAM**: 7 × 16 GB = 112 GB (GLM-4.7 より 2GB 不足)

## 調査結果

### 原因: `llama_params_fit` が row split 未対応

`src/llama.cpp:341-342` で、`split_mode == LLAMA_SPLIT_MODE_ROW` かつ複数デバイスの場合に即座に例外をスローしていた:

```cpp
if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
    throw llama_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
}
```

これにより、row split モードでは `llama_params_fit` による自動 ngl 調整が無効化されていた。

### 重要な発見: llama-bench は `llama_params_fit` を使わない

| ツール | `llama_params_fit` | ngl 処理 |
|--------|:-:|----------|
| **llama-cli** | 使用 (`common_init_result`) | `-ngl` 未指定時は auto(-1), `llama_params_fit` が調整 |
| **llama-bench** | **不使用** | 直接 `llama_model_load_from_file` を呼ぶ。ngl はデフォルト 99 |

ベンチマークレポートの「ロード失敗」は llama-bench を使用しており、`llama_params_fit` は呼ばれていなかった。row split の修正効果は **llama-cli** (auto ngl) で発現する。

### `llama_params_fit` のガード条件

| 行 | 条件 | 適用場面 |
|:--:|------|---------|
| 327 | `n_gpu_layers != -1` (ユーザー指定) | `-ngl 999` 等 → 例外 |
| 331 | `tensor_split` バッファなし | 内部エラー |
| 334 | `tensor_split` ユーザー指定済み | `-ts` 使用時 → 例外 |
| **341** | **`split_mode == ROW`** | **修正対象** |
| 345 | `tensor_buft_overrides` バッファなし | 内部エラー |

row split 修正は行 341 のみ。他のガードは変更不要。

## 修正内容

### 変更ファイル

- `src/llama.cpp` (行 341-378)

### 修正方針

Row split では全レイヤーが全 GPU に比例配分される（`tensor_split` 比率で）。Layer split のようなデバイスごとのレイヤー割り当ては不適切。

修正アプローチ:
1. `tensor_split` は変更しない（デフォルトの VRAM 比例配分を使用）
2. `tensor_buft_overrides` も不要（部分レイヤーのオーバーフローは不要）
3. `n_gpu_layers` のみバイナリサーチで調整

### 実装

```cpp
if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
    // Row split: each layer is distributed across all GPUs proportionally.
    // Binary search for the maximum ngl that fits on all devices.

    uint32_t ngl_lo = 0;
    uint32_t ngl_hi = hp_ngl + 1;

    auto row_split_fits = [&](uint32_t ngl_test) -> bool {
        llama_model_params mparams_test = *mparams;
        mparams_test.n_gpu_layers = ngl_test;
        const dmds_t dmds_test = llama_get_device_memory_data(
            path_model, &mparams_test, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        for (size_t id = 0; id < nd; id++) {
            if (int64_t(dmds_test[id].mb.total()) > int64_t(dmds_test[id].free) - margins[id]) {
                return false;
            }
        }
        return true;
    };

    if (!row_split_fits(ngl_lo)) {
        throw llama_params_fit_exception("row split: model does not fit even with n_gpu_layers=0");
    }

    while (ngl_hi - ngl_lo > 1) {
        uint32_t ngl_mid = ngl_lo + (ngl_hi - ngl_lo) / 2;
        if (row_split_fits(ngl_mid)) {
            ngl_lo = ngl_mid;
        } else {
            ngl_hi = ngl_mid;
        }
    }

    mparams->n_gpu_layers = ngl_lo;
    return;
}
```

バイナリサーチの反復回数は `log2(n_layer + 1)` ≈ 6回（GLM-4.7 の 62 層の場合）。各反復で `llama_get_device_memory_data` を呼ぶが、`no_alloc=true` のため実際の VRAM 割り当ては行わない。

## 再現方法

### ビルド

```bash
bash scripts/rdma-build.sh local
```

### テスト: GLM-4.7 + 7 GPU + row split (auto ngl)

```bash
bash scripts/gpu-lock.sh run build/bin/llama-cli \
  -m /tmp/GLM-4.7-IQ2_M/GLM-4.7-UD-IQ2_M-00001-of-00003.gguf \
  -sm row -c 256 -n 10 -p 'hello' \
  --no-warmup --single-turn --simple-io --log-file /tmp/llama-cli.log
```

## 結果

### 修正前 vs 修正後

| シナリオ | 修正前 | 修正後 |
|---------|:------:|:------:|
| GLM-4.7 + 7GPU + row + auto ngl | ロード失敗 | **成功** (pp=3.5, tg=2.6) |
| GLM-4.7 + 7GPU + layer + auto ngl | 成功 | 成功 (pp=6.5, tg=5.9) |
| gpt-oss-20b + 7GPU + row + auto ngl | 成功 | 成功 (pp=91.5, tg=43.9) |
| gpt-oss-20b + 7GPU + row + ngl=999 | 成功 | 成功 (pp=299.6, tg=42.9) |

### GLM-4.7 row split のメモリ配分

修正後、`llama_params_fit` が自動的に ngl を調整し、一部レイヤーを CPU にオフロード:

| デバイス | VRAM 使用 (MiB) | モデル (MiB) | 空き (MiB) |
|---------|:---------------:|:------------:|:----------:|
| CUDA0 | 14605 | 13882 | 1102 |
| CUDA1 | 14299 | 14201 | 1392 |
| CUDA2 | 14364 | 14266 | 1260 |
| CUDA3 | 13887 | 13790 | 1856 |
| CUDA4 | 13743 | 13646 | 1916 |
| CUDA5 | 13761 | 13663 | 1940 |
| CUDA6 | 12993 | 12821 | 3110 |
| **Host** | — | **18590** | — |

GPU 合計: ~96 GiB + Host: ~18 GiB = ~114 GiB (モデル全体)

### 性能比較 (GLM-4.7, 7 ローカル GPU)

| Split Mode | pp (t/s) | tg (t/s) |
|:----------:|:--------:|:--------:|
| row | 3.5 | 2.6 |
| layer | 6.5 | 5.9 |
| **layer/row** | **1.86x** | **2.27x** |

Layer split が全条件で優位。これは NVLink なし環境での既知の制限 ([参照レポート](2026-02-19_163700_row_vs_layer_split_benchmark.md))。

## 制約事項

1. **llama-bench では効果なし**: `llama_params_fit` を使わないため、`-ngl 999` でモデルが VRAM に収まらない場合は引き続き失敗する
2. **`-ngl` 明示指定時は効果なし**: `llama_params_fit` はユーザー指定の ngl を変更しない（行 327）
3. **`tensor_split` 明示指定時は効果なし**: 行 334-339 のガードで除外される

## 環境情報

| 項目 | 1号機 | 2号機 |
|------|-------|-------|
| Git commit | `6a096a20d (feature/rdma-row-split)` | — |
| NVIDIA Driver | 535.288.01 | 535.288.01 |
| GPU | 7x Tesla P100-PCIE-16GB | 4x Tesla P100-PCIE-16GB |
| GPU 温度 (max) | 35°C | 33°C |
| 電力制限 | 180.00W | 180.00W |
| SM 周波数 (max) | 1328 MHz | 1328 MHz |
| Persistence Mode | Disabled | Disabled |
| nvidia-peermem | loaded | loaded |
| IB デバイス | mlx5_0 (Active, 100) | mlx5_0 (Active, 100) |
| P2P トポロジ | NODE/NV/PHB/PIX/PXB/SYS | NODE/NV/PHB/PIX/PXB/SYS |
| ACS | disabled | disabled |
| IOMMU | disabled | disabled |
| rdma-server | — | running (PID 17159) |
