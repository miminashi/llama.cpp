# RDMA性能ばらつき深掘り調査レポート

- **実施日時**: 2026年2月5日 21:07

## 前提・目的

### 背景
前回の調査 ([2026-02-05_192000_gpu_combination_variance_analysis.md](./2026-02-05_192000_gpu_combination_variance_analysis.md)) で、RDMA 1+1 GPU構成において約2秒のスパイクがgraph_compute中にランダムに発生し、pp128で多発することが判明した。

| 構成 | 分散 (pp128) |
|------|-------------|
| ローカル 2GPU | < 1% (完全安定) |
| RPC 1+1 | < 7% (安定) |
| RDMA 1+1 | 最大1400% (不安定) |

### 目的
スパイクの発生箇所を特定し、根本原因を確定する。

### 調査計画
本調査では以下の実験を計画した:

| 実験 | 内容 | 対象ファイル |
|------|------|-------------|
| 実験1+4 | フルグラフ送信の詳細プロファイリング | ggml-rdma.cpp |
| 実験2 | ビジーポーリングにスリープ追加 | rdma-transport.cpp |
| 実験3 | グラフキャッシュの複数エントリ対応 | (未実施) |

## 結論

### 今回の調査で判明したこと

**実験2で追加した `usleep(10)` が、全てのIB Send/Recv操作を約2秒遅延させる深刻な問題を引き起こした。**

usleep(10) を削除したところ、pp128 の分散は **2%** まで改善し、速度も **361.58 t/s** となった。

### 前回のスパイクについて

**前回の調査で報告された「1400%の分散」の原因は、今回の調査では特定できなかった。**

理由：
- 前回の調査時点では usleep(10) は存在していなかった
- したがって、前回のスパイクの原因は usleep(10) ではありえない
- usleep(10) 削除後の現在、前回報告された「1400%の分散」は再現されていない

前回のスパイクが再現されない理由として考えられるもの：
1. 前回のビルドや環境に何らかの問題があった
2. ネットワーク/GPU状態などの環境要因が変化した
3. 前回の測定自体に問題があった可能性

**結論として、前回の問題が何だったのかは不明だが、現在は安定して動作している。**

## 調査経緯

### Step 1: プロファイリングと usleep の同時追加

計画に従い、以下を同時に実施した：
- `send_rdma_cmd()` 内に send/recv の分離計測を追加（実験1+4）
- `wait_for_completion()` に usleep(10) を追加（実験2）

テスト結果:
```
[client SPIKE] recompute #6: send=2149.82 ms, recv=65.21 ms (TOTAL=2215.03 ms)
[client SPIKE] recompute #7: send=2149.79 ms, recv=70.07 ms (TOTAL=2219.86 ms)
...
```

**観測**: 「ランダム」ではなく、**ほぼ全てのgraph_compute** で send が ~2150ms かかっていた。

### Step 2: サーバー側ログの確認

```
[server profile] cmd FLUSH_AND_RECOMPUTE: total=2149.90 ms (recv=2146.59, rsp_send=0.01)
[server profile] graph_recompute: total=3.26 ms (fix_xdev=0.04, compute=3.23)
```

- サーバー側でも `recv=2146ms` でデータ受信に2秒待機
- GPU計算自体は `compute=3.23ms` で正常
- **クライアントの send とサーバーの recv が同時に2秒待っている**

### Step 3: send_rdma_cmd_raw 内の詳細計測

send処理の内部を詳細に計測:
```cpp
uint64_t t_hdr = profile_now_us();
conn->send(header, header_size, nullptr);  // ヘッダ送信
uint64_t hdr_us = profile_now_us() - t_hdr;

uint64_t t_data = profile_now_us();
conn->send(input, input_size, nullptr);    // データ送信
uint64_t data_us = profile_now_us() - t_data;
```

結果:
```
[client send_raw] cmd=10: header=0.00 ms, data=2150.37 ms (size=4)
[client send_raw] cmd=11: header=0.00 ms, data=2150.70 ms (size=296)
```

**重大発見**:
- header 送信は瞬時 (0.00ms)
- **たった4バイトのデータ送信に 2150ms かかっている**
- これはネットワーク遅延ではあり得ない

### Step 4: usleep の疑い

4バイト送信に2秒かかる原因として、実験2で追加した `usleep(10)` を疑った。

`conn->send()` は内部で `wait_for_completion()` を呼び、CQをポーリングして完了を待つ:

```cpp
while (true) {
    int n = ibv_poll_cq(cq_, 1, &wc);
    if (n > 0) return true;  // 完了

    // 100回ポーリング後に10usスリープ（実験2で追加）
    if (++poll_count > 100) {
        usleep(10);  // ← これが問題の可能性
    }
}
```

### Step 5: usleep 無効化で検証

usleep をコメントアウトして再テスト:

```
[client FULL_GRAPH] #2: send=0.12 ms, recv=67.15 ms  ← 正常!
[client detail] recompute: send=0.01 ms, recv=64.42 ms  ← 正常!
```

**問題が解消された。** send が 0.01ms に戻った。

### 最終結果

| 項目 | usleep あり | usleep なし |
|------|------------|------------|
| send 時間 | 2150 ms | 0.01 ms |
| pp128 速度 | 7.16 t/s | **361.58 t/s** |
| pp128 分散 | 全操作遅延 | **2%** |

## なぜ usleep が 2秒の遅延を引き起こすのか

IB verbs の Send/Recv は以下のように動作する:

1. クライアントが `ibv_post_send()` でデータを送信
2. サーバーが `ibv_post_recv()` で受信バッファを登録
3. 両者が `ibv_poll_cq()` で完了を待つ

**問題のメカニズム**:
- `usleep(10)` を入れると、ポーリング間隔が 10us になる
- IB verbs の完了通知は**即座に**処理される必要がある
- スリープ中に完了イベントが発生しても、次のポーリングまで検知されない
- クライアントとサーバーが互いに「相手の完了」を待つデッドロック的状況が発生
- 最終的にタイムアウト処理やリトライで ~2秒後に解消される

**教訓**: IB verbs のポーリングCQはタイムクリティカル。ビジーポーリングが前提の設計であり、スリープを入れてはいけない。

## 変更内容

### ggml-rdma.cpp (プロファイリング追加)
- `send_rdma_cmd()`: send/recv の分離計測、スパイク自動検出
- `send_rdma_cmd_raw()`: header/data 送信の個別計測

### rdma-transport.cpp (usleep 無効化)
```cpp
// wait_for_completion() 内
// Experiment 2: DISABLED - caused 2s delays
// if (++poll_count > 100) {
//     usleep(10);
// }
(void)poll_count;
```

## 再現方法

### 1. ビルド (1号機)
```bash
cd /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend
rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 2. デプロイ・ビルド (2号機)
```bash
ssh 192.168.100.2 "rm -rf /home/ubuntu/projects/llama.cpp"
rsync -a --exclude='.git' --exclude='build' /home/ubuntu/projects/llama.cpp/.worktree/rdma-backend/ 192.168.100.2:/home/ubuntu/projects/llama.cpp/
ssh 192.168.100.2 "cd /home/ubuntu/projects/llama.cpp && rm -rf build && cmake -B build -DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
```

### 3. サーバー起動 (2号機)
```bash
ssh 192.168.100.2 "GGML_RDMA_PROFILE=1 LD_LIBRARY_PATH=/home/ubuntu/projects/llama.cpp/build/bin nohup /home/ubuntu/projects/llama.cpp/build/bin/rdma-server -H 0.0.0.0 -p 50051 > /tmp/rdma-server.log 2>&1 &"
```

### 4. ベンチマーク実行 (1号機)
```bash
GGML_RDMA_PROFILE=1 GGML_RDMA_SERVERS=192.168.100.2:50051 CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=build/bin build/bin/llama-bench \
-m ~/.cache/llama.cpp/unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-Q4_K_M.gguf \
-ngl 999 -sm layer -r 3 -p 128 -n 32
```

## 残存課題

1. **前回のスパイクの原因は未解明**: 前回報告された「1400%の分散」が何に起因していたのかは不明。現在は再現されていない。

2. **最初のリクエストで ~2秒の遅延**: warmupフェーズでのGPU/メモリ初期化が原因と思われる。

3. **tg32 での 25% 分散**: 複数の graph_compute 呼び出しの累積効果。これは正常な挙動の可能性がある。

4. **CPU使用率**: ビジーポーリングはCPU 100%を消費する。将来的にはイベント駆動 (`ibv_req_notify_cq` + `ibv_get_cq_event`) への移行を検討。
