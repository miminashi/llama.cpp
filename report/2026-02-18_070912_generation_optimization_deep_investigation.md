# Generation 性能最適化 深堀り調査レポート

- **実施日時**: 2026年2月18日 07:09
- **ワークツリー**: `.worktree/rdma-backend`

## 前提・目的

### 背景

RDMA バックエンドの Generation 速度は、同一環境の RPC バックエンド (TCP over IB) に対して約 10% 劣位にある。過去の最適化インベントリ ([2026-02-18 棚卸しレポート](2026-02-18_120000_unimplemented_optimizations_inventory.md)) では「layer split + ConnectX-4 の現環境ではソフトウェア最適化で解消不可」と結論されていた。

### 目的

upstream マージを前提とせず、互換性を壊す手法も含めて Generation 性能改善の可能性を再調査する。2名のエージェントチームによる並行調査を実施:
- **Agent A (内部コード分析)**: RDMA バックエンドのソースコード精読によるボトルネック特定と新最適化案の探索
- **Agent B (外部調査)**: P100/ConnectX-4 の仕様、NCCL/vLLM/DeepSpeed 等の実装、学術論文からの知見収集

### 現在の性能 (GLM-4.7 IQ2_M, 11GPU: 7C+4R)

| バックエンド | Prompt (t/s) | Generation (t/s) |
|-------------|:------------:|:----------------:|
| **RDMA (GDR budget 12GB)** | **6.4** | **6.8** |
| RPC (TCP) | 5.4 | 7.5 |

### 参照した過去のレポート

| # | レポート | 関連 |
|---|---------|------|
| #61 | [包括的レビュー](2026-02-13_044038_rdma_backend_comprehensive_review.md) | Plan A-D の設計調査 |
| #46 | [サーバー堅牢化 & Per-device conn](2026-02-08_020500_server_robustness_and_per_device_connections.md) | Per-device connection 実装 |
| #51 | [Async compute](2026-02-10_214841_rdma_async_compute_optimization.md) | fire-and-forget graph_compute |
| #57 | [Parallel compute dispatch](2026-02-11_214242_parallel_compute_dispatch.md) | サーバー側並列化 |
| #63 | [synchronize 実装](2026-02-13_053658_rdma_synchronize_implementation.md) | RDMA_CMD_SYNC |
| #68 | [synchronize 交絡排除再測定](2026-02-14_055914_synchronize_confound_free_remeasurement.md) | 効果なし確定 |
| -- | [未実施最適化棚卸し](2026-02-18_120000_unimplemented_optimizations_inventory.md) | 全既存案の評価 |

---

## 調査結果の統合

### エージェント間の議論ポイント

#### 合意点: PD 共有 Multi-QP が最有望

両エージェントが独立に同じ結論に到達した。

- **Agent A**: 「Per-device connection の MTT 問題は PD 共有で回避可能。PD に MR が紐づくため、QP が独立していても MR は共有される。MTT エントリ数は shared mode と同じ」
- **Agent B**: 「NCCL は同一 PD 内で複数 QP を運用。同一 PD を共有するため MR は1回だけ登録→MTT 圧力は増加しない」

**結論**: Per-device connection (#46 で実装済み、ConnectX-4 MTT 制限で無効化) の MTT 問題は、PD を全接続で共有することで解消可能な見込み。

#### 新発見: P100 の PCIe Read/Write 非対称性

Agent B が NVIDIA の GPUDirect RDMA ベンチマーク論文から発見した重大な知見:

| 操作方向 | 帯域幅 | 用途 |
|---------|:------:|------|
| RDMA Write (RNIC→GPU) | **9.8 GB/s** | set_tensor (重み転送) |
| RDMA Read (GPU→RNIC) | **3.4 GB/s** | get_tensor (結果取得) |

**現在の get_tensor は RDMA Read (pull) を使用しており、利用可能帯域幅の 35% しか活用できていない。** サーバーが RDMA Write で結果をプッシュする方式に変更すれば、帯域幅が **約 2.8 倍** 改善する。

Agent A の「Compute 完了通知 RDMA Write」提案と Agent B の「get_tensor RDMA Write push」提案は本質的に同じ方向性であり、統合すると以下になる:

> サーバーが graph_compute 完了後に、出力テンソルをクライアントの受信バッファに RDMA Write でプッシュし、RDMA Write with IMM で完了を通知する。

#### 新発見: Hugepage による MTT エントリ削減

Agent B が発見した、ハードウェア制約の回避策:

| ページサイズ | 10GB バッファの MTT エントリ数 | 現在との比 |
|:----------:|:--------------------------:|:--------:|
| 4KB (現在) | ~2,600,000 | 1x |
| 2MB (hugepage) | ~5,120 | **512x 削減** |
| 1GB (hugepage) | 10 | **260,000x 削減** |

GDR バジェット 12GB の制限は MTT キャッシュサイズに起因する。Hugepage を使用すれば同じキャッシュサイズで桁違いに多くのメモリを登録可能。ただし GPU メモリ (cudaMalloc) のページサイズは NVIDIA ドライバ管理のため、**ホストステージングバッファにのみ直接適用可能**。

Agent A は MTT 回避策として PD 共有のみを提案したが、Agent B の Hugepage 案は補完的:
- **PD 共有**: MR 重複登録の排除 → MTT エントリ数を接続数で割る
- **Hugepage**: ページサイズの拡大 → MTT エントリ数を 512 倍削減

**両者の組み合わせが最強**: PD 共有 + Hugepage staging により、ConnectX-4 の MTT 制約を実質的に無力化できる可能性がある。

#### NCCL パターンの適用

Agent B の NCCL 調査から、現在の RDMA バックエンドに直接適用可能な設計パターンが複数見つかった:

| NCCL パターン | 現在の実装 | 改善案 |
|-------------|-----------|--------|
| データ/コントロール QP 分離 | 単一 QP で全通信 | Per-device QP (データ) + 共有 QP (コントロール) |
| RDMA Write + IMM 完了通知 | Send/Recv (RNR リスクあり) | Write+IMM (RNR 排除) |
| Flush QP (GDR 順序保証) | なし (stale data 問題の原因の一つ) | 自己ループバック Read による PCIe barrier |
| Selective signaling | 全 Send が signaled | N 回に 1 回のみ signal |

---

## 最適化提案の総合評価

### Tier 1: 高効果・中実装コスト (推奨実施)

#### 提案 1: PD 共有 Per-device Connection

| 項目 | 詳細 |
|------|------|
| **概要** | 全接続で PD を共有し、MR の重複登録を排除。各デバイスが独立 QP を持つことで RPC と同等の並列性を実現 |
| **推定効果** | **+10-15%** (RPC との差 10% を埋める) |
| **実装複雑度** | 中 |
| **リスク** | 中 (PD 共有が `rdma_create_qp` で動作するか要検証) |
| **Agent A 評価** | 最高優先度 |
| **Agent B 評価** | 最高優先度 (A1) |

**実装方針**:
1. `rdma-transport.cpp:273` の `ibv_alloc_pd` を共有モードに変更
2. 最初の接続で PD を作成し、以降の接続で再利用
3. MR 登録は共有 PD で1回のみ実行
4. 各 QP は独立した CQ を持つ (CQ ロック競合回避)
5. サーバー側: 各 QP ごとにスレッドを割り当て (既存の multi-thread 実装を活用)

**コード変更箇所**:
- `rdma-transport.cpp:273` — PD 共有ロジック
- `rdma-transport.cpp:750-792` — 接続キャッシュの共有 PD 対応
- `ggml-rdma.cpp:2193-2263` — MR 管理の PD 共有対応
- サーバー側 `accept_connection` — 共有 PD での QP 作成

**検証すべき仮説**:
- 同一 `verbs` コンテキストから取得した PD を別の `cm_id` の QP に使用可能か
- PD 共有時の MTT エントリ数が実際に shared mode と同等か

#### 提案 2: Server-Push + RDMA Write with IMM

| 項目 | 詳細 |
|------|------|
| **概要** | get_tensor を pull (RDMA Read) から push (サーバー側 RDMA Write) に変更。完了通知は RDMA Write with IMM |
| **推定効果** | **+5-15%** (GDR 有効時。PCIe 帯域幅 3.4→9.8 GB/s + Send/Recv ラウンドトリップ削減) |
| **実装複雑度** | 高 |
| **リスク** | 中 (プロトコル変更、クライアント側バッファ管理) |
| **Agent A 評価** | 優先度 2 (C3 + B3 の統合) |
| **Agent B 評価** | 優先度 A2 |

**実装方針**:
1. クライアントがホストメモリに結果受信バッファを割り当て、MR 情報をサーバーに通知
2. サーバーは graph_compute 完了後 (`cudaDeviceSynchronize` 後):
   - 出力テンソルを RDMA Write でクライアントバッファに転送
   - RDMA Write with IMM (immediate data にサイズをエンコード) で完了通知
3. クライアントは IMM の CQE を poll して完了検知
4. `compute_pending_` フラグは不要になる (IMM 受信で完了を確認)

**NCCL との類似性**: NCCL のデータ転送プロトコルと同一パターン。NCCL で実証済みの設計。

**注意**: GDR 有効時は GPU VRAM から直接 RDMA Write するため帯域幅の改善が最大。GDR 無効時はホストステージング経由のため改善幅は小さい。

#### 提案 3: Hugepage ステージングバッファ

| 項目 | 詳細 |
|------|------|
| **概要** | ホストステージングバッファを 2MB hugepage 上に確保し、MTT エントリ数を 512 倍削減 |
| **推定効果** | **+5-13%** (GDR バジェット拡大 → GDR パス利用率向上) |
| **実装複雑度** | 低 |
| **リスク** | 低 (hugepage 事前確保が必要) |
| **Agent A 評価** | 直接提案なし (PD 共有に集中) |
| **Agent B 評価** | 優先度 A3 |

**実装方針**:
1. システム設定: `echo 20 > /proc/sys/vm/nr_hugepages` (2MB x 20 = 40MB 確保)
2. ステージングバッファ割り当てを `mmap(MAP_HUGETLB | MAP_HUGE_2MB)` に変更
3. `ibv_reg_mr` は変更不要 (hugepage 上のメモリを自動認識)
4. `GGML_RDMA_GDR_BUDGET_GB` のデフォルト値を引き上げ (12→48GB 等)

**補足**: GPU メモリ (cudaMalloc) のページサイズは NVIDIA ドライバが管理するため、GPU MR の MTT エントリ数はこの手法では削減できない。ホストステージングバッファのみが対象。ただし、ステージングバッファの MTT 圧力が減ることで、GPU MR に使える MTT キャッシュ容量が増加する間接効果がある。

### Tier 2: 中効果・低実装コスト (容易に実施可能)

#### 提案 4: Selective Signaling

| 項目 | 詳細 |
|------|------|
| **概要** | N 回の IB 操作につき 1 回のみ `IBV_SEND_SIGNALED` を付与。CQ DMA 書き込み回数を削減 |
| **推定効果** | **+1-3%** |
| **実装複雑度** | 低 |
| **リスク** | 低 (エラー検出遅延のリスクあり) |

**コード変更箇所**: `rdma-transport.cpp:441` — 全 Send の signaled フラグを条件付きに変更

#### 提案 5: WQE Linked List Batching

| 項目 | 詳細 |
|------|------|
| **概要** | 複数の RDMA 操作を `ibv_post_send` の linked list で1回のシステムコールにまとめる |
| **推定効果** | **+1-3%** |
| **実装複雑度** | 低 |
| **リスク** | 低 |

### Tier 3: 要検証 (実験的)

#### 提案 6: ODP (On-Demand Paging)

| 項目 | 詳細 |
|------|------|
| **概要** | MR 登録時に物理ページをピンしない。MTT キャッシュ問題の根本解決 |
| **推定効果** | MTT 問題の完全解消 (ただし ConnectX-4 の packet damming リスク) |
| **実装複雑度** | 中 |
| **リスク** | 高 (ConnectX-4 の性能劣化の可能性) |

#### 提案 7: NCCL Flush QP

| 項目 | 詳細 |
|------|------|
| **概要** | 自己ループバック RDMA Read による PCIe 順序保証。GDR stale data 対策 |
| **推定効果** | 信頼性向上 (性能改善は間接的) |
| **実装複雑度** | 中 |
| **リスク** | 低 |

---

## 過去の「効果なし」判定の再評価

| 過去の最適化 | 過去の判定 | 今回の再評価 |
|-------------|-----------|------------|
| Per-device connection (Plan C) | MTT 制限で使用不可 | **PD 共有で復活可能** (提案 1) |
| Async compute 完了検知 | compute_pending で Read パス使えず | **Server-Push で解消** (提案 2) |
| Doorbell batching | e2e 効果なし | **Multi-QP + WQE batch で効果見込み** (提案 5) |
| Two-phase loop (Plan A) | layer split でデータ依存 | **依然として効果なし** (変更なし) |
| synchronize | 効果なし | **依然として効果なし** (変更なし) |
| Speculative decoding | MoE で効果限定 | **依然として効果限定** (変更なし) |

---

## 推奨実施順序

```
Phase 1: PD 共有 Per-device Connection (提案 1)
  ├── 検証: PD 共有が ConnectX-4 で MTT 問題を解消するか
  ├── 実装: 共有 PD モードの接続管理
  └── 期待: +10-15% (RPC 同等化)

Phase 2: Hugepage ステージング (提案 3)
  ├── 設定: システムの hugepage 確保
  ├── 実装: mmap(MAP_HUGETLB) によるステージング割り当て
  └── 期待: GDR バジェット拡大 → +5-13%

Phase 3: Server-Push + Write with IMM (提案 2)
  ├── 実装: クライアント受信バッファ + サーバー側 RDMA Write push
  ├── 実装: IMM による完了通知
  └── 期待: +5-15% (PCIe 非対称性の活用)

Phase 4: 微細最適化 (提案 4, 5)
  ├── Selective signaling
  └── WQE batch posting
```

**Phase 1 と Phase 2 は独立しており並行実施可能。Phase 3 は Phase 1 の成果を前提とする。**

---

## 理論上の最大改善

全提案を組み合わせた場合の理論上の改善:

| 施策 | 改善 | 累積 |
|------|:----:|:----:|
| ベースライン (現在) | — | 6.8 t/s |
| + PD 共有 Multi-QP | +10% | 7.5 t/s |
| + Hugepage (GDR バジェット拡大) | +8% | 8.1 t/s |
| + Server-Push | +5% | 8.5 t/s |
| + 微細最適化 | +2% | 8.7 t/s |

**注意**: これは理論上の上限であり、各施策の効果が独立であることを前提としている。実際には効果の重複や新たなボトルネックの出現により、実効改善は小さくなる可能性がある。

---

## 結論

過去の「ソフトウェア最適化では解消不可」という結論は、**upstream 互換性の制約下では正しい**。しかし、以下の3つの新しい切り口により、互換性を壊す前提では改善の余地がある:

1. **PD 共有による Per-device Connection の復活**: MTT キャッシュ問題の回避策として、過去の分析では PD の共有という選択肢が検討されていなかった。IB 仕様上、同一 PD の MR は全 QP から参照可能であり、MR の重複登録が不要。

2. **P100 PCIe Read/Write 非対称性の活用**: get_tensor の方向を RDMA Read (3.4 GB/s) から RDMA Write (9.8 GB/s) に変更することで、帯域幅を 2.8 倍改善。これは Agent B の NVIDIA ベンチマーク論文調査で初めて定量的に明らかになった。

3. **Hugepage による MTT エントリ削減**: 2MB hugepage でエントリ数を 512 倍削減。ConnectX-4 の MTT キャッシュ制約を実質的に無力化できる可能性がある。

**最も重要な発見は、これらの施策が相互補完的であり、組み合わせることで相乗効果が期待できる点**である。PD 共有 Multi-QP で並列性を得つつ、Hugepage で MTT 圧力を軽減し、Server-Push で帯域幅を活用するという三層の最適化により、RPC を超える Generation 性能が理論上実現可能である。

---

## 用語解説

### InfiniBand / RDMA 基本概念

- **RDMA** (Remote Direct Memory Access): CPU やカーネルを介さずに、リモートマシンのメモリに直接読み書きする技術。通常の TCP/IP 通信ではデータがカーネルのネットワークスタックを経由して複数回コピーされるが、RDMA はカーネルバイパスとゼロコピーにより、低レイテンシ・高帯域を実現する。本プロジェクトでは、リモートノードの GPU で実行するテンソル計算のために、重みデータの送信 (set_tensor) や計算結果の取得 (get_tensor) に RDMA を使用している。

- **IB** (InfiniBand): データセンター向けの高速ネットワーク規格で、RDMA のネイティブサポートを特徴とする。Ethernet 上で RDMA を実現する RoCE (RDMA over Converged Ethernet) と比べ、専用スイッチによるロスレスファブリックやクレジットベースのフロー制御を備える。本環境では FDR (Fourteen Data Rate) 規格を使用しており、理論帯域幅は 56 Gbps (片方向 ~6.8 GB/s)。

- **RNIC** (RDMA Network Interface Card): RDMA プロトコル処理をハードウェアでオフロードする専用 NIC。通常の NIC とは異なり、トランスポート層の処理 (再送制御、順序保証等) やメモリアドレス変換 (MTT) をオンチップエンジンで実行する。このオンチップリソースには容量制限があり、特に MTT キャッシュの容量が本プロジェクトでの GDR バジェット制約の直接原因となっている。本環境では Mellanox ConnectX-4 を使用。

- **Verbs**: IB/RDMA のプログラミング API (`libibverbs`)。`ibv_post_send` (送信要求投入)、`ibv_post_recv` (受信バッファ投入)、`ibv_reg_mr` (メモリ領域登録) 等の関数群で構成される。ソケット API と比べて低レベルだが、ハードウェアに近い制御が可能。本バックエンドの `rdma-transport.cpp` が Verbs API を直接使用して全 RDMA 通信を実装している。

### IB リソース

- **PD** (Protection Domain): MR や QP をグループ化するセキュリティ境界。同一 PD に属する QP は、その PD に登録された全ての MR にアクセスできる。異なる PD の QP からは互いの MR にアクセスできないため、マルチテナント環境での分離に使われる。本レポートの「PD 共有 Multi-QP」提案 (提案 1) では、全接続で PD を共有することで MR の重複登録を排除し、MTT エントリの消費を接続数分の1に削減することを狙っている。

- **MR** (Memory Region): `ibv_reg_mr` によって RNIC に登録されたメモリ領域。登録によりカーネルが物理ページをピン (スワップアウト禁止) し、RNIC が DMA で直接アクセスできるようになる。登録時には MTT エントリを消費し (4KB ページあたり 1 エントリ)、大容量 MR ほど多くのエントリを使用する。本プロジェクトではホストステージングバッファ用 MR と、GPUDirect RDMA 用の GPU VRAM MR の 2 種類を使い分けている。

- **QP** (Queue Pair): IB 通信の基本単位で、Send Queue (SQ) と Receive Queue (RQ) のペアで構成される。TCP ソケットに類似するが、ステートマシン (RESET→INIT→RTR→RTS) による明示的な状態遷移が必要。本バックエンドでは現在、全リモートデバイスで 1 つの QP を共有 (shared connection) しているが、提案 1 ではデバイスごとに独立した QP を持つ per-device 方式を PD 共有で実現することを目指している。

- **CQ** (Completion Queue): IB 操作の完了通知を受け取るキュー。RNIC がハードウェア DMA で CQE (完了エントリ) を書き込み、CPU は `ibv_poll_cq` でポーリングして完了を検知する。CQ への DMA 書き込み自体が PCIe トラフィックを生むため、不要な通知を減らす Selective Signaling が最適化手法として有効。本レポートの提案 4 はこの削減を狙っている。

- **CQE** (Completion Queue Entry): CQ 内の個々の完了通知エントリ。操作の成否 (status)、操作種別 (opcode)、転送バイト数 (byte_len)、即値データ (imm_data) 等の情報を含む。RDMA Write with IMM の場合、受信側の CQE に imm_data が格納されるため、追加の制御メッセージなしにメタデータを伝達できる。本レポートの Server-Push 提案 (提案 2) では、この imm_data を計算完了通知に利用する。

- **WQE** (Work Queue Element): QP の SQ または RQ に投入する個々の作業要求。Send、Recv、RDMA Write、RDMA Read 等の操作種別と対象メモリ範囲を指定する。CPU が WQE を作成後、Doorbell (MMIO 書き込み) で RNIC に投入を通知する。`ibv_post_send` の linked list パラメータにより複数の WQE を 1 回の Doorbell で投入可能であり、本レポートの WQE Linked List Batching (提案 5) はこの仕組みを活用する。

- **MTT** (Memory Translation Table): RNIC が仮想アドレスを物理アドレスに変換するためのテーブル。OS のページテーブルに相当するが、RNIC のオンチップ SRAM にキャッシュされる。4KB ページでは 10GB の MR に約 260 万エントリが必要となり、キャッシュ容量 (ConnectX-4 で推定 10-16GB 相当) を超えるとキャッシュミスが頻発して性能が劣化する。本プロジェクトの GDR バジェットシステムはこのキャッシュ容量制限に基づいている。

### IB 操作

- **RDMA Write**: 送信側がリモートの MR に直接データを書き込む片側 (one-sided) 操作。受信側の CPU は一切関与しない。送信側はリモートの MR アドレスと rkey (リモートアクセスキー) を事前に知っている必要がある。本バックエンドでは set_tensor (重みデータのクライアント→サーバー転送) とステージングバッファのフラッシュに使用しており、P100 環境では RNIC→GPU 方向で最大 9.8 GB/s の帯域を達成する。

- **RDMA Read**: 送信側がリモートの MR からデータを直接読み取る片側操作。Write と同様に受信側 CPU の関与なし。ただし P100 の PCIe 実装では Read (GPU→RNIC) 方向の帯域が 3.4 GB/s と、Write 方向の 9.8 GB/s に対して約 1/3 に制限される (PCIe Read/Write 非対称性)。現在の get_tensor は RDMA Read を使用しており、この非対称性がボトルネックとなっている。

- **RDMA Write with IMM**: 通常の RDMA Write に 32bit の即値データ (immediate data) を付加する操作。通常の RDMA Write は受信側に通知されないが、IMM 付きの場合は受信側の CQ に CQE が生成され、imm_data フィールドにその 32bit 値が格納される。これにより、データ書き込みと完了通知を 1 回の操作で行える。本レポートの Server-Push 提案 (提案 2) では、サーバーが計算結果を RDMA Write でプッシュした後、Write with IMM で完了を通知する設計を採用している。

- **Send/Recv**: 送信側と受信側の双方の CPU が関与する両側 (two-sided) 操作。受信側は Send の到着前に `ibv_post_recv()` で受信バッファを投入しておく必要がある。片側操作と比べてレイテンシが高いが、任意のデータを交換できる柔軟性がある。本バックエンドではコマンドの送受信 (graph_compute 要求、set_tensor/get_tensor 要求等) に Send/Recv を使用している。

- **RNR NAK** (Receiver Not Ready): 受信側が `ibv_post_recv()` で受信バッファを投入していない状態で Send が到着した際に、RNIC が返す否定応答。送信側は `min_rnr_timer` で指定された時間待機してから再送する。デフォルト値 (timer=0) は 655.36ms の待機を意味し、3 回のリトライで約 2 秒のスパイクが発生する。本プロジェクトでは初期に 2 秒のレイテンシスパイクの原因となり、`min_rnr_timer=1` (0.01ms) に設定して解決した。

- **Selective Signaling**: N 回の操作につき 1 回のみ `IBV_SEND_SIGNALED` フラグを付与し、CQE の生成を抑制する最適化手法。CQE は RNIC が PCIe DMA でホストメモリに書き込むため、頻繁な CQE 生成は PCIe 帯域を消費する。ただし、signaled でない操作は完了確認ができないため、エラー検出が遅延するトレードオフがある。本レポートの提案 4 でこの手法を適用し +1-3% の改善を見込んでいる。

- **ODP** (On-Demand Paging): MR 登録時に物理メモリのピニングを行わず、RNIC がアクセスするたびにオンデマンドでページをマッピングする方式。ピニング不要のため MTT 管理が簡素化され、登録時のオーバーヘッドも削減される。ただし ConnectX-4 では「packet damming」(ページフォールト処理中にパケットが滞留する現象) のリスクがあり、大規模転送で性能劣化の可能性がある。本レポートの提案 6 (Tier 3) として検討対象。

- **Doorbell**: CPU が RNIC の MMIO レジスタに書き込むことで、新しい WQE の投入を通知する仕組み。1 回の Doorbell で linked list として接続された複数の WQE をまとめて通知できるため、Doorbell の回数を減らすことが微細最適化として有効。本レポートの提案 5 (WQE Linked List Batching) はこの Doorbell 統合を活用する。

### GPUDirect RDMA 関連

- **GDR** (GPUDirect RDMA): RNIC が PCIe バス経由で GPU の VRAM に直接 DMA アクセスする技術。通常のデータ転送では GPU→ホストメモリ→RNIC→ネットワークの 2 段コピーが必要だが、GDR ではホストメモリを経由しないゼロコピー転送を実現する。本プロジェクトでは set_tensor 時のクライアント→GPU VRAM 直接書き込みにより Prompt 処理で RPC 比 +19% の改善を達成した。ただし ConnectX-4 の MTT キャッシュ容量制限により、全 GPU バッファを GDR 登録するとタイムアウトが発生するため、GDR バジェットシステムで登録量を制限している。

- **PCIe** (Peripheral Component Interconnect Express): GPU と RNIC を接続するバス規格。デバイス間のデータ転送は全てこのバスを経由する。本環境の P100 は MaxPayload=256B (パケットあたりの最大ペイロード) がハードウェア上限であり、ConnectX-4 の DevCap 512B より小さいため、P100 側がボトルネックとなっている。また PCIe の Read/Write 方向で帯域幅に非対称性があり (RNIC→GPU Write: 9.8 GB/s vs GPU→RNIC Read: 3.4 GB/s)、これが get_tensor の性能制約となっている。

- **Staging buffer**: GDR が使えない場合 (MTT バジェット超過等) にホスト CPU メモリ上に確保する中間バッファ。データは GPU→ステージング→RNIC→ネットワーク (送信時) またはその逆 (受信時) の 2 段階で転送される。本バックエンドでは `RDMA_WRITE_MAX_BUFFER_SIZE` (4GB) を超えるバッファは RDMA Write ではなく Send/Recv フォールバックを使用し、ステージング経由で転送する。GDR と比べてホストメモリコピーのオーバーヘッドがあるが、MTT キャッシュ圧力は低い。

- **Hugepage**: OS の通常ページサイズ (4KB) より大きいページ (Linux では 2MB または 1GB)。MTT はページ単位でエントリを管理するため、2MB hugepage を使えば同じ 10GB のバッファで MTT エントリ数が ~260 万から ~5,120 に 512 倍削減される。本レポートの提案 3 ではホストステージングバッファを hugepage 上に確保することで MTT キャッシュ圧力を軽減し、GPU MR により多くのキャッシュ容量を割り当てることを狙っている。

- **GDR バジェット**: ConnectX-4 の MTT キャッシュオーバーフローを防ぐために、GPU VRAM の MR 登録総量に上限を設ける仕組み (環境変数 `GGML_RDMA_GDR_BUDGET_GB`)。上限を超えるバッファはホストステージングにフォールバックする。デフォルト値 12GB は ConnectX-4 の MTT キャッシュ容量推定 (~10-16GB) に基づく経験的な値。GLM-4.7 (4 RDMA × ~10GB) で発生した 30 秒タイムアウトがこのシステム導入で解消された。Hugepage (提案 3) で MTT 圧力が軽減されれば、バジェットの引き上げが可能になる。

### ハードウェア

- **ConnectX-4**: Mellanox (現 NVIDIA) の第 4 世代 RNIC。FDR/EDR InfiniBand および 100GbE をサポートする。本プロジェクトでは FDR 56Gbps で使用。`max_qp_rd_atom=16` (同時未完了 RDMA Read 数) をサポートするが、MTT キャッシュ容量が推定 10-16GB と限られており、大規模 MR 登録時の性能制約となっている。また ODP サポートはあるが packet damming のリスクがある。後継の ConnectX-6/7 ではオンチップキャッシュが拡大され、PCIe Gen4 で帯域幅も向上しているため、これらの制約は緩和される。

- **P100**: NVIDIA Tesla P100 GPU (Pascal アーキテクチャ)。16GB HBM2 VRAM を搭載し、本環境では PCIe 版を使用 (SXM2 版とは異なり NVLink 非対応)。PCIe の MaxPayload が 256B とハードウェア上限のため帯域効率が制約される。NVLink がないことから row split (テンソル並列) では GPU 間通信がボトルネックとなり、layer split (パイプライン並列) のみが実用的。本プロジェクトでは 2 ノード合計 16 台 (各ノード 8 台) の P100 クラスタを構成している。

### ソフトウェア / フレームワーク

- **NCCL** (NVIDIA Collective Communications Library): NVIDIA が提供する GPU 間集合通信ライブラリで、AllReduce や AllGather 等の分散学習向け通信プリミティブを実装している。内部実装が RDMA の高効率な使い方の参考になっており、本レポートでは PD 共有 Multi-QP、RDMA Write with IMM による完了通知、Flush QP による PCIe 順序保証などの設計パターンを参考にしている。ただし NCCL は集合通信 (全ノード協調) 向けであるのに対し、本 RDMA バックエンドは推論オフロード (1:N のクライアント-サーバー) 向けという用途の違いがある。

- **RPC**: llama.cpp に標準搭載されたリモート推論バックエンド。TCP ソケット (IB 上の IPoIB を含む) でクライアント-サーバー間を接続する。デバイスごとに独立したソケットを持つため、コマンド送信の並列性が高い (Generation 速度で RDMA 比 +10% の優位)。一方で TCP/IP のカーネルスタックを経由するため、帯域幅効率では RDMA に劣る (Prompt 処理で RDMA 比 -19%)。本 RDMA バックエンドは RPC を性能面で上回ることを目標の一つとしている。

- **MoE** (Mixture of Experts): 入力トークンに応じてモデルの一部のエキスパート層のみを活性化するアーキテクチャ。全パラメータを使う Dense モデルと比べ、同じパラメータ数でも推論時の計算量が少ない。本環境で使用する GLM-4.7 (約 90B パラメータ) がこの方式を採用している。MoE ではトークンごとに異なるエキスパートが選択されるため、クロスデバイスコピー (あるデバイスの計算結果を別デバイスへ転送) の頻度が Dense モデルより高く、Deferred copy の効果が MoE モデルで顕著に現れる理由となっている。

### RDMA バックエンド固有

- **set_tensor**: ggml バックエンドインタフェースの一部で、クライアントからサーバーへテンソルデータを転送する操作。主にモデルの重みデータをリモート GPU に送る際に使用される。小さいバッファ (<4GB) では RDMA Write でサーバーのステージングバッファに直接書き込み、大きいバッファや GDR 有効時は Send/Recv にフォールバックする。Prompt 処理における RDMA の RPC 比 +19% 優位はこの RDMA Write ゼロコピー転送によるもの。

- **get_tensor**: ggml バックエンドインタフェースの一部で、サーバーからクライアントへテンソルデータを取得する操作。主に graph_compute 後の計算結果 (logits 等) を取り戻す際に使用される。GDR MR からは RDMA Read (pull 方式) で直接取得するが、P100 の PCIe Read/Write 非対称性 (3.4 vs 9.8 GB/s) により帯域が制限されている。本レポートの Server-Push 提案 (提案 2) は、この pull を push (サーバー側 RDMA Write) に変更して帯域幅を約 2.8 倍改善することを目指している。

- **graph_compute**: ggml バックエンドインタフェースの一部で、計算グラフの実行をサーバー側 GPU に指示する操作。現在は fire-and-forget 方式 (async compute) で、クライアントはコマンド送信後にサーバーの計算完了を待たずに次の操作に進む。サーバー側では GPU が CUDA カーネルを実行し、完了は後続の get_tensor 時に暗黙的に同期される。サーバー側の parallel compute dispatch により、複数デバイスへの計算指示がワーカースレッドで並列処理される。

- **Per-device connection**: リモートデバイスごとに独立した RDMA 接続 (QP) を確立する方式。RPC がデバイスごとに独立ソケットを持つのと同等の並列性を実現する。shared connection (全デバイスで 1 QP を共有) と比べてコマンド送信を並列化できるが、接続ごとに PD と MR が独立するため MTT エントリが接続数倍に増加する。ConnectX-4 では 4 接続 × ~10GB MR で MTT オーバーフローが発生した。本レポートの提案 1 (PD 共有 Multi-QP) は、PD を全接続で共有して MR 重複登録を排除し、この MTT 問題を解消するものである。

- **Deferred copy**: 異なるリモートデバイス間のテンソルコピー (cpy_tensor) をサーバー内部の D2H+H2D (GPU→ホスト→GPU) で実行する最適化。通常はクライアントが get_tensor + set_tensor で IB ネットワークを 2 回往復する必要があるが、Deferred copy ではサーバー内部の PCIe バスのみでコピーが完結するためネットワーク往復が不要。MoE モデル (GLM-4.7) ではエキスパート間のクロスデバイスコピーが頻繁に発生するため +1.45% の改善効果が確認されている。

- **Flush QP**: NCCL が採用している手法で、自分自身へのループバック RDMA Read を発行することで PCIe の書き込み順序を保証する。GPU VRAM への RDMA Write は PCIe ポステッド書き込みとして処理され、完了通知が返っても実際のデータ書き込みが完了していない可能性がある。Flush QP の Read 要求は PCIe の順序保証規則により先行する全ての Write の完了後に処理されるため、Read の完了で全 Write のコミットが保証される。GDR 環境での stale data 問題への根本対策として本レポートの提案 7 で検討している。
