#!/bin/bash
set -euo pipefail

NODE2="192.168.100.2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="display"
if [[ "${1:-}" == "--markdown" ]]; then
    MODE="markdown"
fi

WARNINGS=()

warn() {
    WARNINGS+=("$1")
}

section() {
    if [[ "$MODE" == "display" ]]; then
        echo ""
        echo "=== $1 ==="
    fi
}

ssh_cmd() {
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$NODE2" "$1" 2>/dev/null
}

check_git() {
    local commit branch dirty=""
    commit=$(git -C "$WORKTREE_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
    branch=$(git -C "$WORKTREE_DIR" branch --show-current 2>/dev/null || echo "unknown")
    if [[ -n $(git -C "$WORKTREE_DIR" status --porcelain 2>/dev/null) ]]; then
        dirty=" (dirty)"
    fi
    GIT_INFO="${commit}${dirty} (${branch})"
}

check_gpu() {
    local host="$1"
    local prefix="$2"
    local cmd='nvidia-smi --query-gpu=name,driver_version,temperature.gpu,memory.used,memory.total,count --format=csv,noheader,nounits'

    local output
    if [[ "$host" == "local" ]]; then
        output=$(eval "$cmd" 2>/dev/null) || { eval "${prefix}_AVAILABLE=no"; return; }
    else
        output=$(ssh_cmd "$cmd" 2>/dev/null) || { eval "${prefix}_AVAILABLE=no"; return; }
    fi

    eval "${prefix}_AVAILABLE=yes"

    local gpu_name driver_ver max_temp gpu_count
    gpu_name=$(echo "$output" | head -1 | cut -d',' -f1 | xargs)
    driver_ver=$(echo "$output" | head -1 | cut -d',' -f2 | xargs)
    max_temp=$(echo "$output" | cut -d',' -f3 | xargs | tr '\n' ' ' | tr -s ' ' | awk '{m=0; for(i=1;i<=NF;i++) if($i+0>m) m=$i+0; print m}')
    gpu_count=$(echo "$output" | wc -l | xargs)

    eval "${prefix}_NAME='${gpu_count}x ${gpu_name}'"
    eval "${prefix}_DRIVER='${driver_ver}'"
    eval "${prefix}_MAX_TEMP='${max_temp}'"

    if [[ "$max_temp" -gt 70 ]]; then
        local node_label="1号機"
        [[ "$host" != "local" ]] && node_label="2号機"
        warn "${node_label} GPU温度が高い: ${max_temp}°C (> 70°C)"
    fi
}

check_gpu_power() {
    local host="$1"
    local prefix="$2"
    local cmd='nvidia-smi --query-gpu=power.limit,clocks.max.sm,persistence_mode --format=csv,noheader,nounits'

    local output
    if [[ "$host" == "local" ]]; then
        output=$(eval "$cmd" 2>/dev/null) || return
    else
        output=$(ssh_cmd "$cmd" 2>/dev/null) || return
    fi

    local power_limit max_sm persistence
    power_limit=$(echo "$output" | head -1 | cut -d',' -f1 | xargs)
    max_sm=$(echo "$output" | head -1 | cut -d',' -f2 | xargs)
    persistence=$(echo "$output" | head -1 | cut -d',' -f3 | xargs)

    eval "${prefix}_POWER_LIMIT='${power_limit}W'"
    eval "${prefix}_MAX_SM='${max_sm} MHz'"
    eval "${prefix}_PERSISTENCE='${persistence}'"
}

check_gdr() {
    local host="$1"
    local prefix="$2"
    local cmd='lsmod | grep nvidia_peermem'

    local output
    if [[ "$host" == "local" ]]; then
        output=$(eval "$cmd" 2>/dev/null) || true
    else
        output=$(ssh_cmd "$cmd" 2>/dev/null) || true
    fi

    if [[ -n "$output" ]]; then
        eval "${prefix}_GDR=loaded"
    else
        eval "${prefix}_GDR=not_loaded"
        local node_label="1号機"
        [[ "$host" != "local" ]] && node_label="2号機"
        warn "${node_label} nvidia-peermem が未ロード"
    fi
}

check_rdma() {
    local host="$1"
    local prefix="$2"
    local cmd='ibstat mlx5_0 2>/dev/null | grep -E "State:|Rate:" | head -2'

    local output
    if [[ "$host" == "local" ]]; then
        output=$(eval "$cmd" 2>/dev/null) || { eval "${prefix}_RDMA='not available'"; return; }
    else
        output=$(ssh_cmd "$cmd" 2>/dev/null) || { eval "${prefix}_RDMA='not available'"; return; }
    fi

    local state rate
    state=$(echo "$output" | grep "State:" | head -1 | awk '{print $NF}')
    rate=$(echo "$output" | grep "Rate:" | head -1 | awk '{print $NF}')

    eval "${prefix}_RDMA='mlx5_0 (${state}, ${rate})'"

    if [[ "$state" != "Active" ]]; then
        local node_label="1号機"
        [[ "$host" != "local" ]] && node_label="2号機"
        warn "${node_label} IB リンクが Active でない: ${state}"
    fi
}

check_p2p() {
    local host="$1"
    local prefix="$2"
    local cmd='nvidia-smi topo -m 2>/dev/null | head -1'

    local output
    if [[ "$host" == "local" ]]; then
        output=$(nvidia-smi topo -m 2>/dev/null | grep -oP '(PHB|NV\d*|PIX|PXB|SYS|NODE)' | sort -u | tr '\n' '/' | sed 's/\/$//' 2>/dev/null) || { eval "${prefix}_P2P='unknown'"; return; }
    else
        output=$(ssh_cmd "nvidia-smi topo -m 2>/dev/null | grep -oP '(PHB|NV[0-9]*|PIX|PXB|SYS|NODE)' | sort -u | tr '\n' '/' | sed 's/\\/\$//'" 2>/dev/null) || { eval "${prefix}_P2P='unknown'"; return; }
    fi

    if [[ -z "$output" ]]; then
        eval "${prefix}_P2P='unknown'"
    else
        eval "${prefix}_P2P='${output}'"
    fi
}

check_acs() {
    local host="$1"
    local prefix="$2"
    local cmd='for dev in $(lspci -D | grep -i "PCI bridge" | cut -d" " -f1); do val=$(setpci -s "$dev" ECAP_ACS+6.w 2>/dev/null); if [ -n "$val" ] && [ "$val" != "0000" ]; then echo "enabled:$dev:$val"; fi; done'

    local output
    if [[ "$host" == "local" ]]; then
        output=$(eval "$cmd" 2>/dev/null) || true
    else
        output=$(ssh_cmd "$cmd" 2>/dev/null) || true
    fi

    if [[ -n "$output" ]]; then
        eval "${prefix}_ACS=enabled"
        local node_label="1号機"
        [[ "$host" != "local" ]] && node_label="2号機"
        warn "${node_label} ACS が有効 (GPUDirect P2P を阻害する可能性)"
    else
        eval "${prefix}_ACS=disabled"
    fi
}

check_iommu() {
    local host="$1"
    local prefix="$2"
    local cmd='cat /proc/cmdline'

    local output
    if [[ "$host" == "local" ]]; then
        output=$(eval "$cmd" 2>/dev/null) || true
    else
        output=$(ssh_cmd "$cmd" 2>/dev/null) || true
    fi

    if echo "$output" | grep -qE 'iommu=(on|force|pt)' 2>/dev/null; then
        eval "${prefix}_IOMMU=enabled"
        local node_label="1号機"
        [[ "$host" != "local" ]] && node_label="2号機"
        warn "${node_label} IOMMU が有効 (GPUDirect RDMA パフォーマンスに影響)"
    else
        eval "${prefix}_IOMMU=disabled"
    fi
}

check_env_vars() {
    ENV_VARS=$(env | grep '^GGML_RDMA' | sort 2>/dev/null) || true
}

check_server() {
    local output
    output=$(ssh_cmd "pgrep -a rdma-server" 2>/dev/null) || true

    if [[ -n "$output" ]]; then
        local pid
        pid=$(echo "$output" | head -1 | awk '{print $1}')
        SERVER_STATUS="running (PID ${pid})"
    else
        SERVER_STATUS="not running"
        warn "2号機 rdma-server が未起動"
    fi
}

output_display() {
    section "Git"
    echo "  Commit: $GIT_INFO"

    section "GPU (1号機)"
    if [[ "${LOCAL_AVAILABLE:-}" == "yes" ]]; then
        echo "  GPU:         $LOCAL_NAME"
        echo "  Driver:      $LOCAL_DRIVER"
        echo "  Max Temp:    ${LOCAL_MAX_TEMP}°C"
        echo "  Power Limit: ${LOCAL_POWER_LIMIT:-N/A}"
        echo "  Max SM:      ${LOCAL_MAX_SM:-N/A}"
        echo "  Persistence: ${LOCAL_PERSISTENCE:-N/A}"
    else
        echo "  nvidia-smi not available"
    fi

    section "GPU (2号機)"
    if [[ "${REMOTE_AVAILABLE:-}" == "yes" ]]; then
        echo "  GPU:         $REMOTE_NAME"
        echo "  Driver:      $REMOTE_DRIVER"
        echo "  Max Temp:    ${REMOTE_MAX_TEMP}°C"
        echo "  Power Limit: ${REMOTE_POWER_LIMIT:-N/A}"
        echo "  Max SM:      ${REMOTE_MAX_SM:-N/A}"
        echo "  Persistence: ${REMOTE_PERSISTENCE:-N/A}"
    else
        echo "  nvidia-smi not available"
    fi

    section "GDR (nvidia-peermem)"
    echo "  1号機: ${LOCAL_GDR:-unknown}"
    echo "  2号機: ${REMOTE_GDR:-unknown}"

    section "RDMA"
    echo "  1号機: ${LOCAL_RDMA:-unknown}"
    echo "  2号機: ${REMOTE_RDMA:-unknown}"

    section "P2P Topology"
    echo "  1号機: ${LOCAL_P2P:-unknown}"
    echo "  2号機: ${REMOTE_P2P:-unknown}"

    section "ACS"
    echo "  1号機: ${LOCAL_ACS:-unknown}"
    echo "  2号機: ${REMOTE_ACS:-unknown}"

    section "IOMMU"
    echo "  1号機: ${LOCAL_IOMMU:-unknown}"
    echo "  2号機: ${REMOTE_IOMMU:-unknown}"

    section "rdma-server (2号機)"
    echo "  Status: $SERVER_STATUS"

    section "GGML_RDMA 環境変数"
    if [[ -n "${ENV_VARS:-}" ]]; then
        echo "$ENV_VARS" | while read -r line; do
            echo "  $line"
        done
    else
        echo "  (none)"
    fi

    if [[ ${#WARNINGS[@]} -gt 0 ]]; then
        echo ""
        echo "==============================="
        echo "  WARNINGS (${#WARNINGS[@]})"
        echo "==============================="
        for w in "${WARNINGS[@]}"; do
            echo "  [!] $w"
        done
    else
        echo ""
        echo "All checks passed."
    fi
}

output_markdown() {
    echo "## 環境情報"
    echo ""
    echo "| 項目 | 1号機 | 2号機 |"
    echo "|------|-------|-------|"
    echo "| Git commit | \`${GIT_INFO}\` | — |"

    if [[ "${LOCAL_AVAILABLE:-}" == "yes" ]]; then
        echo "| NVIDIA Driver | ${LOCAL_DRIVER} | ${REMOTE_DRIVER:-N/A} |"
        echo "| GPU | ${LOCAL_NAME} | ${REMOTE_NAME:-N/A} |"
        echo "| GPU 温度 (max) | ${LOCAL_MAX_TEMP}°C | ${REMOTE_MAX_TEMP:-N/A}°C |"
        echo "| 電力制限 | ${LOCAL_POWER_LIMIT:-N/A} | ${REMOTE_POWER_LIMIT:-N/A} |"
        echo "| SM 周波数 (max) | ${LOCAL_MAX_SM:-N/A} | ${REMOTE_MAX_SM:-N/A} |"
        echo "| Persistence Mode | ${LOCAL_PERSISTENCE:-N/A} | ${REMOTE_PERSISTENCE:-N/A} |"
    fi

    echo "| nvidia-peermem | ${LOCAL_GDR:-unknown} | ${REMOTE_GDR:-unknown} |"
    echo "| IB デバイス | ${LOCAL_RDMA:-unknown} | ${REMOTE_RDMA:-unknown} |"
    echo "| P2P トポロジ | ${LOCAL_P2P:-unknown} | ${REMOTE_P2P:-unknown} |"
    echo "| ACS | ${LOCAL_ACS:-unknown} | ${REMOTE_ACS:-unknown} |"
    echo "| IOMMU | ${LOCAL_IOMMU:-unknown} | ${REMOTE_IOMMU:-unknown} |"
    echo "| rdma-server | — | ${SERVER_STATUS} |"

    echo ""
    echo "GGML_RDMA 環境変数:"
    if [[ -n "${ENV_VARS:-}" ]]; then
        echo "$ENV_VARS" | while read -r line; do
            echo "- \`${line}\`"
        done
    else
        echo "- (none)"
    fi

    if [[ ${#WARNINGS[@]} -gt 0 ]]; then
        echo ""
        echo "**警告:**"
        for w in "${WARNINGS[@]}"; do
            echo "- ${w}"
        done
    fi
}

check_git

check_gpu "local" "LOCAL"
check_gpu "$NODE2" "REMOTE"
check_gpu_power "local" "LOCAL"
check_gpu_power "$NODE2" "REMOTE"
check_gdr "local" "LOCAL"
check_gdr "$NODE2" "REMOTE"
check_rdma "local" "LOCAL"
check_rdma "$NODE2" "REMOTE"
check_p2p "local" "LOCAL"
check_p2p "$NODE2" "REMOTE"
check_acs "local" "LOCAL"
check_acs "$NODE2" "REMOTE"
check_iommu "local" "LOCAL"
check_iommu "$NODE2" "REMOTE"
check_env_vars
check_server

if [[ "$MODE" == "markdown" ]]; then
    output_markdown
else
    output_display
fi
