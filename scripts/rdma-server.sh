#!/bin/bash
set -euo pipefail

NODE2="192.168.100.2"
PROJECT_DIR="/home/ubuntu/projects/llama.cpp"
HOST="0.0.0.0"
PORT="50051"

GDBINIT_REMOTE="${PROJECT_DIR}/scripts/gdbinit-rdma"

usage() {
    echo "Usage: $0 {start|stop|restart|status|log|debug|attach}"
    echo "  start   - Start rdma-server on node 2"
    echo "  stop    - Stop rdma-server on node 2"
    echo "  restart - Restart rdma-server on node 2"
    echo "  status  - Check if rdma-server is running on node 2"
    echo "  log     - Show recent rdma-server log from node 2"
    echo "  debug   - Start rdma-server under gdb on node 2 (foreground)"
    echo "  attach  - Attach gdb to running rdma-server on node 2"
    exit 1
}

server_start() {
    local cuda_vis_env=""
    if [ -n "${RDMA_SERVER_DEVICES:-}" ]; then
        local indices=""
        IFS=',' read -ra devs <<< "${RDMA_SERVER_DEVICES}"
        for dev in "${devs[@]}"; do
            local idx="${dev#CUDA}"
            if [ -n "$indices" ]; then indices="${indices},${idx}"; else indices="${idx}"; fi
        done
        cuda_vis_env="CUDA_VISIBLE_DEVICES=${indices}"
        echo "Starting rdma-server on $NODE2 (${cuda_vis_env})..."
    else
        echo "Starting rdma-server on $NODE2 (all devices)..."
    fi
    ssh "$NODE2" "LD_LIBRARY_PATH=${PROJECT_DIR}/build/bin ${cuda_vis_env} nohup ${PROJECT_DIR}/build/bin/rdma-server -H $HOST -p $PORT > /tmp/rdma-server.log 2>&1 &"
    sleep 2
    server_status
}

server_stop() {
    echo "Stopping rdma-server (port $PORT) on $NODE2..."
    ssh "$NODE2" "pkill -f 'rdma-server.*-p ${PORT}' || true"
    sleep 1
    server_status
}

server_restart() {
    server_stop
    sleep 1
    server_start
}

server_status() {
    if ssh "$NODE2" "pgrep -a rdma-server" 2>/dev/null; then
        echo "SERVER_RUNNING"
    else
        echo "SERVER_NOT_RUNNING"
    fi
}

server_log() {
    ssh "$NODE2" "tail -50 /tmp/rdma-server.log 2>/dev/null || echo 'No log file found'"
}

server_debug() {
    local cuda_vis_env=""
    if [ -n "${RDMA_SERVER_DEVICES:-}" ]; then
        local indices=""
        IFS=',' read -ra devs <<< "${RDMA_SERVER_DEVICES}"
        for dev in "${devs[@]}"; do
            local idx="${dev#CUDA}"
            if [ -n "$indices" ]; then indices="${indices},${idx}"; else indices="${idx}"; fi
        done
        cuda_vis_env="CUDA_VISIBLE_DEVICES=${indices}"
    fi
    echo "Starting rdma-server under gdb on $NODE2 (foreground, interactive)..."
    echo "Note: this requires an interactive terminal (ssh -t)"
    ssh -t "$NODE2" "${cuda_vis_env} LD_LIBRARY_PATH=${PROJECT_DIR}/build/bin gdb -x ${GDBINIT_REMOTE} --args ${PROJECT_DIR}/build/bin/rdma-server -H $HOST -p $PORT"
}

server_attach() {
    echo "Attaching gdb to running rdma-server on $NODE2..."
    ssh -t "$NODE2" "PID=\$(pgrep -f rdma-server | head -1) && echo \"Attaching to PID=\$PID\" && LD_LIBRARY_PATH=${PROJECT_DIR}/build/bin gdb -x ${GDBINIT_REMOTE} -p \$PID || echo 'rdma-server not running'"
}

[ $# -lt 1 ] && usage

case "$1" in
    start)   server_start ;;
    stop)    server_stop ;;
    restart) server_restart ;;
    status)  server_status ;;
    log)     server_log ;;
    debug)   server_debug ;;
    attach)  server_attach ;;
    *)       usage ;;
esac
