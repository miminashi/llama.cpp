#!/bin/bash
set -euo pipefail

LOCK_DIR="/tmp/rdma-workflow"
LOCK_FILE="$LOCK_DIR/gpu.lock"
INFO_FILE="$LOCK_DIR/gpu.info"

mkdir -p "$LOCK_DIR"

usage() {
    echo "Usage: $0 {status|run|wait} [options] [command...]"
    echo "  status                     Show lock state"
    echo "  run <command...>           Non-blocking: acquire lock, run, release (fail if locked)"
    echo "  wait [--timeout N] <cmd>   Blocking: wait for lock, run, release"
    echo "    --timeout N              Max wait seconds (default: unlimited)"
    exit 1
}

write_info() {
    local cmd_str="$*"
    echo "pid=$$" > "$INFO_FILE"
    echo "since=$(date '+%Y-%m-%d %H:%M:%S')" >> "$INFO_FILE"
    echo "cmd=${cmd_str:0:200}" >> "$INFO_FILE"
}

clear_info() {
    rm -f "$INFO_FILE"
}

cmd_status() {
    if [ ! -f "$LOCK_FILE" ]; then
        echo "GPU_UNLOCKED"
        return 0
    fi
    exec 9<"$LOCK_FILE"
    if flock -n 9; then
        flock -u 9
        exec 9<&-
        rm -f "$INFO_FILE"
        echo "GPU_UNLOCKED"
        return 0
    fi
    exec 9<&-
    if [ -f "$INFO_FILE" ]; then
        local pid since cmd
        pid=$(grep '^pid=' "$INFO_FILE" 2>/dev/null | cut -d= -f2- || echo "unknown")
        since=$(grep '^since=' "$INFO_FILE" 2>/dev/null | cut -d= -f2- || echo "unknown")
        cmd=$(grep '^cmd=' "$INFO_FILE" 2>/dev/null | cut -d= -f2- || echo "unknown")
        echo "GPU_LOCKED (pid=$pid, since $since, cmd: $cmd)"
    else
        echo "GPU_LOCKED (no info available)"
    fi
    return 0
}

cmd_run() {
    if [ $# -eq 0 ]; then
        echo "Error: no command specified" >&2
        usage
    fi
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        echo "Error: GPU is locked" >&2
        cmd_status >&2
        exec 9<&-
        return 1
    fi
    trap 'clear_info; flock -u 9; exec 9<&-' EXIT
    write_info "$@"
    "$@"
}

cmd_wait() {
    local timeout=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --timeout)
                timeout="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    if [ $# -eq 0 ]; then
        echo "Error: no command specified" >&2
        usage
    fi
    exec 9>"$LOCK_FILE"
    if [ -n "$timeout" ]; then
        if ! flock -w "$timeout" 9; then
            echo "Error: timed out waiting for GPU lock (${timeout}s)" >&2
            cmd_status >&2
            exec 9<&-
            return 1
        fi
    else
        flock 9
    fi
    trap 'clear_info; flock -u 9; exec 9<&-' EXIT
    write_info "$@"
    "$@"
}

case "${1:-}" in
    status) cmd_status ;;
    run)    shift; cmd_run "$@" ;;
    wait)   shift; cmd_wait "$@" ;;
    *)      usage ;;
esac
