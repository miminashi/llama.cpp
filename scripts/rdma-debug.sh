#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GDBINIT="${SCRIPT_DIR}/gdbinit-rdma"

usage() {
    echo "Usage: $0 {cli|bench|custom} [args...]"
    echo "  cli   [args...]    - Debug llama-cli with gdb"
    echo "  bench [args...]    - Debug llama-bench with gdb"
    echo "  custom <bin> [args...] - Debug arbitrary binary with gdb"
    echo ""
    echo "Environment variables (GGML_RDMA_SERVERS, etc.) are inherited."
    echo "Tip: set GGML_RDMA_TIMEOUT_MS=120000 for debug sessions."
    exit 1
}

[ $# -lt 1 ] && usage

MODE="$1"
shift

export LD_LIBRARY_PATH="${WORKTREE_DIR}/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ulimit -c unlimited 2>/dev/null || true

case "$MODE" in
    cli)
        BINARY="${WORKTREE_DIR}/build/bin/llama-cli"
        ;;
    bench)
        BINARY="${WORKTREE_DIR}/build/bin/llama-bench"
        ;;
    custom)
        if [ $# -lt 1 ]; then
            echo "Error: custom mode requires a binary path"
            usage
        fi
        BINARY="$1"
        shift
        ;;
    *)
        usage
        ;;
esac

if [ ! -f "$BINARY" ]; then
    echo "Error: binary not found: $BINARY"
    echo "Did you build with: bash scripts/rdma-build.sh local debug"
    exit 1
fi

if [ ! -f "$GDBINIT" ]; then
    echo "Warning: gdbinit-rdma not found at $GDBINIT"
fi

exec gdb -x "$GDBINIT" --args "$BINARY" "$@"
