#!/bin/bash
set -euo pipefail

NODE2="192.168.100.2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE_DIR="/home/ubuntu/projects/llama.cpp"
CMAKE_OPTS="-DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=60"

usage() {
    echo "Usage: $0 {local|remote|all}"
    echo "  local   - Build on node 1 (this machine)"
    echo "  remote  - Build on node 2 ($NODE2)"
    echo "  all     - Build on both nodes"
    exit 1
}

build_local() {
    echo "=== Building on node 1 (local) ==="
    cd "$WORKTREE_DIR"
    rm -rf build
    cmake -B build $CMAKE_OPTS > /tmp/cmake_configure.log 2>&1 && echo "CMAKE CONFIGURE OK" || { echo "CMAKE CONFIGURE FAILED"; cat /tmp/cmake_configure.log; exit 1; }
    cmake --build build -- -j "$(nproc)" > /tmp/build.log 2>&1 && echo "BUILD OK" || { echo "BUILD FAILED"; tail -50 /tmp/build.log; exit 1; }
    echo "=== Node 1 build complete ==="
}

build_remote() {
    echo "=== Building on node 2 ($NODE2) ==="
    ssh "$NODE2" "cd ${REMOTE_DIR} && rm -rf build && cmake -B build $CMAKE_OPTS > /tmp/cmake_configure.log 2>&1 && echo 'CMAKE CONFIGURE OK' || { echo 'CMAKE CONFIGURE FAILED'; exit 1; } && cmake --build build -- -j \$(nproc) > /tmp/build.log 2>&1 && echo 'BUILD OK' || { echo 'BUILD FAILED'; exit 1; }"
    echo "=== Node 2 build complete ==="
}

[ $# -lt 1 ] && usage

case "$1" in
    local)  build_local ;;
    remote) build_remote ;;
    all)    build_local && build_remote ;;
    *)      usage ;;
esac
