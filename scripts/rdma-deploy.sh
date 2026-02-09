#!/bin/bash
set -euo pipefail

NODE2="192.168.100.2"
PROJECT_DIR="/home/ubuntu/projects/llama.cpp"
WORKTREE_DIR="${PROJECT_DIR}/.worktree/rdma-backend"
CMAKE_OPTS="-DGGML_RDMA=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=60"

echo "=== Deploying to node 2 ($NODE2) ==="

echo "[1/3] Removing old code on node 2..."
ssh "$NODE2" "rm -rf ${PROJECT_DIR}"

echo "[2/3] Syncing code to node 2..."
rsync -a --exclude='.git' "$WORKTREE_DIR/" "$NODE2:${PROJECT_DIR}/"

echo "[3/3] Building on node 2..."
ssh "$NODE2" "cd ${PROJECT_DIR} && rm -rf build && cmake -B build $CMAKE_OPTS > /tmp/cmake_configure.log 2>&1 && echo 'CMAKE CONFIGURE OK' || { echo 'CMAKE CONFIGURE FAILED'; exit 1; } && cmake --build build -- -j \$(nproc) > /tmp/build.log 2>&1 && echo 'BUILD OK' || { echo 'BUILD FAILED'; exit 1; }"

echo "=== Deploy complete ==="
