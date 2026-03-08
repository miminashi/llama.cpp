#!/bin/bash
set -euo pipefail

# HF_TOKEN loaded from .env
export HF_HUB_ENABLE_HF_TRANSFER=1

exec /home/ubuntu/projects/llama.cpp/.venv/hf/bin/hf "$@"
