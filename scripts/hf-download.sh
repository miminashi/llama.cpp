#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
exec /home/ubuntu/projects/llama.cpp/.venv/hf/bin/hf "$@"
