#!/usr/bin/env bash
set -euo pipefail

HF_CACHE="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"

usage() {
    echo "Usage: $(basename "$0") [--gguf]"
    echo ""
    echo "Options:"
    echo "  --gguf    Output only GGUF file absolute paths (for -m flag)"
    echo "  --help    Show this help"
    exit 0
}

mode="table"
if [[ "${1:-}" == "--gguf" ]]; then
    mode="gguf"
elif [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    usage
fi

if [[ ! -d "$HF_CACHE" ]]; then
    echo "HF cache not found: $HF_CACHE" >&2
    exit 1
fi

model_dirs=()
while IFS= read -r d; do
    model_dirs+=("$d")
done < <(find "$HF_CACHE" -maxdepth 1 -type d -name 'models--*' | sort)

if [[ ${#model_dirs[@]} -eq 0 ]]; then
    echo "No models found in $HF_CACHE" >&2
    exit 0
fi

if [[ "$mode" == "gguf" ]]; then
    for model_dir in "${model_dirs[@]}"; do
        snap_dir="$model_dir/snapshots"
        [[ -d "$snap_dir" ]] || continue
        latest=$(ls -t "$snap_dir" | head -1)
        [[ -n "$latest" ]] || continue
        snap_path="$snap_dir/$latest"
        while IFS= read -r f; do
            echo "$f"
        done < <(find "$snap_path" -name '*.gguf' \( -type f -o -type l \) | sort)
    done
    exit 0
fi

printf "%-40s %8s   %-50s %s\n" "REPO" "SIZE" "GGUF FILES" "PATH"
printf "%-40s %8s   %-50s %s\n" "----" "----" "----------" "----"

for model_dir in "${model_dirs[@]}"; do
    dir_name=$(basename "$model_dir")
    repo_name=${dir_name#models--}
    repo_name=${repo_name//--/\/}

    snap_dir="$model_dir/snapshots"
    if [[ ! -d "$snap_dir" ]]; then
        printf "%-40s %8s   %-50s %s\n" "$repo_name" "(no dl)" "" ""
        continue
    fi

    latest=$(ls -t "$snap_dir" | head -1)
    if [[ -z "$latest" ]]; then
        printf "%-40s %8s   %-50s %s\n" "$repo_name" "(empty)" "" ""
        continue
    fi

    snap_path="$snap_dir/$latest"
    short_hash="${latest:0:6}..."
    display_path="$snap_dir/$short_hash"

    dir_size=$({ du -shL "$snap_path" 2>/dev/null || echo "???"; } | cut -f1)

    gguf_files=()
    while IFS= read -r f; do
        rel="${f#"$snap_path/"}"
        gguf_files+=("$rel")
    done < <(find "$snap_path" -name '*.gguf' \( -type f -o -type l \) | sort)

    gguf_count=${#gguf_files[@]}
    if [[ $gguf_count -eq 0 ]]; then
        gguf_display="(no .gguf files)"
    elif [[ $gguf_count -eq 1 ]]; then
        gguf_display="${gguf_files[0]}"
    else
        gguf_display="${gguf_files[0]} (+$((gguf_count - 1)) files)"
    fi

    printf "%-40s %8s   %-50s %s\n" "$repo_name" "$dir_size" "$gguf_display" "$display_path"
done
