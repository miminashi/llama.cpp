#!/bin/bash
set -euo pipefail

DATA_DIR="/tmp/rdma-workflow"
DATA_FILE="$DATA_DIR/tasks.yaml"
LOCK_FILE="$DATA_DIR/tasks.lock"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKTREE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TASKMD_FILE="$WORKTREE_DIR/TASK.md"

mkdir -p "$DATA_DIR"

usage() {
    echo "Usage: $0 <command> [args]"
    echo "  list                            List all tasks"
    echo "  add \"title\" [worktree]          Add task (state=impl)"
    echo "  test <ID> [notes]               impl → test"
    echo "  done <ID> [notes]               test → done"
    echo "  delete <ID>                     Delete task"
    echo "  show <ID>                       Show task details"
    exit 1
}

init_data() {
    if [ ! -f "$DATA_FILE" ]; then
        echo "next_id: 1" > "$DATA_FILE"
        echo "tasks: []" >> "$DATA_FILE"
    fi
}

with_lock() {
    exec 8>"$LOCK_FILE"
    flock 8
    "$@"
    flock -u 8
    exec 8<&-
}

get_next_id() {
    local id
    id=$(grep '^next_id:' "$DATA_FILE" | awk '{print $2}')
    echo "$id"
}

set_next_id() {
    local new_id="$1"
    local tmp="${DATA_FILE}.tmp"
    sed "s/^next_id: .*/next_id: $new_id/" "$DATA_FILE" > "$tmp"
    mv "$tmp" "$DATA_FILE"
}

format_id() {
    printf "T%03d" "$1"
}

now() {
    date '+%Y-%m-%d %H:%M:%S'
}

find_task_line() {
    local id="$1"
    grep -n "^  - id: $id$" "$DATA_FILE" | head -1 | cut -d: -f1
}

get_task_field() {
    local start_line="$1"
    local field="$2"
    awk -v start="$start_line" -v field="$field" '
        NR > start && /^  - id:/ { exit }
        NR >= start && $0 ~ "^    " field ": " {
            sub("^    " field ": ", "")
            gsub(/^"/, ""); gsub(/"$/, "")
            print
            exit
        }
    ' "$DATA_FILE"
}

set_task_field() {
    local start_line="$1"
    local field="$2"
    local value="$3"
    local tmp="${DATA_FILE}.tmp"
    awk -v start="$start_line" -v field="$field" -v value="$value" '
        NR > start && /^  - id:/ { found_next = 1 }
        NR >= start && !found_next && $0 ~ "^    " field ": " {
            print "    " field ": \"" value "\""
            next
        }
        { print }
    ' "$DATA_FILE" > "$tmp"
    mv "$tmp" "$DATA_FILE"
}

generate_taskmd() {
    local impl_tasks="" test_tasks="" done_tasks=""
    local in_task=0 cur_id="" cur_title="" cur_state="" cur_worktree="" cur_notes="" cur_updated=""
    local line_num=0

    while IFS= read -r line; do
        line_num=$((line_num + 1))
        if [[ "$line" =~ ^"  - id: " ]]; then
            if [ $in_task -eq 1 ]; then
                local entry="- **$cur_id**: $cur_title"
                [ -n "$cur_worktree" ] && entry="$entry [wt: $cur_worktree]"
                [ -n "$cur_updated" ] && entry="$entry (${cur_updated%:*})"
                entry="$entry"
                if [ -n "$cur_notes" ]; then
                    entry="$entry"$'\n'"  - $cur_notes"
                fi
                case "$cur_state" in
                    impl) impl_tasks="${impl_tasks}${entry}"$'\n' ;;
                    test) test_tasks="${test_tasks}${entry}"$'\n' ;;
                    done) done_tasks="${done_tasks}${entry}"$'\n' ;;
                esac
            fi
            in_task=1
            cur_id="${line#*id: }"
            cur_title="" cur_state="" cur_worktree="" cur_notes="" cur_updated=""
        elif [ $in_task -eq 1 ]; then
            case "$line" in
                "    title: "*)    cur_title="${line#*title: }" ; cur_title="${cur_title#\"}"; cur_title="${cur_title%\"}" ;;
                "    state: "*)    cur_state="${line#*state: }" ; cur_state="${cur_state#\"}"; cur_state="${cur_state%\"}" ;;
                "    worktree: "*) cur_worktree="${line#*worktree: }" ; cur_worktree="${cur_worktree#\"}"; cur_worktree="${cur_worktree%\"}" ;;
                "    notes: "*)    cur_notes="${line#*notes: }" ; cur_notes="${cur_notes#\"}"; cur_notes="${cur_notes%\"}" ;;
                "    updated: "*)  cur_updated="${line#*updated: }" ; cur_updated="${cur_updated#\"}"; cur_updated="${cur_updated%\"}" ;;
            esac
        fi
    done < "$DATA_FILE"

    if [ $in_task -eq 1 ]; then
        local entry="- **$cur_id**: $cur_title"
        [ -n "$cur_worktree" ] && entry="$entry [wt: $cur_worktree]"
        [ -n "$cur_updated" ] && entry="$entry (${cur_updated%:*})"
        if [ -n "$cur_notes" ]; then
            entry="$entry"$'\n'"  - $cur_notes"
        fi
        case "$cur_state" in
            impl) impl_tasks="${impl_tasks}${entry}"$'\n' ;;
            test) test_tasks="${test_tasks}${entry}"$'\n' ;;
            done) done_tasks="${done_tasks}${entry}"$'\n' ;;
        esac
    fi

    {
        echo "# Task Board"
        echo ""
        echo "## impl (実装中)"
        if [ -n "$impl_tasks" ]; then
            printf '%s' "$impl_tasks"
        else
            echo "(none)"
        fi
        echo ""
        echo "## test (実験待ち)"
        if [ -n "$test_tasks" ]; then
            printf '%s' "$test_tasks"
        else
            echo "(none)"
        fi
        echo ""
        echo "## done (完了)"
        if [ -n "$done_tasks" ]; then
            printf '%s' "$done_tasks"
        else
            echo "(none)"
        fi
    } > "$TASKMD_FILE"
}

cmd_list() {
    init_data
    local has_tasks=0
    local in_task=0 cur_id="" cur_title="" cur_state="" cur_worktree="" cur_updated=""

    printf "%-6s| %-6s| %-30s| %-20s| %s\n" "ID" "State" "Title" "Worktree" "Updated"
    echo "------+-------+-------------------------------+---------------------+------------------"

    while IFS= read -r line; do
        if [[ "$line" =~ ^"  - id: " ]]; then
            if [ $in_task -eq 1 ]; then
                has_tasks=1
                printf "%-6s| %-6s| %-30s| %-20s| %s\n" \
                    "$cur_id" "$cur_state" "${cur_title:0:30}" "${cur_worktree:0:20}" "${cur_updated%:*}"
            fi
            in_task=1
            cur_id="${line#*id: }"
            cur_title="" cur_state="" cur_worktree="" cur_updated=""
        elif [ $in_task -eq 1 ]; then
            case "$line" in
                "    title: "*)    cur_title="${line#*title: }" ; cur_title="${cur_title#\"}"; cur_title="${cur_title%\"}" ;;
                "    state: "*)    cur_state="${line#*state: }" ; cur_state="${cur_state#\"}"; cur_state="${cur_state%\"}" ;;
                "    worktree: "*) cur_worktree="${line#*worktree: }" ; cur_worktree="${cur_worktree#\"}"; cur_worktree="${cur_worktree%\"}" ;;
                "    updated: "*)  cur_updated="${line#*updated: }" ; cur_updated="${cur_updated#\"}"; cur_updated="${cur_updated%\"}" ;;
            esac
        fi
    done < "$DATA_FILE"

    if [ $in_task -eq 1 ]; then
        has_tasks=1
        printf "%-6s| %-6s| %-30s| %-20s| %s\n" \
            "$cur_id" "$cur_state" "${cur_title:0:30}" "${cur_worktree:0:20}" "${cur_updated%:*}"
    fi

    if [ $has_tasks -eq 0 ]; then
        echo "(no tasks)"
    fi
}

cmd_add() {
    local title="${1:-}"
    local worktree="${2:-}"
    if [ -z "$title" ]; then
        echo "Error: title required" >&2
        usage
    fi
    init_data
    local id_num
    id_num=$(get_next_id)
    local id
    id=$(format_id "$id_num")
    local ts
    ts=$(now)

    local new_entry
    new_entry="  - id: $id
    title: \"$title\"
    state: impl
    worktree: \"$worktree\"
    notes: \"\"
    created: \"$ts\"
    updated: \"$ts\""

    set_next_id $((id_num + 1))

    local tmp="${DATA_FILE}.tmp"
    if grep -q '^tasks: \[\]' "$DATA_FILE"; then
        sed 's/^tasks: \[\]/tasks:/' "$DATA_FILE" > "$tmp"
        echo "$new_entry" >> "$tmp"
        mv "$tmp" "$DATA_FILE"
    else
        echo "$new_entry" >> "$DATA_FILE"
    fi

    echo "Added: $id - $title"
    generate_taskmd
}

cmd_test() {
    local id="${1:-}"
    local notes="${2:-}"
    if [ -z "$id" ]; then
        echo "Error: task ID required" >&2
        usage
    fi
    init_data
    local line_num
    line_num=$(find_task_line "$id")
    if [ -z "$line_num" ]; then
        echo "Error: task $id not found" >&2
        return 1
    fi
    local cur_state
    cur_state=$(get_task_field "$line_num" "state")
    if [ "$cur_state" != "impl" ]; then
        echo "Error: task $id is in '$cur_state' state, expected 'impl'" >&2
        return 1
    fi
    set_task_field "$line_num" "state" "test"
    set_task_field "$line_num" "updated" "$(now)"
    if [ -n "$notes" ]; then
        line_num=$(find_task_line "$id")
        set_task_field "$line_num" "notes" "$notes"
    fi
    echo "Updated: $id → test"
    generate_taskmd
}

cmd_done() {
    local id="${1:-}"
    local notes="${2:-}"
    if [ -z "$id" ]; then
        echo "Error: task ID required" >&2
        usage
    fi
    init_data
    local line_num
    line_num=$(find_task_line "$id")
    if [ -z "$line_num" ]; then
        echo "Error: task $id not found" >&2
        return 1
    fi
    local cur_state
    cur_state=$(get_task_field "$line_num" "state")
    if [ "$cur_state" != "test" ]; then
        echo "Error: task $id is in '$cur_state' state, expected 'test'" >&2
        return 1
    fi
    set_task_field "$line_num" "state" "done"
    set_task_field "$line_num" "updated" "$(now)"
    if [ -n "$notes" ]; then
        line_num=$(find_task_line "$id")
        set_task_field "$line_num" "notes" "$notes"
    fi
    echo "Updated: $id → done"
    generate_taskmd
}

cmd_delete() {
    local id="${1:-}"
    if [ -z "$id" ]; then
        echo "Error: task ID required" >&2
        usage
    fi
    init_data
    local line_num
    line_num=$(find_task_line "$id")
    if [ -z "$line_num" ]; then
        echo "Error: task $id not found" >&2
        return 1
    fi
    local tmp="${DATA_FILE}.tmp"
    awk -v start="$line_num" '
        NR == start { skip = 1; next }
        skip && /^  - id:/ { skip = 0 }
        skip && /^    / { next }
        skip { skip = 0 }
        { print }
    ' "$DATA_FILE" > "$tmp"
    mv "$tmp" "$DATA_FILE"
    echo "Deleted: $id"
    generate_taskmd
}

cmd_show() {
    local id="${1:-}"
    if [ -z "$id" ]; then
        echo "Error: task ID required" >&2
        usage
    fi
    init_data
    local line_num
    line_num=$(find_task_line "$id")
    if [ -z "$line_num" ]; then
        echo "Error: task $id not found" >&2
        return 1
    fi
    echo "ID:        $id"
    echo "Title:     $(get_task_field "$line_num" "title")"
    echo "State:     $(get_task_field "$line_num" "state")"
    echo "Worktree:  $(get_task_field "$line_num" "worktree")"
    echo "Notes:     $(get_task_field "$line_num" "notes")"
    echo "Created:   $(get_task_field "$line_num" "created")"
    echo "Updated:   $(get_task_field "$line_num" "updated")"
}

run_cmd() {
    case "${1:-}" in
        list)   cmd_list ;;
        add)    shift; with_lock cmd_add "$@" ;;
        test)   shift; with_lock cmd_test "$@" ;;
        done)   shift; with_lock cmd_done "$@" ;;
        delete) shift; with_lock cmd_delete "$@" ;;
        show)   shift; cmd_show "$@" ;;
        *)      usage ;;
    esac
}

run_cmd "$@"
