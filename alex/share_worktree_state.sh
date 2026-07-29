#!/usr/bin/env bash
#
# Link an existing worktree's machine-local state to the primary checkout.
#
# Usage:
#   alex/share_worktree_state.sh [--migrate-existing] [WORKTREE]
#
# Example:
#   alex/share_worktree_state.sh --migrate-existing .
#
# New worktrees have no local artifacts and need no flag. For an established
# worktree, --migrate-existing first checks for collisions, copies its artifacts
# into the primary shared directories, and retains the originals in
# .worktree-state-backup/.

set -euo pipefail

migrate_existing=false
if [[ ${1:-} == "--migrate-existing" ]]; then
    migrate_existing=true
    shift
fi
if [[ $# -gt 1 ]]; then
    sed -n '3,12p' "$0"
    exit 2
fi

requested_path=${1:-.}
worktree_path=$(realpath -m -- "$requested_path")
worktree_root=$(git -C "$worktree_path" rev-parse --show-toplevel)
worktree_root=$(realpath -m -- "$worktree_root")
if [[ $worktree_path != "$worktree_root" ]]; then
    echo "[worktree] use the worktree root, not a subdirectory: $worktree_root"
    exit 2
fi

git_common_dir=$(git -C "$worktree_root" rev-parse --git-common-dir)
if [[ $git_common_dir != /* ]]; then
    git_common_dir="$worktree_root/$git_common_dir"
fi
primary_root=$(cd "$(dirname "$git_common_dir")" && pwd -P)
if [[ $worktree_root == "$primary_root" ]]; then
    echo "[worktree] $worktree_root is already the primary shared checkout"
    exit 0
fi

shared_directories=(
    runs
    evals
    logs
    videos
    data
)
shared_resources=(
    .venv
    .envrc
    TurboActivate.dat
    alex/commands.sh
)

same_link() {
    local destination=$1
    local source=$2
    [[ -L $destination ]] \
        && [[ $(realpath -m -- "$destination") == $(realpath -m -- "$source") ]]
}

directory_conflicts() {
    local local_directory=$1
    local shared_directory=$2
    local entry relative shared_entry
    while IFS= read -r -d '' entry; do
        relative=${entry#"$local_directory/"}
        shared_entry="$shared_directory/$relative"
        if [[ ! -e $shared_entry && ! -L $shared_entry ]]; then
            continue
        fi
        if [[ -f $entry && -f $shared_entry ]] \
            && cmp -s -- "$entry" "$shared_entry"; then
            continue
        fi
        if [[ -L $entry && -L $shared_entry ]] \
            && [[ $(readlink "$entry") == $(readlink "$shared_entry") ]]; then
            continue
        fi
        printf '  %s\n' "$relative"
    done < <(find "$local_directory" -mindepth 1 ! -type d -print0)
}

# Preflight everything before changing any path.
has_existing_directories=false
for name in "${shared_directories[@]}"; do
    destination="$worktree_root/$name"
    source="$primary_root/$name"
    if same_link "$destination" "$source"; then
        continue
    fi
    if [[ -L $destination ]]; then
        echo "[worktree] $destination points somewhere other than $source"
        exit 1
    fi
    if [[ -e $destination ]]; then
        has_existing_directories=true
        if [[ $migrate_existing != true ]]; then
            echo "[worktree] $destination already contains local state"
            echo "[worktree] rerun with --migrate-existing after training stops"
            exit 1
        fi
        conflicts=$(directory_conflicts "$destination" "$source")
        if [[ -n $conflicts ]]; then
            echo "[worktree] conflicting files under $name/:"
            printf '%s\n' "$conflicts"
            echo "[worktree] resolve these manually; nothing was changed"
            exit 1
        fi
    fi
done

for name in "${shared_resources[@]}"; do
    destination="$worktree_root/$name"
    source="$primary_root/$name"
    if same_link "$destination" "$source"; then
        continue
    fi
    if [[ -e $destination || -L $destination ]]; then
        echo "[worktree] refusing to replace existing $destination"
        exit 1
    fi
done

if [[ $has_existing_directories == true ]]; then
    for process_cwd in /proc/[0-9]*/cwd; do
        process_root=$(readlink -f "$process_cwd" 2>/dev/null || true)
        if [[ $process_root != "$worktree_root" \
            && $process_root != "$worktree_root/"* ]]; then
            continue
        fi
        process_id=${process_cwd#/proc/}
        process_id=${process_id%/cwd}
        process_command=$(
            tr '\0' ' ' <"/proc/$process_id/cmdline" 2>/dev/null || true
        )
        if [[ $process_command == *train_ant* ]]; then
            echo "[worktree] training process $process_id is still using this worktree"
            echo "[worktree] stop it before migrating shared directories"
            exit 1
        fi
    done
    command -v rsync >/dev/null || {
        echo "[worktree] rsync is required to preserve existing artifacts"
        exit 1
    }
fi

backup_root="$worktree_root/.worktree-state-backup/$(date +%Y%m%d-%H%M%S)"
for name in "${shared_directories[@]}"; do
    destination="$worktree_root/$name"
    source="$primary_root/$name"
    if same_link "$destination" "$source"; then
        echo "[worktree] already shared $name/"
        continue
    fi
    mkdir -p "$source"
    if [[ -e $destination ]]; then
        rsync -a "$destination/" "$source/"
        mkdir -p "$backup_root"
        mv "$destination" "$backup_root/$name"
        echo "[worktree] preserved original $name/ in ${backup_root#$worktree_root/}/"
    fi
    ln -s "$source" "$destination"
    echo "[worktree] shared $name/"
done

for name in "${shared_resources[@]}"; do
    destination="$worktree_root/$name"
    source="$primary_root/$name"
    if same_link "$destination" "$source"; then
        echo "[worktree] already shared $name"
    elif [[ -e $source || -L $source ]]; then
        mkdir -p "$(dirname "$destination")"
        ln -s "$source" "$destination"
        echo "[worktree] shared $name"
    else
        echo "[worktree] skipped missing $source"
    fi
done

if command -v direnv >/dev/null && [[ -e "$worktree_root/.envrc" ]]; then
    if (cd "$worktree_root" && direnv allow); then
        echo "[worktree] allowed .envrc"
    else
        echo "[worktree] warning: run 'cd $worktree_root && direnv allow' manually"
    fi
fi

echo
echo "[worktree] shared state ready: $worktree_root"
echo "[worktree] primary owner: $primary_root"
if [[ -d $backup_root ]]; then
    echo "[worktree] backup retained: $backup_root"
fi
