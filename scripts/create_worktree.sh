#!/usr/bin/env bash
#
# Create a Git worktree whose local runtime state is shared with the primary
# TransformerRL checkout.
#
# Usage:
#   scripts/create_worktree.sh PATH BRANCH [START_POINT]
#
# Examples:
#   scripts/create_worktree.sh ../TransformerRL-bodygen alex/bodygen alex/phase-2
#   scripts/create_worktree.sh ../TransformerRL-review alex/existing-branch
#
# If BRANCH already exists, it is checked out. Otherwise it is created from
# START_POINT, which defaults to the current HEAD.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    sed -n '3,13p' "$0"
    exit 2
fi

worktree_argument=$1
branch=$2
start_point=${3:-HEAD}

current_root=$(git rev-parse --show-toplevel)
git_common_dir=$(git rev-parse --git-common-dir)
if [[ $git_common_dir != /* ]]; then
    git_common_dir="$current_root/$git_common_dir"
fi
primary_root=$(cd "$(dirname "$git_common_dir")" && pwd -P)
worktree_path=$(realpath -m -- "$worktree_argument")
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

git check-ref-format --branch "$branch" >/dev/null

if git show-ref --verify --quiet "refs/heads/$branch"; then
    echo "[worktree] checking out existing branch $branch"
    git worktree add "$worktree_path" "$branch"
else
    echo "[worktree] creating $branch from $start_point"
    git worktree add -b "$branch" "$worktree_path" "$start_point"
fi

"$script_directory/share_worktree_state.sh" "$worktree_path"

echo
echo "[worktree] ready: $worktree_path"
echo "[worktree] branch: $branch"
echo "[worktree] shared state: $primary_root"
