#!/usr/bin/env bash
set -euo pipefail
EXPECTED_VERSION="21.1.5"
root_folder=$(git rev-parse --show-toplevel)

change_in_place=false
if [[ "${1-}" == "--change-in-place" ]]; then
    change_in_place=true
fi

clang-format --version | grep -q $EXPECTED_VERSION || { echo "$(clang-format --version)" != expected $EXPECTED_VERSION; exit 1; }

# Find files
all_files=$(git ls-tree --full-tree -r --name-only HEAD . \
    | grep -E "^(lib|samples)/.*\.(cpp|hpp)$" \
    | sed "s~^~$root_folder/~")

echo "Found $(wc -w <<< "$all_files") files"

if $change_in_place; then
    echo "Formatting in place..."
    xargs clang-format -i <<< "$all_files"
else
    echo "Dry run..."
    xargs clang-format --dry-run --Werror <<< "$all_files"
fi
