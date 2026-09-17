#!/usr/bin/env bash
set -euo pipefail

root=downloads/cs-fleurs
repo=https://huggingface.co/datasets/byan/cs-fleurs
revision=0cdbf166c5517ae4b6eb1c54248522eedec53017

script_dir=$(cd "$(dirname "$0")" && pwd)
recipe_root=$(cd "${script_dir}/.." && pwd)
. "${recipe_root}/asr1/utils/parse_options.sh"

if [ ! -d "${root}/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "${repo}" "${root}"
fi

if [ -n "$(git -C "${root}" status --porcelain)" ]; then
    echo "Error: refusing to change a dirty CS-FLEURS checkout: ${root}" >&2
    exit 1
fi
git -C "${root}" fetch origin "${revision}"
GIT_LFS_SKIP_SMUDGE=1 git -C "${root}" checkout --detach "${revision}"
git -C "${root}" lfs pull

for subset in read/test xtts/train xtts/test1 xtts/test2 mms/test; do
    if [ ! -s "${root}/${subset}/metadata.jsonl" ]; then
        echo "Error: missing ${root}/${subset}/metadata.jsonl" >&2
        exit 1
    fi
done

echo "CS-FLEURS revision: $(git -C "${root}" rev-parse HEAD)" >&2
