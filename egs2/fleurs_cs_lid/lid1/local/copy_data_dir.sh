#!/usr/bin/env bash
set -euo pipefail

# Keep Kaldi's utterance-level speakers separate from classification labels.
# The generic LID helper temporarily treats language IDs as speaker IDs,
# which requires language-prefixed utterance IDs and loses extra sidecars.
validate_opts=
. utils/parse_options.sh
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 [--validate_opts OPTIONS] SOURCE DESTINATION" >&2
  exit 2
fi
source_dir=$1
destination_dir=$2
bash utils/copy_data_dir.sh --validate_opts "${validate_opts}" \
  "${source_dir}" "${destination_dir}"
for name in utt2lang utt2langs lang2utt category2utt utt2category labels.jsonl utt2num_samples; do
  source_name=${name}
  [ "${name}" = category2utt ] && source_name=lang2utt
  if [ -f "${source_dir}/${source_name}" ]; then
    temporary=$(mktemp "${destination_dir}/.${name}.XXXXXX")
    cp "${source_dir}/${source_name}" "${temporary}"
    chmod u+w "${temporary}"
    mv -f "${temporary}" "${destination_dir}/${name}"
  fi
done
