#!/usr/bin/env bash
set -euo pipefail

# Score ASR-style encoder-decoder LID sequence hypotheses.
# This script intentionally searches common ESPnet2 decode layouts, because
# different ESPnet versions may write hypotheses as:
#   <decode_dir>/<set>/text
#   <decode_dir>/<set>/1best_recog/text
#   <decode_dir>/<set>/logdir/output.*/1best_recog/text

python=python3
decode_dir=
label_map=data/local/label_map.used.tsv
test_sets="test_fleurs_lid test_cs_read_test test_cs_xtts_test1 test_cs_xtts_test2 test_cs_mms_test test_cs_all"

. utils/parse_options.sh

if [ -z "${decode_dir}" ]; then
  echo "Usage: $0 --decode_dir <ESPnet decode dir>" >&2
  exit 2
fi

find_hyp_text() {
  local dset=$1
  local base="${decode_dir}/${dset}"
  local c
  for c in "${base}/text" "${base}/1best_recog/text"; do
    if [ -s "${c}" ]; then
      echo "${c}"
      return 0
    fi
  done

  if [ ! -d "${base}" ]; then
    return 1
  fi

  local -a found=()
  while IFS= read -r f; do
    found+=("${f}")
  done < <(find "${base}" -type f \( -path '*/1best_recog/text' -o -name text \) 2>/dev/null | sort)

  if [ "${#found[@]}" -eq 0 ]; then
    return 1
  fi

  local -a preferred=()
  for c in "${found[@]}"; do
    if [[ "${c}" == */1best_recog/text ]]; then
      preferred+=("${c}")
    fi
  done
  if [ "${#preferred[@]}" -eq 0 ]; then
    preferred=("${found[@]}")
  fi

  # Decode may be split into output.1, output.2, ...; duplicates indicate a
  # broken shard merge and must not be hidden by first-occurrence wins.
  local merged="${base}/lidseq_hyp_merged.txt"
  awk '
    NF > 0 {
      if (seen[$1]++) {
        print "duplicate hypothesis utterance id: " $1 > "/dev/stderr"
        bad=1
      } else {
        print
      }
    }
    END {exit bad}
  ' "${preferred[@]}" | sort > "${merged}"
  echo "${merged}"
}

missing=0
scored=0
for dset in ${test_sets}; do
  ref="data/${dset}/utt2langs"
  if [ ! -f "${ref}" ]; then
    ref="dump/raw/${dset}/utt2langs"
  fi
  if [ ! -f "${ref}" ]; then
    echo "Error: missing reference utt2langs for ${dset}" >&2
    missing=1
    continue
  fi
  if ! hyp=$(find_hyp_text "${dset}"); then
    echo "Error: no hypothesis text found under ${decode_dir}/${dset}" >&2
    missing=1
    continue
  fi
  out="${decode_dir}/${dset}/lidseq_score.json"
  mkdir -p "$(dirname "${out}")"
  "${python}" local/score_lidseq.py \
    --ref "${ref}" \
    --hyp "${hyp}" \
    --label_map "${label_map}" \
    --out "${out}" \
    --details_out "${decode_dir}/${dset}/lidseq_details.tsv"
  scored=$((scored + 1))
done

[ "${scored}" -gt 0 ] || missing=1
[ "${missing}" -eq 0 ] || exit 1
