#!/usr/bin/env bash
set -euo pipefail

# FLEURS acquisition follows egs2/fleurs/asr1/local/data.sh:
# source path/cmd/db, create TSV manifests from google/xtreme_s, then convert
# TSV + CS-FLEURS metadata into Kaldi-style data dirs for lid1 and asr1.
[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ] && . ./cmd.sh
[ -f ./db.sh ] && . ./db.sh

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_fleurs_tsvs() {
  ${skip_fleurs} && return 0
  [ -n "${fleurs_manifest_root}" ] && return 0
  local s
  for s in train dev test; do
    [ -s "${fleurs_tsv_root}/${s}.tsv" ] || fail "missing or empty FLEURS TSV: ${fleurs_tsv_root}/${s}.tsv"
  done
}

stage=0
stop_stage=1
python=python3
outdir=data
nlsyms_txt=data/nlsyms.txt

# FLEURS / official ESPnet-FLEURS-style TSV options.
fleurs_config=all
fleurs_download_dir=${FLEURS:-downloads/fleurs}
fleurs_tsv_root=
fleurs_cache_dir=downloads/cache
fleurs_revision=70bb2e84b976b7e960aa89f1c648e09c59f894dd
fleurs_manifest_root=
fleurs_subsample_per_lang=0
skip_fleurs=false
skip_fleurs_download=false
exclude_fleurs_train_overlaps=true

# CS-FLEURS options. The expected local layout is subset/metadata.jsonl plus audio/*.wav.
cs_root=${CS_FLEURS_ROOT:-}
cs_train_subsets="xtts/train"
cs_eval_subsets="read/test,xtts/test1,xtts/test2,mms/test"
cs_dev_ratio=0.02
cs_split_mode=global_hash
cs_pair_field=language

# Optional CS-YODAS source. Download it first with
# ../local/download_extract_cs_yodas.sh and pass --cs_yodas_root.
cs_yodas_root=
cs_yodas_train_ratio=0.8
cs_yodas_valid_ratio=0.1
cs_yodas_split_seed=cs-yodas-v1
cs_yodas_max_train_valid_duration_sec=70.0

# Label / target options.
label_map=
verify_fleurs_config_mapping=true
token_format=angle
allow_unseen_eval_labels=false
require_eval_labels_in_fleurs_lid=true
allow_single_cs=false
strict_labels=true

# Uniform source-level filtering for train/validation only. User request:
# cut train/valid utterances with duration <1 sec or >=30 sec; keep test as-is.
min_train_duration_sec=1.0
max_train_duration_sec=30.0
min_eval_duration_sec=-1.0
max_eval_duration_sec=0.0
duration_missing_policy=warn

. utils/parse_options.sh

if [ -z "${fleurs_tsv_root}" ]; then
  fleurs_tsv_root="${fleurs_download_dir}/${fleurs_config}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  if ! ${skip_fleurs} && ! ${skip_fleurs_download} && [ -z "${fleurs_manifest_root}" ]; then
    log "Stage 0: create FLEURS TSV manifests under ${fleurs_tsv_root}"
    mkdir -p "${fleurs_tsv_root}"
    "${python}" local/create_fleurs_lid_dataset.py \
      --lang "${fleurs_config}" \
      --out_root "${fleurs_tsv_root}" \
      --cache_dir "${fleurs_cache_dir}" \
      --revision "${fleurs_revision}" \
      --source google_fleurs \
      --subsample_per_lang "${fleurs_subsample_per_lang}"
  else
    log "Stage 0: skip FLEURS TSV creation"
  fi
  require_fleurs_tsvs
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: prepare FLEURS + CS-FLEURS data dirs"
  require_fleurs_tsvs
  opts=(
    --outdir "${outdir}"
    --fleurs_config "${fleurs_config}"
    --fleurs_subsample_per_lang "${fleurs_subsample_per_lang}"
    --cs_train_subsets "${cs_train_subsets}"
    --cs_eval_subsets "${cs_eval_subsets}"
    --cs_dev_ratio "${cs_dev_ratio}"
    --cs_split_mode "${cs_split_mode}"
    --cs_pair_field "${cs_pair_field}"
    --token_format "${token_format}"
    --allow_unseen_eval_labels "${allow_unseen_eval_labels}"
    --require_eval_labels_in_fleurs_lid "${require_eval_labels_in_fleurs_lid}"
    --allow_single_cs "${allow_single_cs}"
    --strict_labels "${strict_labels}"
    --nlsyms_txt "${nlsyms_txt}"
    --verify_fleurs_config_mapping "${verify_fleurs_config_mapping}"
    --skip_fleurs "${skip_fleurs}"
    --exclude_fleurs_train_overlaps "${exclude_fleurs_train_overlaps}"
    --min_train_duration_sec "${min_train_duration_sec}"
    --max_train_duration_sec "${max_train_duration_sec}"
    --min_eval_duration_sec "${min_eval_duration_sec}"
    --max_eval_duration_sec "${max_eval_duration_sec}"
    --duration_missing_policy "${duration_missing_policy}"
  )
  if [ -n "${fleurs_manifest_root}" ]; then
    opts+=(--fleurs_manifest_root "${fleurs_manifest_root}")
  elif ! ${skip_fleurs}; then
    opts+=(--fleurs_tsv_root "${fleurs_tsv_root}")
  fi
  [ -n "${fleurs_cache_dir}" ] && opts+=(--fleurs_cache_dir "${fleurs_cache_dir}")
  [ -n "${cs_root}" ] && opts+=(--cs_root "${cs_root}")
  [ -n "${label_map}" ] && opts+=(--label_map "${label_map}")
  "${python}" local/prepare_fleurs_cs_lid_data.py "${opts[@]}"
  if [ ! -s "${nlsyms_txt}" ]; then
    fail "missing canonical nlsyms file: ${nlsyms_txt}"
  fi
  if grep -nE '<[a-z]+_[a-z0-9_]+>|\[[a-z]+_[a-z0-9_]+\]' "${nlsyms_txt}"; then
    fail "raw FLEURS labels or bracket prompts found in ${nlsyms_txt}"
  fi

  if [ -n "${cs_yodas_root}" ]; then
    log "Stage 1: add the six CS-YODAS base-language + English pairs"
    "${python}" ../local/prepare_cs_yodas_views.py \
      --source_data_dir "${outdir}" \
      --outdir "${outdir}" \
      --metadata_dir "${cs_yodas_root}/metadata" \
      --audio_root "${cs_yodas_root}/audio" \
      --train_ratio "${cs_yodas_train_ratio}" \
      --valid_ratio "${cs_yodas_valid_ratio}" \
      --seed "${cs_yodas_split_seed}" \
      --max_yodas_train_valid_duration "${cs_yodas_max_train_valid_duration_sec}"
  fi

  for d in \
    train_fleurs_lid valid_fleurs_lid \
    train_fleurs_lidseq valid_fleurs_lidseq \
    train_lidseq valid_lidseq \
    train_lidseq_csall_yodas valid_lidseq_csall_yodas \
    train_lid_pair_cs valid_lid_pair_cs \
    train_lid_pair_csall_yodas valid_lid_pair_csall_yodas \
    test_fleurs_lid test_cs_read_test test_cs_xtts_test1 \
    test_cs_xtts_test2 test_cs_mms_test test_cs_all \
    test_yodas_lidseq test_lid_pair_yodas; do
    if [ -d "${outdir}/${d}" ]; then
      utils/fix_data_dir.sh "${outdir}/${d}"
      utils/validate_data_dir.sh --no-feats --non-print "${outdir}/${d}"
    fi
  done

  verify_require_cs=true
  if [ -z "${cs_root}" ]; then
    verify_require_cs=false
  fi
  if [ -x local/verify_lid_data.sh ]; then
    verify_max_duration=${max_train_duration_sec}
    verify_extra_sets="train_lid_pair_cs valid_lid_pair_cs"
    if [ -n "${cs_yodas_root}" ]; then
      verify_max_duration=${cs_yodas_max_train_valid_duration_sec}
      verify_extra_sets+=" train_lidseq_csall_yodas valid_lidseq_csall_yodas"
      verify_extra_sets+=" train_lid_pair_csall_yodas valid_lid_pair_csall_yodas"
      verify_extra_sets+=" test_yodas_lidseq test_lid_pair_yodas"
    fi
    local/verify_lid_data.sh \
      --data_dir "${outdir}" \
      --require_cs "${verify_require_cs}" \
      --extra_sets "${verify_extra_sets}" \
      --allow_pair_labels true \
      --max_train_duration_sec "${verify_max_duration}"
  fi
fi
