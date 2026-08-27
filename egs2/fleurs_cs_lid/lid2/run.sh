#!/usr/bin/env bash
set -euo pipefail

# Mixed soft-target LID recipe.  This recipe intentionally does not run data
# preparation.  It reads prepared manifests from lid1/data and writes only under
# lid2/{data,dump,exp,conf/generated}.

profile=mixed
train_set=
valid_set=
test_sets=
tsne_set=

source_data_dir=../lid1/data

ngpu=2
nj=32
dumpdir=dump
expdir=exp
stage=3
stop_stage=5
feats_type=raw
fs=16k
audio_format=wav
min_wav_duration=0.999999
max_wav_duration=
gpu_inference=true
num_nodes=1

lid_config=conf/train_lidseq_mms_ecapa_softtarget.yaml
lid_tag=
lid_args=
lid_label_file=utt2langs
inference_model=valid.accuracy.best.pth
inference_batch_size=4
extract_embd=false

train_batch_size=8
accum_grad=4
effective_batch_size=32
budget_max_epoch=15
target_passes=3.0
warmup_ratio=0.1
round_updates_per_epoch_to=10
fixed_batch_type=sorted
drop_last_iter=true
generated_config_dir=conf/generated
budget_count_file=wav.scp

. utils/parse_options.sh

case "${profile}" in
  mixed)
    : "${train_set:=train_lidseq}"
    : "${valid_set:=valid_lidseq}"
    : "${max_wav_duration:=30}"
    ;;
  csall)
    : "${train_set:=train_lidseq_csall_yodas}"
    : "${valid_set:=valid_lidseq_csall_yodas}"
    : "${max_wav_duration:=70}"
    ;;
  *)
    if [ -z "${train_set}" ] || [ -z "${valid_set}" ] || [ -z "${max_wav_duration}" ]; then
      echo "Error: custom --profile requires train_set, valid_set, and max_wav_duration" >&2
      exit 2
    fi
    ;;
esac
if [ -z "${test_sets}" ]; then
  test_sets="test_fleurs_lid test_cs_read_test test_cs_xtts_test1 test_cs_xtts_test2 test_cs_mms_test test_cs_all"
  [ "${profile}" = csall ] && test_sets+=" test_yodas_lidseq"
fi
if [ -z "${lid_tag}" ]; then
  lid_tag="lid2_${train_set}_softtarget_kl_mms_ecapa_sorted_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_3pass${budget_max_epoch}ep"
fi

if [ "${stage}" -lt 3 ]; then
  echo "Error: lid2 is read-only with respect to data prep. Use stage >= 3." >&2
  echo "       Prepared manifests are read from --source_data_dir ${source_data_dir}." >&2
  exit 2
fi

if [ $((train_batch_size * accum_grad)) -ne "${effective_batch_size}" ]; then
  echo "Error: train_batch_size * accum_grad must equal effective_batch_size" >&2
  exit 2
fi
if [ "${ngpu}" -gt 1 ] && [ $((train_batch_size % ngpu)) -ne 0 ]; then
  echo "Error: train_batch_size=${train_batch_size} must be divisible by ngpu=${ngpu}" >&2
  exit 2
fi

link_from_source_data() {
  local name=$1
  local src="${source_data_dir}/${name}"
  local dst="data/${name}"
  if [ ! -e "${src}" ]; then
    echo "Error: missing source data entry: ${src}" >&2
    exit 2
  fi
  if [ -e "${dst}" ] || [ -L "${dst}" ]; then
    return
  fi
  local rel
  rel=$(realpath --relative-to="$(dirname "${dst}")" "${src}")
  ln -s "${rel}" "${dst}"
}

prepare_readonly_data_links() {
  mkdir -p data
  source_data_dir=$(realpath "${source_data_dir}")
  for d in train_fleurs_lid valid_fleurs_lid train_fleurs_lidseq valid_fleurs_lidseq ${train_set} ${valid_set} ${test_sets}; do
    link_from_source_data "${d}"
  done
  if [ -L data/local ]; then
    echo "Error: data/local must be a lid2-owned directory, not a symlink." >&2
    echo "       Remove the symlink and rerun this wrapper." >&2
    exit 2
  fi
  mkdir -p data/local
  if [ -d "${source_data_dir}/local" ]; then
    for f in \
      label_map.used.tsv \
      label_inventory.tsv \
      fleurs_official_label_map.tsv \
      duration_excluded.tsv \
      duration_summary.tsv; do
      if [ -e "${source_data_dir}/local/${f}" ] && [ ! -e "data/local/${f}" ] && [ ! -L "data/local/${f}" ]; then
        ln -s "$(realpath --relative-to=data/local "${source_data_dir}/local/${f}")" "data/local/${f}"
      fi
    done
  fi
  if [ -e "${source_data_dir}/nlsyms.txt" ] && [ ! -e data/nlsyms.txt ] && [ ! -L data/nlsyms.txt ]; then
    ln -s "$(realpath --relative-to=data "${source_data_dir}/nlsyms.txt")" data/nlsyms.txt
  fi
}

prepare_readonly_data_links

base_name=$(basename "${lid_config}" .yaml)
gen_prefix="${generated_config_dir}/${base_name}_${train_set}_${fixed_batch_type}_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_3pass${budget_max_epoch}ep"

if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  mkdir -p "${generated_config_dir}" data/local
  local/verify_lid_data.sh \
    --data_dir data \
    --require_cs true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --max_train_duration_sec "${max_wav_duration}"
  python3 local/make_fixed_batch_config.py \
    --task lid \
    --base_config "${lid_config}" \
    --output_config "${gen_prefix}.yaml" \
    --train_dir "data/${train_set}" \
    --count_file "${budget_count_file}" \
    --batch_size "${train_batch_size}" \
    --accum_grad "${accum_grad}" \
    --effective_batch_size "${effective_batch_size}" \
    --ngpu "${ngpu}" \
    --max_epoch "${budget_max_epoch}" \
    --target_passes "${target_passes}" \
    --warmup_ratio "${warmup_ratio}" \
    --round_updates_per_epoch_to "${round_updates_per_epoch_to}" \
    --batch_type "${fixed_batch_type}" \
    --drop_last_iter "${drop_last_iter}" \
    --budget_json "${gen_prefix}.budget.json" \
    --budget_tsv "data/local/budget_lid2_${train_set}_bs${train_batch_size}_ag${accum_grad}.tsv" \
    > "data/local/budget_lid2_${train_set}_bs${train_batch_size}_ag${accum_grad}.log"
  cat "data/local/budget_lid2_${train_set}_bs${train_batch_size}_ag${accum_grad}.log" >&2
  lid_config="${gen_prefix}.yaml"
elif [ "${stage}" -gt 5 ] && [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 6 ]; then
  if [ -s "${gen_prefix}.yaml" ]; then
    lid_config="${gen_prefix}.yaml"
  else
    echo "Error: generated config missing for resume: ${gen_prefix}.yaml" >&2
    echo "       Run once with --stage 3 --stop_stage 5 first." >&2
    exit 2
  fi
fi

if [ "${stop_stage}" -gt 5 ]; then
  echo "Error: use local/lid_softmax_logits.py for paper evaluation; run.sh follows template stages 3-5 only." >&2
  exit 2
fi

if [ "${stage}" -le 5 ]; then
  ./lid.sh \
    --stage "${stage}" \
    --stop_stage "${stop_stage}" \
    --ngpu "${ngpu}" \
    --num_nodes "${num_nodes}" \
    --nj "${nj}" \
    --dumpdir "${dumpdir}" \
    --gpu_inference "${gpu_inference}" \
    --feats_type "${feats_type}" \
    --fs "${fs}" \
    --audio_format "${audio_format}" \
    --min_wav_duration "${min_wav_duration}" \
    --max_wav_duration "${max_wav_duration}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --tsne_set "${tsne_set}" \
    --expdir "${expdir}" \
    --lid_config "${lid_config}" \
    --lid_tag "${lid_tag}" \
    --lid_args "${lid_args}" \
    --lid_label_file "${lid_label_file}" \
    --inference_model "${inference_model}" \
    --inference_batch_size "${inference_batch_size}" \
    --extract_embd "${extract_embd}"
fi
