#!/usr/bin/env bash
set -euo pipefail

# FLEURS-only ESPnet lid1 baseline recipe.
# Trains the closed-set LID classifier used for:
#   1) FLEURS top-1 LID accuracy
#   2) CS-FLEURS softmax top-2 unordered set baseline
#
# Training policy mirrors asr1:
#   - default training uses 2 GPUs
#   - ESPnet2 batch_size is global; it is not multiplied by ngpu
#   - use fixed utterance-count mini-batches, not batch_bins/catpow bins
#   - effective batch size = train_batch_size * accum_grad = 32
#   - max_epoch fixed to 30; total exposure about 3 passes over train_fleurs_lid

train_set=train_fleurs_lid
valid_set=valid_fleurs_lid
test_sets="test_fleurs_lid"
tsne_set="test_fleurs_lid"

ngpu=2
nj=32
dumpdir=dump
gpu_inference=true
num_nodes=1
stage=1
stop_stage=7
feats_type=raw
fs=16k
audio_format=wav
# Duration filtering is performed deterministically in local/data.sh. These
# options are retained for compatibility with the LID template interface.
min_wav_duration=0.999999
max_wav_duration=30
expdir=exp
lid_config=conf/train_fleurs_lid_mms_ecapa.yaml
lid_tag=
lid_args=
inference_model=valid.accuracy.best.pth
inference_batch_size=4
extract_embd=false

# Fixed-count 3-pass schedule.  ESPnet2 batch_size is global, not per GPU.
auto_training_budget=true
enforce_training_policy=true
require_cuda_visible_devices=true
train_batch_size=8
batch_size=   # alias: --batch_size 32/16/8/4 may be used instead of --train_batch_size
accum_grad=4
effective_batch_size=32
target_passes=3.0
budget_max_epoch=30
warmup_ratio=0.1
round_updates_per_epoch_to=10
fixed_batch_type=catbel
drop_last_iter=true
generated_config_dir=conf/generated
budget_count_file=wav.scp

run_topk_baselines=false
topk=2
baseline_batch_size=8
baseline_num_workers=4
apply_loss_scale=false

# Data options. cs_root is optional for training but needed for CS-FLEURS eval.
fleurs_config=all
fleurs_download_dir=${FLEURS:-downloads/fleurs}
fleurs_tsv_root=
fleurs_cache_dir=downloads/cache
fleurs_revision=70bb2e84b976b7e960aa89f1c648e09c59f894dd
fleurs_manifest_root=
fleurs_subsample_per_lang=0
skip_fleurs=false
skip_fleurs_download=false
cs_root=${CS_FLEURS_ROOT:-}
cs_train_subsets="xtts/train"
cs_eval_subsets="read/test,xtts/test1,xtts/test2,mms/test"
cs_dev_ratio=0.02
cs_split_mode=global_hash
cs_pair_field=language
cs_yodas_root=
cs_yodas_train_ratio=0.8
cs_yodas_valid_ratio=0.1
cs_yodas_split_seed=cs-yodas-v1
cs_yodas_max_train_valid_duration_sec=70.0
label_map=
verify_fleurs_config_mapping=true
token_format=angle
allow_unseen_eval_labels=false
require_eval_labels_in_fleurs_lid=true
allow_single_cs=false
strict_labels=true
exclude_fleurs_train_overlaps=true
# Apply 1s<=duration<30s only to train/validation; keep test/eval unchanged.
min_train_duration_sec=1.0
max_train_duration_sec=30.0
min_eval_duration_sec=-1.0
max_eval_duration_sec=0.0
duration_missing_policy=error

default_lid_config=${lid_config}

. utils/parse_options.sh

needs_training_or_eval_gpu() {
  [ "${ngpu}" -gt 0 ] && [ "${stage}" -le 8 ] && [ "${stop_stage}" -ge 5 ]
}

check_no_protected_args() {
  local name=$1
  local value=$2
  local opt
  for opt in --batch_size --valid_batch_size --accum_grad --max_epoch --num_iters_per_epoch --batch_type --batch_bins --valid_batch_bins --drop_last_iter --patience --iterator_type --valid_iterator_type; do
    case " ${value} " in
      *" ${opt} "*|*" ${opt}="*)
        echo "Error: ${name} must not override protected training-budget option ${opt}" >&2
        exit 2
        ;;
    esac
  done
}

validate_training_policy() {
  "${enforce_training_policy}" || return 0
  needs_training_or_eval_gpu || return 0
  "${auto_training_budget}" || {
    echo "Error: production training/eval requires --auto_training_budget true. Use --enforce_training_policy false only for explicit debug runs." >&2
    exit 2
  }
  case "${train_batch_size}:${accum_grad}" in
    16:2|8:4|4:8|2:16) ;;
    *)
      echo "Error: allowed fixed LID schedules are 16/2 memory-probe, 8/4 default, 4/8 fallback, or 2/16 OOM fallback; got ${train_batch_size}/${accum_grad}" >&2
      exit 2
      ;;
  esac
  [ "${effective_batch_size}" -eq 32 ] || {
    echo "Error: effective_batch_size must be 32; got ${effective_batch_size}" >&2
    exit 2
  }
  [ "${budget_max_epoch}" -eq 30 ] || {
    echo "Error: budget_max_epoch must be 30; got ${budget_max_epoch}" >&2
    exit 2
  }
  case "${target_passes}" in
    3|3.0|3.00|3.000) ;;
    *)
      echo "Error: target_passes must be 3.0; got ${target_passes}" >&2
      exit 2
      ;;
  esac
  check_no_protected_args lid_args "${lid_args}"
}

validate_cuda_visible_devices() {
  "${require_cuda_visible_devices}" || return 0
  needs_training_or_eval_gpu || return 0
  [ "${stop_stage}" -ge 5 ] || return 0
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || {
    echo "Error: set CUDA_VISIBLE_DEVICES explicitly for GPU training/eval, e.g. CUDA_VISIBLE_DEVICES=0,1" >&2
    exit 2
  }
  local old_ifs=${IFS}
  local -a devs
  IFS=',' read -r -a devs <<< "${CUDA_VISIBLE_DEVICES}"
  IFS=${old_ifs}
  local count=0
  local d
  for d in "${devs[@]}"; do
    [ -n "${d}" ] && count=$((count + 1))
  done
  [ "${count}" -ge "${ngpu}" ] || {
    echo "Error: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} exposes ${count} device(s), but --ngpu ${ngpu} was requested" >&2
    exit 2
  }
}

if [ -n "${batch_size}" ]; then
  train_batch_size=${batch_size}
fi
if [ -z "${fleurs_tsv_root}" ]; then
  fleurs_tsv_root="${fleurs_download_dir}/${fleurs_config}"
fi
if [ $((train_batch_size * accum_grad)) -ne "${effective_batch_size}" ]; then
  echo "Error: train_batch_size * accum_grad must equal effective_batch_size: ${train_batch_size} * ${accum_grad} != ${effective_batch_size}" >&2
  exit 2
fi
if [ "${ngpu}" -gt 1 ] && [ $((train_batch_size % ngpu)) -ne 0 ]; then
  echo "Error: train_batch_size=${train_batch_size} must be divisible by ngpu=${ngpu} for equal per-GPU utterance counts." >&2
  exit 2
fi
validate_training_policy
validate_cuda_visible_devices

base_name=$(basename "${lid_config}" .yaml)
model_label=${base_name#train_fleurs_lid_}
[ "${model_label}" = "${base_name}" ] && model_label=${base_name#train_}
[ -z "${lid_tag}" ] && lid_tag="fleurs_lid_${model_label}_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_3pass30ep"

local_data_opts=(
  --fleurs_config "${fleurs_config}"
  --fleurs_subsample_per_lang "${fleurs_subsample_per_lang}"
  --fleurs_download_dir "${fleurs_download_dir}"
  --fleurs_revision "${fleurs_revision}"
  --fleurs_tsv_root "${fleurs_tsv_root}"
  --skip_fleurs "${skip_fleurs}"
  --skip_fleurs_download "${skip_fleurs_download}"
  --cs_train_subsets "${cs_train_subsets}"
  --cs_eval_subsets "${cs_eval_subsets}"
  --cs_dev_ratio "${cs_dev_ratio}"
  --cs_split_mode "${cs_split_mode}"
  --cs_pair_field "${cs_pair_field}"
  --cs_yodas_root "${cs_yodas_root}"
  --cs_yodas_train_ratio "${cs_yodas_train_ratio}"
  --cs_yodas_valid_ratio "${cs_yodas_valid_ratio}"
  --cs_yodas_split_seed "${cs_yodas_split_seed}"
  --cs_yodas_max_train_valid_duration_sec "${cs_yodas_max_train_valid_duration_sec}"
  --token_format "${token_format}"
  --allow_unseen_eval_labels "${allow_unseen_eval_labels}"
  --require_eval_labels_in_fleurs_lid "${require_eval_labels_in_fleurs_lid}"
  --allow_single_cs "${allow_single_cs}"
  --strict_labels "${strict_labels}"
  --exclude_fleurs_train_overlaps "${exclude_fleurs_train_overlaps}"
  --min_train_duration_sec "${min_train_duration_sec}"
  --max_train_duration_sec "${max_train_duration_sec}"
  --min_eval_duration_sec "${min_eval_duration_sec}"
  --max_eval_duration_sec "${max_eval_duration_sec}"
  --duration_missing_policy "${duration_missing_policy}"
)
[ -n "${cs_root}" ] && local_data_opts+=(--cs_root "${cs_root}")
[ -n "${fleurs_cache_dir}" ] && local_data_opts+=(--fleurs_cache_dir "${fleurs_cache_dir}")
[ -n "${fleurs_manifest_root}" ] && local_data_opts+=(--fleurs_manifest_root "${fleurs_manifest_root}")
[ -n "${label_map}" ] && local_data_opts+=(--label_map "${label_map}")

gen_prefix="${generated_config_dir}/${base_name}_${train_set}_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_3pass30ep"
verify_allow_pair_labels=false
case " ${train_set} ${valid_set} ${test_sets} " in
  *lid_pair*) verify_allow_pair_labels=true ;;
esac

validate_generated_budget() {
  local budget_json=$1
  python3 local/validate_fixed_batch_budget.py \
    --budget_json "${budget_json}" \
    --train_dir "data/${train_set}" \
    --count_file "${budget_count_file}" \
    --task lid \
    --batch_size "${train_batch_size}" \
    --accum_grad "${accum_grad}" \
    --effective_batch_size "${effective_batch_size}" \
    --ngpu "${ngpu}" \
    --max_epoch "${budget_max_epoch}" \
    --target_passes "${target_passes}" \
    --warmup_ratio "${warmup_ratio}" \
    --round_updates_per_epoch_to "${round_updates_per_epoch_to}" \
    --batch_type "${fixed_batch_type}" \
    --drop_last_iter "${drop_last_iter}"
}

budget_prepared_data=false
if "${auto_training_budget}" && [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  if [ ! -s "data/${train_set}/${budget_count_file}" ] || [ "${stage}" -le 1 ]; then
    echo "Preparing data before generating 3-pass training budget for ${train_set}" >&2
    local/data.sh "${local_data_opts[@]}"
    budget_prepared_data=true
  else
    verify_require_cs=true
    [ -z "${cs_root}" ] && verify_require_cs=false
    local/verify_lid_data.sh \
      --data_dir data \
      --require_cs "${verify_require_cs}" \
      --train_set "${train_set}" \
      --valid_set "${valid_set}" \
      --test_sets "${test_sets}" \
      --max_train_duration_sec "${max_wav_duration}" \
      --allow_pair_labels "${verify_allow_pair_labels}"
  fi
  mkdir -p "${generated_config_dir}" data/local
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
    --budget_tsv "data/local/budget_lid_${train_set}_bs${train_batch_size}_ag${accum_grad}.tsv" \
    > "data/local/budget_lid_${train_set}_bs${train_batch_size}_ag${accum_grad}.log"
  cat "data/local/budget_lid_${train_set}_bs${train_batch_size}_ag${accum_grad}.log" >&2
  validate_generated_budget "${gen_prefix}.budget.json" >&2
  lid_config="${gen_prefix}.yaml"
elif "${auto_training_budget}" && [ "${stage}" -gt 5 ] && [ "${stop_stage}" -ge 6 ]; then
  case "${lid_config}" in
    ${generated_config_dir}/*.yaml)
      [ -s "${lid_config}" ] || {
        echo "Error: explicit generated LID config missing: ${lid_config}" >&2
        exit 2
      }
      validate_generated_budget "${lid_config%.yaml}.budget.json" >&2
      ;;
    *)
      if [ -s "${gen_prefix}.yaml" ]; then
        lid_config="${gen_prefix}.yaml"
        validate_generated_budget "${gen_prefix}.budget.json" >&2
      else
        echo "Error: generated LID config missing for resume: ${gen_prefix}.yaml" >&2
        echo "Run this wrapper once with --stage 1 --stop_stage 5 for the same train_batch_size/accum_grad, or pass the matching generated --lid_config explicitly." >&2
        exit 2
      fi
      ;;
  esac
fi

if "${budget_prepared_data}" && ! "${skip_fleurs}" && [ -z "${fleurs_manifest_root}" ]; then
  # Freeze the TSVs that were just used for budget generation before lid.sh
  # calls local/data.sh again.
  local_data_opts+=(--skip_fleurs_download true --fleurs_tsv_root "${fleurs_tsv_root}")
fi
local_data_opts_str="${local_data_opts[*]}"

if [ "${stage}" -le 7 ]; then
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
    --inference_model "${inference_model}" \
    --inference_batch_size "${inference_batch_size}" \
    --extract_embd "${extract_embd}" \
    --local_data_opts "${local_data_opts_str}"
fi

if "${run_topk_baselines}" && [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
  extra=()
  "${apply_loss_scale}" && extra+=(--apply_loss_scale)
  case "${feats_type}" in
    raw) data_feats="${dumpdir}/raw" ;;
    raw_copy) data_feats="${dumpdir}/raw_copy" ;;
    fbank) data_feats="${dumpdir}/fbank" ;;
    extracted) data_feats="${dumpdir}/extracted" ;;
    *) data_feats="${dumpdir}/${feats_type}" ;;
  esac
  local/score_lid_topk.sh \
    --ngpu "${ngpu}" \
    --topk "${topk}" \
    --batch_size "${baseline_batch_size}" \
    --num_workers "${baseline_num_workers}" \
    --expdir "${expdir}" \
    --lid_tag "${lid_tag}" \
    --inference_model "${inference_model}" \
    --data_feats "${data_feats}" \
    --label_map "data/local/label_map.used.tsv" \
    "${extra[@]}"
fi
