#!/usr/bin/env bash
set -euo pipefail

# ESPnet ASR-style variable-cardinality LID recipe.
# Target text is a sequence of language tokens such as <eng> or <ara> <eng>.
#
# Training policy:
#   - default training uses 2 GPUs
#   - ESPnet2 batch_size is global; it is not multiplied by ngpu
#   - use fixed utterance-count mini-batches, not batch_bins/numel/length bins
#   - effective batch size = train_batch_size * accum_grad = 32
#   - sequence MMS runs default to 8 x 4; 16 x 2 can OOM on 30s batches
#   - full runs use 30 epochs; the small-model frozen prefix uses 10 epochs
#   - num_iters_per_epoch and warmup are generated after data prep from the
#     actual filtered train-set size and requested paper schedule
#
# train_mode=mixed  : FLEURS single-label examples + CS-FLEURS code-switch examples
# train_mode=fleurs : FLEURS-only encoder-decoder baseline

ngpu=2
nj=32
dumpdir=dump
inference_nj=1
gpu_inference=true
num_nodes=1
stage=1
stop_stage=13
feats_type=raw
fs=16k
audio_format=wav
# local/data.sh applies the source-specific duration policy first. The ASR
# template then enforces the selected run's common upper bound on train/valid.
min_wav_duration=0.999999
max_wav_duration=
feats_normalize=uttmvn
lang=lidseq
asr_stats_dir=

train_mode=mixed
train_set=
valid_set=
test_sets=

asr_config=conf/train_lidseq_mms_transformer24.yaml
inference_config=conf/decode_lidseq_mms_transformer.yaml
asr_tag=
inference_tag=lidseq_mms_transformer24_beam5
inference_asr_model=valid.loss.best.pth
pretrained_model=
ignore_init_mismatch=false
nlsyms_txt=data/nlsyms.txt
asr_args=
inference_args=

# Fixed-count paper schedules. ESPnet2 batch_size is global, not per GPU.
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
fixed_batch_type=sorted
drop_last_iter=true
generated_config_dir=conf/generated
budget_count_file=wav.scp

run_lidseq_scoring=true
decode_dir=
score_label_map=data/local/label_map.used.tsv

# Data options
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

default_asr_config=${asr_config}

. utils/parse_options.sh

needs_training_or_eval_gpu() {
  [ "${ngpu}" -gt 0 ] && [ "${stage}" -le 13 ] && [ "${stop_stage}" -ge 10 ]
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
    16:2|8:4|4:8) ;;
    *)
      echo "Error: allowed fixed ASR schedules are 16/2 memory-probe, 8/4 default, or 4/8 fallback; got ${train_batch_size}/${accum_grad}" >&2
      exit 2
      ;;
  esac
  [ "${effective_batch_size}" -eq 32 ] || {
    echo "Error: effective_batch_size must be 32; got ${effective_batch_size}" >&2
    exit 2
  }
  case "${target_passes}" in
    1|1.0|1.00|1.000)
      [ "${budget_max_epoch}" -eq 10 ] || {
        echo "Error: the 1-pass frozen-prefix schedule requires budget_max_epoch=10; got ${budget_max_epoch}" >&2
        exit 2
      }
      case "${warmup_ratio}" in
        0.3|.3|0.30|.30|0.300|.300) ;;
        *)
          echo "Error: the 1-pass frozen-prefix schedule requires warmup_ratio=0.3; got ${warmup_ratio}" >&2
          exit 2
          ;;
      esac
      ;;
    3|3.0|3.00|3.000|5|5.0|5.00|5.000) ;;
    *)
      echo "Error: paper schedules support 1 pass/10 epochs or 3/5 passes over 30 epochs; got target_passes=${target_passes}" >&2
      exit 2
      ;;
  esac
  case "${target_passes}" in
    3|3.0|3.00|3.000|5|5.0|5.00|5.000)
      [ "${budget_max_epoch}" -eq 30 ] || {
        echo "Error: 3/5-pass schedules require budget_max_epoch=30; got ${budget_max_epoch}" >&2
        exit 2
      }
      ;;
  esac
  check_no_protected_args asr_args "${asr_args}"
}

validate_cuda_visible_devices() {
  "${require_cuda_visible_devices}" || return 0
  needs_training_or_eval_gpu || return 0
  [ "${stop_stage}" -ge 11 ] || return 0
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

base_name=$(basename "${asr_config}" .yaml)
model_label=${base_name#train_lidseq_}
[ "${model_label}" = "${base_name}" ] && model_label=${base_name#train_}

case "${train_mode}" in
  mixed)
    default_train_set=train_lidseq
    default_valid_set=valid_lidseq
    ;;
  fleurs|fleurs_only|fleurs-only)
    default_train_set=train_fleurs_lidseq
    default_valid_set=valid_fleurs_lidseq
    ;;
  csall|cs_all|cs-all)
    default_train_set=train_lidseq_csall_yodas
    default_valid_set=valid_lidseq_csall_yodas
    [ -n "${cs_yodas_root}" ] || {
      echo "Error: --train_mode csall requires --cs_yodas_root" >&2
      exit 2
    }
    ;;
  *)
    echo "Unsupported --train_mode ${train_mode}. Use fleurs, mixed, or csall." >&2
    exit 2
    ;;
esac

[ -z "${train_set}" ] && train_set=${default_train_set}
[ -z "${valid_set}" ] && valid_set=${default_valid_set}
if [ -z "${max_wav_duration}" ]; then
  case "${train_mode}" in
    csall|cs_all|cs-all) max_wav_duration=70 ;;
    *) max_wav_duration=30 ;;
  esac
fi

case "${target_passes}" in
  1|1.0|1.00|1.000) schedule_passes=1 ;;
  3|3.0|3.00|3.000) schedule_passes=3 ;;
  5|5.0|5.00|5.000) schedule_passes=5 ;;
esac
schedule_name="${schedule_passes}pass${budget_max_epoch}ep"

if [ -z "${asr_tag}" ]; then
  case "${train_mode}" in
    mixed)
      asr_tag="lidseq_mixed_${model_label}_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_${schedule_name}"
      ;;
    fleurs|fleurs_only|fleurs-only)
      asr_tag="lidseq_fleurs_${model_label}_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_${schedule_name}"
      ;;
    csall|cs_all|cs-all)
      asr_tag="lidseq_csall_yodas_${model_label}_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_${schedule_name}"
      ;;
  esac
fi
[ -z "${asr_stats_dir}" ] && asr_stats_dir="exp/asr_stats_${asr_tag}"
if [ -z "${test_sets}" ]; then
  test_sets="test_fleurs_lid test_cs_read_test test_cs_xtts_test1 test_cs_xtts_test2 test_cs_mms_test test_cs_all"
  case "${train_mode}" in
    csall|cs_all|cs-all) test_sets+=" test_yodas_lidseq" ;;
  esac
fi
if [ -z "${cs_root}" ]; then
  echo "Warning: --cs_root is empty. Only FLEURS data will be prepared unless CS_FLEURS_ROOT is set." >&2
  test_sets="test_fleurs_lid"
fi

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
  --nlsyms_txt "${nlsyms_txt}"
)
[ -n "${cs_root}" ] && local_data_opts+=(--cs_root "${cs_root}")
[ -n "${fleurs_cache_dir}" ] && local_data_opts+=(--fleurs_cache_dir "${fleurs_cache_dir}")
[ -n "${fleurs_manifest_root}" ] && local_data_opts+=(--fleurs_manifest_root "${fleurs_manifest_root}")
[ -n "${label_map}" ] && local_data_opts+=(--label_map "${label_map}")

# Need the actual post-filter train-set count before training.  Prepare data once
# here, generate a config, then let asr.sh run normally.  local/data.sh is
# deterministic/idempotent; the second call inside asr.sh should reproduce dirs.
gen_prefix="${generated_config_dir}/${base_name}_${train_set}_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_${schedule_name}"

validate_generated_budget() {
  local budget_json=$1
  python3 local/validate_fixed_batch_budget.py \
    --budget_json "${budget_json}" \
    --train_dir "data/${train_set}" \
    --count_file "${budget_count_file}" \
    --task asr \
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
if "${auto_training_budget}" && [ "${stage}" -le 10 ] && [ "${stop_stage}" -ge 10 ]; then
  if [ ! -s "data/${train_set}/${budget_count_file}" ] || [ "${stage}" -le 1 ]; then
    echo "Preparing data before generating ${schedule_name} training budget for ${train_set}" >&2
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
      --max_train_duration_sec "${max_wav_duration}"
  fi
  mkdir -p "${generated_config_dir}" data/local
  python3 local/make_fixed_batch_config.py \
    --task asr \
    --base_config "${asr_config}" \
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
    --budget_tsv "data/local/budget_asr_${train_set}_bs${train_batch_size}_ag${accum_grad}.tsv" \
    > "data/local/budget_asr_${train_set}_bs${train_batch_size}_ag${accum_grad}.log"
  cat "data/local/budget_asr_${train_set}_bs${train_batch_size}_ag${accum_grad}.log" >&2
  validate_generated_budget "${gen_prefix}.budget.json" >&2
  asr_config="${gen_prefix}.yaml"
elif "${auto_training_budget}" && [ "${stage}" -gt 10 ] && [ "${stop_stage}" -ge 11 ]; then
  case "${asr_config}" in
    ${generated_config_dir}/*.yaml)
      [ -s "${asr_config}" ] || {
        echo "Error: explicit generated ASR config missing: ${asr_config}" >&2
        exit 2
      }
      validate_generated_budget "${asr_config%.yaml}.budget.json" >&2
      ;;
    *)
      if [ -s "${gen_prefix}.yaml" ]; then
        asr_config="${gen_prefix}.yaml"
        validate_generated_budget "${gen_prefix}.budget.json" >&2
      else
        echo "Error: generated ASR config missing for resume: ${gen_prefix}.yaml" >&2
        echo "Run this wrapper once with --stage 1 --stop_stage 10 for the same train_batch_size/accum_grad/train_mode, or pass the matching generated --asr_config explicitly." >&2
        exit 2
      fi
      ;;
  esac
fi

if "${budget_prepared_data}" && ! "${skip_fleurs}" && [ -z "${fleurs_manifest_root}" ]; then
  # Freeze the TSVs that were just used for budget generation before asr.sh
  # calls local/data.sh again.
  local_data_opts+=(--skip_fleurs_download true --fleurs_tsv_root "${fleurs_tsv_root}")
fi
local_data_opts_str="${local_data_opts[*]}"

asr_stop_stage=${stop_stage}
if "${run_lidseq_scoring}" && [ "${stage}" -le 13 ] && [ "${stop_stage}" -ge 13 ]; then
  asr_stop_stage=12
  echo "Info: run_lidseq_scoring=true; passing --stop_stage ${asr_stop_stage} to asr.sh instead of user stop_stage=${stop_stage} to skip ESPnet stage 13 scoring." >&2
  echo "Info: Custom LID-sequence scoring will run from this wrapper after asr.sh." >&2
fi

resolve_lidseq_decode_dir() {
  local tag=${inference_tag}
  if [ -n "${decode_dir}" ]; then
    return 0
  fi
  if [ -z "${tag}" ]; then
    if [ -n "${inference_config}" ]; then
      tag=$(basename "${inference_config}" .yaml)
    else
      tag=inference
    fi
    if [ -n "${inference_args}" ]; then
      tag+="$(echo "${inference_args}" | sed -e "s/--/_/g" -e "s/[ |=]//g")"
    fi
    tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
  fi
  decode_dir="exp/asr_${asr_tag}/${tag}"
}

./asr.sh \
  --stage "${stage}" \
  --stop_stage "${asr_stop_stage}" \
  --ngpu "${ngpu}" \
  --num_nodes "${num_nodes}" \
  --nj "${nj}" \
  --dumpdir "${dumpdir}" \
  --inference_nj "${inference_nj}" \
  --gpu_inference "${gpu_inference}" \
  --feats_type "${feats_type}" \
  --fs "${fs}" \
  --audio_format "${audio_format}" \
  --min_wav_duration "${min_wav_duration}" \
  --max_wav_duration "${max_wav_duration}" \
  --feats_normalize "${feats_normalize}" \
  --lang "${lang}" \
  --asr_stats_dir "${asr_stats_dir}" \
  --use_lm false \
  --use_ngram false \
  --token_type word \
  --nbpe 0 \
  --nlsyms_txt "${nlsyms_txt}" \
  --asr_config "${asr_config}" \
  --inference_config "${inference_config}" \
  --asr_tag "${asr_tag}" \
  --inference_tag "${inference_tag}" \
  --inference_asr_model "${inference_asr_model}" \
  --pretrained_model "${pretrained_model}" \
  --ignore_init_mismatch "${ignore_init_mismatch}" \
  --asr_args "${asr_args}" \
  --inference_args "${inference_args}" \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_sets}" \
  --local_data_opts "${local_data_opts_str}"

if "${run_lidseq_scoring}" && [ "${stage}" -le 13 ] && [ "${stop_stage}" -ge 12 ]; then
  resolve_lidseq_decode_dir
  echo "Info: Running custom LID-sequence scoring with decode_dir=${decode_dir}" >&2
  if [ -n "${decode_dir}" ] && [ -d "${decode_dir}" ]; then
    local/score_lidseq_decodes.sh --decode_dir "${decode_dir}" --label_map "${score_label_map}" --test_sets "${test_sets}"
  else
    echo "Error: custom LID-sequence scoring failed; decode_dir not found: ${decode_dir}" >&2
    exit 1
  fi
fi
