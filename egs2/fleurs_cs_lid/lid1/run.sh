#!/usr/bin/env bash
set -euo pipefail

# One ESPnet LID recipe for hard, soft KL, and multi-hot BCE targets.
# The model and preprocessor are selected by --lid_config.
profile=fleurs_only
train_set=
valid_set=
test_sets=
tsne_set=

ngpu=2
nj=32
dumpdir=dump
gpu_inference=true
num_nodes=1
stage=1
stop_stage=5
feats_type=
fs=16k
audio_format=wav
# Source metadata filtering is followed by a recipe-local actual-length view
# for hard/BCE raw runs. KL mixed and every CS-all run retain source membership.
min_wav_duration=1.0
max_wav_duration=
expdir=exp
lid_config=conf/train_fleurs_lid_mms_ecapa.yaml
lid_tag=
lid_args=
lid_label_file=
lid_stats_dir=
inference_model=valid.accuracy.best.pth
inference_batch_size=4
extract_embd=false

# Fixed-count 3-pass schedule.  ESPnet2 batch_size is global, not per GPU.
auto_training_budget=true
enforce_training_policy=true
require_cuda_visible_devices=true
train_batch_size=
batch_size=   # alias: --batch_size 32/16/8/4 may be used instead of --train_batch_size
accum_grad=
effective_batch_size=32
target_passes=3.0
budget_max_epoch=
warmup_ratio=0.1
round_updates_per_epoch_to=
fixed_batch_type=
drop_last_iter=true
generated_config_dir=conf/generated
budget_count_file=wav.scp

# Data options. cs_root is optional for training but needed for CS-FLEURS eval.
fleurs_config=all
fleurs_download_dir=${FLEURS:-downloads/fleurs}
fleurs_tsv_root=
fleurs_audio_root=
fleurs_cache_dir=downloads/cache
fleurs_revision=70bb2e84b976b7e960aa89f1c648e09c59f894dd
fleurs_manifest_root=
fleurs_subsample_per_lang=0
skip_fleurs=false
skip_fleurs_download=true
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

. utils/parse_options.sh

case "${audio_format}" in
  *ark*)
    echo "Error: this LID recipe trains with sound inputs; use wav or flac, not an archive format." >&2
    exit 2
    ;;
esac

if [ "${stop_stage}" -gt 5 ]; then
  echo "Error: use local/evaluate.sh for LID top-k and threshold evaluation after Stage 5." >&2
  exit 2
fi

# Read only the routing defaults here; ESPnet validates the complete YAML.
config_defaults=$(python3 - "${lid_config}" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
preprocessor = config.get("preprocessor", "lid")
if preprocessor not in {"lid", "lid_softlabel", "lid_multilabel"}:
    raise SystemExit("Expected a LID hard, soft-label, or multi-label preprocessor")
print("utt2lang" if preprocessor == "lid" else "utt2langs")
print(config.get("max_epoch", 30))
print(config.get("batch_type", "catbel"))
print(preprocessor)
PY
)
readarray -t config_defaults <<< "${config_defaults}"
: "${lid_label_file:=${config_defaults[0]}}"
: "${budget_max_epoch:=${config_defaults[1]}}"
: "${fixed_batch_type:=${config_defaults[2]}}"
if [ -z "${round_updates_per_epoch_to}" ]; then
  round_updates_per_epoch_to=10
  # The 15-epoch KL run doubled the 30-epoch budget's iterations exactly.
  [ "${budget_max_epoch}" -eq 15 ] && round_updates_per_epoch_to=20
fi
if [ "${lid_label_file}" != "${config_defaults[0]}" ]; then
  echo "Error: ${lid_config} requires ${config_defaults[0]}, got ${lid_label_file}" >&2
  exit 2
fi

case "${profile}" in
  fleurs_only)
    : "${train_set:=train_fleurs_lid}"
    : "${valid_set:=valid_fleurs_lid}"
    : "${max_wav_duration:=30}"
    ;;
  mixed|csall)
    data_target=lidseq
    [ "${lid_label_file}" = utt2lang ] && data_target=lid_pair_cs
    if [ "${profile}" = csall ]; then
      [ "${data_target}" = lidseq ] && data_target=lidseq_csall_yodas
      [ "${data_target}" = lid_pair_cs ] && data_target=lid_pair_csall_yodas
      : "${max_wav_duration:=70}"
    else
      : "${max_wav_duration:=30}"
    fi
    : "${train_set:=train_${data_target}}"
    : "${valid_set:=valid_${data_target}}"
    ;;
  *) echo "Error: profile must be fleurs_only, mixed, or csall" >&2; exit 2 ;;
esac
historical_feats_type=raw
default_batch=8
default_accum=4
case "${config_defaults[3]}:${profile}" in
  lid:csall|lid_multilabel:mixed) default_batch=4; default_accum=8 ;;
  lid_multilabel:fleurs_only) default_batch=16; default_accum=2 ;;
esac
: "${train_batch_size:=${default_batch}}"
: "${accum_grad:=${default_accum}}"
if [ "${profile}" = csall ] || [ "${config_defaults[3]}" = lid_softlabel ]; then
  historical_feats_type=raw_copy
fi
: "${feats_type:=${historical_feats_type}}"
if [ "${feats_type}" != "${historical_feats_type}" ]; then
  echo "Error: OLD ${profile}/${config_defaults[3]} requires --feats_type ${historical_feats_type}; changing it changes historical membership." >&2
  exit 2
fi
case "${fs}" in
  16k|16000) ;;
  *) echo "Error: OLD duration/sample-count identity requires --fs 16k." >&2; exit 2 ;;
esac
if [ "${min_wav_duration}" != 1.0 ] && [ "${min_wav_duration}" != 1 ]; then
  echo "Error: OLD raw filtering requires the strict 1-second lower bound." >&2
  exit 2
fi
if [ -z "${test_sets}" ]; then
  test_sets="test_fleurs_lid"
  [ -n "${cs_root}" ] && test_sets+=" test_cs_read_test test_cs_xtts_test1 test_cs_xtts_test2 test_cs_mms_test"
  [ -n "${cs_yodas_root}" ] && test_sets+=" test_yodas_lidseq"
fi
if [ "${stage}" -le 1 ] && [ "${profile}" != fleurs_only ] && [ -z "${cs_root}" ]; then
  echo "Error: mixed and csall preparation require --cs_root" >&2
  exit 2
fi
if [ "${stage}" -le 1 ] && [ "${profile}" = csall ] && [ -z "${cs_yodas_root}" ]; then
  echo "Error: csall preparation requires --cs_yodas_root" >&2
  exit 2
fi

needs_training_or_eval_gpu() {
  [ "${ngpu}" -gt 0 ] && [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]
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
  [ "${budget_max_epoch}" -eq 30 ] || [ "${budget_max_epoch}" -eq 15 ] || {
    echo "Error: budget_max_epoch must be 30 or 15; got ${budget_max_epoch}" >&2
    exit 2
  }
  case "${target_passes}" in
    3|3.0|3.00|3.000|5|5.0) ;;
    *)
      echo "Error: target_passes must be 3 or 5; got ${target_passes}" >&2
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
if needs_training_or_eval_gpu && [ $((train_batch_size / ngpu)) -lt 2 ]; then
  echo "Error: ECAPA training requires at least 2 utterances per GPU for BatchNorm." >&2
  exit 2
fi
validate_training_policy
validate_cuda_visible_devices

base_name=$(basename "${lid_config}" .yaml)
model_label=${base_name#train_fleurs_lid_}
[ "${model_label}" = "${base_name}" ] && model_label=${base_name#train_}
schedule_name="${target_passes%.*}pass${budget_max_epoch}ep"
[ -z "${lid_tag}" ] && lid_tag="${train_set}_${model_label}_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_${schedule_name}"
[ -z "${lid_stats_dir}" ] && lid_stats_dir="${expdir}/lid_stats_${lid_tag}"

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
[ -n "${cs_yodas_root}" ] && local_data_opts+=(--cs_yodas_root "${cs_yodas_root}")
[ -n "${fleurs_cache_dir}" ] && local_data_opts+=(--fleurs_cache_dir "${fleurs_cache_dir}")
[ -n "${fleurs_manifest_root}" ] && local_data_opts+=(--fleurs_manifest_root "${fleurs_manifest_root}")
[ -n "${fleurs_audio_root}" ] && local_data_opts+=(--fleurs_audio_root "${fleurs_audio_root}")
[ -n "${label_map}" ] && local_data_opts+=(--label_map "${label_map}")

if [ "${stage}" -le 1 ]; then
  for root in "${fleurs_tsv_root}" "${fleurs_audio_root}" "${fleurs_manifest_root}" "${fleurs_download_dir}" "${fleurs_cache_dir}" "${cs_root}" "${cs_yodas_root}" "${label_map}"; do
    case "${root}" in
      *[[:space:]]*)
        echo "Error: wrapper Stage 1 does not support whitespace in input paths. Run local/data.sh directly with quoted paths, then start the wrapper at Stage 3." >&2
        exit 2 ;;
    esac
  done
fi

gen_prefix="${generated_config_dir}/${base_name}_${train_set}_${fixed_batch_type}_bs${train_batch_size}_ag${accum_grad}_eb${effective_batch_size}_${schedule_name}"
verify_allow_pair_labels=false
case " ${train_set} ${valid_set} ${test_sets} " in
  *lid_pair*) verify_allow_pair_labels=true ;;
esac

validate_generated_budget() {
  local budget_json=$1
  python3 local/validate_fixed_batch_budget.py \
    --budget_json "${budget_json}" \
    --train_dir "${budget_train_dir}" \
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

local_data_opts_str="${local_data_opts[*]}"
if [ "${stage}" -gt 1 ] && [ "${stage}" -le 5 ]; then
  if [ ! -s "data/${train_set}/${budget_count_file}" ]; then
    echo "Error: missing prepared data for ${train_set}; run Stage 1 explicitly. Resume never regenerates data." >&2
    exit 2
  fi
  verify_require_cs=true
  [ -z "${cs_root}" ] && verify_require_cs=false
  local/verify_lid_data.sh \
    --data_dir data --require_cs "${verify_require_cs}" \
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}" \
    --min_train_duration_sec 1.0 --max_train_duration_sec "${max_wav_duration}" \
    --max_non_yodas_train_duration_sec 30.0 --max_yodas_train_duration_sec 70.0 \
    --allow_pair_labels "${verify_allow_pair_labels}"
fi
invoke_template() {
  ./lid.sh \
    --stage "$1" --stop_stage "$2" --dumpdir "$3" \
    --ngpu "${ngpu}" --num_nodes "${num_nodes}" --nj "${nj}" \
    --gpu_inference "${gpu_inference}" --feats_type "${feats_type}" \
    --fs "${fs}" --audio_format "${audio_format}" \
    --min_wav_duration "${min_wav_duration}" --max_wav_duration "${max_wav_duration}" \
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}" \
    --tsne_set "${tsne_set}" --expdir "${expdir}" --lid_config "${lid_config}" \
    --lid_label_file "${lid_label_file}" --lid_stats_dir "${lid_stats_dir}" \
    --lid_tag "${lid_tag}" --lid_args "${lid_args}" \
    --inference_model "${inference_model}" --inference_batch_size "${inference_batch_size}" \
    --extract_embd "${extract_embd}" --local_data_opts "${local_data_opts_str}"
}

# Preserve the standard Stage 3 format / Stage 4 stats / Stage 5 train boundary.
if [ "${stage}" -le 3 ]; then
  preparation_stop=${stop_stage}
  [ "${preparation_stop}" -gt 3 ] && preparation_stop=3
  invoke_template "${stage}" "${preparation_stop}" "${dumpdir}"
fi
training_dumpdir=${dumpdir}
budget_train_dir="data/${train_set}"
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 3 ]; then
  if [ "${feats_type}" = raw ]; then
    training_dumpdir="${dumpdir}/old_duration"
    for dset in "${train_set}" "${valid_set}"; do
      python3 ../local/finalize_duration_view.py \
        --input-dir "${dumpdir}/raw/${dset}" --source-data-dir "data/${dset}" \
        --mode raw --output-dir "${training_dumpdir}/raw/${dset}" --reuse-existing
    done
    budget_train_dir="${training_dumpdir}/raw/${train_set}"
  else
    # Audit only: KL mixed historically reads the direct metadata membership.
    # CS-all likewise keeps OLD raw_copy counts, including the Persian record.
    raw_copy_audit_opts=(--require-no-drops)
    if [ "${profile}" != csall ] && [ "${config_defaults[3]}" = lid_softlabel ]; then
      # Direct-data KL used metadata 1s <= duration < 30s, not strict sample bounds.
      raw_copy_audit_opts=()
      echo "KL raw_copy: preserving direct-data metadata membership (1s inclusive); strict-sample exclusions below are diagnostic only." >&2
    fi
    for dset in "${train_set}" "${valid_set}"; do
      python3 ../local/finalize_duration_view.py \
        --input-dir "${dumpdir}/raw_copy/${dset}" --source-data-dir "data/${dset}" \
        --mode raw_copy --audit-only "${raw_copy_audit_opts[@]}"
    done
  fi
fi

if "${auto_training_budget}" && [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
  if [ ! -s "${budget_train_dir}/${budget_count_file}" ]; then
    echo "Error: missing prepared training view; run Stages 1-3 explicitly. Resume never regenerates source data." >&2
    exit 2
  fi
  mkdir -p "${generated_config_dir}" data/local
  python3 local/make_fixed_batch_config.py \
    --task lid \
    --base_config "${lid_config}" \
    --output_config "${gen_prefix}.yaml" \
    --train_dir "${budget_train_dir}" \
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
fi
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 4 ]; then
  training_stage=${stage}
  [ "${training_stage}" -lt 4 ] && training_stage=4
  invoke_template "${training_stage}" "${stop_stage}" "${training_dumpdir}"
fi
