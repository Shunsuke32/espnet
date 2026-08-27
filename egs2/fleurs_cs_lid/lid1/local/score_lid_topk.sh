#!/usr/bin/env bash
set -euo pipefail

ngpu=0
python=python3
topk=2
batch_size=8
num_workers=4
expdir=exp
lid_tag=fleurs_lid_mms_ecapa
lid_exp=
inference_model=valid.accuracy.best.pth
data_feats=dump/raw
train_set=train_fleurs_lid
test_sets="test_fleurs_lid test_cs_read_test test_cs_xtts_test1 test_cs_xtts_test2 test_cs_mms_test test_cs_all"
label_map=data/local/label_map.used.tsv
outdir=
apply_loss_scale=false
require_dump=true
save_full_output=false
full_outdir=
copy_baselines=true
prediction_mode=individual_topk

. utils/parse_options.sh

if [ -z "${lid_exp}" ]; then
  lid_exp="${expdir}/lid_${lid_tag}"
fi
if [ -z "${outdir}" ]; then
  outdir="${lid_exp}/topk_baselines"
fi
if [ -z "${full_outdir}" ]; then
  full_outdir="${outdir}/posterior_jsonl"
fi

config="${lid_exp}/config.yaml"
model="${lid_exp}/${inference_model}"
lang2utt="${data_feats}/${train_set}/lang2utt"
if [ ! -f "${lang2utt}" ]; then
  lang2utt="data/${train_set}/lang2utt"
fi

if [ ! -f "${config}" ]; then echo "Missing LID config: ${config}" >&2; exit 1; fi
if [ ! -f "${model}" ]; then echo "Missing LID model: ${model}" >&2; exit 1; fi
if [ ! -f "${lang2utt}" ]; then echo "Missing train lang2utt: ${lang2utt}" >&2; exit 1; fi

mkdir -p "${outdir}"
for dset in ${test_sets}; do
  wav_scp="${data_feats}/${dset}/wav.scp"
  ref="${data_feats}/${dset}/utt2langs"
  if [ ! -f "${wav_scp}" ]; then
    if "${require_dump}"; then
      echo "ERROR: Missing formatted wav.scp: ${wav_scp}" >&2
      echo "       Include ${dset} in --test_sets and run ESPnet formatting stage before top-k scoring." >&2
      exit 1
    fi
    echo "WARNING: falling back to unformatted data/${dset}/wav.scp" >&2
    wav_scp="data/${dset}/wav.scp"
  fi
  # utt2langs is a text reference and may remain only in data/ depending on
  # template copy helpers; falling back here is safe.
  if [ ! -f "${ref}" ]; then
    ref="data/${dset}/utt2langs"
  fi
  if [ ! -f "${wav_scp}" ] || [ ! -f "${ref}" ]; then
    echo "ERROR: missing wav.scp or utt2langs for ${dset}: wav_scp=${wav_scp} ref=${ref}" >&2
    exit 1
  fi
  extra=()
  if "${apply_loss_scale}"; then
    extra+=(--apply_loss_scale)
  fi
  tsv="${outdir}/${dset}.top${topk}.tsv"
  json="${outdir}/${dset}.top${topk}.json"
  if "${save_full_output}"; then
    extra+=(--full_output "${full_outdir}/${dset}.top${topk}.posterior.jsonl.gz")
  fi
  "${python}" local/lid_topk_softmax.py \
    --lid_train_config "${config}" \
    --lid_model_file "${model}" \
    --lang2utt "${lang2utt}" \
    --wav_scp "${wav_scp}" \
    --ref_utt2langs "${ref}" \
    --label_map "${label_map}" \
    --topk "${topk}" \
    --prediction_mode "${prediction_mode}" \
    --batch_size "${batch_size}" \
    --num_workers "${num_workers}" \
    --ngpu "${ngpu}" \
    --output "${tsv}" \
    --results "${json}" \
    --details_out "${outdir}/${dset}.cardinality_matched_topk.details.tsv" \
    "${extra[@]}"
  if ! "${copy_baselines}"; then
    continue
  fi
  if [ "${dset}" = "test_fleurs_lid" ]; then
    cp "${tsv}" "${lid_exp}/baseline_fleurs_top${topk}.tsv"
    cp "${json}" "${lid_exp}/baseline_fleurs_top${topk}.json"
  elif [ "${dset}" = "test_cs_all" ]; then
    cp "${tsv}" "${lid_exp}/baseline_cs_top${topk}.tsv"
    cp "${json}" "${lid_exp}/baseline_cs_top${topk}.json"
  fi
done
