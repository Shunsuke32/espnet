#!/usr/bin/env bash
set -euo pipefail

data_dir=data
label_map=
require_cs=true
require_audits=true
train_set=
valid_set=
test_sets=
extra_sets=
min_train_duration_sec=1.0
max_train_duration_sec=30.0
max_non_yodas_train_duration_sec=30.0
max_yodas_train_duration_sec=70.0
allow_pair_labels=false
verify_all_data_dirs=false
check_wav_readable_paths=true

. utils/parse_options.sh

if [ -z "${label_map}" ]; then
  label_map=${data_dir}/local/label_map.used.tsv
fi

fail() { echo "ERROR: $*" >&2; exit 1; }
[ -f "${label_map}" ] || fail "missing ${label_map}"

declare -A selected_sets=()
add_set() {
  local name=$1
  [ -n "${name}" ] || return 0
  [ -d "${data_dir}/${name}" ] || return 0
  selected_sets["${name}"]=1
}
add_set_list() {
  local name
  for name in $*; do
    add_set "${name}"
  done
}

for d in train_fleurs_lid valid_fleurs_lid test_fleurs_lid train_fleurs_lidseq valid_fleurs_lidseq; do
  [ -d "${data_dir}/${d}" ] || fail "missing ${data_dir}/${d}"
  add_set "${d}"
done

add_set_list train_lidseq valid_lidseq test_cs_read_test test_cs_xtts_test1 test_cs_xtts_test2 test_cs_mms_test test_cs_all
add_set "${train_set}"
add_set "${valid_set}"
add_set_list "${test_sets}"
add_set_list "${extra_sets}"

if "${verify_all_data_dirs}"; then
  for d in "${data_dir}"/*; do
    [ -d "${d}" ] || continue
    add_set "$(basename "${d}")"
  done
fi

selected_names=("${!selected_sets[@]}")

check_wav_readable() {
  local wav_scp=$1
  python3 - "${wav_scp}" <<'PY'
import os
import sys

wav_scp = sys.argv[1]
bad = []
checked = 0
with open(wav_scp, encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        parts = line.rstrip("\n").split(maxsplit=1)
        if len(parts) != 2:
            continue
        utt, wav = parts
        wav = wav.strip()
        # ESPnet/Kaldi wav.scp can also contain shell pipelines.  Those are
        # validated later by formatting; here we catch ordinary path mistakes.
        if wav.endswith("|") or " " in wav:
            continue
        checked += 1
        if not os.path.isfile(wav) or not os.access(wav, os.R_OK):
            bad.append((lineno, utt, wav))
            if len(bad) >= 5:
                break
if bad:
    print(f"unreadable wav.scp entries in {wav_scp}", file=sys.stderr)
    for lineno, utt, wav in bad:
        print(f"  line {lineno}: {utt} {wav}", file=sys.stderr)
    sys.exit(1)
print(f"checked readable wav paths: {wav_scp} ({checked})", file=sys.stderr)
PY
}

official_map="${data_dir}/local/fleurs_official_label_map.tsv"
if "${require_audits}"; then
  [ -f "${official_map}" ] || fail "missing ${official_map}; rerun local/data.sh before training"
  [ -f "${data_dir}/local/label_inventory.tsv" ] || fail "missing ${data_dir}/local/label_inventory.tsv; rerun local/data.sh before training"
fi
if [ -f "${official_map}" ]; then
  n_official=$(awk 'END{print (NR > 0 ? NR - 1 : 0)}' "${official_map}")
  n_unique=$(awk -F'\t' 'NR>1 {print $2}' "${official_map}" | sort -u | wc -l)
  [ "${n_official}" -eq 102 ] || fail "expected 102 FLEURS labels, got ${n_official}"
  [ "${n_unique}" -eq 102 ] || fail "expected 102 unique FLEURS canonical labels, got ${n_unique}"
fi

for name in "${selected_names[@]}"; do
  d="${data_dir}/${name}"
  [ -f "${d}/wav.scp" ] || continue
  dup=$(awk '{print $1}' "${d}/wav.scp" | sort | uniq -d | head -n 1)
  [ -z "${dup}" ] || fail "duplicate utterance id in ${d}/wav.scp: ${dup}"
  if "${check_wav_readable_paths}"; then
    check_wav_readable "${d}/wav.scp" || fail "unreadable audio path in ${d}/wav.scp"
  fi
done

check_manifest_alignment() {
  local dir=$1
  python3 - "${dir}" <<'PY'
import json
import pathlib
import sys

data_dir = pathlib.Path(sys.argv[1])


def read_kv(path, labels=False):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) != 2:
                raise SystemExit(f"malformed {path}:{lineno}: {line.rstrip()}")
            utt, value = parts
            if utt in rows:
                raise SystemExit(f"duplicate utterance id in {path}:{lineno}: {utt}")
            if labels:
                value = tuple(x.strip("<>") for x in value.split())
            rows[utt] = value
    return rows


wavs = read_kv(data_dir / "wav.scp")
text = read_kv(data_dir / "text", labels=True)
utt2langs = read_kv(data_dir / "utt2langs", labels=True)
utt2lang = read_kv(data_dir / "utt2lang", labels=True)
metadata = {}
with (data_dir / "labels.jsonl").open(encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        row = json.loads(line)
        utt = str(row["uttid"])
        if utt in metadata:
            raise SystemExit(
                f"duplicate utterance id in {data_dir / 'labels.jsonl'}:{lineno}: {utt}"
            )
        metadata[utt] = row

expected = set(wavs)
for name, rows in (
    ("text", text),
    ("utt2langs", utt2langs),
    ("utt2lang", utt2lang),
    ("labels.jsonl", metadata),
):
    if set(rows) != expected:
        missing = sorted(expected - set(rows))[:5]
        extra = sorted(set(rows) - expected)[:5]
        raise SystemExit(
            f"utterance inventory mismatch in {data_dir}/{name}: "
            f"missing={missing}, extra={extra}"
        )

for utt in sorted(expected):
    labels = utt2langs[utt]
    metadata_labels = tuple(str(x) for x in metadata[utt].get("labels", []))
    if text[utt] != labels or metadata_labels != labels:
        raise SystemExit(
            f"label mismatch for {data_dir}/{utt}: "
            f"text={text[utt]}, utt2langs={labels}, labels.jsonl={metadata_labels}"
        )
    if utt2lang[utt] != labels[:1]:
        raise SystemExit(
            f"utt2lang mismatch for {data_dir}/{utt}: "
            f"utt2lang={utt2lang[utt]}, utt2langs={labels}"
        )
PY
}

for name in "${selected_names[@]}"; do
  d="${data_dir}/${name}"
  if [ -f "${d}/wav.scp" ] && [ -f "${d}/text" ] \
      && [ -f "${d}/utt2lang" ] && [ -f "${d}/utt2langs" ] \
      && [ -f "${d}/labels.jsonl" ]; then
    check_manifest_alignment "${d}" || fail "manifest alignment failed in ${d}"
  fi
done

duration_excluded="${data_dir}/local/duration_excluded.tsv"
if "${require_audits}"; then
  [ -f "${duration_excluded}" ] || fail "missing ${duration_excluded}; rerun local/data.sh before training"
fi
if [ -f "${duration_excluded}" ]; then
  python3 - "${duration_excluded}" "${selected_names[@]}" <<'PY'
import csv
import sys

path = sys.argv[1]
selected = set(sys.argv[2:])
with open(path, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        subset = row.get("set") or row.get("subset") or row.get("data")
        if subset in selected and subset.startswith("test_"):
            print(f"test/eval utterance was duration-filtered: {row}", file=sys.stderr)
            sys.exit(1)
PY
fi

duration_summary="${data_dir}/local/duration_summary.tsv"
if "${require_audits}"; then
  [ -f "${duration_summary}" ] || fail "missing ${duration_summary}; rerun local/data.sh before training"
fi
if [ -f "${duration_summary}" ]; then
  python3 - "${duration_summary}" "${min_train_duration_sec}" "${max_train_duration_sec}" "${selected_names[@]}" <<'PY'
import csv
import sys

path = sys.argv[1]
min_dur = float(sys.argv[2])
max_dur = float(sys.argv[3])
selected = set(sys.argv[4:])
with open(path, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        subset = row.get("set") or row.get("subset") or row.get("data")
        if subset not in selected or subset.startswith("test_") or not subset.startswith(("train_", "valid_")):
            continue
        missing = row.get("num_missing_duration") or row.get("missing_duration") or row.get("missing") or "0"
        if missing not in {"", "0", "0.0"}:
            print(f"missing duration in train/valid set: {row}", file=sys.stderr)
            sys.exit(1)
        kept = (row.get("phase") or row.get("status") or "kept") == "kept"
        if not kept:
            continue
        min_seen = row.get("min_sec") or row.get("min_duration_sec") or row.get("min")
        max_seen = row.get("max_sec") or row.get("max_duration_sec") or row.get("max")
        if min_seen not in {None, ""} and float(min_seen) < min_dur:
            print(f"kept train/valid duration below {min_dur}: {row}", file=sys.stderr)
            sys.exit(1)
        if max_seen not in {None, ""} and float(max_seen) >= max_dur:
            print(f"kept train/valid duration outside < {max_dur}: {row}", file=sys.stderr)
            sys.exit(1)
PY
fi

for name in "${selected_names[@]}"; do
  case "${name}" in
    train_*|valid_*) ;;
    *) continue ;;
  esac
  labels_json="${data_dir}/${name}/labels.jsonl"
  [ -f "${labels_json}" ] || continue
  python3 - "${labels_json}" "${min_train_duration_sec}" \
    "${max_train_duration_sec}" "${max_non_yodas_train_duration_sec}" \
    "${max_yodas_train_duration_sec}" <<'PY'
import json
import sys

path = sys.argv[1]
min_dur = float(sys.argv[2])
run_max_dur = float(sys.argv[3])
non_yodas_max_dur = float(sys.argv[4])
yodas_max_dur = float(sys.argv[5])
with open(path, encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        row = json.loads(line)
        dur = row.get("duration_sec", row.get("duration"))
        if dur is None:
            continue
        dur = float(dur)
        source = str(row.get("source") or "")
        source_max_dur = (
            yodas_max_dur if source.startswith("cs_yodas") else non_yodas_max_dur
        )
        max_dur = min(run_max_dur, source_max_dur)
        if dur < min_dur or dur >= max_dur:
            print(
                f"duration outside [{min_dur}, {max_dur}) for source={source} "
                f"in {path}:{lineno}: {dur}",
                file=sys.stderr,
            )
            sys.exit(1)
PY
done

# FLEURS-only training sets must remain FLEURS-only.
if grep -q '^cs_' "${data_dir}/train_fleurs_lid/utt2spk"; then
  fail "CS-FLEURS utterances leaked into train_fleurs_lid"
fi
if grep -q '^cs_' "${data_dir}/train_fleurs_lidseq/utt2spk"; then
  fail "CS-FLEURS utterances leaked into train_fleurs_lidseq"
fi

# FLEURS-only sequence baseline must have exactly one target label per utterance.
awk 'NF != 2 {print "bad FLEURS-only seq text:", $0; bad=1} END{exit bad}' \
  "${data_dir}/train_fleurs_lidseq/text" || fail "train_fleurs_lidseq/text is not single-label"

for d in train_fleurs_lid valid_fleurs_lid test_fleurs_lid train_fleurs_lidseq valid_fleurs_lidseq; do
  [ -f "${data_dir}/${d}/utt2langs" ] || continue
  awk 'NF != 2 {print "bad FLEURS-only utt2langs:", $0; bad=1} END{exit bad}' \
    "${data_dir}/${d}/utt2langs" || fail "${d}/utt2langs is not single-label"
done

check_text_tokens() {
  local text_file=$1
  python3 - "${label_map}" "${allow_pair_labels}" "${text_file}" <<'PY'
import re
import sys

label_map, allow_pair, text_file = sys.argv[1], sys.argv[2] == "true", sys.argv[3]
labels = set()
with open(label_map, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1] != "canonical":
            labels.add(parts[1])
pair_re = re.compile(r"^[a-z]{3}(?:-[a-z]{3})+$")
with open(text_file, encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        parts = line.rstrip("\n").split()
        for tok in parts[1:]:
            lab = tok.strip("<>")
            if lab in labels:
                continue
            if allow_pair and pair_re.fullmatch(lab):
                continue
            print(f"unknown target token: {lab} in {text_file}:{lineno}: {line.rstrip()}", file=sys.stderr)
            sys.exit(1)
PY
}

check_utt2langs_tokens() {
  local utt2langs_file=$1
  python3 - "${label_map}" "${allow_pair_labels}" "${utt2langs_file}" <<'PY'
import re
import sys

label_map, allow_pair, utt2langs_file = sys.argv[1], sys.argv[2] == "true", sys.argv[3]
labels = set()
with open(label_map, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1] != "canonical":
            labels.add(parts[1])
pair_re = re.compile(r"^[a-z]{3}(?:-[a-z]{3})+$")
with open(utt2langs_file, encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        parts = line.rstrip("\n").split()
        for lab in parts[1:]:
            lab = lab.strip("<>")
            if lab in labels:
                continue
            if allow_pair and pair_re.fullmatch(lab):
                continue
            print(f"unknown utt2langs token: {lab} in {utt2langs_file}:{lineno}: {line.rstrip()}", file=sys.stderr)
            sys.exit(1)
PY
}

for name in "${selected_names[@]}"; do
  d="${data_dir}/${name}"
  [ -f "${d}/text" ] && check_text_tokens "${d}/text"
  [ -f "${d}/utt2langs" ] && check_utt2langs_tokens "${d}/utt2langs"
done

for name in "${selected_names[@]}"; do
  d="${data_dir}/${name}"
  [ -f "${d}/utt2langs" ] || continue
  case "${name}" in
    *pair*) continue ;;
  esac
  awk '$1 ~ /^(cs_|csyodas_)/ && NF != 3 {print "bad CS utt2langs:", $0; bad=1} END{exit bad}' \
    "${d}/utt2langs" || fail "${d}/utt2langs has CS utterances that are not exactly two labels"
done

if [ -d "${data_dir}/${train_set}" ] && "${require_cs}"; then
  case "${train_set}" in
    *pair*)
      awk 'NF == 2 && $2 ~ /^[a-z][a-z][a-z]-[a-z][a-z][a-z]/ {ok=1; exit} END{exit !ok}' \
        "${data_dir}/${train_set}/utt2lang" || fail "${train_set} has no pair-class labels"
      ;;
    *lidseq*|*yodas_lidseq*)
      grep -Eq '^(cs_|csyodas_)' "${data_dir}/${train_set}/utt2spk" || fail "${train_set} contains no CS utterances"
      awk 'NF >= 3 {ok=1; exit} END{exit !ok}' "${data_dir}/${train_set}/text" || \
        fail "${train_set} has no multi-label language sequence"
      ;;
  esac
fi

for d in valid_fleurs_lid test_fleurs_lid; do
  [ -f "${data_dir}/${d}/wav.scp" ] || continue
  overlap=$(awk 'NR==FNR {w[$2]=1; next} $2 in w {print; exit}' \
    "${data_dir}/${d}/wav.scp" "${data_dir}/train_fleurs_lid/wav.scp")
  [ -z "${overlap}" ] || fail "FLEURS train/eval wav overlap with ${d}: ${overlap}"
done

if [ -n "${train_set}" ] && [ -f "${data_dir}/${train_set}/wav.scp" ]; then
  for name in "${selected_names[@]}"; do
    case "${name}" in
      test_*) ;;
      *) continue ;;
    esac
    [ -f "${data_dir}/${name}/wav.scp" ] || continue
    overlap=$(awk 'NR==FNR {w[$2]=1; next} $2 in w {print; exit}' \
      "${data_dir}/${name}/wav.scp" "${data_dir}/${train_set}/wav.scp")
    [ -z "${overlap}" ] || fail "${train_set}/${name} wav overlap: ${overlap}"
  done
fi

echo "verify_lid_data: OK"
