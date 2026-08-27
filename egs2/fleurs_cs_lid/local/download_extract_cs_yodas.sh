#!/usr/bin/env bash
set -euo pipefail

# Download CS-YODAS metadata/audio archives and materialize the wav files that
# are referenced by the selected base+English records.  This script writes only
# under --root.

root=downloads/cs-yodas
metadata_dir=
audio_root=
download_archives=true
extract_audio=true
extract_missing_only=true
revision=e51028041b403f63c99ac91a4af040e72d0cad0e

script_dir=$(cd "$(dirname "$0")" && pwd)
recipe_root=$(cd "${script_dir}/.." && pwd)
if [ -f utils/parse_options.sh ]; then
  . utils/parse_options.sh
elif [ -f "${recipe_root}/asr1/utils/parse_options.sh" ]; then
  . "${recipe_root}/asr1/utils/parse_options.sh"
else
  echo "Error: parse_options.sh not found" >&2
  exit 2
fi

metadata_dir=${metadata_dir:-${root}/metadata}
audio_root=${audio_root:-${root}/audio}
archive_dir=${root}/archives
selected_dir=${root}/selected_paths

mkdir -p "${metadata_dir}" "${archive_dir}" "${audio_root}" "${selected_dir}"

base_url=https://huggingface.co/datasets/byan/cs-yodas/resolve/${revision}
langs=(ara cmn fra hin jpn rus)

download_file() {
  local url=$1
  local out=$2
  if [ -s "${out}" ]; then
    echo "exists: ${out}" >&2
    return 0
  fi
  echo "download: ${url} -> ${out}" >&2
  rm -f "${out}.part"
  curl -fL -sS --retry 5 --retry-delay 10 --connect-timeout 30 "${url}" -o "${out}.part"
  mv "${out}.part" "${out}"
}

for lang in "${langs[@]}"; do
  if [ ! -s "${metadata_dir}/${lang}.jsonl" ]; then
    download_file "${base_url}/${lang}.jsonl" "${metadata_dir}/${lang}.jsonl"
  fi
done

declare -A metadata_sha256=(
  [ara]=19254013052100a5c780d2342c21185babee874cec6a7cb5f14c67bae1435abb
  [cmn]=6cdad8b713ea41a598077dd0fedae14824eb2d6998970132d5b2c885201c7697
  [fra]=76f415663519b4898ddd0540bbf7b5149aae25f181e9262a39e89aae843fbfda
  [hin]=2533311b72283f436ec9a0aca687b8f66983bd2c1fdd2574704a3091cba3e526
  [jpn]=c5fca5676a09f07391aee943996ce6619529bea273b17237f756f1e66c7a5844
  [rus]=f29381d741c611e54e5ad06d7645f4a9fa837508ea8784006d4f600d70be006a
)
for lang in "${langs[@]}"; do
  actual=$(sha256sum "${metadata_dir}/${lang}.jsonl" | awk '{print $1}')
  if [ "${actual}" != "${metadata_sha256[${lang}]}" ]; then
    echo "Error: metadata checksum mismatch for ${lang}.jsonl at revision ${revision}" >&2
    exit 1
  fi
done

if "${download_archives}"; then
  for name in ara.tar cmn.tar hin.tar jpn.tar rus.tar fra.tar.aa fra.tar.ab; do
    if [ ! -s "${archive_dir}/${name}" ]; then
      download_file "${base_url}/${name}" "${archive_dir}/${name}"
    fi
  done
fi

python3 - "${metadata_dir}" "${selected_dir}" <<'PY'
import json
import pathlib
import sys

metadata_dir = pathlib.Path(sys.argv[1])
selected_dir = pathlib.Path(sys.argv[2])
base = {
    "ara": "Arabic",
    "cmn": "Chinese",
    "fra": "French",
    "hin": "Hindi",
    "jpn": "Japanese",
    "rus": "Russian",
}
for lang, name in base.items():
    paths = []
    with (metadata_dir / f"{lang}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            languages = obj.get("languages") or []
            if len(languages) == 2 and set(languages) == {name, "English"}:
                paths.append(obj["wav_path"])
    out = selected_dir / f"{lang}.paths"
    out.write_text("\n".join(sorted(set(paths))) + "\n", encoding="utf-8")
    print(f"{lang}: {len(set(paths))} selected wav paths -> {out}", file=sys.stderr)
PY

if "${extract_audio}"; then
  for lang in ara cmn fra hin jpn rus; do
    if "${extract_missing_only}"; then
      while IFS= read -r wav_path; do
        [ -n "${wav_path}" ] || continue
        [ -f "${audio_root}/${wav_path}" ] || echo "${wav_path}"
      done < "${selected_dir}/${lang}.paths" > "${selected_dir}/${lang}.missing.paths"
    else
      cp "${selected_dir}/${lang}.paths" "${selected_dir}/${lang}.missing.paths"
    fi
  done
  for lang in ara cmn hin jpn rus; do
    if [ -s "${selected_dir}/${lang}.missing.paths" ]; then
      tar -xf "${archive_dir}/${lang}.tar" -C "${audio_root}" -T "${selected_dir}/${lang}.missing.paths"
    fi
  done
  if [ -s "${selected_dir}/fra.missing.paths" ]; then
    cat "${archive_dir}/fra.tar.aa" "${archive_dir}/fra.tar.ab" \
      | tar -xf - -C "${audio_root}" -T "${selected_dir}/fra.missing.paths"
  fi
fi

echo "CS-YODAS metadata: ${metadata_dir}" >&2
echo "CS-YODAS audio root: ${audio_root}" >&2
