# External evaluation data

The ESPnet2 VoxLingua107 README reports evaluation on Babel, FLEURS, ML-SUPERB
2.0 (`dev` and `dev_dialect`), and VoxPopuli. Its default `run.sh` only selects
VoxLingua107 development data. The exact LID-specific raw preparation scripts
for those external corpora are not present in this checkout. Consequently,
this recipe reuses existing ESPnet2-prepared evaluation data; it does not claim
to reconstruct the published external evaluation splits from scratch.

No additional corpus is downloaded automatically. In particular, Babel must
already have been obtained under its applicable license.

## Existing preparation code

The paths below are relative to the ESPnet checkout. Configure the original
recipe's `db.sh`, dependencies and dataset locations before explicitly running
its preparation script. These scripts can download or rewrite that recipe's
data, so do not run them over an existing experiment without checking it first.

| Corpus | Existing ESPnet2 source | Label input for this recipe |
| --- | --- | --- |
| FLEURS | `egs2/fleurs/asr1/local/data.sh --lang all`, `local/create_dataset.py` | Initial `[language]` tags in `data/test_all/text`; provide an explicit source-code-to-ISO3 mapping, e.g. `en_us eng`, `ja_jp jpn`. |
| ML-SUPERB 2.0 | `egs2/ml_superb2/asr1/local/data.sh`, `local/download.py` | Initial `[language]` tags in `data/dev/text` and `data/dev_dialect/text`. Reuse the labels already normalized by this script. |
| Babel | `egs2/babel/asr1/local/data.sh`, `local/setup_languages.sh` | Supply existing `utt2lang`, or an explicit ISO3 `--language` for each monolingual split. Do not infer language from transcript words. |
| VoxPopuli | `egs2/owsm_v3/s2t1/local/prepare_voxpopuli.py` and `.sh` contain ASR/ST preparation | Use the original LID evaluation `wav.scp`/`utt2lang` if available. The OWSM recipe merges segments and includes translation examples, so its output is **not** a substitute for the published LID test split. |

The existing ML-SUPERB script includes source-specific label normalization
(`org_jpn`, `lga`, `ory`, `arb`, and some `ms_speech` dialect examples) and
excludes Norwegian labels from its non-dialect splits. This adapter does not
repeat or override those choices. To reproduce another preparation version,
use its original `utt2lang`; do not silently apply the current ASR rules to it.

## Materialize audio with the existing ESPnet2 formatter

If the data already has utterance-level, 16 kHz `wav.scp` paths, no audio
conversion is required. For recording-level `segments`, pipe commands or
8 kHz Babel audio, reuse the existing formatter, from the original ESPnet2
recipe directory so that its relative paths resolve correctly:

```bash
python pyscripts/audio/format_wav_scp.py \
    --fs 16000 --audio-format wav \
    --segments data/EVAL/segments \
    data/EVAL/wav.scp \
    /mlnas/mitsumori/dataset/lid_evaluation/formatted/EVAL
```

Replace `EVAL` with the selected split. Omit `--segments ...` when the source
does not have that file. Pipe commands are executable input; only use metadata
from a trusted preparation script. The original labels remain in the original
`data/EVAL/utt2lang` or `text`; the formatter only prepares audio.

## Prepare metadata without copying audio

From `egs3/voxlingua107/lid`, import an existing LID split:

```bash
python -m src.external_data \
    --source-dir /path/to/espnet2/dump/raw/EVAL \
    --output-dir /mlnas/mitsumori/dataset/lid_evaluation/EVAL \
    --audio-root /path/to/original/espnet2/recipe
```

`--audio-root` is only relevant when `wav.scp` contains relative paths. Output
paths are absolute. The helper writes only `wav.scp`, `utt2lang`, and `lang2utt`,
and refuses to overwrite an existing output directory.

For ASR text, select `--label-file text`. This reads only a leading
`[language]` tag and rejects untagged transcripts. For example, after preparing
the ML-SUPERB dev audio with the command above:

```bash
python -m src.external_data \
    --source-dir /mlnas/mitsumori/dataset/lid_evaluation/formatted/ml_superb2 \
    --label-file /path/to/espnet2/egs2/ml_superb2/asr1/data/dev/text \
    --output-dir /mlnas/mitsumori/dataset/lid_evaluation/ml_superb2
```

Use `data/dev_dialect/text` and a separate output directory named
`ml_superb2_dialect` for the dialect set. For FLEURS, add
`--language-map /path/to/source_to_iso3.txt`: each line contains the source tag
and the model's ISO-639-3 code. For a monolingual Babel split, use e.g.
`--language asm` instead of reading a label file. Keep multiple monolingual
splits as separate named `dataset.test` entries if no combined LID metadata is
available.

## Inference and language matching

`conf/inference_external.yaml` lists the five external evaluation splits.
Prepare the selected directories, or copy the config and remove entries for
corpora that are not available, then use the normal stages:

```bash
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference_external.yaml \
    --metrics_config conf/metrics.yaml
```

The external Dataset uses ESPnet2's `SoundScpReader` and `read_2columns_text`.
Like ESPnet2's `prepare_ood_test.py`, it evaluates only languages in the actual
model's training `lang2utt`. The exact intersection is tested against that
ESPnet2 function. Labels must match exactly: `yue` is not converted to `cmn`,
and an unseen dialect is excluded unless an explicit preparation mapping was
provided. This is closed-set evaluation, not unknown-language detection.
The stage log reports the retained/total utterance count, supported languages,
and excluded language labels so the evaluation denominator remains visible.

Utterances are sorted by their original IDs, then accessed by integer indices
as in the ASR recipe. Inference output uses those indices; the adapter's
`utterance_ids` list preserves the original-ID correspondence. Samples contain
only `speech` and `lid_labels`, with no extra model input fields. The external
data module is inference-only: preparation is the explicit helper above, not
a new shared stage or an automatic corpus download in `create_dataset`.

Fixture tests cover metadata conversion, source preservation, language
intersection parity, dialect exclusion, stereo handling, and sample-rate
validation. Running those tests does not verify the actual external corpora or
reproduce the published accuracy numbers.
