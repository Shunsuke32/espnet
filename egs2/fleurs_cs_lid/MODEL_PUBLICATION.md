# Model-Only Publication

`local/publish_model.py` prepares **one explicitly selected model**, offline by
default. No model, epoch, repository, or visibility is chosen automatically.
The confirmed account is `shun3232`. The user approved PUBLIC uploads for this
publication session; this is not a global default or approval for future runs.
Each upload still requires an explicitly confirmed checkpoint and destination,
`--public`, `--push`, and matching `--confirm_repo_id`.

Confirmed naming template: `shun3232/mms1b-lid-<family>-<scope>`.

| Family | `<family>` |
| --- | --- |
| ASR Transformer PIT | `transformer-pit` |
| ECAPA hard targets / atomic pairs where used | `softmax-hard` |
| ECAPA soft KL | `softmax-soft` |
| ECAPA AAM BCE, positive weight 50 | `sigmoid` |

The three `<scope>` suffixes are `fleurs` (F-only), `fleurs-csfleurs` (F+CS),
and `fleurs-csfleurs-csyodas` (CS-all). The FLEURS-only `softmax-hard` model is
the shared hard/soft baseline: do not create a separate `softmax-soft-fleurs`
repository or claim an independently trained F-only KL model. The confirmed
publication scope is therefore 11 models. Explicit checkpoint selection and
destination confirmation are still required.

Run from this recipe directory with the ESPnet Python environment activated:

```bash
python local/publish_model.py prepare \
  --task lid --config_file /path/to/experiment/config.yaml \
  --model_file /path/to/experiment/15epoch.pth \
  --lang2utt /path/to/experiment/lang2utt \
  --model_id chosen-model-epoch15 --output_dir /tmp/publication-preview

python local/publish_model.py verify --output_dir /tmp/publication-preview
```

For ASR, use `--task asr` and omit `--lang2utt`. The saved config must contain
the actual token list, either inline or in a readable file. Configured relative
assets require `--recipe_dir /original/recipe/asr1` (or `lid1` for LID); they
never resolve against the experiment directory or an implicitly guessed recipe.
If historical non-linguistic symbols point outside
the experiment, pass `--non_linguistic_symbols_file /actual/frozen/nlsyms.txt`
(`--non_linguistic_symbols` is an alias);
the exact symbols are checked against the saved token list and inlined, without
bracket/raw conversion or guessing from model names. Use a checkpoint-matching
source; today's recipe file is not proof of historical identity. Inline symbol
lists and `bpemodel: null` are also accepted. This helper supports only scoped
LID-sequence ASR: tokens must be canonical `<xxx>` three-letter language codes
or `<blank>`, `<unk>`, `<sos/eos>`; symbols must be canonical language codes
present in that token list, never specials. The checks apply equally to inline
and external lists. BPE and external normalization assets are rejected.
Sensitive nested config keys (token, password, secret, credentials, API/private
keys, authorization) are rejected rather than published.

The output directory must be new. It contains only small `config.yaml`,
`provenance.json`, `README.md`, LID `lang2utt`, optional `utt2langs`, and local-only
`upload-plan.json`.
**No weights are copied or zipped.** Keep the original numbered weight file
available and immutable until upload completes. Symlinks are resolved once at
selection; the resolved filename must be numbered, such as `15epoch.pth`.
There is no latest/best fallback or assertion that this epoch is best.
For an explicitly selected existing average such as `valid.accuracy.ave.pth`,
add `--checkpoint_kind averaged --checkpoint_description 'Existing evaluated average; component epochs unverified'`.
Provenance records `model_type: averaged` and `epoch: null`; the helper never
reconstructs averages, substitutes numbered epochs, or infers component epochs.
Whole `checkpoint.pth` trainer snapshots, including renamed snapshots, are
rejected. Select an existing numbered model-only state dictionary instead.

Preparation and `verify` use CPU `torch.load(weights_only=True, mmap=True)`
with no unsafe-pickle fallback, model construction, GPU use, or downloads.
Hashes stream through the original checkpoint; this takes disk I/O but does
not allocate a second full weight copy. Verification checks file identity,
hashes, and state-dictionary structure, **not model/config semantic equivalence
or inference quality**. Old non-mmap torch serialization is rejected.

## Portable Config And Labels

The historical config is read-only. The public config keeps architecture,
loss/head settings, freeze settings, and ASR tokens; it strips training inputs,
optimizer/runtime settings, initialization sources, and cache paths. It uses
canonical `pit_loss` / `pit_loss_reduction`, rejecting conflicting legacy
aliases. Use matching ESPnet CS-LID code supporting those keys; this is not a
promise that an arbitrary upstream revision supports recipe-specific models.
MMS-1B inference can still need its separately cached backbone; publication
does not fetch it. Unsupported external architecture dependencies are fatal.

LID requires the actual checkpoint-matching ordered training `lang2utt`, not
test labels. Its order and `lang_num` are checked against the original configured
inventory and/or the frozen experiment `lang2utt` snapshot. At least one must
be available; an operator-supplied file of the same length is not sufficient.
For a recipe-relative configured inventory, supply `--recipe_dir` pointing to
the original recipe. All available reference inventories must agree in order.
A frozen experiment snapshot suffices if the original source is unavailable.
Only labels and synthetic
`__inventory_only__` placeholders are published, never training utterance IDs.
The public config uses relative `lang2utt`; for upstream preprocessing/inference
resolve `args.lang2utt` to `<downloaded-bundle>/lang2utt` (or run in the bundle).
Native `LIDTask.build_model_from_file` itself does not consume this path.

**Placeholder rows are not training counts or seen/unseen references.** Optional
`--train_language_sets /path/to/frozen-language-sets.txt` records actual frozen
training language sets in provenance. This file must contain only one or two
individual language codes per line, e.g. `ara eng`, with no IDs or header. It
must be derived from the selected experiment's training references, not inferred
from its head inventory. With this option, the helper also emits `utt2langs`
containing one synthetic ID per distinct supplied language set. The current
`local/evaluate.sh --train_data_dir <bundle>` can use that file plus ordered
`lang2utt` for seen/unseen membership. These rows are **not real utterances or
counts**, and are not evaluation/test references. Without this option, supply
actual frozen training references separately; `lang2utt` placeholders alone
are insufficient.

## Confirmed Upload Only

After the user confirms the exact checkpoint, destination, and visibility,
rerun `prepare` with a **new** output directory and these additional arguments:

```text
--repo_id shun3232/USER_CONFIRMED_REPOSITORY
--push --confirm_repo_id shun3232/USER_CONFIRMED_REPOSITORY
```

Private is the default; add `--public` only for an explicitly approved public
repository. Existing repository visibility must match and is never changed.
Use a repository dedicated to this single model: root config/card/provenance
files are replaced by the commit; unrelated existing files are not removed.
The HF endpoint is fixed to `https://huggingface.co`. Authentication uses the
standard SDK environment/token store; no token goes
into the manifest. The SDK makes one direct commit with `CommitOperationAdd`
referencing the **existing** numbered weights and small public files. The local
upload plan (which contains the source path) is never uploaded.

The model card records CC-BY-NC-4.0 and the MMS-1B base model where applicable,
without invented metrics. Its architecture YAML uses only validated portable
config fields: frontend/backbone and multilayer settings, encoder, task-specific
decoder or pooling/projector/loss settings, model/PIT settings, and freeze settings.
An explicit `--checkpoint_description` is included as operator-provided public
text; provide a human-readable description, not source paths or confidential data.
Original training config and optimizer settings are never copied into the card.
Provenance includes original config/checkpoint
SHA-256, explicit epoch selection, and LID inventory hash/order.

Native ESPnet stages already exist: ASR 14 (pack) / 15 (upload), LID 9 / 10.
Their recipe wrappers are restricted, and their templates build ZIP archives
and assume extra assets such as image directories. This helper is a separate,
model-only route without those packaging assumptions, not a wrapper change.
