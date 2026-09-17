# Reference software versions

The checked CPU validation environment used:

| component | version |
|---|---:|
| Python | 3.10.14 |
| PyTorch | 2.9.1+cu126 |
| Torchaudio | 2.9.1+cu126 |
| datasets | 4.8.5 |
| soundfile | 0.13.1 |
| s3prl | 0.4.18 |
| pycountry | 26.2.16 |
| SciPy | 1.15.3 |
| NumPy | 2.2.6 |
| Transformers (current review environment) | 5.8.0 |
| huggingface-hub (current review environment) | 1.14.0 |
| Matplotlib | 3.10.9 |

S3PRL's installed distribution reports version 0.4.18 but is the ESPnet fork
at commit `572af70a3e47c23292b6a3e93fa13dcf1e281111`, not necessarily the
same code as PyPI's release. This is recorded in its `direct_url.json` and
matches ESPnet's optional dependency. In a new environment install that commit:

```bash
python3 -m pip install 's3prl @ git+https://github.com/espnet/s3prl.git@572af70a3e47c23292b6a3e93fa13dcf1e281111'
python3 -m pip install datasets==4.8.5 soundfile==0.13.1 pycountry==26.2.16 scipy==1.15.3
```

The current review environment is not an archived lockfile of every June-August
training run. In particular, historical Transformers versions were not recovered.
Do not infer full MMS numerical equivalence from the offline frontend tests.

The data revisions are pinned. Historical run selection, initialization
checkpoints, iteration rounding, and saved evaluation exclusions still require
the experiment review; generic base YAMLs alone do not fix those choices.
Exact bitwise equality across CUDA/cuDNN, GPU architecture, and DDP world-size
combinations is not promised.

## MMS Weights

The local reference machine's `facebook/mms-1b` cache points to revision
`0d2f7adb9903d98894d70ae11f7fbdfc8cb71a69`. Its `pytorch_model.bin` symlink
names LFS blob ID `ddf6980ef183118e5873cfb4c4789a90386b87a6fe3fafa8a08a822f557d68f7`.
This records the inspected cache, not proof of every historical run's input.

The installed S3PRL `hf_wav2vec2_custom` adapter does not forward a revision
argument to Transformers. A bare `path_or_url: facebook/mms-1b` can therefore
follow a future Hub update. For a fixed input, first resolve the already cached
snapshot offline and use the returned directory as
`frontend_conf.frontend_conf.path_or_url`:

```bash
HF_HUB_OFFLINE=1 python3 -c 'from huggingface_hub import snapshot_download; print(snapshot_download("facebook/mms-1b", revision="0d2f7adb9903d98894d70ae11f7fbdfc8cb71a69", local_files_only=True))'
```

Record that path and the checkpoint hash with the selected experiment. No
MMS download or full-size MMS training is performed by the offline CPU tests.
If the cache is absent on a different server, downloading that pinned snapshot
is a separate, explicit setup step requiring sufficient disk space. It is not
part of the current review and is never a fallback for missing local inputs.
