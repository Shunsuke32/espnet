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

The data revisions and all training configuration values are pinned in the
recipe.  Exact bitwise equality across CUDA/cuDNN, GPU architecture, and DDP
world-size combinations is not promised; the paper protocol, sample inventory,
optimization budget, targets, metrics, and random seeds are reproducible.
