# pyvae

**Informed Variational Autoencoder (iVAE) for gene expression, in PyTorch.**

`pyvae` implements a VAE whose encoder is *biologically constrained*: each gene
can only influence the pathways that contain it, according to the
[Reactome](https://reactome.org/) database. The result is a latent space whose
dimensions correspond to interpretable biological pathways rather than opaque
features.

[![CI](https://github.com/aipher-group/pyvae/actions/workflows/ci.yml/badge.svg)](https://github.com/aipher-group/pyvae/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/pyvae.svg)](https://anaconda.org/conda-forge/pyvae)
[![PyPI Version](https://img.shields.io/pypi/v/pyvae)](https://pypi.org/project/pyvae/)


---

## Why an *informed* VAE?

A standard Variational Autoencoder learns a compressed latent representation
$z$ of the input $x$ by maximising the Evidence Lower Bound (ELBO):

$$
\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x)\,\|\,p(z))
$$

Gene expression is high-dimensional (~20 000 genes) but structured: genes act
together in *pathways*. `pyvae` restricts the encoder's first layer so only the
gene → pathway connections that exist in Reactome are allowed. A binary mask
enforces this on the weight matrix: $W_{\text{eff}} = W \odot M$, where
$M_{ji}=1$ iff gene $i$ belongs to pathway $j$. This reduces parameters, injects
prior biological knowledge, and makes each latent node interpretable as a
pathway.

## Features

### Models

`InformedLinear` is a masked linear layer that enforces arbitrary connectivity
priors. `InformedVAE` builds a full encoder/decoder around it, using the
reparameterisation trick and a combined reconstruction, KL, and L2 loss. You
configure it through constructor kwargs. `likelihood` picks between
`"gaussian"` (MSE on log1p-normalized expression, the default) and `"nb"` (a
negative-binomial likelihood on raw counts, implemented scVI-style in pure
PyTorch with no `scvi-tools` dependency). `n_cov` sets the width of an
auxiliary one-hot covariate (cell type, condition, or both), which the model
concatenates into the encoder and decoder to support conditional inference.

### Training

`train_ivae` is the legacy training loop: gradient clipping plus early
stopping. `train_ivae_modern` trains the count model instead, adding KL
warmup, AdamW with decoupled weight decay, cosine LR annealing, and
best-weight restore.

### Interpretation

`InformedVAE.predict_counterfactual` (NB models only) encodes a cell under
one covariate and decodes it under another, returning predicted counts. It
uses the posterior mean rather than a stochastic sample, so the prediction
stays deterministic, which is what you want for a question like "what would
this control cell look like if stimulated?" `bayes_factor_da` computes a
signed Bayes factor per pathway between two groups of cells, based on
Monte-Carlo pairing of raw pathway activations. `integrated_gradients`
attributes a single pathway's activation back to each input gene
(Sundararajan et al., 2017), implemented in pure PyTorch.

### Data helpers

Reactome helpers (`build_model_config`, `sync_gexp_adj`) and a Kang PBMC
dataset loader (`load_kang`) cover end-to-end pipelines. `swap_condition`
flips the one-hot condition columns in a covariate DataFrame, which is how
you build the `cov_to` argument for `predict_counterfactual`. `set_all_seeds`
seeds Python's `random`, NumPy, and PyTorch (CPU and CUDA) in a single call,
for reproducible runs.

## Installation

```bash
pip install pyvae                      # from PyPI
conda install -c conda-forge pyvae     # from conda-forge
pixi add pyvae                         # into a pixi project
```

`pyvae` is a pure-Python, `noarch` package and runs on Linux, macOS, and Windows.

### GPU acceleration

`pyvae` only requires `torch>=2.3,<3` and never pins a platform. Which
hardware you can accelerate depends entirely on which PyTorch build your
install command pulls. Use this table to pick the right path:

| Your machine | `pip install pyvae` | `conda install -c conda-forge pyvae` | Accelerated out of the box? |
|---|---|---|---|
| **NVIDIA CUDA (Linux/Windows)** | default `torch` wheel is the CUDA build | conda-forge ships CUDA `pytorch` variants | ✅ yes, on both |
| **Apple Silicon (macOS arm64)** | standard wheel includes **MPS** (Metal) | conda-forge osx-arm64 build includes MPS | ✅ yes (use `device="mps"`) |
| **AMD ROCm (Linux)** | gets the CUDA wheel, which won't drive an AMD GPU | conda-forge has **no ROCm** builds → CPU only | ❌ no (see ROCm steps below) |
| **Intel macOS / generic CPU** | CPU wheel | CPU build | n/a (CPU only) |

**CUDA and Apple-Silicon users need no extra steps.** Verify at runtime with
`python -c "import torch; print(torch.cuda.is_available())"` (CUDA) or
`torch.backends.mps.is_available()` (macOS).

**AMD ROCm users must install torch from PyTorch's ROCm index.** ROCm wheels
exist neither on PyPI nor on conda-forge, so `conda install` is a dead end
for AMD GPUs. Install ROCm torch *first*, then pyvae; pip then sees the
constraint already satisfied and leaves torch alone:

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm7.2   # match your ROCm runtime
pip install pyvae
```

To pin an explicit CUDA stack via conda instead of relying on auto-detection:

```bash
conda create -n pyvae-cuda12 -c conda-forge -c pytorch -c nvidia \
  pyvae pytorch pytorch-cuda=12.*
```

Match the CUDA/ROCm version to your driver/runtime; the official
[PyTorch selector](https://pytorch.org/get-started/locally/) gives exact wheels.

## Quickstart

```python
import pandas as pd
import torch
import pyvae

# 1. Load and preprocess data (Kang PBMC)
adata = pyvae.load_kang(data_folder="./data", n_genes=3000)

# 2. Train/validation split
train_mask = adata.obs["split"] == "train"
x_train = pd.DataFrame(adata[train_mask].X.toarray(), columns=adata.var_names)
x_val   = pd.DataFrame(adata[~train_mask].X.toarray(), columns=adata.var_names)

# 3. Build the biological (Reactome) configuration and adjacency
config = pyvae.build_model_config(
    genes=x_train.columns,
    model_kind="ivae_reactome_hierarchical_d2",
    resources_dir="./resources",
)
adj = torch.tensor(config.model_layer[0].values, dtype=torch.float32)

# 4. Build and train the model
model = pyvae.InformedVAE(adj=adj, latent_dim=64, l2_lambda=1e-5)
model, history = pyvae.train_ivae(
    model, x_train, x_val,
    epochs=500, batch_size=32, patience=50, lr=1e-4, device="cpu",
)

# 5. Encode new cells — `h` is the interpretable pathway activation
z_mean, z_log_var, h = model.encode(torch.tensor(x_val.values, dtype=torch.float32))
```

## Development

This repo is managed with [pixi](https://pixi.sh). The manifest defines several
environments (`default`, `ci`, `cuda12`, `rocm72`, `publish`):

```bash
pixi install              # create the default environment
pixi run pytest tests/ -v # run the test suite
pixi run fixcode          # ruff lint + import sort
pixi run formatcode       # ruff format
```

Without pixi, a standard editable install works too:

```bash
pip install -e ".[dev]" && pytest tests/ -v
```

## Releasing

This repo publishes to both PyPI and conda-forge. Before releasing, bump the
version in `pyproject.toml` (both `[project].version` and
`[tool.pixi.package].version`).

**PyPI** (Trusted Publishing in CI on a `v*` tag, or locally via the isolated
`publish` env):

```bash
pixi run -e publish build-dist     # build sdist + wheel into dist/
pixi run -e publish check-dist     # twine metadata check
pixi run -e publish publish-pypi   # upload (set TWINE_REPOSITORY=testpypi for TestPyPI)
```

**conda** (built as a single `noarch` package by pixi):

```bash
pixi publish --target-dir ./conda-dist          # build + copy locally (no upload)
pixi publish --to https://prefix.dev/<channel>  # or push to a channel directly
```

The [`pyvae-feedstock`](https://github.com/conda-forge/pyvae-feedstock) handles
conda-forge distribution instead of a direct upload, and the package is
already live there. `regro-cf-autotick-bot` picks up new PyPI releases
automatically and opens a version-bump PR on the feedstock, so no action is
needed here. The original submission recipe and instructions for manual
recipe changes live in [`conda-recipe/`](conda-recipe/). On a tagged release,
`.github/workflows/release.yml` builds and verifies both artifacts and
attaches the `noarch` conda package to the GitHub Release.

## Project layout

```
pyvae/
├── __init__.py    public API
├── utils.py       seeding utilities
├── layers.py      InformedLinear (masked layer)
├── components.py  Encoder, DenseDecoder, CountDecoder, likelihoods, helpers
├── models.py      InformedVAE (composition of components)
├── train.py       training loops: train_ivae, train_ivae_modern
├── datasets.py    Kang dataset loader
├── bio.py         Reactome adjacency helpers, swap_condition
└── interpret.py   bayes_factor_da, integrated_gradients
tests/             contract + unit tests
conda-recipe/      conda-forge recipe + submission guide
```

## Acknowledgements

`pyvae` is a PyTorch reimplementation that builds on earlier Keras
informed-VAE work directed by Carlos Loucera, with contributions from Pelin
Gundogdu ([babelomics/ivae_scorer](https://github.com/babelomics/ivae_scorer)),
Alberto Esteban-Medina
([albertoem77/robustness_informed_TFM](https://github.com/albertoem77/robustness_informed_TFM)),
and Sara Fernandez
([saraafdezz/robustness_informed_TFG](https://github.com/saraafdezz/robustness_informed_TFG)).

## License

[MIT](LICENSE) © Amir Aynede, Carlos Loucera
