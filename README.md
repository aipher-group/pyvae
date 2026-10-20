# pyvae

**Informed Variational Autoencoder (iVAE) for gene expression, in PyTorch.**

`pyvae` implements a VAE that carries a gene-set annotation on both sides: each
encoder unit reads only the genes of one [Reactome](https://reactome.org/)
pathway, and each decoder unit writes only to them. Conditioning enters in
pathway coordinates, so a covariate contrast is one signed number per pathway
rather than a quantity the annotation never touches.

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

Gene expression is high-dimensional (about 20,000 genes) but structured: genes
act together in *pathways*. `pyvae` masks the first layer so only the gene to
pathway edges in Reactome carry weight, $W_{\text{eff}} = W \odot M$ with
$M_{ji}=1$ iff gene $i$ belongs to pathway $j$. On the Kang panel that leaves
0.55% of a dense layer's weights, and it keeps a per-gene attribution specific
to one pathway.

Masking the encoder does not on its own make a latent dimension a pathway, since
`z` is a dense mixture of those units. What a covariate contrast reads is the
decoder. Measured against a shuffled annotation of identical set sizes and
overlap, the encoder mask alone is worth +3.24 nats of held-out loss and masking
the decoder as well raises that to +14.52. Both masks are flags, and
[`experiments`](experiments/) reports each.


## Validation

Two Snakemake pipelines under [`experiments/`](experiments/) produce the thesis and future paper results draft, [`experiments`](experiments).

[`experiments/kang/`](experiments/kang/) is the known-answer check. Kang 2018
PBMCs under interferon-beta fix the right pathway before anything runs. Three
grids totalling 40 fits then decide the architecture on evidence: where the
covariate enters, whether the options held fixed matter, and whether
the measure is confounded by how far units sit in the flat tail of `tanh`. The
read-out puts `REACTOME_INTERFERON_ALPHA_BETA_SIGNALING` at rank 25 of
1,615 units with its Reactome parent at 3.

[`experiments/multiple_mieloma/`](experiments/multiple_mieloma/) applies the
selected architecture to bone-marrow T cells across the myeloma precursor
sequence in two independent cohorts.

Each pipeline ends in a verifier that re-derives every number its `FINDINGS.md`
quotes and asserts its own check count, 87 for Kang and 88 for myeloma, so a
claim cannot leave the prose and the checks at the same time.

## Features

### Models

`InformedLinear` is a masked linear layer that enforces arbitrary connectivity
priors. `InformedVAE` builds a full encoder/decoder around it, using the
reparameterisation trick and a combined reconstruction, KL, and L2 loss. You
configure it through constructor kwargs. `likelihood` picks between
`"gaussian"` (MSE on log1p-normalized expression, the default) and `"nb"` (a
negative-binomial likelihood on raw counts, implemented scVI-style in pure
PyTorch with no `scvi-tools` dependency). `n_cov` sets the width of an
auxiliary one-hot covariate (cell type, condition, or both). Where that
covariate enters is a separate choice, and it decides what a conditional
contrast can mean.

The encoder takes `init="fan_in"` for per-unit scaling, `normalize="batch"` for
BatchNorm on the pre-activation, `nonneg_encoder=True` for a softplus
reparameterisation that removes sign freedom, `standardize_input=True` for
BatchNorm on the input, and `informed_encoder=False` to drop its mask
altogether. The decoder takes `informed_decoder=True` for the same mask applied
symmetrically, and `nonneg_decoder=True` so that a unit rising raises its member
genes, which is what gives a pathway's sign a meaning.

Two further flags decide where a covariate enters, and they decide whether the
annotation reaches a conditional contrast at all. `cov_mode="pathway"` routes it
through an `(n_pathways, n_cov)` matrix inside the decoder's nonlinearity
instead of a dense per-gene term added after the mask, and
`cov_in_encoder=False` closes the encoder's covariate route so that term is the
whole conditional effect. Under both, `InformedVAE.pathway_contrast(cov_a,
cov_b)` returns one signed number per pathway, and it raises wherever that
number would be incomplete, naming `unit_contrast` as the read-out that measures
both routes.

### Training

`train_ivae` is the legacy training loop: gradient clipping plus early
stopping. `train_ivae_modern` trains the count model instead, adding KL
warmup, AdamW with decoupled weight decay, cosine LR annealing, and
best-weight restore.

 `train_ivae_modern` also accepts `cov_train` and `cov_val` to thread one-hot
covariates through the model, which the conditional read-out requires and which
is advisable whenever cell composition varies between conditions.
Datasets are moved to the training device once at the start of training, so
batches cost a device-side gather rather than a host-to-device copy per
epoch; a full-panel Kang run takes about 4 minutes on a single A100.

### Interpretation

`pathway_contrast` is the conditional read-out, and `unit_contrast` decomposes
the same quantity per unit: the shift each unit takes, what reaches its member
genes, and how much of the gene-level change passes through the mask rather
than around it. `unit_sign_structure` reports whether a unit has a direction at
all, which under a freely signed decoder it does not. `permute_adjacency`
relabels gene rows while preserving every set size, every gene's degree and the
whole pathway-by-pathway overlap matrix, which is the control that separates
the annotation from its shape. `integrated_gradients` attributes one unit's
activation back to each input gene (Sundararajan et al., 2017); under the
encoder mask a gene outside the set gets exactly zero.

`predict_counterfactual` (NB models only) encodes a cell under one covariate and
decodes it under another, using the posterior mean so the prediction is
deterministic. Read it with care under `cov_mode="gene"`: the masked term then
cancels between the two sides and what is left is one column difference of an
unannotated matrix, which is the defect `experiments/` is about.

The gene-level read-outs remain, as baselines and for model-free comparison.
`differential_expression` samples posterior per-gene differential expression
between two groups (Boyeau et al. 2019), with observed-detection filtering so
decoder-confident but barely-observed genes do not dominate the ranking.
`pathway_activity` runs a tie-corrected competitive Mann-Whitney test of a
pathway's member genes against the rest of the panel on that output, and reports
`top_gene_share` so a hit driven by a single gene is visible.
`pseudobulk_paired_test` collapses cells to donor-level pseudobulk and runs a
paired Wilcoxon per gene, the Squair et al. 2021 correction for cell-level
pseudo-replication.

Two read-outs are retired from our own pipelines, and a third is kept only as a
control. `pathway_unit_fidelity` measures distance from initialisation rather
than fidelity: an untrained non-negative model scores about 0.75 and the score
falls with training, so it detects inverted units and is not a quality gate
([`experiments/kang/README.md`](experiments/kang/README.md) gives the
arithmetic). `bayes_factor_da` compares observed groups of cells, so it is
composition-confounded wherever those groups differ in cell-type mixture, and a
covariate contrast answers the same question conditionally. `pathway_activity`
ranks the right pathway first in all eight architecture arms of the Kang
benchmark, including every arm whose gene-to-pathway map is shuffled, so there
it is evidence about the read-out rather than a result from the model.

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

| Your machine | `pip install pyvae` | `conda install -c conda-forge pyvae` | Accelerated without further configuration? |
|---|---|---|---|
| **NVIDIA CUDA (Linux/Windows)** | default `torch` wheel is the CUDA build | conda-forge ships CUDA `pytorch` variants | ✅ yes, on both |
| **Apple Silicon (macOS arm64)** | standard wheel includes **MPS** (Metal) | conda-forge osx-arm64 build includes MPS | ✅ yes (use `device="mps"`) |
| **AMD ROCm (Linux)** | gets the CUDA wheel, which won't drive an AMD GPU | conda-forge has **no ROCm** builds → CPU only | ❌ no (see ROCm steps below) |
| **Intel macOS / generic CPU** | CPU wheel | CPU build | n/a (CPU only) |

**CUDA and Apple-Silicon users require no further configuration.** Verify at runtime with
`python -c "import torch; print(torch.cuda.is_available())"` (CUDA) or
`torch.backends.mps.is_available()` (macOS).

**AMD ROCm users must install torch from PyTorch's ROCm index.** ROCm wheels
exist neither on PyPI nor on conda-forge, so `conda install` cannot serve AMD GPUs. Install ROCm torch *first*, then pyvae; pip then sees the
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

# 5. Encode new cells. `h` holds one activation per pathway unit
z_mean, z_log_var, h = model.encode(torch.tensor(x_val.values, dtype=torch.float32))
```

For a conditional contrast, build the model with the architecture the
experiments selected and read the effect straight off the decoder:

```python
model = pyvae.InformedVAE(
    adj=adj, latent_dim=adj.shape[1] // 2, likelihood="nb", n_cov=cov.shape[1],
    normalize="batch", standardize_input=True, nonneg_encoder=True,
    informed_decoder=True, nonneg_decoder=True,     # the mask on both sides
    cov_mode="pathway", cov_in_encoder=False,       # the covariate inside it
)
# ... train with pyvae.train_ivae_modern(..., cov_train=..., cov_val=...)
contrast = model.pathway_contrast(cov_control, cov_stimulated)  # (n_pathways,)
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
version with `pixi run bump-version patch` (or `minor`, `major`, or an explicit
`X.Y.Z`). It sets the three declarations in `pyproject.toml` together, and
`tests/unit/test_version_consistency.py` fails if they ever disagree.

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
├── bio.py         Reactome adjacency helpers, swap_condition,
│                   permute_adjacency
├── interpret.py   unit_contrast, unit_sign_structure, integrated_gradients,
│                   differential_expression, pathway_activity,
│                   pseudobulk_paired_test, and two retired read-outs
tests/             contract + unit tests
conda-recipe/      conda-forge recipe + submission guide
experiments/
├── kang/          known-answer validation (Kang 2018 IFN-beta PBMCs)
└── multiple_mieloma/   application to the myeloma precursor sequence
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
