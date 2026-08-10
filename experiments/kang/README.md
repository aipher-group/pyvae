# Kang PBMC — end-to-end pyvae sanity check

An end-to-end run of pyvae on the Kang 2018 PBMC dataset (control vs
interferon-β stimulated), used as the canonical ground-truth benchmark
for informed VAEs on scRNA-seq data.

The goal is to verify that pyvae's modern training loop (`train_ivae_modern`)
and interpretation stack (`bayes_factor_da`, `integrated_gradients`,
`predict_counterfactual`) recover the expected interferon-response biology on
a dataset with a clearly established ground truth.

Similar in spirit to Gundogdu et al. (CMSB 2023, "Cell-Level Pathway Scoring
Comparison with a Biologically Constrained Variational Autoencoder"), which
established this benchmark for informed VAEs at CABD. The comparison is not
head-to-head — pyvae is a PyTorch package built on top of the same modeling
philosophy, with a negative-binomial likelihood, covariate conditioning, and
Bayes-factor / Integrated Gradients interpretation added on top.

## Notebook

- `01_kang_end_to_end.ipynb` — one notebook, ~7 sections, from data loading
  through counterfactual prediction.

## Reproduction

The environment is managed by the parent repo's pixi setup. From the repo
root:

    pixi run jupyter lab

then open `experiments/kang/01_kang_end_to_end.ipynb`.

The Kang dataset is downloaded automatically by `pyvae.load_kang` into
`experiments/kang/data/`, which is git-ignored.
