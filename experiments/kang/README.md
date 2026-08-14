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

## Required resources (not committed)

Both the Kang dataset and the Reactome GMT are gitignored to keep the repo
lean. To reproduce:

- **Kang PBMC dataset:** `experiments/kang/data/kang_counts_25k.h5ad` (~38 MB).
  At the time of writing (Aug 2026), the figshare download endpoint
  (<https://figshare.com/ndownloader/files/34464122>) returns an AWS WAF
  challenge to programmatic clients. Download in a browser or from a
  cached local copy, then place at the path above.

- **Reactome GMT:** `experiments/kang/resources/c2.cp.reactome.v7.5.1.symbols.gmt`
  (~770 KB). Download the MSigDB C2:CP:REACTOME collection from
  <https://www.gsea-msigdb.org/gsea/msigdb/> (registration required).
  1615 pathways over ~11,000 genes.

## Notebook

- `01_kang_end_to_end.ipynb` — one notebook, ~7 sections, from data loading
  through counterfactual prediction. Open in JupyterLab or VS Code.

## Reproduction

The environment is managed by the parent repo's pixi setup. From the repo
root:

    pixi install
    pixi run jupyter lab   # or open the notebook in VS Code with the pixi kernel

then open `experiments/kang/01_kang_end_to_end.ipynb`.
