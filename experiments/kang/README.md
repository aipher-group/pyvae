# Kang PBMC: end-to-end pyvae sanity check

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
head-to-head, since pyvae is a PyTorch package built on the same modeling
philosophy with a negative-binomial likelihood, covariate conditioning, and
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

`01_kang_end_to_end.ipynb` runs from data loading through counterfactual
prediction in about seven sections. Open it in JupyterLab or VS Code.

    pixi install
    pixi run jupyter lab

## Experiment grid

Scripts 02 to 04 are the systematic sweeps behind the notebook's claims.
`config.yaml` holds the whole grid and `Snakefile` runs it.

| script | question | jobs |
| --- | --- | --- |
| `02_architecture_ablation.py` | which architectural knob fixes tanh saturation, and what does it cost in validation loss | 8 configs |
| `03_sample_size_experiment.py` | how few cells can the model rank interferon from, against a model-free baseline | 4 sweeps |
| `04_encoder_fidelity_experiment.py` | does a nonnegative encoder stop pathway units from firing with inverted sign | 2 budgets x 8 configs |

## Running the grid

The grid has its own pixi environment, separate from the repo's development
environment so that a run does not change when someone edits `pyvae/`:

    cd experiments/kang && pixi install

The tasks carry the right flags and run from the repository root, because every
path the scripts default to is repo-root relative:

    cd experiments/kang
    pixi run plan      # dry run, touches nothing
    pixi run all       # run whatever is missing
    pixi run status    # every output with its state
    pixi run report    # provenance: code, config and runtime per output

From anywhere else, point pixi at the manifest:

    pixi run --manifest-path experiments/kang/pixi.toml all

A job is skipped when its `summary.csv` already exists, so a killed sweep
resumes where it stopped and a finished one is a no-op. Nothing needs a
bookkeeping file, because the results are the bookkeeping.

Each job is also benchmarked. Snakemake writes one TSV per job under
`outputs/benchmarks/` with wall time, peak RSS, peak USS and CPU time, and
`collect_benchmarks` gathers them into `outputs/benchmarks.csv`. Two things
to read there: what the grid costs to reproduce, and whether the thread
budget is right. A job whose `cpu_time` is far below `threads` multiplied by
wall time spent that time waiting rather than working, which is the
signature of the oversubscription described below. The collector reports how many of the
grid it measured, so a partial run is visible as such.

One argument-parsing trap: `--resources` takes several values, so a target
named after it gets swallowed. Separate them with `--`:

    snakemake -s experiments/kang/Snakefile -j 63 --resources gpu=3 \
        -- experiments/kang/outputs/ablation/baseline/summary.csv

The two numbers exist for two different reasons. `--resources gpu=3` caps
concurrent GPU jobs at one per card, and `gpu_slot.sh` then hands each job a
real device through an flock, since Snakemake allocates a count rather than a
device id. `-j 63` caps total threads at 63 of 64 cores; each job declares 21,
so three jobs saturate the box exactly once. That second number is not
bookkeeping: three unlimited jobs each spawn one BLAS thread per core, and while
training is GPU-bound and survives it, the metrics phase runs about 1,615 scipy
rank tests per `pathway_activity` call and froze for 5.5 hours under
oversubscription where the same code alone finished in 10 minutes.
