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

`01_kang_showcase.ipynb` trains one model on the full gene panel and examines it
closely: saturation, unit fidelity, the competitive ranking, gene attribution,
and the counterfactual. It is the showcase, not the evidence. The evidence is the
sweep below, and where a claim rests on the sweep the notebook shows one figure
and names the CSV rather than re-deriving an underpowered version of it.

Open it interactively. JupyterLab lives in the repository's own
environment, not the grid's, so this one runs from the repository root:

    pixi run -e cuda12 jupyter lab

Training the full panel takes about 4 minutes on an A100 and the whole notebook
about 5. For a fast pass that exercises every cell on a laptop CPU, set
`PYVAE_SMOKE=1` before starting Jupyter. The numbers under smoke are a syntax
check with numbers attached; throw the figures away.

It also runs as a workflow step, which is how a reproducible run should produce
it. `snakemake` executes it through papermill and keeps the executed copy at
`outputs/01_kang_showcase.executed.ipynb`, with `epochs` and `seed` injected from
`config.yaml` into the cell tagged `parameters`.

`jupyter nbconvert --to notebook --execute` would also save an executed copy, so
papermill is not there for that. It is there for the parameter injection, for
writing the output notebook as it goes (a job killed after an hour leaves a
notebook that names the cell it died on, rather than nothing), and for
`--log-output`, which puts cell stdout in the Snakemake log where training is
otherwise silent for the better part of an hour. The `-k python3` flag is not
optional: papermill otherwise looks for the kernel named in the notebook's own
metadata, which is whoever last saved it and need not exist in this environment.

`01_kang_end_to_end.ipynb` is the earlier notebook that the showcase replaces.

## Experiment grid

Scripts 02 to 04 are the systematic sweeps behind the notebook's claims.
`config.yaml` holds the whole grid and `Snakefile` runs it.

| script | question | jobs |
| --- | --- | --- |
| `02_architecture_ablation.py` | which architectural knob fixes tanh saturation, and what does it cost in validation loss | 8 configs |
| `03_sample_size_experiment.py` | how few cells can the model rank interferon from, against a model-free baseline | 4 sweeps |
| `04_encoder_fidelity_experiment.py` | does a nonnegative encoder stop pathway units from firing with inverted sign | 2 budgets x 8 configs |
| `01_kang_showcase.ipynb` | the selected configuration, examined closely, with figures | 1 notebook |

## Running the grid

The grid has its own pixi environment, separate from the repo's development
environment so that a run does not change when someone edits `pyvae/`.
Everything runs from this directory, and every path in `Snakefile` and
`config.yaml` is relative to it, so `snakemake` finds the workflow by name and
needs no `-s`:

    cd experiments/kang
    pixi install
    pixi run all       # the whole grid, 3 GPUs

Those tasks are one line of snakemake each, so use whichever you prefer:

    pixi run plan      snakemake -n -q            what would run, touching nothing
    pixi run all       snakemake -j 63 --resources gpu=3
    pixi run status    snakemake --summary        every output with its state
    pixi run report    snakemake --report outputs/report.html
    pixi run unlock    snakemake --unlock         after a hard kill

Over ssh, put it in a screen session so a dropped connection does not take the
run with it:

    screen -S kang
    pixi run all
    # Ctrl-A then D to detach, `screen -r kang` to come back

A job is skipped when its output already exists, a `summary.csv` for the scripts
and the executed notebook for the notebook, so a killed sweep resumes where it
stopped and a finished one is a no-op. Nothing needs a bookkeeping file, because
the results are the bookkeeping.

One caveat on resuming after a hard kill. Snakemake removes the outputs of a job
that fails cleanly, but a `SIGKILL` to Snakemake itself leaves them behind, and a
half-written notebook looks like a finished one. `--rerun-incomplete` is what
clears that, and `--unlock` clears the lock that stops the next run starting.

## Starting from scratch

To rebuild everything so the results owe nothing to an earlier state:

    cd experiments/kang
    rm -rf outputs .snakemake
    pixi run all

Do not delete `data/` or `resources/`. Neither can be fetched by a script:
figshare answers programmatic clients with an AWS WAF challenge, and MSigDB
wants a login, so losing either costs a manual browser download. `outputs/` and
`.snakemake/` are the only generated state.

`data/kang_processed.h5ad` can go if it is in the way. It is only ever written,
by `load_kang` with `return_path=True`, and never read back, so it is an orphan
rather than a cache and no stale copy of it can reach a run.

`sync.sh` pushes the tree to the GPU box with `--delete`, so a file removed here
is removed there. Without that the box accumulates code that no longer exists:
four superseded test files lived there for weeks, and `pytest` on the box
collected them and failed against an API that had since been rewritten.
`outputs/`, `data/`, `resources/`, `.snakemake/` and the box's `.vscode/` are
excluded, and rsync protects excluded paths from `--delete`, so none of them are
at risk.

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

    snakemake -j 63 --resources gpu=3 -- outputs/ablation/baseline/summary.csv

The two numbers exist for two different reasons. `--resources gpu=3` caps
concurrent GPU jobs at one per card, and `gpu_slot.sh` then hands each job a
real device through an flock, since Snakemake allocates a count rather than a
device id. `-j 63` caps total threads at 63 of 64 cores; each job declares 21,
so three jobs saturate the box exactly once. That second number is not
bookkeeping: three unlimited jobs each spawn one BLAS thread per core, and while
training is GPU-bound and survives it, the metrics phase runs about 1,615 scipy
rank tests per `pathway_activity` call and froze for 5.5 hours under
oversubscription where the same code alone finished in 10 minutes.
