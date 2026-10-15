# Model-driven biology across the myeloma precursor sequence

What a pathway-informed VAE says changes in bone-marrow T cells between
non-cancer, MGUS, smouldering myeloma and symptomatic myeloma — asked of two
independent datasets, fitted separately under one audited design, and read over
the model's **own pathway units**.

**Read [`FINDINGS.md`](FINDINGS.md) first.** It carries the intent, the methods,
the results in three acts, a guide to what they do and do not support, and a
closing section recording every design decision from each dataset's point of
view.

## Three acts, in order

```
01_architecture.py  ->  02_biology.ipynb  ->  03_donor_test.py  ->  04_figures.py
   which decoder          the biology,          the same read-out,     three pages
   uses the mask          over those units      donors as the unit
```

The first act is not preamble. `InformedVAE` is named for a gene→pathway mask,
and until this rebuild no result from it used the mask: the shipped decoder
computed `logits = dec_out(tanh(dec_latent(z))) + cov_decoder(cov)` with
`dec_out` masked and `cov_decoder` dense and added *after* it, so every
covariate contrast was that dense term and nothing else. A **permuted**
gene→pathway map reproduced the previous results at Spearman +0.97. Act 1 is
what establishes that, selects the fix, and measures what it costs.

| file | what it does |
| --- | --- |
| [`01_architecture.py`](01_architecture.py) | eight fits per run — {real, permuted annotation} × {shipped, pathway, uncond, signed} — scoring each on the fit, the unit rank of a fixed target, the retired gene-level rank, and four forward-pass identities. Four seeds on Zavidij, one on Foster |
| [`02_biology.ipynb`](02_biology.ipynb) | the analysis. Audits every covariate, trains under the selected architecture, and reads the stage contrast off `InformedVAE.pathway_contrast`: one signed number per unit. Once per study, one GPU each |
| [`03_donor_test.py`](03_donor_test.py) | the same read-out with the **donor** as the sampling unit. Refits with the stage block interacted with donor, so each donor's score is one column of `cov_path` minus the healthy mean. The only p-value in the experiment |
| [`04_figures.py`](04_figures.py) | the three paper figures, assembled from the three acts' CSVs. No model, no GPU, seconds |
| [`myeloma.py`](myeloma.py) | the shared library: cell set, phenotype harmonisation, the architecture constant, contrasts, palette. Everything in it has a live caller |
| [`table_figure_metadata.md`](table_figure_metadata.md) | paper-ready title and caption for every figure and table. The figures themselves carry neither |
| [`verify_findings.py`](verify_findings.py) | re-derives every number in `FINDINGS.md` from the outputs — and from the figures, via `fig_panel_data.csv` — printing a pass/fail line each |
| [`Snakefile`](Snakefile) | the workflow; `config.yaml` holds the knobs |

## The architecture, in one table

Three flags, and `01_architecture.py` shows each is necessary:

| flag | what it does |
| --- | --- |
| `cov_mode="pathway"` | the covariate's effect is a vector in pathway coordinates, reaching a gene only through the mask |
| `cov_in_encoder=False` | and it is the *whole* conditional effect, because the covariate cannot also reach `z` |
| `nonneg_decoder=True` | and a unit rising raises its genes, so a pathway's sign means what it says |

It lives in one place, `myeloma.ARCH`, because the notebook, the decision script
and the donor test must fit the *same* model for their read-outs to be
comparable.

**It costs likelihood, and the document leads with that.** A dense per-gene
matrix describes one large diffuse shift with fewer parameters than 1,615 annotated units
can. The justification is not fit; it is that the real annotation then beats a
size-matched shuffle by several nats where it previously beat it by under two,
and that the contrast becomes a signed per-pathway number whose sign means what
it says.

## One pathway collection, two jobs

| collection | sets | job |
| --- | --- | --- |
| Reactome C2:CP | 1,615 | defines the 9,938-gene panel, the encoder mask, **and the read-out** |
| Hallmark | 50 | input to `01_architecture.py` only, where the retired gene-level baseline is scored |

Hallmark used to be the read-out. It is not any more, because that read-out
never saw the annotation — it ranks interferon first even when the gene→pathway
map is shuffled. It survives in act 1 as exactly that evidence.

Reading the units instead brings back the problem Hallmark was adopted to avoid:
Reactome is **strictly nested**, so `interferon alpha beta signaling` (59 genes)
sits inside `interferon signaling` (188) inside `cytokine signaling in immune
system` (682), and three hits there are one finding. The pipeline handles it by
not doing the thing that breaks: **no count over units is ever a result, and no
overlap or enrichment test over units is computed anywhere.** Signed contrasts,
ranks, and a donor-level `q`. The full reasoning, including why pruning
Reactome to its leaves fails, is in
[`resources/PROVENANCE.md`](resources/PROVENANCE.md).

## Running it

```bash
cd experiments/multiple_mieloma
./00_setup.sh                 # fetch the GMTs automatically; prints how to get the atlas
pixi run plan                 # dry run: what would execute
pixi run all                  # snakemake -j 63 --resources gpu=3
pixi run verify               # re-check FINDINGS.md against outputs/
```

Budget on three A100 80GB cards, roughly 3.5 hours wall, dominated by one job:
Foster's architecture run is 8 arms at ~25 minutes each and nothing else is
close. A job is skipped when its output exists, so a killed run resumes and a
finished one is a no-op. Each run's executed notebook, log and benchmark sit in
that run's own output directory.

A saved fit is reused when the design fingerprint matches exactly, so changing a
figure costs seconds rather than half an hour. The fingerprint covers study,
epochs, seed, cell set, covariate design **and the architecture** — never just
the filename, which is what stops a checkpoint from the shipped decoder being
adopted by the new notebook and read as though it came from it.

To run one study interactively instead:

```bash
pixi run lab                  # then open 02_biology.ipynb
```

The first code cell is tagged `parameters`; papermill overrides `study`,
`epochs` and `seed` from the workflow. A run below the canonical 250 epochs
writes to `outputs/biology/<study>__epochs<N>/`, so a quick smoke test cannot
overwrite a finished run's tables and figures.

## What it assumes you know

`InformedVAE`, `pathway_contrast`, `unit_contrast`, `unit_sign_structure`,
`permute_adjacency` and `integrated_gradients` are `pyvae`'s and documented
there. `experiments/kang` is the same architecture decision on a dataset whose
answer is known in advance, and is the external check that this one selected
for the right reason rather than merely selecting.

## Outputs

```
outputs/
  architecture/<study>/            ACT 1, canonical seed
    arms.csv                       one row per arm: fit, both ranks, diagnostics
    units.csv                      the per-unit contrast for every arm
    summary.json                   arms, annotation worth, the four identities
    seed<n>/                       the other seeds, identical filenames
  biology/<study>/                 ACT 2
    summary.json                   the design, the four checks, the ranking
    unit_contrasts.csv             THE RESULT: one row per unit x lineage x
                                   contrast, signed, with its rank
    covariate_audit.csv            what entered, what did not, and why
    table1_design.csv              cells and donors per stage, split by lineage
    table2_drivers.csv             encoder and decoder drivers, side by side
    table3_lineage_agreement.csv   CD4 vs CD8 over all units
    figs/fig1..fig5                the per-study figures
    model.pt, model_fingerprint.json, training_history.csv
    run.ipynb, run.log, benchmark.tsv
  donor_test/<study>/              ACT 3
    donor_scores.csv               one row per donor x lineage x unit x estimator
    donor_test.csv                 trend and endpoint tests, with q
    summary.json                   counts, the power floor, what survives
  figures/                         THE PAPER FIGURES
    fig1_Zavidij_2020.pdf/.png     one cohort, four panels
    fig2_Foster_2026.pdf/.png      the other, identical layout and scale
    fig3_comparison.pdf/.png       what replicates and what does not
    fig_panel_data.csv             every number in the three figures
  verify.txt                       every FINDINGS.md number, re-derived
  benchmarks.csv                   what the run cost
```

Everything a run produced — its executed notebook, log and benchmark — sits in
that run's own directory, so a result is never separated from the thing that
made it.

`fig_panel_data.csv` holds every number that appears in the three figures, and
`verify_findings.py` reads it. The prose is therefore checked against the
figures rather than against a re-derivation that could drift from them.

The figures carry **no title and no caption** — only panel letters, panel
headings and axis labels. Both live in
[`table_figure_metadata.md`](table_figure_metadata.md), so a title never has to
be cropped out of a PDF and a caption never exists in two places at once.

## Four things to know before reading any number

**A unit contrast is estimated from cells, not donors.** Every donor sits at
exactly one stage, so the per-unit numbers in act 2 carry no donor-level
variance at all: a large contrast there is **not** evidence that the effect
would appear in new patients. Act 3 supplies the missing sampling unit, and it
has overturned a cross-study replication once already.

**Units are not independent, so do not count them.** Reactome nests. The top of
any ranking here is families, not independent findings, and every figure caption
that lists units says so.

**Most units are saturated, and a saturated unit writes nothing.** `tanh`
flattens, and this architecture pushes most units deep into the flat tail — far
more than the decoder it replaces. A unit can be biologically involved and
invisible in the ranking because it is pinned. Each fit's `summary.json` carries
the count, and it is a limit on what can be *found*, not a reason to distrust
what was.

**Nothing adjusts for sequencing depth by default.** Detection depth IS a real
confounder here — patients are sequenced deeper than controls — and `depth_bin`
passes the covariate audit in both studies, so it is one word in `config.yaml`
away from entering the design. It is off because neither source paper corrects
for depth and a corrected result cannot be compared one-for-one with their
claims. There is no longer a second depth-adjusted fit per study: conditioning
on `depth_bin` moved the previous read-out by less than 0.02 in z, and carrying
a twin of every fit to re-answer that doubled the compute the workflow requires for a
settled question.
