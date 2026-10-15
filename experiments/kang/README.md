# Kang PBMC: the architecture decision, checked against a known answer

Kang et al. (2018) is PBMCs, interferon-β stimulated against control. The
answer is fixed before anything runs: a model that uses its gene→pathway
annotation should name `REACTOME_INTERFERON_ALPHA_BETA_SIGNALING`. That is what
makes this dataset useful here — not as a demonstration, but as an external check
on an architecture decision taken in `experiments/multiple_mieloma`, where no
ground truth exists.

## What this experiment now says

`InformedVAE` is named for a gene→pathway mask, and until this rebuild no
result from it used the mask. The shipped decoder computed

```
logits = dec_out(tanh(dec_latent(z))) + cov_decoder(cov)
```

with `dec_out` masked by Reactome and `cov_decoder` a dense `(n_genes, n_cov)`
term added **after** it. `predict_counterfactual` encodes under one covariate
and decodes the same `mu` under the other, which makes the masked term identical
on both sides — so every covariate contrast was exactly
`cov_decoder[:, b] - cov_decoder[:, a]`, verified against the decoder's own
forward pass.

Two consequences, both measured in `01_architecture.py`:

- The gene-level competitive test this experiment used to report as its primary result
  ranks interferon **first in all eight arms, including every arm whose
  gene→pathway map is shuffled.** It never sees the annotation.
- Ranking the model's own units, the same pathway sat near the **bottom** of
  1,615 under the shipped decoder.

Three flags fix it, and `01_architecture.py` shows each is necessary:

| flag | what it does |
| --- | --- |
| `cov_mode="pathway"` | the covariate's effect is a vector in pathway coordinates, reaching a gene only through the mask |
| `cov_in_encoder=False` | and it is the *whole* conditional effect, because the covariate cannot also reach `z` |
| `nonneg_decoder=True` | and a unit rising raises its genes, so a pathway's sign means what it says |

Under all three the condition effect **is** `InformedVAE.pathway_contrast`: one
signed number per annotated unit, from the weights, with no cells re-encoded and
no counterfactual convention left to choose.

## Running it

    ./00_setup.sh                 # fetch the data and the Reactome GMT
    pixi run all                  # or: snakemake -j 42 --resources gpu=2

Four GPU jobs and three CPU ones, in order, about three hours in total on an
A100:

| rule | what it is | wall time |
| --- | --- | --- |
| `architecture` | `01_architecture.py`, 8 fits: {real, permuted} × {shipped, pathway, uncond, signed} | 32 min |
| `encoder_ablation` | `01b_encoder_ablation.py`, 18 fits: the four options `BASE` holds fixed, one at a time, real and permuted, plus the two mask arms | ~73 min |
| `saturation` | `01c_saturation.py`, 14 fits: the pre-activation scale against annotation worth | ~55 min |
| `units` | `02_kang_units.ipynb`, one fit under the selected architecture | 5 min |
| `figures` | `05_figures.py`, the four figures from that fit's tables | 2 s, no GPU |
| `verify` | `verify_findings.py`, re-derives every number `FINDINGS.md` quotes | seconds |
| `collect_benchmarks` | what the run cost | seconds |

The two estimates are per-fit cost times the number of fits: those two jobs were
run outside Snakemake and have no benchmark file.

Other useful invocations:

    pixi run plan                 # what would run, touching nothing
    pixi run verify               # re-check FINDINGS.md against outputs/
    snakemake --unlock            # clear the lock left by a hard kill

## The notebook

`02_kang_units.ipynb` is the read-out, and it is run by the workflow through
papermill with `epochs` and `seed` injected into the cell tagged `parameters`.
The executed copy lands at `outputs/units/run.ipynb`.

To open it interactively, from the repository root:

    pixi run -e cuda12 jupyter lab

Training the full panel takes about 4 minutes on an A100. For a fast pass that
exercises every cell on a laptop CPU, run it with `smoke=True` (or set
`PYVAE_SMOKE=1` before starting Jupyter). Smoke writes to its own directory,
`outputs/units_smoke/`, so it cannot overwrite a real run — the numbers under
it are a syntax check with numbers attached, and anything drawn from them
should be discarded.

The notebook asserts four things rather than printing them for a reader to
judge, because nothing below them is readable if any fails: no inverted encoder
unit, no negative decoder weight, no unit writing to its genes against its own
direction, and `pathway_contrast` equal to the forward pass at float32
precision.

## The figures

`05_figures.py` draws all four, and it reads nothing but the tables the
notebook wrote:

    python 05_figures.py          # ~2 seconds, no GPU, no model, no dataset

That is deliberate. Figure work is iterative, and the notebook that computes
these numbers needs a card for five minutes. Keeping the two apart means an
axis can be changed on a laptop, and that editing a figure cannot cost a fit:
the notebook is a declared input of a GPU rule, so touching it makes Snakemake
**delete the read-out and refit**, while `05_figures.py` is an input of a CPU
rule that writes nothing but pictures.

It also means each figure is drawn in exactly one place. The notebook used to
both compute and plot, which left no way to tell which copy of a figure was
current.

Everything the script needs is in `outputs/units/`: the ranked units, the
integrated gradients, and `target_cells.csv`, which carries the target unit's
pre-activation in each of the 24,673 cells together with the scanpy
coordinates. The last of those is the only thing here that a CPU cannot
rebuild, which is why it is written out rather than recomputed.

## Required resources (not committed)

Both are gitignored to keep the repository lean, and `00_setup.sh` fetches both.

- **Kang PBMC dataset**, `data/kang_counts_25k.h5ad` (~38 MB), from the figshare
  **API** endpoint `https://api.figshare.com/v2/file/download/34464122`. An
  earlier version of this README said a browser was required; that described the
  WAF-protected `figshare.com/ndownloader` page. The API endpoint serves the
  file to `curl`, which is also what `pyvae.load_kang` uses.
- **Reactome GMT**, `resources/c2.cp.reactome.v7.5.1.symbols.gmt` (~770 KB),
  MSigDB C2:CP:REACTOME v7.5.1 from
  `https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1/`. 1,615
  pathways over ~11,000 genes; 1,615 units over 7,573 genes once intersected
  with this dataset's panel. The same release
  `experiments/multiple_mieloma` pins, so set membership cannot drift between
  the two experiments.

## What was removed, and why

The previous workflow was a 28-run grid — `02_architecture_ablation.py`,
`03_sample_size_experiment.py`, `04_encoder_fidelity_experiment.py` — over eight
configurations, two epoch budgets and two panel widths, plus two notebooks
(`01_kang_end_to_end.ipynb`, `01_kang_showcase.ipynb`) reporting the gene-level
test as the result.

Its one structural claim is now an `assert` inside both architecture scripts:
zero inverted encoder units. That is asserted rather than scored because
`nonneg_encoder` reparameterises every encoder weight as `softplus(theta) *
mask`, which makes inversion *unrepresentable* rather than merely unlikely.

**The encoder-side questions it asked are back, under the current read-out.**
`01b_encoder_ablation.py` re-asks them: the one-at-a-time sweep over
`normalize` and `init`, the 2×2 of `nonneg_encoder` against
`standardize_input`, and the two mask arms the old grid never had:
`dense_decoder` and `dense_encoder`, which drop one mask each and between them
say which side of the model the annotation acts on. Each
option runs real and permuted, so annotation worth is comparable with
`01_architecture.py`'s, and `01b`'s `selected` arm reproduces `01`'s
`real_signed` because both scripts share one data loader. What changed is the
metrics: the old grid scored these options on `pathway_unit_fidelity` and on a
model-against-naive advantage, and both are retired for cause below.

The rest is retired for cause:

- **`pathway_unit_fidelity` medians measured distance from initialisation.** An
  untrained non-negative model scores about 0.75 on it, because `softplus` of
  random weights is positive and roughly uniform — which already *is* the
  unweighted member mean it correlates against — and the score *falls* with
  training. It detects inverted units and is not a quality gate.
- **The sample-size experiment's model-against-naive advantage is reproduced by a
  permuted annotation**, so it measured masking and not biology.

What that grid never had is the one control that decides the question: an arm
where the annotation itself is shuffled. `permute_adjacency` supplies it. It
relabels genes, so every unit keeps its size, its name, the degree multiset and
the **entire** pathway-by-pathway overlap matrix — only which named genes share
a unit changes. The two arms therefore differ in nothing but the biology in the
map.

## Reading the output

| path | what it is |
| --- | --- |
| `outputs/architecture/arms.csv` | one row per arm: fit, both ranks, every diagnostic |
| `outputs/architecture/units.csv` | the per-unit contrast for every arm |
| `outputs/architecture/summary.json` | the arms, what the annotation is worth, the identities |
| `outputs/encoder_ablation/arms.csv` | one row per encoder-option arm: fit, target rank, saturation, sign structure |
| `outputs/encoder_ablation/summary.json` | the arms, annotation worth per option, and the sign-against-scale 2×2 |

A new arm costs its own two fits rather than the whole grid:

    python 01b_encoder_ablation.py --arms real_dense_encoder perm_dense_encoder --append

`--append` merges into the existing `arms.csv` and `summary.json`, leaving the
rows already there untouched and recomputing every aggregate from the merged
table. Snakemake cannot do this, since a rule either runs or does not, which is
why that grid is run by hand and `rule encoder_ablation` carries no metadata.
| `outputs/units/unit_contrasts.csv` | the read-out: one signed row per unit |
| `outputs/units/ig_target.csv` | what the interferon unit reads the pathway through |
| `outputs/units/target_cells.csv` | the target unit per cell: pre-activation, condition, cell type, UMAP |
| `outputs/units/figs/` | the four figures, drawn from the three tables above |
| `outputs/verify.txt` | every `FINDINGS.md` number, re-derived |

`FINDINGS.md` is the results document and is checked against those files by
`verify_findings.py`. If the two disagree, the verifier fails.
