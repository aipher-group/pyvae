# Kang PBMC informed VAE: what the results show

A guide to the results in `outputs/`, written so you can work from them without
re-running anything. Every number below is in a CSV or the executed notebook, and
each section says where.

The dataset is Kang et al. (2018): 24,673 human PBMCs, control against
interferon-β stimulated, 8 donors, 8 cell types, 15,706 genes. Because interferon
signalling is the known right answer, any ranking the model produces can be
scored against it. The pathway map is Reactome C2:CP, 1,615 pathways over the
7,573 genes that appear in both the panel and the map.

## What you have, and what each part is for

The scripts are the evidence and the notebook is the showcase. That split matters
when you write up.

`02_ablation_merged.csv`, `03_sample_size_merged.csv` and the two
`04_fidelity_e*.csv` files come from 28 independent training runs across a
parameter grid. They are what supports a claim, because each row is a separate
model and the comparison between rows is controlled.

`01_kang_showcase.executed.ipynb` trains one model, the configuration the grid
selected, and examines it closely with figures. It is an illustration of a single
model, not evidence about architecture. Where it makes a claim that rests on the
grid, it prints one figure and names the CSV rather than re-deriving a weaker
version of a comparison the grid already settles. Read it that way, and cite the
CSVs when you argue.

`benchmarks.csv` has wall time, peak memory and CPU time per job, which you need
for any statement about cost.

## Finding 1: a non-negative encoder is what stops pathway units encoding their pathway upside-down

This is the strongest result in the grid and the one to lead with.

`pathway_unit_fidelity` correlates each pathway unit's activation against the
mean z-scored expression of that unit's own member genes. A unit that means what
its name says should correlate positively. In `04_fidelity_e250.csv`:

| `nonneg_encoder` | median \|corr\| | fraction inverted |
| --- | --- | --- |
| on  | 0.66 to 0.88 | **0.000** |
| off | 0.12 to 0.18 | **0.48 to 0.54** |

With the constraint off, roughly half of all units carry their pathway with the
sign flipped. Their activation goes *down* when the pathway's genes go *up*. With
it on, not one unit out of 1,121 tested does that, at either the 100 or 250 epoch
budget, across all four configuration pairs.

The mechanism is worth explaining in the write-up because it is not a tuning
effect. The constraint reparameterises each weight as `softplus(θ) * mask`, so
every live weight is positive by construction. Sign inversion is not discouraged,
it is unrepresentable. That is why the result is a switch rather than a dial, and
why the fraction inverted is exactly zero rather than merely small.

`standardize_input` on its own does nothing: 0.169 against the baseline's 0.170,
both around 50% inverted. It only helps once stacked on top of nonnegativity,
where `both` reaches 0.847 against `nonneg_only` at 0.722. So the honest claim is
that nonnegativity is necessary and standardisation is a refinement.

One caveat to state plainly: 494 of the 1,615 pathways have fewer than 10 member
genes measured and are not tested at all. The fidelity numbers describe the 1,121
that are.

## Finding 2: the informed decoder is what makes counterfactual prediction work

Take control cells, hold their inferred latent state fixed, flip only the
condition covariate, decode, and compare the predicted per-gene shift against the
real one. In `02_ablation_merged.csv`:

| decoder | counterfactual correlation |
| --- | --- |
| informed (3 configs) | 0.978 to 0.981 |
| dense (5 configs) | 0.803 to 0.861 |

The two groups do not overlap. Masking the decoder so each pathway unit writes
only to its own member genes is what buys this.

How the metric is constructed matters, and is worth a short methods paragraph.
Both sides of the comparison come from the decoder: the same cells are decoded
twice, once under their real covariate and once with the condition swapped, and
the shift is the difference between those two decoded profiles. Differencing
against the observed control expression instead would mix a decoder output with a
raw input, so every gene would carry the decoder's reconstruction bias and the
number would measure reconstruction error as much as the covariate effect.
Decoding both sides cancels that bias, because it appears in both terms.

## Finding 3: batch normalisation fixes saturation, layer normalisation does not

The encoder's pathway layer uses tanh, and a unit stuck in the flat tail of tanh
carries no gradient and no information. Counting units where more than half the
cells sit beyond \|h\| > 0.99, from `02_ablation_merged.csv`:

- `normalize_batch`: 0 units, max saturation 0.089
- `baseline` (no normalisation): 8 units, max 0.996
- `normalize_layer`: 18 units, max 1.000

Layer normalisation is worse than no normalisation at all. Do not write
"normalisation fixes saturation" as though it were one knob, because the two
options behave oppositely. Batch normalisation standardises each unit across the
cells in a batch, which is the axis the problem lives on. Layer normalisation
standardises across units within a cell, which is the wrong axis here.

## Finding 4: the model's advantage is panel width, not sample size

The experiment is called the sample-size experiment and its answer is that sample
size is not the variable that matters. From `03_sample_size_merged.csv` and
`figs/08_sample_size.png`, the rank given to interferon alpha/beta, where 1 is
correct:

| training cells | model, full panel | naive, full panel | model, 5k HVG | naive, 5k HVG |
| --- | --- | --- | --- | --- |
| 246 | 1 | 12 | 1 | 1 |
| 986 | 1 | 11 | 1 | 1 |
| 4,934 | 1 | 14 | 1 | 1 |
| 19,704 | 1 | 14 | 1 | 1 |

Four flat lines. Cell count moves neither method. The whole separation is between
the panels: on the full 15,706-gene panel the naive gene-set mean cannot find
interferon no matter how much data it gets, and on 5,000 highly variable genes it
is essentially perfect.

So the claim to make is that the pathway constraint buys robustness to a wide,
noisy gene panel, which is the realistic setting, and not that it needs fewer
cells. Phrasing it as a sample-size advantage would be easy to attack, because
the table plainly shows sample size doing nothing.

The other half of this result is worth stating: the model holds rank 1 with 246
training cells against 28.5 million parameters. The mask is doing enough
regularisation to survive a ratio that would ordinarily be hopeless.

## Finding 5: the counterfactual gets direction right and magnitude half right

From the notebook, section 10. Correlation between predicted and real per-gene
shift is 0.980, the intercept is +0.0005, and the fitted slope is 0.475.

Read that as two separate statements. Direction and relative ordering are close
to perfect, and genes that did not move are predicted not to move. Magnitude is
compressed by about half, so the model systematically under-predicts how far each
gene travels. `figs/07_counterfactual.png` draws the fit against the identity
line so the compression is visible rather than buried.

Under-prediction of this kind is expected when you decode from the posterior
mean, which is what `predict_counterfactual` does deliberately so that repeated
calls are not noisy. Say so, and do not present the 0.980 alone.

## Finding 6: what the gene attributions do and do not say

Section 9 of the notebook reports two numbers that sound similar and measure
different things.

Integrated gradients puts 27.8% of the attribution for the interferon alpha/beta
unit on ISG15, and 58.6% on its top five genes. That is the unit's *sensitivity*:
which genes move the activation.

The decoder-side `top_gene_share` puts 4.9% on IFIT1, against 1.8% if all 57
members contributed equally. That is the pathway's *differential expression*: how
concentrated the actual response is across members.

They disagree, and they name different genes. The pathway responds broadly while
the unit reads it through a handful of members. Both are defensible, and quoting
one as though it were the other would be a mistake. The `top_gene_share` column
exists in every `pathway_activity` output so you can check any pathway for whether
its score is really one gene wearing a pathway's name.

Related, and useful as an illustration: in the notebook's table comparing the
encoder-side Bayes factor against the competitive test, `OAS_ANTIVIRAL_RESPONSE`
sits at encoder rank 5 and competitive rank 1,410. It has 9 genes, 6 of them
shared with the interferon set. A self-contained test cannot discount a gene that
is shared with something genuinely induced; a competitive test, which compares
members against the rest of the panel, can. That single row makes the argument
for competitive testing better than any general statement.

## Four claims to avoid

**Do not use `competitive_z_interferon` to justify the architecture.** It ranges
from 9.20 to 9.44 across all eight ablation configurations, and the plain
baseline scores highest at 9.443. The interferon signal survives every
architecture tested. The metric shows the biology is recoverable, which is worth
saying, but it cannot support a choice between designs. Validation loss,
saturation and the counterfactual are the metrics that separate them.

**Do not claim the units beat the naive baseline at discrimination.**
`unit_vs_naive_auc_diff_mean` runs from 0.002 to 0.010, all positive but tiny,
and the non-negative configurations score *lower* on it than the unconstrained
ones. Fidelity improved enormously while added discriminative power stayed near
zero. This is a real tension and it is better acknowledged than left for an
examiner to find. It also fits Finding 4: the value here is interpretability and
robustness, not raw classification accuracy.

**Do not read the donor-level q-values as effect sizes.** The paired Wilcoxon
test in section 11 runs over 8 donors. With 8 pairs there are 2^8 = 256 possible
sign configurations, so the smallest two-sided p-value attainable is 2/256 =
0.0078, and every gene that moves consistently in all 8 donors hits exactly that
floor. In your results, 3,519 genes share p = 0.007812 and q = 0.034869. You can
verify it is one tie group: 0.0078125 × 15706 / 3519 = 0.034869. Rank that table
by `lfc_mean`, never by q, and say why.

**Do not compare `best_epoch` across the 100 and 250 epoch budgets.** The
learning rate follows a cosine schedule with `T_max` set to the epoch budget, so
the two budgets are separate experiments rather than one being a checkpoint of
the other. `best_epoch == epochs` is the designed outcome, not early stopping
failing to trigger. Compare final validation loss within a budget only.

## Reproducibility, and why you can trust these numbers

The whole grid is a Snakemake workflow. Running it produces 35 jobs from
`config.yaml`, and a job is skipped when its output already exists, so the
results are their own bookkeeping.

Determinism has been checked directly rather than assumed. The notebook trains
from scratch and lands on validation loss 1308.8978, fidelity median 0.791395 and
competitive z 9.167697, which match the `notebook_both` row of
`04_fidelity_e250.csv` to every digit. The grid has also been rebuilt from an
empty directory on separate occasions, returning bit-identical values each time.
Two independent reasons to believe a number here is not an accident of one run.

Cost, from `benchmarks.csv`: the full grid is 2h49m wall on three A100 80GB cards,
4.61 hours of summed job time, peak memory 8.8 GB. One job dominates,
`ss_full_small` at 2.2 hours, because it trains a whole fraction-by-repeat grid.
Training the showcase model itself takes 3.8 minutes.

## Where the argument is still thin

The AUC result in "claims to avoid" is the obvious opening for a critic, and it
deserves a paragraph rather than a footnote. The strongest available answer is
that fidelity and counterfactual behaviour are the properties an interpretable
model is for, and a naive gene-set mean offers neither a generative model nor a
counterfactual.

The sample-size experiment currently varies cell count, which turns out not to
matter, and only has two panel widths, which turns out to be the variable that
does. A sweep over panel width would be the natural follow-up if GPU time ever
becomes available.

Everything here rests on one dataset with one known answer. Interferon stimulation
of PBMCs is a strong, clean perturbation. Claims about how the method behaves on
subtler biology are not supported by what you have, so scope them accordingly.
