# Figure and table metadata

Titles and captions for every display item. The figures themselves carry **no
title and no caption** — only panel letters, panel headings and axis labels — so
that a title never has to be cropped out of a PDF and a caption never exists in
two places at once. Everything a reader needs is here.

There is **one variant** of every item. The depth-adjusted twin is gone:
conditioning on `depth_bin` moved the previous read-out by less than 0.02 in z,
and carrying a second copy of every fit and every figure to re-answer that
doubled the compute the workflow requires for a settled question. `depth_bin` remains an
audited, available covariate — one word in `config.yaml` — and the audit for it
is printed in every run's `covariate_audit.csv` whether or not it enters.

Studies: `Zavidij_2020` (9,863 cells, 30 donors) and `Foster_2026` (125,602
cells, 22 donors). `Foster_2026` is the atlas's own cohort, labelled
`This_study` in the source data and renamed to the `<FirstAuthor>_<Year>`
convention used by the other ten studies.

## What every number in these figures is

A **unit contrast**: how far one stage moves one of the model's 1,615 Reactome
units, in that unit's own pre-activation, holding phenotype fixed. It is a
difference of two columns of `cov_path` — no cell is re-encoded and no cell is
re-weighted — and under this architecture its sign is the direction its member
genes move.

Two quantities appear, and they are not interchangeable:

- **`delta_pre`** is the exact weight difference, cell-independent.
- **`member_shift_z`** is what reaches the genes: `delta_post · w_sum / √n`,
  where `delta_post` is the mean change in `tanh(pre)` over the healthy cells of
  that lineage. The `1/√n` corrects a size tilt that otherwise hands the top of
  the ranking to the smallest units (Spearman between |statistic| and set size
  falls from −0.49 to −0.09). **Every figure ranks and colours by this one.**

### It is not on a common absolute scale across cohorts

`member_shift_z` carries `delta_post`, which depends on how far each fit pushed
its units into the flat tail of `tanh`. `Foster_2026`'s endpoint values run
about **seven times** larger than `Zavidij_2020`'s — the cohort scales are
0.29 and 1.92 — so a shared raw colour scale renders the smaller cohort a blank
grey panel, which is exactly what an earlier version of Figure 1 was.

Figures 1a and 2a therefore colour **`member_shift_z` divided by the cohort's
own scale**, the 99th percentile of its absolute value over all its units,
lineages and contrasts. The two pages then share one norm, and a colour means
"this far out for its own cohort". The raw values and each cohort's divisor are
both in `fig_panel_data.csv`, so the drawn number is reproducible from the file.
Panels b, c and d are in raw units and say so on their axes.

## Three limits that belong in any caption reusing these panels

1. **Reactome nests.** `interferon alpha beta signaling` (59 genes) sits inside
   `interferon signaling` (188) inside `cytokine signaling in immune system`
   (682). Neighbouring rows are often one family. **No count over units is a
   result**, and no overlap or enrichment test over units is computed anywhere.
2. **A saturated unit writes nothing.** Most units sit in the flat tail of
   `tanh` under this architecture, far more than under the decoder it replaces.
   A unit can be biologically involved and invisible because it is pinned. Each
   fit's `summary.json` carries the count.
3. **Cells are not donors.** Every donor sits at exactly one stage, so panels
   a–c of Figures 1 and 2 carry no donor-level variance. Panel d does, and it is
   the only place a p-value appears.

## Production notes

All three figures are **183 × 244 mm** (7.2 × 9.6 in) portrait, vector PDF with
embedded fonts, plus a 300 dpi PNG of the same page. The size is exact and
identical across all three, so Figures 1 and 2 can be laid on top of each other;
that is the point of their shared layout, row order and colour scale, and it is
why the pages are not cropped to their content.

Smallest type is 4.6 pt (the unit labels in Figure 3b) and 5.4 pt (the ladder
row labels in Figures 1a and 2a). If the page is reduced below about 95%, those
fall under the 5 pt floor most journals set; Figure 3b is the one to watch.

### Palette, as measured

Three categorical slots — `#2a78d6` blue, `#eb6834` orange, `#1baf7a` aqua —
against a light print surface `#fcfcfb`. Measured, not judged, with the
`dataviz` skill's `validate_palette.js`:

| check | all pairs | adjacent pairs |
| --- | --- | --- |
| CVD separation (OKLab ΔE ×100) | **9.2** deutan, aqua↔orange | 9.2 deutan, aqua↔orange |
| Normal-vision floor | **24.0**, aqua↔blue | 27.6, aqua↔orange |
| Lightness band, chroma floor | pass | pass |
| Contrast vs surface | **WARN**: aqua at 2.74:1 | same |

All-pairs is the governing column, because these are scatters where every slot
is co-visible. 9.2 sits in the 6–8-and-up band that is legal **only with
secondary encoding**, which is why every categorical mark in these figures is
directly labelled, and the 2.74:1 aqua WARN obligates the same relief rather
than disqualifying the slot. Do not reuse aqua for an unlabelled mark, and do
not add a fourth hue: a fifth series folds into "other".

Signed quantities use a diverging **blue → near-white (`#e8e8e6`) → orange**
ramp centred at zero. Its two poles pass every check on their own (ΔE 24.7
protan, 33.6 normal, both above 3:1 contrast). Running the categorical
validator on the *three* stops instead reports a lightness-band and chroma-floor
failure on the midpoint — that is the ramp being **correct**, not broken: a
diverging midpoint must be neutral and near the surface, and a hue there is the
error the ramp exists to avoid. The validator says as much in its own scope
note. Do not "fix" it.

A sequential ramp on signed data would hide the sign, which is the one property
this architecture exists to provide.

There is no dark variant. These are print figures on a light surface; a dark
mode would be a second palette to validate and maintain for no reader.

`fig_panel_data.csv` holds every plotted number, keyed by figure, panel, unit,
lineage, contrast and quantity. `verify_findings.py` reads it, so the prose is
checked against the figures and not against a re-derivation of them.

---

## Figure 1 — `fig1_Zavidij_2020.pdf`

**Title.** Stage-associated activity of pathway-informed decoder units in
bone-marrow T cells, `Zavidij_2020`.

**Caption.** Each panel reads one fit of a pathway-informed VAE in which the
covariate's effect is a vector in pathway coordinates, so a stage contrast is
one signed number per annotated Reactome unit.
**(a)** The 18 units the stage sequence raises most and the 7 it lowers most,
ordered by the mean endpoint contrast over CD4 and CD8 across *both* studies, so
Figures 1a and 2a show the same rows in the same order. Columns are the three
ladder contrasts against the non-cancer baseline, within each lineage, so a row
reads left to right as a trajectory. Colour is `member_shift_z` on a shared
diverging scale; there is no significance mark, because there is no valid test
over nested units. Reactome nests — read families, not rows.
**(b)** The same endpoint contrast in CD4 against CD8, one point per unit. These
are two near-independent estimates of one disease effect, so agreement is a
diagnostic on the fit rather than a biological claim: the limiting cell count is
annotated, because a weak correlation from a thin arm and a weak correlation
from genuine independence look identical here.
**(c)** What the gene→pathway annotation is worth, per architecture, from
`01_architecture.py`: validation loss under a size-matched **permuted**
adjacency minus validation loss under the real one. The permutation relabels
genes, preserving every unit's size, its name, the degree multiset and the
entire pathway-by-pathway overlap matrix, so the two arms differ in nothing but
the biology in the map. `shipped` is the decoder every previous result used.
**(d)** The same leading units with the **donor** as the sampling unit. Each
point is one donor's score — their own `cov_path` column minus the healthy
donors' mean — scaled per row so six units of different magnitudes share one
axis. Units the donor test could not score, because they have fewer than ten
measured member genes, are left out rather than drawn as an empty row; the
figure log names any that were.

A ring **at the left edge**, beside the row label and offset by lineage, marks a
Spearman trend across all four stages at q < 0.05, from **100,000** donor-label
permutations with Benjamini–Hochberg over the 1,262 scored units. It sits there
rather than inside the point cloud because the trend is a property of the whole
row: drawn at the mean of the points it landed over whichever stage held the
most donors and read as "significant at SMM".

The permutation count is a power parameter. BH multiplies the smallest
attainable p by the number of tests, so the best q this design can produce is
`units / (draws + 1)`: **0.013** at 100,000 draws, and 0.126 at the 10,000 an
earlier version used — where the panel could not have shown a single ring
whatever the data said. The endpoint test cannot reach q < 0.05 in either
cohort at any number of draws, because there are too few donors per stage, and
is reported only beside its floor. Both floors are in
`donor_test/<study>/summary.json`.

BH over nested units is conservative, so a ring is trustworthy and its absence
is uninformative about any single unit.

## Figure 2 — `fig2_Foster_2026.pdf`

**Title.** The same analysis in `Foster_2026`, the atlas's own cohort.

**Caption.** Identical layout, row order and colour scale to Figure 1, so the
two pages can be laid on top of each other. All four panels as in Figure 1.
`Foster_2026` has 13× the cells and fewer donors per arm than `Zavidij_2020`
(3 non-cancer, 1 MGUS, 13 smouldering, 5 symptomatic), which is why panel d's
power floor is reported in `donor_test/summary.json` alongside the result: with
one MGUS donor the endpoint test cannot reach q < 0.05 for any unit, and a null
result there is an impossible one rather than a negative one.

## Figure 3 — `fig3_comparison.pdf`

**Title.** What replicates between the two cohorts, read over the model's own
units.

**Caption.** The two cohorts are fitted independently and never pooled: 69% of
the atlas's non-cancer donors sit in studies containing no patients, so a pooled
disease contrast would substantially be a contrast between studies, and
`study_id` cannot be adjusted for without becoming collinear with the thing
being estimated. This page compares the two fits' **outputs**.
**(a)** Endpoint `member_shift_z` in one cohort against the other, one point per
shared unit, with the Spearman correlation.
**(b)** Each cohort's ten leading units and where the other cohort ranks them.
A filled mark is the unit's rank in its own cohort, an open mark its rank in the
other, joined by a line; the axis is logarithmic over all 1,615 units.
**(c)** The same comparison with the **donor** as the sampling unit: each
cohort's donor-level trend ρ per unit, against the other's. This is the
comparison that matters for replication, and it is the one that has been
overturned before — under the previous gene-level read-out, interferon looked
shared across both cohorts and the donor-level test found it strong in one and
absent in the other, with that cohort's apparent MGUS peak resting on a single
donor.
**(d)** Where each architecture puts the unit the gene-level baseline points at,
both cohorts, logarithmic. Lower is better. This is the decision in one panel.
**(e)** The two cohorts side by side, in words. Counts over nested units are
descriptive and are not results.

---

## Tables, per study

Written to `outputs/biology/<study>/`.

### Table 1 — `table1_design.csv`

**Title.** Cells and donors per disease stage, split by T-cell lineage.

**Caption.** Every number in Figures 1 and 2 is limited by the thinnest arm
here, which is why the table comes first rather than last. Cell counts are after
the atlas's own donor `analysed` flag is honoured, which removes 3 donors and
18,226 cells from `Foster_2026` and nothing from `Zavidij_2020`.

### Table 2 — `table2_drivers.csv`

**Title.** The four leading units, read from both sides.

**Caption.** Two quantities that sound alike and measure different things.
**Encoder side**, integrated gradients: which genes move the unit's activation —
its sensitivity, what it reads the pathway through. **Decoder side**, member
share: `w_out[gene, unit]` over the unit's total, the share of the written
response each member carries. Under a non-negative decoder that is a genuine
share — every term positive, summing to one — which was not true of the decoder
this replaces. The two routinely name different genes; `sides_agree` records
how often they agree on the top one. Neither is a biological claim on its own.

### Table 3 — `table3_lineage_agreement.csv`

**Title.** CD4 against CD8 over all units, with a bootstrap interval.

**Caption.** Spearman correlation of `member_shift_z` between the two lineages
for each ladder contrast, with a 1,000-draw bootstrap percentile interval and
the limiting cell and donor counts. A diagnostic on the fit, not a biological
claim: if agreement tracks the thinnest arm, what is visible is noise
attenuation rather than lineages decoupling.

### `unit_contrasts.csv`

**Title.** The result: every unit's signed stage contrast.

**Caption.** Long format, one row per unit × lineage × contrast, carrying
`delta_pre` (the exact weight difference), `delta_post`, `member_shift_z`, the
unit's member count, the number of baseline cells the post-activation shift was
evaluated on, and the unit's rank within that lineage and contrast. The full
1,615; the figures show its two ends.

### `covariate_audit.csv`

**Title.** Every candidate covariate, and why it did or did not enter.

**Caption.** Four checks per block: does it vary in this study, does it vary
*within* donors, is its missingness aligned with a cohort, and is any level
concentrated in one cohort. Blocks that were not requested are audited anyway,
so the decision is visible rather than implied. A covariate that does not vary
within a donor cannot be separated from donor identity, and cohort is itself a
donor property, so such a block competes with the stage effect for the same
variance.

---

## Architecture outputs, per study

Written to `outputs/architecture/<study>/`, with other seeds in `seed<n>/`.

### `arms.csv`

**Title.** Eight architectures, scored on the same contrast.

**Caption.** One row per arm of {real, permuted annotation} × {shipped,
pathway, uncond, signed}. Carries the fit, the unit rank of the fixed target,
the **gene-level** competitive rank of the same pathway — kept here and only
here, because ranking it first in every arm including the permuted ones is the
evidence that read-out was annotation-blind — `masked_share`, the decoder's
negative-weight fraction, unit-versus-gene sign agreement, and the forward-pass
identities. `units.csv` carries the per-unit contrast behind every row.

### `summary.json`

**Title.** The decision, machine-readable.

**Caption.** The arms, what the annotation is worth per scheme (permuted minus
real validation loss), and the four forward-pass identities with the number of
arms each applies to. One identity is worth reading twice: with the encoder's
covariate route closed, a counterfactual at fixed latent state and a re-encoded
contrast become **exactly** the same computation, so the convention stops
changing the answer. In the shipped decoder it silently determined it.
