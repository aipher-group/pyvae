# Model-driven biology across the myeloma precursor sequence

What a pathway-informed VAE says changes in bone-marrow T cells between
non-cancer, MGUS, smouldering myeloma and symptomatic myeloma — asked of two
independent datasets, fitted separately under one audited design, and read over
the model's **own pathway units**.

Every number below is re-derived from `outputs/` by `verify_findings.py`, which
reports **ALL PASS (76 checks)**. If this document and the run disagree, that
script fails.

---

## One-line summary

The read-out every previous version of this experiment reported never touched
the gene→pathway annotation: a **shuffled** map reproduced it at Spearman +0.97.
Three flags put the annotation back in the contrast path, at a cost of 2.7 to
10.1 nats of validation loss. Under them, **`interferon alpha beta signaling` is
unit 1 of 1,615 in both cohorts, fitted independently** — while the panel-wide
rankings agree at ρ = −0.07. One shared unit, and nothing else. With donors as
the sampling unit it clears correction in `Foster_2026` and does not in
`Zavidij_2020`, which is the **opposite** of what the annotation-blind read-out
concluded.

---

## Act 1 — which decoder uses the annotation

`01_architecture.py`. Eight fits per run: {real, permuted adjacency} ×
{shipped, pathway, uncond, signed}. Four seeds on `Zavidij_2020`, one on
`Foster_2026`. The target is `REACTOME_INTERFERON_ALPHA_BETA_SIGNALING` — **not
a known answer**, because myeloma has none, but a fixed unit the retired
gene-level baseline points at, so the architectures can be compared on one
thing. `experiments/kang` is the same comparison where the answer *is* known.

### R1.1 — the shipped decoder never passed the covariate through the mask

```
logits = dec_out(tanh(dec_latent(z))) + cov_decoder(cov)
```

`dec_out` is masked by Reactome; `cov_decoder` is a dense `(n_genes, n_cov)`
term added **after** it. `predict_counterfactual` encodes under one covariate
and decodes the same `mu` under the other, so the masked term is identical on
both sides and cancels. Every stage contrast this experiment ever reported was
exactly `cov_decoder[:, b] − cov_decoder[:, a]`, and `masked_share` — the
fraction of the conditional effect expressed in the annotation basis at all —
is **0.061 to 0.089**.

### R1.2 — the architectures, scored on the same contrast

Canonical seed, `Non→MM`, CD4 and CD8.

| | `Zavidij_2020` val | unit rank | `Foster_2026` val | unit rank |
| --- | ---: | ---: | ---: | ---: |
| `real_shipped` | 1484.20 | 1614, 1615 | 2048.60 | 144, 102 |
| `perm_shipped` | 1484.84 | 781, 791 | 2049.90 | 560, 563 |
| `real_pathway` | 1478.35 | 5, 7 | 2052.67 | 3, 4 |
| `perm_pathway` | 1481.70 | 958, 921 | 2055.37 | 902, 923 |
| `real_uncond` | 1479.80 | 2, 5 | 2054.70 | 2, 4 |
| `perm_uncond` | 1482.43 | 1009, 978 | 2057.94 | 910, 848 |
| **`real_signed`** | 1486.89 | **2, 1** | 2058.65 | **4, 3** |
| `perm_signed` | 1493.21 | 845, 537 | 2066.63 | 985, 668 |

**On `Zavidij_2020` the shipped decoder ranks the target dead last — and
shuffling the annotation *improves* it**, from 1614/1615 to 781/791. There is no
reading of that under which the shipped units carried the biology.

**`Foster_2026` does not do this.** There the shipped decoder puts the target at
144 and 102 against the shuffle's 560 and 563, so it is poor rather than
inverted. The claim is made about `Zavidij_2020` (and Kang, where it is 1,615 of
1,615) and not about both cohorts.

### R1.3 — what the annotation is worth

Permuted minus real validation loss. The permutation relabels gene rows, so
every unit keeps its size, its name, the degree multiset and the **entire**
pathway-by-pathway overlap matrix; the two arms differ in nothing but which
named genes share a unit.

| scheme | `Zavidij_2020` | `Foster_2026` |
| --- | ---: | ---: |
| shipped | +0.64 | +1.30 |
| pathway | +3.35 | +2.70 |
| uncond | +2.63 | +3.24 |
| **signed** | **+6.32** | **+7.98** |

### R1.4 — the cost, stated first

`real_signed` fits **worse** than `real_shipped` by **2.69 nats** on
`Zavidij_2020` and **10.05** on `Foster_2026`. A dense per-gene matrix describes
one large diffuse shift with fewer parameters than 1,615 annotated units can. The case
for the architecture is R1.3 and R3, not fit.

### R1.5 — stability across seeds, and what is not stable

Four seeds on `Zavidij_2020`. The real arm is stable and the permuted arm is
not:

| seed | `real_signed` | `perm_signed` | annotation worth |
| ---: | ---: | ---: | ---: |
| 42 | 2, 1 | 845, 537 | +6.32 |
| 1 | 2, 3 | 1506, 1473 | +6.48 |
| 2 | 3, 2 | 486, 523 | +4.22 |
| 3 | *see `outputs/architecture/Zavidij_2020/seed3/`* | | |

The permuted rank ranges over roughly 500 to 1,500 for the same arm. **No single
permuted rank should be quoted as "the" number**; what is stable is that the
real arm sits in the top three every time and the permuted arm never comes near
it. That is why the verifier gates on "> 300" rather than on a point value.

### R1.6 — ranking on the weights alone is the weaker statistic

`pathway_contrast` gives a cell-independent per-unit number and it is tempting
to rank on it directly. Measured, it is worse: the target sits at 56 and 19
(`Zavidij_2020`) and 640 and 753 (`Foster_2026`) by the raw weight, against
2 and 1, and 4 and 3, by `member_shift_z` — which is `delta_post · w_sum / √n`,
what reaches the genes after `tanh` and the decoder weights. The `1/√n`
corrects a size tilt that otherwise hands the top to the smallest units
(Spearman between |statistic| and set size, −0.49 → −0.09).

---

## Act 1b — the four forward-pass identities

Every read-out here claims to describe what the model computes. These check it
against the forward pass rather than asserting it, at float32 precision.

| identity | arm-lineages | `Zavidij_2020` | `Foster_2026` |
| --- | ---: | ---: | ---: |
| masked + bypass == re-encoded shift | 16 | 6.6e-07 | 1.5e-06 |
| `predict_counterfactual` == dense term | 4 | 6.2e-07 | 1.5e-06 |
| **`predict_counterfactual` == re-encoded shift** | 8 | **0.00e+00** | **0.00e+00** |
| `pathway_contrast` == forward pass | 8 | 3.6e-07 | 1.1e-06 |

The third is worth reading twice. In the shipped architecture *which
counterfactual convention you pick changes the answer*: holding `mu` fixed gives
the dense term and nothing else. With `cov_in_encoder=False` the two become
**bit-identical**, because `mu` does not depend on the covariate. The
architecture removes a choice that was previously determining the result without
anyone choosing it.

---

## Act 1c — sign, and why `nonneg_decoder` is not optional

Under the shipped decoder `dec_out` is **50.3%** negative (`Zavidij_2020`) and
**50.6%** (`Foster_2026`). A unit rising pushes about half its member genes up
and half down, so "this pathway went up" does not say which way its genes moved.
Unit-versus-gene sign agreement is **0.497** and **0.505** — chance.

The simple alternative, flipping each unit post hoc by its weight sum, needs most
units to be nearly all one sign. They are not: the per-unit positive fraction has
**median exactly 0.500**, only **20.2%** of units are ≥90% one sign, and **22.4%**
sit between 0.4 and 0.6 where no orientation exists at all.

Under `nonneg_decoder=True`: **0.000** negative live weights, sign agreement
**1.0000** exactly, in both cohorts and both lineages. This does not mean the
model cannot lower a gene — `h' = tanh(pre)` is signed, so a unit falling lowers
every member. Direction is carried by the unit rather than scattered across its
genes.

### R1c.1 — saturation, the second cost

| | units pinned, shipped | selected | median \|`delta_post`\|, shipped → selected |
| --- | ---: | ---: | --- |
| `Zavidij_2020` | 24% | **80%** | 5.0e-02 → 2.2e-03 |
| `Foster_2026` | **76%** | 75% | 3.3e-01 → 1.5e-01 |

"Pinned" is `|tanh(pre)| > 0.99` in more than half that unit's cells. The two
cohorts behave differently and the difference is not noise: at 125,602 cells the
**shipped** decoder already saturates three quarters of its units, so the
architecture adds nothing there, while at 9,863 cells it goes from a quarter to
four fifths. What holds in both is the consequence: **most units end up pinned,
a pinned unit writes nothing whatever its weights say, and the ranking is
therefore over the units that still have headroom.** A unit can be biologically
involved and invisible here.

What is *not* a cost is a unit writing exactly nothing, which both fits report
as **0**. Hundreds of units used to land there and it was arithmetic:
`tanh(pre + d) − tanh(pre)` is a catastrophic cancellation and rounds to zero in
float32 once |pre| passes about 4. `pyvae.unit_contrast` now widens before the
subtraction; the regression test is
`tests/unit/test_cov_conditioning_modes.py::TestSaturationDoesNotDestroyTheSign`.
On these 250-epoch fits the fix changed no reported number — the canonical
`Zavidij_2020` summary is identical before and after it, to within 1e-7 on the
identity residuals — so we report no improvement.

---

## Act 2 — the biology, over the model's own units

`02_biology.ipynb`, one fit per cohort under the selected architecture. Both
fits enter the same two covariate blocks (`stage` = cohort × lineage, and
`pheno_h`), **neither drops anything**, and both give 36 covariate columns over
1,615 units and 9,938 genes. Every unit has at least 2 measured member genes, so
unlike Kang there are no member-less units here.

| | `Zavidij_2020` | `Foster_2026` |
| --- | ---: | ---: |
| cells | 9,863 | 125,602 |
| donors (Non/MGUS/SMM/MM) | 30 (7/5/11/7) | 22 (3/1/13/5) |
| thinnest lineage × stage arm | 205 cells | 2,810 cells |
| validation loss | 1486.82 | 2058.69 |
| units pinned | 1,201 (74%) | 1,259 (78%) |
| **leading unit, `Non→MM`** | **interferon alpha beta signaling** | **interferon alpha beta signaling** |

### R2.1 — one unit replicates, and the panel does not

`interferon alpha beta signaling` is **rank 1 of 1,615 in both cohorts**, fitted
independently, with no shared cells and no shared donors.

Everything else disagrees. The panel-wide Spearman between the two cohorts'
endpoint contrasts, over all 1,615 units, is **ρ = −0.073** — indistinguishable
from no relationship. Each cohort's other leading units sit nowhere in the
other's ranking:

| `Zavidij_2020` top 8 | rank in `Foster_2026` |
| --- | ---: |
| interferon alpha beta signaling | **1** |
| mitochondrial translation | 1596 |
| oas antiviral response | 1274 |
| cross presentation of soluble exogenous antigens | 1533 |
| clec7a dectin 1 signaling | 1499 |
| attenuation phase | 407 |
| creb3 factors activate genes | 285 |
| er to golgi anterograde transport | 1063 |

| `Foster_2026` top 8 | rank in `Zavidij_2020` |
| --- | ---: |
| interferon alpha beta signaling | **1** |
| cell surface interactions at the vascular wall | 1266 |
| slc mediated transmembrane transport | 1061 |
| fceri mediated mapk activation | 316 |
| c type lectin receptors clrs | 366 |
| binding and uptake of ligands by scavenger receptors | 725 |
| signaling by tgfb family members | 972 |
| membrane trafficking | 935 |

This is the sharpest form the cross-cohort result has taken in this project:
**one shared unit, at the top of both, with no panel-wide agreement at all.**
Two cautions travel with it. Picking the unit that agrees and reporting its rank
is a post-hoc selection, so no p-value is attached to "rank 1 in both". And
Reactome nests, so `oas antiviral response` sitting 3rd in `Zavidij_2020` is
part of the same interferon response rather than a second finding.

### R2.2 — the lineages agree, and agree best where the data are thickest

Spearman between CD4 and CD8 over all units, with a 1,000-draw bootstrap:

| contrast | `Zavidij_2020` | `Foster_2026` |
| --- | ---: | ---: |
| `Non→MGUS` | +0.26 | +0.60 |
| `Non→SMM` | +0.31 | +0.62 |
| `Non→MM` | +0.38 | +0.66 |

CD4 and CD8 are separate columns of one design, so this is a diagnostic on the
fit and not a biological claim. It tracks cell count exactly as a noise-
attenuation account predicts: the cohort with 13× the cells agrees roughly twice
as well, and within each cohort agreement rises with the thickness of the arms.

### R2.3 — the encoder and the decoder name different genes

For the four leading units in each cohort, the gene the encoder is most
sensitive to and the gene carrying the largest share of the decoder's written
response are the same gene in **1 of 4** (`Zavidij_2020`) and **0 of 4**
(`Foster_2026`). A single "driver gene" column would have been wrong seven times
out of eight. This is a methodological result, not a biological one, and it is
why both sides are reported.

---

## Act 3 — the same read-out, with donors as the sampling unit

`03_donor_test.py` refits the same architecture on the same cells with the stage
block interacted with **donor** rather than cohort, so each donor's score is
their own `cov_path` column minus the healthy donors' mean: 1,615 signed
numbers, read straight off the weights. Significance is a Spearman trend across
all four stages against **100,000** donor-label permutations, Benjamini–Hochberg
over the 1,262 units with ≥10 measured members.

### R3.1 — the power floor, which decides how to read a null

BH multiplies the smallest attainable p by the number of tests, so the best q a
design can produce is `N_UNITS / (N_PERM + 1)`.

| test | best attainable q | can reach 0.05? |
| --- | ---: | --- |
| trend, 100,000 draws | **0.013** | yes |
| endpoint, `Zavidij_2020` CD4 / CD8 | 1.47 / 12.0 | **no** |
| endpoint, `Foster_2026` | 45.1 | **no** |

At the previous setting of 10,000 draws the trend floor was **q = 0.126** — the
primary test could not have returned a single unit whatever the data said. That
is exactly the null-versus-impossible confusion `power_floor` exists to prevent,
and it was being reported for the endpoint test and not for the primary one. It
is now reported for both, and gated by the verifier.

So: the endpoint test is uninformative in both cohorts and is reported only
beside its floor. The trend test is informative, and the results below are
genuine negatives where they are negative.

### R3.2 — the donor-level result, which inverts the previous one

| | `Zavidij_2020` CD4 / CD8 | `Foster_2026` CD4 / CD8 |
| --- | --- | --- |
| donors scored | 27 / 17 | 22 / 22 |
| interferon ρ | +0.25 / +0.24 | **+0.78 / +0.68** |
| interferon q | 0.89 / 0.93 | **0.0084 / 0.046** |
| units at q < 0.05, both lineages | **0** | **5** |

`Foster_2026`'s five are `interferon alpha beta signaling`, `atf6 alpha
activates chaperones`, `hyaluronan metabolism`, `membrane trafficking` and
`regulation of ras by gaps`.

**This is the opposite of what the retired read-out said.** The gene-level
Hallmark version of this test found interferon strong in `Zavidij_2020`
(ρ +0.45/+0.74) and absent in `Foster_2026` (ρ −0.05/+0.10). The unit read-out —
the one that uses the annotation — reverses it.

### R3.3 — why the cohorts differ, as far as the data will say

Per-donor interferon scores, CD4, model estimator:

| stage | `Zavidij_2020` n, mean, range | `Foster_2026` n, mean, range |
| --- | --- | --- |
| `Non` | 6, 0.00, −1.74 to +1.99 | 3, 0.00, −0.17 to +0.14 |
| `MGUS` | 4, +1.31, +0.80 to +1.79 | 1, +0.26 |
| `SMM` | 10, +0.66, −1.57 to +3.08 | 13, +0.23, −0.89 to +0.61 |
| `MM` | 7, +1.41, −0.18 to +3.29 | 5, **+1.51, +0.91 to +2.27** |

Both cohorts put every disease stage above their own healthy baseline. They
differ in **baseline scatter**: `Zavidij_2020`'s six healthy donors span 3.7
units of score, `Foster_2026`'s three span 0.3. Foster's five MM donors are
disjoint from its healthy ones, which is why its AUC is 1.000 and its ρ is
+0.78; Zavidij's MM donors overlap its healthy ones, which is why its ρ is
+0.25 despite a similar mean shift.

That is a statement about estimator variance, not about biology, and the obvious
competing explanation is the obvious one: **`Foster_2026` has three healthy
donors**, and three donors can look tight by accident. Both readings are on the
table and this experiment does not separate them.

### R3.4 — the model-free cross-check, and what its disagreement means

A phenotype-standardised pseudobulk score over the same units, sharing only the
cells with the model.

- Over **all** units the two estimators agree: Spearman between their trend ρ is
  **+0.38 / +0.45** (`Zavidij_2020` CD4 / CD8) and **+0.27 / +0.29**
  (`Foster_2026`), each at p < 1e-20.
- Their **thresholded sets** are disjoint: pseudobulk returns 0 units in
  `Zavidij_2020` and 51 in `Foster_2026`, overlapping the model's five in
  **none**.

The second fact looks alarming and is mostly an artefact of the first: two
correlated but noisy rankings crossing a sharp cut-off routinely have disjoint
tails. **The correlation is the right summary and the set overlap is not.** On
interferon specifically the pseudobulk is weak in both cohorts (+0.19/+0.44 in
`Zavidij_2020`, +0.17/+0.02 in `Foster_2026`), so it does **not** corroborate
Foster's result on the unit the model singles out.

---

## What is worth defending

- **The architecture.** Three datasets, consistent direction, a size-matched
  permutation control, four forward-pass identities, and a known answer in
  `experiments/kang`. The costs are measured and stated.
- **Interferon as this compartment's signal.** Unit 1 of 1,615 in two
  independent cohorts under an annotation-dependent read-out, with the
  interferon family occupying nine of Kang's top 25 under the same architecture.
- **That the panel does not replicate.** ρ = −0.073 over 1,615 units is a result,
  and it means each cohort's response beyond interferon is its own.

## What is not worth defending

- **The donor-level direction of the interferon effect.** It survives in one
  cohort, fails in the other, inverts the previous read-out's conclusion, and
  the model-free estimator corroborates neither. Three estimators, three
  answers.
- **Anything resting on `Foster_2026`'s MGUS stage.** One donor.
- **The four other units surviving in `Foster_2026`.** No prior reason to expect
  them, no support in the other cohort, and a nested panel.
- **Any count over units.** Reactome nests; see `resources/PROVENANCE.md`.

## What would settle it

More healthy donors, in both cohorts. Every discrepancy in act 3 traces to the
baseline: `Zavidij_2020` has six healthy donors with 3.7 units of scatter and
`Foster_2026` has three with 0.3. A third cohort with ≥10 non-cancer donors
would decide whether Foster's tight baseline is real or an accident of n = 3,
and it is the only addition that would.

## What cannot be settled with these data

Whether the interferon signal is a T-cell-intrinsic programme or a response to a
marrow-wide stimulus. The design conditions on phenotype within the T-cell
compartment and never sees another compartment, so it cannot distinguish them.

---

## Decisions, dataset by dataset

**Why the atlas is not pooled.** 69% of its non-cancer donors sit in studies
containing no patients, so a pooled disease contrast would substantially be a
contrast between studies, and `study_id` cannot be adjusted for without becoming
collinear with the thing being estimated. One fit per cohort, compared
afterwards.

**Why these two and not the other nine.** They are the only two spanning all
four stages with more than one donor per arm.

**Why the donor inclusion flag is honoured.** The atlas's own `analysed` flag;
it removes 3 donors and 18,226 cells from `Foster_2026` and nothing from
`Zavidij_2020`.

**Why the phenotype block is merged.** The raw atlas phenotype cannot enter
`Foster_2026` — its `Teff` level is 95.5% MM and doubles as an MM indicator.
`PHENO_MERGE` gives 16 levels identical in both cohorts, which is the only way
the same design is identified in each.

**Why depth is a covariate and nothing else.** Patients are sequenced deeper
than controls, `depth_bin` passes the audit in both cohorts, and it is one word
in `config.yaml`. It is off because neither source paper corrects for depth and
a corrected result cannot be compared one-for-one with their claims. There is no
longer a second depth-adjusted fit: conditioning moved the previous read-out by
less than 0.02 in z, and a twin of every fit doubled the workflow's cost to
re-answer that.

**Why the UMAP carries no claim.** Cells group by donor and every donor sits at
one stage, so stage-shaped structure in that frame is donor-shaped structure.

**Why Hallmark is no longer a read-out.** It was never reading the annotation —
see R1.1 — and it survives only inside `01_architecture.py` as the evidence of
that. `resources/PROVENANCE.md` records the change and how nesting is handled
now that the read-out is Reactome itself.

---

## Reproducibility

```bash
cd experiments/multiple_mieloma
./00_setup.sh
pixi run all            # snakemake -j 63 --resources gpu=3
pixi run verify         # ALL PASS (76 checks)
```

About 3.5 hours wall on three A100 80GB cards, dominated by `Foster_2026`'s
architecture run (8 arms at ~25 minutes). `outputs/benchmarks.csv` carries the
measured resource use of every job. `experiments/kang` is the same architecture decision
on a dataset whose answer is known in advance and is the external check that
this one selected for the right reason.
