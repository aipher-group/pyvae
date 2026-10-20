# Kang PBMC: what the pathway units say, and what they used to say

Kang et al. (2018) is PBMCs, interferon-β stimulated against control. The
answer is fixed before anything runs — `REACTOME_INTERFERON_ALPHA_BETA_SIGNALING`
— which is what makes this dataset the external check on an architecture
decision taken where no ground truth exists.

Every number below is re-derived from `outputs/` by `verify_findings.py`. If
this document and the run disagree, that script fails.

**One-line summary.** The read-out this experiment used to report was
annotation-blind: it ranks interferon first even when the gene→pathway map is
shuffled. Ranking the model's own units instead, the shipped architecture put
interferon **last of 1,615**. Three flags fix that and cost 20.7 nats of
validation loss, and the justification is not fit — it is that the real
annotation then beats a size-matched shuffle by **+14.5 nats** where the
standard conditional decoder beats it by +2.2. Of that +14.5, **78% comes from
the mask on the decoder**: with the encoder masked and the decoder dense, which
is how this line of work was built, the annotation is worth +3.2.

---

## Results 1 — the architecture

`01_architecture.py`, one seed, eight fits: {real, permuted adjacency} ×
{shipped, pathway, uncond, signed}. 24,673 cells, 7,573 genes in the map, 1,615
units, contrast evaluated on the 12,315 control cells.

| arm | val loss | **unit rank** | parent | gene rank (z) | dec_out −ve | sign agree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_shipped` | 1309.12 | **1615** | 1611 | 1 (9.14) | 0.495 | 0.515 |
| `perm_shipped` | 1311.27 | 1130 | 945 | 1 (9.18) | 0.525 | 0.507 |
| `real_pathway` | 1315.87 | 218 | **1** | 1 (9.02) | 0.489 | 0.513 |
| `perm_pathway` | 1322.97 | 1350 | 876 | 1 (8.92) | 0.501 | 0.502 |
| `real_uncond` | 1318.70 | **1** | 3 | 1 (9.37) | 0.494 | 0.494 |
| `perm_uncond` | 1326.17 | 1271 | 601 | 1 (8.74) | 0.481 | 0.487 |
| **`real_signed`** | 1329.86 | **25** | **3** | 1 (10.36) | **0.000** | **1.0000** |
| `perm_signed` | 1344.38 | 1430 | 552 | 1 (8.22) | 0.000 | 1.0000 |

The three flags, each adding to the one before:

| flag | what it does |
| --- | --- |
| `cov_mode="pathway"` | the covariate's effect becomes a vector in pathway coordinates, reaching a gene only through the mask |
| `cov_in_encoder=False` | and it is the *whole* conditional effect, because the covariate cannot also reach `z` |
| `nonneg_decoder=True` | and a unit rising raises its genes, so a pathway's sign means what it says |

### R1.1 — the gene-level read-out never saw the annotation

**Rank 1 in all eight arms**, including every arm whose gene→pathway map is
shuffled, at z 8.22–10.36. That is not a model result. The shipped decoder
computes

```
logits = dec_out(tanh(dec_latent(z))) + cov_decoder(cov)
```

with `dec_out` masked by Reactome and `cov_decoder` a dense `(n_genes, n_cov)`
term added **after** it. `predict_counterfactual` encodes under one covariate
and decodes the same `mu` under the other, so the masked term is identical on
both sides and cancels: the contrast is exactly
`cov_decoder[:, b] − cov_decoder[:, a]`, measured against the decoder's own
forward pass at **1.03e-06**. The annotation is nowhere in it, and a gene-set
test fed those shifts is a gene-list enrichment test with a model-shaped
denoiser in front of it.

### R1.2 — and the unit ranking was worse than chance

Under `real_shipped` the target sits at **1,615 of 1,615** — dead last — while
under `perm_shipped` it sits at 1,130. **Shuffling the annotation improved the
ranking.** There is no reading of that under which the shipped units carried
the biology.

### R1.3 — what the annotation is worth

Permuted minus real validation loss, per scheme. The permutation relabels genes,
so every unit keeps its size, its name, the degree multiset and the entire
pathway-by-pathway overlap matrix; the two arms differ in nothing but which
named genes share a unit.

| scheme | annotation worth (nats) |
| --- | ---: |
| shipped | +2.15 |
| pathway | +7.10 |
| uncond | +7.47 |
| **signed** | **+14.52** |

### R1.4 — the cost, stated first

`real_signed` fits **worse** than `real_shipped` by **20.74 nats** (1329.86
against 1309.12). A dense per-gene matrix describes one large diffuse shift with fewer parameters than 1,615 annotated units can. The
case for the architecture is R1.3 and R3, not fit.

There is a second cost, and it is not a likelihood. **The architecture saturates
far harder than the one it replaces**, measured on the same cells by the same
code:

| fit | units saturated in >half their cells | median \|`delta_post`\| |
| --- | ---: | ---: |
| `real_shipped` | 23% | 2.7e-01 |
| **`real_signed`** | **71%** | **5.4e-02** |

On `Zavidij_2020` the same comparison is 22% → 80% and 6.1e-02 → 2.0e-03. A
unit pinned in the flat tail of `tanh` writes nothing to its genes whatever its
weights say, so `member_shift_z` is in effect a ranking over the units that
still have headroom: **a unit can be biologically involved and invisible here
because it is pinned.** Every fit's `summary.json` carries the count, so the
caveat travels with the numbers rather than living only in this paragraph.

What is *not* a cost is a unit writing exactly nothing, which the count reports
as 0. Hundreds of units used to land there, and it was arithmetic rather than
biology: `tanh(pre + d) − tanh(pre)` is a catastrophic cancellation and rounds
to exactly zero in float32 once |pre| passes about 4. `pyvae.unit_contrast` now
widens before the subtraction, and the regression test for it is
`tests/unit/test_cov_conditioning_modes.py::TestSaturationDoesNotDestroyTheSign`.

### R1.5 — the nesting shows in the ranking, and should

Under `real_signed` the named leaf `INTERFERON_ALPHA_BETA_SIGNALING` is
**25th**, its parent `INTERFERON_SIGNALING` is **3rd**, and its grandparent
`CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM` is **2nd**. Those three sets nest
strictly, so this is **one finding reported at three resolutions**, with the
model preferring the broader resolution once units are size-corrected.

Reading the top 25 as families rather than as rows, the interferon response
occupies nine of them:

| rank | unit | genes |
| ---: | --- | ---: |
| 2 | `CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM` | 605 |
| 3 | `INTERFERON_SIGNALING` | 177 |
| 4 | `OAS_ANTIVIRAL_RESPONSE` | 9 |
| 5 | `TRAFFICKING_AND_PROCESSING_OF_ENDOSOMAL_TLR` | 13 |
| 11 | `TRAF3_DEPENDENT_IRF_ACTIVATION_PATHWAY` | 15 |
| 15 | `STING_MEDIATED_INDUCTION_OF_HOST_IMMUNE_RESPONSES` | 11 |
| 20 | `ZBP1_DAI_MEDIATED_INDUCTION_OF_TYPE_I_IFNS` | 20 |
| 23 | `INFLAMMASOMES` | 21 |
| 25 | `INTERFERON_ALPHA_BETA_SIGNALING` | 57 |

Type-I interferon signalling, the three nucleic-acid sensing arms that induce
it (endosomal TLR, STING, ZBP1), its antiviral effector arm (OAS) and the IRF
activation step that links them. None of that was used in fitting, and none of
it is a separate finding from the others — which is exactly why no count over
these rows is reported.

The strongest single unit is `SIGNALING_BY_GPCR` (385 genes), which is not
interferon biology and is not claimed as any. It is the kind of large, diffuse
unit a size-corrected statistic favours, and it is reported here rather than
cropped out of the figure.

`real_uncond` puts the leaf 1st instead. So adding `nonneg_decoder` moves the
specific answer from 1 to 25 while moving its family to the very top. We report
both rather than selecting the more favourable one: the leaf's rank is better without
the sign constraint, and the sign constraint is what makes any of these numbers
readable as a direction. That trade-off is the decision.

### R1.6 — ranking on the weights alone is worse than ranking on what is written

`pathway_contrast` gives a cell-independent per-unit number, and it is tempting
to rank on it directly. Measured, that is the weaker statistic: the target sits
at **1,599** under `uncond` and **284** under `signed` by the raw weight, against
**1** and **25** by `member_shift_z`, which is `delta_post · w_sum / √n` — what reaches the genes after `tanh` and the decoder weights. The
`1/√n` corrects a size tilt that otherwise hands the top to the smallest units
(Spearman between |statistic| and set size, −0.49 → −0.09).

---


---

## Results 1b — do the options act 1 holds fixed matter?

`01b_encoder_ablation.py`, 18 fits. Act 1 varies the conditioning path and pins
four options in `BASE`, so nothing in act 1 is evidence for them. This grid pins
the conditioning at the selected scheme and moves one `BASE` option at a time,
each arm real and permuted. Both scripts share act 1's data loader, so the
`selected` arm here **is** act 1's `real_signed`: val 1329.86 and unit rank 25
in both, which the verifier checks rather than assumes.

| option | fit | vs selected | annot worth | leaf | parent | pinned | enc −ve | perm leaf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`selected`** | 1329.9 | +0.00 | **+14.52** | 25 | 3 | 71% | 0.000 | 1430 |
| `normalize_none` | 1331.4 | +1.54 | +15.84 | 10 | 4 | 72% | 0.000 | 1385 |
| `normalize_layer` | 1331.0 | +1.16 | +13.47 | 22 | 2 | 74% | 0.000 | 1403 |
| `init_fan_in` | **1320.9** | **−8.96** | +8.07 | **1** | 1196 | **43%** | 0.000 | **450** |
| `free_encoder` | 1330.6 | +0.73 | +14.54 | 29 | 1 | 73% | 0.501 | 1489 |
| `unscaled_input` | 1329.0 | −0.91 | +13.87 | 24 | 2 | 71% | 0.000 | 1461 |
| `free_unscaled` | 1329.8 | −0.06 | +13.96 | 25 | 1 | 72% | 0.500 | 1502 |
| `dense_decoder` | 1330.3 | +0.42 | **+3.24** | — | — | — | — | — |
| `dense_encoder` | 1358.1 | **+28.26** | +13.31 | 19 | 3 | 69% | 0.500 | 1204 |

### R1b.1 — the decoder mask carries most of what the annotation is worth

`dense_decoder` removes the mask from `dec_out` and leaves the encoder mask in
place, which is the configuration this line of work used for years. Annotation
worth falls from **+14.52 to +3.24 nats**: the encoder mask alone captures 22%
of it, and the decoder mask supplies the other **78%**.

It costs **+0.42 nats** of fit, so the mask adds almost no loss. The 20.74 nats the
architecture costs (R1.4) is the conditioning path, not the mask.

That arm is scored on fit and annotation worth alone. With no mask on `dec_out`
every gene is a member of every unit, so `member_shift_z` has no member set to
sum over and the unit read-out is undefined rather than merely different.

`dense_encoder` is the same question asked from the other side, and it was added
after a reader asked what the encoder restriction is for once the biology rests
on the decoder. It drops the encoder mask and keeps the decoder's, so unlike
`dense_decoder` it is still scored over units. Annotation worth lands at
**+13.31**, a fall of 1.21 nats against the decoder mask's 11.28, and the two
arms sit at 69% and 71% pinned, so the R1c curve accounts for only 0.31 nats of that
gap. The target leaf even improves, from 25 to 19.

What the encoder mask contributes is fit. The dense arm is **28.26 nats worse** on
held-out loss while carrying 12,230,395 first-layer weights against the mask's
66,809, which is 0.55% of them. So the annotation in the encoder works as a
regulariser and the annotation in the decoder is what the contrast reads.

Keep the encoder mask anyway, for something this table cannot show.
`integrated_gradients` attributes the encoder's own unit, so under the mask a
gene outside the set has exactly zero attribution, which is what makes R2's
per-gene panel a statement about a pathway. Under a dense encoder it is a
ranking over all 7,573 genes. Both behaviours are asserted in
`tests/unit/test_informed_encoder_decoder_wiring.py` rather than described.

The flag that makes this arm possible, `informed_encoder`, is new. The decoder's
mask has had an ablation flag since Phase 1c and the encoder's did not, so until
now "what is the encoder mask worth" had no measurement in this line of work.

### R1b.2 — the other three options are immaterial

Annotation worth spans **+13.47 to +15.84** across normalisation and the two
encoder constraints, against the **+2.15 to +14.52** the conditioning path moves
it in act 1. The leaf stays between 10 and 29, every permuted arm leaves it past
1,385, and saturation sits at 71% to 74%.

Two specific claims do not survive. `normalize="batch"` was chosen to cut
saturation and does not: 71% against 72% with no normalisation at all. And the
2×2 that `04_encoder_fidelity_experiment.py` was built for is flat:

| | `standardize_input` on | off |
| --- | ---: | ---: |
| `nonneg_encoder` on | +14.52 (rank 25) | +13.87 (rank 24) |
| off | +14.54 (rank 29) | +13.96 (rank 25) |

No main effect from either constraint, no interaction, and a range of 0.67 nats
across four cells. The failure in which one high-variance gene dominates its
unit, which those options were added to prevent does not appear in anything this
read-out measures.

`nonneg_encoder` is retained for a narrower reason than R3 gives. With a
freely signed encoder, `encoder_frac_negative` is 0.501 and unit-versus-gene
sign agreement is still **1.0000**, so the pathway-level sign comes from
`nonneg_decoder` alone. What encoder non-negativity provides is a unit that is not
an upside-down representation of its own pathway, which matters for reading a
unit's activation and not for reading a contrast.

### R1b.3 — fan-in initialisation, and what it revealed about the measure

`init_fan_in` is best on every conventional metric. It fits **8.96 nats
better**, ranks the target leaf **1st** rather than 25th, and cuts saturation
from 71% to **43%**, which is the largest move against that cost anywhere in
this project.

Annotation worth nonetheless falls to **+8.07**. Fan-in is also this grid's only
low-saturation arm, so the two readings are collinear here and cannot be
separated: either the annotation matters less under fan-in, or annotation worth
tracks saturation. `01c_saturation.py` separates them, and the second is
correct. Lifted to 65% pinned, fan-in scores **+13.95** against `selected`'s
+14.52 at 71%, and all three fan-in arms lie on the worth-against-saturation
curve to within 1.7 nats. See Results 1c.

So fan-in is not rejected. `init="xavier"` is kept only because changing it
would mean refitting every model in the project, and the evidence now favours
fan-in: equivalent annotation worth at matched saturation, 8.96 nats better fit,
and 43% of units pinned rather than 71%.

---

## Results 1c — annotation worth is partly a measure of saturation

`01c_saturation.py`, 14 fits. Scaling `dec_latent` and `cov_path` at
initialisation scales the decoder pre-activation, and so moves how far units sit
into the flat tail of `tanh`, while leaving the adjacency and `dec_out`
untouched. Only scaling up holds at convergence: a probe at scale 0.25 landed at
the same 71% pinned as scale 1, because training restores whatever magnitude the
likelihood wants. Fan-in is therefore raised toward `selected` rather than the
reverse.

| arm | pinned | fit | annot worth | leaf |
| --- | ---: | ---: | ---: | ---: |
| xavier, scale 1 | 71% | 1329.9 | +14.52 | 25 |
| xavier, scale 2 | 73% | 1330.4 | +13.21 | 28 |
| xavier, scale 4 | 82% | 1332.6 | +13.21 | 10 |
| xavier, scale 8 | 93% | 1340.0 | +17.96 | 1 |
| fan-in, scale 1 | 43% | 1320.9 | +8.07 | 1 |
| fan-in, scale 2 | 47% | 1322.7 | +9.12 | 1 |
| **fan-in, scale 4** | **65%** | 1326.7 | **+13.95** | 240 |

Worth rises at **+1.72 nats per 10 points** of pinned fraction (r = +0.75 over
the four xavier arms). Fan-in's three arms sit on that curve, residuals **−0.36,
−0.01 and +1.70** against the 6.45-nat deficit R1b.3 had to explain.

**Two consequences.** Annotation worth is comparable between arms at similar
saturation and inflated across a large saturation change, so R1.3's +2.15 to
+14.52 spans 23% to 71% pinned and is confounded to that extent.
`experiments/multiple_mieloma`'s `Foster_2026` is the one architecture
comparison in the project where saturation does not move, at 76% against 75%,
and its +1.30 to +7.98 is the cleanest form of the result.

And the rank statistic is no better: the target climbs from 25 at 71% pinned to
1 at 93%, because the ranking runs over the units with headroom. Leaf rank 1 is
reachable at 43% and at 93% by opposite mechanisms, so neither metric R1b could
have selected on is saturation-free.

## Results 2 — the four forward-pass identities

Every read-out here claims to describe what the model computes. These check it
against the forward pass rather than asserting it, at float32 precision.

| identity | arms it applies to | worst deviation |
| --- | ---: | ---: |
| masked + bypass == the re-encoded shift | 8 | 1.04e-06 |
| `predict_counterfactual` == the dense term | 2 | 1.03e-06 |
| **`predict_counterfactual` == the re-encoded shift** | 4 | **0.00e+00** |
| `pathway_contrast` == the forward pass | 4 | 8.29e-07 |

The third is worth reading twice. In the shipped architecture, *which
counterfactual convention you pick changes the answer*: holding `mu` fixed gives
you the dense term and nothing else, while re-encoding gives you something else
again. With `cov_in_encoder=False` the two become **bit-identical**, because
`mu` does not depend on the covariate. The architecture removes a
choice that was previously determining the result without anyone choosing it.

---

## Results 3 — sign, which is why `nonneg_decoder` is not optional

Under the shipped decoder, `dec_out` is **49.5% negative**. A unit rising
therefore pushes roughly half its member genes up and half down, so "this
pathway went up" does not say which way its genes moved. Unit-versus-gene sign
agreement is **0.515** — chance.

Under `nonneg_decoder=True`, `dec_out` is reparameterised as `softplus(θ)`:
**0.000** negative live weights, sign agreement **1.0000** exactly. The obvious
simple alternative — flip each unit post hoc by its weight sum — needs most units
to be nearly all one sign, and they are not: the per-unit positive fraction has
median exactly 0.500.

This does **not** mean the model cannot lower a gene. `h' = tanh(pre)` is
signed, so a unit falling lowers every member. Direction is carried by the
unit's activity rather than scattered across its genes, which is the whole
point.

### R3.1 — nine units that can never answer

Reactome has 1,615 sets; this dataset measures 7,573 genes; **9 of those sets
share no measured gene with it at all**. Those units exist in the model and the
latent layer writes to their pre-activation, but `dec_out` has no live weight
for them, so whatever they do reaches nothing.

They are reported, not dropped, and not folded into the sign statistic — a unit
with no genes has no member direction to agree or disagree with, so counting it
scores a guaranteed mismatch for a question never asked. An earlier version of
this experiment did count them, which held sign agreement at 0.9944 on a fit
where every unit that writes anything agrees perfectly.

They are also a different failure from saturation, and the two are counted
separately: an empty unit never had a direction, a saturated one has one and
cannot express it.

---

## Results 4 — the read-out, under the selected architecture

`02_kang_units.ipynb`, one fit, control → stimulated read straight off
`InformedVAE.pathway_contrast`. The numbers in `outputs/units/summary.json`;
the four checks that gate them all pass, `pathway_contrast` agreeing with the
forward pass at float32 precision and the shift varying across cells by the
same order, which is what "cell-independent" means operationally.

Two routes agree, and that is the check worth taking from this fit. The
covariate term pushes the interferon unit up by a fixed amount for every cell.
Separately, with **every cell held at `control` in the covariate**, the encoder
still places the stimulated cells higher on that unit — because their
transcriptomes say so, with nothing told to the model. Figure 2 shows the
second; the title carries the first.

The response is strongest in CD14+ monocytes, FCGR3A+ monocytes and dendritic
cells and weakest in megakaryocytes, which is the expected myeloid-dominant
type-I interferon pattern and was not used in fitting.

Integrated gradients inside the interferon unit names **ISG20, IFI6, IFIT1,
HLA-B, ISG15, IFIT3, MX1, IFIT2, HLA-C and SOCS1**, carrying 75% of the unit's
total attribution between them — canonical interferon-stimulated genes plus two
class-I MHC genes, which are themselves interferon-inducible. This is a
within-unit question (what the unit reads the pathway *through*) and not a
ranking of units, and the decoder side names different genes, as it does in
every case this project has measured.

The covariate adds **+2.60** to that unit's pre-activation, the same for every
cell. Under the permuted annotation the same unit sits at **1,430 of 1,615**.

---

## Methods, in brief

- **Model.** `InformedVAE`, NB likelihood, Reactome mask on the encoder's first
  layer and on `dec_out`, `nonneg_encoder`, batch normalisation, latent width
  half the unit count, 250 epochs with cosine annealing and patience pinned to
  the epoch budget.
- **Design.** Condition × cell type, one-hot, 10 columns.
- **Contrast.** `cov_path.weight @ (cov_stim − cov_ctrl)`. No cell re-encoded,
  nothing sampled, no counterfactual convention to choose.
- **Null.** `permute_adjacency` relabels gene rows. It preserves every set size,
  the degree multiset and the **whole** pathway-by-pathway overlap matrix, and
  destroys only which named genes share a unit. Earlier descriptions of this
  null in the project claimed it broke co-membership; it does not — rows travel
  as units — and the null is tighter than that claim implied.

## What this does not establish

- **That this architecture fits better.** It does not; see R1.4.
- **That the leaf is the right resolution.** The model prefers the family; see
  R1.5.
- **That any of this transfers to a dataset without a known answer.** That is
  what `experiments/multiple_mieloma` is for, and this experiment exists to say
  whether its decision was made for the right reason.
- **A sampling-unit claim of any kind.** Kang is one experiment with one
  stimulation; there is no donor-level test here, and `member_shift_z` carries
  no variance estimate.

## What was retired, and why

A 28-run grid — `02_architecture_ablation.py`, `03_sample_size_experiment.py`,
`04_encoder_fidelity_experiment.py` — over eight configurations, two epoch
budgets and two panel widths, plus two notebooks reporting the gene-level test
as the result.

- **Kept, as an `assert`:** zero inverted encoder units. `nonneg_encoder` makes
  inversion unrepresentable rather than unlikely, so it is asserted and not
  scored.
- **Retired:** `pathway_unit_fidelity` medians measure distance from
  initialisation — an untrained non-negative model scores ≈0.75, because
  `softplus` of random weights is positive and roughly uniform, which already
  *is* the unweighted member mean it correlates against, and the score *falls*
  with training.
- **Retired:** the sample-size experiment's model-against-naive advantage is
  reproduced by a permuted annotation, so it measured masking and not biology.

What that grid never had is the control that decides the question: an arm where
the annotation itself is shuffled.
