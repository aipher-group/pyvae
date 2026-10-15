# Pathway-space conditioning makes a biologically informed VAE's covariate contrast depend on its annotation

**Draft manuscript.** The pipelines in `kang/` and `multiple_mieloma/` produced
every number below, and `kang/verify_findings.py` (87 checks) and
`multiple_mieloma/verify_findings.py` (88 checks) re-derive each one from the
`outputs/` trees. Both report ALL PASS.

## Abstract

Biologically informed variational autoencoders mask the decoder with a curated
gene-set collection, so that each latent unit writes only to the genes of one
pathway. Recent work reports that randomising the annotation costs little, and
concludes that the benefit comes from the sparsity rather than from the biology.
We show that this conclusion depends on where the covariate enters the decoder.
In the standard conditional design the covariate term is dense over genes and
enters after the masked layer. A counterfactual covariate contrast then cancels
the masked term exactly and reduces to one column difference of an unannotated
matrix. We verify this as a forward-pass identity. Only 6% to 18% of the
contrast's magnitude passes through the mask, and a size-matched permuted
gene-to-pathway map costs only +0.6 to +2.2 nats of held-out loss, which
reproduces the published negative result. Three changes raise that figure to
+6.3 to +14.5 nats: the covariate enters in pathway coordinates inside the
decoder's nonlinearity, the encoder's covariate route closes, and the masked
decoder becomes non-negative. Together they make a counterfactual contrast and a
re-encoded one bit-identical, and they give every unit a direction, raising
unit-versus-gene sign agreement from 0.50 to exactly 1.000. A second ablation
separates the two masks. Masking the encoder alone, as earlier informed VAEs do,
is worth +3.2 nats, so the decoder mask supplies 78% of the total at a cost of
0.4 nats of fit. Dropping the encoder mask instead leaves worth at +13.3 and
costs 28 nats of held-out loss. A third shows that this measure rises with the fraction of
saturated units, so the cleanest form of the result is the one cohort whose
saturation does not move, where worth goes from +1.3 to +8.0 at a constant 75%
of units pinned. On
interferon-beta-stimulated peripheral blood mononuclear cells, where the
expected answer is fixed in advance, the architecture recovers the type-I
interferon response as nine of its 25 leading units across three nesting levels,
and integrated gradients concentrate 75% of the target unit's attribution on ten
canonical interferon-stimulated genes. Applied to bone-marrow T cells across the
myeloma precursor sequence in two independent cohorts, `interferon alpha beta
signaling` is unit 1 of 1,615 in both, while the panel-wide rankings agree at
rho = -0.07. With donors as the sampling unit the trend clears correction in one
cohort and not the other. We report the architecture's two costs, 2.7 to 20.7
nats of held-out loss and 71% to 80% of units pinned in the flat tail of `tanh`,
and we argue that a null permutation result says nothing until the annotation is
shown to lie in the path that carries the measured quantity.

## 1. Introduction

Masking a decoder with a curated gene-set collection is a standard way to make a
single-cell variational autoencoder interpretable. Each latent unit writes only
to the genes annotated to one pathway, so the unit inherits the pathway's name.
VEGA [1], expiMap [2] and pmVAE [3] are the canonical single-cell instances, and
the question has moved from whether the design works to what the annotation
contributes.

Recent evidence is sceptical. Caranzano et al. [4] rebuilt fifteen
pathway-informed models with randomised counterparts. The randomised versions
performed comparably across metrics and datasets, better in three of fifteen,
and offered no interpretability advantage. The authors conclude that the benefit
comes from the sparsity the annotation imposes rather than from its biological
content. A second line of work shows that biologically informed networks can
reproduce an input-output mapping while failing to recover the gene-to-pathway
weights that generated it, even with noiseless data and correct sparse wiring
[5]. A controlled single-cell comparison on interferon response found a
Hallmark-masked linear decoder at AUROC 0.997 against 0.758 for a permuted mask.
The permuted model's best latent nonetheless reached 0.989, and across ten seeds
that latent was never the interferon-named one; a decoder masked by data-derived
co-expression modules matched curation exactly at 0.998 [6]. The sparsity
carried the signal, and the gene sets determined which latent held it.

This question has a history in our own work, and the present study revisits it.
The `pyvae` codebase descends from an informed VAE in which a single masked
encoder layer carries the annotation, introduced for dimensionality reduction in
single-cell data [10] and then used to score pathway activity at cell level
[11]. That line evaluated the annotation by substituting the bottleneck layer
for gene expression in a standard single-cell workflow: neighbour graph, Leiden
clustering, UMAP, and a Wilcoxon rank-sum comparison of latent activity between
observed groups of cells. On interferon-beta-stimulated PBMC that protocol ranks
interferon pathways at the top [11]. It cannot separate two explanations,
however, because it never shuffled the annotation: the units may carry pathway
identity, or the representation may retain the dominant variance that gene
expression already shows. A later study added randomised-wiring controls at
eight connection densities over 40 seeds and found that clustering coherence
barely separates the two, at adjusted mutual information 0.62 to 0.68 for
informed models against 0.61 to 0.64 for randomised ones, while randomised
wiring achieved slightly lower reconstruction error [12]. What separated them
was the stability of the pathway ranking across seeds. We take that as the
starting point here, and ask what estimand makes the annotation measurable.

We make a narrower claim than either side of that debate. In the standard
*conditional* informed decoder, the annotation lies outside the path that
carries a covariate contrast. A permutation control applied to that contrast
must therefore return null, however informative the annotation is. We specify an
architecture that puts the annotation back into that path, measure what changes
under one permutation control on three datasets, validate on a perturbation
whose answer is known in advance, and apply the result to bone-marrow T cells
across the myeloma precursor sequence.

## 2. Materials

### 2.1 Gene set collection

We use the Reactome subset of MSigDB C2:CP, version 7.5.1, giving `P = 1,615`
sets. The binary adjacency `A` in `{0,1}^(G x P)` has `A[g, p] = 1` when gene `g`
is annotated to set `p`. We retain the collection in full, without pruning to
leaf terms or to a size window, and we handle the resulting nesting at the
inference stage (section 3.11) rather than by editing the collection.

Reactome nests strictly. `interferon alpha beta signaling` (59 genes) sits inside
`interferon signaling` (188), which sits inside `cytokine signaling in immune
system` (682). A parent and its child are therefore near-duplicate hypotheses,
which constrains what we may report over units.

For the baseline gene-level read-out of section 3.9 we additionally use the
MSigDB Hallmark collection, version 7.5.1, 50 sets.

### 2.2 Overview of the three datasets

**Table 1.** The three datasets. Genes in the map are those with at least one
Reactome annotation and non-zero expression; the model reads only these.

| | Kang 2018 | `Zavidij_2020` | `Foster_2026` |
| --- | ---: | ---: | ---: |
| tissue | PBMC | bone marrow | bone marrow |
| cells | 24,673 | 9,863 | 125,602 |
| donors | 8 | 30 | 22 |
| genes measured | 15,706 | 34,849 | 34,849 |
| genes in the map | 7,573 | 9,938 | 9,938 |
| pathway units | 1,615 | 1,615 | 1,615 |
| latent width | 807 | 807 | 807 |
| covariate columns | 10 | 36 | 36 |
| design factors | condition, cell type | stage by lineage, phenotype | stage by lineage, phenotype |
| role | known-answer validation | application | application |

### 2.3 Kang: interferon-beta stimulation of PBMC

Kang et al. (2018) assayed peripheral blood mononuclear cells from eight lupus
patients, each sample split into an interferon-beta-stimulated aliquot and an
unstimulated control. The perturbation fixes the expected answer before any
model runs, which is what makes this dataset a validation set rather than a
discovery set: a type-I interferon programme must rise, and
`REACTOME_INTERFERON_ALPHA_BETA_SIGNALING` names it.

The two conditions are balanced to within 0.2%, at 12,315 control and 12,358
stimulated cells, and the eight annotated cell types cover the expected PBMC
composition.

**Table 2.** Kang cells by annotated cell type and condition.

| cell type | control | stimulated | total |
| --- | ---: | ---: | ---: |
| CD4 T cells | 5,560 | 5,678 | 11,238 |
| CD14+ monocytes | 2,932 | 2,765 | 5,697 |
| B cells | 1,316 | 1,335 | 2,651 |
| NK cells | 855 | 861 | 1,716 |
| CD8 T cells | 811 | 810 | 1,621 |
| FCGR3A+ monocytes | 520 | 569 | 1,089 |
| Dendritic cells | 258 | 271 | 529 |
| Megakaryocytes | 63 | 69 | 132 |
| **total** | **12,315** | **12,358** | **24,673** |

Every donor contributes to both arms, so condition is estimated within donor.

**Table 3.** Kang cells by donor and condition.

| donor | control | stimulated | total |
| --- | ---: | ---: | ---: |
| patient_101 | 858 | 1,166 | 2,024 |
| patient_107 | 517 | 525 | 1,042 |
| patient_1015 | 2,738 | 2,352 | 5,090 |
| patient_1016 | 1,723 | 1,635 | 3,358 |
| patient_1039 | 461 | 641 | 1,102 |
| patient_1244 | 1,879 | 1,464 | 3,343 |
| patient_1256 | 2,108 | 2,026 | 4,134 |
| patient_1488 | 2,031 | 2,549 | 4,580 |
| **total** | **12,315** | **12,358** | **24,673** |

Median genes detected per cell is 507 in control cells (range 12 to 2,049) and
530 in stimulated cells (range 18 to 2,757), so detection depth does not
separate the two arms. The design is condition crossed with cell type, one-hot,
giving `K = 10` columns, and we evaluate the contrast on the 12,315 control
cells.

### 2.4 Myeloma: bone-marrow T cells across the precursor sequence

We draw both application cohorts from one bone-marrow T-cell atlas, taking cells
that pass three filters in order: bone-marrow tissue, non-doublet, and a donor
whose disease stage is recorded. The stage ladder has four levels: non-cancer,
monoclonal gammopathy of undetermined significance (MGUS), smouldering myeloma
(SMM) and symptomatic myeloma (MM). `Foster_2026` is the atlas's own cohort,
labelled `This_study` in the source distribution and renamed here to the
first-author convention the other studies use. We honour the atlas's donor-level
`analysed` flag, which removes 3 donors and 18,226 cells from `Foster_2026`
and nothing from `Zavidij_2020`.

We analyse these two cohorts and not the atlas's other nine because they are the
only two spanning all four stages with more than one donor per arm. We fit them
separately and never pool them, for a reason given in section 4.10.

**Table 4.** Cells, donors and detection depth by disease stage.

| stage | Zavidij donors | Zavidij cells | Zavidij median genes/cell | Foster donors | Foster cells | Foster median genes/cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Non | 7 | 1,775 | 569 | 3 | 12,929 | 732 |
| MGUS | 5 | 1,414 | 693 | 1 | 6,858 | 1,135 |
| SMM | 11 | 3,907 | 646 | 13 | 81,700 | 1,050 |
| MM | 7 | 2,767 | 675 | 5 | 24,115 | 714 |
| **total** | **30** | **9,863** | | **22** | **125,602** | |

The two cohorts differ by a factor of 13 in cells and are comparable in donors,
and the imbalance runs in opposite directions across the ladder. `Zavidij_2020`
has its largest donor group at SMM and a usable non-cancer group of seven, while
`Foster_2026` has thirteen SMM donors but only three non-cancer donors and a
single MGUS donor. Any stage-level claim about `Foster_2026`'s MGUS stage rests
on that one donor, and we do not make one.

Detection depth rises with stage in neither cohort monotonically, but patients
are sequenced deeper than controls in both. `depth_bin` passes our covariate
audit in both cohorts and remains available as a design block. We leave it out
of the reported fits, because neither source study corrects for depth and a
depth-corrected contrast cannot be compared one-for-one against their published
claims. Conditioning on it moved the unit-level read-out by less than 0.02 in z.

**Table 5.** Cells per donor by stage, median (range).

| stage | `Zavidij_2020` | `Foster_2026` |
| --- | --- | --- |
| Non | 183 (36 to 669) | 4,415 (3,429 to 5,085) |
| MGUS | 148 (56 to 915) | 6,858 (single donor) |
| SMM | 301 (70 to 1,093) | 6,289 (1,345 to 11,202) |
| MM | 388 (91 to 682) | 5,122 (2,029 to 8,025) |

**Table 6.** Donor demographics by stage. Sex and age are not recorded for
`Zavidij_2020`'s non-cancer donors in the atlas metadata.

| stage | Zavidij donors | Zavidij female | Zavidij age, median (range) | Foster donors | Foster female | Foster age, median (range) |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Non | 7 | not recorded | not recorded | 3 | 2 | 77 (61 to 87) |
| MGUS | 5 | 3 | 62 (49 to 81) | 1 | 1 | 79 |
| SMM | 11 | 5 | 63 (48 to 81) | 13 | 4 | 65 (32 to 78) |
| MM | 7 | 2 | 68 (58 to 72) | 5 | 3 | 60 (53 to 74) |

### 2.5 T-cell composition by stage

We fit the whole T-cell compartment in one model and read the stage contrast per
lineage from that single fit, rather than training separate models per lineage.
Fitting jointly gives the model two to four times the cells and lets one design
carry both lineages. CD4 and CD8 dominate both cohorts; the three minor subsets
are retained in the fit and are not read out separately.

**Table 7.** Cells by T-cell subset and stage. ISG is the
interferon-stimulated cluster and Prolif. the proliferating cluster, both
lineage-agnostic in the atlas's annotation.

| stage | Zav CD4 | Zav CD8 | Zav MAIT/gdT | Zav ISG | Zav Prolif. | Fos CD4 | Fos CD8 | Fos MAIT/gdT | Fos ISG | Fos Prolif. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Non | 1,056 | 657 | 46 | 7 | 9 | 5,956 | 6,675 | 205 | 30 | 63 |
| MGUS | 1,176 | 205 | 24 | 6 | 3 | 3,722 | 2,810 | 24 | 33 | 269 |
| SMM | 2,779 | 985 | 72 | 62 | 9 | 41,328 | 37,866 | 1,759 | 443 | 304 |
| MM | 1,705 | 906 | 70 | 69 | 17 | 12,819 | 8,527 | 459 | 2,080 | 230 |

The thinnest lineage-by-stage arm is 205 cells (`Zavidij_2020`, CD8 at MGUS) and
2,810 cells (`Foster_2026`, CD8 at MGUS). Every estimate in the per-lineage
ladder is limited by those arms, which is why Table 7 precedes the results
rather than following them.

The phenotype annotation is finer, and we condition on it at cell level so that
a stage contrast is not a composition shift in disguise. The atlas's labels do
not transfer between cohorts unaltered: in `Foster_2026` the `Teff` level is
95.5% MM cells and so doubles as an MM indicator, and `CD8+Tex` is absent from
`Zavidij_2020` entirely. We therefore merge `Teff`, `IFN-stim.` and
`Proliferating` into one lineage-agnostic level and fold `CD8+Tex` into the
adjacent terminal state `CD8+Tte`. The result is 16 levels present in both
cohorts, which is what allows one identified design to be fitted in each.

**Table 8.** Cells by harmonised phenotype and stage, `Zavidij_2020`.

| phenotype | Non | MGUS | SMM | MM |
| --- | ---: | ---: | ---: | ---: |
| CD4+Tn | 623 | 834 | 1,501 | 1,078 |
| CD4+Tcm | 142 | 143 | 278 | 171 |
| CD4+Tem | 95 | 59 | 199 | 105 |
| CD4+Temra | 81 | 13 | 523 | 176 |
| CD4+Th17 | 68 | 56 | 81 | 45 |
| CD4+Treg | 47 | 71 | 197 | 130 |
| CD8+Tn | 176 | 35 | 151 | 109 |
| CD8+Tcm | 6 | 14 | 26 | 30 |
| CD8+Tem.CD69+ | 249 | 43 | 192 | 143 |
| CD8+Tem.S1PR1+ | 5 | 4 | 25 | 27 |
| CD8+Tem.TCF1+ | 99 | 45 | 105 | 55 |
| CD8+Temra | 73 | 30 | 329 | 176 |
| CD8+Trm | 3 | 4 | 4 | 4 |
| CD8+Tte | 46 | 30 | 153 | 362 |
| Invariant | 46 | 24 | 72 | 70 |
| lineage-agnostic | 16 | 9 | 71 | 86 |

**Table 9.** Cells by harmonised phenotype and stage, `Foster_2026`.

| phenotype | Non | MGUS | SMM | MM |
| --- | ---: | ---: | ---: | ---: |
| CD4+Tn | 1,775 | 600 | 19,130 | 8,203 |
| CD4+Tcm | 2,296 | 1,708 | 7,169 | 2,166 |
| CD4+Tem | 456 | 281 | 4,862 | 213 |
| CD4+Temra | 876 | 152 | 6,665 | 927 |
| CD4+Th17 | 349 | 324 | 1,449 | 676 |
| CD4+Treg | 204 | 657 | 2,053 | 634 |
| CD8+Tn | 1,051 | 110 | 7,884 | 2,152 |
| CD8+Tcm | 80 | 274 | 2,662 | 225 |
| CD8+Tem.CD69+ | 3,366 | 880 | 10,127 | 2,569 |
| CD8+Tem.S1PR1+ | 55 | 115 | 2,489 | 115 |
| CD8+Tem.TCF1+ | 876 | 442 | 3,509 | 1,105 |
| CD8+Temra | 474 | 240 | 4,445 | 742 |
| CD8+Trm | 46 | 26 | 190 | 228 |
| CD8+Tte | 727 | 723 | 6,560 | 1,391 |
| Invariant | 205 | 24 | 1,759 | 459 |
| lineage-agnostic | 93 | 302 | 747 | 2,310 |

Three features of Tables 8 and 9 bear on what follows. `CD8+Trm` holds 3 to 4
cells per stage in `Zavidij_2020`, so no phenotype-resolved statement about it
is possible in that cohort. `CD8+Tte` rises from 46 to 362 cells across
`Zavidij_2020`'s ladder against a stable CD4 naive fraction, which is the kind
of composition shift the phenotype block exists to absorb. And
`Foster_2026`'s lineage-agnostic level rises from 93 to 2,310 cells, driven by
the interferon-stimulated and proliferating clusters that we merged into it, so
that level carries composition information we deliberately condition away rather
than interpret.

## 3. Methods

### 3.1 Notation and data representation

Let `G` be the number of genes in the map, `P = 1,615` the number of pathway
units, `K` the number of covariate columns and `L` the latent width. We set
`L = floor(P / 2) = 807` throughout.

For cell `i` we write the raw count vector as `n_i` in `N^G` and its library
size as `l_i = sum_g n_ig`. The encoder reads a normalised expression vector,

```
x_i = log(1 + 10^4 * n_i / l_i)
```

obtained with `scanpy.pp.normalize_total` to a fixed target of `10^4` followed
by `log1p`. We fix the target rather than using the dataset median, because the
median moves whenever cell or gene selection changes and makes losses
incomparable between runs. The likelihood reads the raw counts `n_i` and the
library size `l_i` directly, so normalisation affects the encoder's input alone.
The design matrix holds one-hot covariate columns `c_i` in `R^K`, assembled
without an intercept, so every level carries its own column.

### 3.2 Encoder

The encoder is a single masked linear layer followed by two dense heads, the
architecture of our earlier informed VAE [10, 11] with its non-negativity and
normalisation options enabled. Writing `Theta_enc` in `R^(P x G)` for the unconstrained weight parameter,
`b_enc` in `R^P` for the bias and `odot` for the elementwise product,

```
x~_i   = BatchNorm(x_i)                      per-gene, non-affine
W_enc  = softplus(Theta_enc) odot A^T        (P x G), masked and positive
a_i    = BatchNorm(W_enc x~_i + b_enc)       per-unit, non-affine
h_i    = tanh(a_i)                           in R^P
```

Three constraints act here, and each removes one degree of freedom. The mask
`A^T` restricts unit `p` to reading its own member genes. The softplus
reparameterisation makes every live weight strictly positive, so a unit cannot
encode its pathway upside down; the reconstruction loss never asks unit `p` to
resemble pathway `p`, and without this constraint an inverted unit costs
nothing. The input BatchNorm z-scores each gene before the masked product, so
one high-variance gene cannot dominate its unit by scale alone. The
pre-activation BatchNorm then standardises each unit's activation across the
batch. Both normalisers subtract a mean, which makes `b_enc` inert in this
configuration; we retain it so the layer behaves identically when normalisation
is disabled.

Those are the reasons each constraint was adopted. Section 4.7 tests them, and
finds that only the mask changes any quantity we report: normalisation, encoder
non-negativity and input standardisation are each immaterial to fit, to the
target's rank, to saturation and to annotation worth. We keep them, and we do
not rest any claim on them.

The posterior parameters read `h_i` alone:

```
mu_i          = V_mu h_i + c_mu
log sigma^2_i = clip(V_s h_i + c_s, -3, 3)
z_i           = mu_i + sigma_i odot eps,     eps ~ N(0, I_L)
```

The covariate does not enter the encoder. This is a deliberate restriction
rather than an omission. With a covariate route into `z` as well as into the
decoder, a decoder-side contrast measures only part of the conditional effect,
and the two parts may point in different directions. Closing the encoder route
makes the decoder-side term provably the whole contrast, which is what section
3.5 requires.

### 3.3 Pathway-space conditioning and the non-negative informed decoder

The decoder reconstructs a composition over genes. The covariate enters in
pathway coordinates, inside the nonlinearity:

```
u_i   = U z_i + Q c_i                        U: P x L,  Q: P x K, no bias
h'_i  = tanh(u_i)                            in R^P
W_dec = softplus(Theta_dec) odot A           (G x P), masked and positive
s_i   = softmax(W_dec h'_i + b_dec)          over genes
mu_i  = l_i * s_i
theta = exp(r),  r in R^G
n_i ~ NegativeBinomial(mu_i, theta)
```

`Q` carries no bias because the per-gene baseline belongs to the output layer
and `Q` should carry only the shift attributable to the label. The output bias
`b_dec` is left unmasked, so every gene keeps a free baseline including genes
with no pathway annotation. The softmax normalises over genes and the mask does
not alter that normalisation. Dispersion is per gene, parameterised through an
unconstrained `r_g`.

Non-negativity on `W_dec` is the change that gives a unit a direction. Under an
unconstrained `W_dec`, a unit rising raises some member genes and lowers others,
so "this pathway increased" does not state which way its genes moved. Under the
softplus reparameterisation every live unit-to-gene weight is positive, and the
sign of a unit's shift is the sign shared by all its members. We initialise the
pre-softplus parameter so that `softplus(Theta_dec)` reproduces the magnitude
the chosen initialiser requested, rather than collapsing every weight onto
`softplus(0) = 0.693`.

Non-negativity does not prevent the model from lowering a gene. The activity
`tanh(u)` is signed, so a unit falling lowers every one of its members.
Non-negativity removes the per-gene sign, which is what allows a per-unit sign
to exist.

### 3.4 Objective and optimisation

We minimise the negative evidence lower bound with an activity penalty on the
encoder's pathway layer:

```
Loss = -(1/N) sum_i sum_g log NB(n_ig | mu_ig, theta_g)
       + beta_t * (1/N) sum_i KL( N(mu_i, sigma_i^2) || N(0, I_L) )
       + lambda * (1/N) sum_i ||h_i||^2
```

with `lambda = 10^-5`. The KL weight follows a linear warm-up,
`beta_t = min(t / 10, 1)` over epoch index `t`, reaching 1 at epoch 10. We
evaluate the validation objective at `beta = 1` every epoch, so the reported
held-out loss is a genuine ELBO, comparable across epochs and across
architectures regardless of the warm-up.

Optimisation uses AdamW at learning rate `10^-3` with weight decay `10^-6`,
cosine annealing of the learning rate over the full epoch budget, gradient-norm
clipping at 5.0 and a batch size of 128. We train for 250 epochs and retain the
best-validation checkpoint. We set the early-stopping patience equal to the
epoch budget, so the schedule always completes and selection is by validation
loss rather than by stopping time. Cells split 80/20 into training and
validation, stratified by the design's primary factor crossed with cell type.
When the training split leaves a trailing batch of one cell we move that cell to
validation, because BatchNorm is undefined on a single row. All reported losses
are held-out per-cell negative ELBO in nats.

### 3.5 Covariate contrasts and unit-level statistics

Under sections 3.2 and 3.3 a covariate contrast is a weight difference and
nothing else. For two design vectors `c_a` and `c_b`,

```
delta_pre = Q (c_b - c_a)        in R^P
```

No cell is re-encoded, nothing is sampled, and no counterfactual convention
remains to choose. We report two quantities derived from this, and they are not
interchangeable.

`delta_pre` is the exact pre-activation shift above, identical for every cell.
`member_shift_z` is what reaches the genes. Writing `B` for the baseline cells
of the lineage under test, `w_p = sum_g W_dec[g, p]` for unit `p`'s total
written weight and `n_p` for its measured member count,

```
delta_post_p     = (1/|B|) sum_{i in B} [ tanh(u_ip + delta_pre_p) - tanh(u_ip) ]
member_shift_z_p = delta_post_p * w_p / sqrt(n_p)
```

The `1 / sqrt(n_p)` factor corrects a size tilt that otherwise hands the top of
the ranking to the smallest units, pulling the Spearman correlation between
`|member_shift_z|` and set size from -0.49 to -0.09. Every ranking and every
figure colour in this work uses `member_shift_z`.

We evaluate `delta_post` in float64 before the `tanh`. For small `delta_pre`,
the difference `tanh(u + delta) - tanh(u)` is a catastrophic cancellation that
rounds to exactly zero in float32 once `|u|` exceeds roughly 4, which silently
empties any saturated unit. The regression test is
`tests/unit/test_cov_conditioning_modes.py::TestSaturationDoesNotDestroyTheSign`.

We also report the ranking obtained from `delta_pre` alone, since that quantity
is cell-independent and is the obvious thing to rank on. Section 4.4 shows it is
the weaker statistic.

### 3.6 Architecture ablation

One script, `01_architecture.py`, runs identically on all three datasets and
produces every architecture result. It crosses the three changes, applied
cumulatively, with a shuffled annotation:

| arm | configuration | |
| --- | --- | --- |
| `shipped` | dense `D c` in `R^G` added to the logits after `W_dec`; covariate also enters the encoder; `W_dec` unconstrained | the standard conditional informed decoder |
| `pathway` | `Q c` inside the nonlinearity, no gene-space term | covariate effect becomes a pathway-space vector |
| `uncond` | `+` encoder covariate route closed | and the only route |
| `signed` | `+ W_dec` non-negative | and a unit rising raises its genes |

Each runs as a `real_*` and a `perm_*` arm, giving eight fits per dataset. The
`shipped` arm is the architecture published designs use, and it is the
comparator throughout; the `signed` arm is the architecture of sections 3.2 to
3.3. All four arms share the encoder mask, the encoder non-negativity, both
BatchNorms and every optimisation setting, so the arms differ only in where the
covariate enters and whether `W_dec` carries a sign.

We run four seeds on `Zavidij_2020` and one on each of `Foster_2026` and Kang.

### 3.7 The permutation control

The `perm_*` arms replace `A` with `A~`, obtained by relabelling the gene rows
of `A` under a random permutation `pi` of `{1, ..., G}`, so
`A~[g, p] = A[pi(g), p]`. Every unit keeps its size and its name, the per-gene
degree multiset survives, and so does the entire pathway-by-pathway overlap
matrix `A^T A`. An arm and its permuted twin are therefore matched in capacity,
sparsity and overlap topology, and they differ only in which named genes share a
unit. That match is what lets us attribute the gap between them to the biology
in the map rather than to the shape of the mask.

We define the **annotation worth** of an arm as the held-out loss under the
permuted map minus the held-out loss under the real one, in nats, within that
arm. A positive value means the real annotation describes the data better than a
structurally identical shuffle.

### 3.8 Encoder ablation

The ablation of section 3.6 varies the conditioning path and holds four other
options fixed: `normalize="batch"`, `informed_decoder=True`,
`nonneg_encoder=True` and `standardize_input=True`. Nothing in that grid is
evidence for them, so a second grid tests them directly. It pins the
conditioning at the selected scheme and moves one option at a time, each arm
real and permuted, giving 18 further fits:

| arm | configuration |
| --- | --- |
| `selected` | the four options as above; reproduces `real_signed` from section 3.6 |
| `normalize_none` | no pre-activation normalisation |
| `normalize_layer` | LayerNorm in place of BatchNorm |
| `init_fan_in` | fan-in initialisation |
| `free_encoder` | `nonneg_encoder=False`, restoring encoder sign freedom |
| `unscaled_input` | `standardize_input=False`, restoring input scale freedom |
| `free_unscaled` | both off, completing the 2x2 with `selected` as "both" |
| `dense_decoder` | `informed_decoder=False`, so no mask on the decoder at all |
| `dense_encoder` | `informed_encoder=False`, so no mask on the encoder at all |

Both grids share one data loader, so `selected` is the same fit as
`real_signed`: held-out loss 1329.86 and target rank 25 in each. We check that
identity rather than assume it.

The last two arms each move two options, since a non-negative constraint needs
a mask on its own side to apply to. They are the two halves of one question, and
together they separate what the encoder mask is worth from what the decoder mask
is worth, which no grid in this line of work could previously do.

`dense_decoder` is scored on held-out loss and annotation worth alone. With no
mask on `W_dec` every gene is a member of every unit, so `member_shift_z` has no
member set to sum over and the unit read-out is undefined rather than merely
different. `dense_encoder` keeps the decoder mask, so it is scored like every
other arm; its encoder-side statistics are then taken over every weight rather
than over the live ones, and we check that its encoder is in fact unmasked by
confirming that half its weights are negative.

### 3.9 Baseline gene-level read-out

Submitting a model's denoised or counterfactual fold changes to a competitive
gene-set enrichment test is common practice, and it is the baseline our
unit-level read-out replaces. We compute it in every arm: a competitive rank
test of each Hallmark set's member genes against its non-member genes, on the
gene-level contrast the arm's own decoder produces. It never sees the model's
units, so it provides the control for whether an apparent pathway result
required the model at all.

### 3.10 Donor-level inference

Treating cells as replicates confounds within-donor technical variance with
biological signal [7]. The same applies to a model's latent units, so
`03_donor_test.py` refits the architecture of sections 3.2 to 3.3 on the same
cells with the stage block interacted with donor rather than with cohort. Each
donor's score for unit `p` is then their own column of `Q` minus the mean over
the healthy donors' columns, giving 1,615 signed numbers per donor read directly
off the weights, with no enrichment test anywhere.

The primary test is a Spearman trend of donor score against the four-level stage
ladder, with a null from 100,000 permutations of the donor stage labels, and
Benjamini-Hochberg correction over the 1,262 units carrying at least 10 measured
member genes. We also run the Non-to-MM endpoint contrast as a secondary test.

The permutation count follows from the attainable significance. BH multiplies
the smallest attainable p-value by the number of tests, so the best q a design
can produce is `N_UNITS / (N_PERM + 1)`. At `10^4` draws over 1,262 units that
floor is q = 0.126, above any conventional alpha, so the trend test could not
return a single unit whatever the data said. At `10^5` draws it is q = 0.013. We
report this floor for both tests in every run, because a null result and an
unattainable one require different readings.

As a model-free comparator we compute a phenotype-standardised pseudobulk score
over the same units from the same cells, and test it identically.

### 3.11 Multiplicity under a nested collection

Because Reactome nests (section 2.1), a parent and its child are
near-duplicate hypotheses, and BH over the full collection is conservative. We
accept that conservatism and state it wherever a q-value appears. We report
signed per-unit contrasts, ranks and donor-level q-values, and we report no
count of significant units and no overlap or enrichment test over units
anywhere, since "n pathways were significant" is not well defined over a nested
collection.

### 3.12 Forward-pass identities

Every read-out above claims to describe what the model computes. Rather than
assert that, `01_architecture.py` checks each claim against the forward pass at
float32 precision, in every arm it applies to:

1. masked contribution plus bypass contribution equals the re-encoded shift;
2. a counterfactual at fixed latent state equals the dense gene-space term;
3. a counterfactual at fixed latent state equals the re-encoded shift;
4. `delta_pre` equals the pre-activation difference from a forward pass.

The third identity is the informative one. In the `shipped` arm the
counterfactual convention changes the answer: holding `mu` fixed returns the
dense term alone, while re-encoding returns something different. With the
encoder covariate route closed the two become bit-identical, because `mu`
does not depend on the covariate. The architecture removes a choice
that otherwise determines the result without being chosen.

### 3.13 Implementation

The model is implemented in PyTorch in the `pyvae` package. Both experiments are
Snakemake workflows under pixi, with `pyvae` installed editable so that library
and experiments move together. Integrated gradients use 2,000 interpolation
steps against a matched baseline.

```bash
cd experiments/multiple_mieloma   # or experiments/kang
./00_setup.sh
pixi run all        # snakemake -j 63 --resources gpu=3
pixi run verify     # myeloma: ALL PASS (88 checks); kang: ALL PASS (87)
```

Myeloma takes about 3.5 h wall on three A100 80GB cards, dominated by
`Foster_2026`'s eight architecture arms at roughly 25 min each; Kang takes about
2.5 h, dominated by its 38 ablation fits. `outputs/benchmarks.csv` records the
measured resource use of every job. Every figure in this paper is drawn by a CPU script,
`kang/05_figures.py` and `multiple_mieloma/04_figures.py`, from tables the fits
wrote, so a figure can be redrawn in seconds on a laptop and no figure edit can
invalidate a fit. Each verifier recomputes rather than reads back wherever it
can, checks figures against the panel data the figure script writes at draw time
rather than against a re-derivation, asserts the costs as well as the benefits, and
asserts its own check count, so that a claim cannot leave the prose and the
verifier at the same time.

## 4. Results

### 4.1 The standard conditional decoder excludes the annotation from the contrast

In the `shipped` arm the covariate term is dense over genes and enters after the
masked layer, so the decoder computes

```
logits = W_dec tanh(U z) + D c
```

A counterfactual contrast holds the posterior mean fixed and decodes under two
covariate levels. The masked term is then identical on both sides and cancels,
leaving `D[:, b] - D[:, a]`, one column difference of a matrix the annotation
never touched. This is algebra, and the numbers confirm it. The counterfactual
prediction matches that dense term to 6.2e-07 on `Zavidij_2020`, 1.5e-06 on
`Foster_2026` and 1.03e-06 on Kang.

The quantity `masked_share`, the fraction of the conditional effect expressed in
the annotation basis at all, is 0.061 and 0.078 on `Zavidij_2020`'s two
lineages, 0.089 and 0.066 on `Foster_2026`'s, and 0.179 on Kang. Between 82% and
94% of the contrast bypasses the annotation.

Under pathway-space conditioning `masked_share` is 1.000 by construction, and
the third identity becomes exact.

**Table 10.** Worst deviation per forward-pass identity, over the arms each
applies to.

| identity | arms (Kang / myeloma) | Kang | `Zavidij_2020` | `Foster_2026` |
| --- | ---: | ---: | ---: | ---: |
| masked + bypass = re-encoded shift | 8 / 16 | 1.04e-06 | 6.6e-07 | 1.5e-06 |
| counterfactual = dense term | 2 / 4 | 1.03e-06 | 6.2e-07 | 1.5e-06 |
| **counterfactual = re-encoded shift** | 4 / 8 | **0.00e+00** | **0.00e+00** | **0.00e+00** |
| `delta_pre` = forward pass | 4 / 8 | 8.29e-07 | 3.6e-07 | 1.1e-06 |

The third row is zero rather than small. Once the encoder covariate route
closes, the two computations become the same computation.

### 4.2 The baseline gene-level read-out does not depend on the annotation

On Kang the competitive rank of the interferon programme is 1 in all eight arms,
including every arm whose gene-to-pathway map is shuffled, at z 8.22 to 10.36.
On `Zavidij_2020` it spans ranks 1 to 11 across sixteen arm-by-lineage
combinations, and on `Foster_2026` ranks 1 to 17.

A read-out that returns the right answer under a destroyed annotation is testing
the gene list rather than the model. It is a gene-list enrichment test with a
model-shaped denoiser in front of it. The controlled comparison in [6] reached
the same conclusion from the other direction, finding that gene sets determined
which latent held a signal without contributing measurable signal themselves.

### 4.3 Annotation worth depends on where the covariate enters

**Table 11.** Annotation worth in nats: held-out loss under the permuted map
minus held-out loss under the real one, within each arm.

| arm | Kang | `Zavidij_2020` | `Foster_2026` |
| --- | ---: | ---: | ---: |
| `shipped` | +2.15 | +0.64 | +1.30 |
| `pathway` | +7.10 | +3.35 | +2.70 |
| `uncond` | +7.47 | +2.63 | +3.24 |
| **`signed`** | **+14.52** | **+6.32** | **+7.98** |

The `shipped` row reproduces the published negative result [4] on a conditional
generative model: a shuffled annotation is nearly as good as the real one. The
`signed` row is the same quantity on the same data against the same permuted
maps, three to seven times larger. Across four `Zavidij_2020` seeds the `signed`
figure is +6.32, +6.48, +4.22 and +6.87.

Two of those three columns carry a confound we measure in section 4.7:
annotation worth rises with the fraction of saturated units, and the Kang and
`Zavidij_2020` arms move sharply across that range (23% to 71% and 24% to 80%
of units pinned). Their figures are therefore saturation-confounded.
`Foster_2026` does not move, at 76% against 75%, so its +1.30 to +7.98 is
attributable to the conditioning path alone and is the cleanest form of this
result.

### 4.4 Unit-level recovery of the target

**Table 12.** Rank of `REACTOME_INTERFERON_ALPHA_BETA_SIGNALING` among all 1,615
units by `member_shift_z`. Myeloma columns give CD4 and CD8.

| arm | Kang | `Zavidij_2020` | `Foster_2026` |
| --- | ---: | ---: | ---: |
| `real_shipped` | **1615** | 1614, 1615 | 144, 102 |
| `perm_shipped` | 1130 | 781, 791 | 560, 563 |
| `real_pathway` | 218 | 5, 7 | 3, 4 |
| `perm_pathway` | 1350 | 958, 921 | 902, 923 |
| `real_uncond` | **1** | 2, 5 | 2, 4 |
| `perm_uncond` | 1271 | 1009, 978 | 910, 848 |
| **`real_signed`** | **25** | **2, 1** | **4, 3** |
| `perm_signed` | 1430 | 845, 537 | 985, 668 |

On Kang and on `Zavidij_2020` the `shipped` arm ranks the target last, and
shuffling the annotation improves it, from 1615 to 1130 and from 1614/1615 to
781/791. No reading of that leaves those units carrying the biology.
`Foster_2026` behaves differently: there `real_shipped` puts the target at 144
and 102 against the shuffle's 560 and 563, so it is poor rather than inverted.
We make the inversion claim about Kang and `Zavidij_2020` and not about both
cohorts.

Across the four `Zavidij_2020` seeds the real arm is stable and the permuted arm
is not. The real arm sits in the top four every time, with worst rank per seed
2, 3, 3 and 4, while the permuted rank ranges over roughly 500 to 1,500 for the
same arm. No single permuted rank should be quoted as a point value, so the
verifier gates on a floor of 300 instead.

Ranking on `delta_pre` alone is the weaker statistic. The target sits at 1,599
under `uncond` and 284 under `signed` on Kang by raw weight, against 1 and 25 by
`member_shift_z`. On myeloma it sits at 56 and 19 (`Zavidij_2020`) and 640 and
753 (`Foster_2026`), against 2 and 1, and 4 and 3.

### 4.5 Pathway-level sign

**Table 13.** Sign structure of `W_dec`, `real_shipped` against `real_signed`.
Sign agreement is the fraction of units whose own shift matches the mean shift
of their member genes.

| | `shipped` negative fraction | `shipped` sign agreement | `signed` negative fraction | `signed` sign agreement |
| --- | ---: | ---: | ---: | ---: |
| Kang | 0.495 | 0.515 | **0.000** | **1.0000** |
| `Zavidij_2020` | 0.503 | 0.497 | **0.000** | **1.0000** |
| `Foster_2026` | 0.506 | 0.505 | **0.000** | **1.0000** |

Under the standard decoder a unit rising pushes about half its member genes up
and half down, and sign agreement is at chance. Flipping each unit post hoc by
its weight sum would need most units to be nearly all one sign, and they are
not: the per-unit positive fraction has median exactly 0.500, only 20.2% of
units are at least 90% one sign, and 22.4% to 37.6% sit between 0.4 and 0.6
where no orientation exists at all. Under the non-negative decoder direction
sits in the unit rather than scattered across its genes.

### 4.6 What the architecture costs

The architecture fits worse wherever the dense gene-space term is well
estimated. Held-out loss rises by 20.74 nats on Kang (1329.86 against 1309.12),
2.69 nats on `Zavidij_2020` (1486.89 against 1484.20) and 10.05 nats on
`Foster_2026` (2058.65 against 2048.60). A dense per-gene matrix describes one
large diffuse shift with fewer parameters than 1,615 annotated units can, and the advantage grows with the number of cells available to estimate it. The case for the
architecture rests on sections 4.3 to 4.5 rather than on fit.

The second cost is saturation. A unit is pinned when `|tanh(u)| > 0.99` in more
than half its cells, and a pinned unit writes nothing to its genes whatever its
weights say.

**Table 14.** Pinned units and written magnitude, `real_shipped` against
`real_signed`.

| | pinned, `shipped` | pinned, `signed` | median \|`delta_post`\| |
| --- | ---: | ---: | --- |
| Kang | 23% | **71%** | 2.7e-01 to 5.4e-02 |
| `Zavidij_2020` | 24% | **80%** | 5.0e-02 to 2.2e-03 |
| `Foster_2026` | **76%** | 75% | 3.3e-01 to 1.5e-01 |

The two myeloma cohorts behave differently, and the difference is not noise. At
125,602 cells the standard decoder already saturates three quarters of its
units, so the architecture adds nothing there, while at 9,863 cells it goes from
a quarter to four fifths. The consequence holds everywhere: most units end up
pinned, and the ranking runs over the units that still have headroom. A unit can
be biologically involved and invisible here because it is pinned. Every fit
records the count, so the caveat travels with the numbers. No unit writes
exactly nothing in any of the three fits.

### 4.7 The decoder mask carries the annotation, the encoder mask the fit; saturation contaminates the measure

**Table 15.** The 18 arms of section 3.8, real arms shown with the annotation
worth measured against their own permuted twin. The last column gives where that
twin ranks the target.

| option | fit | vs selected | annot worth | leaf | parent | pinned | enc -ve | perm leaf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`selected`** | 1329.9 | +0.00 | **+14.52** | 25 | 3 | 71% | 0.000 | 1430 |
| `normalize_none` | 1331.4 | +1.54 | +15.84 | 10 | 4 | 72% | 0.000 | 1385 |
| `normalize_layer` | 1331.0 | +1.16 | +13.47 | 22 | 2 | 74% | 0.000 | 1403 |
| `init_fan_in` | **1320.9** | **-8.96** | +8.07 | **1** | 1196 | **43%** | 0.000 | **450** |
| `free_encoder` | 1330.6 | +0.73 | +14.54 | 29 | 1 | 73% | 0.501 | 1489 |
| `unscaled_input` | 1329.0 | -0.91 | +13.87 | 24 | 2 | 71% | 0.000 | 1461 |
| `free_unscaled` | 1329.8 | -0.06 | +13.96 | 25 | 1 | 72% | 0.500 | 1502 |
| `dense_decoder` | 1330.3 | +0.42 | **+3.24** | n/a | n/a | n/a | n/a | n/a |
| `dense_encoder` | 1358.1 | **+28.26** | +13.31 | 19 | 3 | 69% | 0.500 | 1204 |

Removing the mask from the decoder while leaving the encoder masked drops
annotation worth from **+14.52 to +3.24 nats**. The encoder mask alone captures
22% of what the annotation is worth, and the decoder mask supplies the other
**78%**. It costs +0.42 nats of fit, so the mask itself adds almost no loss, and the 20.74 nats of section 4.6 are attributable to the conditioning path rather than to the annotation.

The symmetric arm shows that the two masks do different jobs. Dropping the
encoder's mask while keeping the decoder's leaves annotation worth at **+13.31**,
a fall of 1.21 nats against the decoder mask's 11.28, and the two arms sit at
69% and 71% of units pinned, so the saturation curve below accounts for only 0.31 nats
of that gap. What the encoder mask contributes is held-out loss: the dense arm fits
**28.26 nats worse** while holding 12,230,395 first-layer weights where the mask
holds 66,809, which is 0.55% of them. The annotation in the encoder is acting as
a regulariser, and the annotation in the decoder is what the contrast reads.

The encoder mask has a further justification that this table does not show.
Integrated gradients attribute the encoder's own unit, so under the mask a gene
outside the annotated set has exactly zero attribution, which is what makes the
per-gene read-out of section 4.8 a statement about a pathway at all. Under a
dense encoder that panel becomes a ranking over all 7,573 genes with nothing
tying it to the set. We assert both behaviours in the unit tests rather than
describing them.

This matters beyond our own configuration. A masked encoder with an unmasked
decoder is the architecture of the informed VAE line this work descends from
[10, 11], and it is what the stability framework of [12] evaluated. On this
estimand that configuration recovers roughly a fifth of the annotation's value,
and the measurement now runs in both directions: the mask that line places is
the one that moves the contrast least.

The other three options are immaterial. Annotation worth spans +13.47 to +15.84
across normalisation and the two encoder constraints, against the +2.15 to
+14.52 that the conditioning path moves it in Table 11. The target leaf stays
between 10 and 29, every permuted arm leaves it past 1,385, and saturation sits
between 71% and 74%.

Two specific justifications do not survive. We chose `normalize="batch"` to
reduce saturation, and it does not: 71% against 72% with no normalisation at
all. And the 2x2 of sign freedom against scale freedom in the encoder is flat.

**Table 16.** Annotation worth in nats, with the target's rank in brackets, for
the four cells of `nonneg_encoder` against `standardize_input`.

| | `standardize_input` on | off |
| --- | ---: | ---: |
| `nonneg_encoder` on | **+14.52** (25) | +13.87 (24) |
| off | +14.54 (29) | +13.96 (25) |

No main effect from either constraint, no interaction, and a range of 0.67 nats
across four cells. Encoder non-negativity is retained for a narrower reason than
section 4.5 might suggest: with a freely signed encoder the negative weight
fraction is 0.501 and unit-versus-gene sign agreement is still exactly 1.0000,
so the pathway-level sign comes from the non-negative **decoder** alone. What
the encoder constraint provides is a unit that is not an upside-down
representation of its own pathway, which bears on reading a unit's activation
rather than on reading a contrast.

Fan-in initialisation is the exception, and it turns out to expose a property of
the measure rather than of the initialiser. It fits 8.96 nats better, ranks the
target leaf first rather than 25th, and cuts saturation from 71% to 43%, while
annotation worth falls to +8.07. Read at face value that says the annotation
matters less under fan-in. It is also the only low-saturation arm in the grid,
so the two explanations are collinear and cannot be separated here.

We separate them with a knob that moves saturation without touching the
annotation: scaling the decoder's `U` and `Q` at initialisation scales the
pre-activation `u`, and so moves how far units sit into the flat tail of `tanh`,
while leaving `A` and `W_dec` untouched. Only scaling up holds at convergence,
since training restores whatever magnitude the likelihood wants, so we raise
fan-in toward the selected architecture rather than lowering it toward fan-in.

**Table 17.** Annotation worth against the fraction of units pinned, under the
selected architecture at four pre-activation scales and fan-in at three.

| arm | pinned | fit | annot worth | leaf |
| --- | ---: | ---: | ---: | ---: |
| xavier, scale 1 | 71% | 1329.9 | +14.52 | 25 |
| xavier, scale 2 | 73% | 1330.4 | +13.21 | 28 |
| xavier, scale 4 | 82% | 1332.6 | +13.21 | 10 |
| xavier, scale 8 | 93% | 1340.0 | +17.96 | 1 |
| fan-in, scale 1 | 43% | 1320.9 | +8.07 | 1 |
| fan-in, scale 2 | 47% | 1322.7 | +9.12 | 1 |
| **fan-in, scale 4** | **65%** | 1326.7 | **+13.95** | 240 |

Annotation worth rises with saturation at **+1.72 nats per 10 points** of pinned
fraction (r = +0.75 over the four xavier arms). Lifted into that band, fan-in
scores **+13.95 at 65% pinned against the selected architecture's +14.52 at
71%**. All three fan-in arms lie on the xavier curve, with residuals of -0.36,
-0.01 and +1.70 nats against the 6.45-nat deficit that needed explaining.

Fan-in's low annotation worth was therefore its low saturation, not its
annotation. We report this rather than the earlier reading because it changes
what the measure means: **annotation worth is only comparable between arms at
similar saturation.** Every comparison in this paper that spans a large
saturation change is confounded to that extent, which is why section 4.3 marks
the Kang and `Zavidij_2020` columns and leans on `Foster_2026`.

The rank statistic is no cleaner. Across the xavier ladder the target's rank
improves monotonically as saturation rises, from 25 at 71% pinned to 1 at 93%,
because the ranking runs over the units that still have headroom. Leaf rank 1 is
reachable at 43% pinned through fan-in and at 93% through scaling, by opposite
mechanisms. Neither of the two metrics we might have selected the initialiser on
is saturation-free.

### 4.8 The model learns biology: validation on a known perturbation

![Figure 1. Kang, the ranked units. The fifteen units interferon-beta stimulation raises most and the eight it lowers most, by member_shift_z, with the target leaf and its parent marked. No significance mark and no cut; the full ranking is in unit_contrasts.csv and this shows its two ends.](kang/outputs/units/figs/fig1_ranked_units.png)

Under the selected architecture the named leaf
`INTERFERON_ALPHA_BETA_SIGNALING` ranks 25th of 1,615, its parent
`INTERFERON_SIGNALING` 3rd and its grandparent
`CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM` 2nd. Those three sets nest strictly, so
this is one finding at three resolutions, and the model prefers the broader
resolution once units are size-corrected. Under the permuted annotation the same
leaf sits at 1,430.

Read as families rather than as rows, the interferon response occupies nine of
the 25 leading units.

**Table 18.** Interferon and innate-sensing units among Kang's 25 leading units.

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

Type-I interferon signalling, the three nucleic-acid sensing arms that induce it
through endosomal TLR, STING and ZBP1, its antiviral effector arm through OAS,
and the IRF activation step that links them. None of that entered the fit, and
none of it is a separate finding from the others, which is why we report no
count over these rows. The strongest single unit is `SIGNALING_BY_GPCR` with 385
genes, which is not interferon biology and which we claim as none. It is the kind
of large diffuse unit a size-corrected statistic favours, and we report it
rather than cropping it from the figure.

The arms of the ablation disagree about resolution. `real_uncond` puts the leaf
1st, while `real_signed` puts it 25th and lifts its family to the top two.
The sign constraint therefore provides direction at the expense of leaf precision. We
report both, since the leaf's rank is better without the constraint and the
constraint is what makes any of these numbers readable as a direction.

![Figure 2. Kang, the target unit's pre-activation by condition and cell type, with the saturation line at arctanh(0.99). With every cell held at control in the covariate, the encoder still places the stimulated cells higher on this unit.](kang/outputs/units/figs/fig2_target_activation.png)

Two independent routes agree, which is the check worth taking from this fit. The
covariate term raises the interferon unit's pre-activation by +2.60, identically
for every cell. Separately, with every cell held at `control` in the covariate,
the encoder still places the stimulated cells higher on that unit,
because their transcriptomes say so and the model was told nothing. The response
is strongest in CD14+ monocytes, FCGR3A+ monocytes and dendritic cells, and
weakest in megakaryocytes, which is the expected myeloid-dominant type-I
interferon pattern and played no part in fitting.

![Figure 3. Kang, integrated gradients inside the interferon unit, top 18 genes. Filled bars are annotated members of the set.](kang/outputs/units/figs/fig3_ig_target.png)

Integrated gradients inside the interferon unit name ISG20, IFI6, IFIT1, HLA-B,
ISG15, IFIT3, MX1, IFIT2, HLA-C and SOCS1, carrying 75% of the unit's total
attribution between them. These are canonical interferon-stimulated genes plus
two class-I MHC genes, themselves interferon-inducible. This is a within-unit
question, what the unit reads the pathway through, rather than a ranking of
units, and the decoder side names different genes.

Nine of the 1,615 Reactome sets share no measured gene with this panel. They
exist in the model and the latent layer writes to their pre-activation, but
`W_dec` has no live weight for them, so whatever they do reaches nothing. We
report them and exclude them from the sign statistic, because a unit with no
genes has no member direction to agree or disagree with. Including them would
report sign agreement as 0.9944 on a fit where every unit that writes anything
agrees perfectly.

![Figure 4. Kang, UMAP from raw counts coloured by the target unit's pre-activation. Included for orientation and carrying no claim.](kang/outputs/units/figs/fig4_umap.png)

### 4.9 Stage contrasts in myeloma precursor disease

![Figure 5. Stage-associated activity of pathway-informed decoder units in bone-marrow T cells, Zavidij_2020. (a) The 18 units the stage sequence raises most and the 7 it lowers most, ordered by the mean endpoint contrast over CD4 and CD8 across both cohorts, so this panel and Figure 6a show the same rows in the same order. Columns are the three ladder contrasts against the non-cancer baseline within each lineage, so a row reads left to right as a trajectory. Colour is member_shift_z scaled by the cohort's own 99th percentile. No significance mark, because no valid test exists over nested units. (b) Endpoint contrast in CD4 against CD8, one point per unit, with the limiting cell count annotated. (c) Annotation worth per architecture. (d) The same leading units with the donor as the sampling unit; a ring at the left edge marks a Spearman trend at q < 0.05 from 100,000 donor-label permutations.](multiple_mieloma/outputs/figures/fig1_Zavidij_2020.png)

![Figure 6. The same analysis in Foster_2026. Identical layout, row order and colour scale to Figure 5, so the two pages can be overlaid.](multiple_mieloma/outputs/figures/fig2_Foster_2026.png)

Both fits enter the same two covariate blocks, the stage block crossed with
lineage and the harmonised phenotype block, and neither fit drops a block. Both
give 36 covariate columns over 1,615 units. `Zavidij_2020` reaches a held-out
loss of 1486.82 with 1,201 of 1,615 units pinned, `Foster_2026` 2058.69 with
1,259 pinned.

`interferon alpha beta signaling` leads the `Non` to `MM` contrast in both
cohorts, at rank 1 of 1,615 in each.

Two diagnostics sit beside that. CD4 and CD8 agree over all units at Spearman
+0.26, +0.31 and +0.38 for the three ladder contrasts in `Zavidij_2020`, and
+0.60, +0.62 and +0.66 in `Foster_2026`. The two lineages are separate columns
of one design, so this measures the fit rather than the biology, and it tracks
cell count as noise attenuation predicts: the cohort with 13 times the cells
agrees roughly twice as well. Separately, for the four leading units in each
cohort, the gene the encoder is most sensitive to and the gene carrying the
largest share of the decoder's written response are the same gene in 1 of 4
(`Zavidij_2020`) and 0 of 4 (`Foster_2026`). A single driver-gene column would
have been wrong seven times out of eight, which is why we report both sides.

### 4.10 One unit replicates between cohorts, the panel does not

![Figure 7. What replicates between the two cohorts, read over the model's own units. (a) Endpoint member_shift_z in one cohort against the other, one point per shared unit, with the Spearman correlation. (b) Each cohort's ten leading units and where the other cohort ranks them; filled marks give the unit's rank in its own cohort, open marks its rank in the other, on a logarithmic axis over all 1,615 units. (c) The same comparison with the donor as the sampling unit. (d) Where each architecture puts the target unit, both cohorts, logarithmic, lower better. (e) The two cohorts side by side in words.](multiple_mieloma/outputs/figures/fig3_comparison.png)

We never pool the two cohorts, because 69% of the atlas's non-cancer donors sit
in studies containing no patients. A pooled disease contrast would substantially
be a contrast between studies, and `study_id` cannot be adjusted for without
becoming collinear with the estimand. We therefore compare the two fits'
outputs.

Beyond the leading unit they disagree. The panel-wide Spearman between the two
cohorts' endpoint contrasts, over all 1,615 units, is rho = -0.073, which is
indistinguishable from no relationship.

**Table 19.** Each cohort's eight leading units and their rank in the other
cohort.

| `Zavidij_2020` top 8 | rank in `Foster_2026` | | `Foster_2026` top 8 | rank in `Zavidij_2020` |
| --- | ---: | --- | --- | ---: |
| interferon alpha beta signaling | **1** | | interferon alpha beta signaling | **1** |
| mitochondrial translation | 1596 | | cell surface interactions at the vascular wall | 1266 |
| oas antiviral response | 1274 | | slc mediated transmembrane transport | 1061 |
| cross presentation of soluble exogenous antigens | 1533 | | fceri mediated mapk activation | 316 |
| clec7a dectin 1 signaling | 1499 | | c type lectin receptors clrs | 366 |
| attenuation phase | 407 | | binding and uptake of ligands by scavenger receptors | 725 |
| creb3 factors activate genes | 285 | | signaling by tgfb family members | 972 |
| er to golgi anterograde transport | 1063 | | membrane trafficking | 935 |

One shared unit, at the top of both, with no panel-wide agreement. Two cautions
travel with that. Selecting the unit that agrees and reporting its rank is a
post-hoc choice, so no p-value attaches to "rank 1 in both". And Reactome nests,
so `oas antiviral response` at rank 3 in `Zavidij_2020` is part of the same
interferon response rather than a second finding.

### 4.11 Donor-level inference

The power floor comes first, because it decides how to read a null.

**Table 20.** Best attainable q per test, from `N_UNITS / (N_PERM + 1)`.

| test | best attainable q | reaches 0.05? |
| --- | ---: | --- |
| trend, 100,000 draws | **0.013** | yes |
| endpoint, `Zavidij_2020` CD4 / CD8 | 1.47 / 12.0 | **no** |
| endpoint, `Foster_2026` | 45.1 | **no** |

The endpoint test cannot reach conventional significance in either cohort at any
number of draws, because too few donors occupy each stage, and we report it only
beside its floor. The trend test is informative, so its negatives are genuine
negatives.

**Table 21.** Donor-level trend for the interferon unit, and the number of units
surviving correction in both lineages.

| | `Zavidij_2020` CD4 / CD8 | `Foster_2026` CD4 / CD8 |
| --- | --- | --- |
| donors scored | 27 / 17 | 22 / 22 |
| interferon rho | +0.25 / +0.24 | **+0.78 / +0.68** |
| interferon q | 0.89 / 0.93 | **0.0084 / 0.046** |
| units at q < 0.05, both lineages | **0** | **5** |

`Foster_2026`'s five are `interferon alpha beta signaling`, `atf6 alpha
activates chaperones`, `hyaluronan metabolism`, `membrane trafficking` and
`regulation of ras by gaps`.

The baseline gene-level read-out gives the opposite ordering. Applied to the
same donors it finds interferon strong in `Zavidij_2020` (rho +0.45/+0.74) and
absent in `Foster_2026` (rho -0.05/+0.10). The unit-level read-out, the one that
depends on the annotation, reverses it.

**Table 22.** Per-donor interferon unit score in CD4, by stage.

| stage | `Zavidij_2020` n, mean, range | `Foster_2026` n, mean, range |
| --- | --- | --- |
| Non | 6, 0.00, -1.74 to +1.99 | 3, 0.00, -0.17 to +0.14 |
| MGUS | 4, +1.31, +0.80 to +1.79 | 1, +0.26 |
| SMM | 10, +0.66, -1.57 to +3.08 | 13, +0.23, -0.89 to +0.61 |
| MM | 7, +1.41, -0.18 to +3.29 | 5, **+1.51, +0.91 to +2.27** |

Both cohorts place every disease stage above their own healthy baseline. They
differ in baseline scatter. `Zavidij_2020`'s six healthy donors span 3.7 units
of score and `Foster_2026`'s three span 0.3. Foster's five MM donors are
disjoint from its healthy ones, which gives an AUC of 1.000 and rho +0.78;
Zavidij's MM donors overlap its healthy ones, which gives rho +0.25 despite a
similar mean shift. That is a statement about estimator variance rather than
about biology, and the competing explanation is the obvious one: three donors
can look tight by accident. This design does not separate the two readings.

The model-free comparator agrees globally and disagrees at the threshold. Over
all units the two estimators correlate at Spearman +0.38 and +0.45
(`Zavidij_2020` CD4 and CD8) and +0.27 and +0.29 (`Foster_2026`), each at
p < 1e-20. Their thresholded sets are disjoint: pseudobulk returns 0 units in
`Zavidij_2020` and 51 in `Foster_2026`, overlapping the model's five in none.
The second fact follows largely from the first, since two correlated but noisy
rankings crossing a sharp cut-off routinely have disjoint tails. The correlation
is the right summary and the set overlap is not. On interferon specifically the
pseudobulk is weak in both cohorts, at +0.19/+0.44 in `Zavidij_2020` and
+0.17/+0.02 in `Foster_2026`, so it does not corroborate Foster's result on the
one unit the model singles out.

## 5. Discussion

### 5.1 The sparsity result depends on where the covariate enters

A permutation control applied to an estimand the annotation does not enter must
return null. Our `shipped` arms do exactly that, at +0.64, +1.30 and +2.15 nats,
reproducing the published finding [4] on a conditional generative model. The
same gap at the `signed` arms, on the same data and against the same permuted
maps, is +6.32, +7.98 and +14.52 nats. One codebase, one dataset and one null
produce both answers, and the only thing that changes is where the covariate
enters the decoder.

That suggests a mechanism for the critique literature's result. Shuffling an
annotation that does not lie in the path carrying the measured effect cannot
cost anything, so a null permutation result then says nothing about the
annotation. We do not claim this refutes [4]. Their fifteen models are mostly
supervised predictors evaluated on prediction metrics, ours is a conditional
generative model evaluated on held-out likelihood, and the comparison is not
like-for-like. Our narrower claim is that the check is inexpensive, is an algebraic
identity rather than an empirical argument, and is not routinely reported.


### 5.2 Relation to our previous work

The encoder of section 3.2 is not new. A single masked layer carrying the
annotation, with the pathway activities read off that layer, is the architecture
we introduced for dimensionality reduction [10] and then used for cell-level
pathway scoring [11], and it is the architecture the stability framework of [12]
evaluated. What this work adds is a masked and non-negative decoder, a covariate
term in pathway coordinates, and a covariate contrast as the estimand. Those
additions are what make the annotation measurable, and reassessing the earlier
evaluations in their light should be done explicitly.

The cell-level scoring protocol of [11] substituted the bottleneck layer for
gene expression in a standard single-cell workflow and compared latent activity
between stimulated and control cells with a Wilcoxon rank-sum test. On the same
Kang dataset we use here, that protocol ranks interferon pathways at the top,
and the conclusion drawn was that the model had learned the relevant signalling.
Our results suggest a weaker reading. The estimand there is a marginal
comparison of latent activity between observed groups of cells, and any
representation that preserves the dominant axis of variance will satisfy it,
because the stimulated and control cells separate in gene expression before any
model is fitted. Colouring a UMAP of pathway-space coordinates by condition
shows that the representation retained that separation. It does not show that
unit `p` corresponds to pathway `p`, which is the claim the architecture is
supposed to license. The decisive control, an arm in which the gene-to-pathway
map is shuffled while set sizes and overlap are preserved, was absent.

We can now put a number on what that architecture forgoes. With
the encoder masked and the decoder dense, which is the configuration of [10],
[11] and [12], annotation worth on the Kang contrast is +3.24 nats. Masking the
decoder as well raises it to +14.52, so the encoder mask alone captures 22% of
what the annotation is worth and the decoder mask supplies the other 78%
(section 4.7). The decoder mask costs 0.42 nats of fit, so this is not a
trade-off that the earlier work declined; it is a measurement that was not
available to it, because no grid in that line varied the decoder mask.

Running the ablation the other way sharpens the point. Dropping the encoder mask
and keeping the decoder's leaves annotation worth at +13.31 at matched
saturation, so the mask that line of work places moves the contrast by about a
nat while the mask it omits moves it by eleven. That is not an argument for
removing the encoder mask, which costs 28.26 nats of held-out loss and is what
makes a per-gene attribution specific to a pathway. It is an argument that
informing the first layer answers a different question from the one a covariate
contrast asks.

The stability framework of [12] supplied part of that control and reached a
conclusion consistent with ours. Across eight randomised connection densities
and 40 seeds, clustering coherence barely separated informed from randomised
wiring, at adjusted mutual information 0.62 to 0.68 against 0.61 to 0.64, and
randomised wiring reconstructed slightly better. That is the same direction as
our fit result: the annotation costs likelihood and serves something other than
fit. What [12] found the annotation does provide is stability, since informed
models produced consistent pathway rankings across seeds while randomised
controls did not. We regard that as a second, independent sense in which the
annotation matters, and it is complementary to ours rather than a substitute,
because a stable ranking of units is not yet a statement about any contrast
between conditions.

Three differences from that line bear on how the results compare. First, the
earlier models inform the encoder only and carry no conditional covariate term,
so the bypass identity of section 4.1 does not apply to them; their limitation
is the choice of estimand rather than the wiring. Second, they score marginal
activity over observed cells, which is composition-confounded whenever the
groups differ in cell-type mixture, while a covariate contrast conditions on
phenotype at cell level. Third, their evaluation metrics are clustering
coherence and rank stability, neither of which can quantify the annotation in
the units of the model's own objective; held-out likelihood against a
size-matched and overlap-matched shuffle can.

The practical conclusion we draw for this line of work is that informing the
encoder alone leaves the annotation unable to be measured by the usual read-outs,
and that the quantity to report is not how well the representation clusters but
how much the real annotation beats a structurally identical shuffle on an
estimand the annotation enters.
### 5.3 What distinguishes this architecture

Four elements of this work do not appear, as far as we can establish, in the
single-cell pathway-VAE literature.

The unmasked conditional term is the first. expiMap applies its mask to the
latent weights and leaves the condition matrix dense in the same linear layer
[2], the same structural pattern we describe. Its published read-out is latent
gene-program activity rather than a decoder-side covariate contrast, so the
unmasked term competes with the masked one for condition-associated variance
instead of absorbing all of it, and the consequence there is milder than ours.
We have found no statement anywhere that a read-out taken on the decoder side of
a covariate contrast bypasses the annotation entirely. The companion measurement
is that the decoder mask, not the encoder mask, is where the annotation earns its
keep: 78% of annotation worth against 22% (section 4.7).

Sign is the second. Neither VEGA nor expiMap constrains the sign of its masked
decoder weights, and we are not aware of a single-cell pathway VAE that does.
Our measurement that free signs leave a per-unit positive fraction with median
exactly 0.500, and 22% to 38% of units with no recoverable orientation, puts a
number on what that costs. It means a statement of the form "pathway activity
increased" is not directly interpretable in those models.

The gene-level control is the third. Submitting a model's fold changes to a
competitive enrichment test is widespread, and we show it is annotation-blind:
rank 1 in all eight Kang arms, shuffled maps included. The controlled comparison
in [6] reached the same conclusion independently, and our contribution is the
mechanism rather than the observation.

Donor-level testing of a deep model's latent units is the fourth. Squair et al.
[7] established the sampling-unit problem for differential expression, and
applying it to a biologically informed model's units is uncommon. It also cuts
against us, since this is where our biology thins out.

### 5.4 What the evidence supports

The architecture is the strongest claim. Three datasets, consistent direction, a
size-matched permutation control, four forward-pass identities and a
perturbation whose answer was fixed in advance. Both costs are measured and
reported.

Interferon as this compartment's signal is the second. It ranks 1 of 1,615 in
two independently fitted cohorts under a read-out that depends on the
annotation, with the interferon family occupying nine of Kang's 25 leading units
under the same architecture.

That the panel does not replicate is the third, and it is a result rather than a
failure. rho = -0.073 over 1,615 units means each cohort's response beyond
interferon is its own.

### 5.5 What the evidence does not support

The donor-level direction of the interferon effect is not yet defensible. It
survives in one cohort, fails in the other, reverses the baseline read-out's
ordering, and the model-free comparator corroborates neither. Three estimators
give three answers. Nothing resting on `Foster_2026`'s MGUS stage is defensible
either, since one donor occupies it. Nor are the four other units surviving in
`Foster_2026`: no prior reason to expect them, no support in the other cohort,
and a nested panel.

More healthy donors in both cohorts would settle most of this. Every discrepancy
in section 4.11 traces to the baseline, where `Zavidij_2020` has six healthy
donors with 3.7 units of scatter and `Foster_2026` has three with 0.3. A third
cohort with at least ten non-cancer donors would decide whether Foster's tight
baseline is real or an accident of n = 3, and it is the only addition that
would.

One question lies outside this design altogether. Whether the interferon signal
is a T-cell-intrinsic programme or a response to a marrow-wide stimulus cannot
be settled here, because the design conditions on phenotype within the T-cell
compartment and never observes another compartment.

### 5.6 Limitations

Congruence with Kang's known answer is at the level of the pathway family rather
than the leaf. `real_uncond` ranks the leaf 1st and `real_signed` ranks it 25th,
placing its parent at 3 and its grandparent at 2. The sign constraint provides
direction at the expense of leaf precision.

Most units are pinned. Between 71% and 80% saturate under the selected
architecture, so the ranking runs over the units that still have headroom, and a
biologically involved unit can be invisible here.

The biology is thinner than the architecture. Interferon ranks 1 in both cohorts
while the panel-wide correlation is -0.073, and the donor-level trend holds in
one cohort only.

The forward-pass identities establish that our read-out is the model's own
computation. They do not establish that the model recovered the generating
structure, which is the separate concern raised in [5].

Annotation worth is not a saturation-free measure, which limits where we can
use it. It rises at +1.72 nats per 10 points of pinned fraction, and lifting
fan-in into the selected architecture's saturation band recovers its apparent
deficit almost exactly (section 4.7). Comparisons between arms at similar
saturation are therefore sound, and comparisons across a large saturation change
are inflated by an amount we can estimate but not remove. That is why
`Foster_2026`, where saturation is unchanged between the two architectures,
carries the main claim, and why the Kang and `Zavidij_2020` figures are marked.
The target's rank behaves no better, improving monotonically with saturation, so
we do not select on it either.

We did not refit under fan-in, although the evidence now favours it: at matched
saturation it is equivalent on annotation worth while fitting 8.96 nats better
and leaving 43% rather than 71% of units pinned. Changing the initialiser would
mean refitting every model in the paper, and we report the finding instead.

### 5.7 Future work

A co-expression control is the most demanding test we have not run. The
permutation asks whether the curated map beats a shuffle of itself, not whether
curation beats structure learned from the data, and [6] reports that
data-derived co-expression modules match Hallmark exactly on the interferon
task. A `coexpr_*` arm pair, with modules built on the training split and
size-matched to the Reactome sets, would answer that and requires no change
outside the ablation script.

Simulating from a known gene-to-pathway map would separate reading the model
correctly from the model recovering the truth, which is the identifiability
question in [5].

Saturation now has a measured handle. Fan-in initialisation takes pinned units
from 71% to 43% at no cost in annotation worth once the comparison is
saturation-matched, so refitting the paper under it is the obvious next step and
would test whether a less saturated model also reads out more sharply. Rescaling
the pre-activation does not work: scaled down it is a no-op at convergence,
because training restores whatever magnitude the likelihood wants.

A third cohort with at least ten non-cancer donors would convert the interferon
finding from a hypothesis into a result.

## References

1. Seninge L, Anastopoulos I, Ding H, Stuart J. VEGA is an interpretable
   generative model for inferring biological network activity in single-cell
   transcriptomics. *Nat Commun* 12:5684 (2021). doi:10.1038/s41467-021-26017-0
2. Lotfollahi M, Rybakov S, Hrovatin K, Hediyeh-zadeh S, Talavera-Lopez C,
   Misharin AV, Theis FJ. Biologically informed deep learning to query gene
   programs in single-cell atlases. *Nat Cell Biol* 25:337-350 (2023).
3. Gut G, Stark SG, Raetsch G, Davidson NR. pmVAE: learning interpretable
   single-cell representations with pathway modules. bioRxiv (2021).
   doi:10.1101/2021.01.28.428664
4. Caranzano L, Pancotti C, Rollo C, Sartori F, Lio P, Fariselli P, Sanavia T.
   Sparsity is all you need: rethinking biological pathway-informed approaches
   in deep learning. arXiv:2505.04300 (2025); *Brief Bioinform* (2026).
5. The illusion of interpretability in biologically informed neural networks.
   bioRxiv (2026). doi:10.64898/2026.05.07.723544
6. Miliarakis G. Knowledge or sparsity? Controlled comparison of gene-set
   masked, randomised and co-expression-masked linear-decoder VAEs.
   github.com/gmiliarakis/knowledge-or-sparsity-sc-vae
7. Squair JW, Gautier M, Kathe C, Anderson MA, James ND, Hutson TH, Hudelle R,
   Qaiser T, Matson KJE, Barraud Q, Levine AJ, La Manno G, Skinnider MA,
   Courtine G. Confronting false discoveries in single-cell differential
   expression. *Nat Commun* 12:5692 (2021).
8. Kang HM, Subramaniam M, Targ S, Nguyen M, Maliskova L, McCarthy E, Wan E,
   Wong S, Byrnes L, Lanata CM, Gate RE, Mostafavi S, Marson A, Zaitlen N,
   Criswell LA, Ye CJ. Multiplexed droplet single-cell RNA-sequencing using
   natural genetic variation. *Nat Biotechnol* 36:89-94 (2018).
9. Zavidij O, Haradhvala NJ, Mouhieddine TH, Sklavenitis-Pistofidis R, Cai S,
   Reidy M, Rahmat M, Flaifel A, Ferland B, Su NK, Agius MP, Park J,
   Manier S, Bustoros M, Huynh D, Capelletti M, Berrios B, Liu CJ, He MX,
   Braggio E, Fonseca R, Maruvka YE, Guerriero JL, Goldman M, Van Allen EM,
   McCarroll SA, Azzi J, Getz G, Ghobrial IM. Single-cell RNA sequencing reveals
   compromised immune microenvironment in precursor stages of multiple myeloma.
   *Nat Cancer* 1:493-506 (2020).
10. Gundogdu P, Loucera C, Alamo-Alvarez I, Dopazo J, Nepomuceno-Chamorro IA.
    Integrating pathway knowledge with deep neural networks to reduce the
    dimensionality in single-cell RNA-seq data. *BioData Min* 15(1):1 (2022).
    doi:10.1186/s13040-021-00285-4
11. Gundogdu P, Paya-Milans M, Alamo-Alvarez I, Nepomuceno-Chamorro IA,
    Dopazo J, Loucera C. Cell-level pathway scoring comparison with a
    biologically constrained variational autoencoder. In: Pang J, Niehren J
    (eds) *Computational Methods in Systems Biology*, LNBI 14137, pp. 62-77.
    Springer (2023). doi:10.1007/978-3-031-42697-1_5
12. Fernandez-Malvido S, Esteban-Medina A, Gundogdu P, Dopazo J,
    Nepomuceno-Chamorro IA, Loucera C. A framework for evaluating the stability
    of learned representations in biologically-constrained models in
    single-cell. In: Lopez Fernandez A et al. (eds) *CIABiomed 2025*, LNBI
    16148, pp. 423-437. Springer (2026). doi:10.1007/978-3-032-10661-2_32
