# Prompt: handing the paper figures to a domain specialist

Attach the three figures, in this order:

1. `outputs/figures/fig3_comparison.pdf` — the two cohorts against each other
2. `outputs/figures/fig1_Zavidij_2020.pdf` — one cohort in full
3. `outputs/figures/fig2_Foster_2026.pdf` — the other, identical layout

There is one variant now; the depth-adjusted twin is gone.

> **Figure 3 goes first on purpose.** It is the only one of the three that
> carries replication evidence. Figures 1 and 2 are single-cohort read-outs, and
> a specialist handed one of those first will reliably over-read it — panels a
> to c of those pages are estimated from cells, and every donor sits at exactly
> one stage.

---

You are a specialist in T-cell immunology and plasma-cell dyscrasias. You are
being shown **three figures and nothing else** — no underlying data, no model,
no publications. Interpret them on their own terms and be explicit about the
boundary between what they show and what you are inferring.

## What the figures are

Pathway-unit activity in bone-marrow T cells across the multiple myeloma
precursor sequence — non-cancer (`Non`), monoclonal gammopathy of undetermined
significance (`MGUS`), smouldering myeloma (`SMM`), symptomatic myeloma (`MM`) —
in **two independent patient cohorts, fitted separately under one identical
design**.

The model is a variational autoencoder whose decoder is masked by Reactome:
each of its 1,615 internal units writes only to the genes of one annotated
pathway. The covariate — disease stage, interacted with T-cell lineage — enters
*inside* that masked layer, so the effect of being at one stage rather than
another is **one signed number per annotated unit**, obtained by differencing
two columns of a weight matrix. No cell is re-encoded and no cell is
re-weighted.

Positive means the unit moved up, and under this architecture that means its
member genes moved up: the decoder's weights are constrained non-negative, so a
unit's direction is its genes' direction. That constraint is the reason a sign
here can be read at all.

Colour is effect size. **There is no significance mark on any unit panel**, and
that is deliberate — see trap 1.

## How the columns work

Every heat-map column is a contrast **against the same `Non` baseline** —
`Non→MGUS`, `Non→SMM`, `Non→MM`. A row therefore reads left to right as a
trajectory: how far that unit has moved from healthy, by that stage. Three
columns rising means it keeps moving away; a dip in the middle column means it
moved, came back, and moved again.

CD4 and CD8 are separate blocks. They come from **one model**, not two: lineage
is interacted with stage in the design, so each block is a contrast within that
lineage with cell-type composition held fixed.

## What you are being asked

### 1. Figure 3, panel by panel

- **Panel a** places every shared unit by its endpoint effect in each cohort.
  Which units sit high in **both**? Which are high in one and flat or negative
  in the other? Name them.
- **Panel b** takes each cohort's ten leading units and shows where the other
  cohort ranks them. A filled mark is the unit's rank at home, an open mark its
  rank abroad, on a log axis over all 1,615. A short line is agreement.
- **Panel c** is the same comparison with the **donor** as the sampling unit.
  This is the panel that bears on replication. Where a and c disagree, c takes precedence.
- **Panel d** is methodological, not biological: where four candidate
  architectures put one fixed reference unit. Ignore it for biology.
- **Panel e** is the two cohorts in words.

### 2. Then figures 1 and 2

For each cohort: which units are moving, from which stage, and is the movement
progressive or concentrated at one transition? Where the two cohorts disagree,
say whether the disagreement is in **magnitude**, in **timing**, or in **sign** —
those have different biological readings.

Panel d of each page is the donor-level view of that cohort's own leading units.
Read it against panel a: a unit strong in a and flat in d is a unit whose
apparent effect is not carried by donors.

### 3. The interpretation

- What coordinated T-cell biology would produce the shared pattern, if it is
  real? Name the cell states and mechanisms you would expect alongside it.
- What would produce the cohort-specific patterns? Distinguish biology from
  cohort composition, treatment history and disease burden where you can.
- Which of these would be **novel** against what is established for myeloma
  precursor states, and which are expected?
- For each claim: what is the single most informative experiment or measurement
  that would confirm or kill it?

## Seven traps, stated so you can avoid them

1. **There is no test over units, by design, so do not ask for one.** Reactome
   is strictly nested — `interferon alpha beta signaling` (59 genes) sits inside
   `interferon signaling` (188) inside `cytokine signaling in immune system`
   (682) — so an enrichment or overlap statistic over 1,615 units is not valid,
   and a count of "how many units responded" counts one family however deep the
   hierarchy happens to be. **Read families, not rows.** The only p-value in
   front of you is donor-level, in panel d of figures 1 and 2 and panel c of
   figure 3.
2. **Cells are not donors.** Panels a to c of figures 1 and 2 are estimated from
   cells, and every donor belongs to exactly one stage, so they contain no
   donor-level variance at all. A large effect there is **not** evidence it
   would appear in new patients. This has bitten this analysis before: under the
   previous read-out, interferon looked shared across both cohorts, and the
   donor-level test found it strong in one and absent in the other, with that
   cohort's apparent MGUS peak resting on a single donor.
3. **A saturated unit writes nothing.** The decoder squashes each unit through
   `tanh`, and under this architecture most units sit in the flat tail for most
   cells. A pinned unit cannot register a change whatever its weights say, so a
   unit can be biologically involved and simply absent from these rankings.
   Absence of a unit here is weak evidence of absence in the biology.
4. **Units are pathway *annotations*, not mechanisms.** A unit is a decoder
   coordinate whose weights are confined to one Reactome set. A high value says
   "the genes of this set moved coherently", never "this pathway was activated
   through this route". Do not promote a unit name to a signalling claim.
5. **Values are not a common absolute scale across contrasts.** The statistic is
   standardised per unit by its member count, so it is comparable down a column
   and across the CD4/CD8 blocks, but a +0.02 in one column and a +0.02 in
   another do not mean the same number of molecules.
6. **"Leading in one cohort only" is often a power statement.** The two cohorts
   differ by more than tenfold in cells and have very unequal donor counts per
   stage — `Foster_2026` has a single MGUS donor. Before calling a unit
   cohort-specific, check panel a of figure 3 for whether its effect has the
   same sign and similar magnitude in the other cohort.
7. **Composition is already held fixed, so do not explain a result with it.**
   The contrast is a weight difference at fixed phenotype — no cell is
   re-encoded or re-weighted. "More exhausted T cells at MM" cannot produce
   these numbers. What *can* is a shift in unit activity within the phenotypes
   that are present.

## What to return

A short written assessment, organised as the three questions above. For every
biological claim, say which panel it rests on and which of the seven traps you
had to rule out to make it. Where the figures do not support a claim you would
otherwise want to make, say that instead of hedging it.
