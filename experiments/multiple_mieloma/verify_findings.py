"""Check every number quoted in FINDINGS.md against outputs/.

Writing a results document from memory gets tables wrong. This re-derives each
claim from the JSON and CSV the workflow wrote and prints a pass/fail line, so
the document cannot drift from the run that produced it.

    pixi run verify          # or: python verify_findings.py

Exits non-zero if anything fails.

Four things it does deliberately rather than conveniently.

**It recomputes rather than reads back where it can.** Unit ranks and the
cross-study agreement are recomputed from `unit_contrasts.csv`, not taken from
`summary.json`, so a disagreement between what the notebook printed and the
table it printed from would surface here.

**It checks the figures, not a re-derivation of them.** Every figure number is
taken from `fig_panel_data.csv`, which `04_figures.py` writes at draw time. A
figure that disagrees with this document therefore fails, which is the only way
to keep a caption honest once the figure is a separate artefact.

**It checks the costs as well as the wins.** The architecture is worse on
validation loss and saturates far harder than the one it replaces. Both are
asserted, so a future change that quietly improves the headline by making those
worse cannot pass.

**It counts its own checks.** `EXPECTED_CHECKS` is asserted at the end, which is
what stops a claim being quietly dropped from the prose and the verifier at the
same time.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import myeloma as mm

OUT = Path(os.environ.get("FINDINGS_OUT",
                          Path(__file__).resolve().parent / "outputs"))
ZAV, FOS = mm.ANALYSIS_STUDIES
ALPHA = 0.05
#: The unit the gene-level baseline points at in both cohorts. It is NOT a
#: known answer here -- myeloma has none -- so it is used only to compare
#: architectures against each other on a fixed target. Kang is where a
#: ground truth exists.
TARGET = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"
#: Where the permuted arm must not put that unit. The same floor Kang uses.
PERM_FLOOR = 300
IDENTITY_TOL = 1e-4

ok = True
n_checks = 0


def chk(label, doc, actual, tol=5e-3):
    global ok, n_checks
    n_checks += 1
    if isinstance(doc, str) or isinstance(actual, str):
        good = str(doc) == str(actual)
    elif isinstance(doc, (list, tuple, set)):
        good = list(doc) == list(actual)
    else:
        good = abs(float(doc) - float(actual)) <= tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {label}: doc={doc} actual={actual}")


def gate(label, condition, detail=""):
    """A claim that is pass/fail by construction rather than by comparison."""
    global ok, n_checks
    n_checks += 1
    ok &= bool(condition)
    suffix = f"  ({detail})" if detail else ""
    print(f"{'PASS' if condition else 'FAIL'}  {label}{suffix}")


def rng(label, lo, hi, actual):
    global ok, n_checks
    n_checks += 1
    good = lo <= float(actual) <= hi
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {label}: {actual} in [{lo}, {hi}]")


def need(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"missing {path}\nrun `snakemake` first")
    return path


def load_json(path: Path):
    return json.loads(need(path).read_text())


def arch(study: str, seed: int | None = None):
    sub = "" if seed is None else f"seed{seed}/"
    return load_json(OUT / "architecture" / study / f"{sub}summary.json")


def arch_seeds(study: str) -> list[int | None]:
    """Every seed this study has an architecture run for, canonical first."""
    base = OUT / "architecture" / study
    extra = sorted(int(p.name[4:]) for p in base.glob("seed*") if p.is_dir())
    return [None] + extra


def bio(study: str):
    return load_json(OUT / "biology" / study / "summary.json")


def donor(study: str):
    return load_json(OUT / "donor_test" / study / "summary.json")


def contrasts(study: str) -> pd.DataFrame:
    return pd.read_csv(need(OUT / "biology" / study / "unit_contrasts.csv"))


def endpoint_mean(study: str) -> pd.Series:
    """Each unit's endpoint contrast, averaged over the major lineages."""
    c = contrasts(study)
    end = bio(study)["endpoint_contrast"]
    w = c[c["contrast"] == end].pivot(index="unit", columns="lineage",
                                      values="member_shift_z")
    lin = [l for l in mm.MAJOR_SUBSETS if l in w.columns]
    return w[lin].mean(axis=1)


def unit_rank(study: str, unit: str, lineage: str) -> int:
    c = contrasts(study)
    end = bio(study)["endpoint_contrast"]
    s = (c[(c["contrast"] == end) & (c["lineage"] == lineage)]
         .set_index("unit")["member_shift_z"])
    return int(s.sort_values(ascending=False).index.get_loc(unit)) + 1


def panel(figure: str, panel_name: str, quantity: str, **where):
    d = pd.read_csv(need(OUT / "figures" / "fig_panel_data.csv"))
    sel = d[(d["figure"] == figure) & (d["panel"] == panel_name)
            & (d["quantity"] == quantity)]
    for key, value in where.items():
        sel = sel[sel[key].astype(str) == str(value)]
    if len(sel) != 1:
        sys.exit(f"fig_panel_data.csv: {len(sel)} rows for {figure}/"
                 f"{panel_name}/{quantity} {where}, expected exactly 1")
    return float(sel["value"].iloc[0])


# --------------------------------------------------------------------------- #
print("=" * 72)
print("R1  The architecture: does the annotation reach the contrast at all?")
print("=" * 72)

for study in (ZAV, FOS):
    a = arch(study)
    arms = a["arms"]
    label = study.split("_")[0]

    # The shipped decoder's covariate term is dense and added after the mask,
    # so almost none of the conditional effect is in the annotation basis.
    for lin, v in arms["real_shipped"]["lineages"].items():
        rng(f"R1 {label} {lin} shipped masked_share is small",
            0.0, 0.30, v["masked_share"])
    gate(f"R1 {label} pathway conditioning puts all of it inside the mask",
         all(abs(v["masked_share"] - 1.0) < 1e-6
             for s in ("pathway", "uncond", "signed") for m in ("real", "perm")
             for v in arms[f"{m}_{s}"]["lineages"].values()))

    # The old read-out barely notices the annotation. On Kang it ranks
    # interferon FIRST in all eight arms including the permuted ones; here it
    # degrades, but only from 1 to the high teens -- still the top ~1% of the
    # 50-programme panel in every arm. "Rank 1 everywhere" is a Kang fact and
    # is not asserted here.
    gene = sorted({v["gene_rank_interferon_best"]
                   for arm in arms.values()
                   for v in arm["lineages"].values()})
    gate(f"R1 {label} the gene-level read-out stays near the top in every arm, "
         "permuted included",
         max(gene) <= 20, f"ranks {gene[0]} to {gene[-1]} over 16 arm-lineages")

    # The unit ranking is the read-out that does depend on the annotation.
    shipped = [v["unit_rank_target"]
               for v in arms["real_shipped"]["lineages"].values()]
    signed = [v["unit_rank_target"]
              for v in arms["real_signed"]["lineages"].values()]
    perm = [v["unit_rank_target"]
            for v in arms["perm_signed"]["lineages"].values()]
    gate(f"R1 {label} the selected architecture ranks the target in the top 5%",
         max(signed) <= 0.05 * a["units"],
         f"ranks {signed} of {a['units']:,}")
    gate(f"R1 {label} the permuted map does not",
         min(perm) > PERM_FLOOR, f"ranks {perm} > {PERM_FLOOR}")
    perm_shipped = [v["unit_rank_target"]
                    for v in arms["perm_shipped"]["lineages"].values()]
    gate(f"R1 {label} the shipped decoder's unit ranking is uninformative",
         min(shipped) > 100,
         f"real {shipped} vs permuted {perm_shipped}")

    # Both directions of the trade, so the cost cannot be dropped quietly.
    gate(f"R1 {label} the architecture costs validation loss",
         arms["real_signed"]["best_val"] > arms["real_shipped"]["best_val"],
         f"{arms['real_signed']['best_val']:.1f} vs "
         f"{arms['real_shipped']['best_val']:.1f} nats")
    gate(f"R1 {label} and the annotation is worth more under it",
         a["annotation_worth"]["signed"] > a["annotation_worth"]["shipped"],
         f"+{a['annotation_worth']['signed']:.2f} vs "
         f"+{a['annotation_worth']['shipped']:.2f} nats")

# The sharpest single statement, and it holds in one cohort only: on
# `Zavidij_2020` the shipped decoder ranks the target DEAD LAST while a
# shuffled annotation ranks it mid-table, so shuffling the map *improved* it.
# `Foster_2026` does not do this -- there the shipped decoder puts it at 144
# and 102 against the shuffle's 560 and 563 -- so the claim is made about
# Zavidij and Kang and not about both cohorts.
zav = arch(ZAV)["arms"]
gate("R1 on Zavidij, shuffling the annotation IMPROVES the shipped ranking",
     min(v["unit_rank_target"]
         for v in zav["real_shipped"]["lineages"].values())
     > max(v["unit_rank_target"]
           for v in zav["perm_shipped"]["lineages"].values()),
     f"real {[v['unit_rank_target'] for v in zav['real_shipped']['lineages'].values()]}"
     f" vs permuted "
     f"{[v['unit_rank_target'] for v in zav['perm_shipped']['lineages'].values()]}")
fos = arch(FOS)["arms"]
gate("R1 and on Foster it does not, which the document states rather than "
     "averaging away",
     min(v["unit_rank_target"]
         for v in fos["real_shipped"]["lineages"].values())
     < min(v["unit_rank_target"]
           for v in fos["perm_shipped"]["lineages"].values()),
     f"real {[v['unit_rank_target'] for v in fos['real_shipped']['lineages'].values()]}"
     f" vs permuted "
     f"{[v['unit_rank_target'] for v in fos['perm_shipped']['lineages'].values()]}")

# Stability across seeds is the reason Zavidij runs four of them.
seeds = arch_seeds(ZAV)
gate("R1 Zavidij ran more than one architecture seed",
     len(seeds) > 1, f"{len(seeds)} seeds")
per_seed = [max(v["unit_rank_target"]
                for v in arch(ZAV, s)["arms"]["real_signed"]["lineages"].values())
            for s in seeds]
gate("R1 and every seed keeps the target in the top 5%",
     max(per_seed) <= 0.05 * arch(ZAV)["units"],
     f"worst rank per seed {per_seed}")
worth = [arch(ZAV, s)["annotation_worth"]["signed"] for s in seeds]
gate("R1 and the annotation is worth something in every seed",
     min(worth) > 0, f"{[round(w, 2) for w in worth]} nats")

print()
print("=" * 72)
print("R2  The forward-pass identities, both studies")
print("=" * 72)
for study in (ZAV, FOS):
    label = study.split("_")[0]
    for name, got in arch(study)["identities"].items():
        if not got["rows"]:
            continue
        gate(f"R2 {label} {name}", got["max"] < IDENTITY_TOL,
             f"{got['max']:.2e} over {got['rows']} rows")
    # The one that is exactly zero rather than merely small: with the encoder
    # route closed, holding mu fixed and re-encoding are the same computation,
    # so the choice of counterfactual convention stops mattering.
    gate(f"R2 {label} counterfactual == re-encoded is EXACTLY zero once the "
         "encoder route is closed",
         arch(study)["identities"]
         ["counterfactual_vs_reencoded_maxdev"]["max"] == 0.0)

print()
print("=" * 72)
print("R3  Sign: does a unit rising raise its genes?")
print("=" * 72)
for study in (ZAV, FOS):
    label = study.split("_")[0]
    arms = arch(study)["arms"]
    rng(f"R3 {label} shipped dec_out is about half negative", 0.40, 0.60,
        arms["real_shipped"]["dec_out_frac_negative"])
    rng(f"R3 {label} and its per-unit positive fraction has median 0.5",
        0.45, 0.55, arms["real_shipped"]["frac_pos_median"])
    gate(f"R3 {label} the selected architecture has no negative decoder weight",
         arms["real_signed"]["dec_out_frac_negative"] == 0.0)
    gate(f"R3 {label} nor any negative encoder weight",
         all(a["encoder_frac_negative"] == 0.0 for a in arms.values()))
    for lin, v in arms["real_signed"]["lineages"].items():
        chk(f"R3 {label} {lin} unit-vs-gene sign agreement", 1.0,
            v["sign_agree_unit_vs_genes"], tol=1e-9)
    # Recomputed from units.csv rather than read from a summary key. A unit
    # whose Reactome set shares no measured gene with the panel has no member
    # direction to agree with, so counting it scores a guaranteed mismatch for
    # a question never asked -- on Kang, where 9 of 1,615 sets are empty, that
    # held the statistic at 0.9944 on a fit where every unit that writes
    # anything agrees perfectly. This panel has none, and recomputing says so
    # whether or not the run that produced the summary carried the key.
    u = pd.read_csv(need(OUT / "architecture" / study / "units.csv"))
    u = u[u["arm"] == "real_signed"]
    empty = int((u[u["lineage"] == u["lineage"].iloc[0]]["n_genes"] == 0).sum())
    gate(f"R3 {label} every unit has at least one measured member gene",
         empty == 0, f"{empty} member-less units")
    b = bio(study)
    gate(f"R3 {label} the biology fit agrees: no unit writes against itself",
         b["units_sign_disagree"] == 0)
    gate(f"R3 {label} and no negative weight on either side of it",
         b["dec_out_frac_negative"] == 0.0
         and b["encoder_frac_negative"] == 0.0)

print()
print("=" * 72)
print("R4  The cost: saturation, reported rather than hidden")
print("=" * 72)
def saturation(study: str, arm: str) -> float:
    """Fraction of that arm's units sitting in the flat tail for most cells.

    From `architecture/units.csv`, so the shipped and selected decoders are
    measured on the same cells by the same code -- the only comparison that
    makes "it saturates harder" mean anything.
    """
    u = pd.read_csv(need(OUT / "architecture" / study / "units.csv"))
    u = u[u["arm"] == arm]
    if "lineage" in u.columns:
        u = u[u["lineage"] == u["lineage"].iloc[0]]
    return float((u["saturated_frac"] > 0.5).mean())


for study in (ZAV, FOS):
    label = study.split("_")[0]
    b = bio(study)
    frac = b["units_saturated_gt_half"] / b["mask_units"]
    shipped_sat, signed_sat = saturation(study, "real_shipped"), saturation(
        study, "real_signed")
    print(f"      {label}: {b['units_saturated_gt_half']:,} of "
          f"{b['mask_units']:,} units ({frac:.0%}) saturated in more than half "
          f"their cells; median |delta_post| {b['delta_post_abs_median']:.2e}")
    print(f"      {label}: the same measure inside act 1 -- shipped "
          f"{shipped_sat:.0%}, selected {signed_sat:.0%}")
    gate(f"R4 {label} saturation is carried in summary.json",
         "units_saturated_gt_half" in b and "delta_post_abs_median" in b)
    # Not "harder than shipped" -- that is true on Zavidij (22% -> 80%) and
    # false on Foster (76% -> 75%), where the shipped decoder already saturates
    # most of its units at 125k cells. What holds in both is that most units
    # end up pinned, which is the thing that bounds the read-out.
    gate(f"R4 {label} most units end up pinned under the selected architecture",
         signed_sat > 0.5,
         f"{signed_sat:.0%} of units, against {shipped_sat:.0%} shipped")

print()
print("=" * 72)
print("R5  The biology, over the model's own units")
print("=" * 72)
for study in (ZAV, FOS):
    label = study.split("_")[0]
    b = bio(study)
    gate(f"R5 {label} the notebook fitted the architecture R1 selected",
         (b["architecture"]["cov_mode"] == "pathway"
          and b["architecture"]["cov_in_encoder"] == "False"
          and b["architecture"]["nonneg_decoder"] == "True"))
    gate(f"R5 {label} no covariate block was dropped",
         not b["covariate_blocks_dropped"],
         f"entered {b['covariate_blocks_entered']}")
    # Recomputed from the table rather than read back.
    top = list(endpoint_mean(study).sort_values(ascending=False).index[:10])
    chk(f"R5 {label} top 10 units, recomputed from unit_contrasts.csv",
        b["top_units"], [mm.short(p) for p in top])
    gate(f"R5 {label} the lead unit is the first of them",
         b["lead_unit"] == top[0])

# Whether the two cohorts agree is a result in itself, and it has been
# overturned once already by changing the sampling unit -- so it is recomputed
# here from both studies' tables and checked against the figure.
ea, eb = endpoint_mean(ZAV), endpoint_mean(FOS)
shared = ea.index.intersection(eb.index)
rho_units = float(stats.spearmanr(ea[shared], eb[shared]).statistic)
chk("R5 cross-study agreement over units, figure against recomputation",
    round(rho_units, 3),
    round(panel("comparison", "a", "rho_cross_study"), 3), tol=1e-3)
chk("R5 and its value, which FINDINGS R2.1 quotes", -0.073, rho_units,
    tol=5e-3)

# The headline of act 2: one unit at the top of both independently-fitted
# cohorts, and no agreement anywhere else. Both halves are checked, because
# either one alone would misrepresent it.
for study in (ZAV, FOS):
    label = study.split("_")[0]
    chk(f"R5 {label} the leading unit", TARGET,
        endpoint_mean(study).idxmax())
gate("R5 and the panel-wide rankings do not agree",
     abs(rho_units) < 0.15, f"rho {rho_units:+.3f} over {len(shared):,} units")
# Every OTHER unit in each cohort's top 8 must rank poorly in the other, or
# "one shared unit and nothing else" is the wrong description.
ra, rb = ea.rank(ascending=False), eb.rank(ascending=False)
for src, dst, other, name in ((ea, eb, rb, "Zavidij"), (eb, ea, ra, "Foster")):
    others = [u for u in src.sort_values(ascending=False).index[:8]
              if u != TARGET]
    worst = min(int(other[u]) for u in others)
    gate(f"R5 {name}'s other leading units are not shared",
         worst > 100, f"best rank in the other cohort is {worst} of 1,615")

print()
print("=" * 72)
print("R6  The donor-level test, where the sampling unit is the donor")
print("=" * 72)
for study in (ZAV, FOS):
    label = study.split("_")[0]
    d = donor(study)
    gate(f"R6 {label} the donor fit used the same architecture",
         d["architecture"] == bio(study)["architecture"])
    gate(f"R6 {label} the power floor is recorded for every lineage",
         set(d["power_floor"]) == set(
             l for l in mm.MAJOR_SUBSETS if l in d["donors_per_stage"]),
         str({k: v["donors"] for k, v in d["power_floor"].items()}))
    # The floor has to be below alpha for an empty result to mean anything.
    # BH multiplies the smallest attainable p by the number of tests, so at
    # 10,000 draws over 1,262 units the best q was 0.126 and the primary test
    # could not have returned a single unit whatever the data said. 100,000
    # draws put it at 0.013. This gate is what stops that regressing.
    gate(f"R6 {label} the trend test can actually reach q < {ALPHA}",
         all(f["trend_can_reach_q05"] for f in d["power_floor"].values()),
         ", ".join(f"{l} best q {f['best_attainable_q_trend']}"
                   for l, f in d["power_floor"].items()))
    # And the endpoint test cannot, in either cohort -- too few donors per
    # stage. It is reported beside the trend precisely so that its silence is
    # read as "impossible" rather than "negative".
    gate(f"R6 {label} the endpoint test is known not to, and says so",
         not any(f["endpoint_can_reach_q05"]
                 for f in d["power_floor"].values()),
         ", ".join(f"{l} best q {f['best_attainable_q_endpoint']}"
                   for l, f in d["power_floor"].items()))
    n_model = len(d["trend_both_lineages_model"])
    n_both = len(d["trend_both_lineages_both_estimators"])
    print(f"      {label}: {n_model} units with a rising donor-level trend in "
          f"both lineages, {n_both} of which pseudobulk agrees on")
    gate(f"R6 {label} the two estimators' agreement is a subset, not a claim "
         "of its own",
         set(d["trend_both_lineages_both_estimators"])
         <= set(d["trend_both_lineages_model"]))
    gate(f"R6 {label} BH is over units and the document says so",
         "nested" in d["nesting_note"])

    # The donor-level interferon numbers R3.2 quotes, recomputed per lineage.
    t = pd.read_csv(need(OUT / "donor_test" / study / "donor_test.csv"))
    ifn = t[(t["estimator"] == "model") & (t["unit"] == TARGET)]
    rho = {r["lineage"]: round(float(r["rho_trend"]), 2)
           for _, r in ifn.iterrows()}
    q = {r["lineage"]: float(r["q_trend"]) for _, r in ifn.iterrows()}
    print(f"      {label} interferon donor trend: rho {rho}, q "
          f"{ {k: round(v, 4) for k, v in q.items()} }")
    survives = all(v < ALPHA for v in q.values())
    gate(f"R6 {label} the interferon unit's donor-level verdict",
         survives == (study == FOS),
         "survives in Foster and not in Zavidij, which inverts the retired "
         f"gene-level read-out; here q = { {k: round(v, 4) for k, v in q.items()} }")

    # The two estimators agree over all units even where their thresholded
    # sets are disjoint. The correlation is the honest summary; R3.4 says so.
    for lin in mm.MAJOR_SUBSETS:
        m = t[(t["estimator"] == "model")
              & (t["lineage"] == lin)].set_index("unit")["rho_trend"]
        pb = t[(t["estimator"] == "pseudobulk")
               & (t["lineage"] == lin)].set_index("unit")["rho_trend"]
        both = m.index.intersection(pb.index)
        if len(both) < 50:
            continue
        r = float(stats.spearmanr(m[both], pb[both]).statistic)
        gate(f"R6 {label} {lin} model and pseudobulk agree over all units",
             r > 0.2, f"rho {r:+.3f} over {len(both):,} units")

print()
#: Bumped whenever a claim is added or removed. A verifier that silently stops
#: checking something is worse than no verifier.
EXPECTED_CHECKS = 88
print("=" * 72)
if n_checks != EXPECTED_CHECKS:
    ok = False
    print(f"FAIL  check count: expected {EXPECTED_CHECKS}, ran {n_checks} -- "
          "a claim was added or dropped without updating EXPECTED_CHECKS")
print(f"{'ALL PASS' if ok else 'FAILURES ABOVE'}  ({n_checks} checks)")
sys.exit(0 if ok else 1)
