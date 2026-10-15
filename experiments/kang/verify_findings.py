"""Check every number quoted in FINDINGS.md against outputs/.

Writing a results document from memory gets tables wrong. This re-derives each
claim from the JSON and CSV the workflow wrote and prints a pass/fail line, so
the document cannot drift from the run that produced it.

    pixi run verify          # or: python verify_findings.py

Exits non-zero if anything fails.

Three things it does deliberately rather than conveniently.

**The known answer is a gate, not a report.** Kang is interferon-beta against
control, so `REACTOME_INTERFERON_ALPHA_BETA_SIGNALING` is fixed before the
experiment runs. A model that cannot find it has failed, whatever else it
scores, and the gates below say so rather than printing a number for a reader
to judge.

**It recomputes rather than reads back where it can.** The unit ranks are
recomputed from `unit_contrasts.csv`, not taken from `summary.json`, so a
disagreement between what the notebook printed and the table it printed from
would surface here.

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

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("FINDINGS_OUT", HERE / "outputs"))

TARGET = "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING"
PARENT = "REACTOME_INTERFERON_SIGNALING"
#: Where the permuted arm must NOT put the target. Chosen before the run: a
#: shuffled map should leave the right answer somewhere unremarkable, and
#: "outside the top 300 of 1,615" is unremarkable by any reading.
PERM_FLOOR = 300
#: Every forward-pass identity must hold at float32 precision.
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


def need(path: Path):
    if not path.exists():
        sys.exit(f"missing {path}\nrun `snakemake` first")
    return path


def rank_of(series: pd.Series, name: str) -> int:
    return int(series.sort_values(ascending=False).index.get_loc(name)) + 1


# --------------------------------------------------------------------------- #
arch = json.loads(need(OUT / "architecture" / "summary.json").read_text())
units_run = json.loads(need(OUT / "units" / "summary.json").read_text())
arch_units = pd.read_csv(need(OUT / "architecture" / "units.csv"))
unit_table = pd.read_csv(need(OUT / "units" / "unit_contrasts.csv"),
                         index_col="unit")
arms = arch["arms"]
enc = json.loads(need(OUT / "encoder_ablation" / "summary.json").read_text())
enc_arms = enc["arms"]
enc_worth = enc["annotation_worth"]
satg = json.loads(need(OUT / "saturation" / "summary.json").read_text())
sat_arms = satg["arms"]
sat_worth = satg["annotation_worth"]

print("=" * 72)
print("R1  The architecture: does the annotation reach the contrast at all?")
print("=" * 72)

# The shipped decoder's covariate term is dense and added after the mask, so
# almost none of the conditional effect is expressed in the annotation basis.
rng("R1 shipped masked_share is small", 0.0, 0.30,
    arms["real_shipped"]["masked_share"])
gate("R1 pathway-space conditioning puts all of it inside the mask",
     all(abs(arms[f"{m}_{s}"]["masked_share"] - 1.0) < 1e-6
         for s in ("pathway", "uncond", "signed") for m in ("real", "perm")))

# The old read-out ranked the target first in every arm, permuted included.
# That is the evidence it never saw the annotation.
gene_ranks = sorted({a["gene_rank_target"] for a in arms.values()})
chk("R1 gene-level rank of the target, over all 8 arms", [1], gene_ranks)

# The unit ranking is the read-out that does depend on the annotation.
real_signed = arms["real_signed"]["unit_rank_target"]
perm_signed = arms["perm_signed"]["unit_rank_target"]
gate("R1 shipped ranks the target in the bottom decile of its own units",
     arms["real_shipped"]["unit_rank_target"] > 0.9 * arch["units"],
     f"rank {arms['real_shipped']['unit_rank_target']} of {arch['units']}")
gate("R1 the selected architecture ranks it in the top 2%",
     real_signed <= 0.02 * arch["units"],
     f"rank {real_signed} of {arch['units']}")
gate("R1 the permuted map does not",
     perm_signed > PERM_FLOOR, f"rank {perm_signed} > {PERM_FLOOR}")

# The architecture costs likelihood and buys annotation sensitivity. Both
# directions are checked, because reporting only the second would be a sales
# pitch rather than a result.
gate("R1 the architecture costs validation loss",
     arms["real_signed"]["best_val"] > arms["real_shipped"]["best_val"],
     f"{arms['real_signed']['best_val']:.1f} vs "
     f"{arms['real_shipped']['best_val']:.1f} nats")
gate("R1 and the annotation is worth more under it",
     arch["annotation_worth"]["signed"] > arch["annotation_worth"]["shipped"],
     f"+{arch['annotation_worth']['signed']:.2f} vs "
     f"+{arch['annotation_worth']['shipped']:.2f} nats")

print()
print("=" * 72)
print("R2  The four forward-pass identities")
print("=" * 72)
for name, got in arch["identities"].items():
    if not got["rows"]:
        continue
    gate(f"R2 {name}", got["max"] < IDENTITY_TOL,
         f"{got['max']:.2e} over {got['rows']} rows")
# The one that is exactly zero rather than merely small: with the encoder route
# closed, holding mu fixed and re-encoding are the same computation, so the
# choice of counterfactual convention stops mattering.
gate("R2 counterfactual == re-encoded is EXACTLY zero once the encoder route "
     "is closed",
     arch["identities"]["counterfactual_vs_reencoded_maxdev"]["max"] == 0.0)

print()
print("=" * 72)
print("R3  Sign: does a unit rising raise its genes?")
print("=" * 72)
rng("R3 shipped dec_out is about half negative", 0.40, 0.60,
    arms["real_shipped"]["dec_out_frac_negative"])
gate("R3 the selected architecture has no negative decoder weight",
     arms["real_signed"]["dec_out_frac_negative"] == 0.0)
gate("R3 nor any negative encoder weight",
     all(a["encoder_frac_negative"] == 0.0 for a in arms.values()))
chk("R3 unit-vs-gene sign agreement under the selected architecture",
    1.0, arms["real_signed"]["sign_agree_unit_vs_genes"], tol=1e-9)
rng("R3 and it is near chance without it", 0.40, 0.60,
    arms["real_shipped"]["sign_agree_unit_vs_genes"])

# Nine Reactome sets share no measured gene with this dataset's panel. They are
# a property of the annotation-by-panel intersection, identical in every arm,
# and they are reported rather than dropped -- an earlier version folded them
# into the sign statistic and held it at 0.9944 on a fit where every unit that
# writes anything agrees perfectly.
empty = {a["units_without_members"] for a in arms.values()}
gate("R3 the member-less units are the same in every arm",
     len(empty) == 1, f"{empty.pop()} of {arch['units']:,}")
gate("R3 and they are excluded from the sign statistic rather than counted "
     "against it",
     arms["real_signed"]["sign_agree_unit_vs_genes"] == 1.0)

print()
print("=" * 72)
print("R4  The read-out run, and the known answer")
print("=" * 72)
arch_flags = {k: str(v) for k, v in units_run["architecture"].items()}
gate("R4 the notebook fitted the architecture R1 selected",
     (arch_flags["cov_mode"] == "pathway"
      and arch_flags["cov_in_encoder"] == "False"
      and arch_flags["nonneg_decoder"] == "True"),
     str(units_run["architecture"]))
gate("R4 pathway_contrast equals the forward pass",
     units_run["pathway_contrast_vs_forward"] < IDENTITY_TOL,
     f"{units_run['pathway_contrast_vs_forward']:.2e}")
gate("R4 the contrast is cell-independent",
     units_run["shift_spread_across_cells"] < IDENTITY_TOL,
     f"{units_run['shift_spread_across_cells']:.2e}")
gate("R4 no unit writes against its own shift",
     units_run["units_sign_disagree"] == 0)
chk("R4 the notebook counts the same member-less units as act 1",
    arms["real_signed"]["units_without_members"],
    units_run["units_without_members"], tol=0)

# Recomputed from the table rather than read back from the summary.
recomputed = rank_of(unit_table["member_shift_z"], TARGET)
chk("R4 target rank, recomputed from unit_contrasts.csv",
    units_run["unit_rank_target"], recomputed, tol=0)
gate("R4 the known answer is in the top 2% of units",
     recomputed <= 0.02 * units_run["units"],
     f"rank {recomputed} of {units_run['units']:,}")
gate("R4 and its Reactome parent is beside it",
     units_run["unit_rank_parent"] <= 0.02 * units_run["units"],
     f"{PARENT.replace('REACTOME_', '')} at "
     f"{units_run['unit_rank_parent']}")
gate("R4 the target's shift is positive, as a stimulation should be",
     float(unit_table.loc[TARGET, "delta_pre"]) > 0,
     f"delta_pre {float(unit_table.loc[TARGET, 'delta_pre']):+.4f}")
chk("R4 the covariate shift on the target unit", 2.60,
    float(unit_table.loc[TARGET, "delta_pre"]), tol=0.01)

# The three nesting levels FINDINGS R1.5 quotes, recomputed. Reactome nests, so
# these are one finding at three resolutions and the document says so -- the
# check is that the ranks it prints are the ranks in the table.
GRANDPARENT = "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM"
ranks = {name: rank_of(unit_table["member_shift_z"], name)
         for name in (TARGET, PARENT, GRANDPARENT)}
chk("R4 the interferon family's three nesting levels",
    [25, 3, 2], [ranks[TARGET], ranks[PARENT], ranks[GRANDPARENT]])

# The nine units of the top 25 that FINDINGS R1.5 names as one family. Checked
# as a set membership rather than a count, because a count over nested units is
# exactly what this experiment refuses to report.
IFN_FAMILY = [
    "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM",
    "REACTOME_INTERFERON_SIGNALING",
    "REACTOME_OAS_ANTIVIRAL_RESPONSE",
    "REACTOME_TRAFFICKING_AND_PROCESSING_OF_ENDOSOMAL_TLR",
    "REACTOME_TRAF3_DEPENDENT_IRF_ACTIVATION_PATHWAY",
    "REACTOME_STING_MEDIATED_INDUCTION_OF_HOST_IMMUNE_RESPONSES",
    "REACTOME_ZBP1_DAI_MEDIATED_INDUCTION_OF_TYPE_I_IFNS",
    "REACTOME_INFLAMMASOMES",
    "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING",
]
top25 = set(unit_table["member_shift_z"].sort_values(ascending=False).index[:25])
missing = [u for u in IFN_FAMILY if u not in top25]
gate("R4 every interferon unit FINDINGS names is in the top 25",
     not missing, f"{len(IFN_FAMILY)} named, missing {missing}")
chk("R4 the strongest single unit, named in FINDINGS as not interferon biology",
    "REACTOME_SIGNALING_BY_GPCR", units_run["top_unit"])

# Integrated gradients: the genes FINDINGS quotes, in order, and their share.
ig = pd.read_csv(need(OUT / "units" / "ig_target.csv"), index_col="gene")
chk("R4 the genes the interferon unit reads the pathway through",
    ["ISG20", "IFI6", "IFIT1", "HLA-B", "ISG15",
     "IFIT3", "MX1", "IFIT2", "HLA-C", "SOCS1"],
    list(ig.index[:10]))
chk("R4 and their share of the unit's total attribution", 0.75,
    units_run["ig_top10_share"], tol=0.01)

print()
print("=" * 72)
print("R5  What the architecture costs: saturation")
print("=" * 72)
# Reported, not gated. It is a real limit on what can be found, and burying it
# in a log would be the kind of omission this whole rebuild exists to undo.
sat = units_run["units_saturated_gt_half"]
print(f"      units saturated in more than half their cells: {sat:,} of "
      f"{units_run['units']:,} ({sat / units_run['units']:.0%})")
print(f"      median |delta_post|: {units_run['delta_post_abs_median']:.2e}")
print(f"      units writing nothing at all: {units_run['units_dead']:,}")
gate("R5 saturation is measured and carried in summary.json",
     "units_saturated_gt_half" in units_run and "units_dead" in units_run)

# The architecture arms the notebook quoted must be the ones act 1 wrote.
quoted = units_run.get("architecture_arms", {})
gate("R5 the notebook quoted act 1 rather than recomputing it",
     all(quoted[a]["unit_rank"] == arms[a]["unit_rank_target"]
         for a in quoted),
     f"{len(quoted)} arms cross-checked")

print()
print("=" * 72)
print("R6  Do the options act 1 holds fixed earn their place?")
print("=" * 72)
# `01_architecture.py` pins four options in BASE while it varies the
# conditioning path, so nothing in act 1 is evidence for them.
# `01b_encoder_ablation.py` pins the conditioning and moves them instead.

# The two grids must agree where they overlap, which is what sharing one data
# loader buys. If these drift, the two tables cannot be read side by side.
chk("R6 01b's selected arm reproduces act 1's real_signed fit",
    arms["real_signed"]["best_val"], enc_arms["real_selected"]["best_val"],
    tol=0.05)
chk("R6 and its unit rank", arms["real_signed"]["unit_rank_target"],
    enc_arms["real_selected"]["unit_rank_target"])
chk("R6 01b's perm_selected reproduces act 1's perm_signed fit",
    arms["perm_signed"]["best_val"], enc_arms["perm_selected"]["best_val"],
    tol=0.05)
chk("R6 and its unit rank", arms["perm_signed"]["unit_rank_target"],
    enc_arms["perm_selected"]["unit_rank_target"])
gate("R6 every option ran real and permuted",
     all(f"{m}_{o}" in enc_arms for o in enc_worth for m in ("real", "perm")),
     f"{len(enc_worth)} options")
gate("R6 the conditioning scheme was held fixed",
     enc["conditioning_held_fixed"] == {"cov_mode": "pathway",
                                        "cov_in_encoder": False,
                                        "nonneg_decoder": True})

# Normalisation. Chosen originally to cut saturation, which it does not do.
chk("R6 annotation worth, normalize=batch", 14.52, enc_worth["selected"])
chk("R6 annotation worth, normalize=none", 15.84, enc_worth["normalize_none"])
chk("R6 annotation worth, normalize=layer", 13.47, enc_worth["normalize_layer"])
norm_pinned = [enc_arms[f"real_{o}"]["frac_pinned"]
               for o in ("selected", "normalize_none", "normalize_layer")]
gate("R6 normalisation does not change saturation",
     max(norm_pinned) - min(norm_pinned) < 0.05,
     f"pinned {['%.0f%%' % (p * 100) for p in norm_pinned]}")
gate("R6 so the reason normalize=batch was chosen does not hold",
     abs(enc_arms["real_selected"]["frac_pinned"]
         - enc_arms["real_normalize_none"]["frac_pinned"]) < 0.05)
gate("R6 and the choice moves annotation worth far less than the "
     "conditioning path does",
     (max(enc_worth[o] for o in ("selected", "normalize_none",
                                 "normalize_layer"))
      - min(enc_worth[o] for o in ("selected", "normalize_none",
                                   "normalize_layer")))
     < (arch["annotation_worth"]["signed"]
        - arch["annotation_worth"]["shipped"]) / 3)

# fan-in init: better on every flattering metric, worse on the control.
fan = enc_arms["real_init_fan_in"]
chk("R6 init=fan_in fits better than the selected initialiser",
    -8.96, fan["best_val"] - enc_arms["real_selected"]["best_val"], tol=0.05)
chk("R6 and ranks the target leaf first", 1, fan["unit_rank_target"])
chk("R6 and nearly halves saturation", 0.43, fan["frac_pinned"], tol=0.02)
chk("R6 but the annotation is worth less under it", 8.07,
    enc_worth["init_fan_in"])
gate("R6 so fan-in wins on fit, rank and saturation while losing on the "
     "control", enc_worth["init_fan_in"] < 0.7 * enc_worth["selected"]
     and fan["best_val"] < enc_arms["real_selected"]["best_val"]
     and fan["unit_rank_target"] < enc_arms["real_selected"][
         "unit_rank_target"])
chk("R6 and its permuted arm comes far closer to the right answer", 450,
    enc_arms["perm_init_fan_in"]["unit_rank_target"])
gate("R6 which is why the selected initialiser is kept",
     enc_arms["perm_init_fan_in"]["unit_rank_target"]
     < enc_arms["perm_selected"]["unit_rank_target"] / 2)

# The 2x2 the retired fidelity experiment was built for.
cells = enc["sign_scale_2x2"]
gate("R6 the sign-against-scale 2x2 is recorded", len(cells) == 4)
w2 = [c["annotation_worth"] for c in cells.values()]
gate("R6 and it is flat on annotation worth", max(w2) - min(w2) < 1.0,
     f"{min(w2):+.2f} to {max(w2):+.2f} over four cells")
r2 = [c["unit_rank_target"] for c in cells.values()]
gate("R6 and flat on the target's rank", max(r2) - min(r2) <= 5,
     f"ranks {sorted(r2)}")
gate("R6 encoder sign freedom shows up where it should",
     all(enc_arms[a]["encoder_frac_negative"] > 0.4
         for a in ("real_free_encoder", "real_free_unscaled")))
gate("R6 and nowhere else",
     all(r["encoder_frac_negative"] == 0.0 for r in enc_arms.values()
         if r["scored_on_units"] and r["nonneg_encoder"]))
gate("R6 while the pathway-level sign survives a freely signed encoder",
     all(enc_arms[a]["sign_agree_unit_vs_genes"] == 1.0
         for a in ("real_free_encoder", "real_free_unscaled")),
     "so the sign guarantee comes from nonneg_decoder, not nonneg_encoder")

# The decoder mask against the encoder mask: the split the earlier line of work
# could not measure, because it only ever masked the encoder.
chk("R6 annotation worth with the encoder mask alone", 3.24,
    enc_worth["dense_decoder"])
gate("R6 the decoder mask carries most of what the annotation is worth",
     (enc_worth["selected"] - enc_worth["dense_decoder"])
     / enc_worth["selected"] > 0.7,
     f"{(1 - enc_worth['dense_decoder'] / enc_worth['selected']):.0%} of "
     f"{enc_worth['selected']:+.2f} nats")
gate("R6 and it is close to free on fit",
     abs(enc_arms["real_dense_decoder"]["best_val"]
         - enc_arms["real_selected"]["best_val"]) < 1.0,
     f"{enc_arms['real_dense_decoder']['best_val'] - enc_arms['real_selected']['best_val']:+.2f} nats")
gate("R6 the unmasked-decoder arm is scored on fit alone, and says so",
     enc_arms["real_dense_decoder"]["scored_on_units"] is False
     and enc_arms["real_dense_decoder"]["unit_rank_target"] is None)

# The other half of the same question, which the line of work this one comes
# from never asked, because it only ever masked the encoder.
enc_only = enc_worth["selected"] - enc_worth["dense_encoder"]
dec_only = enc_worth["selected"] - enc_worth["dense_decoder"]
chk("R6 annotation worth with the decoder mask alone", 13.31,
    enc_worth["dense_encoder"])
gate("R6 so the encoder mask carries almost none of it",
     enc_only < 0.2 * dec_only,
     f"{enc_only:+.2f} nats against the decoder mask's {dec_only:+.2f}")
gate("R6 while the encoder mask pays for itself on held-out fit, which the "
     "decoder mask does not",
     enc_arms["real_dense_encoder"]["best_val"]
     - enc_arms["real_selected"]["best_val"] > 10.0,
     f"dropping it costs "
     f"{enc_arms['real_dense_encoder']['best_val'] - enc_arms['real_selected']['best_val']:+.2f}"
     f" nats, against the decoder mask's "
     f"{enc_arms['real_dense_decoder']['best_val'] - enc_arms['real_selected']['best_val']:+.2f}")
gate("R6 the arm really has no encoder mask",
     enc_arms["real_dense_encoder"]["informed_encoder"] is False
     and abs(enc_arms["real_dense_encoder"]["encoder_frac_negative"] - 0.5)
     < 0.05,
     f"{enc_arms['real_dense_encoder']['encoder_frac_negative']:.4f} of its "
     "encoder weights are negative, against 0 under the mask")
live = int(unit_table["n_genes"].sum())
dense = units_run["units"] * units_run["genes"]
gate("R6 and what it drops is 0.55% of a dense first layer",
     abs(live / dense - 0.0055) < 0.0005,
     f"{live:,} live weights against {dense:,}")
gate("R6 and unlike the unmasked decoder it is still scored over units, "
     "because its decoder is masked",
     enc_arms["real_dense_encoder"]["scored_on_units"] is True
     and enc_arms["real_dense_encoder"]["sign_agree_unit_vs_genes"] == 1.0,
     f"target at {enc_arms['real_dense_encoder']['unit_rank_target']}, sign "
     "agreement still 1.0")

# Every permuted arm must leave the right answer unremarkable.
far = {a: r["unit_rank_target"] for a, r in enc_arms.items()
       if a.startswith("perm_") and r["scored_on_units"]}
gate("R6 every permuted arm keeps the target outside the top "
     f"{PERM_FLOOR}", all(v > PERM_FLOOR for v in far.values()),
     f"worst {min(far.values())} over {len(far)} arms")

print()
print("=" * 72)
print("R7  Is annotation worth confounded by saturation?")
print("=" * 72)
# R6 rejected init="fan_in" on annotation worth. Fan-in is also R6's only
# low-saturation arm, so the two explanations were collinear there. This grid
# moves saturation with a knob that leaves the annotation alone.

chk("R7 the unscaled arm reproduces the selected architecture's fit",
    enc_arms["real_selected"]["best_val"], sat_arms["real_scale1"]["best_val"],
    tol=0.05)
chk("R7 and its pinned fraction", enc_arms["real_selected"]["frac_pinned"],
    sat_arms["real_scale1"]["frac_pinned"], tol=0.005)
chk("R7 the unscaled fan-in arm reproduces R6's", 8.07,
    sat_worth["fanin_scale1"])
chk("R7 and its pinned fraction", enc_arms["real_init_fan_in"]["frac_pinned"],
    sat_arms["real_fanin_scale1"]["frac_pinned"], tol=0.005)
gate("R7 the knob moves saturation over a usable range",
     satg["pinned_spread"] > 0.25,
     f"{satg['pinned_spread']:.0%} of units, 43% to 93% pinned")

# The relationship R6's rejection assumed was absent.
reg = satg["worth_vs_pinned"]
chk("R7 annotation worth rises with the pinned fraction, nats per 10 points",
    1.72, reg["slope_nats_per_10pct"], tol=0.05)
gate("R7 so a worth comparison between arms at different saturation is "
     "partly a saturation comparison", reg["slope_nats_per_10pct"] > 0.5,
     f"r = {reg['r']:+.3f} over {reg['n']} xavier arms, p = {reg['p']:.2f}")

# The direct test: fan-in lifted into xavier's saturation band.
f4 = sat_arms["real_fanin_scale4"]
chk("R7 fan-in lifted to 65% pinned", 0.65, f4["frac_pinned"], tol=0.01)
chk("R7 scores the same as xavier at 71%", 13.95, sat_worth["fanin_scale4"])
gate("R7 which is within a nat of the selected architecture's +14.52",
     abs(sat_worth["fanin_scale4"] - enc_worth["selected"]) < 1.0,
     f"{sat_worth['fanin_scale4']:+.2f} at 65% against "
     f"{enc_worth['selected']:+.2f} at 71%")
resid = {k: v["residual"] for k, v in satg["fanin_vs_curve"].items()}
gate("R7 and all three fan-in arms sit on the xavier curve",
     all(abs(r) < 2.0 for r in resid.values()),
     "residuals " + ", ".join(f"{r:+.2f}" for r in resid.values()))
gate("R7 so R6's rejection of fan-in does not stand: its low worth was its "
     "low saturation",
     abs(resid["fanin_scale1"]) < 1.0,
     f"a {enc_worth['selected'] - enc_worth['init_fan_in']:+.2f} nat deficit "
     f"explained to within {abs(resid['fanin_scale1']):.2f} nats")

# The cost of that, stated rather than buried: the rank metric is worse.
ranks = [(sat_arms[f"real_scale{s}"]["frac_pinned"],
          sat_arms[f"real_scale{s}"]["unit_rank_target"]) for s in (1, 2, 4, 8)]
gate("R7 the target's rank improves as saturation rises, so rank is not a "
     "saturation-free read-out either",
     ranks[-1][1] < ranks[0][1],
     " -> ".join(f"{p:.0%}:{r}" for p, r in ranks))

# Which makes R6's two mask arms the comparison that needs no correction.
d_pin = (enc_arms["real_selected"]["frac_pinned"]
         - enc_arms["real_dense_encoder"]["frac_pinned"])
gate("R7 and R6's encoder-mask arm is saturation-matched, so that comparison "
     "stands as measured",
     abs(d_pin) < 0.03,
     f"{enc_arms['real_dense_encoder']['frac_pinned']:.0%} against "
     f"{enc_arms['real_selected']['frac_pinned']:.0%}, a gap this curve would "
     f"price at {abs(d_pin) * 10 * reg['slope_nats_per_10pct']:.2f} nats of "
     f"the {enc_only:.2f} observed")

print()
#: Bumped whenever a claim is added or removed. A verifier that silently stops
#: checking something is worse than no verifier.
EXPECTED_CHECKS = 87
print("=" * 72)
if n_checks != EXPECTED_CHECKS:
    ok = False
    print(f"FAIL  check count: expected {EXPECTED_CHECKS}, ran {n_checks} -- "
          "a claim was added or dropped without updating EXPECTED_CHECKS")
print(f"{'ALL PASS' if ok else 'FAILURES ABOVE'}  ({n_checks} checks)")
sys.exit(0 if ok else 1)
