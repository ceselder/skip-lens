"""Two panels: every arm on ONE judge scale, and the slot-scaling falsification.

Panel A ranks all judged conditions by workspace agreement, recomputed from the
raw per-record judge scores (the stored aggregates mixed a 0-2 and a 0-1 scale).
The split it exposes is the headline: every single-vector condition sits far above
every 8-slot condition.

Panel B kills the hypothesis that the 8-slot collapse was a norm-matching
artifact. Preserving the slots' relative magnitudes ("shared") is WORSE than
amplifying each to full norm ("per_slot"), so what is wrong with the deep slots
is their direction, not their scale.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.expanduser("~/shared/reports/skip-lens-multislot")
rows = json.load(open(f"{R}/data/ALL_rescored.json"))
gap = json.load(open(f"{R}/data/estimate_gap_scaling.json"))

# a condition is single-vector if it comes from an arm trained with one slot
SINGLE = {"armC_raw", "armC_jac"}
for r in rows:
    r["single"] = r["arm"] in SINGLE

fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 7.4),
                           gridspec_kw={"width_ratios": [2.15, 1]})

def label(r):
    a = r["arm"].replace("armA_", "A/").replace("arm", "")
    # the two armC runs were produced by an ad-hoc command that is no longer on
    # disk, so which one fed raw h42 and which fed the Jacobian cannot be
    # verified. Do not assert it in the label.
    if r["condition"] == "?":
        return f"{a} [label unverified]"
    return f"{a}·{r['condition']}"[:34]


lab = [label(r) for r in rows]
y = range(len(rows))
col = ["#c05621" if r["single"] else "#7c8ba1" for r in rows]
a.barh(list(y), [r["workspace"] for r in rows],
       xerr=[r["se"] for r in rows], color=col, height=.72,
       error_kw=dict(ecolor="#33404f", lw=1.1, capsize=2.5))
a.set_yticks(list(y))
a.set_yticklabels(lab, fontsize=9.5)
a.invert_yaxis()
a.set_xlabel("workspace agreement with the J-lens readout (judge 0-2, rescaled to 0-1)",
             fontsize=10.5)
a.axvline(0.5, color="#9b2c2c", ls=":", lw=1.4)
a.text(0.505, len(rows) - 0.4, "half credit", color="#9b2c2c", fontsize=9, rotation=90,
       va="bottom")
for i, r in enumerate(rows):
    if r["degenerate_frac"] and r["degenerate_frac"] > 0.15:
        a.text(r["workspace"] + r["se"] + .012, i,
               f"{r['degenerate_frac']:.0%} degenerate", va="center", fontsize=8,
               color="#9b2c2c")
a.set_xlim(0, 0.72)
a.grid(axis="x", alpha=.25)
a.set_title("A. One vector beats eight, by a margin the old mixed-scale table hid\n"
            "orange = decoder trained on a single vector; grey = eight per-offset slots\n"
            "the two single-vector runs are statistically tied (0.583 vs 0.561, ±0.016), "
            "so a one-vector decoder fed layer 42 scores ~0.57 either way",
            fontsize=11, loc="left")

names = ["per_slot\n(each slot to ‖h‖)", "shared\n(keep relative sizes)"]
vals = [gap["scalings"]["per_slot"]["gap_nats"], gap["scalings"]["shared"]["gap_nats"]]
bars = b.bar(names, vals, color=["#7c8ba1", "#9b2c2c"], width=.6)
for r, v in zip(bars, vals):
    b.text(r.get_x() + r.get_width() / 2, v + .18, f"+{v:.2f}", ha="center",
           fontsize=12, fontweight="bold")
b.axhline(1.0, color="#2f855a", ls="--", lw=1.4)
b.text(1.42, 1.15, "usable (<1 nat)", color="#2f855a", fontsize=9, ha="right")
b.set_ylabel("extra nats to read a span from the averaged-Jacobian\n"
             "estimate instead of the real penultimate state", fontsize=10)
b.set_ylim(0, 10.3)
b.grid(axis="y", alpha=.25)
b.set_title("B. The eight-slot failure is not a scaling bug:\n"
            "keeping faint slots faint makes it WORSE", fontsize=11.5, loc="left")

fig.suptitle("Reading a multi-token span from an averaged Jacobian: single-vector "
             "readouts work, per-token-offset slots do not\n"
             "Qwen3.6-27B, decoder trained at layer 62 and fed layer 42; "
             "judge = Claude Sonnet 5 on 551 held-out prompts",
             fontsize=13, y=1.005)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/rescored_arms.{ext}", dpi=170, bbox_inches="tight")
print(f"wrote {R}/rescored_arms.png/.pdf")

json.dump({"panelA_rows": rows, "panelB_gap_nats":
           {k: v["gap_nats"] for k, v in gap["scalings"].items()},
           "panelB_ce": {k: {"real": v["ce_real"], "estimate": v["ce_estimate"]}
                         for k, v in gap["scalings"].items()}},
          open(f"{R}/data/rescored_arms.json", "w"), indent=1)
print("wrote data/rescored_arms.json")
