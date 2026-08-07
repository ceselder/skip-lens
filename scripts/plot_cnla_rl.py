"""Plot the compositional-NLA RL run from its training log.

Parses the step lines + eval / text_judges lines and produces the headline figure
(reconstruction FVE and Sonnet-5 coherence both rising = NO reconstruction-gaming)
plus the KL-divergence panel (β=0.1 holds ~130 steps then the policy escapes). Writes
PNG+PDF and a replot-ready JSON of every number.

    python scripts/plot_cnla_rl.py --log <rl_log.txt> --out <report_dir>
"""
import argparse, json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP = re.compile(r"step (\d+) \| r (-?[\d.]+) \| FVE ([\d.]+)% \| kl ([\d.]+) \| ent ([\d.]+) \| ext (\d+)%")
EVAL = re.compile(r"\[eval@(\d+)\] reward (-?[\d.]+) \| FVE ([\d.]+)%")
TJ = re.compile(r"\[text_judges@(\d+)\] unique_info ([\d.]+) coherence ([\d.]+) writing_quality ([\d.]+) specificity ([\d.]+) repetitiveness ([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True, help="report dir (writes cnla_rl_curves.{png,pdf} + data/)")
    ap.add_argument("--title-run", default="run 30859 (β=0.1)")
    args = ap.parse_args()
    txt = Path(args.log).read_text(errors="ignore")

    steps = [dict(step=int(a), r=float(b), fve=float(c), kl=float(d), ent=float(e), ext=int(f))
             for a, b, c, d, e, f in STEP.findall(txt)]
    ev = [dict(step=int(a), reward=float(b), fve=float(c)) for a, b, c in EVAL.findall(txt)]
    tj = [dict(step=int(a), unique_info=float(b), coherence=float(c), writing_quality=float(d),
               specificity=float(e), repetitiveness=float(f)) for a, b, c, d, e, f in TJ.findall(txt)]
    data = {"steps": steps, "eval_fve": ev, "text_judges": tj}
    outd = Path(args.out); (outd / "data").mkdir(parents=True, exist_ok=True)
    (outd / "data" / "cnla_rl_curves.json").write_text(json.dumps(data, indent=2))

    x = [s["step"] for s in steps]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    # divergence marker (KL blows up ~here)
    div = next((s["step"] for s in steps if s["kl"] > 0.8), None)

    # (1) reconstruction: per-step train FVE + held-out eval FVE
    ax[0, 0].plot(x, [s["fve"] for s in steps], "-", color="#4C72B0", lw=1.3, alpha=.85, label="train FVE (per step)")
    if ev:
        ax[0, 0].plot([e["step"] for e in ev], [e["fve"] for e in ev], "s", color="#C44E52", ms=9, label="held-out eval FVE")
    ax[0, 0].set_title("Reconstruction FVE (composition of 4 bullets)"); ax[0, 0].set_ylabel("FVE %"); ax[0, 0].legend(fontsize=8)

    # (2) coherence + other Sonnet-5 judge attributes (eval points)
    if tj:
        tx = [t["step"] for t in tj]
        ax[0, 1].plot(tx, [t["coherence"] for t in tj], "-o", color="#C44E52", lw=2, ms=8, label="coherence")
        ax[0, 1].plot(tx, [t["specificity"] for t in tj], "-^", color="#55A868", label="specificity")
        ax[0, 1].plot(tx, [t["unique_info"] for t in tj], "-s", color="#8172B3", label="unique_info")
        ax[0, 1].plot(tx, [t["writing_quality"] for t in tj], "-d", color="#DD8452", label="writing_quality")
        ax[0, 1].set_ylim(0, 10)
    ax[0, 1].set_title("Sonnet-5 judged text quality (held-out)"); ax[0, 1].set_ylabel("score (1–10)"); ax[0, 1].legend(fontsize=8)

    # (3) KL to SFT ref — the divergence story
    ax[1, 0].plot(x, [s["kl"] for s in steps], "-", color="#DD8452", lw=1.6)
    ax[1, 0].axhline(0.2, color="k", ls=":", lw=1, alpha=.5)
    ax[1, 0].set_title("KL(policy‖SFT-ref) — β=0.1 holds then the policy escapes"); ax[1, 0].set_ylabel("KL")

    # (4) policy entropy
    ax[1, 1].plot(x, [s["ent"] for s in steps], "-", color="#55A868", lw=1.4)
    ax[1, 1].set_title("Policy entropy (nats)"); ax[1, 1].set_ylabel("entropy")

    for a in ax.flat:
        a.set_xlabel("RL step")
        if div is not None:
            a.axvline(div, color="0.4", ls="--", lw=1, alpha=.7)
    if div is not None:
        ax[1, 0].annotate(f"KL blow-up\n~step {div}", (div, 0.9), fontsize=8, ha="right", color="0.3")

    fig.suptitle("Compositional NLA (Qwen3.6-27B L62, 4-bullet future-lens): reconstruction AND coherence rise "
                 "TOGETHER — no reconstruction-gaming\n"
                 f"leave-one-out FVE reward, frozen AR, {args.title_run}; KL β=0.1 destabilises after ~130 steps "
                 "(→ use stronger KL / earlier stop). Sonnet-5 judge.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for e in ("png", "pdf"):
        fig.savefig(outd / f"cnla_rl_curves.{e}", dpi=140, bbox_inches="tight")
    # headline KPIs to JSON
    kpi = {
        "eval_fve_start": ev[0]["fve"] if ev else None, "eval_fve_best": max((e["fve"] for e in ev), default=None),
        "coherence_start": tj[0]["coherence"] if tj else None, "coherence_best": max((t["coherence"] for t in tj), default=None),
        "train_fve_peak": max((s["fve"] for s in steps), default=None),
        "kl_max": max((s["kl"] for s in steps), default=None),
        "divergence_step": div, "n_steps_logged": len(steps),
    }
    (outd / "data" / "cnla_kpis.json").write_text(json.dumps(kpi, indent=2))
    print("wrote cnla_rl_curves.png/pdf + data/*.json")
    print("KPIs:", json.dumps(kpi))


if __name__ == "__main__":
    main()
