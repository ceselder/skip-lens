"""Build the self-contained scaled FineWeb OPD experiment report."""

from __future__ import annotations

import argparse
import json
import shutil
from html import escape
from pathlib import Path


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def copy_if_needed(source: Path, target: Path) -> None:
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)


def paired_examples(quality: dict) -> list[tuple[dict, dict, float]]:
    rows = [
        row for row in quality["detail"]
        if row.get("feed") == "L62" and "quality" in row
    ]
    by_arm = {
        arm: {int(row["index"]): row for row in rows if row.get("arm") == arm}
        for arm in ("opd", "sft_matched")
    }
    indices = sorted(by_arm["opd"].keys() & by_arm["sft_matched"].keys())
    scored = [
        (
            by_arm["opd"][index], by_arm["sft_matched"][index],
            float(by_arm["opd"][index]["quality"]["coherence"])
            - float(by_arm["sft_matched"][index]["quality"]["coherence"]),
        )
        for index in indices
    ]
    if not scored:
        return []
    selected = [max(scored, key=lambda item: item[2]), min(scored, key=lambda item: item[2])]
    unique = []
    seen = set()
    for item in selected:
        index = int(item[0]["index"])
        if index not in seen:
            unique.append(item)
            seen.add(index)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    analysis_path, quality_path = Path(args.analysis), Path(args.quality)
    analysis = json.loads(analysis_path.read_text())
    quality = json.loads(quality_path.read_text())
    out = Path(args.out_dir)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    copy_if_needed(analysis_path, data_dir / "scaled_opd_analysis.json")
    copy_if_needed(quality_path, data_dir / "quality_judged.json")

    selection = analysis["selection"]
    step = int(selection["selected_opd_step"])
    selected = analysis["selected_evaluations"]
    comparisons = analysis["paired_comparisons"]
    primary = comparisons["L62"]["reference_student_nll"]
    reverse = comparisons["L62"]["reference_mean_reverse_kl"]
    top1 = comparisons["L62"]["reference_top1_agreement"]
    delta = primary["opd_minus_sft"]
    ci = primary["ci95"]
    if delta < 0:
        title = "Scaled reverse-KL OPD improves last-layer NLL over matched SFT"
        verdict = "beats"
        verdict_class = "good"
    else:
        title = "Scaled reverse-KL OPD still trails matched SFT on last-layer NLL"
        verdict = "trails"
        verdict_class = "bad"

    training = analysis["training"]
    opd_train = training["opd"]["summary"]
    sft_train = training["sft_matched"]["summary"]
    quality_summary = analysis["quality"]
    q_by = quality_summary["by_condition"]["fineweb"]
    q_cmp = quality_summary["paired_comparisons"]["L62"]

    trajectory_rows = []
    for update in analysis["common_steps"]:
        opd = analysis["trajectory"]["L62"]["opd"][str(update)]["aggregate"]
        sft = analysis["trajectory"]["L62"]["sft_matched"][str(update)]["aggregate"]
        trajectory_rows.append(
            f"<tr><td>{update}</td><td>{fmt(opd['reference_student_nll'])}</td>"
            f"<td>{fmt(sft['reference_student_nll'])}</td>"
            f"<td>{fmt(opd['reference_student_nll'] - sft['reference_student_nll'])}</td>"
            f"<td>{fmt(opd['reference_mean_reverse_kl'])}</td>"
            f"<td>{fmt(sft['reference_mean_reverse_kl'])}</td></tr>"
        )

    eval_rows = []
    for feed in ("L62", "L42"):
        for arm, label in (("opd", "Reverse-KL OPD"), ("sft_matched", "Matched SFT")):
            values = selected[feed][arm]["aggregate"]
            judged = q_by[arm][feed]
            eval_rows.append(
                f"<tr><td>{feed}</td><td>{label}</td>"
                f"<td>{fmt(values['reference_student_nll'])}</td>"
                f"<td>{fmt(values['reference_mean_reverse_kl'])}</td>"
                f"<td>{pct(values['reference_top1_agreement'])}</td>"
                f"<td>{fmt(values['mean_length'], 2)}</td>"
                f"<td>{fmt(judged['coherence'], 2)}</td>"
                f"<td>{pct(judged['hallucination_rate'])}</td></tr>"
            )

    examples = []
    for opd, sft, coherence_delta in paired_examples(quality):
        context = escape(opd.get("context", "")[-800:])
        reference = escape(opd.get("reference", ""))
        examples.append(
            f"<h4>Validation row {int(opd['index'])}: coherence difference "
            f"{coherence_delta:+.0f}/5</h4>"
            f"<p><strong>Context tail:</strong> {context}</p>"
            f"<p><strong>One T=1 reference:</strong> {reference}</p>"
            f"<div class='grid-2'><div><strong>Reverse-KL OPD "
            f"({opd['quality']['coherence']}/5)</strong>"
            f"<pre>{escape(opd.get('readout', ''))}</pre></div>"
            f"<div><strong>Matched SFT ({sft['quality']['coherence']}/5)</strong>"
            f"<pre>{escape(sft.get('readout', ''))}</pre></div></div>"
        )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<link rel="stylesheet" href="/reports/static/claude.css"></head><body><main>
<h1>{escape(title)}</h1>
<div class="tldr"><strong>TL;DR.</strong> After a shared 358,400-token SFT warm start,
true student-sampled reverse-KL OPD {verdict} an exactly token-matched continuation-SFT
control by <strong>{abs(delta):.3f} nats/token</strong> at its selected L62 checkpoint
(OPD − SFT {delta:+.3f}, paired 95% CI {ci[0]:+.3f} to {ci[1]:+.3f}). Both arms used
1,024,000 stage-two tokens at effective batch 256. The checkpoint was selected from the
full 50-update trajectory on held-out FineWeb activations; quality was then judged by Sonnet 5.</div>
<div class="kpis">
  <div><span class="kpi-value {verdict_class}">{delta:+.3f}</span><span class="kpi-label">OPD − SFT L62 NLL</span></div>
  <div><span class="kpi-value">{step}</span><span class="kpi-label">selected OPD update</span></div>
  <div><span class="kpi-value">{opd_train['optimized_tokens']:,}</span><span class="kpi-label">tokens per stage-two arm</span></div>
  <div><span class="kpi-value">256</span><span class="kpi-label">effective batch size</span></div>
</div>
<nav class="toc"><strong>Contents</strong><ol>
<li><a href="#trajectory">Validation trajectory</a></li>
<li><a href="#selected">Selected checkpoint</a></li>
<li><a href="#quality">Coherence and hallucination</a></li>
<li><a href="#methods">Methods and audit</a></li></ol></nav>

<section id="trajectory"><h2>1. Scaling trajectory</h2>
<figure><img src="scaled_opd_validation.png" alt="OPD and SFT validation metrics over stage-two training">
<figcaption>Teacher-forced metrics use the same 512 held-out, document-disjoint activations
at every checkpoint. Solid lines feed the training layer L62; dashed lines test raw L42 transfer.</figcaption></figure>
<table><thead><tr><th>Update</th><th>OPD NLL ↓</th><th>SFT NLL ↓</th><th>Difference</th>
<th>OPD reverse KL ↓</th><th>SFT reverse KL ↓</th></tr></thead>
<tbody>{''.join(trajectory_rows)}</tbody></table>
<p>The independently best SFT checkpoint occurs at update {selection['best_sft_nll_step']};
the best OPD reverse-KL checkpoint occurs at update {selection['best_opd_reverse_kl_step']}.
The primary comparison instead uses SFT at OPD's selected update {step}, preserving exact
optimizer-update and token matching.</p></section>

<section id="selected"><h2>2. Selected checkpoint metrics</h2>
<table><thead><tr><th>Fed activation</th><th>Arm</th><th>Reference NLL ↓</th>
<th>Exact reverse KL ↓</th><th>Teacher top-1 ↑</th><th>Mean length /8</th>
<th>Coherence /5 ↑</th><th>Hallucination ↓</th></tr></thead>
<tbody>{''.join(eval_rows)}</tbody></table>
<p>At L62, OPD − SFT exact reverse KL is {reverse['opd_minus_sft']:+.3f}
(95% CI {reverse['ci95'][0]:+.3f} to {reverse['ci95'][1]:+.3f}); top-1 agreement differs by
{100 * top1['opd_minus_sft']:+.1f} points (CI {100 * top1['ci95'][0]:+.1f} to
{100 * top1['ci95'][1]:+.1f}).</p></section>

<section id="quality"><h2>3. Coherence and hallucination</h2>
<p>Sonnet 5 judged {quality_summary['summary']['n']:,} selected-checkpoint readouts with
{quality_summary['summary']['errors']} parse/API errors. At L62, OPD − SFT coherence is
{q_cmp['coherence']['opd_minus_sft']:+.3f}/5 (95% CI
{q_cmp['coherence']['ci95'][0]:+.3f} to {q_cmp['coherence']['ci95'][1]:+.3f}); the hallucination
rate difference is {100 * q_cmp['hallucination']['opd_minus_sft']:+.1f} points (CI
{100 * q_cmp['hallucination']['ci95'][0]:+.1f} to
{100 * q_cmp['hallucination']['ci95'][1]:+.1f}).</p>
<details><summary>Largest paired coherence differences</summary>{''.join(examples)}</details></section>

<section id="methods"><h2>4. Methods and audit</h2>
<p>Base model: Qwen3.6-27B. Data: 100,000 activation/context pairs from 20,000 disjoint
FineWeb documents, five uniformly sampled positions per document, with raw outputs from blocks
62 and 42. Labels are base-model T=1, top-p 1.0, top-k 0 eight-token continuations. The held-out
split is document-disjoint (5%). Training uses rank-64 rsLoRA, learning rate 3e-5 to 3e-6,
physical batch 8 and 32-step gradient accumulation.</p>
<p>The student sees only the activation injected at the Karvonen token; the frozen teacher sees
the actual text prefix plus the student's sampled prefix. OPD uses the per-token sampled
<code>KL(student || teacher)</code> policy-gradient estimator with stored behavior log-probabilities
and importance ratios. Fixed-horizon generation disables EOS as a stopping condition without
masking EOS from the sampled distribution. Thus each arm receives exactly 500 × 256 × 8 =
1,024,000 stage-two tokens.</p>
<p>OPD throughput was {opd_train['optimized_tokens_per_second']:.1f} tokens/s versus
{sft_train['optimized_tokens_per_second']:.1f} for SFT. OPD wall time was
{opd_train['wall_seconds'] / 3600:.2f} GPU-hours and SFT wall time was
{sft_train['wall_seconds'] / 3600:.2f} GPU-hours.</p>
<p><strong>Selection caveat:</strong> {escape(selection['caveat'])} One seed was run.
Automatic uncertainty uses 10,000 paired bootstrap resamples over validation rows. Sonnet 5
quality judgment is model-based and is reported separately from token-level metrics.</p>
<details><summary>Reproducibility</summary><ul>
<li>Collection Slurm array: <code>90805</code>; corrected fixed-horizon training/evaluation:
<code>91188</code>; postprocessing/judging/upload: <code>91189</code>.</li>
<li>Selected OPD checkpoint: <code>{escape(selection['selected_opd_checkpoint'])}</code>.</li>
<li>Matched SFT checkpoint: <code>{escape(selection['matched_sft_checkpoint'])}</code>.</li>
<li>Raw analysis: <code>data/scaled_opd_analysis.json</code>; judge output:
<code>data/quality_judged.json</code>. Plot is available as PNG and PDF.</li>
</ul></details></section>
</main></body></html>"""
    (out / "report.html").write_text(html)
    print(f"wrote {out / 'report.html'}")


if __name__ == "__main__":
    main()
