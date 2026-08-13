"""Upload the selected scaled FineWeb OPD adapter with its model card and evals."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def model_card(analysis: dict, quality: dict, repo_id: str) -> str:
    selection = analysis["selection"]
    selected = analysis["selected_evaluations"]
    paired = analysis["paired_comparisons"]
    q_by = analysis["quality"]["by_condition"]["fineweb"]
    q_cmp = analysis["quality"]["paired_comparisons"]["L62"]
    step = selection["selected_opd_step"]
    rows = []
    for feed in ("L62", "L42"):
        for arm, label in (("opd", "OPD"), ("sft_matched", "Matched SFT")):
            metrics = selected[feed][arm]["aggregate"]
            judged = q_by[arm][feed]
            rows.append(
                f"| {feed} | {label} | {fmt(metrics['reference_student_nll'])} | "
                f"{fmt(metrics['reference_mean_reverse_kl'])} | "
                f"{pct(metrics['reference_top1_agreement'])} | "
                f"{fmt(judged['coherence'], 2)} | "
                f"{pct(judged['hallucination_rate'])} |"
            )
    nll = paired["L62"]["reference_student_nll"]
    rkl = paired["L62"]["reference_mean_reverse_kl"]
    coherence = q_cmp["coherence"]
    hallucination = q_cmp["hallucination"]
    train = analysis["training"]["opd"]["summary"]
    return f"""---
base_model: Qwen/Qwen3.6-27B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- activation-steering
- interpretability
- on-policy-distillation
- skip-lens
---

# Scaled FineWeb Skip-Lens — reverse-KL OPD

This repository contains the selected rank-64 rsLoRA adapter from a scaled
activation-only future-lens experiment. The student receives **only one residual-stream
activation**, injected at the Karvonen token; it cannot attend to the source text. A
frozen Qwen3.6-27B teacher receives the actual FineWeb prefix and scores the student's
own sampled continuation.

This is a research checkpoint, not a normal chat model. Loading it requires the custom
activation-injection hook in [skip-lens](https://github.com/ceselder/skip-lens/pull/1).

## Training

- Base model: `Qwen/Qwen3.6-27B`
- Activations: raw output of decoder block 62 (the penultimate block)
- Data: 100,000 activation/context pairs from 20,000 FineWeb documents; five uniformly
  sampled positions per document; 5% document-disjoint validation
- Labels/control data: base-model continuations sampled at temperature 1.0, top-p 1.0,
  top-k 0, with eight tokens per span
- Shared warm start: 358,400 SFT tokens
- OPD stage: {train['optimized_tokens']:,} tokens, {train['steps']} optimizer updates,
  effective batch 256 (physical 8 × accumulation 32)
- Optimizer: AdamW; learning rate 3e-5 cosine-decayed to 3e-6
- Adapter: rank 64, alpha 16, rsLoRA on attention modules

For OPD, the student samples exactly eight tokens at temperature 1.0. EOS remains in the
sampled distribution but is disabled as a stopping condition, so the OPD and SFT arms are
exactly token-matched. The teacher scores every student-sampled token on the real prefix plus
the sampled prefix. Training uses the sampled per-token reverse-KL policy-gradient estimator
with stored behavior log-probabilities and importance ratios, following
[Thinking Machines' on-policy distillation recipe](https://thinkingmachines.ai/blog/on-policy-distillation/).

## Selection and evaluation

The uploaded checkpoint is update **{step}**, selected for minimum teacher-forced L62
validation NLL over checkpoints saved every 50 updates. Its comparison control is SFT at the
same update and exact token budget.

| Fed activation | Arm | Reference NLL ↓ | Exact reverse KL ↓ | Teacher top-1 ↑ | Coherence /5 ↑ | Hallucination ↓ |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Paired L62 differences (OPD − matched SFT):

- Reference NLL: {nll['opd_minus_sft']:+.3f} nats/token
  (95% CI {nll['ci95'][0]:+.3f} to {nll['ci95'][1]:+.3f})
- Exact reverse KL: {rkl['opd_minus_sft']:+.3f}
  (95% CI {rkl['ci95'][0]:+.3f} to {rkl['ci95'][1]:+.3f})
- Sonnet-5 coherence: {coherence['opd_minus_sft']:+.3f}/5
  (95% CI {coherence['ci95'][0]:+.3f} to {coherence['ci95'][1]:+.3f})
- Sonnet-5 hallucination rate: {100 * hallucination['opd_minus_sft']:+.1f} points
  (95% CI {100 * hallucination['ci95'][0]:+.1f} to
  {100 * hallucination['ci95'][1]:+.1f})

The `evals/` directory contains the complete automatic analysis, raw selected-checkpoint
L62/L42 evaluations, and Sonnet-5 judgments.

## Limitations

- One training seed.
- Checkpoint selection and the reported validation trajectory use the same held-out split;
  these are descriptive post-selection estimates rather than an unbiased final test.
- Middle-layer L42 evaluation feeds raw block-42 activations into a lens trained at L62; it is
  a transfer test, not an independently trained L42 lens.
- Coherence and hallucination are model-judged.
- This adapter requires custom activation injection and is not meaningful without an input
  activation from the matching base model.

## Files

- `adapter_model.safetensors`, `adapter_config.json`: PEFT adapter
- `nla_meta.yaml`: injection token, neighbor-token, layer, and prompt metadata
- `opd_meta.json`: training budget and stopping metadata
- `selection.json`: selected update, checkpoint paths, and selection criterion
- `evals/`: raw and summarized evaluation artifacts

Repository: `{repo_id}`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--repo-id", default="ceselder/skip-lens-qwen36-27b-fineweb-opd-scaled",
    )
    parser.add_argument("--token-path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    analysis_path, quality_path = Path(args.analysis), Path(args.quality)
    analysis = json.loads(analysis_path.read_text())
    quality = json.loads(quality_path.read_text())
    checkpoint = Path(analysis["selection"]["selected_opd_checkpoint"])
    required = {
        "adapter_model.safetensors", "adapter_config.json", "nla_meta.yaml",
        "opd_meta.json",
    }
    missing = [name for name in sorted(required) if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"selected checkpoint is missing {missing}: {checkpoint}")
    token = Path(args.token_path).read_text().strip()
    if not token:
        raise ValueError(f"empty Hugging Face token: {args.token_path}")
    results = Path(args.results)
    step = int(analysis["selection"]["selected_opd_step"])

    with tempfile.TemporaryDirectory(prefix="scaled-opd-upload-") as temporary:
        staging = Path(temporary) / "repo"
        shutil.copytree(checkpoint, staging)
        evals = staging / "evals"
        evals.mkdir()
        shutil.copy2(analysis_path, evals / "scaled_opd_analysis.json")
        shutil.copy2(quality_path, evals / "quality_judged.json")
        for arm in ("opd", "sft_matched"):
            for feed in ("L62", "L42"):
                source = results / "evals" / f"{arm}_{step:07d}_{feed}.json"
                shutil.copy2(source, evals / source.name)
        (staging / "selection.json").write_text(json.dumps(
            analysis["selection"], indent=2,
        ) + "\n")
        (staging / "README.md").write_text(
            model_card(analysis, quality, args.repo_id)
        )

        api = HfApi(token=token)
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(staging),
            commit_message=f"Upload selected scaled OPD checkpoint at update {step}",
        )

    output = {
        "repo_id": args.repo_id,
        "url": f"https://huggingface.co/{args.repo_id}",
        "selected_step": step,
        "checkpoint": str(checkpoint),
        "commit_url": getattr(commit, "commit_url", None),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
