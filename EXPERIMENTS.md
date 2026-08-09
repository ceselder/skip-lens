# Skip Lens experiments

This branch evaluates whether a multi-token activation reader benefits from
on-policy distribution distillation and whether its shallow-layer readouts
track the model's workspace rather than merely predicting likely text.

## 1. On-policy distillation versus future-lens SFT

All arms start from the same short L62 future-lens SFT checkpoint and use the
same training rows. The reported comparison has an externally fixed horizon of
at most eight continuation tokens.

- **OPD:** the student samples a continuation from an injected activation. A
  frozen copy of the base model sees the real source prefix plus that sampled
  prefix and supplies its full next-token distribution. The loss is
  `KL(teacher || student)`.
- **No endogenous stopping:** KL is measured but is not fed back into an EOS
  target. An earlier adaptive-EOS pilot collapsed because emitting EOS changed
  the sampled state distribution, increased measured KL, and created still
  more EOS supervision. That run is retained as a diagnosed negative control,
  not interpreted as activation-information exhaustion.
- **Token-matched SFT:** ordinary teacher-forced continuation CE, stopped after
  the same number of loss-bearing response tokens as OPD.
- **Time-matched SFT:** the same CE arm, stopped after the same GPU wall time as
  OPD. Token and time matching are reported separately.

Primary automatic measures are paired ground-truth continuation NLL,
teacher/student KL and teacher top-1 agreement on the exact same held-out token
positions, plus free-running EOS/effective horizon and reference-prefix
accuracy. Sonnet 5 independently grades coherence, unsupported specificity
(hallucination), and premature EOS.

## 2. Official global-workspace evaluations

The exact Apache-2.0 prompt sets released with Gurnee et al. (2026) are vendored
under `evals/datasets/official/`. The harness honors their specified activation
positions, notably the internal newline for poetry.

For each item and workspace layer it compares:

1. the L62-trained reader fed raw `h_l` (the Skip Lens hypothesis);
2. the reader fed `J_{l->62} h_l`;
3. the ordinary J-lens token readout from the same transported vector;
4. a same-distribution shuffled-activation false-positive control.

Alongside per-concept recovery, the headline natural-language metric is whether
one coherent readout jointly expresses all expected intermediates. This tests
multi-concept/multi-token reporting rather than treating independent token hits
as a coherent workspace description. Answer-skipping and unrelated content are
reported separately.

## 3. CSPRNG repeat-after-me control

The base model is asked to repeat uniformly sampled BIP-39 English words. Words
come from `secrets.SystemRandom` (OS CSPRNG), are restricted to exact single
tokens under the Qwen tokenizer, and are never converted into checksummed
wallet mnemonics or keys. Only token-exact model completions are retained.

Activations are harvested after the model has begun its own completion, with
random target spans of 1--16 tokens. Phrase-disjoint train/validation splits
prevent another position from the same random sequence leaking across. Models
are trained only on L62 and evaluated on aligned L62 and L42 activations. If
raw-L42 transfer survives when semantic priors are useless, that supports the
claim that Skip Lens reads copy/induction workspace state rather than merely
completing plausible web text.

## Reproducibility

Training writes `resolved_config.json`, per-step `metrics.jsonl`, checkpoint
metadata, and WandB runs. Final reports include exact JSON behind every table
and figure, plus both PNG and PDF plots.
