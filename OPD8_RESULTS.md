# Fixed-horizon forward-KL ablation (not true OPD)

All training arms start from the same 200-step future-lens SFT checkpoint.
This initial pilot samples from the current activation-only student and
minimizes full-vocabulary `KL(teacher || student)` for up to eight continuation
tokens. It does **not** use the sampled reverse-KL policy-gradient objective in
Thinking Machines' OPD recipe and must not be interpreted as an OPD result.
There is no KL-triggered EOS target in this experiment.

## Normal completions, last-layer activations

| arm | optimized tokens | seconds | tokens/s | held-out NLL | teacher KL | teacher top-1 |
|---|---:|---:|---:|---:|---:|---:|
| warm start | — | — | — | 2.863 | 0.919 | 53.0% |
| forward-KL ablation | 10,012 | 1,274 | 7.86 | 2.669 | 0.605 | 61.1% |
| SFT, token matched | 10,016 | 605 | 16.55 | **2.625** | 0.775 | 57.9% |
| SFT, time matched | 39,552 | 1,274 | 31.05 | 2.663 | 1.017 | 58.1% |

Against token-matched SFT, forward KL has worse ground-truth NLL by 0.044
nats/token (paired bootstrap 95% CI +0.012 to +0.075), but lower
teacher-student KL by 0.169 nats (CI -0.193 to -0.147) and 3.2 percentage
points higher teacher top-1 agreement. Against time-matched SFT, NLL is tied:
Forward KL minus SFT is +0.005 nats/token (CI -0.054 to +0.063).

The Sonnet-5 paired judge finds no credible forward-KL advantage over equal-token SFT
on coherence (+0.013/5, CI -0.170 to +0.189) or hallucination (-2.5 points,
CI -9.4 to +4.4). Equal-time SFT has the strongest aggregate quality scores.

## Repeat-after-me control

The 200-step repeat warm start was inadequate: its free-running EOS rate was
99.2%. Forward KL reduced this to 94.5%, while both SFT controls reached 0% EOS.
Token-matched SFT also beat forward KL on held-out NLL (2.049 versus 2.157).
This does not answer the reverse-KL question, and it should not be treated as a
clean test of the induction-head hypothesis without a stronger common SFT warm
start.

## Conclusion

At this scale, fixed-eight-token forward KL is slower and does not improve
held-out continuation loss, coherence, or hallucination over ordinary SFT. It
does learn the prefilled teacher's full next-token distribution substantially
better. That distinction persists across all eight evaluated positions. These
numbers do not establish anything about true reverse-KL OPD.

The full replot-ready report is at
`~/shared/reports/skip-lens-opd-workspace/report.html`. The official six-set
workspace readout comparison is queued separately and will be appended.
