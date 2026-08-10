# Eight-token OPD, SFT, and workspace results

## Corrected sampled reverse-KL OPD

The corrected implementation follows the Thinking Machines recipe: the
Karvonen-injected activation-only student samples each trajectory; an
adapter-disabled copy with the real text prefix scores those same sampled
tokens; and the student receives the detached per-token advantage
`-(log p_student - log p_teacher)` through an importance-ratio policy-gradient
loss. Both arms use the same SFT warm start, rows, learning rate, and about
10,000 additional loss-bearing tokens.

| dataset / fed layer | arm | reference NLL ↓ | exact reverse KL ↓ | teacher top-1 ↑ | EOS |
|---|---|---:|---:|---:|---:|
| normal L62 | OPD | 2.807 | **0.749** | 56.6% | 0.0% |
| normal L62 | SFT | **2.625** | 0.777 | **57.9%** | 0.0% |
| normal L42 | OPD | 3.065 | 0.965 | 51.6% | 0.0% |
| normal L42 | SFT | **2.898** | **0.928** | **53.6%** | 0.0% |
| repeat L62 | OPD | 2.696 | **7.817** | 56.4% | 99.6% |
| repeat L62 | SFT | **2.063** | 8.118 | **59.9%** | 0.0% |
| repeat L42 | OPD | 8.427 | **17.592** | 5.0% | 87.9% |
| repeat L42 | SFT | **8.176** | 18.949 | **7.8%** | 0.0% |

On normal L62 validation, OPD worsens reference NLL by 0.182 nats/token
(paired bootstrap 95% CI +0.143 to +0.221). It reduces forward KL by 0.037
nats (CI -0.060 to -0.014), while its exact reverse-KL advantage is marginal
(-0.029, CI -0.058 to +0.001). Raw-L42 NLL is also worse by 0.168 (CI +0.123
to +0.213), with no exact reverse-KL benefit.

On repeat L62, OPD does reduce the objective it optimizes: exact reverse KL is
lower by 0.301 nats (CI -0.490 to -0.105). But reference NLL is worse by 0.633
(CI +0.549 to +0.719), top-1 agreement falls 3.4 points, and 99.6% of rollouts
emit EOS before the eight-token boundary (mean length 5.72). This is
mode-seeking early termination, not evidence that the activation has run out
of information. OPD is also slower: 8.08 versus 16.55 optimized tokens/s on
normal data and 6.50 versus 10.96 on repeat data.

The Sonnet-5 judge finds no normal-L62 coherence difference (+0.011/5, CI
-0.171 to +0.194). Support trends upward by 0.166/5 (CI -0.006 to +0.343)
and hallucination trends downward by 6.9 points (CI -15.4 to +1.7), but neither
is conclusive. Repeat-L62 OPD is 0.582/5 more coherent (CI +0.484 to +0.684)
and 0.262/5 better supported (CI +0.160 to +0.367), because it often emits a
correct short prefix and stops while SFT continues into repetitions. That is a
real readout-quality tradeoff, but not greater continuation reliability: OPD
loses 2.28 tokens of mean horizon (CI -2.38 to -2.18) and has worse reference
NLL and prefix accuracy.

## Retired fixed-horizon forward-KL ablation (not true OPD)

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

## Layer-42 workspace transfer

The corrected repeat control uses an OS-CSPRNG to choose single-token BIP-39
words and a 0%-EOS SFT warm start. On all 551 released Appendix A.6 prompts,
the repeat-only checkpoint recovers 21.7% of expected concepts from raw L42,
versus 27.2% for an ordinary future lens (paired delta -5.4 points, 95% CI
-9.3 to -1.5). Its coherence is only 1.31/5 versus 3.51/5. At the L62 training
depth, recall is comparable (32.1% versus 30.4%; CI includes zero), but the
repeat model remains much less coherent (1.55 versus 3.51).

The ordinary future lens itself has a clear activation-specific signal on the
full official suite. Raw L42 concept recall is 28.2%, compared with 7.3% for a
same-distribution shuffled-activation control. Across L20, L32, L42, L54, and
L62, raw readouts recover 62.9% of expected concepts at least once, versus
18.1% shuffled; J-lens top-20 lexical recovery is 41.7%, with a larger and
non-comparable readout budget. Raw L42 coherence is 3.51/5, but 61.9% of
readouts contain unrelated specific content, so fluent recovery should not be
mistaken for completely faithful reporting.

## Conclusion

At this scale, fixed-eight-token forward KL is slower and does not improve
held-out continuation loss, coherence, or hallucination over ordinary SFT. It
does learn the prefilled teacher's full next-token distribution substantially
better. That distinction persists across all eight evaluated positions. These
numbers do not establish anything about true reverse-KL OPD.

The full replot-ready report is at
`~/shared/reports/skip-lens-opd-workspace/report.html`.
