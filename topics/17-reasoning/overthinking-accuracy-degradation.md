---
id: 17-reasoning/overthinking-accuracy-degradation
title: "Overthinking: Accuracy Loss From Excess Reasoning Length"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Overthinking: Accuracy Loss From Excess Reasoning Length

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/overthinking-accuracy-degradation` · **Status:** empirically-open

## 1. Problem Statement

A reasoning model emits a chain of thought $z$ before an answer $a$. Test-time scaling says accuracy rises with $|z|$. "Overthinking" is the claim that the curve turns over: past some length, *forcing the same model to think longer on the same question makes it less accurate*, not merely less efficient.

Three variants, with different difficulty:

- **Measurement.** Estimate the causal effect of reasoning length on correctness, holding the question fixed. The naive estimand — accuracy conditioned on observed length — is confounded, because a model spends more tokens on questions it finds hard. Solving this variant means producing an intervention on length that does not simultaneously change the policy's competence.
- **Method.** Build a policy that allocates length per question so that accuracy is at least that of the max-budget policy at strictly lower mean cost. Solved-ish in the weak form (compute savings at parity); unsolved in the strong form (accuracy *gains* from truncation that survive a matched-compute control).
- **Theory.** Identify a mechanism that predicts *when* extra steps hurt. Candidate mechanisms — error accumulation per step, self-distraction by irrelevant context, drift from a correct intermediate answer, reward-hacked length in RL training — make different, mostly untested predictions.

Solving it: a reproducible protocol that reports $\hat{\tau}(\ell)$, the interventional accuracy-vs-length curve, with a non-degenerate turnover point on a named benchmark, plus at least one mechanism that predicts turnover on held-out task families.

## 2. Formal Setting

Policy $\pi_\theta(z, a \mid q)$ over traces $z \in \mathcal{V}^*$ and answers $a$, question $q \sim \mathcal{D}$, verifier $v(q,a) \in \{0,1\}$. Length $L = |z|$ in **generated tokens of the reasoning segment only** (between the think-open and think-close markers), not prompt tokens and not answer tokens — the distinction matters because tool-calling traces inflate $L$ with retrieved text the model did not generate.

**Observational curve** (what almost every paper plots):

$$A_{\text{obs}}(\ell) = \mathbb{E}_{q \sim \mathcal{D}}\big[\, v(q,a) \mid L = \ell \,\big].$$

**Interventional curve** (what the claim is about):

$$A_{\text{do}}(\ell) = \mathbb{E}_{q \sim \mathcal{D}}\big[\, v(q, a) \mid \mathrm{do}(L = \ell) \,\big].$$

**Overthinking** at $\ell^\star$: $\exists\, \ell_2 > \ell_1 \ge \ell^\star$ with $A_{\text{do}}(\ell_2) < A_{\text{do}}(\ell_1) - \epsilon$, $\epsilon$ exceeding the paired bootstrap CI over $q$.

Latent difficulty $d(q)$ satisfies $d \to L$ and $d \to v$, so $A_{\text{obs}}$ is confounded by $d$; the two curves differ by the usual back-door term. Estimators used in practice:

- **Budget forcing** (append "Final Answer:" at token $\ell$, or inject "Wait" to extend). Implements $\mathrm{do}(L=\ell)$ only approximately: truncation puts the model off its training distribution, so it measures $A_{\text{do}}$ plus an off-policy artifact.
- **Best-of-$k$ length selection.** Sample $k$ traces per $q$, compare accuracy of the shortest vs longest. Conditions on $L$ *within* a question, removing between-question difficulty but not within-question sample difficulty (a trace goes long *because* it went wrong).
- **Trained length control** (RL with a length target, e.g. L1). Changes $\theta$, so it compares two policies, not two lengths of one policy.

**Overthinking score** as used in agentic work: an LLM-judge score in $[0,10]$ for the ratio of internal simulation to environment interaction. This is a judge output, not a measurement of $L$, and the two should not be conflated.

**Assumptions known to be violated in practice.** (i) $v$ is exact — false for free-form and agentic tasks, where judge error correlates with response length. (ii) Truncation preserves competence — false; forced-stop answers are drawn from a distribution the model never saw in training. (iii) $L$ is exogenous under budget forcing — false; the model adapts its pacing to a stated budget. (iv) Difficulty is one-dimensional — false; distractor density and framing sensitivity move length independently of solution depth.

## 3. State of the Art

**Established (ablated, and reproduced at least once).**

- Length can be cut substantially at accuracy parity. Chen et al. (arXiv:2412.21187, 2024) report ~48% token reduction on MATH500 for QwQ-32B-Preview with accuracy preserved. s1 (Muennighoff et al., arXiv:2501.19393, 2025) shows budget forcing gives a monotone-then-flat curve on AIME24/MATH500 for a 32B model — extension helps, then saturates.
- Within a fixed model and question, longer sampled traces are *less* likely to be correct. Ballon, Algaba & Ginis (arXiv:2502.15631, 2025) show this for o1/o3-mini families on AIME; Hassid et al. (arXiv:2505.17813, 2025) turn it into a decoding rule (short-m@k) and report compute savings around 40% with accuracy retained or improved.
- Test-time compute scaling itself is real and compute-optimal allocation beats uniform allocation (Snell et al., arXiv:2408.03314, 2024). Any overthinking claim must beat this control, not the uniform one.

**Claimed but unablated.**

- *Inverse scaling in test-time compute* (Gema et al., arXiv:2507.14417, 2025) constructs task families where accuracy falls as reasoning length grows, with model-specific failure modes (Claude models distracted by irrelevant detail; DeepSeek-R1 overfitting to problem framings). Established as an existence proof on adversarially built tasks; not established as the behaviour on standard benchmarks.
- Agentic overthinking (Cuadron et al., arXiv:2502.08235, 2025) reports that a higher judge-assigned overthinking score tracks lower SWE-bench Verified resolution, with a selection heuristic improving resolution while cutting cost. The score is judge-derived and the causal direction (failing runs deliberate more) is not separated.

**Benchmark-number-only.** Most "reduces tokens by X% with no accuracy loss" claims are single-seed, single-benchmark, with no matched-compute arm. Treat them as unreplicated.

## 4. What Is Known

- **Numbers, at scale.** QwQ-32B-Preview emits on the order of $10^3$ tokens on trivial arithmetic where a non-reasoning model uses tens (Chen et al., 2024, 32B). s1-32B: budget forcing from ~512 to ~4k thinking tokens raises AIME24 by roughly 20 points, then flattens; forcing further does not recover more. Sprague et al. (ICLR 2025, arXiv:2409.12183, meta-analysis over 100+ papers) find CoT's mean gain concentrated on math/symbolic tasks (double-digit points) and near zero elsewhere — so "more reasoning" has no headroom to help on most task families.
- **Sign of the observational slope.** Negative within-question at frontier scale (o1/o3-mini, AIME 2024/2025), positive across-model. Both are consistent with zero causal effect of length.
- **Theory of why length helps.** Chain of thought strictly extends transformer expressivity: polynomial-length CoT lets a constant-depth transformer simulate polytime computation (Merrill & Sabharwal, ICLR 2024; Li et al., ICLR 2024). No theorem says extra steps beyond sufficiency degrade accuracy — degradation must come from the learned policy, not from the computational model.
- **Length is trainable.** L1 (Aggarwal & Welleck, arXiv:2503.04697, 2025) controls output length by RL to a prompt-specified budget, at 1.5B scale, and reports a short-CoT model matching a much larger non-reasoning model at equal token budget.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** $A_{\text{do}}(\ell)$ has no accepted estimator. Budget forcing, within-question length ranking and RL length control give three different curves on the same model; nobody has shown which, if any, identifies the causal effect. Until then "overthinking" is defined by the estimator used.
- **Empirically open.** Whether turnover exists on *natural* benchmarks (AIME, GPQA, SWE-bench) for frontier models, as opposed to constructed inverse-scaling tasks. The experiment is runnable today; it needs paired per-question budget sweeps at $\ge 10$ seeds, which nobody has published at frontier scale.
- **Empirically open.** Whether truncation gains are compute-artifacts: at fixed total token budget, is "one short trace" ever better than "majority vote over $k$ short traces"? The matched-compute control is usually missing.
- **Theoretically open.** No model predicting $\ell^\star$ from properties of $q$ and $\theta$. A per-step error-accumulation model predicts $A(\ell) \approx A_{\max}(1-\eta)^{\ell/s}$ with recovery probability; it has not been fit against data and its parameters are not identified separately from difficulty.

## 6. Why It Is Hard

**Non-identifiability under a confounded, self-selected treatment.** Length is assigned by the same policy whose accuracy is being measured, and it is assigned as a *function of the model's own difficulty estimate*. Every available intervention breaks something else:

- Truncation moves the model off-distribution, so it changes competence and length together.
- "Wait"-injection extension changes the prompt, so it changes the conditional distribution over answers directly.
- RL length control changes $\theta$.

There is no known way to hold competence fixed while moving length. Compounding this: judge-based scores used in agentic settings have a known length bias, so an "overthinking score" partly measures verbosity, and the evaluation does not measure the thing it names. Compute cost is secondary but real — a per-question budget sweep over 8 budgets $\times$ 16 seeds $\times$ 500 problems at 32k tokens is roughly $2 \times 10^9$ generated tokens per model.

## 7. Current Research (as of 2026)

- **Efficient reasoning / length compression**: surveyed by Sui et al. (arXiv:2503.16419, 2025). Dominant lines are RL length penalties, adaptive routing between think and no-think modes, and latent/compressed reasoning.
- **Adversarial inverse-scaling suites**: Anthropic-affiliated and Edinburgh authors (Gema et al., 2025) building task families where length hurts by construction. Expect extension to agentic settings *(frontier — verify)*.
- **Model-side budget APIs**: frontier vendors now expose explicit thinking budgets, which makes budget sweeps cheap to run and is the most likely source of a decisive dataset *(frontier — verify)*.
- **Confidence-gated early exit**: stop when internal answer confidence stabilises. Reports parity at lower cost; a causal claim about overthinking is not established by parity results.

## 8. Concrete Next Experiment

**Question:** does $A_{\text{do}}(\ell)$ turn over on a natural benchmark, or is the negative slope entirely selection?

- **Scale.** One open-weights reasoning model at 32B (so weights are fixed and traces are reproducible), 500 questions: 250 AIME-2024/2025-style, 250 GPQA-Diamond. Budgets $\ell \in \{0.5, 1, 2, 4, 8, 16, 32\}$k thinking tokens, 16 seeds per (question, budget). ~$1.5\times10^9$ tokens; a few thousand GPU-hours on 8×H100.
- **Treatment arm.** Budget forcing at $\ell$ (hard stop + answer prompt).
- **Control arm 1 (the one usually missing).** Matched-compute majority vote: at total budget $B$, compare one trace of length $B$ against $k = B/\ell_0$ traces of length $\ell_0$.
- **Control arm 2 (off-distribution control).** Same budgets applied to a model RL-trained to that budget (L1-style), isolating truncation artifact from length effect.
- **Deciding number.** Per-question paired difference $\Delta = A_{\text{do}}(\ell^{\text{peak}}) - A_{\text{do}}(32\text{k})$, averaged over questions, with a paired bootstrap 95% CI. **Overthinking is real on natural tasks iff $\Delta > 0$ with CI excluding 0 in the treatment arm *and* the same sign in control arm 2.** If $\Delta \le 0$ in arm 2, the effect is truncation artifact, and the literature's within-question negative slope is selection.

## 9. Key References

- **[Foundational]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[Foundational]** Chen et al. *Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs.* 2024. — arXiv:2412.21187
- **[SOTA]** Muennighoff, Yang, Shi, Li, Fei-Fei, Hajishirzi, Zettlemoyer, Liang, Candès, Hashimoto. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- **[SOTA]** Gema et al. *Inverse Scaling in Test-Time Compute.* 2025. — arXiv:2507.14417
- **[SOTA]** Aggarwal, Welleck. *L1: Controlling How Long a Reasoning Model Thinks with Reinforcement Learning.* 2025. — arXiv:2503.04697
- **[Empirical]** Ballon, Algaba, Ginis. *The Relationship Between Reasoning and Performance in Large Language Models — o3 (mini) Thinks Harder, Not Longer.* 2025. — arXiv:2502.15631
- **[Empirical]** Hassid et al. *Don't Overthink It: Preferring Shorter Thinking Chains for Improved LLM Reasoning.* 2025. — arXiv:2505.17813
- **[Empirical]** Cuadron et al. *The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks.* 2025. — arXiv:2502.08235
- **[Theory]** Merrill, Sabharwal. *The Expressive Power of Transformers with Chain of Thought.* ICLR 2024.
- **[Theory]** Li, Liu, Zhou, Ma. *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.* ICLR 2024.
- **[Survey]** Sui et al. *Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models.* 2025. — arXiv:2503.16419
- **[Survey]** Sprague, Yin, Rodriguez, Jiang, Wadhwa, Singhal, Zhao, Ye, Mahowald, Durrett. *To CoT or Not to CoT? Chain-of-Thought Helps Mainly on Math and Symbolic Reasoning.* ICLR 2025. — arXiv:2409.12183

## 10. Worked Example

A 500-question set, two latent strata the evaluator cannot see.

| stratum | share | accuracy | mean $L$ |
|---|---|---|---|
| easy | 0.70 | 0.95 | 300 |
| hard | 0.30 | 0.40 | 2000 |

Suppose the true causal effect of length is exactly zero within each stratum. The observational curve still falls hard:

$$A_{\text{obs}}(300) = 0.95, \qquad A_{\text{obs}}(2000) = 0.40, \qquad \text{slope} \approx -3.2\ \text{points per }100\text{ tokens}.$$

Pooled accuracy is $0.7(0.95)+0.3(0.40) = 0.785$. A paper plotting accuracy against length here reports a 55-point "overthinking" effect that does not exist.

Now try the standard fix: stratify on a difficulty proxy. The only proxies available at eval time are model-derived — the model's own confidence, or its length. Length *is* the treatment, so it cannot be the covariate. Confidence is produced by the same forward pass that chose the length, so conditioning on it blocks part of the causal path and leaves residual confounding of unknown sign.

Try the second fix: budget-force the hard stratum to 300 tokens. Accuracy drops to, say, 0.22. Is that the causal effect of shortening ($-0.18$), or the cost of answering from a truncated, off-distribution state? Run control arm 2 — an L1-style model trained to a 300-token budget scores 0.35 on the same items. The gap $0.35 - 0.22 = 0.13$ is pure truncation artifact, and it is *larger* than most published overthinking effects.

That is the obstruction in one number: the measurement artifact of the standard intervention is bigger than the effect being measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*