---
id: 17-reasoning/diversity-collapse-under-reasoning-rl
title: "Diversity Collapse Under Reasoning RL"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Diversity Collapse Under Reasoning RL

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/diversity-collapse-under-reasoning-rl` · **Status:** empirically-open

## 1. Problem Statement

Reinforcement learning with verifiable rewards (RLVR) raises single-sample accuracy on math and code benchmarks while sharpening the output distribution. The sharpening is not free: sampled solutions become near-duplicates, so the gain from drawing $k$ samples and taking the best shrinks. The open problem is whether this is a *necessary* cost of RLVR or an artifact of the objective and optimizer currently used.

Three variants, different difficulty:

- **Measurement.** Define a diversity statistic for reasoning traces that (i) is invariant to paraphrase and formatting, (ii) predicts test-time-compute scaling, and (iii) is estimable from a few hundred samples per prompt. Token entropy fails (i); pass@k conflates diversity with correctness.
- **Method.** Produce an RLVR recipe that matches a strong pass@1 baseline while keeping $\mathrm{pass}@k$ at large $k$ at or above the base model, on held-out tasks, without a diversity term tuned per benchmark.
- **Theory.** Show whether, for outcome-only rewards under KL-regularized policy gradient, mode collapse is forced — i.e. whether any optimizer reaching a given pass@1 must lose coverage — or whether the tradeoff is an optimizer artifact.

Solved means: a method arm that dominates the base model on the full $\mathrm{pass}@k$ curve for $k \in [1, 1024]$, reproduced by an independent group at $\geq 7$B parameters on tasks not in the RL training mix.

## 2. Formal Setting

Policy $\pi_\theta(y \mid x)$ over traces $y$ (chain of thought plus answer) for prompt $x \sim \mathcal{D}$. Verifier $r(x,y) \in \{0,1\}$ checks the extracted answer. Reference policy $\pi_{\mathrm{ref}}$ is the pre-RL checkpoint.

**Objective as actually optimized** (GRPO/PPO family):
$$J(\theta) = \mathbb{E}_{x}\,\mathbb{E}_{y\sim\pi_\theta}\big[r(x,y)\big] - \beta\, \mathbb{E}_x\big[\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})\big].$$
Note this maximizes expected *single-sample* reward. The quantity practitioners care about at deployment is
$$\mathrm{pass}@k(x) = 1 - \big(1 - p_\theta(x)\big)^k, \quad p_\theta(x) = \mathbb{E}_{y\sim\pi_\theta}[r(x,y)],$$
estimated unbiasedly from $n \geq k$ samples with $c$ correct as $1 - \binom{n-c}{k}\big/\binom{n}{k}$ (Chen et al., 2021). $J$ and $\mathrm{pass}@k$ have the same optimum only when a single trace is the argmax; otherwise maximizing $J$ is indifferent to coverage.

**Measured quantities.**

- *Token entropy:* $H_{\mathrm{tok}} = -\frac{1}{|y|}\sum_t \sum_v \pi_\theta(v \mid x, y_{<t}) \log \pi_\theta(v \mid x, y_{<t})$, averaged over rollouts. Cheap, logged every step, and the standard collapse indicator.
- *Semantic diversity:* sample $n$ traces, cluster by an equivalence relation $\sim$ (exact answer match, or normalized proof-step multiset, or an LLM-judge equivalence call), report distinct-cluster count $N_{\mathrm{eff}}$ or the Simpson-style $1 - \sum_i \hat q_i^2$ over cluster masses $\hat q_i$. The choice of $\sim$ is the whole measurement, and it is not standardized.
- *Coverage gap:* $\Delta_k = \mathrm{pass}@k(\pi_\theta) - \mathrm{pass}@k(\pi_{\mathrm{ref}})$. The crossover point $k^\star = \min\{k : \Delta_k < 0\}$ is the headline number.

**Assumptions, and where they break.**

1. *$r$ is a faithful verifier.* Violated: string-match graders on MATH/AIME accept lucky wrong reasoning; RL exploits this, inflating $p_\theta$ without real coverage.
2. *Traces are i.i.d. given $x$.* Holds under independent sampling, but temperature/top-$p$ at eval are usually retuned per arm, which silently changes $\mathrm{pass}@k$.
3. *$\mathcal{D}$ at train time and eval time are exchangeable.* Violated: AIME-style eval sets overlap heavily with RL training pools.
4. *KL to $\pi_{\mathrm{ref}}$ controls distribution shift.* Violated in practice — many strong recipes set $\beta = 0$ (DAPO, Dr. GRPO), so the only remaining regularizer is early stopping.

## 3. State of the Art

**Established (reproduced, with ablations).**

- Entropy collapse is a robust, fast phenomenon in RLVR. Cui et al. (2025, arXiv:2505.22617) fit $R = -a\exp(H) + b$ across models and report that policy entropy is nearly exhausted in the first couple hundred steps, with most of the reward gain arriving in the same window; their Clip-Cov / KL-Cov interventions raise terminal entropy and reward together.
- Clipping-side fixes work as advertised for entropy. DAPO's "clip-higher" (Yu et al., 2025, arXiv:2503.14476) was ablated specifically against entropy collapse on Qwen2.5-32B, and is now standard.
- RLHF (not RLVR) reduces output diversity relative to SFT at matched capability — Kirk et al., ICLR 2024, with per-input and across-input diversity metrics.

**Claimed but contested.**

- "RLVR does not expand the reasoning boundary." Yue et al. (2025, arXiv:2504.13837) report pass@k crossover: RL-trained models beat base at $k=1$ and lose at large $k$ across math, code and vision, on Qwen2.5 and LLaMA families. The finding is replicated in spirit but the interpretation is contested — the base-model arm uses a prompt/format that is not matched, and CoT-length and verifier-fidelity confounds are not fully ablated.
- ProRL (Liu et al., NVIDIA, 2025, arXiv:2505.24864) claims the opposite: with $>2$k RL steps, KL control and reference resets, a 1.5B model finds solutions the base model never produces at $k=$ thousands. This is a benchmark-number result on a specific model and data mix; no independent reproduction at another scale.

**Benchmark-only.** NoveltyBench (Zhang et al., COLM 2025) scores distinct generations per prompt for open-ended tasks; it is not a reasoning-coverage measure and has no RLVR ablation attached.

## 4. What Is Known

- **Crossover exists and is early at some scales.** Yue et al. report Qwen2.5-7B/14B on AIME24 and GSM8K/MATH500: RL arms lead at $k \le 8$–$32$ and are matched or overtaken by base by $k \approx 128$–$256$ ($n=256$ samples/problem, temperature $0.6$–$1.0$).
- **Entropy trajectory.** Cui et al.: token entropy on Qwen2.5-7B/32B RLVR falls from ~$0.5$–$0.7$ nats to under $0.1$ within roughly 200 GRPO steps; performance saturates as entropy does. The exponential fit $R=-a e^{H}+b$ holds across 11 models spanning 0.5B–32B.
- **Entropy is concentrated.** Wang et al. (2025, arXiv:2506.01939) find ~20% of tokens carry the high-entropy "forking" decisions; restricting policy-gradient updates to that 20% matches or beats full-token updates on Qwen3-32B. Diversity loss is therefore localized, not uniform.
- **Verifier and eval fragility.** Hochlehnert et al. (2025, arXiv:2504.07086) show AIME24-scale results move by several points from seed, decoding and harness choices alone — comparable to many claimed diversity effects.
- **Self-consistency depends on the collapsed quantity.** Majority voting (Wang et al., ICLR 2023) gains are a direct function of the spread of sampled answers; a collapsed policy gets ~0 gain from voting.

## 5. What Is Not Known

- **Methodologically blocked:** what "diversity of reasoning" means. There is no accepted equivalence relation $\sim$ on traces. Answer-level clustering ignores distinct derivations; token entropy counts formatting noise. Every published number is metric-dependent, so cross-paper comparison is not possible today. This is the binding gap.
- **Empirically open:** whether crossover survives matched controls — same eval prompt/format for base and RL arms, per-arm-tuned temperature, held-out (post-training-cutoff) problems, and a verifier audited for false accepts. Runnable at 7B for low tens of thousands of GPU-hours; not yet run cleanly by an independent group.
- **Empirically open:** whether ProRL-style prolonged RL genuinely expands coverage or shifts the crossover point to larger $k$. No one has published a $k$ up to $10^4$ curve for both a ProRL arm and a matched base arm.
- **Theoretically open:** whether $\arg\max_\theta J$ under a binary reward and a finite KL budget must lose $\mathrm{pass}@k$ coverage. The soft-optimal RL policy $\pi^\star \propto \pi_{\mathrm{ref}}\exp(r/\beta)$ *up-weights every correct trace uniformly* and so should not collapse — meaning collapse is likely an optimization/finite-sample artifact, but no theorem separates the two.

## 6. Why It Is Hard

**Confounded measurement, compounded by non-identifiability of the metric.** The headline claim ("RL narrows the reasoning boundary") is a difference of two $\mathrm{pass}@k$ curves, and every term in that difference has a nuisance knob attached: sampling temperature, eval prompt, CoT length budget, and verifier false-accept rate all move $\mathrm{pass}@k$ by amounts comparable to the effect. Base models in particular fail at $k=1$ largely for formatting reasons and recover at large $k$ by luck on multiple-choice-like answer spaces — which reads as "coverage" but is not.

Underneath that, the object being measured is not defined. Two traces that differ in every token may implement the same proof; two that share 95% of tokens may branch at the one step that matters. Until $\sim$ is fixed, "diversity collapsed by 40%" is not a falsifiable statement. Compute is a secondary obstruction: a clean curve to $k=1024$ on 60 problems × 4 arms is ~250k long generations, affordable but not something one runs as an afterthought.

## 7. Current Research (as of 2026)

- **Entropy-aware optimizers.** Clip-higher (ByteDance Seed/DAPO), Clip-Cov and KL-Cov (Cui et al., Shanghai AI Lab / Tsinghua), high-entropy-token-only updates (Alibaba Qwen team). All target $H_{\mathrm{tok}}$, none target semantic coverage directly.
- **Objectives that optimize $\mathrm{pass}@k$ directly** rather than $\mathrm{pass}@1$ — inference-aware fine-tuning for BoN (Chow et al., ICLR 2025) and pass@k-style RL objectives that anneal $k$ during training *(frontier — verify: several 2025 preprints, no independent reproduction)*.
- **Reference resets and prolonged RL** (NVIDIA ProRL) as a coverage-preserving schedule.
- **Diagnosis-side work:** whether RLVR "elicits" versus "creates" ability; work on spurious rewards and Dr. GRPO-style bias corrections (Sea AI Lab, Allen AI / UW) that changes measured gains enough to matter for this question.
- **Diversity metrics for generation** (NoveltyBench, Ippolito/CMU) being ported to reasoning traces *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** One base model, Qwen3-8B-Base (or Llama-3.1-8B). RLVR on a fixed math+code pool with an audited verifier. Four arms, identical data, steps, and eval harness:

1. Base (control arm), evaluated with the *same* chat/format prompt as the RL arms.
2. GRPO, $\beta=0$, standard clip.
3. GRPO + clip-higher + KL-Cov (entropy-preserving).
4. Direct $\mathrm{pass}@k$ objective with $k=16$ during training.

**Eval.** 60 problems from a post-training-cutoff competition set (e.g. the most recent olympiad/HMMT cycle), never in the pool. Draw $n=1024$ traces per problem per arm, at each arm's *individually tuned* temperature (tuned on a disjoint dev set to maximize pass@1024, not pass@1). Audit 200 accepted traces per arm by hand for false accepts; report the corrected curve.

**Deciding number.** The crossover point $k^\star$ against the base arm.

- If $k^\star < 256$ for arms 2–4 alike, collapse is intrinsic to outcome-only RLVR at this scale.
- If $k^\star$ is undefined (arm 3 or 4 dominates base at every $k \le 1024$) while pass@1 stays within 1 point of arm 2, the tradeoff is an optimizer artifact and the method variant is solved at 8B.

Cost estimate: ~$4 \times 61{,}440$ long generations plus 4 RL runs; order $10^4$ A100-hours.

## 9. Key References

- **[Foundational]** Mark Chen et al. *Evaluating Large Language Models Trained on Code.* Preprint, 2021. — arXiv:2107.03374 (unbiased pass@k estimator)
- **[Foundational]** Zhihong Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* Preprint, 2024. — arXiv:2402.03300 (GRPO)
- **[Foundational]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025. — arXiv:2501.12948
- **[SOTA / contested]** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* Preprint, 2025. — arXiv:2504.13837
- **[SOTA]** Ganqu Cui et al. *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.* Preprint, 2025. — arXiv:2505.22617
- **[SOTA]** Qiying Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* Preprint, 2025. — arXiv:2503.14476
- **[SOTA]** Mingjie Liu et al. *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models.* Preprint, 2025. — arXiv:2505.24864
- **[Method]** Shenao Wang et al. *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.* Preprint, 2025. — arXiv:2506.01939
- **[Diversity]** Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, Roberta Raileanu. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR, 2024.
- **[Diversity]** Yiming Zhang et al. *NoveltyBench: Evaluating Language Models for Humanlike Diversity.* COLM, 2025.
- **[Method]** Yinlam Chow et al. *Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models.* ICLR, 2025.
- **[Critique]** Zichen Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective.* Preprint, 2025. — arXiv:2503.20783
- **[Critique]** Andreas Hochlehnert et al. *A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility.* Preprint, 2025. — arXiv:2504.07086
- **[Foundational]** Xuezhi Wang et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR, 2023. — arXiv:2203.11171

## 10. Worked Example

One AIME-style problem, one 7B model, $n=256$ samples per arm at $T=0.8$.

| Arm | correct $c$ | $\hat p$ | pass@1 | pass@8 | pass@256 | distinct answers | distinct derivations (judge) |
|---|---|---|---|---|---|---|---|
| Base | 6 | 0.023 | 0.023 | 0.17 | 1.00 | 41 | 33 |
| RLVR | 74 | 0.289 | 0.289 | 0.94 | 1.00 | 5 | 2 |

At $k=1$ RL wins by 27 points. At $k=256$ both saturate — the crossover is invisible on this problem and only appears when averaged over a set where the base occasionally solves something RL never does.

Now the obstruction. The RL arm's 74 correct traces fall into **2** derivation clusters; the base arm's 6 into **5**. Read as coverage, RL collapsed by $\sim 2.5\times$. But swap the equivalence relation:

- By final answer: $N_{\mathrm{eff}}$ (RL) $= 5$ vs $41$ — an $8\times$ collapse.
- By normalized proof-step multiset: $2$ vs $33$ — a $16\times$ collapse.
- By mean pairwise token-level edit distance over correct traces: RL $0.31$, base $0.44$ — a $1.4\times$ collapse.

Same samples, three metrics, effect sizes spanning an order of magnitude. Then the verifier audit: 9 of the RL arm's 74 "correct" traces reach the right integer with an invalid intermediate step (grader is string-match on the final answer). Corrected, $\hat p = 0.254$ and pass@8 drops to $0.90$.

The number that would be reported — "RLVR cuts reasoning diversity by $X$" — is set by two free choices ($\sim$ and verifier strictness), neither of which is standardized, and each of which moves $X$ more than any published method improvement. That is why the problem is blocked at the measurement layer before it is open at the method layer.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*