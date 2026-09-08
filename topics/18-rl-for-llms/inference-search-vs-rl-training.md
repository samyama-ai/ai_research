---
id: 18-rl-for-llms/inference-search-vs-rl-training
title: "Inference-Time Search Versus RL Training Trade-Off"
topic: 18-rl-for-llms
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Inference-Time Search Versus RL Training Trade-Off

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/inference-search-vs-rl-training` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed total compute budget, how should it be split between (a) reinforcement learning on verifiable rewards (RLVR) that changes the policy weights, and (b) inference-time search — sampling many candidates, reranking with a verifier, tree search, or long chain-of-thought — that leaves weights untouched?

- **Input:** a base policy $\pi_0$, a task distribution $\mathcal{D}$ with a verifier, a training budget $C_{\text{train}}$ FLOPs, a per-query inference budget $C_{\text{inf}}$ FLOPs, and an expected deployment volume $N$ queries.
- **Output:** an allocation $(C_{\text{train}}, C_{\text{inf}})$ and the resulting policy.
- **Decision predicate:** does there exist an allocation rule, computable from cheap probes of $\pi_0$ and $\mathcal{D}$, that beats both corner solutions (all-training, all-search) at fixed $C_{\text{total}} = C_{\text{train}} + N\,C_{\text{inf}}$?

Three variants, of very different difficulty:

- **Measurement.** Is RL adding capability, or only concentrating probability mass on solutions the base model already samples at large $k$? Requires comparing pass@$k$ curves, not pass@1.
- **Method.** Find the allocation rule. Currently done by intuition and vendor defaults.
- **Theory.** Prove a scaling law $\text{Err}(C_{\text{train}}, C_{\text{inf}})$ with an exchange rate between the two axes, analogous to Chinchilla's parameter/token exchange rate.

## 2. Formal Setting

Policy $\pi_\theta(y \mid x)$ over token sequences. Binary verifier $v(x,y) \in \{0,1\}$ (unit tests, exact-match on a boxed answer, formal proof check). Success at $k$ i.i.d. samples:

$$\text{pass@}k(\pi,x) = 1 - \big(1 - p_\pi(x)\big)^k, \quad p_\pi(x) = \mathbb{E}_{y\sim\pi(\cdot|x)}[v(x,y)].$$

Measured as the unbiased Chen et al. (2021) estimator from $n \ge k$ samples: $\widehat{\text{pass@}k} = 1 - \binom{n-c}{k}/\binom{n}{k}$ with $c$ successes. Selection under a *scorer* $s$ (majority vote, PRM, learned verifier) gives the deployable quantity

$$\text{acc}@k(\pi,s,x) = \mathbb{E}\big[v(x, \arg\max_{i\le k} s(x,y_i))\big] \le \text{pass@}k(\pi,x).$$

Compute accounting, all in FLOPs and all measurable from logs:

$$C_{\text{train}} \approx 6 P T_{\text{grad}} + 2 P T_{\text{roll}}, \qquad C_{\text{inf}}(k) \approx 2 P \, k \, \bar{L} + C_{\text{scorer}},$$

with $P$ non-embedding parameters, $T_{\text{grad}}$ tokens in gradient updates, $T_{\text{roll}}$ tokens generated during rollouts (RLVR generates far more tokens than it backprops — this term dominates), $\bar L$ mean tokens per response. The trade-off surface is

$$\text{Err}(C_{\text{train}}, C_{\text{inf}}) = 1 - \mathbb{E}_{x\sim\mathcal D}\big[\text{acc}@k(\pi_{\theta(C_{\text{train}})}, s, x)\big],$$

and the question is the shape of its iso-error contours, in particular the local exchange rate $\left.\frac{\partial C_{\text{train}}}{\partial C_{\text{inf}}}\right|_{\text{Err}}$ and whether it is constant, task-dependent, or sign-changing.

**Assumptions, with the violated ones flagged:**

1. *Verifier soundness*, $v$ has no false positives. **Violated:** unit tests admit reward hacking; boxed-answer matching accepts right answers from wrong reasoning.
2. *Scorer quality independent of $k$*. **Violated:** Stroebl et al. (2024) show imperfect verifiers make acc@$k$ non-monotone — it rises then falls as $k$ grows.
3. *Fungible compute*. **Violated:** training FLOPs are capital, inference FLOPs are amortized over $N$; the optimum depends on $N$, which is unknown at training time (Sardana et al., ICML 2024).
4. *Test items i.i.d. and uncontaminated*. **Violated:** AIME/MATH contamination and $n{=}30$ item counts on AIME dominate the noise budget (Hochlehnert et al., 2025).

## 3. State of the Art

**Established (ablated, replicated):**

- Repeated sampling raises coverage far beyond pass@1. Brown et al. (2024) took DeepSeek-Coder-V2-Instruct on SWE-bench Lite from 15.9% (1 sample) to 56% (250 samples) with an oracle verifier — above the then-SOTA single-attempt agent at 43%.
- Compute-optimal test-time scaling is question-difficulty-dependent. Snell et al. (2024) showed adaptive allocation of a PRM-guided search beats best-of-$N$ at $4\times$ less test compute on MATH, and a small model plus search can beat a $14\times$ larger model on easy/medium questions — but *not* on the hardest bucket, where pretraining wins.
- RLVR reliably improves pass@1. DeepSeek-R1-Zero (DeepSeek-AI, *Nature*, 2025) moved AIME 2024 pass@1 from 15.6% to 71.0%, and 86.7% with majority-vote@64 — pure RL from a base model, no SFT cold start.

**Claimed but unablated / benchmark-number-only:**

- That RL and search are *complementary* rather than substitutable. Vendor system cards report both used together; no published run holds $C_{\text{total}}$ fixed and sweeps the split.
- Yue et al. (2025) report that across model families and tasks, RLVR models' pass@$k$ curves *cross below* the base model's at large $k$ ($k \approx 128$–$256$), concluding RLVR reshapes the sampling distribution rather than extending the reasoning boundary. This is a strong, replicated-in-parts claim, but its dependence on RL recipe, entropy regularization, and training duration is not fully ablated — later work reports the crossing recedes with longer RL and higher-entropy exploration *(frontier — verify)*.

## 4. What Is Known

- **Coverage scales like a log-linear law in $k$** over 4 orders of magnitude on GSM8K, MATH, MiniF2F, CodeContests, SWE-bench Lite (Brown et al., 2024; models 70M–70B).
- **Selection, not generation, is the binding constraint.** On CodeContests, coverage at $k{=}10{,}000$ far exceeds what majority vote or a reward model recovers; the gap between pass@$k$ and acc@$k$ widens with $k$.
- **Imperfect verifiers cap search.** Stroebl et al. (2024): with a false-positive-prone verifier, resampling gains saturate and can reverse; weak-verifier regimes make a *weaker* model with more samples worse, not better.
- **Sequential ≠ parallel scaling.** s1 (Muennighoff et al., 2025): 1,000 SFT examples plus "budget forcing" (appending "Wait" to extend thinking) gave up to +7 points on AIME 2024 over o1-preview at 32B scale — sequential test-time scaling with negligible training compute.
- **Distillation beats small-scale RL.** DeepSeek-R1's distilled 32B model outperformed a 32B trained with large-scale RL directly from base, at far lower training compute — evidence that the exchange rate is not monotone in model size.
- **Inference cost is not amortization-free.** Sardana et al. (ICML 2024) show Chinchilla-optimal sizing is wrong once $N$ is large: overtraining smaller models is compute-optimal end-to-end.

## 5. What Is Not Known

- **Empirically open (the load-bearing gap).** No published experiment fixes $C_{\text{total}}$ and sweeps the split between RLVR and inference search at frontier scale with a matched verifier. Every comparison in the literature varies both axes and the recipe at once. The experiment is runnable today at 7B–32B for roughly $10^{22}$–$10^{23}$ FLOPs.
- **Empirically open.** Whether RLVR's apparent pass@$k$ regression is a property of RLVR or of under-trained, low-entropy RL runs.
- **Theoretically open.** No proof that a joint scaling law of the form $\text{Err} = A C_{\text{train}}^{-\alpha} + B C_{\text{inf}}^{-\beta} + E$ (separable) is correct rather than a coupled form; no lower bound on how much search can substitute for policy improvement given a verifier of precision $\rho$.
- **Methodologically blocked.** "Capability" has no agreed operationalization independent of $k$. pass@1 favors RL, pass@$k$ at large $k$ favors the base model, and acc@$k$ depends on a scorer that is itself trained. The field lacks a $k$-free capability measure, so the trade-off's *dependent variable* is under-specified.

## 6. Why It Is Hard

**Confounded measurement, compounded by non-identifiability of the dependent variable.** The comparison requires the same $x$-axis for two things measured in different units — gradient FLOPs versus generated-token FLOPs — while the $y$-axis (accuracy) moves under a nuisance parameter $k$ that each arm optimizes differently. RL is tuned for pass@1; search is tuned for pass@$k$. Whichever metric you pick pre-decides the winner.

Secondary obstruction: **cost asymmetry.** A single RLVR run at 32B with meaningful rollout volume is $10^{22}$-plus FLOPs and weeks of wall-clock; a matched search arm is cheap. So the sweep needs $\ge 5$ training arms, and nobody runs 5 RL arms to publish a scaling law. Third: **verifier leakage** — the scorer used at inference is often trained on the same data as the RL reward, so the "search" arm silently contains training compute that the accounting misses.

## 7. Current Research (as of 2026)

- **Entropy-preserving RLVR.** Work on entropy collapse in GRPO-style training, clip-higher and KL-free objectives, aimed at keeping pass@$k$ from regressing *(frontier — verify)*.
- **Learning to allocate test-time compute.** Training the policy to decide its own thinking length, rather than fixing a budget — meta-RL framings from CMU (Setlur, Kumar and colleagues) and adaptive-budget work following Snell et al.
- **Verifier scaling.** Generative reward models and self-verification, targeting the acc@$k$–pass@$k$ gap rather than the generator.
- **Evaluation hygiene.** Hochlehnert et al. (2025) and follow-ups on AIME variance; seed-averaged, contamination-controlled protocols are becoming standard for exactly this comparison.
- **Amortized-compute frontier laws.** Extensions of Sardana et al. to reasoning models, treating $N$ as a first-class variable.

## 8. Concrete Next Experiment

**Scale.** One base model, Qwen3-8B-Base or Llama-3.1-8B, one dataset (MATH train + DeepScaleR-style problems), one held-out verifier. Total budget fixed at $C_{\text{total}} = 3\times10^{21}$ FLOPs, assuming $N = 10^5$ deployment queries.

**Arms (5), all at identical $C_{\text{total}}$:**

| Arm | $C_{\text{train}}$ share | inference $k$ |
|---|---|---|
| A (control) | 0% — base model | $k$ set to consume full budget ($\approx 512$) |
| B | 25% | $\approx 384$ |
| C | 50% | $\approx 256$ |
| D | 75% | $\approx 128$ |
| E | 100% RLVR | $k = 1$, extended CoT only |

All arms use the *same frozen* verifier trained before the experiment, on data disjoint from the RL reward set (closes verifier leakage). Report both pass@$k$ (oracle) and acc@$k$ (frozen verifier), 5 seeds, on AIME 2025 + MATH-500 + a held-out olympiad set.

**The deciding number.** $\text{acc}@k$ of the best interior arm (B, C, or D) minus $\max(\text{acc of A}, \text{acc of E})$, on the held-out set, with 95% CI over seeds. If that difference is $> 2$ points and CI excludes zero, the frontier is genuinely interior and an allocation rule is worth learning. If it is $\le 0$, one corner dominates and the "trade-off" is a false dichotomy at this scale. Secondary readout: the $k$ at which arm E's pass@$k$ crosses arm A's — that number tests the Yue et al. claim under matched compute for the first time.

## 9. Key References

- **[Foundational]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Anthony, Tian, Barber. *Thinking Fast and Slow with Deep Learning and Tree Search.* NeurIPS, 2017. — arXiv:1705.08439
- **[Foundational]** Cobbe et al. *Training Verifiers to Solve Math Word Problems.* 2021. — arXiv:2110.14168
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* ICLR, 2025. — arXiv:2408.03314
- **[SOTA]** Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* Nature, 2025. — arXiv:2501.12948
- **[SOTA]** Muennighoff, Yang, Shi, Li, Fei-Fei, Hajishirzi, Zettlemoyer, Liang, Candès, Hashimoto. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- **[Critique]** Yue, Chen, Lu, Zhu, Zhao, Zheng, Huang. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837
- **[Critique]** Stroebl, Kapoor, Narayanan. *Inference Scaling Flaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[Economics]** Sardana, Portes, Doubov, Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML, 2024. — arXiv:2401.00448
- **[Method]** Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[Survey/Hygiene]** Hochlehnert, Bhatnagar, Udandarao, Albanie, Prabhu, Bethge. *A Sober Look at Progress in Language Model Reasoning.* 2025. — arXiv:2504.07086

## 10. Worked Example

Take an 8B model, $\bar L = 4{,}000$ tokens per solution. Inference cost per sample: $2 \times 8\times10^9 \times 4\times10^3 = 6.4\times10^{13}$ FLOPs. At $k=256$: $1.6\times10^{16}$ FLOPs per query. Over $N=10^5$ queries: $1.6\times10^{21}$ FLOPs.

An RLVR run generating $2\times10^{10}$ rollout tokens and backpropagating $2\times10^{9}$ of them costs $2PT_{\text{roll}} + 6PT_{\text{grad}} = 3.2\times10^{20} + 9.6\times10^{19} \approx 4.2\times10^{20}$ FLOPs. **So one full RLVR run costs about a quarter of what 256-way search costs over the deployment lifetime.** That already tells you the corner solutions are not far apart in price — the trade-off is real, not a formality.

Now the obstruction. Suppose on AIME-style problems the base model has $p = 0.06$ and RLVR raises it to $p = 0.30$.

- pass@1: $0.06 \to 0.30$. RL wins by 24 points.
- pass@256: base $1-(0.94)^{256} = 0.9999$; RL, if it collapsed diversity so that solvable-set coverage shrank, can sit *below* this. Yue et al.'s reported crossings are of exactly this form.
- acc@256 with a verifier of precision 0.9 on false positives: the selected answer is right with probability well under pass@256 — Stroebl et al.'s regime, where adding samples past a few dozen buys nothing.

The three metrics rank the two arms **in three different orders**, and all three are computed from the same 256 samples. Nothing in the experiment is wrong; the dependent variable is under-defined. That is why the problem is empirically open and partly methodologically blocked at once: the experiment in §8 is cheap enough to run, but only decides anything if $k$ and the verifier are frozen and declared *before* the arms are compared.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*