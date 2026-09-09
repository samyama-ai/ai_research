---
id: 17-reasoning/generator-verifier-gap-scaling
title: "Generator-Verifier Gap Scaling"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Generator-Verifier Gap Scaling

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/generator-verifier-gap-scaling` · **Status:** empirically-open

## 1. Problem Statement

Inference-time scaling rests on an asymmetry: for many reasoning tasks it is easier to *recognise* a correct solution than to *produce* one. Sample $n$ candidates, score them, keep the best. The **generator-verifier gap** is the size of that asymmetry, and the open question is how it behaves as you scale.

Three variants, with different difficulty:

- **Measurement.** Given a generator, a verifier, a task distribution and a sample budget $n$, what number reports the gap? Best-of-$n$ minus pass@1 confounds verifier quality with sampling diversity. A normalised quantity — fraction of the oracle-selection headroom actually recovered — is better but is not standard, and is undefined when coverage saturates.
- **Method.** Build verifiers whose recovered fraction stays flat or rises as $n$ grows and as the generator gets stronger. Currently it falls.
- **Theory.** Prove or refute: for a fixed task family, does the gap shrink to zero as generator capability grows (verification becomes redundant), stay constant (a permanent multiplier on test-time compute), or widen (verification is the cheaper axis to scale)?

Solving it means a predictive scaling law: given generator FLOPs $C_g$, verifier FLOPs $C_v$ and budget $n$, predict end-to-end selected accuracy to within a few points on held-out, uncontaminated tasks.

## 2. Formal Setting

Task distribution $\mathcal{D}$ over problems $x$ with a ground-truth predicate $c(x,y)\in\{0,1\}$. Generator $\pi_\theta(\cdot\mid x)$ trained with compute $C_g$; verifier $v_\phi(x,y)\in\mathbb{R}$ trained/run with compute $C_v$. Draw $y_1,\dots,y_n \stackrel{iid}{\sim} \pi_\theta(\cdot\mid x)$ at a fixed decoding temperature.

**Measured quantities.**

$$p_1 = \mathbb{E}_x\big[\Pr_{y\sim\pi}[c(x,y)=1]\big], \qquad \mathrm{cov}(n) = \mathbb{E}_x\Big[1-\big(1-\Pr[c=1]\big)^n\Big]$$

$p_1$ is pass@1, estimated unbiasedly from $N \gg n$ samples; $\mathrm{cov}(n)$ is pass@$n$ (oracle selection), estimated with the unbiased Chen-style estimator rather than by literally drawing $n$.

$$A_v(n) = \mathbb{E}_x\Big[c\big(x,\ \arg\max_{i\le n} v_\phi(x,y_i)\big)\Big]$$

**Selection efficiency**, the normalised gap:

$$\eta(n; C_g, C_v) \;=\; \frac{A_v(n) - p_1}{\mathrm{cov}(n) - p_1} \in (-\infty, 1].$$

$\eta=1$ is an oracle verifier, $\eta=0$ is random selection, $\eta<0$ is an actively harmful verifier. The scaling question is the sign and size of the exponents in
$$1-\eta \;\approx\; a\,C_g^{-\gamma} n^{-\delta} C_v^{-\zeta}.$$

**Verifier error rates**, measured per candidate: false-positive rate $\varepsilon = \Pr[v \text{ accepts} \mid c=0]$ and true-positive rate $\tau = \Pr[v \text{ accepts} \mid c=1]$. For a binary verifier with random tie-breaking among accepted candidates, expected precision at budget $n$ is
$$\Pr[\text{selected correct}] \approx \frac{p\tau}{p\tau + (1-p)\varepsilon},$$
independent of $n$ — the ratio $\varepsilon/p$, not $n$, is the binding constraint.

**Assumptions, and which are violated.**
1. *Verifier errors independent of generator errors.* Violated. Generator and verifier usually share a base model and pretraining corpus, so they fail on the same problems; measured $\varepsilon$ on generator samples exceeds $\varepsilon$ on adversarial or human-written wrong answers.
2. *$c$ is computable.* Violated in practice. Final-answer string matching accepts right answers reached by wrong reasoning and rejects equivalent forms; on code, hidden test suites are incomplete.
3. *i.i.d. sampling.* Held by construction, but breaks for revision/tree search methods, where $\mathrm{cov}(n)$ is no longer the right ceiling.
4. *No contamination.* Violated for MATH, GSM8K, HumanEval at frontier scale; inflates $p_1$ and deflates the apparent gap.

## 3. State of the Art

**Established.** Cobbe et al. (2021, arXiv:2110.14168) introduced trained verifiers on GSM8K: a 6B generator with a 6B verifier over 100 samples roughly matched a 175B fine-tuned generator — an early, reproduced demonstration that verification substitutes for generator scale. Lightman et al. (*Let's Verify Step by Step*, ICLR 2024, arXiv:2305.20050) established that **process** supervision beats outcome supervision: 78.2% on a MATH subset at 1,860 samples, with the process reward model's advantage *widening* with $n$ while the outcome model's flattened. This is the single strongest evidence that verifier design changes the scaling exponent, not just the intercept.

**Established, negative.** Stroebl, Kapoor and Narayanan (*Inference Scaling Flaws*, 2024, arXiv:2411.17501) showed that with imperfect verifiers, resampling accuracy saturates and can decline with $n$; on HumanEval with unit-test verification, weaker models with large $n$ do not overtake stronger models at $n=1$. Huang et al. (ICLR 2024, arXiv:2310.01798) showed intrinsic self-correction without an external signal degrades accuracy.

**Claimed but unablated / benchmark-only.** Snell et al. (arXiv:2408.03314) report test-time compute outperforming a $14\times$ larger model on MATH, but only on easy and medium difficulty bins, with the ordering reversing on hard bins — the compute-optimal policy is difficulty-dependent and the difficulty estimator is itself a model. Zhao et al. (*Sample, Scrutinize and Scale*, arXiv:2502.01839) report Gemini 1.5 Pro with ~200 samples and self-verification exceeding o1-preview on AIME; the result is a benchmark number, with no ablation isolating self-verification from sampling diversity. Weaver (Stanford, arXiv:2506.18203) reports that ensembling weak verifiers recovers a large fraction of oracle pass@$k$; the reported gains are per-benchmark, and generalisation to uncontaminated tasks is untested.

**Theory SOTA** is thin. Huang, Foster et al. (*Self-Improvement in Language Models: The Sharpening Mechanism*, ICLR 2025, arXiv:2412.01951) give sample-complexity results for the special case where the verifier is the generator's own likelihood; the general case, with an independently trained verifier, has no comparable analysis.

## 4. What Is Known

- **Coverage rises far past any deployed selector.** Brown et al. (*Large Language Monkeys*, arXiv:2407.21787): DeepSeek-Coder-V2-Instruct on SWE-bench Lite goes from 15.9% at one sample to 56% coverage at 250 samples. On MATH and GSM8K, coverage keeps climbing while majority voting and reward-model selection plateau within roughly an order of magnitude of samples. Measured at 8B–236B generators.
- **Majority voting saturates.** Self-consistency (Wang et al., ICLR 2023, arXiv:2203.11171) gains most of its benefit by $n\approx 40$ and is flat after; measured on PaLM-540B and GPT-3-scale models.
- **Process > outcome at large $n$.** Lightman et al., 1,860 samples, GPT-4-scale generator; the gap between PRM and ORM selection grows with $n$ over the measured range.
- **Dense/advantage-shaped process rewards improve compute efficiency.** Setlur et al. (*Rewarding Progress*, ICLR 2025, arXiv:2410.08146) report ~$5{-}6\times$ better sample efficiency for test-time search on MATH with advantage-based process verifiers versus outcome rewards, at 2B–9B generator scale.
- **The gap is generator-scale-dependent.** Song et al. (*Mind the Gap*, ICLR 2025, arXiv:2412.02674) measure a generation-verification gap that varies monotonically with pretraining compute at fixed task family — the closest thing to a scaling-law measurement that exists.
- **Solve-vs-verify allocation matters.** Singhi et al. (arXiv:2504.01005) find the compute-optimal split between generating more solutions and running generative verification shifts with model and task; no closed form.

## 5. What Is Not Known

- **Empirically open (the core gap).** Nobody has measured $\eta(n; C_g, C_v)$ on a single grid with generator scale, verifier scale and $n$ varied independently, on uncontaminated tasks. Every result above fixes two of the three axes. The sign of $\gamma$ — whether verification advantage shrinks or grows with generator capability — is unresolved, and the two literatures point opposite ways (Song et al. suggest growth; Stroebl et al. suggest collapse under realistic $\varepsilon$).
- **Empirically open.** Whether the PRM-over-ORM advantage at $n\sim10^3$ persists at $n\sim10^5$, or is a transient in the measured range.
- **Theoretically open.** No proof that a verifier with per-candidate false-positive rate bounded away from zero admits *any* selection rule whose accuracy increases with $n$ once generator and verifier errors are correlated. The independent-error case is easy; the correlated case has no theorem either way.
- **Methodologically blocked.** $\varepsilon$ itself is not measurable on the standard benchmarks, because the ground-truth predicate $c$ is final-answer matching, which is exactly the failure mode ($\text{right answer} \wedge \text{wrong reasoning}$) that inflates apparent verifier accuracy. Until $c$ is faithful, $\eta$ is measured against a corrupted ceiling.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by a non-identifiable decomposition**. Best-of-$n$ accuracy is a product of three things — generator coverage, verifier discrimination, and the correlation between their errors — and a single end-to-end benchmark number identifies none of them. Two systems with identical $A_v(1000)$ can have $\eta = 0.9$ with low coverage or $\eta = 0.3$ with high coverage; the second is the one whose verifier will fail when you scale $n$.

Secondary: cost. A clean $3\times3\times$($n$ up to $10^4$) grid on contamination-free tasks is roughly $10^5$–$10^6$ full reasoning traces per cell. And the honest ground-truth predicate requires human or formal-checker grading of *reasoning*, not answers, which does not scale with the sample budget — the measurement instrument is the bottleneck, not the GPUs.

## 7. Current Research (as of 2026)

- **Process and advantage-shaped verifiers**: CMU/Google (Setlur, Kumar) on PAVs; OpenAI's PRM line.
- **Weak-verifier ensembling and verifier distillation**: Stanford Hazy Research (Weaver).
- **Self-verification at scale as an implicit scaling axis**: Google DeepMind (Zhao et al.).
- **Limits-of-resampling analysis**: Princeton CITP (Kapoor, Narayanan, Stroebl).
- **Theory of sharpening and self-improvement**: MIT/Microsoft Research (Foster, Huang, Krishnamurthy).
- *(frontier — verify)* Reports that RL-trained long-CoT models internalise verification, shrinking the external-verifier gap on math while leaving it wide on open-ended and agentic tasks. Consistent with public leaderboard behaviour but not established by a controlled ablation.

## 8. Concrete Next Experiment

**Scale.** Three generator sizes spanning $\sim30\times$ compute (e.g. 1B, 8B, 70B from one open family, identical data recipe) $\times$ three verifier sizes (1B, 8B, 70B ORM/PRM trained on identical labels) $\times$ $n \in \{1,10,10^2,10^3,10^4\}$. Task set: 400 problems released after every model's cutoff — recent olympiad/Putnam-style problems plus code tasks with author-written hidden suites — so contamination is ruled out by construction.

**Ground truth.** For a stratified subsample of 2,000 (problem, candidate) pairs, grade *reasoning* not answers, via formal checking or two independent human graders. This yields the honest $\varepsilon$ and $\tau$, and the correction factor between answer-match accuracy and true accuracy.

**Control arms.** (a) Oracle selection — computes $\mathrm{cov}(n)$, the denominator of $\eta$. (b) Majority vote — verifier-free baseline. (c) Random selection — $\eta=0$ anchor. (d) Shuffled-verifier arm: scores permuted across candidates within a problem, to confirm any apparent gain is not sampling artefact.

**The deciding number.** Fit $1-\eta(n{=}10^3) = a\,C_g^{-\gamma}$ across the three generator scales at fixed verifier scale. Report $\gamma$ with a bootstrap CI. $\gamma > 0$ (CI excluding 0) means verification advantage compounds with generator scale and inference-time selection is a durable axis. $\gamma \le 0$ means the gap closes as generators improve, and best-of-$n$ pipelines are a transitional artefact of weak generators. One number, three orders of magnitude of compute, a decision either way.

## 9. Key References

- **[Foundational]** Cobbe, Kosaraju, Bavarian, et al. *Training Verifiers to Solve Math Word Problems.* 2021. — arXiv:2110.14168
- **[Foundational]** Wang, Wei, Schuurmans, et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023. — arXiv:2203.11171
- **[SOTA]** Lightman, Kosaraju, Burda, et al. *Let's Verify Step by Step.* ICLR 2024. — arXiv:2305.20050
- **[SOTA]** Brown, Juravsky, Ehrlich, et al. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** Setlur, Nagpal, Fisch, et al. *Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning.* ICLR 2025. — arXiv:2410.08146
- **[Negative result]** Stroebl, Kapoor, Narayanan. *Inference Scaling Flaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[Negative result]** Huang, Chen, Mishra, et al. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024. — arXiv:2310.01798
- **[Theory]** Huang, Geuter, Foster, Krishnamurthy, et al. *Self-Improvement in Language Models: The Sharpening Mechanism.* ICLR 2025. — arXiv:2412.01951
- **[Measurement]** Song, Zhang, et al. *Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models.* ICLR 2025. — arXiv:2412.02674
- **[Allocation]** Singhi, et al. *When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning.* 2025. — arXiv:2504.01005
- **[Systems]** Chen, Zaharia, Zou. *Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems.* NeurIPS 2024. — arXiv:2403.02419

## 10. Worked Example

A code-generation task where the verifier is a hidden unit-test suite — the most favourable case, since verification is executable.

**Easy regime.** $p_1 = 0.55$. Suite catches most bugs: $\tau = 0.90$, $\varepsilon = 0.02$. At $n=1000$: expected accepted-and-correct $= 1000(0.55)(0.90) = 495$; accepted-and-wrong $= 1000(0.45)(0.02) = 9$. Precision $495/504 = 0.982$. Coverage at $n=1000$ is $\approx 1.0$, so
$$\eta = \frac{0.982 - 0.55}{1.00 - 0.55} = 0.96.$$
Verification looks nearly perfect. This is the regime most published numbers sit in.

**Hard regime — same verifier, same $\varepsilon$.** $p_1 = 0.02$. At $n = 10{,}000$: accepted-and-correct $= 10{,}000(0.02)(0.90) = 180$; accepted-and-wrong $= 10{,}000(0.98)(0.02) = 196$. Precision $= 180/376 = 0.48$. Coverage $\approx 1-(0.98)^{10^4} \approx 1.0$, so
$$\eta = \frac{0.48 - 0.02}{1.00 - 0.02} = 0.47.$$

**Where the obstruction becomes visible.** The verifier did not get worse — $\varepsilon$ is fixed at 2% in both rows. What changed is $\varepsilon/p_1$: $0.036$ versus $0.98$. Precision is $p\tau/(p\tau+(1-p)\varepsilon)$, which has **no $n$ in it**. Raising the budget from $10^3$ to $10^5$ moves precision not at all; it adds correct and incorrect accepted candidates in the same ratio. All the apparent inference-time scaling in the easy row comes from coverage, and coverage is already 1.

Two consequences. First, aggregate benchmark accuracy is a mixture over difficulty bins, so a reported best-of-$n$ curve that rises is mostly the easy bins saturating — exactly Snell et al.'s difficulty-dependence, seen from the verifier side. Second, and worse, $\varepsilon = 0.02$ was assumed. On a math benchmark graded by final-answer match, the measured $\varepsilon$ is *definitionally* the rate at which a wrong answer matches the reference — near zero — while the quantity that governs selection is the rate at which flawed reasoning gets accepted, which the benchmark cannot see. The 0.47 in the hard row is unmeasurable with current instruments. That is the methodological block, not a compute shortage.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*