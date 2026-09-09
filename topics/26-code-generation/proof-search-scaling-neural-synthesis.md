---
id: 26-code-generation/proof-search-scaling-neural-synthesis
title: "Proof Search Scaling in Neural Theorem-Prover-Guided Synthesis"
topic: 26-code-generation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Proof Search Scaling in Neural Theorem-Prover-Guided Synthesis

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/proof-search-scaling-neural-synthesis` · **Status:** empirically-open

## 1. Problem Statement

In verified synthesis, a program and its correctness proof are produced together: the search emits a candidate implementation and a machine-checked proof that it meets a specification. A neural policy proposes tactics or proof steps; a kernel (Lean, Rocq/Coq, Isabelle) accepts or rejects them. The practical question is how the success rate scales with the **inference compute spent inside the search**, holding the model fixed.

Three variants, usually conflated:

- **Measurement.** Given a prover, a benchmark, and a compute budget $C$, what is the success-rate curve $s(C)$, and is $C$ counted in tokens, kernel calls, or wall-clock? No convention exists; papers report pass@$k$ with incomparable $k$.
- **Method.** Which search operator — independent sampling, best-first, MCTS-style, subgoal decomposition, repair loops — has the best exponent in $s(C)$ at fixed model, and does the ranking change with model scale?
- **Theory.** Is there a compute-exchange law $\; \mathrm{FLOPs}_{\text{train}} \leftrightarrow \mathrm{FLOPs}_{\text{search}}$ for formal proof search, as there is for board games, and does the curve saturate at a finite ceiling set by the policy's support?

Solving it means: a budget-normalized scaling law for $s(C)$ over at least two decades of $C$, with the search operator as an ablated variable, on a benchmark that is verifiably not contaminated.

## 2. Formal Setting

A proof state is a node in a search tree. Let $\Sigma$ be the tactic alphabet, $g_0$ the goal (for synthesis, `∃ p, spec p`), and $\pi_\theta(a \mid g)$ the policy. The environment is the kernel $K$: applying $a$ to $g$ yields subgoals $K(g,a) \in \Sigma^* \cup \{\bot\}$. A proof is an AND-tree with all leaves closed; success is the kernel's `True`, not a test suite.

**Budget, as measured.** Report all three; they do not move together.

$$C_{\text{tok}} = \sum_{i=1}^{N} \big(|p_i| + |o_i|\big), \qquad C_{\text{ker}} = \\#\{\text{kernel invocations}\}, \qquad C_{\text{wall}} = \text{seconds} \times \text{accelerators}$$

$C_{\text{tok}}$ is prompt plus output tokens over all $N$ model calls. $C_{\text{ker}}$ dominates wall-clock when elaboration is slow (Lean `simp`/`omega` calls of 1–30 s are routine); $C_{\text{tok}}$ dominates when generation is long. pass@$k$ is *not* a budget: a 6000-token whole-proof sample and a 40-token tactic step both count as 1.

**Success curve.** For problem set $\mathcal{D}$,

$$s(C) = \frac{1}{|\mathcal{D}|}\sum_{x \in \mathcal{D}} \mathbb{1}\big[\exists\, \text{closed tree found under budget } C\big],$$

with the standard unbiased pass@$k$ estimator (Chen et al., 2021) when $n>k$ samples are drawn. The hypothesized form is a saturating power law

$$1 - s(C) \;=\; (1-s_\infty)\;+\;\alpha\, C^{-\beta},$$

where $s_\infty < 1$ is the **support ceiling**: problems whose proofs lie outside the policy's reachable set at any budget. $\beta$ is the scaling exponent; the exchange-rate question is whether $\Delta \log C_{\text{train}}$ and $\Delta \log C_{\text{search}}$ trade at a fixed ratio.

**Assumptions, and where they break.**
- *Independence of samples.* Assumed by pass@$k$; violated — samples from one policy at temperature $T$ are strongly correlated, so pass@$k$ curves flatten faster than an i.i.d. model predicts.
- *Kernel calls are $O(1)$.* Violated: cost varies by 3+ orders of magnitude across tactics, so $C_{\text{ker}}$ and $C_{\text{wall}}$ decouple.
- *The benchmark is unseen.* Violated for miniF2F and much of mathlib; both predate current pretraining corpora.
- *Verified = correct.* Holds only up to the specification. A synthesized `sorry`-free proof of a vacuous or under-constrained spec passes the kernel.

## 3. State of the Art

**Empirical SOTA (established, kernel-checked).** DeepSeek-Prover-V2-671B reports 88.9% on miniF2F-test (Ren et al., 2025). Kimina-Prover-72B reports 80.7% (Wang et al., 2025). Goedel-Prover-SFT reports 57.6% pass@32 (Lin et al., 2025). These are benchmark numbers under *differing* budgets, not a controlled comparison — none normalizes $C_{\text{tok}}$ across systems, so they do not establish that any method has a better exponent than any other.

**Search-operator results (established, ablated).** HyperTree Proof Search (Lample et al., NeurIPS 2022) ablates AND-tree search against best-first on the same policy and reports 58.6% on miniF2F-valid. DeepSeek-Prover-V1.5 (Xin et al., ICLR 2025) ablates RMaxTS (intrinsic-reward MCTS) against plain sampling at matched sample counts: 63.5% vs. 60.2% miniF2F-test at $32 \times 6400$ samples. This is the cleanest published operator ablation; its budget axis is samples, not tokens.

**Claimed but unablated.** AlphaProof's IMO-2024 silver-medal result (DeepMind; Nature, 2025) attributes the gain to test-time RL on problem variants, with per-problem budgets reported as days of TPU time. The decomposition of that gain into policy quality versus search budget is not published *(frontier — verify)*.

**Theory SOTA.** Nothing specific to proof search. The nearest result is Jones (2021) on Hex: train-time and test-time compute are substitutable along a smooth log-log frontier with an approximately fixed exchange rate. Hex has a dense value signal and bounded depth; formal proof search has neither. No transfer proof exists.

## 4. What Is Known

- **Coverage grows log-linearly in samples over 4 decades.** Brown et al. (2024) show fraction-solved rises near-log-linearly in $k$ from 1 to $10^4$ across MATH, CodeContests, and SWE-bench Lite (11.7% → 56% single-model on SWE-bench Lite). Measured at 7B–70B.
- **Formal domains inherit this, with a hard verifier.** DeepSeek-Prover-V1.5 shows monotone miniF2F gains from pass@128 to pass@$32{\times}6400$ — roughly 2.5 decades of budget for single-digit percentage points at the top end. 7B scale.
- **Expert iteration converts search into policy.** Polu et al. (ICLR 2023) reach 41.2% miniF2F-test from a 774M model via curriculum-driven expert iteration; the mechanism — search finds proofs, proofs retrain the policy — is independently reproduced (InternLM2.5-StepProver; Goedel-Prover; Lean-STaR).
- **Retrieval helps premise selection.** ReProver (Yang et al., NeurIPS D&B 2023) improves over non-retrieval baselines on LeanDojo's `novel_premises` split, which is constructed to defeat memorization.
- **Ceilings are real and benchmark-dependent.** PutnamBench (Tsoukalas et al., NeurIPS D&B 2024): best reported solves are tens out of 657 formalizations, versus ~89% on miniF2F. The gap is not closed by budget at any published scale.
- **Verification is not the bottleneck it is assumed to be.** For whole-proof methods, kernel time is often the majority of wall-clock; DeepSeek-Prover-V1.5 reports batched Lean 4 verification as a first-class engineering cost.

## 5. What Is Not Known

- **Empirically open.** The exponent $\beta$ and ceiling $s_\infty$ for any fixed prover, measured in $C_{\text{tok}}$ over $\ge 3$ decades, with the search operator ablated at matched budget. Every ingredient exists; nobody has published the matched-budget grid.
- **Empirically open.** Whether the operator ranking (MCTS vs. sampling vs. decomposition) inverts with model scale. The plausible hypothesis — search helps weak policies and stops helping strong ones — has never been tested with the same operator across a 7B/70B/671B ladder.
- **Theoretically open.** Whether a Jones-style compute-exchange law exists for AND-tree search with a sparse terminal reward. No proof either way; no lower bound on samples needed as a function of policy KL to the proof distribution.
- **Methodologically blocked.** "Success" on synthesis benchmarks. A kernel-checked proof of a weak specification is scored identically to a proof of the intended one, and there is no accepted specification-strength metric. Until that exists, $s(C)$ measures proof-finding under whatever spec the benchmark shipped.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by contamination**, not compute alone.

1. **The budget axis is not shared.** Whole-proof sampling and step-level search have per-unit costs differing by ~100×. Comparing at equal $k$ compares nothing. Recomputing published results in $C_{\text{tok}}$ is impossible: token counts are almost never reported.
2. **Policy and search co-adapt.** Expert iteration trains on proofs its own search found, so the "search gain" at iteration $t$ is partly a policy gain from iteration $t-1$. The two are non-identifiable from final numbers alone.
3. **miniF2F is saturated and contaminated.** At 88.9%, the residual is ~50 problems; between-method differences fall inside the seed variance of a 6400-sample run. Its 2021 publication predates current pretraining data, so an unknown fraction of the curve is retrieval, not search.
4. **Cost.** A single 3-decade budget sweep at 671B, ablated over 3 operators and 3 seeds, is on the order of $10^5$–$10^6$ GPU-hours plus a Lean farm. That is why it has not been run, not because it is conceptually hard.

## 7. Current Research (as of 2026)

- **Subgoal decomposition as the scaling primitive.** DeepSeek (V2's recursive subgoal pipeline), Kimina (Numina/Kimi), and Goedel-Prover (Princeton) all convert budget into *decompositions* rather than resamples. Whether this changes $\beta$ or only $\alpha$ is unmeasured.
- **Test-time RL.** AlphaProof-style per-problem adaptation, i.e. spending search budget on gradient steps rather than rollouts *(frontier — verify: the published decomposition is thin)*.
- **Verified software synthesis, not competition math.** Rango (Thompson et al., ICSE 2025) and Baldur (First et al., FSE 2023) target Rocq/Isabelle proofs in real repositories, where specifications are given rather than benchmark-authored — the setting where $s_\infty$ actually matters.
- **Benchmark replacement.** PutnamBench and mathlib-derived `novel_premises` splits, motivated explicitly by miniF2F saturation.
- **Contamination-controlled evaluation.** Petrov et al. (2025) on USAMO-2025 report near-zero rigorous-proof rates on problems released after training cutoffs — informal, but the same contamination logic applies to formal sets.

## 8. Concrete Next Experiment

**Question.** Does search-operator choice change the exponent $\beta$, or only the constant $\alpha$?

**Scale.** One fixed open-weights prover (e.g. a 7B Lean prover) and one 70B-class variant of the same family. Benchmark: PutnamBench (657 problems, unsaturated) plus a held-out mathlib `novel_premises` split. Budget grid in $C_{\text{tok}}$: $\{10^5, 10^6, 10^7, 10^8\}$ tokens per problem — 3 decades, 3 seeds.

**Arms.**
- **Control:** independent whole-proof sampling at $T=1.0$, no tree, no repair. This is the arm every paper skips.
- **A:** best-first tactic search.
- **B:** RMaxTS / MCTS.
- **C:** recursive subgoal decomposition with a fixed decomposer.

All arms share the same policy weights, the same Lean toolchain version, and the same token accountant (prompt + output, cached prefills counted at their real cost).

**Deciding number.** Fit $\log(1-s) = \log\alpha - \beta \log C_{\text{tok}}$ per arm and report $\Delta\beta = \beta_{\text{B or C}} - \beta_{\text{control}}$ with bootstrap CI over seeds and problems. **If $\Delta\beta \le 0.02$ with the CI excluding 0.05, structured search buys a constant factor, not a better exponent** — and the field should spend on policy, not on trees. If $\Delta\beta \ge 0.1$, search is a genuine scaling axis and the exchange-rate question becomes worth a training-side sweep.

Secondary readout: the fitted $s_\infty$ per arm. If all arms share $s_\infty$ within 2%, the ceiling is a policy-support property and no search operator escapes it.

## 9. Key References

- **[Foundational]** Stanislas Polu, Ilya Sutskever. *Generative Language Modeling for Automated Theorem Proving.* 2020. — arXiv:2009.03393
- **[Foundational]** Stanislas Polu, Jesse Michael Han, Kunhao Zheng, Mantas Baksys, Igor Babuschkin, Ilya Sutskever. *Formal Mathematics Statement Curriculum Learning.* ICLR, 2023. — arXiv:2202.01344
- **[Foundational]** Guillaume Lample, Timothée Lacroix, Marie-Anne Lachaux, Aurélien Rodriguez, Amaury Hayat, Thibaut Lavril, Gabriel Ebner, Xavier Martinet. *HyperTree Proof Search for Neural Theorem Proving.* NeurIPS, 2022. — arXiv:2205.11491
- **[SOTA]** Huajian Xin et al. *DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search.* ICLR, 2025. — arXiv:2408.08152
- **[SOTA]** Z.Z. Ren et al. *DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition.* 2025. — arXiv:2504.21801
- **[SOTA]** Yong Lin et al. *Goedel-Prover: A Frontier Model for Open-Source Automated Theorem Proving.* 2025. — arXiv:2502.07640
- **[Benchmark]** Kunhao Zheng, Jesse Michael Han, Stanislas Polu. *miniF2F: A Cross-System Benchmark for Formal Olympiad-Level Mathematics.* ICLR, 2022. — arXiv:2109.00110
- **[Benchmark]** George Tsoukalas et al. *PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2407.11214
- **[Benchmark/Method]** Kaiyu Yang et al. *LeanDojo: Theorem Proving with Retrieval-Augmented Language Models.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.15626
- **[Scaling]** Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[Scaling]** Andy L. Jones. *Scaling Scaling Laws with Board Games.* 2021. — arXiv:2104.03113
- **[Systems]** Emily First, Markus N. Rabe, Talia Ringer, Yuriy Brun. *Baldur: Whole-Proof Generation and Repair with Large Language Models.* ESEC/FSE, 2023.
- **[Systems]** Kyle Thompson, Nuno Saavedra, Pedro Carrott, Kevin Fisher, Alex Sanchez-Stern, Yuriy Brun, João F. Ferreira, Sorin Lerner, Emily First. *Rango: Adaptive Retrieval-Augmented Proving for Automated Software Verification.* ICSE, 2025.
- **[Survey]** Kaiyu Yang et al. *Formal Mathematical Reasoning: A New Frontier in AI.* 2024. — arXiv:2412.16075

## 10. Worked Example

Take one PutnamBench problem and a 7B Lean prover. Suppose the measured per-problem success at four budgets is:

| $C_{\text{tok}}$ | samples (whole-proof, ~3k tok each) | $s$ (over 657 problems) | $1-s$ |
|---|---|---|---|
| $10^5$ | 33 | 0.030 | 0.970 |
| $10^6$ | 333 | 0.052 | 0.948 |
| $10^7$ | 3 333 | 0.076 | 0.924 |
| $10^8$ | 33 333 | 0.098 | 0.902 |

Fit $\log(1-s)$ against $\log C$: the drop from 0.970 to 0.902 over 3 decades gives

$$\beta = -\frac{\log(0.902/0.970)}{3\ln 10 / \ln 10 \cdot \ln 10}\Big/\ldots \;\approx\; \frac{0.0727}{3\ln 10} \approx 0.0105.$$

At $\beta \approx 0.01$, reaching $s = 0.5$ from $s=0.098$ requires $\log_{10} C$ to increase by roughly $\log(0.902/0.5)/(\beta \ln 10) \approx 25$ decades. That is the obstruction made numeric: **the residual is not budget-limited, it is support-limited.** The fitted curve is indistinguishable, over the observable range, from $1-s = 0.89 + \alpha C^{-\beta'}$ with $s_\infty \approx 0.11$ and a much larger $\beta'$ — two models with opposite implications (spend more compute vs. change the policy) that the same four points fit equally well.

Distinguishing them needs a fifth decade, which costs ~10× everything already spent — and even then, on miniF2F rather than PutnamBench the whole exercise is void, because at 88.9% the 50 remaining problems give a standard error on $s$ of about 1.2%, larger than the between-operator differences the fit is trying to resolve. That is why the problem is empirically open and not merely unfinished.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*