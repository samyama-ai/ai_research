---
id: 17-reasoning/compute-matched-reasoning-comparison-protocol
title: "Compute-Matched Comparison Protocol for Reasoning Methods"
topic: 17-reasoning
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Matched Comparison Protocol for Reasoning Methods

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/compute-matched-reasoning-comparison-protocol` · **Status:** methodologically-blocked

## 1. Problem Statement

Inference-time reasoning methods — chain-of-thought, self-consistency, best-of-$N$ with a reward model, tree/beam search over steps, sequential revision, long-CoT RL models — are routinely compared by accuracy on a benchmark. They consume wildly different amounts of compute per question. A method that samples 64 chains and reranks them is not comparable to greedy decoding, and a "reasoning model" that emits 8,000 thinking tokens is not comparable to a base model that emits 300.

**The problem:** define a comparison protocol under which the claim "method $A$ beats method $B$ at equal compute" is well posed, measurable from artifacts a paper can publish, and stable across the free choices (hardware, sampling temperature, judge, seed, benchmark subset) that the protocol does not fix.

Three variants, of different difficulty:

- **Measurement.** What is the right compute unit — FLOPs, generated tokens, wall-clock, dollars, energy? Each induces a different ranking, and no accepted reduction between them exists. This is the blocked variant.
- **Method.** Given a fixed unit, how do you allocate a budget $C$ to each arm fairly, when each arm has its own tunable knobs ($N$, temperature, depth, verifier size) and each has a different budget–accuracy curve shape?
- **Theory.** Under what conditions does a compute-matched ranking at budget $C$ predict the ranking at $10C$? Empirically, curves cross; no theory says when.

A solution is a written protocol plus a reference implementation such that two independent groups, given the same models and benchmark, produce budget–accuracy curves that agree within stated error bars, and such that the induced ranking is invariant to the unit chosen from a declared set.

## 2. Formal Setting

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n}$ be a benchmark, $M$ a model with $N_{\text{act}}$ active parameters per token, and $\mathcal{A}$ a reasoning method with knob vector $\theta$ (samples, depth, temperature, verifier). One run of $\mathcal{A}$ on $x_i$ produces a transcript.

**Compute, as measured.** For a dense transformer, the standard accounting is $2N$ FLOPs per token forward:

$$C_{\text{FLOP}}(\mathcal{A},\theta,x_i) = 2N_{\text{act}}\big(T^{\text{in}}_i + T^{\text{out}}_i\big) + 2N^{v}_{\text{act}}\big(T^{v,\text{in}}_i + T^{v,\text{out}}_i\big) + c_{\text{attn}},$$

with $T^{\text{in}}, T^{\text{out}}$ prefill and generated token counts summed over all calls, superscript $v$ the verifier/reward model, and $c_{\text{attn}} \approx 2 L\, d\, T^2$-scale attention terms usually dropped. Report $\bar{C} = \frac1n\sum_i C(\cdot,x_i)$ and the full distribution — reasoning-token counts are heavy-tailed, so the mean is not the median.

Alternative units, none reducible to another without hardware assumptions: generated tokens $T^{\text{out}}$; serving cost $\$ = p_{\text{in}}T^{\text{in}} + p_{\text{out}}T^{\text{out}}$; wall-clock latency $L$, which depends on batch size and on whether the method is sequential (revision) or parallel (best-of-$N$); energy in joules.

**Accuracy.** $\hat{a}(\mathcal{A},\theta) = \frac1n \sum_i \mathbb{1}[g(\hat{y}_i, y_i)]$ with grader $g$ (exact match, symbolic equivalence, or an LLM judge). Sampling variance from $K$ seeds: report $\hat{a} \pm z\sqrt{\hat{a}(1-\hat{a})/n + s^2_{\text{seed}}/K}$ — both terms, since benchmark-size variance and seed variance are separate and both large at $n=30$ (Miller, 2024).

**The comparison predicate.** Define the budget–accuracy frontier

$$F_{\mathcal{A}}(C) = \max_{\theta \,:\, \bar{C}(\mathcal{A},\theta)\le C} \; \hat{a}(\mathcal{A},\theta),$$

taken over a *declared* knob grid, with the $\theta$-selection done on a held-out split (otherwise $F$ is a max over noise and is upward-biased by $\approx z\, s\sqrt{2\log|\Theta|}$). The claim "$A$ beats $B$ at compute $C$" is $F_A(C) > F_B(C)$ with non-overlapping intervals.

**Assumptions, and which are violated.**

1. *Compute is one-dimensional.* Violated: parallel and sequential compute have the same FLOPs and different latency; the ranking flips between the $C_{\text{FLOP}}$ and $L$ units.
2. *Verifier compute is charged to the method.* Routinely violated — most best-of-$N$ papers report the generator's samples and omit the reward-model passes.
3. *Frontier is monotone and non-crossing.* Violated: pass@$k$-style coverage curves cross accuracy curves once an imperfect verifier is in the loop (Stroebl et al., 2024).
4. *Grader is method-independent.* Violated when a judge model rewards long, structured outputs, which the high-compute arm produces by construction.
5. *Benchmark items are i.i.d. and uncontaminated.* Violated for MATH/GSM8K-era sets and unmeasurable for closed models.

## 3. State of the Art

**Established.** Compute-matched frontiers exist as a research object. Snell et al. (ICLR 2025) plot accuracy against a generation-FLOP budget for revision and search on MATH with PaLM-2 and report that a compute-optimal allocation is up to $4\times$ more token-efficient than best-of-$N$, and that at small budgets test-time compute can beat a $\sim14\times$ larger model — while stating the reverse at large budgets and hard questions. Brown et al. (2024) hold the axis fixed at number of samples and report coverage curves that are near log-linear in $k$. Wu et al. (2025) fit inference-compute scaling laws across model sizes and strategies. These are the only widely-used protocols, and they are *pairwise* protocols, not a standard.

**Claimed but unablated.** Most "our method beats CoT" results in the reasoning literature report a single operating point per arm with no budget axis, no verifier compute, and one seed. Claims that long-CoT RL models "reason better" typically compare against a base model at $\sim20\times$ fewer output tokens. The s1 result (Muennighoff et al., 2025) — budget forcing with 1k training samples — is an honest test-time-scaling curve on AIME24/MATH500, but AIME24 has 30 items, so each item is 3.3 accuracy points.

**Benchmark-number-only.** Leaderboard entries for reasoning models (AIME, GPQA-Diamond, ARC-AGI) publish accuracy with, at best, a coarse "reasoning effort" label. Except where a vendor publishes a cost axis, these numbers cannot be placed on any compute-matched frontier at all.

## 4. What Is Known

- **Repeated sampling buys coverage at predictable rates.** Brown et al. (2024): DeepSeek-Coder-V2-Instruct on SWE-bench Lite goes from 15.9% with one sample to 56% coverage at 250 samples; on GSM8K/MATH with Llama-3-8B-Instruct, coverage grows near log-linearly over 4 orders of magnitude in $k$. Scale: 8B–236B open models.
- **Coverage is not accuracy.** With an imperfect verifier, resampling gains saturate and can invert; Stroebl et al. (2024) show false-positive selection dominates past a modest $k$ on code tasks. Same scale.
- **Compute-optimal allocation is question-dependent.** Snell et al. (ICLR 2025), PaLM-2-S* on MATH: easy questions favor sequential revision, hard questions favor parallel search; a fixed allocation loses to a difficulty-conditioned one by a factor of $\sim4$ in tokens.
- **Seed and format variance is comparable to reported deltas.** Hochlehnert et al. (2025), "A Sober Look at Progress in Language Model Reasoning": on AIME'24-scale sets, seed-to-seed swings of several points are routine and many published gains fall inside them. Sclar et al. (ICLR 2024) show prompt-format changes alone move accuracy by tens of points on some tasks at LLaMA-2-13B scale.
- **Metric choice manufactures qualitative claims.** Schaeffer et al. (NeurIPS 2023): discontinuous metrics produce apparent emergence that vanishes under continuous ones. The same mechanism applies to thresholded reasoning accuracies.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted compute unit, no accepted rule for charging verifier/judge/retry compute, and no accepted rule for whether the knob grid is searched on the test set. Until these are fixed, "compute-matched" is not a defined predicate, and two honest groups can reach opposite rankings from identical raw data.
- **Empirically open.** Nobody has published a multi-arm, multi-unit frontier study — same models, same benchmark, four or more methods, curves in FLOPs *and* latency *and* dollars — large enough to test whether the unit changes the ranking. The experiment is runnable today for well under \$50k.
- **Theoretically open.** No result gives conditions under which $F_A(C) > F_B(C)$ implies $F_A(\lambda C) > F_B(\lambda C)$ for $\lambda > 1$. Crossings are observed; their structure is uncharacterized.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the compute axis combined with an evaluation that does not measure what it names**.

Non-identifiability: FLOPs, latency, and dollars are not monotone transforms of each other. Best-of-64 and a 64-step sequential revision chain can have identical FLOPs, a $64\times$ latency ratio, and different dollar costs under any batched serving stack. So "equal compute" picks out at least three distinct experiments with different answers, and nothing in the phenomenon privileges one.

Confounded measurement compounds it: the arm with more compute also produces longer outputs, which changes judge behavior, changes exact-match parsing failure rates, and changes contamination exposure. Charging the verifier honestly can double an arm's cost and flip the ranking, so the choice is outcome-determining and is currently unstated. Add benchmark sizes of $n=30$–$500$, where the binomial standard error alone is 2–9 points, and the protocol's free choices are larger than the effects it is used to adjudicate.

## 7. Current Research (as of 2026)

- **Scaling-law framing of inference compute** — Snell/Kumar (Berkeley, CMU, DeepMind), Wu et al. (CMU) — fitting $\hat{a}$ as a function of budget rather than reporting points.
- **Evaluation-hygiene work** — Hochlehnert et al. (Tübingen/Bethge lab) on seed and harness variance in reasoning evals; Miller (Anthropic) on error bars; the HELM/Eleuther harness maintainers on standardized decoding configs. *(frontier — verify: whether any harness has landed first-class token-cost logging as a required field.)*
- **Verifier-limited scaling** — Stroebl, Kapoor, Narayanan (Princeton) on imperfect verifiers bounding resampling gains.
- **Efficiency-aware leaderboards** — vendor-published cost-vs-accuracy axes for reasoning modes, and ARC-AGI's \$-per-task reporting, which is the closest existing thing to a normative protocol. *(frontier — verify: adoption outside ARC.)*
- **Token-budget-controlled training** — s1-style budget forcing (Stanford/UW) makes the budget an explicit input, which is a prerequisite for clean matching.

## 8. Concrete Next Experiment

**The unit-inversion test.** One number decides whether the problem is real.

- **Scale.** Two open-weight models (e.g. an 8B and a 70B dense instruct model), one clean benchmark with $n \ge 500$ items (MATH500 plus a held-out 500-item set for knob selection), 5 seeds per configuration.
- **Arms.** (i) greedy CoT; (ii) self-consistency at $N \in \{4,16,64\}$; (iii) best-of-$N$ with a 7B reward model, $N \in \{4,16,64\}$, verifier FLOPs charged; (iv) sequential revision at depth $\in \{4,16,64\}$. Knobs selected on the held-out split only.
- **Control arm.** The larger model at greedy decoding, which is the "just buy more parameters" baseline every method must beat, and which fixes the FLOP axis independently of any test-time trick.
- **Measurement.** Log per-question $T^{\text{in}}, T^{\text{out}}$, verifier tokens, and batched wall-clock at fixed batch size. Emit three frontiers per arm: $F(C_{\text{FLOP}})$, $F(L)$, $F(\$)$.
- **Deciding number.** The **rank-inversion rate**: the fraction of budget decades in which the top-ranked arm under $C_{\text{FLOP}}$ differs from the top-ranked arm under $L$, counting only budgets where the accuracy gap exceeds the 95% interval. If that rate is $0$, a single unit suffices and the problem collapses to bookkeeping. If it exceeds $\sim20\%$, every single-axis compute-matched claim in the literature is under-specified and the protocol must report a frontier per unit.

Cost estimate: roughly $4 \times 3 \times 5 \times 1000$ questions $\times \le 64$ samples, on the order of $10^{9}$ generated tokens — a few thousand GPU-hours on 8×H100.

## 9. Key References

- **[Foundational]** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS, 2022. — arXiv:2201.11903
- **[Foundational]** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR, 2023. — arXiv:2203.11171
- **[SOTA]** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* ICLR, 2025. — arXiv:2408.03314
- **[SOTA]** Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- **[Critique]** Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan. *Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[Critique]** Andreas Hochlehnert, Hardik Bhatnagar, Vishaal Udandarao, Samuel Albanie, Ameya Prabhu, Matthias Bethge. *A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility.* 2025. — arXiv:2504.07086
- **[Critique]** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004
- **[Methods]** Melanie Sclar, Yejin Choi, Yulia Tsvetkov, Alane Suhr. *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.* ICLR, 2024. — arXiv:2310.11324
- **[Methods]** Evan Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* Anthropic, 2024. — arXiv:2411.00640
- **[Verifiers]** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[Scaling]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556 — the precedent: a compute axis defined precisely enough to settle a debate.

## 10. Worked Example

Take an 8B dense model ($N_{\text{act}} = 8\times10^9$) on one MATH-style question. Prompt: 200 tokens. A CoT solution: 500 output tokens.

**Arm A — greedy CoT.** $C = 2(8\times10^9)(700) = 1.12\times10^{13}$ FLOPs. Latency: 500 sequential decode steps.

**Arm B — best-of-16 with a 7B reward model.** Generator: 16 samples, prefill shared, $C_{\text{gen}} = 2(8\times10^9)(200 + 16\cdot500) = 1.31\times10^{14}$. Verifier scores 16 candidates of 700 tokens each: $C_{\text{ver}} = 2(7\times10^9)(16\cdot700) = 1.57\times10^{14}$. Total $2.88\times10^{14}$ — **26× arm A**, and *the verifier is 55% of it*.

**Arm C — the control, a 70B model, greedy.** $C = 2(7\times10^{10})(700) = 9.8\times10^{13}$ FLOPs — **one third of arm B**.

Now the obstruction. Suppose arm B scores 62%, arm C scores 60%, arm A 45%, each on $n=500$ with a 95% interval of $\pm4.3$ points.

- Under $C_{\text{FLOP}}$ *with verifier charged*: arm B costs $2.9\times$ arm C for a 2-point gain inside the error bar. Best-of-$N$ loses; "buy a bigger model" wins.
- Under $C_{\text{FLOP}}$ *without verifier charged* — the convention in most papers — arm B costs $1.3\times$ arm C and is reported as the compute-matched winner.
- Under **latency**, arm B's 16 samples run in parallel: its critical path is 500 decode steps on an 8B model, roughly $2$–$3\times$ *faster* than arm C's 500 steps on a 70B model. Arm B wins decisively.
- Under **dollars** at typical served prices, arm B's 8,200 generated tokens versus arm C's 500 at ~9× the per-token rate puts the two within ~2× of each other, direction depending on the vendor's price ratio.

One set of transcripts. Three defensible units and one unstated bookkeeping choice. Four rankings, of which three can be published truthfully. That is the problem: not that the experiment is expensive, but that its conclusion is a free parameter until the protocol fixes the axis.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*