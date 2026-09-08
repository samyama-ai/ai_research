---
id: 17-reasoning/compute-optimal-pretrain-vs-inference-allocation
title: "Compute-Optimal Allocation Between Pretraining and Inference-Time Search"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Allocation Between Pretraining and Inference-Time Search

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/compute-optimal-pretrain-vs-inference-allocation` · **Status:** empirically-open

## 1. Problem Statement

Given a total compute budget $C_{\text{tot}}$ split between training a model and running search at inference, how should it be split to maximise task accuracy?

- **Input:** a lifetime budget $C_{\text{tot}}$ (FLOPs), an expected query load $Q$, a task distribution $\mathcal{D}$, a training recipe, and an inference-time search procedure (best-of-$k$, beam/lookahead search over a process reward model, sequential revision, long chain-of-thought with budget forcing).
- **Output:** a triple $(N^\star, D^\star, k^\star)$ — parameters, training tokens, per-query search budget.
- **Predicate:** $(N^\star, D^\star, k^\star)$ maximises $\mathbb{E}_{x\sim\mathcal{D}}[\,\text{acc}\,]$ subject to $C_{\text{train}}(N,D) + Q\cdot C_{\text{inf}}(N,k) \le C_{\text{tot}}$.

Three variants, with different difficulty:

- **Measurement:** does an exchange rate between pretraining FLOPs and inference FLOPs exist and is it stable across tasks, model families and difficulty strata? Currently the weakest link — the exchange rate is measured only on MATH/GSM8K-scale benchmarks with saturating metrics.
- **Method:** an allocation rule computable *before* training, from small-scale probes, that predicts the frontier at 10–100× the probe scale.
- **Theory:** conditions under which inference search is a *substitute* for parameters (accuracy depends only on a scalar aggregate of the two) versus a *complement* (search gains are gated by base-model coverage, so they cannot replace pretraining at any budget).

## 2. Formal Setting

Model with $N$ non-embedding parameters trained on $D$ tokens. Measured quantities:

$$C_{\text{train}} = 6ND \quad\text{FLOPs (Kaplan et al. 2020 accounting; measured as } 6ND\text{, not wall-clock).}$$

Inference cost for one sampled solution of $T$ generated tokens, prompt length $P$, with $k$ samples and a verifier/reward model of size $N_v$:

$$C_{\text{inf}}(N,k) = k\big[2N(P+T)\big] + k\big[2N_v(P+T)\big] + C_{\text{agg}}$$

Measured as: forward-pass FLOPs counted per token emitted, $T$ taken as the *empirical mean* generation length under the search policy (not the cap — budget-forced long-CoT has heavy-tailed $T$), $C_{\text{agg}}$ the aggregation/majority-vote cost, usually negligible. KV-cache reuse across samples reduces prefill but not decode; report both.

Accuracy of policy $\pi_{N,D}$ under search operator $S_k$ with verifier $v$:

$$A(N,D,k) = \mathbb{E}_{x\sim\mathcal{D}}\big[\mathbb{1}\{\text{correct}(S_k(\pi_{N,D},v,x))\}\big].$$

Two ceilings, both measured, not assumed:

$$\underbrace{\text{cov}_k(x) = 1-\big(1-p(x)\big)^k}_{\text{pass@}k\text{, oracle verifier}} \qquad A(N,D,k) \le \mathbb{E}_x[\text{cov}_k(x)].$$

The **allocation problem** is $\max_{N,D,k} A(N,D,k)$ s.t. $6ND + Q\,C_{\text{inf}}(N,k) \le C_{\text{tot}}$.

Assumptions, with those known to be violated flagged:

1. **Separability** — $A$ factorises into a training-quality term and a search-gain term. *Violated:* search gain depends on per-example $p(x)$, which is a function of the trained model, so the terms are coupled (Brown et al. 2024).
2. **Verifier is free and correct.** *Violated:* on GSM8K, automatic verification via answer matching admits false positives; learned reward models are the binding constraint on realised accuracy (Stroebl et al. 2024).
3. **Uniform query load** — every query gets $k$ samples. *Violated:* difficulty-adaptive allocation beats uniform by a large margin (Snell et al. 2024).
4. **FLOPs are the currency.** *Violated in deployment:* decode is memory-bandwidth-bound; a 7B model at $k{=}64$ and a 70B model at $k{=}1$ with equal FLOPs have different latency and different \$/token.
5. **Stationary $\mathcal{D}$ over the model's lifetime**, needed to fix $Q$ at training time.

## 3. State of the Art

**Established (ablated, FLOP-matched):**

- Snell et al. (2024, ICLR 2025) compare test-time compute against a $\sim$14× larger pretrained model under FLOP matching on MATH with PaLM 2-S\*. On easy/medium questions and at low inference load, optimal test-time scaling beats the larger model; on the hardest strata, and once the inference:pretraining token ratio grows large, the larger model wins. Their "compute-optimal" strategy — choose revision-vs-search per difficulty bin — beats best-of-$N$ at $\sim$4× less test-time compute.
- Wu et al. (2024/2025) show smaller models with better search dominate: Llemma-7B with REBASE tree search matches or beats Llemma-34B at equal FLOPs on MATH500/GSM8K, and majority voting is *not* compute-optimal at large budgets (it saturates).
- Sardana et al., *Beyond Chinchilla-Optimal* (ICML 2025), modify Chinchilla to include inference demand: with substantial $Q$, optimal models are smaller than Chinchilla and trained on many more tokens. This is a **cost model**, not an accuracy model — it holds loss fixed and minimises FLOPs/dollars.

**Claimed but unablated / benchmark-only:** long-CoT RL results (OpenAI o1, 2024; DeepSeek-R1, 2025) report accuracy rising with test-time tokens, but no FLOP-matched arm holds pretraining constant against an equivalently-priced larger base model. s1 (Muennighoff et al. 2025) reports test-time scaling from 1k SFT examples with budget forcing — a benchmark number with a strong control on *data*, none on *pretraining allocation*.

## 4. What Is Known

- **Coverage scales log-linearly in $k$ over orders of magnitude.** Brown et al. (2024): DeepSeek-V2-Coder-Instruct on SWE-bench Lite goes from 15.9% (1 sample) to 56% (250 samples) — above the 43% single-attempt SOTA of the time. Across GSM8K/MATH/MiniF2F, $\log(\text{cov})$ is near-linear in $\log k$.
- **Selection, not generation, is the bottleneck.** In the same work, on GSM8K/MATH with no automatic verifier, majority vote and reward-model selection plateau by $k\approx100$ while coverage keeps rising. Stroebl et al. (2024) prove/measure that with an imperfect verifier of fixed false-positive rate, accuracy is non-monotone in $k$ and eventually *decreases*.
- **Search gains are difficulty-gated.** Snell et al. (2024) find test-time compute substitutes for pretraining only on the two easiest MATH difficulty quintiles; the deficit on the hardest quintile is not closed at any tested budget.
- **RL-tuned reasoning models often do not raise pass@$k$ at large $k$ over their base model** (Yue et al., 2025, on math/code benchmarks) — evidence that some inference-time gains are sharpening, not new capability.
- **Chinchilla scaling** (Hoffmann et al. 2022, 70B/1.4T): $N\propto C^{0.5}$, $D\propto C^{0.5}$ at training-compute optimum — the baseline the allocation problem perturbs.

## 5. What Is Not Known

- **Theoretically open.** Whether a joint scaling law $A(N,D,k)$ with an interaction term exists in a form that is identifiable from small-scale fits. Candidate parametric models exist (Levi 2024; Chen et al., NeurIPS 2024, for compound systems) but none has been shown to extrapolate. No theorem states when search is a substitute versus a complement for parameters.
- **Empirically open (the main gap).** No published study sweeps $(N, D, k)$ jointly on a shared grid at $\ge 10^{22}$ FLOPs with a fixed search operator and a fixed verifier. Existing work varies one axis and holds the others at convenient values. The experiment is runnable at a few hundred thousand GPU-hours; nobody has run it publicly.
- **Methodologically blocked.** The *cross-over query count* $Q^\star$ — the load at which retraining beats searching — is not well-defined without a fixed verifier and a fixed difficulty distribution, and both are typically re-tuned per arm. Also blocked: comparing FLOPs across arms with different memory-boundedness.

## 6. Why It Is Hard

Three named obstructions.

1. **Non-identifiability of the interaction term.** Accuracy gain from search is a functional of the per-example success distribution $p(x)$, which shifts with $N$ and $D$. Fitting $A(N,D,k)$ from a low-$k$, small-$N$ probe cannot distinguish "search adds a fixed offset" from "search amplifies existing coverage", and the two extrapolate to opposite recommendations.
2. **The verifier confounds the measurement.** Reported test-time scaling curves mix generator quality with verifier quality. A study that improves the reward model between arms measures the reward model, not the allocation.
3. **Benchmarks saturate before the exchange rate stabilises.** MATH500/GSM8K accuracies above ~90% compress differences into noise, so the estimated exchange rate is fit on the flat part of the curve — an evaluation that does not measure the thing it names.
4. **Cost:** an honest grid is $\ge 5$ model sizes $\times$ 3 token multipliers $\times$ 5 search budgets, each with held-out inference sweeps.

## 7. Current Research (as of 2026)

- **Inference-aware scaling laws:** Sardana/Frankle-line work extending Chinchilla to inference demand; extensions to accuracy (not loss) targets are active *(frontier — verify)*.
- **Difficulty-adaptive allocation and routing:** per-query budget prediction, early exit, "don't overthink" length control — Google DeepMind and UC Berkeley lines from Snell et al.
- **Verifier scaling:** generative reward models and self-verification, where the open question is how to split budget between generator and verifier *(frontier — verify)*.
- **RL-for-reasoning vs. search substitution:** whether RL-trained long CoT is a cheaper way to buy the same accuracy as parallel search (DeepSeek, Qwen, open replications).
- **Compound-system scaling:** Chen et al.'s law for number of LLM calls, extended to agent scaffolds.

## 8. Concrete Next Experiment

**Scale.** Pretrain a $3\times3$ grid on one corpus: $N \in \{0.5\text{B}, 1.5\text{B}, 7\text{B}\}$, $D \in \{1\times, 4\times, 16\times\}$ Chinchilla-optimal tokens. Total $\approx 5\times10^{22}$ training FLOPs.

**Search arms.** One frozen, externally-trained verifier (same for every arm, never re-tuned) and one search operator (best-of-$k$ with the frozen PRM), $k \in \{1,4,16,64,256\}$.

**Evaluation.** A non-saturating, difficulty-stratified set — e.g. held-out competition math plus SWE-bench-style tasks — with per-example $p(x)$ recorded so coverage and selection are separable.

**Control arm.** The Chinchilla-optimal 7B model at $k{=}1$, and its FLOP-matched partner: the largest model trainable if the entire inference budget of the $k{=}256$ arm were moved into pretraining.

**The deciding number.** Fit $A(N,D,k)$ on the $\{0.5\text{B},1.5\text{B}\}$ rows only, predict the 7B row, and report the **signed error in predicted iso-accuracy exchange rate** $\rho = \partial \log C_{\text{train}} / \partial \log C_{\text{inf}}$ at fixed $A$. If $|\hat\rho - \rho_{\text{true}}| / \rho_{\text{true}} < 0.25$ at the held-out scale, allocation is predictable from cheap probes and the method variant is solved for this task family. If not, the interaction term is not identifiable at probe scale — which is itself the result.

## 9. Key References

- **[Foundational]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Jared Kaplan et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[SOTA]** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* ICLR 2025. — arXiv:2408.03314
- **[SOTA]** Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, Yiming Yang. *Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models.* ICML 2025. — arXiv:2408.00724
- **[SOTA]** Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML 2024/2025. — arXiv:2401.00448
- **[Empirical]** Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[Empirical]** Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan. *Inference Scaling $f$Laws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[Empirical]** Niklas Muennighoff, Zitong Yang, Weijia Shi, et al. *s1: Simple Test-Time Scaling.* 2025. — arXiv:2501.19393
- **[Theory]** Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Ion Stoica, Matei Zaharia, James Zou. *Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems.* NeurIPS 2024.
- **[Related]** Yang Yue et al. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* 2025. — arXiv:2504.13837

## 10. Worked Example

Take a 7B model trained on 2T tokens.

$$C_{\text{train}} = 6 \times 7\times10^9 \times 2\times10^{12} = 8.4\times10^{22}\ \text{FLOPs}.$$

One solution at 500 generated tokens, prompt 200, no verifier: $2N(P+T) = 2(7\times10^9)(700) = 9.8\times10^{12}$ FLOPs. At $k=64$: $6.3\times10^{14}$ FLOPs per query.

Break-even query count against retraining a 70B model ($C_{\text{train}} = 8.4\times10^{23}$, i.e. $7.6\times10^{23}$ extra):

$$Q^\star = \frac{7.6\times10^{23}}{6.3\times10^{14}} \approx 1.2\times10^{9}\ \text{queries.}$$

So under ~1.2B hard queries, $k{=}64$ search on the 7B model is the cheaper way to spend the marginal FLOP — *if* it buys the same accuracy.

Now the obstruction. On a hard math stratum, suppose the 7B model has $\text{pass@1} = 0.31$ and $\text{pass@64} = 0.87$ (coverage). With a PRM of 85% pairwise selection accuracy, realised best-of-64 accuracy lands near 0.55, not 0.87 — and rises to only ~0.57 at $k{=}256$, because false-positive selections grow with $k$ (Stroebl et al. 2024). The 70B model at $k{=}1$ might sit at 0.52 and at $k{=}64$ at 0.71.

The allocation answer therefore flips on a quantity absent from the budget equation: verifier quality. At oracle verification the 7B+search arm wins by 16 points; at the realised PRM it wins by 3 points, inside the noise band of a 500-problem eval ($\pm 4.4$ points at 95%). Two labs running the "same" FLOP-matched comparison with different reward models will report opposite compute-optimal allocations, and both will be right about their own system. That is why the problem is empirically open rather than merely unmeasured: the measurement is only well-posed once the verifier is fixed and reported as a third budget axis.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*