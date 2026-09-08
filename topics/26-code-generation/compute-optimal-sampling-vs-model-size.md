---
id: 26-code-generation/compute-optimal-sampling-vs-model-size
title: "Compute-Optimal Sampling versus Model Size for Program Synthesis"
topic: 26-code-generation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Sampling versus Model Size for Program Synthesis

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/compute-optimal-sampling-vs-model-size` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed inference-time compute budget $C$ (FLOPs, or dollars, or wall-clock) and a program-synthesis task, how should the budget be split between **model size** and **number of samples**?

Input: a specification $s$ (natural-language prompt plus optional visible tests), a family of code models $\{M_N\}$ indexed by parameter count $N$, a sampling procedure, and a selector (execution against tests, a reward model, or majority vote over outputs).
Output: a single program $\hat p$.
Objective: maximise $\Pr[\hat p \text{ correct}]$ subject to $C_{\text{inference}} \le C$.

Three variants, with different difficulty:

- **Measurement.** Does the optimal $(N, k)$ frontier for code exist and is it stable across benchmarks, or is it an artifact of test-suite quality? Currently the weakest link.
- **Method.** Build a selector whose precision does not degrade as $k$ grows, so that coverage gains convert into solve-rate gains.
- **Theory.** Derive $N^*(C)$, $k^*(C)$ from the distributional form of per-problem success probabilities — an inference-side analogue of Chinchilla.

Solving it means: a published, reproduced allocation rule $k^*(C) \propto C^{\beta}$, $N^*(C) \propto C^{1-\beta}$ with a measured exponent $\beta$ and stated validity range, verified out-of-sample on a benchmark with contamination controls and a verifier whose false-positive rate is quantified.

## 2. Formal Setting

Let $\mathcal{D}$ be a distribution over problems $x$. For model $M_N$ at temperature $T$, define per-problem single-sample success

$$p_N(x) = \Pr_{y \sim M_N(\cdot \mid x, T)}\big[V(x, y) = 1\big],$$

where $V$ is the **verifier** actually used at evaluation time. Measured by drawing $n \gg k$ samples and counting $c$ passes; $\hat p_N(x) = c/n$.

**Coverage** (pass@$k$, the Codex unbiased estimator, Chen et al. 2021):

$$\text{pass@}k(N) = \mathbb{E}_{x}\left[1 - \binom{n-c}{k}\Big/\binom{n}{k}\right].$$

Coverage is an upper bound on any selector. **Solve rate** under selector $\sigma$ mapping $k$ candidates to one:

$$\text{acc}_\sigma(N,k) = \mathbb{E}_x\Pr\big[V^\star(x, \sigma(y_{1:k})) = 1\big],$$

with $V^\star$ the *true* correctness oracle (held-out tests, human review, or a proof). The gap between $V$ and $V^\star$ is the false-positive rate $\phi = \Pr[V=1 \mid V^\star=0]$ — measured, not assumed.

**Compute.** Inference cost per token $\approx 2N$ FLOPs; a sample of $L_{\text{out}}$ tokens with prompt $L_{\text{in}}$ costs $\approx 2N(L_{\text{in}} + L_{\text{out}})$ ignoring attention and KV-cache reuse. Total:

$$C(N,k) \approx 2Nk(L_{\text{in}} + L_{\text{out}}) + C_{\text{verify}} \cdot k.$$

$C_{\text{verify}}$ (sandboxed test execution) is not FLOPs and is usually dropped — an assumption violated whenever test suites are slow or the verifier is itself a model.

**Allocation problem.** $\max_{N,k} \text{acc}_\sigma(N,k)$ s.t. $C(N,k) \le C$. Fit $k^*(C) \propto C^\beta$.

**Assumptions and their violations:**
1. *Samples are i.i.d.* — violated; temperature sampling from one prompt has strongly correlated failure modes, so coverage saturates faster than a binomial predicts.
2. *$V = V^\star$* — violated. HumanEval/MBPP test suites admit false positives (HumanEval+ / MBPP+, Liu et al. 2023, cut reported pass rates by up to ~19 points).
3. *Cost is linear in $N$* — violated by MoE models, quantisation, batching, and speculative decoding; $2N$ per token is a dense-model idealisation.
4. *Problems are exchangeable* — violated; $p_N(x)$ is heavy-tailed, and the frontier is dominated by problems with $p \approx 0$.
5. *No train/test contamination* — violated on HumanEval and MBPP for models after 2023.

## 3. State of the Art

**Established (reproduced, ablated):**
- *Coverage is log-linear in $\log k$ over 4+ orders of magnitude.* Brown et al., *Large Language Monkeys* (2024), across CodeContests, SWE-bench Lite, GSM8K, MiniF2F, models 70M–70B.
- *Selector precision, not coverage, is the binding constraint.* Same paper: on CodeContests and SWE-bench, coverage far exceeds achievable solve rate when only model-based or majority-vote selection is available.
- *Repeated sampling from a small model can beat one sample from a large model at equal cost when a strong verifier exists.* Hassid et al., *The Larger the Better? Improved LLM Code-Generation via Budget Reallocation* (COLM 2024) — and the same paper shows the ordering **reverses** without a ranker.

**Claimed but unablated / benchmark-number-only:**
- Snell et al. (2024) report that test-time-compute-optimal scaling lets a small model beat a $\sim$14$\times$ larger one under FLOPs-matched comparison — measured on MATH, with math verifiers, not on code. Transfer to program synthesis is asserted, not shown.
- Wu et al., *An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models* (2024): Llemma-7B with REBASE tree search beats Llemma-34B at equal budget on MATH/GSM8K. Again math, not code; single model family.
- AlphaCode (Li et al., *Science* 2022): $10^6$ samples filtered to 10 submissions, 54.3% Codeforces percentile. A benchmark number under a very specific filtering pipeline; the $(N,k)$ trade-off was not swept.
- Vendor claims that "reasoning" models make sampling obsolete for code: no public FLOPs-matched ablation against sampling from a non-reasoning model of matched size.

**Theory SOTA.** Schaeffer et al. (2025) explain the log-linear coverage curve: if per-problem $p_N(x)$ has a distribution with polynomial mass near $0$, then $1 - \mathbb{E}[(1-p)^k]$ follows a power law in $k$ even though each problem's curve is exponential. This is a mechanism, not an allocation rule — it does not predict $\beta$.

## 4. What Is Known

- **Codex 12B (2021):** HumanEval pass@1 = 28.8%, pass@100 = 72.3% ($T$ tuned per $k$). A $2.5\times$ gap between one sample and 100.
- **Llama-3-8B-Instruct on SWE-bench Lite (Brown et al. 2024):** coverage 15.9% at $k=1$ → 56% at $k=250$. DeepSeek-Coder-V2-Instruct at $k=250$ reached 56% solve rate with a real verifier, exceeding the then-best single-attempt agent (43%) at roughly $30\times$ lower cost.
- **Chinchilla (Hoffmann et al., NeurIPS 2022):** at *training* time, $N^\ast \propto C^{0.5}$, $D^\ast \propto C^{0.5}$, $\approx 20$ tokens/parameter. The inference-side analogue has no comparable fit.
- **Verifier ceiling (Stroebl et al. 2024):** with imperfect verifiers, resampling gains plateau and can *invert* — accuracy under $V^\star$ falls as $k$ grows because false positives accumulate. Shown on HumanEval and MBPP.
- **HumanEval+ (Liu et al., NeurIPS 2023):** 80$\times$ more tests; reported pass@1 for many models drops by 10–19 absolute points. This is the size of the $V$ vs $V^\star$ gap on the field's most-used code benchmark.

## 5. What Is Not Known

- **Empirically open.** No FLOPs-matched sweep over $(N, k)$ for code across a model family spanning $\geq 10\times$ in $N$, on a contamination-controlled benchmark, with $k$ up to $10^3$ and a quantified-$\phi$ verifier. The experiment is runnable today for well under $10^5$ GPU-hours. Nobody has published it.
- **Empirically open.** Whether $\beta$ depends on task difficulty (competitive programming vs repository patching) or is a property of the model family.
- **Theoretically open.** No result relating the tail of the $p_N(x)$ distribution to $N$. Without $\partial p_N/\partial N$, the allocation problem cannot be solved analytically even given Schaeffer et al.'s mechanism.
- **Methodologically blocked.** "Correct program" has no cheap oracle. Coverage under a weak $V$ is not the quantity of interest, and the standard fix (more tests) shifts $\phi$ by an unknown amount rather than driving it to zero.

## 6. Why It Is Hard

The obstruction is **confounded measurement, not compute**. Every published $(N,k)$ comparison entangles four factors that move together: (i) the verifier's false-positive rate $\phi$, which sets the ceiling for large $k$; (ii) benchmark contamination, which inflates $p_N$ for large $N$ specifically, since larger models memorise more; (iii) cost accounting, since MoE, batching and KV-cache reuse make $2Nk L$ wrong by up to an order of magnitude and in a direction that favours large $k$; (iv) temperature, which is optimal at different values for different $k$ and is almost never re-tuned per arm.

A second obstruction is **non-identifiability of the frontier from aggregate scores**. Two model families with identical pass@1 and identical pass@100 can have completely different $p_N(x)$ distributions — one with many mid-$p$ problems, one bimodal — and therefore different $k^*$. Aggregate benchmark numbers do not identify the distribution that determines the answer.

## 7. Current Research (as of 2026)

- **Inference-scaling-law fitting.** Stanford (Brown, Ré and collaborators), CMU, and Google DeepMind on parametric fits to coverage curves and their mechanism *(frontier — verify current results)*.
- **Verifier-limited scaling.** Princeton CITP (Stroebl, Kapoor, Narayanan) on the resampling-with-imperfect-verifiers ceiling; direct extensions into agentic SWE-bench settings.
- **Agentic budget allocation.** CodeMonkeys (Ehrlich et al., 2025) sweeps serial versus parallel test-time compute on SWE-bench Verified with test-generation-based selection — the closest existing thing to the missing sweep, but over one model.
- **Reasoning models as a confound.** Long-chain-of-thought models change $L_{\text{out}}$ by 10–100$\times$, so a "sample" is no longer a fixed unit of compute; the trade-off must be restated in tokens, not samples *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** One open model family with shared training data at four sizes — e.g. Qwen2.5-Coder 1.5B / 7B / 14B / 32B. Benchmarks: SWE-bench Verified (500 instances) plus LiveCodeBench problems released *after* every checkpoint's cutoff (contamination control). Draw $n = 1024$ samples per problem per model, temperature re-tuned per arm on a held-out split. Cost: roughly $2\times10^{21}$–$10^{22}$ FLOPs, days on a 64-GPU cluster.

**Control arm.** The largest model at $k=1$, greedy, at each budget level — i.e. "spend it all on parameters".

**Deciding number.** The fitted exponent $\beta$ in $k^*(C) \propto C^{\beta}$, reported **twice**: once under the benchmark's own tests $V$, once under a strengthened oracle $V^\star$ (EvalPlus-style test amplification plus human adjudication of a 200-instance random subsample, giving a measured $\phi$ with a confidence interval).

- If $\beta_{V^\star} > 0.5$ with 95% CI excluding 0.5, sampling deserves more than half the marginal budget and the field's parameter-first default is wrong for code.
- If $\beta_{V^\star} \le 0$ while $\beta_V > 0$, the entire repeated-sampling literature on code is a verifier artifact.

## 9. Key References

- **[Foundational]** Jared Kaplan et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Mark Chen et al. *Evaluating Large Language Models Trained on Code.* 2021. — arXiv:2107.03374
- **[SOTA]** Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, Yiming Yang. *An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models.* 2024. — arXiv:2408.00724
- **[SOTA]** Michael Hassid, Tal Remez, Jonas Gehring, Roy Schwartz, Yossi Adi. *The Larger the Better? Improved LLM Code-Generation via Budget Reallocation.* COLM, 2024. — arXiv:2404.00725
- **[SOTA]** Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan. *Inference Scaling Flaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[Method]** Yujia Li et al. *Competition-Level Code Generation with AlphaCode.* Science 378(6624), 2022.
- **[Method]** Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.* NeurIPS, 2023. — arXiv:2305.01210
- **[Theory]** Rylan Schaeffer, Joshua Kazdan, John Hughes, Jordan Juravsky, Sara Price, et al. *How Do Large Language Monkeys Get Their Power (Laws)?* 2025. — arXiv:2502.17578
- **[Survey]** Andrea Stocco et al. and others have surveyed test-time scaling; see Qiyuan Zhang et al. *What, How, Where, and How Well? A Survey on Test-Time Scaling in Large Language Models.* 2025. — arXiv:2503.24235

## 10. Worked Example

Budget: $C = 10^{16}$ FLOPs per problem. Prompt + output $\approx 2{,}000$ tokens. Cost per sample $\approx 2N \cdot 2000 = 4000N$.

| Arm | $N$ | $k = C/(4000N)$ |
|---|---|---|
| A | 32B | 78 |
| B | 7B | 357 |
| C | 1.5B | 1,667 |

Take plausible SWE-bench-Verified-like numbers: $p_{32\text{B}} = 0.30$, $p_{7\text{B}} = 0.15$, $p_{1.5\text{B}} = 0.05$ on *solvable* problems, with a solvable fraction of $0.6$, $0.5$, $0.35$ respectively (smaller models simply cannot reach some problems at any $k$).

Coverage $= f_{\text{solv}}\,(1 - (1-p)^k)$:

- A: $0.6(1 - 0.7^{78}) \approx 0.600$
- B: $0.5(1 - 0.85^{357}) \approx 0.500$
- C: $0.35(1 - 0.95^{1667}) \approx 0.350$

Under coverage, the large model wins. Now apply the selector. Suppose the verifier accepts a wrong patch with $\phi = 0.03$ per candidate — realistic for auto-generated test suites. The chance that at least one of $k$ *incorrect* candidates is falsely accepted is $1-(1-\phi)^{k}$ on the incorrect ones alone: $0.91$ for A, $0.999$ for B, $\approx 1$ for C. If the selector takes the first accepted candidate, true solve rate collapses toward the fraction of cases where a correct sample is drawn *early*, and the ranking among arms is now set almost entirely by $\phi$ and $k$, not by $N$.

That is the obstruction. Flip $\phi$ from $0.03$ to $0.003$ and arm C's false-accept probability drops from $\approx 1$ to $0.99$; drop $k$ to 78 and it is $0.21$. The published $(N,k)$ comparisons do not report $\phi$, so their orderings are not identified — the same experiment with a stronger test suite can reverse the conclusion without any change to the models.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*