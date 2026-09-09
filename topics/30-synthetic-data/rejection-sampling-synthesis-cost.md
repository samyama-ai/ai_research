---
id: 30-synthetic-data/rejection-sampling-synthesis-cost
title: "Exact Cost of Rejection Sampling in Data Synthesis Pipelines"
topic: 30-synthetic-data
status: solved-but-impractical
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Exact Cost of Rejection Sampling in Data Synthesis Pipelines

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/rejection-sampling-synthesis-cost` · **Status:** solved-but-impractical

## 1. Problem Statement

Modern synthetic-data pipelines generate candidates from a proposal model and keep the ones a filter accepts: unit tests pass, a verifier agrees with a reference answer, a reward model scores above threshold, a dedup hash is fresh. The question is how much compute one accepted example costs, and what distribution the accepted set is actually drawn from.

Three variants, with different difficulty:

- **Measurement.** Given a fixed pipeline, predict FLOPs (or wall-clock, or dollars) per accepted, retained, deduplicated example — before running it. Requires modelling per-prompt acceptance heterogeneity, truncation of rejected generations, and batched-inference effects.
- **Method.** Given a target yield $N$ and a budget $C$, choose the sampling policy (samples per prompt $k$, temperature, threshold $\tau$, early-abort rule) that maximizes downstream task gain per FLOP. Currently done by grid search on $k \in \{4,16,64\}$.
- **Theory.** Characterize the distribution the filter induces and its divergence from the intended target. Classical rejection sampling gives an *exact* answer — expected trials $=M=\sup_x p(x)/q(x)$, output exactly $p$ — which is why the status here is **solved-but-impractical**: in a synthesis pipeline $M$ is unknown, unbounded, and the filter is not a density ratio but a noisy binary verifier, so the classical guarantee does not transfer.

Solving it means: a cost model that predicts accepted-example FLOPs within, say, $\pm 20\%$ on a held-out pipeline, plus a bound on the divergence between the accepted distribution and the one the pipeline designer thinks they are sampling.

## 2. Formal Setting

Prompts $x \sim \mathcal{D}$. Proposal (generator) $q_\theta(y \mid x)$. Acceptance predicate $A(x,y) \in \{0,1\}$, possibly stochastic (an LLM judge). Per-prompt acceptance rate

$$\alpha(x) \;=\; \mathbb{E}_{y \sim q_\theta(\cdot\mid x)}\big[A(x,y)\big], \qquad \bar{\alpha} \;=\; \mathbb{E}_{x\sim\mathcal{D}}[\alpha(x)].$$

**Measured as:** draw $k$ samples for each of $m$ prompts, $\hat\alpha(x)=\frac{1}{k}\sum_i A(x,y_i)$. Note $\bar\alpha$ is an average of rates, not the rate of the average; pipelines that report "acceptance rate" usually report pooled accepts/total, which equals $\mathbb{E}[\alpha]$ only under uniform $k$.

The induced distribution over accepted outputs is the tilt

$$p_A(y\mid x) \;=\; \frac{q_\theta(y\mid x)\,\mathbb{E}[A(x,y)]}{\alpha(x)},$$

which equals a true target $p$ only if $A$ is a likelihood-ratio test with the right threshold. It is not.

**Cost.** Let $\mathrm{FLOP}(x,y) \approx 2P(|x|+|y|)$ for a dense model of $P$ non-embedding parameters (prefill plus decode, ignoring attention quadratic terms). Cost per *accepted* example under sample-until-accept:

$$C_{\text{acc}} \;=\; \mathbb{E}_x\!\left[\frac{\mathbb{E}_{y}\big[\mathrm{FLOP}(x,y)\big]}{\alpha(x)}\right] \;+\; C_{\text{verify}}\cdot\mathbb{E}_x\!\left[\frac{1}{\alpha(x)}\right].$$

The $1/\alpha(x)$ **inside** the expectation is where the practical cost lives: by Jensen, $\mathbb{E}[1/\alpha] \ge 1/\mathbb{E}[\alpha]$, and the gap is large when $\alpha(x)$ is bimodal (most prompts easy, a tail near zero). With a hard cap of $k$ attempts per prompt, expected generations per prompt is $\big(1-(1-\alpha)^k\big)/\alpha$ and *yield* is $1-(1-\alpha(x))^k$ — so capped pipelines silently drop the hard tail rather than paying for it.

**Deduplicated yield.** After near-dup filtering with rate $\rho$, useful examples per prompt $u(x,k)$ grows sublinearly in $k$; unique-solution counts empirically saturate, so the marginal cost of the $j$-th distinct example rises with $j$.

**Assumptions and where they break.**
1. *$A$ is deterministic and correct.* Violated: verifier false-accept rates on GSM8K-style final-answer matching are nonzero (correct answer, wrong derivation); LLM judges have position and length bias.
2. *$\mathrm{FLOP} \propto$ tokens.* Violated under batched serving: throughput depends on batch composition, KV-cache pressure, and speculative decoding; accepted and rejected samples share a cached prefix, so the $k$ samples for one prompt cost far less than $k$ independent generations.
3. *Samples are i.i.d. given $x$.* Violated by nucleus/top-$k$ truncation, by beam-like schedulers, and by any dedup-aware resampling.
4. *$\alpha(x)$ is stationary.* Violated in iterated self-training: each round changes $q_\theta$, so the cost model must be re-estimated per round.

## 3. State of the Art

**Theory (established).** Classical rejection sampling: expected trials $=M$, output exact (von Neumann 1951; Devroye 1986). For best-of-$n$, the KL from the base policy is bounded and, for continuous reward, exactly $\log n - \frac{n-1}{n}$ nats (Stiennon et al. 2020; Gao et al. 2023; tightened and made rigorous with finite-$n$ and tie-handling by Beirami et al. 2024). Chatterjee & Diaconis (*Annals of Applied Probability*, 2018) prove that importance-sampling-style reweighting needs sample size on the order of $e^{D_{\mathrm{KL}}(p\|q)}$ — the exponential-in-divergence wall that also governs rejection.

**Systems/empirical (established).** Rejection-sampling fine-tuning is standard: STaR (Zelikman et al., NeurIPS 2022), RFT (Yuan et al. 2023), RAFT (Dong et al., TMLR 2023), ReST (Gulcehre et al. 2023), ReST$^{EM}$ (Singh et al., TMLR 2024), and the rejection-sampling stage of Llama 2 (Touvron et al. 2023). Repeated-sampling coverage curves (Brown et al. 2024) are reproduced widely.

**Claimed but unablated.** That rejection-sampled synthetic data is compute-optimal versus RL (PPO/GRPO) at matched FLOPs. Most comparisons match *steps* or *tokens*, not generator FLOPs including rejected samples. Reported synthetic-data pipeline costs in industrial technical reports (e.g. phi-4, Abdin et al. 2024) are benchmark numbers with no accompanying generation-cost ledger — rejected-sample compute is essentially never reported.

## 4. What Is Known

- **BoN KL is exactly $\log n - \frac{n-1}{n}$ nats** for continuous reward and distinct samples (Beirami et al. 2024). At $n=16$: $2.77-0.94=1.83$ nats.
- **Reward overoptimization follows $d\,(\alpha_{\text{bon}} - \beta_{\text{bon}} d)$ with $d=\sqrt{\mathrm{KL}}$** (Gao et al., ICML 2023), fitted on proxy reward models from 3M to 3B parameters against a 3B gold RM, with $n$ up to $\sim\!6\times10^4$. Gold reward peaks and then falls — so more rejection sampling past a point buys negative value.
- **Coverage scales as an exponentiated power law in samples.** Brown et al. (2024): DeepSeek-Coder-V2-Instruct solves 15.9% of SWE-bench Lite with one sample, 56% with 250 samples — a ~3.5× gain for 250× the generator compute, and only where a verifier can pick the winner.
- **Rejection-sampled augmentation gives real but bounded gains.** Yuan et al. (2023) report LLaMA-7B GSM8K rising from 35.9% (SFT) to ~49% with rejection-sampled, deduplicated reasoning paths pooled across models at $k \approx 100$.
- **Iterated self-training saturates fast.** ReST$^{EM}$ (Singh et al., TMLR 2024) on PaLM 2 for MATH and APPS: most of the gain lands in iteration 1–2, and further iterations overfit the train set.
- **Prefix sharing is a first-order cost effect.** In speculative decoding the analogous accounting is exact: expected accepted tokens per draft block of $\gamma$ is $(1-\alpha^{\gamma+1})/(1-\alpha)$ (Leviathan et al., ICML 2023). No comparably exact accounting exists for pipeline-level rejection.

## 5. What Is Not Known

- **Theoretically open.** No characterization of $D_{\mathrm{KL}}(p_A \| p_{\text{intended}})$ when $A$ is a *noisy* verifier with asymmetric error rates $(\epsilon_{\text{FP}}, \epsilon_{\text{FN}})$ that correlate with $y$'s features (length, style, memorized solutions). The classical exactness guarantee has no known analogue here.
- **Empirically open.** The FLOP-matched comparison between rejection-sampling synthesis and on-policy RL, counting rejected generations, at $\ge 10^{22}$ training FLOPs. Runnable today on a few hundred GPU-days; not published.
- **Empirically open.** The shape of $\alpha(x)$'s distribution across realistic prompt sets, and how much of $\mathbb{E}[1/\alpha]$ the tail contributes. Cheap to measure; almost never reported.
- **Methodologically blocked.** "Cost per useful example" is not well defined without a fixed downstream measurement, because usefulness is a property of the *set* (diversity, coverage), not the example. There is no accepted set-level utility metric, so per-example cost cannot be normalized.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability between three causes of the same acceptance rate**, compounded by **confounded cost measurement**.

A pipeline with $\bar\alpha=0.3$ can arise from (i) uniform per-prompt difficulty, (ii) 30% of prompts at $\alpha\approx 1$ and 70% at $\alpha\approx 0$, or (iii) a verifier with a 30% false-accept rate on garbage. These have wildly different costs ($\mathbb{E}[1/\alpha]$ is $3.3$, unbounded, and $3.3$-but-worthless respectively) and are indistinguishable from the pooled statistic every pipeline reports. Distinguishing them needs per-prompt $k \ge 32$ *and* an independent verifier audit — the second of which requires the ground truth the pipeline exists to manufacture.

Second: FLOP accounting is not additive under batched serving. Rejected samples for a prompt share its prefill and often much of its KV cache, so a naive $k\times$ cost estimate can be 2–5× too high, while continuous batching means marginal cost depends on queue state. Reported dollar costs therefore do not transfer across serving stacks, and reported token counts omit rejects entirely.

## 7. Current Research (as of 2026)

- **Adaptive sample allocation.** Spend $k(x)$ proportional to estimated difficulty rather than uniformly; test-time-compute scaling laws (Snell et al. 2024) supply the framework. Extending them to *training-data* synthesis budgets is active *(frontier — verify)*.
- **Distilling BoN into a single forward pass.** BoNBoN (Gui, Gârbacea, Veitch, NeurIPS 2024) and Variational Best-of-$n$ (Amini, Vieira, Cotterell 2024) aim to get the BoN tilt without paying $n$ generations — turning a sampling cost into a training cost.
- **Verifier-cost-aware pipelines.** Treating verification as the dominant term when the verifier is itself a large model or a sandboxed test suite; cascade designs with cheap filters first.
- **Reward-model ensembles and thresholds** to delay overoptimization (Coste et al., ICLR 2024), which changes where the useful-$n$ ceiling sits.

## 8. Concrete Next Experiment

**Question:** does per-prompt acceptance heterogeneity, not the pooled rate, determine synthesis cost — and by how much?

- **Scale.** One 7–8B open generator (e.g. Llama-3.1-8B-Instruct). $m=5{,}000$ MATH/GSM8K-style prompts spanning difficulty levels. $k=64$ samples per prompt at $T=1.0$: $3.2\times10^5$ generations, roughly 300–500 A100-hours — one week on 4 GPUs.
- **Instrumentation.** Record $\hat\alpha(x)$ per prompt, exact prefill/decode token counts, and wall-clock under a fixed vLLM configuration with prefix caching on and, separately, off.
- **Control arm.** A homogeneous surrogate: an artificial pipeline with the *same* pooled $\bar\alpha$ but per-prompt rates drawn as $\alpha(x) \equiv \bar\alpha$ (achieved by resampling accepts across prompts). Both arms produce the same number of accepted examples; train identical 8B SFT runs on each.
- **Deciding number.** The ratio $R = \mathbb{E}_x[1/\hat\alpha(x)] \big/ (1/\bar\alpha)$ , estimated with a censored estimator for prompts with zero accepts at $k=64$ (report the censored mass explicitly). If $R < 1.5$, pooled acceptance is an adequate cost model and the problem is largely a bookkeeping exercise. If $R > 3$, every published per-example cost that uses pooled acceptance is wrong by that factor, and adaptive $k(x)$ is mandatory rather than an optimization.
- **Secondary readout.** Downstream accuracy difference between the two SFT arms at matched accepted-example count — isolating the *distributional* cost of heterogeneity from the compute cost.

## 9. Key References

- **[Foundational]** J. von Neumann. *Various techniques used in connection with random digits.* National Bureau of Standards, Applied Mathematics Series 12, 1951.
- **[Foundational]** L. Devroye. *Non-Uniform Random Variate Generation.* Springer, 1986.
- **[Foundational]** S. Chatterjee, P. Diaconis. *The sample size required in importance sampling.* Annals of Applied Probability, 2018.
- **[Foundational]** E. Zelikman, Y. Wu, J. Mu, N. D. Goodman. *STaR: Bootstrapping Reasoning With Reasoning.* NeurIPS, 2022. — arXiv:2203.14465
- **[SOTA]** L. Gao, J. Schulman, J. Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[SOTA]** A. Beirami, A. Agarwal, J. Berant, A. D'Amour, J. Eisenstein, C. Nagpal, A. T. Suresh. *Theoretical guarantees on the best-of-n alignment policy.* 2024. — arXiv:2401.01879
- **[SOTA]** B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. Ré, A. Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** A. Singh et al. *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models.* TMLR, 2024. — arXiv:2312.06585
- **[SOTA]** H. Dong, W. Xiong, D. Goyal, Y. Zhang, W. Chow, R. Pan, S. Diao, J. Zhang, K. Shum, T. Zhang. *RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment.* TMLR, 2023. — arXiv:2304.06767
- **[SOTA]** Z. Yuan, H. Yuan, C. Li, G. Dong, K. Lu, C. Tan, C. Zhou, J. Zhou. *Scaling Relationship on Learning Mathematical Reasoning with Large Language Models.* 2023. — arXiv:2308.01825
- **[SOTA]** Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- **[SOTA]** L. Gui, C. Gârbacea, V. Veitch. *BoNBoN Alignment for Large Language Models and the Sweetness of Best-of-n Sampling.* NeurIPS, 2024. — arXiv:2406.00832
- **[Survey]** H. Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models.* 2023. — arXiv:2307.09288 (rejection-sampling fine-tuning stage, §3.2)
- **[Survey]** C. Snell, J. Lee, K. Xu, A. Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314

## 10. Worked Example

Target: 100,000 accepted math solutions from an 8B generator ($P=8\times10^9$), 400 prompt tokens + 600 output tokens, so $\mathrm{FLOP}\approx 2\cdot 8\times10^9 \cdot 1000 = 1.6\times10^{13}$ per generation.

**Naive ledger.** Pooled acceptance $\bar\alpha=0.25$. Expected generations $=100{,}000/0.25=4\times10^5$. Total $=6.4\times10^{18}$ FLOPs $\approx$ 20 A100-days at 40% MFU.

**Actual ledger.** Suppose the prompt set is bimodal: 25% of prompts have $\alpha=0.95$, 75% have $\alpha=0.02$. Pooled rate is $0.25\cdot0.95+0.75\cdot0.02 = 0.253$ — identical to above. But

$$\mathbb{E}_x[1/\alpha] = 0.25\cdot\tfrac{1}{0.95} + 0.75\cdot\tfrac{1}{0.02} = 0.26 + 37.5 = 37.8,$$

against $1/\bar\alpha = 3.95$. Ratio $R = 9.6$. Covering the prompt set uniformly costs almost **10× the naive estimate**, or 190 A100-days.

**What the pipeline actually does.** It caps at $k=16$. Yield on hard prompts is $1-0.98^{16}=27.6\%$; on easy prompts, ~100%. Cost stays near the naive figure, but 72% of hard prompts contribute nothing, and the accepted set is ~55% easy-prompt solutions by mass instead of the intended 25%. The dataset is skewed toward problems the model already solves — exactly the examples with the least training signal.

**The obstruction made visible.** Both the uniform and bimodal worlds report `acceptance rate: 25.3%`. From that logged number alone, the 20-day estimate and the 190-day estimate are indistinguishable, and so are a well-covered dataset and one that quietly dropped three quarters of its hard tail. Recovering the difference requires per-prompt $\hat\alpha$ at $k\ge32$ — which almost no published pipeline reports, and which is the cheapest missing measurement in the whole area.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*