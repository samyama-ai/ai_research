---
id: 08-loss-and-heads/cross-entropy-versus-downstream-utility
title: "Cross-Entropy Versus Downstream Task Utility"
topic: 08-loss-and-heads
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Entropy Versus Downstream Task Utility

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/cross-entropy-versus-downstream-utility` · **Status:** methodologically-blocked

## 1. Problem Statement

Language models are trained to minimise token-level cross-entropy (CE) and judged by downstream task performance. The two are not the same objective, and the map between them is not known.

Three variants, of very different difficulty:

- **Measurement.** Given two checkpoints with CE gap $\Delta$ on a held-out corpus, what interval on downstream utility $U$ does $\Delta$ imply? Currently there is no non-vacuous answer, and — the core claim of this page — the question is not yet well posed, because $U$ is defined by scoring rules (prompt format, answer extraction, normalisation) that are not part of the CE measurement at all.
- **Method.** Can a training objective be constructed whose value is a *monotone, low-variance* predictor of $U$ for a target task family, at a cost comparable to CE?
- **Theory.** Under what conditions on the data distribution and the task does $\varepsilon$-optimality in CE imply $\delta(\varepsilon)$-optimality in $U$, with $\delta \to 0$ as $\varepsilon \to 0$ and a rate that is not exponential in sequence length?

Solving the measurement variant means: a stated procedure that, given only pretraining loss and a task specification, returns a calibrated prediction interval for accuracy that holds out-of-sample across model families.

## 2. Formal Setting

Let $\mathcal{V}$ be the vocabulary, $p^\star$ the data distribution over sequences $x_{1:T}\in\mathcal{V}^T$, and $p_\theta$ the model. Measured cross-entropy is the empirical average over a held-out set $D$ of $N$ tokens:

$$\hat{L}(\theta) = -\frac{1}{N}\sum_{x \in D}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}),$$

reported in nats/token. **As measured**, $\hat L$ depends on the tokenizer (so cross-tokenizer comparison requires nats/byte or nats/character), on $D$'s domain mix, and on context length; none of these is fixed across published numbers.

A task is a distribution $q$ over $(c, y)$ with prompt $c$ and answer $y$, plus a **scoring map** $S$ turning $p_\theta(\cdot\mid c)$ into a prediction. Utility is

$$U(\theta) = \mathbb{E}_{(c,y)\sim q}\big[\mathbb{1}\{S(p_\theta(\cdot\mid c)) = y\}\big].$$

$S$ is where the trouble lives. Common choices — argmax over answer-letter logits; length-normalised sequence log-likelihood $\frac{1}{|y|}\log p_\theta(y\mid c)$; unnormalised log-likelihood; PMI-normalised $\log \frac{p_\theta(y\mid c)}{p_\theta(y)}$; free-form generation with regex extraction — give different $U$ for the *same* $p_\theta$.

The natural bridge is the KL decomposition $\hat L(\theta) = H(p^\star) + \mathrm{KL}(p^\star \| p_\theta) + O(N^{-1/2})$, so CE gaps between models equal KL gaps. Pinsker gives $\mathrm{TV}(p^\star, p_\theta) \le \sqrt{\tfrac12 \mathrm{KL}}$ at the *sequence* level, and $|U(\theta) - U^\star| \le \mathrm{TV}$.

Assumptions this framing rests on, and their status:

1. **$D \sim p^\star$ and $q \ll p^\star$** (task distribution is covered by pretraining). *Violated*: benchmark prompts are template-formatted text near-absent from web corpora.
2. **KL is dominated by task-relevant tokens.** *Violated by orders of magnitude*: benchmark-relevant token mass is $\sim 10^{-5}$ of a pretraining corpus (§10).
3. **$S$ is fixed across compared models.** *Violated in practice*: harnesses differ, and format sensitivity moves MMLU accuracy by several points.
4. **Test data is not in training data.** *Violated at unknown rate* for web-scale corpora.

## 3. State of the Art

**Established (reproduced, ablated).**
- CE itself is predictable from compute: Kaplan et al. (2020) and Hoffmann et al. (2022) fit power laws with residuals of a few percent across $\sim$3 orders of magnitude of compute.
- *Aggregate* downstream error is predictable from CE. Gadre et al. (2024) fit average top-1 error over 17 tasks as an exponential function of perplexity and predict it from models trained with $\sim 20\times$ less compute, with relative error around a few percent on the average. Per-task predictions are markedly worse.
- Equal CE does not imply equal downstream. Liu, Xie, Li & Ma (ICML 2023) show checkpoints matched on pretraining loss differ measurably on downstream fine-tuning, attributing the gap to implicit bias — the sharpest single refutation of loss-as-sufficient-statistic.

**Claimed but unablated / benchmark-only.**
- "Downstream ability is a function of pretraining loss alone" (Du et al., 2024): the loss-threshold curves for MMLU-style emergence are reported for one model family and one tokenizer; the claim is not tested across tokenizers or data mixes.
- Observational scaling laws (Ruan, Maddison & Hashimoto, NeurIPS 2024) fit a low-dimensional capability space over $\sim$100 public models and predict held-out benchmark scores well. This is a *benchmark-number* result: the latent factors are not identified with any measurable property of $p_\theta$, and the fit reuses models whose benchmark scores were part of their own selection pressure.
- Schaeffer et al. (2024) argue predicting individual-task accuracy remains elusive because the scoring map $S$ destroys the smooth per-token signal; they show the argument on multiple-choice tasks but do not extend it to generative tasks.

**Theory SOTA** is weak: Saunshi et al. (ICLR 2021) prove that low pretraining loss implies good linear-probe performance on tasks representable as natural-language classification, under assumptions (task expressible as a sentence-completion with bounded quantifier) that no benchmark satisfies exactly.

## 4. What Is Known

- Pinsker bounds are vacuous at realistic scale. A per-token CE gap of $0.01$ nats over a 500-token sequence gives sequence KL $\approx 5$ nats, so $\mathrm{TV}\le\sqrt{2.5} > 1$.
- Emergence is partly an artefact of $S$. Schaeffer, Miranda & Koyejo (NeurIPS 2023) show that replacing exact-match with continuous, per-token metrics turns discontinuous accuracy curves into smooth ones on arithmetic tasks in the GPT-3 and LaMDA families; 92% of the "emergent" claims in BIG-bench they audit sit on discontinuous metrics.
- Scaling is not always monotone in utility: McKenzie et al. (TMLR 2023) document inverse-scaling tasks where larger, lower-CE models score *worse*.
- Architecture changes decouple the two: Tay et al. (ICLR 2022; and *Scaling Laws vs Model Architectures*, 2022) show model-shape variants matched on upstream perplexity differ by several points on SuperGLUE, with rank order not preserved.
- RLHF moves utility while raising CE. Stiennon et al. (NeurIPS 2020) report summarisation policies preferred over reference summaries by human raters while their likelihood under the pretrained model falls.
- CE is a strictly proper scoring rule (Gneiting & Raftery, 2007) — it is uniquely minimised at $p^\star$ — but propriety is a statement about the *limit*, not about ordering imperfect models by any downstream functional.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no agreed definition of $U$ independent of the scoring map $S$. Until "downstream utility" names a quantity invariant to prompt format and answer extraction, the CE→$U$ map has a free variable on the right-hand side, and disagreements between papers are unresolvable.
- **Theoretically open.** Whether any non-vacuous bound $|U(\theta)-U(\theta')| \le f(\Delta)$ exists under assumptions that real corpora satisfy. No proof either way; the only known lower-bound construction (adversarially reallocating KL onto a rare subpopulation) shows the *unconstrained* bound is vacuous, not that a constrained one fails.
- **Empirically open.** Nobody has trained a matched grid of models — identical loss, deliberately varied data mix / tokenizer / optimiser — and measured the resulting spread in $U$ at fixed CE at $\geq$7B scale. Liu et al. (2023) did this at small scale only.

## 6. Why It Is Hard

The specific obstruction is **measurement confounding by subpopulation mass**. CE aggregates over a token distribution in which task-relevant tokens are a $\sim 10^{-5}$ fraction. A CE gap therefore places a constraint on the task-conditional distribution weaker than the gap by five orders of magnitude (§10), so a fixed $\Delta$ is consistent with essentially any downstream ordering. This is not a compute problem; buying more tokens does not shrink the ratio.

Second obstruction: **the evaluation does not measure what it names**. Accuracy under $S$ is a thresholded functional of $p_\theta$; two models with identical rank orderings over answers but different margins score identically, and two with tiny margin differences can score far apart. Contamination adds an unmeasurable bias term of unknown sign.

## 7. Current Research (as of 2026)

- **Loss-conditioned capability laws.** Following Du et al. (2024), groups at Tsinghua/Zhipu and in the open-weights community are re-fitting benchmark accuracy against pretraining loss rather than compute. *(frontier — verify whether these hold across tokenizers.)*
- **Observational / latent-capability scaling.** Ruan, Maddison & Hashimoto (Stanford/Toronto) and follow-ups extend the PCA-over-public-models approach to agentic benchmarks. *(frontier — verify.)*
- **Metric redesign.** Continuous, margin-based reformulations of multiple-choice scoring (Brier score over answer options, per-token log-likelihood margins) as replacements for exact-match — pushed by the emergence-mirage line.
- **Utility-aligned objectives.** Reward-model-weighted and task-weighted CE variants; data-mixture optimisation targeting downstream loss directly. *(frontier — verify; most reported gains are single-seed.)*

## 8. Concrete Next Experiment

**Question:** at fixed pretraining CE, how much downstream utility spread do training-recipe choices produce?

- **Scale:** 8 models at 1.4B parameters, 30B tokens each (~$2.5\times10^{20}$ FLOPs, roughly 400 A100-days total). Vary two factors orthogonally: data mix (4 levels, web-heavy → code/math-heavy) and optimiser/schedule (2 levels, AdamW cosine vs. muP + WSD). Train each to a **loss target**, not a step count: stop when held-out CE on a *fixed, neutral* corpus (measured in nats/byte, shared tokenizer across all arms) reaches $2.40 \pm 0.005$ nats/byte.
- **Control arm:** 3 seeds of the single web-heavy/AdamW recipe trained to the same loss target. This gives the seed-noise floor for $U$ — the spread that is not attributable to recipe.
- **Evaluation:** 6 tasks (MMLU, ARC-C, GSM8K, HumanEval, HellaSwag, TriviaQA), each scored under 3 scoring maps $S$ (letter-argmax, length-normalised likelihood, PMI-normalised) and 5 prompt templates, giving 15 measurements per task per model.
- **Deciding number:** $R = \dfrac{\text{spread in mean } U \text{ across the 8 recipes at fixed CE}}{\text{spread across the 3 control seeds}}$, in accuracy points, computed per task.
  - $R < 2$ on all six tasks ⟹ CE is close to a sufficient statistic at this scale; the problem is largely one of noise, not decoupling.
  - $R > 5$ on any task ⟹ CE at 0.005 nats/byte resolution does not determine utility to within 2× seed noise, and loss-matched comparison is unsound as currently practised.
  - Secondary number: variance of $U$ attributable to $S$ and template, as a fraction of total. If that fraction exceeds the recipe fraction, the block is in the metric, not the objective — which is the outcome this page predicts.

## 9. Key References

- **[Foundational]** Kaplan, J., McCandlish, S., Henighan, T., et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, J., Borgeaud, S., Mensch, A., et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Gneiting, T. & Raftery, A. E. *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102(477), 2007.
- **[SOTA]** Gadre, S. Y., Smyrnis, G., Shankar, V., et al. *Language models scale reliably with over-training and on downstream tasks.* 2024. — arXiv:2403.08540
- **[SOTA]** Ruan, Y., Maddison, C. J. & Hashimoto, T. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024. — arXiv:2405.10938
- **[SOTA]** Du, Z., Zeng, A., Dong, Y. & Tang, J. *Understanding Emergent Abilities of Language Models from the Loss Perspective.* NeurIPS, 2024. — arXiv:2403.15796
- **[Key result]** Liu, H., Xie, S. M., Li, Z. & Ma, T. *Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models.* ICML, 2023. — arXiv:2210.14199
- **[Key result]** Schaeffer, R., Miranda, B. & Koyejo, S. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004
- **[Key result]** Schaeffer, R., Schoelkopf, H., Miranda, B., et al. *Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?* 2024. — arXiv:2406.04391
- **[Theory]** Saunshi, N., Malladi, S. & Arora, S. *A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks.* ICLR, 2021. — arXiv:2010.03648
- **[Counterexample]** McKenzie, I. R., Lyzhov, A., Pieler, M., et al. *Inverse Scaling: When Bigger Isn't Better.* TMLR, 2023. — arXiv:2306.09479
- **[Counterexample]** Tay, Y., Dehghani, M., Rao, J., et al. *Scale Efficiently: Insights from Pretraining and Finetuning Transformers.* ICLR, 2022. — arXiv:2109.10686
- **[Empirical]** Stiennon, N., Ouyang, L., Wu, J., et al. *Learning to Summarize from Human Feedback.* NeurIPS, 2020. — arXiv:2009.01325
- **[Survey]** Liang, P., Bommasani, R., Lee, T., et al. *Holistic Evaluation of Language Models.* TMLR, 2023. — arXiv:2211.09110

## 10. Worked Example

Two 7B checkpoints, A and B, evaluated on the same 100M-token held-out corpus with the same tokenizer.

- $\hat L_A = 2.100$ nats/token, $\hat L_B = 2.110$ nats/token. Gap $\Delta = 0.010$ nats/token — about 0.5% relative, a gap that in practice separates a good recipe from a mediocre one.
- Standard error: with $N = 10^8$ tokens and per-token log-loss standard deviation $\approx 2.5$ nats, $\mathrm{SE} \approx 2.5/\sqrt{10^8} = 2.5\times10^{-4}$ nats. So $\Delta$ is $\sim$40 SE — statistically unambiguous. (Correlation within documents inflates this by roughly $\sqrt{\text{docs}}$ effects; even so $\Delta$ is significant.)

Now ask what $\Delta$ constrains on MMLU. MMLU has 14,042 test items; scoring reads exactly one answer token per item, so the scored subpopulation is $\sim 1.4\times10^4$ tokens. Against a 100M-token corpus its mass is $\alpha \approx 1.4\times10^{-4}$; against a 10T-token pretraining corpus, $\alpha \approx 1.4\times10^{-9}$.

Decompose the CE gap by subpopulation: $\Delta = \alpha\,\Delta_{\text{task}} + (1-\alpha)\,\Delta_{\text{rest}}$. Even forcing $\Delta_{\text{rest}} = 0$, the bound on the task subpopulation is

$$\Delta_{\text{task}} \le \Delta/\alpha = 0.010 / 1.4\times10^{-4} \approx 71 \ \text{nats/token}.$$

Seventy-one nats is more than $\log|\mathcal{V}| \approx \log(128{,}000) = 11.8$ nats — the bound exceeds the maximum possible cross-entropy of a uniform predictor by 6×. It is *fully vacuous*: A and B could differ from "identical on MMLU" to "one at chance and the other perfect", and the observed $\Delta$ is consistent with either.

The reverse direction is equally loose. Suppose B is *better* on MMLU, with a mean answer-token log-prob advantage of 0.35 nats. Its contribution to corpus CE is $1.4\times10^{-4}\times 0.35 = 4.9\times10^{-5}$ nats/token — 200× below the observed $\Delta$, and buried a factor of 5 under the measurement's own standard error at $N=10^8$.

**What this makes visible.** The obstruction is not sample size, model scale, or benchmark quality. It is that CE is an average over a population in which the decision-relevant tokens carry $10^{-4}$–$10^{-9}$ of the mass. Improving the CE estimate improves the estimate of the *average*, not of the conditional the task depends on. A CE-based prediction of downstream utility is therefore an extrapolation across a five-to-nine-order-of-magnitude mass gap, licensed only by an unstated and untested smoothness assumption — that models which are better on average are better on the rare subpopulation too. Liu et al. (2023) and the inverse-scaling tasks are exactly the cases where that assumption fails. Until the utility metric is defined on the subpopulation directly and independently of the scoring map, there is no well-posed quantity to bound.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*