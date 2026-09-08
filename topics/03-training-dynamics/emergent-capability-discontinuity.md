---
id: 03-training-dynamics/emergent-capability-discontinuity
title: "Emergent Capability Discontinuity"
topic: 03-training-dynamics
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Emergent Capability Discontinuity

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/emergent-capability-discontinuity` · **Status:** methodologically-blocked

## 1. Problem Statement

Some capabilities of large models appear absent at small scale and present at large scale, with a transition sharper than the smooth power-law decay of pretraining loss. The question is whether that sharpness is a property of the model, of the training dynamics, or only of the metric used to look.

Three variants, with different difficulty:

- **Measurement.** Given a task $T$ and a model family indexed by compute $C$, decide whether the capability curve $A_T(C)$ has a genuine discontinuity or high-order phase transition, under a metric choice that is not itself responsible for the shape. Solving it means: a metric-invariant test that returns the same verdict across all reasonable scoring rules for $T$.
- **Method.** Predict, from runs at $C \le C_0$, the value $A_T(10 C_0)$ to within a stated error bar, before the large run. Solving it means calibrated forecasts on held-out tasks and held-out scales.
- **Theory.** Prove for a concrete learning problem that SGD on a transformer exhibits a loss curve smooth in $C$ while a natural task metric is non-analytic in $C$ at some $C^*$ — or prove no such $C^*$ can exist for a given metric class.

Status is **methodologically blocked**: the measurement variant is not yet well posed, so the method and theory variants inherit an ambiguous target.

## 2. Formal Setting

Let $\theta(C) \in \mathbb{R}^d$ be parameters produced by a fixed training recipe at compute $C$ (FLOPs), $N$ parameters, $D$ tokens. Pretraining loss is measured as held-out mean per-token cross-entropy on a fixed corpus:

$$L(C) = -\frac{1}{|\mathcal{V}|}\sum_{(x_{<t},x_t)\in\mathcal{V}} \log p_{\theta(C)}(x_t \mid x_{<t}).$$

Task performance for task $T$ with examples $(q_i, y_i)_{i=1}^n$ and scoring rule $s$:

$$A_T^{s}(C) = \frac{1}{n}\sum_{i=1}^n s\big(\hat{y}_i(\theta(C)), y_i\big),$$

where $\hat y_i$ is the decoded output under a fixed prompt template, shot count, and decoding rule. Two scoring rules matter:

- **Discrete:** $s_{\mathrm{em}} = \mathbb{1}[\hat y = y]$ (exact match), or multiple-choice argmax.
- **Continuous:** $s_{\mathrm{cont}} \in \{$token edit distance, Brier score, $\log p_\theta(y\mid q)\}$.

A useful bridge: if the answer is $k$ tokens and per-token correctness is roughly independent with probability $p(C)$, then $\mathbb{E}[s_{\mathrm{em}}] \approx p(C)^k$. Smooth $p$ gives sharp exact-match onset for $k \gtrsim 5$ (Schaeffer et al., 2023).

**Discontinuity predicate.** Fit a family $\mathcal{F}$ of smooth laws (power law, sigmoid in $\log C$, broken power law). Declare emergence at level $\varepsilon$ if

$$\min_{f \in \mathcal{F}} \max_{C \in [C_{\min}, C_{\max}]} \big| A_T^{s}(C) - f(C) \big| > \varepsilon,$$

with $\varepsilon$ set by seed-to-seed variance. Equivalently, define a *predictability gap*: the extrapolation error of $\mathcal{F}$ fit on $C \le C_0$, evaluated at $10C_0$.

**Assumptions, and which are violated.**
1. *Single-parameter family.* Assumed: models differ only in $C$. Violated — public model families change data mixture, tokenizer, and hyperparameters across sizes.
2. *Task independent of training data.* Violated — benchmark contamination is measurable and uncontrolled at frontier scale.
3. *Metric measures the capability it names.* Violated by construction for exact match on multi-token answers.
4. *Resolution.* $A_T^s$ estimated on $n \sim 100$–$1000$ items has standard error $\ge 1.5$ points; below-chance regions are censored, not measured.
5. *Deterministic recipe.* Violated — seed variance at small scale can exceed the effect (Sellam et al., MultiBERTs, ICLR 2022).

## 3. State of the Art

**Empirical SOTA — established.**
- Aggregate pretraining loss is predictable across $\sim$4 orders of magnitude of compute (Kaplan et al., 2020; Hoffmann et al., 2022). GPT-4's loss was forecast from runs using $\le 10^{-4}$ of final compute (OpenAI, 2023) — reported, not independently reproducible.
- Metric choice changes the verdict. Schaeffer, Miranda & Koyejo (NeurIPS 2023, Best Paper) show for GPT-3-family arithmetic and 92 BIG-Bench tasks that discrete metrics produce sharp curves while continuous surrogates on the same outputs are smooth, and that emergence can be induced or removed by choosing the metric. Established for the tasks tested.
- Downstream accuracy is better predicted from *loss* than from compute. Du et al. (2024) show task accuracy is a near-deterministic function of pretraining loss across model sizes, with sharp behavior appearing only below a loss threshold.
- Observational scaling laws (Ruan, Maddison & Hashimoto, NeurIPS 2024) fit a low-dimensional capability space over ~80 public models and predict "emergent" task curves as sigmoids in that latent space, with held-out predictions on GPT-4-class models.

**Claimed but unablated.**
- That emergence reflects a discrete internal circuit forming. Mechanistic cases exist (induction heads, Olsson et al. 2022; modular addition, Nanda et al. 2023) but have not been shown to cause any BIG-Bench emergence curve.
- Broken Neural Scaling Laws (Caballero et al., ICLR 2023) fit sharp transitions post hoc; the ablation showing they *forecast* unseen breaks is not there.
- Lu et al. (ACL 2024) argue emergent abilities reduce to in-context learning plus instruction tuning — an argument over ~20 tasks, not a general result.

**Theory SOTA.** Barak et al. (NeurIPS 2022) prove SGD learns $k$-sparse parities with a long plateau then abrupt drop, giving a rigorous mechanism for hidden progress under a flat metric. Michaud et al. (NeurIPS 2023) give the quantization model: discrete "quanta" with power-law frequency yield smooth aggregate loss and step-like per-quantum performance. Neither is proved for transformers on natural language.

## 4. What Is Known

- Wei et al. (TMLR 2022) catalog 137 BIG-Bench tasks with sharp onset; on 3-digit addition, GPT-3 exact-match goes from $<1\%$ at 6.7B to $\sim8\%$ at 13B to $\sim25\%$ at 175B.
- Schaeffer et al. (2023): substituting token edit distance for exact match on the same GPT-3 outputs converts that curve into a smooth monotone trend; on 5-digit multiplication, per-token accuracy rises smoothly while $p^k$ stays near zero until $p \approx 0.8$.
- BIG-Bench (Srivastava et al., TMLR 2023): across 204 tasks, most curves are smooth; sharp onsets are a minority and concentrate in multi-step, exact-match tasks.
- Induction heads form in a narrow window early in training (roughly 2.5–5B tokens for small models), coinciding with a visible bump in the loss curve — the clearest case of a real discontinuity in training dynamics (Olsson et al., 2022).
- Grokking: modular addition at 113 modulus, ~40% train fraction, test accuracy jumps from chance to $>99\%$ thousands of steps after train accuracy saturates (Power et al., 2022); Nanda et al. (2023) construct progress measures that rise smoothly through that jump.
- Hu et al. (ICLR 2024, PassUntil) show that with $\sim 10^5$ samples per item, task performance on code and math is measurable at $10^{-5}$ resolution and follows a power law two orders of magnitude below the apparent emergence point.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No accepted criterion for a "non-metric-induced" discontinuity. Every candidate metric-invariance test either restricts to continuous scoring rules — which by construction cannot show a jump for smooth $p$ — or admits any user-chosen thresholding. The predicate in §2 is not identified without fixing $\mathcal{F}$, $s$, and $\varepsilon$, and there is no principled way to fix them.
- **Theoretically open.** Whether any transformer trained by SGD on a natural-language distribution has a genuinely non-analytic capability curve in $C$, as opposed to a steep analytic one. No proof either way. Also open: whether the quantization model's discrete quanta exist as identifiable objects in real models.
- **Empirically open.** Whether a *single* family trained under one fixed recipe, one data mixture, decontaminated, at $\ge 6$ scales spanning $10^{18}$–$10^{23}$ FLOPs, with $\ge 3$ seeds each, shows any task whose curve exceeds seed noise against the best smooth fit under *all* continuous metrics. Runnable; not run.
- **Empirically open.** Whether emergence points are predictable in advance rather than fitted after. Snell et al. (2024) forecast via small-scale finetuning; not validated at frontier scale.

## 6. Why It Is Hard

Four named obstructions, in order of bite.

1. **Non-identifiability of metric versus model.** $A_T^s(C) = s \circ g(\theta(C))$. Sharpness can enter through $s$ or through $g$, and observations of $A$ alone do not separate them. Choosing a continuous $s$ to "remove" the artifact begs the question: continuous surrogates cannot exhibit jumps, so the test is rigged either way.
2. **Confounded measurement across families.** Public model series vary data, tokenizer and hyperparameters with size; the "scale axis" is not one axis. The clean experiment requires training the family yourself.
3. **Resolution floor.** With $n=500$ items and binary scoring, anything below $\sim0.2\%$ is indistinguishable from zero. PassUntil shows the signal is there but needs $\sim10^5$ samples per item — a $200\times$ inference cost.
4. **Compute cost of the decisive arm.** Six scales $\times$ 3 seeds up to $10^{23}$ FLOPs is $\sim10^{24}$ FLOPs total, a frontier-lab budget, and the question is not commercially urgent.

Absent ground truth compounds all four: there is no independent oracle saying whether a model "has" a capability at a given scale.

## 7. Current Research (as of 2026)

- **Loss-as-x-axis.** Replacing compute with pretraining loss (Du et al., 2024, Tsinghua/Zhipu) as the predictor. Extending this to per-domain loss on held-out task-adjacent corpora is active *(frontier — verify)*.
- **Infinite-resolution evaluation.** PassUntil-style estimators (Hu et al., ICLR 2024) applied to agentic and tool-use tasks *(frontier — verify)*.
- **Observational scaling.** Ruan/Maddison/Hashimoto (Stanford/Toronto) latent-capability fits across public models; being extended to post-trained and reasoning models *(frontier — verify)*.
- **Mechanistic phase transitions.** Anthropic and independent groups tracking circuit formation against loss-curve features; developmental-interpretability work using the local learning coefficient (Lau, Murfet et al.) to detect degeneracy changes during training *(frontier — verify)*.
- **Predicting capability from small-scale finetuning** (Snell, Levine et al., Berkeley) — "emergence laws" fit to finetuned small models.
- **Evaluation-science pushback.** Growing insistence on error bars and item-level scoring in benchmark reporting (Miller, 2024, "Adding Error Bars to Evals").

## 8. Concrete Next Experiment

**Question.** Under a fixed recipe, does any task curve depart from the best smooth fit by more than seed noise, under *every* continuous scoring rule?

**Scale.** Train one Chinchilla-optimal family at 6 sizes — 70M, 160M, 410M, 1B, 2.8B, 6.9B parameters ($\approx 10^{18}$ to $\approx 10^{23}$ FLOPs), 3 seeds each, 18 runs. Identical data mixture, tokenizer, and hyperparameter schedule; decontaminate against the eval suite by 13-gram overlap. Cost: roughly $10^{23}$–$10^{24}$ FLOPs, order 200k–500k H100-hours.

**Tasks.** The 20 BIG-Bench / arithmetic / MMLU-subset tasks with the sharpest reported onsets. Evaluate each with 4 scoring rules: exact match, token edit distance, per-token accuracy, and answer log-likelihood. Use PassUntil sampling ($10^4$ samples/item) so accuracies down to $10^{-4}$ are measured, not censored.

**Control arm.** Synthetic tasks with a *known* answer: (a) $k$-token copy, where $s_{\mathrm{em}} = p^k$ exactly and the discontinuity is provably metric-induced; (b) $k$-sparse parity, where Barak et al. guarantee a real plateau-then-jump in the underlying computation. These calibrate the detector's false-positive and false-negative rates.

**Deciding number.** For each task and each continuous metric, the **maximum residual of the best smooth fit, in units of cross-seed standard deviation**:

$$Z_T = \min_{f\in\mathcal{F}} \max_C \frac{|A_T^{s}(C) - f(C)|}{\hat\sigma_{\text{seed}}(C)}.$$

If $\max_s Z_T < 3$ for all 20 natural tasks while the parity control gives $Z > 10$, the discontinuity is metric-induced and the measurement variant closes. If any natural task holds $Z_T > 3$ under all four metrics, a genuine transition exists and the theory variant becomes the live problem.

## 9. Key References

- **[Foundational]** J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, et al. *Emergent Abilities of Large Language Models.* TMLR, 2022. — arXiv:2206.07682
- **[SOTA]** R. Schaeffer, B. Miranda, S. Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004
- **[Theory]** B. Barak, B. L. Edelman, S. Goel, S. Kakade, E. Malach, C. Zhang. *Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit.* NeurIPS, 2022. — arXiv:2207.08799
- **[Theory]** E. J. Michaud, Z. Liu, U. Vaintrob, M. Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS, 2023. — arXiv:2303.13506
- **[SOTA]** Y. Ruan, C. J. Maddison, T. Hashimoto. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024. — arXiv:2405.10938
- **[SOTA]** S. Hu, X. Liu, X. Han, X. Zhang, C. He, W. Zhao, et al. *Predicting Emergent Abilities with Infinite Resolution Evaluation.* ICLR, 2024. — arXiv:2310.03262
- **[Method]** Z. Du, A. Zeng, Y. Dong, J. Tang. *Understanding Emergent Abilities of Language Models from the Loss Perspective.* NeurIPS, 2024. — arXiv:2403.15796
- **[Mechanism]** C. Olsson, N. Nanda, N. Elhage, et al. *In-context Learning and Induction Heads.* Transformer Circuits Thread, Anthropic, 2022.
- **[Mechanism]** N. Nanda, L. Chan, T. Lieberum, J. Smith, J. Steinhardt. *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR, 2023. — arXiv:2301.05217
- **[Mechanism]** A. Power, Y. Burda, H. Edwards, I. Babuschkin, V. Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* 2022. — arXiv:2201.02177
- **[Survey]** A. Srivastava et al. *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models.* TMLR, 2023. — arXiv:2206.04615
- **[Foundational]** D. Ganguli, D. Hernandez, L. Lovitt, et al. *Predictability and Surprise in Large Generative Models.* ACM FAccT, 2022. — arXiv:2202.07785
- **[Scaling]** J. Hoffmann, S. Borgeaud, A. Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Fitting]** E. Caballero, K. Gupta, I. Rish, D. Krueger. *Broken Neural Scaling Laws.* ICLR, 2023. — arXiv:2210.14891
- **[Method]** C. Snell, E. Wallace, D. Klein, S. Levine. *Predicting Emergent Capabilities by Finetuning.* 2024. — arXiv:2411.16035

## 10. Worked Example

Take 5-digit integer addition, scored by exact match on the 6-token answer.

Suppose per-token accuracy is smooth and roughly linear in $\log_{10} C$ over the observed range:

| Compute (FLOPs) | per-token $p$ | $p^6$ (exact match) |
|---|---|---|
| $10^{20}$ | 0.50 | 1.6% |
| $10^{21}$ | 0.65 | 7.5% |
| $10^{22}$ | 0.80 | 26% |
| $10^{23}$ | 0.92 | 61% |

$p$ gains 0.14–0.15 per decade — perfectly smooth, no break. Exact match gains 5.9 points, then 19, then 35: an accelerating curve that any eye reads as onset near $10^{22}$. Fit a power law to the first two exact-match points and extrapolate to $10^{23}$: $1.6\% \to 7.5\%$ is a factor 4.7 per decade, predicting $\approx 35\%$ at $10^{23}$ against an actual 61%. A 26-point miss, produced entirely by the exponent $k=6$, with no change in the model's underlying trend.

Now the obstruction. Rescore with token edit distance. The curve is smooth — but that was guaranteed: any metric averaging over tokens turns $p^6$ back into $p$. The continuous metric cannot report a jump even if one exists, and the discrete metric reports a jump even when none does. Neither observation identifies $g(\theta(C))$.

The only escape is a case where $k$ is known and the metric's exponent can be divided out — which requires knowing the task's compositional structure in advance. For 5-digit addition, $k=6$ is known. For "causal judgment" or "multi-step reasoning", it is not, and no procedure currently estimates it from data. That is what makes the problem methodologically blocked rather than merely unrun.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*