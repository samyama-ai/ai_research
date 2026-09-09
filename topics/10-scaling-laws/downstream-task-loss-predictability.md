---
id: 10-scaling-laws/downstream-task-loss-predictability
title: "Downstream Task Loss Predictability"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Downstream Task Loss Predictability

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/downstream-task-loss-predictability` · **Status:** open

## 1. Problem Statement

Pretraining loss is predictable from compute to within a few percent. Downstream task performance is not, reliably. The problem: given a family of models trained under a fixed recipe at compute budgets $C_1 < \dots < C_k$, predict the score of a model at target budget $C^\star \gg C_k$ on a named benchmark, before training it.

Three variants, routinely conflated:

- **Measurement.** Define a task metric whose scaling behaviour is estimable at all. Accuracy on a 4-way multiple-choice benchmark is pinned at chance (0.25) across the whole small-model regime, so it carries no signal to fit.
- **Method.** Given a well-behaved metric, fit an extrapolator with calibrated error bars. Decision predicate: is $|\hat{M}(C^\star) - M(C^\star)| \le \epsilon$ with stated coverage, at extrapolation ratio $C^\star/C_k \ge 100$?
- **Theory.** Explain *why* a link between next-token loss and task success exists, and predict which tasks have a sharp threshold in loss and which do not.

Solving it means: for an unseen benchmark and an unseen model family, produce a prediction interval at $100\times$ extrapolation that contains the truth at its nominal rate. Nobody has done this. The measurement variant is where the failure begins.

## 2. Formal Setting

A model $\theta(N, D)$ has $N$ non-embedding parameters trained on $D$ tokens, $C \approx 6ND$ FLOPs. Pretraining loss, measured as mean token-level cross-entropy in nats on a held-out shard of the *training* distribution:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

A benchmark is $T = \{(x_i, y_i, \mathcal{Y}_i)\}_{i=1}^n$: prompt, correct answer, candidate set. Two measured quantities per example.

**Task negative log-likelihood** — the continuous surrogate, measured by teacher-forcing the gold continuation:

$$\ell_i(\theta) = -\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log p_\theta(y_{i,t} \mid x_i, y_{i,<t}), \qquad L_T(\theta) = \frac{1}{n}\sum_i \ell_i(\theta)$$

**Task metric** — what is actually reported:

$$M(\theta) = \frac{1}{n}\sum_i \mathbb{1}\Big[\arg\max_{y \in \mathcal{Y}_i} s_\theta(y \mid x_i) = y_i\Big], \quad s_\theta \in \{\text{sum-NLL}, \text{length-normalized}, \text{PMI-normalized}\}$$

The standard pipeline is two-step: $C \mapsto L_T$ by a power law, then $L_T \mapsto M$ by a monotone link, typically

$$M = M_{\text{chance}} + \frac{1 - M_{\text{chance}}}{1 + \exp\!\big(k\,(L_T - \ell_0)\big)}$$

Prediction error is $|\hat M(C^\star) - M(C^\star)|$ in accuracy points at extrapolation ratio $\rho = C^\star / C_k$.

**Assumptions, with the violated ones flagged:**

1. Fixed data mixture across the ladder — *violated*: frontier runs change mixture and add mid-training/annealing phases; Hoffmann-style laws fitted on one mixture do not transfer.
2. Single link function $f$ shared across scales — *violated in practice*: $s_\theta$'s ranking depends on the probability mass on *incorrect* options, which scales differently from mass on the correct one (Schaeffer et al. 2024).
3. Eval examples i.i.d. and uncontaminated — *violated*: benchmark leakage into web corpora is documented and unquantified per run.
4. Metric is a deterministic function of the checkpoint — *violated* by prompt format, few-shot ordering, and seed; the same checkpoint moves several points under these.
5. No post-training — *violated* for every deployed model; the law is fitted on base models and used to argue about instruction-tuned ones.

## 3. State of the Art

**Established (reproduced, ablated).**
- Two-step $C \to L_T \to M$ extrapolation works *within a family, on tasks already above chance*. Gadre et al., "Language models scale reliably with over-training and on downstream tasks" (ICLR 2025, arXiv:2403.08540) predict average top-1 error over 17 tasks at relative error ≈0.05 from models using ~300× less compute, with a public 104-model testbed.
- Emergence is partly a metric artifact. Schaeffer, Miranda & Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?" (NeurIPS 2023, arXiv:2304.15004) show discontinuous-looking curves become smooth under continuous metrics, and induce sharp curves in vision models by swapping the metric.
- Loss, not scale, is the better index. Du et al., "Understanding Emergent Abilities of Language Models from the Loss Perspective" (arXiv:2403.15796) find models of different sizes at equal pretraining loss have near-equal task performance, with above-chance behaviour beginning below a task-specific loss threshold.

**Claimed but unablated / single-instance.**
- The GPT-4 Technical Report (OpenAI, 2023, arXiv:2303.08774) predicts mean log pass-rate on a HumanEval subset from runs at ≤1/1000 the compute. One family, one task, subset chosen post hoc; no held-out replication. It also reports a task (Hindsight Neglect) where the trend inverted — a counterexample from the same paper.
- Compute-efficient "task ladders" (Bhagia et al., arXiv:2412.04403) report a few points of absolute error predicting OLMo 2 7B/13B task accuracy, but per-task error varies widely and the good tasks are selected.
- **Benchmark-number-only:** most published "we predicted X" claims are single points on a single family. There is no cross-family, pre-registered prediction leaderboard.

**Theory SOTA.** Weak. Broken Neural Scaling Laws (Caballero et al., ICLR 2023, arXiv:2210.14891) fit smoothly-joined power-law segments and can *describe* breaks, but the break location is a fitted parameter, not predicted. Observational scaling laws (Ruan, Maddison & Hashimoto, NeurIPS 2024, arXiv:2405.10938) regress on a low-dimensional capability space recovered by PCA over ~100 public models — strong fits, but it is interpolation across an existing model population, not forward extrapolation of a new run.

## 4. What Is Known

- Pretraining loss extrapolates well: Kaplan et al. (2020) and Hoffmann et al. (NeurIPS 2022, arXiv:2203.15556) fit $L(N,D)$ over ~3 orders of magnitude of compute with residuals of order 1%; Chinchilla's 70B/1.4T prediction from a ladder up to ~$10^{22}$ FLOPs held.
- The loss-to-metric map is where error enters. Aggregate downstream error over ~17 tasks is predictable at ~5% relative error (Gadre et al.); *individual* task accuracy is far worse, and aggregation is doing the work.
- Chance floors destroy signal. On 4-way multiple choice, models below roughly 1B parameters trained on ~100B tokens sit within noise of 25% on MMLU-style items — so a 5-point ladder contributes ~1 usable point.
- Benchmark noise is not negligible. Madaan et al., "Quantifying Variance in Evaluation Benchmarks" (arXiv:2406.10229) find seed- and continuation-level spread on the order of a point on standard multiple-choice suites — the same order as the errors being claimed as successes.
- Scoring-rule choice moves ranking. Length-normalized vs. unnormalized vs. PMI-normalized likelihood changes accuracy by several points at fixed checkpoint; BIG-bench (Srivastava et al., TMLR 2023, arXiv:2206.04615) documents the metric sensitivity across 200+ tasks.
- Infinite-resolution evaluation works: PassUntil (Hu et al., ICLR 2024, arXiv:2310.03262) recovers signal below the resolution of pass@1 by sampling until first success, making generative-task curves fittable at small scale.

## 5. What Is Not Known

- **Methodologically blocked** (the dominant gap): there is no metric definition for discriminative multiple-choice benchmarks that is simultaneously (a) continuous and non-degenerate at small scale, (b) monotonically related to the reported accuracy, and (c) invariant to prompt format and scoring rule. Until this exists, "predict MMLU" is not a well-posed estimation problem. Schaeffer et al., "Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?" (arXiv:2406.04391) localizes the block: accuracy depends on the *gap* between correct and incorrect option scores, and the incorrect-mass dynamics are not captured by any fitted law.
- **Empirically open**: whether a link fitted on family $\mathcal{A}$ transfers to family $\mathcal{B}$ with a different tokenizer and mixture. Runnable — needs two independent ~10-point ladders to $10^{23}$ FLOPs — and unrun publicly.
- **Theoretically open**: no proof that a monotone $L_T \to M$ link must exist, nor any characterization of which tasks admit one. No lower bound on prediction error as a function of $\rho$.
- **Open**: the effect of post-training. All laws are base-model laws; nobody has shown RLHF/instruction-tuning gains are a predictable function of base loss.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure what it names, compounded by a floor**. Reported accuracy is a thresholded functional of a likelihood *ratio* over the candidate set. Its small-scale value is pinned at chance not because the model has no partial knowledge, but because $\arg\max$ discards the partial knowledge. So the regime where compute is cheap enough to sample densely is exactly the regime where the observable is constant. The identifiable-signal window opens only above the threshold $\ell_0$ — and $\ell_0$, $k$, and the ceiling $M_\infty$ are three free parameters fitted from the handful of points that lie inside it. That is near non-identifiability: many $(k, \ell_0)$ pairs fit the observed points equally well and diverge by 10+ points at $C^\star$. Adding compute does not fix it; it moves the window, and you are re-fitting on the same few points.

Secondary: per-task noise (~1 point) is comparable to claimed accuracy (~2 points), so single-task successes are not distinguishable from luck without repeated seeds, which multiplies cost.

## 7. Current Research (as of 2026)

- **Ladder methodology.** AI2 (OLMo) continues compute-efficient task ladders with public checkpoints; DataComp-LM/Gadre-style open testbeds remain the only reproducible substrate.
- **Observational scaling** (Stanford — Ruan, Hashimoto) extended to agentic and post-trained models *(frontier — verify)*.
- **Resolution-recovering metrics**: PassUntil-style sampling, per-token Brier scores, and rank-of-correct-answer statistics as fittable surrogates.
- **Loss-threshold theory** (Tsinghua/Zhipu line, Du et al.) linking task emergence to a critical pretraining loss.
- **Estimation hygiene**: Choshen et al., "A Hitchhiker's Guide to Scaling Law Estimation" (arXiv:2410.11840) quantifies how many models and what scale range a usable fit needs.
- **Frontier labs** report predicted-vs-actual benchmark tables in system cards; methodology is undisclosed and unfalsifiable externally *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does a $L_T \to M$ link fitted on one model family transfer to another?

- **Scale.** Two ladders, each 8 checkpoints, $10^{18}$–$10^{22}$ FLOPs (roughly 50M–1.4B params, Chinchilla-optimal tokens). Family A: Llama-style tokenizer, DCLM mixture. Family B: different tokenizer, different mixture (e.g. heavy code). Plus one target run per family at $10^{24}$ FLOPs ($\rho = 100$). Budget: ~$2.2\times10^{24}$ FLOPs total, dominated by the two targets; the ladders are under 2% of it.
- **Arms.** Fit the two-step law on family A's ladder. Predict family B's target. **Control arm:** fit on family B's *own* ladder and predict B's target. Second control: 3 seeds at the largest ladder rung to bound noise.
- **Tasks.** 6 benchmarks: 3 multiple-choice (ARC-C, HellaSwag, MMLU), 3 generative scored by PassUntil (HumanEval, GSM8K, a translation set).
- **Deciding number.** Mean absolute error in accuracy points on family B's target, cross-family minus within-family. If the excess is $\le 2$ points across all 6 tasks, links transfer and downstream prediction is a solved engineering problem. If it exceeds 5 points on any multiple-choice task, the link is family-specific and every published single-family success is uninformative about frontier practice.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Wei, Tay, Bommasani, et al. *Emergent Abilities of Large Language Models.* TMLR, 2022. — arXiv:2206.07682
- **[SOTA]** Gadre, Smyrnis, Shankar, et al. *Language Models Scale Reliably With Over-Training and on Downstream Tasks.* ICLR, 2025. — arXiv:2403.08540
- **[SOTA]** Ruan, Maddison, Hashimoto. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024. — arXiv:2405.10938
- **[SOTA]** Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004
- **[Analysis]** Schaeffer, Schoelkopf, Miranda, et al. *Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?* 2024. — arXiv:2406.04391
- **[Method]** Hu, Song, Cui, et al. *Predicting Emergent Abilities with Infinite Resolution Evaluation.* ICLR, 2024. — arXiv:2310.03262
- **[Method]** Bhagia, Liu, Wettig, et al. *Establishing Task Scaling Laws via Compute-Efficient Model Ladders.* 2024. — arXiv:2412.04403
- **[Analysis]** Du, Zeng, Lu, et al. *Understanding Emergent Abilities of Language Models from the Loss Perspective.* 2024. — arXiv:2403.15796
- **[Method]** Caballero, Gupta, Rish, Krueger. *Broken Neural Scaling Laws.* ICLR, 2023. — arXiv:2210.14891
- **[Measurement]** Madaan, Esiobu, Stenetorp, et al. *Quantifying Variance in Evaluation Benchmarks.* 2024. — arXiv:2406.10229
- **[Survey]** Srivastava, Rastogi, Rao, et al. *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models.* TMLR, 2023. — arXiv:2206.04615
- **[Practice]** Choshen, Zhang, Andreas. *A Hitchhiker's Guide to Scaling Law Estimation.* 2024. — arXiv:2410.11840

## 10. Worked Example

Predict MMLU (57 subjects, 4-way, chance $=0.25$) at $C^\star = 10^{24}$ FLOPs from a 5-point ladder.

Suppose the ladder gives (accuracy, task NLL in nats):

| $C$ (FLOPs) | $N$ | accuracy | $L_T$ |
|---|---|---|---|
| $10^{19}$ | 70M | 0.252 | 1.86 |
| $10^{20}$ | 160M | 0.254 | 1.74 |
| $10^{21}$ | 410M | 0.259 | 1.63 |
| $10^{22}$ | 1.0B | 0.271 | 1.52 |
| $10^{23}$ | 2.8B | 0.305 | 1.41 |

Step 1 is fine: $L_T$ falls ~0.11 nats per decade, a clean power law, extrapolating to $L_T(10^{24}) \approx 1.30$ with residuals under 0.01 nats.

Step 2 is where it breaks. Fit $M = 0.25 + 0.75/(1+e^{k(L_T-\ell_0)})$. Only the top two points are meaningfully above chance — the first three sit inside the ±0.01 seed-noise band, so they constrain nothing beyond "below threshold." Two informative points, two free parameters.

- Fit A: $k = 12$, $\ell_0 = 1.05$ → $\hat M(1.30) = 0.30$.
- Fit B: $k = 6$, $\ell_0 = 0.85$ → $\hat M(1.30) = 0.29$.
- Fit C: $k = 25$, $\ell_0 = 1.24$ → $\hat M(1.30) = 0.36$.

All three reproduce the observed accuracies to within 0.005 — inside noise. At $C^\star$ they predict 29%, 30%, 36%. One decade further, at $10^{25}$ ($L_T \approx 1.19$), they give 31%, 33%, **59%**. The spread is 26 points, and no data on the ladder can adjudicate.

That is the obstruction in one table: the loss extrapolation is not the problem, and the link is not overfit in any conventional sense — it is *under-determined*, because the metric's chance floor deletes the observations that would identify $k$ and $\ell_0$. Refitting on the same ladder with a continuous surrogate (mean rank of the correct option, or per-option Brier score) restores three usable points and collapses the spread; that substitution, and whether the surrogate's own link transfers across families, is the actual open question.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*