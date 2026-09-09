---
id: 10-scaling-laws/finetuning-transfer-scaling-law
title: "Transfer Scaling Law for Fine-Tuning"
topic: 10-scaling-laws
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Transfer Scaling Law for Fine-Tuning

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/finetuning-transfer-scaling-law` · **Status:** partially-solved

## 1. Problem Statement

Given a pretrained model and a target task, predict fine-tuned performance *before* fine-tuning, as a function of the knobs a practitioner actually controls: base model size $N$, pretraining tokens $D_P$, pretraining/target distribution overlap, fine-tuning set size $D_F$, and adaptation method (full fine-tuning, LoRA, prompt tuning).

Three variants, of very different difficulty:

- **Measurement variant.** Does fine-tuned loss on a fixed target distribution admit a smooth, low-parameter functional form in $(N, D_P, D_F)$, with residuals at the level of seed noise? Largely answered *yes* for target-distribution cross-entropy; not answered for downstream task metrics.
- **Method variant.** Can the coefficients of that form be estimated from cheap probes — small $N$, small $D_F$ — so that a practitioner spends $\ll 1\%$ of the target run's compute to choose between candidate base models and data budgets? Partially: rank-ordering works better than absolute prediction.
- **Theory variant.** Why do the exponents take the values they do, and what determines the transfer exponent as a function of the pretrain/target relationship? Open.

A solution to the practical problem is a predictor that, from probes costing $\le 1\%$ of the target run, forecasts final target loss within the run-to-run seed standard deviation, and gets the *argmax over configurations* right.

## 2. Formal Setting

Pretraining distribution $\mathcal{P}$, target distribution $\mathcal{T}$. A base model $f_{\theta_0}$ with $N$ non-embedding parameters, trained on $D_P$ tokens from $\mathcal{P}$. Fine-tuning uses $D_F$ tokens from $\mathcal{T}$, producing $\theta$. The measured quantity is held-out target cross-entropy in nats/token:

$$L(N, D_P, D_F) = -\frac{1}{|S|}\sum_{(x,y)\in S} \log p_\theta(y \mid x), \quad S \sim \mathcal{T}, \ S \cap \text{train} = \emptyset .$$

**Additive (Chinchilla-style) form**, fitted to fine-tuning:

$$L(N, D_F) = E + \frac{A}{N^{\alpha}} + \frac{B}{D_F^{\beta}} ,$$

with $E$ the irreducible entropy of $\mathcal{T}$, and $\alpha,\beta>0$. Fitted by Huber loss on $\log L$ over a grid, as in Hoffmann et al. (2022).

**Effective-data-transferred form** (Hernandez et al., 2021). Define $D_T(D_F, N)$ as the extra target tokens a from-scratch model of the same $N$ would need to reach the loss the fine-tuned model reaches with $D_F$:

$$D_T = k\, D_F^{\alpha}\, N^{\beta}, \qquad \text{effective data} = D_F + D_T .$$

$D_T$ is *measured*, not assumed: run the from-scratch sweep, invert its $L(D)$ curve at the fine-tuned model's loss. This inversion is the expensive part and the source of most of the error bars.

**Rectified form** (Lin et al., 2024), for the small-$D_F$ regime:

$$L(D_F) = E + \frac{B}{(D_\ell + D_F)^{\beta}},$$

where $D_\ell$ is a fitted "pre-learning" offset that produces a plateau before the power-law phase begins.

**Assumptions, and which are violated.**

1. *Fixed target distribution across the sweep.* Violated whenever $D_F$ grows by adding data from a different source than the seed set.
2. *Optimal or at least consistent hyperparameters at every grid point.* Violated in practice: learning rate, epoch count and LoRA rank are usually tuned at one scale and reused. This biases $\alpha$ and $\beta$ by an unquantified amount.
3. *Single epoch / no repetition.* Violated — fine-tuning sets are small and typically seen 2–5 times. Repetition changes the effective $D_F$ (Muennighoff et al., 2023).
4. *Loss is the objective.* Violated for every deployed use: the decision is made on accuracy, BLEU, pass@1, or a preference score.
5. *$E$ is identifiable.* In practice $E$, $B$ and $\beta$ are jointly degenerate over the two-decade $D_F$ ranges people actually sweep (see §10).

## 3. State of the Art

**Established.**

- Hernandez, Kaplan, Henighan & McCandlish, *Scaling Laws for Transfer* (2021, arXiv:2102.01293). Text→Python transfer. Effective data transferred fits $D_T \propto D_F^{\alpha} N^{\beta}$ with small $\alpha \approx 0.18$ and $\beta \approx 0.38$ over $N$ from ~$10^5$ to ~$10^9$ parameters. Sublinear in $D_F$: the *fraction* of your fine-tuning data that is "free" from pretraining shrinks as $D_F$ grows. Pretraining substitutes for target data most strongly in the low-data limit.
- Zhang et al., *When Scaling Meets LLM Finetuning* (ICLR 2024, arXiv:2402.17193). Bilingual models 1B–16B, translation and summarization, full fine-tuning vs. LoRA vs. prompt tuning. A multiplicative joint law in $(N, D_P, D_F)$ fits better than additive; scaling base model size helps more than scaling $D_P$ or $D_F$; PET methods' fine-tuning-data exponent is smaller than full fine-tuning's.
- Isik et al., *Scaling Laws for Downstream Task Performance in Machine Translation* (ICLR 2025, arXiv:2402.04177). Downstream BLEU/COMET follows a log-law in $D_P$ **only when** pretrain and target distributions are aligned; under misalignment BLEU is non-monotonic in $D_P$ while downstream cross-entropy still falls monotonically. This is the sharpest existing evidence that loss-based transfer laws do not transfer to task metrics.

**Claimed but unablated.** The multiplicative-beats-additive claim in Zhang et al. rests on fit quality within one model family and two tasks; no independent replication at a different family or task set. The specific exponents in Hernandez et al. have not been re-measured on modern data mixes at $\ge 10^{10}$ parameters.

**Benchmark-number-only.** Most "our law predicts X" claims for instruction tuning are reported as a single held-out point on one benchmark suite, without a from-scratch control or a seed-noise bar. Treat them as anecdotes.

## 4. What Is Known

- **Sublinear transfer exponent.** $\alpha < 1$ in $D_T \propto D_F^{\alpha}$ (Hernandez et al., 2021; measured $\approx 0.18$ for text→Python across $10^5$–$10^9$ params). Consequence: doubling fine-tuning data does not double effective data.
- **Model size dominates.** Zhang et al. (ICLR 2024), 1B–16B: the exponent on $N$ exceeds the exponent on $D_F$ across all three fine-tuning methods tested. Choosing a bigger base beats collecting more target data at typical budgets.
- **LoRA is not a free substitute.** Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024, arXiv:2405.09673), Llama-2 7B/13B on code and math: LoRA trails full fine-tuning on target-domain gains, especially in the continued-pretraining regime ($\sim 20$B tokens), while degrading source-domain performance less. The gap is *method*-dependent, not a constant offset — so a single law cannot cover both.
- **Phase transition at small $D_F$.** Lin et al. (ICML 2024, arXiv:2402.02314): loss is flat then power-law; ignoring the pre-learning phase makes naive power-law fits mis-rank candidate models.
- **Repetition is cheap up to a point.** Muennighoff et al., NeurIPS 2023: up to ~4 epochs of repeated data is close to fresh data; beyond ~16 epochs the return is near zero. Directly limits how far $D_F$ can be faked.
- **Very small $D_F$ can suffice for style.** LIMA (Zhou et al., NeurIPS 2023): 1,000 curated examples. Capability transfer and format transfer scale differently.

## 5. What Is Not Known

- **Theoretically open.** No derivation of the transfer exponent from a stated property of $\mathcal{P}$ and $\mathcal{T}$. Data-manifold-dimension arguments (Sharma & Kaplan, JMLR 2022; Bahri et al., PNAS 2024) explain pretraining exponents but say nothing about how the pretrained initialization changes them.
- **Empirically open.** Whether Hernandez-style exponents hold above $10^{10}$ parameters, on modern heavily-filtered mixes, is runnable and unrun — it needs a from-scratch control at frontier scale, which nobody pays for.
- **Methodologically blocked.** Predicting *downstream metric* rather than loss. Isik et al. show the two can move in opposite directions. Until there is an agreed way to map a loss law onto a bounded, thresholded, non-differentiable metric, the practically relevant law is not merely unmeasured — it is undefined.
- **Open, mixed.** A single law spanning full FT, LoRA at rank $r$, and prompt tuning, with $r$ as a continuous knob.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of $(E, B, \beta)$ over the sweep ranges that are affordable**, compounded by a control arm that costs more than the thing being predicted.

- $E$ (irreducible target entropy) is not separately measurable. Fitting it jointly with $B$ and $\beta$ over $\le 2$ decades of $D_F$ leaves a valley of near-equivalent optima (§10). Two fits indistinguishable in-range differ by ~2× when extrapolated one decade out.
- The effective-data-transferred definition requires a **from-scratch** sweep at every $N$. That control costs more compute than every fine-tuning run in the study combined, which is why the measurement exists only at small $N$.
- **Confounded measurement:** hyperparameters are re-tuned unevenly across the grid, so a fitted exponent partly measures tuning effort.
- **Evaluation does not measure what it names:** "transfer" is scored by target cross-entropy, but the decision is made on a task metric that can move the other way.

## 7. Current Research (as of 2026)

- **Observational scaling laws** — Ruan, Maddison & Hashimoto (NeurIPS 2024): fit low-dimensional capability factors across ~100 existing public models instead of running the sweep. Sidesteps compute cost; inherits selection bias from which models got released.
- **Rectified / phase-aware laws** for model selection under tiny $D_F$ (Lin et al. and follow-ups).
- **Data-mixture laws** applied to the fine-tuning mixture rather than the pretraining mixture — predicting target loss as a function of mixture proportions. *(frontier — verify)*
- **PEFT-aware laws** treating LoRA rank as a capacity variable alongside $N$. Active at Meta AI, Google DeepMind, and Databricks/Mosaic. *(frontier — verify)*
- **RL/post-training compute laws** — whether the same transfer form covers preference optimization and RLVR. Early, contested. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** is the fine-tuning-data exponent $\beta$ invariant to base model scale, or does it drift?

- **Scale.** Base models at $N \in \{160\text{M}, 410\text{M}, 1.4\text{B}, 2.8\text{B}, 6.9\text{B}\}$ from one open family with public checkpoints (Pythia, or OLMo). One target domain with a large clean corpus (e.g. legal text or a single programming language). Fine-tune at $D_F \in \{3\times10^5, 10^6, \ldots, 10^9\}$ tokens — **four decades**, not two. 3 seeds per cell. Learning rate re-swept at every cell over 5 values; report the best. Cost: order $10^{21}$ FLOPs total, feasible on ~64 H100s in under two weeks.
- **Control arm.** From-scratch training at the same $N$ and the same $D_F$ grid, same LR sweep. This is what makes $D_T$ measurable rather than assumed. A second control: LoRA at $r=16$ on the same grid.
- **The deciding number.** Fit $L = E + A/N^{\alpha} + B/D_F^{\beta}$ *independently at each $N$* and report $\beta(N)$ with bootstrap 95% CIs. **If the CIs for $\beta(160\text{M})$ and $\beta(6.9\text{B})$ overlap, a single global $\beta$ is justified and cheap small-$N$ probes can set fine-tuning budgets at large $N$. If they are disjoint by more than 0.03, every published transfer law fitted at one scale is being extrapolated illegitimately**, and the practical recommendation becomes: probe at the deployment scale or not at all.

Secondary readout: $\hat{D}_T$ from the from-scratch inversion versus $D_T$ predicted by the Hernandez form, as a ratio. A ratio outside $[0.5, 2]$ at the largest $N$ falsifies the 2021 exponents for modern mixes.

## 9. Key References

- **[Foundational]** Hernandez, D., Kaplan, J., Henighan, T., McCandlish, S. *Scaling Laws for Transfer.* arXiv preprint, 2021. — arXiv:2102.01293
- **[Foundational]** Hoffmann, J. et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Foundational]** Kaplan, J. et al. *Scaling Laws for Neural Language Models.* arXiv preprint, 2020. — arXiv:2001.08361
- **[SOTA]** Zhang, B. et al. *When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method.* ICLR, 2024. — arXiv:2402.17193
- **[SOTA]** Isik, B., Ponomareva, N., Hazimeh, H., Paparas, D., Vassilvitskii, S., Koyejo, S. *Scaling Laws for Downstream Task Performance in Machine Translation.* ICLR, 2025. — arXiv:2402.04177
- **[SOTA]** Lin, H. et al. *Selecting Large Language Model to Fine-tune via Rectified Scaling Law.* ICML, 2024. — arXiv:2402.02314
- **[SOTA]** Biderman, D. et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[Method]** Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Method]** Muennighoff, N. et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- **[Method]** Ruan, Y., Maddison, C. J., Hashimoto, T. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024. — arXiv:2405.10938
- **[Theory]** Bahri, Y., Dyer, E., Kaplan, J., Lee, J., Sharma, U. *Explaining Neural Scaling Laws.* PNAS, 2024. — arXiv:2102.06701
- **[Context]** Zhou, C. et al. *LIMA: Less Is More for Alignment.* NeurIPS, 2023. — arXiv:2305.11206

## 10. Worked Example

A team fine-tunes a 7B base on a domain corpus. They can afford $D_F \in \{10^4, 10^5\}$ tokens of measurement and want to know whether collecting $10^7$ tokens is worth it. They fit $L = E + B/D_F^{\beta}$ with $E$ fixed at a plausible 1.80 nats.

Measured (3 seeds, seed SD $\approx 0.006$ nats):

| $D_F$ | measured $L$ | reducible part $L-E$ |
|---|---|---|
| $10^4$ | 1.863 | 0.063 |
| $10^5$ | 1.834 | 0.034 |

Two fits:

- **Fit A:** $\beta = 0.30$, $B = 1.00$. Predicts $0.0631$ at $10^4$, $0.0316$ at $10^5$.
- **Fit B:** $\beta = 0.20$, $B = 0.398$. Predicts $0.0631$ at $10^4$, $0.0398$ at $10^5$.

Residual gap between them at $10^5$: $0.0082$ nats — about $1.4\times$ the seed SD. With 3 seeds the standard error of the mean is $\approx 0.0035$, so the two fits are separated at roughly $2.3\sigma$: suggestive, not decisive, and entirely erased if $E$ is allowed to float by $\pm 0.01$.

Extrapolate to $D_F = 10^7$:

- Fit A: $1.00/10^{2.1} = 0.0079$ → $L = 1.808$.
- Fit B: $0.398/10^{1.4} = 0.0158$ → $L = 1.816$.

The reducible loss differs by **2.0×**. Under Fit A, going from $10^5$ to $10^7$ tokens buys 75% of the remaining headroom; under Fit B it buys 54%. The data-collection decision — roughly $10^7$ tokens of expert-labeled domain text — flips on a difference the affordable measurement cannot resolve.

**The obstruction, made visible:** separating $\beta = 0.30$ from $\beta = 0.20$ at the same $E$ requires either a third decade of $D_F$ or driving the standard error below $\sim 0.001$ nats, which needs $\sim 36$ seeds per cell. And if $E$ is fitted rather than fixed, the $(E, B, \beta)$ valley absorbs the difference entirely — the fit is degenerate, not merely noisy. This is why §8 insists on four decades of $D_F$ and per-$N$ bootstrap CIs rather than a single global fit.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*