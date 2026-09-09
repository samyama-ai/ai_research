---
id: 10-scaling-laws/loss-to-loss-prediction
title: "Loss-to-Loss Prediction Across Datasets"
topic: 10-scaling-laws
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Loss-to-Loss Prediction Across Datasets

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/loss-to-loss-prediction` · **Status:** partially-solved

## 1. Problem Statement

A compute scaling law is fit per pretraining distribution. Change the data mixture and the fit is discarded: the constants, the exponent, and the irreducible term all move, and re-deriving them costs a fresh model ladder. **Loss-to-loss prediction** asks whether the dependence on data can be factored out — whether the map from a model's loss on one distribution to its loss on another is a low-parameter, scale-free function that can be fit cheaply at small scale and used at large scale.

Three variants, of increasing difficulty:

- **Measurement.** Given a family of models trained to varying compute on pretraining set $D_1$, and the same family trained on $D_2$, does there exist a fixed monotone $f$ such that $L_{D_2}(\text{model}) \approx f(L_{D_1}(\text{model}))$, with the map independent of model size and token count? This is answerable with existing runs.
- **Method.** Fit $f$ from small models only, then use a known scaling law on $D_1$ plus $f$ to predict the large-model loss on $D_2$ without training any large model on $D_2$. Solving this means: relative error on held-out large-scale points below what an independently fit scaling law on $D_2$ achieves, at a fraction of the compute.
- **Theory.** Explain why $f$ takes the observed shifted-power-law form, and predict its parameters from measurable properties of $D_1, D_2$ (overlap, entropy, token distribution) rather than fitting them.

The measurement variant is largely settled for pretraining-corpus pairs. The method variant works within its tested envelope. The theory variant is untouched. Hence *partially-solved*.

## 2. Formal Setting

Let $\mathcal{D}$ index distributions over token sequences. A training run is $(N, D, \mathcal{P})$: $N$ non-embedding parameters, $D$ training tokens, pretraining distribution $\mathcal{P} \in \mathcal{D}$. Define the measured loss on evaluation distribution $\mathcal{Q}$ as the token-averaged cross entropy on a held-out sample:

$$L_{\mathcal{Q}}(N, D, \mathcal{P}) = -\frac{1}{|S|}\sum_{(x_{<t}, x_t) \in S} \log p_{\theta(N,D,\mathcal{P})}(x_t \mid x_{<t}), \quad S \sim \mathcal{Q}.$$

In practice $|S|$ is $10^6$–$10^7$ tokens, giving a standard error on $L$ of order $10^{-3}$ nats — small relative to the effects of interest, but not relative to the disagreements between fits.

**Loss-to-loss map.** Fix two pretraining sets $\mathcal{P}_1, \mathcal{P}_2$ and an evaluation set $\mathcal{Q}$. Pair up runs at matched compute $C = 6ND$. The claim under test is the existence of a **shifted power law**

$$L_{\mathcal{Q}}\!\left(\mathcal{P}_2\right) \;=\; K\,\big(L_{\mathcal{Q}'}(\mathcal{P}_1) - E_1\big)^{\alpha} \;+\; E_2,$$

with four free parameters $(K, \alpha, E_1, E_2)$ and no dependence on $N$ or $D$ beyond what enters through the argument. Three instantiations:

- **train-to-train:** $\mathcal{Q} = \mathcal{P}_2$, $\mathcal{Q}' = \mathcal{P}_1$ — each model evaluated on its own pretraining distribution.
- **train-to-test:** $\mathcal{Q} = \mathcal{Q}' = $ a third, held-out distribution; predict downstream-corpus loss of a $\mathcal{P}_2$-trained model from that of a $\mathcal{P}_1$-trained model.
- **test-to-test:** one model, two evaluation sets; the map relates $L_{\mathcal{Q}_a}$ to $L_{\mathcal{Q}_b}$ along a single training trajectory or ladder.

**Assumptions, and which fail.**

1. *Matched compute-optimality.* Pairing requires both families to sit on the same $N/D$ trade-off (Chinchilla-style, $D \approx 20N$). Violated whenever one arm is over-trained; over-training moves $L$ at fixed $C$ by amounts comparable to the between-dataset gap (Gadre et al., 2025).
2. *Tokenizer invariance.* Cross-entropy per token is not comparable across tokenizers. Held fixed by construction in existing work; violated the moment anyone compares published models.
3. *Deduplication of eval from train.* Contamination inflates $L_{\mathcal{Q}}$ downward non-uniformly across $\mathcal{P}_1,\mathcal{P}_2$, and breaks monotonicity of $f$.
4. *Single architecture family.* All published fits use one decoder-only recipe. Whether $f$ survives an architecture change (MoE, state-space, different depth/width ratio) is untested.
5. *Irreducible-entropy shift $E$ identifiable.* $E_1, E_2$ are fit, not measured. They are only weakly constrained by data confined to $1.5$ orders of magnitude of compute.

## 3. State of the Art

**Established.** Brandfonbrener, Anand, Vyas, Malach, and Kakade, *Loss-to-Loss Prediction: Scaling Laws for All Datasets* (TMLR, 2025; arXiv:2411.12925) is the reference result. They train matched ladders on six pretraining corpora (C4, FineWeb, FineWeb-Edu, ProofPile 2, SlimPajama, SmolLM Corpus), pair runs at matched compute, and show all three map types are well fit by the shifted power law above. The empirical ordering they report is robust: **train-to-train fits are the tightest, test-to-test next, train-to-test loosest.** Extrapolation is demonstrated from small models out to roughly an order of magnitude more compute than the fitting range. This is a measurement result with an ablation over dataset pairs — the strongest form available.

**Established, adjacent.** Ruan, Maddison, and Hashimoto, *Observational Scaling Laws* (NeurIPS 2024) show that ~100 public models' capabilities collapse onto a low-dimensional (≈3 PC) latent space, which is the loss-to-loss claim in a weaker, correlational form across uncontrolled training recipes. Gadre et al., *Language Models Scale Reliably with Over-Training and on Downstream Tasks* (ICLR 2025) fit an exponential map from average pretraining perplexity to average downstream top-1 error over 104 models, predicting error on 17 tasks with reported ~2 percentage-point accuracy for a 6.9B/138B-token model extrapolated from ~300× less compute.

**Claimed but unablated.** That the shifted power law is the *correct* functional form rather than a flexible three-parameter curve fit over a narrow dynamic range. Broken Neural Scaling Laws (Caballero et al., ICLR 2023) shows that four-parameter families fit almost any monotone loss curve over 1–2 decades; no published work has compared shifted power law against a matched-capacity alternative on held-out compute decades.

**Benchmark-number-only.** Every claim that loss-to-loss transfers to *downstream task accuracy* rests on aggregate benchmark scores, not per-task fits. Individual-task fits degrade sharply; Bhagia et al., *Establishing Task Scaling Laws via Compute-Efficient Model Ladders* (2024) needs a two-step compute→task-loss→accuracy decomposition precisely because the one-step map fails per task.

## 4. What Is Known

- Loss-to-loss maps across pretraining corpora are **monotone, smooth, and near-affine over the observed range**, with curvature captured by a shift term. Measured on models spanning roughly $20\text{M}$ to $3$B parameters at Chinchilla-optimal token counts, over six corpora (Brandfonbrener et al., 2025).
- **Train-to-train is more predictable than train-to-test.** The practical consequence is stated in the paper: predicting a new dataset's *own* loss is easier than predicting how a dataset change moves loss on a fixed third distribution — the reverse of what a practitioner choosing data actually wants.
- **Downstream error is a smooth function of pretraining loss in aggregate.** Gadre et al. (2025): exponential-decay fit, aggregate over 17 tasks, ~2 pp extrapolation error at 6.9B parameters; per-task fits are visibly worse and some tasks are unfittable at that scale.
- **Data mixture effects are themselves predictable from small runs.** Data Mixing Laws (Ye et al., 2024) and RegMix (Liu et al., 2024) fit mixture→loss surfaces from proxy models under 1B parameters and select mixtures that beat human-chosen baselines — evidence that the dataset axis is low-dimensional, consistent with loss-to-loss.
- **Negative result on the naive alternative.** Kaplan-style laws refit per dataset do not share exponents across corpora; the irreducible term $E$ moves the most (visible in Hoffmann et al., 2022, and in the data-constrained regime of Muennighoff et al., NeurIPS 2023).

## 5. What Is Not Known

- **Theoretically open.** No derivation of the shifted power law from a data-distribution model. There is no theorem connecting $(K, \alpha, E_1, E_2)$ to any measurable divergence between $\mathcal{P}_1$ and $\mathcal{P}_2$ — not KL, not $n$-gram overlap, not embedding-space coverage. Nothing rules the form out either; competing forms are not distinguished by existing data.
- **Empirically open.** Whether $f$ holds across an architecture change, across tokenizers, across the over-trained regime ($D/N \gg 20$), and past roughly $10^{23}$ FLOP. All runnable; none run at that scale in public.
- **Empirically open.** Whether $f$ fit on pretraining corpora transfers to post-training: does the map survive instruction tuning or RL, where the loss being minimized is no longer next-token cross entropy on $\mathcal{P}$?
- **Methodologically blocked.** Extension to *capabilities* rather than losses. There is no agreed measurement of "task performance" that is continuous, tokenizer-invariant, and free of the discretization that makes emergence artifactual (Schaeffer et al., NeurIPS 2023). Until a task metric is defined that behaves like a loss, loss-to-*capability* prediction cannot be posed cleanly, let alone solved.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the shift parameters over the accessible compute range**. $E_1$ and $E_2$ are asymptotic irreducible losses — quantities that only become visible as $C \to \infty$. Fits are performed over 1.5–2 decades of compute where $L$ moves by perhaps $0.4$–$0.8$ nats. In that window, a large $E$ with small $K$ and a small $E$ with large $K$ are nearly indistinguishable: the residual difference between such fits is below the $\sim 10^{-2}$ nat scatter from seed and data-order variation, while the two fits diverge by $O(10^{-1})$ nats when extrapolated two further decades. The parameter that controls extrapolation is exactly the one the data does not constrain.

Compounding it: the arm that matters for practice — train-to-test, "will switching corpora help me on the thing I care about?" — is the arm with the worst fit quality, and its ground truth requires training the large model on $\mathcal{P}_2$, which is the cost the method exists to avoid.

## 7. Current Research (as of 2026)

- **Harvard Kempner Institute (Kakade, Malach, Brandfonbrener, Vyas)** — extension of loss-to-loss beyond matched-compute pairing and toward mixture interpolation *(frontier — verify)*.
- **Allen Institute for AI** — compute-efficient task ladders (OLMo line, Bhagia et al.), pushing the two-step loss→task-loss→accuracy decomposition to more tasks.
- **Stanford / Hashimoto group** — observational scaling extended to post-trained and reasoning models *(frontier — verify)*.
- **Epoch AI** — auditing benchmark predictability from compute (Owen, *How Predictable Is Language Model Benchmark Performance?*, 2024) as the negative control on capability-level claims.
- **Data-mixture optimization** (Ye et al.; Liu et al.; and industrial mixture search at frontier labs) — the applied consumer of loss-to-loss, generally without testing the functional form's extrapolation.

## 8. Concrete Next Experiment

**Question:** does the shifted power law extrapolate, or is it a flexible fit?

**Scale.** Two pretraining corpora ($\mathcal{P}_1 =$ FineWeb-Edu, $\mathcal{P}_2 =$ ProofPile 2), one tokenizer, one architecture. Matched Chinchilla-optimal ladders at $N \in \{40\text{M}, 80\text{M}, 160\text{M}, 320\text{M}, 640\text{M}, 1.3\text{B}\}$. Fit $f$ on the four smallest points only ($\le 320$M, total ≈ $3\times10^{20}$ FLOP for both arms). Hold out $640$M and $1.3$B.

**Control arm.** Two controls, both mandatory. (a) A four-parameter alternative of matched capacity — logistic-in-log-loss — fit on the identical four points. (b) An independently fit Chinchilla law on $\mathcal{P}_2$ using the same four small $\mathcal{P}_2$ points, ignoring $\mathcal{P}_1$ entirely. Three seeds per configuration to establish the noise floor.

**The deciding number.** Absolute prediction error, in nats, on $L_{\mathcal{P}_2}$ at $N = 1.3$B. The shifted power law is a real functional form, not a curve fit, iff its error is **below $0.01$ nats and at least $2\times$ smaller than the matched-capacity alternative's**, with the gap exceeding the three-seed standard deviation. If the two forms land within noise of each other, the form is unidentified and the reported extrapolation success is a property of the narrow fitting range, not of the law. Cost: under 5,000 A100-hours.

## 9. Key References

- **[SOTA]** David Brandfonbrener, Nikhil Anand, Nikhil Vyas, Eran Malach, Sham Kakade. *Loss-to-Loss Prediction: Scaling Laws for All Datasets.* Transactions on Machine Learning Research, 2025. — arXiv:2411.12925
- **[Foundational]** Jared Kaplan, Sam McCandlish, Tom Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Yangjun Ruan, Chris J. Maddison, Tatsunori Hashimoto. *Observational Scaling Laws and the Predictability of Language Model Performance.* NeurIPS, 2024. — arXiv:2405.10938
- **[SOTA]** Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, et al. *Language Models Scale Reliably with Over-Training and on Downstream Tasks.* ICLR, 2025. — arXiv:2403.08540
- **[SOTA]** Akshita Bhagia, Jiacheng Liu, Alexander Wettig, et al. *Establishing Task Scaling Laws via Compute-Efficient Model Ladders.* Allen Institute for AI, 2024. — arXiv:2412.04403
- Ethan Caballero, Kshitij Gupta, Irina Rish, David Krueger. *Broken Neural Scaling Laws.* ICLR, 2023. — arXiv:2210.14891
- Jiasheng Ye, Peiju Liu, Tianxiang Sun, et al. *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance.* 2024. — arXiv:2403.16952
- Niklas Muennighoff, Alexander M. Rush, Boaz Barak, et al. *Scaling Data-Constrained Language Models.* NeurIPS, 2023. — arXiv:2305.16264
- Rylan Schaeffer, Brando Miranda, Sanmi Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004
- **[Survey]** David Owen. *How Predictable Is Language Model Benchmark Performance?* Epoch AI, 2024. — arXiv:2401.04757

## 10. Worked Example

Take two candidate fits of a train-to-train map, both fit on four points at $N \le 320$M where the measured $L_{\mathcal{P}_1}$ falls from $3.20$ to $2.80$ nats.

| Fit | $K$ | $\alpha$ | $E_1$ | $E_2$ | RMSE on the 4 fit points |
|---|---|---|---|---|---|
| A | $1.05$ | $1.00$ | $1.60$ | $1.50$ | $0.004$ nats |
| B | $0.62$ | $0.80$ | $1.00$ | $1.90$ | $0.004$ nats |

Both reproduce the fitting window to $0.004$ nats — below the $\approx 0.008$-nat seed-to-seed scatter observed across three data orderings at 320M. They are empirically the same fit.

Now extrapolate. At $1.3$B, suppose $L_{\mathcal{P}_1} = 2.55$:

- Fit A: $1.05 \times (2.55 - 1.60)^{1.00} + 1.50 = 2.498$
- Fit B: $0.62 \times (2.55 - 1.00)^{0.80} + 1.90 = 0.62 \times 1.402 + 1.90 = 2.769$

A gap of $0.27$ nats — roughly two-thirds of the entire loss improvement the ladder bought over 1.5 decades of compute. Push to $L_{\mathcal{P}_1} = 2.20$ (about $10^{23}$ FLOP) and the gap widens to $0.35$ nats.

The obstruction is now visible and is not a matter of noisier data. Two parameterizations that are **indistinguishable to five times the measurement precision inside the fitting window** disagree by more than a compute decade's worth of loss outside it. The data does not constrain $E_1$; $E_1$ alone determines the extrapolation. Reporting a good in-window $R^2$ — as most loss-to-loss results do — says nothing about the quantity the method is used for. Only held-out large-scale points, as in §8, discriminate the fits.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*