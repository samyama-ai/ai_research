---
id: 10-scaling-laws/ensembling-merging-scaling-law
title: "Scaling Laws for Model Ensembling and Merging"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws for Model Ensembling and Merging

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/ensembling-merging-scaling-law` · **Status:** empirically-open

## 1. Problem Statement

Given a total training budget $C$, you can spend it on one model or split it across $K$ models that are later combined — by output ensembling (average the predictive distributions) or by weight merging (average or arithmetically combine the parameter vectors). The question: **is there a predictive scaling law that says which split wins, and at what budget the answer flips?**

Three variants, different difficulty:

- **Measurement.** Fit $L(C, K)$ — loss as a joint function of budget and member count — on a controlled grid, and report the compute-equivalent gain (CEG): the factor by which a single model's budget must grow to match the combination. Runnable today; not run at language-model scale with a proper control.
- **Method.** Design a merge whose CEG grows with $K$ rather than saturating. Current merges saturate fast; most gain arrives by $K=3$–$5$.
- **Theory.** Prove when merging is lossless — i.e. when $\ell(\bar\theta) \le \frac{1}{K}\sum_k \ell(\theta_k)$ — from properties of the loss landscape, and derive the exponent of the residual in $K$. Open.

Solving it means: a fitted law with held-out predictive error small enough to choose $K$ before spending the budget, plus an identified condition under which the law fails.

## 2. Formal Setting

Members $\theta_1,\dots,\theta_K \in \mathbb{R}^N$, each trained with $N$ parameters on $D$ tokens, so per-member compute is $C_1 \approx 6ND$ and total $C = K C_1$. Loss is next-token cross-entropy in nats on a held-out corpus $\mathcal{D}_{\text{val}}$:

$$L(\theta) = -\frac{1}{|\mathcal{D}_{\text{val}}|}\sum_{(x,y)} \log p_\theta(y \mid x).$$

**Output ensemble** (measured by running all $K$ forward passes and averaging probabilities before the log):

$$L_{\text{ens}}(K) = -\frac{1}{|\mathcal{D}_{\text{val}}|}\sum \log\Big(\tfrac{1}{K}\sum_{k} p_{\theta_k}(y\mid x)\Big).$$

Jensen gives $L_{\text{ens}}(K) \le \frac{1}{K}\sum_k L(\theta_k)$ unconditionally; the gap is the ensemble's Bregman diversity term, measured directly as the difference of those two numbers.

**Weight merge** (one forward pass, $K$ training runs): $\theta_{\text{merge}} = \sum_k \lambda_k \theta_k$ with $\sum \lambda_k = 1$, or task-arithmetic form $\theta_0 + \sum_k \lambda_k (\theta_k - \theta_0)$ around a shared pretrained anchor $\theta_0$.

**Baseline law.** Hoffmann et al. (2022) fit $L(N,D) = E + A N^{-\alpha} + B D^{-\beta}$ with $E=1.69$, $A=406.4$, $\alpha=0.34$, $B=410.7$, $\beta=0.28$.

**Decision quantity.** The compute-equivalent gain, measured by inverting the single-model law:

$$\mathrm{CEG}(K) = C^{\ast}\big(L_{\text{comb}}(K)\big) \big/ (K C_1),$$

where $C^\ast(\ell)$ is the budget at which a compute-optimal single model reaches loss $\ell$. $\mathrm{CEG}>1$ means the split beat the monolith at equal training compute. Report separately at **fixed training compute** and **fixed inference compute** — an ensemble costs $K\times$ inference, a merge costs $1\times$.

Assumptions and their status:

- *Members are exchangeable and identically distributed.* Violated whenever members are fine-tunes of a shared $\theta_0$ on different data — the usual case.
- *Losses are on a single common validation distribution.* Violated in the merging literature, which reports per-task accuracy averages, not one likelihood.
- *The single-model law is valid at the interpolated budgets.* Violated near the fit's edges; Chinchilla's constants were fit over $\sim 7\times10^{18}$–$5\times10^{23}$ FLOPs.
- *Linear mode connectivity holds between members.* Violated for independent initializations (Frankle et al., 2020); merging only works inside one basin.

## 3. State of the Art

**Empirical SOTA — established.**
- *Model soups* (Wortsman et al., ICML 2022): greedy averaging of fine-tunes from one pretrained checkpoint; a ViT-G/14 soup reached 90.94% ImageNet top-1, above the best individual member of the sweep, at unchanged inference cost. Ablated over hyperparameter sweeps.
- *Deep ensembles obey a power law in $K$* (Lobacheva et al., NeurIPS 2020): calibrated log-likelihood improves as a power law in $K$ with a saturating "deep ensemble equivalent" model size, measured on CIFAR/ImageNet CNNs.
- *Ensembles of small models can beat one large model at matched FLOPs* (Kondratyuk et al., 2020; Mustafa et al., "Wisdom of committees", 2022) — vision only.
- *Linear mode connectivity is basin-local* (Frankle et al., ICML 2020; Ainsworth et al., ICLR 2023 for the permutation-quotient version; Jordan et al., ICLR 2023 for the variance-collapse repair).

**Claimed but unablated.**
- *What Matters for Model Merging at Scale?* (Yadav et al., 2024) merges up to 8 experts on PaLM-2 bases from 1B to 64B and reports that merging quality improves with base size and with expert count, and that large merged models approach multitask training on held-out tasks. This is the closest thing to a merging scaling law, but it reports task-accuracy averages, not a fitted $L(N,D,K)$, and does not include an equal-training-compute single-model control arm.
- *Benchmark-number-only results.* TIES-Merging (Yadav et al., NeurIPS 2023), DARE (Yu et al., ICML 2024) and most merge variants are reported as averages over task suites at fixed $K$ and one base size. No exponent is extractable from them.

**Theory SOTA.** Weakest layer. DiWA (Ramé et al., NeurIPS 2022) gives a bias–variance–covariance–locality decomposition explaining when weight averaging approximates output ensembling. Ortiz-Jimenez et al. (NeurIPS 2023) show task arithmetic works to the extent the model is in a weight-disentangled, near-linear (NTK) regime. Neither yields a rate in $K$ or in $N$.

## 4. What Is Known

- **Ensembling never hurts the log-loss** (Jensen), exactly; merging can and does hurt, catastrophically, across initializations — interpolating two independently trained networks typically lands near chance-level accuracy before permutation alignment (Frankle et al., 2020).
- **Returns in $K$ saturate early.** In deep-ensemble studies at CIFAR-scale ResNets, most of the calibrated-log-likelihood gain arrives by $K \approx 5$; beyond that the marginal member buys a decreasing power-law increment (Lobacheva et al., 2020).
- **Merging gain grows with base scale.** On PaLM-2 bases at 1B/8B/24B/64B, held-out generalization of merged experts improves monotonically with base size (Yadav et al., 2024) — measured in task accuracy, at $K \le 8$.
- **Averaging beats single fine-tunes at zero inference cost** in the shared-anchor regime: soups (ImageNet, ViT-B through ViT-G), WiSE-FT robustness interpolation (Wortsman et al., CVPR 2022), WARM reward-model averaging (Ramé et al., ICML 2024).
- **The routed-model analogue has a fitted law.** Clark et al. (ICML 2022) fit a joint law in parameters and expert count for MoE language models up to ~$10^{21}$ FLOPs, and find the expert-count benefit shrinks as dense size grows. This is the nearest existing precedent, and it predicts saturation.

## 5. What Is Not Known

- **Empirically open.** No published joint fit $L(N,D,K)$ for language models with an equal-training-compute single-model control. The grid is affordable at $10^{19}$–$10^{21}$ FLOPs; nobody has published it. Whether $\mathrm{CEG}(K)>1$ anywhere on the compute-optimal frontier for LM cross-entropy is unresolved.
- **Empirically open.** Whether the crossover point moves with scale — i.e. whether ensembling's advantage vanishes as $N$ grows, as the MoE result suggests it should.
- **Theoretically open.** No proof of a rate for the merge residual $L(\bar\theta) - \frac{1}{K}\sum L(\theta_k)$ as a function of $K$ and inter-member distance $\max_k \|\theta_k-\theta_0\|$, beyond second-order Taylor arguments that assume a shared positive-definite Hessian.
- **Methodologically blocked.** "Merging quality" has no agreed measurement. Task-accuracy averages over heterogeneous suites are not comparable to a cross-entropy scaling law, and are not invertible into a compute-equivalent budget. Until merging is reported as a likelihood on a fixed distribution, its law cannot be fitted at all.

## 6. Why It Is Hard

Four named obstructions:

1. **Confounded measurement.** Merging results are reported on task suites where member models were trained on different data. Any gain mixes three causes — data coverage, regularization from averaging, and ensemble diversity — and no published ablation separates them.
2. **Non-identifiability.** $L(N,D,K)$ has strongly correlated directions: adding members, adding parameters and adding tokens all decrease loss. With $K \le 8$ and two base sizes, the $K$-exponent is not identified. You need $\ge 4$ values of $K$ crossed with $\ge 4$ values of $N$ — 16+ full training runs.
3. **The control arm is the expensive one.** To test $K=8$ at member budget $C_1$, you must also train one model at $8C_1$. The control dominates the cost, which is exactly why it is usually omitted.
4. **An evaluation that does not measure what it names.** "Merged model matches multitask training" is measured on suites whose ceiling effects hide the loss difference. Two models 0.05 nats apart can score identically on MMLU.

## 7. Current Research (as of 2026)

- **Merging at scale.** Google DeepMind and UNC-Chapel Hill (Yadav, Raffel and collaborators) on base-size and expert-count dependence.
- **Weight averaging in alignment.** Ramé and colleagues (Google DeepMind) — WARM, WARP, rewarded soups: averaging as a variance-reduction device for reward hacking.
- **Landscape theory.** Permutation-symmetry and cross-task-linearity work (Ainsworth, Jordan, Zhou et al.) attempting to explain when a basin admits lossless averaging.
- **Observational laws.** Ruan, Maddison and Hashimoto (NeurIPS 2024) fit scaling laws across existing public models via latent capability axes — a cheap route to a first $K$-exponent estimate from checkpoints that already exist *(frontier — verify: no published application to merged models)*.
- **Open-weight merge ecosystems.** Thousands of community merges exist as artifacts; they are an uncontrolled observational dataset, not an experiment.

## 8. Concrete Next Experiment

**Scale.** A $4\times4$ grid at $10^{19}$–$3\times10^{20}$ FLOPs. Decoder-only LMs at $N \in \{70\text{M}, 160\text{M}, 410\text{M}, 1\text{B}\}$, each trained Chinchilla-optimal ($D = 20N$) on disjoint shards of one corpus, $K \in \{1,2,4,8\}$ members per cell. Report three combination arms: output ensemble, uniform weight merge from independent inits, and uniform merge from a shared 10%-of-budget pretrained anchor $\theta_0$.

**Control arm.** For each cell, one single model trained at the *full* budget $K \cdot 6ND$, allocated compute-optimally. This is non-negotiable; without it, only relative claims are possible.

**Deciding number.** $\mathrm{CEG}(K=8)$ at $N=1$B, measured in held-out nats on a fixed validation set and converted to budget through the single-model fit. If $\mathrm{CEG}(8) > 1.0$ for the output ensemble but the trend in $N$ is decreasing, splitting is a small-model artifact. If $\mathrm{CEG}(8) > 1.0$ and flat or rising in $N$, compute allocation for frontier training is wrong. Secondary number: the merge residual $L(\bar\theta) - L_{\text{ens}}$, which quantifies exactly what one pays to collapse $K$ forward passes into one.

Cost estimate: ~40 runs totalling $\sim 4\times10^{21}$ FLOPs — days on a 64-GPU cluster.

## 9. Key References

- **[Foundational]** Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Lakshminarayanan, Pritzel, Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. — arXiv:1612.01474
- **[SOTA]** Lobacheva, Chirkova, Kodryan, Vetrov. *On Power Laws in Deep Ensembles.* NeurIPS 2020. — arXiv:2007.08483
- **[SOTA]** Wortsman, Ilharco, Gadre, et al. *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy Without Increasing Inference Time.* ICML 2022. — arXiv:2203.05482
- **[SOTA]** Yadav, Raffel, Muqeeth, et al. *What Matters for Model Merging at Scale?* 2024. — arXiv:2410.03617
- **[SOTA]** Clark, de las Casas, Guy, et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- Ilharco, Ribeiro, Wortsman, et al. *Editing Models with Task Arithmetic.* ICLR 2023. — arXiv:2212.04089
- Yadav, Tam, Choshen, Raffel, Bansal. *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS 2023. — arXiv:2306.01708
- Matena, Raffel. *Merging Models with Fisher-Weighted Averaging.* NeurIPS 2022. — arXiv:2111.09832
- Frankle, Dziugaite, Roy, Carbin. *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* ICML 2020. — arXiv:1912.05671
- Ainsworth, Hayase, Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* ICLR 2023. — arXiv:2209.04836
- Ortiz-Jimenez, Favero, Frossard. *Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models.* NeurIPS 2023. — arXiv:2305.12827
- Ramé, Kirchmeyer, Rahier, et al. *Diverse Weight Averaging for Out-of-Distribution Generalization.* NeurIPS 2022. — arXiv:2205.09739
- Ramé, Vieillard, Hussenot, et al. *WARM: On the Benefits of Weight Averaged Reward Models.* ICML 2024. — arXiv:2401.12187
- **[Survey]** Yang, Shen, Wang, et al. *Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities.* 2024. — arXiv:2408.07666

## 10. Worked Example

Take the Chinchilla fit and the 70B/1.4T point, $C = 6ND = 5.88\times10^{23}$ FLOPs:

$$L = 1.69 + \frac{406.4}{(7\times10^{10})^{0.34}} + \frac{410.7}{(1.4\times10^{12})^{0.28}} = 1.69 + 0.083 + 0.159 = 1.932 \text{ nats.}$$

Now split the same budget four ways. Each member gets $C/4 = 1.47\times10^{23}$ FLOPs, compute-optimally $N = 35$B, $D = 700$B:

$$L_1 = 1.69 + \frac{406.4}{(3.5\times10^{10})^{0.34}} + \frac{410.7}{(7\times10^{11})^{0.28}} = 1.69 + 0.106 + 0.198 = 1.994 \text{ nats.}$$

**The bar is 0.062 nats.** A 4-member ensemble must extract at least that much from prediction averaging just to tie the monolith on training compute — while paying $4\times$ at inference. A 4-way merge must extract it *and* survive collapsing four parameter vectors into one.

The obstruction is visible here: nobody knows the left-hand side. The ensemble gain is the Bregman diversity term, and it depends on how correlated four 35B models' errors are — a quantity never measured for language models on a common validation likelihood. Vision-scale evidence (Lobacheva et al., 2020) suggests $K=4$ buys the equivalent of roughly a 2× model at CIFAR scale, which would clear 0.062 nats easily; the MoE law (Clark et al., 2022) suggests the benefit shrinks with dense size, which at 35B could put it under the bar. Both extrapolations are three orders of magnitude out of their fitted range. The 40-run grid in §8 replaces the extrapolation with a number.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*