---
id: 03-training-dynamics/gradient-noise-scale-diagnostic
title: "Gradient Noise Scale as Training Diagnostic"
topic: 03-training-dynamics
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Gradient Noise Scale as Training Diagnostic

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/gradient-noise-scale-diagnostic` · **Status:** methodologically-blocked

## 1. Problem Statement

The gradient noise scale (GNS) is the ratio of gradient variance to squared gradient norm. McCandlish et al. (2018) derived it as a predictor of the **critical batch size** $B_\text{crit}$ — the batch size past which doubling the batch stops halving the step count. The proposal: measure GNS cheaply during training and use it to set batch size, detect when a run has entered a noise-dominated regime, or schedule batch-size ramps.

Three variants, of different difficulty:

- **Measurement.** Given a run, produce an estimate $\hat{B}_\text{simple}$ with a stated error bar and a stated invariance class. *Open in the strict sense: no standard estimator, no published error bars, no agreed treatment of preconditioning.*
- **Method.** Given $\hat{B}_\text{simple}$ at step $t$, choose a batch size that provably lands within a factor $c$ of the compute-optimal one. *Requires the constant $c$ to be universal; it is not known to be.*
- **Theory.** Prove that $B_\text{simple}$ (the Hessian-free approximation actually measured) bounds $B_\text{crit}$ for a non-quadratic loss under a preconditioned optimizer. *Theoretically open.*

Solving it means: a GNS estimator whose value at step $t$ predicts, within a stated factor, the measured knee of a batch-size sweep on a held-out run — across at least two optimizers and two architectures, without per-setting refitting.

## 2. Formal Setting

Parameters $\theta \in \mathbb{R}^d$, per-example loss $\ell(\theta, x)$, true gradient $G = \mathbb{E}_x[\nabla \ell]$, per-example covariance $\Sigma = \operatorname{Cov}_x(\nabla \ell)$. A batch of size $B$ gives $G_B$ with $\mathbb{E}[G_B] = G$, $\operatorname{Cov}(G_B) = \Sigma/B$.

Second-order expansion of the expected loss decrease under step size $\epsilon$:

$$\mathbb{E}[\Delta L] = -\epsilon |G|^2 + \tfrac{1}{2}\epsilon^2\left(G^\top H G + \frac{\operatorname{tr}(H\Sigma)}{B}\right)$$

Optimising over $\epsilon$ gives the exact noise scale

$$B_\text{noise} = \frac{\operatorname{tr}(H\Sigma)}{G^\top H G},$$

and the measurable surrogate under $H \approx cI$:

$$B_\text{simple} = \frac{\operatorname{tr}(\Sigma)}{|G|^2}.$$

**How it is measured.** Two batch sizes $B_\text{small} < B_\text{big}$ (usually per-device vs. global batch). Unbiased estimators:

$$|\hat G|^2 = \frac{B_\text{big}|G_{B_\text{big}}|^2 - B_\text{small}|G_{B_\text{small}}|^2}{B_\text{big} - B_\text{small}}, \qquad \hat S = \frac{|G_{B_\text{small}}|^2 - |G_{B_\text{big}}|^2}{1/B_\text{small} - 1/B_\text{big}}$$

$\hat B_\text{simple} = \hat S / |\hat G|^2$ is a **ratio of two unbiased estimates and is therefore biased**; both are averaged over an EMA window in practice.

The predicted step/example tradeoff is a hyperbola: $(S/S_\text{min} - 1)(E/E_\text{min} - 1) = 1$, with $B_\text{crit} = E_\text{min}/S_\text{min}$.

**Assumptions, and their status:**

| Assumption | Status |
|---|---|
| $H \approx cI$ (justifies $B_\text{simple}$ for $B_\text{noise}$) | **Violated.** Transformer Hessians have wide spectra and large outlier eigenvalues. |
| Gradient noise has finite variance | **Contested.** Şimşekli et al. (ICML 2019) report heavy tails; Zhang et al. (NeurIPS 2020) report heavy-tailed noise in attention layers. |
| Optimizer is plain SGD | **Violated.** Real runs use Adam/AdamW; $\Sigma$ must be measured in the preconditioned metric $P^{-1}$, and $P$ itself depends on the noise. |
| Batch samples are IID | **Violated.** Sequence packing, curriculum ordering, and data-parallel sharding correlate microbatches. |
| Loss curvature is locally constant over the EMA window | **Violated during warmup and LR decay**, precisely where the diagnostic is most used. |

Note that $B_\text{simple}$ is not invariant to loss scaling in fp16, to weight decay contributions to the gradient, or to gradient clipping.

## 3. State of the Art

**Established (reproduced, ablated):**

- McCandlish, Kaplan, Amodei & the OpenAI Dota Team, *An Empirical Model of Large-Batch Training* (arXiv:1812.06162, 2018). The hyperbolic step/example tradeoff fits observed batch-size sweeps well across supervised and RL tasks. Established: the *functional form*. Not established: that $B_\text{simple}$ predicts $B_\text{crit}$ across settings without a fitted constant.
- Shallue, Lee, Antognini, Sohl-Dickstein, Frostig & Dahl, *Measuring the Effects of Data Parallelism on Neural Network Training* (JMLR 2019). ~100k training runs. Finds the perfect-scaling → diminishing-returns → saturation shape universally, and that the transition point varies by **orders of magnitude** with model, optimizer and dataset. Explicitly reports that simple gradient-noise metrics did **not** reliably predict where the transition falls.
- Zhang, Li, Nado, Martens, Sachdeva, Dahl, Shallue & Grosse, *Which Algorithmic Choices Matter at Which Batch Sizes?* (NeurIPS 2019). A noisy quadratic model reproduces the qualitative ordering of critical batch sizes across optimizers — momentum and preconditioning raise $B_\text{crit}$.

**Claimed but unablated / benchmark-number only:**

- Kaplan et al. (2020) fit $B_\text{crit}(L) \approx 2\times10^8 \text{ tokens} / L^{4.76}$ for LM loss $L$ in nats/token. This is a *fit to a family of runs*, not a validated per-run diagnostic; the exponent has not been independently reproduced at other tokenizers or data mixes.
- Per-device GNS logging in production training stacks is widely used to justify batch-size ramps. No public ablation shows the ramp beat a fixed batch size at matched compute.
- Hilton, Cobbe & Schulman, *Batch Size-Invariance for Policy Optimization* (NeurIPS 2022) gives an RL-side mechanism making PPO behaviour batch-size invariant — evidence that the batch-size knee is partly an *algorithmic* artifact, not a pure property of the noise.

## 4. What Is Known

- **Ranges.** McCandlish et al. report $B_\text{simple}$ spanning roughly $10^2$ (MNIST/SVHN classification) to $10^3$–$10^4$ (ImageNet, Billion Word LM) to $\sim10^6$–$10^7$ observations for Dota 1v1 late in training. OpenAI Five ran at batch sizes near $10^6$ observations, consistent with the high end.
- **GNS grows during training**, typically by 1–2 orders of magnitude from init to convergence, because $|G|^2$ decays faster than $\operatorname{tr}(\Sigma)$. Reproduced in the original paper and in LM settings.
- **Sweep-measured $B_\text{crit}$ varies over ~3 orders of magnitude** across the 6 model families / 7 datasets in Shallue et al. (2019), with batch sizes up to $2^{16}$.
- **Optimizer changes $B_\text{crit}$ at fixed data.** Zhang et al. (2019): preconditioned methods extend perfect scaling to larger batches than plain SGD on the same task.
- **Modern LM result:** Zhang, Morwani, Vyas, Wu, Zou, Ghai, Foster & Kakade, *How Does Critical Batch Size Scale in Pre-training?* (arXiv:2410.21676; ICLR 2025), across models roughly $85$M–$1.2$B, find $B_\text{crit}$ scales primarily with **data size**, not model size — a dependence $B_\text{simple}$ as usually logged does not obviously encode.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no standard GNS estimator. Choice of $(B_\text{small}, B_\text{big})$, EMA half-life, whether to measure in the Adam-preconditioned metric, whether to include weight decay, and whether clipping is applied before measurement each move the reported number by factors of 2–10. No published work reports $\hat B_\text{simple}$ with a confidence interval. Until the estimator is pinned, "GNS predicted $B_\text{crit}$ to within 2×" is not a falsifiable claim.
- **Theoretically open.** No bound relating $B_\text{simple}$ to $B_\text{noise}$ for realistic Hessian spectra; no extension of the derivation to Adam, where the preconditioner is a function of the same noise being measured. Whether $B_\text{crit}$ is even well-defined under heavy-tailed noise (infinite $\operatorname{tr}\Sigma$) is unresolved.
- **Empirically open.** Nobody has published a paired experiment: log GNS continuously through a run *and* perform a full batch-size sweep on the same recipe at $\geq$1B parameters, then report the ratio $B_\text{crit}/\hat B_\text{simple}$ across settings. The compute exists at several labs; the result does not exist publicly.

## 6. Why It Is Hard

**The specific obstruction is estimator conditioning, compounded by non-identifiability.**

$|\hat G|^2$ is a difference of two large, nearly equal quantities scaled by $B_\text{big}/(B_\text{big}-B_\text{small})$. When $B_\text{simple} \gg B_\text{big}$ — exactly the regime the diagnostic is consulted in — the true signal $|G|^2$ is a small residual of the measured $|G_{B_\text{big}}|^2$, so a few-percent error in the large batch norm propagates to a several-fold error in the ratio (§10). The estimator is worst-conditioned where it matters most.

The non-identifiability: $B_\text{crit}$ measured by a sweep depends on the learning-rate schedule tuned at each batch size. A sweep with per-batch-size LR retuning and one with a $\sqrt{B}$ heuristic yield different knees on the same data. So the target the diagnostic is validated against is itself a function of tuning effort — there is no ground truth $B_\text{crit}$, only a $B_\text{crit}$-under-a-tuning-protocol.

## 7. Current Research (as of 2026)

- **Critical-batch-size scaling laws for LMs.** Kakade's group (Harvard) and collaborators, following arXiv:2410.21676, on how $B_\text{crit}$ depends on tokens vs. parameters and on how much of it is explained by LR/warmup interaction.
- **Preconditioner-aware noise scales.** Reformulating GNS in the metric induced by Adam's second-moment estimate, so the measured quantity matches the optimizer actually used *(frontier — verify)*.
- **Batch-size-invariant algorithms.** Extending the Hilton et al. (2022) line: if the algorithm is made invariant, the diagnostic becomes moot for that algorithm.
- **Muon/second-order optimizers.** Whether orthogonalized-momentum updates shift $B_\text{crit}$ enough to break existing GNS-fitted constants *(frontier — verify)*.
- **Production ramp schedules.** Batch-size warmup is standard in large open LM training runs; the justification is usually cited to McCandlish et al. rather than measured in-run.

## 8. Concrete Next Experiment

**Scale.** A 1.4B-parameter decoder-only LM, ~30B tokens, AdamW, fixed data order and fixed cosine schedule shape.

**Arms.**
1. **Sweep arm (ground truth):** train at global batch $\in \{0.25, 0.5, 1, 2, 4, 8\}$M tokens, with peak LR independently tuned at each batch size over a 5-point grid. Fit the hyperbola $(S/S_\text{min}-1)(E/E_\text{min}-1)=1$ to steps-to-target-loss; read off $B_\text{crit}$ at three target losses.
2. **Diagnostic arm:** on the 1M-token run only, log $\hat B_\text{simple}$ every 50 steps under **six** estimator configurations: raw vs. Adam-preconditioned gradient × EMA half-life $\in \{100, 1000, 5000\}$ steps. Bootstrap a 90% CI over the EMA window.
3. **Control arm:** a batch-size-agnostic baseline — predict $B_\text{crit}$ from current loss alone via the Kaplan power law $B \propto L^{-4.76}$, refit only on the first target loss.

**Deciding number.** The spread of $\log_2\!\big(B_\text{crit}/\hat B_\text{simple}\big)$ across the three target losses and six estimator configurations. If that spread is $\leq 1$ bit (all ratios within a factor of 2 of a single constant), GNS is a usable diagnostic and the constant should be published. If the spread exceeds 2 bits, or exceeds the spread of the loss-only control, GNS as currently measured carries no information beyond the loss, and the problem is confirmed methodologically blocked. Cost: ~30 runs, dominated by the LR grid; roughly $10^{21}$–$10^{22}$ FLOPs total.

## 9. Key References

- **[Foundational]** McCandlish, Kaplan, Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch Training.* arXiv, 2018. — arXiv:1812.06162
- **[Foundational]** Shallue, Lee, Antognini, Sohl-Dickstein, Frostig, Dahl. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019. — arXiv:1811.03600
- **[SOTA]** Zhang, Morwani, Vyas, Wu, Zou, Ghai, Foster, Kakade. *How Does Critical Batch Size Scale in Pre-training?* ICLR, 2025. — arXiv:2410.21676
- **[SOTA]** Zhang, Li, Nado, Martens, Sachdeva, Dahl, Shallue, Grosse. *Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model.* NeurIPS, 2019.
- **[Related]** Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei. *Scaling Laws for Neural Language Models.* arXiv, 2020. — arXiv:2001.08361
- **[Related]** Şimşekli, Sagun, Gürbüzbalaban. *A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks.* ICML, 2019.
- **[Related]** Zhang, Karimireddy, Veit, Kim, Reddi, Kumar, Sra. *Why are Adaptive Methods Good for Attention Models?* NeurIPS, 2020.
- **[Related]** Hilton, Cobbe, Schulman. *Batch Size-Invariance for Policy Optimization.* NeurIPS, 2022.
- **[Related]** Goyal, Dollár, Girshick, Noordhuis, Wesolowski, Kyrola, Tulloch, Jia, He. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv, 2017. — arXiv:1706.02677

## 10. Worked Example

A run logs gradient norms at $B_\text{small}=32$ and $B_\text{big}=512$ sequences. Observed squared norms, EMA-averaged:

$$|G_{32}|^2 = 4.100\times10^{-2}, \qquad |G_{512}|^2 = 2.800\times10^{-3}$$

Signal:
$$|\hat G|^2 = \frac{512(2.800\times10^{-3}) - 32(4.100\times10^{-2})}{480} = \frac{1.4336 - 1.3120}{480} = 2.533\times10^{-4}$$

Noise:
$$\hat S = \frac{4.100\times10^{-2} - 2.800\times10^{-3}}{1/32 - 1/512} = \frac{3.820\times10^{-2}}{2.9297\times10^{-2}} = 1.304$$

$$\hat B_\text{simple} = 1.304 / 2.533\times10^{-4} \approx 5{,}150 \text{ sequences}$$

The engineer reads "critical batch size ≈ 5k sequences" and proposes a ramp to 4k.

**Now perturb $|G_{512}|^2$ by $\pm 5\%$** — well inside EMA sampling error, and smaller than the shift caused by turning gradient clipping on:

| $|G_{512}|^2$ | $|\hat G|^2$ | $\hat B_\text{simple}$ |
|---|---|---|
| $2.66\times10^{-3}$ (−5%) | $1.040\times10^{-4}$ | **12,580** |
| $2.80\times10^{-3}$ | $2.533\times10^{-4}$ | **5,150** |
| $2.94\times10^{-3}$ (+5%) | $4.027\times10^{-4}$ | **3,230** |

A 5% measurement wobble moves the answer across a **4× range**. The cause is visible in the middle column: $|\hat G|^2$ is $0.12$ of a difference between $1.43$ and $1.31$ — a 9% residual, so relative error is amplified about 11×. The amplification factor is roughly $B_\text{simple}/B_\text{big}$, so it grows exactly as the diagnostic reports larger, more consequential numbers.

Before any question about whether $B_\text{simple}$ tracks $B_\text{noise}$ under a non-isotropic Hessian, or whether Adam's preconditioner invalidates the derivation, the estimator has already lost the decision. That is the blockage: the measurement is not defined tightly enough for the theory question to be reachable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*