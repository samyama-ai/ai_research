---
id: 03-training-dynamics/critical-batch-size-prediction
title: "Critical Batch Size Prediction"
topic: 03-training-dynamics
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Critical Batch Size Prediction

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/critical-batch-size-prediction` · **Status:** empirically-open

## 1. Problem Statement

The critical batch size $B_{\mathrm{crit}}$ is the batch size beyond which increasing data parallelism stops buying a proportional reduction in optimization steps. Below it, doubling the batch roughly halves the steps to a target loss; above it, steps stop falling and the extra tokens per step are wasted compute.

**Input.** A training recipe: architecture and parameter count $N$, token budget $D$, optimizer and its tuning protocol, data distribution, and a target loss $L^\star$.

**Output.** A prediction $\hat{B}_{\mathrm{crit}}$ — a token count — made *before* the run, from cheap proxy runs or from a closed-form law.

**Decision predicate.** The prediction is useful if, at the target scale, the realized time-to-loss at $\hat{B}_{\mathrm{crit}}$ is within a stated tolerance (say 10%) of the best achievable over a batch-size sweep, and the compute spent producing $\hat{B}_{\mathrm{crit}}$ is $\ll$ the cost of that sweep.

Three variants, of very different difficulty:

- **Measurement.** Given unlimited compute, is $B_{\mathrm{crit}}$ even a well-defined property of a (model, data, optimizer) triple? It is defined only relative to a tuning protocol and a target loss, and both are choices.
- **Method.** Predict $B_{\mathrm{crit}}$ at scale $10^{24}$ FLOPs from runs at $10^{20}$ FLOPs. Currently done by fitting power laws; extrapolation error is unbounded.
- **Theory.** Derive $B_{\mathrm{crit}}$ from properties of the loss surface — gradient covariance, Hessian spectrum, noise structure. This is where the gradient-noise-scale program lives, and where it has repeatedly failed to match measurement.

## 2. Formal Setting

Let $\theta \in \mathbb{R}^N$, population loss $L(\theta) = \mathbb{E}_{x\sim\mathcal{D}}[\ell(\theta,x)]$, minibatch gradient $\hat{g}_B = \frac{1}{B}\sum_{i=1}^{B}\nabla\ell(\theta,x_i)$, true gradient $G = \nabla L(\theta)$, per-example gradient covariance $\Sigma(\theta) = \operatorname{Cov}_{x}[\nabla \ell(\theta,x)]$, Hessian $H(\theta)$.

**Steps-to-target.** $S(B) = \min\{ t : L(\theta_t) \le L^\star \}$ under the best hyperparameters found by a fixed tuning protocol $\mathcal{T}$ at that $B$. Examples-to-target $E(B) = B\,S(B)$. Measured by running training and reading off the first step crossing $L^\star$ on a held-out set — noisy, so in practice one fits a smoothed loss curve.

**Pareto definition (McCandlish et al., 2018).** Empirically $S(B)$ and $E(B)$ trace a hyperbola
$$\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1, \qquad B_{\mathrm{crit}} = \frac{E_{\min}}{S_{\min}},$$
so $B_{\mathrm{crit}}$ is the batch size at which $S = 2S_{\min}$ and $E = 2E_{\min}$ — a 2× penalty in both currencies. Measured by sweeping $B$ over a geometric grid, retuning at each point, and fitting two constants.

**Noise-scale definition.** For a quadratic model with a small learning rate,
$$B_{\mathrm{noise}} = \frac{\operatorname{tr}(H\Sigma)}{G^\top H G}, \qquad B_{\mathrm{simple}} = \frac{\operatorname{tr}(\Sigma)}{\lVert G\rVert^2}.$$
$B_{\mathrm{simple}}$ is measured cheaply from the variance of gradients across data-parallel shards: with per-shard gradients at sizes $B_{\text{small}}, B_{\text{big}}$, unbiased estimates of $\lVert G\rVert^2$ and $\operatorname{tr}(\Sigma)$ follow from the two squared norms. $B_{\mathrm{noise}}$ needs Hessian-vector products and is rarely measured.

**Scaling-law form.** Kaplan et al. (2020) posit $B_{\mathrm{crit}}(L) = B_\star / L^{1/\alpha_B}$, a function of loss alone.

**Assumptions, and which are violated.**
1. *Local quadratic loss, fixed $\Sigma$ over the step.* Violated — $\Sigma$ and $H$ change by orders of magnitude across training, and $B_{\mathrm{simple}}$ typically grows monotonically during a run.
2. *Optimal learning rate found at every $B$.* Violated in most published sweeps; tuning budgets are capped, and under-tuning at large $B$ manufactures a spurious $B_{\mathrm{crit}}$.
3. *$B_{\mathrm{crit}}$ depends only on loss.* Contested — later work finds dependence on data budget $D$ and on weight decay.
4. *SGD-like analysis transfers to Adam.* Violated — Adam's preconditioner changes the noise geometry, and $B_{\mathrm{noise}}$ was derived for the unpreconditioned case.
5. *Single-epoch, i.i.d. data.* Violated in multi-epoch and curriculum regimes.

## 3. State of the Art

**Theory SOTA.** McCandlish, Kaplan, Amodei et al., *An Empirical Model of Large-Batch Training* (2018, arXiv:1812.06162) — the hyperbolic Pareto model plus the $B_{\mathrm{noise}}$ derivation. Established: the hyperbola fits well within a workload. Claimed but not established: that $B_{\mathrm{simple}}$ *predicts* $B_{\mathrm{crit}}$ across workloads; the paper reports order-of-magnitude agreement, not calibrated prediction.

**Empirical SOTA.** Shallue, Lee, Antognini, Sohl-Dickstein, Frostig, Dahl, *Measuring the Effects of Data Parallelism on Neural Network Training* (JMLR 2019) — batch sizes from 2 to $2^{16}$ across six model families, with per-batch-size hyperparameter search. Established: universal three-phase shape (perfect scaling → diminishing returns → saturation) and that the transition point varies by model, optimizer and dataset over more than an order of magnitude. Established negative: no tested simple statistic, including gradient noise scale, predicted the transition across workloads. Zhang, Li, Nado et al., *Which Algorithmic Choices Matter at Which Batch Sizes?* (NeurIPS 2019) show the noise-scale model tracks behavior once the optimizer's curvature handling is matched, and that momentum extends the scaling regime.

**LLM-scale SOTA.** Kaplan et al. (2020) report $B_\star \approx 2\times10^8$ tokens and $\alpha_B \approx 0.21$ for language modeling — a benchmark fit on one lab's data, never independently reproduced at the same scale. DeepSeek-AI, *DeepSeek LLM* (2024, arXiv:2401.02954), fits batch size directly against compute budget, $B_{\mathrm{opt}} \propto C^{\approx 0.33}$ with $\eta_{\mathrm{opt}} \propto C^{-\approx 0.12}$ — a benchmark number from an internal sweep, with no ablation isolating batch from schedule. Zhang, Morwani, Vyas, Brandfonbrener, Kakade et al., *How Does Critical Batch Size Scale in Pre-training?* (arXiv:2410.21676, 2024) run controlled sweeps and report that CBS is governed mainly by **data size**, not model size — the most direct challenge to the loss-only form. Bergsma et al., *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training* (2025) tie the optimal batch size to the AdamW timescale $B/(\eta\lambda)$, implying $B_{\mathrm{crit}}$ is not identifiable without fixing weight decay.

## 4. What Is Known

- **Three-phase curve is real.** Reproduced across MLPs, CNNs, transformers, and LMs; Shallue et al. observed it in all six workloads, batch 2–65,536.
- **The transition is workload-specific.** In Shallue et al., the perfect-scaling limit ranged from roughly $10^2$ (simple CNN on MNIST) to $>10^4$ examples (ResNet-50/ImageNet, Transformer/LM1B) — a two-orders-of-magnitude spread not predicted by any single statistic.
- **$B_{\mathrm{crit}}$ grows during training.** $B_{\mathrm{simple}}$ rises by 10–100× from start to end of a run (McCandlish et al., across Atari, ImageNet, and LM tasks), which is why fixed-batch runs are inefficient early and adequate late.
- **Learning rate must co-scale.** Linear-scaling $\eta \propto B$ holds in the small-batch regime (Goyal et al., 2017, ImageNet in 1 hour, batch 8k); square-root scaling matches better under Adam in several LLM sweeps. Failure to retune converts a tuning artifact into a false $B_{\mathrm{crit}}$.
- **Loss-only form is at best incomplete.** Zhang et al. (2024) find data budget $D$ dominates model size $N$ in setting CBS across models from $\sim$85M to $\sim$1.2B parameters, contradicting the pure $B_{\mathrm{crit}}(L)$ parameterization.
- **Equivalence of schedule and batch.** Smith et al., *Don't Decay the Learning Rate, Increase the Batch Size* (ICLR 2018) — batch increase and LR decay produce near-identical curves below $B_{\mathrm{crit}}$, verified on ResNet/CIFAR and ImageNet at batch up to 65k.

## 5. What Is Not Known

- **Theoretically open.** No proof that $B_{\mathrm{noise}}$ equals $B_{\mathrm{crit}}$ for any non-quadratic loss, and no theory of $B_{\mathrm{crit}}$ under adaptive preconditioning. There is no derivation of the exponent in $B_{\mathrm{crit}} \propto D^{\beta}$ from first principles; $\beta$ is a fitted number.
- **Empirically open.** Whether any power law fit below $10^{21}$ FLOPs extrapolates to $10^{25}$ FLOPs. The frontier-scale sweep — a full batch-size ladder with retuned LR and weight decay at $\ge$70B parameters — has been run inside labs but not published with the arm-by-arm losses needed to check it.
- **Methodologically blocked.** $B_{\mathrm{crit}}$ is not well defined without fixing (a) the tuning protocol, (b) the target loss, and (c) the AdamW timescale $B/(\eta\lambda)$. Different published values are not commensurable because these three are not held fixed. Until a reference protocol exists, cross-paper comparison of $B_{\mathrm{crit}}$ numbers is not meaningful.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**, not compute alone.

Measuring $S(B)$ requires the optimal $(\eta, \lambda, \text{warmup}, \beta_2)$ at each $B$. Under-tuning at large $B$ inflates $S(B)$ and moves the fitted knee left; over-tuning at small $B$ does the same. Since tuning cost grows with $B$ (each trial is more expensive), the tuning budget is systematically anticorrelated with $B$ — the bias always points the same way, toward a *smaller* apparent $B_{\mathrm{crit}}$.

Non-identifiability: Bergsma et al.'s timescale result means $(B, \eta, \lambda)$ enter through fewer effective degrees of freedom than three. A "batch size effect" measured at fixed $\lambda$ is partly a weight-decay effect. Two labs reporting different $B_{\mathrm{crit}}$ may be measuring the same surface at different points on the same ridge.

Finally, the standard evaluation does not measure the named quantity: papers report the batch size minimizing *steps at fixed token budget*, which is a wall-clock optimum under a specific cluster topology, not the Pareto knee $E_{\min}/S_{\min}$.

## 7. Current Research (as of 2026)

- **Data-size-driven CBS laws.** Kempner Institute / Harvard (Kakade, Morwani, Brandfonbrener, Vyas, Zhang) — controlled ladders establishing $D$-dependence and testing invariance to model shape.
- **Hyperparameter-transfer unification.** Extending $\mu$P (Yang & Hu, 2021; Yang et al., *Tensor Programs V*, 2022) to jointly transfer batch size, LR and weight decay — the "Power Lines" line at Cerebras and related groups. *(frontier — verify)* Whether $\mu$P-style width transfer holds for $B_{\mathrm{crit}}$ is unresolved; width transfer of $\eta$ does not imply batch transfer.
- **Preconditioner-aware noise scales.** Second-order and Shampoo/SOAP-adjacent work asks whether $B_{\mathrm{crit}}$ moves right under better preconditioning; small-scale results say yes, frontier-scale confirmation absent. *(frontier — verify)*
- **Systems-side pressure.** Frontier runs use batches of $10^7$–$10^8$ tokens with batch-size ramps; whether the ramp schedule is near-optimal or merely convenient is undocumented publicly. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does a CBS law fitted at $10^{19}$–$10^{20}$ FLOPs predict the measured knee at $10^{22}$ FLOPs to within 2×?

**Scale.** Decoder-only transformers, Chinchilla-ratio token budgets, at four compute points: $10^{19}$, $10^{20}$, $10^{21}$, $10^{22}$ FLOPs (roughly 90M → 3B parameters). At each compute point, a batch ladder of seven points spanning $2^{16}$ to $2^{22}$ tokens.

**Protocol (removes the confound).** At every $(C, B)$ cell, tune $\eta$ and $\lambda$ jointly on a fixed 12-point grid in $(\eta, B/(\eta\lambda))$ — the same *number* of trials at every $B$, so the tuning budget is not anticorrelated with batch size. Fixed cosine schedule, fixed warmup fraction, fixed data order seed set of 3.

**Control arm.** The same ladder at $10^{22}$ FLOPs, measured directly. Fit the law on the three cheap points only; the $10^{22}$ ladder is held out.

**The deciding number.** $\rho = \hat{B}_{\mathrm{crit}}(10^{22}) / B_{\mathrm{crit}}^{\text{measured}}(10^{22})$, with $B_{\mathrm{crit}}$ defined as the Pareto knee $E_{\min}/S_{\min}$ at target loss $L^\star$ = the best loss achievable at the smallest batch. **$\rho \in [0.5, 2]$ across all three seeds means extrapolation works and the problem moves from empirically-open to partially-solved. $\rho$ outside that band, or seed spread exceeding 1.5×, means the published laws are fits, not predictions.** Cost: about $1.3\times10^{22}$ FLOPs total, roughly one week on 256 H100s.

## 9. Key References

- **[Foundational]** Sam McCandlish, Jared Kaplan, Dario Amodei, et al. *An Empirical Model of Large-Batch Training.* arXiv preprint, 2018. — arXiv:1812.06162
- **[Foundational]** Christopher J. Shallue, Jaehoon Lee, Joseph Antognini, Jascha Sohl-Dickstein, Roy Frostig, George E. Dahl. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019. — arXiv:1811.03600
- **[Foundational]** Priya Goyal, Piotr Dollár, Ross Girshick, et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv preprint, 2017. — arXiv:1706.02677
- **[Foundational]** Samuel L. Smith, Pieter-Jan Kindermans, Chris Ying, Quoc V. Le. *Don't Decay the Learning Rate, Increase the Batch Size.* ICLR 2018. — arXiv:1711.00489
- **[SOTA]** Hanlin Zhang, Depen Morwani, Nikhil Vyas, David Brandfonbrener, Sham Kakade, et al. *How Does Critical Batch Size Scale in Pre-training?* ICLR 2025. — arXiv:2410.21676
- **[SOTA]** Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, Joel Hestness. *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training.* 2025.
- **[SOTA]** Guodong Zhang, Lala Li, Zachary Nado, James Martens, et al. *Which Algorithmic Choices Matter at Which Batch Sizes? Insights from a Noisy Quadratic Model.* NeurIPS 2019. — arXiv:1907.04164
- **[Context]** Jared Kaplan, Sam McCandlish, Tom Henighan, et al. *Scaling Laws for Neural Language Models.* arXiv preprint, 2020. — arXiv:2001.08361
- **[Context]** DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* 2024. — arXiv:2401.02954
- **[Context]** Greg Yang, Edward J. Hu, Igor Babuschkin, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS 2021. — arXiv:2203.03466

## 10. Worked Example

Take Kaplan et al.'s form $B_{\mathrm{crit}}(L) = B_\star L^{-1/\alpha_B}$ with $B_\star = 2\times10^8$ tokens, $\alpha_B = 0.21$, so $1/\alpha_B \approx 4.76$.

At $L = 3.0$ nats/token:
$$B_{\mathrm{crit}} = 2\times10^{8}\cdot 3.0^{-4.76} = 2\times10^{8} / e^{4.76\ln 3} = 2\times10^{8}/196 \approx 1.0\times10^{6}\ \text{tokens}.$$

At $L = 2.2$ nats/token:
$$B_{\mathrm{crit}} = 2\times10^{8}/2.2^{4.76} = 2\times10^{8}/38.6 \approx 5.2\times10^{6}\ \text{tokens}.$$

At $L = 1.8$:
$$B_{\mathrm{crit}} = 2\times10^{8}/1.8^{4.76} = 2\times10^{8}/16.1 \approx 1.2\times10^{7}\ \text{tokens}.$$

**Where the obstruction becomes visible.** The exponent 4.76 makes the prediction violently sensitive to the loss estimate. A 5% error in $L$ — well inside the spread between tokenizers, or between two held-out sets — moves $B_{\mathrm{crit}}$ by $1.05^{4.76} \approx 1.26$, i.e. 26%. A 15% loss difference, which is roughly what separating a BPE-32k from a BPE-100k tokenizer produces in nats/token, moves the prediction by $1.15^{4.76} \approx 1.96$ — a factor of two, which is exactly the tolerance band the whole prediction is supposed to hit.

Now compare against measurement. Frontier runs at $L \approx 1.8$ use batches in the $10^7$–$10^8$ token range. The law's $1.2\times10^7$ sits at the very bottom of that range, and the observed spread across labs is itself an order of magnitude. So the law is neither clearly confirmed nor clearly falsified — and it cannot be, because each lab's reported batch is chosen under a different weight decay, a different LR schedule, and a different cluster topology. The prediction error and the measurement ambiguity are the same size. That is the problem: not that $B_{\mathrm{crit}}$ is expensive to measure, but that nobody has fixed the protocol that would make two measurements of it comparable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*