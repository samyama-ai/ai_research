---
id: 08-loss-and-heads/leakage-from-head-overconfidence
title: "Membership Leakage Attributable to Output Head Overconfidence"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Membership Leakage Attributable to Output Head Overconfidence

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/leakage-from-head-overconfidence` · **Status:** open

## 1. Problem Statement

A trained classifier leaks membership: an adversary given $(x,y)$ and query access can guess better than chance whether $(x,y)$ was in the training set. Empirically, leakage tracks the *confidence gap* — members get near-one softmax probability on the true label, non-members do not. The folk explanation is that the **output head is overconfident**, and that fixing calibration fixes privacy.

The problem is to establish, refute, or make precise the causal claim:

- **Measurement variant.** Define a quantity $L_{\text{head}}$ — the portion of membership leakage removable by reparameterizing the final linear layer and the loss's confidence shaping, holding the representation $\phi$ fixed. Is $L_{\text{head}}$ well defined and estimable?
- **Method variant.** Does any head-only intervention (temperature scaling, logit-norm penalty, label smoothing, confidence masking, entropy regularization) reduce leakage at a *fixed* utility, measured by TPR at low FPR rather than by attack accuracy or AUC?
- **Theory variant.** Is there a separation theorem: a data distribution and training procedure where the representation $\phi$ is $\varepsilon$-membership-private under any label-only adversary, yet the composed model $g\circ\phi$ leaks $\Omega(1)$ through the head alone — or, conversely, a proof that head-only leakage is always removable up to a monotone relabeling?

Solved means: a defensible estimator for $L_{\text{head}}$, plus evidence at ImageNet or LLM scale that head-only fixes either do or do not move worst-case leakage.

## 2. Formal Setting

Dataset $D=\{z_i=(x_i,y_i)\}_{i=1}^n \sim \mathbb{D}^n$. Training algorithm $\mathcal{T}$ produces $f_\theta = g_W \circ \phi_\psi$, with representation $\phi_\psi:\mathcal{X}\to\mathbb{R}^d$ and head $g_W(h)=\mathrm{softmax}\!\left((Wh+b)/T\right)$.

**Leakage, as measured.** A membership adversary $\mathcal{A}$ outputs a score $s(z)\in\mathbb{R}$; the operating curve is the ROC of $s$ over members vs. non-members. The reported quantity is not AUC but
$$\mathrm{TPR}@\alpha \;=\; \Pr_{z\in D}\!\left[s(z) > \tau_\alpha\right], \qquad \Pr_{z\notin D}\!\left[s(z)>\tau_\alpha\right]=\alpha,\quad \alpha \in \{10^{-3},10^{-2}\},$$
estimated by training $N$ shadow models (typically $N=64$–$256$) with each example in half of them (Carlini et al., 2022). The LiRA score is the likelihood ratio
$$s_{\text{LiRA}}(z)=\log\frac{\mathcal{N}\!\left(\ell(z);\mu_{\text{in}},\sigma_{\text{in}}^2\right)}{\mathcal{N}\!\left(\ell(z);\mu_{\text{out}},\sigma_{\text{out}}^2\right)},\qquad \ell(z)=\log\frac{f_\theta(x)_y}{1-f_\theta(x)_y}.$$

**Overconfidence, as measured.** Binned expected calibration error over $M=15$ equal-mass bins,
$$\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\Big|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\Big|,$$
computed on *held-out* data. Companion statistics: mean logit norm $\mathbb{E}\|Wh+b\|_2$, and the member/non-member confidence gap $\Delta = \mathbb{E}_{z\in D}[f_\theta(x)_y]-\mathbb{E}_{z\notin D}[f_\theta(x)_y]$.

**The attributable-leakage quantity.** For a class $\mathcal{G}$ of admissible head reparameterizations (those preserving top-1 accuracy within $\delta$),
$$L_{\text{head}} \;=\; \mathrm{TPR}@\alpha\big(g_W\circ\phi_\psi\big)\;-\;\inf_{g'\in\mathcal{G}}\ \mathrm{TPR}@\alpha\big(g'\circ\phi_\psi\big).$$

**Assumptions, with the violated ones flagged.**
1. Shadow models are drawn from the same $\mathcal{T}$ and $\mathbb{D}$ as the target. *Violated for pretrained LLMs*: the pretraining corpus and recipe are not reproducible, so $\mu_{\text{out}}$ is unestimable.
2. Members and non-members are exchangeable. *Violated in LLM benchmarks* where "non-members" are collected after the cutoff and differ distributionally (Duan et al., 2024).
3. The head is the last linear layer only. *Violated* when the loss shapes intermediate features (label smoothing changes $\phi$, not just $W$; Müller et al., 2019).
4. Gaussianity of $\ell$ under in/out. Approximately true for logit-scaled confidences at CIFAR scale; untested for open-vocabulary next-token losses.

## 3. State of the Art

**Established.**
- Per-example difficulty-calibrated attacks (LiRA, Carlini et al., S&P 2022; Attack-R/Attack-P, Ye et al., CCS 2022; Watson et al., ICLR 2022) dominate global-threshold attacks in the low-FPR regime. Any head fix must be evaluated against these, not against Shokri-style accuracy.
- **Label-only attacks** (Choquette-Choo et al., ICML 2021) reach comparable strength using only the predicted label plus decision-boundary distance. This is the central negative result for the head hypothesis: an intervention that only rescales confidences leaves the label-only attack surface untouched.
- Confidence-vector masking (MemGuard, Jia et al., CCS 2019) was reported to drop attack accuracy to near 50%, then shown to be largely defeated by label-only and calibrated attacks.
- DP-SGD (Abadi et al., CCS 2016) gives a leakage bound that holds against *any* adversary; empirical attacks approach the bound only under strong adversary instantiation (Nasr et al., S&P 2021).

**Claimed but unablated.** That label smoothing, focal loss, logit-norm penalties, or temperature scaling reduce privacy risk *because* they reduce overconfidence. Reported results are typically single-architecture CIFAR numbers at AUC or attack-accuracy granularity, without a fixed-utility control and without a label-only arm. Reports that label smoothing *increases* membership vulnerability exist alongside reports that it decreases it; the sign is architecture- and $\alpha$-dependent in published tables.

**Benchmark-number-only.** Most "calibration improves privacy" entries are single rows in a defense table (one dataset, one attack, one seed), not ablations.

**Defense SOTA that works but is not head-only.** RelaxLoss (Chen, Yu, Fritz, ICLR 2022) targets the *loss distribution* — it stops gradient descent once per-example loss falls below a target, flattening the member/non-member loss gap. SELENA (Tang et al., USENIX Security 2022) uses self-distillation over sub-model ensembles. Both change the training trajectory, not merely the head.

## 4. What Is Known

- **Overfitting is sufficient, not necessary.** Yeom et al. (CSF 2018) prove that for the loss-threshold adversary, advantage is bounded by the generalization gap under bounded loss; but Carlini et al. (2022) show models with small train/test gaps still yield TPR $\approx 8\%$ at FPR $10^{-3}$ on CIFAR-10 (WideResNet, $n=25{,}000$, 64 shadow models), where prior attacks gave $<0.1\%$ — a >100× gap at the same AUC.
- **AUC hides the leak.** Several defenses reduce attack AUC from $\approx0.70$ to $\approx0.55$ on CIFAR-100 while leaving low-FPR TPR nearly unchanged; the metric choice, not the defense, produces the improvement.
- **Calibration is cheap and post-hoc.** Temperature scaling with a single $T$ cuts ECE on CIFAR-100/ResNet-110 from $\approx16.5\%$ to $\approx1$–$2\%$ (Guo et al., ICML 2017) with zero accuracy change — and, being a strictly monotone map on the true-class logit, it leaves the per-example loss *ranking* unchanged.
- **Class-difficulty dominates.** In Ye et al. (CCS 2022) and Watson et al. (ICLR 2022), most of the naive attack's apparent signal is example hardness, removable by per-example calibration; leakage that survives is concentrated on outliers.
- **LLM scale is unsettled.** Duan et al. (COLM 2024) find MIA on pretrained LLMs (Pythia 160M–12B, MIMIR) near chance, AUC $\approx0.5$–$0.6$, and attribute much reported success to member/non-member distribution shift rather than memorization.

## 5. What Is Not Known

- **Methodologically blocked.** $L_{\text{head}}$ has no agreed definition, because $\mathcal{G}$ is unspecified. Restrict $\mathcal{G}$ to monotone rescalings and $L_{\text{head}}\approx 0$ by construction (rank-preserving). Widen $\mathcal{G}$ to arbitrary retrained heads and the representation is no longer held fixed in any meaningful sense. The counterfactual "same $\phi$, non-overconfident head" is not operationally pinned down.
- **Theoretically open.** No separation theorem either way: no construction showing a head that leaks $\Omega(1)$ over a provably non-leaking representation, and no proof that head-only leakage is bounded by a calibration functional of the head.
- **Empirically open.** No study runs LiRA *and* a label-only arm across a calibration sweep (temperature, label-smoothing $\alpha$, logit-norm weight) at fixed test accuracy, at ImageNet-1k scale with $\geq 64$ shadow models. Compute, not method, is the blocker.
- **Open for generative heads.** Whether the same decomposition is even meaningful for a 128k-way tied softmax over sequences is unresolved.

## 6. Why It Is Hard

**Non-identifiability of the head/representation split.** Temperature scaling and any strictly monotone recalibration are rank-preserving on the true-class confidence, so any attack that thresholds a *calibrated per-example* statistic is invariant to them. Confidence-level leakage and representation-level leakage are therefore not separable by observing outputs alone: the same output distribution is produced by "confident head over benign features" and "modest head over memorized features."

**Confounded measurement.** Interventions marketed as head-only are not. Label smoothing changes the penultimate geometry (tighter class clusters — Müller et al., 2019), so its privacy effect cannot be attributed to the head.

**Evaluation that does not measure what it names.** "Attack accuracy" and AUC average over the whole ROC; the privacy harm lives at FPR $\le 10^{-3}$. A defense can move the average while leaving the tail intact.

**Compute.** A credible answer requires shadow-model ensembles: $64\times$ full training runs per configuration, times a calibration sweep of $\sim6$ points, times a control arm. At ImageNet-1k that is $\sim800$ ResNet-50 trainings.

## 7. Current Research (as of 2026)

- **Low-FPR-first evaluation** is now the norm in the Carlini/Tramèr line (Google DeepMind, ETH Zürich) and the Shokri line (NUS) — new defenses without a TPR@0.1%FPR column are discounted.
- **Loss-shaping defenses** (RelaxLoss-style flattening of the per-example loss distribution) remain the strongest non-DP direction at fixed utility; whether they act through the head or through the trajectory is exactly the open question.
- **LLM membership auditing** has partly shifted from MIA to extraction and to canary-based auditing with injected duplicates, after the null results on pretraining MIA. *(frontier — verify)*
- **Privacy auditing in one run** (tight empirical $\varepsilon$ from a single training run, Steinke, Nasr, Jagielski, NeurIPS 2023) offers a route around the shadow-model cost and has not yet been applied to a calibration sweep. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** CIFAR-100, WideResNet-28-2, $n=25{,}000$ members drawn from a 50k pool; $N=128$ shadow models per configuration (about 5 GPU-days per configuration on one A100-class device).

**Arms.** Calibration sweep at *fixed test accuracy within $\pm0.5$ pp*: (i) baseline cross-entropy; (ii) post-hoc temperature scaling $T\in\{1.5,3\}$; (iii) label smoothing $\alpha\in\{0.05,0.1\}$; (iv) logit-norm penalty $\lambda\|Wh+b\|_2^2$ tuned to match arm (iii)'s ECE.

**Control arm (the point of the design).** A *frozen-representation* arm: take the baseline $\phi_\psi$, discard $W$, retrain only a fresh head on the same members with each calibration objective. This isolates head effects from trajectory effects. Any configuration must be scored by **both** LiRA and the label-only boundary-distance attack.

**The deciding number.** $\Delta \mathrm{TPR}@\mathrm{FPR}=10^{-3}$ between baseline and the best calibrated head in the frozen-representation arm, under the label-only attack. If it is $<1$ pp absolute while ECE falls by $\ge 10$ pp, the head-overconfidence hypothesis is refuted as a *causal* account: overconfidence is a symptom. If it exceeds 3 pp, head design is a real privacy lever and $\mathcal{G}$ can be defined around it.

## 9. Key References

- **[Foundational]** R. Shokri, M. Stronati, C. Song, V. Shmatikov. *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P, 2017. — arXiv:1610.05820
- **[Foundational]** S. Yeom, I. Giacomelli, M. Fredrikson, S. Jha. *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE CSF, 2018. — arXiv:1709.01604
- **[SOTA]** N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramèr. *Membership Inference Attacks From First Principles.* IEEE S&P, 2022. — arXiv:2112.03570
- **[SOTA]** J. Ye, A. Maddi, S. K. Sankararaman, R. Shokri et al. *Enhanced Membership Inference Attacks against Machine Learning Models.* ACM CCS, 2022. — arXiv:2111.09679
- **[SOTA]** L. Watson, C. Guo, G. Cormode, A. Sablayrolles. *On the Importance of Difficulty Calibration in Membership Inference Attacks.* ICLR, 2022. — arXiv:2111.08440
- C. A. Choquette-Choo, F. Tramèr, N. Carlini, N. Papernot. *Label-Only Membership Inference Attacks.* ICML, 2021. — arXiv:2007.14321
- C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML, 2017. — arXiv:1706.04599
- R. Müller, S. Kornblith, G. Hinton. *When Does Label Smoothing Help?* NeurIPS, 2019. — arXiv:1906.02629
- D. Chen, N. Yu, M. Fritz. *RelaxLoss: Defending Membership Inference Attacks without Losing Utility.* ICLR, 2022. — arXiv:2207.05801
- X. Tang, S. Mahloujifar, L. Song, V. Sehwag, R. Shokri, P. Mittal. *Mitigating Membership Inference Attacks by Self-Distillation Through a Novel Ensemble Architecture.* USENIX Security, 2022. — arXiv:2110.08324
- J. Jia, A. Salem, M. Backes, Y. Zhang, N. Z. Gong. *MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples.* ACM CCS, 2019.
- **[Survey]** L. Song, P. Mittal. *Systematic Evaluation of Privacy Risks of Machine Learning Models.* USENIX Security, 2021. — arXiv:2003.10595
- M. Duan, A. Suri, N. Mireshghallah, S. Min, W. Shi, L. Zettlemoyer, Y. Tsvetkov, Y. Choi, D. Evans, H. Hajishirzi. *Do Membership Inference Attacks Work on Large Language Models?* COLM, 2024. — arXiv:2402.07841
- M. Abadi et al. *Deep Learning with Differential Privacy.* ACM CCS, 2016. — arXiv:1607.00133
- T. Steinke, M. Nasr, M. Jagielski. *Privacy Auditing with One (1) Training Run.* NeurIPS, 2023. — arXiv:2305.08846

## 10. Worked Example

Take a CIFAR-100 ResNet-110 of the kind measured by Guo et al.: test accuracy $\approx 71.5\%$, ECE $\approx 16.5\%$ before temperature scaling and $\approx 1.3\%$ after, with $T^\star \approx 2.5$. Suppose the attacker uses the logit-scaled confidence $\ell(z)=\log\frac{p_y}{1-p_y}$.

Post-hoc temperature scaling replaces logits $u$ by $u/T$. For the true class,
$$p_y(T)=\frac{e^{u_y/T}}{\sum_k e^{u_k/T}},$$
which is strictly increasing in $u_y$ for fixed $T>0$ and fixed competitors. So the induced ordering of examples by $\ell$ is preserved almost everywhere. An ROC depends only on the ordering of scores. Therefore:

- Global-threshold attack: **ROC unchanged**, TPR@FPR$=10^{-3}$ unchanged, to numerical precision.
- LiRA: the in/out Gaussians $\mathcal{N}(\mu_{\text{in}},\sigma^2_{\text{in}})$, $\mathcal{N}(\mu_{\text{out}},\sigma^2_{\text{out}})$ both shift and shrink by roughly the same factor; the likelihood ratio is near-invariant. Concretely, if $\mu_{\text{in}}-\mu_{\text{out}}=2.0$ and $\sigma=1.0$ before scaling, both scale by $\approx 1/2.5$, leaving $d'=(\mu_{\text{in}}-\mu_{\text{out}})/\sigma \approx 2.0$ intact.
- Label-only attack: **exactly unchanged** — $\arg\max_k u_k/T = \arg\max_k u_k$.

So a 13-point ECE reduction buys $\approx 0$ points of TPR at low FPR. The obstruction is visible: the metric that names the hypothesis (ECE) and the metric that names the harm (low-FPR TPR) are decoupled by a transformation that moves one and provably cannot move the other. Any real head-based defense must therefore be *non-monotone* in the per-example loss — it must reorder members relative to non-members, which is what loss-flattening methods like RelaxLoss do and what recalibration by construction does not. Whether such reordering is achievable without touching $\phi$ is the open question.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*