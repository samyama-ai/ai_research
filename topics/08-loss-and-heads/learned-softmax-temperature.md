---
id: 08-loss-and-heads/learned-softmax-temperature
title: "Softmax Temperature as a Learned Parameter"
topic: 08-loss-and-heads
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Softmax Temperature as a Learned Parameter

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/learned-softmax-temperature` · **Status:** open

## 1. Problem Statement

A softmax head maps logits $z \in \mathbb{R}^K$ to $p = \mathrm{softmax}(z/\tau)$. The temperature $\tau > 0$ is almost always either fixed by hand (LM training: $\tau = 1$), tuned post hoc on held-out data (calibration), or made a trained parameter (CLIP's `logit_scale`, AdaCos, several contrastive methods). The problem: **when is learning $\tau$ jointly with the weights a real degree of freedom, and when is it a reparameterization that the optimizer and the regularizer — not the data — resolve?**

Three variants, different difficulty:

- **Theory.** For which head architectures is $\tau$ identifiable from the training loss? For unconstrained logits $z = Wh$, $(\tau, W)$ and $(c\tau, cW)$ give identical predictions and identical loss, so $\tau$ is non-identifiable at the level of the function; it is identified only through weight decay, parameterization-dependent gradient dynamics, or a norm constraint. Open: whether the *dynamics* of learned $\tau$ reach a solution unreachable by any fixed $\tau$ with the same effective step size.
- **Method.** Does a learned $\tau$ beat the best fixed $\tau$ found by grid search, at matched compute *including* the search cost, on held-out NLL / zero-shot accuracy / calibration?
- **Measurement.** Learned $\tau$ is usually reported as a single scalar at the end of training. Whether that number means anything — versus being pinned by a clamp, an init, or the weight-decay coefficient — is not routinely measured.

Solving it means: a criterion that predicts, before training, whether learning $\tau$ helps, plus a matched-compute experiment confirming the sign and magnitude of the effect.

## 2. Formal Setting

Encoder $f_\theta: \mathcal{X} \to \mathbb{R}^d$, head $W \in \mathbb{R}^{K \times d}$, logits $z = W f_\theta(x)$, inverse temperature $\beta = 1/\tau$. Loss on $n$ examples:

$$\mathcal{L}(\theta, W, \beta) = -\frac{1}{n}\sum_{i=1}^n \log \frac{\exp(\beta z_{i,y_i})}{\sum_{k}\exp(\beta z_{i,k})} + \lambda\|W\|_F^2 .$$

**Measured quantities.**

- **Effective scale** $s = \beta \cdot \mathbb{E}_i\big[\mathrm{std}_k(z_{i,k})\big]$ — the logit spread actually entering the softmax. This, not $\beta$, is the invariant; log it every 100 steps.
- **Predictive entropy** $H = -\frac{1}{n}\sum_i \sum_k p_{i,k}\log p_{i,k}$, in nats.
- **ECE** with $M=15$ equal-mass bins: $\mathrm{ECE} = \sum_{m}\frac{|B_m|}{n}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|$. Equal-mass, not equal-width — equal-width bins under-report ECE for over-confident models.
- **Held-out NLL** in nats/token (LM) or top-1 (retrieval/classification), 3 seeds, report seed std.
- **Gradient on $\beta$:** $\partial\mathcal{L}/\partial\beta = -\frac{1}{n}\sum_i (z_{i,y_i} - \mathbb{E}_{p_i}[z_i])$ — the mean logit margin. It vanishes exactly when the model's expected logit equals the target logit, i.e. $\beta$ is driven by *average margin*, not by accuracy.

**Non-identifiability.** With $z = Wf_\theta(x)$ unconstrained, $\mathcal{L}$ at $\lambda = 0$ depends on $(\beta, W)$ only through $\beta W$. The parameterization changes the gradient flow but not the loss surface's level sets in function space.

**Assumptions and where they break.**

1. *$\tau$ is a single global scalar.* Violated: per-class and per-instance temperatures are known to matter under class imbalance (Kukleva et al., ICLR 2023; Qiu et al., ICML 2023).
2. *Logits are norm-free.* Violated in contrastive and metric heads, where $\|f\|_2 = 1$ makes $z \in [-1, 1]$ and **restores identifiability** of $\beta$.
3. *Train and test temperature coincide.* Violated by design in distillation and post-hoc calibration.
4. *$\tau$ is stationary.* Violated: learned $\tau$ in CLIP-style runs moves monotonically toward its clamp.

## 3. State of the Art

**Established.**

- **Post-hoc temperature scaling** (Guo et al., ICML 2017): fit one scalar on a validation split after training. Accuracy is exactly unchanged (argmax is scale-invariant); ECE drops by ~10× on CIFAR-100 CNNs. Reproduced widely; this is the strongest result in the area and it is *not* a learned-during-training temperature.
- **Contrastive $\tau$ matters and is not a reparameterization.** With $\ell_2$-normalized embeddings, InfoNCE performance varies by several points top-1 across $\tau \in \{0.05, 0.1, 0.5, 1.0\}$ (Chen et al., SimCLR, ICML 2020), with $\tau \approx 0.1$ best for ImageNet. Mechanism characterized by Wang & Liu (CVPR 2021): $\tau$ trades *uniformity* against *tolerance* to semantically similar negatives; small $\tau$ up-weights hard negatives.

**Claimed but unablated.**

- **CLIP's learned `logit_scale`** (Radford et al., ICML 2021): initialized at $\tau = 0.07$, trained, and clamped at $\beta \le 100$. No published ablation isolates learned-$\tau$ against a matched grid over fixed $\tau$ at CLIP scale. The clamp existing at all is evidence the parameter runs away.
- **AdaCos** (Zhang et al., CVPR 2019) derives an adaptive cosine scale for face recognition and reports gains on LFW/MegaFace — benchmark numbers, with the scale rule and the margin change entangled in the same arm.
- **Temperature schedules** (Kukleva et al., ICLR 2023): cosine oscillation between $\tau = 0.1$ and $0.5$ improves long-tail self-supervised transfer; reported as benchmark deltas, not against a per-run tuned constant $\tau$ with equal search budget.

**Theory SOTA.** Agarwala et al., *Temperature check* (2020): in the wide/large-logit regime, softmax-CE training under different $\tau$ maps onto different effective learning-rate and initialization scales; low temperature induces near-linear (NTK-like) dynamics. This is the sharpest statement that fixed-$\tau$ choice is largely a *dynamics* knob for unnormalized heads.

## 4. What Is Known

- **Temperature scaling, ResNet-110 / CIFAR-100 (Guo et al. 2017):** ECE $\approx 12.7\% \to \approx 1.3\%$ with a single fitted $T \approx 2.2$; accuracy identical to 4 significant figures. Scale: 1.7M–1.7B FLOP-class CNNs, 50k train images.
- **SimCLR, ResNet-50, ImageNet linear eval:** $\tau = 0.1$ best among $\{0.05, 0.1, 0.5, 1.0\}$ with $\ell_2$ normalization; spread across that grid is a few top-1 points, and removing normalization while keeping $\tau=1$ costs substantially more. Scale: 1.28M images, batch 4096.
- **CLIP `logit_scale` clamp:** $\beta_{\max} = 100 \Leftrightarrow \tau_{\min} = 0.01$; init $\tau_0 = 0.07$. Open-source reproductions (OpenCLIP) report the parameter sitting at or near the clamp late in training — the reported "learned" value is then the clamp value.
- **Logit-growth instability is real at LM scale:** attention/output logits grow without bound and destabilize training; mitigations are auxiliary z-loss (PaLM, Chowdhery et al. 2022), QK-norm and related (Wortsman et al., ICLR 2024), and $\sigma$Reparam (Zhai et al., ICML 2023). All are effective-scale controllers under other names.
- **Gumbel-Softmax / Concrete** (Jang et al.; Maddison et al., ICLR 2017): $\tau$ controls the bias–variance trade of the relaxed gradient; annealed schedules beat fixed $\tau$ on structured-latent tasks at small scale (MNIST-class VAEs). Learning $\tau$ here is known to collapse toward the low-bias/high-variance end without a penalty.

## 5. What Is Not Known

- **Theoretically open.** No proof either way that for an unnormalized softmax head with weight decay $\lambda > 0$, joint $(\beta, W)$ gradient descent converges to a function not reachable by fixed-$\beta$ descent with a matched effective step size. The non-identifiability is trivial; whether the *dynamics* break it usefully is not settled.
- **Theoretically open.** No characterization of the optimal $\tau$ in InfoNCE as a function of the data's cluster structure (intra-/inter-class similarity gap), beyond the qualitative uniformity–tolerance account.
- **Empirically open.** Learned $\tau$ vs. compute-matched grid-searched fixed $\tau$, at $\ge 10^{21}$ FLOP, 3 seeds, on both an autoregressive LM and a CLIP-style dual encoder. Runnable today; not published in that form.
- **Empirically open.** Whether learned $\tau$ in CLIP-style training is doing anything except walking to the clamp — i.e. whether an ablation with clamp removed diverges, and whether clamp value is the real hyperparameter.
- **Methodologically blocked.** "Is $\tau$ learned well?" has no agreed target. Held-out NLL, ECE, and zero-shot top-1 select different $\tau$; a model can improve NLL and worsen ECE with the same change. Until a single decision metric is fixed per use case, cross-paper comparison of learned-$\tau$ claims is not meaningful.

## 6. Why It Is Hard

**Non-identifiability plus confounded measurement.** For unnormalized heads $\beta$ and $\|W\|$ enter the loss only as a product, so the training objective does not select $\tau$; the weight-decay coefficient does. Every reported "learned temperature" from such a head is therefore a readout of the regularizer and the parameterization, not of a property of the data. Papers report the final $\tau$ as if it were a discovered constant.

Compounding it: the natural control arm is expensive. A fair comparison must charge the fixed-$\tau$ arm its grid-search cost and the learned-$\tau$ arm its own sensitivity to $\tau_0$, the $\beta$ learning rate, and the clamp — three hyperparameters that replace the one they eliminate. At LM scale a 5-point grid $\times$ 3 seeds is 15 runs, which is why the ablation does not exist.

## 7. Current Research (as of 2026)

- **Per-instance and per-class temperatures.** Qiu et al. (ICML 2023) individualize $\tau$ per anchor in contrastive learning; Kukleva et al. (ICLR 2023) schedule it for long-tail data. Active in self-supervised vision.
- **Effective-logit-scale control as a stability tool.** QK-norm, z-loss, $\sigma$Reparam are converging on "constrain the scale entering the softmax" rather than "learn it" — Google DeepMind, Apple, EleutherAI lineages. *(frontier — verify current defaults per codebase.)*
- **Temperature and RLHF/inference-time sampling.** Learned or predicted decoding temperature conditioned on context, and the distinction between training $\tau$ and sampling $\tau$ in preference-tuned models. *(frontier — verify.)*
- **Uncertainty-carrying temperature.** Per-sample $\tau$ read as an uncertainty estimate (Zhang, Wu, Bayrooti, Goodman, *Temperature as Uncertainty in Contrastive Learning*, 2021). Small-scale only.

## 8. Concrete Next Experiment

**Setting.** OpenCLIP ViT-B/32 dual encoder, CC12M (12M pairs), 32 epochs, batch 8192, AdamW, cosine LR. ~1.3×10<sup>20</sup> FLOP per run; ~10 A100-days each on 8×A100. Normalized embeddings, so $\beta$ is genuinely identifiable — this is the setting where learned $\tau$ *could* be real.

**Arms** (3 seeds each):

1. **Control:** fixed $\tau \in \{0.01, 0.03, 0.07, 0.15\}$, 4 runs × 3 seeds = 12 runs. Best-of-grid is the control number.
2. **Learned:** $\beta$ trained, init $\tau_0 = 0.07$, clamp $\beta \le 100$ (CLIP default), 3 runs.
3. **Learned, unclamped:** identical but no clamp, 3 runs — tests whether the clamp is the real hyperparameter.
4. **Learned, $\tau_0 = 0.5$**, clamped, 3 runs — tests init-dependence.

Total 21 runs, ~210 A100-days. Log $s = \beta\,\mathbb{E}[\mathrm{std}_k(z)]$ every 100 steps.

**Deciding number.** $\Delta = \text{(ImageNet zero-shot top-1, best learned arm)} - \text{(best fixed-}\tau\text{ arm)}$, seed std expected $\approx 0.3$ pt.

- $\Delta > +0.5$ pt: learned $\tau$ is a real degree of freedom under normalization.
- $|\Delta| \le 0.5$ pt: learned $\tau$ is a *search-cost saver only* — report it as such, and the field should stop citing it as a modeling contribution.
- Arm 3 diverging or landing far from arm 2's endpoint: the clamp, not the data, sets $\tau$ — a publishable negative result on its own.

**Cheap companion (1 GPU-day):** same four arms on a 124M-param GPT-2-class LM, 2B tokens, *unnormalized* head, with weight decay $\lambda \in \{0, 0.1\}$ on the output layer. Prediction from §6: at $\lambda = 0$ the learned-$\tau$ and fixed-$\tau$ arms match in val NLL to within seed noise ($<0.005$ nats/token), and the final $\tau$ varies by $>2\times$ across seeds — non-identifiability made visible.

## 9. Key References

- **[Foundational]** Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NIPS 2014 Deep Learning Workshop. — arXiv:1503.02531
- **[Foundational]** Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. — arXiv:1706.04599
- **[Foundational]** Jang, Gu, Poole. *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017. — arXiv:1611.01144
- **[Foundational]** Maddison, Mnih, Teh. *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.* ICLR 2017. — arXiv:1611.00712
- **[SOTA]** Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. — arXiv:2103.00020
- **[SOTA]** Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020. — arXiv:2002.05709
- **[SOTA]** Wang, Liu. *Understanding the Behaviour of Contrastive Loss.* CVPR 2021. — arXiv:2012.09740
- **[SOTA]** Kukleva, Böhle, Schiele, Kuehne, Rupprecht. *Temperature Schedules for Self-Supervised Contrastive Methods on Long-Tail Data.* ICLR 2023.
- **[SOTA]** Qiu et al. *Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization.* ICML 2023.
- **[SOTA]** Zhang, Zhao, Qiao, Wang, Li. *AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations.* CVPR 2019.
- **[Theory]** Agarwala, Pennington, Dauphin, Schoenholz. *Temperature check: theory and practice for training models with softmax-cross-entropy losses.* 2020. — arXiv:2010.07344
- **[Systems]** Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR 2024. — arXiv:2309.14322
- **[Systems]** Zhai et al. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML 2023.
- **[Survey]** Kull, Nieto, Kängsepp, Silva Filho, Song, Flach. *Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration.* NeurIPS 2019.

## 10. Worked Example

**Unnormalized head, $K = 3$.** Take $h$ with $Wh = (2.0, 0.0, -1.0)$ and $\beta = 1$. Then $p = \mathrm{softmax}(2, 0, -1) = (0.836, 0.113, 0.042)$, NLL on the true class $= 0.179$ nats.

Now take $\beta' = 8$, $W' = W/8$, so $W'h = (0.25, 0, -0.125)$ and $\beta' W' h = (2, 0, -1)$. **Identical** probabilities, identical NLL, identical accuracy, identical ECE. The reported temperature is $\tau = 1$ in one case and $\tau = 0.125$ in the other, for the same function.

What separates them is only the penalty: with $\lambda\|W\|_F^2$ and $\|Wh\|$ fixed, the second configuration carries $1/64$ of the first's weight-decay cost. So gradient descent drives $\beta$ up and $\|W\|$ down until the $\beta$-side of the parameterization runs out of gradient — $\partial\mathcal{L}/\partial\beta = -(z_y - \mathbb{E}_p[z])$, the mean margin, which shrinks as $W$ shrinks. **The learned temperature is a readout of $\lambda$, not of the data.** Rerun the same setup at $\lambda = 0.1$ and $\lambda = 0.01$ and the final $\tau$ moves by roughly the ratio of the shrinkage scales while val NLL is unchanged.

**Contrast: normalized head.** Set $\|f\|_2 = \|w_k\|_2 = 1$, so $z_k = \cos\angle(f, w_k) \in [-1,1]$. The rescaling above is now illegal — $W' = W/8$ leaves the unit sphere. With a realistic positive/hard-negative cosine gap of $0.90$ vs $0.75$, the logit margin is $0.15$; at $\tau = 0.07$ the softmax sees a margin of $2.14$, at $\tau = 0.01$ it sees $15$. That is the difference between a graded loss and a hinge on the single hardest negative. Here $\tau$ is identifiable and consequential.

**Where the obstruction shows.** CLIP's learned $\beta$ lives in the second regime, so learning it is defensible — yet the clamp is set at $\beta = 100$ and the parameter is reported approaching it. The measured quantity is then $\min(\beta_{\text{learned}}, 100) = 100$: the value came from a line of code, not from CC12M. No published ablation removes the clamp and reports what $\beta$ does. That single missing run is why the problem is still open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*