---
id: 24-multimodal/contrastive-temperature-batch-coupling
title: "Optimal Contrastive Temperature and Batch Size Coupling"
topic: 24-multimodal
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Contrastive Temperature and Batch Size Coupling

> **Topic:** Multimodal Models · **ID:** `24-multimodal/contrastive-temperature-batch-coupling` · **Status:** partially-solved

## 1. Problem Statement

In CLIP-style contrastive pretraining, two hyperparameters jointly control the loss landscape: the softmax temperature $\tau$ and the contrastive batch size $B$ (the number of in-batch negatives). Both set how much gradient mass concentrates on the hardest negatives. The question: **is there a law $\tau^*(B)$ giving the downstream-optimal temperature as a function of batch size, and is it a power law $\tau^* \propto B^{-\alpha}$, a log law $\tau^* \propto 1/\log B$, or neither?**

Three variants, of very different difficulty:

- **Measurement.** Given a fixed architecture, data distribution, and compute budget, estimate $\tau^*(B)$ on a grid and fit an exponent. Runnable; almost never run with $\tau$ and $B$ properly decoupled.
- **Method.** Produce a rule that transfers — set $B$ from hardware, read off $\tau$, and match a tuned sweep to within noise. Analogous to the linear scaling rule for SGD learning rate (Goyal et al., 2017), which succeeded in exactly this role.
- **Theory.** Prove that some invariant of the negative-softmax distribution (entropy, effective negative count, gradient bias) is the quantity that must be held fixed, and derive $\tau^*(B)$ from it.

Solving it means: a closed-form or tabulated $\tau^*(B)$ that predicts the argmin of downstream zero-shot error to within the run-to-run standard deviation, across at least two orders of magnitude in $B$ and one in model scale.

## 2. Formal Setting

Encoders $f_\theta: \mathcal{X}\to\mathbb{S}^{d-1}$ (image) and $g_\phi: \mathcal{Y}\to\mathbb{S}^{d-1}$ (text), $\ell_2$-normalised. For a batch of $B$ pairs, $s_{ij} = \langle f_\theta(x_i), g_\phi(y_j)\rangle \in [-1,1]$. The symmetric InfoNCE objective:

$$\mathcal{L}(\tau,B) = -\frac{1}{2B}\sum_{i=1}^{B}\left[\log\frac{e^{s_{ii}/\tau}}{\sum_{j}e^{s_{ij}/\tau}} + \log\frac{e^{s_{ii}/\tau}}{\sum_{j}e^{s_{ji}/\tau}}\right].$$

**Measured quantities.**

- $\tau$: in CLIP it is *learned*, parameterised as $\log(1/\tau)$ and clipped at $1/\tau \le 100$. Measure the value at end of training, not the initialisation ($0.07$ in Radford et al., 2021).
- $B$: the number of pairs in the softmax denominator — **not** the optimizer batch size. With sharded all-gather these coincide; with cached-negative or memory-bank methods they do not. Always report the softmax $B$.
- Negative posterior: $p_{ij} = e^{s_{ij}/\tau}/\sum_{k\ne i} e^{s_{ik}/\tau}$ for $j\ne i$. Its **effective negative count** is $N_{\mathrm{eff}} = \exp(H(p_{i\cdot}))$ averaged over $i$, with $H$ the Shannon entropy in nats. $N_{\mathrm{eff}} \in [1, B-1]$; this is the natural candidate invariant.
- $\tau^*(B) = \arg\min_\tau \mathbb{E}[\text{err}_{\text{zs}}(\theta_T(\tau,B))]$, where $\text{err}_{\text{zs}}$ is zero-shot ImageNet top-1 error and $T$ is a **fixed number of samples seen**, not a fixed number of steps.
- Coupling exponent: $\alpha = -\,d\log \tau^*/d\log B$, estimated by regression over a $B$-grid.

**Assumptions, and which are violated.**

1. *Negatives are i.i.d. from the marginal.* Violated: web batches contain near-duplicates and shard-correlated samples; false negatives are common in LAION-scale data.
2. *$\tau$ is a single global scalar.* Violated by construction in iSogCLR (Qiu et al., 2023), which shows per-sample optima differ.
3. *Learned $\tau$ converges to $\tau^*$.* Not established — the learned $\tau$ minimises training loss, and clipping at $1/\tau=100$ is a binding constraint in many CLIP runs, so the observed value is partly an artifact of the clip.
4. *The other hyperparameters are held fixed as $B$ varies.* Violated in essentially all published comparisons, where learning rate and warmup are re-tuned with $B$. This is the central confound.

## 3. State of the Art

**Theory SOTA.** InfoNCE is a variational lower bound on mutual information capped at $\log B$ nats (Poole et al., ICML 2019) — established, and it explains why large $B$ helps but says nothing about $\tau$. Wang & Isola (ICML 2020) decompose the $\tau$-scaled loss into alignment and uniformity terms and prove asymptotic optimality of perfectly uniform, perfectly aligned encoders as $B\to\infty$; the limit is $\tau$-independent, so the theory is silent exactly where the problem lives. Wang & Liu (CVPR 2021) show $\tau$ controls a hardness-aware penalty, with a uniformity–tolerance tradeoff. Yuan et al. (ICML 2022, SogCLR) give the sharpest result: the finite-$B$ InfoNCE gradient is a **biased** estimator of the full-data contrastive gradient, and a moving-average correction attains $O(1/\epsilon^4)$ convergence with $B=256$. Established.

**Empirical SOTA.** SigLIP (Zhai et al., ICCV 2023) replaces the softmax with a pairwise sigmoid loss carrying learned temperature *and* bias, breaking the global normalisation over $B$. Its batch-size sweep is the best single piece of evidence: performance rises to roughly $B=32{,}000$ and then flattens or slightly declines out to $B \ge 10^5$. Established as a benchmark number; the mechanism is claimed, not ablated — the paper does not isolate whether saturation comes from the loss form, the temperature, or the data.

**Claimed but unablated.** The folk rule "larger $B$ wants smaller $\tau$" is widespread in practice and has no controlled study behind it at CLIP scale. OpenCLIP reproductions (Cherti et al., CVPR 2023) fit scaling laws in compute, model, and data — but hold $\tau$ learnable and $B$ tied to the hardware configuration, so $\tau^*(B)$ is not recoverable from them.

## 4. What Is Known

- **The $\log B$ ceiling.** InfoNCE cannot certify more than $\log B$ nats of mutual information (Poole et al., 2019). At $B=32{,}768$, that is $10.4$ nats.
- **Batch-size returns saturate.** SimCLR (Chen et al., ICML 2020) at ResNet-50, ImageNet, 100 epochs: linear-probe top-1 rises with $B$ from 256 to 8192, with the gap shrinking sharply after ~2048 and vanishing at 1000 epochs. SigLIP: saturation near $B\approx32$k at ViT-B scale.
- **Small batches are not fatal.** SogCLR at $B=256$, ResNet-50/ImageNet, matches SimCLR at $B=8192$ (~66–69% linear probe, 800 epochs) once the gradient bias is corrected. This is the strongest evidence that the observed $B$-dependence is partly an *optimizer* artifact, not an information-theoretic one.
- **Temperature dominates over a narrow band.** SimCLR reports a clear optimum near $\tau=0.1$; performance degrades by several points at $\tau=0.5$ and at $\tau=0.01$. CLIP initialises $\tau=0.07$ and learns downward against a $\tau=0.01$ clip.
- **Long-tail data prefers non-constant $\tau$.** Kukleva et al. (ICLR 2023) show a cosine $\tau$ schedule improves ImageNet-LT and full-ImageNet linear probes over any constant $\tau$, by ~1–2 points at ResNet-50 scale.
- **Per-sample $\tau$ beats global $\tau$.** iSogCLR (Qiu et al., ICML 2023) improves zero-shot and retrieval over a tuned global temperature at CC3M/CC12M scale.

## 5. What Is Not Known

- **Empirically open.** The exponent $\alpha$ itself. No published study sweeps $\tau$ on a grid at three or more values of $B$ spanning $\ge 2$ decades with everything else — learning rate, schedule, samples seen, data order — held fixed. The experiment is entirely runnable; it costs roughly one CLIP-B/32 training run times the grid size.
- **Theoretically open.** Whether any invariant of $p_{i\cdot}$ ($N_{\mathrm{eff}}$, entropy, gradient bias norm) is provably the right thing to hold fixed under $B$. No proof either way. A conjecture with a clean prediction: holding $N_{\mathrm{eff}}$ fixed requires $1/\tau^* \propto \log B$ when similarities have $B$-independent spread, giving $\tau^* \propto 1/\log B$ — much flatter than any power law.
- **Methodologically blocked.** "Optimal" is not well defined across objectives. $\tau$ that minimises zero-shot ImageNet error, retrieval R@1, and linear-probe error need not coincide, and the alignment/uniformity tradeoff predicts they should not. Until the target metric is fixed, $\tau^*(B)$ is a family of curves, not a curve.
- **Unresolved.** Whether the learned $\tau$ in CLIP is at its downstream optimum or merely at its clip boundary.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**. Changing $B$ in a real run changes at least four things at once: the number of negatives in the softmax, the gradient noise scale, the number of optimizer steps per epoch, and (because practitioners re-tune) the learning rate. Any measured $\tau^*(B)$ therefore mixes an information-theoretic effect with an optimization effect. SogCLR's result — $B=256$ matching $B=8192$ — proves the mixture is real and that the optimization component is large enough to invert conclusions.

Second obstruction: **cost**. A defensible $\alpha$ needs a $4\times4$ grid ($B \in \{1\text{k},4\text{k},16\text{k},64\text{k}\}$, four $\tau$ each) with $\ge2$ seeds, at a scale where zero-shot numbers are not noise-dominated. That is ~32 CLIP-B/32 runs, ~$10^4$ GPU-hours. Cheaper than a frontier run, more than any single academic group has spent on a hyperparameter coupling question.

## 7. Current Research (as of 2026)

- **Loss forms that decouple $B$ from normalisation.** SigLIP and successors (Google DeepMind) remove the global softmax; the temperature question survives but changes shape, since the sigmoid bias absorbs part of what $\tau$ did. *(frontier — verify)* whether SigLIP's optimal $(\tau, b)$ pair obeys a simpler $B$-law than InfoNCE's $\tau$.
- **Adaptive and individualised temperature.** Yuan/Qiu group (Texas A&M / RPI) — DRO-derived per-sample $\tau$; the natural next step is showing individualisation makes $B$-dependence disappear.
- **Gradient-bias correction at scale.** Whether SogCLR-style corrections hold at LAION-2B and ViT-L is untested publicly. *(frontier — verify)*
- **Temperature schedules** (Kukleva et al. line, Oxford VGG) applied to multimodal rather than unimodal SSL.

## 8. Concrete Next Experiment

**Scale.** ViT-B/32 + 12-layer text encoder, DataComp-1B or LAION-400M subset, **fixed 3B samples seen** for every arm (not fixed epochs, not fixed steps). ~$3\times10^3$ A100-hours per arm at this budget.

**Grid.** $B \in \{1024, 4096, 16384, 65536\}$ (softmax batch, via all-gather) $\times$ fixed $\tau \in \{0.2, 0.1, 0.05, 0.02\}$, temperature **frozen**, 2 seeds. 32 runs.

**Control arm.** For each $B$, one run with learned $\tau$ (CLIP default, init 0.07) and one with $\tau$ set by the $1/\log B$ rule anchored at $\tau=0.1$ for $B=1024$: $\tau(B) = 0.1 \cdot \log(1024)/\log(B)$, giving $0.1, 0.083, 0.071, 0.062$. Learning rate held at the same value across all $B$ *and* additionally swept in a $2\times$ ladder for $B=1024$ and $B=65536$ only, to bound the LR–$B$ confound.

**Deciding number.** Fit $\log \tau^* = c - \alpha \log B$ over the four batch sizes, with $\tau^*$ the parabola-vertex interpolation of zero-shot ImageNet top-1 over the $\tau$ grid. **The single number is $\alpha$ with its bootstrap 95% CI.** Outcomes: CI containing $0$ ⇒ $\tau^*$ is $B$-independent and the folk rule is wrong; CI around $0.15$–$0.25$ ⇒ power-law coupling; CI excluding $0$ but with a $1/\log B$ fit achieving lower residual than any power law ⇒ the entropy-invariance conjecture. Report $N_{\mathrm{eff}}$ at convergence for every arm; if $N_{\mathrm{eff}}$ is constant along the fitted $\tau^*(B)$ curve, the invariant is identified.

## 9. Key References

- **[Foundational]** van den Oord, Li, Vinyals. *Representation Learning with Contrastive Predictive Coding.* 2018. — arXiv:1807.03748
- **[Foundational]** Poole, Ozair, van den Oord, Alemi, Tucker. *On Variational Bounds of Mutual Information.* ICML 2019. — arXiv:1905.06922
- **[Foundational]** Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020. — arXiv:2002.05709
- **[Foundational]** Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. — arXiv:2103.00020
- **[Theory]** Wang, Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML 2020. — arXiv:2005.10242
- **[Theory]** Wang, Liu. *Understanding the Behaviour of Contrastive Loss.* CVPR 2021. — arXiv:2012.09740
- **[Theory]** Yuan, Wang, Xu, Jin, Yang. *Provable Stochastic Optimization for Global Contrastive Learning: Small Batch Does Not Harm Performance.* ICML 2022. — arXiv:2202.12387
- **[SOTA]** Zhai, Mustafa, Kolesnikov, Beyer. *Sigmoid Loss for Language Image Pre-Training.* ICCV 2023. — arXiv:2303.15343
- **[SOTA]** Qiu, Hu, Yuan, Zhou, Yang. *Not All Semantics Are Created Equal: Contrastive Self-Supervised Learning with Automatic Temperature Individualization.* ICML 2023. — arXiv:2305.11965
- **[SOTA]** Kukleva, Böhle, Schiele, Kuehne, Rupprecht. *Temperature Schedules for Self-Supervised Contrastive Methods on Long-Tail Data.* ICLR 2023. — arXiv:2303.13664
- **[Scaling]** Cherti et al. *Reproducible Scaling Laws for Contrastive Language-Image Learning.* CVPR 2023. — arXiv:2212.07143
- **[Related]** Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* 2017. — arXiv:1706.02677
- **[Related]** Robinson, Chuang, Sra, Jegelka. *Contrastive Learning with Hard Negative Samples.* ICLR 2021. — arXiv:2010.04592
- **[Survey]** Balestriero et al. *A Cookbook of Self-Supervised Learning.* 2023. — arXiv:2304.12210

## 10. Worked Example

Take a trained CLIP-B/32 and measure $N_{\mathrm{eff}}$ directly. Suppose off-diagonal similarities within a batch are approximately Gaussian with mean $\mu = 0.10$ and standard deviation $\sigma = 0.05$ — typical for a converged CLIP image–text similarity matrix on web data. For a softmax over $B-1$ such logits at temperature $\tau$, the entropy of $p_{i\cdot}$ is approximately

$$H \approx \log(B-1) - \tfrac{1}{2}\left(\sigma/\tau\right)^2 \quad \text{(second-order, valid while } \sigma/\tau \lesssim 1).$$

At $\tau = 0.05$, $\sigma/\tau = 1.0$, so $H \approx \log(B-1) - 0.5$ and $N_{\mathrm{eff}} \approx 0.61\,(B-1)$: the softmax is still spread over most of the batch. Going from $B=1024$ to $B=65536$ multiplies $N_{\mathrm{eff}}$ by $64$ at fixed $\tau$.

Now hold $N_{\mathrm{eff}}$ fixed at its $B=1024$, $\tau=0.05$ value ($\approx 624$). At $B=65536$ this requires $\tfrac12(\sigma/\tau)^2 = \log(65535) - \log(624) = 4.16$, so $\sigma/\tau = 2.88$ and $\tau = 0.017$. The implied exponent is $\alpha = \log(0.05/0.017)/\log(64) = 0.26$ — a power law, not the $1/\log B$ law the entropy heuristic suggests when the quadratic expansion holds. **That is the obstruction made visible:** the exponent you predict depends entirely on which regime the expansion is in, and the expansion is already invalid at $\sigma/\tau = 2.88$ (the true entropy deficit saturates rather than growing quadratically). Redoing the calculation in the saturated regime gives $\alpha$ closer to $0.05$ — a $5\times$ disagreement in the predicted rule, driven by a quantity ($\sigma$, the empirical spread of in-batch negative similarities) that nobody publishes and that itself drifts during training. Until $\sigma(t, B)$ is measured and reported, the analytic route cannot distinguish the candidate laws, and the grid experiment in §8 is the only way through.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*