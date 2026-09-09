---
id: 08-loss-and-heads/output-head-initialization-landscape
title: "Loss Landscape Effect of Output Head Initialization"
topic: 08-loss-and-heads
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Loss Landscape Effect of Output Head Initialization

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/output-head-initialization-landscape` · **Status:** empirically-open

## 1. Problem Statement

The final linear layer (the "readout" or "output head") maps a $d$-dimensional representation to $C$ logits. Its initialization scale — from zero, through $1/\sqrt{d}$ Glorot/He scaling, up to large — is a single scalar knob, and folklore across three subfields says it matters: focal-loss detectors need a specific bias prior or they diverge; $\mu$P prescribes a zero-initialized readout; LP-FT says a randomly initialized head destroys pretrained features.

The question: **does output-head initialization change the loss landscape the optimizer actually traverses, or only the first few steps of the trajectory?** Three variants, of very different difficulty:

- **Measurement.** Define landscape quantities (Hessian spectrum, barrier height between solutions, sharpness at convergence) that are invariant to the reparameterizations head scaling induces, and measure how they move with head init scale $\sigma_W$ at fixed final loss.
- **Method.** Find a head-init rule that dominates the alternatives on final loss/generalization at fixed compute, with the mechanism ablated rather than asserted.
- **Theory.** Prove a statement of the form: for architecture class $\mathcal{A}$ and loss $\ell$, head scale $\sigma_W$ below a threshold puts training in the feature-learning regime, above it in the lazy/kernel regime, with a quantified boundary at finite width.

A solution to the measurement variant is a reparameterization-invariant statistic plus a scaling curve; to the method variant, a rule plus its ablation; to the theory variant, a finite-width theorem. They are currently conflated.

## 2. Formal Setting

Backbone $f_\theta:\mathcal{X}\to\mathbb{R}^d$ with parameters $\theta$, features $h=f_\theta(x)$, head $(W,b)$ with $W\in\mathbb{R}^{C\times d}$. Logits

$$z(x)=\alpha\,W h(x)+b,\qquad \alpha>0 \text{ a fixed multiplier},\qquad W_{ij}\sim\mathcal{N}(0,\sigma_W^2).$$

Loss $L(\theta,W,b)=\mathbb{E}_{(x,y)}\,\ell(z(x),y)$, cross-entropy unless stated. The knob is $(\sigma_W,\alpha,b_0)$; note $\sigma_W$ and $\alpha$ are **not** separately identifiable at initialization (only the product sets logit scale) but are identifiable under gradient descent, because the gradient w.r.t. $W$ carries $\alpha$ and the gradient w.r.t. $\theta$ carries $\alpha\sigma_W$. Non-identifiability at $t=0$ and identifiability at $t>0$ is the source of most of the confusion in the literature.

Measured quantities, as they would actually be logged:

- **Initial logit scale** $s_0=\mathrm{sd}_{x,c}[z_c(x)]$ at step 0, over a held-out batch of $\geq 2^{13}$ examples.
- **Initial loss gap** $L_0-\log C$ (for balanced $C$-way classification, $\log C$ is the loss of the zero-logit head). Non-zero $s_0$ makes $L_0>\log C$ in expectation.
- **Head/backbone gradient ratio** $\rho_t=\|\nabla_W L\|_F/\|\nabla_\theta L\|_2$ at step $t$, per-parameter-normalized.
- **Feature movement** $\Delta_T=\mathbb{E}_x\|h_T(x)-h_0(x)\|_2/\mathbb{E}_x\|h_0(x)\|_2$ — the lazy/feature-learning order parameter of Chizat–Oyallon–Bach.
- **Hessian outliers.** $\nabla^2 L$ of a $C$-class net has $\approx C$ eigenvalues separated from the bulk (Sagun et al. 2017; Papyan 2019); report $\lambda_1$ and the outlier/bulk-edge ratio, computed by Lanczos on $\geq 10^4$ examples.
- **Linear interpolation barrier** between two seeds sharing backbone init but differing in head init: $B=\max_{u\in[0,1]}L(\text{lerp}(u))-\max\{L(0),L(1)\}$ (Frankle et al. 2020 instability analysis).

Assumptions, and which fail. (i) *Comparisons at fixed $L_0$ isolate the head* — violated, since changing $\sigma_W$ changes $\rho_0$ and thus the whole early trajectory. (ii) *Hessian sharpness is comparable across head scales* — violated: $W\!\to\!cW$, $h\!\to\!h/c$ leaves $z$ fixed but rescales the Hessian, so raw $\lambda_1$ is not a landscape property, only a parameterization property. (iii) *Balanced classes with $b_0=0$ optimal* — violated in detection and in language modeling, where unigram-prior bias init changes $L_0$ by $>1$ nat. (iv) *Fixed learning rate is a fair control* — violated, because the optimal LR itself moves with $\sigma_W$.

## 3. State of the Art

**Established (ablated, reproduced).**

- **$\mu$P / zero-init readout.** Yang & Hu (ICML 2021) and Yang et al., *Tensor Programs V* (NeurIPS 2021, arXiv:2203.03466) prescribe readout weights initialized at zero with a $1/\text{fan\_in}$ output multiplier; under this parameterization the optimal LR transfers across width. The transfer claim is ablated over widths up to 4096 and a 6.7B-parameter transfer target.
- **Bias prior init.** Lin et al., *Focal Loss for Dense Object Detection* (ICCV 2017) initialize the classification-head bias so the predicted foreground prior is $\pi=0.01$. The paper reports that without it, training with a large class imbalance is unstable in the first epoch. This is a **bias**, not a weight-scale, result.
- **Head init in fine-tuning.** Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (ICLR 2022) show a randomly initialized head produces large early gradients that move features; linear-probe-then-fine-tune (LP-FT) removes the effect, with OOD gains of roughly 10 points on the reported WILDS/DomainNet-style shifts.
- **Zero-init of residual-branch outputs.** Fixup (Zhang, Dauphin, Ma, ICLR 2019), SkipInit (De & Smith, NeurIPS 2020), ReZero (Bachlechner et al., UAI 2021). These are branch outputs, not the task head — frequently cited as if they settled the head question. They do not.

**Claimed but unablated.** That zero readout init "flattens the landscape" or "removes bad minima". No paper isolates $\sigma_W$ with backbone init, data order, and tuned LR all held fixed and reports a reparameterization-invariant landscape statistic.

**Benchmark-number-only.** Most LLM recipes (small trunc-normal readout, tied vs. untied embeddings, z-loss) are reported as final perplexity in a training report, without a head-init sweep.

## 4. What Is Known

- **Output scale controls lazy vs. feature learning.** Chizat, Oyallon & Bach (NeurIPS 2019) prove that as the output multiplier $\alpha\to\infty$ with the initial output forced to zero, training converges to the linearized (NTK) dynamics; $\Delta_T\to 0$ at rate $O(1/\alpha)$. Measured on CIFAR-10 CNNs at width $\sim10^2$–$10^3$, lazy training generalizes worse — several points of test accuracy.
- **The head produces the Hessian outliers.** Sagun et al. (2017) and Papyan (JMLR 2020) find exactly $C$ outlier eigenvalues traceable to class-mean logit structure; at $C=10$ on CIFAR-10, outliers exceed the bulk edge by one to two orders of magnitude. Head scale directly moves these.
- **Head init decides seed instability.** Frankle et al. (ICML 2020): whether two runs are linearly mode connected depends on the first few hundred SGD steps; on ResNet-20/CIFAR-10 barriers drop to $\approx 0$ after $\sim$1–3% of training.
- **Logit growth is a real instability at scale.** Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* (ICLR 2024, arXiv:2309.14322) reproduce output-logit divergence at $\sim$10M–1.2B parameters and control it with a z-loss of coefficient $10^{-4}$; the instability's LR onset shifts predictably with scale.
- **Neural collapse.** Papyan, Han & Donoho (PNAS 2020): at convergence, last-layer features collapse to class means and the head aligns to a simplex ETF — measured across many architectures and datasets, meaning the *endpoint* head geometry is largely init-independent, even where the path is not.

## 5. What Is Not Known

- **Empirically open.** Whether head-init scale changes anything at convergence beyond a transient, at LLM pretraining scale ($\geq$1B parameters, $\geq$20 tokens/parameter) with LR retuned per arm. The experiment is runnable today; it costs a full sweep and nobody has published one with the LR control.
- **Empirically open.** Whether zero readout init's benefit under $\mu$P is a landscape effect or purely an LR-transfer bookkeeping effect. Untangling needs the same $\mu$P run with $\sigma_W\in\{0,\;0.1/\sqrt d,\;1/\sqrt d\}$ at matched tuned LR.
- **Theoretically open.** A finite-width threshold $\sigma_W^\ast(d,C,n)$ separating feature-learning from lazy dynamics. Chizat et al. give an asymptotic-in-$\alpha$ statement; no non-asymptotic boundary exists.
- **Methodologically blocked.** "Flatness" as a landscape property. Under the head rescaling symmetry $W\!\to\!cW,\,h\!\to\!h/c$, $\lambda_1$ changes while the function does not (the Dinh et al., ICML 2017, argument, which applies verbatim to a linear head). No accepted invariant sharpness measure exists for this symmetry, so "head init flattens the loss surface" is currently not a well-posed claim.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a gauge symmetry**.

1. *Gauge.* Head scale is partly a reparameterization. Any sharpness statistic that is not invariant to $W\!\to\!cW,\,h\!\to\!h/c$ will report a difference that has no functional content. Every popular one ($\lambda_1$, trace, SAM-radius loss increase) fails this test.
2. *LR confound.* Changing $\sigma_W$ changes $\rho_0$ by orders of magnitude, so the optimal LR moves. Fixed-LR comparisons — the norm in the literature — measure "which init is best at this LR", not "which init is better".
3. *Erasure at convergence.* Neural collapse says the endpoint geometry is nearly init-independent, so the effect, if any, lives in the transient, and transients are exactly where seed noise is largest. Detecting a 0.5% perplexity effect against seed noise needs $\geq 3$ seeds per arm.
4. *Scale.* Instabilities that head init plausibly controls (logit divergence) only appear above roughly 1B parameters at aggressive LR, which is where sweeps become expensive.

## 7. Current Research (as of 2026)

- **$\mu$P and depth-$\mu$P** (Yang, Hu, Littwin and collaborators; Tensor Programs VI, 2023) continue to treat readout init as prescribed rather than as a variable to study. *(frontier — verify)* Extensions to optimizer-specific scaling (Adam-$\mu$P, "unit scaling") are active.
- **Training-stability work** downstream of Wortsman et al. (2024): z-loss, QK-norm, logit soft-capping. These are output-head interventions studied as stability fixes, not as landscape probes.
- **Fine-tuning head init** (LP-FT descendants, surgical fine-tuning) remains the one area where the head-init ablation is routinely run — but at $\leq$1B scale and on classification heads.
- **Invariant sharpness** — attempts at reparameterization-invariant flatness (adaptive/relative sharpness, Fisher-normalized measures). No consensus measure. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Decoder-only transformer, 1.4B parameters, 30B tokens (Chinchilla-ish), $\mu$P, untied output embedding, standard AdamW.

**Arms.** Readout init $\sigma_W\in\{0,\;0.02/\sqrt d,\;0.2/\sqrt d,\;1/\sqrt d\}$, each with an **independently tuned** peak LR over a 5-point grid at 1/10 the token budget, then run at full budget. 3 seeds per arm at the tuned LR.

**Control arm.** $\sigma_W=0$ (the $\mu$P prescription), identical backbone init seed, identical data order.

**Deciding number.** Final validation loss gap between the best and worst arm, each at its own tuned LR, in nats: $\Delta L^\ast=\max_\sigma L^\ast(\sigma)-\min_\sigma L^\ast(\sigma)$, compared against the seed standard deviation $s_{\text{seed}}$.

- $\Delta L^\ast < 2 s_{\text{seed}}$ (expected $s_{\text{seed}}\approx 0.003$ nats at this scale) → head init is an LR-bookkeeping effect, not a landscape effect. The problem closes negatively.
- $\Delta L^\ast > 0.01$ nats → a real effect survives LR tuning; then report $\Delta_T$ (feature movement) and the barrier $B$ between arms to identify the mechanism.

Secondary readout: $\Delta_T$ at 1% of training, which the lazy-training theory predicts should scale as $1/\sigma_W$ across the four arms. A clean $1/\sigma_W$ fit with a flat $\Delta L^\ast$ would be the sharpest possible statement: the regime changes, the outcome does not.

## 9. Key References

- **[Foundational]** Xavier Glorot, Yoshua Bengio. *Understanding the difficulty of training deep feedforward neural networks.* AISTATS, 2010.
- **[Foundational]** Lénaïc Chizat, Edouard Oyallon, Francis Bach. *On Lazy Training in Differentiable Programming.* NeurIPS, 2019. — arXiv:1812.07956
- **[Foundational]** Laurent Dinh, Razvan Pascanu, Samy Bengio, Yoshua Bengio. *Sharp Minima Can Generalize For Deep Nets.* ICML, 2017. — arXiv:1703.04933
- **[SOTA]** Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[SOTA]** Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, Simon Kornblith. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[SOTA]** Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR, 2022. — arXiv:2202.10054
- **[Established]** Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. *Focal Loss for Dense Object Detection.* ICCV, 2017. — arXiv:1708.02002
- **[Established]** Hongyi Zhang, Yann N. Dauphin, Tengyu Ma. *Fixup Initialization: Residual Learning Without Normalization.* ICLR, 2019. — arXiv:1901.09321
- **[Established]** Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin. *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* ICML, 2020. — arXiv:1912.05671
- **[Established]** Vardan Papyan, X. Y. Han, David L. Donoho. *Prevalence of neural collapse during the terminal phase of deep learning training.* PNAS 117(40), 2020. — arXiv:2008.08186
- **[Survey]** Vardan Papyan. *Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra.* JMLR 21(252), 2020.

## 10. Worked Example

CIFAR-10, ResNet-18, $d=512$, $C=10$, SGD+momentum, 200 epochs. Two arms differing only in head init.

**Arm A ($\sigma_W=0$).** All logits are 0 at step 0, so $L_0=\log 10=2.3026$ exactly. $\nabla_\theta L=0$ (the backbone gradient is $\alpha W^\top(\cdot)$ and $W=0$): **the backbone does not move on step 1 at all**. $\rho_0=\infty$. The head alone absorbs the first steps.

**Arm B (He init, $\sigma_W=\sqrt{2/512}=0.0625$).** With unit-scale features, $s_0\approx 0.0625\sqrt{512}\cdot\mathrm{sd}(h)\approx 1.4$. Then $\mathbb{E}[L_0]\approx\log C+\tfrac{1}{2}s_0^2\cdot\frac{C-1}{C}\approx 2.30+0.88=3.18$ nats — a 0.88-nat handicap that is gone within ~200 steps.

**Where the obstruction becomes visible.** Suppose after training Arm B reports $\lambda_1=180$ and Arm A $\lambda_1=95$, and someone concludes zero init "finds a flatter minimum". Now take Arm B's converged network and apply $W\to 2W$, $h\to h/2$ (absorb the $1/2$ into the preceding BN scale). The function is bit-identical; test accuracy is unchanged; $L$ is unchanged everywhere on the data manifold. But the Hessian block structure rescales, and the measured $\lambda_1$ moves — typically by a factor between $c^{-2}$ and $c^{2}$ depending on which block dominates. A single gauge choice can turn 180 into 45 or into 720.

So the reported "flatness gap" of $180\to 95$ is inside the range that a null reparameterization can manufacture. Until the statistic is made invariant to head rescaling — or until the comparison is made on a functional quantity like final loss at independently tuned LR (Section 8) — the landscape claim is not measuring what it names.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*