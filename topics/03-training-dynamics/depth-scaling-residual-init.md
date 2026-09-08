---
id: 03-training-dynamics/depth-scaling-residual-init
title: "Depth Scaling of Residual Branch Initialization"
topic: 03-training-dynamics
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Depth Scaling of Residual Branch Initialization

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/depth-scaling-residual-init` · **Status:** partially-solved

## 1. Problem Statement

A residual network of depth $L$ computes $x_{\ell+1} = x_\ell + \alpha_L \mathcal{F}_\ell(x_\ell)$. The question is how $\alpha_L$ — the scale applied to the residual branch at initialization — and the accompanying learning rate must depend on $L$ so that the network trains stably and improves with depth.

Three variants, different difficulty:

- **Measurement.** Given a fixed architecture family, find the exponent $\beta$ in $\alpha_L = \Theta(L^{-\beta})$ under which optimal hyperparameters (peak learning rate $\eta^\star$, warmup, weight decay) are invariant in $L$. Solved by grid search; the cost is the obstruction.
- **Method.** Produce an initialization + learning-rate rule that makes depth a *free* axis: train at $L=8$, tune once, extrapolate to $L=128$ with no retuning and monotone loss improvement. Partially solved for single-layer residual branches.
- **Theory.** Prove that a given $\beta$ yields a well-defined, non-degenerate $L \to \infty$ limit in which (i) forward activations and backward gradients stay $\Theta(1)$, (ii) features change by $\Theta(1)$ during training, and (iii) blocks remain *diverse* (do not collapse to a common update direction). Solved for a restricted class; open for the architectures actually deployed.

Solving it means: a rule $(\alpha_L, \eta_L)$ with a proof of the non-degenerate limit for pre-LN transformer blocks of internal depth $\geq 2$ with LayerNorm and Adam, plus empirical hyperparameter transfer verified across at least a $16\times$ depth range at $\geq 10^9$ parameters.

## 2. Formal Setting

Let $x_\ell \in \mathbb{R}^n$ be the residual-stream activation at block $\ell \in \{0,\dots,L\}$, width $n$, and

$$x_{\ell+1} = x_\ell + \alpha_L\, \mathcal{F}_\ell(x_\ell; \theta_\ell), \qquad \alpha_L = L^{-\beta}, \ \beta \geq 0 .$$

$\mathcal{F}_\ell$ is a block of *internal depth* $d$ (an MLP with $d$ weight matrices, or attention+MLP). Weights are initialized $\theta_\ell \sim \mathcal{N}(0, \sigma^2/n)$ (fan-in / He scaling), so that $\mathbb{E}\|\mathcal{F}_\ell(x)\|^2 / n = \Theta(\|x\|^2/n)$.

Measured quantities, as they would actually be logged:

- **Stream growth.** $G_L = \frac{1}{n}\mathbb{E}\|x_L\|^2 \big/ \frac{1}{n}\mathbb{E}\|x_0\|^2$, estimated from activation norms on a held-out batch at step 0. Under independence of blocks, $G_L \approx (1 + \alpha_L^2 c)^L$, so $\beta = 1/2$ gives $G_L \to e^{c}$ and $\beta < 1/2$ gives $G_L \to \infty$.
- **Feature learning.** $\Delta_\ell(t) = \frac{1}{\sqrt n}\|x_\ell(t) - x_\ell(0)\|$ on a fixed probe batch after $t$ steps. Non-degeneracy requires $\Delta_L(t) = \Theta(1)$ in both $n$ and $L$.
- **Block diversity.** $\rho_L = \mathbb{E}_{\ell \neq \ell'} \cos\big(\delta_\ell, \delta_{\ell'}\big)$ where $\delta_\ell = \mathcal{F}_\ell(x_\ell;\theta_\ell(t)) - \mathcal{F}_\ell(x_\ell;\theta_\ell(0))$. Degenerate (ODE-like) limits have $\rho_L \to 1$; the useful limit keeps $\rho_L$ bounded away from $1$.
- **Transfer.** $\eta^\star(L) = \arg\min_\eta \mathcal{L}_{\text{val}}$ on a log-spaced sweep (typically 7 points, factor-2 spacing). Transfer holds iff $\eta^\star(L)$ is constant in $L$ up to one grid step.

Assumptions, with the ones known to break in practice flagged:

1. Blocks are i.i.d. and independent of the stream — **violated** after a few steps of training; correlations are exactly what $\rho_L$ measures.
2. $n \to \infty$ before or jointly with $L$ — **violated**: at $n = 4096$, $L = 128$ the ratio $L/n$ is not small enough for the width-first limit to be tight (Li, Nica & Roy, NeurIPS 2021, show the joint limit is log-Gaussian, not Gaussian).
3. No normalization inside the block — **violated**: LayerNorm rescales $\mathcal{F}_\ell$'s input and partly cancels the $\alpha_L$ the theory prescribes.
4. SGD-like updates — **violated**: Adam's sign-like normalization changes the correct $\eta_L$ exponent relative to SGD.

## 3. State of the Art

**Established (proved or independently reproduced).**
- $\beta = 1/2$ ("$1/\sqrt{L}$ branch scaling") gives a finite NTK and stable forward/backward signal at initialization: *Stable ResNet* (Hayou, Clerico, He, Deligiannidis, Doucet, Rousseau; AISTATS 2021).
- Depth-$\mu$P (Yang, Yu, Zhu, Hayou, *Tensor Programs VI*, 2023): for **block internal depth $d = 1$**, the pair $\alpha_L = L^{-1/2}$, per-layer learning rate $\eta_\ell \propto L^{-1/2}$ is the *unique* scaling admitting feature learning and diversity in the $L\to\infty$ limit. Same paper proves that for $d \geq 2$ **no** power-law scaling gives hyperparameter transfer — a negative result, not a gap.
- Zero-initializing the branch output (ReZero, Bachlechner et al., UAI 2021; SkipInit, De & Smith, NeurIPS 2020; Fixup, Zhang, Dauphin & Ma, ICLR 2019) removes the need for normalization to control depth-wise growth.

**Claimed but not fully ablated.**
- DeepNet (Wang et al., 2022) sets residual scale $\alpha = (2L)^{1/4}$-style constants derived from a model-update bound and trains a 1000-layer encoder-decoder; the ablation isolating $\alpha$ from the accompanying init-downscaling is thin.
- LayerNorm-scaling fixes for the "curse of depth" in pre-LN LLMs (Sun et al., 2025) report better use of deep layers, measured mainly as pretraining perplexity at $\leq 1$B parameters.
- **Benchmark-number-only:** DeepNet's $+4.4$ BLEU on OPUS-100 multilingual translation is a single-benchmark result, not a controlled depth-scaling curve.

## 4. What Is Known

- **Fixup** (ICLR 2019) trains a 110-layer ResNet on CIFAR-10 to $\approx 5\%$ test error and a 10,000-layer network to convergence with no normalization, by scaling residual-branch weights by $L^{-1/(2d-2)}$ and zeroing the last layer of each block.
- **SkipInit** (NeurIPS 2020): a single learnable scalar initialized to $0$ per branch recovers batch-norm-level trainability on ResNet-1000 at CIFAR-10 scale; the paper's finding is that BN's main depth effect is *downscaling the residual branch*, not variance control.
- **ReZero** (UAI 2021): $\alpha$ as a learned scalar starting at $0$; 12-layer transformer converges $\approx 56\%$ faster in steps on enwiki8, and a 128-layer transformer trains without warmup.
- **Depth-$\mu$P**: empirical transfer of $\eta^\star$ across $L \in \{2,\dots,64\}$ for $d=1$ residual MLPs and small transformers; measured non-transfer for $d=2$ blocks over the same range.
- **Joint limit**: at $L/n$ fixed, pre-activation norms are log-normal with variance growing as $L/n$ (Li, Nica, Roy, NeurIPS 2021) — so the width-first analysis is quantitatively wrong at $L \approx n$, e.g. $L=96$, $n=1024$.
- **Transformers specifically**: rank collapse in attention is depth-driven and $\alpha_L$ shifts where it sets in (Noci et al., NeurIPS 2022); $\sigma^2_{\text{res}} \propto 1/L$ delays the collapse in models up to $\approx 300$M parameters.

## 5. What Is Not Known

- **Theoretically open.** No non-degenerate $L\to\infty$ limit is known for blocks with internal depth $d \geq 2$ under any $\alpha_L$ — Depth-$\mu$P proves power laws fail but does not rule out non-power-law or per-block-heterogeneous schedules. No limit theory covers LayerNorm-inside-the-block plus Adam jointly.
- **Empirically open.** Whether $\beta = 1/2$ with $\eta \propto L^{-1/2}$ actually transfers for a standard pre-LN transformer ($d=2$ MLP, attention, RMSNorm, AdamW) at $L=16 \to 128$ and $\geq 10^9$ parameters. Runnable today; nobody has published the full grid at that scale.
- **Methodologically blocked.** "Depth helps" has no agreed metric at fixed compute. Comparing $L=32$ and $L=128$ at equal parameters changes width, attention-head count, and tokens-per-parameter simultaneously; loss-vs-depth curves in the literature are not controlled for this, so the *outcome variable itself* is ill-defined.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a quadratic sweep cost**. Deciding the exponent requires a 2-D sweep over $(\eta, L)$ — 7 learning rates $\times$ 5 depths $\times$ 3 seeds $=105$ runs — at a scale where the width-first limit is not already tight, i.e. $L \gtrsim n/8$. At $n=2048$ that means $L \geq 256$ and $\gtrsim 10^{20}$ FLOPs for the grid. Cheaper proxies fail for a reason that is not budgetary: at small $L$, $\alpha_L$ variants differ by $O(1)$ constants that are absorbed into $\eta^\star$, so any $L \leq 16$ sweep cannot separate $\beta = 1/2$ from $\beta = 1$ — the exponent is **non-identifiable** in the regime that is affordable. Additionally, LayerNorm makes $\alpha_L$ partially unidentifiable at initialization: scaling the branch weights and scaling the branch output are the same function until the norm's learnable gain moves, so two "different" schemes are the same model at step 0 and diverge only through the optimizer.

## 7. Current Research (as of 2026)

- **Depth-$\mu$P extensions** to attention and to $d \geq 2$ blocks — Hayou (Oxford/NUS), Yang (xAI), and collaborators; the open direction is heterogeneous per-block scaling rather than a single power law *(frontier — verify)*.
- **Normalization redesign as depth control**: LayerNorm scaling / $1/\sqrt{\ell}$-style per-depth gains reported to unlock deep-layer contribution in pre-LN LLMs (Sun et al., 2025 and follow-ups) *(frontier — verify)*.
- **Geometric signal-propagation predictors** of trainability (Cowsik, Nakamura-Zimmerer, Hanin et al., 2024) aiming to replace grid search with an initialization-time statistic.
- **Joint width–depth limits**: characterizing when the two limits commute (Hayou & Yang, ICML 2023) and what breaks when they do not.
- Practical deep-stack recipes in production LLMs (e.g. very deep, narrow models) where $\alpha_L$ is folded into per-layer init constants and rarely reported.

## 8. Concrete Next Experiment

**Scale.** Pre-LN decoder transformer, $n = 1024$, standard $d=2$ SwiGLU MLP, RMSNorm, AdamW, 20B tokens of a fixed corpus. Depths $L \in \{16, 32, 64, 128\}$ with tokens and *total non-embedding parameters held fixed* by shrinking the MLP ratio, so depth is the only varying axis. Peak learning rate swept over 7 log-spaced points, 2 seeds. Cost: $\approx 112$ runs of $\approx 1.5\times10^{19}$ FLOPs each.

**Arms.** (a) $\alpha_L = 1$ with standard init — control. (b) $\alpha_L = L^{-1/2}$, global $\eta$. (c) $\alpha_L = L^{-1/2}$ with residual-branch learning rate $\eta_\ell \propto L^{-1/2}$ (Depth-$\mu$P as literally prescribed). (d) ReZero: learned scalar at $0$.

**Deciding number.** $R = \eta^\star(128)/\eta^\star(16)$ per arm, on the shared grid. Transfer holds iff $R \in [0.5, 2]$ (one grid step). Report alongside $\mathcal{L}_{\text{val}}(128) - \mathcal{L}_{\text{val}}(16)$ at each arm's own optimum. The claim under test — that Depth-$\mu$P's $d=1$ non-transfer proof is vacuous in practice for $d=2$ transformers — is refuted if arm (c) gives $R \notin [0.5,2]$ while arm (a) does not, and supported if $R \approx 1$ for (c) and $R \leq 0.25$ for (a).

## 9. Key References

- **[Foundational]** Hongyi Zhang, Yann N. Dauphin, Tengyu Ma. *Fixup Initialization: Residual Learning Without Normalization.* ICLR 2019. — arXiv:1901.09321
- **[Foundational]** Soham De, Samuel L. Smith. *Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks.* NeurIPS 2020. — arXiv:2002.10444
- **[Foundational]** Thomas Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao, Garrison W. Cottrell, Julian McAuley. *ReZero is All You Need: Fast Convergence at Large Depth.* UAI 2021. — arXiv:2003.04887
- **[SOTA]** Greg Yang, Dingli Yu, Chen Zhu, Soufiane Hayou. *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks.* 2023. — arXiv:2310.02244
- **[SOTA]** Soufiane Hayou, Eugenio Clerico, Bobby He, George Deligiannidis, Arnaud Doucet, Judith Rousseau. *Stable ResNet.* AISTATS 2021. — arXiv:2010.12859
- **[SOTA]** Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, Furu Wei. *DeepNet: Scaling Transformers to 1,000 Layers.* 2022. — arXiv:2203.00555
- Mufan Bill Li, Mihai Nica, Daniel M. Roy. *The Future is Log-Gaussian: ResNets and Their Infinite-Depth-and-Width Limit at Initialization.* NeurIPS 2021.
- Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, Aurelien Lucchi. *Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse.* NeurIPS 2022.
- Soufiane Hayou, Greg Yang. *Width and Depth Limits Commute in Residual Networks.* ICML 2023.
- Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, Jiawei Han. *Understanding the Difficulty of Training Transformers.* EMNLP 2020.
- **[Survey]** Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, Jascha Sohl-Dickstein. *Deep Information Propagation.* ICLR 2017.

## 10. Worked Example

Take $L = 100$ blocks, each with $\mathbb{E}\|\mathcal{F}_\ell(x)\|^2/n = \|x\|^2/n$ at init (He scaling), blocks independent.

- **$\beta = 0$** ($\alpha=1$): $G_{100} = 2^{100} \approx 1.3\times10^{30}$. Logits saturate; gradients through the last block are $2^{100}$ times those through the first. Unusable without normalization.
- **$\beta = 1/2$**: $\alpha^2 = 0.01$, $G_{100} = 1.01^{100} = 2.70 \to e$. Bounded, depth-independent. Backward pass symmetric.
- **$\beta = 1$**: $\alpha^2 = 10^{-4}$, $G_{100} = 1.0001^{100} = 1.010$. Stable, but each block contributes $O(1/L)$ — the ODE limit; measured $\rho_L \to 1$, blocks become copies of one vector field, and added depth buys nothing.

Now the obstruction. Run the same three at $L = 8$: $G_8 = 256$, $2.14$, $1.13$. Sweep $\eta$ and all three arms hit essentially the same best validation loss, because a constant rescaling of the stream at $L=8$ is absorbed by the first LayerNorm and by a $\approx 2\times$ shift in $\eta^\star$ — within one grid step. The three hypotheses are **indistinguishable at the depth you can afford to sweep**. They separate only where $G_L$ diverges super-polynomially or $\rho_L$ saturates, i.e. $L \gtrsim 64$ — and there a 7-point $\eta$ sweep with 2 seeds is already $\sim 10^{19}$ FLOPs per depth. That gap between where the effect is identifiable and where the sweep is cheap is the whole problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*