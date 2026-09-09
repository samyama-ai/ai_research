---
id: 03-training-dynamics/mup-beyond-width
title: "Maximal Update Parametrization Beyond Width"
topic: 03-training-dynamics
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Maximal Update Parametrization Beyond Width

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/mup-beyond-width` · **Status:** partially-solved

## 1. Problem Statement

μP (maximal update parametrization) prescribes how initialization variance, per-layer learning rates, and output multipliers must scale with **width** $n$ so that a hyperparameter tuned on a small proxy model is optimal on a large target model. Width transfer works. The open problem is every other scaling axis.

Input: a proxy model with width $n_0$, depth $L_0$, batch size $B_0$, token budget $D_0$, trained under some optimizer, and its tuned hyperparameter vector $\theta^\star_0$ (peak LR, warmup, weight decay, output multiplier, $\epsilon$).
Output: a map $\theta^\star_0 \mapsto \theta^\star_1$ for a target $(n_1, L_1, B_1, D_1)$.
Decision predicate: the transferred $\theta^\star_1$ lands within a tolerance $\delta$ of the loss achieved by directly tuning at the target scale.

Three variants, different difficulty:

- **Theory.** Does a nontrivial joint $(n, L)$ or $(n, D)$ limit exist in which per-layer feature updates are $\Theta(1)$ and the optimal hyperparameters are scale-free? Settled for width; partially settled for depth in residual nets; open for token horizon.
- **Method.** A concrete scaling rule that practitioners apply. Exists for width and for $1/\sqrt{L}$ residual branches; nothing established for $D$.
- **Measurement.** What counts as "the optimal LR transferred"? The loss-vs-LR curve is flat near its minimum, and the flatness itself grows with scale. This is where most disagreement lives.

## 2. Formal Setting

Network $f(x;\vartheta)$ with $L$ residual blocks of width $n$. Layer $\ell$ has weights $W^\ell \in \mathbb{R}^{n\times n}$. An **abc-parametrization** (Yang & Hu, ICML 2021) fixes three exponents per layer:

$$W^\ell = n^{-a_\ell}w^\ell,\qquad w^\ell_{ij}\sim\mathcal{N}(0, n^{-2b_\ell}),\qquad \eta_\ell = \eta\, n^{-c_\ell}.$$

**Measured quantities.**

- Feature update size: $\Delta h^\ell_t = h^\ell_t - h^\ell_0 \in \mathbb{R}^n$, measured as the RMS coordinate $\|\Delta h^\ell_t\|_2/\sqrt{n}$ over a fixed held-out batch. μP is the choice of $(a,b,c)$ making this $\Theta_n(1)$ for every $\ell$ — no layer's contribution vanishing or blowing up as $n\to\infty$.
- Alignment: $\rho_\ell = \dfrac{\|\Delta W^\ell h^{\ell-1}\|_2}{\|\Delta W^\ell\|_F\,\|h^{\ell-1}\|_2/\sqrt{n}}$, measured directly from activations. μP's derivation assumes $\rho_\ell = \Theta(1)$ (full alignment); the null assumption is $\rho_\ell = \Theta(n^{-1/2})$.
- Transfer error: with $\mathcal{L}_S(\eta)$ the final loss at scale $S$, define $\eta^\star_S = \arg\min_\eta \mathcal{L}_S(\eta)$ over a log-spaced grid, and the **transfer gap** $\;g = \mathcal{L}_{S_1}(\eta^\star_{S_0}) - \mathcal{L}_{S_1}(\eta^\star_{S_1})$, in nats/token. This — not $|\log \eta^\star_1 - \log \eta^\star_0|$ — is the quantity that matters.
- Sharpness: $\lambda_{\max}(\nabla^2\mathcal{L})$ by power iteration on a fixed batch; a proposed mechanistic explanation of transfer (Noci et al., ICML 2024).

**Assumptions known to be violated in practice.**

1. $n\to\infty$ at fixed $L$, $B$, $D$. Real scaling moves all four together (Chinchilla-style), so the limit taken is not the limit used.
2. Full alignment $\rho_\ell=\Theta(1)$. Everett et al. (ICML 2024) measured intermediate, layer- and time-dependent alignment.
3. Adam is scale-invariant in the gradient. False once $\epsilon>0$: as $n$ grows, per-coordinate gradients shrink and $\epsilon$ stops being negligible, silently changing the effective parametrization.
4. Fixed number of steps. The limit theorems hold for $t$ fixed as $n\to\infty$; production runs grow $t$ with $n$.
5. No weight decay, or decoupled weight decay treated as scale-free. Decoupled AdamW introduces a second timescale $\eta\lambda$ that is not covered by the width derivation.

## 3. State of the Art

**Theory SOTA (established).**
- Yang & Hu, *Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks* (ICML 2021): classification of abc-parametrizations into kernel-regime and feature-learning regime; μP is the unique maximal one.
- Yang & Littwin, *Tensor Programs IVb: Adaptive Optimization in the ∞-Width Limit* (ICLR 2023): extends the limit to Adam-family optimizers.
- Yang, Yu, Zhu, Hayou, *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks* (ICLR 2024): "Depth-μP" — residual branch multiplier $1/\sqrt{L}$ plus per-block LR scaling $\eta_\ell \propto L^{-1/2}$ gives a well-defined infinite-depth feature-learning limit **for block depth 1**. The paper itself reports that transfer degrades for blocks of depth $\geq 2$ — i.e. for real transformer blocks.
- Hayou & Yang, *Width and Depth Limits Commute in Residual Networks* (ICML 2023): with $1/\sqrt{L}$ branch scaling the two limits commute, so the order of taking $n,L\to\infty$ does not matter.

**Empirical SOTA (established).**
- Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (NeurIPS 2021): μTransfer from a 40M proxy to GPT-3 6.7B, matching a 13B baseline at total tuning cost ≈7% of pretraining.
- Everett et al., *Scaling Exponents Across Parameterizations and Optimizers* (ICML 2024): the largest controlled sweep, tens of thousands of runs up to ~26.8B parameters. Established: (a) several parametrizations beyond μP achieve LR transfer once per-layer LRs are allowed; (b) Adam's $\epsilon$ must be scaled or removed (their `Adam-atan2` variant); (c) the alignment assumption underlying μP is not empirically satisfied as stated.

**Claimed but unablated.**
- That depth-μP transfers on production transformers. Reported at modest depth on small models; no independent replication at $L>100$ with modern blocks.
- u-μP (Blake et al., 2024), combining unit scaling with μP for FP8: promising loss/LR-robustness curves, but results are single-lab and mostly ≤7B.

**Benchmark-number-only.** Cerebras-GPT (Dey et al., 2023) and BTLM-3B-8K report using μP and cite improved loss-per-FLOP; these are model-release numbers, not controlled μP-vs-SP ablations at fixed compute.

## 4. What Is Known

- **Width transfer is real.** Tensor Programs V: optimal LR is stable across widths from 128 to 4096 on transformers on Wikitext-103, and proxy-to-target transfer at 6.7B improved on the tuned baseline.
- **Depth transfer needs $1/\sqrt{L}$.** Without branch scaling, optimal LR drifts monotonically with $L$; with it, drift largely disappears for depth-1 blocks (Tensor Programs VI, ICLR 2024).
- **Sharpness is the mechanism, plausibly.** Noci et al. (ICML 2024) show $\lambda_{\max}$ at the end of training is roughly width-independent under μP but grows under standard parametrization, on transformers and ResNets up to a few hundred million parameters — which would explain why the same $\eta$ stays at the edge of stability.
- **$\epsilon$ breaks transfer.** Everett et al. show that at fixed $\epsilon=10^{-8}$, transfer visibly degrades past ~1B parameters; the failure is arithmetic, not conceptual.
- **μP does not remove instabilities.** Wortsman et al., *Small-Scale Proxies for Large-Scale Transformer Training Instabilities* (ICLR 2024): attention-logit growth and output-logit divergence still occur; qk-layernorm and z-loss extend the stable LR range independently of parametrization.
- **Some components do not transfer.** Lingle, *A Large-Scale Exploration of μ-Transfer* (2024, up to ~1.2B): LR transfer is robust across many architectural changes, but trainable per-layer gains and some norm/decay choices break it.
- **Token horizon shifts the optimum.** Optimal LR decreases as the token budget $D$ grows at fixed model size — reported by Bjorck et al. (Microsoft, 2024) and consistent with Chinchilla-era practice. μP says nothing about $D$.

## 5. What Is Not Known

- **Theoretically open.** No joint $(n, L, D)$ limit with $\Theta(1)$ feature learning in all three. Depth-μP for blocks of depth $\geq 2$ (i.e. real MLP+attention blocks) has no limit theorem. No parametrization theory for MoE, where the relevant width is per-expert but the routing is discrete; no theory covering weight decay's $\eta\lambda$ timescale.
- **Empirically open.** Whether a $D$-dependent correction $\eta^\star \propto D^{-\gamma}$ has a universal exponent $\gamma$: runnable today at 1B×(20–500) tokens/param, unrun as a controlled sweep. Whether depth transfer survives $L=32\to 128$ with modern blocks at fixed width. Whether μP still helps once per-layer LRs are tuned in standard parametrization — Everett's result suggests much of μP's benefit is recoverable.
- **Methodologically blocked.** "Optimal LR" is not well defined. The loss-vs-$\log\eta$ curve is near-quadratic and shallow; the minimizer's location has estimation error comparable to the grid spacing, while the *loss* penalty for being wrong varies by orders of magnitude across scales. Papers report drift in $\log\eta^\star$; practitioners care about the transfer gap $g$. These can disagree in sign of importance.

## 6. Why It Is Hard

Three specific obstructions.

1. **Confounded measurement.** LR interacts with warmup, schedule shape, weight decay, $\epsilon$, and batch size. A drift attributed to depth may be a schedule artifact: shorter runs at larger $L$ change the fraction of training spent in warmup. Almost no published μP study holds all five fixed while varying one axis.
2. **Non-identifiability at the minimum.** Near $\eta^\star$, $\mathcal{L}(\eta)\approx \mathcal{L}^\star + \tfrac{\kappa}{2}(\log\eta - \log\eta^\star)^2$ with $\kappa$ itself scale-dependent. With seed noise $\sigma$, the minimizer is identifiable only to $\pm\sqrt{2\sigma/\kappa}$ in $\log\eta$ — often $\pm 0.3$ nats of $\log_2\eta$, comparable to the drift being measured.
3. **Compute cost of the only decisive test.** Deciding whether transfer holds requires the control arm — a *direct* tuning sweep at the target scale. That is exactly the cost μP exists to avoid. So the claim "μP transfers to 100B" is structurally almost never tested; it is assumed and then a single run is reported.

## 7. Current Research (as of 2026)

- **Per-layer exponent search** (Google DeepMind, following Everett et al.): treating $(a_\ell,b_\ell,c_\ell)$ as fit parameters rather than derived ones, and reporting alignment exponents measured from activations.
- **Numerics-aware parametrization**: u-μP (Graphcore) and FP8/MX-format training, where the parametrization must also keep every tensor inside a narrow exponent range.
- **Dynamical mean-field theory of scaling**: Bordelon & Pehlevan (Harvard) — DMFT of kernel evolution (NeurIPS 2022) and a dynamical model of neural scaling laws (ICML 2024) — gives width, depth and *time* in one framework, and is the most likely route to a $D$-axis result. *(frontier — verify)*
- **Batch-size axis**: the $\sqrt{B}$ rule for Adam (Malladi et al., NeurIPS 2022) composed with μP; whether critical batch size grows with $D$ under μP is actively contested. *(frontier — verify)*
- **μP for MoE and for attention variants** (GQA, latent attention): several industrial labs report internal recipes; no published controlled ablation. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does the optimal peak LR under μP depend on the token horizon $D$, and with what exponent?

**Scale.** Decoder-only transformer, μP with $1/\sqrt{L}$ branch scaling, $L=24$, widths $\{512, 1024, 2048\}$ (≈40M–800M non-embedding params). Token budgets $D \in \{4, 20, 100, 400\}$ tokens/parameter — four horizons spanning 100×. Fixed batch size, fixed cosine schedule shape (decay to 10% of peak, warmup a fixed 500 steps, not a fixed fraction), AdamW with $\epsilon = 10^{-15}$ or `atan2`, weight decay fixed at $\lambda = 0$ in arm A and $\eta\lambda$ held constant in arm B. LR grid: 7 points, half-octave spacing. 3 seeds at the two grid points nearest the minimum. ≈250 runs; ~15k A100-hours.

**Control arm.** The same grid in standard parametrization with a single global LR — the parametrization μP claims to beat — plus a second control that holds $D$ fixed and varies width only, to confirm the known width-transfer result reproduces in this pipeline.

**The deciding number.** Fit $\log_2 \eta^\star(n, D) = \alpha - \gamma \log_2 D$ and report $\gamma$ with a bootstrap CI over seeds. If the CI for $\gamma$ excludes 0 by more than the identifiability floor ($\pm 0.3$ in $\log_2\eta$ over 100× in $D$, i.e. $|\gamma| > 0.045$), μP is incomplete on the data axis and $\gamma$ is the correction exponent. Report alongside it the transfer gap $g$ in nats/token from using the $D=4$ optimum at $D=400$: if $g < 0.005$ nats/token, the drift is real but practically irrelevant, which is itself a publishable answer.

## 9. Key References

- **[Foundational]** Greg Yang, Edward J. Hu. *Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks.* ICML, 2021. — arXiv:2011.14522
- **[Foundational]** Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[SOTA]** Greg Yang, Dingli Yu, Chen Zhu, Soufiane Hayou. *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks.* ICLR, 2024. — arXiv:2310.02244
- **[SOTA]** Katie Everett, Lechao Xiao, Mitchell Wortsman, Alexander A. Alemi, Roman Novak, Peter J. Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling, Jaehoon Lee, Jeffrey Pennington. *Scaling Exponents Across Parameterizations and Optimizers.* ICML, 2024. — arXiv:2407.05872
- **[Theory]** Greg Yang, Etai Littwin. *Tensor Programs IVb: Adaptive Optimization in the ∞-Width Limit.* ICLR, 2023.
- **[Theory]** Soufiane Hayou, Greg Yang. *Width and Depth Limits Commute in Residual Networks.* ICML, 2023.
- **[Mechanism]** Lorenzo Noci, Alexandru Meterez, Thomas Hofmann, Antonio Orvieto. *Why Do Learning Rates Transfer? Reconciling Optimization and Scaling Limits for Deep Learning.* ICML, 2024.
- **[Empirical]** Mitchell Wortsman et al. *Small-Scale Proxies for Large-Scale Transformer Training Instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[Empirical]** Lucas Lingle. *A Large-Scale Exploration of μ-Transfer.* 2024. — arXiv:2404.05728
- **[Numerics]** Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Y. Prince, Björn Deiseroth, Andres Felipe Cruz-Salinas, Carlo Luschi, Samuel Weinbach, Douglas Orr. *u-μP: The Unit-Scaled Maximal Update Parametrization.* 2024. — arXiv:2407.17465
- **[Theory/adjacent]** Blake Bordelon, Cengiz Pehlevan. *Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks.* NeurIPS, 2022.
- **[Survey]** Blake Bordelon, Alexander Atanasov, Cengiz Pehlevan. *A Dynamical Model of Neural Scaling Laws.* ICML, 2024.

## 10. Worked Example

Take the identifiability obstruction concretely. Two μP transformers, $n_0=256$ and $n_1=2048$ (8× width), $L=12$, trained on 4B tokens. Run a 5-point LR grid at half-octave spacing, 3 seeds each.

Proxy ($n_0$): losses at $\log_2\eta \in \{-11,-10.5,-10,-9.5,-9\}$ are $3.412, 3.397, 3.391, 3.398, 3.421$. Seed std $\sigma = 0.004$. Fitting the quadratic gives $\log_2\eta^\star_0 = -10.03$, curvature $\kappa \approx 0.24$ nats per $(\log_2\eta)^2$. Identifiability radius $\sqrt{2\sigma/\kappa} = \sqrt{0.008/0.24} \approx 0.18$ — so $\eta^\star_0$ is pinned only to $\pm 0.18$ octaves.

Target ($n_1$): the true minimum sits at $\log_2\eta^\star_1 = -10.35$, curvature $\kappa \approx 0.09$ — flatter, because larger models are more LR-tolerant near the optimum.

- Apparent drift: $|{-10.35} - ({-10.03})| = 0.32$ octaves. This looks like a transfer failure: it exceeds the $0.18$ identifiability radius of the proxy measurement, and a paper reporting $\log\eta^\star$ drift would call it one.
- Actual cost: $g = \tfrac{\kappa}{2}(0.32)^2 = \tfrac{0.09}{2}(0.1024) \approx 0.0046$ nats/token. At 4B tokens that is worth roughly a 1.5% shift in effective data — below the run-to-run spread of most published comparisons.

The obstruction is visible here: the same experiment supports "μP fails on this axis" (drift is statistically resolvable) and "μP works fine" (the loss penalty is negligible), because $\kappa$ shrinks as scale grows while the drift does not. Any claim about μP beyond width must report $g$ in nats/token, not drift in $\log\eta^\star$ — and almost none of the literature does. That, more than compute, is why the axis is unresolved.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*