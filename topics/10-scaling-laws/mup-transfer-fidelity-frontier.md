---
id: 10-scaling-laws/mup-transfer-fidelity-frontier
title: "muP Transfer Fidelity at Frontier Scale"
topic: 10-scaling-laws
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# muP Transfer Fidelity at Frontier Scale

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/mup-transfer-fidelity-frontier` · **Status:** empirically-open

## 1. Problem Statement

The Maximal Update Parametrization (muP) rescales initialization, per-layer learning rates, and output multipliers so that the optimal hyperparameters of a proxy model transfer unchanged to a much wider target model. In practice a lab tunes at $10^{7}$–$10^{8}$ parameters and copies the learning rate to $10^{11}$–$10^{12}$.

The problem: **quantify the loss penalty incurred by that copy, at the width ratios and token budgets actually used at the frontier, and determine whether the penalty grows with scale.**

Three variants, different difficulty:

- **Measurement.** Define and estimate the *transfer gap* — final loss under transferred hyperparameters minus final loss under hyperparameters tuned directly at the target — separating it from seed noise and from data-order effects. Currently the gap is almost never measured at target scale, because measuring it requires the sweep muP exists to avoid.
- **Method.** Find a parameterization whose transfer gap stays flat as width, depth, batch size, token budget, and vocabulary all grow together — the realistic joint limit, not the width-only limit muP was derived in.
- **Theory.** Prove that the finite-width correction to the optimal learning rate vanishes at a stated rate, e.g. $\eta^{*}(n) = \eta^{*}_{\infty}(1 + \Theta(n^{-\gamma}))$ with $\gamma > 0$, for Adam on a transformer with LayerNorm, weight decay, and a nonzero $\epsilon$. No such theorem exists.

Solved means: a published rate $\gamma$, or a measured gap curve over at least three target scales spanning $\ge 100\times$ compute with the tuned control arm actually run.

## 2. Formal Setting

Let $n$ be model width (`d_model`), $L$ depth, $D$ tokens, $B$ batch size in tokens, $N(n,L)$ non-embedding parameters, $C \approx 6ND$ training FLOPs.

**Parameterization.** In the abc-parameterization, layer $l$ has weights $W_l = n^{-a_l} w_l$, with $w_l \sim \mathcal{N}(0, n^{-2b_l})$ and learning rate $\eta_l = \eta\, n^{-c_l}$. muP fixes $(a_l,b_l,c_l)$ so that every layer's activations change by $\Theta(1)$ per step in the $n\to\infty$ limit. For Adam-family optimizers the standard table is:

$$
\text{embed: } \sigma^2=\Theta(1),\ \eta_l=\Theta(1); \quad
\text{hidden: } \sigma^2=\Theta(1/n),\ \eta_l=\Theta(1/n); \quad
\text{readout: } \sigma^2=\Theta(1/n^2),\ \eta_l=\Theta(1/n),
$$

with attention logits scaled $1/d_{\text{head}}$ rather than $1/\sqrt{d_{\text{head}}}$.

**Transfer gap, as measured.** Fix a proxy width $n_0$ and a hyperparameter vector $\theta = (\eta, \text{init scale}, \text{multipliers}, \dots)$. Let $\hat\theta(n_0)$ minimize validation loss at $n_0$ over a grid, and let $T_{n_0\to n}$ be the muP map. Then

$$
\Delta(n) \;=\; \mathbb{E}_s\big[L_n\big(T_{n_0\to n}\hat\theta(n_0); s\big)\big] \;-\; \min_{\theta}\ \mathbb{E}_s\big[L_n(\theta; s)\big],
$$

expectation over seeds $s$ (init and data order). Both terms are held-out cross-entropy in nats/token on a fixed validation set, at matched $D$, matched tokenizer, matched data order distribution. $\Delta(n) \ge 0$ by construction; the estimator is *biased upward* by grid coarseness in the second term and *downward* by seed noise, so a usable estimate needs $\ge 3$ seeds per arm and a grid spacing of $\le 2\times$ in $\eta$.

**Interpretable units.** Convert nats to compute using the local scaling law slope. With $L(C) = E + A C^{-\alpha}$ and $\alpha \approx 0.05$ (Hoffmann et al., 2022 regime), the compute multiplier equivalent to a gap $\Delta$ is

$$
\rho = \exp\!\left(\frac{\Delta}{\alpha\,(L - E)}\right).
$$

**Assumptions, and which are violated.**
1. *Width is the only growing dimension.* Violated — frontier runs grow $L$, $B$, $D$, and vocabulary jointly.
2. *Alignment*: the update $\Delta W_l$ and the incoming activation are asymptotically non-aligned, giving $\Theta(1)$ coordinate change. Everett et al. (2024) report empirical alignment exponents that differ from the muP-assumed values, so per-layer exponents inferred empirically do not match the derivation.
3. *Adam is scale-invariant.* Violated: $\epsilon > 0$ breaks it once per-coordinate gradients fall below $\epsilon$, which happens at large $n$ precisely because gradients shrink.
4. *Loss is smooth and unimodal in $\log\eta$.* Violated near the instability edge (attention-logit growth, loss spikes).
5. *Optimum is interior.* Weight decay, gradient clipping, and $z$-loss all interact with $\eta$ and are not part of the muP map.

## 3. State of the Art

**Theory SOTA.** Tensor Programs IV (Yang & Hu, ICML 2021) establishes the infinite-width feature-learning limit; Tensor Programs V (Yang et al., NeurIPS 2021) derives muTransfer for SGD and Adam; Tensor Programs IVb (Yang & Littwin, ICLR 2023) extends the limit to adaptive optimizers. These are limit theorems: they state that $\eta^{*}(n)$ converges, not *how fast*. **No finite-width rate is proven.** Depth-muP (Yang, Yu, Zhu, Hayou, ICLR 2024) gives a $1/\sqrt{L}$ residual-branch scaling for depth transfer; Bordelon, Atanasov & Pehlevan (ICLR 2024) derive a compatible depthwise limit for residual networks.

**Empirical SOTA.** *Established:* Yang et al. (2021) transferred from a 13M proxy to a 6.7B GPT-3 model and reported it outperforming the released 6.7B baseline, with tuning cost about 7% of one pretraining run. Cerebras-GPT (Dey et al., 2023) applied muP across a 111M–13B family. Lingle (2024) ran the largest independent replication published, up to roughly 1.2B parameters, and found width transfer of the learning rate holds while several common tricks break it. Everett et al. (ICML 2024) swept parameterizations and optimizers and found per-layer learning-rate exponents that beat standard muP on their grid.

*Claimed but unablated:* the industrial claim that muP "just works" at $10^{11}$+ parameters. Every large deployment (MiniCPM, Tele-FLM, and various frontier models) reports using muP; none publishes the tuned control arm at target scale. Without that arm the reported number is a benchmark number, not a measurement of $\Delta$.

## 4. What Is Known

- **Optimal $\log_2 \eta$ is flat in width over ~2 decades.** Yang et al. (2021) show the loss-vs-$\eta$ curve minimum stationary from width 256 to 8192 on transformers; the 6.7B transfer used a 13M proxy, a $\sim500\times$ parameter ratio.
- **Tuning cost.** The muTransfer sweep for the 6.7B model cost about **7% of a single pretraining run** — the number that makes muP economically attractive.
- **Standard parameterization does not transfer.** Under SP, optimal $\eta$ drifts roughly as $1/n$ across the same width range; this is the reproduced control.
- **Specific breakages (Lingle, 2024, $\le$ 1.2B).** Trainable per-layer gain/scale parameters, and multiplicative-style modifications, degrade or destroy transfer; the $1/d_{\text{head}}$ attention scaling matters; decoupled weight decay must be handled as a separate axis rather than transferred.
- **Adam $\epsilon$ is not neutral.** Everett et al. (2024) show that at large width the default $\epsilon = 10^{-8}$ is not small relative to per-coordinate gradient magnitudes, and propose an $\epsilon$ scaling; u-muP (Blake et al., 2024) makes the same point via unit scaling and reports muP-competitive or better loss with hyperparameters that are more separable.
- **Instabilities are width-dependent.** Wortsman et al. (ICLR 2024) reproduce attention-logit growth and output-logit divergence at small scale and show muP narrows but does not eliminate the sensitive region.

## 5. What Is Not Known

- **Theoretically open.** The finite-width correction rate. No proof of $|\eta^{*}(n) - \eta^{*}_{\infty}| = O(n^{-\gamma})$ for any $\gamma$, for Adam with $\epsilon>0$, LayerNorm, weight decay, and clipping. Also open: whether $\Delta(n)$ is monotone, or has an interior maximum at the widths labs actually use.
- **Theoretically open.** The joint limit. muP is a width limit; depth-muP is a depth limit. Whether the two compose, and how batch-size and token-budget scaling interact with both, has no theorem. The critical batch size itself grows with $D$, which shifts $\eta^{*}$ independently of $n$.
- **Empirically open.** $\Delta(n)$ has never been published with a tuned control arm above roughly 1B parameters. The experiment is runnable at 7B today for well under $10^{6}$ GPU-hours. Nobody has published it.
- **Methodologically blocked.** "Transfer succeeded" has no agreed definition. Papers show overlapping loss-vs-$\eta$ curves by eye. The seed-to-seed standard deviation of final loss at a given scale is rarely reported, so there is no noise floor against which to call an overlap significant.

## 6. Why It Is Hard

The obstruction is **circular verification cost**. Estimating $\Delta(n)$ requires $\min_\theta L_n(\theta)$ — a hyperparameter sweep at the target scale, which is exactly the expense muP was introduced to remove. At $10^{26}$ FLOPs, a single run on 100k H100-class accelerators at 40% MFU ($4\times10^{19}$ FLOP/s aggregate) takes about **29 days**; a 7-point learning-rate sweep with 3 seeds is 21 runs, i.e. **1.7 years of the full cluster**. No lab will spend that to measure a quantity it hopes is zero.

Two secondary obstructions:

- **Confounded measurement.** At frontier scale, muP is never the only change. Data mix, sequence length, MoE routing, and optimizer variants all move between the proxy and the target. Any observed gap is attributable to at least four things.
- **Absent ground truth on the objective.** The quantity that matters is downstream capability, not validation cross-entropy, and the map from a 0.01-nat loss difference to benchmark deltas is noisy and task-dependent. An evaluation reporting "muP matched the baseline on MMLU" does not measure transfer fidelity.

## 7. Current Research (as of 2026)

- **Per-layer exponent search.** Google DeepMind / Google Brain lineage (Everett, Xiao, Wortsman, Pennington, Sohl-Dickstein and colleagues): treat $(a_l,b_l,c_l)$ as fit parameters rather than derived ones, and measure alignment exponents directly.
- **Unit-scaled and $\epsilon$-corrected variants.** Graphcore's u-muP line, aiming for hyperparameter *separability* (independent axes) as well as transfer, with low-precision training as the motivator.
- **Joint width–depth transfer.** Harvard (Pehlevan group) and the Tensor Programs line, on whether depth-muP's $1/\sqrt{L}$ branch scaling composes with width muP under Adam.
- **Batch-size and schedule coupling.** Work relating $\eta^{*}$ to critical batch size and to warmup/decay shape, which muP does not cover. *(frontier — verify)*
- **Industrial silent adoption.** Multiple frontier labs report muP-derived setups in model cards without control arms. *(frontier — verify)* Treat these as existence proofs of usability, not as measurements of $\Delta$.

## 8. Concrete Next Experiment

**Scale.** Proxy: $n_0 = 256$, $L=12$, ~40M non-embedding parameters. Targets: 1B, 3B, 7B ($n = 2048, 3072, 4096$), each trained at Chinchilla-optimal $D = 20N$ tokens on a fixed corpus, fixed tokenizer, fixed data order per seed.

**Arms.**
- *Transfer arm:* $\hat\theta(n_0)$ tuned at the proxy over a 9-point $\log_2\eta$ grid plus init scale and readout multiplier, then mapped by muP. 3 seeds per target.
- *Control arm (the expensive one, and the point):* at each target, a 7-point $\log_2\eta$ grid centred on the transferred value, 3 seeds at the argmin. This arm is what every prior paper omits above 1B.
- *Noise arm:* 5 seeds at the transferred $\eta$ at each target, to establish $\sigma_{\text{seed}}$.

**Cost.** 7B control sweep $\approx 7 \times 6 \times 7\times10^{9} \times 1.4\times10^{11} = 4.1\times10^{22}$ FLOPs, about 29 hours on 1024 H100s at 40% MFU. Full three-scale design lands near $8\times10^{22}$ FLOPs — roughly $10^{5}$ GPU-hours. Affordable for an academic consortium.

**Deciding number.** The slope

$$
\beta \;=\; \frac{d\log \Delta(n)}{d\log N}
$$

fit over the three targets, reported with $\sigma_{\text{seed}}$ as the noise floor. **$\beta \le 0$ with $\Delta(7\text{B}) < 2\sigma_{\text{seed}}$ vindicates muP at frontier extrapolation. $\beta > 0$ with $\Delta(7\text{B}) > 0.01$ nats means muP is a decaying approximation and frontier runs are leaving measurable compute on the table.**

## 9. Key References

- **[Foundational]** Greg Yang, Edward J. Hu. *Feature Learning in Infinite-Width Neural Networks* (Tensor Programs IV). ICML, 2021. — arXiv:2011.14522
- **[Foundational / SOTA]** Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[Theory]** Greg Yang, Etai Littwin. *Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit.* ICLR, 2023. — arXiv:2308.01814
- **[Theory]** Greg Yang, Dingli Yu, Chen Zhu, Soufiane Hayou. *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks.* ICLR, 2024. — arXiv:2310.02244
- **[Theory]** Blake Bordelon, Lorenzo Noci, Mufan Bill Li, Boris Hanin, Cengiz Pehlevan. *Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit.* ICLR, 2024. — arXiv:2309.16620
- **[SOTA]** Katie Everett, Lechao Xiao, Mitchell Wortsman, Alexander A. Alemi, Roman Novak, Peter J. Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling, Jaehoon Lee, Jeffrey Pennington. *Scaling Exponents Across Parameterizations and Optimizers.* ICML, 2024. — arXiv:2407.05872
- **[Replication]** Lucas Lingle. *A Large-Scale Exploration of $\mu$-Transfer.* 2024. — arXiv:2404.05728
- **[SOTA]** Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Y. Prince, Björn Deiseroth, Andres Felipe Cruz-Salinas, Carlo Luschi, Samuel Weinbach, Douglas Orr. *u-$\mu$P: The Unit-Scaled Maximal Update Parametrization.* 2024. — arXiv:2407.17465
- **[Empirical]** Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, Simon Kornblith. *Small-scale Proxies for Large-scale Transformer Training Instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[Application]** Nolan Dey, Gurpreet Gosal, Zhiming Chen, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin Tom, Joel Hestness. *Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster.* 2023. — arXiv:2304.03208
- **[Context]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556

## 10. Worked Example

Take a 7B target, $D = 1.4\times10^{11}$ tokens, and suppose the tuned optimum sits one grid step below the transferred value: transferred $\eta = 2^{-8} = 3.9\times10^{-3}$, tuned optimum $\eta = 2^{-8.5} = 2.8\times10^{-3}$.

Suppose the measured losses are $L_{\text{transfer}} = 1.938$ nats and $L_{\text{tuned}} = 1.926$ nats, so $\Delta = 0.012$ nats, against an assumed seed standard deviation $\sigma_{\text{seed}} \approx 0.003$ nats (this is the number nobody publishes — flagged as assumed).

Convert to compute using $\alpha = 0.05$ and $L - E \approx 0.25$ nats:

$$
\rho = \exp\!\left(\frac{0.012}{0.05 \times 0.25}\right) = \exp(0.96) \approx 2.6.
$$

A 0.012-nat gap — visually invisible on a loss-vs-$\eta$ plot, four seed-sigmas wide — is worth **about $2.6\times$ the training compute**. At $10^{26}$ FLOPs that is roughly $6\times10^{25}$ FLOPs discarded, far more than the entire tuning sweep would have cost.

Now the obstruction. To know that $L_{\text{tuned}} = 1.926$, you must run the 7-point sweep. If you run it, you did not need muP. If you do not run it, the two arms are indistinguishable from the transfer arm alone: a loss of 1.938 nats is a perfectly healthy-looking 7B training curve, and nothing in it signals that 2.6x compute is missing. The failure mode of muP is not divergence or a visible spike — it is a run that looks completely normal and is quietly 60% efficient. That is why the gap is empirically open rather than empirically settled: the measurement is cheap at 7B and prohibitive at $10^{26}$ FLOPs, and only the cheap end has ever been checked.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*