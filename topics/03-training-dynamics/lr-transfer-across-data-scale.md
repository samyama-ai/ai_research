---
id: 03-training-dynamics/lr-transfer-across-data-scale
title: "Learning Rate Transfer Across Data Scale"
topic: 03-training-dynamics
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learning Rate Transfer Across Data Scale

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/lr-transfer-across-data-scale` · **Status:** empirically-open

## 1. Problem Statement

μP (maximal update parametrization) makes the optimal learning rate approximately invariant to **width**. Nothing comparable exists for **training duration**. In practice a small proxy run is tuned at $D_0$ tokens and the resulting $\eta^\star$ is used for a target run at $D \gg D_0$ — 100× to 10,000× more tokens — with no principled correction.

Three distinct problems, routinely conflated:

- **Measurement.** How much does $\eta^\star$ move with $D$ at fixed model size $N$, fixed batch size $B$, and fixed schedule shape? Nobody has published a clean sweep with all three controlled.
- **Method.** Is there a parametrization, schedule, or optimizer normalization under which $\eta^\star$ is $D$-invariant, so that a $D_0$-token proxy transfers zero-shot?
- **Theory.** Under what conditions on the loss landscape and the optimizer does a $D$-invariant limit exist at all? μP's width invariance follows from an infinite-width feature-learning limit. There is no analogous infinite-data limit with a fixed nonzero $\eta$.

Solved would mean: a rule $\eta^\star(N, D, B)$ — or a parametrization making the $D$ argument vanish — that predicts the loss-minimizing LR at $D = 10^{13}$ tokens to within the width of the LR optimum's flat region, from runs at $D_0 \le 10^{10}$.

## 2. Formal Setting

Objects, each as measured:

- $N$ — non-embedding parameter count.
- $D$ — training tokens. Measured as $S \cdot B \cdot T$ (steps × sequence batch × context length), single epoch, no repeats.
- $B$ — tokens per optimizer step.
- $\eta$ — peak learning rate of the schedule $\eta_t = \eta \cdot s(t/S)$, with $s$ the shape (warmup + cosine, or constant + cooldown), $\max_u s(u) = 1$.
- $L(N, D; \eta, B)$ — final loss on a held-out shard of the same distribution, measured at the last step, averaged over $\ge 3$ seeds. Seed noise at 1B scale is typically $\sigma_L \approx 0.002$–$0.005$ nats.

The optimum and the object of interest:

$$\eta^\star(N,D,B) = \arg\min_{\eta} \; \mathbb{E}_{\text{seed}}\,[\,L(N,D;\eta,B)\,], \qquad \Delta(D_0 \!\to\! D) = \log_2 \frac{\eta^\star(N,D,B)}{\eta^\star(N,D_0,B)}.$$

$\Delta$ is measured in LR doublings, because sweeps are run on a log grid. The transfer question is whether $\Delta \to 0$ as the grid refines, or grows like $\log D$.

The optimum is flat, so the sign of $\Delta$ is not itself decisive. Define the **regret** of transferring $\eta_0 = \eta^\star(N,D_0,B)$:

$$R(D_0 \!\to\! D) = L(N,D;\eta_0,B) - L(N,D;\eta^\star(N,D,B),B) \;\ge\; 0 .$$

$R$ is the quantity that matters and is directly measurable. Convert it to a compute-equivalent using a fitted scaling law $L(C)$: $\rho = C_{\text{eq}}/C - 1$, the extra compute needed at $\eta_0$ to reach the loss achieved at $\eta^\star$.

Assumptions, with the violated ones flagged:

1. *Single loss minimum in $\eta$, locally quadratic in $\log \eta$.* Approximately holds below the instability edge; **violated** above it, where the curve is not smooth but cliff-shaped — divergence, loss spikes, or silent degradation.
2. *Schedule shape fixed under $D$-rescaling.* **Violated by construction** for cosine: the same $s$ at larger $S$ spends more absolute steps at high LR, so $\eta$ and $D$ are not separable. This is the central confound.
3. *$B$ independent of $D$.* **Violated in practice** — batch size is scaled with compute in every frontier recipe, and Adam's LR–batch coupling is roughly $\eta \propto \sqrt{B}$ (Malladi et al., 2022), so an uncontrolled $B(D)$ injects a spurious $D$-dependence into $\eta^\star$.
4. *Weight decay held fixed.* **Violated in effect**: AdamW's decay acts through the timescale $\tau = 1/(\eta\lambda)$ measured in steps, so fixed $(\eta,\lambda)$ means a *different* effective regularization at different $S$.
5. Single epoch, fixed data mixture, fixed context length.

## 3. State of the Art

**Established.**

- μP transfers the optimal LR across width. Yang et al., *Tensor Programs V* (NeurIPS 2021, arXiv:2203.03466), tuned a 40M proxy and transferred to 6.7B GPT-3, beating the baseline 6.7B run. The transferred axis is width; **depth, data, and batch size were not shown to transfer** in the same sense.
- Optimal LR falls with compute budget. DeepSeek LLM (Bi et al., 2024, arXiv:2401.02954) fit $\eta_{\text{opt}} \approx 0.31\,C^{-0.125}$ and $B_{\text{opt}} \approx 0.29\,C^{0.327}$ over budgets $10^{17}$–$10^{20}$ FLOPs. Since $C$ mixes $N$ and $D$, this does not isolate the $D$ axis.
- Everett et al., *Scaling Exponents Across Parameterizations and Optimizers* (ICML 2024, arXiv:2407.05872) — the largest published sweep of its kind, tens of thousands of runs to 1.2B params. Finding: μP's width transfer is real but incomplete; per-layer LR and Adam's $\epsilon$ matter, and alignment assumptions underlying μP are violated at realistic widths.
- Porian et al., *Resolving Discrepancies in Compute-Optimal Scaling of Language Models* (NeurIPS 2024, arXiv:2406.19146) showed the Kaplan-vs-Chinchilla exponent gap is largely an artifact of LR tuning and schedule, not of the data. Direct evidence that mis-transferred LR corrupts scaling-law conclusions, not just single runs.

**Claimed but unablated.**

- Explicit $\eta^\star(N,D)$ power-law fits — e.g. the StepLaw line of work (Li et al., *Predictable Scale*, 2025, arXiv:2503.04715), reporting $\eta^\star$ decreasing in $N$ and *increasing* in $D$, with $B^\star$ a function of $D$ alone. Single-lab fits on one tokenizer, one mixture, one architecture family; the positive $D$ exponent is in apparent tension with the DeepSeek $C^{-0.125}$ fit and has not been independently reproduced. *(frontier — verify)*
- Schedules advertised as duration-agnostic: WSD / constant-plus-cooldown (Hägele et al., NeurIPS 2024, arXiv:2405.18392) and the Power scheduler (Shen et al., 2024, arXiv:2408.13359). WSD's established claim is that a run can be branched and cooled at any $D$ with loss matching a bespoke cosine run; that the *peak* LR is thereby $D$-invariant is asserted with limited ablation.
- u-μP (Blake et al., 2024, arXiv:2407.17465): unit-scaled μP with claimed lower LR sensitivity. Benchmark numbers at sub-1B scale only.

## 4. What Is Known

- Width transfer works over $\ge$128× width at fixed $D$; μP reduced a 6.7B run's loss below its baseline using a 40M proxy (arXiv:2203.03466).
- Optimum flatness: at 1B scale the loss penalty inside a 2× LR window is typically $<0.01$ nats — smaller than the drop from 20 to 40 tokens/param. A 4× miss usually costs $0.02$–$0.05$ nats; an 8× miss risks crossing the instability edge.
- Empirical LR decrease with budget: exponent $-0.125$ in $C$ over three decades of FLOPs (DeepSeek, 2024).
- Instability grows with scale and LR jointly: attention-logit and output-logit divergence appear at smaller LR as width grows; qk-layernorm and z-loss shift the edge (Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, ICLR 2024, arXiv:2309.14322).
- Mechanism for width transfer: sharpness $\lambda_{\max}(\nabla^2 L)$ at the edge of stability becomes width-independent under μP (Noci et al., ICML 2024, arXiv:2402.17457). The corresponding statement across $D$ is not established — sharpness demonstrably drifts over training (Cohen et al., ICLR 2021).
- Weight decay's optimum tracks tokens-per-parameter, not $N$ or $D$ alone (Bergsma et al., *Power Lines*, 2025, arXiv:2505.13738), which entangles $\lambda$ with any $\eta$-vs-$D$ measurement.
- Frontier practice is 100–2000 tokens/param (Llama 3 8B: 15T tokens ≈ 1900), i.e. 5–100× past Chinchilla's ~20 (Hoffmann et al., 2022, arXiv:2203.15556). Published LR sweeps cluster near 20.

## 5. What Is Not Known

- **Empirically open (the core gap).** No published sweep measures $R(D_0 \to D)$ at fixed $N$, fixed $B$, fixed $\lambda$, and a $D$-agnostic schedule across $\ge 100$× in $D$ into the 500+ tokens/param regime. The experiment is runnable at 1B scale for a few hundred thousand GPU-hours. Nobody has published it.
- **Theoretically open.** Whether a parametrization exists under which $\eta^\star$ is asymptotically $D$-invariant at fixed $N$. μP's argument is a width limit; the large-$D$ limit at fixed $N$ ends in a different regime — approaching interpolation or the data-limited floor — where no feature-learning fixed point is known.
- **Methodologically blocked.** "The optimal LR" is not well defined without fixing the schedule, and no schedule family is both $D$-agnostic and standard. Under cosine, $\eta^\star(D)$ is partly an artifact of decay shape; under constant+cooldown, the peak LR's optimum is flatter but the cooldown fraction becomes a second free knob. Until the community agrees on a $D$-reparametrization-invariant definition, cross-paper exponents are not comparable — which is exactly why the DeepSeek and StepLaw signs can disagree without either being wrong.

## 6. Why It Is Hard

The obstruction is **confounded measurement plus a flat optimum measured against seed noise**, not compute alone.

- Varying $D$ at fixed $N$ changes four things at once: total steps, effective weight-decay timescale $1/(\eta\lambda S)$, the fraction of training under high LR (cosine), and warmup's share. The measured $\Delta(D_0 \to D)$ is a sum of these, and no published design separates them.
- The quantity to detect is small. Regret from a 2× LR miss ($\sim0.005$–$0.01$ nats) is 1–3 seed standard deviations at 1B. Resolving it needs $\ge 5$ seeds per LR per $D$, multiplying an already expensive grid by 5.
- Non-identifiability: $(\eta, B, \lambda)$ enter through approximately two effective quantities — a noise scale $\eta/B$-like term and a decay timescale $\eta\lambda S$. A sweep over $\eta$ alone cannot tell which coordinate moved.
- Failure at large $D$ is often not a smooth loss increase but a spike or a slow late-training divergence, so the objective is not the smooth function the fitting procedure assumes.

## 7. Current Research (as of 2026)

- **Hyperparameter scaling laws.** Explicit $\eta^\star(N,D)$, $B^\star(D)$ fits — StepLaw (StepFun), DeepSeek, Cerebras' *Power Lines*. Direction: replace transfer with an extrapolated fit. Weakness: extrapolation is only as good as the fitted regime, and none cover $>1000$ tokens/param. *(frontier — verify current exponents.)*
- **Duration-agnostic schedules.** WSD/cooldown variants now default at several labs; the open question is whether the peak LR under WSD is genuinely $D$-invariant or merely less sensitive.
- **Parametrization beyond width.** depth-μP (Yang, Littwin et al., *Tensor Programs VI*, arXiv:2310.02244); u-μP; per-layer LR exponents (Everett et al.).
- **Optimizer replacement.** Muon, Shampoo/SOAP-family second-order methods, and sign-based updates change the LR–scale coupling. Whether they widen the flat region in $D$ is largely unmeasured. *(frontier — verify.)*
- **Theory.** Sharpness/edge-of-stability accounts of *why* transfer works (Noci et al.), and warmup mechanism work (Kalra & Barkeshli, NeurIPS 2024) — both about width and early training, not duration.

## 8. Concrete Next Experiment

**Question.** At fixed $N$ and fixed $B$, does the transfer regret $R(D_0 \to D)$ grow with $D$?

**Scale.** $N = 610$M non-embedding, μP, decoder-only Transformer, one clean mixture, one epoch. $D \in \{12\text{B}, 40\text{B}, 120\text{B}, 400\text{B}\}$ tokens — 20 to 650 tokens/param, 33× span. $B$ pinned at 2M tokens/step at every $D$ (the deliberate departure from practice; it removes confound 3). AdamW with $\lambda$ set so $\eta\lambda S$ is constant across $D$ (removes confound 4). Constant LR with a fixed 20% cooldown, branch-and-cool from one trunk per LR (removes confound 2 and cuts cost ~3×).

**Grid.** 7 peak LRs, half-decade spacing, centred on $\eta^\star(12\text{B})$. 5 seeds at the 3 LRs nearest each optimum, 2 elsewhere. Cost with branch-and-cool $\approx 1.2\times10^{22}$ FLOPs, ≈ 90k H100-hours.

**Control arm.** The same grid with cosine-to-zero and $B$ scaled as $D^{0.33}$ — the standard recipe. If $\Delta$ is large under the control and near zero under the treatment, the observed "LR drift with data scale" in the literature is a schedule-and-batch artifact, not a property of data scale.

**Deciding number.** $R(12\text{B} \to 400\text{B})$ in nats, with a bootstrap CI over seeds. **$R < 0.005$ nats** (below 2 seed-$\sigma$, and below the loss gained from ~4% more compute) ⇒ LR transfers across data scale once schedule and batch are controlled; proxy tuning at 20 tokens/param is safe. **$R > 0.02$ nats** (equivalent to ~15–25% wasted compute at these budgets) ⇒ it does not, and $\eta^\star$ must be fit in $D$, not transferred.

## 9. Key References

- **[Foundational]** Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS 2021. — arXiv:2203.03466
- **[Foundational]** Jared Kaplan et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[SOTA]** Katie Everett, Lechao Xiao, Mitchell Wortsman, Alexander A. Alemi, Roman Novak, Peter J. Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling, Jaehoon Lee, Jeffrey Pennington. *Scaling Exponents Across Parameterizations and Optimizers.* ICML 2024. — arXiv:2407.05872
- **[SOTA]** Tomer Porian, Mitchell Wortsman, Jenia Jitsev, Ludwig Schmidt, Yair Carmon. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models.* NeurIPS 2024. — arXiv:2406.19146
- **[SOTA]** Alexander Hägele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro von Werra, Martin Jaggi. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.* NeurIPS 2024. — arXiv:2405.18392
- **[SOTA]** Mitchell Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR 2024. — arXiv:2309.14322
- **[Theory]** Lorenzo Noci, Alexandru Meterez, Thomas Hofmann, Antonio Orvieto. *Why do Learning Rates Transfer? Reconciling Optimization and Scaling Limits for Deep Learning.* ICML 2024. — arXiv:2402.17457
- **[Theory]** Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, Sanjeev Arora. *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms.* NeurIPS 2022.
- **[Empirical]** Xiao Bi et al. (DeepSeek-AI). *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* 2024. — arXiv:2401.02954
- **[Empirical]** Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, Joel Hestness. *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training.* 2025. — arXiv:2505.13738
- **[Empirical]** Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Y. Prince, Björn Deiseroth, Andres Felipe Cruz-Salinas, Carlo Luschi, Samuel Weinbach, Douglas Orr. *u-μP: The Unit-Scaled Maximal Update Parametrization.* 2024. — arXiv:2407.17465

## 10. Worked Example

A 1B-param μP model, proxy-tuned at Chinchilla ratio, then scaled out in data.

- Proxy: $N=10^9$, $D_0 = 2\times10^{10}$ (20 tok/param), $B = 10^6$ tokens/step ⇒ $S_0 = 2\times10^4$ steps. Cosine to zero, AdamW $\lambda = 0.1$. Sweep gives $\eta^\star_0 = 3\times10^{-3}$.
- Target: $D = 2\times10^{12}$ (2000 tok/param), same $N$, $B$ raised to $4\times10^6$ ⇒ $S = 5\times10^5$ steps.

Two corrections apply before any "data-scale" effect is visible:

1. **Batch.** Adam's square-root rule gives $\eta \mathrel{\times}= \sqrt{4} = 2$ ⇒ $6\times10^{-3}$.
2. **Decay timescale.** $\eta\lambda S$ went from $3\times10^{-3}\cdot0.1\cdot2\times10^4 = 6$ to $6\times10^{-3}\cdot0.1\cdot5\times10^5 = 300$, a 50× stronger effective decay. Holding $\eta\lambda S$ fixed instead means $\lambda \to 0.002$ — or, if $\lambda$ is left at 0.1, the LR sweep will report a *lower* $\eta^\star$ purely to compensate.

Suppose the target sweep returns $\eta^\star = 1.5\times10^{-3}$, i.e. $\Delta = -2$ doublings from the naive proxy value. **That number is uninterpretable.** It is the sum of: $+1$ doubling from batch, an unknown negative contribution from the 50× decay-timescale shift, an unknown contribution from cosine spending 25× more absolute steps at high LR, and whatever the true data-scale effect is. Four terms, one measurement, no identification.

The cost of being wrong is real but small enough to hide. At 1B/2T tokens, $C \approx 1.2\times10^{22}$ FLOPs; a fitted $L(C)$ with exponent $\approx0.05$ means $0.02$ nats of regret is worth roughly $1-(1-0.02/(0.05 \cdot L))\dots$ — concretely, about 25% more compute, or ~$1.5\times10^{21}$ wasted FLOPs. That is large in absolute terms and invisible in a loss curve, which is precisely why the question stays open: everyone pays it, nobody sees it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*