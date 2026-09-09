---
id: 13-parameter-efficient-adaptation/learning-rate-transfer-adapter-rank
title: "Learning-Rate Transfer Across Adapter Rank"
topic: 13-parameter-efficient-adaptation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Learning-Rate Transfer Across Adapter Rank

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/learning-rate-transfer-adapter-rank` · **Status:** partially-solved

## 1. Problem Statement

Tuning a LoRA adapter requires picking a learning rate. Practitioners tune it at a cheap rank ($r=8$ or $16$) and reuse it at the rank they ship ($r=128$, $256$, $512$). The question is whether that reuse is sound.

- **Measurement variant.** Does the loss-optimal learning rate $\eta^\ast(r)$ actually stay fixed as $r$ grows, under a stated parametrization, and how would you certify that it does without a full sweep at every rank?
- **Method variant.** Find a reparametrization — a scaling of the adapter factors, their initialization, and their per-factor learning rates — under which $\eta^\ast$ is provably rank-independent, and which does not cost final loss relative to per-rank tuning.
- **Theory variant.** Prove a rank-scaling law: derive the exponent $c$ in the adapter scale $\alpha/r^{c}$ and the per-factor learning-rate ratio for which forward activations, backward signals, and weight updates all stay $\Theta(1)$ in $r$, for Adam, for a *pretrained* (non-random) base weight, at finite $r$.

Solving it means: a rule that, given a sweep at $r_0=8$, predicts $\eta^\ast(256)$ to within one grid step of a $\sqrt{2}$-spaced sweep, and loses no more than the seed noise in final validation loss.

## 2. Formal Setting

A frozen base matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is adapted as

$$W = W_0 + \frac{\alpha}{r^{c}} B A, \qquad B \in \mathbb{R}^{d_{\text{out}} \times r},\; A \in \mathbb{R}^{r \times d_{\text{in}}},\; r \ll \min(d_{\text{in}}, d_{\text{out}}).$$

Standard LoRA sets $c=1$; rsLoRA sets $c=1/2$. Initialization is $A_{kj} \sim \mathcal{N}(0, \sigma^2)$ with $\sigma^2 = 1/d_{\text{in}}$, $B = 0$. Optimizer is Adam with learning rates $\eta_A = \eta$, $\eta_B = \lambda \eta$.

**Measured quantities.**

- *Optimal learning rate.* Train on a fixed token budget $T$ and dataset $\mathcal{D}$; measure held-out loss $L(\eta, r, s)$ at seed $s$. Sweep $\eta$ on a log grid of ratio $\sqrt{2}$, fit a parabola in $\log_2 \eta$ over the three lowest points, and define $\eta^\ast(r) = \arg\min$ of that fit. Report $\hat{\sigma}_{\log}$, the seed standard deviation of $\log_2 \eta^\ast$.
- *Transfer predicate.* Transfer holds at $(r_0, r)$ iff $|\log_2 \eta^\ast(r) - \log_2 \eta^\ast(r_0)| \le \tfrac12$ **and** $L(\eta^\ast(r_0), r) - L(\eta^\ast(r), r) \le 2\hat{\sigma}_L$. Both conditions are needed: a flat basin can satisfy the second while the argmin drifts, and a sharp basin can fail the second on a drift of a quarter grid step.
- *Update size.* The thing the parametrization is supposed to hold fixed is the relative first-step weight change $\rho_1(r) = \|\Delta_1 W\|_F / \|W_0\|_F$, measurable directly.
- *Alignment.* $\gamma_t = \dfrac{\|\Delta_t B \cdot A_t\|_F}{\sqrt{r}\,\|\Delta_t B\|_F \|A_t\|_F / \sqrt{r}}$ — normalized so $\gamma \approx r^{-1/2}$ for independent factors and $\gamma \approx 1$ for fully aligned ones. This is the quantity that sets $c$, and it is directly loggable.

**Assumptions, and which are violated.**

1. *Adam is scale-invariant in the parameter.* Violated by $\epsilon$ (updates below $\epsilon$ are damped) and by decoupled weight decay.
2. *$W_0$ has i.i.d. entries.* False — $W_0$ is pretrained, has a heavy-tailed spectrum, and its top singular directions are exactly the ones the adapter must interact with. Every Tensor-Programs-style limit argument assumes the first, and none of the LoRA rank results have been re-derived without it.
3. *$r \to \infty$.* The regime of interest is $r \in [8, 512]$ against $d = 4096$. Asymptotic exponents are being applied at $r=8$.
4. *Loss is locally quadratic in $\log \eta$.* Holds near the optimum; breaks on the divergence side, so the parabola must be fit one-sided.
5. *Single epoch, no LR schedule interaction.* In practice PEFT runs use cosine decay over 2–3 epochs; $\eta^\ast$ under a schedule is not the same object as under constant LR.

## 3. State of the Art

**Established.**

- **$\mu$P (Yang et al., *Tensor Programs V*, NeurIPS 2021)** gives zero-shot LR transfer across *width* for full training, verified up to 6.7B. It says nothing about rank; the adapter bottleneck is not a width in the $\mu$P sense because $B$ and $A$ are trained jointly and $B$ starts at zero.
- **rsLoRA (Kalajdzievski, 2023, arXiv:2312.03732)** shows $c=1$ makes adapter outputs shrink as $r$ grows and proves $c=1/2$ is the unique exponent keeping forward activations $\Theta(1)$ under the paper's independence assumption. Verified on Llama-2-7B, OpenOrca, ranks 4–2048.
- **LoRA+ (Hayou, Ghosh, Yu, ICML 2024)** proves that $\eta_A = \eta_B$ gives suboptimal (non-"efficient") feature learning in the width limit, and that $\lambda = \eta_B/\eta_A = \Theta(d)$ is required. Empirically $\lambda = 16$ is the recommended default. The analysis is a width-limit argument at fixed $r$; the $r$-dependence is not derived.

**Claimed but unablated.**

- That $c=1/2$ plus $\lambda=16$ *jointly* deliver rank transfer. No published sweep varies $c$, $\lambda$, and $r$ in a full factorial with a per-rank LR sweep as control. rsLoRA's rank results use a single LR; LoRA+'s $\lambda$ results use a single rank.
- That $\alpha = 2r$ (the widely copied HuggingFace-era default, which is $c=1$ with $\alpha$ growing linearly, hence effectively $c=0$) is "good practice." This is folklore with no ablation behind it.

**Benchmark-number-only.** Most rank-vs-quality tables in DoRA, PiSSA, and AdaLoRA papers report accuracy at several ranks with one shared learning rate. Those numbers cannot distinguish "rank $r$ is worse" from "the shared LR was tuned for the other rank."

## 4. What Is Known

- **Biderman et al., *LoRA Learns Less and Forgets Less* (TMLR 2024)** ran the largest published joint LR × rank sweep: Llama-2-7B and 13B, ranks 16–256, code (Magicoder) and math (MetaMathQA), LR grid $10^{-5}$–$5\times10^{-4}$. LoRA's best LR clustered near $1\text{–}5\times10^{-4}$ across ranks — roughly an order of magnitude above full fine-tuning's $\sim10^{-5}$ — and the rank-induced drift was within about one factor-of-2 grid step. Under $c=1$, this is *weaker* drift than the naive $\sqrt{r}$ prediction (a factor of 4 over $16\to256$).
- **Rank quality saturates before LR sensitivity does.** In the same study, going $r=16 \to 256$ moved final loss by less than the gap between LoRA and full fine-tuning, so the loss-vs-rank signal is small compared to the loss-vs-LR signal — the sweep is dominated by the wrong axis.
- **Initialization asymmetry is real.** Hayou et al. (NeurIPS 2024) show init-$A$-random/$B$-zero and its mirror give different optimal $\eta$ by an amount that grows with width; the two inits are not interchangeable, and papers differ silently on which they use.
- **Parametrization choice changes measured exponents more than architecture does.** Everett et al. (ICML 2024) found, at up to 1.2B params, that per-layer LR alignment and $\epsilon$ handling shift transfer exponents materially — the same confound applies to adapters and is not controlled in any LoRA rank study.

## 5. What Is Not Known

- **Theoretically open.** No proof of the correct $c$ for Adam with a *pretrained* $W_0$, at finite $r$, with the alignment $\gamma_t$ evolving during training. rsLoRA's $c=1/2$ is derived under an independence assumption that training violates within the first few steps. There is no theorem giving $\eta^\ast(r)$ up to a constant.
- **Empirically open.** The full factorial — $c \in \{0, 1/2, 1\}$ × $\lambda \in \{1, 4, 16\}$ × $r \in \{8, 32, 128, 512\}$ × 8-point LR grid × 3 seeds — has never been run at 7B. It is roughly 900 fine-tuning runs of a few hundred GPU-hours total; entirely feasible, simply unrun.
- **Methodologically blocked.** "Optimal learning rate" is not a well-defined function of rank until the token budget, the schedule, the target-module set, and the epoch count are pinned. Published $\eta^\ast$ values are not comparable across papers because these differ. Worse, the LoRA loss basin in $\log \eta$ is flat enough that the argmin is seed-unstable; no paper reports $\hat{\sigma}_{\log}$, so "transfer holds" claims have no error bar.

## 6. Why It Is Hard

The obstruction is **non-identifiability between three knobs that all rescale the same product**. The effective update to $W$ depends on $\alpha$, $r^{-c}$, $\eta$, $\lambda$, and $\sigma$ only through a small number of combinations. Changing $\alpha$ by $2\times$ and $\eta$ by $1/2$ is nearly a no-op away from $\epsilon$ and weight-decay effects. So an experiment that reports "LR transferred across rank" may only be reporting that the authors' $\alpha$ convention absorbed the drift — and since $\alpha=2r$ is a common default, the drift is silently absorbed by construction in a large fraction of the literature.

Compounding it: the exponent that *should* appear is set by $\gamma_t$, the alignment between $\Delta B$ and $A$, which is $r^{-1/2}$ at initialization and heads toward $1$ as training aligns the factors. Neither $c=1$ nor $c=1/2$ is right throughout a run. The measurement is therefore confounded by a quantity that changes during the very run being measured, and nobody logs it.

## 7. Current Research (as of 2026)

- **Unit-scaled parametrizations.** The u-$\mu$P line (Blake et al., 2024) removes the $\alpha$/$\eta$ degeneracy by forcing unit-scale tensors, which is exactly the fix the identifiability problem needs. Extension to adapters is being attempted but not published at scale *(frontier — verify)*.
- **Spectral / init-aware adapters.** PiSSA (Meng et al., NeurIPS 2024) initializes $BA$ from the top singular subspace of $W_0$, which changes $\gamma_0$ from $r^{-1/2}$ to $\approx 1$ and should therefore change the correct $c$ outright. The interaction with LR transfer is untested.
- **Automated per-layer LR.** Work on adapter-side preconditioning (LoRA-GA-style gradient alignment, and Riemannian/scale-invariant LoRA optimizers) aims to make $\eta^\ast$ rank-free by construction rather than by parametrization *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B, LoRA on all attention and MLP projections, 200M tokens of a fixed instruction mix, constant LR (no schedule), bf16, Adam $\epsilon=10^{-8}$, no weight decay. Held-out loss on 5M tokens.

**Grid.** $r \in \{8, 64, 512\}$ × $c \in \{1/2, 1\}$ × $\lambda \in \{1, 16\}$ × 7 LRs spaced $\sqrt{2}$ around $10^{-4}$ × 3 seeds = 252 runs. At ~15 GPU-hours each on H100s, ~3,800 GPU-hours.

**Control arm.** Per-rank tuned LR at each $(r, c, \lambda)$ cell — this is the ceiling the transfer rule is measured against. Log $\gamma_t$ every 100 steps in all runs.

**The deciding number.** $D = \max_{r \in \{64,512\}} |\log_2 \eta^\ast(r) - \log_2 \eta^\ast(8)|$, with the seed error bar $\hat\sigma_{\log}$ reported alongside. If some $(c, \lambda)$ cell achieves $D \le 0.5$ while $c=1, \lambda=1$ gives $D \ge 1.5$, the parametrization fix is real and the exponent is settled empirically. If every cell gives $D \le 0.5$ **and** $\hat\sigma_{\log} \ge 0.4$, the honest conclusion is that the basin is too flat to measure and the problem is methodologically blocked, not solved.

## 9. Key References

- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466
- **[SOTA]** Soufiane Hayou, Nikhil Ghosh, Bin Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML, 2024. — arXiv:2402.12354
- **[SOTA]** Damjan Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* Preprint, 2023. — arXiv:2312.03732
- **[SOTA]** Soufiane Hayou, Nikhil Ghosh, Bin Yu. *The Impact of Initialization on LoRA Finetuning Dynamics.* NeurIPS, 2024. — arXiv:2406.08447
- **[Empirical]** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[Empirical]** Katie Everett, Lechao Xiao, Mitchell Wortsman, Alexander A. Alemi, Roman Novak, Peter J. Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling, Jaehoon Lee, Jeffrey Pennington. *Scaling Exponents Across Parameterizations and Optimizers.* ICML, 2024. — arXiv:2407.05872
- **[Related]** Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Y. Prince, Björn Deiseroth, Andres Felipe Cruz-Salinas, Carlo Luschi, Samuel Weinbach, Douglas Orr. *u-µP: The Unit-Scaled Maximal Update Parametrization.* Preprint, 2024. — arXiv:2407.17465
- **[Related]** Fanxu Meng, Zhaohui Wang, Muhan Zhang. *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.* NeurIPS, 2024. — arXiv:2404.02948
- **[Survey]** Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR, 2024. — arXiv:2403.14608

## 10. Worked Example

Take $d_{\text{in}} = d_{\text{out}} = 4096$, $\sigma = 1/\sqrt{4096} = 0.0156$, $\alpha = 16$, $c = 1$, $\eta = 10^{-4}$, $B = 0$. On step 1, Adam's update is sign-like: every entry of $B$ moves by about $\eta$. So

$$\Delta_1 W_{ij} = \frac{\alpha}{r}\sum_{k=1}^{r} \Delta B_{ik} A_{kj}.$$

The sum of $r$ terms of magnitude $\eta\sigma$ scales as $\eta\sigma\, r\,\gamma_1$.

- **If $\gamma_1 = r^{-1/2}$ (independent signs):** $|\Delta_1 W_{ij}| = \alpha\eta\sigma r^{-1/2}$. At $r=8$: $16 \times 10^{-4} \times 0.0156 / 2.83 = 8.8\times10^{-6}$. At $r=256$: $1.56\times10^{-6}$. The update shrinks $5.7\times$, so $\eta^\ast$ should rise by $\sqrt{32} \approx 5.7\times$ — about **2.5 grid steps** of a factor-2 sweep.
- **If $\gamma_1 = 1$ (aligned):** $|\Delta_1 W_{ij}| = \alpha\eta\sigma$, independent of $r$. $\eta^\ast$ should not move at all.

The measured drift in the largest published sweep is roughly $2\times$ over $r=16\to256$ — between the two predictions, and consistent with neither. Because $\Delta_1 B \propto (\nabla_{\text{out}} L)(Ax)^\top$ is rank-one and therefore *partly* aligned with $A$ from the first step, $\gamma$ starts near $r^{-1/2}$ and climbs. The observed exponent is an average over a trajectory-dependent $\gamma_t$.

That is the obstruction made concrete: the exponent $c$ that would make LR transfer exactly is not a property of the parametrization at all — it is a property of the run. Any fixed $c$ ($1$ or $1/2$) is an approximation whose error is bounded by the drift in $\gamma_t$, and no published experiment measures $\gamma_t$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*