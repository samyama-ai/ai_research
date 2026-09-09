---
id: 13-parameter-efficient-adaptation/lora-initialization-scale-convergence
title: "Initialization Scale for LoRA Convergence"
topic: 13-parameter-efficient-adaptation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Initialization Scale for LoRA Convergence

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/lora-initialization-scale-convergence` · **Status:** partially-solved

## 1. Problem Statement

LoRA reparameterizes a frozen weight $W_0 \in \mathbb{R}^{m \times n}$ as $W_0 + \frac{\alpha}{r} BA$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$. The product must start at zero so the adapted model equals the base model at step 0. That forces an asymmetry: one factor is random, the other is zero. Three free knobs follow — which factor is zeroed, the variance of the random factor, and the scaling $\gamma_r$ that multiplies $BA$ — and they interact with the learning rate.

**The problem.** Given $(m, n, r, W_0, \text{task})$ and a compute budget, choose $\gamma_r$, $\mathrm{Var}[A_{ij}]$, and the per-factor learning rates so that training is both *stable* (no loss spike, no divergence) and *feature-learning efficient* (the update $\Delta W$ changes the layer's output by $\Theta(1)$, not $\Theta(n^{-1/2})$ or $\Theta(n^{1/2})$), uniformly in width $n$ and rank $r$.

Three variants, different difficulty:

- **Measurement.** Define an observable that separates "converged fast because init was well-scaled" from "converged fast because the effective learning rate was higher." Currently these are confounded in every published comparison.
- **Method.** Produce an init rule that transfers: tuned once at width 256 / rank 8, applied without retuning at width 8192 / rank 256. This is $\mu$P-style zero-shot transfer for adapters.
- **Theory.** Prove, for a stated architecture and loss class, which $(\gamma_r, \sigma_A, \eta_A, \eta_B)$ family admits stable $\Theta(1)$ feature learning as $n, r \to \infty$, and whether the LoRA optimum is reachable from that init.

## 2. Formal Setting

Layer input $x \in \mathbb{R}^n$ with $\|x\|_2 = \Theta(\sqrt{n})$ (measured: RMS of the layer's pre-adapter activations over a held-out batch of 256 sequences). Adapter output

$$\Delta h = \gamma_r\, B A x, \qquad \gamma_r = \alpha / r \ \text{(LoRA)} \quad\text{or}\quad \alpha/\sqrt{r} \ \text{(rsLoRA)}.$$

**Init[A]:** $A_{ij} \sim \mathcal{N}(0, \sigma_A^2)$, $B = 0$. **Init[B]:** $B_{ij} \sim \mathcal{N}(0, \sigma_B^2)$, $A = 0$. Reference HuggingFace PEFT default is Init[A] with Kaiming-uniform $A$, i.e. $\sigma_A^2 = \Theta(1/n)$.

Quantities as measured:

- **Feature-learning scale** $\;\delta_t = \mathbb{E}_x\!\left[\|\Delta h_t - \Delta h_{t-1}\|_2\right] / \mathbb{E}_x\!\left[\|h_t\|_2\right]$, logged per layer per step. The target regime is $\delta_t = \Theta(1)$ in $n$: measure $\delta_{10}$ at $n \in \{512, 1024, 2048, 4096\}$ and fit the exponent $p$ in $\delta_{10} \propto n^{p}$. Stable feature learning means $\hat p = 0 \pm 0.05$.
- **Instability** $\;I = \max_t \big(\mathcal{L}_t - \min_{s \le t}\mathcal{L}_s\big)$ over the first 500 steps, in nats/token.
- **Effective LR.** Because the loss is invariant to $A \to cA$, $B \to c^{-1}B$, only the products $\eta_A \sigma_A^{-1}$-type combinations are identifiable. Report the *layerwise update ratio* $\rho = \|\Delta W_t\|_F / \|W_0\|_F$, which is invariant to that rescaling.
- **Convergence** $\;T(\epsilon)$ = optimizer steps to first reach validation loss $\epsilon$, with $\eta$ swept over a log grid of at least 7 points and the best arm reported.

**Assumptions, and which are violated.** (i) Gaussian, mean-zero $W_0$ with i.i.d. entries — *violated*: pretrained weights have heavy-tailed spectra and strong low-rank structure, which is exactly what PiSSA exploits. (ii) Infinite width with $r$ fixed — *violated* at $r=256$ on $n=4096$. (iii) SGD analysis extended to Adam by analogy — *violated*: Adam's per-coordinate normalization removes the $\sigma_A$ dependence of the first step almost entirely, so scale-of-init theory derived for SGD does not transfer verbatim. (iv) No weight decay, no dropout, no gradient clipping — *violated*: clipping interacts directly with the early-step spike that init scale controls.

## 3. State of the Art

**Theory SOTA (established).** Hayou, Ghosh & Yu, *The Impact of Initialization on LoRA Finetuning Dynamics* (NeurIPS 2024), analyze the $n \to \infty$ limit and show Init[A] and Init[B] are **not** symmetric: Init[A] tolerates a larger stable learning rate and yields more efficient feature learning, at the cost of "internal instability" in the adapter's own activations; Init[B] is stable but the induced updates are suboptimally small. The same group's **LoRA+** (ICML 2024) proves that a single learning rate for $A$ and $B$ is width-suboptimal and that $\eta_B / \eta_A = \Theta(n)$ restores $\Theta(1)$ feature learning.

**Kalajdzievski, rsLoRA (2023)** — established by direct calculation: $\gamma_r = \alpha/r$ makes adapter output scale as $\Theta(1/\sqrt{r})$, causing gradient collapse at large $r$; $\gamma_r = \alpha/\sqrt{r}$ is the unique rank-stabilizing choice. This is the cleanest settled sub-result on the page.

**Empirical SOTA (claimed, partially unablated).** Data-dependent inits: **PiSSA** (Meng et al., NeurIPS 2024) initializes $A, B$ from the top-$r$ SVD of $W_0$ and trains the residual; **LoRA-GA** (Wang et al., NeurIPS 2024) aligns the first LoRA update with the full-fine-tuning gradient via SVD of $\nabla_W \mathcal{L}$ at step 0; **OLoRA** uses QR orthonormalization. All three report faster early convergence, but the reported comparisons are against default-LoRA at a shared or lightly-swept learning rate. Because these inits change the effective step size, part of the gain is a learning-rate confound rather than an init effect. **This is the central unablated claim in the area.** LoRA-GA's reported GLUE/T5-base numbers and PiSSA's LLaMA-2-7B GSM8K numbers exist only as benchmark tables, not as matched-$\rho$ comparisons.

## 4. What Is Known

- **rank-scaling.** With $\gamma_r = \alpha/r$, per-layer gradient norms on $A$ shrink by roughly $\sqrt{r}$ as $r$ grows; rsLoRA restores monotone improvement with rank. Measured on LLaMA-2-7B-class models, ranks $r \in \{4, \ldots, 512\}$ (Kalajdzievski, 2023).
- **Init asymmetry is real and reproduced.** Init[A] beats Init[B] at the tuned learning rate on RoBERTa-base/large GLUE and LLaMA-7B MNLI-class tasks; the gap is small at the best LR (fractions of a point) but large at shared LRs — consistent with the theoretical claim that the mechanism is *the size of the stable LR*, not the init per se (Hayou et al., NeurIPS 2024).
- **LoRA+.** Setting $\eta_B/\eta_A \in [2^4, 2^6]$ gives ~1–2% accuracy gain and up to ~2× fewer steps to a target loss on GLUE with RoBERTa-base and on LLaMA-7B, at $r = 8$–$64$ (Hayou, Ghosh & Yu, ICML 2024).
- **$\alpha$ is not a free parameter.** With Adam, $\alpha$ and $\eta$ are near-degenerate at fixed $r$ — a widely reproduced practitioner regularity, and a direct consequence of Adam's scale invariance in the first step.
- **The starting point matters for the endpoint, not just the speed.** LoRA solutions contain "intruder dimensions" — singular vectors absent from full fine-tuning — and these correlate with worse out-of-distribution and sequential-task behavior at matched in-distribution loss (Shuttleworth et al., 2024, LLaMA-2-7B / RoBERTa). Init is one of the levers that decides which subspace is entered.
- **Capacity floor.** LoRA underperforms full fine-tuning on code and math continued-pretraining at 7B, and the gap is rank-limited, not init-limited (Biderman et al., TMLR 2024). Init tuning cannot close it.

## 5. What Is Not Known

- **Theoretically open.** No proof exists that any init family gives $\Theta(1)$ feature learning *jointly* in width $n$ and rank $r$ under Adam. Existing results are $n \to \infty$ at fixed $r$ (LoRA+, Init[A]/Init[B]) or fixed $n$ in $r$ (rsLoRA). The joint limit is unaddressed.
- **Theoretically open.** Whether Init[A] with a stated $\sigma_A$ makes the global LoRA optimum reachable at all, or only a basin determined by the random $A$'s row space. Non-convexity of $BA$ means no known result rules out init-dependent local minima at practical $r$.
- **Empirically open.** Whether any data-dependent init (PiSSA, LoRA-GA, OLoRA) retains an advantage over default LoRA once *both* arms are learning-rate swept and matched on $\rho$. The experiment is a few thousand GPU-hours at 7B — runnable, unrun in public.
- **Empirically open.** Whether an init rule tuned at $n = 256$ transfers zero-shot to $n = 8192$, the $\mu$P question for adapters.
- **Methodologically blocked.** "Better initialization" has no agreed observable. Steps-to-target confounds LR; final accuracy confounds regularization; $\delta_t$ is not logged by any standard PEFT library. Until $\rho$-matched comparison is standard, init papers are not comparable to each other.

## 6. Why It Is Hard

**Non-identifiability plus a measurement confound, compounding.** The loss is exactly invariant under $A \to cA,\ B \to c^{-1}B$, so "initialization scale" alone is not a well-defined quantity — only its interaction with the optimizer is. Under SGD that interaction is computable; under Adam, the update magnitude is $\eta$ times a normalized direction, so the first-step effect of $\sigma_A$ cancels to leading order and reappears only through second-order coupling in the $B$ gradient. The consequence: every published init comparison that fixes $\eta$ across arms is measuring a *different effective step size*, not a different init. The fix — sweeping $\eta$ independently per arm and matching on $\rho$ — multiplies experiment cost by 7–10×, which is why it is not done at 7B+. The obstruction is not that the experiment is conceptually hard; it is that the honest version costs an order of magnitude more than the version that gets published.

## 7. Current Research (as of 2026)

- **Spectral / gradient-aligned inits.** PiSSA, LoRA-GA, and the one-step-gradient line (LoRA-One, ICML 2025, Zhang, Liu & Chen) — the direction is to prove that one full gradient step determines the correct init subspace. *(frontier — verify)* Reported recovery guarantees hold under low-rank-plus-noise assumptions on $\nabla_W\mathcal{L}$ that are not verified on real checkpoints.
- **Optimizer-geometry views.** Riemannian preconditioned LoRA (Zhang & Pilanci, ICML 2024) argues the init problem partly dissolves under the right preconditioner — the scale-invariance is absorbed into the metric.
- **$\mu$P for adapters.** Extending Tensor Programs V (Yang & Hu) to the two-factor low-rank case. Groups: Hayou/Yu (Singapore/Berkeley line), and industrial labs training adapter fleets where per-scale retuning is the actual cost driver.
- **Magnitude/direction decomposition.** DoRA (Liu et al., ICML 2024) separates the update into norm and direction, which changes the init question's shape rather than answering it.

## 8. Concrete Next Experiment

**The $\rho$-matched init ablation.**

- **Scale.** LLaMA-3-8B base, instruction-tuning on a fixed 100k-example mixture, $r \in \{8, 64, 256\}$, adapters on all linear projections, 3 seeds. ~1,500 A100-hours.
- **Arms.** (1) Default LoRA, Init[A], Kaiming, $\gamma_r = \alpha/r$. (2) rsLoRA, $\gamma_r = \alpha/\sqrt{r}$. (3) PiSSA. (4) LoRA-GA. (5) LoRA+ with $\eta_B/\eta_A = 16$.
- **Control arm — the point of the design.** Each arm gets its *own* 7-point log LR sweep, and is reported at its own best LR. Then the decisive comparison is made at matched $\rho_{100} = \|\Delta W_{100}\|_F/\|W_0\|_F$ measured on the layer-16 query projection, by adjusting each arm's LR to hit $\rho_{100} = 0.01$.
- **The deciding number.** $\Delta T = T_{\text{default}}(\epsilon^\star) - T_{\text{init}}(\epsilon^\star)$, steps to validation loss $\epsilon^\star$ = the default arm's loss at step 2000, under $\rho$-matching. If $|\Delta T| / T_{\text{default}} < 0.10$ for all data-dependent arms at all three ranks, the claimed init advantages are a learning-rate artifact and the field should standardize on rsLoRA + LoRA+ and stop publishing inits. If any arm holds $\ge 25\%$ at $r = 256$, the effect is real and rank-dependent.

## 9. Key References

- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[SOTA — theory]** Soufiane Hayou, Nikhil Ghosh, Bin Yu. *The Impact of Initialization on LoRA Finetuning Dynamics.* NeurIPS, 2024. — arXiv:2406.08447
- **[SOTA — theory]** Soufiane Hayou, Nikhil Ghosh, Bin Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML, 2024. — arXiv:2402.12354
- **[SOTA — scaling]** Damjan Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* 2023. — arXiv:2312.03732
- **[SOTA — empirical]** Fanxu Meng, Zhaohui Wang, Muhan Zhang. *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.* NeurIPS, 2024. — arXiv:2404.02948
- **[SOTA — empirical]** Shaowen Wang, Linxi Yu, Jian Li. *LoRA-GA: Low-Rank Adaptation with Gradient Approximation.* NeurIPS, 2024. — arXiv:2407.05000
- **[Related]** Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML, 2024. — arXiv:2402.09353
- **[Related]** Fangzhao Zhang, Mert Pilanci. *Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models.* ICML, 2024.
- **[Evidence]** Dan Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[Evidence]** Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* 2024. — arXiv:2410.21228
- **[Foundational — transfer]** Greg Yang, Edward J. Hu, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS, 2021. — arXiv:2203.03466

## 10. Worked Example

Take a single $4096 \times 4096$ query projection, $r = 128$, $\alpha = 16$, Kaiming-uniform $A$ so $\sigma_A^2 = 1/(3n) \approx 8.1\times10^{-5}$, $B = 0$, input RMS 1.0.

**Step 0.** $\Delta h = 0$ by construction. The $A$ gradient is $\propto B^\top(\cdot) = 0$: **$A$ receives exactly zero gradient at step 0.** Only $B$ moves. So the first update is $\Delta W_1 = \gamma_r (\eta_B G_B) A$ — its row space is fixed entirely by the random $A$. The init's *direction* is decided before any data-dependent signal reaches $A$.

**Scaling.** With $\gamma_r = \alpha/r = 16/128 = 0.125$, and $\|A x\|_2 \approx \sigma_A \sqrt{r n} \cdot 1 = 8.1{\times}10^{-5\,/2}\sqrt{128 \cdot 4096} \approx 0.0090 \times 724 \approx 6.5$. Now raise the rank to $r = 512$: $\gamma_r$ drops 4× to $0.03125$ while $\|Ax\|$ grows only $2\times$. Net adapter output falls by $2\times$ — quadrupling rank *halves* the update. rsLoRA's $\alpha/\sqrt r$ makes the two effects cancel exactly.

**Where the obstruction bites.** Under Adam with $\eta = 10^{-4}$, the $B$ update at step 1 is $\approx \eta$ per coordinate regardless of $\|Ax\|$ — the normalizer eats it. So $\rho_1 = \gamma_r \eta \sqrt{mr}/\|W_0\|_F$ depends on $\gamma_r$ but **not** on $\sigma_A$. Halve $\sigma_A$ and rerun: step-1 loss is unchanged to 3 decimals, but by step 100 the arms diverge, because the $A$ gradient (now nonzero, and $\propto B$) has a different conditioning. A paper reporting "init scale $\sigma_A/2$ converges 8% faster" at fixed $\eta$ has measured $\gamma_r$-coupled step-size drift over 100 steps, not an init effect. Only the $\rho$-matched sweep in §8 separates them.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*