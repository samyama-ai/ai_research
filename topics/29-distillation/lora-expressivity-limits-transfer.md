---
id: 29-distillation/lora-expressivity-limits-transfer
title: "LoRA Expressivity Limits for Transfer"
topic: 29-distillation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# LoRA Expressivity Limits for Transfer

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/lora-expressivity-limits-transfer` · **Status:** partially-solved

## 1. Problem Statement

Low-rank adaptation (LoRA) constrains a fine-tuning update to $\Delta W = BA$ with $\mathrm{rank}(\Delta W) \le r \ll \min(d,k)$. The question: **for which transfer targets does this constraint cost accuracy, and how does the minimum sufficient rank scale with the distance between source and target task?**

Three variants, routinely conflated:

- **Theory.** Given a frozen network $f_0$ and a target function $g$, what rank suffices for some LoRA parametrization to *represent* $g$ exactly or to $\epsilon$? This is a representability question and is largely answered.
- **Method.** Given a fixed compute and data budget, does gradient descent on $(A,B)$ *reach* a solution as good as full fine-tuning? Representability does not imply reachability.
- **Measurement.** Is there a task-side quantity — computable before training — that predicts the critical rank $r^*$? This is the variant that is blocked.

Solving the problem means: a predictor $\hat r$, computable from the frozen model and a sample of the target data, such that LoRA at rank $\hat r$ matches full fine-tuning within a stated tolerance across held-out task families.

## 2. Formal Setting

Frozen base weights $W_0 \in \mathbb{R}^{d\times k}$ per adapted module, update $\Delta W = \frac{\alpha}{r} B A$, $B \in \mathbb{R}^{d\times r}$ initialized to $0$, $A \in \mathbb{R}^{r\times k}$ initialized Gaussian. Adapted module set $\mathcal{M}$ (attention projections only, or $+$ MLP).

**Transfer gap**, measured as the difference in target-task loss at matched wall-clock or matched token budget $C$:
$$G(r; \tau, C) = \mathcal{L}_\tau\big(\theta^{\text{LoRA}}_{r}(C)\big) - \mathcal{L}_\tau\big(\theta^{\text{full}}(C)\big)$$
Both arms tuned independently over learning rate (LoRA optima sit $\sim 10\times$ higher), warmup, and epochs; report the best of $\ge 5$ seeds and the seed variance, since $G$ is often smaller than seed noise.

**Critical rank**: $r^*(\tau,\epsilon) = \min\{r : G(r;\tau,C) \le \epsilon\}$, measured by bisection over $r \in \{1,2,4,\dots,512\}$.

**Update spectrum**: run full fine-tuning, form $\Delta W^{\text{full}} = W^{\text{full}} - W_0$, take singular values $\sigma_1 \ge \sigma_2 \ge \dots$. Report the **stable rank** $\mathrm{srank} = \|\Delta W\|_F^2/\|\Delta W\|_2^2$ and the effective rank $r_{0.9} = \min\{j : \sum_{i\le j}\sigma_i^2 \ge 0.9\|\Delta W\|_F^2\}$.

**Assumptions, with the ones known to fail marked:**

1. Full fine-tuning is the ceiling. **Violated** — LoRA sometimes wins under distribution shift because it forgets less (Biderman et al., 2024).
2. $r^*$ is a property of the task. **Violated** — it moves with $\alpha$, initialization, and $\mathcal{M}$; the $\alpha/r$ scaling is itself unstable at large $r$ and should be $\alpha/\sqrt r$ (Kalajdzievski, 2023).
3. $r_{0.9}$ of the full update upper-bounds $r^*$. **Not established** — a rank-$r$ update in a *different* subspace can match the loss without approximating $\Delta W^{\text{full}}$.
4. Linearized (NTK) dynamics. **Violated** at the scales and learning rates actually used.

## 3. State of the Art

**Theory SOTA — established.** Zeng & Lee (ICLR 2024) give exact-representation results: for a frozen fully-connected network of width $D$ and depth $L$, a target network of depth $\bar L$ is exactly representable by LoRA of rank $r \ge D\lceil \bar L / L\rceil$; for transformers, rank $\approx D/2$ suffices under stated conditions. This bounds the *worst case* and is far above ranks used in practice — the gap between $D/2$ and $r=16$ is the entire empirical question. Jang et al. (ICML 2024) show that in the NTK regime with $r \gtrsim \sqrt{N}$ ($N$ = training examples), LoRA has no spurious local minima; the NTK assumption is the load-bearing one.

**Empirical SOTA — established.** Biderman et al. (TMLR 2024) is the cleanest negative result: matched-budget LoRA vs full fine-tuning on Llama-2 7B/13B, code and math, both continued pretraining (up to 20B tokens) and instruction tuning.

**Claimed but unablated.** DoRA (Liu et al., ICML 2024), PiSSA (Meng et al., NeurIPS 2024), LoRA-GA, and rank-adaptive AdaLoRA (Zhang et al., ICLR 2023) all report closing the gap. Most gains are benchmark deltas of 0.5–2 points on GLUE/commonsense suites at a single rank, without a rank sweep against a learning-rate-tuned full-FT control. Treat as benchmark numbers, not established mechanism.

## 4. What Is Known

- **Instruction tuning is low-rank; skill acquisition is not.** Biderman et al. (2024), Llama-2 7B: on instruction tuning LoRA at $r=16$–$256$ tracks full fine-tuning closely; on continued pretraining over 20B tokens of code, LoRA at every tested rank stays clearly behind, and the gap does not close by raising $r$ within the tested range.
- **Full fine-tuning updates are high-rank.** Same paper: measured singular spectra of $\Delta W^{\text{full}}$ have effective rank 10–100$\times$ larger than the LoRA ranks that are typically deployed.
- **Task subspaces can be tiny.** Aghajanyan et al. (ACL 2021): RoBERTa-large reaches 90% of full fine-tuning performance on MRPC within a random subspace of a few hundred dimensions ($d_{90} \approx 200$ for MRPC, larger for QQP). Intrinsic dimension *shrinks* with pretraining quality and model size.
- **Forgetting is the compensating benefit.** LoRA degrades base-model capabilities (HellaSwag, HumanEval on the non-target domain) less than full fine-tuning at matched target gain — measured at 7B/13B.
- **The learned subspace is structurally different.** Shuttleworth et al. (2024) report "intruder dimensions": singular vectors in the LoRA update with little overlap with the pretrained spectrum, correlating with worse out-of-distribution behavior even at matched in-distribution loss.
- **Hyperparameters, not rank, explain much of the reported gap.** LoRA+ (Hayou et al., ICML 2024) shows $B$ needs a larger learning rate than $A$; rsLoRA fixes the $\alpha/r$ scaling. Both move accuracy by amounts comparable to published rank effects.

## 5. What Is Not Known

- **Theoretically open.** No non-vacuous lower bound: no theorem saying "for target family $\mathcal{G}$ and frozen model $f_0$, every rank-$r$ adaptation with $r < r_0$ incurs loss $\ge \epsilon$." All current results are sufficiency (upper) bounds. Also open: whether the optimization gap at fixed $r$ vanishes with better initialization or is intrinsic.
- **Empirically open.** Whether $r^*$ grows with pretraining scale, shrinks, or is flat. Every clean measurement is at $\le 13$B. The experiment — rank sweep $\times$ model scale $\times$ task distance, with a tuned full-FT control — is runnable and unrun.
- **Methodologically blocked.** "Task distance" has no agreed measurement. Candidates (KL between base and target-tuned output distributions, gradient-subspace overlap, $d_{90}$ intrinsic dimension, Fisher-metric distance) have never been compared on the same tasks against measured $r^*$. Without this, $r^*$ can be measured per task but not predicted, so no scaling law can be fit.

## 6. Why It Is Hard

**The measurement is confounded by the optimizer, and the confound is the same size as the effect.** $G(r)$ is a difference between two independently tuned training runs. LoRA's optimal learning rate is roughly an order of magnitude above full fine-tuning's, its optimal $\alpha$ interacts with $r$, and the standard $\alpha/r$ scaling systematically underuses large ranks. A poorly tuned $r=256$ run can lose to a well-tuned $r=16$ run, producing the false conclusion that rank saturates. Doing it properly means a 2-D sweep (rank $\times$ learning rate, $\ge 5$ seeds) against a *separately* swept full-FT control — roughly $50$–$100$ full fine-tuning runs per model scale.

Second obstruction: **non-identifiability of the subspace.** Two rank-$r$ updates with near-zero subspace overlap can reach the same loss, so $\Delta W^{\text{full}}$'s spectrum does not certify what rank is needed — only what rank one particular optimizer used.

## 7. Current Research (as of 2026)

- **Rank-restarting and high-rank-from-low-rank training** — ReLoRA (Lialin et al., ICLR 2024) and GaLore (Zhao et al., ICML 2024) accumulate high-rank updates from low-rank steps; the open question is whether they close Biderman's continued-pretraining gap. *(frontier — verify)*
- **Initialization as the lever** — PiSSA (principal singular vectors), LoRA-GA (gradient alignment), and Hayou et al.'s init-asymmetry analysis. Direction: show the gap is optimization, not expressivity.
- **Spectral diagnostics of adapters** — intruder-dimension analyses and their link to OOD degradation (MIT/Bosch-adjacent groups). *(frontier — verify)*
- **Extreme compression** — VeRA (Kopiczko et al., ICLR 2024) shares frozen random matrices and trains only scaling vectors, pushing the lower end of the expressivity curve.
- **PEFT surveys** — Han et al. (TMLR 2024) as the current map.

## 8. Concrete Next Experiment

**Question:** does $r^*$ scale with model size?

**Scale.** Three base models: Llama-3.2 1B, 3B, and 8B (or Qwen2.5 equivalents). Two targets at different distances: (a) *near* — instruction tuning on 100M tokens of a chat mixture; (b) *far* — continued pretraining on 20B tokens of a language absent from the pretraining mix (e.g. Telugu or Kazakh). Ranks $r \in \{4,16,64,256,1024\}$, LoRA on all linear modules, rsLoRA scaling ($\alpha/\sqrt r$), learning rate swept over 4 values per cell, 3 seeds.

**Control arm.** Full fine-tuning on identical data, identical token budget, its own 4-point learning-rate sweep, 3 seeds. Report seed standard deviation alongside every mean; a gap smaller than $2\sigma$ is not a gap.

**Deciding number.** $r^*(\epsilon)$ with $\epsilon = 0.01$ nats of held-out target loss, per (model size, target). The decision: fit $\log r^* = a\log P + b$ over parameter count $P$. **If $a > 0.15$, LoRA's expressivity limit worsens with scale and the method is a stopgap; if $|a| < 0.05$, $r^*$ is scale-free and rank can be set from task distance alone; if $a < -0.15$, larger models need less rank and LoRA improves with scale.** One number, $a$, decides it.

## 9. Key References

- **[Foundational]** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. — arXiv:2106.09685
- **[Foundational]** Aghajanyan, Zettlemoyer, Gupta. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL 2021. — arXiv:2012.13255
- **[Theory SOTA]** Zeng, Lee. *The Expressive Power of Low-Rank Adaptation.* ICLR 2024. — arXiv:2310.17513
- **[Theory]** Jang, Kim, Hwang, Yun. *LoRA Training in the NTK Regime has No Spurious Local Minima.* ICML 2024. — arXiv:2402.11867
- **[Empirical SOTA]** Biderman, Ortiz, Portes, Paul, Greengard, Jennings, King, Havens, Chiley, Frankle, Blakeney, Cunningham. *LoRA Learns Less and Forgets Less.* TMLR 2024. — arXiv:2405.09673
- **[Diagnostic]** Shuttleworth, Andreas, Torralba, Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence.* 2024. — arXiv:2410.21228
- **[Method]** Liu, Wang, Yin, Molchanov, Wang, Cheng, Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024. — arXiv:2402.09353
- **[Method]** Hayou, Ghosh, Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models.* ICML 2024. — arXiv:2402.12354
- **[Method]** Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* 2023. — arXiv:2312.03732
- **[Method]** Lialin, Muckatira, Shivagunde, Rumshisky. *ReLoRA: High-Rank Training Through Low-Rank Updates.* ICLR 2024. — arXiv:2307.05695
- **[Method]** Zhao, Zhang, Chen, Wang, Anandkumar, Tian. *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* ICML 2024. — arXiv:2403.03507
- **[Survey]** Han, Gao, Liu, Zhang, Zhang. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR 2024. — arXiv:2403.14608

## 10. Worked Example

Take one attention output projection of a 7B model: $d = k = 4096$, so $\mathrm{rank}(\Delta W)$ can be up to 4096 and full fine-tuning of this matrix trains $16.8$M parameters. LoRA at $r=16$ trains $2 \times 4096 \times 16 = 131{,}072$ parameters — **0.78%**.

Now the two competing predictions of $r^*$:

- **Intrinsic-dimension view.** Aghajanyan measured $d_{90} \approx 200$ *for the whole model* on MRPC with RoBERTa-large. Spread over $\sim 300$ adapted matrices, that is well under one rank-1 direction per matrix. Predicted $r^* = 1$.
- **Spectral view.** Zeng & Lee's sufficiency bound for transformers is $r \approx D/2 = 2048$. Biderman's measured effective rank of $\Delta W^{\text{full}}$ on code continued-pretraining is 10–100$\times$ typical LoRA rank, i.e. $r_{0.9}$ in the hundreds to low thousands. Predicted $r^* \approx 10^2$–$10^3$.

Three orders of magnitude apart, and both are *measurements*, not guesses. The obstruction is visible here: they measure different things. $d_{90}$ measures how few directions suffice when they are chosen *jointly across the whole model* by optimization; $r_{0.9}$ measures how many directions one SGD trajectory *happened to use per matrix*, inflated by gradient noise that contributes small singular values without contributing loss reduction.

Concretely: if 90% of $\|\Delta W^{\text{full}}\|_F^2$ sits in 400 directions but 88% sits in 12, then $r_{0.9} = 400$ while the loss-relevant rank may be 12. Nothing in the current literature separates these, because no one has projected $\Delta W^{\text{full}}$ onto its top-$j$ subspace and re-measured target loss as a function of $j$. That single ablation — cheap, one extra forward pass per $j$ — would tell you which of the two predictions above is measuring the thing its name claims.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*