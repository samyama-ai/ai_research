---
id: 15-mixture-of-experts/dense-to-moe-upcycling-optimality
title: "Upcycling Dense Checkpoints Into MoE Optimally"
topic: 15-mixture-of-experts
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Upcycling Dense Checkpoints Into MoE Optimally

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/dense-to-moe-upcycling-optimality` · **Status:** partially-solved

## 1. Problem Statement

You have a trained dense transformer checkpoint $\theta_D$ that cost $C_D$ FLOPs, and a remaining budget $C_U$ FLOPs. You may either continue training $\theta_D$ densely, or apply an **upcycling map** $U$ that converts some or all FFN blocks into Mixture-of-Experts layers and then train the result. The question: for which $(C_D, C_U, N, E, k)$ does upcycling win, and which map $U$ is optimal?

Three variants, of very different difficulty:

- **Measurement.** Under what budget accounting does "upcycling wins" hold? Training FLOPs, total tokens, wall-clock on a fixed cluster, and serving cost under a fixed inference load give different answers and different winners. Most published claims fix one and are silent on the others.
- **Method.** Which $U$ minimizes final loss? Naive replication (copy the dense FFN $E$ times, random router), partial re-initialization (Drop-Upcycling), expert splitting (slice $d_{ff}$ into $E$ disjoint chunks), and merge-based construction (Branch-Train-MiX) are all in use. No head-to-head at matched budget across more than two scales exists.
- **Theory.** Is there a characterization of the **crossover budget** $C^*(C_D, N, E)$ — the extra compute above which the upcycled model overtakes the dense continuation — and does it follow a scaling law? Nothing resembling a proof exists.

Solved means: a predictive rule, fit at small scale and validated at large, that outputs both $C^*$ and the choice of $U$ from $(C_D, N, E, k, G)$ alone.

## 2. Formal Setting

Dense model $\theta_D$ with $N$ non-embedding parameters, of which each FFN layer holds $W_{\text{in}} \in \mathbb{R}^{d \times d_{ff}}$, $W_{\text{out}} \in \mathbb{R}^{d_{ff} \times d}$. Upcycling produces expert set $\{W^{(e)}\}_{e=1}^{E}$ and a router $W_r \in \mathbb{R}^{d \times E}$, with top-$k$ gating

$$g(x) = \mathrm{top}\text{-}k\big(\mathrm{softmax}(W_r^\top x)\big), \qquad y = \sum_{e} g_e(x)\, f_{W^{(e)}}(x).$$

**Quantities as measured.**

- **Init map.** Naive: $W^{(e)} \leftarrow W$ for all $e$, $W_r \sim \mathcal{N}(0,\sigma^2)$. Drop-Upcycling with ratio $r$: sample index set $S_e \subset [d_{ff}]$, $|S_e| = r\,d_{ff}$, re-initialize those columns/rows from the *empirical statistics* of the dense weights (per-tensor mean and variance of $W_{\text{in}}$), keep the rest. Splitting: partition $[d_{ff}]$ into $E$ blocks, $d_{ff}^{\text{expert}} = d_{ff}/E$. Granularity $G = d_{ff}/d_{ff}^{\text{expert}}$.
- **Training compute.** $C \approx 6 \cdot N_{\text{act}} \cdot T$ for $T$ tokens, $N_{\text{act}}$ the per-token active parameters. This ignores router, all-to-all communication, and expert-parallel bubbles, which at $E \ge 64$ can add 10–30% wall-clock at fixed FLOPs — the first assumption known to be violated.
- **Crossover.** With $L_{\text{MoE}}(C_U \mid U)$ and $L_{\text{dense}}(C_U)$ the validation losses after spending $C_U$ from the same $\theta_D$,
$$C^\star(U) = \inf\{\,C_U : L_{\text{MoE}}(C'\mid U) < L_{\text{dense}}(C') \ \forall C' \ge C_U\,\},$$
reported as the ratio $\rho = C^\star / C_D$.
- **Effective-parameter accounting.** Routed scaling laws express MoE loss as a function of active parameters and $E$; the fine-grained MoE law of Ludziejewski et al. (ICML 2024) adds $G$ as a third axis. Upcycling violates the law's premise that training starts from random init, so its coefficients are not transferable without refitting — the second violated assumption.
- **Serving cost.** $\mathrm{Cost}_{\text{inf}} \propto N_{\text{act}} \cdot T_{\text{served}} + \alpha N_{\text{total}}$, with $\alpha$ the memory-residency penalty. Papers reporting FLOP-matched wins almost never report this term.

**Symmetry problem.** Under naive upcycling all $E$ experts are identical, so the loss is invariant to any permutation of router columns and the gradient of expert diversity is zero at init; differentiation is driven entirely by router noise and batch stochasticity. This is the structural reason $U$ matters.

## 3. State of the Art

**Established (ablated, reproduced in some form).**

- *Sparse Upcycling* (Komatsuzaki et al., ICLR 2023, arXiv:2212.05055) — the reference method. On T5-Base/Large/XL and ViT-B/32, upcycled MoEs beat both the original checkpoint and a dense continuation, but only after a non-trivial fraction of the original pretraining budget is re-spent; below that, dense continuation wins. The paper's own ablations cover expert count, router type, and the choice to upcycle every-other layer.
- *Drop-Upcycling* (Nakamura et al., ICLR 2025, arXiv:2502.19261) — partial re-initialization of expert weights using the dense weights' own statistics. Ablated over re-init ratio $r$; intermediate $r$ (around $0.5$) beats both $r=0$ (naive) and $r=1$ (random init) at the same token budget, at 1.5B-class and 3.7B-class dense sources trained for hundreds of billions of tokens.
- *Sparse Upcycling: Inference Inefficient Finetuning* (Doubov, Sardana, Chiley, 2024, arXiv:2411.08968) — the counter-result. Under inference-aware accounting, upcycling's advantage shrinks or inverts; dense continuation is preferable at high serving-to-training ratios.

**Claimed but unablated.**

- NVIDIA's upcycling recipe (He et al., 2024, arXiv:2410.07524) introduces "virtual group" initialization for fine-grained MoE and a weight-scaling factor, reporting roughly a 1.5-point MMLU gain over continued dense training for a Nemotron-class 15B dense source at a 1T-token budget. Reported at one scale; the scaling factor is not ablated against the softmax-normalization alternative.
- *Qwen1.5-MoE-A2.7B* (Qwen team, 2024) — 14.3B total / 2.7B active, upcycled from a 1.8B dense model, with a claimed ~75% reduction in training cost versus a 7B dense peer. This is a benchmark and cost number in a blog release, not a controlled experiment.
- *LLaMA-MoE* (Zhu et al., 2024) and *Branch-Train-MiX* (Sukhbaatar et al., 2024, arXiv:2403.07816) construct MoEs by splitting or by merging separately-trained domain experts. Neither is compared against naive upcycling at matched compute from the same source checkpoint.

**Theory SOTA.** Unified scaling laws for routed models (Clark et al., ICML 2022) and fine-grained MoE scaling laws (Ludziejewski et al., ICML 2024) are from-scratch laws. There is no scaling law with $C_D$ as an input variable. $C^\star$ has never been predicted before being measured.

## 4. What Is Known

- Upcycling has a **crossover**, not a uniform advantage. Komatsuzaki et al. observe dense continuation ahead early and upcycled MoE ahead later; the crossing occurs at a substantial fraction of the original budget (order 10–20% of $C_D$ in their T5 and ViT settings, varying by scale).
- **Identical experts are a real cost.** Drop-Upcycling's central measured result is that breaking init symmetry by partial re-init improves final loss at fixed tokens, and that the effect grows with token budget — naive upcycling's experts stay correlated for a long time.
- **Re-init ratio has an interior optimum.** $r \approx 0.5$ beat $r \in \{0, 1\}$ in the ICLR 2025 ablations at 1.5B–3.7B dense sources.
- **Router init matters more than expert init early.** Small router noise plus load-balancing loss coefficient (typically $\lambda \approx 10^{-2}$) determines whether experts specialize at all; without balancing loss, upcycled MoEs collapse to a single expert.
- **Inference accounting can flip the verdict** (Databricks 2024): at fixed quality, MoEs are cheaper per training FLOP and more expensive per served token in memory-bound regimes.
- **Fine-grained experts help from scratch.** Granularity $G$ in the 8–32 range improves the compute-optimal frontier (Ludziejewski et al., ICML 2024), measured up to ~7B total parameters. Whether the same $G$ optimum survives upcycling is untested.

## 5. What Is Not Known

- **Theoretically open.** No characterization of $C^\star(C_D, N, E, k)$. No proof that any $U$ dominates another, and no theory of how long the identical-expert symmetry persists under SGD with load-balancing regularization.
- **Empirically open.** A four-way head-to-head — naive / Drop-Upcycling / splitting / BTX — from one shared dense checkpoint at matched FLOPs, run at two scales to check whether the ranking is scale-stable. Fully runnable today; nobody has published it.
- **Empirically open.** Optimal $E$ and $G$ *as a function of $C_D$*. Every recipe fixes $E \in \{8, 16, 64\}$ by convention.
- **Methodologically blocked.** "Upcycling wins" has no agreed budget definition. Training FLOPs, wall-clock, total parameter memory, and amortized serving cost yield different orderings, and no paper reports all four. Until the objective is pinned, the comparisons are not commensurable.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability of the init**. The upcycled model and the dense control differ simultaneously in parameter count, active parameters, optimizer-state reset, learning-rate schedule restart, and data ordering. A reported win can come from any of these — in particular, re-warming the learning rate on a converged dense checkpoint is itself worth measurable loss, and most upcycling arms get that for free while the dense arm does not.

Second, the decisive comparison is expensive in exactly the regime where it matters. Crossover happens at $\rho \sim 0.1$–$0.2$ of the original pretraining run; for a 15B model trained on 8T tokens, that is ~1T tokens *per arm*, and you need at least four arms. Small-scale proxies are unreliable because $\rho$ appears to depend on how converged $\theta_D$ is, which is precisely the quantity that does not transfer downward.

## 7. Current Research (as of 2026)

- **Partial re-init and diversification.** Drop-Upcycling (LLM-jp / Swallow, ICLR 2025) is the current default for symmetry-breaking; follow-ups explore statistics-preserving noise per-tensor rather than per-layer *(frontier — verify)*.
- **Fine-grained upcycling.** NVIDIA's virtual-group init targets $G > 1$ experts from a single dense FFN; the open question is whether granularity gains observed from scratch survive.
- **Inference-aware upcycling.** Databricks/Mosaic-line work on serving-cost-inclusive comparisons; the natural next step is a joint train+serve optimality frontier.
- **Merge-then-route.** Branch-Train-MiX descendants building experts from domain-specialized continuations rather than copies (Meta and academic groups) *(frontier — verify)*.
- **Upcycling attention.** Extending $U$ beyond FFNs to KV heads or attention experts; reported at small scale only.

## 8. Concrete Next Experiment

**Scale.** One dense source: a 1.5B-parameter transformer pretrained on 300B tokens ($C_D \approx 2.7 \times 10^{21}$ FLOPs). Budget per arm: 100B tokens ($\rho = 0.33$), five arms, ~$10^{21}$ FLOPs total per arm.

**Arms.**
1. **Control:** continued dense training, identical LR re-warm schedule, identical data order.
2. Naive upcycling, $E=8$, top-2.
3. Drop-Upcycling, $r=0.5$, $E=8$, top-2.
4. Expert splitting, $E=8$, $d_{ff}/8$ each (active params matched to arm 1).
5. Drop-Upcycling at granularity $G=8$, $E=64$, top-16 (active params matched to arm 3).

Every arm shares the LR schedule, data order, and load-balance coefficient. Arms 4 and 5 are the ones that isolate capacity from init — arm 4 has the *same* active parameter count as the dense control.

**Deciding number.** $\rho^\star$ = fraction of $C_D$ at which each arm's validation loss first drops below the control's and stays below, read off logged curves at 5B-token resolution. The question is settled for this scale if the ordering of $\rho^\star$ across arms 2–5 is stable under a repeat at 400M parameters. Secondary: expert pairwise cosine similarity $\bar{s} = \mathbb{E}_{e \ne e'} \cos(W^{(e)}, W^{(e')})$ at 10B tokens — the prediction is that $\rho^\star$ is monotone increasing in $\bar{s}$. If it is not, symmetry-breaking is not the mechanism and the field's current explanation is wrong.

## 9. Key References

- **[Foundational]** Komatsuzaki, Puigcerver, Lee-Thorp, Ruiz, Mustafa, Ainslie, Tay, Dehghani, Houlsby. *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints.* ICLR, 2023. — arXiv:2212.05055
- **[SOTA]** Nakamura, Mizuki, Fujii, et al. *Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization.* ICLR, 2025. — arXiv:2502.19261
- **[SOTA]** He, Sun, Shen, et al. *Upcycling Large Language Models into Mixture of Experts.* NVIDIA, 2024. — arXiv:2410.07524
- **[Counterpoint]** Doubov, Sardana, Chiley. *Sparse Upcycling: Inference Inefficient Finetuning.* 2024. — arXiv:2411.08968
- **[Theory]** Clark, de las Casas, Guy, et al. *Unified Scaling Laws for Routed Language Models.* ICML, 2022. — arXiv:2202.01169
- **[Theory]** Ludziejewski, Krajewski, Adamczewski, et al. *Scaling Laws for Fine-Grained Mixture of Experts.* ICML, 2024. — arXiv:2402.07871
- **[Related]** Sukhbaatar, Golovneva, Sharma, et al. *Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM.* 2024. — arXiv:2403.07816
- **[Related]** Zhu, Wei, Qu, et al. *LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training.* EMNLP, 2024.
- **[Systems]** Wei, Zhu, Zhao, et al. *Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models.* 2024. — arXiv:2406.06563

## 10. Worked Example

Take a 1.5B dense model, $d = 2048$, $d_{ff} = 8192$, 24 layers. FFN parameters: $24 \times 2 \times 2048 \times 8192 \approx 0.80$B, roughly 55% of non-embedding parameters.

Naive upcycle to $E=8$, top-2, all layers. Total parameters go $1.5\text{B} \to 1.5 + 7 \times 0.80 = 7.1$B. Active parameters: $1.5 - 0.80 + 2 \times 0.10 = 0.90$B... but only if experts are $d_{ff}/8$; with full-size copies active params are $1.5 - 0.80 + 2 \times 0.80 = 2.3$B. So the "same active compute" framing is already false: naive upcycling at top-2 with full-width experts costs **1.53× the dense forward FLOPs**, before any router or all-to-all overhead.

Now the accounting. Dense control at 100B tokens: $6 \times 1.5\text{e}9 \times 1\text{e}11 = 9.0 \times 10^{20}$ FLOPs. Upcycled arm at the same 100B tokens: $6 \times 2.3\text{e}9 \times 1\text{e}11 = 1.38 \times 10^{21}$ FLOPs — 53% more. A FLOP-matched comparison must therefore give the upcycled arm only 65B tokens.

**This is where the obstruction becomes visible.** Published upcycling wins are frequently token-matched, not FLOP-matched, and the two differ by exactly this 1.53× at $E=8$/top-2 — a gap large enough to account for a 1–2 point benchmark move on its own. Add the memory term: 7.1B total parameters at bf16 is 14.2 GB of weights versus 3.0 GB dense, so at low batch size the upcycled model is memory-bandwidth-bound and its per-token latency tracks 7.1B, not 2.3B. Under a serving load of $10^{13}$ tokens, the residency and bandwidth penalty exceeds the entire training saving.

The measured quantity that decides it — $\rho^\star$ computed under a *stated* budget definition, with the LR schedule held identical across arms — has not been reported for any model above 4B parameters.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*