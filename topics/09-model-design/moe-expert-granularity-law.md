---
id: 09-model-design/moe-expert-granularity-law
title: "Mixture-of-Experts Granularity Law"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Mixture-of-Experts Granularity Law

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/moe-expert-granularity-law` · **Status:** empirically-open

## 1. Problem Statement

A sparse Mixture-of-Experts (MoE) layer replaces one dense feed-forward network (FFN) of hidden width $d_{ff}$ with $E$ experts of width $d_e$, of which $k$ are activated per token. **Granularity** is $G = d_{ff}/d_e$: how finely the same active parameter budget is chopped up. Holding active parameters fixed means holding $k \cdot d_e$ fixed, so raising $G$ means raising $k$ proportionally.

The question: **is there a law $G^\*(C, N, D)$ giving the loss-optimal granularity as a function of compute budget $C$, total parameters $N$, and tokens $D$ — and does it survive when cost is measured in wall-clock rather than FLOPs?**

Three variants, of different difficulty:

- **Measurement.** Given a fixed cost metric, fit $\mathcal{L}(N, D, G)$ over a grid and read off $\arg\min_G$. Runnable today; the disagreement is over which cost metric is legitimate.
- **Method.** Find the architecture/routing/kernel changes that let large $G$ be realized without losing throughput. Open engineering.
- **Theory.** Explain *why* finer experts help — the expected mechanism is combinatorial: $\binom{E}{k}$ reachable expert subsets grows super-exponentially in $G$, so the layer approximates more distinct token-conditional functions per active FLOP. No proof connects this counting argument to a loss bound.

A solution is a law that predicts $G^\*$ at a held-out scale to within one factor-of-two granularity step, under a stated cost metric, and that is falsified when wrong.

## 2. Formal Setting

Let $x_t \in \mathbb{R}^{d}$ be the residual-stream activation at token $t$. A router $W_r \in \mathbb{R}^{d \times E}$ gives logits $h = W_r^\top x_t$; top-$k$ selection yields index set $\mathcal{K}_t$ and gates $g_i = \mathrm{softmax}(h)_i$. The layer output is

$$y_t = \sum_{i \in \mathcal{K}_t} g_i \, f_i(x_t), \qquad f_i(x) = W^{(i)}_{2}\,\sigma\!\left(W^{(i)}_{1} x\right),\quad W^{(i)}_1 \in \mathbb{R}^{d_e \times d}.$$

**Measured quantities.**

- $G = d_{ff}/d_e$, where $d_{ff}$ is the width the *dense* baseline of equal active parameters would use. $G$ is an integer by construction in every published grid.
- $N_{\text{act}}$: parameters touched per token, counted from the actual forward graph (routed experts $+$ shared experts $+$ attention $+$ embeddings), not from a config file.
- $N_{\text{tot}}$: all parameters, the quantity that sets HBM footprint.
- $C_{\text{flop}} \approx 6 N_{\text{act}} D$ — the Chinchilla convention, excluding router, permutation and all-to-all.
- $C_{\text{wall}}$: device-seconds at fixed hardware and a *tuned* parallelism plan. This is the metric that differs by $3\times$ from $C_{\text{flop}}$ at large $G$ (§10).
- $\mathcal{L}$: token-averaged cross-entropy in nats on a held-out split of the *training* distribution, measured after a full cosine decay — not at an intermediate checkpoint.

The candidate law, in the form reported by Krajewski et al. (ICML 2024), separates a granularity-dependent coefficient on the parameter term:

$$\mathcal{L}(N, D, G) \;=\; c \;+\; \left(\frac{g}{G^{\gamma}} + a\right) N^{-\alpha} \;+\; b\, D^{-\beta}.$$

Fitting gives $\hat{G}^\*(C)$ by minimizing under $C_{\text{flop}} = 6N_{\text{act}}D$.

**Assumptions, and which are violated.**

1. *FLOPs are the cost.* Violated. Dispatch traffic scales as $k \propto G$ at fixed active parameters while expert FLOPs stay constant.
2. *Routing quality is $G$-independent.* Violated. Load-balance loss coefficient, router z-loss and expert-capacity factor all have $G$-dependent optima; grids that fix them confound granularity with router tuning.
3. *Optimal LR/batch are $G$-independent.* Violated in practice; most grids inherit one dense-tuned schedule.
4. *One loss surface.* Violated. Downstream task accuracy and loss dissociate for MoEs more than for dense models — memorization-heavy evaluations favor large $N_{\text{tot}}$ irrespective of $G$.
5. *Separability of the $N$, $D$, $G$ terms.* Assumed, not derived. No interaction term between $G$ and $D$ is fitted, so the law cannot express "finer experts need more tokens to be worth it."

## 3. State of the Art

**Established.**

- Fine-grained experts plus shared experts beat coarse experts at equal active parameters. DeepSeekMoE (Dai et al., ACL 2024) segments each expert into 4 and isolates shared experts; the 16B model activates 2.8B parameters and matches LLaMA2-7B. Ablated against a same-budget GShard-style baseline in the paper.
- OLMoE (Muennighoff et al., ICLR 2025) ran the cleanest public granularity ablation: 64 experts, top-8, 6.9B total / 1.3B active, with controlled comparisons against fewer/larger experts at matched active parameters. Fine-grained won; the margin is small (low single-digit percent in perplexity terms) but consistent.
- The law $\mathcal{L}(N,D,G)$ of Krajewski et al. was fit over granularities up to $G=16$ at model sizes below roughly 1.3B parameters, trained on C4. Its qualitative prediction — $G^\*$ increases with compute budget — is a fit-extrapolation, not a tested extrapolation.

**Claimed but unablated.**

- That $G^\*$ keeps growing past $G = 32$. Every frontier model that ships high granularity (DeepSeek-V3: 256 routed experts of width 2048 plus 1 shared, 8 routed active, 671B total / 37B active; Qwen3 and Llama 4 in the same family) reports a *benchmark number*, not a same-budget ablation against a coarser variant. These are existence proofs that fine-grained MoE trains stably at scale, not evidence about $G^\*$.
- That Clark et al. (ICML 2022) were wrong. Their "Unified Scaling Laws for Routed Language Models" found routing gains vanishing near 900M parameters; the fine-grained literature attributes this to fixed $G=1$ and untuned expert counts. Plausible, never re-run as a direct replication.

**Systems SOTA.** MegaBlocks (Gale et al., MLSys 2023) removed token dropping via block-sparse expert kernels; grouped-GEMM kernels in production stacks make $G \approx 8$–$32$ trainable at acceptable MFU. No public kernel makes $G = 64$ throughput-neutral.

## 4. What Is Known

- Granularity gains are real but small and diminishing in loss, at the ≤1.3B-parameter, ~100B-token scale where the law was fit ($G \in \{1,\dots,16\}$, C4).
- Shared experts (always-on, $k_{\text{shared}} \ge 1$) recover part of the benefit of coarse experts by absorbing common computation; DeepSeekMoE at 2B and 16B scale.
- Sparsity and granularity are separate axes. Abnar et al. (ICML 2025) show the loss-optimal *sparsity* ($1 - k/E$) rises with compute at fixed total parameters — measured at up to a few B parameters, with granularity largely held fixed.
- Total parameters buy loss cheaply: at fixed $N_{\text{act}}$, loss falls roughly log-linearly in $N_{\text{tot}}$ over ~1.5 decades before flattening (Switch Transformer, JMLR 2022, at 7B–1.6T total).
- Routing instability grows with $E$: expert collapse and load imbalance both worsen, requiring auxiliary-loss or bias-correction schemes (DeepSeek-V3 uses auxiliary-loss-free bias updates).

## 5. What Is Not Known

- **Empirically open (primary).** Whether $G^\*$ predicted from sub-2B fits holds at 30B+ active parameters and multi-trillion-token budgets. The experiment is runnable — it is a granularity sweep at one large scale — and nobody has published it. Cost, not method, is the barrier.
- **Empirically open.** Whether $G^\*$ under $C_{\text{wall}}$ is the same integer as under $C_{\text{flop}}$. Nobody has published an iso-wall-clock granularity sweep on fixed hardware.
- **Methodologically blocked.** Whether the granularity gain is a *capability* gain or a *loss-only* gain. Downstream benchmarks at these deltas sit inside the noise of pretraining-data variation; no accepted measurement isolates the effect.
- **Theoretically open.** No lower or upper bound relates $\binom{E}{k}$ expressivity to achievable cross-entropy. The combinatorial argument is a heuristic. Also open: whether $\mathcal{L}(N,D,G)$ is separable at all, or whether a $G \times D$ interaction term is required.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by a cost metric that omits the dominant cost**.

At fixed active parameters, raising $G$ leaves expert matrix-multiply FLOPs unchanged but multiplies $k$ — and all-to-all dispatch/combine traffic is $\Theta(k \cdot B \cdot d)$ bytes, linear in $G$. So the objective being minimized ($C_{\text{flop}}$) is provably not the objective the practitioner faces ($C_{\text{wall}}$), and the gap widens exactly along the axis under study. A law fitted in FLOPs will recommend granularities that are throughput-negative.

Second obstruction: **non-identifiability against router hyperparameters**. $G$, capacity factor, and load-balance coefficient interact. A sweep that fixes the latter two attributes to granularity whatever the fixed setting happened to favor. Isolating $G$ needs a nested tuning loop, multiplying an already 8-figure-FLOP grid by the tuning budget.

## 7. Current Research (as of 2026)

- **Fine-grained + shared-expert designs at frontier scale** — DeepSeek, Qwen, Moonshot, Meta. All ship $G \gtrsim 8$ with $E \ge 128$; none publish granularity ablations.
- **Joint scaling laws including memory/serving cost** — the IDEAS NCBR / Warsaw group behind the fine-grained law has extended it toward memory-aware objectives (*Joint MoE Scaling Laws*, 2025). Direction is right; scales remain small. *(frontier — verify)*
- **Sparsity-axis laws** — Apple (Abnar et al.) and follow-ups, treating $k/E$ as the free variable.
- **Kernel work to make large $G$ cheap** — grouped GEMM, fused permute-dispatch, expert-parallel overlap. If throughput at $G=64$ reaches parity, the FLOP-based law becomes the right law. *(frontier — verify)*
- **Upcycling laws** — converting dense checkpoints to MoE constrains $G$ to divisors of $d_{ff}$, giving a natural, cheap granularity grid. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Five models, each ~3B active / ~40B total parameters, trained on 300B tokens of one fixed corpus. Granularity $G \in \{1, 2, 8, 32, 64\}$ with $k$ scaled as $k = G$ (so $k \cdot d_e$ is constant), $E$ scaled to hold $N_{\text{tot}}$ constant. Cost: roughly $6 \times 3\!\times\!10^9 \times 3\!\times\!10^{11} \approx 5.4 \times 10^{21}$ FLOPs each, ~$2.7 \times 10^{22}$ total — days on a 512-GPU cluster.

**Control arms.** (a) A dense 3B model on the identical data and schedule. (b) At each $G$, a 3-point sweep of load-balance coefficient and learning rate, taking the best — this is what removes the router confound. (c) Every run reported twice: once at iso-FLOP, once at iso-wall-clock on the same hardware with a tuned parallelism plan (the iso-wall-clock arm trains the higher-$G$ models on proportionally fewer tokens).

**The deciding number.** $\Delta = G^\*_{\text{flop}} - G^\*_{\text{wall}}$, expressed in doublings, where each $G^\*$ is the arg-min of validation loss over the five arms. $\Delta = 0$ means the FLOP-based law is safe to deploy. $\Delta \ge 2$ (two or more doublings) means the published granularity law optimizes the wrong objective and every fit needs redoing against $C_{\text{wall}}$. Secondary readout: whether $\mathcal{L}(N,D,G)$ fitted on the $\le$1.3B data predicts these five losses to within 0.01 nats.

## 9. Key References

- **[Foundational]** Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR, 2022. — arXiv:2101.03961
- **[Foundational]** Clark, de las Casas, Guy, Mensch, et al. *Unified Scaling Laws for Routed Language Models.* ICML, 2022. — arXiv:2202.01169
- **[SOTA]** Krajewski, Ludziejewski, Adamczewski, et al. *Scaling Laws for Fine-Grained Mixture of Experts.* ICML, 2024. — arXiv:2402.07871
- **[SOTA]** Dai, Deng, Zhao, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL, 2024. — arXiv:2401.06066
- **[SOTA]** Muennighoff, Soldaini, Groeneveld, et al. *OLMoE: Open Mixture-of-Experts Language Models.* ICLR, 2025. — arXiv:2409.02060
- **[SOTA]** Abnar, Shah, Busbridge, et al. *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models.* ICML, 2025.
- **[Systems]** Gale, Narayanan, Young, Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[Systems]** Zhou, Lei, Liu, et al. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS, 2022. — arXiv:2202.09368
- **[Empirical]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Empirical]** Jiang, Sablayrolles, Roux, et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Reference]** Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Survey]** Cai, Jiang, Wang, et al. *A Survey on Mixture of Experts.* 2024.

## 10. Worked Example

One MoE layer. $d = 2048$, dense-equivalent $d_{ff} = 8192$, gated FFN (3 weight matrices), $B = 16{,}384$ tokens per step per expert-parallel group, bf16.

Expert FLOPs, identical at every $G$ because $k \cdot d_e = 8192$ is fixed:

$$6 \cdot B \cdot d \cdot d_{ff} = 6 \times 16384 \times 2048 \times 8192 \approx 1.65 \times 10^{12}\ \text{FLOPs}.$$

At 400 TFLOP/s achieved, that is **4.1 ms**.

Dispatch plus combine traffic is $2 \cdot k \cdot B \cdot d \cdot 2$ bytes:

| $G$ | $k$ | $d_e$ | all-to-all bytes | comms @ 400 GB/s | expert compute | layer step |
|---|---|---|---|---|---|---|
| 8 | 8 | 1024 | 1.07 GB | 2.7 ms | 4.1 ms | ~4.1 ms (overlapped) |
| 32 | 32 | 256 | 4.29 GB | 10.7 ms | 4.1 ms | ~10.7 ms |
| 64 | 64 | 128 | 8.59 GB | 21.5 ms | 4.1 ms | ~21.5 ms |

Compute is unchanged; communication grows $8\times$ from $G=8$ to $G=64$ and becomes the critical path. Wall-clock per layer rises **$\approx 5.2\times$** while $C_{\text{flop}}$ says the three rows cost the same.

Now the obstruction. Extrapolating the fitted term $g/G^{\gamma}$ from the published sub-2B fits, the loss improvement from $G=8$ to $G=64$ is on the order of $0.01$–$0.02$ nats — worth roughly 20–40% extra tokens under a Chinchilla-style trade. But the $5.2\times$ wall-clock penalty buys, at constant budget, a $5.2\times$ token reduction, which costs far more than $0.02$ nats. Under $C_{\text{flop}}$, $G^\* = 64$; under $C_{\text{wall}}$ on this hardware, $G^\* = 8$. The law and the machine disagree by three doublings, and no published experiment measures which side is right at frontier scale. That gap — not the fit quality — is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*