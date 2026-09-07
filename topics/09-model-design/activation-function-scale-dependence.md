---
id: 09-model-design/activation-function-scale-dependence
title: "Activation Function Choice at Scale"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Activation Function Choice at Scale

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/activation-function-scale-dependence` · **Status:** empirically-open

## 1. Problem Statement

Every dense transformer feed-forward block picks a nonlinearity: ReLU, GELU, SwiGLU, GEGLU, squared ReLU. The field converged on SwiGLU (PaLM, LLaMA, Qwen, Mistral) on the strength of ablations run at $10^8$–$10^9$ parameters and $\sim 10^{19}$ FLOPs. The open question is whether that choice survives extrapolation.

Three distinct variants, usually conflated:

- **Measurement.** Does the loss gap between activation $a$ and activation $b$, at matched parameters *and* matched training FLOPs, shrink, hold, or grow as compute $C \to 10^{24}$? Equivalently: does the activation change the *coefficient* of the scaling law, or its *exponent*?
- **Method.** Given a fixed budget of small-scale runs, is there a procedure that predicts the large-scale winner? Small-scale ablation is the current procedure and it is known to be unreliable for other architectural knobs.
- **Theory.** Is there any account — signal-propagation, kernel, or optimization-geometry — that predicts the *sign* of the gap between two smooth gated activations before running the experiment? No such account exists.

Solving it means: a scaling-law fit for at least three activation families over $\geq 3$ decades of compute, with confidence intervals on the exponent difference tight enough to reject "the exponent is the same."

## 2. Formal Setting

FFN block on input $x \in \mathbb{R}^{d}$, hidden width $d_\text{ff}$:

$$\text{FFN}_\sigma(x) = W_2\,\sigma(W_1 x), \qquad \text{FFN}_{\sigma\text{GLU}}(x) = W_2\big(\sigma(W_1 x) \odot V x\big)$$

Gated variants carry three matrices, so parameter matching requires $d_\text{ff}^{\text{GLU}} = \tfrac{2}{3} d_\text{ff}^{\text{dense}}$ (in practice $\tfrac{8}{3}d$ against $4d$).

**Measured quantities.**

- $N$ — non-embedding parameter count, counted exactly, not nominal.
- $D$ — training tokens; $C \approx 6ND$ FLOPs. Report both $N$ and $D$, never $C$ alone.
- $L_\sigma(N, D)$ — validation cross-entropy in nats/token on a held-out shard of the *training* distribution, tokenizer fixed across arms.
- Fitted law per activation:
$$L_\sigma(N,D) = E_\sigma + \frac{A_\sigma}{N^{\alpha_\sigma}} + \frac{B_\sigma}{D^{\beta_\sigma}}$$
The decision predicate is on $\Delta\alpha = \alpha_a - \alpha_b$ and $\Delta E = E_a - E_b$. A pure coefficient effect ($\Delta\alpha = 0$, $A_a < A_b$) means the gap is a constant compute *offset* — worth a fixed multiplier, never more. An exponent effect means the choice compounds.
- **Compute-matched gap.** $g(C) = L_a(C) - L_b(C)$ along each arm's own compute-optimal frontier, not at fixed $N$. Reporting at fixed $N$ silently gives the gated arm more FLOPs per token.
- **Wall-clock gap.** $\tilde g$ = same comparison at matched accelerator-seconds. SwiGLU costs ~1.5× the FFN matmul FLOPs of a dense-ReLU FFN at equal $d_\text{ff}$, and the elementwise gate is memory-bound.

**Assumptions, and which are violated.**

1. *Hyperparameters are transferred fairly.* Violated by default: optimal learning rate depends on the activation; a single tuned LR favors whichever arm it was tuned on. µP (Yang & Hu, 2021) partially fixes this but its transfer guarantees are not established for gated activations specifically.
2. *A single power law holds across the sweep.* Violated near the small end and under data repetition.
3. *Validation loss is the objective.* Violated: downstream task deltas at these effect sizes ($<0.02$ nats) are inside benchmark noise.
4. *Numerics are irrelevant.* Violated in bf16 — the gate product $\sigma(W_1x)\odot Vx$ has a wider dynamic range than a single activation and interacts with attention-logit growth and loss spikes.

## 3. State of the Art

**Empirical SOTA (established).** Shazeer, *GLU Variants Improve Transformer* (2020, arXiv:2002.05202): T5-Base-scale (~220M params) span-corruption pretraining, GEGLU/SwiGLU improve log-perplexity over ReLU FFN by roughly 0.05 nats, with parameter-matched $d_\text{ff}$. The paper explicitly declines to explain the result ("divine benevolence"). This is the single ablation the whole field's default rests on.

**Independent replication (established).** Narang et al., *Do Transformer Modifications Transfer Across Implementations and Applications?* (EMNLP 2021, arXiv:2102.11972) re-ran dozens of architectural modifications in one codebase and found most do not transfer; GLU variants were among the small set that did. Scale: T5-Base/Large, still $\leq 10^9$ params.

**Claimed but unablated.** PaLM (Chowdhery et al., JMLR 2023) and LLaMA (Touvron et al., 2023) both adopt SwiGLU and cite Shazeer; neither runs a controlled ReLU arm at their own scale. Every 100B+ SwiGLU model is evidence that SwiGLU *works*, not that it *wins*.

**Counter-direction.** Mirzadeh et al., *ReLU Strikes Back* (ICLR 2024, arXiv:2310.04564): ReLU yields ~90% activation sparsity in the FFN, cutting inference FLOPs several-fold, at close-to-parity quality for OPT/Llama-class models after finetuning. So et al., *Primer* (NeurIPS 2021, arXiv:2109.08668) found squared ReLU by architecture search and reported training-compute savings — a benchmark number on a specific search setup, not a scaling-law claim.

**Theory SOTA.** Signal-propagation theory (Poole et al. 2016; Schoenholz et al. 2017) predicts trainability at initialization from activation-dependent order/chaos boundaries. It says nothing about the converged-loss gap between two activations that are both trainable.

## 4. What Is Known

- ~0.05 nats log-perplexity for GEGLU/SwiGLU over ReLU-FFN, T5-Base ~220M params, parameter-matched, single seed (Shazeer 2020). No confidence interval published.
- The gap replicates in a second codebase at T5-Base/Large scale (Narang et al. 2021).
- Parameter-matching to $\tfrac{8}{3}d$ is required; without it the gated arm simply has ~50% more FFN parameters and the comparison is void.
- ReLU FFN activations are ~90% zero at inference in trained LLMs; GELU/SwiGLU are not sparse but their pre-activations are similarly concentrated (Mirzadeh et al. 2024).
- Chinchilla-style laws (Hoffmann et al., NeurIPS 2022) fit $L(N,D)$ with $\alpha \approx 0.34$, $\beta \approx 0.28$ over $70$M–$16$B params. These were fit on a *single* architecture; no published fit isolates $\alpha_\sigma$ per activation.
- Loss-spike behavior is architecture-sensitive and small-proxy-predictable for some knobs (Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, 2023, arXiv:2309.14322) — but activation family was not one of the knobs swept there.

## 5. What Is Not Known

- **Empirically open.** Whether $\Delta\alpha \neq 0$ between ReLU, GELU, SwiGLU and squared ReLU. The experiment is fully runnable — a 4-arm IsoFLOP sweep from $10^{18}$ to $10^{22}$ FLOPs — and has not been published. This is the core gap.
- **Empirically open.** Whether the ~0.05-nat gap persists at $10^{23}$ FLOPs and 15+ tokens/param, or is absorbed once training is heavily data-dominated.
- **Methodologically blocked.** Whether activation choice affects *downstream capability* independent of loss. At $\Delta L \approx 0.02$ nats, no existing benchmark suite has the resolution to separate arms; the measurement is not defined at the effect size in question.
- **Theoretically open.** No proof or principled derivation of why multiplicative gating beats a pointwise nonlinearity at matched parameters. Shazeer's own paper concedes this.
- **Empirically open.** Whether µP LR transfer holds across activation families, which determines whether any cross-activation comparison is fair at all.

## 6. Why It Is Hard

Two obstructions, both specific.

**The effect size is below the noise floor of the affordable experiment.** The claimed gap is ~0.05 nats at 220M and plausibly ~0.01–0.02 nats at 10B. Seed-to-seed validation-loss variance at 1B scale is of the same order. Separating a 0.01-nat coefficient effect from a 0.005-per-decade exponent effect needs multiple seeds × multiple scales × 4 arms — the cost is the number of *cells*, not the size of the largest run.

**The comparison is confounded by hyperparameter optimality.** Activation choice moves the optimal learning rate, warmup, and init scale. Any observed gap decomposes into (architecture) + (how well each arm's HP was tuned), and the two are non-identifiable without a per-arm HP sweep — which multiplies the cost again. Published ablations almost never report a per-arm LR sweep, so the sign of the reported gap is not safely attributable.

Secondary: the winner under *loss per FLOP* and the winner under *loss per second served* differ (ReLU's sparsity is an inference-time win invisible to pretraining loss), so the evaluation does not measure the quantity practitioners optimize.

## 7. Current Research (as of 2026)

- **Sparsity-driven revival of ReLU.** Apple's *ReLU Strikes Back* line and follow-on sparse-inference work; motivated by decode cost, not pretraining loss.
- **Scaling-law-native architecture comparison.** Everett et al., *Scaling Exponents Across Parameterizations and Optimizers* (ICML 2024, arXiv:2407.05872) established the methodology — fit exponents per configuration rather than compare at one scale. Applying it to activations is the obvious open slot.
- **Learnable / parameterized activations.** xIELU and related trainable-slope activations reported in 2025 open-model work claim small pretraining wins at multi-B scale *(frontier — verify; treat published deltas as unablated)*.
- **Gate-free simplification.** Ongoing work asking whether normalization and gating are jointly redundant *(frontier — verify)*.
- Groups with the compute to settle this — Google DeepMind, Meta AI, Alibaba Qwen, Allen AI (OLMo) — publish architecture choices but not the controlled sweep.

## 8. Concrete Next Experiment

**Four-arm IsoFLOP scaling-law fit.**

- **Arms:** (1) dense ReLU, $d_\text{ff}=4d$ — *control*; (2) dense GELU, $4d$; (3) SwiGLU, $\tfrac{8}{3}d$; (4) squared ReLU, $4d$. All parameter-matched to within 1%.
- **Scale:** 6 model sizes, 70M → 3B non-embedding params; 4 IsoFLOP slices per size spanning $3\times10^{18}$ – $3\times10^{21}$ FLOPs; 3 seeds at the two smallest sizes to estimate noise. One fixed data mixture, one tokenizer, no data repetition.
- **Fairness control:** µP with a per-arm base-LR sweep (5 points) at the smallest width only, then transferred. Report the per-arm sweep curves.
- **Deciding number:** $\Delta\alpha = \alpha_{\text{SwiGLU}} - \alpha_{\text{ReLU}}$ with a bootstrap 95% CI. If the CI contains 0 and $|\Delta\alpha| < 0.005$, the gap is a pure coefficient effect — SwiGLU is worth a fixed ~1.2–1.4× compute multiplier and nothing more, and the field should optimize activation choice for inference cost. If the CI excludes 0 with $\Delta\alpha > 0.01$, the choice compounds and every large model without it is leaving exponent on the table.
- **Cost:** ~$3\times10^{22}$ FLOPs total — a few thousand H100-days. Within a single well-funded lab's quarterly budget; outside any academic group's.

## 9. Key References

- **[Foundational]** Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. — arXiv:1706.03762
- **[Foundational]** Hendrycks & Gimpel. *Gaussian Error Linear Units (GELUs).* 2016. — arXiv:1606.08415
- **[Foundational]** Dauphin, Fan, Auli, Grangier. *Language Modeling with Gated Convolutional Networks.* ICML 2017. — arXiv:1612.08083
- **[SOTA]** Shazeer. *GLU Variants Improve Transformer.* 2020. — arXiv:2002.05202
- **[SOTA]** Narang et al. *Do Transformer Modifications Transfer Across Implementations and Applications?* EMNLP 2021. — arXiv:2102.11972
- **[SOTA]** Mirzadeh et al. *ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models.* ICLR 2024. — arXiv:2310.04564
- **[SOTA]** So et al. *Primer: Searching for Efficient Transformers for Language Modeling.* NeurIPS 2021. — arXiv:2109.08668
- **[Method]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Method]** Yang & Hu et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* NeurIPS 2021. — arXiv:2203.03466
- **[Method]** Everett et al. *Scaling Exponents Across Parameterizations and Optimizers.* ICML 2024. — arXiv:2407.05872
- **[Context]** Ramachandran, Zoph, Le. *Searching for Activation Functions.* 2017. — arXiv:1710.05941
- **[Context]** Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities.* 2023. — arXiv:2309.14322

## 10. Worked Example

Take the Shazeer number at face value and extrapolate it two ways.

Observed: $\Delta L = 0.05$ nats at $N = 2.2\times10^8$. Suppose ReLU follows $L = 1.69 + 406/N^{0.34}$ (Chinchilla-form, data-unbounded).

**Hypothesis A — coefficient only.** SwiGLU has the same $\alpha = 0.34$, smaller $A$. Solve for the compute multiplier: a 0.05-nat drop at $N=2.2\times10^8$ requires $\Delta \ln N = 0.05 / (0.34 \cdot A N^{-0.34})$. With $A N^{-0.34} \approx 0.63$ nats at that scale, $\Delta \ln N \approx 0.23$, i.e. a 1.26× parameter (≈1.26× compute) equivalent. That multiplier is *scale-invariant*: at $10^{12}$ params SwiGLU is still worth 1.26×, and the nat gap has shrunk to $0.05 \times (10^{12}/2.2{\times}10^8)^{-0.34} \approx 0.0035$ nats.

**Hypothesis B — exponent shift.** $\alpha_{\text{SwiGLU}} = 0.35$, $\alpha_{\text{ReLU}} = 0.34$, coefficients set to agree at $2.2\times10^8$. At $10^{12}$ params the gap is $\approx 0.05 \times [(N/N_0)^{-0.34} - (N/N_0)^{-0.35}] / (\cdot)$ — expanding, the gap *grows* to roughly $0.09$ nats, a ~2.5× compute multiplier.

**The obstruction, made visible.** At $N = 2.2\times10^8$ the two hypotheses predict the *same* 0.05 nats. At $N = 10^{9}$ they predict 0.031 vs 0.034 nats — a 0.003-nat separation. Single-seed validation loss at 1B scale varies by roughly ±0.005 nats across seeds and data-order shuffles. The discriminating signal is smaller than the noise until $N$ passes ~$10^{10}$, and at that point one cell of the experiment costs more than most groups' annual compute. That is why the field's default activation rests on one 2020 ablation at 220M parameters, and why the question is empirically open rather than merely unanswered.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*