---
id: 15-mixture-of-experts/moe-low-precision-training-stability
title: "Stability of MoE Training in Low Precision"
topic: 15-mixture-of-experts
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Stability of MoE Training in Low Precision

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/moe-low-precision-training-stability` · **Status:** partially-solved

## 1. Problem Statement

Sparse Mixture-of-Experts (MoE) transformers diverge, spike, or silently lose quality more often than dense models of matched compute when trained in reduced precision (bf16, and now fp8 for GEMM inputs). The cause is structurally specific: MoE contains a **discrete** operation — $\mathrm{top}\text{-}k$ over router logits — inside the differentiable graph. Rounding error that would be a small perturbation in a dense model can flip a token's expert assignment, which is a discontinuous change in the function being optimized and in the load seen by every expert.

Three variants, of different difficulty:

- **Measurement.** Given a training run, decide whether an observed loss spike or quality gap is *caused* by precision, by routing collapse, or by ordinary optimization noise. Currently no accepted estimator separates these.
- **Method.** Find the minimal set of components that must stay in high precision (router logits, gate multiply, expert-gradient accumulation, second moments) such that fp8 training matches a bf16 control to within a stated loss tolerance, at fixed token budget.
- **Theory.** Prove a bound on the excess loss, or on divergence probability, induced by quantizing the router — as a function of the *logit-gap distribution* $\Delta$, the quantization step, and $k$. No such bound exists.

Solved means: a rule of the form "keep components $S$ in precision $p$" with a proof or a scaling-validated ablation showing final loss within $\varepsilon$ of the bf16 control across seeds, at $\geq 10^{12}$ tokens.

## 2. Formal Setting

Layer input token $x \in \mathbb{R}^d$, $E$ experts, router $W_r \in \mathbb{R}^{E \times d}$, logits $h = W_r x$, gates $p = \mathrm{softmax}(h)$, selected set $\mathcal{T}_k(x) = \operatorname{arg\,top}_k p$, output $y = \sum_{i \in \mathcal{T}_k} p_i \, E_i(x)$.

**Quantizer.** $Q_{\Delta}(v) = \Delta \cdot \mathrm{rn}(v/\Delta)$ with $\Delta$ the per-tensor/per-tile scale and $\mathrm{rn}$ round-to-nearest-even in the target format. For fp8 E4M3, representable magnitudes lie in $[2^{-9}, 448]$; E5M2 reaches $57344$. Measured as: the scale actually emitted by the delayed-scaling recipe, not an idealized one.

**Logit gap.** $\Delta_k(x) = h_{(k)}(x) - h_{(k+1)}(x)$, the margin between the last selected and first rejected expert. Measured empirically as a histogram over a held-out batch of $10^5$ tokens per layer.

**Router flip rate.** With $\tilde h$ the low-precision logits,
$$R_{\text{flip}} = \Pr_{x}\!\left[\, \mathcal{T}_k(\tilde h) \neq \mathcal{T}_k(h) \,\right] \;\lesssim\; \Pr\!\left[\Delta_k(x) < 2\epsilon_{\text{eff}}\right],$$
where $\epsilon_{\text{eff}}$ is the realized logit perturbation. Measured by running both precisions on the same batch from the same checkpoint and comparing index sets — this requires a shadow forward pass and is the only direct measurement in the list.

**Load imbalance.** With $c_i$ tokens routed to expert $i$ in a batch and $\bar c = \frac{1}{E}\sum_i c_i$, $\mathrm{MaxVio} = (\max_i c_i - \bar c)/\bar c$ (Wang et al., 2024). Under capacity factor $C$, tokens beyond $\lceil C\bar c\rceil$ are dropped; drop rate is measured directly.

**Spike detector.** $s_t = \mathbb{1}\big[\ell_t - \mathrm{med}(\ell_{t-w:t}) > \kappa\,\mathrm{MAD}(\ell_{t-w:t})\big]$, $w=100$, $\kappa=8$. Crude but reproducible; the field has no better convention.

**Precision-gap metric.** $\mathrm{RelErr} = |L_{\text{fp8}} - L_{\text{bf16}}| / L_{\text{bf16}}$ at matched tokens and seed.

**Assumptions, and which fail.** (i) Quantization error is zero-mean and independent of the input — *violated*: error is deterministic given the value, and activation outliers concentrate on specific channels and tokens. (ii) The perturbation to $y$ is $O(\Delta)$ — *violated* whenever a flip occurs; then $\|\Delta y\| \sim \|E_i(x) - E_j(x)\|$, an $O(1)$ jump. (iii) Expert function distributions are exchangeable — *violated*: experts specialize, so flips are not symmetric. (iv) The loss surface is locally smooth — *violated* at flip boundaries; the MoE objective is piecewise smooth with measure-zero discontinuities that a batch of $10^6$ tokens hits constantly.

## 3. State of the Art

**Established (ablated).**
- *Selective precision.* Switch Transformer (Fedus, Zoph, Shazeer, JMLR 2022) casts the router to fp32 locally while keeping bf16 elsewhere; their ablation shows the fp32-router variant recovers full-fp32 quality with bf16 speed, whereas full-bf16 is unstable. This is a real ablation, repeated by later work, and is now standard.
- *Router z-loss.* ST-MoE (Zoph et al., 2022) adds $L_z = \frac{1}{B}\sum_b (\log\sum_i e^{h_i^{(b)}})^2$ with coefficient $10^{-3}$, penalizing large logit magnitudes so that the softmax input stays in a well-conditioned range. Ablated to fix stability at 32B-parameter scale with no quality cost; adopted by OLMoE (Muennighoff et al., 2024) with an explicit ablation.

**Claimed but under-ablated.**
- *fp8 for MoE at scale.* DeepSeek-V3 (2024) trains a 671B-total / 37B-active MoE on 14.8T tokens with fp8 GEMMs, fine-grained $1\times128$ tile and $128\times128$ block scaling, and promotion out of tensor cores to fp32 accumulation every $N_C=128$ elements. It keeps the router, embeddings, normalization, and attention softmax in higher precision. The recipe works, but the report gives no component-wise ablation isolating *which* exclusion is load-bearing for the MoE specifically — the exclusions are inherited from dense fp8 practice.
- *Benchmark-only numbers.* DeepSeek-V3's "relative loss error below $0.25\%$ vs. BF16" was validated at roughly 16B and 230B scale on ~1T tokens, then extrapolated to the 671B run. It is a benchmark number for the recipe as a whole, not evidence about any single component.

**Theory SOTA.** Mixed-precision training theory (Micikevicius et al., ICLR 2018; FP8 Formats, 2022) covers dense linear layers, loss scaling, and accumulation width. There is no theory covering a quantized argmax inside the graph. The gap between theory and systems practice is total here.

## 4. What Is Known

- Routers in reduced precision destabilize training; fp32 router logits fix it. Measured at Switch-Base/Large scale (~1–7B params, T5 setup), JMLR 2022.
- Router z-loss at $\lambda_z = 10^{-3}$ eliminated divergences in ST-MoE runs up to 269B total parameters (ST-MoE-32B, 2022) with no measured quality penalty.
- fp8 tensor-core accumulation on H800 retains only about 14 bits of mantissa, well below fp32; DeepSeek-V3 measures this as the reason periodic fp32 promotion is required.
- fp8 divergence can be *late*: Fishman et al. (2024) find SwiGLU-driven outlier amplification produces divergence only after roughly 200B tokens in a 7B dense Llama-style model, invisible in short runs; Smooth-SwiGLU fixes it and they report ~34% throughput gain over bf16 on 2T tokens.
- Instabilities have small-scale proxies: Wortsman et al. (ICLR 2024) reproduce attention-logit growth and output-logit divergence at $\leq$ 1B parameters by raising learning rate, and show the interventions transfer. Dense-only; the MoE analogue is untested.
- Auxiliary-loss-free balancing via per-expert bias (Wang et al., 2024) reduces MaxVio relative to aux-loss baselines at 1B/3B scale without gradient interference — relevant because aux-loss gradients are themselves a small, precision-sensitive signal.

## 5. What Is Not Known

- **Theoretically open.** Any bound of the form $\mathbb{E}[L_{\text{quant}}] - \mathbb{E}[L] \le F(\Delta, k, \text{law of } \Delta_k)$. Also open: whether router flips act as beneficial exploration noise (a stochastic-routing regularizer) or as pure gradient corruption. Both stories fit current data.
- **Empirically open.** The component-wise fp8 ablation for MoE — router logits, gate multiply $p_i E_i(x)$, expert down-projection, all-to-all payload, optimizer second moment — each toggled independently at $\geq$ 100B tokens with 3 seeds. Runnable today on ~500 H100-days; nobody has published it. Also open: whether the ~200B-token late-divergence phenomenon is worse in MoE, where per-expert token counts are $E/k$ times smaller and outlier statistics per expert correspondingly noisier.
- **Methodologically blocked.** Attribution of a loss spike to precision. The shadow-bf16 forward pass needed to measure $R_{\text{flip}}$ costs a second copy of the model and is not instrumented in any public trainer, so no published MoE run reports $R_{\text{flip}}$ over time. Without it, "precision caused this spike" is not a measurable claim.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by a discontinuity**. In a dense model you can bound the effect of quantization by propagating a norm. In MoE the effect passes through $\mathrm{top}\text{-}k$, so the sensitivity is controlled by $\Pr[\Delta_k < \epsilon]$ — a quantity that changes during training as the router sharpens, and that nobody logs. Meanwhile every candidate cause is entangled: lowering precision also changes the all-to-all payload, hence the drop pattern under a fixed capacity factor, hence expert load, hence the aux-loss gradient, hence the router. A single-arm run cannot separate these, and the multi-arm run is expensive because the failure mode appears late (>100B tokens) and is seed-dependent — so the design needs $\geq 3$ seeds $\times \geq 5$ arms at a scale where each arm costs tens of thousands of GPU-hours. That is the actual barrier: the cheap experiment does not reproduce the phenomenon.

## 7. Current Research (as of 2026)

- Frontier-lab fp8 MoE recipes: fine-grained tile/block scaling plus selective high-precision islands (DeepSeek; NVIDIA Transformer Engine; Microsoft FP8-LM lineage). Convergent practice, divergent justifications.
- Outlier-suppression at the source — Smooth-SwiGLU, activation clipping, per-channel scaling (Intel Habana, Fishman et al.) — now being applied to expert FFNs *(frontier — verify)*.
- Aux-loss-free and bias-corrected balancing, which removes one precision-sensitive gradient path entirely.
- Microscaling formats (MX-FP8/FP6/FP4, OCP spec, 2023) with hardware block scaling on Blackwell; sub-fp8 MoE training claims are appearing but component ablations are not public *(frontier — verify)*.
- Small-proxy instability research (Wortsman-style) being extended to sparse models; no published MoE result yet *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Is the fp32 router the load-bearing exclusion in fp8 MoE training, or is it the gate multiply and expert-gradient accumulation?

**Scale.** 8B total / 1.3B active parameters, $E=64$, $k=8$, 32 MoE layers, 150B tokens, sequence 4096. Roughly 500–700 H100-days per arm at bf16 speed; five arms $\times$ 3 seeds.

**Control arm.** Full bf16, fp32 router, z-loss $10^{-3}$, capacity factor 1.25 — the known-stable configuration.

**Treatment arms.** (A) fp8 everywhere except router in fp32; (B) A, plus gate multiply in bf16; (C) A, plus fp32 expert-gradient accumulation; (D) fp8 router logits too. All arms share data order and init per seed.

**Instrumentation.** Every 500 steps, run a shadow bf16 forward on a fixed 100k-token probe batch and log $R_{\text{flip}}$ per layer, the $\Delta_k$ histogram, MaxVio, drop rate, and $s_t$.

**Deciding number.** $\mathrm{RelErr}$ at 150B tokens, averaged over 3 seeds, against a threshold of $0.25\%$ (DeepSeek's own claimed tolerance). If arm A clears $0.25\%$ and arm D fails it, the fp32 router is load-bearing and the folklore is confirmed. If A also fails and C clears it, the accumulator — not the router — is the real constraint, and the field's standard recipe is mis-specified. Secondary readout: the correlation between $R_{\text{flip}}$ and $\mathrm{RelErr}$ across arms, which is the first evidence on whether flips are noise or damage.

## 9. Key References

- **[Foundational]** N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, J. Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** P. Micikevicius et al. *Mixed Precision Training.* ICLR, 2018. — arXiv:1710.03740
- **[Foundational]** W. Fedus, B. Zoph, N. Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23(120), 2022. — arXiv:2101.03961
- **[SOTA]** B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, W. Fedus. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* 2022. — arXiv:2202.08906
- **[SOTA]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[SOTA]** M. Fishman, B. Chmiel, R. Banner, D. Soudry. *Scaling FP8 Training to Trillion-Token LLMs.* 2024. — arXiv:2409.12517
- **[SOTA]** L. Wang, H. Gao, C. Zhao, X. Sun, D. Dai. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* 2024. — arXiv:2408.15664
- **[Method]** P. Micikevicius et al. *FP8 Formats for Deep Learning.* 2022. — arXiv:2209.05433
- **[Method]** H. Peng et al. *FP8-LM: Training FP8 Large Language Models.* 2023. — arXiv:2310.18313
- **[Method]** M. Wortsman et al. *Small-scale Proxies for Large-scale Transformer Training Instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[Method]** N. Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models.* 2024. — arXiv:2409.02060
- **[Survey]** D. Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668

## 10. Worked Example

Take one MoE layer, $E=64$, $k=8$, $d=2048$, batch of $10^6$ tokens.

Suppose the router is computed in fp8 E4M3 with per-tensor scaling. Logits after a few thousand steps have $|h| \lesssim 8$; with the scale set to $\Delta_{\max}=8$, the E4M3 spacing near $h \approx 4$ is about $2^{-2}\cdot 2^{-3} = 0.031$, so $\epsilon_{\text{eff}} \approx 0.016$ per logit and the effective margin threshold is $\approx 0.03$.

Now the empirical part. In a trained router, the top-8 boundary gap $\Delta_8$ is small for a large fraction of tokens — the 8th and 9th experts are near-ties by construction. Take a plausible measured value: $\Pr[\Delta_8 < 0.03] = 0.02$. Then $R_{\text{flip}} \approx 2\%$ of tokens per layer. Across 32 MoE layers, the probability that a token's *entire* routing path is unchanged is $0.98^{32} \approx 0.52$. **Half the tokens in the batch take a different path through the network than the bf16 reference.**

Now the obstruction. Per-token loss change from one flip is $\|p_8(E_i(x) - E_j(x))\|$ — but $p_8 \approx p_9$ at the boundary, so the gate weight is *small*, and the expected loss change is second-order. So a 2% flip rate might be harmless. Or the flips shift $c_i$ enough to change which tokens get dropped at capacity 1.25, which changes the aux-loss gradient, which sharpens the router, which changes $\Pr[\Delta_8 < 0.03]$ at the next step — a feedback loop with no small parameter.

Both accounts predict the same observable at step 1000: a normal-looking loss curve. They diverge at 200B tokens. That is why the cheap experiment settles nothing, and why the question is still open despite fp8 MoE runs shipping in production.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*