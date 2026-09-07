---
id: 09-model-design/activation-function-choice-at-scale
title: "Feedforward Activation Function Choice at Scale"
topic: 09-model-design
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Feedforward Activation Function Choice at Scale

> **Topic:** Architecture & Model Design · **ID:** `09-model-design/activation-function-choice-at-scale` · **Status:** empirically-open

## 1. Problem Statement

Every dense transformer contains a position-wise feedforward block (FFN) whose only nonlinearity is a scalar activation $\sigma$ — ReLU, GELU, SiLU/Swish, squared ReLU — optionally in gated form (GLU family: ReGLU, GEGLU, SwiGLU). SwiGLU became the de facto default after 2020 on the strength of one small-scale comparison. The problem is to determine whether that choice matters at frontier scale, and along which axis.

Three variants, with different difficulty:

- **Measurement.** Given a compute budget $C$ and a fixed everything-else recipe, is $\Delta L = L_{\sigma_1}(C) - L_{\sigma_2}(C)$ nonzero, and does $|\Delta L|$ grow, shrink, or hold constant as $C \to 10^{24}$ FLOPs? A solution is a scaling-law fit per activation with confidence intervals that separate the curves — or fail to.
- **Method.** Choose $\sigma$ to optimize a *joint* objective: loss per FLOP *and* inference cost (activation sparsity), *and* quantization/low-precision robustness (outlier magnitudes), *and* training stability. Different activations win different terms; no method selects on the joint objective today.
- **Theory.** Predict $\Delta L$ from properties of $\sigma$ (smoothness, gating, degree of homogeneity, dead-unit fraction) without training. No such theory exists; even the sign of $\Delta L$ is not derivable.

The problem is **empirically open**: the deciding experiment is a well-defined, runnable, expensive ablation that has not been published at scale with matched hyperparameters.

## 2. Formal Setting

An FFN layer maps $x \in \mathbb{R}^{d}$ to

$$\text{FFN}_\sigma(x) = W_2\,\sigma(W_1 x), \qquad W_1 \in \mathbb{R}^{d_{ff}\times d},\; W_2 \in \mathbb{R}^{d \times d_{ff}}$$

and its gated counterpart, with a third matrix $V$:

$$\text{FFN}_{\sigma\text{GLU}}(x) = W_2\big(\sigma(W_1 x) \odot V x\big).$$

**FLOP/parameter matching (measured, not assumed).** Non-gated FFN cost per token is $4\,d\,d_{ff}$ FLOPs (forward+backward $\approx 3\times$); gated is $6\,d\,d_{ff}$. Matching requires $d_{ff}^{\text{GLU}} = \tfrac{2}{3} d_{ff}$. Measure both arms as **params** ($\sum |W|$, embeddings reported separately) and **realized FLOPs** from a profiler, not the analytic formula — kernel-level cost of $\sigma$ itself (an $\exp$ in GELU/SiLU vs a compare in ReLU) is nonzero and is what a profiler catches.

**Objective.** Validation loss in nats/token on a held-out shard $\mathcal{D}_{val}$: $L = -\frac{1}{|\mathcal{D}_{val}|}\sum \log p_\theta(x_t \mid x_{<t})$. Fit per activation

$$L_\sigma(C) = L_\infty^\sigma + a_\sigma C^{-\alpha_\sigma},$$

and report the *joint* posterior over $(L_\infty, a, \alpha)$ from $\geq 5$ compute points and $\geq 3$ seeds. The decision predicate is whether the 95% intervals for $L_{\sigma_1}(C^\star)$ and $L_{\sigma_2}(C^\star)$ overlap at extrapolated $C^\star$.

**Secondary measured quantities.**
- Activation sparsity: $s = \Pr_{t,i}\big[\,|\sigma(W_1x_t)_i| < \epsilon\,\big]$, $\epsilon$ chosen so masking those units changes $L$ by $<10^{-3}$ nats. Report $\epsilon$; "sparsity" without a threshold is not a measurement.
- Outlier severity: $\kappa = \max_i \|h_i\|_\infty / \text{RMS}(h)$ over FFN inputs/outputs — the quantity that decides whether INT8/FP8 post-training quantization survives.
- Instability rate: fraction of seeds with a loss spike $>0.1$ nats, plus max stable learning rate $\eta^\star$.

**Assumptions known to be violated.** (i) *Hyperparameters transfer across activations* — false; $\eta^\star$ differs measurably between ReLU and GLU variants, so any single-$\eta$ comparison is confounded. (ii) *$d_{ff}^{\text{GLU}} = \tfrac23 d_{ff}$ makes arms equivalent* — it equalizes FLOPs but changes width, so it entangles activation with aspect ratio. (iii) *Rank ordering at $10^{20}$ FLOPs persists at $10^{24}$* — untested, and this is the whole question. (iv) *Loss ordering implies downstream ordering* — violated for sparsity-motivated activations, where perplexity is matched but MMLU-style scores move within noise.

## 3. State of the Art

**Empirical SOTA (established).** Shazeer, *GLU Variants Improve Transformer* (2020, arXiv only, no venue) — the field's load-bearing citation. T5-Base-scale (~220M) encoder-decoder, span-corruption pretraining, parameter- and FLOP-matched via the $2/3$ rule. GEGLU and SwiGLU improve pretraining log-perplexity over ReLU by roughly $0.05$ nats (order $1.997 \to 1.94$ as reported). The paper is explicit that it offers no explanation and reports essentially single-run numbers; downstream GLUE/SuperGLUE deltas are within fine-tuning variance. This is one scale, one objective, one seed regime — **claimed but unablated** as a scaling claim.

**Adoption, not evidence.** PaLM (Chowdhery et al., JMLR 2023) and LLaMA (Touvron et al., 2023) use SwiGLU citing Shazeer; Gemma, Qwen, Mistral follow. No frontier report includes an activation ablation at its own scale. Adoption is a benchmark-number cascade, not a replication.

**Counter-SOTA (established at mid scale).** Mirzadeh et al., *ReLU Strikes Back* (ICLR 2024): replacing GELU/SiLU with ReLU in OPT- and Llama-family models up to ~7B costs little or nothing in perplexity while producing large FFN activation sparsity, cutting FFN inference FLOPs by a factor of order 2–3. So et al., *Primer* (NeurIPS 2021) found squared ReLU as a searched improvement over ReLU in decoder-only LMs; Zhang et al., *ReLU²Wins* (2024) argues squared ReLU is the best sparsity/quality tradeoff for sparse LLM inference.

**Theory SOTA.** Essentially absent. Approximation theory (any non-polynomial $\sigma$ is universal), NTK/infinite-width analyses, and the Ramachandran et al. *Searching for Activation Functions* (2017) search that produced Swish all fail to predict a nats-per-FLOP gap. The strongest theory-adjacent result is negative: search-derived activations transfer inconsistently across architectures.

## 4. What Is Known

- **~$0.05$ nats** advantage for GEGLU/SwiGLU over ReLU at **220M params**, span-corruption objective, FLOP-matched (Shazeer 2020). Not reproduced at $\geq 10$B with matched $\eta^\star$ in public literature.
- **FFN activations are >90% near-zero per token** in trained transformers, sparsity increasing with depth and width, measured on T5 models up to 11B and on ViTs (Li et al., *The Lazy Neuron Phenomenon*, ICLR 2023). Sparsity is a property of trained FFNs, not only of ReLU — but ReLU makes it exactly exploitable.
- **ReLU-family activations reduce quantization outliers.** Bondarenko et al., *Quantizable Transformers* (NeurIPS 2023) traces INT8 failure to a few extreme-magnitude channels driven by attention/FFN interaction; activation choice is one of the levers.
- **Small-scale proxies for instability exist but are partial.** Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* (2023) shows learning-rate sensitivity reproduces at small scale — which means $\eta^\star$ per activation is measurable cheaply, and therefore that single-$\eta$ comparisons are avoidable and inexcusable.
- **Parameterization confounds scaling comparisons.** Everett et al., *Scaling Exponents Across Parameterizations and Optimizers* (ICML 2024) shows fitted exponents shift with parameterization; Yang & Hu's $\mu$P (ICML 2021/2022) is the standard mitigation.

## 5. What Is Not Known

- **Empirically open.** Whether $\Delta L(\sigma)$ persists, shrinks, or reverses from $10^{20}$ to $10^{24}$ FLOPs. Whether GLU's edge is the gating or the induced width/aspect-ratio change. Whether SwiGLU's advantage survives $\mu$P with per-arm tuned $\eta$ and $\geq 3$ seeds. All runnable; none run publicly.
- **Theoretically open.** No bound relating any property of $\sigma$ to $L_\infty$ or $\alpha$. Whether two activations can differ in $\alpha$ (not just $a$) is unproven either way — this is the difference between a constant offset and an eventually-unbounded gap.
- **Methodologically blocked.** The joint objective. There is no accepted scalarization of {nats/FLOP, inference sparsity, quantization headroom, stability}, so "best activation" is currently ill-posed: ReLU can win deployment cost while SwiGLU wins loss, with no defined exchange rate.

## 6. Why It Is Hard

Three named obstructions.

1. **Confounded measurement, not compute cost alone.** Changing $\sigma$ changes $\eta^\star$, initialization scale, and (under FLOP-matching) $d_{ff}$. Any published $\Delta L$ is a sum of an activation effect and a tuning effect, and no public comparison separates them at scale.
2. **The effect size is near the noise floor.** $0.05$ nats at 220M is comparable to seed-to-seed and data-order variance in many setups. Distinguishing $\alpha_{\sigma_1} \neq \alpha_{\sigma_2}$ requires seeds at several compute points — a multiplicative, not additive, cost.
3. **An evaluation that does not measure what it names.** "Better activation" is scored by pretraining loss, but the deployed quantity is quality per dollar of inference, where sparsity and quantizability dominate. ReLU's 2–3× FFN FLOP reduction at matched perplexity is invisible to the metric the field decides on.

## 7. Current Research (as of 2026)

- **Sparsity-exploiting activations for inference.** Apple (Mirzadeh et al.), and the ProSparse/ReLU² line from Tsinghua-affiliated groups, pushing ReLU-family activations to recover deployable sparsity in Llama-class models.
- **Learned/parametric activations** with trainable slope and gain, reported as small gains in early-training loss at sub-1B scale *(frontier — verify: the xIELU line and related parametric-ELU variants circulating 2025 lack independent replication)*.
- **Activation choice as a quantization/FP8 decision** — co-design with per-channel scaling and outlier suppression; active at NVIDIA, Qualcomm AI Research, Meta *(frontier — verify)*.
- **$\mu$P-clean architecture ablations.** Google DeepMind and EleutherAI-adjacent groups publishing scaling comparisons with per-arm tuned learning rates; activation is usually a side arm, not the headline.

## 8. Concrete Next Experiment

**Question.** Does the SwiGLU-over-ReLU gap have a nonzero scaling exponent difference, or is it a constant offset that a 2× data increase erases?

**Design.** Decoder-only LMs, $\mu$P, identical data order, 5 compute points: $\{3\times10^{18}, 10^{19}, 3\times10^{19}, 10^{20}, 3\times10^{20}\}$ FLOPs (~100M to ~3B params, Chinchilla-optimal tokens). Three arms: **(a) control — ReLU, $d_{ff}=4d$**; (b) SwiGLU, $d_{ff}=\tfrac83 d$ (FLOP-matched); (c) SwiGLU at $d_{ff}=4d$ (FLOP-*mismatched*, isolating gating from width). 3 seeds each; per-arm learning-rate sweep at the two smallest points, then $\mu$P transfer. Total ≈ $2\times10^{21}$ FLOPs — a few thousand H100-hours.

**Decision number.** The 95% credible interval on $\alpha_{\text{SwiGLU}} - \alpha_{\text{ReLU}}$ from the joint fit. If it excludes 0, activation choice is a scaling-law property and must be re-ablated at every new frontier scale. If it contains 0 while the interval on $\log(a_{\text{ReLU}}/a_{\text{SwiGLU}})$ excludes 0, the gap is a fixed compute multiplier — report it as "SwiGLU $\equiv$ ReLU trained on $r\times$ compute" and check whether $r$ exceeds ReLU's measured 2–3× inference saving. If both intervals contain 0, the 2020 result does not survive matched tuning.

**Secondary readouts on the same runs:** $s$ at $\epsilon$ calibrated to $10^{-3}$ nats, $\kappa$ for INT8 headroom, $\eta^\star$ per arm.

## 9. Key References

- **[Foundational]** Nair, Hinton. *Rectified Linear Units Improve Restricted Boltzmann Machines.* ICML, 2010.
- **[Foundational]** Hendrycks, Gimpel. *Gaussian Error Linear Units (GELUs).* 2016. — arXiv:1606.08415
- **[Foundational]** Ramachandran, Zoph, Le. *Searching for Activation Functions.* 2017. — arXiv:1710.05941
- **[Foundational]** Dauphin, Fan, Auli, Grangier. *Language Modeling with Gated Convolutional Networks.* ICML, 2017. — arXiv:1612.08083
- **[SOTA]** Shazeer. *GLU Variants Improve Transformer.* 2020. — arXiv:2002.05202
- **[SOTA]** So, Mańke, Liu, Dai, Shazeer, Le. *Primer: Searching for Efficient Transformer Language Models.* NeurIPS, 2021. — arXiv:2109.08668
- **[SOTA]** Mirzadeh, Alizadeh, Mehta, Del Mundo, Tuzel, Samei, Rastegari, Farajtabar. *ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models.* ICLR, 2024.
- **[SOTA]** Li, Chen, Zhang, Aggarwal, et al. *The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers.* ICLR, 2023.
- **[Method]** Yang, Hu et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* 2022. — arXiv:2203.03466
- **[Method]** Everett, Xiao, Wortsman, et al. *Scaling Exponents Across Parameterizations and Optimizers.* ICML, 2024.
- **[Context]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Context]** Touvron et al. *LLaMA: Open and Efficient Foundation Language Models.* 2023. — arXiv:2302.13971
- **[Context]** Bondarenko, Nagel, Blankevoort. *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.* NeurIPS, 2023.
- **[Survey]** Dubey, Singh, Chaudhuri. *Activation functions in deep learning: A comprehensive survey and benchmark.* Neurocomputing, 2022.

## 10. Worked Example

Take the 2020 result at face value and push it through a scaling law.

Observed: $\Delta L \approx 0.05$ nats at $N \approx 220$M, i.e. $C \approx 6ND \approx 2\times10^{19}$ FLOPs. Chinchilla-style fits give $L(C) \approx L_\infty + a C^{-\alpha}$ with $\alpha \approx 0.05$ for compute-optimal scaling. If SwiGLU is a pure *compute multiplier* $r$ — same $\alpha$, smaller $a$ — then

$$0.05 = a\,C^{-\alpha}\big(1 - r^{-\alpha}\big).$$

With $aC^{-\alpha} \approx 0.8$ nats of reducible loss at that scale, $1 - r^{-0.05} = 0.0625$, so $r = 0.0625$-solved as $r = (1-0.0625)^{-1/0.05} \approx e^{1.29} \approx 3.6$. Read plainly: the 2020 gap, if it is a constant multiplier, is worth ~3.6× training compute — enormous.

Now the obstruction. Suppose instead the gap is a constant *offset* in $L_\infty$ of $0.05$ nats: at $C = 3\times10^{23}$ FLOPs, reducible loss is smaller, the multiplier interpretation implies a gap of only $0.8 \cdot (3\times10^{23}/2\times10^{19})^{-0.05} \cdot 0.0625 \approx 0.026$ nats, while the offset interpretation keeps it at $0.05$. Both are consistent with the single published data point, and they differ by ~2× at frontier scale.

Against this, ReLU's measured inference saving is a factor of 2–3 in FFN FLOPs at matched perplexity. A 2–3× serving saving and a 0.026–0.05 nat loss penalty are not comparable on any published scale.

So the honest state is: one 220M-parameter, single-seed, single-objective number, with no error bars and no per-arm learning-rate tuning, is doing all the work in a decision replicated across every frontier model — and it cannot distinguish a 3.6× compute win from a vanishing offset that a cheaper activation more than repays at inference. That is why this entry is empirically open, and why the deciding statistic in §8 is an interval on $\alpha_{\sigma_1}-\alpha_{\sigma_2}$, not another loss table.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*