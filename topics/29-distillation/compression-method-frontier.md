---
id: 29-distillation/compression-method-frontier
title: "Distillation versus Pruning versus Quantization Frontier"
topic: 29-distillation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Distillation versus Pruning versus Quantization Frontier

> **Topic:** Distillation & Transfer · **ID:** `29-distillation/compression-method-frontier` · **Status:** empirically-open

## 1. Problem Statement

Given a trained model and a fixed serving budget, three families of compression compete: **distillation** (train a smaller student on the teacher's outputs), **pruning** (remove weights or structures), and **quantization** (reduce numeric precision). Each is reported to be near-lossless in its own papers, on its own budget axis, against its own baseline. The problem is to determine the **Pareto frontier over the joint space** — which method, or which composition of methods, minimizes loss at a given inference cost, and how that answer moves with model scale, token budget, and the metric used.

Three variants, of very different difficulty:

- **Measurement.** Define one cost axis that all three methods can be plotted on. Bits-per-parameter, parameter count, and FLOPs are not interchangeable: a 2:4-sparse model and an INT4 model both claim "4×" and neither means the same thing on an H100.
- **Method.** Find the compositional policy $\pi$ mapping (budget, architecture, data) to a compression recipe that dominates the best single-method recipe at every budget.
- **Theory.** Prove that one family is asymptotically preferred in some regime — e.g. that quantization dominates pruning below a parameter-count threshold because it preserves the dimension of the representation while pruning does not.

A solution to the measurement variant is a published table where all three arms share one hardware-realized cost metric, one training-token accounting, and one evaluation suite that includes tail behavior. No such table exists at frontier scale.

## 2. Formal Setting

Teacher $f_\theta$, $\theta \in \mathbb{R}^N$, trained on $D$ tokens. A compression operator $C$ produces $\tilde{f} = C(f_\theta, B)$ under budget $B$.

**Cost, as measured.** Not parameter count. Define

$$B = \big(M,\; T_{\text{tok}},\; C_{\text{prep}}\big)$$

where $M$ is peak resident bytes of weights + KV cache at batch $b$ and context $L$ (measured by `nvidia-smi` / allocator high-water mark, not computed from a formula); $T_{\text{tok}}$ is measured wall-clock tokens/s at fixed $(b, L)$ on a named accelerator; $C_{\text{prep}}$ is the one-off FLOP cost of producing $\tilde f$ (calibration forward passes, retraining steps, teacher inference for distillation logits). $C_{\text{prep}}$ is the term routinely omitted: post-training quantization costs $\sim10^{-4}$ of pretraining, while distillation of a student on $D_s$ tokens costs at least $D_s \cdot (2N_s + 2N_t)$ FLOPs including teacher forward passes.

**Quality, as measured.** $\mathcal{L}(\tilde f) = \mathbb{E}_{x\sim P}[-\log \tilde f(x)]$ on a held-out corpus, plus a task vector $\mathbf{a}(\tilde f) \in [0,1]^K$. Report both: perplexity and mean accuracy are known to hide the failures that compression actually causes.

**The frontier.** $\tilde f$ is Pareto-optimal if no $\tilde f'$ has $\mathcal{L}(\tilde f') \le \mathcal{L}(\tilde f)$, $M' \le M$, $T'_{\text{tok}} \ge T_{\text{tok}}$ with one strict. The open object is

$$\mathcal{F}(M, T) \;=\; \min_{C \in \{\text{distill},\,\text{prune},\,\text{quant}\}^*} \mathcal{L}\big(C(f_\theta)\big) \quad \text{s.t. } M(\tilde f) \le M,\; T_{\text{tok}}(\tilde f) \ge T.$$

The Kleene star matters: compositions (prune → distill → quantize) are in the search space and are what production systems actually ship.

**Assumptions known to be violated.**
1. *Cost is monotone in bit-width.* False. INT3 and 2:4 sparsity often run slower than FP16 without a fused kernel; unstructured sparsity gives no speedup at all on dense tensor cores.
2. *Held-out perplexity is a sufficient statistic for quality.* False — Hooker et al. (2019) showed compression error concentrates on a small, identifiable subset of inputs while aggregate accuracy moves ~1%.
3. *The teacher's data is available.* Usually false; distillation arms are run on proxy corpora, which confounds every distillation-vs-pruning comparison ever published.
4. *Methods compose additively.* Claimed, not established: quantizing a pruned-and-distilled model has interaction terms nobody has fit.

## 3. State of the Art

**Established, with ablations.**
- *Quantization.* GPTQ (Frantar et al., ICLR 2023) and AWQ (Lin et al., MLSys 2024) give 4-bit post-training weight quantization of 7B–175B models with small perplexity cost and $C_{\text{prep}}$ measured in GPU-hours. Dettmers & Zettlemoyer (ICML 2023) established the *k*-bit inference scaling law: at fixed total bits, 4-bit is near-optimal across 19M–176M–66B parameter sweeps.
- *Pruning.* SparseGPT (Frantar & Alistarh, ICML 2023) and Wanda (Sun et al., ICLR 2024) reach 50% unstructured sparsity one-shot on LLaMA-scale models. Frantar et al. (ICLR 2024) fit joint scaling laws for sparsity, parameters and data on ViT/T5.
- *Structured pruning + distillation.* Minitron (Muralidharan et al., NeurIPS 2024) derived an 8B and 4B model from Nemotron-15B using ~$40\times$ fewer training tokens than from-scratch training, with the distillation loss ablated against plain fine-tuning.
- *Distillation.* Busbridge et al. (2025) fit distillation scaling laws over teacher/student/token grids, and found distillation beats supervised pretraining only when the student token budget is below a computable threshold — above it, from-scratch training wins.

**Claimed but unablated.** That "prune then distill then quantize" is jointly optimal — shipped in Gemma-2 and Llama-3.2 style pipelines, never compared against a matched-$C_{\text{prep}}$ single-method arm. Extreme-quantization results (BitNet b1.58, Ma et al. 2024) exist mainly as benchmark tables against a differently-trained FP16 baseline.

**Benchmark-number-only.** Most head-to-head "quantization beats pruning" claims. Jaiswal et al. (ICLR 2024) showed those rankings flip when evaluation moves from perplexity to harder generative tasks.

## 4. What Is Known

- 4-bit weight quantization is close to free: Dettmers & Zettlemoyer report 4-bit as the bit-width maximizing zero-shot accuracy per total model bit across 35,000+ experiments, 19M–66B.
- 50% one-shot unstructured sparsity on OPT-175B costs <0.1 perplexity via SparseGPT (4.5 hours on one A100). Below ~60–70% sparsity, degradation is steep.
- Structured pruning costs far more than unstructured at equal sparsity: LLM-Pruner (Ma et al., NeurIPS 2023) at 20% structured sparsity on LLaMA-7B loses several points of zero-shot accuracy without recovery training.
- Sheared-LLaMA (Xia et al., ICLR 2024) produced 1.3B/2.7B models from LLaMA2-7B using 50B tokens, outperforming equally-sized open models trained on far more tokens — evidence that pruning+continued-training beats from-scratch at small token budgets.
- Distillation has a crossover: below the Busbridge et al. threshold, a distilled student beats a from-scratch student at matched compute; above it, the ordering reverses.
- Compression error is non-uniform. Hooker et al. found pruned ImageNet models with ~1% top-1 loss misclassify a "CIE" (compression-identified exemplar) subset at rates far above baseline, concentrated on rare attributes.

## 5. What Is Not Known

- **Empirically open.** The three-arm comparison at matched $M$, matched $T_{\text{tok}}$, and matched $C_{\text{prep}}$, at $\ge$7B, has not been run. Every ingredient exists; the experiment costs perhaps $10^4$–$10^5$ GPU-hours and no group has an incentive to publish a null.
- **Empirically open.** Whether composition is super- or sub-additive. No fitted interaction term $\gamma$ in $\mathcal{L} \approx \mathcal{L}_0 + \alpha_{\text{prune}} + \beta_{\text{quant}} + \gamma$.
- **Theoretically open.** No proof that any family dominates in any regime. The sparsity scaling law and the precision scaling law (Kumar et al., 2024) are separately fitted, never unified into one exponent.
- **Methodologically blocked.** "Capability retained" has no accepted definition. Perplexity, mean benchmark accuracy, and tail-subset accuracy give different orderings, so the frontier is a function of a metric choice nobody has standardized.

## 6. Why It Is Hard

The obstruction is **confounded measurement across incommensurable cost axes**, compounded by **absent tail ground truth**.

Concretely: pruning removes parameters, quantization removes bits per parameter, distillation removes both but spends training compute. Papers report the axis flattering to their method. A 2× memory reduction from INT8 and a 2× reduction from 50% sparsity are equal on paper and not equal on hardware — the sparse model needs a 2:4 pattern and a supported kernel to run faster, the INT8 model needs a fused dequant path. So the ranking is a property of the kernel stack, not of the method, and changes with each CUDA release.

Second, distillation's $C_{\text{prep}}$ is 3–5 orders of magnitude above post-training quantization's. Any comparison that ignores it makes distillation look free; any comparison that amortizes it over serving volume makes the answer depend on a deployment parameter (tokens served) that is outside the paper.

Third, the failure mode is tail-concentrated. Without a labeled tail set per model, you cannot detect the degradation that matters, so aggregate numbers systematically favor whichever method damages rare behavior most cheaply.

## 7. Current Research (as of 2026)

- **Unified scaling laws for compression.** Kumar et al.'s precision scaling laws and Frantar et al.'s sparsity laws are the two halves; work merging them into a single $(N, D, \text{sparsity}, \text{bits})$ law is active *(frontier — verify)*.
- **Prune-then-distill pipelines at production scale.** NVIDIA (Minitron/Nemotron), Google DeepMind (Gemma distillation), Meta (Llama quantized releases). Ablations are partial; recipes are shipped as artifacts.
- **Quantization-aware training and sub-4-bit regimes.** Microsoft Research (BitNet line), ISTA (Alistarh group) on sparse+quantized composition.
- **Inference-aware training-budget theory.** Sardana et al. (ICML 2024) reframed Chinchilla to account for inference cost — the natural home for a compression frontier, not yet extended to it.
- **Compression auditing.** Extension of Hooker's CIE methodology to LLM generation *(frontier — verify)*; no standard tail benchmark yet.

## 8. Concrete Next Experiment

**Scale.** One teacher: an open 8B dense model with published training data (e.g. an OLMo/Pythia-class model, so the distillation arm uses the *teacher's own* corpus and the data confound is removed).

**Arms**, all targeting the same measured $M = 4$ GB weights and reported at their own measured $T_{\text{tok}}$ on one A100 at $b{=}32$, $L{=}2048$:
1. INT4 GPTQ/AWQ of the 8B model.
2. 2:4 structured sparsity + INT8, SparseGPT-style, with recovery training.
3. Structured prune to 4B + distillation on $D_s$ tokens (Minitron recipe).
4. **Control arm:** a 4B model trained from scratch on the same corpus for the same total FLOPs as arm 3's $C_{\text{prep}}$.

Sweep $D_s \in \{5, 20, 80\}$B tokens so arm 3 crosses the distillation threshold.

**The deciding number.** $\Delta = \mathcal{L}_{\text{best-single-method}} - \mathcal{L}_{\text{best-composition}}$ in nats/token on held-out data, at matched $M$, matched $T_{\text{tok}}$ within 5%, and matched $C_{\text{prep}}$ within 2×. If $\Delta \le 0.01$ nats across all $D_s$, composition buys nothing and the field's default pipeline is unjustified. If $\Delta \ge 0.05$ nats, the interaction term is real and worth fitting. Report the same $\Delta$ on a tail subset (bottom-decile-frequency entities) — the sign is allowed to differ, and if it does, that is the paper's main result.

## 9. Key References

- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Song Han, Jeff Pool, John Tran, William Dally. *Learning both Weights and Connections for Efficient Neural Networks.* NeurIPS, 2015. — arXiv:1506.02626
- **[SOTA]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[SOTA]** Elias Frantar, Dan Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML, 2023. — arXiv:2301.00774
- **[SOTA]** Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter. *A Simple and Effective Pruning Approach for Large Language Models.* ICLR, 2024. — arXiv:2306.11695
- **[SOTA]** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[SOTA]** Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, et al. *Compact Language Models via Pruning and Knowledge Distillation.* NeurIPS, 2024. — arXiv:2407.14679
- **[SOTA]** Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, Danqi Chen. *Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning.* ICLR, 2024. — arXiv:2310.06694
- **[Scaling law]** Tim Dettmers, Luke Zettlemoyer. *The case for 4-bit precision: k-bit Inference Scaling Laws.* ICML, 2023. — arXiv:2212.09720
- **[Scaling law]** Elias Frantar, Carlos Riquelme, Neil Houlsby, Dan Alistarh, Utku Evci. *Scaling Laws for Sparsely-Connected Foundation Models.* ICLR, 2024. — arXiv:2309.08520
- **[Scaling law]** Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, et al. *Scaling Laws for Precision.* 2024. — arXiv:2411.04330
- **[Scaling law]** Dan Busbridge, Amitis Shidani, Floris Weers, et al. *Distillation Scaling Laws.* Apple, 2025. — arXiv:2502.08606
- **[Critique]** Sara Hooker, Aaron Courville, Gregory Clark, Yann Dauphin, Andrea Frome. *What Do Compressed Deep Neural Networks Forget?* 2019. — arXiv:1911.05248
- **[Critique]** Ajay Jaiswal, Zhe Gan, Xianzhi Du, et al. *Compressing LLMs: The Truth is Rarely Pure and Simple.* ICLR, 2024. — arXiv:2310.01382
- **[Context]** Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML, 2024. — arXiv:2401.00448

## 10. Worked Example

Target: serve an 8B FP16 model (16 GB weights) from 4 GB. Three routes claim it.

| Route | Weights | Measured speedup (A100, $b{=}32$) | $C_{\text{prep}}$ (FLOPs) |
|---|---|---|---|
| INT4 GPTQ | 4.0 GB | ~1.5–2× (memory-bound decode) | $\sim10^{18}$ (128 calibration seqs) |
| 50% unstructured + INT8 | 4.0 GB | ~1.0× — no dense-tensor-core speedup | $\sim10^{18}$ |
| Prune to 4B + distill, 20B tokens | 8.0 GB → INT8 4.0 GB | ~2× (half the FLOPs) | $20\text{e}9 \times 2(N_s{+}N_t) \approx 4.8\times10^{20}$ |

The distillation route costs **~500× more preparation compute** than the quantization route, and route 2 delivers the memory saving with no throughput gain at all — the 4 GB figure is real and the "2× faster" implied by it is not.

Now the confound. Suppose the published numbers read: INT4 at 6.20 perplexity, prune+distill at 6.05, FP16 teacher at 5.95. The prune+distill arm looks better by 0.15. But (a) it burned $4.8\times10^{20}$ FLOPs, which at 8B-scale is enough to train a 4B model on ~10B tokens from scratch — the control arm nobody ran; (b) its 20B distillation tokens came from a proxy corpus, so part of the 0.15 is domain adaptation to the eval set, not compression quality; (c) on a rare-entity tail subset the ordering may invert, because distillation on 20B tokens cannot re-teach facts the teacher saw once in 15T.

Three numbers — 0.15 nats, 500× compute, and an unmeasured tail delta — and the reported comparison controls for none of them. That is the obstruction: the frontier is currently defined by which axis the author chose to plot.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*