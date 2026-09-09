---
id: 13-parameter-efficient-adaptation/merging-adapters-into-quantized-weights
title: "Merging Adapters Into Quantized Base Weights"
topic: 13-parameter-efficient-adaptation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Merging Adapters Into Quantized Base Weights

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/merging-adapters-into-quantized-weights` · **Status:** partially-solved

## 1. Problem Statement

A LoRA adapter trained on a quantized base model cannot be folded back into that base model without leaving the quantization grid. QLoRA trains $BA$ in $\mathrm{bf16}$ against a frozen 4-bit base $Q(W)$; the effective forward weight is $Q(W) + BA$, which is not a 4-bit tensor. Deployment then faces three bad options: keep the adapter separate (extra kernel, extra latency, no fusion), dequantize-merge-requantize (accuracy drops, sometimes below the unadapted model), or retrain.

- **Method variant.** Find $\widehat{W} \in \mathcal{Q}$ (the set of tensors representable in the target quantization format) minimizing the deployed model's loss gap to $Q(W)+BA$, at inference cost equal to the unadapted quantized model.
- **Measurement variant.** Decide what "merge succeeded" means. Weight-space error $\|\widehat{W} - (Q(W)+BA)\|$ is cheap and wrong; task loss is right and expensive; the two disagree in known cases.
- **Theory variant.** Characterize when a rank-$r$ update is exactly absorbable into a group-wise affine quantizer's scales and zero-points, and bound the loss when it is not.

Solved means: a merge procedure that, for $r \le 64$ adapters on 4-bit models, reaches task accuracy within noise of the unmerged $Q(W)+BA$ model with zero inference overhead — and does so without access to the fine-tuning data.

## 2. Formal Setting

Let $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ be one linear layer. Group-wise affine quantization partitions each row into groups $g$ of size $G$ (typically $G \in \{32, 64, 128\}$) and stores integers $\bar{w}_{ij} \in \{0,\dots,2^b-1\}$ with per-group scale $s_g$ and zero-point $z_g$:

$$Q(W)_{ij} = s_g \left( \bar{w}_{ij} - z_g \right), \qquad s_g = \frac{\max_g W - \min_g W}{2^b - 1}.$$

**Measured quantities.**
- *Bits per parameter*: $b + (\text{bits}(s_g) + \text{bits}(z_g))/G$. NF4 with $G=64$ and fp16 scales is $4 + 32/64 = 4.5$; with double quantization (QLoRA) it is measured at $\approx 4.127$.
- *Adapter update*: $\Delta = \frac{\alpha}{r} BA$, $B \in \mathbb{R}^{d_\text{out}\times r}$, $A \in \mathbb{R}^{r \times d_\text{in}}$, $\mathrm{rank}(\Delta) \le r$.
- *Merge residual*: $R = \widehat{W} - (Q(W) + \Delta)$, reported as relative Frobenius error $\|R\|_F / \|Q(W)+\Delta\|_F$.
- *Deployment gap*: $\mathcal{G} = \mathcal{L}_{\text{task}}(\widehat{W}) - \mathcal{L}_{\text{task}}(Q(W)+\Delta)$, evaluated on held-out task data, not on the calibration set.
- *Overhead*: added tokens/s cost and added bytes at batch size 1 and at batch size 64 (multi-tenant serving changes the answer).

**Assumptions, and which fail.**
1. *Layer-wise decomposability* — that minimizing $\|R\|_F$ per layer minimizes $\mathcal{G}$. Violated: GPTQ-family results show activation-weighted objectives $\|(\widehat{W}-W)X\|_F^2$ beat plain $\|R\|_F$, and neither tracks end-task loss monotonically.
2. *Group statistics are stable under merging* — that $\max_g$ and $\min_g$ barely move. Violated where $\Delta$ has outlier rows; a single entry can inflate $s_g$ and coarsen all $G$ weights in the group.
3. *Calibration data availability* — merge-time requantization (GPTQ, AWQ) needs calibration text. In adapter distribution (a hub adapter merged by a third party) the fine-tuning data is often absent.
4. *Adapter smallness* — that $\|\Delta\| \ll$ quantization step, so merging is a perturbation. Violated in both directions: often $\|\Delta_{ij}\|$ is *below* half a step (the update is rounded away entirely), and occasionally far above it.

## 3. State of the Art

**Established (ablated, reproduced).**
- **QLoRA** (Dettmers et al., NeurIPS 2023) established the setting and stated the constraint explicitly: the NF4 base plus bf16 adapter is not itself 4-bit. Matching 16-bit full-finetuning quality on Guanaco/Vicuna benchmarks is established *for the unmerged* model.
- **QA-LoRA** (Xu et al., ICLR 2024) is the strongest exact-merge result. It constrains $A$ so that each quantization group of columns sees a *constant* adapter contribution; the rank-$r$ update then folds exactly into the group zero-points $z_g$, giving a genuinely 4-bit merged model with no residual. Ablated against QLoRA + post-hoc GPTQ on LLaMA 7B–65B.
- **LoftQ** (Li et al., ICLR 2024) and **LQ-LoRA** (Guo et al., ICLR 2024) attack the mirror-image problem: initialize $Q(W)$ and $BA$ jointly so $Q(W)+BA \approx W$ at step 0. Established gains at 2–3 bits, where naive QLoRA initialization diverges.

**Claimed but unablated.**
- The common practitioner recipe — dequantize to fp16, merge, re-run GPTQ/AWQ — has no controlled study isolating requantization error from adapter loss. Reported degradations circulate as issue threads and blog benchmarks, not ablations.
- "Merge quality scales with rank" is folklore; no published sweep of $r$ against $\mathcal{G}$ at fixed bit-width.

**Benchmark-number-only.** Most merged-quantized results are single MMLU/GSM8K deltas on one 7B model with one adapter. They do not separate adapter-capacity loss from grid loss, and none report variance across adapter seeds.

## 4. What Is Known

- QLoRA at 4-bit NF4, LLaMA-65B, reaches 99.3% of ChatGPT's Vicuna benchmark score (NeurIPS 2023) — unmerged.
- QA-LoRA reports LLaMA-7B 4-bit merged MMLU (5-shot) above QLoRA followed by post-training quantization, with the gap widening as bit-width falls to 3 and 2 bits (ICLR 2024). The mechanism is exact, not empirical: zero-point absorption has zero residual by construction.
- LoftQ recovers usable 2-bit LLaMA-2 fine-tuning where QLoRA's naive NF4 initialization fails to converge (ICLR 2024); the effect is largest at 2 bits and near-zero at 4 bits.
- GPTQ quantizes OPT-175B and BLOOM-176B to 3–4 bits in about 4 GPU-hours with small perplexity loss (ICLR 2023) — so requantizing after merge is cheap in compute; the cost is accuracy and calibration data, not time.
- AWQ (MLSys 2024) shows ~1% of weight channels, selected by activation magnitude, dominate quantization loss. Adapter updates concentrated on those channels are exactly the ones a uniform requantizer handles worst.
- Serving systems (S-LoRA, MLSys 2024; Punica, MLSys 2024) make *not merging* cheap at high batch: thousands of adapters served concurrently with batched gather-GEMM. Merging is worth most at batch size 1 and on edge devices.

## 5. What Is Not Known

- **Theoretically open.** No characterization of which rank-$r$ updates are exactly representable as perturbations of $(s_g, z_g)$ beyond the column-constant construction QA-LoRA uses. The expressivity price of that constraint — how much of the rank-$r$ function class is lost — has no bound either way.
- **Empirically open.** No controlled study across (bit-width $\in\{2,3,4,8\}$) × (rank $\in \{8,64\}$) × (3+ model families) × (5+ adapter seeds) reporting $\mathcal{G}$ with error bars. Every ingredient is a standard 8×A100 run; nobody has run the grid.
- **Empirically open.** Whether merged-quantized models degrade *disproportionately* on safety and refusal behaviour versus the capability benchmarks usually reported.
- **Methodologically blocked.** "Merge fidelity" has no agreed metric. Weight-space residual, activation-space residual, task delta, and KL to the unmerged model rank methods differently, and no paper reports more than one.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the target under a rank-blind objective**, compounded by a **step-size mismatch**. Two distinct effects are routinely confounded:

1. The adapter's per-entry magnitude is typically *below* half the quantization step, so round-to-nearest requantization deletes most of it while amplifying the surviving entries to a full step. The merge is not a small perturbation of $\Delta$ — it is a sparse, high-variance caricature of it.
2. Merging changes group extrema, so it perturbs *base* weights that the adapter never touched. Attributing post-merge loss to the adapter is therefore wrong by default; the control arm (requantize the base with the same calibration set, no adapter) is almost never run.

Secondary: exact-merge methods buy their exactness by constraining $A$, so they change the training problem, not just the deployment step — comparisons against QLoRA confound merge fidelity with training capacity.

## 7. Current Research (as of 2026)

- **Quantization-aware adapter design** — QA-LoRA-style structural constraints and their successors; the open question is how far the constraint can be relaxed while keeping exactness *(frontier — verify)*.
- **Rotation-based pipelines** — QuaRot (Ashkboos et al., NeurIPS 2024) and RoLoRA make weights outlier-free before quantization, which shrinks group ranges and should make merged updates survive rounding. Interaction with LoRA merging is the active thread.
- **Joint decomposition at extreme bits** — LQ-LoRA, ApiQ (Liao & Monz, EMNLP 2024), IR-QLoRA (Qin et al., ICML 2024) at 2–3 bits.
- **Merge-free serving** — Punica/S-LoRA lineage, arguing merging is the wrong goal above batch ~16; groups at UW, Berkeley, and MIT-HAN.
- **Adapter-aware requantization** — using the adapter itself to weight the requantization objective (an AWQ-style saliency signal derived from $\Delta$) *(frontier — verify; no strong published result)*.

## 8. Concrete Next Experiment

**Question.** Does dequantize-merge-requantize lose accuracy because of the adapter, or because requantization perturbs the base?

**Scale.** Llama-3-8B and Qwen2.5-7B; 4-bit and 3-bit, group size 128; LoRA $r \in \{8, 64\}$, $\alpha = 2r$, all linear projections; 5 seeds per configuration; fine-tune on GSM8K train + a 20k-sample instruction mix. About 60 fine-tuning runs at ~2 A100-hours each plus requantization — under 200 GPU-hours.

**Arms.**
1. Unmerged $Q(W) + \Delta$ — the reference.
2. Dequantize, merge, requantize with GPTQ (128 calibration sequences of C4).
3. **Control:** dequantize, requantize with the *same* GPTQ call, **no adapter merged**, then attach $\Delta$ at inference. This isolates base-perturbation from adapter-deletion. This arm is the point of the experiment and is the one missing from the literature.
4. QA-LoRA exact merge, trained under its column-constant constraint.

**Deciding number.** $\mathcal{G}_2 - \mathcal{G}_3$: the GSM8K exact-match drop attributable to the adapter *after* subtracting the base requantization drop, averaged over seeds with 95% CIs. If $|\mathcal{G}_2 - \mathcal{G}_3| < 0.5$ points, merging is essentially free and the field's reported degradations are base requantization mislabeled — the practical recommendation becomes "merge freely, but requantize with good calibration." If it exceeds 2 points, adapter deletion is real and constrained-merge methods like QA-LoRA are the necessary path.

## 9. Key References

- **[Foundational]** Hu, E., Shen, Y., Wallis, P., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. — arXiv:2106.09685
- **[Foundational]** Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS, 2023. — arXiv:2305.14314
- **[SOTA]** Xu, Y., Xie, L., Gu, X., et al. *QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models.* ICLR, 2024. — arXiv:2309.14717
- **[SOTA]** Li, Y., Yu, Y., Liang, C., et al. *LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models.* ICLR, 2024. — arXiv:2310.08659
- **[SOTA]** Guo, H., Greengard, P., Xing, E., Kim, Y. *LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning.* ICLR, 2024. — arXiv:2311.12023
- **[Systems]** Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR, 2023. — arXiv:2210.17323
- **[Systems]** Lin, J., Tang, J., Tang, H., et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys, 2024. — arXiv:2306.00978
- **[Systems]** Sheng, Y., Cao, S., Li, D., et al. *S-LoRA: Serving Thousands of Concurrent LoRA Adapters.* MLSys, 2024. — arXiv:2311.03285
- **[Related]** Ashkboos, S., Mohtashami, A., Croci, M., et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.* NeurIPS, 2024. — arXiv:2404.00456
- **[Related]** Liao, B., Monz, C. *ApiQ: Finetuning of 2-Bit Quantized Large Language Model.* EMNLP, 2024.
- **[Survey]** Han, Z., Gao, C., Liu, J., et al. *Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey.* TMLR, 2024. — arXiv:2403.14608

## 10. Worked Example

One group of a 4-bit `q_proj` in a 7B model. Group size $G=128$, symmetric int4, and the group's weights span $[-0.05, 0.05]$ — typical for LLaMA-family attention projections.

Step size:

$$s_g = \frac{0.05 - (-0.05)}{2^4 - 1} = \frac{0.10}{15} = 6.67\times10^{-3}.$$

Round-to-nearest deletes any update with $|\Delta_{ij}| < s_g/2 = 3.33\times10^{-3}$.

Now the adapter. With $r=16$, $\alpha=32$, $A$ initialized $\mathcal{N}(0, 1/d_\text{in})$ and $B=0$, trained LoRA deltas on this layer have entrywise standard deviation of order $\sigma \approx 2\times10^{-3}$ — smaller than half a step. Modelling $\Delta_{ij} \sim \mathcal{N}(0,\sigma^2)$:

$$P\left(|\Delta_{ij}| < s_g/2\right) = \mathrm{erf}\!\left(\frac{3.33\times10^{-3}}{2\times10^{-3}\sqrt{2}}\right) = \mathrm{erf}(1.18) \approx 0.90.$$

**Nine of ten adapter entries in this group are rounded away.** The remaining 10% get moved by a full $6.67\times10^{-3}$ — over three times their intended magnitude. The merged tensor is not a slightly-degraded $\Delta$; it is a 10%-sparse version with $3.3\times$ the intended amplitude on the survivors. Its Frobenius norm is roughly $\sqrt{0.1}\cdot 3.3 \approx 1.05$ times $\|\Delta\|$ — *the weight-space residual metric looks fine*, while the direction is almost orthogonal to the trained update.

Then the second effect. Suppose one $\Delta_{ij}$ in the group equals $0.012$ and lands on a weight already at $0.05$. The group max becomes $0.062$, so $s_g$ rises to $7.5\times10^{-3}$, a 12% coarsening applied to all 128 weights — 127 of which the adapter never touched.

The obstruction is visible here: a norm-preserving, direction-destroying merge, plus collateral damage to untouched base weights. Any evaluation that reports only $\|R\|_F$ or only post-merge task accuracy without the no-adapter requantization control cannot tell these apart.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*