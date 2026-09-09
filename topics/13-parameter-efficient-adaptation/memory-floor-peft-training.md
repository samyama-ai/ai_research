---
id: 13-parameter-efficient-adaptation/memory-floor-peft-training
title: "Memory Floor of Parameter-Efficient Training"
topic: 13-parameter-efficient-adaptation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memory Floor of Parameter-Efficient Training

> **Topic:** Parameter-Efficient Adaptation · **ID:** `13-parameter-efficient-adaptation/memory-floor-peft-training` · **Status:** partially-solved

## 1. Problem Statement

Parameter-efficient fine-tuning (PEFT) is sold as a memory result: LoRA trains 0.1% of the parameters, so training should cost roughly inference. It does not. Trainable-parameter count controls only the gradient and optimizer-state terms. The weight term and the **activation** term — the intermediate tensors the backward pass must read — are unchanged by making the update low-rank.

The problem: **characterize and reach the true lower bound on peak device memory for adapting a pretrained model to within $\epsilon$ of full fine-tuning quality, under a bounded compute overhead.**

Three variants, different difficulty:

- **Measurement.** Given a method, decompose measured peak memory into weight / gradient / optimizer / activation / allocator-fragmentation terms, reproducibly across kernels and frameworks. Currently done ad hoc; papers report peak GB on one stack and call it a method property.
- **Method.** Build an adapter family whose activation footprint is $o(L\,B\,T\,d)$ *without* the accuracy loss of gradient-free or backbone-detached training and without more than a constant-factor FLOP penalty.
- **Theory.** Prove a lower bound on memory for any first-order adaptation procedure achieving $\epsilon$-competitive loss, as a function of recomputation budget. No such bound exists for learned adapters; the only rigorous bounds are for exact reverse-mode AD scheduling.

Solving it means: a method plus a matching lower bound showing it is within a constant factor.

## 2. Formal Setting

Model: $L$ transformer blocks, hidden width $d$, frozen weights $W \in \mathbb{R}^{P}$, $P \approx 12Ld^2$. Batch $B$, sequence length $T$. Trainable adapter parameters $\theta \in \mathbb{R}^{p}$, $p \ll P$. Objective $\mathcal{L}(W \oplus \theta)$ minimized with a first-order optimizer.

Peak resident memory, in bytes, as a profiler would report it (`torch.cuda.max_memory_allocated` plus reserved-minus-allocated):

$$M \;=\; \underbrace{b_W P}_{\text{weights}} \;+\; \underbrace{b_g p}_{\text{gradients}} \;+\; \underbrace{b_o s\, p}_{\text{optimizer}} \;+\; \underbrace{M_A}_{\text{activations}} \;+\; \underbrace{M_{\text{frag}}}_{\text{allocator}}$$

with $b_W$ bytes/weight (2 for bf16, $\approx 0.52$ for NF4 with double quantization), $b_g = 2$, $s$ optimizer states per parameter ($s=2$ for Adam, $b_o=4$ if fp32), and $M_{\text{frag}}$ the difference between reserved and allocated bytes — measured, not derived, and routinely 10–25% of peak.

Activation term under uniform checkpointing with segment length $\ell$:

$$M_A \;=\; \underbrace{2\,\frac{L}{\ell}\,B\,T\,d}_{\text{stored boundaries}} \;+\; \underbrace{2\,c\,\ell\,B\,T\,d}_{\text{recompute window}}, \qquad \ell^\star = \sqrt{L/c}$$

where $c$ counts live intermediates per block (for LLaMA-style blocks with fused attention, $c \approx 18\text{–}22$ measured, not $O(T^2)$ since FlashAttention removed the score matrix). Compute overhead is $F = (1+\alpha)F_0$ with $\alpha \approx 1/3$ for one extra forward.

**Define the floor.** For tolerance $\epsilon$ and overhead cap $\kappa$:

$$M^\star(\epsilon,\kappa) \;=\; \min_{\mathcal{A}} \Big\{ \max_t M_t(\mathcal{A}) \;:\; \mathcal{L}(\mathcal{A}) - \mathcal{L}_{\text{full-FT}} \le \epsilon,\; F(\mathcal{A}) \le \kappa F_0 \Big\}$$

**Assumptions, and which are violated.**
- *Activations are stored densely at bf16.* Violated by ActNN/GACT-style 2–4-bit activation compression, which changes $M_A$ but injects gradient noise.
- *$\epsilon$ is measurable.* Violated: fine-tuning quality is an average over a task suite with run-to-run variance often exceeding the method gap.
- *$M_{\text{frag}}$ is method-independent.* Violated: paged optimizers and CPU offload shift the peak into host memory and change the allocator's behaviour, so two methods are not measured on the same axis.
- *Compute overhead is the only cost of recomputation.* Violated at scale: recompute changes the communication schedule under FSDP/ZeRO, so wall-clock does not track $\alpha$.

## 3. State of the Art

**Established (ablated, independently reproduced).**
- **Gradient checkpointing.** Chen et al. (2016) give $O(\sqrt{n})$ memory with one extra forward; Griewank & Walther's *revolve* (ACM TOMS 2000) is optimal for exact reverse-mode AD: $c$ checkpoints and $r$ recompute sweeps reverse $\binom{c+r}{c}$ steps. This is the only tight theory in the area, and it bounds *AD scheduling*, not *adaptation*.
- **NF4 + double quantization (QLoRA, Dettmers et al., NeurIPS 2023).** $b_W \approx 0.52$ bytes/param, 65B fine-tuned on one 48 GB GPU, with a matched-quality ablation against 16-bit LoRA.
- **8-bit block-wise optimizers (Dettmers et al., ICLR 2022).** $b_o$ from 4 to 1, no quality loss at up to 1.5B, ablated on LM, GLUE and ImageNet.

**Claimed but under-ablated.**
- **GaLore (Zhao et al., ICML 2024).** 82.5% optimizer-state reduction, 63.3% total, LLaMA-7B pretraining on a 24 GB RTX 4090. The memory number is solid; the claim that projected gradients match full-rank *fine-tuning* quality is a benchmark number on a narrow suite, not a controlled ablation.
- **Ladder Side-Tuning (Sung et al., NeurIPS 2022).** ~69% memory reduction vs full FT by removing backprop through the backbone — the only mainstream method that attacks $M_A$ directly. Shown at T5 scale; not established at 7B+.
- **MeZO (Malladi et al., NeurIPS 2023).** Zeroth-order, inference-level memory, 12× reduction, OPT-30B on one 80 GB A100. Quality parity holds for prompt-based classification; convergence is slow on generation, and the parity claim outside its suite is a benchmark number.

**Not established.** No published method reaches full-FT quality at 7B with $M_A$ below the $\ell^\star$-checkpointing curve at $\kappa \le 1.5$.

## 4. What Is Known

Numbers, with scale named:

- **LoRA does not reduce activation memory.** Backprop to adapter $B$ at layer $j$ still requires the full backward chain through frozen weights above $j$, hence the same stored activations. Confirmed by profiling across LoRA ranks: LLaMA-2 7B, $B{=}4$, $T{=}2048$, ranks 8→64 moves peak by <1 GB.
- **Activations dominate under quantized weights.** At 7B, $B{=}4$, $T{=}2048$ with layer-boundary checkpointing, $M_A \approx 3.5$ GB against 3.5 GB of NF4 weights and 0.2 GB of adapter state (§10).
- **Checkpointing buys ~20× on $M_A$ for ~33% FLOPs** at LLaMA-2 7B — a larger factor than any PEFT method delivers on the same term.
- **Activation compression works at 2 bits.** ActNN (Chen et al., ICML 2021) reports up to 12× activation-memory reduction and 6–14× larger batches on CNNs; ImageNet-scale, not LLM-scale.
- **Rank is not free on quality.** Biderman et al. (TMLR 2024) show LoRA underperforms full fine-tuning on code and math continued pretraining at 7B/13B while forgetting less — so $\epsilon$ in §2 is not zero for the cheapest configurations.
- **Adapters have an intrinsic-dimension justification** (Aghajanyan et al., ACL 2021): RoBERTa-large reaches 90% of full-FT MRPC performance in ~200 trainable dimensions. This argues $p$ can be tiny; it says nothing about $M_A$.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on $M^\star(\epsilon,\kappa)$ for adaptation. Revolve bounds exact gradient replay; adaptation does not need exact gradients. Whether an $\epsilon$-competitive update can be computed from $o(L B T d)$ retained bytes is unproven either way.
- **Empirically open.** Whether backbone-detached training (LST-style) or 2-bit activation compression reaches full-FT quality at 7B–70B. The runs are affordable (hundreds of GPU-hours); nobody has published them with a matched full-FT control at that scale.
- **Methodologically blocked.** Cross-paper memory comparison. Reported peak GB confounds framework, kernel set, attention implementation, allocator config, offload, and sequence length. There is no standard *memory profile* artifact, so the field cannot tell a method improvement from a stack improvement.

## 6. Why It Is Hard

**Confounded measurement, and an evaluation that does not measure what it names.** "Parameter-efficient" names $p$; the binding constraint is $M_A$, which is independent of $p$. A method can cut trainable parameters 1000× and peak memory 3%, and still be reported as a memory advance because the headline metric is the wrong term.

Second: **non-identifiability of the source of savings.** QLoRA's 48 GB result is NF4 + paged optimizers + checkpointing + LoRA. Removing LoRA and keeping the rest changes peak memory by a few percent at 7B. The four factors are almost never crossed, so the marginal contribution of the PEFT component is unmeasured.

Third: **absent ground truth for $\epsilon$.** Deciding whether a memory-cheaper method is $\epsilon$-competitive requires a full fine-tuning control at the same scale — which is exactly the run the method exists to avoid. Papers proposing memory reductions at 7B+ therefore usually omit the control arm.

## 7. Current Research (as of 2026)

- **Projection-based optimizer compression.** GaLore and its quantized successor Q-GaLore (Zhang, Zhao, Tian et al., 2024) — attacks $M_O$, leaves $M_A$. *(frontier — verify current variants.)*
- **Block-coordinate full-parameter updates.** BAdam (Luo et al., NeurIPS 2024) and LOMO/AdaLomo (Lv et al., ACL 2024) — fuse the update into the backward to eliminate $M_G$ and $M_O$ entirely; still full activation cost per block.
- **Backbone-detached and side-network adapters.** LST descendants; the only line targeting the actual floor. Underexplored at LLM scale.
- **Zeroth-order and sparse-ZO.** MeZO variants trading FLOPs and convergence for near-inference memory; active at Princeton and follow-ups. *(frontier — verify.)*
- **Compressed-activation training** (ActNN/GACT lineage, Berkeley) applied to LLM fine-tuning. *(frontier — verify; few LLM-scale results published.)*

## 8. Concrete Next Experiment

**The 2×2×2 memory-attribution grid.**

- **Scale.** LLaMA-2 7B, single 80 GB H100, $B{=}4$, $T{=}2048$, 2000 steps on a fixed instruction-tuning mixture. FlashAttention-2, fixed allocator config, fixed framework build.
- **Factors, fully crossed:** weight precision {bf16, NF4} × update {LoRA $r{=}16$, full-parameter via BAdam-style fused block update} × activation policy {layer-boundary checkpointing, $\ell^\star$ revolve schedule}. 8 arms.
- **Control arm.** bf16 full fine-tuning, no checkpointing, Adam fp32 — sharded across 4 GPUs if it does not fit, with per-device *and* aggregate memory reported. This is the $\epsilon = 0$ reference; without it the quality axis is uninterpretable.
- **Instrumentation.** Per-arm memory profile split into the five §2 terms, plus reserved-minus-allocated, plus wall-clock and FLOPs.
- **The deciding number.** The **marginal peak-memory reduction attributable to the PEFT factor with weight precision and activation policy held fixed**, as a fraction of control peak. Prediction from §10: $\le 5\%$. If it is $\le 5\%$, "parameter-efficient" is established as a compute-and-storage claim, not a training-memory claim, and the field's memory effort should move to $M_A$. If it exceeds 15%, the current accounting is wrong and needs republishing.

## 9. Key References

- **[Foundational]** Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin. *Training Deep Nets with Sublinear Memory Cost.* arXiv, 2016. — arXiv:1604.06174
- **[Foundational]** Andreas Griewank, Andrea Walther. *Algorithm 799: Revolve — An Implementation of Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation.* ACM Transactions on Mathematical Software, 2000.
- **[Foundational]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. — arXiv:2106.09685
- **[SOTA]** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. — arXiv:2305.14314
- **[SOTA]** Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian. *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* ICML 2024. — arXiv:2403.03507
- **[SOTA]** Yi-Lin Sung, Jaemin Cho, Mohit Bansal. *LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning.* NeurIPS 2022. — arXiv:2206.06522
- **[SOTA]** Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora. *Fine-Tuning Language Models with Just Forward Passes.* NeurIPS 2023. — arXiv:2305.17333
- **[SOTA]** Tim Dettmers, Mike Lewis, Sam Shleifer, Luke Zettlemoyer. *8-bit Optimizers via Block-wise Quantization.* ICLR 2022. — arXiv:2110.02861
- **[SOTA]** Jianyi Chen, Lianmin Zheng, Zhuohan Yang, Joseph E. Gonzalez, Michael W. Mahoney, et al. *ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training.* ICML 2021. — arXiv:2104.14129
- **[Evidence]** Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, et al. *LoRA Learns Less and Forgets Less.* TMLR, 2024. — arXiv:2405.09673
- **[Survey]** Vladislav Lialin, Vijeta Deshpande, Anna Rumshisky. *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning.* arXiv, 2023. — arXiv:2303.15647
- **[Context]** Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* ACL 2021. — arXiv:2012.13255

## 10. Worked Example

**QLoRA on LLaMA-2 7B, one 24 GB GPU.** $P = 6.74 \times 10^9$, $L=32$, $d=4096$, $B=4$, $T=2048$, LoRA $r=16$ on $\{q,k,v,o\}$.

Adapter parameters: $32 \times 4 \times (2 \times 4096 \times 16) = 1.68\times10^7$.

| Term | Formula | Bytes |
|---|---|---|
| Weights (NF4 + double quant) | $0.52 \times 6.74\text{e}9$ | 3.50 GB |
| Adapter weights + grads (bf16) | $4 \times 1.68\text{e}7$ | 0.07 GB |
| Adam states (fp32) | $8 \times 1.68\text{e}7$ | 0.13 GB |
| Stored boundaries ($\ell{=}1$) | $2 L B T d$ | 2.15 GB |
| Recompute window ($c{=}20$) | $2 c B T d$ | 1.34 GB |
| **Total (pre-fragmentation)** | | **7.19 GB** |

**Where the obstruction shows.** The parameter-efficient part of this configuration — adapter weights, gradients and optimizer states together — is 0.20 GB, **2.8% of peak**. The activation term is 3.49 GB, **49%**. Raising rank from 16 to 64 adds 0.6 GB; a method that eliminated the adapter entirely would save less than one-third of what a single checkpointing decision saves.

Turn checkpointing off and $M_A = 2cLBTd = 42.9$ GB — the run does not fit on a 24 GB card at any LoRA rank. The 20× reduction that makes single-GPU 7B tuning possible comes from a 2016 AD-scheduling trick, not from parameter efficiency.

Now the theory gap. Revolve says the $\ell^\star = \sqrt{L/c} \approx 1.26$ schedule is optimal *for exactly replaying the reverse sweep*. But adaptation only needs a descent direction within $\epsilon$. Nobody has proven whether $M_A$ can go below $\Theta(\sqrt{L}\,BTd)$ while remaining $\epsilon$-competitive — LST and 2-bit activation compression are existence attempts, both unvalidated at this scale against a full-FT control. That missing pair — a sub-$\sqrt{L}$ construction and a matching lower bound — is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*