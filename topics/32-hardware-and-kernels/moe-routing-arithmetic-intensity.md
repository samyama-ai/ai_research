---
id: 32-hardware-and-kernels/moe-routing-arithmetic-intensity
title: "Arithmetic Intensity Limits of Mixture-of-Experts Routing"
topic: 32-hardware-and-kernels
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Arithmetic Intensity Limits of Mixture-of-Experts Routing

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/moe-routing-arithmetic-intensity` · **Status:** open

## 1. Problem Statement

A sparse mixture-of-experts (MoE) layer routes each token to $k$ of $E$ experts. It cuts FLOPs per token by $E/k$ relative to a dense layer of the same parameter count. It does **not** cut the bytes moved: every activated expert's weights must be read from HBM at least once per forward pass. The consequence is that MoE trades FLOPs for arithmetic intensity, and arithmetic intensity is the quantity that determines whether a kernel runs at peak.

The problem has three variants that are routinely conflated.

- **Measurement.** Given a deployed MoE layer, what is its true arithmetic intensity $I$ (FLOP per byte moved to/from DRAM), decomposed into expert GEMMs, routing (top-$k$, sort, scatter/gather), and all-to-all? No standard decomposition exists; published "MFU" numbers for MoE models are aggregate and do not isolate the routing tax.
- **Method.** Construct a router that preserves the quality of top-$k$ routing while guaranteeing every expert receives at least $m^\*$ tokens per invocation, where $m^\*$ is the hardware ridge point. Expert-choice routing solves this for training and is not causal at decode.
- **Theory.** Is there a genuine tradeoff theorem? Informally: does any router whose token-to-expert assignment carries $H$ bits of input-dependent information necessarily admit a batch-size-dependent upper bound on achievable arithmetic intensity? No such theorem is known in either direction.

Solving it means: a router plus kernel schedule that reaches $\geq 50\%$ of dense-equivalent hardware utilization at decode batch sizes achievable on a single node, with no loss in downstream quality — or a proof that this is impossible.

## 2. Formal Setting

Let a layer process $T$ tokens with hidden size $d$. Each expert $e \in \{1,\dots,E\}$ is an FFN with weights of $P_e$ elements stored at $b$ bytes each (bf16: $b=2$; fp8: $b=1$). The router produces $g_t \in \binom{[E]}{k}$ and the per-expert token count is

$$m_e = |\{t : e \in g_t\}|, \qquad \sum_e m_e = kT, \qquad \bar m = kT/E.$$

**Arithmetic intensity, as measured.** For a grouped GEMM where expert $e$ maps $m_e$ tokens through weights of shape $d \times d_{\text{ff}}$ (both up and down projections, $P_e = 2 d d_{\text{ff}}$):

$$I_e = \frac{2 m_e P_e}{\underbrace{b P_e}_{\text{weights}} + \underbrace{2 b m_e (d + d_{\text{ff}})}_{\text{activations}}} \;\xrightarrow{\;m_e \ll d\;}\; \frac{2 m_e}{b}.$$

So in bf16, **arithmetic intensity in FLOP/byte is numerically equal to the tokens-per-expert count**, to within the activation term. The layer-level intensity is $I = \frac{2\sum_e m_e P_e}{\sum_{e:\,m_e>0} b P_e + \text{act} + \text{routing bytes}}$.

The ridge point is $m^\* = b\,\pi/2$ where $\pi = F_{\text{peak}}/B_{\text{peak}}$ is the machine balance (peak FLOP/s over peak DRAM bandwidth). Roofline (Williams et al., CACM 2009) gives achievable throughput $\min(F_{\text{peak}}, I \cdot B_{\text{peak}})$.

**Measured quantities.** $m_e$ from the router's histogram (a device-side counter, cheap). Bytes from hardware counters — `dram__bytes_read.sum` + `dram__bytes_write.sum` in Nsight Compute, not from an analytic model. FLOPs analytically from $2\sum_e m_e P_e$ (tensor-core instruction counters undercount padded grouped GEMMs). Load imbalance as $\rho = \max_e m_e / \bar m$.

**Assumptions, and which are violated.**

1. *Weights are read exactly once from DRAM.* Violated in both directions. With fine-grained experts (DeepSeekMoE: $d_{\text{ff}}$ per expert as low as 2048), a single expert's bf16 weights are ~8 MB and can stay resident in H100's 50 MB L2 across microbatches, raising effective intensity above the model. Conversely, tiled grouped GEMMs with poor scheduling re-read weight tiles.
2. *Perfect load balance*, $m_e = \bar m$. Violated: auxiliary-loss balancing leaves $\rho \approx 1.2$–$2$ at layer level even in well-tuned models, and much worse in the first and last layers.
3. *Routing cost is negligible.* Violated at small $\bar m$: top-$k$, sort, and scatter/gather have $I \approx 0.1$–$1$ FLOP/byte and become a nontrivial fraction of layer time once expert GEMMs are short.
4. *All-to-all overlaps compute.* Only holds when there is enough compute to hide it — precisely the regime that fails when $\bar m$ is small. The assumption is self-defeating.

## 3. State of the Art

**Systems SOTA (established).** MegaBlocks (Gale, Narayanan, Zaharia, Zoph; MLSys 2023) reformulates the MoE FFN as block-sparse matmul, removing token dropping and the padding waste of capacity-factor batching; reported up to $40\%$ end-to-end training speedup over Tutel and $2.4\times$ over dense Megatron-LM at billion-parameter scale. Tutel (Hwang et al., MLSys 2023) adds adaptive parallelism switching and reports $4.96\times$ single-layer speedup at 2048 A100s. Both are established and reproduced. Neither reports arithmetic intensity; both report time.

**Claimed but unablated.** DeepSeek-V3 (2024) attributes its inference throughput to large-scale expert parallelism plus DualPipe overlap, running prefill and decode on separate large EP groups. The technical report gives throughput but does not ablate the contribution of batch aggregation versus kernel work, so the fraction of the gain attributable to raising $\bar m$ past the ridge point is not established.

**Theory SOTA.** There is essentially none for this specific tradeoff. Roofline gives the bound; nothing connects routing entropy to $I$. Scaling-law work (Clark et al., *Unified Scaling Laws for Routed Language Models*, ICML 2022; Krajewski et al., *Scaling Laws for Fine-Grained Mixture of Experts*, 2024) models quality against $E$ and granularity but treats hardware cost as a FLOP count — exactly the abstraction that fails here.

**Benchmark-only results.** Most "MoE is $N\times$ faster" claims are single-configuration wall-clock numbers on one GPU generation, with no roofline decomposition. They do not transfer across machine balance.

## 4. What Is Known

- **The ridge point is large and rising.** H100 SXM: 989 TFLOP/s bf16 dense, 3.35 TB/s HBM3 → $\pi \approx 295$ FLOP/byte, so $m^\* \approx 295$ tokens per expert in bf16. B200: ~2250 TFLOP/s bf16, 8 TB/s → $\pi \approx 281$, $m^\* \approx 281$. Precision changes both terms: fp8 halves bytes ($I = 2m/1$) and doubles peak, leaving $m^\*$ near-invariant at $\approx 300$ tokens/expert.
- **The required batch scales as $E/k$.** With balanced routing, $\bar m \geq m^\*$ requires $T \geq m^\* E / k$. For Mixtral 8×7B ($E{=}8, k{=}2$): $T \geq 1180$. For DeepSeek-V3 ($E{=}256$ routed, $k{=}8$): $T \geq 9440$ concurrent tokens per layer invocation.
- **Fine-grained experts raise quality and lower intensity simultaneously.** DeepSeekMoE (Dai et al., ACL 2024) and Krajewski et al. (2024) both find that splitting experts finer improves the loss/FLOP frontier; both increase $E/k$ ratios that push $m^\*E/k$ up.
- **Imbalance is persistent.** Switch Transformer (Fedus et al., JMLR 2022) and ST-MoE (Zoph et al., 2022) both need an auxiliary load-balancing loss; even with it, capacity factors of 1.25–2.0 are used, implying $\rho$ well above 1.
- **Expert-choice routing (Zhou et al., NeurIPS 2022) makes $m_e$ exactly constant** by having experts select tokens — a hardware-perfect histogram. It is non-causal within a batch and so is unavailable for autoregressive decode without approximation.

## 5. What Is Not Known

- **Theoretically open.** No lower bound of the form "any causal router achieving conditional-computation gain $G$ over a dense model of equal active parameters admits $\bar m \leq f(G, T, E)$." It is not known whether balanced *and* input-dependent *and* causal routing is achievable without quality loss, or whether the three conflict.
- **Empirically open.** Nobody has published a roofline decomposition of an MoE layer at production scale that separates expert-GEMM bytes, routing bytes, and all-to-all bytes with hardware counters, across $\bar m$ swept from 8 to 1024. The experiment is a few thousand GPU-hours; the instrumentation exists.
- **Methodologically blocked.** "MFU" for MoE is not well defined: numerator FLOPs can be counted as active FLOPs, padded FLOPs, or dense-equivalent FLOPs, and the three differ by up to $E/k$. Papers do not state which they use. Until the numerator is fixed by convention, cross-paper efficiency comparison is meaningless.

## 6. Why It Is Hard

**Confounded measurement, specifically.** Three effects move layer time in the same direction and cannot be separated by wall-clock ablation: (i) $\bar m$ falling below $m^\*$, (ii) L2 residency of small expert weights, which *raises* effective intensity in a way analytic models miss, and (iii) all-to-all latency, which stops being hidden exactly when compute shrinks. Any experiment that changes $E$, $k$, or granularity moves all three at once. Isolating (i) requires holding bytes-to-DRAM fixed while varying $m_e$ — which the hardware will not do, because the cache decides.

Secondary: absent ground truth for the counterfactual. There is no way to ask "what would this router's quality be if it had produced a balanced histogram" without retraining, so the quality cost of forced balance is never measured against the same model.

## 7. Current Research (as of 2026)

- **Batch aggregation as the answer.** Disaggregated prefill/decode plus very wide expert parallelism to raise $T$ per invocation — DeepSeek's published inference architecture is the reference design; several inference-serving groups have followed. This concedes the routing problem and solves it with concurrency.
- **Grouped/scattered GEMM kernels** that avoid materializing padded expert batches: ScatterMoE (Tan et al., 2024), cuBLAS/CUTLASS grouped-GEMM paths, and Triton grouped kernels now standard in vLLM/SGLang. Reduces the activation-byte term, not the weight-byte term.
- **Communication–computation fusion** for all-to-all overlap at the kernel level *(frontier — verify)*.
- **Router regularization targeting the histogram directly** rather than an entropy surrogate — loss-free balancing via per-expert bias adjustment (DeepSeek-V3) is the deployed instance.
- **Theory of routing capacity** connecting router entropy to achievable batching remains essentially unstaffed. This is the gap.

## 8. Concrete Next Experiment

**Question.** How much of the MoE efficiency loss is arithmetic intensity, as opposed to all-to-all and routing overhead?

**Scale.** One 8×H100 node. A 16B-total / 2.4B-active MoE, $E{=}64$, $k{=}6$, $d{=}2048$, $d_{\text{ff}}^{\text{expert}}{=}1408$, 24 layers, bf16, expert parallelism 8. Sweep tokens per forward $T \in \{64, 256, 1024, 4096, 16384\}$, giving $\bar m \in \{6, 24, 96, 384, 1536\}$ — straddling $m^\* \approx 295$.

**Control arm.** The same layer with routing frozen to a fixed, perfectly balanced round-robin assignment (identical $m_e$, identical GEMM shapes, identical all-to-all volume, zero router compute). This isolates dynamic routing cost from shape cost. Second control: a dense FFN with the same *active* parameter count.

**The deciding number.** $\eta(\bar m) = \dfrac{\text{measured achieved FLOP/s}}{\min(F_{\text{peak}},\, I_{\text{measured}} \cdot B_{\text{peak}})}$, with $I_{\text{measured}}$ from `dram__bytes.*` counters. If $\eta \geq 0.85$ across the whole sweep, MoE inefficiency *is* arithmetic intensity and the fix is batching or bigger experts. If $\eta$ collapses below $0.6$ at small $\bar m$ while the balanced control stays high, the loss is routing and communication, and kernel work is the fix. Report $\eta$ and $\rho$ per layer; both are single numbers per configuration.

## 9. Key References

- **[Foundational]** N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, J. Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — arXiv:1701.06538
- **[Foundational]** S. Williams, A. Waterman, D. Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM 52(4), 2009.
- **[Foundational]** W. Fedus, B. Zoph, N. Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23, 2022. — arXiv:2101.03961
- **[SOTA]** T. Gale, D. Narayanan, M. Zaharia, B. Zoph. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys 2023. — arXiv:2211.15841
- **[SOTA]** C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram, J. Chau, P. Cheng, F. Yang, M. Yang, Y. Xiong. *Tutel: Adaptive Mixture-of-Experts at Scale.* MLSys 2023. — arXiv:2206.03382
- **[SOTA]** Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Zhao, A. Dai, Z. Chen, Q. Le, J. Laudon. *Mixture-of-Experts with Expert Choice Routing.* NeurIPS 2022. — arXiv:2202.09368
- **[SOTA]** R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, A. Levskaya, J. Heek, K. Xiao, S. Agrawal, J. Dean. *Efficiently Scaling Transformer Inference.* MLSys 2023. — arXiv:2211.05102
- **[Systems]** D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, Z. Chen. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR 2021. — arXiv:2006.16668
- **[Systems]** J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, Q. Li. *FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models.* PPoPP 2022.
- **[Scaling]** A. Clark, D. de las Casas, A. Guy, A. Mensch, M. Paganini, J. Hoffmann, et al. *Unified Scaling Laws for Routed Language Models.* ICML 2022. — arXiv:2202.01169
- **[Scaling]** J. Krajewski, J. Ludziejewski, K. Adamczewski, et al. *Scaling Laws for Fine-Grained Mixture of Experts.* 2024. — arXiv:2402.07871
- **[Model]** D. Dai, C. Deng, C. Zhao, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL 2024. — arXiv:2401.06066
- **[Model]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Model]** A. Q. Jiang, A. Sablayrolles, A. Roux, et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Kernel]** S. Tan, Y. Shen, R. Panda, A. Courville. *Scattered Mixture-of-Experts Implementation.* 2024. — arXiv:2403.08245

## 10. Worked Example

Mixtral 8×7B decode on one H100 SXM. $E{=}8$, $k{=}2$, $d{=}4096$, $d_{\text{ff}}{=}14336$. Each expert has three matrices ($w_1, w_2, w_3$), $P_e = 3 \times 4096 \times 14336 = 1.76\times10^8$ params $= 352$ MB in bf16.

Take a decode batch of $T=32$ tokens. Then $\bar m = 32 \times 2 / 8 = 8$ tokens per expert. With any realistic imbalance, all 8 experts are touched, so all 2.8 GB of expert weights are read. FLOPs $= 2 \times 32 \times 2 \times 1.76\times10^8 = 2.25\times10^{10}$.

$$I = \frac{2.25\times10^{10}}{2.82\times10^{9} + \text{act}} \approx 8.0 \text{ FLOP/byte}.$$

Roofline ceiling: $8.0 \times 3.35\times10^{12} = 26.8$ TFLOP/s, i.e. **2.7% of the 989 TFLOP/s peak**. Time $\approx 2.82\text{ GB} / 3.35 \text{ TB/s} = 0.84$ ms per layer — pure bandwidth.

Now raise the batch to $T = 1180$, so $\bar m = 295 = m^\*$. Weight bytes are unchanged at 2.82 GB. FLOPs rise $37\times$. Time per layer rises only to $\approx 0.84$ ms (still bandwidth-bound, just barely), and utilization goes to ~99%. **The same bytes now buy 37× the work.**

Here is the obstruction, made visible. At $T=32$ the layer is bandwidth-bound and the router is irrelevant to performance — any assignment reads all the weights. At $T=1180$ the layer is at the ridge and the router's *histogram* becomes the only thing that matters: an imbalance of $\rho = 1.5$ means the slowest expert has $m_e = 443$ while some other has ~150, and with expert parallelism the layer waits for the slowest. So the routing decision is performance-neutral in the regime where it is cheap to change, and performance-critical in the regime where changing it costs quality. No experiment run at $T=32$ — which is where most single-GPU MoE benchmarks live — can observe the effect it claims to measure.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*