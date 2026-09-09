---
id: 15-mixture-of-experts/moe-inference-batch-efficiency-cliff
title: "MoE Inference Batch-Size Efficiency Cliff"
topic: 15-mixture-of-experts
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# MoE Inference Batch-Size Efficiency Cliff

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/moe-inference-batch-efficiency-cliff` · **Status:** empirically-open

## 1. Problem Statement

A sparse MoE model with $E$ experts and top-$k$ routing activates $k/E$ of its expert parameters per token. At batch size 1 this buys a real speedup over a dense model of equal total size. As the decode batch grows, the union of experts touched by the batch grows until it covers all $E$, and the weight traffic per step equals that of the dense model — while the useful FLOPs still only count $k$ experts per token. Throughput per GPU therefore does not scale with batch the way it does for a dense model. The gap has a shape: a sublinear segment at small batch, then a long memory-bound plateau, then a compute-bound regime reached only at batch sizes $E/k$ times larger than a dense model needs.

Three variants:

- **Measurement.** Given a model, a serving stack, and a hardware target, produce $T(B)$ — steady-state decode tokens/s per GPU as a function of concurrent sequences $B$ — and locate $B^\star$, the batch at which $T$ reaches 90% of its asymptote. Currently no cross-family measurement on a common harness exists.
- **Method.** Design a serving strategy that reaches a given fraction of peak at batch $B \ll I_r E/k$ (see §2), without changing the model. Candidates: expert caching, batch-aware routing, expert offload, speculative expert prefetch, replica-level request routing that correlates expert use.
- **Theory.** Given a routing distribution and a memory hierarchy, lower-bound the bytes any correct execution must move for a batch of $B$ tokens. No such bound exists.

Solving it means: a predictive model of $T(B)$ accurate to within 15% across $(E, k, d, \text{hardware})$, plus a method that shifts $B^\star$ down by a stated factor.

## 2. Formal Setting

An MoE layer has $E$ experts, top-$k$ routing, model width $d$, expert hidden width $d_{ff}$, and a gated-MLP expert with three matrices, so $P_e = 3 d\, d_{ff}$ parameters per expert. Weights are stored in $b$ bytes/element (measured as the on-device dtype, not the checkpoint dtype). Hardware has peak dense throughput $F$ FLOP/s and HBM bandwidth $W$ B/s; the **ridge point** is $I_r = F/W$ FLOP/byte (H100 SXM BF16: $989\times10^{12}/3.35\times10^{12} \approx 295$).

Let $Z(B)$ be the number of **distinct** experts touched by a decode step of $B$ tokens. Under i.i.d. uniform routing,

$$\mathbb{E}[Z(B)] = E\left(1 - \left(1 - \tfrac{k}{E}\right)^{B}\right).$$

Per layer per step: useful FLOPs $\Phi(B) = 2 B k P_e$; weight bytes $M(B) = b\, P_e\, \mathbb{E}[Z(B)]$. Arithmetic intensity is

$$I(B) = \frac{\Phi(B)}{M(B)} = \frac{2Bk}{b\,\mathbb{E}[Z(B)]} \;\xrightarrow[B \gg E/k]{}\; \frac{2Bk}{bE}.$$

With $b=2$, saturation ($I(B)=I_r$) needs

$$\boxed{B^{\text{ridge}}_{\text{MoE}} = I_r \cdot \frac{E}{k}}, \qquad B^{\text{ridge}}_{\text{dense}} = I_r,$$

for a dense model of the same *active* parameter count. **The MoE needs $E/k$ times more concurrency to reach the same efficiency.** Mixtral 8×7B: $E/k=4$, $B^{\text{ridge}}\approx 1180$. DeepSeek-V3: $E/k=32$, $B^{\text{ridge}}\approx 9440$ tokens per expert-parallel group.

Measured quantities, as instrumented:
- $T(B)$: output tokens/s divided by GPU count, steady-state decode only, prefill excluded, fixed context length, measured over ≥200 steps after warmup.
- $\eta_{\text{act}}(B) = \Phi_{\text{total}}(B) / (F \cdot t_{\text{step}})$ — MFU counting only active-expert FLOPs.
- $\eta_{\text{iss}}(B)$ — MFU counting FLOPs actually issued, including tile padding when per-expert row count $Bk/E$ is below the GEMM tile height $m_{\text{tile}}$ (128 for typical BF16 tensor-core kernels).
- Imbalance $\rho(B) = \max_j n_j / \bar{n}$ over expert token counts $n_j$; under expert parallelism, step time scales with $\rho$.

Assumptions and their status: **uniform i.i.d. routing** — violated; routing is skewed and strongly correlated within a sequence and across a domain, so $\mathbb{E}[Z(B)]$ overestimates coverage at small $B$ and $\rho \gg 1$ at large $B$. **Weights HBM-resident** — violated under offload. **Attention cost negligible** — violated beyond ~4K context, where KV traffic dominates and masks the expert cliff. **All-to-all overlapped with compute** — only partially true; under expert parallelism the collective adds a latency term that itself depends on $B$.

## 3. State of the Art

**Systems/empirical SOTA.** Expert parallelism with fused grouped GEMMs is the standard: MegaBlocks (Gale et al., MLSys 2023) removes token-dropping and padding via block-sparse kernels; Tutel (Hwang et al., MLSys 2023) adds adaptive parallelism switching; DeepSpeed-MoE (Rajbhandari et al., ICML 2022) reports up to 7.3× better inference latency/cost than a quality-equivalent dense model. DeepSeek-V3 (2024) is the clearest existence proof that the cliff is real *and* addressable by deployment scale: it serves prefill at EP32 and **decode at EP320 across 40 nodes with one routed expert per GPU plus redundant hot experts**, an architecture that exists specifically to assemble the aggregate batch and balance $\rho$ that a 256-expert top-8 model requires.

**Theory SOTA.** Pope et al., *Efficiently Scaling Transformer Inference* (MLSys 2023) gives the dense roofline and the batch-vs-latency Pareto analysis that the MoE formula above extends; there is no equivalent published MoE-specific lower bound.

**Established:** the roofline formula for $B^{\text{ridge}}$ follows directly from parameter counts and is arithmetic, not an empirical claim. Expert offload with prefetch and caching (Eliseev & Mazur, 2023) works at batch 1 on consumer GPUs.

**Claimed but unablated:** vendor and framework throughput tables for MoE serving are reported at one or two batch points, usually the favourable one. Speedup claims over dense baselines rarely fix *active* parameters, hardware, quantization, and context length simultaneously — so the reported ratio confounds the sparsity benefit with the model-size difference. **Benchmark-number-only:** most published MoE tokens/s figures come from a single harness at a single concurrency and cannot be used to reconstruct $T(B)$.

## 4. What Is Known

- **Coverage saturates fast.** Mixtral 8×7B ($E=8,k=2$): $\mathbb{E}[Z]$ = 2.0 at $B{=}1$, 5.5 at $B{=}4$, 7.2 at $B{=}8$, 7.92 at $B{=}16$. DeepSeek-V3 ($E=256,k=8$): $\mathbb{E}[Z]=222$ at $B{=}64$, 252 at $B{=}128$. Past ~$(E/k)\ln E$ tokens, every step reads every expert.
- **Model sizes, at the scales named.** Mixtral 8×7B: 46.7B total / 12.9B active, $d{=}4096$, $d_{ff}{=}14336$, 32 layers → 352 MB/expert in BF16, 2.82 GB per MoE layer, ~93 GB total. DeepSeek-V3: 671B total / 37B active, 256 routed + 1 shared expert, top-8, $d{=}7168$, expert $d_{ff}{=}2048$ → 44.0M params/expert, 22.5 GB of routed-expert weights per layer in BF16 — more than one 80 GB GPU can hold for even two layers. Qwen3-30B-A3B: 30.5B total / 3.3B active, 128 experts, top-8, 48 layers.
- **Routing is not uniform.** Switch Transformer (Fedus et al., JMLR 2022) and GShard (Lepikhin et al., ICLR 2021) both require an explicit auxiliary load-balancing loss precisely because unregularized routing collapses; DeepSeek-V3 replaces it with a bias-based balancing scheme and adds redundant copies of hot experts at serving time. Both facts are evidence that $\rho > 1$ persists at deployment scale.
- **The step is memory-bound over a wide range.** For Mixtral at $B{=}128$, per-layer weight traffic is 2.82 GB (841 µs on H100) against 90.2 GFLOP of useful work (91 µs) — a 9.2× memory-bound margin. Even with 4× tile padding at $Bk/E = 32$ rows/expert, compute stays under the memory time, so the padding waste is invisible in wall-clock and appears only in $\eta_{\text{iss}}$.

## 5. What Is Not Known

- **Empirically open (primary).** No published measurement sweeps $T(B)$ across $B \in \{1,\dots,4096\}$ for two or more MoE families and a matched dense control on one harness and one hardware target. The prediction $B^\star_{\text{MoE}}/B^\star_{\text{dense}} \approx E/k$ is untested. The experiment costs one 8-GPU node-week and nobody has run it.
- **Empirically open.** Whether routing correlation within realistic request mixes (same domain, same prompt template, shared prefix) reduces $Z(B)$ enough to matter, and whether an expert-affinity request router can exploit it. Correlation strength has not been measured on production traces.
- **Theoretically open.** A lower bound on bytes moved for $B$ tokens under a given routing distribution and cache size. Whether an online expert-caching policy can be competitive against the offline optimum for correlated routing sequences is unproven either way.
- **Methodologically blocked.** *MFU is not well defined for MoE.* $\eta_{\text{act}}$, $\eta_{\text{iss}}$, and a "total-parameter" MFU can differ by more than 4× on the same run. Papers report "MFU" without saying which. Until the field fixes one convention, cross-paper efficiency comparisons for MoE are uninterpretable.

## 6. Why It Is Hard

**Confounded measurement, plus an evaluation that does not measure what it names.** Three effects move together as $B$ rises and no published protocol separates them: (i) expert-coverage growth $Z(B)$, (ii) GEMM tile utilization $\min(1, Bk/(E\,m_{\text{tile}}))$, (iii) load imbalance $\rho(B)$, which *falls* with $B$ under uniform routing but is set by skew in practice. A throughput curve is the product of all three, and any of the three can be credited for a knee. Attention KV traffic is a fourth term that grows with $B \times$ context and can dominate the whole step, hiding the expert cliff entirely at long context — which is why measurements at 32K context and at 512 context tell different stories about the same model.

The naming failure compounds it: "MFU" reported for an MoE almost always means $\eta_{\text{act}}$, which credits the model for FLOPs it skipped while charging it nothing for the bytes it moved anyway. A model can show rising MFU across exactly the batch range where per-GPU throughput is flat.

Compute cost is secondary but real: reproducing the DeepSeek-V3 regime needs $B \approx 9400$ concurrent tokens per EP group, which means a multi-node deployment, not a single GPU.

## 7. Current Research (as of 2026)

- **Large-EP disaggregated serving.** DeepSeek's EP320 decode design and the open reimplementations of it (SGLang, vLLM expert-parallel paths) are the main production line of attack: aggregate enough concurrency across nodes that $B$ clears the ridge, then fight $\rho$ with redundant expert replicas and dynamic rebalancing. *(frontier — verify current EP degree and rebalancing cadence in released configs.)*
- **Prefill/decode disaggregation** (DistServe, OSDI 2024; Splitwise, ISCA 2024) is orthogonal but interacts: it makes the decode pool's batch composition controllable, which is exactly the knob this problem needs.
- **Fine-grained expert architectures.** DeepSeekMoE-style designs push $E$ up and expert size down, which *raises* $E/k$ and pushes $B^{\text{ridge}}$ out. Whether the quality gain per unit of serving concurrency is favourable is an open architecture-serving co-design question. *(frontier — verify.)*
- **Expert caching and offload** for the single-user regime: Eliseev & Mazur (2023) and successors. Effective at $B{=}1$, and the exact regime that the cliff analysis says degrades fastest as $B$ rises.
- **Quantization interaction.** Dropping expert weights to 4 bits cuts $M(B)$ by 4× and therefore cuts $B^{\text{ridge}}$ by 4×. This is arguably the highest-leverage lever and is undermeasured as a *batch-efficiency* intervention rather than a memory-footprint one.

## 8. Concrete Next Experiment

**Scale.** One 8×H100-80GB node, NVLink. Three models: Mixtral 8×7B ($E/k=4$), Qwen3-30B-A3B ($E/k=16$), and **control arm** Llama-3.1-8B dense (matched to Mixtral's 12.9B active within a factor of 1.6). All BF16, all on the same serving stack (vLLM or SGLang, one version, pinned).

**Protocol.** Decode-only steady state: fixed 512-token prefix, generate 256 tokens, sweep concurrency $B \in \{1,2,4,8,16,32,64,128,256,512,1024,2048,4096\}$. Report per-$B$: tokens/s/GPU, per-step time, $\eta_{\text{act}}$, $\eta_{\text{iss}}$, measured $Z(B)$ from router logits, and $\rho(B)$. Repeat the whole sweep at 8K context to quantify KV masking. Repeat Mixtral at 4-bit expert weights. Instrumentation must log $Z$ and $\rho$ per layer per step — this is the part existing harnesses lack and the reason the experiment is unrun.

**The deciding number.** $B^\star$ = smallest batch reaching 90% of asymptotic tokens/s/GPU. Compute the ratio

$$R = \frac{B^\star_{\text{MoE}}}{B^\star_{\text{dense}}}.$$

The roofline predicts $R \approx E/k$: 4 for Mixtral, 16 for Qwen3-30B-A3B. **If $R$ lands within $[0.7, 1.4] \times E/k$ for both models, the cliff is fully explained by expert coverage and the problem reduces to an engineering target ($B^{\text{ridge}} \propto E/(k b)$, attack it with quantization and aggregation). If $R$ is materially below $E/k$, routing correlation is doing real work and expert-affinity request routing becomes the lever. If $R$ exceeds $E/k$, the residual is all-to-all and imbalance, and $\rho(B)$ from the same logs identifies which.** One node-week.

## 9. Key References

- **[Foundational]** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Dmitry Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* ICLR, 2021. — arXiv:2006.16668
- **[Foundational]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 23, 2022. — arXiv:2101.03961
- **[SOTA — roofline]** Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean. *Efficiently Scaling Transformer Inference.* MLSys, 2023. — arXiv:2211.05102
- **[SOTA — kernels]** Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* MLSys, 2023. — arXiv:2211.15841
- **[SOTA — systems]** Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He. *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.* ICML, 2022. — arXiv:2201.05596
- **[SOTA — systems]** Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, Joe Chau, Peng Cheng, Fan Yang, Mao Yang, Yongqiang Xiong. *Tutel: Adaptive Mixture-of-Experts at Scale.* MLSys, 2023. — arXiv:2206.03382
- **[SOTA — deployment]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Model]** Albert Q. Jiang et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Serving]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Serving]** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI, 2024. — arXiv:2401.09670
- **[Offload]** Artyom Eliseev, Denis Mazur. *Fast Inference of Mixture-of-Experts Language Models with Offloading.* 2023. — arXiv:2312.17238
- **[Survey]** Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, Jiayi Huang. *A Survey on Mixture of Experts.* 2024. — arXiv:2407.06204

## 10. Worked Example

Mixtral 8×7B, one MoE layer, H100 SXM ($W = 3.35$ TB/s, $F = 989$ TFLOP/s BF16). Per expert: $3 \times 4096 \times 14336 = 176.2$M params = 352 MB. Useful FLOPs per token: $2 \times 2 \times 176.2\text{M} = 0.705$ GFLOP.

| $B$ | $\mathbb{E}[Z]$ | bytes (GB) | mem time | FLOP | cmp time | step (32 layers) | tok/s | $\eta_{\text{act}}$ | rows/expert |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.00 | 0.70 | 210 µs | 0.7 G | 0.7 µs | 6.7 ms | 149 | 0.10% | 0.25 |
| 8 | 7.20 | 2.53 | 756 µs | 5.6 G | 5.7 µs | 24.2 ms | 331 | 0.75% | 2 |
| 16 | 7.92 | 2.79 | 832 µs | 11.3 G | 11 µs | 26.6 ms | 601 | 1.4% | 4 |
| 128 | 8.00 | 2.82 | 841 µs | 90 G | 91 µs | 26.9 ms | 4,755 | 10.9% | 32 |
| 1024 | 8.00 | 2.82 | 841 µs | 722 G | 730 µs | 26.9 ms | 38,041 | 86% | 256 |
| 4096 | 8.00 | 2.82 | 841 µs | 2.89 T | 2.92 ms | 93.4 ms | 43,812 | 99% | 1024 |

Read the throughput column. From $B{=}1$ to $B{=}16$ — a 16× batch increase — throughput rises only **4.0×**. That is the cliff: each added token in that range drags in fresh expert weights, so the marginal token costs bandwidth, not just compute. From $B{=}16$ to $B{=}1024$ throughput is exactly linear (64× for 64×), because the weight bill is already fully paid at 2.82 GB and every extra token is free. Saturation arrives at $B \approx 1180 = 295 \times 4$, as the formula says. A dense 12.9B-active model reading 0.352 GB/layer would have saturated at $B \approx 147$.

**Where the obstruction becomes visible.** At $B{=}128$ the model is running at 10.9% $\eta_{\text{act}}$ — but the per-expert GEMM has only 32 rows against a 128-row tile, so the hardware issues 4× the useful FLOPs and $\eta_{\text{iss}} \approx 44\%$. Both numbers are "MFU". One says the kernel is idle; the other says it is nearly half-busy. Neither predicts wall-clock, because wall-clock is set by the 841 µs of weight traffic that appears in *neither* metric. A paper reporting either number in isolation, at this single batch point, is reporting something that does not determine throughput — which is why §5 classifies the measurement as blocked and §8 insists on logging $Z(B)$, $\rho(B)$, and both MFU variants together.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*