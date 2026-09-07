---
id: 11-inference-and-serving/communication-lower-bounds-decoding
title: "Communication Lower Bounds for Distributed Decoding"
topic: 11-inference-and-serving
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Communication Lower Bounds for Distributed Decoding

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/communication-lower-bounds-decoding` · **Status:** open

## 1. Problem Statement

A transformer with parameters $\theta$ is sharded across $P$ accelerators. Autoregressive decoding emits one token per forward pass, so every inter-device exchange sits on the critical path of user-visible latency. The question:

**How many bits must cross the interconnect, and in how many synchronization rounds, to emit one token from a model whose parameters and KV cache do not fit on one device?**

Three variants, with different difficulty:

- **Theory variant.** Fix a sharding of $\theta$ across $P$ devices (each device holds a disjoint slice, none holds the whole model). Lower-bound the bits $C$ and rounds $R$ any protocol must use to produce a token sampled from a distribution within total variation $\varepsilon$ of the true $p_\theta(\cdot \mid x_{<t})$. Open even for a single attention-plus-MLP block.
- **Method variant.** Construct a decoding protocol that beats $2\times$ all-reduce-per-layer tensor parallelism at fixed output quality. Speculative decoding, expert parallelism with dropped tokens, and low-rank activation compression are all attacks; none has a matching lower bound saying how far they can go.
- **Measurement variant.** Report communication in a way that composes. Bytes-on-the-wire, rounds, and achieved latency are three different numbers, and current papers report whichever is favorable.

Solving it means: a lower bound $C \geq f(P, d, L, \varepsilon)$ that holds for *all* protocols under a stated sharding, plus a protocol matching it to constants.

## 2. Formal Setting

Model: $L$ layers, hidden width $d$, vocabulary $V$, KV cache of $S$ tokens, batch $B$, weights in $b_w$ bits, activations in $b_a$ bits. $P$ devices, tensor-parallel degree $T$, pipeline depth $D$, $P = TD$.

**Cost model.** Hockney $\alpha$–$\beta$: a message of $n$ bytes costs
$$t(n) = \alpha + \beta n,$$
$\alpha$ = per-message latency (measured: round-trip of a 4-byte all-reduce, NCCL, warm, median of $10^4$ trials), $\beta$ = inverse achieved bandwidth (measured: 1 GB all-reduce time divided by bytes moved per device, *not* vendor peak).

**Per-token communication.** Megatron-style tensor parallelism issues two all-reduces per layer on a tensor of $B \cdot d$ activations. With a bandwidth-optimal ring (Patarasuk & Yuan, 2009), bytes moved per device per all-reduce are $2\frac{T-1}{T} B d b_a/8$, so
$$C_{\text{token}} = 2L \cdot 2\frac{T-1}{T} \cdot \frac{B d\, b_a}{8}, \qquad R_{\text{token}} = 2L \cdot 2(T-1),$$
and modeled latency
$$t_{\text{token}} \;=\; R_{\text{token}}\,\alpha \;+\; C_{\text{token}}\,\beta \;+\; \underbrace{\frac{|\theta| b_w/8}{P \cdot \mathrm{BW}_{\text{HBM}}}}_{\text{weight read}}.$$

**Quality constraint.** Let $q$ be the protocol's output distribution. The protocol is $\varepsilon$-faithful if $\mathbb{E}_{x_{<t}}\,\mathrm{TV}(q(\cdot\mid x_{<t}),\,p_\theta(\cdot\mid x_{<t})) \le \varepsilon$. Measured as: mean TV over $\ge 10^5$ held-out prefixes against an unsharded fp32 reference — not as downstream benchmark accuracy, which is a lossy proxy.

**Assumptions, and which are false in practice.**
- *Linear $\alpha$–$\beta$, congestion-free.* Violated: NVLink switch contention and NIC incast make $\beta$ batch-size-dependent; the ring's $2(T-1)$ rounds are not the achieved round count on tree/NVLS algorithms.
- *Communication does not overlap compute.* Violated in decode-heavy regimes only partially — with $B\cdot d$ small there is nothing to overlap with.
- *Fixed sharding across the token.* Violated by MoE (routing changes the sharding per token) and by disaggregated prefill/decode.
- *Devices are honest and synchronous.* Holds inside one node; fails across a WAN.

## 3. State of the Art

**Theory SOTA (established).** Classical collective bounds: any all-reduce of $n$ bytes on $P$ nodes needs $\ge \lceil\log_2 P\rceil \alpha$ latency and $\ge 2\frac{P-1}{P} n\beta$ bandwidth under the one-port model (Chan, Heimlich, Purkayastha & van de Geijn, *CCPE* 2007); ring all-reduce attains the bandwidth term (Patarasuk & Yuan, *JPDC* 2009). Two-party communication complexity (Yao, STOC 1979) and lower bounds on information transfer in distributed computation (Abelson, 1980) give the general machinery. Distributed-optimization lower bounds (Tsitsiklis & Luo 1987; Arjevani & Shamir, NeurIPS 2015) show the template for "bits vs. accuracy" trade-offs — but for *training*, not decoding.

**No published bound treats decoding as the primitive.** Existing bounds are per-collective, so they lower-bound a *given algorithm*'s collectives, not the token-emission task. A protocol allowed to change the sharding, approximate the softmax, or speculate is not covered.

**Systems SOTA (established, reproduced).** Megatron-LM tensor parallelism (Shoeybi et al., 2019); *Efficiently Scaling Transformer Inference* (Pope et al., MLSys 2023) — partitioning layouts chosen analytically per regime, with the 2D weight-gathered layout shifting communication from activations to weights at large batch; PagedAttention/vLLM (Kwon et al., SOSP 2023) for memory, not communication; Ring Attention (Liu, Zaharia & Abbeel, 2023) and DeepSpeed-Ulysses (Jacobs et al., 2023) for sequence-dimension sharding; DistServe (Zhong et al., OSDI 2024) and Splitwise (Patel et al., ISCA 2024) for prefill/decode disaggregation, which converts collective traffic into KV-cache transfer.

**Claimed but unablated.** Activation-compression schemes for tensor-parallel inference report end-to-end speedups without holding TV distance fixed; the quality control arm is usually a benchmark score, not a distributional distance. Speculative decoding (Leviathan, Kalman & Matias, ICML 2023; Chen et al., 2023) is *provably* distribution-preserving for the sampled token, but its communication accounting across devices exists only as wall-clock benchmark numbers.

## 4. What Is Known

- **Decode is latency-bound, not bandwidth-bound.** At $B{=}32$, $d{=}8192$, bf16, one all-reduce payload is 512 KB; at $L{=}80$ that is 160 all-reduces per token. The $R\alpha$ term dominates $C\beta$ on NVLink for all $B \lesssim 10^2$ (derived from the model in §2; see §10 for the arithmetic).
- **Bandwidth-optimality is settled for the collective.** Ring all-reduce is within a factor 1 of the $2\frac{P-1}{P}n\beta$ bound (Patarasuk & Yuan 2009), measured on clusters of $P \le 64$.
- **Partitioning strategy changes the exponent in $B$.** Pope et al. (MLSys 2023) show, on PaLM 540B at TP up to 64 chips, that the optimal layout switches between activation-communicating and weight-gathered as batch grows; they report reaching 76% of peak MFU in prefill but far lower in decode, at 29 ms/token targets.
- **Speculation is exactly faithful.** Leviathan et al. prove the modified rejection sampler emits from $p_\theta$ exactly ($\varepsilon = 0$), with $2$–$3\times$ measured walltime speedup on T5-XXL 11B and 7B–70B decoder models.
- **Tensor parallelism does not scale past a node boundary.** Widely reproduced: crossing from NVLink ($\alpha \approx 2$–$5\,\mu$s) to InfiniBand ($\alpha \approx 10$–$20\,\mu$s, order-of-magnitude lower $1/\beta$) makes $T > 8$ slower than $T = 8$ for 70B-class decode.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on bits-per-token for $\varepsilon$-faithful decoding under *any* sharding of $\theta$. Even the simplest case — one MLP layer, weights column-split across two devices, output token sampled to TV $\varepsilon$ — has no published $\Omega(\cdot)$ bound. Nor is there a round-complexity bound: is $\Omega(L)$ synchronization rounds necessary, or can a constant-round protocol exist with polynomially more bits?
- **Empirically open.** Whether an approximate all-reduce at fixed measured TV distance yields real end-to-end gains at 70B–400B scale is runnable today and unrun with the right control (see §8).
- **Methodologically blocked.** "Communication cost" is not a defined scalar. Papers report bytes, rounds, or achieved µs interchangeably; NCCL's algorithm selection changes $R$ silently between runs, so the same code has different round complexity on different topologies. Until the reported quantity is fixed, cross-paper comparison is not meaningful.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the sharding from the task**. A lower bound needs a fixed input partition — the standard trick is to assign each party an input and argue the output depends on both. Here the "input" is the *model*, and the protocol designer chooses how to split it. Any bound proved for column-parallel MLPs is escaped by re-sharding, by replicating small tensors, or by speculating with a local draft model that needs no communication at all. Two devices each holding a full copy communicate zero bits, so the bound must be stated relative to a memory budget per device: $\Omega(C)$ subject to $\text{mem}_i \le M$. That coupled memory-communication trade-off is the hard object, and it is not a standard communication-complexity setting.

Secondary: the measurement is confounded. $\alpha$ and $\beta$ are not constants — they depend on message size, concurrency, and NCCL's runtime algorithm choice — so an "improved protocol" and "a different collective algorithm got selected" are indistinguishable from wall-clock alone.

## 7. Current Research (as of 2026)

- **Disaggregation** (DistServe, Splitwise, Mooncake): move traffic off the per-token critical path into KV transfer. Microsoft, PKU, Moonshot.
- **Sequence/context parallelism** for long contexts (Ring Attention, Ulysses; Berkeley, Microsoft) — changes what is communicated from activations to KV blocks.
- **Communication-aware quantization of activations** for the tensor-parallel all-reduce *(frontier — verify)*: several 2025–2026 preprints claim int4/int8 all-reduce for decode; faithfulness controls are weak.
- **MoE all-to-all lower bounds** *(frontier — verify)*: expert parallelism replaces all-reduce with all-to-all whose cost depends on the routing distribution; a bound would need to be over random routings.
- **Compiler-side collective scheduling** (XLA/Mosaic, JAX `shard_map`; Google) — overlap and fusion, treating $R$ as the target.

No group is, to my knowledge, working the lower-bound side directly. This is the gap.

## 8. Concrete Next Experiment

**Question:** does reducing bits per all-reduce buy latency at fixed faithfulness, or is decode purely round-bound?

- **Scale.** Llama-3-70B ($L{=}80$, $d{=}8192$), 8×H100 single node, NVLink, TP $=8$, bf16 baseline. Batch sweep $B \in \{1, 8, 32, 128\}$, decode-only, 512-token prefix, 256 tokens generated, $10^5$ prompts from a held-out mixture.
- **Arms.** (a) Control: bf16 ring all-reduce, NCCL algorithm pinned via `NCCL_ALGO=Ring` so $R$ is fixed and known. (b) fp8 all-reduce, identical round count. (c) int4 all-reduce. (d) Round-reduction arm: fuse the two per-layer all-reduces into one by deferring the attention-output reduction into the MLP input reduction, halving $R$ at unchanged $C$.
- **Instrumentation.** Per-arm: measured bytes-on-wire (NCCL counters), measured round count, per-token latency (p50/p99), and $\mathrm{TV}(q, p)$ against the fp32 single-device reference on the full vocabulary.
- **Deciding number.** The latency reduction of arm (d) minus that of arm (b), at $B = 32$, at $\mathrm{TV} \le 10^{-3}$. If (d) $-$ (b) $> 0$ — halving rounds beats halving bits — decode is round-bound and the research target is round complexity, not compression, which retires an entire line of activation-quantization work. If (b) wins, bit-level bounds are the right object. Expected, from §10: (d) wins by roughly 1.5–2 ms/token at $B{=}32$; (b) buys under 0.2 ms.

## 9. Key References

- **[Foundational]** A. C.-C. Yao. *Some complexity questions related to distributive computing.* STOC, 1979.
- **[Foundational]** H. Abelson. *Lower bounds on information transfer in distributed computations.* Journal of the ACM, 1980.
- **[Foundational]** E. Chan, M. Heimlich, A. Purkayastha, R. van de Geijn. *Collective communication: theory, practice, and experience.* Concurrency and Computation: Practice and Experience, 2007.
- **[Foundational]** P. Patarasuk, X. Yuan. *Bandwidth optimal all-reduce algorithms for clusters of workstations.* Journal of Parallel and Distributed Computing, 2009.
- **[Foundational]** J. N. Tsitsiklis, Z.-Q. Luo. *Communication complexity of convex optimization.* Journal of Complexity, 1987.
- **[Theory]** Y. Arjevani, O. Shamir. *Communication complexity of distributed convex learning and optimization.* NeurIPS, 2015.
- **[SOTA]** R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, A. Levskaya, J. Heek, K. Xiao, S. Agrawal, J. Dean. *Efficiently scaling transformer inference.* MLSys, 2023. — arXiv:2211.05102
- **[SOTA]** M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, B. Catanzaro. *Megatron-LM: Training multi-billion parameter language models using model parallelism.* 2019. — arXiv:1909.08053
- **[SOTA]** Y. Leviathan, M. Kalman, Y. Matias. *Fast inference from transformers via speculative decoding.* ICML, 2023.
- **[SOTA]** W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, I. Stoica. *Efficient memory management for large language model serving with PagedAttention.* SOSP, 2023.
- **[SOTA]** Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, H. Zhang. *DistServe: Disaggregating prefill and decoding for goodput-optimized LLM serving.* OSDI, 2024.
- **[Survey]** T. Hoefler, T. Schneider, A. Lumsdaine. *Performance modeling for systematic performance tuning.* SC, 2011.

## 10. Worked Example

Llama-3-70B, 8×H100, TP $= 8$, bf16, $B = 32$, $d = 8192$, $L = 80$.

Per all-reduce payload: $32 \times 8192 \times 2 = 512$ KB.
Bytes moved per device per all-reduce (ring): $2 \cdot \frac{7}{8} \cdot 512\,\text{KB} = 896$ KB.
All-reduces per token: $2L = 160$. Total $C_{\text{token}} = 143$ MB per device.

With $\beta^{-1} = 400$ GB/s achieved unidirectional NVLink:
$$C\beta = \frac{143\ \text{MB}}{400\ \text{GB/s}} = 0.36\ \text{ms}.$$

Rounds: $160 \times 2(8-1) = 2240$. With $\alpha = 2\,\mu$s per step this is nominally 4.5 ms — but NCCL pipelines a 512 KB ring, so the measured per-collective latency is closer to $\approx 25\,\mu$s, giving $160 \times 25\,\mu\text{s} = 4.0$ ms.

Weight read: $70\text{B} \times 2\,\text{B} / 8 = 17.5$ GB per device at 3.35 TB/s HBM3 $= 5.2$ ms.

So per token: $\approx 5.2$ ms compute-bound floor, $\approx 4.4$ ms communication, of which **92% is the round/latency term and 8% is bytes**.

**The obstruction, made visible.** Halving the bits (fp8) removes 0.18 ms — 2% of the token. Halving the rounds removes 2.2 ms — 24%. Yet the only communication lower bound available, $2\frac{P-1}{P}n\beta$, bounds the *8% term* and says nothing about the 92%. And the latency term is not a property of the task: it is a property of the chosen sharding, which the protocol designer may change. A device holding two adjacent layers instead of a slice of each halves $R$ for free at higher memory cost. That is exactly the memory-communication trade-off with no theorem behind it — which is why "is 4.4 ms near-optimal or $10\times$ off?" is currently unanswerable.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*