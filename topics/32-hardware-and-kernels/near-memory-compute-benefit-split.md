---
id: 32-hardware-and-kernels/near-memory-compute-benefit-split
title: "Near-Memory Compute Benefit for Attention Versus Feedforward"
topic: 32-hardware-and-kernels
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Near-Memory Compute Benefit for Attention Versus Feedforward

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/near-memory-compute-benefit-split` · **Status:** open

## 1. Problem Statement

Processing-in-memory (PIM) and near-memory compute (NMC) put MAC units inside or beside DRAM banks, trading low peak FLOPs for bandwidth that never crosses the memory bus. Every vendor demo and every academic PIM-for-LLM paper reports an end-to-end speedup. None of them answers the question a system architect actually has:

**How much of the benefit comes from attention, and how much from the feedforward/projection weights — at a fixed batch size, fixed KV precision, and fixed host?**

The question splits three ways.

- **Measurement variant.** Given a real PIM device and a real transformer decode workload, produce a per-operator attribution of the observed speedup that survives a serialization control. Currently unresolved: PIM and host run concurrently, so "time spent in attention" is not additive.
- **Method variant.** Given a model, a context length distribution, and a device with host bandwidth $B_h$ and PIM bandwidth $B_p$, decide which operators to place on PIM to minimise per-token latency at a target throughput. Solving it means a placement rule that beats both all-host and all-PIM on held-out workloads.
- **Theory variant.** Characterise the batch size $B^\*$ at which the PIM benefit for feedforward overtakes the benefit for attention, as a function of GQA group size, KV quantization, and $B_p/B_h$. Prove that the crossover exists and is unique under a roofline model, or exhibit the regime where it is not.

Solving it means: a decision rule, validated on at least two PIM substrates, that predicts the sign and rough magnitude of $\rho_\text{attn} - \rho_\text{ffn}$ without running the workload.

## 2. Formal Setting

Decode step, model with $L$ layers, hidden $d$, $n_q$ query heads, $n_{kv}$ KV heads, head dim $d_h$, GQA group $g = n_q/n_{kv}$, batch $B$, context length $S$, KV element width $b_{kv}$ bytes, weight width $b_w$ bytes.

**KV bytes touched per decode step:**
$$M_\text{attn} = 2\,B\,S\,L\,n_{kv}\,d_h\,b_{kv}$$

**Attention FLOPs** (scores plus value-weighted sum, both GEMV):
$$F_\text{attn} = 4\,B\,S\,L\,n_q\,d_h$$

Arithmetic intensity, the quantity that decides everything:
$$I_\text{attn} = \frac{F_\text{attn}}{M_\text{attn}} = \frac{2g}{b_{kv}} \quad\text{FLOP/byte, independent of } B \text{ and } S.$$

**Feedforward + projections:** weight bytes $M_\text{ffn} = P b_w$ for $P$ non-embedding parameters, FLOPs $F_\text{ffn} = 2BP$, so
$$I_\text{ffn} = \frac{2B}{b_w},$$
linear in batch because weights are shared across the batch.

**Measured, not nominal, bandwidth.** $B_h$ is the sustained streaming read rate of the host memory system under the actual access pattern (paged KV, 16–64 KB pages), measured by a pointer-chasing-free strided read microbenchmark, not the datasheet pin rate. $B_p$ is the sustained *aggregate* PIM read rate measured by running the device's own MAC primitive over a resident array larger than all row buffers, so refresh and row-activation overhead are included. Vendor "internal bandwidth" figures are bank-array peaks and are not $B_p$.

**Per-operator benefit ratio,** the object of the problem:
$$\rho_{o} = \frac{T_o^\text{host}}{T_o^\text{pim}}, \qquad T_o^{x} = \max\!\left(\frac{M_o}{B_x},\ \frac{F_o}{C_x}\right) + \tau_o^{x},$$
with $C_x$ the compute peak of substrate $x$ and $\tau_o^x$ the fixed cost of moving operands and results across the boundary (for PIM: activation broadcast in, partial sums out; for host: nothing). The decision predicate is the sign of $\Delta = \rho_\text{attn} - \rho_\text{ffn}$ and the crossover $B^\*$ solving $\Delta(B^\*)=0$.

**Assumptions, and which are violated.**

1. *Roofline additivity* — operator times sum. Violated: real PIM systems overlap host and PIM execution (NeuPIMs' sub-batch interleaving exists precisely to do this), so end-to-end time is not $\sum_o T_o$.
2. *KV cache resident in PIM-addressable memory.* Violated whenever the cache exceeds device capacity; then $\tau$ includes host staging and dominates.
3. *Uniform $S$ across the batch.* Violated in every serving system; continuous batching mixes prefill chunks (compute-bound, $I \gg I_\text{attn}$) into the same step.
4. *Weights static.* Violated by MoE, where per-step expert weight traffic scales with $B$ until experts saturate, making $I_\text{ffn}$ sublinear.
5. *PIM MACs are usable at full precision.* Violated: commercial PIM units are FP16/BF16 MAC-only; INT4 weight-only quantization — the standard serving configuration — has no PIM datapath, so the host arm and PIM arm are not running the same numerics.

## 3. State of the Art

**Silicon (established).** Samsung's FIMDRAM/HBM-PIM (ISSCC 2021; ISCA 2021 for the stack) places 16 programmable compute units per die, 1.2 TFLOPS FP16 per HBM2 stack. SK hynix's GDDR6-AiM (ISSCC 2022) puts a MAC per bank, 1 TFLOPS-class per die, plus activation functions. UPMEM ships 2,560 general-purpose DPUs in DDR4 DIMMs. These are real parts with published measurements.

**Systems results (claimed; ablation partial).** AttAcc (ASPLOS 2024) argues explicitly that attention, not FFN, is the PIM-worthy operator, and reports up to $2.81\times$ speedup and $2.67\times$ energy reduction over a DGX-A100 for GPT-3 175B — from a simulator, against a baseline whose attention kernel is not FlashDecoding-class. NeuPIMs (ASPLOS 2024) reports $2.4\times$ throughput over NPU-only and $1.6\times$ over a naive NPU+PIM composition, with the gain attributed to dual row-buffers plus sub-batch interleaving; the two contributions are ablated against each other but not against operator placement. Duplex (MICRO 2024) targets the low-intensity phases including MoE. CENT (ASPLOS 2025) reports $2.3\times$ throughput and energy efficiency over A100 for Llama-2-70B on a GPU-free CXL-PNM system.

**Benchmark-number-only.** Samsung's "$2.5\times$ system performance, $\sim$60% energy reduction" for HBM-PIM comes from RNN-T speech recognition, not transformer decode, and is a vendor benchmark figure with no public per-operator breakdown. Treat every LLM-scale PIM speedup above as a simulator number unless the paper names the silicon it ran on; only UPMEM results (PrIM, IEEE Access 2022) are measured on shipping PIM hardware, and PrIM's headline finding is the opposite of encouraging — host-to-DPU transfer dominates for most kernels.

**What is genuinely established:** the arithmetic-intensity asymmetry (§4.1) and the silicon capabilities. **What is claimed but unablated:** every per-operator attribution of end-to-end speedup.

## 4. What Is Known

- **The intensity gap is exact, not empirical.** For Llama-2-70B ($g=8$, $b_{kv}=2$), $I_\text{attn} = 8$ FLOP/byte at every batch and context. At $B=64$, $I_\text{ffn} = 64$. The A100-80GB ridge point is $312\ \text{TFLOP/s} / 2.039\ \text{TB/s} \approx 153$ FLOP/byte; H100 SXM is $\approx 295$. Both operators are memory-bound at $B=64$; attention is $8\times$ deeper into the bound.
- **GQA moved the target.** MHA gives $I_\text{attn}=1$ FLOP/byte at fp16; GQA-8 gives 8; DeepSeek-V2-style MLA (2024) compresses the cache further and raises it again. A PIM benefit case computed against MHA overstates the 2026 case by up to $8\times$.
- **KV capacity, at scale.** Llama-2-70B: $2\times 8\times 128\times 2 = 4096$ B per token per layer, $\times 80$ layers $= 320$ KB/token. At $S=4096$, $B=64$: 82 GB — larger than one A100's HBM and larger than the KV-resident capacity of a single HBM-PIM stack.
- **Real PIM hardware is transfer-bound.** PrIM (Gómez-Luna et al., IEEE Access 2022), on 2,560 UPMEM DPUs: kernels with low operational intensity and little inter-DPU communication scale well; anything requiring host round-trips does not. This is the only large body of *measured* PIM data at any scale.
- **Bank-level MAC throughput is real but narrow.** GDDR6-AiM's 1 TFLOPS is FP16 MAC-only with a fixed dataflow; general GEMM is not expressible.

## 5. What Is Not Known

- **Empirically open.** No published experiment holds hardware, batch, context, and KV precision fixed while swapping *only* which operator runs on PIM, on real silicon. The experiment is runnable today on an SK hynix AiMX-class card or a UPMEM DIMM. Nobody has run it at 8B–70B scale.
- **Empirically open.** Whether the PIM attention advantage survives an FP8/INT4 KV cache. Quantizing KV to 4 bits raises $I_\text{attn}$ to $4g$ and cuts $M_\text{attn}$ by $4\times$ — but commercial PIM MACs are FP16, so the PIM arm cannot use the quantized format. The comparison has never been run with both arms at matched numerics.
- **Theoretically open.** Whether $B^\*$ is unique. Under strict roofline with fixed $\tau$, $\Delta(B)$ is monotone and the crossover is unique; with capacity-dependent $\tau$ (spilling KV to host past a threshold batch) $\Delta$ can change sign twice. No proof of either regime's boundary exists.
- **Methodologically blocked.** Per-operator attribution under concurrent host/PIM execution. When PIM attention overlaps host FFN, the marginal contribution of each placement is not a partition of wall-clock time, and no PIM paper defines the counterfactual it is measuring against.

## 6. Why It Is Hard

**Non-identifiability of the speedup source.** A PIM arm differs from its host baseline in at least four ways simultaneously: (i) bandwidth amplification $B_p/B_h$; (ii) avoided host-bus traffic; (iii) added memory capacity, which permits a larger batch, which changes $I_\text{ffn}$; (iv) a different kernel implementation, usually a less-tuned baseline. A single end-to-end throughput ratio is one equation in four unknowns. Published results report the one equation.

**Compounding this: the evaluation does not measure what it names.** "Speedup on LLM inference" for a system whose PIM capacity forces a different batch size than the baseline is measuring a capacity effect and labelling it a bandwidth effect. Distinguishing them requires either capacity-matched arms (throw away PIM memory) or batch-matched arms (throw away PIM throughput) — and no paper reports both.

**Simulator dependence.** The LLM-scale results are all simulated, and the DRAM timing models differ across papers, so cross-paper comparison of $\rho$ is uncalibrated.

## 7. Current Research (as of 2026)

- **Heterogeneous NPU+PIM decomposition** — Seoul National University and KAIST groups behind AttAcc, NeuPIMs, and Duplex continue on attention-offload scheduling and MoE-aware placement.
- **CXL-attached near-memory** — CENT (ASPLOS 2025) pushes the GPU-free end of the design space; the CXL type-2 memory-expander path is where commercial NMC is most likely to ship *(frontier — verify)*.
- **Vendor productisation** — SK hynix AiMX accelerator cards and Samsung LPDDR-PIM demos target LLM decode explicitly; publicly available per-operator breakdowns remain absent *(frontier — verify)*.
- **Algorithmic erosion of the premise** — MLA, KV quantization, and sparse/selective attention all cut $M_\text{attn}$, shrinking the attention-side prize that motivates PIM in the first place.

## 8. Concrete Next Experiment

**Scale.** Llama-3-8B (GQA, $n_{kv}=8$, $g=4$) decode on one SK hynix AiMX-class card or one UPMEM 2,560-DPU node with a host GPU, $S \in \{4\text{k}, 32\text{k}\}$, $B \in \{1, 8, 32, 128, 256\}$, KV cache resident on the near-memory device at all points (cap $B$ so it never spills — spilling is a separate experiment).

**Four arms, one hardware configuration, identical numerics (BF16 weights *and* BF16 KV in every arm):**

| Arm | Attention | FFN + projections |
|---|---|---|
| A0 (control) | host | host |
| A1 | near-memory | host |
| A2 | host | near-memory |
| A3 | near-memory | near-memory |

**Control arm requirement.** A0 must use a FlashDecoding-class attention kernel and paged KV. A baseline with a naive attention kernel invalidates the whole measurement.

**Serialization control.** Run every arm twice: once with host/device overlap enabled, once with a barrier forcing serial execution. The serial run gives an additive time decomposition; the difference between the two quantifies how much of the reported gain is overlap rather than placement.

**The deciding number.** The crossover batch
$$B^\* = \arg\min_B \left| \rho_\text{attn}(B) - \rho_\text{ffn}(B) \right|,$$
with $\rho_\text{attn} = T_\text{A0}/T_\text{A1}$ and $\rho_\text{ffn} = T_\text{A0}/T_\text{A2}$ from the *serialized* runs. If $B^\* $ falls below the batch sizes real serving systems use (32–256), near-memory compute is an FFN technology and the attention framing is wrong. If $B^\*$ exceeds 256 or does not exist in range, the attention framing survives. One number, one plot, decides it.

## 9. Key References

- **[Foundational]** Kwon, Y.-C. et al. *A 20nm 6GB Function-In-Memory DRAM, Based on HBM2 with a 1.2TFLOPS Programmable Computing Unit Using Bank-Level Parallelism, for Machine Learning Applications.* ISSCC, 2021.
- **[Foundational]** Lee, S. et al. *Hardware Architecture and Software Stack for PIM Based on Commercial DRAM Technology.* ISCA, 2021.
- **[Foundational]** Lee, S. et al. *A 1ynm 1.25V 8Gb, 16Gb/s/pin GDDR6-based Accelerator-in-Memory Supporting 1TFLOPS MAC Operation and Various Activation Functions for Deep-Learning Applications.* ISSCC, 2022.
- **[SOTA]** Park, J. et al. *AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference.* ASPLOS, 2024.
- **[SOTA]** Heo, G. et al. *NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing.* ASPLOS, 2024.
- **[SOTA]** Yun, S. et al. *Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching.* MICRO, 2024.
- **[SOTA]** Gu, Y. et al. *PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference.* ASPLOS, 2025.
- **[Empirical baseline]** Gómez-Luna, J. et al. *Benchmarking a New Paradigm: Experimental Analysis and Characterization of a Real Processing-in-Memory System.* IEEE Access, 2022.
- **[Related]** Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP, 2023.
- **[Survey]** Mutlu, O., Ghose, S., Gómez-Luna, J., Ausavarungnirun, R. *A Modern Primer on Processing in Memory.* In *Emerging Computing: From Devices to Systems*, Springer, 2022.

## 10. Worked Example

Llama-2-70B, $B=64$, $S=4096$, BF16 throughout, host = A100-80GB ($B_h \approx 1.7$ TB/s sustained, $C_h = 312$ TFLOP/s).

Per decode step:

```
attention:  M = 82 GB    F = 0.69 TFLOP   I = 8.4 FLOP/B
ffn+proj:   M = 138 GB   F = 8.8  TFLOP   I = 64  FLOP/B
host time:  attn = 82/1700   = 48 ms
            ffn  = 138/1700  = 81 ms      (both bandwidth-bound)
```

Now add a near-memory device with $B_p = 4B_h$ and MAC peak $C_p = 4$ TFLOP/s.

```
attn on PIM: max(82/6800, 0.69/4000)  = 12.1 ms  -> rho_attn = 4.0
ffn  on PIM: max(138/6800, 8.8/4000)  = 20.3 ms  -> rho_ffn  = 4.0
```

Identical. At this batch, both operators are bandwidth-bound on both substrates, so both get exactly the bandwidth ratio and $\Delta = 0$: **the attention-versus-FFN distinction does not exist in the roofline.** It only appears when the PIM MAC peak binds. FFN hits $C_p$ when $2BP/C_p > Pb_w/B_p$, i.e. $B > C_p b_w / (2 B_p) = 4000 \times 2 / (2 \times 6800) \approx 0.6$ — already exceeded, so the max above should have been checked: $8.8/4000 = 2.2$ ms $< 20.3$ ms, so FFN is still bandwidth-bound. It binds only for $B \gtrsim 590$.

The obstruction is now visible. Under a clean roofline the answer is "no difference below $B\approx590$" — yet AttAcc reports $2.81\times$ and attributes it to attention. That gap is entirely $\tau$, capacity, and baseline-kernel quality, none of which the roofline models and none of which the published papers separate. Attention's real PIM advantage, if it exists, lives in the terms nobody has measured: the KV cache is *written* every step (PIM handles this; FFN weights are read-only and cacheable), and attention operands are per-sequence, so they never amortise across the batch the way a weight tile does. Those are the terms the §8 experiment isolates — and the arithmetic above shows that without isolating them, a headline speedup number carries no information about which operator earned it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*