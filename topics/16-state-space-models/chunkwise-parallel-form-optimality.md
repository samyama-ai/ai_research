---
id: 16-state-space-models/chunkwise-parallel-form-optimality
title: "Chunkwise Parallel Form Optimality"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Chunkwise Parallel Form Optimality

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/chunkwise-parallel-form-optimality` · **Status:** open

## 1. Problem Statement

Every modern linear-attention / state-space layer has three algebraically equivalent evaluation forms: a **recurrent** form (O(1) state per step, sequential, depth $L$), a **parallel/quadratic** form (depth $O(\log L)$, work $O(L^2 d)$), and a **chunkwise** form that splits the sequence into blocks of length $C$, runs the quadratic form inside a block and the recurrent form across blocks. Chunkwise is the form every fast kernel actually ships. The question is whether it is *optimal*, and in what sense.

Three variants, which are routinely conflated:

- **Measurement.** Given a layer, a hardware target, and a sequence length, what is the achievable Pareto frontier over (wall-clock, HBM traffic, peak memory, numerical error)? Chunkwise with tuned $C$ is a point on it. Is it *on the frontier*, or merely the best thing anyone has written a kernel for?
- **Method.** Does a decomposition exist that is neither pure-recurrent, pure-quadratic, nor two-level chunkwise — e.g. a multi-level / hierarchical scan, a Strassen-style bilinear identity, or a form that fuses the state update into the intra-chunk matmul — and beats tuned chunkwise at fixed accuracy?
- **Theory.** Is there a lower bound on work $\times$ depth $\times$ I/O for exactly evaluating a linear recurrence with matrix-valued state, matching the chunkwise upper bound? No such bound is known.

Solving it means either (a) a lower bound that chunkwise provably meets, or (b) a kernel that beats it by a measured margin at fixed loss.

## 2. Formal Setting

Sequence length $L$, head dimension $d$ (key/value dims taken equal), chunk length $C$, $N=L/C$ chunks. The layer family is the **generalized linear recurrence**

$$S_t = M_t\, S_{t-1} + v_t k_t^\top \in \mathbb{R}^{d\times d}, \qquad o_t = S_t^\top q_t,$$

with transition $M_t$ drawn from a structured family: $M_t=\alpha_t I$ (RetNet, Mamba-2/SSD), $M_t=\mathrm{diag}(a_t)$ (GLA, mLSTM), $M_t = \alpha_t(I-\beta_t k_tk_t^\top)$ (Gated DeltaNet), or a product of $n$ such generators (DeltaProduct).

**Chunkwise form.** With $S^{(i)}$ the state entering chunk $i$, outputs are $O_i = \underbrace{\tilde Q_i S^{(i)}}_{\text{inter}} + \underbrace{\big((\tilde Q_i \tilde K_i^\top)\odot \Gamma\big)V_i}_{\text{intra}}$, where $\Gamma$ is the $C\times C$ causal decay mask and tildes denote decay-rescaled projections.

**Quantities, as measured.**

- **Work** $F(C)$ — FLOPs counted from the kernel's own matmul shapes, not from a model:
$$F(C) \;=\; \underbrace{c_1 L C d}_{\text{intra}} \;+\; \underbrace{c_2 L d^2}_{\text{state update + query}} \;+\; \underbrace{c_3 L C^2}_{\text{UT/WY transform, delta-rule only}}$$
with $c_1,c_2$ small integers ($\approx 4$ each). Note $F$ is *increasing* in $C$: chunking trades extra FLOPs for parallelism.
- **Depth** — critical path $= N + O(\log C) = L/C + O(\log C)$ sequential steps.
- **I/O** $Q(C)$ — HBM bytes, measured with `ncu`/`nsys` DRAM counters, not estimated. Dominant term: state traffic $b\,N d^2 = b\,Ld^2/C$ bytes, *decreasing* in $C$; plus $b\,Ld$ for activations.
- **Arithmetic intensity** $I = F(C)/Q(C)$, compared against the machine balance ratio (H100 SXM: $\approx 989$ bf16 TFLOP/s over 3.35 TB/s $\approx 295$ FLOP/byte).
- **Numerical error** $\epsilon = \|O_{\text{kernel}} - O_{\text{fp64 recurrent}}\|_\infty / \|O_{\text{fp64}}\|_\infty$, measured against a float64 sequential reference on the same inputs.
- **Quality** — validation loss in nats/token at matched tokens and matched parameters. Not downstream benchmark scores.

**Assumptions, and which are violated.**
1. *$F$ predicts time.* Violated. Tensor-core utilization is a step function of tile shape; $C=64$ and $C=32$ can cost the same wall-clock despite $2\times$ FLOP difference.
2. *The identity is exact.* Holds for diagonal and rank-1-plus-diagonal $M_t$. Violated for dense $M_t$, where chunk-level transition products need approximation.
3. *State fits in SRAM.* $d^2$ fp32 floats: $d=128 \Rightarrow$ 64 KB, near the 228 KB/SM limit on H100 once tiles are added. Violated at $d=256$.
4. *bf16 matmul error is benign.* Violated for delta-rule variants, where the $C\times C$ triangular inverse in the WY representation amplifies conditioning; production kernels keep the state in fp32 for this reason.

## 3. State of the Art

**Established (ablated, reproduced).**
- Chunkwise form for linear attention: Hua et al., *Transformer Quality in Linear Time* (ICML 2022) — mixed chunk local-quadratic/global-linear.
- RetNet (Sun et al., 2023) named the chunkwise recurrent form for scalar decay.
- **GLA** (Yang, Wang, Shen, Panda, Kim, ICML 2024): chunkwise form for *diagonal, data-dependent* decay with secondary chunking to keep non-matmul FLOPs off the critical path. Trained 340M and 1.3B at ~100B tokens; kernels released in `flash-linear-attention`.
- **SSD / Mamba-2** (Dao & Gu, ICML 2024): recasts selective SSMs as semiseparable matrix multiplication; the block decomposition *is* the chunkwise form. Reported 2–8× faster than Mamba-1's selective scan, and faster than FlashAttention-2 beyond ~2K sequence length.
- **DeltaNet chunkwise** (Yang, Wang, Zhang, Kim, NeurIPS 2024): the key algorithmic result — the delta rule's rank-1 non-diagonal transition admits an exact chunkwise form via the WY / UT-transform representation, cost $O(LCd + LC^2)$. This converted a model believed inherently sequential into a matmul-bound one.

**Claimed but not independently ablated.**
- Tiled Flash Linear Attention (Beck et al., 2025) adds a second level of tiling *within* the chunk for mLSTM; the speedups are reported on the authors' kernels against their own baselines.
- Vendor-reported throughput for hybrid production models (MiniMax-01 lightning attention; Kimi Linear/KDA, 2025) exists **only as benchmark numbers** — no controlled study isolating the decomposition from the model change. *(frontier — verify)*

**Not established at all:** any lower bound. Nothing in the literature says chunkwise cannot be beaten.

## 4. What Is Known

- **Chunk size is empirically flat, not sharp.** Production kernels converge on $C \in \{64, 128\}$ across GLA, Mamba-2, DeltaNet, and mLSTM — chosen by autotuning, and reported as a plateau rather than a peak. The FLOP model predicts the optimum near $C \approx d$, which matches $d=128$ heads only by coincidence of two unrelated constants.
- **Exactness.** For diagonal $M_t$ and for the delta rule, the chunkwise output equals the recurrent output up to floating-point error; there is no accuracy/speed trade to tune. Confirmed by the algebraic derivations in the SSD and DeltaNet papers.
- **The crossover with attention is length-dependent.** Mamba-2's SSD kernel is reported to beat FlashAttention-2 above roughly 2K tokens at $d=64$ head dim; below that, quadratic attention wins outright. Measured on A100/H100 in the SSD paper.
- **Chunking does not change expressivity.** The known expressivity ceiling — uniform-$\mathrm{TC}^0$, hence no $S_5$ word problem without depth growth (Merrill, Petty & Sabharwal, ICML 2024) — is a property of the recurrence, not the evaluation order. Chunk size is not a lever on state tracking.
- **Depth/work trade is real but not tight.** Chunkwise sits at depth $L/C$; a Blelloch-style associative scan reaches depth $O(\log L)$ at work $O(L d^3)$ for matrix state — asymptotically worse work, and never competitive in practice for $d\ge 64$.

## 5. What Is Not Known

- **Theoretically open.** No work–depth–I/O lower bound for exact evaluation of $S_t = M_tS_{t-1}+v_tk_t^\top$ over structured $M_t$. Red-blue pebbling arguments give I/O bounds for dense matmul; nobody has instantiated them for the semiseparable structure these layers have. So "chunkwise is optimal" is an unproven folk claim.
- **Empirically open.** Whether a three-level (token / chunk / super-chunk) decomposition beats two-level at $L \ge 128$K. Runnable today on 8×H100; nobody has published the controlled comparison.
- **Methodologically blocked.** "Optimality" has no agreed metric. Papers report TFLOP/s (rewards a form that does *more* FLOPs), tokens/s at one length, or end-to-end training time with the model also changed. There is no shared harness fixing $(L, d, \text{dtype}, \text{hardware})$ and reporting the (time, HBM bytes, $\epsilon$) triple.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus a non-convex, hardware-quantized objective**.

1. *TFLOP/s is the wrong number and is the number reported.* Chunkwise deliberately inflates FLOPs ($c_1LCd$ term) to hit tensor cores. A form that reduces work looks *worse* under the standard metric.
2. *Every new decomposition needs a new hand-written Triton/CUDA kernel.* An unoptimized implementation of a better algorithm loses to a tuned implementation of a worse one, so a negative result is uninterpretable. Cost per candidate: weeks of kernel engineering, not GPU-hours.
3. *No ground truth for "best possible".* Absent a lower bound, one cannot tell a 5% gap from a 5× gap.

## 7. Current Research (as of 2026)

- **Kernel libraries as the de facto venue.** `flash-linear-attention` (Songlin Yang, Yu Zhang and contributors) hosts most published chunkwise kernels; changes land there before they land in papers.
- **Richer transitions.** DeltaProduct (Siems et al., 2025) uses products of Householder generators, raising the per-chunk cost from $O(LC^2)$ to $O(nLC^2)$ — this makes chunk-size choice matter far more, and is where a better decomposition would pay. *(frontier — verify)*
- **Hierarchical tiling.** NXAI/JKU (Beck, Hochreiter et al.) on TFLA; multi-level chunking is the closest active line to the method variant of this problem.
- **Hybrid stacks** (Nvidia, MiniMax, Moonshot) where a chunkwise linear layer is interleaved with full attention — changes the optimal $C$ because the KV-cache and the state now compete for SRAM. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does a three-level chunkwise decomposition beat tuned two-level at long context?

- **Scale.** Single H100 80GB, one Gated DeltaNet layer, $d=128$, 16 heads, batch 1, $L \in \{8\text{K}, 32\text{K}, 128\text{K}, 512\text{K}\}$, bf16 compute / fp32 state.
- **Arms.** (A) *Control*: `flash-linear-attention` chunkwise kernel, $C$ autotuned over $\{16,32,64,128,256\}$, best time reported. (B) *Treatment*: three-level — inner tile $C_0=32$, chunk $C_1=128$, super-chunk $C_2=1024$, with super-chunk states held in registers across chunk iterations. (C) *Numerical control*: fp64 sequential reference for $\epsilon$.
- **The deciding number.** Forward+backward wall-clock at $L=128$K, arm B over arm A, at $\epsilon_B \le \epsilon_A$. **$\ge 1.15\times$ speedup refutes two-level optimality; $\le 1.02\times$ is evidence chunkwise is at a local optimum that is plausibly global.** Report DRAM bytes from `ncu` alongside, so the result is attributable to I/O rather than to occupancy luck.
- **Cost.** ~2 GPU-days of measurement; ~3 engineer-weeks of Triton.

## 9. Key References

- **[Foundational]** Katharopoulos, Vyas, Pappas, Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020. — arXiv:2006.16236
- **[Foundational]** Hua, Dai, Liu, Le. *Transformer Quality in Linear Time.* ICML 2022. — arXiv:2202.10447
- **[Foundational]** Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei. *Retentive Network: A Successor to Transformer for Large Language Models.* 2023. — arXiv:2307.08621
- **[SOTA]** Yang, Wang, Shen, Panda, Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024. — arXiv:2312.06635
- **[SOTA]** Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** Yang, Wang, Zhang, Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024. — arXiv:2406.06484
- **[SOTA]** Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. — arXiv:2412.06464
- **[Related]** Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023. — arXiv:2312.00752
- **[Related]** Beck, Pöppel, Lippe, Hochreiter. *Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels.* 2025.
- **[Related]** Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024. — arXiv:2404.08819
- **[Survey]** Tiezzi, Casoni, Betti, Guidi, Gori, Melacci. *Back to Recurrent Processing at the Crossroad of Transformers and State-Space Models.* Nature Machine Intelligence, 2025.

## 10. Worked Example

GLA-style layer, $d=128$, $L=32{,}768$, one head, bf16 activations, fp32 state. Take $c_1=c_2=4$.

| $C$ | intra $4LCd$ | state $4Ld^2$ | total GFLOP | state traffic $4\cdot(L/C)d^2$ B | intensity FLOP/B |
|---|---|---|---|---|---|
| 32 | 0.54 G | 2.15 G | 2.69 | 268 MB | 10.0 |
| 64 | 1.07 G | 2.15 G | 3.22 | 134 MB | 24.0 |
| 128 | 2.15 G | 2.15 G | 4.30 | 67 MB | 64.2 |
| 256 | 4.29 G | 2.15 G | 6.44 | 34 MB | 192.4 |
| 512 | 8.59 G | 2.15 G | 10.74 | 17 MB | 641.3 |

The obstruction is visible in the last column against H100's machine balance of $\approx 295$ FLOP/byte. Below $C=256$ the kernel is **memory-bound** — the reported 24 FLOP/B at $C=64$ means the GPU is idle most of the time, so the $3.2\times$ FLOP saving versus $C=512$ buys nothing. Above $C=256$ it is compute-bound and the extra intra-chunk work is now paid in full. The crossing sits between $C=256$ and $C=512$, i.e. the FLOP-minimizing choice ($C$ small) and the intensity-maximizing choice ($C$ large) point in opposite directions and the optimum is set by a hardware constant, not by the algorithm.

Now the part nobody has resolved. At $C=256$, total work is $6.44$ GFLOP against an information-theoretic floor of the $2.15$ GFLOP state term — chunkwise is doing **3× the necessary arithmetic** to stay on tensor cores. Any decomposition that recovered even half of that overhead while keeping intensity above 295 would be a $1.5\times$ win. There is no theorem saying such a decomposition does not exist, and no experiment ruling it out. That is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*