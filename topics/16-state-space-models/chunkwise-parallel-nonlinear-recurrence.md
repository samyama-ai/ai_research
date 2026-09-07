---
id: 16-state-space-models/chunkwise-parallel-nonlinear-recurrence
title: "Chunkwise Parallel Form for Nonlinear Recurrences"
topic: 16-state-space-models
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Chunkwise Parallel Form for Nonlinear Recurrences

> **Topic:** State-Space & Recurrent Models · **ID:** `16-state-space-models/chunkwise-parallel-nonlinear-recurrence` · **Status:** open

## 1. Problem Statement

Linear-state recurrences (linear attention, Mamba-2, GLA, DeltaNet) have a *chunkwise parallel form*: split the sequence into chunks of length $C$, compute inside each chunk with dense matmuls, and pass one state between chunks. This is why they train at transformer-like hardware utilization. Nonlinear recurrences — GRU, LSTM, any $h_t = f_\theta(h_{t-1}, x_t)$ with an elementwise nonlinearity inside the state update — have no such form. They train at sequential depth $T$.

The problem: **does a chunkwise parallel form exist for nonlinear state updates, and if so at what error and what cost?**

Three variants, different difficulty:

- **Theory.** Is there an algorithm evaluating $h_{1:T}$ for a general nonlinear $f_\theta$ in depth $o(T)$ and work $O(T \cdot \mathrm{poly}(d))$, exactly? Almost certainly no in the general case (§6), so the real question is *which restricted families of $f$ admit one*.
- **Method.** Given an approximation tolerance $\epsilon$, produce an iterative scheme whose per-iteration inner loop is matmul-dominated and whose iteration count $K$ does not grow with $T$.
- **Measurement.** Existing "parallel nonlinear RNN" results are reported as speedup over a sequential `scan` baseline. That baseline is often not the right control (§3). A defensible metric is model FLOPs utilization (MFU) at fixed loss, not a speedup ratio.

Solving it means: a nonlinear recurrent layer trained end to end at $\geq 30\%$ MFU on an A100/H100-class GPU at $T \geq 8192$, matching the sequential-scan reference loss to within run-to-run noise.

## 2. Formal Setting

Sequence length $T$, state width $d$, input $x_t \in \mathbb{R}^{d_x}$. The recurrence is

$$h_t = f_\theta(h_{t-1}, x_t), \qquad h_0 = 0, \qquad y_t = g_\theta(h_t).$$

**Linear case.** $f_\theta(h,x) = A(x) h + b(x)$ with $A$ diagonal or diagonal-plus-rank-1. Then for a chunk $[iC{+}1, (i{+}1)C]$ the chunk-local map is itself affine, and the intra-chunk computation collapses to a masked $C \times C$ matmul plus a $d \times d$ state carry. Cost:

$$\text{work} = O\!\left(T d C + T d^2\right), \qquad \text{depth} = O(T/C), \qquad \text{matmul fraction} \to 1 \text{ as } C \to d.$$

**Nonlinear case.** Write the whole-sequence residual $F(h_{1:T}) = 0$ with $F_t(h) = h_t - f_\theta(h_{t-1}, x_t)$. Newton's iteration gives

$$h^{(k+1)}_t = f_\theta(h^{(k)}_{t-1}, x_t) + J_t^{(k)}\left(h^{(k+1)}_{t-1} - h^{(k)}_{t-1}\right), \qquad J_t^{(k)} = \tfrac{\partial f_\theta}{\partial h}\Big|_{h^{(k)}_{t-1}},$$

which is a *linear* recurrence in $h^{(k+1)}$ and therefore parallel-scannable. This is the DEER construction (Lim et al., ICLR 2024). Everything hard is in the constants.

**Measured quantities.**

- $K(\epsilon, T)$: iterations until $\max_t \|h^{(k)}_t - h^\star_t\|_\infty / (\|h^\star_t\|_\infty + 10^{-6}) < \epsilon$, with $h^\star$ from float64 sequential evaluation.
- Work: $\Theta(K T d^2)$ for dense $J_t$ vs. $\Theta(KTd)$ for a diagonal approximation.
- Memory: full Jacobians cost $T d^2$ floats — at $T=8192$, $d=512$, fp32 that is $8.6$ TB. This is the binding constraint, not FLOPs.
- $\rho = \sup_t \|J_t\|_2$: the local contraction factor, measured empirically over the training distribution.
- MFU $= \text{model FLOPs} / (\text{wall clock} \times \text{device peak FLOPs})$, counting only the FLOPs a sequential implementation would do — iterations $2..K$ are overhead, not credit.

**Assumptions and their violations.**

| Assumption | Status in practice |
|---|---|
| $f_\theta$ differentiable in $h$ | Violated at ReLU/hard-sigmoid kinks; measure-zero but hit by Newton |
| $\rho < 1$ (contractive state) | Violated by design in state-tracking layers; trained LSTMs sit near $\rho \approx 1$ |
| Newton basin contains $h^{(0)}=0$ | Unverified at any scale; failures are observed as divergence |
| $K$ independent of $T$ | Empirically holds for contractive $f$; **no proof**, and observed to break at $\rho \to 1$ |

## 3. State of the Art

**Theory SOTA.** Blelloch's associative-scan bound (1990) is the ceiling: depth $O(\log T)$ iff the update composes associatively — true for affine maps, false for general $f$. Martin & Cundy (ICLR 2018) made the linear-RNN case explicit. Merrill & Sabharwal (TACL 2023) and Merrill, Petty & Sabharwal (ICML 2024, "The Illusion of State in State-Space Models") establish the converse direction: models expressible in $\mathsf{TC}^0$ cannot do $\mathsf{NC}^1$-hard state tracking, so *any* fully parallel form implies an expressivity ceiling. Established.

**Systems SOTA (linear).** Mamba-2 / SSD (Dao & Gu, ICML 2024) and GLA (Yang et al., ICML 2024) reach 30–60% MFU with chunk sizes $C \in [64, 256]$. DeltaNet's chunkwise form (Yang, Wang, Zhang, Kim, NeurIPS 2024) extends this to a rank-1 non-diagonal update via a WY-style representation; Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025) adds decay. Established and independently reproduced in `flash-linear-attention`.

**Systems SOTA (nonlinear).** DEER (Lim et al., ICLR 2024) parallelizes GRU/NeuralODE evaluation by Newton + parallel scan. ELK / quasi-DEER (Gonzalez et al., NeurIPS 2024) replaces the dense Jacobian with a diagonal one and adds trust-region damping, cutting memory from $Td^2$ to $Td$ and fixing the divergence DEER shows on long sequences.

**Claimed but unablated.** Reported speedups are against a sequential `scan` in the same framework, at *inference/evaluation*, not against a tuned linear-recurrence layer of equal parameter count at equal loss. No published result shows a nonlinear recurrence trained end to end this way beating a chunkwise linear layer on quality-per-wall-clock. The DEER/ELK numbers are benchmark numbers for a fixed forward evaluation, not training throughput.

## 4. What Is Known

- **Associativity is necessary and sufficient** for $O(\log T)$-depth exact evaluation by scan. Affine $f$ qualifies; $\tanh(Wh+Ux)$ does not.
- **Chunkwise linear forms deliver.** GLA at $C=64$, $d=1024$: within ~10% of FlashAttention-2 throughput at $T=4096$ and faster beyond $T=8192$ (ICML 2024, 1.3B params, A100).
- **Newton converges in few iterations when contractive.** DEER reports convergence in roughly $10$–$30$ iterations for GRUs at $T \sim 10^3$–$10^4$, giving order-of-magnitude to ~$100\times$ wall-clock speedups over sequential evaluation on GPU at small $d$ (tens to a few hundred).
- **Dense Jacobians do not fit.** $T d^2$ memory is the reason quasi-DEER exists; ELK reports diagonal-Jacobian variants scaling to sequence lengths where DEER OOMs or diverges.
- **Expressivity is bought back by structure, not nonlinearity.** DeltaNet with eigenvalues extended to $[-1,1]$ (Grazzi et al., ICLR 2025) and DeltaProduct (Siems et al., 2025) recover parity and modular-arithmetic state tracking while keeping a chunkwise form — at 340M–1.3B scale.

## 5. What Is Not Known

- **Theoretically open.** No lower bound rules out a depth-$o(T)$, work-$\tilde O(T d^2)$ *exact* algorithm for the specific family $h_t = \sigma(Ah_{t-1}+Bx_t)$ with $\sigma$ a fixed saturating nonlinearity. General $f$ is P-complete (§6); this restricted family is not known to be.
- **Theoretically open.** Whether $K(\epsilon,T) = O(1)$ in $T$ for Newton on non-contractive ($\rho \geq 1$) recurrences. No convergence theorem covers the trained regime.
- **Empirically open.** Whether *training* (not evaluation) a nonlinear RNN via chunkwise Newton beats a matched chunkwise linear layer at equal wall-clock, at $\geq 1$B params and $T \geq 8192$. The experiment is runnable today. Nobody has published it.
- **Methodologically blocked.** "Speedup over sequential scan" does not measure the quantity of interest. There is no agreed protocol pairing quality (loss at fixed tokens) with iteration-count overhead, so published numbers are not comparable.

## 6. Why It Is Hard

The obstruction is **P-completeness combined with a memory wall**, not compute cost.

1. *P-completeness.* Evaluating a general nonlinear recurrence simulates an arbitrary sequential circuit; the Circuit Value Problem is P-complete (Ladner 1975), so a depth-polylog algorithm for arbitrary $f$ implies $\mathsf{NC}=\mathsf{P}$. Every workable method must therefore restrict $f$ or accept approximation — and the restriction is exactly what costs expressivity.
2. *Memory, not FLOPs.* The Newton route trades depth for a $T d^2$ Jacobian tensor. At $T=8192,d=512$ that is 8.6 TB in fp32 versus 16 MB for the sequential states. Diagonal approximations fit but reduce Newton to a Picard-like fixed point whose rate degrades as $\rho \to 1$ — precisely the regime where the nonlinearity was doing useful work.
3. *Confounded measurement.* Speedup is reported against the slowest possible baseline. A nonlinear layer needing $K=20$ iterations must be $20\times$ more useful per FLOP than a linear one, and no evaluation currently measures that ratio.

## 7. Current Research (as of 2026)

- **Structured-linear expansion.** MIT/CMU (Songlin Yang, Yoon Kim), NVIDIA: DeltaNet, Gated DeltaNet, DeltaProduct — recover state tracking inside a chunkwise form rather than reintroducing nonlinearity. This is the direction with reproduced wins.
- **Hierarchical chunking.** Log-linear attention (Guo, Yang et al., 2025) — $O(T\log T)$ work, $O(\log T)$ state, chunkwise-compatible. *(frontier — verify)*
- **Fixed-point / implicit sequence models.** Parallelizing implicit and diffusion-style token recurrences via Picard/Jacobi iteration; Schöne et al. (2025) on implicit LMs as RNNs. *(frontier — verify)*
- **Quasi-Newton scans.** Follow-ups to ELK (Gonzalez, Warrington et al.) on damping, trust regions, and low-rank Jacobian sketches. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** 340M-parameter decoder-only LM, 15B tokens of FineWeb-Edu, $T = 8192$, $d = 1024$, 8×H100.

**Arms.**
1. *Treatment:* GRU-style nonlinear recurrent layer trained with chunkwise quasi-Newton ($C=128$, diagonal Jacobian, damping, $K$ capped at 8, backward through the fixed point via implicit differentiation).
2. *Control A:* Gated DeltaNet, identical parameter count, block layout, optimizer, data order.
3. *Control B:* the same nonlinear layer trained by sequential scan (correctness reference — gives $h^\star$ and the reference loss).

**Deciding number.** Validation loss at **fixed wall-clock budget** (24 GPU-hours), treatment minus Control A. If treatment $\leq$ Control A $- 0.02$ nats, nonlinear chunkwise parallelism is worth its iteration overhead. If $\geq +0.02$ nats, the linear-with-structure route wins and the problem is answered negatively at this scale.

**Secondary reporting (required for the result to be interpretable).** Median and 99th-percentile $K$ per step; measured $\rho$; treatment-vs-Control-B loss gap at equal *tokens*, which must be $< 0.01$ nats or the approximation is changing the model, not accelerating it.

## 9. Key References

- **[Foundational]** Guy E. Blelloch. *Prefix Sums and Their Applications.* CMU-CS-90-190, 1990.
- **[Foundational]** Richard E. Ladner. *The Circuit Value Problem is Log Space Complete for P.* SIGACT News, 1975.
- **[Foundational]** Eric Martin, Chris Cundy. *Parallelizing Linear Recurrent Neural Nets Over Sequence Length.* ICLR 2018. — arXiv:1709.04057
- **[SOTA]** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. — arXiv:2405.21060
- **[SOTA]** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training.* ICML 2024. — arXiv:2312.06635
- **[SOTA]** Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024.
- **[SOTA]** Songlin Yang, Jan Kautz, Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025.
- **[SOTA]** Yi Heng Lim, Qi Zhu, Joshua Selfridge, Muhammad Firmansyah Kasim. *Parallelizing Non-Linear Sequential Models over the Sequence Length.* ICLR 2024.
- **[SOTA]** Xavier Gonzalez, Andrew Warrington, Jimmy T.H. Smith, Scott W. Linderman. *Towards Scalable and Stable Parallelization of Nonlinear RNNs.* NeurIPS 2024.
- **[Theory]** William Merrill, Ashish Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL, 2023.
- **[Theory]** William Merrill, Jackson Petty, Ashish Sabharwal. *The Illusion of State in State-Space Models.* ICML 2024.
- **[Theory]** Riccardo Grazzi, Julien Siems, Simon Schrodi, Thomas Brox, Frank Hutter. *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025.

## 10. Worked Example

Take $h_t = \tanh(W h_{t-1} + u_t)$, $d = 512$, $T = 8192$, chunk $C = 128$ (64 chunks).

**Sequential reference.** $2d^2 = 524{,}288$ FLOPs per step $\Rightarrow 4.3$ GFLOP per sequence. On an H100 (≈990 TFLOP/s bf16) this is $4\,\mu$s of arithmetic — but at depth 8192 with a ~5 µs kernel-launch-bound step, wall clock is ~41 ms. Utilization: $\approx 0.001\%$. That gap is the whole motivation.

**Dense Newton.** One iteration needs $J_t = \mathrm{diag}(1 - h_t^2)W$ for every $t$: $8192 \times 512^2 \times 4$ B $= 8.6$ GB per iteration per sequence, and the parallel scan over $512\times512$ matrices costs $T d^3 = 1.1$ PFLOP — $250\times$ more arithmetic than the sequential version *per iteration*. Ten iterations: $2500\times$. It is faster in wall clock only because it is depth-$\log T$; it is catastrophically wasteful in work. At batch size 8 the Jacobians alone exceed 68 GB and OOM.

**Diagonal quasi-Newton (the practical arm).** Approximate $J_t \approx \mathrm{diag}(1-h_t^2)\,\mathrm{diag}(W)$. Memory drops to $Td = 4096$ floats/sequence, work per iteration to $\approx 4.3$ GFLOP. With $K=8$ that is $34$ GFLOP — $8\times$ the sequential work, but arranged as 64 chunk-parallel matmuls at depth $\log_2(8192)=13$ per iteration. Plausible wall clock: ~2 ms, a $20\times$ speedup.

**Where it breaks.** Convergence rate is governed by $\rho = \max_t \|(1-h_t^2)W\|_2$. A trained LSTM/GRU language model sits near $\rho \approx 0.95$–$1.0$ (it must, to carry information across thousands of tokens). The residual after $K$ diagonal-Jacobi iterations decays like $\rho^K$ only for $\rho<1$; at $\rho = 0.98$, reaching $\epsilon=10^{-4}$ needs $K \approx \ln(10^{-4})/\ln(0.98) \approx 456$ iterations, not 8. At $K=8$ the residual is $0.98^8 = 0.85$ — the "parallel" layer is computing a different function.

**The obstruction, made visible.** The method is fast exactly when $\rho \ll 1$, and $\rho \ll 1$ means the state forgets in $O(1/(1-\rho))$ steps — i.e. it is a short-memory layer that a chunkwise *linear* recurrence already handles at higher MFU and with exact semantics. The regime where nonlinearity is needed is the regime where the parallelization fails. No published experiment reports $\rho$ for a trained model, which is why §8 makes it a required secondary measurement.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*