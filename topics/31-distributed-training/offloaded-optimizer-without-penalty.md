---
id: 31-distributed-training/offloaded-optimizer-without-penalty
title: "Offloaded Optimizer States Without Step-Time Penalty"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Offloaded Optimizer States Without Step-Time Penalty

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/offloaded-optimizer-without-penalty` · **Status:** open

## 1. Problem Statement

Adam keeps two moments and an FP32 master copy per parameter — 12–16 bytes/param against 2 bytes for a BF16 weight. Moving that state off the accelerator to host DRAM or NVMe frees the majority of HBM. The cost is that the update now crosses a link two orders of magnitude slower than HBM and runs on a device two orders of magnitude slower at FLOPs.

**The problem:** design an offloaded optimizer whose per-step wall-clock is within $\epsilon$ (say 2%) of the co-located baseline *and* whose loss-vs-tokens curve is indistinguishable from it, in the regime that matters — small per-GPU micro-batch, few or no gradient-accumulation steps, hundreds to thousands of GPUs.

Three variants, different difficulty:

- **Measurement.** Does a given offload system actually pay zero penalty? Reported MFU on a single GPU with 32 accumulation steps does not answer this. No standard harness reports exposed-offload time separately from compute at cluster scale.
- **Method.** Build the system. The candidate levers are overlap (hide transfer under backward), compression (quantize the state), staleness (delayed parameter update), and hardware (coherent CPU–GPU links).
- **Theory.** Staleness is the only lever that buys unbounded latency tolerance. How much does a $\tau$-step delayed update cost on the loss curve of a non-convex LLM pretraining run? Open.

Solving it means: a system with $\rho \le 0.02$ (below) at $\ge 512$ GPUs with $k \le 2$ accumulation steps, and a matched-token loss gap inside seed noise.

## 2. Formal Setting

Let $N$ be parameters, $P$ data-parallel ranks, ZeRO-3 sharding so each rank owns $N/P$. Let $k$ be gradient-accumulation micro-steps per optimizer step, $T$ tokens per micro-batch per GPU.

**Compute time per micro-step**, as measured by CUDA events around forward+backward:
$$T_{\text{fb}} = \frac{6NT}{\eta\,F_{\text{peak}}}$$
with $\eta$ the achieved fraction of peak (measured: total FLOPs / wall-clock / $F_{\text{peak}}$; recompute FLOPs counted if activation checkpointing is on).

**Transfer time.** With $g$ bytes/param of gradient sent down and $p$ bytes/param of updated weight sent up:
$$T_{\text{xfer}} = \frac{(g+p)\,N/P}{B_{\text{link}}}$$
$B_{\text{link}}$ measured by a pinned-memory `cudaMemcpyAsync` microbenchmark on the *populated* node, all GPUs transferring at once — not the vendor number.

**Host update time**, bandwidth-bound not FLOP-bound. FP32 Adam touches $m$, $v$, master $w$, grad (read) and $m$, $v$, $w$ (write) plus the cast-down output: $c \approx 30$ bytes/param of DRAM traffic.
$$T_{\text{host}} = \frac{c\,N/P}{B_{\text{mem}}/P_{\text{gpu/socket}}}$$

**Exposed overhead.** Only $(k-1)/k$ of the accumulation window is available to hide gradient transfer, and the parameter upload must complete before the next forward:
$$T_{\text{exposed}} = \max\!\left(0,\; T_{\text{xfer}} + T_{\text{host}} - W\right),\qquad W = \frac{k-1}{k}\,k\,T_{\text{fb}}$$

**The objective.** Overhead ratio and loss gap:
$$\rho = \frac{k T_{\text{fb}} + T_{\text{exposed}}}{k T_{\text{fb}}} - 1, \qquad \Delta(t) = L_{\text{off}}(t) - L_{\text{base}}(t)$$
Solved iff $\rho \le \epsilon$ and $|\Delta(t)| \le \delta$ for all $t$ in the token budget, with $\delta$ set by the seed-to-seed standard deviation of the baseline (typically 0.005–0.01 nats at 1B–7B scale).

**Assumptions, and which are violated.**
- *Additive, hideable transfer.* Violated: on non-NVLink nodes the offload DMA and the ZeRO all-gather/reduce-scatter share the same PCIe root complex; they serialize, so $T_{\text{xfer}}$ is not free even inside $W$.
- *$B_{\text{link}}$ is per-GPU.* Violated: 4 or 8 GPUs typically share one socket's root complex and DRAM controllers, so effective per-GPU bandwidth is $B/4$ or $B/8$, not $B$.
- *Host CPU is idle.* Violated: the dataloader, tokenizer and NCCL progress threads contend for the same cores and memory channels.
- *$k$ is large.* Violated at scale: global batch is capped by the critical batch size (McCandlish et al., 2018), so $k \to 1$ as $P$ grows.
- *Update semantics unchanged.* Violated by every system that uses delayed parameter update.

## 3. State of the Art

**Systems, established.** ZeRO-Offload (Ren et al., USENIX ATC 2021) fixed the partition: FP32 optimizer state and the Adam step on CPU, forward/backward on GPU, with a one-step delayed parameter update (DPU) option. ZeRO-Infinity (Rajbhandari et al., SC 2021) extended the hierarchy to NVMe and added bandwidth-centric partitioning, so aggregate offload bandwidth grows with $P$ rather than being pinned to one link. PatrickStar (Fang et al., IEEE TPDS 2023) replaced static partitioning with chunk-based dynamic placement. STRONGHOLD (Sun et al., SC 2022) added a working-window prefetcher. These are reproduced and in production (DeepSpeed, Colossal-AI).

**Established, orthogonal:** 8-bit block-wise Adam (Dettmers et al., ICLR 2022) cuts optimizer bytes 4× with matched quality; GaLore (Zhao et al., ICML 2024) projects gradients to low rank; Adam-mini (Zhang et al., 2024) collapses per-parameter second moments to block-wise ones. Each shrinks $g$, $p$ or $c$ directly and so shrinks $T_{\text{xfer}}$ and $T_{\text{host}}$ — they are the cheapest available progress on this problem and are underexploited *in combination with* offload.

**Claimed but unablated.** Fuyou (Liao et al., 2024) reports 156 TFLOP/s fine-tuning GPT-3 175B on a single RTX 4090 versus ~45 TFLOP/s for ZeRO-Infinity — a single-paper benchmark number on one consumer GPU with very large accumulation, not an independently reproduced cluster result. Smart-Infinity (Jang et al., HPCA 2024) moves the update into computational-storage devices; the speedup is real on their prototype but the hardware is not deployed anywhere at scale. ZeRO-Offload's own DPU convergence evidence is GPT-2-scale and one delay step; it is not an ablation over $\tau$, model size, or learning-rate schedule.

**Nobody has published** a matched-loss, matched-token comparison of offloaded versus co-located Adam at $\ge 512$ GPUs with $k \le 2$. That absence is the state of the art.

## 4. What Is Known

- ZeRO-Offload trains 13B on one 32 GB V100 at ~40 TFLOP/s/GPU, against a PyTorch ceiling of 1.4B at ~30 TFLOP/s (ATC 2021; single DGX-2, and up to 128 GPUs for 70B).
- ZeRO-Infinity sustains >25 TFLOP/s/GPU on 512 V100s for a 1T-parameter model — roughly 20% of the 125 TFLOP/s FP16 peak (SC 2021).
- 8-bit Adam matches FP32 Adam perplexity on GPT-2 up to 1.5B and on GLUE/ImageNet while removing 75% of optimizer memory (ICLR 2022).
- Roofline arithmetic, not measured: PCIe Gen4 ×16 delivers ~25 GB/s of a 32 GB/s peak; Gen5 ~50 of 64. An 8-channel DDR5-4800 socket peaks at ~307 GB/s, ~200 achievable. NVLink-C2C on GH200 gives ~450 GB/s each direction — an 18× jump that changes the sign of the whole calculation and has never been evaluated as a *training-quality* question, only as a throughput one.
- Delayed-gradient SGD has $O(1/\sqrt{n})$ rates with delay-dependent constants for smooth convex objectives (Agarwal & Duchi, NeurIPS 2011; Stich & Karimireddy, JMLR 2020). PipeDream (SOSP 2019) showed weight staleness is empirically tolerable at ImageNet/translation scale.

## 5. What Is Not Known

- **Empirically open.** Does $\rho \le 0.02$ hold at $P \ge 512$, $k \le 2$? Runnable today on any H100 or GH200 cluster. Nobody has published it.
- **Empirically open.** The loss cost of $\tau$-step DPU as a function of $\tau$, $N$, and position in the LR schedule. Runnable at 1B; the interesting question is whether the penalty grows or shrinks with $N$, which needs 1B/3B/7B.
- **Theoretically open.** No convergence result for delayed *parameter* updates (as opposed to delayed gradients) under Adam-style adaptivity on non-convex objectives with the delay interacting with a warmup–cosine schedule.
- **Methodologically blocked.** There is no accepted metric that separates exposed offload time from compute. MFU folds them together; profiler traces attribute host-side Adam to gaps, not to a line item. Two systems with identical $\rho$ can report MFU differing by 10 points because of unrelated kernel choices.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure what it names**, sitting on top of a hard scaling limit.

Offload papers report throughput at the operating point where offload is free: single node, tiny GPU, $k = 16$–$32$, so $W \gg T_{\text{xfer}} + T_{\text{host}}$ and $T_{\text{exposed}} = 0$ by construction. Frontier training sits at the opposite point. Global batch cannot exceed the critical batch size without wasting tokens, so as $P$ grows, $k$ falls to 1 and $W$ falls to 0. Every reported "no penalty" result is measured in the regime where the question is trivial, and the published number is not wrong — it just does not name the regime it holds in.

Secondary: the loss-gap half needs matched-token runs at $\ge 1$B to have any power, which is $\sim 10^{21}$ FLOPs per arm, and the effect size ($\delta \approx 0.01$ nats) is the same order as seed noise, so it needs multiple seeds per arm.

## 7. Current Research (as of 2026)

- **Coherent-memory offload.** GH200/GB200 NVLink-C2C makes host DRAM a ~450 GB/s tier. DeepSpeed and Colossal-AI have Grace-aware paths; the open question is whether the update should move to the Grace cores at all once the link is fast enough to just page state back. *(frontier — verify)*
- **State compression as offload enabler.** Stacking 8-bit or low-rank optimizer state under an offload engine to shrink $g$, $p$, $c$ simultaneously. Reported ad hoc, not systematically ablated.
- **Near-storage and near-memory update** (Smart-Infinity line, HPCA/ISCA communities). Prototype hardware.
- **Staleness-tolerant schedules.** Interest in $\tau > 1$ DPU with LR compensation; no published pretraining-scale ablation. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** 1.3B decoder-only, 64 H100s, ZeRO-3, sequence 4096, micro-batch 4 seqs/GPU, $k=1$ (global batch ~1.05M tokens — at or under critical batch for this scale), 100B tokens, 3 seeds per arm.

**Arms.**
1. *Control:* co-located FP32 Adam, all state in HBM.
2. Host-DRAM offload, synchronous update ($\tau=0$).
3. Host-DRAM offload, $\tau=1$ DPU.
4. Host-DRAM offload, $\tau=4$ DPU.

**Instrumentation.** Report $T_{\text{exposed}}$ as a first-class number via CUDA-event brackets on the copy and host-update streams, not inferred from MFU.

**The deciding number.** The pair $(\rho, \Delta)$ at 100B tokens. The problem moves if any offload arm shows $\rho \le 0.02$ with $|\Delta| \le 2\sigma_{\text{seed}}$. Prediction, from Section 10's arithmetic: arm 2 gives $\rho \approx 0.5$; arm 3 gives $\rho \approx 0.05$; arm 4 gives $\rho \le 0.01$ but is the one at risk on $\Delta$. **The single decisive quantity is $\Delta$ for arm 4** — if a 4-step delayed parameter update costs less than seed noise at 1.3B/100B tokens, offload is solved for this regime and the remaining work is scaling the check to 7B+.

## 9. Key References

- **[Foundational]** S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC 2020. — arXiv:1910.02054
- **[Foundational]** J. Ren, S. Rajbhandari, R. Y. Aminabadi, O. Ruwase, S. Yang, M. Zhang, D. Li, Y. He. *ZeRO-Offload: Democratizing Billion-Scale Model Training.* USENIX ATC 2021. — arXiv:2101.06840
- **[SOTA]** S. Rajbhandari, O. Ruwase, J. Rasley, S. Smith, Y. He. *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning.* SC 2021. — arXiv:2104.07857
- **[SOTA]** J. Fang, Z. Zhu, S. Li, H. Su, Y. Yu, J. Zhou, Y. You. *PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Dynamic Memory Management.* IEEE TPDS, 2023. — arXiv:2108.05818
- **[SOTA]** X. Sun et al. *STRONGHOLD: Fast and Affordable Billion-Scale Deep Learning Model Training.* SC 2022.
- **[SOTA]** T. Dettmers, M. Lewis, S. Shleifer, L. Zettlemoyer. *8-bit Optimizers via Block-wise Quantization.* ICLR 2022. — arXiv:2110.02861
- **[SOTA]** J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, Y. Tian. *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* ICML 2024. — arXiv:2403.03507
- **[SOTA]** Y. Zhang et al. *Adam-mini: Use Fewer Learning Rates To Gain More.* ICLR 2025. — arXiv:2406.16793
- **[Empirical]** C. Liao et al. *Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU.* 2024. — arXiv:2403.06504
- **[Empirical]** H. Jang et al. *Smart-Infinity: Fast Large Language Model Training using Near-Storage Processing on a Real System.* HPCA 2024.
- **[Theory]** A. Agarwal, J. C. Duchi. *Distributed Delayed Stochastic Optimization.* NeurIPS 2011.
- **[Theory]** S. U. Stich, S. P. Karimireddy. *The Error-Feedback Framework: Better Rates for SGD with Delayed Gradients and Compressed Updates.* JMLR, 2020.
- **[Context]** S. McCandlish, J. Kaplan, D. Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch Training.* 2018. — arXiv:1812.06162
- **[Survey]** J. Duan et al. *Efficient Training of Large Language Models on Distributed Infrastructures: A Survey.* 2024. — arXiv:2407.20018

## 10. Worked Example

7B model, 8 H100s, ZeRO-3. Per-rank shard $N/P = 875$M params. Gradients down in FP32 ($g=4$), weights up in BF16 ($p=2$). Node: PCIe Gen4 ×16 per GPU, 4 GPUs per socket, DDR5 ~200 GB/s achievable per socket.

$$T_{\text{xfer}} = \frac{6 \times 875\text{M}}{25\ \text{GB/s}} = 0.21\ \text{s}, \qquad T_{\text{host}} = \frac{30 \times 875\text{M}}{200/4\ \text{GB/s}} = 0.53\ \text{s}$$

Total offload work: **0.74 s per optimizer step**, and $T_{\text{host}}$ is a roofline floor — measured CPU-Adam runs slower.

**Case A — the published regime.** Micro-batch 8192 tokens, $k=32$. $T_{\text{fb}} = 6 \cdot 7\text{e}9 \cdot 8192 / (0.45 \cdot 990\text{e}12) = 0.77$ s. Window $W = 31 \times 0.77 = 23.9$ s. Offload work is 3% of the window. $T_{\text{exposed}} = 0$, $\rho = 0$. Paper reports "no penalty." True.

**Case B — the frontier regime.** Same model, 4096 GPUs, global batch pinned at the critical batch size, so $k = 1$. One micro-step per optimizer step: $W = 0$.
$$T_{\text{exposed}} = 0.74\ \text{s}, \qquad \rho = \frac{0.77 + 0.74}{0.77} - 1 = 0.96$$

**A 96% step-time penalty on exactly the same system that reported zero.** Halving the state with 8-bit moments takes $T_{\text{host}}$ to ~0.27 s and $\rho$ to ~0.62 — better, still nowhere near 2%. Moving to GH200's ~450 GB/s C2C takes $T_{\text{xfer}}$ to 0.012 s but leaves $T_{\text{host}} = 0.53$ s untouched, because the host update is DRAM-bandwidth-bound, not link-bound: $\rho \approx 0.70$. Only staleness closes the gap — $\tau = 1$ gives the transfer and update a full step to hide in, dropping $\rho$ to a few percent.

The obstruction is now visible: **every purely mechanical lever (faster link, coherent memory, compression) leaves $\rho$ an order of magnitude above target at $k=1$, and the only lever that reaches the target changes the optimizer's semantics in a way nobody has measured at pretraining scale.** That is why this is an open problem and not an engineering task.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*