---
id: 31-distributed-training/memory-fragmentation-long-runs
title: "Memory Fragmentation Bounds for Long Distributed Runs"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memory Fragmentation Bounds for Long Distributed Runs

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/memory-fragmentation-long-runs` · **Status:** open

## 1. Problem Statement

A large training job survives step 1 and dies at step 40,000 with `CUDA out of memory` while the sum of live tensors is unchanged. The allocator holds enough total free bytes; no single free block is large enough. The question is whether this failure can be **bounded in advance** rather than discovered.

Three variants, of very different difficulty:

- **Measurement.** Given a run, report a single number that says how much of reserved device memory is unusable because of fragmentation, and that number must be stable under re-runs and comparable across allocators. Not currently well defined (§5).
- **Method.** Design an allocator (or a memory plan) whose peak reserved memory over $T$ steps exceeds the peak live memory by a factor that does not grow with $T$. Partially achieved by CUDA virtual-memory stitching; unproven.
- **Theory.** Prove a bound on the reserved/live ratio for the *specific* allocation workload of distributed transformer training — near-periodic, few distinct sizes, with bounded aperiodic perturbations (variable sequence length, checkpointing, evaluation passes, elastic re-sharding). Classical worst-case allocator bounds are $\Theta(\log n)$ and are far too loose to be useful here.

Solving it means: given the first $T_0 \ll T$ steps of a trace, emit a certificate that the job will not OOM from fragmentation before step $T$, or emit the step at which it will.

## 2. Formal Setting

An **allocation trace** is a sequence of requests $\sigma = (r_1, r_2, \dots)$, $r_i \in \{\texttt{alloc}(s_i), \texttt{free}(p_i)\}$ with sizes $s_i \in [s_{\min}, s_{\max}]$, $n = s_{\max}/s_{\min}$ the size ratio. *Measured as:* the record emitted by `torch.cuda.memory._record_memory_history()`, one entry per allocator call with size, stream, and stack.

- **Live bytes** $L(t) = \sum_{\text{live } p} s_p$. *Measured as:* `torch.cuda.memory_allocated()`.
- **Reserved bytes** $R(t)$ = bytes the allocator holds from the driver. *Measured as:* `torch.cuda.memory_reserved()`, equal to the sum of segment sizes in `memory_stats()`.
- **Fragmentation ratio** $$\Phi(t) \;=\; \frac{R(t) - L(t)}{R(t)}, \qquad \Phi \in [0,1).$$ This is the number practitioners quote, and it is the wrong one: it charges the allocator for deliberately cached free blocks that will be reused next step.
- **Blocking fragmentation**, the honest quantity: $$\Phi_{\mathrm{blk}}(t) \;=\; \frac{1}{R(t)}\Big(R(t)-L(t) - \max_{b \in \mathcal{F}(t)} |b|\Big),$$ where $\mathcal{F}(t)$ is the set of free blocks. It measures free bytes that cannot serve the largest pending request. It requires the free-list, which no framework exposes as a first-class counter.
- **Overhead factor** $\rho(T) = \max_{t \le T} R(t) \,/\, \max_{t\le T} L(t)$. The object to be bounded. A run is *drift-free* if $\rho(T) = O(1)$ in $T$; the empirical question is whether real runs are.
- **OOM predicate.** Failure at $t$ iff request $s_t$ arrives, $\max_b |b| < s_t$, and the driver cannot supply a new segment: $R(t) + \lceil s_t \rceil_{\text{seg}} > C$ for device capacity $C$ (80 GiB on H100 SXM, 141 GiB on H200).

**Allocator model.** PyTorch's caching allocator: two pools (small, $<1$ MiB requests, 2 MiB segments; large, 20 MiB segments), best-fit within a size-ordered free list, splitting of oversized blocks, no coalescing across segments, and a full `cudaFree`-everything fallback (`empty_cache`) before raising OOM. With `expandable_segments:True` it instead reserves a large virtual address range via the CUDA VMM API and maps physical pages into it, which makes non-adjacent physical pages contiguous in the virtual address space.

**Assumptions, and which are violated.**

| Assumption | Status in practice |
|---|---|
| Trace is periodic across steps | **Violated.** Variable sequence length, dynamic MoE expert routing, bucketed padding, and periodic eval/checkpoint passes inject off-cycle sizes. |
| Single stream, one allocation order | **Violated.** Comm streams, NCCL buffers, and CUDA graphs allocate concurrently; order is nondeterministic. |
| Allocator state is per-rank independent | **Violated in effect.** Collectives are synchronous, so one rank's OOM kills the job — the relevant statistic is $\max$ over $10^3$–$10^4$ ranks, not the mean. |
| Sizes drawn from a fixed finite set | Approximately true for dense transformers, false for MoE with capacity-factor drop. |
| Free is immediate | **Violated.** Autograd graph retention and async collectives delay frees by an unbounded number of steps. |

## 3. State of the Art

**Theory SOTA (established, but not for this workload).** Robson's bounds for dynamic storage allocation: any allocator needs, in the worst case, at least about $\tfrac{1}{2} M \log_2 n$ memory for max-live $M$ and size ratio $n$ (Robson, *JACM* 1974), and there exists a strategy achieving $O(M \log_2 n)$ (Robson, *JACM* 1971). First-fit is within a constant of that; best-fit has $\Omega(M\sqrt{n})$-type worst cases. These are adversarial-trace bounds. With $n \approx 10^7$ (1 KiB to 10 GiB) they permit $\rho \approx 23$, which is vacuous: real runs live at $\rho \approx 1.05$–$1.4$. **No non-trivial bound exists for the near-periodic trace class of transformer training.**

**Systems SOTA.**
- **Segment stitching.** GMLake (Guo et al., ASPLOS 2024) uses CUDA VMM to stitch non-contiguous physical blocks; reports up to **25 GB** GPU memory reduction and up to $\sim 33\%$ reduction in reserved memory on 8–32 GPU fine-tuning workloads. PyTorch's `expandable_segments` is the same mechanism upstream. *Established* as a peak-memory reduction on the reported workloads; **unablated** as a *drift* fix — no published experiment runs it for $>10^4$ steps and reports $\rho(T)$ versus $T$.
- **Static planning.** Checkmate (Jain et al., MLSys 2020) solves optimal rematerialization as an ILP; Korthikanti et al. (MLSys 2023) cut activation memory by $\sim 5\times$ with selective recomputation at 530B scale. Both bound *live* memory, not reserved — they shrink $L$, and say nothing about $\rho$.
- **Sharding.** ZeRO (Rajbhandari et al., SC 2020) and PyTorch FSDP (Zhao et al., VLDB 2023) reduce $L$ per rank but *increase* the aperiodic component: all-gather buffers of parameter-shard size appear and vanish per layer, which is exactly the churn that fragments.
- **Serving.** PagedAttention/vLLM (Kwon et al., SOSP 2023) eliminates KV-cache fragmentation by fixed-size paging, reporting waste under $4\%$ versus $60$–$80\%$ for contiguous allocation. This is the existence proof that paging kills fragmentation — for one tensor of one shape. Nobody has paged the whole training heap.

**Only-a-benchmark-number:** every published "fragmentation reduced by X%" figure, including GMLake's, is a $\Phi$ or peak-reserved delta on a short run, not a bound and not a $T$-extrapolation.

## 4. What Is Known

- **Long runs fail often, for mixed causes.** Llama 3 405B on 16,384 H100s: **466 job interruptions in 54 days**, 419 unexpected, of which about **78%** were confirmed hardware (Llama 3 herd of models, Meta, 2024). Software-side OOM is a minority but a nonzero, unbudgeted fraction.
- **OOM is a named top-level failure class** in datacenter LLM-development traces: Hu et al. (NSDI 2024) characterize six months of Acme cluster jobs and list CUDA OOM among the dominant framework-level failures.
- **Fragmentation is real at reported magnitude.** vLLM measured $60$–$80\%$ waste in pre-paging KV allocation (SOSP 2023). GMLake measured tens of GB of reserved-but-unusable memory in training (ASPLOS 2024).
- **The workaround is universal and undocumented as science.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `max_split_size_mb`, and periodic `torch.cuda.empty_cache()` are standard practice. `empty_cache` costs a device synchronize plus `cudaFree`, typically $10$–$10^2$ ms; at every step on a 300 ms step this is a $3$–$30\%$ throughput tax.
- **Baseline live-memory arithmetic is settled.** Mixed-precision Adam holds $\approx 16$ bytes/parameter of states (fp32 master + two moments + fp16 grad), so a 70B model needs $\approx 1.1$ TB of optimizer state before activations — the well-known ZeRO accounting.

## 5. What Is Not Known

- **Methodologically blocked.** There is no agreed definition of the fragmentation number. $\Phi$ conflates caching with waste; $\Phi_{\mathrm{blk}}$ is right but not exposed by any framework counter and depends on the *next* request's size. Until this is fixed, cross-paper comparisons of "fragmentation reduced by X%" are not commensurable.
- **Empirically open.** Does $\rho(T)$ drift? Nobody has published $\rho$ as a function of $t$ over $\ge 10^5$ steps at $\ge 10^3$ ranks with per-rank distributions. The run exists inside several labs' telemetry; the curve is not in the literature. Nor is the tail: at $10^4$ ranks the job dies when the *worst* rank dies, so the relevant object is the max-order statistic of $\rho$, never reported.
- **Theoretically open.** No bound on $\rho$ for near-periodic traces. Concretely: for a trace that is exactly periodic with period $P$ plus at most $k$ aperiodic requests per period, is $\rho = 1 + O(k s_{\max}/M)$, independent of $T$? No proof either way. Also open: whether VMM stitching provably gives $\rho = 1 + O(\text{page}/M)$, or merely moves fragmentation into the virtual address space, where 2 MiB granularity and finite VA reservation reintroduce it.

## 6. Why It Is Hard

**The obstruction is confounded measurement compounded by non-identifiability of the failure cause.** When a 10,000-GPU job OOMs at step 40,000, the observable is one rank's exception. Distinguishing (a) allocator fragmentation, (b) a genuine live-memory spike from a long sequence in that microbatch, (c) an NCCL buffer growth, and (d) a memory leak in user code requires the free-list history for that rank at that instant — which is not retained, because recording it (`_record_memory_history`) costs $5$–$20\%$ throughput and unbounded host memory, so it is off in production. The evidence is destroyed by the cost of collecting it.

Second obstruction: **the experiment is expensive and non-reusable.** The drift signal is defined at $T \sim 10^5$ steps. A control arm needs the same schedule, same data order, same hardware, differing only in allocator config — a second full pretraining run. At 1,000 H100-days per arm, roughly $10^5$ GPU-hours, the ablation costs more than most groups' entire compute budget, and cannot be borrowed from a production run because production runs change config mid-flight.

Third: **the adversarial theory is useless and the average-case theory does not exist.** Robson's $\log n$ is tight for adversarial traces, so any improvement must exploit near-periodicity — which requires a trace model nobody has fitted to real jobs.

## 7. Current Research (as of 2026)

- **Virtual-memory allocators.** Upstream PyTorch `expandable_segments` (Meta) is now default-on in several internal configs *(frontier — verify)*; GMLake (Alibaba/SJTU) is the published academic instance. Direction: make stitching the default and remove segment-level fragmentation by construction.
- **Paged training memory.** Extending the vLLM insight from KV cache to activations and optimizer state — fixed-size pages for all large tensors, accepting internal fragmentation of at most one page per tensor in exchange for zero external fragmentation. Prototypes exist in serving-adjacent stacks; no published training-scale result.
- **Allocator telemetry as a first-class metric.** Proposals to export $\max_b |b|$ and free-list histograms as cheap counters, enabling $\Phi_{\mathrm{blk}}$ without full history recording *(frontier — verify)*.
- **Fault-tolerance framing.** MegaScale (ByteDance, NSDI 2024) and Llama 3 treat OOM as one entry in a fast-restart taxonomy: rather than bound fragmentation, detect and restart in seconds. This is currently winning on engineering grounds and is why the bound remains unproven — the failure is survivable, so nobody pays to eliminate it.
- **Deterministic memory planning for compiled graphs.** `torch.compile` + CUDA graphs fix the allocation sequence, which in principle makes offline planning exact; the open part is the dynamic-shape escape hatch.

## 8. Concrete Next Experiment

**Question.** Does $\rho(T)$ drift with $T$ under the default caching allocator, and does VMM stitching remove the drift?

**Scale.** A 7B dense transformer, FSDP across **256 H100s**, $10^5$ steps, variable sequence length bucketed to $\{2048, 4096, 8192\}$ with the real length distribution (not fixed padding — fixed padding removes the aperiodicity being tested). Roughly 6–8 wall-clock days per arm, $\approx 4\times10^4$ GPU-hours per arm. This is the smallest scale with both real length variance and enough ranks for a tail.

**Arms.**
- **Control:** `expandable_segments:False`, no `empty_cache`, default `max_split_size_mb`.
- **Treatment A:** `expandable_segments:True`, otherwise identical (same seed, same data order, same schedule).
- **Treatment B (cost baseline):** control plus `empty_cache()` every 100 steps.

**Instrumentation.** Every 100 steps, on every rank, log `memory_allocated`, `memory_reserved`, and the largest free block (one extra allocator counter, $O(1)$ cost). Never enable full history.

**The deciding number.** The slope of the per-rank maximum overhead factor: $$\hat\beta \;=\; \frac{\rho_{\max}(10^5) - \rho_{\max}(10^4)}{\log_{10} 10} ,\qquad \rho_{\max}(T) = \max_{r \le 256}\ \rho^{(r)}(T).$$ If $\hat\beta \le 0.01$ per decade in the control, fragmentation drift is not real and the problem collapses to a peak-memory problem. If $\hat\beta \ge 0.05$ per decade in control and $\le 0.01$ in Treatment A, VMM stitching is the fix and the open question becomes proving it. If both drift, the problem is genuinely open and paging is the next thing to try.

## 9. Key References

- **[Foundational]** J. M. Robson. *An Estimate of the Store Size Necessary for Dynamic Storage Allocation.* Journal of the ACM 18(3), 1971.
- **[Foundational]** J. M. Robson. *Bounds for Some Functions Concerning Dynamic Storage Allocation.* Journal of the ACM 21(3), 1974.
- **[Survey]** P. R. Wilson, M. S. Johnstone, M. Neely, D. Boles. *Dynamic Storage Allocation: A Survey and Critical Review.* International Workshop on Memory Management (IWMM), 1995.
- **[Foundational]** M. S. Johnstone, P. R. Wilson. *The Memory Fragmentation Problem: Solved?* ISMM, 1998.
- **[SOTA]** C. Guo et al. *GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching.* ASPLOS, 2024. — arXiv:2401.08156
- **[SOTA]** W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, I. Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Foundational]** S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC, 2020. — arXiv:1910.02054
- **[SOTA]** Y. Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.* VLDB, 2023. — arXiv:2304.11277
- **[SOTA]** V. Korthikanti, J. Casper, S. Lym, L. McAfee, M. Andersch, M. Shoeybi, B. Catanzaro. *Reducing Activation Recomputation in Large Transformer Models.* MLSys, 2023. — arXiv:2205.05198
- **[SOTA]** P. Jain, A. Jain, A. Nrusimha, A. Gholami, P. Abbeel, J. Gonzalez, K. Keutzer, I. Stoica. *Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization.* MLSys, 2020. — arXiv:1910.02653
- **[Empirical]** Z. Jiang et al. *MegaScale: Scaling Large Model Training to More Than 10,000 GPUs.* NSDI, 2024. — arXiv:2402.15627
- **[Empirical]** Q. Hu et al. *Characterization of Large Language Model Development in the Datacenter.* NSDI, 2024.
- **[Empirical]** Llama Team, Meta AI. *The Llama 3 Herd of Models.* 2024. — arXiv:2407.21783
- **[Foundational]** T. Chen, B. Xu, C. Zhang, C. Guestrin. *Training Deep Nets with Sublinear Memory Cost.* 2016. — arXiv:1604.06174

## 10. Worked Example

A 7B model, FSDP over 8 H100-80GB, activation checkpointing on. Steady-state live memory per rank: $L \approx 62$ GiB. Capacity $C = 79.2$ GiB usable. Headroom: $17.2$ GiB.

Per step the allocator sees, in the large pool, roughly the same 340 requests. Two of them are aperiodic: the all-gather buffer for the largest parameter shard, $1.75$ GiB, and the activation block for the current sequence bucket, which takes one of $\{1.2, 2.4, 4.8\}$ GiB depending on the bucket drawn.

Step $t$: bucket 8192 is drawn. The allocator carves a $4.8$ GiB activation block out of a fresh 20 MiB-granular segment run. Step $t+1$: bucket 2048. The $4.8$ GiB block is freed and a $1.2$ GiB block is split from it. The residual $3.6$ GiB stays on the free list. Step $t+2$: bucket 8192 again, but the $3.6$ GiB residual is not adjacent to anything free — the $1.2$ GiB piece is still live because the backward pass has not run. A **new** $4.8$ GiB region is taken from the driver.

Reserved has grown by $4.8$ GiB while live is unchanged. If exactly one such stranded residual survives per 500 steps and is never recovered:

$$R(t) \approx 62 + 3.6\,\frac{t}{500}\ \text{GiB}, \qquad \text{OOM at } t \approx \frac{17.2}{3.6}\times 500 \approx 2{,}390 \text{ steps}.$$

If instead one survives per 50,000 steps, the job runs 239,000 steps and finishes clean. **The two scenarios differ by a factor of 100 in a rate nobody measures.** That is the whole problem: $\Phi$ at step 100 is $\approx 0.03$ in both, and identical. The distinguishing observable is $\max_b |b|$ — the largest free block — which falls monotonically in the first scenario and is flat in the second, and which no framework reports. The $\hat\beta$ in §8 is exactly this rate, and the reason it is unknown is that the counter costs nothing and is simply not exported.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*