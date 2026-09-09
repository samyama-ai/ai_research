---
id: 32-hardware-and-kernels/gnn-kernel-autoscheduling
title: "Compiler Autoscheduling for Irregular Graph Neural Network Kernels"
topic: 32-hardware-and-kernels
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compiler Autoscheduling for Irregular Graph Neural Network Kernels

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/gnn-kernel-autoscheduling` · **Status:** open

## 1. Problem Statement

Dense-tensor autoschedulers (Halide, Ansor, MetaSchedule) work because the cost of a loop nest is a function of the *program* — shapes are compile-time constants, so a learned model maps schedule features to runtime. GNN kernels break this. The dominant kernels — SpMM (sparse-dense matmul, the aggregation step), SDDMM (sampled dense-dense matmul, the attention step), and segment reductions — have runtimes that depend on the *values* of the sparsity structure, not just its dimensions. Two graphs with identical $(n, \text{nnz})$ can differ 3–5× in achieved bandwidth under one schedule and invert their ranking under another.

Three variants, in increasing difficulty:

- **Measurement.** Given a graph $G$, a kernel, and a hardware target, define a benchmark protocol whose measured runtime is attributable to the schedule rather than to vertex ordering, caching state, or format-conversion cost amortized off-book. *Open in the weak sense that no community standard exists.*
- **Method.** Learn $\hat{T}(s, G, H)$ — a cost model over schedule $s$, graph $G$, hardware $H$ — accurate enough that search over $s$ without on-device measurement beats hand-written libraries (cuSPARSE, dgSPARSE) on unseen graphs. *This is the operational problem.*
- **Theory.** Characterize the schedule-selection problem: is there a poly-time computable statistic of $G$ that is sufficient for the argmax over a given schedule space? *No answer either way.*

Solving it means: a compiler that, given an unseen graph and a fixed tuning budget of $B$ on-device measurements with $B \le 10$, matches within 5% the runtime found by exhaustive search, across a distribution of graphs spanning at least three orders of magnitude in average degree.

## 2. Formal Setting

**Graph and kernel.** $G = (V, E)$, $n = |V|$, $m = |E|$, adjacency $A \in \{0,1\}^{n \times n}$ stored in a format $f$. Features $X \in \mathbb{R}^{n \times d}$. Aggregation is
$$Y = A X, \qquad Y_{i:} = \sum_{j \in \mathcal{N}(i)} X_{j:}.$$
Attention scoring is SDDMM: $S_{ij} = \langle Q_{i:}, K_{j:}\rangle$ for $(i,j) \in E$ only.

**Schedule.** $s \in \mathcal{S}$ is a tuple of loop transformations: tile sizes over the feature dimension $d$, row-block size, thread-per-row vs. warp-per-row assignment, load-balance strategy (row-split / nnz-split / merge-based), use of shared memory for $X$ tiles, vector width, and format $f$ (CSR, ELL, blocked-ELL, hybrid). $|\mathcal{S}|$ is $10^4$–$10^7$ for realistic spaces.

**Measured quantities.**
- $T(s, G, H)$: median wall-clock over $\ge 100$ iterations after $\ge 20$ warmup iterations, L2 flushed between iterations. Without the flush, small graphs measure L2-resident performance and the number means nothing for a training loop.
- Useful work $W = 2md$ FLOP for SpMM.
- Compulsory traffic $Q_{\min} = 4(nd + m + n)$ bytes (fp32, CSR indices), assuming perfect reuse of $X$.
- Actual traffic $Q(s,G)$: measured with `dram__bytes.sum` (Nsight Compute) or equivalent.
- **Reuse deficit** $\rho(s, G) = Q(s,G)/Q_{\min} \in [1, \, md/(nd)] = [1, \bar{k}]$ where $\bar{k}=m/n$ is average degree. This is the single quantity that governs SpMM performance and it is *not* a function of $(n,m,d)$.
- Regret of a predictor: $R = \mathbb{E}_G\!\left[ \frac{T(\hat{s}_G, G, H)}{\min_{s} T(s, G, H)} - 1 \right]$, with $\hat{s}_G = \arg\min_s \hat{T}(s,G,H)$. **Regret, not speedup over a baseline, is the right target** — speedup over cuSPARSE conflates model quality with baseline weakness.

**Assumptions, and which are violated.**
1. *Runtime is a deterministic function of $(s,G,H)$.* Violated: clock throttling and CTA scheduling nondeterminism give 2–5% run-to-run spread on A100/H100, which is the same order as the differences between the top few schedules.
2. *Graph ordering is part of $G$.* Usually violated in practice — papers report a graph name, not a permutation, while reordering (RCM, Rabbit Order) moves runtime 1.3–2× on a *fixed* schedule.
3. *Format conversion is amortized.* Holds for full-batch training, fails for sampled minibatch training where a new subgraph arrives every step and conversion can exceed kernel time.
4. *The schedule space contains the optimum.* Unfalsifiable in practice; $\min_s T$ is measured over the searched space, so reported regret is a lower bound on true regret.

## 3. State of the Art

**Sparse compiler infrastructure (established).** TACO (Kjolstad et al., OOPSLA 2017) gives format-generic code generation; the sparse iteration-space scheduling language (Senanayake et al., OOPSLA 2020) makes $\mathcal{S}$ addressable. SparseTIR (Ye et al., ASPLOS 2023) adds composable formats and tensor-core-eligible decompositions inside TVM; Finch (Ahrens et al., 2023–2024) extends codegen to structured non-zero patterns. These are established as *expressiveness* results — the schedules exist and compile.

**GNN-specific backends (established, narrow).** FeatGraph (SC 2020) and Graphiler (MLSys 2022) supply fusion of message/aggregate stages; Seastar (EuroSys 2021) gives vertex-centric fusion; GE-SpMM (SC 2020) and TC-GNN (USENIX ATC 2023) supply hand-tuned kernel families. All rely on *heuristic* schedule selection or a small hand-enumerated set, not learned search.

**Learned selection (claimed, partly unablated).** WACO (Won, Mendis, Emer, Amarasinghe; ASPLOS 2023) is the closest direct attack: it learns a co-optimizer over format *and* schedule from a sparse-pattern embedding of the matrix, trained and evaluated over hundreds of SuiteSparse matrices, and reports beating TACO and ASpT. What is *not* ablated in the literature: how much of the gain is format choice vs. loop schedule, and how performance degrades under distribution shift to graphs unlike SuiteSparse (e.g., power-law social graphs with $\bar k > 50$).

**Asymptotic autoscheduling (established, coarse).** Ahrens, Kjolstad and Amarasinghe (PLDI 2022) give an autoscheduler for sparse tensor algebra driven by an *asymptotic* cost model — provably picks the right complexity class, explicitly does not target constant factors, which is where the 2× lives.

**Reported end-to-end numbers exist only as benchmark numbers.** SparseTIR reports roughly 1.2–2.3× on single GNN operators and ~1.3–1.5× end-to-end over vendor/framework baselines on a handful of OGB and SuiteSparse graphs. TC-GNN reports ~1.7× over DGL. These are single-hardware, few-graph point measurements, not regret curves.

## 4. What Is Known

- **The dense-tensor recipe transfers poorly.** TenSet (Zheng et al., NeurIPS 2021 Datasets) established that learned cost models for dense operators generalize across networks on fixed hardware. No comparable dataset exists for sparse kernels; the analogous claim is untested.
- **Structure dominates size.** On SuiteSparse and OGB graphs at $d=128$ on A100-class GPUs, SpMM achieved bandwidth spans roughly 15%–80% of peak across matrices at *identical* $(n,m)$ scale — a >4× spread attributable to structure alone.
- **Reordering is a first-order effect.** Rabbit Order (Arai et al., IPDPS 2016) and related locality-optimizing permutations give order-1.3–2× on graph workloads without touching the kernel. Any schedule search that fixes the ordering is measuring a confounded quantity.
- **Tensor cores are usable but only conditionally.** TC-GNN (ATC 2023) shows dense-core execution of sparse aggregation wins when the graph has exploitable block density, and loses otherwise; the crossover is not characterized as a computable predicate.
- **Load balance is the dominant schedule axis for power-law graphs.** Merge-based / nnz-split SpMV and SpMM (Merrill & Garland, SC 2016) give bounded work per thread independent of degree skew; on uniform-degree meshes the same schedule loses to row-split from index overhead.

## 5. What Is Not Known

- **Theoretically open.** Whether any polynomially computable statistic $\phi(G)$ of size $o(m)$ is *sufficient* for schedule selection — i.e. $\arg\min_s T(s,G,H)$ depends on $G$ only through $\phi(G)$ — for a nontrivial schedule space. No hardness result and no construction. Related: no lower bound on $\rho$ achievable by any schedule given a fixed vertex ordering.
- **Empirically open.** Whether a cost model trained on $\sim 10^3$ graphs achieves regret $< 5\%$ on held-out graphs from a different family. The experiment is runnable today on one 8-GPU node; nobody has published the regret curve.
- **Methodologically blocked.** Cross-hardware transfer. There is no agreed protocol for holding the graph, ordering, format-conversion accounting and L2 state fixed while varying the target, so "our cost model transfers to H100" is currently unmeasurable rather than unproven.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement plus absent ground truth for the label the model needs.** The quantity a cost model must predict is the reuse deficit $\rho(s,G)$, which is determined by the interaction of vertex ordering, cache capacity and the schedule's traversal order. It is not a function of any feature that is cheap to compute: computing it exactly requires simulating the cache on the actual access trace, which costs more than running the kernel. So the model is asked to regress a quantity whose only ground truth is the measurement it is meant to replace.

Second obstruction: the label is noisy at the resolution that matters. Top-5 schedules typically sit within 5–10% of each other, while measurement spread is 2–5%; a large share of the training signal is at or below noise.

Third: distribution. SuiteSparse is dominated by scientific meshes ($\bar k \approx 5$–$50$, near-banded); GNN workloads include power-law graphs with $\bar k$ in the hundreds and heavy-tailed degree. A model fit on the former is out of distribution on the latter, and the failure shows up as regret on exactly the graphs practitioners care about.

## 7. Current Research (as of 2026)

- **Sparse abstractions in mainstream compilers.** MLIR's sparse tensor dialect and the TVM/SparseTIR line (Ye, Shao, Chen — UW/CMU/OctoML lineage) continue to widen $\mathcal{S}$; the selection problem is explicitly left to the user.
- **Pattern-conditioned cost models.** WACO's line at MIT (Amarasinghe, Mendis) — learned matrix embeddings feeding schedule search. *(frontier — verify)* Extensions to fused GNN layers rather than single operators are in progress.
- **Structured-sparsity codegen.** Finch / Looplets (Ahrens, MIT) — compiling over runtime-discovered structure rather than assuming a format.
- **Relational and heterogeneous GNNs.** Hector (ASPLOS 2024) targets RGNNs where the schedule space additionally includes per-relation-type layout choices, expanding $\mathcal{S}$ by another factor.
- **Search-free selection.** *(frontier — verify)* Several groups are reported to be trying decision-tree or nearest-neighbour selection over precomputed graph fingerprints as a cheaper alternative to learned regression; no published regret numbers.

## 8. Concrete Next Experiment

**Question.** Does a pattern-conditioned cost model achieve regret $< 5\%$ on out-of-family graphs at a tuning budget of 10 measurements?

**Scale.** 1,000 graphs: 600 SuiteSparse (train), 200 SuiteSparse held-out (in-family test), 200 out-of-family test (OGB — `ogbn-arxiv`, `ogbn-products`, `ogbn-proteins`, `ogbl-citation2` — plus synthetic power-law graphs with $\bar k \in \{4, 32, 256\}$). Kernel: fp32 SpMM at $d \in \{32,128,512\}$. Hardware: one A100 80GB. Schedule space: 512 points (row-split/nnz-split/merge × 4 feature tiles × 4 row blocks × shared-memory on/off × CSR/blocked-ELL). Exhaustive ground truth = $512 \times 1000 \times 3 \approx 1.5$M kernel runs; at 1 ms amortized including L2 flush and warmup, ≈ 25 GPU-hours. **This is a one-node, two-day experiment.** Each graph is measured under two orderings (original, Rabbit) so ordering enters as a covariate rather than a confound.

**Control arms.** (a) Best single fixed schedule chosen on the training set — the "no model" arm. (b) The library baseline: cuSPARSE `cusparseSpMM` plus dgSPARSE. (c) Random search with the same budget of 10 on-device measurements. Any learned model must beat (a) and (c) at equal budget; beating (b) alone proves nothing.

**Deciding number.** Median regret $R$ on the 200 out-of-family graphs at budget $B=10$. $R < 5\%$ with the fixed-schedule arm above 25% settles the method variant affirmatively. $R > 20\%$, while in-family regret is $< 5\%$, localizes the failure to distribution shift and turns the problem into a graph-representation problem rather than a search problem — a different and more tractable target.

## 9. Key References

- **[Foundational]** Fredrik Kjolstad, Shoaib Kamil, Stephen Chou, David Lugato, Saman Amarasinghe. *The Tensor Algebra Compiler.* OOPSLA 2017.
- **[Foundational]** Ryan Senanayake, Changwan Hong, Ziheng Wang, Amalee Wilson, Stephen Chou, Shoaib Kamil, Saman Amarasinghe, Fredrik Kjolstad. *A Sparse Iteration Space Transformation Framework for Sparse Tensor Algebra.* OOPSLA 2020.
- **[Foundational]** Tianqi Chen et al. *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.* OSDI 2018.
- **[Foundational]** Lianmin Zheng et al. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI 2020.
- **[SOTA]** Jaeyeon Won, Charith Mendis, Joel Emer, Saman Amarasinghe. *WACO: Learning Workload-Aware Co-optimization of the Format and Schedule of a Sparse Tensor Program.* ASPLOS 2023.
- **[SOTA]** Zihao Ye, Ruihang Lai, Junru Shao, Tianqi Chen, Luis Ceze. *SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning.* ASPLOS 2023.
- **[SOTA]** Peter Ahrens, Fredrik Kjolstad, Saman Amarasinghe. *Autoscheduling for Sparse Tensor Algebra with an Asymptotic Cost Model.* PLDI 2022.
- **[SOTA]** Yuke Wang, Boyuan Feng, Zheng Wang, Guyue Huang, Yufei Ding. *TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs.* USENIX ATC 2023.
- **[Systems]** Yuwei Hu et al. *FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems.* SC 2020.
- **[Systems]** Guyue Huang, Guohao Dai, Yu Wang, Huazhong Yang. *GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph Neural Networks.* SC 2020.
- **[Systems]** Zhiqiang Xie, Minjie Wang, Zihao Ye, Zheng Zhang, Rui Fan. *Graphiler: Optimizing Graph Neural Networks with Message Passing Data Flow Graph.* MLSys 2022.
- **[Systems]** Duane Merrill, Michael Garland. *Merge-Based Parallel Sparse Matrix-Vector Multiplication.* SC 2016.
- **[Systems]** Junya Arai, Hiroaki Shiokawa, Takeshi Yamamuro, Makoto Onizuka, Sotetsu Iwamura. *Rabbit Order: Just-in-Time Parallel Reordering for Fast Graph Analysis.* IPDPS 2016.
- **[Dataset]** Timothy A. Davis, Yifan Hu. *The University of Florida Sparse Matrix Collection.* ACM TOMS, 2011.
- **[Dataset]** Weihua Hu et al. *Open Graph Benchmark: Datasets for Machine Learning on Graphs.* NeurIPS 2020.
- **[Survey]** Lianmin Zheng et al. *TenSet: A Large-Scale Program Performance Dataset for Learned Tensor Compilers.* NeurIPS 2021 Datasets and Benchmarks.

## 10. Worked Example

`ogbn-arxiv`: $n = 169{,}343$, $m = 1{,}166{,}243$ directed edges, $\bar k \approx 6.9$. SpMM at $d = 128$, fp32, on an A100 (1.55 TB/s HBM, 19.5 TFLOP/s fp32).

Useful work: $W = 2md = 2 \times 1.166\text{e}6 \times 128 \approx 2.99\text{e}8$ FLOP $= 0.30$ GFLOP.

Two traffic regimes for the *same* arithmetic:

| Regime | Traffic $Q$ | HBM-bound time |
|---|---|---|
| Perfect reuse of $X$ ($\rho=1$): $4(nd + m + n)$ | 91.1 MB | 59 µs |
| No reuse ($\rho = \bar k$): $4(md + m)$ | 601 MB | 388 µs |

The span between the two is 6.6×, and *every point in it is reachable by some schedule on this one graph*. Arithmetic intensity is $0.30/0.091 = 3.3$ FLOP/B at best, against the A100 ridge point of $19.5/1.55 = 12.6$ FLOP/B — memory-bound throughout, so predicting runtime *is* predicting $\rho$.

Now the obstruction. Take two schedules: $s_1$ = row-split, one warp per row, no shared memory; $s_2$ = nnz-split with merge-based balancing and a 32-wide feature tile staged in shared memory. On `ogbn-arxiv` in its native (roughly chronological, hence temporally local) ordering, rows touch neighbours with close indices, $X$ tiles stay L2-resident, and $s_1$ is competitive. Apply Rabbit Order and $\rho$ drops further; apply a random permutation and $\rho$ approaches $\bar k = 6.9$, at which point $s_2$'s shared-memory staging wins and the ranking of $s_1$ vs. $s_2$ *inverts* — with $n$, $m$, $d$, $\bar k$, degree histogram and every other cheap statistic held exactly constant, because a permutation changes none of them.

That is the whole problem in one instance: the feature that decides the argmax is the *permutation*, an object of size $n$ that no fixed-length embedding of a graph summary preserves. A cost model that consumes $(n, m, \bar k, \text{degree histogram})$ is provably unable to distinguish these two cases, and one that consumes the full sparsity pattern costs $O(m)$ to evaluate — the same order as running the kernel it is trying to avoid running.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*