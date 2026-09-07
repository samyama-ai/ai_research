---
id: 32-hardware-and-kernels/activation-checkpointing-optimal-policy
title: "Optimal Recomputation-Versus-Memory Tradeoff in Activation Checkpointing"
topic: 32-hardware-and-kernels
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Recomputation-Versus-Memory Tradeoff in Activation Checkpointing

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/activation-checkpointing-optimal-policy` · **Status:** partially-solved

## 1. Problem Statement

Backpropagation needs the forward activations of every layer. Storing all of them costs memory linear in depth × batch × sequence. Activation checkpointing (rematerialization) stores a subset and recomputes the rest during the backward pass, trading FLOPs for bytes.

**Input.** A computation graph $G=(V,E)$ for one training step, per-node compute cost $c_v$ (seconds on the target device), per-node output size $m_v$ (bytes), and a device memory budget $M$ (bytes).

**Output.** A schedule: an ordering of forward, recompute, backward, free operations that never exceeds $M$ live bytes and produces correct gradients.

**Objective.** Minimize wall-clock time of the step subject to the memory constraint.

Three variants, differing in difficulty:

- **Theory.** For which graph classes is the optimal schedule computable in polynomial time, and what is the tight bound on the recompute factor as a function of $M$? Settled for homogeneous chains; open for general DAGs with heterogeneous costs beyond NP-hardness.
- **Method.** Given a real model and a real accelerator, find a near-optimal schedule fast enough to be used at compile time. Solved to within a few percent for single-device, static, dense graphs; unsolved once parallelism (tensor/pipeline/sequence), offloading, fused kernels, and dynamic shapes enter.
- **Measurement.** Define "memory used" and "cost of a node" so that the optimizer's model matches what the allocator and the kernels actually do. This is where deployed systems lose most of the theoretical gain.

Solving it means: a policy that, for a given model and device, is provably within a stated factor of the true minimum step time at the given budget, and whose predicted peak memory matches the measured peak within a few percent.

## 2. Formal Setting

Let the forward graph be a DAG with topological order $v_1,\dots,v_n$. A schedule is a sequence $S=(s_1,\dots,s_T)$ of operations $s_t \in \{\text{compute}(v), \text{free}(v)\}$; $R_t \subseteq V$ is the resident set after step $t$. Correctness requires that when $\text{compute}(v)$ executes, $\text{parents}(v)\subseteq R_{t-1}$, and that every backward node executes once.

$$\min_S \ \sum_{t:\,s_t=\text{compute}(v)} c_v \quad \text{s.t.} \quad \max_t \sum_{v\in R_t} m_v \;+\; M_{\text{fixed}} \;\le\; M .$$

**How each quantity is measured.**

- $m_v$ — the allocator's rounded block size for the tensor, not $\prod_i d_i \cdot \text{sizeof}(\text{dtype})$. PyTorch's caching allocator rounds to 512 B blocks and segments to 2 MB/20 MB pools; the gap between logical and reserved bytes is the fragmentation term.
- $c_v$ — measured by isolated kernel timing at the exact shape and dtype, with the device clock-locked. On a fused kernel the per-node decomposition does not exist: $c_v$ is only defined for the whole fusion group.
- $M_{\text{fixed}}$ — parameters + optimizer state + gradients + workspace + CUDA context. Under ZeRO/FSDP this is itself time-varying (sharded weights are gathered and released), so treating it as a constant is an approximation.
- Peak memory — `torch.cuda.max_memory_allocated()` (logical) versus `max_memory_reserved()` (what triggers OOM). The constraint binds on the latter.

Define the **recompute factor** $\rho = (\text{total FLOPs with recomputation}) / (\text{FLOPs of one forward+backward})$ and the **compression ratio** $\kappa = M_{\text{no ckpt}}/M$.

**Assumptions, and which are violated.**

1. *Additive, static $m_v$* — violated by fragmentation and by variable sequence lengths.
2. *Fixed $c_v$ independent of context* — violated: recompute of a node in a memory-pressured, cache-cold state can be 1.2–2× its isolated time.
3. *Compute and memory traffic are the only costs* — violated when recompute is bandwidth-bound; then $\rho$ overstates the true slowdown for compute-bound layers and understates it for elementwise chains.
4. *Sequential execution* — violated by async collectives, which recomputation can overlap with (making some recompute free) or serialize behind (making it superlinearly costly).
5. *Bitwise reproducibility of recompute* — violated by dropout and by nondeterministic kernels unless RNG state is captured and replayed.

## 3. State of the Art

**Theory SOTA (established).**
- Griewank & Walther, *Algorithm 799: revolve* (ACM TOMS, 2000): for a homogeneous chain of length $l$ with $c$ checkpoints, binomial checkpointing is provably optimal, and $l \le \binom{c+r}{c}$ where $r$ is the maximum number of forward recomputations of any step — i.e. logarithmic growth of $\rho$ in $\kappa$.
- Beaumont et al., *Optimal checkpointing for heterogeneous chains* (NeurIPS 2019): exact dynamic program for chains with per-layer heterogeneous costs and sizes, shipped as `rotor`.
- Jain et al., *Checkmate* (MLSys 2020): rematerialization on general DAGs with heterogeneous costs is NP-complete; they give an ILP plus a two-phase LP rounding approximation.
- Kusumoto et al. (NeurIPS 2019): polynomial-time optimal recomputation for restricted graph families.

**Systems SOTA (established).**
- Chen et al. (2016), `torch.utils.checkpoint`: $O(\sqrt{n})$ memory with one extra forward pass, $\rho \approx 1.3$.
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (MLSys 2023): selective recomputation — recompute only the attention block whose activations are large relative to their FLOPs — plus sequence parallelism.
- Kirisame et al., *Dynamic Tensor Rematerialization* (ICLR 2021): an online greedy eviction heuristic with a cost-per-byte-per-staleness score; needs no static graph.

**Claimed but unablated.** Checkmate's "1.73× throughput" and "up to 5.1× larger batch" are benchmark numbers on V100 for vision graphs; they have not been reproduced independently at transformer scale, and the ILP's solve time (minutes to hours) is excluded from the reported speedup. MONeT (Shah et al., ICLR 2021) jointly picks rematerialization and low-precision/implementation choices, reporting ~3× memory reduction at ~9–16% overhead — again a benchmark table, with the operator-implementation axis not separately ablated from the rematerialization axis.

## 4. What Is Known

- **Sublinear memory works.** Chen et al. (2016), ResNet-1001 and LSTM-scale models: memory linear→$O(\sqrt{n})$ at ~30% extra time.
- **Full recompute is expensive at LLM scale.** Korthikanti et al. (MLSys 2023) measure full activation recomputation overhead at 36% for a 22B transformer, 39–40% for 175B/530B (A100, Megatron-LM). Selective recomputation + sequence parallelism cuts activation memory ~5× and drops the overhead to ~2–4%; for 530B they report 29.7% of achievable-FLOPs-utilization gain over full recompute.
- **The optimum is very flat near the knee, then a cliff.** Empirically, on transformer stacks $\rho$ rises slowly as $\kappa$ goes 1→4 and steeply past the point where the remaining resident set is dominated by non-recomputable state.
- **Heuristics are close to the ILP on chains.** `rotor` and DTR both land within a few percent of the exact solution on sequential models; DTR reports near-Checkmate performance without a static graph (ICLR 2021).
- **The kernel can dominate the schedule.** FlashAttention (Dao et al., NeurIPS 2022) makes attention recomputation nearly free by never materializing the $N\times N$ matrix — changing $m_v$ by $O(N)$ and thereby moving the optimum, not just the objective value.

## 5. What Is Not Known

- **Theoretically open.** No tight approximation-ratio bound for rematerialization on general heterogeneous DAGs. NP-completeness is proved (Checkmate, MLSys 2020); whether a constant-factor polynomial-time approximation exists is unresolved. The joint recompute + offload + parallelism-degree problem has no known exact algorithm at all.
- **Empirically open.** Nobody has published the exact-optimal-versus-shipped-heuristic gap at $\ge$70B parameters on a modern node (H100/H200/B200) with FSDP or 3D parallelism. All exact-solver comparisons are single-device and mostly pre-2022 vision models. The experiment is runnable; the ILP is the bottleneck, not the hardware.
- **Methodologically blocked.** "Memory used" is not a well-defined scalar under a caching allocator with fragmentation and async collectives, and $c_v$ is undefined inside a fused kernel. Until the objective is defined over fusion groups and reserved (not allocated) bytes, an "optimal" schedule is optimal for a model of the machine that the machine does not implement.

## 6. Why It Is Hard

**Confounded measurement, in two specific places.**

1. *The memory constraint is not the quantity the optimizer constrains.* The solver bounds $\sum_{v\in R_t} m_v$; the OOM is triggered by reserved segments. A schedule that is feasible at 78 GB logical can OOM at 80 GB reserved because freeing a 2 MB tensor does not return a usable 40 MB block. The residual is workload-dependent and not modeled by any published rematerialization solver.
2. *The cost model assumes node-level decomposability that compilers destroy.* Under `torch.compile`/Triton fusion, the graph the solver sees is not the graph that runs. Recomputing "one node" may force materialization of a whole fusion group's inputs, so the realized $\rho$ diverges from the planned one — and the divergence is in the direction that makes measured results look like heuristic noise.

Add compute cost as a secondary obstruction: the ILP has $O(n^2)$ binary variables, so exact solutions are out of reach for graphs with $n$ in the thousands, which is every real transformer training step after decomposition.

## 7. Current Research (as of 2026)

- **Selective, analytic policies.** Megatron-LM/NVIDIA and the DeepSpeed line have converged on hand-derived rules (recompute attention core, keep MLP) rather than solvers — closed-form, robust, and provably suboptimal by an unmeasured margin.
- **Compiler-integrated rematerialization.** PyTorch's `min-cut` partitioner in AOTAutograd chooses saved-versus-recomputed tensors inside the compiled graph, using a max-flow formulation over the joint forward/backward graph. This is the first widely deployed policy that operates on the post-fusion graph.
- **Joint recompute + offload + sharding.** Beaumont et al. and follow-ons on combining rematerialization with CPU/NVMe offloading; with HBM3e capacity rising and NVLink/PCIe bandwidth not keeping pace, the crossover point between "recompute" and "fetch from host" is moving. *(frontier — verify)*
- **Long-context regimes.** With sequence parallelism and ring/context parallelism, activation memory scales with $S/P$ and communication competes with recompute for the same overlap slots; the correct objective becomes a scheduling problem with a critical path, not a knapsack. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** How much step time do deployed heuristics leave on the table versus the exact optimum, at LLM scale, on the post-fusion graph?

**Scale.** Llama-3-8B-class dense transformer (32 layers, $d=4096$), sequence length 8192, one 8×H100 node, FSDP + `torch.compile`, batch size fixed at the largest value that fits with full recomputation.

**Arms.**
1. *Control:* Megatron-style selective recomputation (attention only), the current default.
2. Full recomputation (upper bound on $\rho$, lower bound on memory).
3. AOTAutograd min-cut partitioner, default settings.
4. *Treatment:* Checkmate-style ILP solved on the **post-fusion** graph (~2–4k nodes after layer-level coarsening), with $m_v$ set to allocator block sizes and $c_v$ measured per fusion group by clock-locked isolated timing; solved offline, replayed at runtime.

**Decider.** Median step time (ms/step over 200 steps, first 20 discarded) at matched peak *reserved* memory within ±1%. **If arm 4 beats arm 1 by <3%, the method variant is closed for dense transformers and effort should move to the joint offload/parallelism problem. If the gap is >10%, the deployed heuristics are leaving a full accelerator-generation of throughput unclaimed.** Secondary number: predicted-versus-measured peak reserved bytes for arm 4 — if the prediction error exceeds 5%, the measurement variant is confirmed as the binding blocker.

## 9. Key References

- **[Foundational]** Andreas Griewank, Andrea Walther. *Algorithm 799: revolve — an implementation of checkpointing for the reverse or adjoint mode of computational differentiation.* ACM Transactions on Mathematical Software, 26(1), 2000.
- **[Foundational]** Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin. *Training Deep Nets with Sublinear Memory Cost.* 2016. — arXiv:1604.06174
- **[SOTA/theory]** Paras Jain, Ajay Jain, Aniruddha Nrusimha, Amir Gholami, Pieter Abbeel, Kurt Keutzer, Ion Stoica, Joseph E. Gonzalez. *Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization.* MLSys 2020. — arXiv:1910.02653
- **[SOTA/theory]** Olivier Beaumont, Lionel Eyraud-Dubois, Julien Herrmann, Alexis Joly, Alena Shilova. *Optimal checkpointing for heterogeneous chains: how to train deep neural networks with limited memory.* NeurIPS 2019.
- **[SOTA/systems]** Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, Bryan Catanzaro. *Reducing Activation Recomputation in Large Transformer Models.* MLSys 2023. — arXiv:2205.05198
- **[SOTA/systems]** Marisa Kirisame, Steven Lyubomirsky, Altan Haan, Jennifer Brennan, Mike He, Jared Roesch, Tianqi Chen, Zachary Tatlock. *Dynamic Tensor Rematerialization.* ICLR 2021. — arXiv:2006.09616
- **[Related]** Aashaka Shah, Chao-Yuan Wu, Jayashree Mohan, Vijay Chidambaram, Philipp Krähenbühl. *MONeT: Memory Optimization for Deep Networks.* ICLR 2021. — arXiv:2010.14501
- **[Related]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. — arXiv:2205.14135
- **[Related]** Mitsuru Kusumoto, Takuya Inoue, Gentaro Watanabe, Takuya Akiba, Masanori Koyama. *A Graph Theoretic Framework of Recomputation Algorithms for Memory-Efficient Backpropagation.* NeurIPS 2019.

## 10. Worked Example

A 32-layer transformer, $d=4096$, sequence $S=8192$, micro-batch $b=1$, bf16, no tensor parallelism. Per-layer activation memory without recomputation, using the Megatron accounting $sbh(34 + 5\,aS/h)$ bytes with $a=32$ heads:

- Term 1: $34 \cdot 8192 \cdot 1 \cdot 4096 \approx 1.14$ GB.
- Term 2 (attention scores): $5 \cdot 32 \cdot 8192^2 \approx 10.7$ GB.
- Total $\approx 11.9$ GB per layer; ×32 layers $\approx 380$ GB. Does not fit on an 80 GB H100.

Three policies:

| Policy | Activation memory | Extra FLOPs | Predicted $\rho$ |
|---|---|---|---|
| None | 380 GB | 0 | 1.00 |
| Full recompute | ~2 GB (layer inputs only) | +1 forward | 1.33 |
| Selective (attention only) | ~36 GB | attention only | ~1.02 |

The analytic model says selective recompute buys a 10× memory cut for ~2% time — matching Korthikanti et al.

Now the obstruction. Replace attention with FlashAttention. The $10.7$ GB score matrix is never materialized, so term 2 vanishes and per-layer memory falls to ~1.14 GB, ~36 GB total — the same figure selective recomputation was reaching for, at $\rho = 1.0$. The solver's input $m_v$ for the attention node was wrong by a factor of ten, and the "optimal" policy it derived is now optimal for a machine that no longer exists. Worse, the remaining 36 GB is dominated by many small elementwise tensors that the allocator rounds up; measured `max_memory_reserved` on this configuration typically runs 8–15% above `max_memory_allocated`, which is larger than the 3% margin the experiment in §8 is trying to resolve.

The tradeoff curve is not a property of the model. It is a property of the model, the kernel library, and the allocator jointly — and only the first of those is in any published optimizer's input.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*