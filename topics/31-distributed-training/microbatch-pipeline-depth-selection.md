---
id: 31-distributed-training/microbatch-pipeline-depth-selection
title: "Optimal Micro-Batch and Pipeline Depth Co-Selection"
topic: 31-distributed-training
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Micro-Batch and Pipeline Depth Co-Selection

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/microbatch-pipeline-depth-selection` · **Status:** partially-solved

## 1. Problem Statement

Pipeline-parallel training splits a model over $p$ stages and splits each optimizer step's batch into $m$ micro-batches of size $b$. The two knobs interact: raising $m$ shrinks the pipeline bubble but shrinks $b$ (hurting per-GPU kernel efficiency) or raises the global batch $B$ (hurting sample efficiency). Raising $p$ buys memory headroom and lets you shard a bigger model, but costs bubble and stage-boundary communication.

**Input.** A model graph, a cluster (device count $N$, per-device memory $M$, intra- and inter-node bandwidths), a token budget $D$, and a target loss $L^\*$.

**Output.** A configuration $(b, p, d, t, v, \text{recompute policy})$ — micro-batch size, pipeline depth, data-parallel width, tensor-parallel width, interleaved virtual stages.

**Objective.** Minimize wall-clock time to reach $L^\*$, not throughput. Throughput-optimal and time-to-loss-optimal configurations differ whenever the throughput-optimal choice pushes $B$ past the critical batch size.

Three variants, of very different difficulty:

- **Measurement:** given a candidate $(b,p)$, predict iteration time and peak memory to within a few percent without running it. Largely solved for fixed schedules; profile-plus-analytic models are accurate.
- **Method:** search $(b,p,d,t,v)$ efficiently. Solved for the throughput objective under a fixed $B$ (Alpa, Piper, Galvatron). Not solved when $B$ is itself a decision variable.
- **Theory:** prove the joint optimum, or bound the loss from decoupling the two choices. Open.

## 2. Formal Setting

Let $N = d \cdot p \cdot t$ be the device count. Per data-parallel replica the number of micro-batches is
$$m = \frac{B}{b \cdot d},$$
with $B$ the global batch in sequences and $b$ the micro-batch size in sequences per replica. *Measured as:* $B$ is read off the data loader; $m$ is the count of forward passes between optimizer steps, verifiable from a profile trace.

**Bubble fraction.** For the 1F1B (one-forward-one-backward) schedule with equal stage times,
$$\beta_{1F1B} = \frac{p-1}{m},$$
and for the interleaved schedule with $v$ virtual stages per device (Narayanan et al., SC 2021),
$$\beta_{\text{int}} = \frac{1}{v}\cdot\frac{p-1}{m}.$$
*Measured as:* $\beta = 1 - (\text{summed per-stage busy time}) / (p \cdot T_{\text{iter}})$ from a device trace, which folds in imbalance and communication stalls that the formula omits.

**Iteration time.** With $t_f(b), t_b(b)$ the per-stage forward/backward times and $c(b)$ the activation transfer time per stage boundary,
$$T_{\text{iter}}(b,p,m,v) \approx (m + p/v - 1)\,\big(t_f(b)+t_b(b)\big) + 2(p-1)\,c(b) + T_{\text{allreduce}}(d).$$
*Measured as:* median step latency over $\geq 50$ steady-state steps, excluding warmup.

**Memory.** Stage $i$ holds up to $p - i$ in-flight micro-batch activation sets under 1F1B:
$$M_i \approx \underbrace{\frac{P}{p\,t}\,\kappa}_{\text{weights, grads, optimizer}} + (p-i)\,A(b,\ell/p) + M_{\text{frag}} \le M.$$
*Measured as:* `torch.cuda.max_memory_allocated()` per rank; $M_\text{frag}$ (allocator fragmentation, typically 3–10%) is the term prediction models most often get wrong.

**Time to loss.** Let $S(B)$ be optimizer steps to reach $L^\*$ at batch $B$. The objective is
$$\min_{b,p,d,t,v}\; S(B)\cdot T_{\text{iter}}(b,p,m,v)\quad\text{s.t. } M_i \le M,\ dpt = N.$$
Under the noise-scale model of McCandlish et al. (2018), $S(B) \approx S_{\min}(1 + B_{\text{crit}}/B)$, so total examples processed grows once $B > B_{\text{crit}}$.

**Assumptions, and which are violated.**
1. *Equal stage times* — violated: embedding and LM-head stages are heavier; Llama-3-class runs use uneven layer assignment.
2. *$t_f, t_b$ linear in $b$* — violated below the GEMM saturation point; at $b=1$ with short sequences, small-$m$ kernels run at a fraction of peak.
3. *$B_{\text{crit}}$ constant* — violated: it grows through training.
4. *Communication overlaps compute* — partially violated; overlap quality depends on the schedule and on NIC contention with the data-parallel all-reduce.

## 3. State of the Art

**Established (ablated, reproduced).**
- The $(p-1)/m$ bubble law and its $1/v$ interleaved reduction (Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*, SC 2021). The paper ablates $p$, $t$, $m$, and interleaving separately and reports 502 pFLOP/s on 3072 A100s (~52% of peak) for a 1T-parameter model.
- Selective activation recomputation cuts activation memory ~5× with ~2% throughput cost at 530B scale (Korthikanti et al., MLSys 2023). This directly changes the feasible $(b,p)$ region, so recompute policy cannot be chosen after $(b,p)$.
- Automated joint search beats hand tuning: Alpa (Zheng et al., OSDI 2022) matches or exceeds hand-written Megatron/DeepSpeed configurations on models it was not specialized for; Piper (Tarnawski et al., NeurIPS 2021) gives an exact DP-based planner over $(p, b, t)$ under memory constraints.
- Zero Bubble PP (Qi et al., ICLR 2024) splits the backward pass into input-grad and weight-grad halves; reported up to 23% throughput over 1F1B at matched memory.

**Claimed but unablated.** That schedule-level bubble elimination (zero-bubble, DualPipe in DeepSeek-V3, Hanayo's wave schedule, SC 2023) makes pipeline depth "free" enough that $p$ can be chosen purely from memory. The throughput numbers are real; the claim that the optimal $b$ is unchanged under these schedules has not been ablated.

**Benchmark-number-only.** Most large-run configuration tables (Megatron, DeepSeek-V3, Llama 3) report the chosen $(b,p,t,d)$ and the achieved MFU. They are not sweeps: no counterfactual arm is reported, so they establish feasibility, not optimality.

## 4. What Is Known

- **Bubble law holds to first order.** At $p=8$, $m=8$: $\beta = 7/8 = 0.875$ predicted idle *if the pipeline never refills* — the operative form is $m/(m+p-1) = 0.53$ utilization. Megatron SC'21 measures interleaving gains of roughly 10% at small $m$, vanishing as $m \gtrsim 4p$.
- **Empirical rule of thumb $m \geq 4p$** keeps $\beta \le 25\%$; most production runs sit at $m/p$ between 4 and 16.
- **Tensor parallelism should not cross node boundaries.** Megatron SC'21 shows $t \le 8$ (one NVLink domain) dominates; beyond that pipeline parallelism is preferred. This is one of the few robustly reproduced cross-dimension results.
- **Critical batch size scales primarily with data, weakly with parameters.** Zhang, Morwani et al. (*How Does Critical Batch Size Scale in Pre-training?*, ICLR 2025) fit $B_{\text{crit}}$ over models up to ~1.2B params trained on C4 and find dependence on tokens seen dominating parameter count. Shallue et al. (JMLR 2019) established the earlier, workload-dependent shape of the steps-vs-batch curve at ResNet/Transformer scale.
- **Uneven stage assignment matters.** Llama 3 (Grattafiori et al., 2024) reports reducing the first and last stage's layer counts to fix the embedding/head imbalance — evidence the equal-stage assumption fails at production scale.

## 5. What Is Not Known

- **Theoretically open.** No approximation guarantee for the joint problem when $B$ is a decision variable. Existing planners (Piper, Alpa) are exact or near-exact for fixed $B$; adding $S(B)$ makes the objective non-separable across the DP and PP dimensions, and no hardness result or approximation bound is published for that version.
- **Theoretically open.** Whether decoupling — pick $B$ from a noise-scale schedule, then optimize $(b,p)$ for throughput at that $B$ — is bounded-loss. Practitioners do this universally; nobody has bounded the regret.
- **Empirically open.** A full $(b,p)$ grid run to a fixed target loss, rather than to fixed steps, at $\geq 7$B parameters. Every published sweep measures throughput; the sample-efficiency arm is missing.
- **Empirically open.** Whether zero-bubble-class schedules shift the optimal $b$. Runnable on 64 GPUs; unrun.
- **Methodologically blocked.** $B_{\text{crit}}$ has no agreed estimator. Noise-scale (McCandlish), the branch point of the steps-vs-batch curve (Shallue), and the loss-matched-at-fixed-tokens definition (Zhang et al.) give different numbers on the same run, and the gap between them is comparable to the effect size the co-selection question turns on.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by an objective whose ground truth costs a full training run**.

- Changing $b$ changes three things at once: kernel efficiency, activation memory (hence feasible recompute policy), and — if $d$ and $m$ are held fixed — the global batch. Any single throughput delta is a sum of these; the published sweeps do not disentangle them.
- The real objective, $S(B)\cdot T_{\text{iter}}$, requires knowing $S(B)$, which requires training to $L^\*$. At 7B a single arm is $\sim$10$^{21}$ FLOPs; a $6\times6$ grid is a frontier-lab-scale budget. Proxy runs at 100M–1B do not transfer because $B_{\text{crit}}$ moves with token count.
- The evaluation in near-universal use — MFU or tokens/sec — does not measure the named quantity. A configuration can win on MFU by raising $B$ past $B_{\text{crit}}$ and lose on time-to-loss.

## 7. Current Research (as of 2026)

- **Schedule design to remove $p$ from the objective:** zero-bubble and its V-shaped memory-efficient variants (Qi et al., Sea AI Lab / NUS), DualPipe (DeepSeek), wave-like schedules (Hanayo, SC 2023). Direction: if $\beta \to 0$, $p$ is chosen by memory alone and the coupling weakens.
- **Auto-parallelization planners** extended to MoE and long context: Alpa lineage, Galvatron (Miao et al., VLDB 2023), Megatron's built-in heuristics. *(frontier — verify)* Recent planners reportedly add a batch-size term to the search objective; published evidence remains throughput-only.
- **Batch-size scaling laws** as first-class inputs to system planning — Harvard/Kempner (Morwani, Brandfonbrener, Kakade) and DeepMind-lineage work on optimal batch/LR co-scaling.
- **Long-context regimes** where sequence parallelism and $b=1$ are forced, collapsing the micro-batch axis and shifting the whole problem to the $(p, v, \text{context-parallel})$ subspace. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does the throughput-optimal $(b,p)$ equal the time-to-loss-optimal $(b,p)$?

**Scale.** A 7B dense decoder, 64 H100s (8 nodes, NVLink intra-node), $t=8$ fixed, trained to a fixed target loss on a 150B-token corpus (~4 arms/week at 40% MFU).

**Arms.** Nine configurations spanning $p \in \{2,4,8\}$ and $b \in \{1,2,4\}$, with $d$ set by $d = 64/(pt)$ and $B$ held **fixed** at 1024 sequences across all arms so that $m = B/(bd)$ varies but sample efficiency does not. Then a second block of three arms at the throughput-winning $(b,p)$ with $B \in \{512, 1024, 2048\}$.

**Control arm.** The standard practice configuration: $p$ minimal subject to memory fit, $b$ maximal subject to memory fit, $B$ from a noise-scale schedule.

**Deciding number.** $\Delta = \dfrac{T_{\text{loss}}(\text{best-throughput config})}{T_{\text{loss}}(\text{best time-to-loss config})} - 1$, wall-clock hours to reach $L^\*=$ the control arm's final loss. If $\Delta < 0.03$, decoupling is safe and the problem is a solved engineering search. If $\Delta > 0.15$, throughput-driven planners are systematically misconfiguring large runs and the joint objective must be adopted.

## 9. Key References

- **[Foundational]** Y. Huang, Y. Cheng, A. Bapna, et al. *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.* NeurIPS, 2019. — arXiv:1811.06965
- **[Foundational]** D. Narayanan, A. Harlap, A. Phanishayee, et al. *PipeDream: Generalized Pipeline Parallelism for DNN Training.* SOSP, 2019.
- **[SOTA]** D. Narayanan, M. Shoeybi, J. Casper, et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC, 2021. — arXiv:2104.04473
- **[SOTA]** P. Qi, X. Wan, G. Huang, M. Lin. *Zero Bubble Pipeline Parallelism.* ICLR, 2024. — arXiv:2401.10241
- **[SOTA]** L. Zheng, Z. Li, H. Zhang, et al. *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.* OSDI, 2022. — arXiv:2201.12023
- **[SOTA]** J. Tarnawski, D. Narayanan, A. Phanishayee. *Piper: Multidimensional Planner for DNN Parallelization.* NeurIPS, 2021.
- **[Foundational]** S. McCandlish, J. Kaplan, D. Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch Training.* Preprint, 2018. — arXiv:1812.06162
- **[Foundational]** C. J. Shallue, J. Lee, J. Antognini, et al. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 20(112), 2019.
- **[SOTA]** H. Zhang, D. Morwani, N. Vyas, et al. *How Does Critical Batch Size Scale in Pre-training?* ICLR, 2025. — arXiv:2410.21676
- **[SOTA]** V. Korthikanti, J. Casper, S. Lym, et al. *Reducing Activation Recomputation in Large Transformer Models.* MLSys, 2023. — arXiv:2205.05198
- **[Survey]** X. Miao, X. Nie, H. Zhang, et al. *Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism.* VLDB, 2023.

## 10. Worked Example

64 H100s, 7B model, $t=8$, sequence length 8192, global batch $B=1024$ sequences.

| $p$ | $d$ | $b$ | $m=B/(bd)$ | $\beta=(p-1)/m$ | utilization $m/(m+p-1)$ |
|---|---|---|---|---|---|
| 2 | 4 | 4 | 64 | 0.016 | 0.985 |
| 4 | 2 | 4 | 128 | 0.023 | 0.977 |
| 8 | 1 | 4 | 256 | 0.027 | 0.973 |
| 8 | 1 | 1 | 1024 | 0.007 | 0.993 |

By the bubble law, $p=2,b=4$ and $p=8,b=1$ look near-identical (98.5% vs 99.3%). Measured behavior diverges for reasons the law does not model:

- At $b=1$, the per-stage GEMM is $8192 \times d_{\text{model}}$ — enough to saturate an H100 at this sequence length, so the usual small-$b$ penalty is muted. At sequence length 512 it would not be, and the same table would mislead by ~20%.
- At $p=2, d=4$, the data-parallel all-reduce covers 7B params across 4 replicas; at $p=8, d=1$ it disappears entirely. The $\beta$ column does not contain this term at all.
- At $p=2$, stage-0 memory holds $(p-i)=2$ activation sets over $\ell/2 = 16$ layers at $b=4$; at $p=8$ it holds 8 sets over 4 layers. The products are comparable, but only the $p=8$ arm fits without full recomputation, and full recomputation costs ~30% throughput — a term that dwarfs the 0.8-point bubble difference.

Now the obstruction. Suppose the $p=8,b=1$ arm wins on tokens/sec by 6%. To use that, you must hold $B$ fixed at 1024. If instead you let $B$ float to 2048 to raise $m$ further, throughput rises another ~2% — but if $B_{\text{crit}} \approx 1100$ sequences at this token budget, $S(B)$ rises by roughly $(1 + 1100/2048)/(1 + 1100/1024) \approx 0.74$ per-step count against $2\times$ the tokens per step, i.e. ~1.48× total tokens, and time-to-loss gets *worse* by ~40% while every dashboard shows an improvement. Which side of that you land on depends on a $B_{\text{crit}}$ estimate whose three published definitions disagree by more than the 6% the systems choice is worth.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*