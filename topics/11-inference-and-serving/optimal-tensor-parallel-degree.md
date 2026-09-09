---
id: 11-inference-and-serving/optimal-tensor-parallel-degree
title: "Optimal Tensor-Parallel Degree at Fixed Latency Target"
topic: 11-inference-and-serving
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Tensor-Parallel Degree at Fixed Latency Target

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/optimal-tensor-parallel-degree` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed pool of $N$ accelerators, a model, a request arrival process, and a latency service-level objective (SLO), choose the tensor-parallel degree $t$ — how many GPUs one model replica is sharded across — that maximises SLO-attaining throughput. The pool then holds $N/t$ replicas. Larger $t$ cuts per-token latency (each GPU reads $1/t$ of the weights) but adds two all-reduce collectives per layer and shrinks each GEMM, so the speedup is sublinear; and it cuts the replica count, so aggregate throughput can fall.

Three variants, of different difficulty:

- **Measurement.** Given a deployed $(t, \text{batching policy})$ pair, report its goodput at the SLO. Solved in principle, contaminated in practice (§6).
- **Method.** Pick $t$ for a new model/hardware/SLO triple *without* sweeping all of them. Currently done by sweeping. This is the open problem.
- **Theory.** Prove that goodput as a function of $t$ is unimodal (single-peaked), so a $\log$-time search over $t \in \{1,2,4,8,\ldots\}$ is exact. No proof either way exists.

Solving it means: a predictor that, from hardware specs and model shape alone, names the goodput-maximising $t$ correctly on held-out (model, GPU, SLO) triples, with a stated error bar on the goodput at the chosen $t$.

## 2. Formal Setting

Model: $L$ layers, hidden size $h$, $n_q$ query heads, $n_{kv}$ KV heads, head dim $d$, FFN intermediate $h_f$, parameter count $P$, weight precision $b_w$ bytes. Hardware per GPU: HBM bandwidth $\beta$ (B/s), dense FLOP rate $\phi$, and an intra-node interconnect characterised by latency $\alpha$ (s) and per-GPU bandwidth $\beta_c$.

**Decode step time** at TP degree $t$, batch $B$, mean context $S$:

$$
T_{\text{dec}}(t,B) \;=\; \underbrace{\frac{b_w P + 2 b_{kv} B S n_{kv} d L}{t\,\beta\,\eta_{\text{mem}}(t,B)}}_{\text{memory}} \;+\; \underbrace{\frac{2PB}{t\,\phi\,\eta_{\text{flop}}(t,B)}}_{\text{compute}} \;+\; \underbrace{2L\Big[\alpha\log_2 t + \tfrac{2(t-1)}{t}\cdot\tfrac{2Bh}{\beta_c}\Big]}_{\text{2 all-reduces/layer}} \;+\; L\,\kappa(t)
$$

- $\eta_{\text{mem}}, \eta_{\text{flop}} \in (0,1]$: **achieved** fractions of peak, measured by dividing bytes moved / FLOPs issued by wall-clock kernel time from a profiler (Nsight, `torch.profiler`). They are functions of $t$ because sharding shrinks the GEMM's output dimension to $h/t$ or $h_f/t$.
- $\kappa(t)$: per-layer fixed overhead (kernel launch, Python/scheduler dispatch, collective enqueue), measured as the intercept of $T_{\text{dec}}$ regressed on $B$ at $B \to 0$.
- **TTFT** (time to first token) and **TPOT** (time per output token) are the two SLO axes. Goodput:

$$
G(t) \;=\; \frac{N}{t}\cdot \max\Big\{ \lambda \;:\; \Pr[\text{TTFT} \le \tau_1] \ge q \;\wedge\; \Pr[\text{TPOT} \le \tau_2] \ge q \Big\}
$$

measured as the sustained arrival rate $\lambda$ (req/s per replica) at which the $q$-th percentile (typically $q=0.9$ or $0.99$) of both latencies still meets targets $\tau_1,\tau_2$. The decision predicate is $t^\star = \arg\max_t G(t)$.

**Assumptions, and which are violated.**
1. *All-reduce cost is a ring/tree closed form.* Violated: NCCL switches algorithms by message size, and at small $Bh$ the collective is latency-bound and jitter-dominated.
2. *$\eta_{\text{mem}}$ is constant in $t$.* Violated — this is the crux. Sharding narrows GEMMs and reduces arithmetic intensity per kernel.
3. *Batch size is fixed.* Violated: continuous batching (Orca, vLLM) makes $B$ a random variable coupled to $\lambda$ and to the KV cache capacity, which itself depends on $t$.
4. *Uniform GPUs, no contention.* Violated on shared clusters and with NUMA/PCIe-attached nodes.
5. *One $t$ serves both phases.* Violated: prefill is compute-bound, decode memory-bound; their optimal $t$ differ (DistServe, Splitwise).

## 3. State of the Art

**Systems/empirical SOTA.** The operative practice is a sweep: enumerate $t \in \{1,2,4,8\}$ (and pipeline degree), benchmark each under the target workload, pick the winner. vLLM (Kwon et al., SOSP 2023), TensorRT-LLM, and SGLang all expose $t$ as a user-set flag with no automatic selection. Vidur (Agrawal et al., MLSys 2024) is the strongest published attempt to replace the sweep with simulation: it fits per-operator latency models from profiled traces and searches the deployment configuration space, reporting inference latency error under 9% and a large reduction in search cost. This is *established* for the models and GPUs it was fit on; **not established** is transfer to an unprofiled architecture or GPU generation — the operator models are fit per hardware.

Alpa (Zheng et al., OSDI 2022) solves intra-/inter-operator parallelism placement by ILP+DP for *training*; its cost model is throughput-oriented and does not carry a tail-latency SLO. AlpaServe (Li et al., OSDI 2023) is the clearest statement that model parallelism is a *latency* tool: it reports serving up to 10× tighter SLOs (or ~2.3× higher rates) at the same attainment by using model parallelism for statistical multiplexing rather than raw speed.

**Theory SOTA.** There is none specific to this problem. The relevant results are generic: the Amdahl/Hockney $\alpha$–$\beta$ collective model, and the roofline argument in Pope et al. (*Efficiently Scaling Transformer Inference*, MLSys 2023), which derives partitioning layouts analytically for TPU and shows they match measurement. No paper proves unimodality of $G(t)$.

**Claimed but unablated.** Vendor tables recommending "TP=8 for 70B, TP=4 for 34B" are benchmark numbers at one batch size and one input/output length mix; the $t$ ranking is not shown to be invariant to $(\tau_1,\tau_2,\lambda,S)$, and it is not.

## 4. What Is Known

- **Sublinearity is real and quantified at TPU scale.** Pope et al. (MLSys 2023) report PaLM 540B decode at 29 ms/token latency in a low-latency int8 configuration on 64 TPU v4 chips, versus a distinctly higher-throughput/higher-latency configuration on the same hardware — the same weights, different partitioning, order-of-magnitude different MFU.
- **The two phases want different $t$.** DistServe (Zhong et al., OSDI 2024) reports up to 7.4× more requests or 12.6× tighter SLOs by disaggregating prefill and decode onto separately configured resources; Splitwise (Patel et al., ISCA 2024) reports the same qualitative result on A100/H100 clusters. Established.
- **Batching policy changes the answer.** Sarathi-Serve (Agrawal et al., OSDI 2024) shows chunked prefill removes decode stalls and shifts the achievable throughput-at-latency frontier by up to ~5.6× serving capacity on Falcon-180B across 8 A100s — a shift large enough to reorder $t$ candidates that a fixed-batch model would rank confidently.
- **GQA changes the KV term.** Ainslie et al. (EMNLP 2023) reduce KV bytes by $n_q/n_{kv}$ (8× for Llama-3-70B), which shrinks the $S$-dependent part of the memory term and makes weight reads dominate — pushing $t^\star$ upward for latency-tight SLOs.
- **Collective overhead per layer is microseconds, not milliseconds.** On NVLink-4 (450 GB/s/GPU unidirectional), an all-reduce of $2Bh$ bytes at $B{=}32$, $h{=}8192$ moves under 1 MB; transfer is ~2 µs and fixed cost ~5–10 µs dominates. Measured routinely by `nccl-tests`.

## 5. What Is Not Known

- **Empirically open.** Whether $t^\star$ can be predicted from $(P, h, L, n_{kv}, \beta, \phi, \alpha, \beta_c, \tau_1, \tau_2)$ without a per-hardware profiling pass. The experiment is a full factorial sweep over ~4 model sizes × 3 GPU types × 4 TP degrees × 4 SLO points — hundreds of GPU-hours, entirely runnable, and no published paper reports it as a single controlled grid.
- **Theoretically open.** Unimodality of $G(t)$ in $t$. If $G$ has two local maxima (plausible: a KV-capacity cliff at the $t$ where the working set first fits in HBM, plus a communication-dominated decline later), then greedy/bisection search over $t$ is unsound. No proof, no counterexample published.
- **Methodologically blocked.** $\eta_{\text{mem}}(t,B)$ and $\kappa(t)$ are not separately identifiable from end-to-end step time. Both are decreasing-return terms that grow in importance as $t$ rises; a single wall-clock number cannot attribute the sublinearity between "kernels got narrower" and "we launched more of them". Until they are separated, the analytic model cannot be corrected in the right place.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability plus a confounded control**.

- *Non-identifiability.* The three sublinear terms — collective fixed cost $\alpha \log t$, degraded $\eta_{\text{mem}}(t)$, and launch overhead $\kappa(t)$ — all scale roughly with $L$ and are all weakly dependent on $B$. Fitting them from end-to-end latency sweeps yields a valid fit with wrong coefficients; the fit then mispredicts on a model with different $L/h$ ratio.
- *Confounded control.* Changing $t$ simultaneously changes KV cache capacity per replica, which changes the batch size continuous batching reaches, which changes the operating point on the roofline. Comparing TP=4 and TP=8 "at the same load" therefore compares two different batch-size distributions. Holding $B$ fixed instead makes the comparison clean but no longer answers the deployment question.
- *Cost.* Each grid cell needs a multi-minute steady-state run at a controlled arrival rate to get a stable $p99$; a $4\times3\times4\times4$ grid with 3 seeds is a multi-thousand-GPU-hour experiment on hardware that is scarce precisely because it is serving.

## 7. Current Research (as of 2026)

- **Simulation-first configuration search.** Vidur (Microsoft Research India — Agrawal, Ramjee et al.) is the reference line; follow-on work extending fitted operator models to unseen GPUs is active *(frontier — verify)*.
- **Disaggregated serving** (Peking University / UCSD — DistServe; Microsoft Azure — Splitwise; and the Mooncake system from Moonshot AI, FAST 2025) makes $t$ a per-phase decision, which raises the search dimension while making each sub-decision cleaner.
- **Throughput-maximal single-node pipelines.** NanoFlow (Zhu et al., University of Washington, arXiv:2408.12757) argues intra-device operator overlap, not TP degree, is the binding constraint at high load — if right, it lowers $t^\star$ for throughput-oriented SLOs *(frontier — verify)*.
- **Sequence/expert parallelism interaction.** For MoE models the tensor-parallel/expert-parallel split reopens the same question with a different cost structure; published guidance is vendor benchmark tables, not ablation.

## 8. Concrete Next Experiment

**Scale.** One 8×H100 SXM node (NVLink-4). Model: Llama-3.1-70B, FP16 (140 GB weights, $L{=}80$, $h{=}8192$, $n_{kv}{=}8$). Serving stack: vLLM with chunked prefill. Workload: ShareGPT-derived trace, replayed as a Poisson arrival process at a swept rate $\lambda$.

**Arms.** $t \in \{2,4,8\}$ with $N/t \in \{4,2,1\}$ replicas, so total GPUs is 8 in every arm. SLO grid: $\tau_2 \in \{20, 40, 80\}$ ms TPOT at $q{=}0.9$, $\tau_1 = 2$ s TTFT.

**Control arm.** $t{=}8$, single replica — the current default deployment for 70B on one node. Report every other arm as a ratio to it. A second control isolates the confound: rerun each $t$ with the KV cache size *artificially capped* to the TP=2 per-replica capacity, so batch-size distributions match across arms.

**Deciding number.** The **goodput ratio $G(t{=}4)/G(t{=}8)$ at $\tau_2 = 40$ ms**, in SLO-attaining requests/second per node, with 3 seeds and a bootstrap 95% interval. If that ratio exceeds 1.10 at any $\tau_2$ in the grid, the field-default "shard as wide as the node" rule is wrong at that SLO and the sweep is not optional. If the ratio stays within $[0.95, 1.05]$ across the whole grid, $t$ is a second-order knob and the problem downgrades. The capped-KV control arm tells you whether any observed difference came from capacity or from kernel efficiency.

## 9. Key References

- **[Foundational]** Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* arXiv, 2019. — arXiv:1909.08053
- **[Foundational]** Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean. *Efficiently Scaling Transformer Inference.* MLSys, 2023. — arXiv:2211.05102
- **[SOTA]** Amey Agrawal, Nitin Kedia, Jayashree Mohan, Ashish Panwar, Nipun Kwatra, Bhargav Gulavani, Ramachandran Ramjee, Alexey Tumanov. *Vidur: A Large-Scale Simulation Framework for LLM Inference.* MLSys, 2024. — arXiv:2405.05465
- **[SOTA]** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI, 2024. — arXiv:2401.09670
- **[SOTA]** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, Ramachandran Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024. — arXiv:2403.02310
- **[SOTA]** Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. *AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving.* OSDI, 2023. — arXiv:2302.11665
- **[Foundational]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Foundational]** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI, 2022.
- **[SOTA]** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA, 2024. — arXiv:2311.18677
- **[SOTA]** Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica. *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.* OSDI, 2022. — arXiv:2201.12023
- **[Survey]** Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Hongyi Jin, Tianqi Chen, Zhihao Jia. *Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems.* arXiv, 2023. — arXiv:2312.15234

## 10. Worked Example

Llama-3.1-70B, FP16, 8×H100 SXM: $\beta = 3.35$ TB/s, NVLink $\beta_c = 450$ GB/s, $L=80$, $h=8192$, weights 140 GB. Take $B=32$, GQA so the KV term is small; ignore it for the arithmetic.

**Weight-read time per decode step**, assuming $\eta_{\text{mem}} = 0.85$ at both degrees:

| $t$ | bytes/GPU | ideal read | at $\eta{=}0.85$ | all-reduce (160×) | step | tok/s/replica | replicas | node tok/s |
|---|---|---|---|---|---|---|---|---|
| 4 | 35 GB | 10.4 ms | 12.3 ms | 160×7 µs = 1.1 ms | 13.4 ms | 75 | 2 | 150 |
| 8 | 17.5 GB | 5.2 ms | 6.1 ms | 160×8 µs = 1.3 ms | 7.4 ms | 135 | 1 | 135 |

Read the table naively: at a 20 ms TPOT SLO both arms pass, and TP=4 wins on node throughput, 150 vs 135 tok/s. Recommendation: TP=4.

Now relax assumption 2. At $t=8$ the column-parallel FFN GEMM has output width $h_f/8 = 3584$ and the row-parallel projection has $h/8 = 1024$ — narrow enough that the kernel is no longer bandwidth-saturating. Suppose $\eta_{\text{mem}}$ falls to 0.65 at $t{=}8$ and holds at 0.85 at $t{=}4$. TP=8 step time becomes $17.5/3.35/0.65 = 8.0$ ms $+\,1.3 = 9.3$ ms, i.e. 108 tok/s. TP=4 still wins on throughput but its *latency* margin narrows less than expected, and at a 10 ms TPOT SLO the TP=8 arm now **fails** where the first table said it passed by 26%.

The obstruction is visible here: the entire decision flipped on $\eta_{\text{mem}}(8) \in \{0.85, 0.65\}$ — a quantity nobody publishes, that varies by kernel library version, and that cannot be recovered from an end-to-end step-time measurement because a 1.9 ms increase is equally consistent with $\kappa(8)$ being 24 µs/layer instead of 8. Both explanations fit the same wall-clock number and extrapolate to opposite recommendations on a model with different $L/h$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*