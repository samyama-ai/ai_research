---
id: 32-hardware-and-kernels/energy-optimal-serving-batch-size
title: "Energy-Optimal Batch Size for LLM Serving"
topic: 32-hardware-and-kernels
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Energy-Optimal Batch Size for LLM Serving

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/energy-optimal-serving-batch-size` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed model, fixed hardware, and a request stream with a latency service-level objective (SLO), choose the batch size (and, jointly, the clock frequency and replica count) that minimizes **energy per output token** subject to the SLO holding at a stated quantile.

Three variants, of different difficulty:

- **Measurement variant.** What is the true energy cost of serving one token at batch size $B$? Requires a defensible boundary (accelerator die? node? rack? wall plug including cooling?) and an attribution rule for shared static power. Currently the weakest link.
- **Method variant.** Given a working energy oracle, find an online controller that picks $B_t$ under a non-stationary arrival process. This is a constrained stochastic control problem, runnable today.
- **Theory variant.** Prove that energy per token is quasi-convex (or monotone) in $B$ under a roofline-plus-static-power model, so that a scheduler's local search is globally optimal. No proof either way exists.

Solving it means: a published controller that beats a well-tuned fixed-batch baseline in joules per token at equal SLO attainment and equal output quality, on hardware someone else can rent, with wall-plug energy reported.

## 2. Formal Setting

Model $\theta$ with $P$ parameters at $b$ bytes/parameter. Decode step at batch size $B$ moves weights $W = Pb$ bytes plus KV-cache $K(B, L) = 2 n_{\text{layer}} n_{\text{kv}} d_{\text{head}} b \sum_{i=1}^{B} L_i$ bytes for context lengths $L_i$.

**Step time**, measured as the wall-clock delta between two consecutive token emissions on the server, averaged over $\geq 10^3$ steps after warm-up:

$$T_{\text{step}}(B, f) \;=\; \max\!\left(\frac{W + K(B,L)}{\beta(f)}, \; \frac{2PB}{\gamma(f)}\right) + \tau_{\text{ov}}(B)$$

with $\beta$ the achieved HBM bandwidth (measured by a bandwidth microbenchmark at the same clock, not the spec sheet) and $\gamma$ the achieved FLOP/s. $\tau_{\text{ov}}$ is kernel-launch and scheduler overhead, measured as the residual after fitting the max-term.

**Power.** $\bar{p}(B,f)$ is average board power over the step window. Measured three ways, which do not agree: NVML/`nvidia-smi` device counters (accelerator only, ~100 ms internal averaging window), node-level BMC/IPMI or PDU telemetry (adds CPU, DRAM, NIC, fans, PSU loss), and facility wall-plug (adds cooling, via PUE). Define $P_{\text{idle}}$ as node power with the model resident and no active requests.

**Objective.**

$$E_{\text{tok}}(B,f) \;=\; \frac{\bar{p}(B,f)\, T_{\text{step}}(B,f)}{B_{\text{eff}}}, \qquad B_{\text{eff}} = \mathbb{E}[\text{tokens emitted per step}]$$

$B_{\text{eff}} < B$ whenever a slot is idle or the batch is padded. The optimization is

$$\min_{B,f} \; E_{\text{tok}}(B,f) \quad \text{s.t.} \quad \Pr[\text{TTFT} > s_1] \le \epsilon, \; \Pr[\text{TPOT} > s_2] \le \epsilon$$

with TTFT the time to first token and TPOT the inter-token latency, both at quantile $1-\epsilon$ (typically $p99$).

**Assumptions, and where they break.** (i) Roofline separability of prefill and decode — violated by chunked prefill and continuous batching, which interleave the two in one kernel launch. (ii) Power constant within a step — violated; decode power is bursty at millisecond scale, below NVML's sampling window. (iii) $L_i$ known at admission — violated; output length is unknown until EOS. (iv) Static power independent of $B$ — violated; leakage rises with die temperature, which rises with sustained utilization, so $P_{\text{idle}}$ measured cold understates it by several percent.

## 3. State of the Art

**Systems/empirical SOTA.** Continuous (iteration-level) batching — Orca (Yu et al., OSDI 2022) and vLLM's PagedAttention (Kwon et al., SOSP 2023) — raised achievable $B$ by removing head-of-line blocking and KV fragmentation; both report throughput, not energy. Sarathi-Serve (Agrawal et al., OSDI 2024) adds chunked prefill and stall-free batching, improving throughput at fixed TBT SLO. DynamoLLM (Stojkovic et al., 2024; HPCA 2025) is the closest direct attack: it reconfigures instance count, tensor parallelism, and GPU frequency by request-class and reports ~53% energy savings at SLO compliance. That number is a system-level benchmark result on a specific trace mix, not an ablation isolating the batch-size term from the frequency and parallelism terms.

**Established vs. claimed.** Established: larger decode batches amortize the weight read and reduce joules per token in the memory-bound regime. Claimed but unablated: that jointly tuned $(B, f)$ beats frequency capping alone by a large margin. Zeus (You, Chung, Chowdhury, NSDI 2023) showed batch-size/power-limit joint tuning matters for *training*; the inference analogue has not been separated cleanly.

**Measurement SOTA.** MLPerf Power (Tschand et al., MLSys 2025) is the only cross-vendor harness with an audited wall-plug measurement methodology for ML systems. Its inference power results are per-submission benchmark numbers under fixed scenarios — they do not sweep batch size at fixed SLO.

**Theory SOTA.** None specific. The nearest formal object is the roofline model (Williams, Waterman, Patterson, CACM 2009), which predicts the shape but not the static-power interaction.

## 4. What Is Known

- Decode at $B=1$ is memory-bandwidth bound with arithmetic intensity $\approx 2$ FLOP/byte, far below the ~200+ FLOP/byte ridge point of A100/H100 in fp16. Established by roofline analysis and reproduced across serving stacks.
- Samsi et al. (*From Words to Watts*, IEEE HPEC 2023) measured LLaMA-65B inference on V100 and A100 nodes and reported energy per token falling sharply as batching increases, with power capping (250 W vs 400 W on A100) costing modest latency for large energy savings — measured at single-node scale, batch sizes up to the low tens.
- Luccioni, Jernite, Strubell (*Power Hungry Processing*, FAccT 2024) measured inference energy across 88 models and 10 tasks on A100s: roughly $10^{-3}$–$10^{-2}$ kWh per 1,000 inferences for task-specific models, with generative tasks 1–3 orders of magnitude costlier. Scale: single A100, batch effects not swept.
- Splitwise (Patel et al., ISCA 2024) measured prefill and decode phases having distinct power profiles on A100/H100 and showed phase-splitting across machine pools cuts power ~20% at iso-throughput. Established at cluster-emulation scale.
- Perseus (Chung et al., SOSP 2024) established that in large-model *training* pipelines a large fraction of energy is "bloat" removable at zero throughput cost. Inference has no equivalent established number.

## 5. What Is Not Known

- **Empirically open (the core gap).** Nobody has published a clean sweep of $E_{\text{tok}}$ over $B \in \{1, \dots, 512\}$ crossed with GPU clock $f$, at fixed p99 TPOT, on a current-generation node, with wall-plug energy. Every ingredient is rentable today. The experiment is unrun at the right scale.
- **Theoretically open.** Whether $E_{\text{tok}}(B)$ is quasi-convex under the model in §2 with temperature-dependent leakage. If it is not, hill-climbing schedulers can land in local minima, and no published controller checks for this.
- **Methodologically blocked.** Attribution of static and shared power. There is no accepted convention for charging idle node power, cooling, or memory-refresh energy to a token. Two papers reporting "joules per token" that differ by 3x may both be right under their own boundary. MLPerf Power fixes a boundary for its own scenarios but is not applied to batch sweeps.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under confounded measurement**. Raising $B$ simultaneously (a) increases utilization, which raises dynamic power; (b) raises die temperature, which raises leakage and can trigger DVFS down-clocking; (c) increases KV-cache footprint, which changes the memory-bound term; and (d) increases queueing delay, which changes which requests are in flight. A single throughput-and-power trace cannot separate these four. NVML's ~100 ms averaging window is coarser than a decode step (5–30 ms), so per-step power is not directly observable — it is inferred from an average over a window containing a varying mix of prefill and decode steps.

Second obstruction: the SLO constraint is measured on a workload whose output-length distribution is itself dependent on the model and prompt mix, so "equal SLO attainment" across two configurations is not a controlled comparison unless the trace is replayed token-exactly.

## 7. Current Research (as of 2026)

- **Frequency- and phase-aware serving.** Microsoft Azure Research (DynamoLLM, Splitwise, *Towards Greener LLMs*) continues on request-classified reconfiguration and prefill/decode disaggregation.
- **Energy measurement tooling.** Michigan SymbioticLab (Zeus, Perseus, the ML.Energy leaderboard) publishes per-model inference energy at fixed configurations; extending it to SLO-constrained batch sweeps is the obvious next step *(frontier — verify)*.
- **Benchmark standardization.** MLCommons Power working group is extending audited power measurement to more inference scenarios *(frontier — verify)*.
- **Carbon-aware and heterogeneous placement.** Routing between accelerator generations by marginal joules per token, largely simulation-based so far *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** One 8×H100 SXM node (or 8×A100 80GB). Two models: Llama-3-8B (TP=1) and Llama-3-70B (TP=4), fp16 and fp8. Replay a fixed 100k-request trace with token-exact prompt and output lengths captured once and pinned, so every arm decodes identical token counts.

**Sweep.** $B \in \{1,2,4,8,16,32,64,128,256,512\}$ × SM clock $f \in \{$ 6 levels from 900 MHz to max $\}$, continuous batching on, chunked prefill on. Measure NVML device energy, node BMC energy, and PDU wall-plug energy simultaneously at 1 kHz where possible.

**Control arm.** Fixed $B = 256$ at stock clocks with default vLLM scheduling — the configuration a throughput-tuned deployment would ship.

**Deciding number.** Wall-plug joules per output token at the best feasible $(B,f)$ divided by the same quantity for the control arm, where "feasible" means p99 TPOT $\le 50$ ms and p99 TTFT $\le 2$ s. **If that ratio is below 0.75, batch-size selection is a first-class energy knob and the problem is live; if it is above 0.95, batching is already saturated by throughput tuning and the remaining energy lever is frequency and placement, not $B$.** Report the same ratio under the NVML boundary to quantify how much of the answer is a boundary artifact.

## 9. Key References

- **[Foundational]** Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM, 2009.
- **[Foundational]** Yu, Jeong, Kim, Kim, Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI, 2022.
- **[SOTA]** Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[SOTA]** Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov, Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024. — arXiv:2403.02310
- **[SOTA]** Stojkovic, Choukse, Zhang, Goiri, Torrellas. *DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency.* HPCA, 2025. — arXiv:2408.00741
- **[SOTA]** Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA, 2024. — arXiv:2311.18677
- **[Measurement]** You, Chung, Chowdhury. *Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training.* NSDI, 2023. — arXiv:2208.06102
- **[Measurement]** Chung, Gu, Jeong, Jang, Ye, Chowdhury. *Perseus: Reducing Energy Bloat in Large Model Training.* SOSP, 2024. — arXiv:2312.06902
- **[Measurement]** Tschand et al. *MLPerf Power: Benchmarking the Energy Efficiency of Machine Learning Systems from Microwatts to Megawatts for Sustainable AI.* MLSys, 2025. — arXiv:2410.12032
- **[Empirical]** Samsi, Zhao, McDonald, Li, Michaleas, Jones, Bergeron, Kepner, Tiwari, Gadepally. *From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference.* IEEE HPEC, 2023. — arXiv:2310.03003
- **[Survey]** Luccioni, Jernite, Strubell. *Power Hungry Processing: Watts Driving the Cost of AI Deployment?* ACM FAccT, 2024. — arXiv:2311.16863

## 10. Worked Example

Llama-2-7B, fp16, single A100 80GB (spec 2.0 TB/s HBM, ~1.6 TB/s achieved on a stream benchmark; 400 W TDP; ~70 W idle with weights resident). Weights $W = 13.5$ GB. Short contexts, so ignore KV traffic.

**Decode step time (memory-bound), $B \le 64$:** $T_{\text{step}} \approx 13.5/1600 = 8.4$ ms, roughly independent of $B$.

| $B$ | tok/s | board power (NVML) | J/token (NVML) | node power (+130 W) | J/token (node) |
|---|---|---|---|---|---|
| 1 | 119 | 160 W | 1.34 | 290 W | 2.44 |
| 8 | 952 | 210 W | 0.22 | 340 W | 0.36 |
| 64 | 7,620 | 330 W | 0.043 | 460 W | 0.060 |
| 256 | ~18,000 (compute-bound, $T_{\text{step}}\approx14$ ms) | 390 W | 0.022 | 520 W | 0.029 |

Read naively, energy per token falls monotonically — pick the largest $B$ that fits. Two things break that.

**First, the boundary changes the answer's shape.** Going $B{=}8 \to 64$ saves 5.1x under NVML but 6.0x under node power; going $B{=}64 \to 256$ saves 1.95x under NVML and 2.07x under node. The static term $P_{\text{idle}}$ dominates at small $B$ and vanishes at large $B$, so the *marginal* value of another doubling depends entirely on where you draw the line. Add a PUE of 1.2 and the numbers move again. Nothing in the literature fixes this boundary for batch sweeps.

**Second, the SLO turns monotone into non-monotone.** At $B=256$ the step is 14 ms, so TPOT is 14 ms — fine. But prefill for 256 queued requests at 512 prompt tokens each is $2 \times 7\text{e}9 \times 131{,}072 / 3\text{e}14 \approx 6.1$ s of compute, so p99 TTFT blows past a 2 s target. Chunked prefill fixes TTFT by interleaving prefill chunks into decode steps — which lengthens $T_{\text{step}}$ and pushes TPOT up. The feasible $B$ is therefore set by an interaction between two SLOs and a scheduler policy, not by the energy curve.

The obstruction is visible here: the energy curve alone says "as large as possible", the measured curve depends on an unstandardized boundary, and the constraint that actually binds is produced by a scheduler whose behavior is not in the model. That is why the answer is empirically open rather than arithmetic.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*