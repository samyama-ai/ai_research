---
id: 11-inference-and-serving/optimal-chunked-prefill-size
title: "Optimal Chunked Prefill Chunk Size"
topic: 11-inference-and-serving
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Chunked Prefill Chunk Size

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/optimal-chunked-prefill-size` · **Status:** empirically-open

## 1. Problem Statement

Chunked prefill splits a request's prompt into token blocks and schedules each block into a batch alongside ongoing decode steps, so one GPU kernel launch carries both compute-bound prefill work and memory-bound decode work. The scheduler needs a token budget $C$ — the maximum number of tokens per iteration (`max_num_batched_tokens` in vLLM, `chunk_size` in Sarathi-Serve). Choosing $C$ trades time-to-first-token (TTFT) against inter-token latency (ITL) and against total throughput.

The problem has three variants that are routinely conflated:

- **Measurement.** Given a workload trace, a model, and a hardware target, is there a well-defined objective $C$ optimizes? Latency SLOs are per-request and percentile-valued; throughput is aggregate. There is no agreed scalarization.
- **Method.** Compute $C^\star$ — or a per-iteration adaptive $C_t$ — from observable state (queue depth, prompt-length distribution, KV-cache occupancy) without an offline sweep per model/hardware pair.
- **Theory.** Prove that the achievable (TTFT, ITL, throughput) frontier is traced by a *scalar* chunk budget at all, versus requiring a richer scheduling policy. No such result exists in either direction.

**Solved** would mean: a policy that, without per-deployment sweeping, lands within a few percent of the offline-optimal $C$ on the deciding metric across at least three model scales and two accelerator generations.

## 2. Formal Setting

Let the model have $L$ layers, hidden size $d$, $n_{kv}$ KV heads of dimension $d_h$, and per-token KV bytes $b = 2 L n_{kv} d_h s$ for $s$ bytes per element. A request $r$ has prompt length $P_r$ and output length $D_r$.

**Iteration cost.** At iteration $t$ the batch carries $m_t$ prefill tokens (across at most one or a few chunked requests) and $k_t$ decode tokens, with $m_t + k_t \le C$. Measured cost is the wall-clock duration of the forward pass:

$$T_t = \underbrace{\alpha (m_t + k_t)}_{\text{linear-layer GEMMs}} + \underbrace{\beta \, \mathrm{KV}_t / \mathrm{BW}}_{\text{attention memory traffic}} + \underbrace{\gamma}_{\text{fixed launch/allreduce}}$$

$\alpha$ (s/token) is measured by fitting a line to forward-pass time versus batch token count at fixed context; $\gamma$ is the intercept — measure it as the time of a 1-token decode step with empty KV. $\mathrm{BW}$ is achieved HBM bandwidth from a memory-bound microbenchmark, not the spec sheet.

**Chunking overhead.** Chunking a prompt of length $P$ into $\lceil P/C\rceil$ chunks forces each chunk to re-read the KV of all preceding chunks. Total attention KV bytes read for one prompt:

$$B(C) \;=\; b\sum_{i=1}^{\lceil P/C\rceil}\big((i-1)C + \tfrac{C}{2}\big) \;\approx\; \frac{b P^2}{2C} + \frac{bP}{2}$$

against $bP/2$ for an unchunked prefill. The overhead is $\Theta(P^2/C)$ — it diverges as $C\to 0$ and vanishes as $C \ge P$. FLOPs are unchanged; only traffic grows.

**Objective.** Per request, $\mathrm{TTFT}_r$ = admission to first token, $\mathrm{ITL}_r$ = mean gap between successive output tokens. The commonly used decision predicate:

$$C^\star = \arg\max_C \; \Lambda(C) \quad \text{s.t.} \quad Q_{p}\big(\mathrm{TTFT}\big) \le \tau_1, \;\; Q_{p}\big(\mathrm{ITL}\big) \le \tau_2$$

with $\Lambda$ the sustainable arrival rate (req/s), $Q_p$ the $p$-th percentile (usually $p=99$), and $\tau_1,\tau_2$ the SLOs.

**Assumptions, and which break.**
1. *$T_t$ is affine in token count.* Violated: GEMM time is piecewise-flat across tile-quantization boundaries, so $\alpha$ is a staircase. A chunk of 513 tokens can cost the same as 640.
2. *Prefill and decode overlap freely inside one kernel.* Violated: standard batched attention serializes the prefill and decode attention phases; POD-Attention (ASPLOS 2025) exists precisely because they do not overlap by default.
3. *Output lengths are known or exchangeable.* Violated: $D_r$ is unknown at admission and heavy-tailed.
4. *One scalar $C$ per deployment.* Violated whenever the prompt-length distribution is bimodal (short chat + long RAG).

## 3. State of the Art

**Established.**
- **Sarathi** (Agrawal et al., 2023, arXiv:2308.16369) introduced chunked prefill with decode piggybacking and showed the mechanism removes decode-side pipeline bubbles.
- **Sarathi-Serve** (Agrawal et al., OSDI 2024) formalized *stall-free batching*: admit prefill chunks only into the slack left by decodes, so no decode is preempted. Reported up to $2.6\times$ higher serving capacity under SLO for Mistral-7B on one A100 and up to $5.6\times$ for Falcon-180B on a multi-GPU node. Independently reproduced *as a mechanism* — chunked prefill is now default in vLLM V1 and present in TensorRT-LLM and SGLang.
- **DeepSpeed-FastGen** (Holmes et al., 2024, arXiv:2401.08671) shipped Dynamic SplitFuse, the same idea with a fixed target token count per forward pass.
- The $\Theta(P^2/C)$ re-read cost of §2 is arithmetic, not empirical.

**Claimed but unablated.**
- That $C$ is a *sufficient* control knob. Sarathi-Serve tunes $C$ against a decode-latency SLO but does not ablate $C$ against alternative controls (prefill-token quota per request, decode-batch floor, priority queueing) on a common trace.
- Prefill/decode **disaggregation** — DistServe (Zhong et al., OSDI 2024, arXiv:2401.09670) and Splitwise (Patel et al., ISCA 2024, arXiv:2311.18677) — claims to dominate chunked prefill by removing interference entirely. The head-to-head comparisons are cross-paper, on different hardware and traces, and are not an ablation.

**Benchmark numbers only.** The chunk sizes that circulate as folklore — 512 in the Sarathi papers, vLLM's default `max_num_batched_tokens` of 2048 (raised to 8192 in throughput-oriented configs) — are single-configuration measurements, not optima transported across hardware.

## 4. What Is Known

- **Chunking helps decode tail latency.** On Sarathi-Serve's setup (Mistral-7B, A100-80GB, chunk 512), P99 ITL under mixed load drops sharply versus Orca-style continuous batching (Yu et al., OSDI 2022), which admits whole prefills and stalls decodes for the length of one prompt.
- **Chunking hurts TTFT and raw prefill throughput.** Because $B(C)\propto P^2/C$, small chunks on long prompts add real HBM traffic. For a Llama-3-8B-shaped model ($b \approx 128$ KiB/token at FP16 with GQA), a 32 k prompt at $C=512$ reads $\approx \tfrac{b P^2}{2C} \approx 128\,\mathrm{KiB}\times 32768^2/1024 \approx 134$ GB of KV — about 67 ms of pure re-read at 2 TB/s, versus $\approx 2$ GB unchunked.
- **The optimum is hardware-dependent.** $\alpha$ scales with FLOPs and $\beta/\mathrm{BW}$ with bandwidth; H100 has roughly $3\times$ the FP16 FLOPs of A100-80GB against roughly $1.6$–$2\times$ the bandwidth, so the token count at which a batch becomes compute-saturating shifts materially between them. The direction is established by roofline arithmetic; the magnitude at the level of a specific $C^\star$ is not published as a clean sweep.
- **Fusing does not mean overlapping.** POD-Attention (Kamath et al., ASPLOS 2025) reports up to $\approx 1.75\times$ faster attention and $\approx 1.2\times$ end-to-end over vLLM by co-scheduling prefill and decode attention within an SM — evidence that pre-POD chunked-prefill measurements were taken on a serialized attention kernel, so any $C^\star$ fit before it is kernel-specific.
- **Vidur** (Agrawal et al., MLSys 2024, arXiv:2405.05465) shows iteration time is predictable enough from a fitted operator model to search scheduler configs in simulation, with reported errors in the low single-digit percent on the configurations tested.

## 5. What Is Not Known

- **Empirically open.** No published sweep of $C \in \{256,\dots,16384\}$ crossed with $\{$model scale$\}\times\{$A100, H100, MI300X$\}\times\{$chat, RAG, agentic long-context trace$\}$ reporting the full (TTFT-P99, ITL-P99, throughput) surface. The experiment costs GPU-hours, not new science. This is the primary gap.
- **Empirically open.** Whether adaptive per-iteration $C_t$ beats the best fixed $C$ by enough to matter (say $>10\%$ capacity at fixed SLO). No controlled comparison exists.
- **Methodologically blocked.** "Optimal" is undefined until the TTFT/ITL/throughput scalarization is fixed. Metron/Etalon (Agrawal et al., 2024, arXiv:2407.07000) argues mean ITL hides stalls and proposes a fluidity index; there is no community-standard objective, so two papers can both be right about different $C^\star$.
- **Theoretically open.** Whether the SLO-feasible throughput $\Lambda(C)$ is unimodal in $C$. Everyone tunes as if it is. Assumption 1 of §2 (staircase $\alpha$) predicts local non-monotonicity at tile boundaries, and no one has proven or disproven unimodality even under the affine-cost idealization.

## 6. Why It Is Hard

**Confounded measurement.** $C$ is not an independent variable. Changing it simultaneously changes (i) the number of decode slots per iteration, (ii) KV re-read traffic, (iii) GEMM tile efficiency, (iv) admission timing and therefore queueing, and (v) KV-cache high-water mark and hence preemption rate. A throughput delta from $C{=}512$ to $C{=}2048$ cannot be attributed without holding the other four fixed, and no published harness does. Consequence: reported chunk sizes are properties of an entire (kernel, scheduler, trace, hardware) tuple and do not transport.

Secondary: **absent ground truth for output length.** The optimal chunk admission depends on how many decode steps the currently-running requests still owe, which is unknown at decision time. Even an oracle over $C$ is defined only against a specific trace realization.

## 7. Current Research (as of 2026)

- **Kernel-level fusion** to make the chunk-size trade-off less sharp: POD-Attention (Microsoft Research India / Georgia Tech lineage, ASPLOS 2025); NanoFlow's intra-device pipelining of compute-, memory-, and network-bound ops (Zhu et al., OSDI 2025, arXiv:2408.12757). If fusion fully hides decode inside prefill, $C^\star$ moves toward "as large as memory allows."
- **Disaggregation as the alternative answer** — DistServe, Splitwise, and production Mooncake-style deployments. *(frontier — verify)* The open question is the crossover: at what request rate and prompt-length mix does disaggregation beat well-tuned chunked prefill on a fixed GPU budget.
- **Simulation-driven configuration search** (Vidur and successors) to replace physical sweeps.
- **SLO-aware adaptive scheduling** *(frontier — verify)*: schedulers that shrink $C$ when decode SLO headroom is thin and grow it when the queue is prefill-dominated. Present in production serving stacks; not, to our knowledge, ablated in a peer-reviewed head-to-head against best-fixed-$C$.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B on 1×H100-80GB and Llama-3.1-70B (TP=4) on 4×H100. Two traces: (a) ShareGPT-like chat, median prompt $\approx 250$ tokens; (b) long-context RAG, prompts sampled log-uniformly in $[4\mathrm{k}, 64\mathrm{k}]$. Sweep $C \in \{256, 512, 1024, 2048, 4096, 8192, 16384\}$. Run each cell at 8 arrival rates spanning 40–110% of saturation, 20 min each. $\approx 900$ GPU-hours.

**Control arms.** (1) Orca-style continuous batching, no chunking. (2) Best fixed $C$ per (model, trace) — the arm the adaptive policy must beat. (3) Same sweep with POD-style fused prefill-decode attention, to test whether $C^\star$ is a kernel artifact.

**Deciding number.** $\Lambda^{\text{SLO}}(C)$: maximum sustained req/s with P99 TTFT $\le 2$ s and P99 ITL $\le 50$ ms. The question is settled if
$$\frac{\max_C \Lambda^{\text{SLO}}(C) - \Lambda^{\text{SLO}}(C_{\text{default}}=2048)}{\Lambda^{\text{SLO}}(2048)} \;<\; 0.05$$
holds in every cell — then $C$ is a nuisance parameter and tuning it is not worth the engineering. If any cell exceeds $0.20$, $C$ is a first-class control and per-deployment tuning (or adaptation) is mandatory. Report $\Lambda^{\text{SLO}}$ against $C$ as a curve, so unimodality (§5) is directly checkable.

## 9. Key References

- **[Foundational]** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI, 2022.
- **[Foundational]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[Foundational]** Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee. *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* 2023. — arXiv:2308.16369
- **[SOTA]** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024. — arXiv:2403.02310
- **[SOTA]** Aditya K. Kamath, Ramya Prabhu, Jayashree Mohan, Simon Peter, Ramachandran Ramjee, Ashish Panwar. *POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference.* ASPLOS, 2025.
- **[Contrast]** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI, 2024. — arXiv:2401.09670
- **[Contrast]** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA, 2024. — arXiv:2311.18677
- **[Systems]** Connor Holmes et al. *DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference.* 2024. — arXiv:2401.08671
- **[Methods]** Amey Agrawal, Nitin Kedia, Anmol Agarwal, Jayashree Mohan, Nipun Kwatra, Souvik Kundu, Ramachandran Ramjee, Alexey Tumanov. *Etalon / Metron: Holistic Performance Evaluation Framework for LLM Inference Systems.* 2024. — arXiv:2407.07000
- **[Methods]** Amey Agrawal et al. *Vidur: A Large-Scale Simulation Framework for LLM Inference.* MLSys, 2024. — arXiv:2405.05465

## 10. Worked Example

Llama-3.1-8B, FP16, H100-80GB. GQA gives $n_{kv}=8$, $d_h=128$, $L=32$, so $b = 2\times32\times8\times128\times2 = 131{,}072$ B $= 128$ KiB per token. Take achieved HBM bandwidth $\mathrm{BW}=2.6$ TB/s.

One request, $P=32{,}768$ prompt tokens, served alongside 32 decodes.

| $C$ | chunks | KV re-read $\approx bP^2/2C$ | re-read time | GEMM time ($\alpha\!\approx\!0.29$ µs/tok) | decode slots served |
|---|---|---|---|---|---|
| 512 | 64 | 134 GB | 52 ms | 9.5 ms | 64 iterations |
| 2048 | 16 | 34 GB | 13 ms | 9.5 ms | 16 iterations |
| 8192 | 4 | 8.4 GB | 3.2 ms | 9.5 ms | 4 iterations |

The GEMM work is identical in all three rows — same tokens, same weights. Only the attention traffic and the decode opportunity change.

Now the obstruction. At $C=512$ the prompt costs $52+9.5 \approx 62$ ms of extra attention traffic but hands the 32 waiting requests **64** decode opportunities, each arriving roughly every $\approx 1$ ms — a clean ITL. At $C=8192$ the prompt costs $\approx 13$ ms total but the 32 decoders wait $\approx 3.2$ ms between token emissions, and each of the 4 iterations is one long stall. Which is better depends entirely on whether the SLO is written as mean ITL (favours $C=8192$: less total work, higher throughput) or P99 ITL (favours $C=512$: no stall exceeds 1 ms).

Both configurations are "optimal." The 16× spread in $C$ is not a measurement disagreement — it is the absence of a fixed objective. That is why §5 lists the methodological block first: the sweep in §8 is cheap and runnable, but its answer is only meaningful once someone commits to a percentile and a threshold before running it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*