---
id: 32-hardware-and-kernels/memory-bandwidth-scaling-laws
title: "Memory Bandwidth Scaling Laws for Model Serving"
topic: 32-hardware-and-kernels
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memory Bandwidth Scaling Laws for Model Serving

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/memory-bandwidth-scaling-laws` · **Status:** empirically-open

## 1. Problem Statement

Autoregressive decoding moves more bytes than it does arithmetic. The folk model is that serving throughput is proportional to HBM bandwidth. The problem is to establish whether that proportionality is a law with a measurable exponent, or a first-order approximation that breaks down exactly where deployment decisions are made.

- **Input:** a serving configuration — model $\mathcal{M}$, batch size $B$, context length $S$, parallelism plan $\pi$, and a device with peak bandwidth $\beta$ (bytes/s) and peak dense throughput $\phi$ (FLOP/s).
- **Output:** predicted tokens/s at a target latency percentile.
- **Decision predicate:** does there exist an exponent $\alpha$ and a slowly-varying prefactor such that $\text{throughput} \propto \beta^{\alpha}$ over at least a 3× bandwidth range, with $\alpha$ stable across model shapes and hardware generations?

Three variants, with different difficulty:

- **Measurement:** can achieved bandwidth for a serving workload be attributed to weights, KV cache, activations, and collective traffic, separately? Currently only aggregate counters exist. This variant is the blocker.
- **Method:** given a bandwidth budget, what is the throughput-optimal $(B, S, \text{quantization}, \text{KV layout})$? Solved heuristically, not optimally.
- **Theory:** is there a lower bound on bytes moved per generated token for a given model and cache policy, analogous to I/O complexity lower bounds for matrix multiply? Open.

Solving it means: a fitted law that predicts tokens/s on unseen hardware within 10% from the model's byte budget alone.

## 2. Formal Setting

Let a decoder-only transformer have $L$ layers, hidden size $d$, $h$ query heads, $h_{kv}$ key/value heads, head dimension $d_h$, and $N$ parameters. Let $b_w$ be bytes per weight and $b_{kv}$ bytes per cached element.

**Per-step byte traffic** (one decode step, batch $B$, mean context $\bar{S}$):

$$
\mathcal{B}(B,\bar{S}) = \underbrace{N b_w}_{\text{weights}} + \underbrace{2 B \bar{S} L h_{kv} d_h b_{kv}}_{\text{KV read}} + \underbrace{2 B L h_{kv} d_h b_{kv}}_{\text{KV write}} + \underbrace{c_{\text{act}} B L d}_{\text{activations}} + \underbrace{\kappa(\pi, B, d)}_{\text{collectives}}
$$

Weight traffic is independent of $B$; KV traffic is linear in $B\bar{S}$. This is the whole tension.

**Arithmetic intensity** $I = \mathcal{F}(B,\bar{S}) / \mathcal{B}(B,\bar{S})$ where $\mathcal{F} \approx 2NB + 4B\bar{S}Lh d_h$. Roofline (Williams et al., CACM 2009) gives

$$
T_{\text{step}} \ \geq\ \max\!\left(\frac{\mathcal{F}}{\phi},\ \frac{\mathcal{B}}{\beta}\right), \qquad \text{ridge point } I^\* = \phi/\beta .
$$

**Model bandwidth utilization**, the quantity actually reported by practitioners:

$$
\mathrm{MBU} = \frac{\mathcal{B}(B,\bar{S})}{\beta \, T_{\text{step}}^{\text{measured}}} \in (0,1].
$$

The proposed law: $\text{tok/s} = B/T_{\text{step}} \propto \beta^{\alpha}$, and the claim under test is $\alpha \to 1$ in the memory-bound regime.

**How each quantity is measured.** $\beta$: STREAM-analogue on device, not the datasheet number (datasheet = pin rate × width; achievable is typically 80–93% of it). $\mathcal{B}$: hardware counters (`dram__bytes` on NVIDIA, sampled per kernel) — these count DRAM traffic including L2 misses on reload, so $\mathcal{B}^{\text{measured}} \geq \mathcal{B}^{\text{analytic}}$. $T_{\text{step}}$: inter-token latency at a fixed percentile, not mean, and excluding prefill.

**Assumptions, and which are violated.**

1. *Weights are read exactly once per step.* Violated under tensor parallelism with re-materialization, and under some MoE routings where expert weights are read per-microbatch.
2. *KV cache is read exactly once.* Violated by paged allocators with fragmentation, and by chunked prefill interleaving (Sarathi-Serve) which re-touches cache pages.
3. *Bandwidth is a scalar.* Violated: read and write bandwidth differ, and HBM row-buffer locality means effective $\beta$ depends on access stride. Paged KV with page size 16 tokens has measurably worse locality than contiguous.
4. *Batches are homogeneous in $\bar{S}$.* Violated in every production trace; continuous batching mixes sequences spanning 3 orders of magnitude in length, so $\bar{S}$ is a summary of a heavy-tailed distribution.
5. *Compute and memory overlap perfectly.* Violated — the roofline max is a lower bound, not an estimate.

## 3. State of the Art

**Established (ablated, reproduced):**

- *Efficiently Scaling Transformer Inference* (Pope et al., MLSys 2023) gives the analytic per-step byte budget for PaLM-scale decoding and shows the ridge-point crossover as $B$ grows. The partitioning-strategy ablations are real ablations.
- *PagedAttention / vLLM* (Kwon et al., SOSP 2023) reports 2–4× throughput over Orca at the same latency, attributed to KV memory fragmentation reduction — measured, with a memory-waste ablation (waste falls from 60–80% to under 4%).
- *GQA* (Ainslie et al., EMNLP 2023) reduces KV bytes by $h/h_{kv}$; for Llama-3-70B that is 8×. Quality-vs-bytes tradeoff is ablated against MHA and MQA.
- *FlashAttention* (Dao et al., NeurIPS 2022; FA-2, 2023) — IO-aware tiling with an explicit HBM-access analysis. The $O(S^2 d^2 M^{-1})$ HBM access count is proved, not just measured.
- Prefill/decode disaggregation: DistServe (Zhong et al., OSDI 2024) and Splitwise (Patel et al., ISCA 2024) both show the two phases have different bottlenecks (compute vs bandwidth) and that co-locating them costs goodput.

**Claimed but unablated:**

- "Decode is bandwidth-bound, so throughput scales linearly with HBM bandwidth." Stated widely; the controlled bandwidth sweep at fixed compute has not been published. Vendor generation-over-generation numbers (A100 2.0 TB/s → H100 3.35 TB/s → H200 4.8 TB/s → B200 ~8 TB/s) confound bandwidth with SM count, cache size, NVLink width, and software stack.
- MBU as a portable metric. Introduced in Databricks' inference-performance engineering writeup (2023); widely quoted, no cross-vendor calibration study establishing that equal MBU means equal efficiency.

**Benchmark-number-only:** most published tokens/s figures for a given GPU are single-config MLPerf-Inference or vendor entries. They fix the software stack per submission, so they cannot be differenced to isolate bandwidth.

## 4. What Is Known

- **Ridge points are far to the right.** H100 SXM: $\phi/\beta \approx 989\ \text{TFLOP/s} / 3.35\ \text{TB/s} \approx 295$ FLOP/byte (BF16 dense). A100 80GB: $312/2.039 \approx 153$. Decode with $B=1$ has $I \approx 2$ FLOP/byte — two orders of magnitude below the ridge.
- **Achieved MBU plateaus well below 1.** Reported values for well-tuned 7B–70B decode cluster in the 55–75% range on A100/H100 at moderate batch; small-batch single-stream is commonly 30–50%. Scale: single node, 1–8 GPUs.
- **The memory-wall trend is quantified.** Gholami et al., *AI and Memory Wall* (IEEE Micro, 2024): peak compute grew roughly 3× per 2 years while DRAM bandwidth grew about 1.6× per 2 years over the 2000s–2020s window. The gap is structural, not a one-generation artifact.
- **Batching converts a bandwidth problem into a capacity problem.** Weight traffic amortizes as $1/B$; KV traffic does not. For Llama-3-70B (GQA, $h_{kv}=8$, $L=80$, $d_h=128$, FP16 KV), KV cost is $2\cdot 8\cdot 128\cdot 2\cdot 80 = 327{,}680$ B/token $\approx 0.33$ MB/token — so at 8k context a single sequence carries 2.7 GB of cache.
- **Weight-only quantization moves the line.** AWQ (Lin et al., MLSys 2024) and GPTQ (Frantar et al., ICLR 2023) at 4 bits cut $N b_w$ by ~4×; measured end-to-end decode speedups are typically 2.5–3.5×, not 4× — the deficit is real and unexplained by any published attribution.
- **Speculative decoding raises $I$ without changing $\beta$** (Leviathan et al., ICML 2023; Chen et al., 2023): 2–3× wall-clock on greedy-equivalent output, because $k$ tokens are verified per weight read.

## 5. What Is Not Known

- **Empirically open.** The controlled experiment — vary $\beta$ alone, holding SM clock, cache hierarchy, driver, and kernel binary fixed, and fit $\alpha$ — is runnable today on any GPU with lockable memory clocks. Nobody has published it across model shapes. The exponent $\alpha$ is unmeasured.
- **Empirically open.** Whether MBU is invariant across vendors (NVIDIA HBM3 vs AMD MI300X HBM3e vs TPU v5e) for the same model and batch. If it is not, MBU is a per-vendor efficiency knob, not a law.
- **Methodologically blocked.** Per-source byte attribution. `dram__bytes` counters aggregate all traffic; separating weight reads from KV reads from allocator overhead requires either kernel-level instrumentation that perturbs timing or a simulator whose fidelity is itself unvalidated. Without attribution, the 4-bit-quantization deficit (3× measured vs 4× predicted) cannot be diagnosed.
- **Theoretically open.** No lower bound on bytes-per-generated-token for a decoder with an arbitrary cache-compression policy. FlashAttention bounds HBM accesses for a *fixed* attention computation; the question of the minimum traffic over all algorithms producing the same token distribution is unaddressed.
- **Empirically open.** Whether inference-aware scaling laws (Sardana et al., ICML 2024, which reweights Chinchilla for inference FLOPs) change qualitatively when the cost model is *bytes* rather than FLOPs. Sardana et al. optimize FLOPs; decode is not FLOP-bound.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus non-identifiability**.

Bandwidth cannot be varied in the field without varying something else. Across GPU generations, $\beta$, $\phi$, L2 size, and the compiler all move together, so a generation-over-generation throughput ratio identifies nothing. Within a single GPU, memory clock can be locked — but on datacenter SKUs (A100, H100) the exposed memory-clock set is small and coarse, and lowering it changes DVFS behavior for the SMs too on some firmware.

Second: the model $T_{\text{step}} = \mathcal{B}/(\eta\beta) + \tau_{\text{fixed}}$ has two unknowns, $\eta$ and $\tau_{\text{fixed}}$ (launch overhead, collectives, scheduler). A single throughput measurement cannot separate them. Any $\alpha < 1$ observed is equally explained by "the workload is not fully bandwidth-bound" and by "there is a constant overhead per step". Attribution counters would break the tie; they do not exist at the required granularity. That is why the measurement variant blocks the method variant.

Third: the standard evaluation does not measure what it names. "Tokens/s" on a fixed benchmark trace conflates scheduling policy, batch composition, and hardware. Two systems at identical MBU can differ 2× in tokens/s because one admits longer sequences.

## 7. Current Research (as of 2026)

- **KV-cache byte reduction as the primary lever.** DeepSeek's multi-head latent attention (DeepSeek-V2, 2024) compresses KV into a low-rank latent, cutting cache bytes by roughly an order of magnitude versus MHA; adoption in other frontier open models is broadening *(frontier — verify)*.
- **Disaggregated serving at cluster scale.** Following DistServe and Splitwise, production stacks route prefill to compute-dense parts and decode to bandwidth-dense parts. Whether a dedicated "decode SKU" with high $\beta$ and low $\phi$ is economically optimal is an active argument *(frontier — verify)*.
- **HBM4 and near-memory processing.** Sampling and stacking roadmaps push per-package bandwidth toward 1.5–2× HBM3e. PIM-based attention offload (Samsung/SK hynix research lines) targets exactly the decode KV read *(frontier — verify)*.
- **Roofline-style analysis for LLM inference** as a survey-and-tooling direction: Yuan et al., *LLM Inference Unveiled: Survey and Roofline Model Insights* (2024) is the reference framing.
- Groups: Berkeley Sky/BAIR (vLLM lineage), Microsoft Research Systems (Splitwise, Sarathi), UCSD/PKU (DistServe), Stanford Hazy Research (IO-aware kernels), CMU Catalyst (speculative decoding).

## 8. Concrete Next Experiment

**The bandwidth sweep with a compute control arm.**

- **Scale:** 8× H100 SXM (single node, NVLink), and 8× A100 80GB as a second point. Models: Llama-3.1-8B (TP1) and Llama-3.1-70B (TP8), BF16 and 4-bit AWQ. Serving stack: vLLM, pinned commit, paged KV, continuous batching off (fixed batch) to remove scheduler variance.
- **Treatment arm:** lock SM clock at a fixed value; sweep memory clock across the full supported set via `nvidia-smi -lmc`, spanning at least a 2.5× range in achieved $\beta$ (verify achieved $\beta$ per setting with a device-side STREAM triad, do not trust the clock ratio).
- **Control arm:** lock memory clock at maximum; sweep SM clock over a matched 2.5× range via `nvidia-smi -lgc`. If decode is bandwidth-bound, this arm must be nearly flat.
- **Grid:** $B \in \{1, 8, 32, 128\}$, $\bar{S} \in \{512, 4096, 32768\}$, three seeds, report p50 and p95 inter-token latency.
- **The deciding number:** the fitted exponent $\alpha$ in $\log(\text{tok/s}) = \alpha \log \beta^{\text{achieved}} + c$, per cell, with 95% CI.
  - $\alpha \geq 0.90$ across the grid → the linear-bandwidth folk law holds; publish the prefactor as the real content.
  - $\alpha \leq 0.70$ in any cell with $B \leq 32$ → the law fails in the regime that matters, and $\tau_{\text{fixed}}$ (obtained as the intercept of $T_{\text{step}}$ vs $1/\beta$) is the quantity to attack instead.
- **Cost:** roughly 200 GPU-hours. This is small. It has not been published, which is the point.

## 9. Key References

- **[Foundational]** Samuel Williams, Andrew Waterman, David Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM, 2009.
- **[Foundational]** Wm. A. Wulf, Sally A. McKee. *Hitting the Memory Wall: Implications of the Obvious.* ACM SIGARCH Computer Architecture News, 1995.
- **[Foundational]** John D. McCalpin. *Memory Bandwidth and Machine Balance in Current High Performance Computers.* IEEE TCCA Newsletter, 1995.
- **[SOTA]** Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean. *Efficiently Scaling Transformer Inference.* MLSys, 2023. — arXiv:2211.05102
- **[SOTA]** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[SOTA]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS, 2022. — arXiv:2205.14135
- **[SOTA]** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP, 2023. — arXiv:2305.13245
- **[SOTA]** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI, 2024. — arXiv:2401.09670
- **[SOTA]** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA, 2024. — arXiv:2311.18677
- **[SOTA]** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI, 2024. — arXiv:2403.02310
- **[SOTA]** Yaniv Leviathan, Matan Kalman, Yossi Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- **[SOTA]** Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML, 2024. — arXiv:2401.00448
- **[Survey]** Amir Gholami, Zhewei Yao, Sehoon Kim, Coleman Hooper, Michael W. Mahoney, Kurt Keutzer. *AI and Memory Wall.* IEEE Micro, 2024.
- **[Survey]** Zhihang Yuan, Yuzhang Shang, Yang Zhou, Zhen Dong, Chenhao Xue, Bingzhe Wu, Zhikai Li, Qingyi Gu, Yong Jae Lee, Yan Yan, Beidi Chen, Guangyu Sun, Kurt Keutzer. *LLM Inference Unveiled: Survey and Roofline Model Insights.* 2024. — arXiv:2402.16363
- **[Survey]** Sehoon Kim, Coleman Hooper, Thanakul Wattanawong, Minwoo Kang, Ruohan Yan, Hasan Genc, Grace Dinh, Qijing Huang, Kurt Keutzer, Michael W. Mahoney, Sophia Shao, Amir Gholami. *Full Stack Optimization of Transformer Inference: a Survey.* 2023. — arXiv:2302.14017

## 10. Worked Example

**Llama-3.1-70B, BF16, TP8 on one 8×H100 SXM node, $B=1$, $\bar{S}=8192$.**

Bytes per decode step, aggregated across the node:

- Weights: $70\times10^9 \times 2 = 140$ GB.
- KV read: $0.33$ MB/token $\times$ 8192 tokens $= 2.68$ GB.
- Activations + collectives at $B=1$: under 0.1 GB.

Total $\mathcal{B} \approx 142.8$ GB. Aggregate peak bandwidth $= 8 \times 3.35 = 26.8$ TB/s.

$$
T_{\text{step}}^{\text{roofline}} = \frac{142.8\ \text{GB}}{26.8\ \text{TB/s}} = 5.33\ \text{ms} \quad \Rightarrow \quad 188\ \text{tok/s}.
$$

Measured single-stream decode for this configuration on a well-tuned stack sits around 40–60 tok/s, i.e. $T_{\text{step}} \approx 17$–25 ms. So $\mathrm{MBU} \approx 0.21$–$0.32$.

**Where the obstruction becomes visible.** Roughly 12 ms per step is unaccounted for. Candidate explanations, all consistent with the single number 50 tok/s:

| Candidate | Predicted signature |
|---|---|
| 8 all-reduces × 80 layers of NVLink latency | scales with $L$, not with $\beta$ |
| Kernel launch overhead (~640 kernels/step) | constant, independent of $\beta$ and $B$ |
| Weights re-read from HBM due to L2 thrash | scales with $1/\beta$ |
| Paged KV row-buffer misses | scales with $1/\beta$, worse at small page size |

Rows 1–2 are $\beta$-independent; rows 3–4 are $\beta$-proportional. A throughput measurement at one bandwidth cannot distinguish them — the model $T = \mathcal{B}/(\eta\beta) + \tau_{\text{fixed}}$ is unidentified from a single point. That is precisely why the sweep in §8 is the experiment: it turns one equation with two unknowns into a regression with an intercept. If the intercept lands near 12 ms, the "bandwidth-bound" framing is wrong for $B=1$ serving and the engineering target is launch and collective overhead, not HBM.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*