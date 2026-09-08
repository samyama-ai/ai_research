---
id: 11-inference-and-serving/energy-per-token-lower-bound
title: "Energy per Token Lower Bound for Served Models"
topic: 11-inference-and-serving
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Energy per Token Lower Bound for Served Models

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/energy-per-token-lower-bound` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a model family, a quality target, and a serving workload, what is the minimum energy required to produce one output token?

- **Input:** a task distribution $\mathcal{D}$, a quality floor $q^\star$ on that distribution, a hardware class $\mathcal{H}$, and a workload trace (arrival rate, prompt/decode length distribution, latency SLO).
- **Output:** a number $E^\star$ in joules per output token, plus a certificate that no serving system in the admissible class beats it.
- **Decision predicate:** for a claimed $E^\star$ and a candidate system with measured $\hat{E}$, decide whether $\hat{E} < E^\star$ is achievable.

Three variants, with sharply different difficulty:

- **Measurement.** Define and measure $\hat{E}$ for a deployed system so two labs get the same number for the same system. Currently blocked — the boundary of what counts is not agreed.
- **Method.** Find the lowest $\hat{E}$ achievable today at fixed quality. Empirically open, and progress is fast.
- **Theory.** Prove a lower bound $E \geq E^\star$ that is not vacuous. Theoretically open; the only rigorous bounds available (Landauer) are ~7 orders of magnitude below practice.

Solving it means: a reproducible measurement protocol, plus a bound within one order of magnitude of the best measured system.

## 2. Formal Setting

Let a served model be $f_\theta$ with $N$ non-embedding parameters. A request $r$ has prompt length $p_r$ and generates $g_r$ tokens. Over a trace window $[0,T]$ with request set $R$:

$$\hat{E} \;=\; \frac{\int_0^T P(t)\,dt}{\sum_{r \in R} g_r}$$

Every term needs an operational definition:

- $P(t)$ — **instantaneous power, in watts, at a named boundary.** Options, all in use: (a) accelerator die power from `nvidia-smi`/NVML or DCGM; (b) node wall power at the PDU; (c) node power $\times$ PUE (power usage effectiveness, facility draw over IT draw); (d) (c) plus embodied manufacturing energy amortized over device lifetime. These differ by more than $3\times$. NVML samples at ~10–100 Hz with vendor-defined smoothing and excludes HBM PHY, NVLink switches, CPU host, NIC, and storage.
- **Idle attribution.** $\int_0^T P(t)dt$ includes power drawn while the GPU is idle between requests. Whether idle energy is charged to tokens, to capacity provisioning, or discarded changes $\hat{E}$ by $2$–$10\times$ at low utilization.
- $g_r$ — output tokens. Not comparable across tokenizers: the same string is 15–30% more tokens under a smaller vocabulary. Energy per *character* or per *task* is comparable; per *token* is not.
- **Quality constraint.** $\hat{E}$ is only meaningful at fixed quality: $\;\min \hat{E}$ subject to $q(f_\theta, \mathcal{D}) \ge q^\star$ and $\mathrm{P}_{95}[\text{TPOT}] \le \tau$ (time per output token). Quantization, distillation, and speculative decoding all trade $q$ for $E$; without the constraint the minimum is trivially zero.

**Physical floors.** Landauer's bound gives $E_{\min} = k_B T \ln 2 = 2.87 \times 10^{-21}$ J per irreversibly erased bit at $T=300$K. A memory-movement floor is closer to binding:

$$E_{\text{mem}} \;\ge\; \frac{8 \cdot \min(N \cdot b_\theta,\, M_{\text{HBM}}) \cdot \varepsilon_{\text{bit}}}{B}$$

where $b_\theta$ is bytes per parameter, $\varepsilon_{\text{bit}} \approx 3$–$5$ pJ/bit for HBM3, and $B$ is the decode batch size sharing one weight sweep.

**Assumptions known to be violated:** that weights are read once per forward pass (KV-cache traffic often dominates at long context); that batch size is a free variable (SLOs cap it); that power is proportional to utilization (idle draw is 15–30% of TDP); that DVFS is fixed (clocks throttle thermally mid-trace).

## 3. State of the Art

**Theory SOTA.** No non-trivial lower bound exists for transformer inference. Landauer (1961) and Bennett's reversible-computation result (1973) bound bit erasure, not matrix multiplication on real hardware. Horowitz (ISSCC 2014) gives *measured* per-operation energies at 45 nm — 0.9 pJ for a 32-bit FP add, 3.7 pJ for a multiply, ~640 pJ for a 32-bit DRAM access — which are engineering data points, not proven bounds. Established: the gap between arithmetic and memory-access energy is $10^2$–$10^3\times$, so inference energy is a data-movement problem.

**Systems/empirical SOTA.** Established and independently reproduced: continuous batching plus paged KV cache (vLLM, Kwon et al., SOSP 2023) raises decode throughput several-fold at fixed hardware, which lowers J/token roughly proportionally. Prefill/decode disaggregation (Splitwise, Patel et al., ISCA 2024) reports meaningful power reduction at matched throughput. Weight-only quantization to 4 bits (GPTQ, Frantar et al., ICLR 2023) cuts the weight-read term ~$4\times$.

**Claimed but unablated.** Google's 2025 report of a 0.24 Wh median Gemini Apps text prompt is a single-vendor number with an internally defined boundary; it is not independently reproducible and not per-token. Vendor claims of "$X\times$ more efficient inference" for new accelerator generations are benchmark numbers at unstated batch size and quality. MLPerf Inference (Reddi et al., ISCA 2020) has an optional power measurement track using SPEC PTDaemon at the wall; submissions are sparse, and the LLM benchmark's fixed accuracy target is a floor rather than a match, so J/token across submissions is not quality-controlled.

## 4. What Is Known

- **Per-operation energy asymmetry (45 nm, Horowitz 2014):** 32-bit int add 0.1 pJ; 32-bit FP multiply 3.7 pJ; 8 KB SRAM read ~10 pJ; DRAM access ~1.3–2.6 nJ. Ratio DRAM:add exceeds $10^4$.
- **Inference dominates lifetime energy.** BLOOM-176B: training 433 MWh, deployment measured at ~0.0036 kWh per request over 18 days of an API serving ~558 requests/hour (Luccioni et al., JMLR 2023 / arXiv:2211.02001). Sardana et al. (ICML 2024) formalize the consequence: inference-aware scaling favors smaller, over-trained models.
- **Task type dominates model choice.** Luccioni, Jernite & Strubell (FAccT 2024) measured ~88 models on 30 datasets: text generation cost roughly $10^3\times$ more energy per inference than text classification; image generation ranged to ~2.9 Wh per image.
- **Batch size dominates everything else at fixed hardware.** Samsi et al. (IEEE HPEC 2023) measured LLaMA-65B on A100 and V100 nodes and found per-token energy falling steeply with batch size and rising with tensor-parallel degree.
- **Datacenter overhead is not negligible:** hyperscale PUE ~1.1, typical enterprise ~1.5–1.6 (Masanet et al., Science 2020).
- **Order of magnitude for a served chat token, 2023–2025 published figures:** roughly $0.1$–$3$ J per output token depending on model size, batch, and boundary — a 30$\times$ spread that is mostly definitional, not physical.

## 5. What Is Not Known

- **Methodologically blocked (the primary gap).** There is no agreed measurement boundary, idle-attribution rule, or tokenizer normalization. Two honest labs measuring the same deployment differ by $3$–$10\times$. Until $\hat{E}$ is well defined, "lower bound on $\hat{E}$" has no referent. This is why the page is not merely empirically open.
- **Theoretically open.** No proof that any function of the form "produce one token at quality $q^\star$" requires $\Omega(g(N, q^\star))$ joules on physically realizable hardware. Communication-complexity lower bounds for matrix multiplication (Hong–Kung style I/O bounds) apply to a fixed algorithm, not to the task; nothing rules out a $100\times$ cheaper architecture at equal quality.
- **Empirically open.** Nobody has published a controlled sweep that holds task quality fixed while varying model size, quantization, batch, and hardware generation, and reports wall-plug J/token with a stated boundary. The experiment is runnable on a few nodes; it has not been run.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by an undefined feasible set**.

1. **The number is a ratio of two quantities that both move under the same intervention.** Raising batch size lowers J/token and raises latency; the "minimum" is a point on a Pareto surface, not a scalar. Reporting a scalar requires fixing the SLO, and no SLO is canonical.
2. **The infimum is over an open set.** $E^\star$ is defined as a minimum over admissible systems, but "admissible" includes future architectures, sparsity patterns, and caches. A prefix-cache hit costs near-zero decode energy; a system that caches enough drives measured J/token arbitrarily low without doing less computation.
3. **The physical floor is not binding.** Landauer sits ~7 orders of magnitude below practice (see §10), so it constrains nothing. The binding floor is architectural, and architectural floors are engineering estimates, not theorems.
4. **Vendor telemetry is the only cheap instrument and it measures the wrong boundary.** NVML reports die power, excluding host, network, cooling, and facility — the majority of the delta between published estimates.

## 7. Current Research (as of 2026)

- **Energy-aware serving schedulers.** DynamoLLM (Stojkovic et al., HPCA 2025) and related work co-tune parallelism, batch, and DVFS against SLOs; reported energy savings of tens of percent at matched latency. Microsoft Azure Research and UIUC are active here.
- **Measurement standardization.** MLPerf Inference power track (MLCommons) is the only cross-vendor protocol with a defined wall boundary; coverage of LLM serving remains thin. *(frontier — verify current submission counts.)*
- **Regulatory pressure on disclosure.** EU AI Act GPAI energy-reporting obligations and US datacenter reporting proposals are pushing vendors toward per-query numbers, which raises the cost of an undefined denominator. *(frontier — verify.)*
- **Hardware-side floors.** Analog/in-memory compute and 3D-stacked memory groups (Mythic, IBM Research, academic PIM efforts) argue the memory term is reducible by $10$–$100\times$; none has served a frontier model. *(frontier — verify.)*
- **Estimation from the outside.** Epoch AI, Hugging Face (Luccioni, Strubell), and de Vries (Joule 2023) publish independent bottom-up estimates of per-query energy; these disagree with vendor figures by roughly $2$–$5\times$, which is itself evidence for the methodological gap.

## 8. Concrete Next Experiment

**Goal:** produce the first quality-controlled, wall-plug J/token curve with a published boundary.

- **Scale:** one 8×H100 node (or 8×MI300X), metered at the PDU with a 1 Hz wall-power logger, plus synchronized NVML traces. Four models spanning $8$B–$70$B, each at FP16 and 4-bit weight quantization. One serving stack (vLLM), fixed version.
- **Workload:** a fixed 10,000-request replay trace with realistic prompt/decode length distribution, run at arrival rates producing 10%, 40%, 70%, and 95% of saturation throughput.
- **Quality control (the part usually missing):** each (model, quantization) arm must be *matched*, not merely above a floor — accept an arm only if its score on a held-out task suite lies within $\pm 0.5$ points of the reference arm. Arms that fail are excluded, not reported.
- **Control arm:** FP16 70B at batch 1 with idle energy charged to tokens — the worst-case, most-conservative accounting.
- **Deciding number:** the ratio $\rho = \hat{E}_{\text{wall}} / \hat{E}_{\text{NVML}}$, reported per arm. If $\rho$ is stable within $\pm 15\%$ across all arms, NVML is a valid proxy and the field can standardize on it cheaply. If $\rho$ varies by more than $2\times$ across arms, every published NVML-based J/token figure is incomparable, and the measurement problem must be solved before any bound is meaningful. Secondary output: the J/token vs. utilization curve, whose asymptote is the first empirically defensible candidate for $E^\star$ on that hardware.

Cost: roughly 200 GPU-hours plus a $500$ power meter.

## 9. Key References

- **[Foundational]** R. Landauer. *Irreversibility and Heat Generation in the Computing Process.* IBM Journal of Research and Development, 1961.
- **[Foundational]** C. H. Bennett. *Logical Reversibility of Computation.* IBM Journal of Research and Development, 1973.
- **[Foundational]** M. Horowitz. *Computing's Energy Problem (and what we can do about it).* ISSCC, 2014.
- **[Foundational]** E. Strubell, A. Ganesh, A. McCallum. *Energy and Policy Considerations for Deep Learning in NLP.* ACL, 2019. — arXiv:1906.02243
- **[SOTA]** A. S. Luccioni, S. Viguier, A.-L. Ligozat. *Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model.* JMLR, 2023. — arXiv:2211.02001
- **[SOTA]** A. S. Luccioni, Y. Jernite, E. Strubell. *Power Hungry Processing: Watts Driving the Cost of AI Deployment?* ACM FAccT, 2024. — arXiv:2311.16863
- **[SOTA]** S. Samsi et al. *From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference.* IEEE HPEC, 2023. — arXiv:2310.03003
- **[SOTA]** W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023. — arXiv:2309.06180
- **[SOTA]** P. Patel et al. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA, 2024. — arXiv:2311.18677
- **[SOTA]** N. Sardana et al. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML, 2024. — arXiv:2401.00448
- **[Method]** P. Henderson et al. *Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning.* JMLR, 2020. — arXiv:2002.05651
- **[Method]** V. J. Reddi et al. *MLPerf Inference Benchmark.* ISCA, 2020. — arXiv:1911.02549
- **[Survey]** E. Masanet, A. Shehabi, N. Lei, S. Smith, J. Koomey. *Recalibrating Global Data Center Energy-Use Estimates.* Science, 2020.
- **[Survey]** A. de Vries. *The Growing Energy Footprint of Artificial Intelligence.* Joule, 2023.
- **[Context]** D. Patterson et al. *Carbon Emissions and Large Neural Network Training.* 2021. — arXiv:2104.10350

## 10. Worked Example

**Setting.** Llama-3-70B, FP16, decode phase, 2×H100 SXM (700 W TDP each, 80 GB HBM3, 3.35 TB/s each), batch $B$.

**Weight-sweep time.** $140$ GB of weights over $6.7$ TB/s aggregate $= 20.9$ ms per forward pass. At $B=64$, that is $0.33$ ms per token.

**Measured-style estimate.** $2 \times 700\ \mathrm{W} \times 3.3\times10^{-4}\ \mathrm{s} = 0.46$ J/token at the die boundary. Add host and network ($\times 1.5$) and PUE 1.1: $\approx 0.76$ J/token at the wall. So $\rho \approx 1.65$ here.

**Memory floor.** $1.12\times10^{12}$ bits $\times\ 4$ pJ/bit $= 4.5$ J per weight sweep; at $B=64$, $0.070$ J/token.

**Landauer floor.** Decode costs $\approx 2N = 1.4\times10^{11}$ FLOPs. Charging a generous 64 bit-erasures per FLOP: $1.4\times10^{11} \times 64 \times 2.87\times10^{-21} = 2.6\times10^{-8}$ J $=26$ nJ/token.

| Quantity | J/token | Ratio to wall estimate |
|---|---|---|
| Wall-plug estimate | $0.76$ | $1\times$ |
| Die-only (NVML-style) | $0.46$ | $0.61\times$ |
| HBM-movement floor, $B{=}64$ | $0.070$ | $0.09\times$ |
| Landauer floor | $2.6\times10^{-8}$ | $3\times10^{-8}$ |

**Where the obstruction becomes visible.** The only rigorous bound is $1.8\times10^7$ times below practice — useless. The one floor within an order of magnitude, $0.070$ J, is not a constant: it scales as $1/B$. At $B=8$ it is $0.56$ J; at $B=512$ it is $0.0088$ J. Batch size is set by the latency SLO, which is a product decision. So the "physical lower bound on energy per token" is, in the regime anyone cares about, a restatement of a business constraint — and changing the reporting boundary alone moves the measured number by $1.65\times$ before any engineering happens.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*