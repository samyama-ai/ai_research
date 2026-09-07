---
id: 10-scaling-laws/communication-limited-scaling
title: "Scaling Laws Under Hardware Communication Limits"
topic: 10-scaling-laws
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws Under Hardware Communication Limits

> **Topic:** Scaling Laws & Compute Allocation · **ID:** `10-scaling-laws/communication-limited-scaling` · **Status:** open

## 1. Problem Statement

Neural scaling laws predict loss from parameter count $N$ and token count $D$ under a *compute* budget $C \approx 6ND$ FLOPs. Real training is not bought in FLOPs. It is bought in GPU-hours on a topology with finite interconnect bandwidth, and the fraction of peak FLOPs actually realized depends on how much data the parallelism strategy must move per step. The problem: **derive and validate a scaling law whose budget constraint is wall-clock time on a specified topology, not FLOPs, and determine whether the compute-optimal allocation $(N^\*, D^\*)$ moves under that constraint.**

Three variants, different difficulty:

- **Measurement.** Given a fixed cluster, measure the achievable loss frontier $L^\*(T)$ as a function of wall-clock hours $T$, and the $(N, D, \text{parallelism})$ that attains it. Runnable today; expensive.
- **Method.** Design training procedures (low-communication optimizers, sparsity, asynchronous updates) that shift the frontier, and show the shift persists as scale grows rather than closing.
- **Theory.** Prove or refute that the Chinchilla exponents are invariant to the communication constraint — i.e. that bandwidth changes the *achievable* $C$ but not the optimal $D/N$ ratio. No proof exists in either direction.

Solving it means: a fitted law $L(N, D, B, k)$ in bandwidth $B$ and device count $k$ that extrapolates one order of magnitude in $k$ with error comparable to the FLOP-only law's extrapolation error.

## 2. Formal Setting

**Objects.** Model with $N$ non-embedding parameters, trained on $D$ tokens, loss $L$ (nats/token on a held-out split of the training distribution). Cluster: $k$ accelerators, per-device peak $F_{\text{peak}}$ (FLOP/s, dense BF16 — the vendor number, e.g. $9.9\times10^{14}$ for H100 SXM), intra-node bandwidth $B_{\text{fast}}$ and inter-node bandwidth $B_{\text{slow}}$ (bytes/s per device, measured by all-reduce bus bandwidth from `nccl-tests`, not the datasheet link rate).

**Baseline law** (Hoffmann et al., 2022):
$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B_c}{D^{\beta}}, \qquad \alpha \approx 0.34,\ \beta \approx 0.28.$$

**Utilization.** Model FLOPs utilization is measured, not assumed:
$$\mathrm{MFU} = \frac{6ND}{T \cdot k \cdot F_{\text{peak}}},$$
with $T$ the observed wall-clock seconds for the run and $6ND$ the analytic forward+backward FLOPs (excluding attention-score FLOPs and recomputation, so MFU $\le$ hardware FLOPs utilization).

**Communication volume per optimizer step.** For 3D parallelism with tensor degree $t$, pipeline degree $p$, data degree $d$ ($tpd = k$), micro-batch $b$, sequence $s$, hidden $h$, $\ell$ layers, precision $\rho$ bytes:
$$V_{\text{tp}} = 4\,\ell\, b\, s\, h\, \rho \cdot \tfrac{t-1}{t}, \quad V_{\text{pp}} = 2(p-1)\, b\, s\, h\, \rho, \quad V_{\text{dp}} = 2N\rho\cdot\tfrac{d-1}{d}.$$
$V_{\text{tp}}$ is the four all-reduces per transformer layer (two forward, two backward); $V_{\text{dp}}$ is the ring gradient all-reduce. The step time is
$$T_{\text{step}} = \max\!\Big(\tfrac{\text{FLOPs}_{\text{step}}}{k F_{\text{peak}}},\ \tfrac{V_{\text{tp}}}{B_{\text{fast}}} + \tfrac{V_{\text{pp}} + V_{\text{dp}}}{B_{\text{slow}}}\Big) + T_{\text{bubble}} + T_{\text{stragglers}}.$$

**The constrained problem.** Instead of $\min_{N,D} L$ s.t. $6ND \le C$, solve
$$\min_{N, D, (t,p,d), b} L(N,D) \quad \text{s.t.} \quad T_{\text{step}}(\cdot)\cdot \frac{D}{B_{\text{global}}} \le T_{\text{budget}},$$
where $B_{\text{global}} = b\,d\,(\text{grad-accum})$ tokens per step. The question is whether the minimizer's $D^\*/N^\*$ differs from the unconstrained $\approx 20$.

**Assumptions, and which are violated.**
1. *Compute and communication perfectly overlap* (the $\max$). Violated: NCCL kernels contend for SMs; observed overlap efficiency is typically 60–90%.
2. *Loss depends only on $(N, D)$, not on batch size.* Violated beyond the critical batch size $B_{\text{crit}}$ (McCandlish et al., 2018) — and large $d$ is the main way to hide communication, so the constraint pushes directly into the regime where the assumption fails.
3. *Bandwidth is homogeneous within a tier.* Violated by oversubscribed fat-trees, rail-optimized topologies, and failure-induced re-placement.
4. *Straggler-free.* Violated: Llama 3 405B training saw 466 job interruptions over 54 days.

## 3. State of the Art

**Systems SOTA (established).** 3D parallelism with selective activation recomputation reaches 52% of peak on 3072 A100s for a 1T-parameter model (Narayanan et al., SC 2021); PaLM 540B reached 46.2% MFU / 57.8% hardware FLOPs utilization on 6144 TPU v4 chips (Chowdhery et al., JMLR 2023); Llama 3 405B reported 38–43% BF16 MFU on up to 16k H100s (Grattafiori et al., 2024). These are measured end-to-end numbers, reproduced in spirit across labs.

**Low-communication SOTA (partly established).** DiLoCo (Douillard et al., 2023) reduces inter-worker communication by ~$500\times$ by synchronizing every $H \approx 500$ inner steps with an outer Nesterov step, at parity or better perplexity on C4 at ~150M–400M scale. "Scaling Laws for DiLoCo" (Charles et al., 2025) fits a law across model sizes to ~10B parameters and reports that DiLoCo's evaluation loss improves *relative* to data-parallel as $N$ grows, and that it tolerates much larger global batch sizes. This is the closest existing instance of the object this page asks for — but it is a law over $(N, D, H, M)$ (replica count), not over measured bandwidth $B$, and it holds $F_{\text{peak}}$ fixed.

**Claimed but unablated.** (a) That low-communication methods are "free" at frontier scale — the DiLoCo scaling fits stop ~2 orders of magnitude below frontier $N$, and the relative advantage is fitted, not derived. (b) Vendor and lab MFU comparisons across clusters: MFU is reported against different peak-FLOPs conventions (with/without sparsity, with/without attention FLOPs), so cross-paper MFU numbers are benchmark numbers, not commensurable measurements.

**Theory SOTA.** No scaling law with bandwidth as an argument has been derived from a training-dynamics model. The available theory is (i) the noise-scale / critical-batch-size analysis of McCandlish et al. (2018), which bounds how far $d$ can grow before step-efficiency collapses, and (ii) classical communication lower bounds for parallel matrix multiplication (Ballard et al., 2011), which bound $V$ from below but say nothing about $L$.

## 4. What Is Known

- **Chinchilla exponents at scale.** $\alpha \approx 0.34$, $\beta \approx 0.28$, $D^\*/N^\* \approx 20$ tokens/param, fitted over 400+ runs, 70M–16B parameters, up to $5\times10^{23}$ FLOPs (Hoffmann et al., 2022). The fit was later shown to have understated confidence intervals (Besiroglu et al., 2024) — the point estimate survived; the error bars widened.
- **Bandwidth ratios.** H100 NVLink gives 900 GB/s bidirectional per GPU intra-node; a 400 Gb/s InfiniBand NIC gives 50 GB/s. The intra/inter ratio is ~18×, and it has *widened* across generations relative to FLOPs growth.
- **Communication share.** For tensor parallelism beyond a node ($t > 8$), measured throughput drops sharply; Megatron-LM's published guidance is to keep $t \le$ node size, which is a bandwidth constraint expressed as an architecture rule.
- **Critical batch size grows with data, not model.** Zhang et al. (2024) report $B_{\text{crit}}$ scaling primarily with $D$ rather than $N$ at 85M–1.2B scale — directly relevant, because $d$ (hence tolerable communication) is capped by $B_{\text{crit}}$.
- **Inference-aware allocation already moves the optimum.** Sardana et al. (2024) show that adding an inference-cost term shifts optimal models smaller and $D/N$ much larger than 20. This is an existence proof that the Chinchilla optimum is not robust to changing the budget's definition.

## 5. What Is Not Known

- **Theoretically open.** Whether $\alpha, \beta$ are invariant under a bandwidth constraint. No derivation connects gradient staleness or reduced-precision all-reduce error to the loss exponents. Also open: the achievable-loss lower bound given $(k, B_{\text{slow}}, T)$ — a communication-limited analogue of Ballard et al.'s bounds, in loss rather than in bytes.
- **Empirically open.** Nobody has run a two-arm scaling-law sweep in which the *only* varied factor is inter-node bandwidth, with everything else (data order, seed, hyperparameters, token budget) matched. Bandwidth throttling is trivially available (NCCL rate limits, `NCCL_MAX_NCHANNELS`, subnet shaping); the experiment costs GPU-hours, not new science.
- **Methodologically blocked.** "Communication-optimal" has no agreed measurement. MFU is not comparable across clusters, loss-vs-wall-clock is not comparable across price points, and no public benchmark fixes $(k, B, T)$ and reports loss. Until the budget variable is standardized, method claims cannot be ranked.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus non-identifiability**. A run's wall-clock time mixes at least five terms — kernel efficiency, communication volume, overlap failure, pipeline bubbles, stragglers/restarts — and only their sum is observed. Two clusters with the same nominal $B_{\text{slow}}$ can differ 1.5× in step time from placement alone. So when a low-communication method wins on loss-vs-hours, the win cannot be attributed to communication without a matched-hardware control, and matched hardware at frontier scale is exactly what is scarce.

Second obstruction: **the confound is with batch size, not just with engineering.** Every method that reduces communication does so by increasing the work per synchronization, which raises the effective global batch. Past $B_{\text{crit}}$, extra batch buys less loss per token. A measured advantage therefore always admits the alternative explanation "this is just a large-batch result," and separating the two requires sweeping $B_{\text{crit}}$ at each $N$ — multiplying the cost of the sweep.

Third: **cost, honestly stated.** A credible bandwidth-arm sweep needs ≥6 model sizes × ≥2 bandwidth arms × ≥3 seeds. At 1B–8B scale that is order $10^{22}$–$10^{23}$ FLOPs, roughly a mid-size lab's quarterly budget, for a negative-result-prone measurement.

## 7. Current Research (as of 2026)

- **Low-communication pretraining.** DiLoCo and Streaming DiLoCo (Google DeepMind); OpenDiLoCo and INTELLECT-1/2 (Prime Intellect), which trained a 10B model across geographically distributed nodes — the first frontier-adjacent demonstration that WAN-bandwidth training is feasible. *(frontier — verify current model sizes.)*
- **Scaling laws for the communication-reduced regime.** Charles et al. (2025) is the reference fit; extensions to heterogeneous replicas and to bandwidth as an explicit variable are in progress *(frontier — verify)*.
- **Hardware-trend modeling.** Work analyzing diminishing returns from adding accelerators to a single job (Fernandez et al., 2024) fits throughput-vs-$k$ curves showing sublinear scaling well below cluster size.
- **Systems co-design.** FP8/FP6 gradient reduction, in-network aggregation (SHARP), and communication-aware optimizer sharding (ZeRO++). These reduce $V$ by constants; whether constants change exponents is exactly the open question.

## 8. Concrete Next Experiment

**The bandwidth-arm IsoFLOP sweep.**

- **Scale.** 6 model sizes: 150M, 300M, 600M, 1.2B, 2.4B, 4.8B non-embedding parameters. Each trained at 3 token budgets spanning $D/N \in \{10, 20, 40\}$. 64 H100s, 8 nodes. Total ≈ $4\times10^{21}$ FLOPs per arm; ~2 weeks per arm on 64 GPUs.
- **Arms.** Identical data order, seed, tuned LR per size, identical software stack. The *only* difference is inter-node all-reduce bandwidth, throttled in NCCL:
  - **Control arm:** full 400 Gb/s inter-node.
  - **Treatment arms:** 40 Gb/s (10×) and 4 Gb/s (100×) — the datacenter-to-WAN range.
  - Parallelism config re-optimized *within* each arm (that is the point: the arm gets to adapt).
- **Deciding number.** Fit $L = E + A N^{-\alpha} + B_c D^{-\beta}$ separately per arm under the *wall-clock* budget, and report
  $$\Delta = \frac{(D^\*/N^\*)_{4\text{ Gb/s}}}{(D^\*/N^\*)_{400\text{ Gb/s}}}.$$
  **If $\Delta \in [0.9, 1.1]$ with bootstrap CI excluding 1.3, the Chinchilla ratio is bandwidth-invariant and communication only rescales the effective compute budget** — the useful, boring answer, and it licenses the standard practice of planning in FLOPs. If $\Delta > 1.3$, communication-limited clusters should train *smaller models on more tokens*, and every frontier allocation plan built on FLOP budgets is mis-specified.
- **Secondary readouts.** Measured MFU per arm; $B_{\text{crit}}$ per size per arm (to check the batch-size confound); $\alpha, \beta$ CIs per arm.

## 9. Key References

- **[Foundational]** J. Hoffmann, S. Borgeaud, A. Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** J. Kaplan, S. McCandlish, T. Henighan, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** S. McCandlish, J. Kaplan, D. Amodei, et al. *An Empirical Model of Large-Batch Training.* 2018. — arXiv:1812.06162
- **[Systems SOTA]** D. Narayanan, M. Shoeybi, J. Casper, et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC 2021. — arXiv:2104.04473
- **[Systems SOTA]** V. Korthikanti, J. Casper, S. Lym, et al. *Reducing Activation Recomputation in Large Transformer Models.* MLSys 2023. — arXiv:2205.05198
- **[Systems SOTA]** S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC 2020. — arXiv:1910.02054
- **[Empirical]** A. Chowdhery, S. Narang, J. Devlin, et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR 24(240), 2023. — arXiv:2204.02311
- **[Empirical]** A. Grattafiori et al. (Llama Team, Meta AI). *The Llama 3 Herd of Models.* 2024. — arXiv:2407.21783
- **[SOTA, low-communication]** A. Douillard, Q. Feng, A. A. Rusu, et al. *DiLoCo: Distributed Low-Communication Training of Language Models.* 2023. — arXiv:2311.08105
- **[SOTA, low-communication]** Z. Charles, G. Teston, L. Dery, et al. *Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo.* 2025. — arXiv:2503.09799
- **[Related]** N. Sardana, J. Portes, S. Doubov, J. Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML 2024. — arXiv:2401.00448
- **[Replication]** T. Besiroglu, E. Erdil, M. Barnett, J. You. *Chinchilla Scaling: A Replication Attempt.* 2024. — arXiv:2404.10102
- **[Theory]** G. Ballard, J. Demmel, O. Holtz, O. Schwartz. *Minimizing Communication in Numerical Linear Algebra.* SIAM J. Matrix Anal. Appl. 32(3), 2011.
- **[Survey]** T. Ben-Nun, T. Hoefler. *Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis.* ACM Computing Surveys 52(4), 2019. — arXiv:1802.09941

## 10. Worked Example

Take a 7B-parameter transformer ($\ell = 32$, $h = 4096$, $s = 8192$), 512 H100s in 64 nodes, BF16 ($\rho = 2$).

**Gradient all-reduce.** With $t = 8$ (intra-node), $p = 1$, $d = 64$ across nodes:
$$V_{\text{dp}} = 2 \cdot 7\times10^9 \cdot 2 \cdot \tfrac{63}{64} \approx 2.76\times10^{10}\ \text{bytes per device per step}.$$
At a *measured* inter-node bus bandwidth of 45 GB/s (400 Gb/s link, ~90% achieved):
$$T_{\text{comm}} \approx 2.76\times10^{10} / 4.5\times10^{10} \approx 0.61\ \text{s}.$$

**Compute.** Global batch 4M tokens: step FLOPs $= 6 \cdot 7\times10^9 \cdot 4.2\times10^6 \approx 1.76\times10^{17}$. At 512 GPUs × $9.9\times10^{14}$ FLOP/s × 40% MFU $= 2.03\times10^{17}$ FLOP/s:
$$T_{\text{compute}} \approx 0.87\ \text{s}.$$

Communication is 0.61 s against 0.87 s of compute — hideable, if overlap is perfect. Now throttle to 4.5 GB/s (a WAN-like link): $T_{\text{comm}} = 6.1$ s, and step time rises from ~0.87 s to ~6.1 s. **Effective MFU falls from 40% to 5.7%.** In FLOP-budget terms the same wall-clock now buys $7\times$ less compute — which, through $L \propto C^{-0.05}$ near the Chinchilla frontier, costs roughly $1 - 7^{-0.05} \approx 10\%$ of the loss gap above the irreducible term $E$.

**Where the obstruction becomes visible.** The obvious fix is to raise the global batch from 4M to 28M tokens, so $T_{\text{compute}}$ rises to 6.1 s and communication hides again — restoring 40% MFU with *zero* extra communication (the all-reduce volume $2N\rho$ is batch-independent). If $B_{\text{crit}}$ at this scale is ≥28M tokens, bandwidth costs nothing and $\Delta = 1$. If $B_{\text{crit}}$ is 4M, the larger batch wastes most of its tokens and bandwidth costs the full 10%.

The two hypotheses — "bandwidth is free, just batch harder" and "bandwidth is a real constraint" — make *identical* predictions for step time and MFU, and differ only in loss-per-token at large batch. Every published MFU number in Section 4 is therefore silent on the question this page asks. That is the non-identifiability: the standard reported metric cannot distinguish the two arms, and only the §8 experiment, which measures loss under matched wall-clock and swept $B_{\text{crit}}$, can.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*