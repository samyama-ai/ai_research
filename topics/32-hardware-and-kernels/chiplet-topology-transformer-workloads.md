---
id: 32-hardware-and-kernels/chiplet-topology-transformer-workloads
title: "Chiplet Interconnect Topology Optimal for Transformer Workloads"
topic: 32-hardware-and-kernels
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Chiplet Interconnect Topology Optimal for Transformer Workloads

> **Topic:** Hardware & Kernels · **ID:** `32-hardware-and-kernels/chiplet-topology-transformer-workloads` · **Status:** open

## 1. Problem Statement

A chiplet package holds $N$ dies connected by die-to-die links on an organic substrate, silicon interposer, or bridge. The **topology** is which dies are wired to which, and how wide each link is. Given a transformer workload (prefill or decode, dense or mixture-of-experts, a fixed parallelism plan), which topology minimizes cost-normalized latency or maximizes cost-normalized throughput?

Three variants, with different difficulty:

- **Measurement.** Given two packaged topologies and one workload, decide which is faster at equal silicon area and equal package power. Requires either silicon or a simulator whose die-to-die model is validated against silicon.
- **Method.** Given a workload distribution and a fabrication cost model, *synthesize* the link graph and per-link width. This is a constrained graph design problem, currently solved by hand plus a small sweep.
- **Theory.** Prove a lower bound on time-to-token for any degree-$d$, bisection-$B$ topology executing a given collective schedule, and exhibit a topology meeting it. Only fragments exist.

Solving it means: a topology $T^\star$ plus a proof or a controlled measurement that no topology in the same area/power/cost envelope beats it by more than a stated margin on a stated workload mix.

## 2. Formal Setting

Package graph $G=(V,E)$, $|V|=N$ chiplets. Each edge $e$ carries $w_e$ lanes at $r$ Gb/s/lane, so link bandwidth $\beta_e = w_e r/8$ B/s. Measured quantities:

- **Shoreline budget.** Each die has beachfront $L_i$ (mm). With bandwidth density $\rho$ (GB/s per mm of shoreline, measured on a PHY test chip, not from a datasheet), $\sum_{e \ni i} \beta_e \le \rho L_i$.
- **Link energy** $\epsilon$ (pJ/bit), measured as package wall power delta under a saturating traffic pattern divided by bits moved — *not* the PHY-only figure vendors quote, which excludes the on-die network and serdes clocking.
- **Per-hop latency** $\lambda$ (ns), measured as a one-way ping-pong across one hop at zero load.
- **Bisection bandwidth** $B(G) = \min_{S \subset V, |S|=N/2} \sum_{e \in \delta(S)} \beta_e$.

Workload: batch $b$, sequence $s$, hidden $H$, layers $L$, tensor-parallel degree $p$, expert-parallel degree $q$. Per-layer collective volume under tensor parallelism is two all-reduces of $bsH$ elements at $\gamma$ bytes/element. Ring all-reduce time on a topology admitting a Hamiltonian cycle:

$$
T_{\text{AR}} = 2(p-1)\left(\frac{bsH\gamma}{p\,\beta} + \lambda_{\text{eff}}\right),
$$

with $\lambda_{\text{eff}}$ the measured per-step software+hop latency. Compute time per layer is $T_{\text{cmp}} = \max\!\left(\frac{F}{p\,\Phi},\ \frac{W\gamma}{p\,\mathrm{BW}_{\text{mem}}}\right)$ for $F$ FLOPs, $W$ weights, peak rate $\Phi$, memory bandwidth $\mathrm{BW}_{\text{mem}}$. Objective:

$$
T^\star = \arg\min_{G}\ \mathbb{E}_{\text{workload}}\big[L(T_{\text{cmp}} + T_{\text{AR}} + T_{\text{a2a}})\big]
\quad \text{s.t.}\quad \text{shoreline},\ P_{\text{link}} = \epsilon \cdot \text{bits/s} \le P_{\max},\ \text{cost}(G) \le C.
$$

Assumptions and their status:
- *Uniform link bandwidth and latency.* **Violated.** Simba measured non-uniform inter-chiplet latency and bandwidth that materially changed the optimal tiling (MICRO 2019).
- *Compute and communication overlap perfectly.* **Violated.** Tensor-parallel all-reduces sit on the critical path; overlap is partial and kernel-dependent.
- *Traffic is uniform-random.* **Violated.** Transformer traffic is a fixed, known collective pattern — which is why generic NoC topology results transfer poorly.
- *Cost is linear in die area.* **Violated.** Yield is superlinear in area; the chiplet argument rests on this.

## 3. State of the Art

**Systems / empirical.** Production packages are shallow and near-uniform, not exotic. AMD MI300X uses 8 accelerator dies over 4 I/O dies with Infinity Fabric; the EPYC/Ryzen chiplet program is documented by Naffziger et al. (ISCA 2021). NVIDIA GB200 NVL72 pairs two dies per Blackwell package and pushes topology off-package to NVLink (72 GPUs, 1.8 TB/s bidirectional per GPU). Google TPU v4 solves the same problem one level up with a 3D torus plus optical circuit switches that reconfigure topology per job (Jouppi et al., ISCA 2023) — the strongest evidence that topology reconfigurability pays, but at rack scale, not package scale.

**Research.** Simba (Shao et al., MICRO 2019) is the reference multi-chip-module DNN accelerator: 36 chiplets, 128 TOPS peak, ground-referenced signaling at 11 Gb/s/pin, and the first careful account of MCM-specific non-uniformity. Chiplet Cloud (Peng et al., 2023, arXiv:2307.02666) argues LLM serving TCO favors many small chiplets with large on-die SRAM; the result is a cost model plus simulation, **not** silicon. UCIe 1.0+ standardizes the interface: advanced-package targets ~0.25 pJ/bit and <2 ns link latency; standard package roughly 0.5 pJ/bit.

**Established vs. claimed.** Established: chiplet disaggregation improves yield-adjusted cost; die-to-die links are ~1–2 orders of magnitude worse in energy/bit than on-die wires; non-uniform placement matters (Simba, on silicon). Claimed but unablated: that any *particular* topology family (mesh vs. flattened butterfly vs. dragonfly vs. fully connected) is best for transformers. The published comparisons are simulator sweeps with an assumed link model; none holds area, power, and package cost fixed while varying only topology on fabricated parts.

## 4. What Is Known

- **Topology theory, pre-transformer.** Dally's k-ary n-cube analysis under fixed wire bisection (IEEE TC, 1990) shows low-dimensional networks win when bisection, not node degree, is the binding constraint. Flattened butterfly (Kim, Balfour, Dally, ISCA 2007) and dragonfly (Kim et al., ISCA 2008) reduce hop count at fixed bisection cost. These assume uniform-random traffic; transformer traffic is not.
- **Energy gap.** UCIe advanced-package target ~0.25 pJ/bit versus on-die NoC traversal in the tens of fJ/bit. At 1 TB/s sustained, 0.25 pJ/bit is 2 W of pure link power.
- **Collective cost is scale-invariant in the ring.** Ring all-reduce bandwidth term $2(p-1)/p \cdot V/\beta$ is nearly flat in $p$; the latency term $2(p-1)\lambda$ is not. So topology matters via $\lambda$ and hop count, mostly at small messages.
- **Reconfiguration pays at rack scale.** TPU v4's OCS reported up to ~1.2–2.3× throughput improvement for jobs whose communication pattern matches a reshaped torus (ISCA 2023, 4096-chip scale).
- **Non-uniformity is measurable and exploitable.** Simba, 36 chiplets: communication-aware tiling and placement changed achieved performance materially versus uniform tiling on the same silicon.

## 5. What Is Not Known

- **Empirically open.** No study varies package topology alone — fixed total die area, fixed link power, fixed HBM — across ≥3 topology families on real transformer prefill *and* decode. The parts exist (UCIe test vehicles, FPGA-emulated D2D); nobody has run the controlled comparison publicly.
- **Theoretically open.** No lower bound on time-to-token for degree-$d$, bisection-$B$ packages under the specific transformer collective schedule (all-reduce + MoE all-to-all interleaved with GEMMs). Existing bounds are for uniform-random or permutation traffic.
- **Methodologically blocked.** "Optimal topology" is not well posed until cost is defined. Package cost, yield, and NRE are vendor-confidential; every public comparison substitutes a guessed cost model, so two papers can rank the same topologies oppositely without either being wrong.

## 6. Why It Is Hard

**The comparison is confounded at the point where it matters.** Changing topology changes floorplan, which changes die area, PHY count, shoreline use, and thermal density — all of which change clock and memory bandwidth. A measured 8% throughput difference between mesh and torus cannot be attributed to topology without holding four things fixed that fabrication does not let you hold fixed.

**And the signal is small where the money is.** For dense decode, the workload is memory-bandwidth-bound; the all-reduce is a few percent of step time (Section 10). Topology differences land inside run-to-run variance. Where the signal is large — MoE all-to-all, long-context prefill, very high tensor-parallel degree — the workload itself is changing fastest, so the target moves faster than a 24-month silicon cycle. Simulation avoids the confound but reintroduces it as an unvalidated die-to-die model: the ranking is a function of assumed $\lambda$ and $\epsilon$, and those are exactly the numbers vendors do not publish honestly.

## 7. Current Research (as of 2026)

- **UCIe ecosystem hardening** — UCIe 2.0/3.0 add manageability and higher rates; the practical effect is that topology becomes a design variable for non-vertically-integrated teams *(frontier — verify specific rate figures against the current spec release)*.
- **Cost-model-driven chiplet synthesis** — Chiplet Cloud line (Washington/Microsoft) and successors optimizing TCO per token rather than peak FLOPs.
- **Reconfigurable and optical package-level interconnect** — extending the TPU v4 OCS idea downward to the package; several groups and startups *(frontier — verify)*.
- **Communication-aware mapping** — Alpa-style automatic parallelization (Zheng et al., OSDI 2022) applied with non-uniform link costs; the search space is the same, the cost matrix is not.
- **MoE-driven interest in all-to-all** — as sparse models dominate serving, the binding collective shifts from all-reduce to all-to-all, which rewards higher bisection over lower diameter.

## 8. Concrete Next Experiment

**Scale.** One 8-chiplet test vehicle, or an FPGA/emulation platform with programmable D2D links, running a 70B-class dense model at $p=8$ tensor parallel plus a 8×A2 MoE layer stack.

**Design.** Fix total die area, HBM bandwidth per die, and total link power budget $P_{\max}$. Vary only the link graph across three arms at identical $\sum_e \beta_e$:
1. **Ring** ($d=2$, low bisection, Hamiltonian);
2. **2D mesh/torus** $4\times2$ ($d=3$–4);
3. **Fully connected** $K_8$ ($d=7$, links narrowed to keep $\sum \beta_e$ constant).

**Control arm.** Arm 2 (mesh) is the control — it is what production ships. Additionally run a *monolithic-equivalent* control: the same total area as one simulated die with zero D2D cost, to bound the maximum possible topology gain.

**The deciding number.** Median tokens/s/W at fixed 99th-percentile time-per-output-token, reported for each arm, as a ratio to the mesh control. If no arm differs from mesh by more than **5%** on dense decode but $K_8$ beats mesh by more than **20%** on the MoE all-to-all layer, the answer is "topology is workload-conditional, and MoE is what should drive it" — which would redirect the field from generic topology search to all-to-all-specific bisection provisioning.

## 9. Key References

- **[Foundational]** W. J. Dally. *Performance Analysis of k-ary n-cube Interconnection Networks.* IEEE Transactions on Computers, 1990.
- **[Foundational]** J. Kim, J. Balfour, W. J. Dally. *Flattened Butterfly Topology for On-Chip Networks.* MICRO, 2007.
- **[Foundational]** J. Kim, W. J. Dally, S. Scott, D. Abts. *Technology-Driven, Highly-Scalable Dragonfly Topology.* ISCA, 2008.
- **[SOTA]** Y. S. Shao, J. Clemons, R. Venkatesan, et al. *Simba: Scaling Deep-Learning Inference with Multi-Chip-Module-Based Architecture.* MICRO, 2019.
- **[SOTA]** N. Jouppi, G. Kurian, S. Li, et al. *TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings.* ISCA, 2023.
- **[SOTA]** H. Peng, S. Davidson, R. Shi, S. L. Song, M. Taylor. *Chiplet Cloud: Building AI Supercomputers for Serving Large Generative Language Models.* 2023 — arXiv:2307.02666.
- **[Systems]** S. Naffziger, N. Beck, T. Burd, et al. *Pioneering Chiplet Technology and Design for the AMD EPYC and Ryzen Processor Families.* ISCA, 2021.
- **[Systems]** L. Zheng, Z. Li, H. Zhang, et al. *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.* OSDI, 2022.
- **[Standard]** UCIe Consortium. *Universal Chiplet Interconnect Express (UCIe) Specification*, rev. 1.0 (2022) and later revisions.

## 10. Worked Example

Eight chiplets, $p=8$ tensor parallel, dense decode. $H=8192$, $L=80$, bf16 ($\gamma=2$), batch $b=32$, one new token per step ($s=1$). Per chiplet: 1.5 TB/s HBM, 256 GB/s per D2D link, $\lambda_{\text{eff}}=0.2\ \mu$s.

**Compute (memory-bound).** Weights per layer $\approx 12H^2 = 8.05\times10^8$ params $= 1.61$ GB; per chiplet $201$ MB; at 1.5 TB/s: $T_{\text{cmp}} = 134\ \mu$s per layer.

**Communication.** Two all-reduces per layer, each $bsH\gamma = 32\cdot 8192\cdot 2 = 0.52$ MB. Ring, $p=8$: 14 steps of 65 KB chunks.
$$
T_{\text{AR}} = 14\left(\frac{65\text{ KB}}{256\text{ GB/s}} + 0.2\,\mu s\right) = 14(0.26 + 0.2) = 6.4\ \mu s.
$$
Two per layer: $12.8\ \mu$s, i.e. **8.7%** of the $147\ \mu$s layer time.

**Now the topology swap.** $K_8$ at constant total bandwidth: each of 28 links gets $8\cdot 256/28 = 73$ GB/s. One-shot all-reduce: each chiplet sends $0.52$ MB $\times 7/8 = 0.46$ MB split over 7 links, $65$ KB per link at 73 GB/s $= 0.9\ \mu$s, plus reduce-broadcast $\approx 2\times$: $\approx 2.0\ \mu$s. Two per layer: $4.0\ \mu$s. Layer time $138\ \mu$s.

**Predicted end-to-end gain: 6%.** That is the whole prize for replacing a ring with a fully connected package on dense decode — and it sits under the 3–8% run-to-run spread typical of a serving benchmark, under the clock variation induced by re-floorplanning 28 PHYs instead of 8, and under the error bar on $\lambda_{\text{eff}}$, which was assumed, not measured. The obstruction is not that the experiment is expensive. It is that at the workload point that pays the bills, the effect size and the confound are the same size. Move to an MoE layer where all-to-all volume is $bsH\gamma$ per expert-parallel hop and the bisection term grows by roughly $8\times$ — there the arms separate, and that is where the experiment in Section 8 must be aimed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*