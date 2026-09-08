---
id: 31-distributed-training/communication-aware-scaling-laws
title: "Scaling Laws That Include Communication Cost"
topic: 31-distributed-training
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Scaling Laws That Include Communication Cost

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/communication-aware-scaling-laws` · **Status:** open

## 1. Problem Statement

Compute-optimal scaling laws (Kaplan et al. 2020; Hoffmann et al. 2022) predict loss from parameters $N$ and tokens $D$ under a **FLOP** budget $C \approx 6ND$. FLOPs are not what a training run spends. A run spends wall-clock time on a fixed cluster, and a growing share of that time is spent moving bytes: all-reduce of gradients, all-gather/reduce-scatter of sharded parameters, point-to-point pipeline transfers, and all-to-all in MoE layers.

The problem: **produce a scaling law whose budget variable is accelerator-seconds on a specified interconnect topology, not FLOPs, and show that its predicted optimum differs from the FLOP-optimal one by a measurable amount.**

Three variants, different difficulty:

- **Measurement.** Given a run, decompose wall-clock into compute, exposed communication, and stall, in a way that is reproducible across frameworks. Hard because overlap makes "communication time" not additively separable.
- **Method.** Given a cluster $(P, B_{\text{intra}}, B_{\text{inter}}, \ell)$ and a time budget $T$, choose $(N, D, b, \text{parallelism plan}, \text{sync period } H)$ minimizing final loss. Currently done by search plus intuition.
- **Theory.** Is there a closed-form $L(N, D, \text{comm})$ with transferable exponents, or is the communication term irreducibly hardware-specific so that only the *shape* of the law transfers and every constant must be refit per cluster?

Solving it means: a fitted law on cluster $A$ that predicts, within a stated error bar, the loss-optimal configuration on cluster $B$ with different bandwidth, and beats the Chinchilla-optimal configuration in iso-wall-clock loss.

## 2. Formal Setting

**Objects and how each is measured.**

- $N$ — non-embedding parameters, counted from the model definition.
- $D$ — training tokens consumed, counted post-packing (not pre-dedup corpus size).
- $L$ — loss in nats/token on a held-out set drawn from the *training* distribution, measured at the final step, not the minimum over steps.
- $P$ — accelerator count; $T$ — wall-clock seconds, measured from first optimizer step to last, excluding checkpoint restarts. Restart time must be reported separately, because at $P > 10^4$ it is not negligible.
- $B_{\text{intra}}, B_{\text{inter}}$ — achieved unidirectional bandwidth (bytes/s) inside a node and across the fabric, measured by a NCCL/RCCL microbenchmark at the message size the run actually uses, not the vendor peak.
- $\ell$ — end-to-end latency (s) of a zero-byte collective at the relevant group size.

**Collective cost.** Under the $\alpha$–$\beta$ model, a bandwidth-optimal ring all-reduce over $p$ ranks on $m$ bytes costs
$$t_{\text{ar}}(m,p) = 2(p-1)\,\ell + \frac{2(p-1)}{p}\cdot\frac{m}{B}.$$

**Time model.** With microbatch compute time $t_c$ and exposed (non-overlapped) communication $t_e$ per step, and $S = D/(b\,s)$ steps at global batch $b$ tokens:
$$T = S\,(t_c + t_e) + T_{\text{restart}}, \qquad \text{MFU} = \frac{6ND}{P\,T\,F_{\text{peak}}}.$$

**The object sought.** A law of the form
$$L(N, D \mid \mathcal{H}) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + E, \quad \text{minimized subject to} \quad T(N, D, \mathcal{H}) \le T_{\max},$$
where $\mathcal{H} = (P, B_{\text{intra}}, B_{\text{inter}}, \ell, F_{\text{peak}})$ enters only through the *constraint*, not through $L$ — the weak form — or through both, if communication-reduction methods (large $b$, low-precision gradients, infrequent sync) degrade loss at fixed $(N,D)$ — the strong form. Whether the strong form is needed is itself open.

**Assumptions known to be violated.**

1. $C = 6ND$ ignores activation recomputation (+~30% forward FLOPs) and attention's $O(s^2)$ term, which is >10% of FLOPs at $s \ge 16$k.
2. The $\alpha$–$\beta$ model assumes congestion-free, contention-free links. Real fabrics see incast and oversubscription; measured all-reduce on a busy cluster deviates from it by tens of percent.
3. Single-epoch, IID data. Violated in data-constrained regimes.
4. Loss is a function of $(N, D)$ alone. Violated: batch size, optimizer state precision, and gradient compression all move loss at fixed $(N,D)$ — which is exactly the coupling that makes the strong form necessary.

## 3. State of the Art

**Theory SOTA.** No published law makes bandwidth an explicit variable of the loss surface. The closest structural precedent is Sardana & Frankle (ICML 2024), *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, which adds a non-training cost term to the Chinchilla objective and shows the optimum shifts toward smaller $N$, larger $D$. The communication analogue has not been written down and fitted. Established.

**Systems SOTA.** Performance models that predict step time from $(N, b, \text{parallelism}, \mathcal{H})$ exist and are validated: Narayanan et al. (SC 2021, Megatron-LM on 3072 A100s), Zheng et al. (*Alpa*, OSDI 2022, automatic inter/intra-operator parallelism search), Isaev et al. (*Calculon*, SC 2023, high-level co-design of LLM and system). These predict **time**, then hand the loss question back to Chinchilla. Nobody joins the two surfaces and refits the exponents.

**Communication-reduction SOTA.** DiLoCo (Douillard et al. 2023) trains with inner-optimizer steps between syncs, reporting ~500× fewer communications; Streaming DiLoCo (Douillard et al. 2025) reports further bandwidth reduction by partial, overlapped parameter exchange. These are **claimed at small scale and largely unablated against a matched-wall-clock baseline** — the comparisons are typically iso-step or iso-token, not iso-seconds-on-a-named-cluster, which is the comparison the scaling-law question needs.

**Benchmark-number-only results.** Llama 3 (Grattafiori et al. 2024) reports 405B params, 16,384 H100s, BF16 MFU in the 38–43% band; DeepSeek-V3 (2024) reports 2.788M H800 GPU-hours for 14.8T tokens with DualPipe overlap. Both are single points. Neither sweeps bandwidth, so neither constrains a law.

## 4. What Is Known

- **Chinchilla exponents.** Hoffmann et al. (2022) fit $\alpha \approx 0.34$, $\beta \approx 0.28$, $E \approx 1.69$ over 400 runs, 70M–16B params, up to 400B tokens; $D^\star/N^\star \approx 20$ tokens/param. Reproduced qualitatively by others; the fit's confidence intervals were shown to be understated by Besiroglu et al. (2024), *Chinchilla Scaling: A Replication Attempt*.
- **Critical batch size.** McCandlish et al. (2018) show a gradient-noise-scale threshold above which increasing $b$ buys no step-count reduction. This is the ceiling on "just use a bigger batch to hide communication," and it grows during training.
- **Communication share is large and topology-dependent.** Megatron-LM at 3072 A100s achieved 52% of peak with tensor parallelism confined inside 8-GPU NVLink nodes (Narayanan et al., SC 2021); the design constraint is explicitly that tensor-parallel all-reduce must not cross the slower fabric.
- **Sharding trades bytes for memory.** ZeRO (Rajbhandari et al., SC 2020) stage-3 removes redundant optimizer state at the price of extra parameter all-gathers per step — a known, quantified bandwidth-for-memory exchange.
- **Local SGD converges.** Stich (ICLR 2019) gives convergence rates for local SGD with $H$ local steps, showing the same asymptotic rate as mini-batch SGD for $H$ below a threshold in the convex setting. The threshold's non-convex, LLM-scale analogue is not established.

## 5. What Is Not Known

- **Theoretically open.** Whether the loss surface admits a *separable* communication term at all — i.e. whether $L$ depends on $\mathcal{H}$ only through the achievable $(N, D, b)$ (weak form) or irreducibly through sync frequency and gradient precision (strong form). No proof either way.
- **Empirically open.** The bandwidth sweep. Fit Chinchilla-style laws at 3–4 bandwidth settings on the same hardware and test whether $\alpha, \beta$ move. Runnable today with a bandwidth-throttled cluster; nobody has published it. Cost is the only barrier: roughly $10^{2}$ runs × $10^{2}$–$10^{3}$ GPU-hours.
- **Empirically open.** Whether the DiLoCo family's loss-neutrality survives past ~10B params. All public matched comparisons are below that.
- **Methodologically blocked.** "Communication cost" has no agreed measurement under overlap. When a collective runs concurrently with GEMMs on separate SMs, it both hides latency and steals compute throughput. Charging it zero seconds (fully overlapped) and charging it its isolated duration are both wrong, and the framework profilers disagree.

## 6. Why It Is Hard

**Non-identifiability between the batch-size effect and the communication effect.** Every practical way to reduce communication also changes optimization: larger $b$ reduces sync count *and* moves the loss trajectory; longer sync period $H$ reduces bytes *and* changes the effective update. So a run that reaches lower loss per second cannot be attributed to the communication saving without a matched-$b$, matched-$H$ control — and such a control usually cannot be run, because the point of the change was that the control does not fit the cluster.

Compounding this: **the profiler does not measure the thing it names.** Overlapped-collective time is reported inconsistently, so the exposed-communication fraction — the single input the law most needs — is framework-dependent, not a property of the run.

Third: **each data point is a full pretraining run.** Fitting exponents needs $\sim10^2$ runs; doing that at each of several bandwidths multiplies it. Extrapolation from 100M-param proxies is what is in question, so cannot be assumed.

## 7. Current Research (as of 2026)

- **Low-communication distributed training.** Google DeepMind's DiLoCo line; open replications by Prime Intellect (INTELLECT-1, 10B, geographically distributed) and Nous Research (DisTrO/DeMo). *(frontier — verify current scale claims.)*
- **Co-design performance models.** Calculon-style analytic models extended toward joint loss/time optimization *(frontier — verify)*.
- **Cost-inclusive scaling laws.** Follow-ons to Sardana & Frankle adding serving and hardware cost terms; the communication term is the obvious missing one.
- **Overlap-aware kernels.** DeepSeek's DualPipe, NVIDIA's fine-grained tensor-parallel overlap. These change the constant that the law would fit, which is why the law must specify the overlap regime.

## 8. Concrete Next Experiment

**Question.** Do the Chinchilla exponents move when bandwidth changes, or is bandwidth purely a constraint?

**Scale.** One cluster, 64 nodes × 8 H100 (512 GPUs). Model grid $N \in \{300\text{M}, 1\text{B}, 3\text{B}, 7\text{B}\}$; token grid $D \in \{20N, 40N, 80N\}$ — 12 runs per arm, ~$10^4$ GPU-hours per arm.

**Arms.** Inter-node bandwidth throttled (NCCL socket/QP limits or NIC rate limiting) to $\{400, 100, 25\}$ Gb/s, with the *same* parallelism plan and global batch in all three. **Control arm:** the 400 Gb/s arm at Chinchilla-optimal $(N^\star, D^\star)$ for the wall-clock budget the 25 Gb/s arm achieves — i.e. the FLOP-optimal recommendation, penalized only by slower steps.

**Deciding number.** Refit $L = A/N^\alpha + B/D^\beta + E$ per arm. **If $|\alpha_{25} - \alpha_{400}| < 0.02$ and $|\beta_{25} - \beta_{400}| < 0.02$** (against bootstrap CIs from the run grid), the weak form holds: communication is a constraint, and the existing law plus a step-time model suffices. **If either exponent moves by $>0.05$**, the strong form is required and a communication term belongs inside $L$.

**Secondary readout.** Iso-wall-clock final loss of the communication-aware optimum minus the control, in nats. A gap $>0.01$ nats at 7B is practically material — comparable to a ~2× compute change near that scale.

## 9. Key References

- **[Foundational]** Jared Kaplan, Sam McCandlish, et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS 2022. — arXiv:2203.15556
- **[Foundational]** Sam McCandlish, Jared Kaplan, Dario Amodei, et al. *An Empirical Model of Large-Batch Training.* 2018. — arXiv:1812.06162
- **[SOTA]** Nikhil Sardana, Jacob Portes, Sasha Doubov, Jonathan Frankle. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws.* ICML 2024. — arXiv:2401.00448
- **[SOTA]** Deepak Narayanan, Mohammad Shoeybi, Jared Casper, et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC 2021. — arXiv:2104.04473
- **[SOTA]** Lianmin Zheng, Zhuohan Li, Hao Zhang, et al. *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.* OSDI 2022. — arXiv:2201.12023
- **[SOTA]** Mikhail Isaev, Nic McDonald, Larry Dennison, Richard Vuduc. *Calculon: A Methodology and Tool for High-Level Codesign of Systems and Large Language Models.* SC 2023.
- **[SOTA]** Arthur Douillard, Qixuan Feng, Andrei Rusu, et al. *DiLoCo: Distributed Low-Communication Training of Language Models.* 2023. — arXiv:2311.08105
- **[Foundational]** Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC 2020. — arXiv:1910.02054
- **[Foundational]** Sebastian U. Stich. *Local SGD Converges Fast and Communicates Little.* ICLR 2019. — arXiv:1805.09767
- **[Replication]** Tamay Besiroglu, Ege Erdil, Matthew Barnett, Josh You. *Chinchilla Scaling: A Replication Attempt.* 2024. — arXiv:2404.10102
- **[Survey/Report]** Aaron Grattafiori et al. (Llama Team, Meta AI). *The Llama 3 Herd of Models.* 2024. — arXiv:2407.21783

## 10. Worked Example

**Setup.** 7B dense model, 512 H100s, data-parallel across 64 nodes, ZeRO-1, BF16 gradients. Gradient payload $m = 7\times10^9 \times 2 = 14$ GB. Global batch $4\text{M}$ tokens.

**Compute per step.** $6ND_{\text{step}} = 6 \times 7\times10^9 \times 4\times10^6 = 1.68\times10^{17}$ FLOPs. At 40% of $989$ TFLOP/s BF16 across 512 GPUs $\approx 2.03\times10^{17}$ FLOP/s, so $t_c \approx 0.83$ s.

**Communication per step.** Ring all-reduce over 64 nodes, $\frac{2(p-1)}{p}\frac{m}{B} \approx 1.97 \times 14\text{GB} / B$.

| Inter-node $B$ | all-reduce time | if fully exposed, step time | throughput vs 400 Gb/s |
|---|---|---|---|
| 400 Gb/s (50 GB/s) | 0.55 s | 1.38 s | 1.00× |
| 100 Gb/s (12.5 GB/s) | 2.21 s | 3.04 s | 0.45× |
| 25 Gb/s (3.1 GB/s) | 8.83 s | 9.66 s | 0.14× |

**The FLOP-optimal answer.** Chinchilla says train 7B on 140B tokens regardless of $B$. At 25 Gb/s that costs $7.0\times$ the wall-clock of the 400 Gb/s run.

**Where the obstruction appears.** The obvious fix at 25 Gb/s is to raise $b$ to $16$M tokens, cutting sync count $4\times$ and bringing throughput back to ~0.45×. Suppose the resulting run lands 0.03 nats *above* the small-batch run at equal tokens. Is that penalty (a) the batch exceeding critical batch size, (b) a learning-rate schedule not retuned for the new $b$, or (c) something intrinsic to low-bandwidth training? The matched control — $b = 16$M at 400 Gb/s — separates (a)+(b) from (c), and it is cheap. **The control that cannot be run is the reverse:** $b = 4$M at 25 Gb/s to full 140B tokens, which needs 9.66 s/step × 35,000 steps ≈ 94 hours of exclusive 512-GPU time, and it is precisely the arm the law needs to anchor its constant.

Second obstruction, visible in the same numbers: with DualPipe-style overlap, the 400 Gb/s row's 0.55 s is mostly hidden, and the profiler reports exposed communication near 0.05 s — but measured step time is 1.02 s, not 0.88 s. The missing 0.14 s is SM contention. Charge it to compute and MFU looks poor; charge it to communication and the "communication fraction" is 3× the profiler's number. Until that attribution is fixed by convention, the law's own independent variable is ambiguous at the ~10% level — larger than the exponent shift the experiment in §8 is designed to detect.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*