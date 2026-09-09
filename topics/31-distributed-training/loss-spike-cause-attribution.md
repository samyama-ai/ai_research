---
id: 31-distributed-training/loss-spike-cause-attribution
title: "Attribution of Loss Spikes to Systems Versus Optimization Causes"
topic: 31-distributed-training
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Attribution of Loss Spikes to Systems Versus Optimization Causes

> **Topic:** Distributed Training Systems · **ID:** `31-distributed-training/loss-spike-cause-attribution` · **Status:** methodologically-blocked

## 1. Problem Statement

A large pretraining run shows a sudden jump in training loss — 0.1 to 2+ nats over a handful of steps, sometimes recovering, sometimes not. **Input:** the run's telemetry (per-step loss, gradient norms, optimizer state moments, activation statistics, the data batches consumed, and the systems log: node health, ECC counters, NCCL timings, checkpoint boundaries, elastic-restart events). **Output:** an attribution of the spike to a cause class, with a calibrated confidence:

1. **Optimization** — curvature/edge-of-stability dynamics, Adam second-moment collapse, attention-logit or output-logit growth;
2. **Data** — a pathological batch (repeated n-grams, encoding artifacts, a corrupt shard);
3. **Systems** — silent data corruption (SDC), a bit flip in a weight or activation, a mis-ordered or partial all-reduce, a stale parameter after elastic restart, a mismatched optimizer-state shard on resume.

**Decision predicate:** given a spike $s$, output $\hat{c}(s) \in \{\text{opt}, \text{data}, \text{sys}\}$ such that the intervention implied by $\hat c$ (change LR/$\epsilon$/normalization; skip batches; fence and replace hardware) prevents recurrence.

The three variants differ sharply:
- **Measurement variant** (the blocked one): define an estimand for "the cause of this spike" that is identifiable from a *single* run, given that the run cannot be replayed bitwise.
- **Method variant:** build an online classifier with useful precision/recall at 10k-GPU scale.
- **Theory variant:** prove that a given optimizer/architecture pair either admits or excludes spikes of magnitude $\ge \delta$ under bounded gradient noise.

Solving it means: a detector that fires within $k$ steps of the spike and whose recommended intervention is right more often than the standing default (rewind and skip batches).

## 2. Formal Setting

Let $\theta_t \in \mathbb{R}^d$ be parameters at step $t$, $B_t$ the global batch, $L(\theta; B)$ the mean token loss in nats. Define the **observed** series $\ell_t = L(\theta_t; B_t)$ — this is what the logger writes, and it is *not* $L(\theta_t;\mathcal{D})$: it confounds parameter quality with batch difficulty.

**Spike detector (as measured).** With a causal median filter of width $w$ ($w=50$ typical) and robust scale $\hat\sigma_t = 1.4826\,\mathrm{MAD}(\ell_{t-w:t-1})$:
$$z_t = \frac{\ell_t - \mathrm{med}(\ell_{t-w:t-1})}{\hat\sigma_t},\qquad \text{spike} \iff z_t > \tau\ (\tau \approx 8).$$
$\tau$ is a convention, not a physical threshold; nothing in the literature fixes it.

**Batch-difficulty control.** The counterfactual $\ell_t^{\mathrm{ctl}} = L(\theta_{t-1}; B_t)$ — the same batch under the *previous* parameters — separates "hard batch" from "broken parameters". It costs one extra forward pass ($\approx 1/3$ of a step) and is almost never logged.

**Systems estimand.** Let $g_t^{(r)}$ be rank $r$'s local gradient and $\bar g_t = \frac{1}{R}\sum_r g_t^{(r)}$ the all-reduced result. Define the **reduction residual**
$$\rho_t = \frac{\|\bar g_t^{\text{obs}} - \frac{1}{R}\sum_r g_t^{(r)}\|_2}{\|\bar g_t^{\text{obs}}\|_2},$$
which is nonzero only from numerics or corruption. Measuring $\rho_t$ requires a redundant reduction (a second all-reduce in a different dtype or order), costing $\approx 5\text{–}15\%$ of step time at 1k+ ranks.

**Optimization estimand.** Adam's update $u_t = \hat m_t / (\sqrt{\hat v_t} + \epsilon)$. Molybog et al. (2023) locate instability where $\sqrt{\hat v_t} \sim \epsilon$, so define the **$\epsilon$-domination fraction** $\phi_t = |\{i : \sqrt{\hat v_{t,i}} < \epsilon\}|/d$. Also track sharpness $\lambda_t = \lambda_{\max}(\nabla^2 L)$ against the edge-of-stability threshold $2/\eta_t$, and max attention logit $A_t = \max_{h,i,j} q_i^\top k_j/\sqrt{d_h}$.

**Assumptions, and which are violated.**
- *Replayability* (rerunning from checkpoint $t-k$ on the same data reproduces $\ell_t$): **violated.** Non-deterministic reduction order, atomics, and cuBLAS/FlashAttention kernel selection make trajectories diverge; PaLM (2023) reports spikes that did **not** recur on replay from an earlier checkpoint with the same batches.
- *Single cause per spike:* **violated.** An SDC-induced outlier gradient and an already-marginal LR compose.
- *Stationary noise:* **violated** across curriculum/LR-schedule phases, so $\hat\sigma_t$ is not comparable across a run.
- *Detectable hardware faults:* **violated by construction** — SDC is silent (Hochschild et al., HotOS 2021; Dixit et al., 2021).

## 3. State of the Art

**Established (reproduced, ablated).**
- *Small-scale proxies* — Wortsman et al. (ICLR 2024) reproduce attention-logit growth and output-logit divergence at 20M–4.8B parameters by pushing LR, and show **qk-layernorm** and **z-loss** widen the stable LR range by roughly an order of magnitude. This is a genuine ablation, not a log anecdote.
- *Attention entropy collapse* — Zhai et al. (ICML 2023) tie instability to entropy collapse of attention maps and fix it with $\sigma$Reparam; ablated across ViT/ASR/MT.
- *QK-LayerNorm at scale* — Dehghani et al. (ViT-22B, ICML 2023) adopt it to remove divergences at 22B.
- *Initialization/scale conditions* — Takase et al. (2023) give sufficient conditions on sub-layer gradient norms ("Spike No More") with matching small-scale experiments.

**Claimed but unablated (log anecdote or single run).**
- *PaLM* (Chowdhery et al., JMLR 2023): ~20 spikes at 540B; mitigation = rewind ~100 steps, skip 200–500 batches. The paper explicitly reports that the same batches at a *different* checkpoint did not spike — evidence against pure data causation, but $n=1$ per event and no control arm.
- *Adam instability theory* (Molybog et al., 2023): a mechanism, evidenced by correlation on a small number of large runs; no intervention ablation at scale.
- *DeepSeek-V3* (2024): "no irrecoverable loss spikes or rollbacks" across 14.8T tokens. A benchmark-style claim with no counterfactual — it is not known which of the dozen concurrent design choices bought it.
- *Data-side attribution*: OLMo 2 (2025) reports spikes associated with repeated-n-gram documents and improved stability after filtering — confounded with several simultaneous changes.
- *Detection tooling*: MegaScale (NSDI 2024) and Llama 3 (Grattafiori et al., 2024) describe fault-detection and stragglers, not spike attribution.

There is **no published method that outputs a cause label with measured precision/recall.** SOTA practice is a heuristic ladder: rewind, skip batches, lower LR, and if it recurs, drain the node.

## 4. What Is Known

- **Spikes are rare and expensive.** PaLM 540B: ~20 spike events over the run. Each rewind of ~100 steps at 6144 TPU v4 chips discards on the order of $10^4$ chip-hours.
- **Hardware failures dominate interruptions, and are separately measured.** Llama 3 405B, 54-day snapshot on 16,384 H100s: 466 job interruptions, 419 unexpected, ~78% attributed to hardware (GPU issues ~58.7%), effective training time ~90% (Grattafiori et al., 2024). These are *detected* faults; spike attribution concerns the undetected tail.
- **SDC is real at fleet scale.** Google reports "mercurial cores" at a rate of about one per several thousand machines (Hochschild et al., HotOS 2021); Meta reports SDC at roughly one per thousand machines over months of fleet testing (Dixit et al., 2021).
- **Optimizer $\epsilon$ matters.** Raising Adam $\epsilon$ from $10^{-8}$ toward $10^{-6}$–$10^{-5}$ is a standard large-run stabilizer, consistent with the $\sqrt{\hat v}\sim\epsilon$ mechanism (Molybog et al., 2023).
- **Edge-of-stability is the default regime.** Full-batch GD drives $\lambda_{\max}$ to $\approx 2/\eta$ and hovers (Cohen et al., ICLR 2021); curvature-driven catapults produce spikes with no systems cause at all (Gilmer et al., 2021), measured at ResNet/ViT scale ($10^7$–$10^8$ params).
- **Spike-aware optimizers help on benchmarks.** SPAM (Huang et al., ICLR 2025) reports improved perplexity via momentum reset and spike-aware clipping at 60M–1B LLaMA-style models — benchmark numbers, not a causal attribution.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No agreed estimand for "the cause of spike $s$" that is identifiable without bitwise-deterministic replay. Non-determinism means the interventional counterfactual "same batch, same weights, healthy hardware" cannot be sampled; the standard rewind-and-replay confounds the intervention with the trajectory change. There is also no public labeled corpus of spikes with ground-truth causes — no benchmark exists to score any detector against.
- **Empirically open.** Whether qk-layernorm + z-loss + raised $\epsilon$ removes *systems-triggered* spikes as well as optimization-triggered ones. Runnable at 7B–70B with fault injection; nobody has published it. Also open: the base rate — what fraction of spikes in a 10k-GPU run are systems-caused? No published run reports a denominator.
- **Theoretically open.** No theorem bounding $\Pr[\Delta\ell > \delta]$ for Adam under heavy-tailed gradient noise plus a per-step bit-flip probability $p$. Molybog et al.'s account is a mechanism, not a bound. Whether an $\ell$-only observer can distinguish a single-parameter corruption from a curvature catapult is not known even in a two-layer linear model.

## 6. Why It Is Hard

**Non-identifiability under non-deterministic replay.** The one experiment that would settle a given spike — replay it — destroys the thing being measured. Reduction order in NCCL all-reduce, tensor-core accumulation, and kernel autotuning make two runs from the same checkpoint on the same data diverge; in a chaotic optimization trajectory, an initial discrepancy of $\sim 10^{-6}$ relative grows over tens of steps to the same order as the spike itself. So a non-reproducing spike is consistent with *both* "it was a transient hardware fault" and "it was a knife-edge optimization event". PaLM's own observation — same batches, no spike — is therefore evidence, but not decisive evidence, against data causation.

**Compounding obstructions:** (i) *absent ground truth* — SDC is silent by definition, so there is no label even in principle unless you inject the fault yourself; (ii) *compute cost* — the events are rare (~20 per full run), so collecting $n=30$ labeled spikes naturally means $\sim$10 frontier runs; (iii) *confounded observable* — $\ell_t$ mixes batch difficulty with parameter state, and the cheap control $L(\theta_{t-1};B_t)$ is not logged by default frameworks; (iv) *evaluation mismatch* — "did the fix work?" is scored by absence of recurrence, which for a base rate of one spike per $10^4$ steps has almost no power.

## 7. Current Research (as of 2026)

- **Small-scale proxy program** (Google DeepMind, following Wortsman et al.): reproduce instabilities at 100M–1B by LR-scaling, then transfer fixes upward. Strongest established line.
- **Architectural prophylaxis** at frontier labs: qk-layernorm, z-loss, embedding scaling, $\mu$P-style LR transfer — deployed widely; the DeepSeek-V3 and OLMo 2 reports read as a shift toward "never spike" rather than "attribute spikes". *(frontier — verify: which specific choice carries the effect is not disclosed by any lab.)*
- **Bitwise-deterministic training stacks** — deterministic reduction ordering and fixed kernel selection, at a reported single-digit-percent throughput cost. If it lands as default, it dissolves the identifiability obstruction. *(frontier — verify: no peer-reviewed measurement at 1k+ GPUs.)*
- **Fault-injection / SDC detection in ML training** (Meta, Google infrastructure groups; MegaScale-lineage systems work): redundant-compute checks and per-rank gradient-norm outlier fencing.
- **Spike-aware optimizers** (SPAM and successors): treat spikes as nuisance to suppress rather than signal to attribute.

## 8. Concrete Next Experiment

**Goal:** measure whether an online classifier can separate injected systems faults from genuine optimization spikes, and establish the first labeled spike corpus.

**Scale.** A 7B decoder, 256 A100/H100 GPUs, 300B tokens (~30k steps, roughly 20k GPU-hours). Large enough to exhibit natural spikes at a raised LR; small enough to run 3 seeds.

**Arms.**
1. **Injection arm** — at 60 pre-registered steps, flip one bit in the exponent field of one parameter shard on one rank, or drop one rank's contribution from one all-reduce. Ground truth is known by construction.
2. **Control arm (essential)** — identical seed, identical schedule, *no* injections, with LR set so the natural spike rate is comparable ($z_t>8$ at $\ge 20$ steps per run). Both arms log $\ell_t$, the batch-difficulty control $L(\theta_{t-1};B_t)$, $\phi_t$, $A_t$, per-rank gradient norms, and $\rho_t$ from a redundant fp32 reduction.
3. **Determinism arm** — the control arm rerun with deterministic kernels and fixed reduction order, to quantify how much replay-based attribution recovers.

**Deciding number.** AUROC of a classifier that must, within 5 steps of detection, label a spike systems vs. non-systems, using only signals available online. **Threshold: AUROC $\ge 0.90$ on held-out seeds.** Below $\approx 0.75$, the "attribute then intervene" framing is dead at this scale and the field should default to prophylaxis plus deterministic replay. Secondary number: the fraction of *natural* control-arm spikes that the classifier calls "systems" — a false-positive rate above 20% means every deployment would drain healthy nodes.

## 9. Key References

- **[Foundational]** Chowdhery, A. et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR, 2023. — arXiv:2204.02311
- **[Foundational]** Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., Talwalkar, A. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR, 2021. — arXiv:2103.00065
- **[SOTA]** Wortsman, M. et al. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[SOTA]** Molybog, I. et al. *A Theory on Adam Instability in Large-Scale Machine Learning.* Preprint, 2023. — arXiv:2304.09871
- **[SOTA]** Zhai, S. et al. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML, 2023. — arXiv:2303.06296
- **[SOTA]** Takase, S., Kiyono, S., Kobayashi, S., Suzuki, J. *Spike No More: Stabilizing the Pre-training of Large Language Models.* Preprint, 2023. — arXiv:2312.16903
- **[Systems]** Hochschild, P. H. et al. *Cores that don't count.* HotOS, 2021.
- **[Systems]** Dixit, H. D., Pendharkar, S., Beadon, M. et al. *Silent Data Corruptions at Scale.* Preprint, 2021. — arXiv:2102.11245
- **[Systems]** Jiang, Z. et al. *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs.* NSDI, 2024. — arXiv:2402.15627
- **[Empirical]** Grattafiori, A. et al. (Llama 3 team). *The Llama 3 Herd of Models.* Preprint, 2024. — arXiv:2407.21783
- **[Empirical]** Zhang, S. et al. *OPT: Open Pre-trained Transformer Language Models.* Preprint, 2022. — arXiv:2205.01068 (see the accompanying training logbook)
- **[Empirical]** OLMo Team. *2 OLMo 2 Furious.* Preprint, 2025. — arXiv:2501.00656
- **[Method]** Huang, T. et al. *SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training.* ICLR, 2025. — arXiv:2501.06842
- **[Related]** Gilmer, J. et al. *A Loss Curvature Perspective on Training Instability in Deep Learning.* Preprint, 2021. — arXiv:2110.04369
- **[Related]** Dehghani, M. et al. *Scaling Vision Transformers to 22 Billion Parameters.* ICML, 2023. — arXiv:2302.05442

## 10. Worked Example

A 7B run, 1024 GPUs, global batch 4M tokens, $\eta = 3\times10^{-4}$, Adam $\epsilon=10^{-8}$. Baseline $\ell_t \approx 2.10$ nats with $\hat\sigma_t = 0.004$. At step 41,220, $\ell = 2.45$: $\Delta\ell = 0.35$, so $z = 87.5$. Cost of the standard response (rewind 100 steps, skip 400 batches): $100 \times 1024$ GPUs $\times$ 12 s/step $\approx 340$ GPU-hours discarded, plus 1.6B tokens skipped.

Now try to attribute it.

- **Data hypothesis.** Re-score batch $B_{41220}$ under $\theta_{41219}$: $L = 2.13$. Only $+0.03$ over baseline, i.e. $z\approx 7$ on batch difficulty. The batch is mildly hard, nowhere near enough to explain $0.35$. Data is largely exonerated — but this control existed only because it was logged in advance; standard frameworks do not log it.
- **Optimization hypothesis.** $\phi_t$ (fraction of coordinates with $\sqrt{\hat v}<\epsilon$) is $0.031$ at the spike vs. a trailing median of $0.028$. Max attention logit $A_t = 41$ vs. median $38$. Both are elevated, neither is anomalous by the $z>8$ convention. Suggestive, not diagnostic.
- **Systems hypothesis.** Per-rank pre-reduce gradient norms: 1023 ranks in $[0.81, 0.95]$, rank 617 at $1.62$ — a $\approx 1.7\times$ outlier. That is consistent with a bit flip in an activation on that rank, and *also* consistent with rank 617 holding the shard of a genuinely hard microbatch.

**Where it dies.** Replay from the step-41,120 checkpoint on the identical data: no spike ($\ell_{41220} = 2.11$). By step 41,150 the replay's loss already differs from the original by $\sim 3\times10^{-3}$ — the same order as $\hat\sigma_t$ — purely from non-deterministic reduction order. So the replay trajectory is not the original trajectory by the time the spike would have occurred, and non-recurrence discriminates nothing: a transient SDC on rank 617 and a knife-edge curvature catapult both predict exactly this outcome. The team drains rank 617's node. Whether that was the right call is unknown, and will stay unknown, because the run continues spike-free either way — one observation, base rate one per $10^4$ steps, statistical power near zero.

This is the obstruction in full: three plausible causes, one non-reproducible event, no ground truth, and an intervention whose success is unfalsifiable. The experiment in §8 exists to manufacture the ground truth that this instance lacks.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*