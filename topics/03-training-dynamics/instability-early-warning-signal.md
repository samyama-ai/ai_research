---
id: 03-training-dynamics/instability-early-warning-signal
title: "Training Instability Early Warning Signal"
topic: 03-training-dynamics
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Training Instability Early Warning Signal

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/instability-early-warning-signal` · **Status:** empirically-open

## 1. Problem Statement

Large language model pretraining runs fail in discrete events: loss spikes, divergences, and slow "silent" degradations that never recover. Today these are handled reactively — detect the spike after it lands, roll back to a checkpoint, skip data, lower the learning rate, restart. The problem is whether they can be handled **proactively**.

**Input.** The telemetry stream available at optimizer step $t$ without extra forward passes: per-step loss, gradient norms per parameter group, Adam moment statistics, activation and logit statistics, weight norms, and the identity of the batch about to be consumed.

**Output.** A binary alarm $a_t \in \{0,1\}$, raised at step $t$, claiming that a spike will occur within the next $h$ steps.

**Decision predicate.** Does there exist a detector achieving recall $\ge 0.9$ at a false-alarm rate low enough that acting on every alarm costs less than the spikes it prevents, with lead time $h$ large enough for the intervention to take effect (order $10^2$ steps, not $10^0$)?

Three variants, different difficulty:

- **Measurement.** Define "instability" so that two labs labelling the same loss curve agree. Currently unresolved — see §5.
- **Method.** Build a detector meeting the predicate above. Empirically open.
- **Theory.** Prove that some observable is a *necessary* precursor — that no spike occurs without it crossing a threshold. Theoretically open; only sufficient-condition results exist.

Solving it means: a run of $10^{5}$ steps at $\ge 10^{10}$ parameters completes with zero rollbacks, and an ablation shows the alarms fired before, not after, the spikes they claim to predict.

## 2. Formal Setting

Parameters $\theta_t \in \mathbb{R}^d$, minibatch $B_t$, per-step training loss $\ell_t = \mathcal{L}(\theta_t; B_t)$.

**Spike label (measured).** Fit a robust local baseline — median over a trailing window $W$ (typically $W = 100$ steps) — and its scale:
$$\mu_t = \mathrm{med}(\ell_{t-W:t-1}), \qquad s_t = \mathrm{MAD}(\ell_{t-W:t-1}), \qquad z_t = \frac{\ell_t - \mu_t}{1.4826\, s_t}.$$
A spike starts at $\tau$ if $z_\tau > \kappa$ (commonly $\kappa \in [5,10]$) and persists for $\ge m$ steps. It is **recoverable** if $\ell$ returns within $\epsilon$ of the pre-spike trend inside $R$ steps, **divergent** otherwise. $\kappa, m, R$ are conventions, not derived quantities — this is the measurement problem.

**Detector and lead time.** A causal map $a_t = f(\mathcal{H}_t)$ on history $\mathcal{H}_t = \{\ell_{\le t}, g_{\le t}, \dots\}$. For spike onset $\tau$, lead time $\lambda = \tau - \min\{t : a_t = 1,\ t \le \tau\}$. Detection counts only if $\lambda \ge h$.

**Candidate precursors, as measured.**

- Attention logit magnitude: $A^{(\ell)}_t = \max_{i,j} |q_i^\top k_j / \sqrt{d_k}|$ in layer $\ell$, logged from a hook.
- Output logit drift: $Z_t = \log \sum_v \exp(z_v)$, the z-loss term.
- Gradient-norm ratio: $\rho_t = \|g_t\| / \mathrm{med}(\|g_{t-W:t-1}\|)$.
- Adam update-to-parameter ratio: $u_t = \|\eta\, \hat m_t / (\sqrt{\hat v_t} + \varepsilon)\| / \|\theta_t\|$.
- Sharpness: $\lambda_{\max}(\nabla^2 \mathcal{L})$, estimated by power iteration on Hessian-vector products — 10–20 extra backward passes, so it is sampled every $10^3$ steps, not per step.
- Critical-slowing-down statistics from the dynamical-systems EWS literature: lag-1 autocorrelation and variance of the detrended residual $\ell_t - \mu_t$ over $W$.

**Assumptions, and which fail.**

1. *Stationarity of the baseline within $W$.* Violated during learning-rate warmup, batch-size ramps, and data-mixture switches — exactly when spikes cluster.
2. *Spikes are events in $\theta$-dynamics, not batch artifacts.* Violated: PaLM found spikes reproduce only with a specific (data, parameter-state) pair, so the trigger is partly exogenous.
3. *One-dimensional order parameter.* Assumed by every scalar detector; unverified — instabilities in attention entropy, output logits, and optimizer moments may be distinct failure modes with distinct precursors.
4. *Small-scale proxies transfer.* Partially supported (§4), but the mapping is via learning-rate sensitivity curves, not identity.

## 3. State of the Art

**Established (reproduced, ablated).**

- **qk-LayerNorm** (Dehghani et al., ViT-22B, ICML 2023) removes attention-logit-growth divergence; independently confirmed on decoder LMs by Wortsman et al. (ICLR 2024). Mechanism identified, fix ablated.
- **z-loss** on output logits (PaLM, 2022; Chowdhery et al.) controls logit-norm drift; retained in later runs.
- **Small-scale proxies** (Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities*, ICLR 2024, arXiv:2309.14322): instabilities that appear at large scale can be induced in models of $10^7$–$10^9$ parameters by raising the learning rate, and the LR at which loss diverges shrinks predictably with scale. This is the strongest existing bridge between cheap experiments and expensive ones.
- **$\sigma$Reparam / attention entropy collapse** (Zhai et al., ICML 2023, arXiv:2303.06296): entropy of attention distributions collapses *before* the loss degrades — a genuine precursor with a mechanism, ablated on ViT and LM training.

**Claimed but unablated as early warning.**

- Gradient-norm spikes as precursors. Widely used in practice (skip-batch heuristics, spike-aware optimizers such as momentum-reset variants of Adam). Reported precision/recall/lead-time numbers on production-scale runs are essentially absent; the claim is that the fix works, not that the signal predicts.
- **Adam instability theory** (Molybog et al., *A Theory on Adam Instability in Large-Scale Machine Learning*, 2023, arXiv:2304.09871): argues spikes follow from $\hat v$ becoming stale relative to $\hat m$ for rarely-updated parameters. Explanatory, derived from OPT-175B logs; not converted into a validated detector.
- **Spike No More** (Takase et al., 2023/2024, arXiv:2312.16903): gives sufficient conditions on embedding-layer gradient norms for spike-free training. A design rule, not a runtime alarm.

**Benchmark-number-only.** Public loss curves with annotated spikes (OPT-175B logbook, BLOOM, OLMo) are the de facto evaluation set. There is no shared benchmark with a fixed spike labelling, a fixed telemetry schema, and a held-out split — so cross-paper detector comparisons do not exist.

## 4. What Is Known

- **PaLM 540B** (Chowdhery et al., JMLR 2023): about 20 loss spikes in the run. Restarting from a checkpoint ~100 steps before onset and skipping 200–500 batches avoided the spike; replaying the same batches from a *different* checkpoint did not reproduce it. Interaction, not data alone.
- **OPT-175B** (Zhang et al., 2022, arXiv:2205.01068): dozens of manual interventions — LR reductions, restarts, optimizer-state resets — logged over a 175B-parameter run. Evidence that no automated precursor was available in 2022.
- **Edge of stability** (Cohen et al., ICLR 2021, arXiv:2103.00065): full-batch GD drives $\lambda_{\max}$ to $\approx 2/\eta$ and then hovers, with non-monotone loss. Measured at CIFAR-10 scale ($10^5$–$10^7$ params). Adaptive-optimizer analogue at $2/\eta$ in preconditioned norm (Cohen et al., 2022, arXiv:2207.14484).
- **Curvature perspective** (Gilmer et al., ICLR 2022, arXiv:2110.04369): warmup and clipping work by keeping $\eta \lambda_{\max}$ below the stability threshold; measured across ResNet/Transformer scales up to $\sim 10^8$ params.
- **LR sensitivity scaling** (Wortsman et al. 2024): across 20M–4.8B parameters, the divergence LR falls roughly as a power law in width; qk-LayerNorm widened the stable LR band by roughly an order of magnitude at the largest proxy scale tested.
- **Outlier features** (He et al., NeurIPS 2024): activation kurtosis grows with depth and predicts quantization difficulty and instability-prone configurations; measured at $\le 1.2$B parameters.

## 5. What Is Not Known

- **Methodologically blocked.** No agreed spike definition. $(\kappa, m, R, W)$ are chosen per paper; changing $\kappa$ from 5 to 10 changes the event count on the same curve by an order of magnitude, which changes every precision/recall figure. Until the label is fixed, detector comparison is meaningless.
- **Empirically open.** Whether any known observable achieves recall $\ge 0.9$ with lead time $\ge 100$ steps and false-alarm rate $\le 10^{-5}$/step at $\ge 10^{10}$ parameters. Every ingredient exists; the experiment costs a spike-rich frontier-scale run with full telemetry, which no lab has published.
- **Empirically open.** Whether small-scale-proxy spikes are the *same* phenomenon as frontier-scale spikes, or only share a symptom. Wortsman et al. show LR-induced proxies; nobody has matched proxy precursors to production precursors event-by-event.
- **Theoretically open.** No necessity result. There is no theorem of the form "if a spike occurs at $\tau$, then some named statistic exceeds threshold on $[\tau - h, \tau)$." All results are sufficient conditions for stability.
- **Theoretically open.** Whether loss spikes are bifurcations in a slow-fast system (which would license critical-slowing-down EWS: rising variance and lag-1 autocorrelation) or noise-driven barrier crossings (which would not).

## 6. Why It Is Hard

**The base rate.** Spikes are rare per step. At 20 events in $\sim 2.5 \times 10^5$ steps the prior is $\approx 8 \times 10^{-5}$. Precision is dominated by the false-alarm rate, so a detector must be calibrated two orders of magnitude tighter than typical ML classifiers to be useful (§10).

**Confounded measurement.** Every candidate precursor is also a *consequence* of ordinary schedule events. Gradient norm rises during warmup; attention logits grow monotonically through training; $u_t$ jumps at every LR change. Separating precursor from schedule requires a baseline the schedule itself violates (assumption 1, §2).

**Absent ground truth.** Labels come from a single realization. Whether a spike would have occurred absent intervention is unobservable once someone intervenes — and production logs are full of interventions. Counterfactual labels require deliberate non-intervention at $10^{10}$-parameter scale.

**Non-identifiability.** Detector and controller are entangled. A run using qk-LayerNorm plus z-loss has few spikes, so it cannot validate a detector; a run without them spikes often but is not the system anyone deploys.

**Compute.** One spike-rich, telemetry-complete, intervention-free run at $\ge 10$B parameters is a multi-hundred-thousand-GPU-hour experiment whose deliverable is a *negative* engineering result.

## 7. Current Research (as of 2026)

- **Architectural prophylaxis over detection** — qk-LayerNorm, QK-norm variants, z-loss, sandwich/normalization placement, careful residual scaling. Industrial default at Google DeepMind, Meta, and the open-weights labs. Effectively a decision that prevention beats prediction.
- **Optimizer-side mitigation** — momentum reset and spike-aware clipping in Adam variants; Muon/Shampoo-family second-order preconditioners reported to have different and generally milder spike profiles *(frontier — verify: spike-rate comparisons between Muon-family and Adam at $\ge 10$B scale are mostly informal reports, not ablations)*.
- **$\mu$P and scaling-law transfer** (Yang & Hu, *Tensor Programs V*, 2022, arXiv:2203.03466) used to move stable-LR estimates from proxy to target — an indirect early-warning system: predict the unstable region before the run rather than during it.
- **Dynamical-systems EWS imported into ML** — critical-slowing-down statistics (Scheffer et al., *Nature* 2009) applied to loss residuals. Exploratory; no published production validation.
- **Telemetry infrastructure** — open logging of per-step norms and activation statistics in OLMo-family releases (AI2). The most likely source of a public benchmark.

## 8. Concrete Next Experiment

**Scale.** Twelve runs at 1.4B parameters, 30B tokens each (Chinchilla-ish, roughly $2\times10^4$ H100-hours total). Deliberately spike-rich: set peak LR at $0.9\times$ the empirically measured divergence LR, no qk-LayerNorm, no z-loss. Vary only seed (×4) and data order (×3). Log per step: $\ell_t$, $\rho_t$, $u_t$, per-layer $A^{(\ell)}_t$, $Z_t$, attention entropy, activation kurtosis. Sample $\lambda_{\max}$ every 500 steps. **No interventions** — let every spike run to completion or divergence.

**Labels.** Freeze $(\kappa, m, R, W) = (6, 3, 2000, 100)$ before looking at any curve. Publish the label file.

**Control arm.** Two controls, both required. (a) *Post-hoc oracle*: a detector allowed to see $\ell_{t+1}$ — the reactive baseline everyone already has, lead time $0$. (b) *Schedule-only*: a detector using solely step index, LR, and batch size, no telemetry. Any candidate must beat (b) to show it uses dynamics rather than memorizing when in training spikes happen.

**Deciding number.** Recall at a fixed false-alarm budget of $10^{-4}$ per step, with lead time $\ge 100$ steps, on runs held out by seed. **Threshold: recall $\ge 0.5$.** Below that, no scalar precursor is worth deploying and the field should keep buying prevention instead of prediction. Above it, the same protocol scales to 10B for the frontier-transfer question.

Cost of the answer: about $10^4$–$10^5$ GPU-hours. Cost of one unpredicted divergence at frontier scale: comparable. That asymmetry is why the experiment is worth running and why nobody has.

## 9. Key References

- **[Foundational]** Jeremy Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR, 2021. — arXiv:2103.00065
- **[Foundational]** Justin Gilmer, Behrooz Ghorbani, Ankush Garg, Sneha Kudugunta, Behnam Neyshabur, David Cardoze, George Dahl, Zachary Nado, Orhan Firat. *A Loss Curvature Perspective on Training Instability in Deep Learning.* ICLR, 2022. — arXiv:2110.04369
- **[SOTA]** Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, Simon Kornblith. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[SOTA]** Shuangfei Zhai, Tatiana Likhomanenko, Eric Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, Josh Susskind. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML, 2023. — arXiv:2303.06296
- **[Empirical]** Aakanksha Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR, 2023. — arXiv:2204.02311
- **[Empirical]** Susan Zhang et al. *OPT: Open Pre-trained Transformer Language Models.* 2022 (includes the training logbook). — arXiv:2205.01068
- **[Theory]** Igor Molybog et al. *A Theory on Adam Instability in Large-Scale Machine Learning.* 2023. — arXiv:2304.09871
- **[Method]** Sho Takase, Shun Kiyono, Sosuke Kobayashi, Jun Suzuki. *Spike No More: Stabilizing the Pre-training of Large Language Models.* 2023. — arXiv:2312.16903
- **[Method]** Mostafa Dehghani et al. *Scaling Vision Transformers to 22 Billion Parameters.* ICML, 2023. — arXiv:2302.05442
- **[Method]** Greg Yang, Edward J. Hu, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* 2022. — arXiv:2203.03466
- **[Adjacent]** Marten Scheffer et al. *Early-warning signals for critical transitions.* Nature 461, 2009.
- **[Empirical]** Bobby He, James Martens, et al. *Understanding and Minimising Outlier Features in Transformer Training.* NeurIPS, 2024.

## 10. Worked Example

Take PaLM 540B as the target regime: $\approx 20$ spikes over a run of order $2.5 \times 10^5$ steps. Per-step prior:
$$\pi = \frac{20}{2.5\times10^5} = 8 \times 10^{-5}.$$

Suppose a gradient-norm detector — alarm when $\rho_t > 3$ — achieves recall $r = 0.9$ and a per-step false-alarm rate $\phi = 10^{-3}$ (one false alarm per 1,000 steps; optimistic for a threshold rule on a non-stationary series). Precision:
$$P = \frac{r\pi}{r\pi + \phi(1-\pi)} = \frac{7.2\times10^{-5}}{7.2\times10^{-5} + 9.999\times10^{-4}} = 0.067.$$

**14 false alarms for every true one.** If the response is PaLM's own remedy — roll back ~100 steps and skip 300 batches — the detector triggers roughly 250 rollbacks per run to prevent 18 spikes. At ~100 wasted steps each, that is $2.5\times10^4$ steps of throughput, about 10% of the run, to avoid 20 events that cost perhaps 100 steps each to fix reactively. **The detector is a net loss by a factor of ten.**

Invert for break-even. Requiring $P \ge 0.9$ at $r = 0.9$:
$$\phi \le \frac{r\pi(1-P)}{P(1-\pi)} \approx \frac{7.2\times10^{-5} \times 0.1}{0.9} = 8\times10^{-6}.$$

So the false-alarm rate must be **125× lower** than the already-optimistic figure above — under one false alarm per 125,000 steps, roughly one per run.

That is the obstruction, made numeric. The literature reports precursors that *correlate* with spikes; correlation at $\phi \sim 10^{-3}$ is worthless here. The binding constraint is not sensitivity but specificity against a background of $10^5$ ordinary steps, and no published detector has ever been measured against that background — because measuring it requires the intervention-free, spike-rich, fully-instrumented run described in §8.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*