---
id: 03-training-dynamics/loss-spike-mechanism
title: "Loss Spike Mechanism and Prevention"
topic: 03-training-dynamics
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Loss Spike Mechanism and Prevention

> **Topic:** Training Dynamics & Optimization · **ID:** `03-training-dynamics/loss-spike-mechanism` · **Status:** open

## 1. Problem Statement

During large-scale language model pre-training, the training loss sometimes jumps by 0.1–5 nats within a few steps, then either recovers over hundreds to thousands of steps or diverges permanently. The problem has three variants that are usually conflated.

- **Measurement.** Given a training run's telemetry, decide whether step $t$ is a spike, and attribute it to a cause: a data batch, an optimizer state pathology, an activation-scale blow-up, or numerical error. Solving this means a detector with a stated false-positive rate and an attribution that predicts the counterfactual — "remove this cause, the spike does not occur."
- **Method.** Produce an intervention $\mathcal{I}$ (architecture, initialization, optimizer, or clipping rule) such that a run at $N \gtrsim 10^{11}$ parameters completes with zero irrecoverable spikes, at no loss in final validation loss relative to an unmodified control at matched compute. Interventions that only lower the learning rate are excluded — they trade stability for loss.
- **Theory.** Prove, for a stated model class and optimizer, either (a) a sufficient condition on architecture/hyperparameters under which the loss is monotone-in-expectation outside a bounded set, or (b) that spikes are intrinsic to adaptive optimization at the edge of stability and can only be bounded, not eliminated.

Status **open** in all three: measurement is methodologically blocked, method is empirically open, theory is theoretically open.

## 2. Formal Setting

Parameters $\theta_t \in \mathbb{R}^d$; per-step loss $L_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t} \ell(x_i; \theta_t)$ on the batch actually consumed at step $t$ (this is what training logs record; it is *not* a fixed-distribution estimate, since $\mathcal{B}_t$ changes).

**Spike detector (as measured).** Let $\tilde{L}_t$ be the median of $\{L_s\}_{s=t-W}^{t-1}$ and $\hat{\sigma}_t$ the median absolute deviation over the same window, $W \approx 100$. Step $t$ is a spike if
$$ z_t \;=\; \frac{L_t - \tilde{L}_t}{1.4826\,\hat{\sigma}_t} \;>\; \kappa, \qquad \kappa \in [6, 10]. $$
Recovery time $\tau = \min\{s > t : L_{t+s} \le \tilde{L}_t + \hat{\sigma}_t\}$; the spike is *irrecoverable* if $\tau$ exceeds the remaining budget. Note $\hat\sigma_t$ shrinks as training proceeds, so fixed $\kappa$ makes late spikes easier to trip — a known artifact.

**Candidate mechanistic quantities, each with its measurement.**

- Sharpness $\lambda_t = \lambda_{\max}(\nabla^2 L(\theta_t))$, estimated by 20–50 Lanczos iterations with Hessian-vector products on a fixed held-out batch of size $\ge 2^{19}$ tokens. Cost: $\approx 50\times$ one forward-backward.
- Preconditioned sharpness $\lambda_t^{P} = \lambda_{\max}(P_t^{-1}\nabla^2 L)$ with $P_t = \mathrm{diag}(\sqrt{v_t} + \varepsilon)$, Adam's second moment. Edge-of-stability arguments apply to $\lambda^P$, not $\lambda$.
- Max attention logit $A_t = \max_{\text{layer},\,\text{head},\,i,j} q_i^\top k_j / \sqrt{d_h}$, logged per step at negligible cost.
- Update-to-parameter RMS ratio $\rho_t^{(l)} = \mathrm{RMS}(\Delta\theta_t^{(l)})/\mathrm{RMS}(\theta_t^{(l)})$ per layer $l$.
- Second-moment staleness $s_t = \|g_t\|/\|\sqrt{v_t}\|$: Adam's implicit trust that the gradient distribution has not shifted.

**Assumptions and their violations.** (i) Edge-of-stability analyses assume full-batch deterministic descent — violated; batch gradients at $B = 4\text{M}$ tokens still have relative noise $\gtrsim 0.1$. (ii) Adam convergence results assume bounded gradients and $\beta_2$-stationarity — violated exactly at the spike, which is when the gradient distribution shifts. (iii) NTK/linearization arguments assume $\|\Delta\theta\|$ small — violated; a spike *is* the large-displacement event. (iv) Spikes are assumed reproducible given identical data order — falsified at 540B (see §4).

## 3. State of the Art

**Established (ablated, and reproduced by at least one independent group):**

- **qk-layernorm** — LayerNorm on queries and keys before the dot product. Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* (ICLR 2024) show attention-logit growth is the cause of one instability class and that qk-layernorm removes it, extending the stable learning-rate range by roughly an order of magnitude at 1.2B parameters. Now standard (Gemma 2, Chameleon, dozens of others).
- **z-loss** — auxiliary $10^{-4}\log^2 Z$ on the softmax normalizer, from PaLM (Chowdhery et al., JMLR 2023); ablated in Wortsman et al. as fixing output-logit divergence.
- **Update clipping in the optimizer** (StableAdamW / Adafactor-style clipping of $\rho_t$ rather than of the gradient) reduces spike rate without the loss penalty of a lower learning rate.
- **Scaled/small embedding-output initialization and residual-branch scaling** — Takase et al., *Spike No More*, give a sufficient condition on the embedding-layer gradient norm and show spikes vanish under it at up to 13B.

**Claimed but unablated, or benchmark-only:**

- DeepSeek-V3's report of *no irrecoverable loss spikes and no rollbacks* over 14.8T tokens is a run outcome, not an ablation: the control arm (same recipe minus the stabilizers) was never trained.
- GLM-130B's embedding gradient shrink ($\alpha = 0.1$) is reported as decisive at 130B, but only with a single seed and no matched control.
- SPAM (spike-aware Adam with momentum reset, ICLR 2025), σReparam (Zhai et al., ICML 2023), and nGPT-style normalization all report reduced instability; none has a published $\ge$100B control-arm comparison.
- Claims that spikes are caused by "bad data" persist widely and are contradicted by the PaLM batch-restart result (§4).

## 4. What Is Known

- **PaLM 540B** (Chowdhery et al., JMLR 2023): about 20 spikes in one run. Restarting from a checkpoint ~100 steps earlier and skipping 200–500 batches removed each spike; restarting from the same checkpoint *with the same data* did not reproduce it. Conclusion: spikes require the coincidence of a particular batch **and** a particular optimizer state, not bad data alone.
- **OPT-175B** (Zhang et al., 2022, plus the public logbook): 35+ manual restarts, learning-rate reductions, and loss-scale adjustments over the run; instabilities correlated with fp16 dynamic-loss-scale collapse and hardware failures.
- **Adam-$\varepsilon$ mechanism** (Molybog et al., *A theory on Adam instability in large-scale machine learning*, 2023): at 175B scale, when a layer's gradient becomes small relative to $\varepsilon$ the update collapses toward zero, gradient estimates decorrelate in time, and a subsequent large gradient produces an outsized step. Predicts the observed dependence on model size, since per-parameter gradient magnitude shrinks with width.
- **Attention entropy collapse** (Zhai et al., ICML 2023): entropy of the attention distribution falls sharply immediately before divergence across ViT, machine translation, and speech; σReparam prevents it.
- **Edge of stability** (Cohen et al., ICLR 2021): full-batch GD at step size $\eta$ drives $\lambda_{\max}$ to hover just above $2/\eta$ with non-monotone loss — measured at $\sim 10^5$–$10^7$ parameters on vision tasks. Whether this transfers to Adam at $10^{11}$ parameters is unestablished.
- **Outlier features** (He, Noci et al., NeurIPS 2024): kurtosis of hidden activations grows through training and tracks instability; normalization and optimizer choices measurably reduce it at $\le$1.2B.
- Spike frequency rises with model size, with learning rate, and with $\beta_2 \to 1$; it falls with warmup length. These are consistent across PaLM, OPT, GLM-130B, and Wortsman et al., but no scaling law with a fitted exponent exists.

## 5. What Is Not Known

- **Theoretically open.** No theorem gives a sufficient condition for spike-freeness of Adam on a transformer. No proof that any specific mechanism (logit growth, $\varepsilon$-collapse, entropy collapse) is necessary rather than one of several sufficient routes. No proof either way that spikes are intrinsic to adaptive methods operating near their stability boundary.
- **Empirically open.** Whether the mechanism identified at $\le$1.2B in small-scale proxies is the *same* mechanism at $\ge$100B. Whether stabilizers cost final loss: no published matched-compute A/B at $\ge$70B with and without qk-layernorm + z-loss. Whether spikes are net-harmful — one hypothesis is that a recovered spike acts as noise injection that improves the final solution; nobody has run the control.
- **Methodologically blocked.** Attribution. "This spike was caused by X" has no accepted operational definition, because the counterfactual re-run is stochastic (nondeterministic kernels, different device assignment) and, per PaLM, does not reproduce. Without reproducible counterfactuals there is no ground truth for any attribution method.

## 6. Why It Is Hard

Three specific obstructions.

1. **Non-identifiability of the trigger.** The candidate signals — $A_t$, $\rho_t$, kurtosis, $\lambda^P_t$ — rise together in the ~50 steps before a spike. They are functions of the same underlying activation-scale growth, so single-run telemetry cannot separate cause from co-symptom. Distinguishing them needs interventions, and each intervention changes the whole trajectory.
2. **The counterfactual is not reproducible.** The event that would define ground truth (same state, same batch, spike again) fails at scale. This is the reason attribution is blocked, not merely unfunded.
3. **Compute cost of the decisive scale.** Spikes are rare below 10B and common above 100B. A single ablation arm at 100B $\times$ 2T tokens is $\sim 1.2\times10^{24}$ FLOPs — roughly $10^4$ H100-days. Three arms $\times$ three seeds is a frontier-lab training budget spent on a negative result, which is why the ablations in §3 stop at 13B.

A fourth, softer obstruction: the reported metric is training loss on the consumed batch, which mixes model state with batch difficulty. Runs that log only this cannot tell a 0.3-nat spike from a hard shard.

## 7. Current Research (as of 2026)

- **Small-scale proxy design** — extending the Wortsman-style protocol (predict $\ge$100B instabilities from $\le$1B runs by sweeping learning rate to the divergence boundary) to more mechanism classes. Google DeepMind, ETH Zürich (Hofmann group, outlier features).
- **Architectural elimination** — normalization-everywhere designs (nGPT, σReparam, QK-norm variants) that aim to make the instability manifold unreachable. NVIDIA, EPFL. *(frontier — verify)*
- **Optimizer-side fixes** — momentum reset and spike-aware clipping (SPAM), Muon and other spectral-norm-constrained updates whose per-layer update norm is bounded by construction, which if it holds removes the $\rho_t$ blow-up route. Whether Muon-family runs are empirically spike-free at $\ge$100B is claimed in several 2025–2026 reports and not independently ablated *(frontier — verify)*.
- **Low-precision interaction** — FP8 and MX-format training reintroduce spikes that BF16 recipes had removed; DeepSeek and NVIDIA report per-tensor scaling fixes. *(frontier — verify)*
- **Reframing spikes as beneficial** — a minority line arguing spike-and-recover improves generalization. Little evidence either way.

## 8. Concrete Next Experiment

**Question:** is attention-logit growth the *necessary* precursor of spikes at scale, or one of several routes?

- **Scale.** 7B parameters, 300B tokens (Chinchilla-ish, $\approx 1.3\times10^{22}$ FLOPs/arm), learning rate deliberately set at $1.5\times$ the largest stable value found in a 1B sweep, so spikes occur at usable rates ($\ge$10 per run).
- **Arms.** (A) control: no qk-layernorm, no z-loss, standard AdamW, $\varepsilon = 10^{-8}$; (B) qk-layernorm only; (C) $\varepsilon = 10^{-15}$ only; (D) update-RMS clipping only. Three seeds each; identical data order across seeds within an arm. 12 runs, $\approx 1.6\times10^{23}$ FLOPs total, feasible on ~2k H100s in under two weeks.
- **Instrumentation.** Log $A_t$, $\rho_t^{(l)}$, activation kurtosis, $s_t$ every step; $\lambda^P_t$ every 500 steps.
- **Deciding number.** In arm A, the fraction of spikes ($z_t > 8$) preceded within 50 steps by $A_t$ exceeding $10^4$. If that fraction is $\ge 0.9$ **and** arm B's spike count is $\le 0.1\times$ arm A's, logit growth is the dominant route at 7B and the mechanism question narrows to whether it survives to 100B. If the fraction is $\le 0.5$, or arm B still spikes at $\ge 0.5\times$ arm A's rate, the single-mechanism hypothesis is falsified and attribution must become multi-cause. Secondary number: final validation loss gap between A (spike-free segments) and B at matched tokens; a gap $>0.01$ nats means the stabilizer is not free.

## 9. Key References

- **[Foundational]** Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR 24, 2023. — arXiv:2204.02311
- **[Foundational]** Cohen, Kaur, Li, Kolter, Talwalkar. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR 2021. — arXiv:2103.00065
- **[SOTA]** Wortsman et al. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR 2024. — arXiv:2309.14322
- **[SOTA]** Takase, Kiyono, Kobayashi, Suzuki. *Spike No More: Stabilizing the Pre-training of Large Language Models.* 2023/2025. — arXiv:2312.16903
- **[Mechanism]** Molybog et al. *A Theory on Adam Instability in Large-Scale Machine Learning.* 2023. — arXiv:2304.09871
- **[Mechanism]** Zhai et al. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML 2023. — arXiv:2303.06296
- **[Mechanism]** He, Noci, Paliotta, Schlag, Hofmann. *Understanding and Minimising Outlier Features in Transformer Training.* NeurIPS 2024.
- **[Empirical]** Zhang et al. *OPT: Open Pre-trained Transformer Language Models.* 2022 (with the public training logbook). — arXiv:2205.01068
- **[Empirical]** Zeng et al. *GLM-130B: An Open Bilingual Pre-trained Model.* ICLR 2023. — arXiv:2210.02414
- **[Empirical]** DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. — arXiv:2412.19437
- **[Method]** Huang et al. *SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training.* ICLR 2025.

## 10. Worked Example

A 1.4B decoder, $d = 2048$, 24 layers, $d_h = 128$, AdamW with $\eta = 3\times10^{-4}$, $\beta_2 = 0.95$, $\varepsilon = 10^{-8}$, batch 2M tokens. At step 41,300 the logged loss goes $2.412 \to 2.407 \to 3.98 \to 3.11 \to 2.65$, returning to 2.42 after 900 steps. With $W = 100$, $\hat\sigma \approx 0.006$, so $z = (3.98 - 2.41)/0.006 \approx 260$ — an unambiguous spike by any $\kappa$.

Telemetry over the preceding 60 steps:

```
step      L      A_t (max logit)   kurtosis(h)   rho_max   s_t
41240   2.414        890               41         1.1e-3   0.31
41270   2.413       3,400              78         1.9e-3   0.29
41295   2.412      1.4e4              310         6.2e-3   0.24
41302   3.980      3.1e4              540         4.1e-2   0.08
```

Every candidate signal moves together: logits $\times 35$, kurtosis $\times 13$, update ratio $\times 37$, and $s_t$ falls as $\sqrt{v}$ lags the gradient surge. Any of the four would have "predicted" the spike with 30 steps of lead time.

Now the obstruction. Re-run from the step-41,200 checkpoint with the identical data shard: on 3 of 5 replays the spike does not occur — bit-level nondeterminism in the reduction order changes $\theta$ by $\sim10^{-7}$ relative, which is enough near the stability boundary. So the observed correlations cannot be converted into a cause. The only test that separates them is intervention, and turning on qk-layernorm changes $A_t$ *by construction*, so a subsequent spike-free run is consistent both with "logit growth was the cause" and with "the intervention also shrank $\rho$, and $\rho$ was the cause." That is the non-identifiability in §6, made concrete: four correlated signals, no reproducible counterfactual, and every available intervention moves more than one signal at once.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*