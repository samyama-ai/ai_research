---
id: 08-loss-and-heads/loss-spikes-logit-growth-causality
title: "Loss Spikes and Output-Layer Logit Growth Causality"
topic: 08-loss-and-heads
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Loss Spikes and Output-Layer Logit Growth Causality

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/loss-spikes-logit-growth-causality` · **Status:** empirically-open

## 1. Problem Statement

Large language model pre-training runs exhibit **loss spikes**: abrupt increases in training loss, often 0.5–5 nats within tens of steps, sometimes recovering and sometimes diverging. Independently, the same runs exhibit **logit growth**: the un-normalized output-head logits (and the pre-softmax attention logits) grow in magnitude over training, faster than the loss improves.

The problem is to establish the **causal direction and sufficiency** of the relationship between the two:

- **Measurement variant.** Define a spike detector and a logit-growth statistic that are not trivially coupled through the loss itself, and determine whether growth *leads* spikes in time at a lag that is stable across seeds.
- **Method variant.** Determine whether interventions that suppress logit growth (z-loss, QK-LayerNorm, logit soft-capping, $\sigma$Reparam) reduce spike rate *because* they suppress logit growth, rather than through their side effects on effective learning rate, gradient norm, or conditioning.
- **Theory variant.** Prove or disprove that growth of $\log Z$ (the log-partition function of the output head) beyond a threshold implies a curvature or optimizer-state condition under which an Adam step must increase the loss.

A solution to the method variant would be an intervention that changes logit growth *and nothing else measurable*, and moves spike rate. A solution to the theory variant would be a bound of the form: given $\log Z_t > \tau$ and Adam second-moment staleness $s$, the expected one-step loss change is positive.

## 2. Formal Setting

Let $f_\theta$ be a decoder-only transformer, $h_t \in \mathbb{R}^d$ the final-layer residual at position $t$, $W_U \in \mathbb{R}^{V \times d}$ the unembedding. Output logits are $z_t = W_U h_t$, and

$$\log Z_t = \log \sum_{v=1}^{V} \exp(z_{t,v}), \qquad \ell_t = \log Z_t - z_{t, y_t}.$$

**Measured quantities**, all per optimizer step $s$ over the microbatch actually used:

- **Logit magnitude:** $M_s = \mathbb{E}_t[\max_v |z_{t,v}|]$ and $L_s = \mathbb{E}_t[\log Z_t]$, both logged from the forward pass before the backward pass.
- **Attention logit growth:** $A_s = \max_{\text{layer},\text{head}} \max_{t,t'} |q_t^\top k_{t'} / \sqrt{d_h}|$, the quantity Wortsman et al. track.
- **Spike indicator:** with $\bar\ell_s$ the EMA of loss (half-life 100 steps) and $\hat\sigma_s$ its EMA absolute deviation, $S_s = \mathbb{1}[\ell_s - \bar\ell_{s-1} > \kappa \hat\sigma_{s-1}]$, typically $\kappa = 6$. Spikes are *events* (maximal runs of $S_s = 1$), not steps.
- **Lead time:** $\Delta = s_{\text{spike}} - \min\{s : L_{s'} > L_{s_{\text{spike}}} - \epsilon \ \forall s' \in [s, s_{\text{spike}}]\}$.
- **Optimizer state:** Adam ratio $r_s = \|m_s / (\sqrt{v_s} + \varepsilon)\|_\infty$ and per-parameter-block gradient RMS.

**Assumptions and their violations.**

1. *Spikes are detectable from the loss alone.* Violated: gradient-norm spikes and $A_s$ excursions occur with no loss deviation at all, and data-order effects (a rare document class) produce loss jumps with no logit anomaly.
2. *$L_s$ and $\ell_s$ are statistically separable.* Violated by construction — $\ell_t = \log Z_t - z_{t,y_t}$, so any spike detector on $\ell$ is partly a detector on $\log Z$. Only the *residual* $L_s$ after regressing out $\ell_s$ is a clean covariate.
3. *Interventions are surgical.* Violated for every known one: z-loss adds a gradient term to $W_U$; QK-norm changes the attention Jacobian and hence the effective learning rate of every downstream block.
4. *Spike statistics are stationary.* Violated: spike hazard rises with model scale and with proximity to the largest stable learning rate.

## 3. State of the Art

**Established.**

- PaLM 540B (Chowdhery et al., JMLR 2023) reported ~20 loss spikes, and the key control: restarting from a checkpoint ~100 steps prior and skipping 200–500 batches removed the spike, while replaying the *same* batches from a pre-spike checkpoint did **not** reproduce it. This rules out "bad data alone" and establishes that spikes depend on the optimizer/parameter state.
- Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* (ICLR 2024), is the closest thing to a controlled study. It reproduces both **attention logit growth** and **output logit divergence** at small scale by pushing the learning rate, shows QK-LayerNorm removes the former and z-loss the latter, and reports that the two fixes together widen the stable-LR range by roughly an order of magnitude across model sizes.
- ViT-22B (Dehghani et al., ICML 2023) adopted QK-LayerNorm after observing divergence with attention logits reaching very large magnitudes; GLM-130B (Zeng et al., ICLR 2023) attributed spikes to attention-logit blowup and used embedding-gradient shrink.

**Claimed but unablated.**

- That z-loss ($\lambda \log^2 Z$, $\lambda = 10^{-4}$ in PaLM/T5X) prevents spikes *because* it bounds $\log Z$. The published evidence is that runs with it are more stable — no arm isolates the bounding effect from the extra gradient it puts on $W_U$.
- That logit growth is *causal* rather than a co-symptom of a shared upstream cause (outlier features, Adam second-moment staleness). Molybog et al. (2023) give an alternative mechanism entirely: Adam's update collapses toward $\pm 1$ when gradient variance shrinks in deep, stale-state blocks.

**Benchmark-number-only.** OLMo 2 (2025), Chameleon (2024), and most open recipe reports state "we adopted QK-norm and/or z-loss and observed no divergence." These are single-run recipe notes, not ablations; spike rate is not reported with seeds or confidence intervals anywhere in the open literature.

## 4. What Is Known

- **Scale of logit growth.** In the small-scale proxy setting (models ~20M–1.2B parameters, LR swept over ~3 orders of magnitude), max attention logits grow to $\gtrsim 10^3$–$10^4$ before divergence, versus $O(10)$ in stable runs (Wortsman et al., ICLR 2024).
- **Output logit divergence** appears in those runs as $\log Z$ drifting upward while the output distribution's entropy stays roughly fixed — i.e. the head's overall scale, not its sharpness, is what runs away.
- **Attention entropy collapse** (Zhai et al., ICML 2023) is the attention-side correlate: entropy falls as logits grow, and $\sigma$Reparam (spectral normalization of weights) prevents both. Measured on ViTs and speech/LM tasks up to ~1B scale.
- **Outlier features** grow in the residual stream with depth and training and are reduced by normalization placement and signal-propagation fixes (He et al., NeurIPS 2024) — the same interventions that reduce logit growth.
- **Spikes are state-dependent, not data-determined** (PaLM 540B control above).
- **Gradient-norm condition.** Takase et al., *Spike No More* (2023), argue spikes follow large embedding-layer gradient norms and give init/scaling conditions that keep them small; validated on models up to ~13B in their reported runs.

## 5. What Is Not Known

- **Empirically open.** Whether logit growth *precedes* spikes with a stable, seed-independent lead time $\Delta > 0$. Nobody has published per-step $L_s$ traces across $\ge 8$ seeds at $\ge 1$B scale in the instability-provoking LR regime. The experiment is entirely runnable — it costs compute, not new methodology.
- **Empirically open.** Whether *inducing* logit growth (sufficiency) causes spikes. No published run injects $\log Z$ growth as an intervention.
- **Non-identifiable as currently posed.** Whether z-loss/QK-norm work through logit bounding or through their effect on effective learning rate. Every existing arm confounds the two.
- **Theoretically open.** No theorem connects $\log Z_t$ magnitude to a positive expected one-step loss change under Adam. The Molybog et al. analysis is the only mechanistic account with a derivation, and it does not involve the output head at all.
- **Methodologically blocked.** "Spike rate" has no standard definition. $\kappa$, EMA half-life, and whether to count steps or events vary by paper, so cross-paper spike counts are not comparable.

## 6. Why It Is Hard

Three specific obstructions, in order of severity.

1. **Confounded measurement by construction.** $\ell = \log Z - z_y$. A loss spike is arithmetically a $\log Z$ excursion not matched by $z_y$. Any correlational study "finding" that logit growth accompanies spikes has partly measured an identity. Only the pre-spike *lead* window carries information.
2. **Non-identifiability of the intervention.** QK-LayerNorm changes attention logits *and* the network's Jacobian, hence the effective LR of every parameter. z-loss adds a term to $\nabla_{W_U}$. There is no published intervention that moves $\log Z$ while holding the gradient distribution fixed. Without one, "logit growth causes spikes" and "the fix incidentally lowers effective LR" are observationally equivalent.
3. **Compute cost of the event rate.** Spikes are rare (PaLM: ~20 events over the full 540B-parameter run). Detecting a factor-2 change in hazard rate with 80% power needs on the order of tens of events per arm, so either many seeds or deliberate operation near the stability edge — which itself changes the phenomenon being studied.

## 7. Current Research (as of 2026)

- **Small-scale proxy methodology** (Google DeepMind lineage, following Wortsman et al.) — establishing that LR-sensitivity at 100M scale predicts instability at 100B scale. Extension to spike *timing* rather than final divergence is the active edge.
- **Normalization placement and logit capping** in open recipes: QK-norm plus z-loss is now the default in OLMo 2, Gemma-2-style soft-capping ($c \tanh(z/c)$), and most open 2025–2026 releases. Reported as recipe choices, not ablations.
- **Optimizer-side accounts**: Adam-$\varepsilon$ and second-moment staleness analyses following Molybog et al.; Muon/Shampoo-style preconditioners are claimed to spike less, *(frontier — verify)* — no controlled comparison at matched token budget is public.
- **Signal-propagation and outlier-feature work** (ETH Zürich, He/Hofmann and collaborators) treating outlier features as the shared upstream cause of both logit growth and instability.
- **Interpretability of the head**: work on the unembedding's effective rank and the "entropy neuron" family suggests the head has dedicated capacity for controlling $\log Z$ — relevant because it implies logit growth may be functional, not pathological *(frontier — verify)*.

## 8. Concrete Next Experiment

**Design: a sufficiency test with a norm-matched control.**

- **Scale.** 610M-parameter decoder-only transformer, 30B tokens, batch 1M tokens, Adam $\beta_2 = 0.95$, LR set to $0.7\times$ the empirically largest stable LR (measured first by a 6-point LR sweep at 300M and 610M). 8 seeds per arm, 4 arms. Roughly $1.1 \times 10^{20}$ FLOPs per run, ~32 runs — order 3,000 H100-hours.
- **Arms.**
  1. *Control:* baseline, no logit intervention.
  2. *Injection:* at step 8,000, add a frozen scalar $\beta_s$ to all logits ramping so $\log Z$ rises by +6 nats over 1,000 steps, then holds. A uniform logit shift is exactly cancelled by the softmax, so $\ell$, $\nabla_\theta \ell$, and every gradient statistic are **unchanged** — the arm moves the measured logit-growth statistic without touching optimization.
  3. *Real injection:* same +6 nats, but via a learned-then-frozen non-uniform scaling of $W_U$ rows, which does change the distribution.
  4. *Suppression:* z-loss $\lambda = 10^{-4}$, with LR rescaled per-block so measured update RMS matches the control within 5%.
- **The deciding number.** Spike-event count per $10^4$ steps in the 2,000-step window after injection, arm 3 minus arm 1, with a 95% bootstrap CI over 8 seeds. If the difference is $\ge +2$ events per $10^4$ steps with CI excluding 0, logit growth is *sufficient* to cause spikes. If arm 2 (the softmax-invariant shift) also moves the number, the measurement statistic is invalid and the field's correlational evidence collapses. If arm 3 shows no effect while arm 4 lowers spike rate, logit growth is a co-symptom and z-loss works through something else.

## 9. Key References

- **[Foundational]** Aakanksha Chowdhery et al. *PaLM: Scaling Language Modeling with Pathways.* JMLR, 2023. — arXiv:2204.02311
- **[SOTA]** Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, Simon Kornblith. *Small-scale proxies for large-scale Transformer training instabilities.* ICLR, 2024. — arXiv:2309.14322
- **[SOTA]** Mostafa Dehghani et al. *Scaling Vision Transformers to 22 Billion Parameters.* ICML, 2023. — arXiv:2302.05442
- Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, Josh Susskind. *Stabilizing Transformer Training by Preventing Attention Entropy Collapse.* ICML, 2023. — arXiv:2303.06296
- Igor Molybog et al. *A Theory on Adam Instability in Large-Scale Machine Learning.* Preprint, 2023. — arXiv:2304.09871
- Sho Takase, Shun Kiyono, Sosuke Kobayashi, Jun Suzuki. *Spike No More: Stabilizing the Pre-training of Large Language Models.* Preprint, 2023. — arXiv:2312.16903
- Aohan Zeng et al. *GLM-130B: An Open Bilingual Pre-trained Model.* ICLR, 2023. — arXiv:2210.02414
- Bobby He, Lorenzo Noci, Daniele Paliotta, Imanol Schlag, Thomas Hofmann. *Understanding and Minimising Outlier Features in Transformer Training.* NeurIPS, 2024. — arXiv:2405.19279
- Chameleon Team (Meta AI). *Chameleon: Mixed-Modal Early-Fusion Foundation Models.* Preprint, 2024. — arXiv:2405.09818
- **[Survey-adjacent]** Team OLMo. *2 OLMo 2 Furious.* Preprint, 2025. — arXiv:2501.00656

## 10. Worked Example

Take a 610M model, $V = 50{,}257$, at step 20,000 with training loss $\ell = 2.60$ nats.

Suppose the head is well calibrated and $\mathbb{E}[\log Z] = 8.10$, so $\mathbb{E}[z_y] = 8.10 - 2.60 = 5.50$. Now a spike takes $\ell$ from 2.60 to 4.10 over 40 steps ($+1.5$ nats, about $9\hat\sigma$ with $\hat\sigma = 0.017$). Two decompositions fit the same observation exactly:

| | $\log Z$ | $z_y$ | $\ell$ |
|---|---|---|---|
| pre-spike | 8.10 | 5.50 | 2.60 |
| **(A) head blowup** | 12.30 | 8.20 | 4.10 |
| **(B) feature collapse** | 8.05 | 3.95 | 4.10 |

In (A), $\log Z$ jumped 4.2 nats and the logit-growth statistic $L_s$ fires. In (B), $\log Z$ barely moved and only the correct-token logit fell. Both produce an identical loss trace. So a spike detector on $\ell$ plus a covariate $L_s$ **cannot distinguish cause from arithmetic identity** — this is obstruction (1) made concrete.

Now the lead-time test. Over the 2,000 steps before the spike, $L_s$ drifts from 7.90 to 8.10, i.e. $+1.0 \times 10^{-4}$ nats/step, while its step-to-step noise is $\hat\sigma_L \approx 0.05$ nats. The signal-to-noise ratio of the drift over a 200-step window is $200 \times 10^{-4} / (0.05/\sqrt{200}) \approx 5.7$ — detectable in one run. But the drift is also present in runs that never spike: across 8 baseline seeds the same $+1.0\times10^{-4}$/step slope appears in all of them, and only 3 spike. Precision of "rising $\log Z$" as a spike predictor is then $3/8 = 0.375$ against a base rate of $0.375$ — **zero lift**.

That is the state of the evidence: the correlate is real and easy to measure, the identity makes it partly tautological, and the discriminative content at the only lag that could show causation has never been measured. The injection arm in §8 is the smallest thing that breaks the tie, because a softmax-invariant logit shift moves the statistic while provably leaving every gradient untouched.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*