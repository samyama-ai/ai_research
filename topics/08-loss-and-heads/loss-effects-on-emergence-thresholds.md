---
id: 08-loss-and-heads/loss-effects-on-emergence-thresholds
title: "Loss Function Effects on Emergent Capability Thresholds"
topic: 08-loss-and-heads
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Loss Function Effects on Emergent Capability Thresholds

> **Topic:** Loss Functions & Output Heads · **ID:** `08-loss-and-heads/loss-effects-on-emergence-thresholds` · **Status:** methodologically-blocked

## 1. Problem Statement

Some downstream capabilities appear abruptly as pretraining proceeds: near-chance accuracy for a long stretch, then a fast rise over a short interval of compute or loss. The question here is whether the **training objective** — the loss function and output head, not the data, architecture, or budget — moves the point at which that rise happens.

Three variants, of very different difficulty:

- **Measurement.** Given two models trained with different objectives $\mathcal{L}_A, \mathcal{L}_B$, place them on a *common* x-axis and report the value at which a task crosses a fixed performance level. This is the blocked variant: no objective-invariant axis is agreed on.
- **Method.** Find an objective (auxiliary term, head, denoiser mixture) that moves a target capability's threshold earlier by a stated margin at fixed compute and data. Partially answered — some interventions move some tasks.
- **Theory.** Prove or refute: for a fixed architecture and data distribution, the compute at which task $T$ crosses threshold $\tau$ is a function of the achieved *density estimate* alone, invariant to which proper scoring rule produced it. Wholly open.

A solution to the measurement variant is a statistic $\Phi$, computable for any objective, such that emergence thresholds expressed in $\Phi$ agree across objectives to within measurement error — or a demonstration that no such $\Phi$ exists.

## 2. Formal Setting

Corpus distribution $\mathcal{D}$ over byte strings. A model $p_\theta$ is trained with objective $\mathcal{L}$ under budget $C \approx 6ND$ FLOPs ($N$ non-embedding parameters, $D$ tokens).

**Held-out loss, as measured.** For a causal LM with tokenizer $V$,
$$\ell(\theta) = -\frac{1}{|x|}\sum_{i} \log p_\theta(x_i \mid x_{<i}) \quad \text{nats/token}.$$
Tokenizer-invariant form, bits per byte:
$$\mathrm{BPB}(\theta) = \frac{\ell(\theta)}{\ln 2 \cdot \bar{b}}, \qquad \bar{b} = \text{mean bytes per token on the held-out set}.$$
$\mathrm{BPB}$ is comparable across tokenizers **only** if $p_\theta$ is a normalized density over the same byte strings. It is not defined for span-corruption or masked objectives, whose losses are conditionals under a corruption process $q$, not $\log p(x)$.

**Task metric.** Task $T$ with items $(q_j, a_j)$ and scorer $s$. Report both a discontinuous metric $M_{\text{disc}} = \frac{1}{n}\sum_j \mathbb{1}[\hat{a}_j = a_j]$ and a continuous one, e.g. Brier score $M_{\text{cont}} = 1 - \frac{1}{n}\sum_j (1 - p_\theta(a_j\mid q_j))^2$, since the choice of metric alone can create or remove an apparent break (§3).

**Threshold.** For criterion level $\tau$ and axis $\Phi \in \{\log C, \ell, \mathrm{BPB}\}$,
$$\Phi^*_T(\mathcal{L}, \tau) = \inf\{\Phi : M_T \ge \tau\},$$
estimated by fitting a monotone curve to checkpoints and inverting; report a bootstrap CI over checkpoints and eval items. The object of study is the contrast $\Delta\Phi^*_T = \Phi^*_T(\mathcal{L}_A,\tau) - \Phi^*_T(\mathcal{L}_B,\tau)$ at matched $C$, $\mathcal{D}$, and architecture.

**Assumptions, and which fail.**
1. *$M_T$ monotone in $\Phi$* — violated: inverse and U-shaped scaling exist (McKenzie et al. 2023; Wei et al. 2023).
2. *Losses comparable across objectives* — violated by construction for denoising and MoE auxiliary losses.
3. *Held-out set is uncontaminated* — routinely violated at web scale; contamination inflates $M_T$ non-uniformly across checkpoints.
4. *One evaluation format* — violated: threshold shifts with prompt format and few-shot count, often by more than the objective effect.

## 3. State of the Art

**Established (empirical).** Du et al., *Understanding Emergent Abilities of Language Models from the Loss Perspective* (2024, arXiv:2403.15796), trained 1.5B/6B/32B models on up to 1T tokens and showed downstream accuracy on 12 English and Chinese benchmarks is well predicted by pretraining loss alone, collapsing across model size and token count. Several tasks stay at chance until loss falls below ≈2.2 nats/token, then rise — and the break persists under continuous metrics, which is evidence against the pure-metric-artifact account. **All models in that study share one objective and one tokenizer**, so it says nothing about $\Delta\Phi^*$ across losses.

**Established (methodological critique).** Schaeffer, Miranda & Koyejo, *Are Emergent Abilities of Large Language Models a Mirage?* (NeurIPS 2023, arXiv:2304.15004): claimed emergent abilities on BIG-Bench concentrate overwhelmingly under two discontinuous scorers (Multiple Choice Grade, Exact String Match); replacing them with continuous scorers removes most breaks, and sharp curves can be induced in vision models by choosing a discontinuous metric.

**Claimed but not ablated against a loss-matched control.**
- Tay et al., *UL2* (ICLR 2023, arXiv:2205.05131): mixture-of-denoisers changes which tasks a 20B model does well on relative to a causal-LM baseline. Reported as benchmark numbers at one scale; no threshold curve, no common loss axis.
- Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (ICML 2024, arXiv:2404.19737): $n$-token-prediction heads give no benefit at small scale and clear gains at ≥3B — a scale-dependent sign flip attributable to the head. Reported as endpoint accuracies, not thresholds.
- Auxiliary z-loss ($10^{-4}\log^2 Z$, PaLM; router z-loss, ST-MoE): justified by stability, never evaluated for threshold effects.

**Prediction SOTA.** Hu et al., *Predicting Emergent Abilities with Infinite Resolution Evaluation* (ICLR 2024, arXiv:2310.03262) resolve sub-1% task signal by massive sampling, predicting a 2.4B model's performance from ≤0.03B-scale runs. Single objective.

## 4. What Is Known

- Loss–performance collapse across $N$ and $D$ holds within one objective: 1.5B–32B, ≤1T tokens, 12 benchmarks; task-specific loss thresholds around **2.2 nats/token** (Du et al. 2024).
- Metric choice can manufacture a break: >90% of BIG-Bench "emergent" claims sit under Exact String Match / Multiple Choice Grade (Schaeffer et al. 2023), across 204 tasks (BIG-Bench, TMLR 2023).
- Regularization strength moves a threshold in a controlled toy setting: *Omnigrok* (Liu et al., ICLR 2023) shows grokking onset is governed by weight norm, and adjusting weight decay or initialization scale removes or induces the delay entirely — modular addition, ≤10⁵ steps.
- Objective terms change endpoint capability at fixed compute: 13B multi-token-prediction models solve ~12% more HumanEval and ~17% more MBPP problems than next-token baselines; the effect is absent at 0.3B (Gloeckle et al. 2024).
- Downstream error is predictable from loss under over-training across 0.011B–6.9B and token/param ratios to 640 (Gadre et al. 2024, arXiv:2403.08540) — again single-objective.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted axis $\Phi$ on which a span-corruption model, a causal LM, and a model with auxiliary losses can be compared. $\ell$ is objective-specific; $\mathrm{BPB}$ requires a normalized density the denoising model does not provide. Without $\Phi$, $\Delta\Phi^*$ is undefined, not merely unmeasured.
- **Empirically open.** Even restricted to objectives that *are* densities (plain CE vs. CE + label smoothing vs. CE + z-loss vs. multi-token heads), no one has run a matched grid across ≥3 scales with per-checkpoint task curves. Runnable today; ~10³ GPU-days.
- **Theoretically open.** Whether emergence onset is a functional of the density estimate alone, or of the optimization path the loss induces. No proof either way. The Quantization Model (Michaud et al., NeurIPS 2023) predicts discrete "quanta" acquired in frequency order but does not derive objective-dependence of acquisition order.

## 6. Why It Is Hard

**Non-identifiability of the x-axis.** Two models trained with different objectives have no shared coordinate. Substituting compute for loss reintroduces the confound the loss axis was meant to remove: objectives differ in FLOPs-per-token efficiency (denoising sees a fraction of positions as targets), so equal $C$ is not equal effective supervision.

**Confounded measurement compounds it.** The threshold estimate is sensitive to (a) the scorer's continuity, (b) prompt format, (c) tokenizer $\bar b$, and (d) contamination. Published objective effects are single-scale endpoint numbers whose error bars are not reported at all, and are plausibly smaller than (a)–(d).

**Absent ground truth.** There is no independent definition of "the capability appeared." $\tau$ is a convention; results can be flipped by moving it.

## 7. Current Research (as of 2026)

- Loss-as-x-axis scaling work extended to multilingual and multimodal mixtures, following Du et al.; Tsinghua/Zhipu and DeepMind lines. *(frontier — verify)*
- Objective-design at scale: multi-token and lookahead heads moving from research to production decoders, motivated by speculative decoding as much as by capability. *(frontier — verify)*
- Tokenizer-free / byte-level models (MegaByte, byte-latent architectures) are the closest thing to a fix for the axis problem, since BPB is native there. Whether they can host denoising objectives comparably is untested. *(frontier — verify)*
- Developmental interpretability (Michaud-style quanta, circuit-formation timing) aims to supply the missing ground truth for "appeared."

## 8. Concrete Next Experiment

**Scale.** Four sizes — 150M, 400M, 1.4B, 6.9B non-embedding params — Chinchilla-optimal tokens (~20 tokens/param), one fixed data mixture, one fixed 32k tokenizer, ≥30 logged checkpoints each. Five arms: (1) plain CE **[control]**, (2) CE + z-loss $10^{-4}\log^2 Z$, (3) CE + label smoothing 0.1, (4) CE + 4-token-prediction auxiliary heads (main head unchanged at inference), (5) UL2 mixture-of-denoisers. Cost ≈ $4\times10^{22}$ FLOPs total, ~1.1k H100-days at $4\times10^{14}$ effective FLOP/s.

**Axis.** For arms 1–4, plot every task curve against held-out $\mathrm{BPB}$ measured with the *same* main-head causal density on the same byte stream — the auxiliary terms change training but not the density's definition. Arm 5 has no such density; report it against $\log C$ only, and record that fact as the result for the blocked variant.

**Tasks.** Eight with documented breaks (3-digit addition, modular arithmetic, IPA transliteration, word unscrambling, HumanEval, MMLU, GSM8K 8-shot, Persian QA). Score each with both Exact Match and Brier.

**Deciding number.** $\Delta\mathrm{BPB}^*$ at $\tau = 0.5 \times$ (ceiling − chance), bootstrapped over checkpoints and items. Decision rule: an objective moves thresholds iff $|\Delta\mathrm{BPB}^*| > 0.02$ bits/byte with a 95% CI excluding zero, **on the Brier axis**, for ≥3 of 8 tasks. 0.02 bits/byte is roughly the BPB gain from a 1.6× compute increase at these scales — below that, the effect is not worth an objective change.

## 9. Key References

- **[Foundational]** Wei, Tay, Bommasani, Raffel, et al. *Emergent Abilities of Large Language Models.* TMLR, 2022. — arXiv:2206.07682
- **[Foundational]** Srivastava et al. *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models.* TMLR, 2023. — arXiv:2206.04615
- **[SOTA]** Du, Zeng, Hou, Dong, Tang, et al. *Understanding Emergent Abilities of Language Models from the Loss Perspective.* 2024. — arXiv:2403.15796
- **[SOTA]** Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023. — arXiv:2304.15004
- **[SOTA]** Hu, Song, Zhu, et al. *Predicting Emergent Abilities with Infinite Resolution Evaluation.* ICLR, 2024. — arXiv:2310.03262
- **[SOTA]** Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction.* ICML, 2024. — arXiv:2404.19737
- **[SOTA]** Gadre et al. *Language Models Scale Reliably with Over-training and on Downstream Tasks.* 2024. — arXiv:2403.08540
- **[Method]** Tay, Dehghani, Tran, et al. *UL2: Unifying Language Learning Paradigms.* ICLR, 2023. — arXiv:2205.05131
- **[Theory]** Michaud, Liu, Girit, Tegmark. *The Quantization Model of Neural Scaling.* NeurIPS, 2023. — arXiv:2303.13506
- **[Theory]** Liu, Michaud, Tegmark. *Omnigrok: Grokking Beyond Algorithmic Data.* ICLR, 2023. — arXiv:2210.01117
- **[Context]** Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Survey]** McKenzie et al. *Inverse Scaling: When Bigger Isn't Better.* TMLR, 2023. — arXiv:2306.09479

## 10. Worked Example

Take Du et al.'s reported break at $\ell^* \approx 2.2$ nats/token and try to transfer it to another model.

Model A: 32k-vocab tokenizer, $\bar b_A = 3.7$ bytes/token on English held-out text.
$$\mathrm{BPB}_A = \frac{2.2}{0.6931 \times 3.7} = 0.858 \text{ bits/byte}.$$

Model B: 100k-vocab tokenizer, $\bar b_B = 4.3$ bytes/token. The same BPB corresponds to
$$\ell_B = 0.858 \times 0.6931 \times 4.3 = 2.56 \text{ nats/token}.$$

So "the threshold is 2.2 nats" becomes 2.56 nats — a **16% shift from tokenization alone**. In the same study the loss gap between "chance" and "clearly above chance" checkpoints on the sharpest tasks is on the order of 0.1–0.2 nats. The tokenizer artifact (0.36 nats) is **two to three times larger than the effect being measured**. Any cross-objective comparison must therefore run on BPB, not nats.

Now add arm 5. UL2's span-corruption loss is $-\log q$-weighted conditional on corrupted context; it is not $-\log p(x)$, so $\bar b$ does not convert it. Numerically it can sit *below* 2.2 while the model's byte-level compression is worse, because the target positions are a strict, easier subset. The conversion is not merely unknown — no scalar rescaling exists, since the two quantities score different events. That is the block: the causal-LM arms can be put on one axis and answered empirically; the denoising arm cannot be put on that axis at all without first defining a normalized density for it (e.g. by importance-weighting over corruption masks, at cost exponential in span count). Until someone supplies that estimator with bounded variance, the headline question "does the loss function move emergence thresholds?" is answerable only inside the family of objectives that are already densities.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*