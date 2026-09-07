---
id: 18-rl-for-llms/catastrophic-forgetting-rl-finetuning
title: "Catastrophic Forgetting During RL Fine-Tuning"
topic: 18-rl-for-llms
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Catastrophic Forgetting During RL Fine-Tuning

> **Topic:** Reinforcement Learning for LLMs · **ID:** `18-rl-for-llms/catastrophic-forgetting-rl-finetuning` · **Status:** partially-solved

## 1. Problem Statement

**Input.** A pretrained (usually also instruction-tuned) policy $\pi_0$, a task-specific reward $r$ (preference model, verifier, or rubric), an RL algorithm (PPO, GRPO, RLOO, DPO-family), and a compute budget.

**Output.** A policy $\pi_\theta$ that raises expected reward on the target task without degrading capabilities that were present in $\pi_0$ and are not covered by $r$: factual recall, multilinguality, calibration, instruction following, safety refusals, output diversity.

**Objective.** Reach a target reward $R^\ast$ while keeping the drop on a held-out retention suite below a tolerance $\varepsilon$.

Three variants, routinely conflated:

- **Measurement.** Given $\pi_0$ and $\pi_\theta$, decide how much capability was *lost* rather than *hidden*. Open — see §6.
- **Method.** Given $r$, produce a training procedure whose retention loss is below $\varepsilon$ at matched reward. Partially solved: KL penalties, pretraining-gradient mixing, and weight interpolation each buy real retention, none is free.
- **Theory.** Predict, before training, which capabilities a given $(\pi_0, r, \text{algorithm})$ will lose. Open.

Solving it means: an intervention that, at fixed target-task reward, provably (or reliably, across ≥3 base models and ≥5 retention domains) reduces retention loss to within noise of $\pi_0$.

## 2. Formal Setting

Policy $\pi_\theta(y \mid x)$ over token sequences. RL fine-tuning maximizes

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}_{\text{RL}},\, y \sim \pi_\theta(\cdot\mid x)}\big[r(x,y)\big] - \beta\, \mathbb{E}_{x}\big[\mathrm{KL}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_0(\cdot\mid x)\right)\big].$$

**Measured quantities.**

- **KL budget.** Not computed exactly. In practice it is the Monte-Carlo estimate over sampled rollouts, $\widehat{\mathrm{KL}} = \frac{1}{N}\sum_{i=1}^{N} \log\frac{\pi_\theta(y_i\mid x_i)}{\pi_0(y_i\mid x_i)}$ with $y_i \sim \pi_\theta$, or the low-variance $k_3$ estimator $\mathbb{E}[\rho - \log\rho - 1]$, $\rho = \pi_0/\pi_\theta$. Both are on-policy, on $\mathcal{D}_{\text{RL}}$ prompts only — they say nothing about drift on retention prompts.
- **Forgetting.** For retention tasks $t = 1..T$ with metrics $m_t$ (accuracy, pass@1, BLEU), $$F = \frac{1}{T}\sum_{t=1}^{T} \frac{m_t(\pi_0) - m_t(\pi_\theta)}{m_t(\pi_0) - m_t(\text{chance})}.$$ Normalizing by headroom matters: 3 points off a 90% task is not 3 points off a 30% task.
- **Diversity.** Per-prompt distinct-$n$ or mean pairwise embedding distance over $k$ samples at fixed temperature. RL collapses this even when $F \approx 0$.
- **Weight displacement.** $\|\theta - \theta_0\|_2$, and the sparsity of the update: fraction of parameters with $|\Delta\theta_i| > 0$ after a threshold.

**Assumptions, and how they break.**

1. *The retention suite spans the capabilities at risk.* Violated. Suites are 5–20 benchmarks; the pretraining distribution is not.
2. *$m_t$ measures a capability, not a format.* Violated routinely. RL on verifier-checked math rewrites answer formatting; strict-match graders then score format loss as knowledge loss.
3. *KL to $\pi_0$ on RL prompts bounds drift elsewhere.* Violated. KL is prompt-distribution-specific and unbounded off $\mathcal{D}_{\text{RL}}$.
4. *$\pi_0$ is fixed.* Violated in iterated RLHF, where $\pi_0$ is reset each round and drift compounds.

## 3. State of the Art

**Established (ablated, reproduced).**

- **PPO-ptx** (Ouyang et al., NeurIPS 2022): add a pretraining log-likelihood term $\gamma\,\mathbb{E}_{x\sim\mathcal{D}_{\text{pretrain}}}[\log \pi_\theta(x)]$ to the PPO objective. Ablated against PPO at 1.3B/6B/175B; it removes most of the observed "alignment tax" on public NLP tasks at essentially unchanged preference win-rate. This is the strongest, best-controlled result in the area, and it is four years old.
- **KL regularization to $\pi_0$.** Ablated in Stiennon et al. (NeurIPS 2020) and quantified by Gao, Schulman & Hilton (ICML 2023): proxy-vs-gold reward follows $R_{\text{RL}}(d) = d(\alpha - \beta\log d)$ with $d = \sqrt{\mathrm{KL}}$, over reward models from 3M to 3B parameters. This gives a *dial*, not a solution — retention and reward trade off along $\beta$.
- **Weight-space interpolation.** WiSE-FT (Wortsman et al., CVPR 2022) established for CLIP that $\theta_\lambda = (1-\lambda)\theta_0 + \lambda\theta$ improves OOD robustness at little ID cost. Transferred to RLHF by model-averaging work (e.g. Lin et al., *Mitigating the Alignment Tax of RLHF*, EMNLP 2024), with ablations but on fewer base models.

**Claimed but not fully ablated.**

- **RL forgets less than SFT at matched target performance**, with forgetting predicted by the forward KL $\mathrm{KL}(\pi_\theta\|\pi_0)$ ("RL's Razor", Shenfeld, Pari & Agrawal, 2025). Mechanism plausible and the KL–forgetting correlation is striking, but the matched-performance comparison depends heavily on how the SFT arm is tuned.
- **RL updates a small subnetwork** (5–30% of parameters), unlike SFT (2025 preprints, *frontier — verify*). Reported at 7B–70B; not yet shown to be causal for retention.
- **Reference-policy resetting** (ProRL, NVIDIA, 2025) as an enabler of thousand-step RL without collapse. Reported as a training recipe; no isolated ablation of the reset against a fixed-reference control at matched steps.

**Benchmark-number-only.** Most frontier reasoning-RL model cards report a retention table (MMLU, IFEval, safety) alongside the reasoning gain. These are single-run, single-seed, no matched-KL control. They are evidence that labs *watch* forgetting, not evidence about its mechanism.

## 4. What Is Known

- **The alignment tax is real and scale-dependent.** InstructGPT (175B, NeurIPS 2022) showed few-shot regressions on SQuADv2, DROP, HellaSwag and WMT Fr→En under plain PPO, largely recovered by PPO-ptx.
- **Pretraining scale reduces forgetting.** Ramasesh et al. (ICLR 2022) show, for T5 and ViT variants across roughly two orders of magnitude of parameters, that larger *pretrained* models forget sequentially-learned tasks substantially less; forgetting correlates with representational drift in early layers.
- **But instruction-tuned continual fine-tuning gets worse with scale in some setups.** Luo et al. (2023) report increasing forgetting from BLOOMZ 1.1B to 7.1B during continual instruction tuning — a direct tension with Ramasesh et al., unresolved, and likely due to differing task sequences.
- **Some "forgetting" is inference-time, not weight-time.** Kotha, Jain & Raghunathan (ICLR 2024) show fine-tuned LMs recover apparently-lost capabilities when the prompt is disambiguated toward the pretraining task ("implicit inference"): the knowledge is in the weights, the prior over tasks moved.
- **Fine-tuning distorts features.** Kumar et al. (ICLR 2022) prove and demonstrate that full fine-tuning from a randomly-initialized head distorts pretrained features and underperforms OOD versus linear-probe-then-fine-tune.
- **RLHF cuts output diversity.** Kirk et al. (ICLR 2024) find RLHF improves OOD generalization over SFT while reducing per-input output diversity across every measure tested, at 7B scale.
- **Theory exists only for linear models.** Evron et al. (COLT 2022) give tight bounds on forgetting for sequential linear regression, including cases where forgetting is bounded by $O(1/k)$ under random task ordering. Nothing comparable for policy-gradient updates on transformers.

## 5. What Is Not Known

- **Methodologically blocked.** How to separate *erasure* (information gone from $\theta$) from *suppression* (information present, prior shifted). Kotha et al. show the distinction is real; there is no accepted probe that decides it per capability. Until this is settled, every reported $F$ is an upper bound of unknown tightness.
- **Theoretically open.** Any nonvacuous bound on retention loss as a function of $\mathrm{KL}(\pi_\theta \| \pi_0)$ measured on $\mathcal{D}_{\text{RL}}$ and evaluated off $\mathcal{D}_{\text{RL}}$. Also open: whether the sparsity of RL updates is causal for their milder forgetting.
- **Empirically open.** Whether PPO-ptx-style replay still works for verifier-reward RL at $10^3$+ steps and 30B+ scale — runnable today, not published with controls. Whether forward vs reverse KL choice changes retention at matched reward.
- **Empirically open.** Whether forgetting during long-horizon RL is monotone or has a recoverable phase (drop then partial return as the policy consolidates).

## 6. Why It Is Hard

The primary obstruction is **confounded measurement**, not compute. RL fine-tuning changes three things at once: the answer distribution, the output format, and the implicit task prior. Standard retention benchmarks use strict-match parsing and short-form prompting, so all three register as accuracy loss. A model that has learned to emit a long chain of thought before answering will score near zero on a strict-match MMLU harness while having lost nothing.

Secondary: **non-identifiability**. Given only $(\pi_0, \pi_\theta)$ and benchmark scores, no known procedure identifies whether a capability was overwritten or de-prioritized — the two produce identical scores under the standard eval and differ only under prompts nobody has agreed on.

Tertiary and real, but not primary: matched-reward comparisons need the control arm trained to the *same* target performance, which means a sweep, which multiplies an already expensive RL run by 5–10×.

## 7. Current Research (as of 2026)

- **KL-geometry accounts of forgetting** — MIT Improbable AI (Shenfeld, Pari, Agrawal) on forward-KL as the predictor; follow-ups asking whether an explicit forward-KL penalty beats the implicit one *(frontier — verify)*.
- **Long-horizon RL stability** — NVIDIA (ProRL) and DeepSeek-lineage work on reference resets, entropy control, and KL scheduling over $10^3$–$10^4$ steps.
- **Sparse/subnetwork RL updates** — several 2025–26 preprints on whether restricting updates to a small mask preserves retention at matched reward *(frontier — verify)*.
- **Model merging as post-hoc tax removal** — interpolation and task-arithmetic between $\pi_0$ and $\pi_\theta$, now standard practice in open-weight release pipelines.
- **Eval reform** — log-likelihood and re-prompted scoring of retention suites to strip format confounds; not yet standardized across labs.

## 8. Concrete Next Experiment

**Question.** Is measured forgetting during verifier-reward RL mostly format/prior shift rather than weight-level loss?

**Scale.** One 7–8B base model (e.g. Qwen2.5-7B or Llama-3.1-8B), GRPO on a math/code verifier reward, 500 optimizer steps, ~$3\times10^3$ H100-hours total including controls. Three seeds.

**Arms.**
1. GRPO, $\beta = 0$.
2. GRPO, $\beta$ tuned to a fixed $\widehat{\mathrm{KL}} = 10$ nats.
3. **Control:** SFT on rollouts from arm 1, tuned to *matched* target-task pass@1 (±0.5 pt).

**Retention suite,** each scored three ways: (a) strict-match, standard prompt; (b) log-likelihood scoring over answer options (format-free); (c) strict-match with a task-disambiguating prefix (Kotha-style).

**Deciding number.** $\Delta = F_{\text{strict}} - F_{\text{loglik}}$, averaged over the suite, for arm 1. If $\Delta > 0.5\,F_{\text{strict}}$ — more than half the apparent forgetting vanishes under format-free scoring — the field's retention tables are primarily measuring format drift, and the methodological block in §5 is the binding constraint. If $\Delta < 0.2\,F_{\text{strict}}$, forgetting is substantive and the method question in §1 is the binding one.

## 9. Key References

- **[Foundational]** Kirkpatrick, Pascanu, Rabinowitz, et al. *Overcoming catastrophic forgetting in neural networks.* PNAS 114(13), 2017.
- **[Foundational]** Ouyang, Wu, Jiang, et al. *Training language models to follow instructions with human feedback.* NeurIPS, 2022. — arXiv:2203.02155
- **[Foundational]** Stiennon, Ouyang, Wu, et al. *Learning to summarize from human feedback.* NeurIPS, 2020. — arXiv:2009.01325
- **[SOTA]** Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[SOTA]** Kotha, Jain, Raghunathan. *Understanding Catastrophic Forgetting in Language Models via Implicit Inference.* ICLR, 2024.
- **[SOTA]** Kirk, Mediratta, Nalmpantis, et al. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR, 2024.
- **[SOTA]** Shenfeld, Pari, Agrawal. *RL's Razor: Why Online Reinforcement Learning Forgets Less.* Preprint, 2025.
- **[Theory]** Evron, Moroshko, Ward, Srebro, Soudry. *How catastrophic can catastrophic forgetting be in linear regression?* COLT, 2022.
- **[Theory]** Ramasesh, Lewkowycz, Dyer. *Effect of scale on catastrophic forgetting in neural networks.* ICLR, 2022.
- **[Method]** Kumar, Raghunathan, Jones, Ma, Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR, 2022. — arXiv:2202.10054
- **[Method]** Wortsman, Ilharco, Kim, et al. *Robust fine-tuning of zero-shot models.* CVPR, 2022. — arXiv:2109.01903
- **[Survey]** Wang, Zhang, Su, Zhu. *A Comprehensive Survey of Continual Learning: Theory, Method and Application.* IEEE TPAMI, 2024.

## 10. Worked Example

Take an 8B instruct model, RL-trained with a math verifier for 400 GRPO steps. Suppose the observed table (illustrative arithmetic, structure taken from the pattern reported in open model cards; the point is the *decomposition*, not the digits):

| Task | $\pi_0$ | $\pi_\theta$ strict | $\pi_\theta$ log-lik |
|---|---|---|---|
| MATH500 (target) | 42.0 | 68.0 | — |
| MMLU | 68.0 | 62.0 | 67.4 |
| IFEval strict | 74.0 | 66.0 | — |
| HumanEval | 61.0 | 59.5 | — |

Normalized forgetting under strict scoring, with chance = 25 for MMLU, 0 for IFEval and HumanEval:

$$F_{\text{strict}} = \tfrac13\left(\tfrac{68-62}{68-25} + \tfrac{74-66}{74} + \tfrac{61-59.5}{61}\right) = \tfrac13(0.140 + 0.108 + 0.025) = 0.091.$$

Under log-likelihood scoring the MMLU term collapses to $(68-67.4)/43 = 0.014$ — a factor of 10. The 6-point strict-match MMLU drop was 5.4 points of "the model now writes a derivation instead of the letter B" and 0.6 points of anything a weight-level account would call forgetting.

IFEval is worse: it is *not* format-free by construction, since the format *is* the capability. Its 8-point drop cannot be decomposed by this trick at all — the RL policy's learned preamble genuinely violates instructions like "answer in exactly one sentence."

The obstruction is now visible. The headline $F = 0.091$ mixes three incommensurable quantities: a measurement artifact (MMLU), a real capability loss (IFEval), and noise (HumanEval, 1.5 points, inside seed variance at this scale). Reporting their mean is not a measurement of forgetting. Any method claim — "our KL schedule reduces forgetting by 40%" — evaluated against this aggregate is unfalsifiable, because moving the format term alone moves the aggregate more than any weight-level intervention does.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*