---
id: 04-alignment/alignment-tax-constant-compute
title: "Alignment Tax Measurement at Constant Compute"
topic: 04-alignment
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Alignment Tax Measurement at Constant Compute

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/alignment-tax-constant-compute` · **Status:** empirically-open

## 1. Problem Statement

The "alignment tax" is the capability lost when a base language model is turned into a helpful, harmless assistant. It is quoted constantly and measured almost never under a fair control.

The standard comparison is *aligned checkpoint vs. base checkpoint*. That comparison is confounded: the aligned model has consumed extra training compute (SFT + reward model + RL rollouts), extra data, and a different output format. The base model gets none of these. Any capability difference mixes at least four causes: (i) compute spent on alignment rather than on more pretraining, (ii) forgetting induced by the alignment objective, (iii) refusals and hedging that suppress otherwise-correct answers, (iv) prompt-format mismatch that under-elicits the base model.

Three variants, of very different difficulty:

- **Measurement.** Define and estimate $\tau(C)$ — the capability gap between an aligned model and a control that received *the same total training FLOPs* — with the elicitation and refusal components separated out. This is the open problem the page is about.
- **Method.** Reduce $\tau$ at fixed alignment quality (KL-regularized RL, pretraining-gradient mixing, weight interpolation). Partially solved; see §3.
- **Theory.** Predict $\tau$ as a function of model scale $N$, alignment compute $C_A$, and preference-data size $D_p$. Essentially untouched.

**Solved** would mean: an iso-FLOP protocol with a named control arm, under which two independent labs report $\tau$ at $\geq 3$ model scales with overlapping confidence intervals, and with the refusal and format components reported separately from the capability component.

## 2. Formal Setting

Base model $\theta_0$ trained with pretraining compute $C_0 \approx 6ND_0$ FLOPs ($N$ non-embedding parameters, $D_0$ tokens; Kaplan et al. 2020).

An alignment procedure $A$ maps $\theta_0 \mapsto \theta_A$ at cost
$$C_A = C_{\text{SFT}} + C_{\text{RM}} + C_{\text{RL}},$$
where $C_{\text{RL}}$ must include *rollout* FLOPs (generation dominates: $\approx 2N$ per token generated, times $k$ samples per prompt), not just gradient FLOPs. In practice papers report epochs, not FLOPs; this is the first measurement gap.

**Control arm.** $\theta_C$ = $\theta_0$ trained for a further $C_A$ FLOPs on the pretraining mixture. Then
$$\tau(C_0, C_A) \;=\; \mathbb{E}_{t \sim \mathcal{T}}\big[ m_t(\theta_C) \big] \;-\; \mathbb{E}_{t \sim \mathcal{T}}\big[ m_t(\theta_A) \big],$$
with $\mathcal{T}$ a capability suite and $m_t \in [0,1]$ a per-task score. $\tau > 0$ is a tax; $\tau < 0$ is an alignment *bonus*.

**Elicitation-corrected tax.** Each model is scored under its own best prompting policy $\pi$ from a fixed pool $\Pi$ (few-shot, zero-shot, chat template, "answer directly" prefix):
$$\tilde\tau \;=\; \max_{\pi \in \Pi} \mathbb{E}[m_t(\theta_C, \pi)] - \max_{\pi \in \Pi} \mathbb{E}[m_t(\theta_A, \pi)].$$
Without this max, $\tilde\tau$ measures template mismatch, not capability.

**Refusal decomposition.** Let $r_t$ be the refusal/abstention rate on task $t$. Report
$$\tilde\tau = \underbrace{\tilde\tau^{\text{ans}}}_{\text{scored on answered items only}} + \underbrace{\Delta_r \cdot \bar m}_{\text{refusal-attributed loss}},$$
so that a model that is merely more cautious is not scored as less capable.

**Iso-quality constraint.** $\tau$ is meaningless without fixing alignment strength. Bind either the win rate $w(\theta_A)$ against a reference policy, or the KL budget $\mathrm{KL}(\pi_{\theta_A} \| \pi_{\theta_0})$ measured on a held-out prompt set. Report $\tau$ as a curve $\tau(w)$ or $\tau(\mathrm{KL})$, never a scalar.

**Assumptions, and which are violated.**
1. *$m_t$ is a stable capability measure.* Violated — benchmark contamination and format sensitivity move MMLU-style scores by several points independent of capability.
2. *The control's extra pretraining is on the same distribution as $\theta_0$'s.* Usually violated — frontier labs anneal the data mixture late in training, so "more pretraining" is not well defined at the margin.
3. *$C_A \ll C_0$, so the control's gain is small.* Roughly holds (RLHF is commonly $10^{-3}$–$10^{-2}$ of pretraining), which makes $\tau$ a small difference of large numbers — noise-dominated.
4. *Preference data is i.i.d. from a single population.* Violated; annotator disagreement on helpfulness/harmlessness is 20–40% in released HH data.

## 3. State of the Art

**Established (ablated, reproduced):**
- *PPO-ptx.* Ouyang et al., InstructGPT (NeurIPS 2022) added a pretraining-gradient term to the PPO objective and showed it removes most of the regression on SQuAD, DROP, HellaSwag and translation relative to plain PPO, at matched human-preference win rate. This is the canonical demonstration that a large part of the tax is *forgetting*, not a capability/alignment trade-off.
- *KL-regularized RL controls the tax knob.* Stiennon et al. (NeurIPS 2020) and Gao, Schulman, Hilton (ICML 2023) establish that gold-reward degradation is a smooth function of $\sqrt{\mathrm{KL}}$, giving a principled iso-quality axis.
- *Weight interpolation recovers robustness.* Wortsman et al., WiSE-FT (CVPR 2022) showed linear interpolation between pre- and post-finetuning weights recovers distribution-shift robustness at no accuracy cost in CLIP; Lin et al. (EMNLP 2024, *Mitigating the Alignment Tax of RLHF*) transfer model-averaging to RLHF and report reduced tax at matched reward.

**Claimed but unablated:**
- "The alignment tax vanishes at scale." Traced to Askell et al. (2021), who found the tax negligible for their largest models (up to 52B) and positive for small ones. This is a base-vs-aligned comparison, not iso-FLOP, and used context distillation rather than RLHF. It is repeated far beyond its evidence.
- Post-2023 frontier model cards asserting "no capability regression from safety training." These are benchmark numbers on internal suites with no control arm and no elicitation matching.

**Benchmark-only results:** essentially all published tax numbers — MMLU/GSM8K/HumanEval deltas between a base and a chat checkpoint. No public paper reports $\tau$ against a compute-matched continued-pretraining control.

## 4. What Is Known

- **Small models pay, large models pay less (base-vs-aligned).** Askell et al. (2021): imitation-learning alignment costs measurable loss at 13M–1B; approximately zero at 52B on their evaluations.
- **Preference optimization degrades diversity.** Kirk et al. (ICLR 2024) measured RLHF vs. SFT on Llama-7B-class models: RLHF improved out-of-distribution generalization but cut per-input output diversity substantially (single-digit to low-tens percent, task dependent), with SFT preserving more diversity.
- **Overoptimization is lawful.** Gao et al. (ICML 2023), reward models 3M–3B: gold reward follows $R(d)=d(\alpha-\beta \log d)$ for best-of-$n$ with $d=\sqrt{\mathrm{KL}}$, so past a KL threshold, proxy reward rises while true quality falls. The tax at high KL is partly reward hacking, not alignment.
- **Alignment is fragile to small updates.** Qi et al. (ICLR 2024) removed safety behavior from GPT-3.5-class and Llama-2-Chat with ~100 fine-tuning examples for under \$0.20 — evidence that the aligned solution sits very close to the base weights, which bounds how much capability RLHF can plausibly destroy.
- **Alignment can be a bonus.** InstructGPT 1.3B was preferred to 175B GPT-3 by human raters ~85% of the time — a $100\times$ effective-compute *gain* on the preference metric. Whichever sign $\tau$ takes depends entirely on which metric is called "capability."

## 5. What Is Not Known

- **Empirically open (the core gap).** No one has run the iso-FLOP control. Spending $C_A$ FLOPs on continued pretraining instead of on RLHF, at $\geq 3$ scales, with elicitation-matched evaluation, is a runnable experiment at 1B–8B for well under \$100k. Nobody has published it.
- **Empirically open.** Whether $\tau$ actually decreases with $N$ under a fair control, or whether the apparent decrease is an artifact of large models being less format-sensitive.
- **Methodologically blocked.** The refusal/capability split. There is no agreed operationalization of "would have answered correctly had it not refused"; counterfactual scoring requires a ground truth the abstaining model does not emit.
- **Theoretically open.** No scaling law $\tau(N, C_A, D_p)$, and no proof that any nonzero tax is necessary. There is no theorem saying a policy maximizing a preference objective under a KL ball of radius $\epsilon$ must lose $\Omega(f(\epsilon))$ on an unrelated capability metric.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by a small effect size**. $C_A$ is $10^{-3}$–$10^{-2}$ of $C_0$, so the honest control ("more pretraining") moves benchmarks by well under one point — comparable to seed variance and prompt-template variance on MMLU-scale suites. Detecting $\tau$ therefore requires either many seeds or a deliberately inflated $C_A$, which changes the regime being measured.

Second obstruction: **the evaluation does not measure the thing it names**. Base models are scored few-shot with log-likelihood ranking; chat models are scored zero-shot with generative parsing. A large fraction of reported "alignment tax" is an answer-extraction artifact. Third: **non-identifiability** — refusal, hedging, verbosity and genuine capability loss all reduce the same score, and no public protocol separates them.

## 7. Current Research (as of 2026)

- **Model merging / interpolation as tax mitigation.** Follow-ups to WiSE-FT and Lin et al. (2024) in open-weight post-training stacks; the reported wins are at matched reward, not matched compute. *(frontier — verify)*
- **Online DPO and iterative preference optimization** (Meta, Cohere, academic groups) — cheaper alignment shifts $C_A$ down by an order of magnitude, which shrinks the honest tax but also shrinks measurability.
- **Regularization to the base policy** beyond KL: Fisher-weighted penalties, LoRA-restricted post-training, replay of pretraining data during RL.
- **Inference-compute alignment** — safety and preference behavior obtained through reasoning at test time rather than weight updates, which relocates the tax from parameters to token budget and needs a different accounting entirely. *(frontier — verify)*
- **Evaluation-format robustness** work (Anthropic, EleutherAI harness maintainers) is the prerequisite for any credible $\tilde\tau$.

## 8. Concrete Next Experiment

**Scale.** Three open base models with public pretraining data: ~1B, ~7B, ~14B (e.g. the OLMo or Qwen base families, where the pretraining mixture is available or reproducible).

**Arms** (per scale, 3 seeds each):
1. **Aligned:** SFT + DPO (or PPO) to a fixed KL budget $\mathrm{KL} = 10$ nats on held-out prompts. Log all FLOPs including rollouts; call it $C_A$.
2. **Control (the point of the experiment):** continued pretraining on the original mixture for exactly $C_A$ FLOPs.
3. **Null control:** $\theta_0$ untouched, to bound seed and format noise.

**Evaluation.** MMLU, GSM8K, HumanEval, ARC-C, plus a held-out-perplexity probe. Every arm scored under all four prompting policies in $\Pi$; take the per-arm max. Report refusal rate per task and score answered-only separately.

**The deciding number.** $\tilde\tau^{\text{ans}}$ at 7B, in benchmark points, with a 95% CI from the 3 seeds. If $\tilde\tau^{\text{ans}} < 1.0$ point and the CI includes zero at all three scales, the alignment tax as commonly quoted is a format-and-refusal artifact, and the field should stop citing it as a capability cost. If $\tilde\tau^{\text{ans}} > 2$ points with a CI excluding zero, and it shrinks monotonically in $N$, the scale-attenuation folklore is confirmed for the first time under a fair control.

**Cost.** Dominated by arm 2. At $C_A \approx 10^{-2} C_0$ for a 7B model, that is roughly $10^{20}$ FLOPs per seed — order 100–300 A100-hours. All three scales, three arms, three seeds: comfortably under \$100k.

## 9. Key References

- **[Foundational]** Amanda Askell, Yuntao Bai, Anna Chen, et al. *A General Language Assistant as a Laboratory for Alignment.* arXiv preprint, 2021. — arXiv:2112.00861
- **[Foundational]** Long Ouyang, Jeff Wu, Xu Jiang, et al. *Training language models to follow instructions with human feedback.* NeurIPS, 2022. — arXiv:2203.02155
- **[Foundational]** Nisan Stiennon, Long Ouyang, Jeff Wu, et al. *Learning to summarize with human feedback.* NeurIPS, 2020. — arXiv:2009.01325
- **[SOTA]** Leo Gao, John Schulman, Jacob Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[SOTA]** Yong Lin, Hangyu Lin, Wei Xiong, et al. *Mitigating the Alignment Tax of RLHF.* EMNLP, 2024. — arXiv:2309.06256
- **[SOTA]** Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, et al. *Robust fine-tuning of zero-shot models.* CVPR, 2022. — arXiv:2109.01903
- **[Empirical]** Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, et al. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR, 2024. — arXiv:2310.06452
- **[Empirical]** Xiangyu Qi, Yi Zeng, Tinghao Xie, et al. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR, 2024. — arXiv:2310.03693
- **[Background]** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[Background]** Rafael Rafailov, Archit Sharma, Eric Mitchell, et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS, 2023. — arXiv:2305.18290

## 10. Worked Example

Take a 7B base model, $C_0 = 6 \times 7\times10^9 \times 2\times10^{12} \approx 8.4\times10^{22}$ FLOPs.

Alignment budget: SFT on 50k examples (~$5\times10^7$ tokens, $6ND = 2.1\times10^{18}$), a 7B reward model trained on 100k pairs ($\approx 4\times10^{18}$), and PPO over 100k prompts with 4 rollouts of 512 tokens each. Rollout cost alone: $2 \times 7\times10^9 \times 100{,}000 \times 4 \times 512 \approx 2.9\times10^{20}$ FLOPs — **two orders of magnitude above the SFT gradient cost**, and routinely omitted from "alignment is cheap" claims. Total $C_A \approx 3\times10^{20}$, i.e. $0.36\%$ of $C_0$.

Now the control. $3\times10^{20}$ FLOPs of continued pretraining buys $D = C_A/6N \approx 7\times10^9$ tokens, or $+0.36\%$ on a 2T-token budget. Using a Chinchilla-style loss curve $L(D) = L_\infty + B D^{-\beta}$ with $\beta \approx 0.28$, the loss improvement is
$$\Delta L / (L - L_\infty) \approx \beta \times 0.0036 \approx 0.1\%,$$
which on a typical $L - L_\infty \approx 0.4$ nats gives $\Delta L \approx 4\times10^{-4}$ nats. Translated through observed loss-to-MMLU slopes, that is well under $0.1$ benchmark points.

**The obstruction, made visible.** The compute-matched control is *indistinguishable from the base model* — the honest iso-FLOP correction to $\tau$ is smaller than a single seed's variance. So "constant compute" cannot be the whole framing: measured against a fair control, essentially the entire reported alignment tax must be attributed to forgetting, refusal, or elicitation, not to compute diverted from pretraining. Meanwhile the raw base-vs-chat gap on the same 7B model is commonly reported at 2–5 MMLU points, and switching the base model from 5-shot log-likelihood scoring to zero-shot generative scoring alone can cost it more than that. The quantity the field calls the alignment tax is, at 7B, dominated by the measurement protocol rather than by anything the alignment procedure did to the weights — which is exactly why the experiment in §8 has to score every arm under every prompting policy before the number means anything.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*