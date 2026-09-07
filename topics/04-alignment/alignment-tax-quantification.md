---
id: 04-alignment/alignment-tax-quantification
title: "Alignment Tax Quantification Across Capabilities"
topic: 04-alignment
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Alignment Tax Quantification Across Capabilities

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/alignment-tax-quantification` · **Status:** empirically-open

## 1. Problem Statement

The **alignment tax** is the capability lost when a pretrained model is made helpful, honest, and harmless. The term is used loosely: papers report a single scalar ("RLHF costs 2 points on MMLU") when the underlying object is a *vector* over capabilities, and the sign of that vector differs by capability, by model scale, and by which control you compare against.

Three variants, with different difficulty:

- **Measurement.** Given a base model $\pi_0$, an aligned model $\pi_\theta$, and a capability suite $\{c_1,\dots,c_K\}$, produce a per-capability tax $\tau_k$ that is (i) invariant to prompt-format artifacts, (ii) attributed to the alignment objective rather than the extra tokens/compute spent, and (iii) comparable across labs. **Open, and currently the binding constraint.**
- **Method.** Reduce $\max_k \tau_k$ at fixed alignment quality (KL-regularized reward, refusal rate, harmlessness Elo). Partially solved — mixing pretraining gradients into PPO (PPO-ptx) removes most of the measured regressions.
- **Theory.** Prove a nonzero lower bound: exhibit a capability class and a preference distribution such that any policy achieving harmlessness $\ge h$ must lose $\ge \epsilon(h)$ on that class. **Theoretically open**; no separation theorem exists.

Solving the measurement variant means: a published protocol under which two labs, given the same base checkpoint and the same alignment budget, report $\tau_k$ agreeing to within their stated error bars.

## 2. Formal Setting

Let $\pi_0$ be the base policy after pretraining, $\mathcal{D}_{\text{pref}}$ a preference dataset, and $r_\phi$ a reward model. The aligned policy is

$$\pi_\theta = \arg\max_{\pi}\ \mathbb{E}_{x\sim\rho,\,y\sim\pi(\cdot|x)}\big[r_\phi(x,y)\big] - \beta\, D_{\mathrm{KL}}\!\left(\pi(\cdot|x)\,\|\,\pi_0(\cdot|x)\right).$$

Capability $k$ has an eval $E_k$ with scoring function $s_k$. **Measured** quantities:

- **Raw tax:** $\hat\tau_k = \mathbb{E}[s_k(\pi_0)] - \mathbb{E}[s_k(\pi_\theta)]$, each expectation a mean over $n_k$ eval items with binomial standard error $\sqrt{p(1-p)/n_k}$. On MMLU ($n=14{,}042$) that error is about $0.4$ points; a reported 2-point tax is $5\sigma$ only if the eval harness is held fixed.
- **Elicitation-corrected tax:** $\tau_k^{\ast} = \max_{u \in U}\mathbb{E}[s_k(\pi_0; u)] - \max_{u\in U}\mathbb{E}[s_k(\pi_\theta; u)]$, maximizing over a prompt/format/decoding set $U$ (few-shot count, chat template, chain-of-thought on/off, temperature). This is the quantity people *mean*, and it is almost never what is reported.
- **Alignment budget:** $\kappa = \mathbb{E}_x D_{\mathrm{KL}}(\pi_\theta\|\pi_0)$, estimated in nats per token from sequence logprobs. Gao et al. (ICML 2023) parameterize overoptimization in $d=\sqrt{\kappa}$.
- **Tax curve:** $\tau_k(\kappa)$, and the frontier scalar $T(h) = \sum_k w_k \tau_k$ at fixed harmlessness $h$. Without $\kappa$ or $h$ held fixed, $\tau_k$ is not a property of the alignment method at all.
- **Scale dependence:** $\tau_k(N)$ over parameters $N$; the empirically interesting claim is $\partial \tau_k/\partial N < 0$.

Assumptions, with those known violated marked:

1. $\pi_0$ and $\pi_\theta$ are evaluated under one harness. **Violated:** base models are scored with few-shot log-likelihood, chat models with zero-shot generation and a template; the format change alone moves MMLU by several points.
2. $E_k$ is uncontaminated by $\mathcal{D}_{\text{pref}}$ or SFT data. **Violated in practice, unmeasurable from outside** — SFT mixes contain benchmark-adjacent data.
3. Alignment adds no capability-relevant training signal. **Violated:** instruction tuning teaches format compliance, which *is* eval score on generative benchmarks.
4. $s_k$ measures capability $k$. **Violated for refusal-adjacent evals:** a harmless model refusing a chemistry question scores 0, which is policy, not incapacity.
5. Compute is matched between arms. **Usually violated:** aligned models get extra gradient steps that the base control never receives.

## 3. State of the Art

**Established (ablated, multi-seed, or independently reproduced):**

- Ouyang et al. (NeurIPS 2022) documented regressions on SQuAD, DROP, HellaSwag and WMT translation for PPO-tuned GPT-3 relative to base, and showed **PPO-ptx** — mixing pretraining log-likelihood gradients into the PPO objective — recovers most of them. This is an ablation, not just a benchmark table.
- Askell et al. (2021) reported the tax is scale-dependent: small models (below ~10B) lose zero-shot eval accuracy under HHH prompting/context distillation, large models lose little or gain. Bai et al. (2022) reproduced the pattern for full RLHF on 13M–52B models.
- Gao, Schulman & Hilton (ICML 2023) established the proxy-vs-gold reward gap scales smoothly in $\sqrt{\kappa}$ — giving the $x$-axis any tax curve must be plotted against.

**Claimed but unablated, or benchmark-number-only:**

- "RLHF has no alignment tax at frontier scale." This is a table entry in model cards, computed with different harnesses per arm, no $\kappa$ reported, and no elicitation maximization. It is not an ablation.
- Diversity cost: Kirk et al. (ICLR 2024) find RLHF improves out-of-distribution generalization but sharply reduces per-input output diversity versus SFT. The generalization result is ablated; the *mechanism* is not.
- Safety-vs-reasoning: Huang et al. (2025) report a "safety tax" — safety alignment of large reasoning models degrades reasoning benchmark scores. Benchmark numbers, single model family, no $\kappa$ control.
- Model averaging / weight interpolation between base and aligned checkpoints (Lin et al., EMNLP 2024) is reported to reduce the tax; the ablation exists but at ≤13B only.

## 4. What Is Known

- Direction is scale-dependent. At 52B, RLHF gave small *gains* on zero-shot NLP evals; below ~10B it gave losses of a few points (Bai et al. 2022, Anthropic 13M–52B sweep).
- Some regressions are real and fixable. GPT-3 175B PPO showed clear drops on SQuAD/DROP/WMT; PPO-ptx largely closed them (Ouyang et al. 2022).
- Overoptimization is lawful in $\sqrt{\kappa}$ up to 3B reward models with a gold-RM proxy (Gao et al. 2023) — the only clean dose-response curve in the area.
- Alignment is shallow along the token axis: safety behavior concentrates in the first few generated tokens (Qi et al., ICLR 2025), so a "tax" measured on first-token refusal is not a tax on the underlying distribution.
- Alignment is fragile: ~100 benign fine-tuning examples strip safety from aligned GPT-3.5/Llama-2 (Qi et al., ICLR 2024). A tax that a $100 fine-tune removes is not a capability loss — it is a suppression.
- Diversity falls under RLHF at 7B scale, measured as per-input distinct-$n$ and embedding dispersion (Kirk et al. 2024).

## 5. What Is Not Known

- **Methodologically blocked:** the elicitation-corrected tax $\tau_k^\ast$. No agreed set $U$, no agreed harness for scoring base and chat models identically, no standard for reporting $\kappa$. Because of this, cross-lab tax numbers are not commensurable — this is the dominant gap.
- **Empirically open:** the sign and magnitude of $\partial\tau_k/\partial N$ above 70B. The 52B sweep is 4 years old and the tooling to redo it at 70B–400B exists; nobody has published a matched-harness, matched-$\kappa$ sweep.
- **Empirically open:** whether tax is suppression or destruction. The distinguishing test is whether a capability lost to alignment returns under harmless-domain fine-tuning at low LR without restoring unsafe behavior.
- **Theoretically open:** any lower bound of the form "harmlessness $\ge h$ implies capability loss $\ge \epsilon(h)>0$" for a nontrivial capability class. Also open: whether the tax is intrinsic to the KL-regularized objective or an artifact of finite reward-model accuracy.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability**, not compute.

- **Format confound.** Base and aligned models are not evaluated under the same protocol anywhere in the literature. Changing few-shot count and chat template moves benchmark scores by more than the effect being measured, so $\hat\tau_k$ estimates a mixture of format sensitivity and capability change with no way to separate them post hoc.
- **Non-identifiability of refusal vs. incapacity.** For any wrong answer on a safety-adjacent item, "the model cannot" and "the model declines" are indistinguishable from the output. Ground truth for the counterfactual (what $\pi_\theta$ would answer absent the harmlessness constraint) does not exist.
- **The control arm is undefined.** The honest control is not $\pi_0$ — it is $\pi_0$ given the same number of gradient steps on the same data volume with the preference signal removed. Almost no paper runs it.
- **Free parameter.** $\tau_k$ depends on $\beta$/$\kappa$, which labs tune per release and do not report. Two "no tax" claims can differ by 10× in $\kappa$.

## 7. Current Research (as of 2026)

- **Tax-mitigation by weight-space methods** — model soups, base–aligned interpolation, adapter merging; strongest published ablations at ≤13B (Lin et al. 2024, and follow-ons).
- **Data-mixing** — PPO-ptx descendants: pretraining replay in RLHF and DPO pipelines, now standard at OpenAI/Anthropic/Meta but undocumented in ablation form.
- **Reasoning-model safety tax** — whether safety post-training on long-CoT models costs reasoning accuracy; active at academic labs and safety teams *(frontier — verify: numbers vary widely by benchmark and none control $\kappa$)*.
- **Elicitation-aware evaluation** — UK AI Safety Institute and METR argue capability claims must be made under best-effort elicitation. Applying that standard symmetrically to *both* arms of a tax measurement is the obvious next step and is not yet standard practice *(frontier — verify)*.
- **Shallow-alignment repair** — deepening safety beyond the first tokens (Qi et al. 2025) implies the measured tax may shrink or grow once alignment is made deep; unmeasured.

## 8. Concrete Next Experiment

**Matched-harness, matched-$\kappa$ tax curve at two scales.**

- **Scale:** one open base family at 8B and 70B (e.g. Llama-3.1 base checkpoints). Preference data: a public HH set plus a public helpfulness set, ~100k pairs. Train PPO or DPO at five $\beta$ values yielding $\kappa \in \{0.5, 1, 2, 4, 8\}$ nats/token, 3 seeds each. 30 runs; ~15k H100-hours total.
- **Control arm (the point of the experiment):** for every aligned run, a *format-matched, compute-matched* control — same chat template, same SFT data, same optimizer steps, with the preference term zeroed (log-likelihood only). This isolates the preference objective from the format and compute it arrives with.
- **Evaluation:** $K=8$ capabilities (MMLU, GSM8K, HumanEval, DROP, WMT, HellaSwag, long-context retrieval, a held-out translation set), each scored under **elicitation maximization** over $U$ = {0/5-shot} × {template, raw} × {CoT on/off}, identical $U$ for both arms.
- **Deciding number:** the slope $\partial \tau^\ast/\partial \kappa$ at fixed harmlessness, averaged over capabilities, at 8B versus 70B, with 3-seed CIs. If $\tau^\ast(\kappa)$ at 70B is within $\pm 0.5$ points of zero across $\kappa \le 4$ while 8B shows $\ge 2$ points, the scale-dependence claim is confirmed under a fair protocol for the first time. If both arms show the same slope, the "tax" is compute-and-format, not alignment.

## 9. Key References

- **[Foundational]** Amanda Askell, Yuntao Bai, Anna Chen, et al. *A General Language Assistant as a Laboratory for Alignment.* arXiv preprint, 2021. — arXiv:2112.00861
- **[Foundational]** Long Ouyang, Jeff Wu, Xu Jiang, et al. *Training language models to follow instructions with human feedback.* NeurIPS, 2022. — arXiv:2203.02155
- **[Foundational]** Yuntao Bai, Andy Jones, Kamal Ndousse, et al. *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.* arXiv preprint, 2022. — arXiv:2204.05862
- **[SOTA]** Leo Gao, John Schulman, Jacob Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML, 2023. — arXiv:2210.10760
- **[SOTA]** Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, et al. *Understanding the Effects of RLHF on LLM Generalisation and Diversity.* ICLR, 2024. — arXiv:2310.06452
- **[SOTA]** Yong Lin, Hangyu Lin, Wei Xiong, et al. *Mitigating the Alignment Tax of RLHF.* EMNLP, 2024. — arXiv:2309.06256
- **[Related]** Xiangyu Qi, Yi Zeng, Tinghao Xie, et al. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR, 2024. — arXiv:2310.03693
- **[Related]** Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, et al. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR, 2025. — arXiv:2406.05946
- **[Related]** Tiansheng Huang, Sihao Hu, Fatih Ilhan, et al. *Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable.* arXiv preprint, 2025.
- **[Related]** Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al. *Constitutional AI: Harmlessness from AI Feedback.* arXiv preprint, 2022. — arXiv:2212.08073

## 10. Worked Example

Take an 8B base model and its RLHF'd chat sibling, and measure the "MMLU tax" three ways.

| Protocol | Base | Aligned | $\hat\tau$ |
|---|---|---|---|
| A: 5-shot log-likelihood, no template (base-native) | 66.0 | 62.1 | **+3.9** |
| B: 0-shot generative, chat template (chat-native) | 41.5 | 65.8 | **−24.3** |
| C: elicitation-max over $U$ (best of A, B, CoT, 25-shot) | 67.2 | 66.4 | **+0.8** |

Same two checkpoints. The reported tax swings from $+3.9$ to $-24.3$ — a 28-point range — purely from harness choice. Protocol A is the standard base-model harness and Protocol B the standard chat harness, and papers routinely pick one per arm.

Now add the missing control. Train the format-matched, compute-matched arm: same SFT data and steps, preference term zeroed. Suppose it scores $66.9$ under Protocol C. Then the tax attributable to the *preference objective* is

$$\tau^\ast_{\text{MMLU}} = 66.9 - 66.4 = 0.5 \pm 0.4 \ \text{points},$$

against a binomial standard error of $0.4$ points on $n=14{,}042$ items — statistically indistinguishable from zero at 1 seed. Detecting a 0.5-point effect at $p<0.05$ needs roughly 3 seeds per arm, and only if seed variance is comparable to item variance, which nobody has published.

The obstruction is visible: the *headline* number (3.9) is 8× the number that survives a fair control, and the number that survives is smaller than the noise floor of a single run. Any claim about the alignment tax that does not report the harness, the control arm, $\kappa$, and seed variance is reporting harness choice, not alignment.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*