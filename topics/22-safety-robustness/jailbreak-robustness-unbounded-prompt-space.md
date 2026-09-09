---
id: 22-safety-robustness/jailbreak-robustness-unbounded-prompt-space
title: "Jailbreak Robustness Under Unbounded Prompt Space"
topic: 22-safety-robustness
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Jailbreak Robustness Under Unbounded Prompt Space

> **Topic:** Safety & Robustness · **ID:** `22-safety-robustness/jailbreak-robustness-unbounded-prompt-space` · **Status:** open

## 1. Problem Statement

A safety-trained language model refuses some requests. A **jailbreak** is any input that induces the model to comply anyway. Unlike an $\ell_p$-bounded image perturbation, the attacker's search space is the set of all token strings — unbounded in length, discrete, and with no metric that makes "small perturbation" meaningful. The question: *can a deployed system be made robust against an adversary who may submit arbitrary strings, and can that robustness be measured or certified rather than merely not-yet-broken?*

Three variants, of very different difficulty:

- **Measurement.** Given model $M$ and defense $D$, estimate the probability that an adversary with budget $B$ (queries, compute, human hours) elicits a target behavior. Open because attack success rate (ASR) against a *fixed* attack suite is an upper-bounded proxy for a supremum over an infinite set.
- **Method.** Build $D$ such that no adversary at budget $B$ succeeds above rate $\epsilon$, with acceptable over-refusal and latency cost. Partially achieved for narrow harm categories with external classifiers; unachieved for the base model itself.
- **Theory.** Prove *any* nontrivial lower bound on worst-case robustness for a transformer over $\mathcal{V}^*$, or prove no such bound exists. Essentially untouched; the only certificates are for artificially restricted attack classes.

Solving it means: a defense whose reported robustness is a bound, not a benchmark number, and that survives an adaptive attacker told the defense.

## 2. Formal Setting

Vocabulary $\mathcal{V}$, $|\mathcal{V}| \approx 1.3\times10^5$ for current tokenizers. Prompt space $\mathcal{X} = \mathcal{V}^*$. Model $M: \mathcal{X} \to \Delta(\mathcal{V}^*)$, sampled at temperature $T$.

**Harm judge.** $J: \mathcal{X}\times\mathcal{V}^* \to \{0,1\}$, measured in practice as an LLM classifier (HarmBench's fine-tuned Llama-2-13B judge) or a human panel. $J$ is not ground truth; inter-annotator agreement on borderline completions runs 0.6–0.8 Cohen's $\kappa$, so every ASR below ~10% is partly judge noise.

**Behavior-conditioned ASR.** For target behavior $b$ (e.g. "synthesis route for VX"), attack $\mathcal{A}$ with query budget $B$:

$$\mathrm{ASR}(M,\mathcal{A},b,B) \;=\; \Pr\big[\,\exists\, i \le B:\; J(x_i, y_i) = 1\,\big],\quad x_i \sim \mathcal{A}(M, b, \text{history}_{<i}),\ y_i \sim M(x_i).$$

Measured as a Monte Carlo estimate over behaviors and seeds; a 100-behavior suite at 5 seeds gives $\pm 4$pp at 95% confidence near 50%.

**Worst-case robustness** — the quantity actually named by "jailbreak robustness":

$$R(M, b) \;=\; 1 - \sup_{x \in \mathcal{X}} \Pr_{y\sim M(x)}\big[J(x,y)=1\big].$$

This supremum is over a countably infinite set and is **not estimable from below by any finite attack suite**. Every published number is $\hat{R} = 1 - \max_{\mathcal{A}\in \text{suite}} \widehat{\mathrm{ASR}}$, an *upper* bound on true robustness that ratchets down as attacks improve.

**Budget-indexed curve.** The empirically useful object is $\mathrm{ASR}(B)$ as a function of sampling budget. Best-of-$N$ jailbreaking (Hughes et al., 2024) fits $-\log(1-\mathrm{ASR}) \propto B^{\alpha}$ over four orders of magnitude in $B$, i.e. ASR $\to 1$ for any $\alpha>0$. Robustness is then not a number but the pair $(\alpha, \text{intercept})$.

**Over-refusal cost.** $\rho = \Pr[\text{refuse} \mid x \sim \mathcal{D}_{\text{benign}}]$, measured on XSTest/WildJailbreak-benign. Any defense must be scored on $(\hat{R}, \rho, \text{latency}, \text{FLOP overhead})$ jointly.

**Assumptions known to be violated.** (i) $J$ is accurate — violated; StrongREJECT showed judges systematically overcount success on low-quality outputs. (ii) Behaviors are i.i.d. draws from a harm distribution — violated; the suites are hand-curated and correlated. (iii) The attacker cannot modify weights or see logits — violated for open-weight releases, where fine-tuning removes alignment for <$0.20 of compute. (iv) The defense is fixed while the attack adapts — realistic, and the reason static evaluations mislead.

## 3. State of the Art

**Empirical attack SOTA (established).**
- **GCG** (Zou et al., 2023): greedy-coordinate gradient suffix search; 88% ASR on Vicuna-7B, 84% transfer to GPT-3.5. Requires white-box gradients; suffixes are high-perplexity and filterable.
- **PAIR** (Chao et al., 2023) / **TAP** (Mehrotra et al., 2023): black-box LLM-as-attacker; jailbreaks in ~20 queries, fluent prompts that perplexity filters miss.
- **Simple adaptive attacks** (Andriushchenko et al., 2024): 100% ASR on AdvBench against Llama-2-7B-chat, GPT-3.5/4, Claude, Gemini, using hand-designed templates plus random search over a suffix. The headline finding is that per-model adaptivity, not attack sophistication, is what wins.
- **Best-of-$N$** (Hughes et al., 2024): pure input augmentation (shuffling, capitalization, audio/image variants), 89% on GPT-4o and 78% on Claude 3.5 Sonnet at $N=10{,}000$, with power-law scaling.
- **Many-shot jailbreaking** (Anil et al., NeurIPS 2024): ASR rises as a power law in the number of in-context faux-dialogue demonstrations, exploiting long context windows.

**Defense SOTA.**
- **Constitutional Classifiers** (Sharma et al., Anthropic, 2025): input/output classifiers trained on constitution-generated synthetic data. Reported: zero universal jailbreaks found in >3,000 hours of red-teaming across 183 participants; automated ASR 4.4% vs 86% undefended; +0.38pp production refusal rate; +23.7% inference compute. **Established** as a red-team result at large human scale; **not** a bound — it is a statement about what a specific bounty population found in a specific window.
- **Circuit breakers / RR** (Zou et al., NeurIPS 2024): representation rerouting cuts HarmBench ASR roughly an order of magnitude at near-zero capability cost. Independently reproduced in part; adaptive attacks tuned against the representation objective reduce the gap, and the full adaptive picture is **claimed but unablated**.
- **Certified defenses**: *erase-and-check* (Kumar et al., 2023) certifies against adversarial suffixes up to $\sim20$ tokens by checking all erasure subsets — cost grows combinatorially and the guarantee covers only suffix insertion, not the attack class anyone uses. **SmoothLLM** (Robey et al., 2023) gives empirical, not certified, reduction.

Benchmark-only numbers: most ASR figures on AdvBench are **not** capability-checked; StrongREJECT (Souly et al., 2024) showed several published attacks trade harmfulness for coherence, so their ASR overstates real uplift.

## 4. What Is Known

- No published defense survives an attacker who knows it. Andriushchenko et al. broke every deployed frontier model in 2024 with per-model prompt templates plus random search — 100% on a 50-behavior AdvBench subset.
- Attack success scales with sampling budget as a power law over $10^0$–$10^4$ samples (Hughes et al., 2024), for text, vision and audio inputs alike. Extrapolation implies no finite-ASR plateau within measured range.
- Safety alignment is shallow: Qi et al. (ICLR 2025) show refusal behavior in Llama-2-7B-chat and Gemma is concentrated in the first ~5 generated tokens; forcing an affirmative prefix collapses refusal, and deepening alignment over more token positions measurably raises resistance to fine-tuning attacks.
- Fine-tuning removes alignment cheaply: Qi et al. (ICLR 2024) reported GPT-3.5 safety guardrails removed with 10 adversarial examples at ~$0.20 API cost.
- Classifier-based external defenses substantially raise attacker cost without large refusal regressions (Anthropic 2025: +0.38pp refusals), at ~24% compute overhead.
- Judge disagreement is a real error floor: HarmBench and StrongREJECT disagree on a nontrivial fraction of borderline completions, so single-digit ASRs are not reliably distinguishable from zero.

## 5. What Is Not Known

- **Theoretically open.** Whether any nontrivial lower bound on $R(M,b)$ exists for a transformer over $\mathcal{V}^*$. No impossibility theorem and no positive certificate for the unrestricted class. Related open question: does universal jailbreakability follow from the model's own generality (an in-context-learning argument), which would make robustness-by-training impossible in principle?
- **Empirically open.** Whether the best-of-$N$ power law holds to $B = 10^6$–$10^8$ against a classifier-defended stack. Runnable today; nobody has published it at that scale because it costs real inference money.
- **Empirically open.** Whether defense-in-depth composes: does stacking $k$ independent defenses multiply attacker cost, or does one universal prompt defeat all $k$ because they share training data?
- **Methodologically blocked.** "Robustness" itself. With no metric on $\mathcal{X}$, there is no perturbation budget, so there is no analogue of certified $\ell_\infty$ radius. Until someone defines the attack class formally, every claim is "not broken yet by these people."
- **Methodologically blocked.** Marginal-uplift measurement — how much real-world capability a jailbreak confers over a search engine. Without it, ASR counts refusals overturned, not harm caused.

## 6. Why It Is Hard

The core obstruction is **an evaluation that does not measure the thing it names**, compounded by **absent ground truth**.

$R(M,b)$ is a supremum over $|\mathcal{V}|^L$ strings — $\approx 10^{101}$ at $L=20$ tokens with a 128k vocabulary. No search, and no finite red-team, samples a meaningful fraction. So the reported quantity is $\max$ over a suite, which is monotonically non-increasing in the community's attack ingenuity and cannot certify anything. A defense that scores 0% ASR and one that is genuinely robust are observationally identical until someone finds the attack.

Second obstruction: the harm judge is itself a model with 0.6–0.8 $\kappa$ against humans. Defenses are now pushed into the single-digit-ASR regime, where measured differences are inside judge noise. Third: adaptivity is unbudgeted — reviewers cannot check whether an author's adaptive attack was tried hard enough, so negative results are unfalsifiable. Fourth: discreteness kills the tools. Randomized smoothing needs a noise distribution respecting a metric; token space has none that corresponds to semantic distance.

## 7. Current Research (as of 2026)

- **Classifier-guard stacks with public bug bounties** — Anthropic's constitutional classifiers line, extended to output streaming and multimodal inputs *(frontier — verify current deployment details)*.
- **Representation-level defenses** — circuit breakers, latent adversarial training (Sheshadri et al., 2024), and probing-based monitors; the open question is whether they generalize off-distribution or merely move the decision boundary.
- **Deep alignment / tamper resistance** — extending refusal supervision past the first tokens (Qi et al.), and tamper-resistant safeguards for open weights (Tamirisa et al., 2024). Fine-tuning attacks still win at modest budgets *(frontier — verify)*.
- **Instruction hierarchy** (Wallace et al., OpenAI, 2024) — treating jailbreaks as privilege escalation and training explicit trust levels. Reframes the problem as access control, which admits a cleaner threat model.
- **Attack-cost economics** — reporting dollars-per-successful-jailbreak rather than ASR. The most promising direction because it is measurable and does not pretend to be a bound.

## 8. Concrete Next Experiment

**Question:** does defense-in-depth change the *exponent* of attacker scaling, or only the intercept?

**Scale.** Fix 100 HarmBench behaviors. Four arms: (A) undefended open-weight 70B chat model; (B) + input/output constitutional-style classifier; (C) + circuit-breaker training; (D) B+C stacked. Run best-of-$N$ augmentation attacks at $N \in \{10^1, 10^2, 10^3, 10^4, 10^5\}$ per behavior. Total $\approx 5\times10^7$ generations at ~200 tokens: roughly $10^{10}$ output tokens, order $10^4$–$10^5$ GPU-hours on 70B inference — one week on 512 H100s.

**Control arm.** Arm A run at identical $N$ with identical augmentation seeds, plus a *judge control*: 2,000 completions dual-scored by the HarmBench judge, StrongREJECT, and 3 humans, to bound judge error separately per arm.

**Deciding number.** Fit $-\log(1-\mathrm{ASR}) = c B^{\alpha}$ per arm. The decision is $\alpha_D - \alpha_A$. If $\alpha_D \approx \alpha_A$ within CI (defenses shift $c$ only), then stacked defenses buy a constant factor in attacker cost and every ASR-at-fixed-$N$ table in the literature is reporting an intercept while calling it robustness. If $\alpha_D < \alpha_A - 0.05$ with non-overlapping bootstrap CIs, defense-in-depth genuinely bends the curve, and the research target becomes maximizing $-\Delta\alpha$ per unit inference cost.

## 9. Key References

- **[Foundational]** Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* 2023. — arXiv:2307.15043
- **[Foundational]** Alexander Wei, Nika Haghtalab, Jacob Steinhardt. *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS 2023. — arXiv:2307.02483
- **[Foundational]** Nicholas Carlini et al. *Are Aligned Neural Networks Adversarially Aligned?* NeurIPS 2023. — arXiv:2306.15447
- **[SOTA attack]** Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion. *Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks.* ICLR 2025. — arXiv:2404.02151
- **[SOTA attack]** John Hughes et al. *Best-of-N Jailbreaking.* 2024. — arXiv:2412.03556
- **[SOTA attack]** Cem Anil et al. *Many-Shot Jailbreaking.* NeurIPS 2024.
- **[SOTA defense]** Mrinank Sharma et al. *Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming.* Anthropic, 2025. — arXiv:2501.18837
- **[SOTA defense]** Andy Zou et al. *Improving Alignment and Robustness with Circuit Breakers.* NeurIPS 2024. — arXiv:2406.04313
- **[Certification]** Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, Hima Lakkaraju. *Certifying LLM Safety against Adversarial Prompting.* 2023. — arXiv:2309.02705
- **[Evaluation]** Mantas Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML 2024. — arXiv:2402.04249
- **[Evaluation]** Alexandra Souly et al. *A StrongREJECT for Empty Jailbreaks.* NeurIPS 2024 Datasets & Benchmarks. — arXiv:2402.10260
- **[Mechanism]** Xiangyu Qi et al. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR 2025. — arXiv:2406.05946
- **[Mechanism]** Xiangyu Qi et al. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR 2024. — arXiv:2310.03693
- **[Threat model]** Eric Wallace et al. *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions.* 2024. — arXiv:2404.13208
- **[Baseline]** Neel Jain et al. *Baseline Defenses for Adversarial Attacks Against Aligned Language Models.* 2023. — arXiv:2309.00614

## 10. Worked Example

Take a defended model reporting **4.4% automated ASR** and **zero universal jailbreaks in 3,000 human red-team hours** — the constitutional-classifier figures. Ask what that licenses.

Model the red team as $H$ independent Bernoulli trials at per-attempt success $p$. Zero successes in $H$ trials gives the rule-of-three 95% upper bound $p \le 3/H$. At an optimistic 1 attempt per hour, $H=3000$, so $p \le 10^{-3}$. Now put that against deployment: a system handling $10^7$ requests/day faces an expected $10^7 \times 10^{-3} = 10^4$ successful universal jailbreaks per day if adversarial traffic matched red-team quality. Even at $10^{-4}$ of traffic being adversarial, that is ~1 success per day. **The strongest human red-team result yet published is four to seven orders of magnitude short of the per-query rate deployment needs.**

Now the second half. The 4.4% automated ASR was measured against a fixed attack suite at some fixed $N$. Fit the best-of-$N$ law $-\log(1-\mathrm{ASR}) = cB^{\alpha}$ with the published Claude 3.5 Sonnet point (78% at $N=10^4$, so $c\,(10^4)^{\alpha} = 1.51$) and a plausible $\alpha = 0.25$: $c = 0.151$. Suppose a defense cuts $c$ by $30\times$ to $0.005$ and leaves $\alpha$ untouched. Required budget for the same 78%: $B = (1.51/0.005)^{4} \approx 8\times10^9$ samples — impressive, and at $10^{-5}$ USD per short query, about $80{,}000$. A motivated actor pays it.

The obstruction is visible in the arithmetic: both headline numbers — the red-team hours and the ASR — are *intercept* measurements. Neither constrains $\alpha$, and $\alpha$ is what decides whether robustness survives contact with a well-funded adversary. Nothing in the current evaluation regime reports $\alpha$, which is exactly why §8 asks for it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*