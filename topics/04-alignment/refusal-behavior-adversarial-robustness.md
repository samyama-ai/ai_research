---
id: 04-alignment/refusal-behavior-adversarial-robustness
title: "Adversarial Robustness of Refusal Behavior"
topic: 04-alignment
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adversarial Robustness of Refusal Behavior

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/refusal-behavior-adversarial-robustness` · **Status:** open

## 1. Problem Statement

An aligned language model is trained to refuse a set of requests (weapons synthesis, targeted harassment, CSAM, etc.). The problem: make that refusal hold under adversarial input, and *know* that it holds.

- **Input:** a policy $\pi_\theta$, a set of forbidden behaviors $\mathcal{D}$, and an adversary who may rewrite the prompt, prefill the assistant turn, stuff the context, or fine-tune the weights.
- **Output:** either a model whose refusal survives the adversary, or a certificate that it does.
- **Decision predicate:** does there exist an input in the adversary's budget that elicits a compliant, *useful* answer to $b \in \mathcal{D}$?

Three variants, different difficulty:

- **Measurement.** Estimate worst-case attack success rate (ASR). Currently only a lower bound is computable — every reported ASR is "the best attack we ran", never "the best attack that exists".
- **Method.** Train or wrap a model so measured ASR under strong adaptive attack is near zero without destroying helpfulness. Partial progress; no defense has survived unrestricted adaptive attack.
- **Theory.** Prove a nonvacuous upper bound on ASR over a combinatorially large discrete input space. Essentially untouched — certified defenses cover only tiny perturbation sets.

Solving it means: a defense with $\le 1\%$ ASR against an adversary given $10^4$ queries and full white-box access, at $\le 2$ points of capability loss, reproduced by an independent red team.

## 2. Formal Setting

Let $\mathcal{V}$ be the token vocabulary, $\pi_\theta: \mathcal{V}^* \to \Delta(\mathcal{V}^*)$ the model. A behavior $b$ is a natural-language specification of forbidden output (HarmBench and JailbreakBench both ship $b$ as a string plus a rubric).

**Adversary.** An attack set $\mathcal{A}(b) \subseteq \mathcal{V}^*$ — all inputs the adversary may submit while still "asking for $b$". Budget: query count $q$, token length $L$, access level $a \in \{\text{black-box}, \text{logits}, \text{gradients}, \text{weights}\}$.

**Judge.** $J: (b, y) \mapsto \{0,1\}$, measured in practice by an LLM classifier (HarmBench's fine-tuned Llama-2-13B judge; StrongREJECT's rubric scoring specificity and usefulness, not just non-refusal) or by human annotation on a subsample.

**Objective.**
$$\mathrm{ASR}_{\mathcal{A}}(\pi_\theta) \;=\; \mathbb{E}_{b\sim\mathcal{D}}\Big[\max_{x \in \mathcal{A}(b)}\; \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\, J(b,y)\Big]$$

The $\max$ is the whole problem. Any concrete attack $A$ gives $\widehat{\mathrm{ASR}}_A \le \mathrm{ASR}_\mathcal{A}$: **every published robustness number is a one-sided lower bound on vulnerability**, i.e. an upper bound on safety that nothing verifies.

**Refusal margin.** For a refusal-token set $R$ ("I cannot", "I'm sorry", …), the per-prompt margin
$$m(x) \;=\; \log \pi_\theta(R \mid x) \;-\; \log \pi_\theta(y^{+}_{1:k} \mid x)$$
where $y^+$ is a target-affirmative prefix ("Sure, here is"). GCG minimizes $-\log\pi_\theta(y^+\mid x)$ directly; $m(x)$ is measurable for any open-weight model and is the closest thing to a continuous robustness signal.

**Assumptions, and which are false in practice.**

| Assumption | Status |
|---|---|
| $\mathcal{D}$ is well specified — harm is a property of the output | **False.** Dual-use requests have no ground-truth label; annotator agreement on borderline items is far from 1. |
| $J$ is accurate | **Approximately false.** LLM judges score fluent-but-useless outputs as successes; StrongREJECT (Souly et al., NeurIPS 2024) showed several published jailbreaks lose most of their apparent effect under a usefulness-aware rubric. |
| $\mathcal{A}(b)$ is fixed and known | **False.** The attack set is open-ended natural language; new families (many-shot, prefill, best-of-$N$ augmentation, fine-tuning) keep appearing. |
| Weights are held by the defender | **False for open weights.** Fine-tuning removes refusal at negligible cost. |

## 3. State of the Art

**Attacks (established).**
- **GCG** — Zou et al. (2023), greedy coordinate gradient over a universal suffix: near-100% ASR on Vicuna-7B/13B on AdvBench (520 behaviors), with nontrivial transfer to closed models.
- **PAIR / TAP** — Chao et al. (2023), Mehrotra et al. (2023): attacker-LLM query loops, jailbreaks in ~20 queries black-box.
- **Adaptive attacks** — Andriushchenko, Croce, Flammarion (ICLR 2025): random search over a suffix plus logit access plus assistant prefilling reached 100% ASR on all 50 JailbreakBench behaviors across every model tested, including ones reported robust to GCG. This is the strongest single result in the area and its lesson is methodological: **defenses evaluated against fixed attacks report inflated robustness**.
- **Best-of-$N$** — Hughes et al. (2024): random augmentation resampling, ASR growing as a power law in $N$; high success on frontier models at $N \approx 10^4$.
- **Many-shot jailbreaking** — Anil et al. (NeurIPS 2024): ASR increases log-linearly with the number of in-context faux-dialogue demonstrations; long context windows are themselves an attack surface.
- **Fine-tuning** — Qi et al. (ICLR 2024): ~10 adversarial examples, under $0.20 of API spend, collapsed GPT-3.5 Turbo's refusal behavior.

**Defenses.**
- *Established as a real improvement:* **circuit breakers** (Zou et al., NeurIPS 2024) — representation rerouting on harmful-completion activations cuts HarmBench ASR by roughly an order of magnitude at small capability cost; **latent adversarial training** (Sheshadri et al., 2024) attacks the residual stream rather than tokens and improves robustness to held-out attacks.
- *Claimed but not fully ablated:* **constitutional classifiers** (Sharma et al., Anthropic, 2025) — thousands of hours of red teaming produced no universal jailbreak; the reported production cost was a small absolute increase in refusal rate and a double-digit percentage inference overhead. The result is strong evidence about *that* red-team distribution, not a bound.
- *Largely superseded:* perplexity filters (Jain et al., 2023) and SmoothLLM (Robey et al., 2023) defeat GCG-style gibberish suffixes but not fluent or prefill attacks.
- *Benchmark-number-only:* most "our defense achieves $X\%$ ASR" claims in the 2024–2026 literature are static-attack evaluations. Treat them as untested until re-attacked adaptively.

**Certification.** Kumar et al. (2023) "erase-and-check" certifies only against short adversarial suffixes ($\le 20$ tokens) at exponential-in-length cost. No nonvacuous certificate exists for unconstrained natural-language attacks.

## 4. What Is Known

- Refusal is **shallow**. Qi et al. (ICLR 2025) showed safety behavior in Llama-2 and Gemma is concentrated in the first few generated tokens; forcing an affirmative prefix of ~5 tokens flips most refusals. Deepening the safety objective over more token positions measurably improves robustness.
- Refusal is **low-rank**. Arditi et al. (NeurIPS 2024): a single residual-stream direction mediates refusal across 13 open chat models from 1.5B to 72B parameters; ablating it removes refusal, adding it induces refusal on harmless prompts. A one-dimensional safety mechanism is a one-dimensional attack surface.
- **Mismatched generalization** (Wei, Haghtalab, Steinhardt, NeurIPS 2023): capabilities generalize past the safety-training distribution (base64, low-resource languages, role-play), so failure modes are predictable from what safety data does *not* cover.
- **Attack success scales with attacker compute** as a power law in $N$ (best-of-$N$) and log-linearly in shot count (many-shot). No defense has been shown to break this scaling — only to shift its intercept.
- **Defender inference compute helps, partially.** Zaremba et al. (2025) found ASR falls with reasoning-time compute for several attack families, but not for attacks that corrupt the goal the model is reasoning about.
- **Evaluation is noisy.** HarmBench (ICML 2024) and JailbreakBench (NeurIPS 2024 D&B) disagree on ranking for some model/attack pairs, because judge and behavior set differ.

## 5. What Is Not Known

- **Theoretically open.** Whether any nonvacuous upper bound on $\mathrm{ASR}_\mathcal{A}$ is achievable for unconstrained natural-language $\mathcal{A}$. No impossibility theorem and no positive construction. Also open: whether refusal robustness and capability are in unavoidable tension, or whether observed tension is an artifact of current training objectives.
- **Empirically open.** Whether robustness improves with model scale at fixed safety-training recipe — runnable (train a matched safety recipe at 8B/70B/400B, attack identically) and, publicly, unrun. Whether circuit-breaker-style representation defenses survive an unrestricted adaptive attacker given $10^5$ queries and white-box gradients.
- **Methodologically blocked.** Worst-case ASR is not estimable, only lower-boundable. There is no accepted way to report "robustness" that is not "robustness against the attacks I happened to run". Harm labels for dual-use behaviors have no ground truth, so the target set $\mathcal{D}$ is itself undefined at its boundary.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the maximum over a discrete, unbounded, semantically open input space**, compounded by a judge that does not measure what it names.

1. **No inner maximizer.** Vision robustness has $\ell_p$ balls and PGD, a strong, standard, near-optimal attack. Language has no metric ball: the set of "prompts that ask for $b$" is defined semantically. So the inner $\max$ is approximated by heuristic search, and defenses can trivially defeat *the specific search used*, gradient masking by another name. Andriushchenko et al. is the empirical proof.
2. **Confounded measurement.** LLM judges reward non-refusal, not uplift. A model that emits confident nonsense scores as jailbroken; a model that refuses while leaking the answer in a "safety warning" scores as safe. StrongREJECT quantified this gap.
3. **Absent ground truth at the boundary.** For dual-use chemistry or security questions there is no label to be robust *to*, so ASR and over-refusal are measured on incompatible sets.
4. **Asymmetric cost.** Defender pays training compute once; attacker pays $10^4$ queries. Power-law attack scaling means each 10× attacker spend buys a fixed ASR increment, and defenses so far shift the constant, not the exponent.

## 7. Current Research (as of 2026)

- **Latent-space and representation defenses**: circuit breakers, latent adversarial training, refusal-direction hardening (Gray Swan, EleutherAI/MATS-affiliated groups, Anthropic).
- **Classifier sandwiches / defense-in-depth**: input–output constitutional classifiers with streaming, plus instruction-hierarchy training (Wallace et al., OpenAI, 2024). Deployment-favored because it decouples the safety layer from the capable model.
- **Deep safety alignment**: extending the safety objective beyond the first tokens (Qi et al., Princeton).
- **Robustness scaling laws**: whether ASR-vs-attacker-compute exponents change with model scale or defense class *(frontier — verify)*.
- **Tamper-resistant open weights**: making fine-tuning-based removal expensive (TAR and successors) — early results show attacks with more fine-tuning steps still recover the capability *(frontier — verify)*.
- **Adaptive-attack standardization**: pushing red teams and bug bounties toward a mandatory adaptive arm before any robustness claim.

## 8. Concrete Next Experiment

**Question:** does a leading representation-level defense reduce *worst-case* ASR, or only shift the attacker-compute constant?

- **Scale.** One open-weight base, Llama-3.1-8B-Instruct (cheap enough to run $10^5$ white-box attack queries per behavior; three seeds).
- **Arms.** (a) base model; (b) **control**: standard refusal SFT+DPO on the same safety data, matched token budget — this is the arm most papers omit; (c) circuit breakers; (d) latent adversarial training.
- **Attack ladder.** For each arm, run GCG, PAIR, prefill+random-search (Andriushchenko), and best-of-$N$ at $N \in \{10, 10^2, 10^3, 10^4\}$ on the 100 JailbreakBench/HarmBench behaviors, with StrongREJECT rubric scoring plus 200 human-labeled samples for judge calibration. Attacks must be re-tuned per arm (adaptive, not transferred).
- **Deciding number.** Fit $\mathrm{ASR}(N) = cN^{\alpha}$ per arm. Report $\hat\alpha_{\text{defense}} - \hat\alpha_{\text{control}}$ with bootstrap CI.
  - If the CI excludes 0 and $\hat\alpha$ drops (say from $0.35$ to $\le 0.20$), the defense changes the scaling and is a real advance.
  - If only $\hat c$ drops with $\hat\alpha$ unchanged, the defense buys the attacker's budget a constant factor — useful for deployment, not a solution.
- **Cost estimate.** ~4 arms × 4 attacks × 100 behaviors × up to $10^4$ queries $\approx 10^7$ generations at 8B; feasible on 8×H100 in single-digit GPU-weeks.

## 9. Key References

- **[Foundational]** Zou, Wang, Carlini, Nasr, Kolter, Fredrikson. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* 2023. — arXiv:2307.15043
- **[Foundational]** Wei, Haghtalab, Steinhardt. *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS 2023. — arXiv:2307.02483
- **[Foundational]** Carlini, Nasr, Choquette-Choo, Jagielski, Gao, Awadalla, Koh, Ippolito, Lee, Tramèr, Schmidt. *Are Aligned Neural Networks Adversarially Aligned?* NeurIPS 2023. — arXiv:2306.15447
- **[SOTA — attack]** Andriushchenko, Croce, Flammarion. *Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks.* ICLR 2025. — arXiv:2404.02151
- **[SOTA — attack]** Anil et al. *Many-shot Jailbreaking.* NeurIPS 2024.
- **[SOTA — attack]** Hughes et al. *Best-of-N Jailbreaking.* 2024. — arXiv:2412.03556
- **[SOTA — defense]** Zou, Phan, Wang, Duenas, Lin, Andriushchenko, Wang, Kolter, Fredrikson, Hendrycks. *Improving Alignment and Robustness with Circuit Breakers.* NeurIPS 2024. — arXiv:2406.04313
- **[SOTA — defense]** Sheshadri et al. *Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs.* 2024. — arXiv:2407.15549
- **[SOTA — defense]** Sharma et al. *Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming.* Anthropic, 2025. — arXiv:2501.18837
- **[Mechanism]** Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024. — arXiv:2406.11717
- **[Mechanism]** Qi, Panda, Wang, Chen, Xie, Ma, Mittal, Henderson. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep.* ICLR 2025. — arXiv:2406.05946
- **[Benchmark]** Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML 2024. — arXiv:2402.04249
- **[Benchmark]** Chao, Debenedetti, Robey, Andriushchenko, Croce, Sehwag, Dobriban, Flammarion, Pappas, Tramèr, Hassani, Wong. *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models.* NeurIPS 2024 Datasets & Benchmarks. — arXiv:2404.01318
- **[Measurement critique]** Souly et al. *A StrongREJECT for Empty Jailbreaks.* NeurIPS 2024. — arXiv:2402.10260
- **[Threat model]** Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson. *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!* ICLR 2024. — arXiv:2310.03693
- **[Certification]** Kumar, Agarwal, Srinivas, Feizi. *Certifying LLM Safety against Adversarial Prompting.* 2023. — arXiv:2309.02705

## 10. Worked Example

Take one HarmBench behavior $b$ and a model reported at 5% ASR.

Suppose a defended 8B model is evaluated on $n = 100$ behaviors against GCG and reports $\widehat{\mathrm{ASR}}_{\text{GCG}} = 0.05$. Binomial standard error is $\sqrt{0.05\cdot0.95/100} \approx 0.022$ — the headline number's own CI is roughly $[0.01, 0.09]$, before any judge error.

Now add the judge. If the classifier's false-positive rate on fluent-but-useless completions is $\varepsilon_+ = 0.10$ and its false-negative rate on genuinely useful completions is $\varepsilon_- = 0.15$, the observed rate relates to the true rate $p$ by
$$\hat p = p(1-\varepsilon_-) + (1-p)\varepsilon_+ \;\Rightarrow\; p = \frac{\hat p - \varepsilon_+}{1 - \varepsilon_- - \varepsilon_+}.$$
With $\hat p = 0.05$, $p = (0.05-0.10)/0.75 = -0.067$. **The estimator returns a negative probability.** At the 5% level the measurement carries no information: judge noise exceeds the signal.

Then add the adaptive arm. Take the same 100 behaviors and, instead of GCG's gibberish suffix, prefill the assistant turn with "Sure, here are the step-by-step instructions:" and run random search over a 25-token suffix for $10^3$ steps. Because refusal is concentrated in the first tokens (Qi et al.) and mediated by a low-rank direction (Arditi et al.), this class of attack has repeatedly taken models from single-digit reported ASR to near-total failure — that is precisely the Andriushchenko et al. result, 100% on all 50 JailbreakBench behaviors including models with published low ASR.

The obstruction is visible in both halves. The reported 5% is (i) statistically indistinguishable from judge noise, and (ii) an artifact of the attack chosen, not a property of the model. Any page, paper, or model card reporting a single ASR figure without a judge-calibration term and an adaptive-attack arm is reporting a number that does not constrain the quantity it names.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*