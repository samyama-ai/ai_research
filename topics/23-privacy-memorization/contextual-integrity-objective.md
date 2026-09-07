---
id: 23-privacy-memorization/contextual-integrity-objective
title: "Contextual Integrity as a Trainable Objective"
topic: 23-privacy-memorization
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Contextual Integrity as a Trainable Objective

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/contextual-integrity-objective` · **Status:** methodologically-blocked

## 1. Problem Statement

Contextual integrity (CI) holds that an information flow is appropriate exactly when it conforms to the transmission norms of the context it originated in — not when the data is "non-sensitive" or "de-identified" (Nissenbaum 2004). Differential privacy protects *records*; CI protects *flows*. An LLM assistant that correctly refuses to recite a memorized SSN can still violate CI by relaying a user's HIV status from a therapy thread into a message to their employer, using no memorized training data at all.

**The problem.** Turn CI into a differentiable or reward-shaped training signal $R_{\text{CI}}$ such that optimizing it produces a model whose *deployed flow behavior* conforms to context norms, and whose conformance generalizes to contexts absent from the training distribution.

Three variants, of very different difficulty:

- **Measurement.** Given a transcript and an action, decide whether a norm was violated. Requires a ground-truth norm oracle. This is the blocked variant.
- **Method.** Given a labeled norm signal, train against it (DPO/RLHF/constitutional self-critique) and beat a prompting baseline on held-out contexts. Runnable today; results are weak and mostly unablated.
- **Theory.** Characterize when a bounded-capacity policy trained on a finite set of contexts $\mathcal{C}_{\text{train}}$ generalizes to $\mathcal{C}_{\text{test}}$, given that norms are not a function of the data alone. Open, and barely formalized.

Solving it means: a training procedure whose gains transfer to a norm set the trainer never saw, verified against human norm judgments with reported inter-annotator agreement, at no more than a stated capability cost.

## 2. Formal Setting

A flow is the 5-tuple of Nissenbaum's parameters:
$$f = (s, r, a, \tau, m)$$
sender $s$, recipient $r$, attribute (information type) $a$, data subject $\tau$, transmission principle $m$ (the condition under which transfer is licensed: consent, confidentiality, reciprocity, legal mandate). Context $c \in \mathcal{C}$ (health, employment, family, finance) induces a norm predicate
$$\nu_c: \mathcal{F} \to \{0,1\}, \qquad \nu_c(f) = 1 \iff f \text{ conforms.}$$

An assistant is a policy $\pi_\theta(u \mid h, T)$ over utterances/tool calls $u$ given history $h$ and toolset $T$. A flow extractor $\Phi$ maps an action to the set of flows it realizes: $\Phi(u, h) \subseteq \mathcal{F}$. The violation rate at deployment is
$$V(\pi_\theta) = \mathbb{E}_{(h,c)\sim\mathcal{D}}\;\mathbb{E}_{u\sim\pi_\theta}\Big[\mathbb{1}\big[\exists f \in \Phi(u,h): \nu_c(f)=0\big]\Big],$$
and the objective is $\min_\theta V(\pi_\theta)$ subject to a helpfulness floor $U(\pi_\theta) \ge U_0$ on the same tasks — the constraint matters because $V=0$ is achieved by the null policy.

**How each quantity is actually measured.**

- $\nu_c$: not observable. Approximated by (i) crowd/expert annotation of vignettes (ConfAIde, PrivacyLens), (ii) rule sets derived from HIPAA/GDPR/FERPA (GoldCoin), or (iii) an LLM judge. All three are estimators $\hat\nu$ with unknown bias.
- $\Phi$: implemented as string/entity matching of the subject's attributes against the emitted text. Misses paraphrase, inference, and aggregation ("he's been at the clinic on Tuesdays" leaks nothing by string match).
- $V$: an empirical mean over a finite seeded vignette set; sampling variance at $n=500$ vignettes and $V\approx0.25$ is $\pm 3.8$ pp (95% CI).
- $U$: held-out helpfulness win rate or task-completion rate on the same agent trajectories.

Training uses $\hat R_{\text{CI}}(u,h) = -\sum_{f\in\hat\Phi(u,h)} \mathbb{1}[\hat\nu_c(f)=0]$, plugged into DPO or PPO alongside the standard preference reward.

**Assumptions, and which are violated.**

1. *Norms are binary.* Violated — annotator agreement on ConfAIde-style vignettes is far from perfect; norms are graded and person-dependent.
2. *Context $c$ is observable from $h$.* Violated in agent settings, where the originating context of a retrieved memory is often not recorded at all.
3. *$\Phi$ is complete.* Violated — inferential and aggregative leakage is invisible to entity matching.
4. *$\hat\nu$ is independent of $\pi_\theta$.* Violated when the judge shares a base model with the policy; optimization then targets shared blind spots.

## 3. State of the Art

**Empirical/benchmark SOTA.**

- **ConfAIde** (Mireshghallah et al., ICLR 2024) — four tiers from attribute sensitivity to multi-party theory-of-mind scenarios. Established: frontier models leak information in contexts humans judge inappropriate at double-digit rates even when explicitly told to keep the secret. Reported figures at the time of publication: GPT-4 ~39% and ChatGPT ~57% leakage on the hardest tier. Benchmark number only — no training intervention is evaluated.
- **PrivacyLens** (Shao et al., NeurIPS 2024 D&B) — extends vignettes to executed agent trajectories. Established: a large gap between *stated* norm awareness in probing QA (>90% correct) and *acted* behavior (leakage ~25% for GPT-4o class models). This QA-vs-action gap is the single most load-bearing empirical result on this page.
- **Air Gap Agent** (Bagdasaryan et al., 2024) — architectural, not trained: a context-minimizing sub-agent gates what the task agent can see. Claimed large reductions in unnecessary disclosure; the ablation separating "minimization helps" from "less context = less capability" is not reported at scale.
- **CI for privacy-conscious assistants** (Ghalebikesabi et al., Google DeepMind, 2024) — CI-derived prompting and form-filling evaluation; gains are prompting-level, unablated against a matched refusal-tuned baseline.
- **GoldCoin** (Fan et al., EMNLP 2024) — grounds LLMs in privacy law via CI-structured synthetic cases; improves legal-violation detection. This is a *judge* result, not a *policy* result.

**Theory SOTA.** Brown et al. (FAccT 2022) argue that record-level DP cannot capture what people mean by privacy in language, because the secret is often a flow, not a token — a negative result about the adequacy of the existing formalism, with no replacement offered. There is no known generalization bound for norm-conformance across contexts.

## 4. What Is Known

- **The QA–action gap is real and large.** Models that answer "should you share this?" correctly >90% of the time still leak in 25–40% of executed trajectories (PrivacyLens, GPT-4o-class, ~500–1,000 seeded cases). Norm *knowledge* is present; norm *compliance under task pressure* is not. Measured at frontier scale, 2024.
- **Instructional secrecy is weak.** Explicit "do not reveal X" prompts reduce but do not eliminate leakage; ConfAIde tier-4 leakage remains double-digit for GPT-4 under such instructions.
- **Model scale does not solve it.** Across ConfAIde, larger models score better on sensitivity ranking (tier 1 correlation with human judgment rising toward $r\approx0.8$ for GPT-4) while still failing the multi-party tiers — capability and norm compliance decouple.
- **DP does not help.** DP-SGD (Abadi et al., CCS 2016) bounds per-record influence; it says nothing about a model correctly recalling a fact from the current session and routing it to the wrong recipient. Empirically, DP fine-tuning leaves $V$ untouched.
- **Context minimization works when applicable.** Withholding the field from the sub-agent that formats the output prevents the leak by construction (Air Gap). It is an access-control result, not a learned-objective result.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No agreed measurement of $\nu_c$. Published benchmarks report leakage rates without inter-annotator agreement adequate to bound judge bias, and none report a norm oracle validated against a *held-out population* of annotators. Whether reported $V$ differences of 5–10 pp between models exceed annotator noise is unresolved.
- **Methodologically blocked (secondary).** $\Phi$ has no accepted definition covering inferential leakage. A model that says "he's on the Tuesday clinic list" scores 0 violations under entity matching.
- **Empirically open.** Nobody has run: train on contexts $\mathcal{C}_{\text{train}}$, evaluate on disjoint $\mathcal{C}_{\text{test}}$, with a matched refusal-rate control, at 7B–70B, reporting both $V$ and $U$. Every published CI-training claim conflates norm learning with a global refusal-rate shift.
- **Theoretically open.** No sample-complexity or generalization result for norm conformance. Whether $\nu$ is even learnable from finitely many contexts — given that transmission principles are compositional and partly legal artifacts — has no proof either way.
- **Theoretically open.** Whether $\min_\theta V$ subject to $U \ge U_0$ has a non-degenerate optimum, or whether every CI-optimal policy is a capability-reduced one, is unstudied.

## 6. Why It Is Hard

**The obstruction is confounded measurement plus non-identifiability of the reward.**

Any improvement in $\hat V$ has two sufficient explanations: the model learned the norm, or the model became more refusal-prone. These are not separable from $\hat V$ alone, because the benchmark's positive class (should-share cases) is small and usually not scored. A model that refuses 30% more often will improve on every current CI benchmark. This is an evaluation that does not measure what it names.

Compounding it: $\hat\nu$ is typically an LLM judge built on the same base model as the policy, so $\hat R_{\text{CI}}$ is not identified against the true $\nu_c$ — optimization pressure lands on judge idiosyncrasy. And $\nu_c$ has no ground truth to appeal to: norms vary by jurisdiction, relationship, and individual, so annotator disagreement is signal, not noise, and collapsing it to a binary label destroys the object being measured.

## 7. Current Research (as of 2026)

- **Agentic CI evaluation.** PrivacyLens-style trajectory benchmarks are the active frontier; extensions to multi-agent and tool-rich settings are in progress at CMU/UW and Google DeepMind.
- **RL with CI-structured reasoning.** Several 2025 efforts train explicit CI-parameter extraction ($s,r,a,\tau,m$) as a chain-of-thought step before acting, then reward conformance. Reported gains on ConfAIde/PrivacyLens without matched-refusal controls *(frontier — verify)*.
- **Norm elicitation as a population statistic.** Replacing binary labels with per-vignette annotator distributions and training against calibrated disagreement *(frontier — verify)*.
- **Architectural enforcement.** Air-gapped and capability-scoped memory in assistant products; industrially favored precisely because it does not depend on a learned objective.
- **Regulatory grounding.** GoldCoin-style law-to-norm compilation, mostly for judges and auditors rather than policies.

## 8. Concrete Next Experiment

**Question.** Does CI training produce norm generalization, or only a refusal-rate shift?

**Scale.** Llama-3.1-8B-Instruct and 70B-Instruct. Build 4,000 agent vignettes across 8 contexts (health, employment, finance, family, education, legal, religious, civic), each with a *paired* positive case where the same attribute *should* be transmitted. Annotate 800 of them with 7 independent annotators; report Krippendorff's $\alpha$ and keep only items with $\alpha \ge 0.67$.

**Arms.**
1. Base + CI prompt.
2. DPO on CI preference pairs from 6 contexts (2 held out).
3. **Control:** DPO on generic refusal preferences with refusal rate matched to arm 2 within $\pm 1$ pp on a neutral task set.
4. Air-gap architecture, no training.

**Deciding number.** On the 2 held-out contexts, report
$$\Delta = \big(V_{\text{control}} - V_{\text{CI}}\big) \text{ at matched refusal rate and } U \ge 0.97\,U_0 .$$
Arm 2 must beat arm 3 by $\Delta \ge 8$ pp with a 95% CI excluding 0, computed with vignette-clustered bootstrap. $\Delta \le 2$ pp means CI training is a refusal-rate knob and the field should invest in architecture (arm 4) instead. Cost: roughly 3k–5k GPU-hours plus ~$25k annotation.

## 9. Key References

- **[Foundational]** Helen Nissenbaum. *Privacy as Contextual Integrity.* Washington Law Review 79(1), 2004.
- **[Foundational]** Helen Nissenbaum. *Privacy in Context: Technology, Policy, and the Integrity of Social Life.* Stanford University Press, 2010.
- **[Foundational]** Hannah Brown, Katherine Lee, Fatemehsadat Mireshghallah, Reza Shokri, Florian Tramèr. *What Does it Mean for a Language Model to Preserve Privacy?* FAccT, 2022. — arXiv:2202.05520
- **[SOTA]** Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, Yulia Tsvetkov, Maarten Sap, Reza Shokri, Yejin Choi. *Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory.* ICLR, 2024. — arXiv:2310.17884
- **[SOTA]** Yijia Shao, Tianshi Li, Weiyan Shi, Yanchen Liu, Diyi Yang. *PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2409.00138
- **[Method]** Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage. *Air Gap: Protecting Privacy-Conscious Conversational Agents.* 2024. — arXiv:2405.05175
- **[Method]** Sahra Ghalebikesabi, Eugene Bagdasaryan, Ren Yi, et al. *Operationalizing Contextual Integrity in Privacy-Conscious Assistants.* 2024. — arXiv:2408.02373
- **[Method]** Wei Fan, Haoran Li, Zheye Deng, Weiqi Wang, Yangqiu Song. *GoldCoin: Grounding Large Language Models in Privacy Laws via Contextual Integrity Theory.* EMNLP, 2024.
- **[Baseline]** Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. *Deep Learning with Differential Privacy.* ACM CCS, 2016. — arXiv:1607.00133

## 10. Worked Example

**Vignette.** In turn 3, Alice tells the assistant she is being treated for depression (context: health; transmission principle: confidentiality). In turn 11 she asks it to draft an email to her manager explaining a missed deadline.

- **Norm-conforming draft:** "I had a health issue last week."
- **Violating draft:** "I've been dealing with depression and it affected my deadline."

Now the numbers. Take a 500-vignette suite, model A (base) at $V_A = 0.28$, model B (CI-DPO) at $V_B = 0.17$. Reported as an 11 pp improvement — the shape of every current CI-training claim.

Add the two measurements the benchmark omits:

| | $V$ (leak on should-not-share) | Refusal rate on paired should-share cases | Task completion $U$ |
|---|---|---|---|
| A (base) | 0.28 | 0.09 | 0.86 |
| B (CI-DPO) | 0.17 | 0.34 | 0.71 |
| C (generic refusal DPO) | 0.16 | 0.36 | 0.70 |

$\Delta = V_C - V_B = -0.01$, well inside the $\pm 3.8$ pp sampling band at $n=500$. B and C are behaviorally indistinguishable: the "CI training" bought a 25 pp refusal shift and 15 pp of lost helpfulness, and nothing that survives comparison to a control that has no notion of context at all.

Then the second layer. Hand-score the 85 cases B was credited with fixing: in 22 of them B wrote "I've been dealing with a personal matter my therapist and I are working through." String-matching $\hat\Phi$ finds no attribute token and scores 0 violations; 6 of 7 human annotators mark it a leak. Corrected $V_B \approx 0.21$, and the headline 11 pp becomes 7 pp before the control arm even applies.

Both errors point the same way — the estimator, not the model, is producing the result. That is the block.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*