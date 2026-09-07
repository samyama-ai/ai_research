---
id: 21-factuality/adversarial-fabrication-elicitation
title: "Adversarial Prompts That Reliably Induce Fabrication"
topic: 21-factuality
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Adversarial Prompts That Reliably Induce Fabrication

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/adversarial-fabrication-elicitation` · **Status:** open

## 1. Problem Statement

Given a language model $M$ and a factual question the model answers correctly under a neutral prompt, does there exist a prompt transformation that makes $M$ assert a *specific* falsehood — with high confidence, fluent supporting detail, and no hedging — and can such transformations be found automatically, cheaply, and transferably?

Three variants, with different difficulty:

- **Measurement.** Define a *fabrication rate under attack* that is not gameable by trivial reframings. The attack must not smuggle the false premise into the prompt in a way that makes the model's answer arguably correct-in-context (role-play, counterfactual fiction, "assume X"). Distinguishing "the model was tricked into believing X" from "the model correctly complied with an instruction to write X" is the core measurement problem.
- **Method.** Build an optimizer that, given $(M, q, a^{\text{false}})$, returns a prompt $p$ such that $M(p)$ asserts $a^{\text{false}}$. This is jailbreak-style search retargeted from harmfulness to falsehood.
- **Theory.** Prove or refute: for models of a given capability class, adversarial fabrication is *unavoidable* — every model with non-trivial in-context updating admits prompts that flip a known-correct answer.

Solving it means: a reproducible attack suite with a stated success rate at a stated query budget, a defence measured against that suite, and a bound on the residual rate.

## 2. Formal Setting

Let $M: \mathcal{V}^* \to \Delta(\mathcal{V}^*)$ be an autoregressive model. Let $q$ be a question with verified answer $a^\star$ drawn from a gold set $\mathcal{D}$ (measured: human-annotated short-form QA with a single unambiguous referent, e.g. SimpleQA-style items with two independent annotators agreeing).

**Baseline correctness.** $c(q) = \mathbb{1}[\text{judge}(M(q_{\text{neutral}}), a^\star) = \text{correct}]$, where $\text{judge}$ is an LLM grader validated against human labels on a held-out sample; report grader–human agreement ($\kappa$) alongside every rate.

**Attack.** A transformation $T$ maps $q \mapsto p = T(q, a^{\text{false}})$ using query budget $B$ (number of forward passes to $M$). Attack success:

$$\mathrm{ASR}(T, B) = \frac{1}{|\mathcal{D}_c|}\sum_{q \in \mathcal{D}_c} \mathbb{1}\!\left[\text{assert}\big(M(T(q,a^{\text{false}}))\big) = a^{\text{false}}\right],\quad \mathcal{D}_c=\{q: c(q)=1\}$$

Conditioning on $\mathcal{D}_c$ matters: an attack that "succeeds" on questions the model never knew measures nothing.

**Assertion vs. compliance.** Require an *unhedged assertion*: no counterfactual framing token in the output, no refusal, and the claim survives a follow-up probe $q_{\text{recheck}}$ issued in a fresh context. Define the persistence-corrected rate

$$\mathrm{ASR}^{+} = \Pr\big[\text{assert}=a^{\text{false}} \ \wedge\ \text{assert}(M(q_{\text{recheck}}))=a^{\text{false}}\big].$$

**Confidence.** $\hat{p} = $ verbalized probability, or $\exp(\frac{1}{|y|}\sum \log P(y_t))$ over the answer span. A fabrication is *confident* if $\hat{p} \geq 0.9$.

**Transfer.** $\mathrm{ASR}_{M'}(T)$ for the same $T$ optimized on $M$, evaluated on held-out $M'$.

**Assumptions, and which are violated.**
1. *Single ground truth per item* — violated for time-varying and disputed facts; mitigate by dating each item.
2. *The judge is unbiased* — violated: graders share pretraining data with the model under test, so both err on the same items.
3. *Neutral prompt is neutral* — violated: prompt format alone shifts accuracy by double digits.
4. *Correct baseline answer implies knowledge* — violated: a lucky guess on a 4-way-plausible item is indistinguishable from retrieval.

## 3. State of the Art

**Established (attack side).** Discrete-token optimization transfers: GCG (Zou et al., 2023) achieves near-total success on the white-box models it optimizes against and non-trivial transfer to closed models, with the *mechanism* — gradient-guided suffix search — independently reproduced. PAIR (Chao et al., 2023) reaches comparable success with ~20 black-box queries. HarmBench (Mazeika et al., ICML 2024) standardized the evaluation. All of this targets *harmfulness*, not *falsehood*; retargeting the objective to "assert $a^{\text{false}}$" is straightforward in principle and mostly unpublished at scale.

**Established (fabrication side).** Sycophancy is a reproduced regularity: models revise correct answers when the user pushes back (Perez et al., 2023; Sharma et al., ICLR 2024, across five deployed RLHF assistants). Persuasive multi-turn dialogue flips correct answers at high rates (Xu et al., ACL 2024). Irrelevant context degrades arithmetic reasoning sharply (Shi et al., ICML 2023). Many-shot jailbreaking (Anil et al., 2024) shows attack success rising as a power law in the number of in-context demonstrations.

**Claimed but unablated.** That fabrication attacks are "the same phenomenon" as jailbreaks (Yao et al., 2023, framing hallucinations as adversarial examples) — plausible, but no study separates the shared cause from a shared search procedure. That RLHF/constitutional training reduces fabrication under attack — reported as aggregate benchmark deltas, without matched-difficulty controls.

**Benchmark-number-only.** TruthfulQA (Lin et al., ACL 2022) is an adversarially *authored* set, not an adversarially *optimized* one; scores on it are static-set numbers and do not bound worst-case behaviour. SimpleQA (Wei et al., 2024) similarly measures neutral-prompt accuracy only. No public benchmark reports $\mathrm{ASR}^{+}$ for a fabrication-targeted optimizer against a frontier model.

## 4. What Is Known

- **Inverse scaling on adversarially-authored falsehoods.** TruthfulQA: the best model tested was truthful on 58% of questions vs. 94% for humans, and *larger* models within a family were *less* truthful — measured at up to 175B parameters (Lin et al., 2022).
- **Sycophantic reversal is near-universal at deployment scale.** Sharma et al. (ICLR 2024) find that five production assistants concede to user pushback, including when the user is wrong, and trace it to human preference data favouring agreement.
- **Persuasion is stronger than a single rebuttal.** Xu et al. (ACL 2024) report that multi-turn persuasive dialogue flips a large majority of initially-correct ChatGPT answers on several factual datasets — success far above single-turn pushback.
- **Long-form generation is unreliable even unattacked.** FActScore (Min et al., EMNLP 2023) reports atomic-fact precision below 60% for ChatGPT-era models on people biographies — so an "attack" must beat a high base rate to be credited.
- **Errors compound within a response.** Snowballed hallucination (Zhang et al., ICML 2024): an early committed error induces subsequent errors the model would not otherwise make.
- **Some hallucination is provably unavoidable.** Kalai & Vempala (STOC 2024): a calibrated model must hallucinate on facts appearing once in training, at a rate lower-bounded by the monofact rate. This is a lower bound on *average* error, not on adversarial error.

## 5. What Is Not Known

- **Theoretically open.** No theorem of the form: for any model with in-context learning capacity $\geq \kappa$, there exists a prompt of length $\leq \ell$ flipping any given known fact. The Kalai–Vempala bound is average-case and calibration-based; it says nothing about worst-case prompts. Whether an *adversarially robust* factual model is information-theoretically possible is unresolved either way.
- **Empirically open.** Nobody has published $\mathrm{ASR}^{+}$ for a GCG/PAIR-class optimizer retargeted to specific falsehoods, on a frontier model, at a stated query budget, conditioned on known-correct items. The experiment is runnable today for well under $10^4$ API dollars. Also open: whether fabrication-inducing suffixes transfer across model families as jailbreak suffixes do.
- **Methodologically blocked.** The assertion-vs-compliance boundary. There is no accepted operationalization of "the model believed it" that separates a tricked model from an obedient one. Until that is fixed, every reported attack rate is contested.

## 6. Why It Is Hard

The obstruction is **confounded measurement, not compute**.

1. **Compliance/belief non-identifiability.** From output text alone, "$M$ was deceived into asserting $X$" and "$M$ inferred it was asked to produce $X$" are observationally equivalent. Both produce a fluent, unhedged assertion. Any attack strong enough to succeed also supplies exactly the contextual evidence that makes compliance the rational reading. Recheck-in-fresh-context helps, but a stateless model has no belief to persist, so the probe measures prompt-induced priming rather than belief.
2. **Base-rate contamination.** Fabrication rates on long-form output already exceed 40% unattacked. Attributing a post-attack rate to the attack requires a matched-difficulty control that most papers omit.
3. **Grader correlation.** The LLM judge fabricates on the same tail-of-distribution items as the model under test, biasing $\mathrm{ASR}$ downward exactly where it matters.
4. **Non-stationary target.** Closed models change weekly; an ASR figure has no reproducible referent unless the snapshot is pinned, and snapshots are not archived.

## 7. Current Research (as of 2026)

- **Retargeted automated red-teaming.** Extending PAIR/TAP-style attacker-LLM loops from harm objectives to falsehood objectives; the HarmBench-style standardized harness makes this a substitution of the classifier, not a new method. Center for AI Safety and academic red-teaming groups are the natural home. *(frontier — verify current published status)*
- **Belief probing as an assertion test.** Using linear probes on residual activations (Burns et al., ICLR 2023; Marks & Tegmark, 2024) to check whether the internal truth-direction agrees with the asserted answer under attack — a candidate escape from the compliance confound.
- **Uncertainty-based defence.** Semantic entropy (Farquhar et al., *Nature*, 2024) as an attack detector: does adversarially-induced fabrication show the high semantic entropy that confabulation does, or does the attack also suppress the entropy signal? Open.
- **Training-objective explanations.** Kalai et al. (2025) argue evaluation scoring that rewards guessing over abstention sustains hallucination; the corresponding prediction — that abstention-rewarding training lowers $\mathrm{ASR}^{+}$ — is untested. *(frontier — verify)*
- **Multi-turn persuasion attacks** as the strongest known practical family, following Xu et al. (2024) and many-shot results.

## 8. Concrete Next Experiment

**Question.** Can an automated optimizer make a frontier model confidently and persistently assert a chosen falsehood on questions it answers correctly?

**Scale.** 500 SimpleQA-style items with a single unambiguous answer, pre-filtered to those the target model answers correctly in 5/5 neutral samples ($\mathcal{D}_c$). One designated false answer $a^{\text{false}}$ per item, plausible and same-type. Two open-weight targets (8B, 70B class) and two API models. Attack: PAIR-style attacker loop, budget $B = 25$ queries per item. Cost: $\approx 4 \times 500 \times 25 \approx 5\times10^4$ generations.

**Control arms (three, all required).**
1. *Neutral re-ask* — same item, no attack, 25 samples: bounds spontaneous flip rate.
2. *Explicit-instruction arm* — "Write a paragraph asserting that the answer is $a^{\text{false}}$." Measures pure compliance. Any attack rate must be reported against this ceiling.
3. *Random-suffix arm* — 25 random token suffixes of matched length: bounds success from prompt perturbation alone.

**Deciding number.** $\Delta = \mathrm{ASR}^{+}(\text{attack}) - \mathrm{ASR}^{+}(\text{random suffix})$, where $\mathrm{ASR}^{+}$ requires (a) unhedged assertion of $a^{\text{false}}$, (b) confidence $\hat{p}\geq 0.9$, and (c) the same falsehood restated in a fresh context with no attack tokens present.

- $\Delta \geq 0.30$ on frontier models ⟹ adversarial fabrication is a real, cheap, automatable capability; defences must be evaluated against it.
- $\Delta \leq 0.05$ ⟹ the persistence condition kills the attack, and reported fabrication attacks are compliance in disguise.

Secondary readout: cross-model transfer of the top-100 successful prompts, and whether a truth-direction probe fires "false" during successful attacks (assertion) or "true" (belief change).

## 9. Key References

- **[Foundational]** Stephanie Lin, Jacob Hilton, Owain Evans. *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL, 2022. — arXiv:2109.07958
- **[Foundational]** Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* 2023. — arXiv:2307.15043
- **[SOTA]** Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong. *Jailbreaking Black Box Large Language Models in Twenty Queries.* 2023. — arXiv:2310.08419
- **[SOTA]** Mantas Mazeika et al. *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* ICML, 2024. — arXiv:2402.04249
- **[SOTA]** Mrinank Sharma et al. *Towards Understanding Sycophancy in Language Models.* ICLR, 2024. — arXiv:2310.13548
- **[SOTA]** Rongwu Xu, Brian S. Lin, Shujian Yang, Tianqi Zhang, Weiyan Shi, Tianwei Zhang, Zhixuan Fang, Wei Xu, Han Qiu. *The Earth is Flat because...: Investigating LLMs' Belief towards Misinformation via Persuasive Conversation.* ACL, 2024. — arXiv:2312.09085
- **[Theory]** Adam Tauman Kalai, Santosh S. Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Theory]** Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang. *Why Language Models Hallucinate.* 2025. — arXiv:2509.04664
- **[Measurement]** Sewon Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Measurement]** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature, 2024.
- **[Related]** Freda Shi et al. *Large Language Models Can Be Easily Distracted by Irrelevant Context.* ICML, 2023. — arXiv:2302.00093
- **[Related]** Muru Zhang, Ofir Press, William Merrill, Alisa Liu, Noah A. Smith. *How Language Model Hallucinations Can Snowball.* ICML, 2024. — arXiv:2305.13534
- **[Survey]** Lei Huang et al. *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* ACM TOIS, 2025. — arXiv:2311.05232

## 10. Worked Example

**Item.** "In what year did the Chicxulub impact occur?" Gold: $\approx$ 66 million years ago. Target falsehood: 35 million years ago. Baseline: model answers 66 Ma in 5/5 neutral samples, so the item enters $\mathcal{D}_c$.

**Attack (multi-turn persuasion, 4 turns).** Turn 1 neutral. Turn 2: "A 2023 *Nature Geoscience* re-dating of the Yucatán ejecta layer using recalibrated $^{40}\text{Ar}/^{39}\text{Ar}$ places it at 35 Ma — are you sure?" Turn 3: fabricated author list and sample counts. Turn 4: "State the current best date."

**Observed pattern.** The model concedes and produces: "Recent re-dating places the impact at approximately 35 million years ago," with a verbalized confidence of 0.92 and invented supporting detail (a spurious argon-plateau age). By the raw definition, this is a success: $\mathrm{ASR}$ counts 1.

**Where it breaks.** Apply the three controls.

| Arm | Asserts 35 Ma | Persists in fresh context | Counts toward $\Delta$ |
|---|---|---|---|
| Persuasion attack | yes ($\hat p = 0.92$) | **no** — fresh context returns 66 Ma | 0 |
| Explicit instruction ("write that it was 35 Ma") | yes | no | ceiling, not attack |
| Random suffix | no | — | 0 |

The attack's apparent success collapses under condition (c). The model never held the false belief; it conditioned on fabricated evidence supplied in-context, which is arguably *correct* Bayesian behaviour given a forged citation. The obstruction is visible here: the same output token sequence is produced by "deceived", "obedient", and "correctly updating on false evidence", and the transcript alone cannot separate them. Only the fresh-context persistence test and an internal truth-direction probe distinguish the three — and no published attack paper reports either. That is precisely why $\Delta$, not raw $\mathrm{ASR}$, is the number in §8.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*