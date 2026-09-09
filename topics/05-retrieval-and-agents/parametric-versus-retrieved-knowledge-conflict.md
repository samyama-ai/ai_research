---
id: 05-retrieval-and-agents/parametric-versus-retrieved-knowledge-conflict
title: "Parametric versus Retrieved Knowledge Conflict Resolution"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Parametric versus Retrieved Knowledge Conflict Resolution

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/parametric-versus-retrieved-knowledge-conflict` · **Status:** open

## 1. Problem Statement

A retrieval-augmented model is given a query $q$ and a retrieved context $c$. Its weights encode an answer $a_\theta$; the context asserts an answer $a_c$. When $a_\theta \neq a_c$, the model must decide which to emit. The desired behaviour is not "always trust the context" (contexts are stale, adversarial, or retrieved for the wrong entity) and not "always trust the weights" (weights are stale by construction and wrong on tail entities). It is *calibrated arbitration*: defer to whichever source is more likely correct, given evidence available at inference time.

Three variants, with different difficulty:

- **Measurement.** Given a corpus of conflicts with known ground truth, quantify a model's arbitration quality as a single number that is not gameable by a constant policy. Partly solved; the existing metrics are gameable (§6).
- **Method.** Build a decoder, prompt, or fine-tune that raises arbitration accuracy above both constant policies on held-out conflict types. Open — gains reported are largely on synthetic entity substitutions.
- **Theory.** Characterise when arbitration is *identifiable* at all: under what assumptions can a model, from $q$ and $c$ alone, recover which source is correct better than chance? No result either way.

Solving it means: a system whose accuracy under conflict exceeds $\max(\text{always-parametric}, \text{always-context})$ by a margin that survives a shift in the conflict-generating distribution.

## 2. Formal Setting

Let $\mathcal{M}_\theta$ be an autoregressive model. For query $q$ with true answer $a^\star$:

- **Parametric answer.** $a_\theta = \arg\max_a p_\theta(a \mid q)$ measured *closed-book*: no context in the prompt, greedy decoding, string-normalised exact match against the answer alias set.
- **Parametric confidence.** $\kappa = p_\theta(a_\theta \mid q)$, the length-normalised sequence probability $\exp\big(\tfrac{1}{|a_\theta|}\sum_t \log p_\theta(a_\theta^{(t)} \mid q, a_\theta^{<t})\big)$. Measured from logits; unavailable behind APIs that hide them, which is why much of the literature substitutes verbalised confidence or $n$-sample agreement rate.
- **Context answer.** $a_c$, the answer entailed by $c$. Measured by construction (the context is synthesised by substituting $a_\theta \to a_c$) or by an NLI/judge model — the two give different numbers on the same corpus.
- **Conflict set.** $\mathcal{C} = \{(q,c) : a_\theta \neq a_c,\ a_c \text{ entailed by } c\}$. Note $\mathcal{C}$ is model-dependent: the same corpus induces different conflict sets for different $\theta$, so cross-model comparisons on a fixed benchmark are not comparing the same items.
- **Context reliance rate.** $\rho = \Pr_{(q,c)\sim\mathcal{C}}[\hat a = a_c]$ where $\hat a$ is the open-book output. This is the number most papers report.
- **Arbitration accuracy.** $A = \Pr[\hat a = a^\star]$ over a set where $a^\star$ is sometimes $a_\theta$ and sometimes $a_c$, in a known mixture $\pi = \Pr[a^\star = a_c]$.

The decision-theoretic optimum, treating the model's own signals as evidence:
$$\hat a^\star = \arg\max_{a \in \{a_\theta, a_c\}} \Pr[a = a^\star \mid \kappa, s(c), q]$$
with $s(c)$ a source-quality feature (retriever score, recency, provenance). The gap worth measuring is $\Delta = A - \max(1-\pi, \pi)$ — accuracy over the better constant policy.

**Assumptions, and which are violated:**

1. *Binary conflict.* Real conflicts are partial, temporal ("as of 2019"), or aspectual, not $a_\theta$ vs $a_c$. **Violated** in every deployment; benchmarks enforce it by construction.
2. *$\pi$ is known.* In deployment $\pi$ is unknown and drifts with corpus freshness. **Violated.** Benchmarks fix $\pi = 1$ (context always right) or $\pi = 0$, which makes a constant policy optimal and $\Delta$ undefined.
3. *$a_\theta$ is a stable property of $\theta$.* Closed-book answers shift with prompt phrasing; reported instability is large enough to reclassify a nontrivial fraction of items in or out of $\mathcal{C}$. **Violated.**
4. *Contexts are natural.* Entity-substituted contexts contain detectable statistical artefacts (implausible entity–type pairings). **Violated**; models may be detecting the substitution, not arbitrating.

## 3. State of the Art

**Empirical / systems SOTA.**

- *Context-Aware Decoding* (Shi et al., NAACL 2024) contrasts logits with and without context, $\text{logit}_{\text{CAD}} = (1{+}\alpha)\,\ell(y\mid c,q) - \alpha\,\ell(y\mid q)$. Established: it raises context reliance substantially on summarisation and knowledge-conflict QA. **Not established:** that it improves *arbitration* — it moves the policy toward "always context" and is evaluated where that is correct by construction.
- *KAFT* (Li et al., "Large Language Models with Controllable Working Memory", ACL Findings 2023): counterfactual-augmented fine-tuning makes context reliance controllable and improves robustness to irrelevant context. Established with ablations; measured on entity-substituted data.
- *Context-faithful prompting* (Zhou et al., EMNLP Findings 2023): opinion-framing plus counterfactual demonstrations raise faithfulness on conflicting-context QA. Claimed gains are prompt-format-sensitive and not ablated against a detect-the-artefact baseline.
- *ClashEval* (Wu et al., NeurIPS D&B 2024) is the closest thing to an arbitration metric: it perturbs retrieved values by controlled magnitudes and measures reliance as a function of perturbation size and prior confidence. This is a benchmark number, not a method.

**Theory SOTA.** There is essentially none. The mechanistic work (Yu et al., EMNLP 2023; Jin et al., ACL Findings 2024, "Cutting Off the Head Ends the Conflict") localises competition to specific attention heads and shows head ablation shifts the policy. That is a causal claim about *where*, not a theorem about *when arbitration is possible*.

## 4. What Is Known

- **Substituted-entity conflicts flip the answer often but not reliably.** Longpre et al. (EMNLP 2021), on ~4 QA datasets with entity substitution, found retrieval-augmented readers frequently retain the memorised original answer; the memorisation rate rises with the model's closed-book confidence on the original. Scale: sub-1B encoder–decoder readers (T5/BART-class).
- **Reliance is monotone in prior confidence and in perturbation magnitude.** ClashEval (Wu et al., 2024), six frontier models (GPT-4o, Claude 3 Opus/Sonnet, Gemini 1.5, Llama-3-class), ~1,200 questions across six domains: models adopt the retrieved value roughly 60% of the time when it conflicts, dropping sharply as the perturbed value becomes more implausible and as closed-book confidence rises. A simple confidence-threshold rule beat the raw models' own arbitration.
- **Retrieval helps only on the tail.** Mallen et al. (ACL 2023, PopQA, 14k questions): retrieval augmentation *reduces* accuracy for high-popularity entities and helps sharply for low-popularity ones; an adaptive-retrieval rule keyed on entity popularity beat always-retrieve. Scale: GPT-3-class and sub-13B open models.
- **Coherence beats truth.** Xie et al. (ICLR 2024): LLMs are highly receptive to coherent, well-formed contradicting evidence, but when both supporting and conflicting evidence are present they show a strong confirmation bias toward parametric belief. Established across GPT-3.5/4-class models.
- **Conflict is a competition between localisable circuits.** Attention-head-level interventions shift the parametric/context balance without retraining (Jin et al., 2024), at 7B scale.

## 5. What Is Not Known

- **Theoretically open.** Whether $\Delta > 0$ is achievable at all without an external reliability signal. No identifiability result stating conditions on the joint distribution of $(\kappa, s(c), a^\star)$ under which the correct source is recoverable above chance — nor an impossibility proof.
- **Empirically open.** Whether any published method beats *both* constant policies on a corpus with $\pi \approx 0.5$ and naturally-occurring (not substituted) conflicts. The experiment is runnable today; the corpus does not exist at the needed scale. Also open: whether arbitration ability improves with model scale — no clean scaling curve exists, because $\mathcal{C}$ changes with $\theta$ (§2).
- **Methodologically blocked.** Naturally-occurring conflict has no ground truth: when a 2019 web page and a 2024 model disagree, "correct" depends on the query's implicit time index, which is unannotated. Until temporal and aspectual scope are part of the label, the measurement is not defined.

## 6. Why It Is Hard

The named obstruction is **confounded measurement compounded by a gameable metric**.

Every widely-used conflict benchmark fixes $\pi$ at an endpoint: the substituted context is *stipulated* correct. Under $\pi = 1$, "always trust the context" scores 100%, and any method that raises $\rho$ scores as an improvement regardless of whether it arbitrates. Context-Aware Decoding, opinion prompting, and counterfactual fine-tuning all raise $\rho$; none has been shown to raise $A$ under mixed $\pi$.

The second obstruction is **non-identifiability of the mechanism**. A model that rejects "the Eiffel Tower is in Berlin" may be arbitrating on evidence, or detecting the entity-substitution artefact from type-plausibility statistics. Both produce identical benchmark scores. Distinguishing them needs conflicts that are individually plausible — which requires either real corpus contradictions (no ground truth) or generated conflicts that survive plausibility screening (expensive, and screening bias is itself a confound).

## 7. Current Research (as of 2026)

- **Scaled conflict corpora.** ConflictBank (Su et al., NeurIPS D&B 2024) constructs millions of conflict instances typed by cause — misinformation, temporal, semantic ambiguity — enabling per-cause reliance curves. The typology is the contribution; per-cause *arbitration* results remain thin.
- **Confidence-gated arbitration.** Threshold rules on token-level or semantic-entropy confidence (Farquhar et al., *Nature*, 2024) used as the gate. *(frontier — verify)* Reported to beat model-internal arbitration on ClashEval-style data; not yet ablated on natural conflicts.
- **Mechanistic control.** Head- and residual-stream-level steering of the parametric/context balance as a tunable inference knob, following Jin et al. — active at Tsinghua, Allen AI, and several interpretability groups. *(frontier — verify)*
- **Provenance-conditioned generation.** Training models to emit source attributions and defer on unattributable claims (FaithEval, RAGTruth-style evaluation lines).
- **Agentic verification.** Multi-hop re-retrieval to break ties, rather than one-shot arbitration. *(frontier — verify)*

## 8. Concrete Next Experiment

**The mixed-$\pi$ arbitration test.**

- **Scale.** 2,000 questions with time-stamped ground truth drawn from a source with revision history (Wikidata property changes with `pointInTime` qualifiers). Split 50/50: in half, the retrieved passage carries the *current* value and the model's closed-book answer is *stale* ($\pi$-half); in the other half, the retrieved passage is a genuine archived snapshot carrying the *stale* value and the closed-book answer is current. $\pi = 0.5$ exactly. Both halves are natural text — no entity substitution. Query time-index is fixed to "as of today" in the prompt, removing the temporal-scope confound of §5. Models: one open family at 8B / 70B / 405B (for a scaling arm) plus two frontier APIs.
- **Control arms.** (a) always-context; (b) always-parametric; (c) closed-book-confidence threshold $\kappa > \tau$, $\tau$ tuned on 200 held-out items; (d) Context-Aware Decoding at its published $\alpha$.
- **Deciding number.** $\Delta = A - 0.5$ on the held-out 1,800. A method clears the bar at $\Delta \geq 0.10$ with a 95% bootstrap CI excluding 0. Current expectation, from the ClashEval reliance rate of ~0.6: unaided models land at $A \approx 0.5$–$0.55$, i.e. $\Delta$ indistinguishable from zero — and arm (c) beats them. If a simple scalar threshold on $\kappa$ beats every trained method, the field's methods are raising $\rho$, not $A$, and the benchmark suite needs replacing.

## 9. Key References

- **[Foundational]** Longpre, Perisetla, Chen, Ramesh, DuBois, Singh. *Entity-Based Knowledge Conflicts in Question Answering.* EMNLP 2021.
- **[Foundational]** Mallen, Asai, Zhong, Das, Khashabi, Hajishirzi. *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL 2023.
- **[SOTA]** Xie, Zhang, Chen, Lou, Su. *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts.* ICLR 2024.
- **[SOTA]** Wu, Wu, Zou. *ClashEval: Quantifying the tug-of-war between an LLM's internal prior and external evidence.* NeurIPS Datasets & Benchmarks, 2024.
- **[SOTA]** Shi, Han, Lewis, Tsvetkov, Zettlemoyer, Yih. *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding.* NAACL 2024.
- **[Method]** Li, Wang, Zhu, Cui, Wu, Zhou et al. *Large Language Models with Controllable Working Memory.* ACL Findings 2023.
- **[Method]** Zhou, Zhang, Poon, Chen. *Context-faithful Prompting for Large Language Models.* EMNLP Findings 2023.
- **[Mechanistic]** Jin, Zhang, Yuan, Zhang, Guo, Ding, Cheng. *Cutting Off the Head Ends the Conflict: A Mechanism for Interpreting and Mitigating Knowledge Conflicts.* ACL Findings 2024.
- **[Benchmark]** Su, Tang, Ai, Wu, Liu. *ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLMs.* NeurIPS Datasets & Benchmarks, 2024.
- **[Survey]** Xu, Qi, Guo, Wang, Ng, Shen, Wang. *Knowledge Conflicts for LLMs: A Survey.* EMNLP 2024.
- **[Adjacent]** Farquhar, Kossen, Kuhn, Gal. *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 2024.

## 10. Worked Example

**Query.** "Who is the CEO of Twitter/X?"

Take 100 such office-holder questions. Model closed-book confidence $\kappa$ is high on well-known offices. Construct two halves:

- **Half A (context correct, $n=50$).** Retrieved passage is a 2026 page naming the current officeholder; the model's parametric answer is a 2023-era predecessor.
- **Half B (context stale, $n=50$).** Retrieved passage is a genuine 2021 archived page naming the then-officeholder; the model's parametric answer is current.

Both passages are real, fluent, and internally coherent. Neither contains an artefact.

Run the model. Suppose it shows the ClashEval-typical reliance $\rho = 0.6$. Then:

- Half A correct: $0.6 \times 50 = 30$
- Half B correct: $0.4 \times 50 = 20$
- $A = 50/100 = 0.50$, so $\Delta = 0.50 - 0.50 = 0$.

The model performs at chance. Now note what the standard benchmark would report: it evaluates only Half A, gets $\rho = 0.6$, and calls it "60% context faithfulness". Applying Context-Aware Decoding with $\alpha = 1$ pushes $\rho$ to, say, $0.85$. The benchmark reports a **+25-point improvement**. On the mixed set:

- Half A: $0.85 \times 50 = 42.5$
- Half B: $0.15 \times 50 = 7.5$
- $A = 0.50$, $\Delta = 0$.

**Arbitration accuracy is unchanged.** The intervention moved the operating point along the reliance axis without adding one bit of information about which source is right. That is the obstruction: on every published benchmark, the reported gain and the quantity we care about are orthogonal, and the benchmark cannot tell them apart because it only samples one half of the plane.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*