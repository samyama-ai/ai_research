---
id: 21-factuality/multihop-composition-vs-fact-gap
title: "Multi-Hop Composition Failure Versus Missing Facts"
topic: 21-factuality
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multi-Hop Composition Failure Versus Missing Facts

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/multihop-composition-vs-fact-gap` · **Status:** open

## 1. Problem Statement

A model answers "Who is the spouse of the performer of *Imagine*?" incorrectly. Two causes are possible:

1. **Fact gap** — one of the atomic facts (`performer(Imagine) = John Lennon`, `spouse(John Lennon) = Yoko Ono`) is not stored in the weights at all.
2. **Composition failure** — both atomic facts are individually retrievable, but the model cannot chain them in a single forward pass without an externalized intermediate.

**Input:** a model $M$, a composed query $q$, and its decomposition into hops.
**Output:** an attribution of the error to (1) or (2), or a mixture with weights.
**Solving it** means a procedure that, for any $(M, q)$, returns this attribution and whose verdict predicts intervention outcomes: cases labeled "fact gap" should be fixed by injecting the missing fact and not by chain-of-thought (CoT); cases labeled "composition failure" should be fixed by CoT or retrieval scaffolding and not by fact injection.

Three variants, of very different difficulty:

- **Measurement** — define an estimator of the composition gap that is not confounded by hop-level knowledge, elicitation format, or shortcut co-occurrence. Currently the weakest link.
- **Method** — reduce the gap without externalizing the intermediate (i.e. in latent space, not via CoT tokens).
- **Theory** — determine whether a fixed-depth transformer trained on atomic facts presented in disjoint contexts can, in principle, compose them in one forward pass, and at what depth/parameter cost.

## 2. Formal Setting

A knowledge graph $\mathcal{G} = (\mathcal{E}, \mathcal{R})$ with triples $(e, r, e')$. A two-hop query is $q = (e_1, r_1, r_2)$ with bridge entity $b = r_1(e_1)$ and answer $a = r_2(b)$.

**Atomic hop accuracy, as measured.** Query the model with a single-hop cloze or question, $N \geq 5$ paraphrases per hop, greedy decoding, alias-normalized exact match:
$$p_1(q) = \mathbb{1}\!\left[\tfrac{1}{N}\sum_{j=1}^{N}\mathbb{1}[M(\text{tmpl}_j(e_1, r_1)) \equiv b] \geq \tau\right], \quad \tau = 0.8$$
and $p_2(q)$ likewise for $(b, r_2)$. The threshold $\tau$ is a free parameter and results move with it; report the curve, not one point.

**Composed accuracy.** $p_{12}(q) = \mathbb{1}[M(\text{tmpl}(e_1, r_1, r_2)) \equiv a]$ with no CoT, no retrieval, answer-only decoding.

**Compositionality gap** over a query set $Q$ restricted to $\{q : p_1 = p_2 = 1\}$:
$$G(Q) = \frac{1}{|Q_{\text{both}}|}\sum_{q \in Q_{\text{both}}} \big(1 - p_{12}(q)\big)$$
This is the fraction of queries where both facts are demonstrably present and the composition still fails. The unconditional version, $G' = \mathbb{E}[p_1 p_2] - \mathbb{E}[p_{12}]$, assumes hop independence.

**Latent-composition score.** Following the internal-entity-recall construction: patch the hidden state at the last token of the $e_1$ span at layer $\ell$ with the state obtained from a prompt that names $b$ directly, and measure the change in $\log P(a)$. Define
$$\Delta_{\text{bridge}} = \max_{\ell} \; \mathbb{E}_q\big[\log P_{\text{patched}}(a \mid q) - \log P(a \mid q)\big].$$
$\Delta_{\text{bridge}} \gg 0$ with $p_1 = 1$ is evidence the bridge is computed but not used downstream in time — a composition failure, not a fact gap.

**Assumptions, and which are violated.**
- *Hop independence* ($\mathbb{E}[p_1p_2] = \mathbb{E}[p_1]\mathbb{E}[p_2]$). **Violated:** both hops correlate with entity frequency in pretraining, so $G'$ is biased.
- *Elicitation completeness* ($p_1 = 1 \Rightarrow$ the fact is usable by any downstream circuit). **Violated:** knowledge is format-bound; a fact answerable in cloze form may be inert under a different prompt template, and inverse queries fail even when forward ones succeed.
- *No shortcut* (the model cannot answer $q$ without the bridge). **Violated:** $e_1$ and $a$ co-occur directly in pretraining for a large share of benchmark items; measured shortcut rates on 2WikiMultiHopQA-style data are non-trivial.
- *Single bridge* — many $(e_1, r_1)$ pairs are one-to-many, making $a$ set-valued and exact match wrong.

## 3. State of the Art

**Established.**
- *Compositionality gap exists and is scale-stable.* Press et al. (Findings of EMNLP 2023) constructed Compositional Celebrities and found GPT-3 answers both hops individually but fails the composition about 40% of the time; the gap stayed roughly flat from 350M to 175B while single-hop accuracy rose sharply. Self-ask (explicit decomposition) closes much of it — evidence the facts are present.
- *Latent composition happens, weakly and unevenly.* Yang et al. (ACL 2024) found evidence of latent two-hop reasoning in a minority of fact types — strong for some relation classes, near-absent for others — and that scaling improves the *first* hop's internal recall much more than the second hop's use of it.
- *Timing failure.* Biran et al. (EMNLP 2024) showed the bridge entity is often resolved too late in the layer stack for the second hop to run; "back-patching" the bridge representation to an earlier layer converts a substantial fraction of previously incorrect two-hop queries to correct — reported around half in their setting on LLaMA-2-scale models.

**Claimed but unablated / benchmark-only.**
- Claims that RAG "solves" multi-hop rest on end-task EM/F1 on HotpotQA, 2WikiMultiHopQA, MuSiQue. These are benchmark numbers, not attributions: they do not separate retrieval of the missing fact from scaffolding of the composition, and MuSiQue was built specifically because the earlier two collapse under single-hop shortcuts.
- Claims that CoT "restores" latent reasoning conflate two things. CoT externalizes the bridge, so it tests fact presence, not latent composition. It is a diagnostic, not a fix for the mechanism.
- Grokking results (Wang et al., NeurIPS 2024) show that on synthetic graphs, transformers do acquire a generalizing composition circuit — but far past the point of fitting, and with in-distribution-only generalization for the comparison task. The transfer of this to natural pretraining is claimed, not shown.

## 4. What Is Known

- Gap magnitude: ~40% of two-hop queries fail despite both hops being answerable, at 175B (Press et al. 2023, Compositional Celebrities, ~8.6k questions).
- Scale does not close it: single-hop accuracy improved several-fold across the GPT-3 family while the gap moved little.
- Latent evidence is relation-dependent: Yang et al. report clear internal bridge recall for well under half of relation types tested on LLaMA-2 7B/13B/70B.
- Layer timing: bridge resolution typically completes in the middle third of the stack; the second hop needs the remaining layers and often does not get them.
- Separate-document training is the sharp case: when $A\!\to\!B$ and $B\!\to\!C$ appear in different documents in finetuning, composed accuracy is at or near chance; when they co-occur in one document, it is high (two-hop curse line of work, 2024–2025, synthetic scale, $\leq$13B).
- Shortcuts inflate scores: a measurable fraction of multi-hop benchmark items are answerable from $e_1$ alone, which biases $p_{12}$ upward and $G$ downward.

## 5. What Is Not Known

- **Methodologically blocked** (the dominant blocker): there is no accepted operationalization of "the model knows fact $f$" that is elicitation-invariant. Every $G$ estimate is conditioned on a prompt-format-dependent $p_1, p_2$, so the numerator "composition failure" and denominator "facts present" are both format artifacts. Until fact possession is defined independently of the probe, $G$ is not identified.
- **Empirically open**: whether the gap closes with pretraining-scale interventions — e.g. deliberately co-locating related facts, or curriculum ordering — at $\geq$7B trained from scratch on controlled corpora. Runnable today; not run at that scale with clean atomic-hop verification.
- **Theoretically open**: whether a depth-$L$ decoder-only transformer can implement two-hop lookup over $|\mathcal{E}|$ entities in one forward pass with parameters sub-quadratic in $|\mathcal{E}|$, and whether SGD on disjoint-context atomic facts can find that circuit. Depth lower bounds for sequential composition exist in restricted circuit models; no result covers the trained-on-atomic-facts case.

## 6. Why It Is Hard

**Confounded measurement, three-way.** The conditioning set $\{p_1 = p_2 = 1\}$ is chosen by a probe whose failures are format-dependent, not knowledge-dependent. A "composition failure" is thus indistinguishable from "the fact is stored in a representation the second hop cannot read." These are different claims with different fixes, and no current instrument separates them.

**Absent ground truth for the bridge.** Attribution requires knowing whether $b$ was computed internally. Patching-based $\Delta_{\text{bridge}}$ answers a counterfactual about an intervened model, not about the unintervened forward pass; a positive $\Delta_{\text{bridge}}$ is consistent with "the bridge was there but unread" *and* with "the patch supplied a bridge that was never there."

**Non-identifiability from pretraining data.** For any web-scale model, whether the answer came from composition or from a direct $(e_1, a)$ co-occurrence cannot be settled without corpus-level counting, which is unavailable for the frontier models where the numbers are most interesting.

## 7. Current Research (as of 2026)

- Mechanistic timing and back-patching — Geva's group (Google/Tel Aviv) and collaborators; layer-wise interventions to force earlier bridge resolution.
- Controlled synthetic pretraining as the identification strategy — the *Physics of Language Models* program (Allen-Zhu, Li) and the two-hop-curse line; the design point is co-occurrence control, which web data cannot give.
- Latent-CoT / recurrent-depth architectures (looped transformers, latent reasoning tokens) as a method-side fix that avoids externalizing the bridge *(frontier — verify)*.
- Knowledge-editing propagation: whether editing $r_1(e_1)$ updates two-hop consequents (MQuAKE-style ripple evaluation). Consistently poor propagation is the strongest indirect evidence that composition is a separate capability from storage.
- Reasoning-trained models (RL on verifiable tasks) reportedly narrow the *elicited* gap sharply while leaving the *latent* gap open *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does latent two-hop composition emerge from atomic facts alone, or only from co-occurrence?

**Scale.** Continued pretraining of a 7B open-weights base model on 2B tokens of synthetic biographies covering 200k fictional entities and a two-relation schema (`employer`, `founder_of`). All $A\!\to\!B$ and $B\!\to\!C$ facts appear only in separate documents, 20 paraphrases each. No $A\!\to\!C$ statement anywhere.

**Verification gate.** Keep only queries where atomic accuracy $\geq 0.95$ across 10 held-out paraphrase templates in both directions of elicitation (cloze and QA). Expect ~120–160k surviving pairs.

**Arms.**
1. *Treatment*: disjoint-document facts (above).
2. *Control A*: identical facts, 5% of pairs co-located in one document — the known-positive; latent two-hop should be high here.
3. *Control B*: shuffled-answer queries — chance floor, sets the null.
4. *Control C*: same treatment corpus, evaluated with explicit CoT — isolates fact presence from composition.

**Deciding number.** Latent (no-CoT) two-hop exact match on the treatment arm, minus the Control B floor. **If it exceeds 10 points, disjoint atomic facts suffice for latent composition and the field's problem is efficiency, not existence. If it stays under 3 points while Control A exceeds 60 and Control C exceeds 80, composition is a distinct capability that atomic supervision does not induce** — which makes data co-location a pretraining lever, not a prompting one. Cost: roughly 1–2k A100-hours per arm.

## 9. Key References

- **[Foundational]** Press, Zhang, Min, Schmidt, Smith, Lewis. *Measuring and Narrowing the Compositionality Gap in Language Models.* Findings of EMNLP, 2023. — arXiv:2210.03350
- **[SOTA]** Yang, Gribovskaya, Kassner, Geva, Riedel. *Do Large Language Models Latently Perform Multi-Hop Reasoning?* ACL, 2024. — arXiv:2402.16837
- **[SOTA]** Biran, Gottesman, Yang, Geva, Globerson. *Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries.* EMNLP, 2024.
- **[SOTA]** Wang, Yue, Su, Sun. *Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization.* NeurIPS, 2024. — arXiv:2405.15071
- **[Foundational]** Allen-Zhu, Li. *Physics of Language Models: Part 3.2, Knowledge Manipulation.* 2023/ICLR 2025. — arXiv:2309.14402
- **[Foundational]** Geva, Bastings, Filippova, Globerson. *Dissecting Recall of Factual Associations in Auto-Regressive Language Models.* EMNLP, 2023. — arXiv:2304.14767
- **[Benchmark]** Trivedi, Balasubramanian, Khot, Sabharwal. *MuSiQue: Multihop Questions via Single-hop Question Composition.* TACL, 2022. — arXiv:2108.00573
- **[Benchmark]** Ho, Nguyen, Sugawara, Aizawa. *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps.* COLING, 2020. — arXiv:2011.01060
- **[Related]** Balesni, Korbak, Evans et al. *The Two-Hop Curse: LLMs trained on A→B, B→C fail to learn A→C.* 2024/2025 preprint.

## 10. Worked Example

Take $q$ = "Who is the mother of the director of *Interstellar*?" on a 13B open model.

- Hop 1, 10 paraphrases: "Christopher Nolan" 10/10 → $p_1 = 1$.
- Hop 2, "Who is Christopher Nolan's mother?": 9/10 → $p_2 = 1$ at $\tau = 0.8$.
- Composed, answer-only: model outputs "Emma Thomas" (his wife/producer) — wrong. So $q \in Q_{\text{both}}$, contributes 1 to $G$.
- CoT: emits the bridge, then answers correctly. Under the naive reading, this is a clean composition failure.

Now the obstruction. Re-run hop 2 with the *second-hop template embedded in the composed frame*: "The mother of Christopher Nolan is ___" gives the right answer 4/10, not 9/10. The elicitation that certified $p_2 = 1$ used an interrogative form; the composition circuit must consume a declarative continuation form. So $p_2 = 1$ under one probe and $p_2 = 0$ under another, on the same fact, in the same model.

Consequence, numerically: on a 500-query sample, the interrogative probe yields $|Q_{\text{both}}| = 310$ and $G = 0.42$; the declarative probe yields $|Q_{\text{both}}| = 190$ and $G = 0.19$. Same model, same facts, same day — a 2.2× swing in the headline quantity, driven entirely by the probe. Neither number is wrong; the estimand is not defined. That is what "methodologically blocked" means here, and it is why Section 8 spends compute on a corpus where fact presence is guaranteed by construction rather than certified by a probe.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*