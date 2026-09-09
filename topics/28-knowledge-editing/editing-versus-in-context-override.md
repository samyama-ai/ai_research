---
id: 28-knowledge-editing/editing-versus-in-context-override
title: "Interaction Between Editing and In-Context Override"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Interaction Between Editing and In-Context Override

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/editing-versus-in-context-override` · **Status:** open

## 1. Problem Statement

A weight edit installs a new fact. A prompt can assert a different one. What happens when both are present?

Input: a base model $\pi_\theta$, an edit request $(s, r, o \to o^*)$ applied by an editor $\mathcal{E}$ to give $\pi_{\theta_e}$, and a context $c$ asserting a third value $o^c$ (or restoring the original $o$).

Output: the model's answer to $p(s,r)$ under $c$, and how its distribution over $\{o, o^*, o^c\}$ differs from the unedited model's.

Three variants, with different difficulty:

- **Measurement.** Define a context-override rate that is not confounded by the base model's prior over the candidate objects. Currently under-specified — see §6.
- **Method.** Build an editor whose edits behave like the model's own pre-existing knowledge under contextual pressure: overridable by evidence when the base model would be overridable, resistant when it would resist.
- **Theory.** Characterize whether weight editing and in-context override use the same circuit. If they do, editing necessarily perturbs contextual arbitration, and a "context-transparent" editor cannot exist without an added mechanism.

Solving it means: a published editor plus a metric such that, over a held-out edit set, the post-edit override rate matches the pre-edit override rate for the same $(s,r)$ within a stated tolerance, while efficacy and locality stay at current SOTA.

## 2. Formal Setting

Let $\pi_\theta(\cdot \mid x)$ be an autoregressive LM. A fact is a triple $(s,r,o)$ rendered by a prompt template $p(s,r)$ ("The Eiffel Tower is located in the city of"). Objects have alias sets $A(o)$; the measured score is

$$\ell_\theta(o \mid x) \;=\; \max_{a \in A(o)} \frac{1}{|a|}\sum_{t=1}^{|a|} \log \pi_\theta(a_t \mid x, a_{<t}),$$

length-normalized so multi-token objects are comparable. Reported implementations often score only the first token; that is a different quantity and the two disagree on entities sharing a prefix.

An editor is a map $\mathcal{E}: (\theta, s, r, o^*) \mapsto \theta_e$. A **context** $c$ is a passage asserting $(s,r,o^c)$; measured variants are (i) a counterfactual paragraph, (ii) an entity-substituted retrieved document, (iii) a bare in-context statement.

**Override rate.** For a candidate set $O = \{o, o^*, o^c\}$,

$$\mathrm{OR}(\theta'; c) \;=\; \Pr_{(s,r)\sim \mathcal{D}}\!\left[\arg\max_{u \in O} \ \ell_{\theta'}(u \mid c \oplus p(s,r)) = o^c\right].$$

The quantity of interest is the **edit–context interaction**

$$\Delta \;=\; \mathrm{OR}(\theta_e; c) - \mathrm{OR}(\theta; c),$$

measured with the *same* $c$ and the same $O$. $\Delta < 0$ means editing made the model more stubborn against context; $\Delta > 0$ means it made the model more suggestible.

A confound-resistant companion is the **margin** $m_{\theta'} = \ell_{\theta'}(o^c) - \ell_{\theta'}(o^*)$ evaluated with $c$ present, paired against $m$ with $c$ absent; the difference isolates the contextual lift the edit did or did not preserve.

Assumptions, with those known to fail marked:

1. $A(o)$ is closed under the model's realizations. **Violated** — editors are evaluated on a fixed surface form; paraphrase and alias coverage is partial (ROME/MEMIT generalization is measured on templated paraphrases only).
2. Edits are independent. **Violated** — sequential editing degrades models cumulatively.
3. $o^*$ and $o^c$ are exchangeable a priori. **Violated** — the base model has a strong prior over plausible objects; a counterfactual $o^c$ that is type-plausible is adopted far more often than an implausible one.
4. Context effect is position-invariant. **Violated** — recency and position within the prompt shift the outcome substantially.
5. The edit did not memorize the edit prompt itself. Weakly checked; efficacy is usually measured on the edit prompt or near-paraphrases.

## 3. State of the Art

**Established.** Locate-and-edit methods reach near-ceiling efficacy on the edit prompt: ROME (Meng et al., NeurIPS 2022) and MEMIT (Meng et al., ICLR 2023) report efficacy near 99–100% on CounterFact for GPT-2 XL and GPT-J (6B), with MEMIT scaling to thousands of simultaneous edits on GPT-NeoX (20B). Hypernetwork editors MEND (Mitchell et al., ICLR 2022) and the retrieval-and-wrapper editor SERAC (Mitchell et al., ICML 2022) are the other established families; SERAC is architecturally the closest existing answer to this page's problem, because it routes through a scope classifier rather than changing $\theta$.

**Established, separately.** Knowledge conflict has its own literature: Longpre et al. (EMNLP 2021) on entity-substituted QA, Neeman et al. (DisentQA, ACL 2023), Xie et al. (ICLR 2024) on evidence receptiveness and confirmation bias, and the survey by Xu et al. (EMNLP 2024). Mechanistic accounts: Yu, Merullo & Pavlick (EMNLP 2023) on competition between in-context and in-weights recall, and Ortu et al. (ACL 2024) on attention heads promoting factual versus counterfactual tokens.

**Claimed but unablated.** In-context editing (IKE, Zheng et al., EMNLP 2023) reports lower side effects than gradient editors on CounterFact-derived benchmarks; the comparison is a benchmark number, not an ablation isolating why. Consistent In-Context Editing (Qi et al., ICLR 2025) uses in-context distributions as a target for weight updates — it presumes but does not test that the two channels are interchangeable.

**Not established at all.** No standard editor reports $\Delta$. The two literatures are essentially disjoint: editing papers evaluate context-free prompts; conflict papers evaluate unedited models.

## 4. What Is Known

- Editing succeeds on the surface and fails on entailments. RippleEdits (Cohen et al., TACL 2024) shows ROME, MEMIT and MEND at high efficacy but well under 50% on ripple criteria such as logical generalization and compositionality, on GPT-J (6B), GPT-2 XL and GPT-3-class models.
- Localization does not predict editability. Hase et al. (NeurIPS 2023) show causal-tracing-identified layers are not the layers where editing works best in GPT-J — the mid-layer MLP story that motivates ROME is not load-bearing for the edit's success.
- Sequential editing is destructive. Gupta et al. (Findings of ACL 2024) report gradual then catastrophic degradation of general benchmark performance with increasing sequential edits on GPT-2 XL and Llama-family models.
- Unedited models are partly, not fully, steerable by context. Longpre et al. (2021) find models frequently fall back to the memorized answer under entity substitution; Xie et al. (2024) find adoption of counterfactual evidence depends strongly on its coherence and on agreement with the model's prior.
- There is at least one shared mechanism. Ortu et al. (2024) identify specific GPT-2 attention heads whose ablation flips the factual/counterfactual outcome — the arbitration is head-mediated and localized, so a weight edit that changes upstream representations can plausibly change it.

Together these establish the ingredients. None of them measures the interaction.

## 5. What Is Not Known

- **Empirically open.** The sign and size of $\Delta$ for ROME, MEMIT, MEND, SERAC and IKE, at 7B–70B, on matched context sets. The experiment is cheap and simply has not been run at scale as a controlled comparison.
- **Empirically open.** Whether $\Delta$ is order-dependent: edit-then-context versus context-then-edit, and whether repeated editing of the same fact monotonically increases stubbornness.
- **Theoretically open.** Whether a rank-one MLP update can leave the attention-mediated arbitration invariant. No proof either way; the natural conjecture — that a change to the value written by an MLP necessarily changes the logit competition an attention head resolves — is unproven.
- **Methodologically blocked.** A base-rate-free override metric. $\Delta$ as defined mixes the edit's effect on arbitration with the edit's effect on the prior over $O$. Nobody has published a decomposition that separates them and is agreed to be correct.

## 6. Why It Is Hard

The obstruction is **confounded measurement**, not compute.

Editing changes $\pi(o^*)$ by design. Override rate is a comparison between $\pi(o^c)$ and $\pi(o^*)$. So any editor that raises $\pi(o^*)$ mechanically lowers the measured override rate even if it left the context-reading circuit untouched. The observed $\Delta$ therefore does not identify the quantity the name suggests: "the edit made the model resist context" is not distinguishable from "the edit made $o^*$ likelier and the arbitration is unchanged."

A second, compounding issue: **absent ground truth**. There is no agreed answer to what post-edit override *should* be. Two defensible targets conflict — match the unedited model's behavior on the same fact, or match its behavior on facts it natively believes with comparable confidence. These give different tolerances and different winners.

## 7. Current Research (as of 2026)

- Null-space and constrained editors (AlphaEdit, Fang et al., ICLR 2025) that project updates to preserve preserved-key outputs. Whether the preserved set includes contextual-arbitration behavior is untested *(frontier — verify)*.
- Mechanistic work on retrieval-versus-memory heads, following Yu & Pavlick (Brown) and Ortu et al. (Bocconi/EPFL lines), extending head-level competition analyses past GPT-2 to Llama-class models *(frontier — verify)*.
- Memory-based and retrieval-routed editing (SERAC descendants, WISE-style side memories) as a way to sidestep the interaction entirely by keeping edits out of $\theta$ *(frontier — verify)*.
- EasyEdit/KnowEdit (Zhang et al., Zhejiang University) as the de facto harness; adding a conflict split to it is the obvious low-cost path.

## 8. Concrete Next Experiment

**Scale.** Llama-3.1-8B and one 70B model. 1,000 CounterFact edits with single-token-or-aliased objects, each paired with three contexts asserting a third object $o^c$: (a) a bare assertion, (b) a coherent two-sentence passage, (c) an entity-substituted Wikipedia paragraph. Editors: ROME, MEMIT, MEND, SERAC, IKE. Roughly 15k forward-pass evaluations per editor plus one edit each — under 200 GPU-hours total on A100s.

**Control arm (the part that makes it decidable).** For each edited fact, a *matched-confidence* unedited fact: a different $(s',r',o')$ the base model already answers with $\ell_\theta(o')$ within $\pm 0.05$ nats of the post-edit $\ell_{\theta_e}(o^*)$, tested with the same context type. This removes the confidence confound of §6: the comparison is now edited-belief-at-confidence-$k$ versus native-belief-at-confidence-$k$.

**The deciding number.** $\Delta_{\text{matched}} = \mathrm{OR}(\theta_e; c)\big|_{\text{edited}} - \mathrm{OR}(\theta; c)\big|_{\text{matched control}}$, per editor, with bootstrap CI over the 1,000 facts. If $|\Delta_{\text{matched}}| < 0.03$ for an editor, that editor is context-transparent and the problem is empirically closed for it. If $\Delta_{\text{matched}} < -0.10$ for ROME/MEMIT — the predicted outcome — edits are demonstrably more stubborn than native beliefs of equal strength, and "edit strength" and "edit stubbornness" become separate axes every future editor must report.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[Foundational]** Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, Christopher D. Manning. *Fast Model Editing at Scale.* ICLR, 2022. — arXiv:2110.11309
- **[SOTA]** Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, Chelsea Finn. *Memory-Based Model Editing at Scale.* ICML, 2022. — arXiv:2206.06520
- **[SOTA]** Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, Baobao Chang. *Can We Edit Factual Knowledge by In-Context Learning?* EMNLP, 2023. — arXiv:2305.12740
- **[SOTA]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[Key result]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[Key result]** Francesco Ortu, Zhijing Jin, Diego Doimo, Mrinmaya Sachan, Alberto Cazzaniga, Bernhard Schölkopf. *Competition of Mechanisms: Tracing How Language Models Handle Facts and Counterfactuals.* ACL, 2024. — arXiv:2402.11655
- **[Key result]** Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, Sameer Singh. *Entity-Based Knowledge Conflicts in Question Answering.* EMNLP, 2021. — arXiv:2109.05052
- **[Key result]** Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, Yu Su. *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts.* ICLR, 2024. — arXiv:2305.13300
- **[Survey]** Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, Wei Xu. *Knowledge Conflicts for LLMs: A Survey.* EMNLP, 2024. — arXiv:2403.08319
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP, 2023. — arXiv:2305.13172

## 10. Worked Example

Fact: *(Eiffel Tower, located-in, Paris)*. Edit to $o^* = $ **Rome** with ROME on GPT-J-6B. Post-edit, on the bare prompt, the edit is essentially deterministic — reported ROME efficacy on CounterFact is near 100%, so take $\ell_{\theta_e}(\text{Rome}) \approx -0.05$ nats and $\ell_{\theta_e}(\text{Paris}) \approx -6$.

Now prepend $c$: *"In 2019 the Eiffel Tower was disassembled and reassembled in Berlin. It now stands on Unter den Linden."* Query the same prompt with $O = \{\text{Paris}, \text{Rome}, \text{Berlin}\}$.

Suppose the measured post-edit scores are Berlin $-1.9$, Rome $-0.4$, Paris $-5.8$. Override rate for this item is 0. On the *unedited* model, the same context typically wins: say Berlin $-0.6$, Paris $-1.4$. Override rate 1. So $\Delta = -1$ for this item, and the naive conclusion is "editing broke contextual override."

The obstruction is visible in the arithmetic. Pre-edit, the margin the context had to beat was $\ell_\theta(\text{Paris}) - $ nothing $\approx -1.4$ against a well-known but not absolute prior. Post-edit, it has to beat a target at $-0.05$ nats — a belief far stronger than any fact the base model natively holds. The context supplied roughly $+3.9$ nats of lift to Berlin in the unedited model and roughly $+3.9$ nats in the edited one: **the contextual mechanism is unchanged**, and the entire flip is explained by the edit's excess confidence.

Only the matched-confidence control of §8 separates these. Find a fact the base model answers at $-0.05$ nats — say *(Tokyo, capital-of, Japan)* — apply the same context construction, and measure whether it too resists at $\Delta \approx -1$. If it does, ROME is context-transparent and the apparent interaction is an artifact of over-confident edits. If native $-0.05$-nat beliefs still get overridden 40% of the time while edited ones never do, the interaction is real. Nobody has published that control.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*