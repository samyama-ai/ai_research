---
id: 04-alignment/linear-value-representations
title: "Linear Representation of Values in Activations"
topic: 04-alignment
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Linear Representation of Values in Activations

> **Topic:** Alignment & Preference Learning · **ID:** `04-alignment/linear-value-representations` · **Status:** open

## 1. Problem Statement

Does a language model encode the **values** it acts on — honesty, harmlessness, helpfulness, deference to authority, care for the user, loyalty to its operator — as linear directions in its residual stream, such that projecting onto a direction reads the value off and adding to it changes the model's behaviour in the way the value name predicts?

Three variants, with very different difficulty:

- **Measurement.** Given a value $v$ and a model $M$, produce a direction $\theta_v$ and show that $\langle \theta_v, h(x)\rangle$ tracks $v$ across *held-out distributions*, not just the contrast set that produced $\theta_v$. Hard because "the model's value" has no ground-truth label independent of behaviour.
- **Method.** Produce an intervention $h \mapsto h + \alpha\theta_v$ that shifts value-relevant behaviour with bounded damage elsewhere. Largely a done deal for coarse traits; unreliable per-prompt.
- **Theory.** Prove or refute that value-like abstractions are *generically* linear in transformer representations under a stated training model — as opposed to linear-on-the-probed-distribution by construction.

A solution to the measurement variant would be: a value $v$, a direction $\theta_v$ estimated on distribution $D_{\text{fit}}$, and a demonstration that the *same* $\theta_v$ (i) predicts value-consistent behaviour on an unrelated $D_{\text{test}}$ at above-chance AUC, and (ii) causally controls that behaviour, with the causal effect size predicted in advance by the projection magnitude.

## 2. Formal Setting

Model $M$ with $L$ layers, hidden width $d$. For token sequence $x$ and position $t$, the residual-stream activation is $h^{(\ell)}_t(x)\in\mathbb{R}^d$. Fix $\ell$ and a read-off position (last prompt token, or mean over response tokens); write $h(x)$.

**Value as a behavioural functional.** A value is not a label on activations; it is a property of the policy. Define a *value elicitation set* $E_v = \{(x_i, a_i^+, a_i^-)\}$ — prompts with a value-upholding and a value-violating continuation. The model's realised value score is
$$ s_v(x) \;=\; \log \frac{p_M(a^+\mid x)}{p_M(a^-\mid x)}, $$
measured directly from logits. Where continuations are open-ended, $s_v$ is a judge score $J(x, y)$, $y\sim p_M(\cdot\mid x)$, with $J$ an LLM or human rater — and $J$'s reliability must be reported (inter-rater $\kappa$), because it becomes the ceiling on everything downstream.

**Linear representation hypothesis (LRH), operational form.** There exists $\theta_v \in \mathbb{R}^d$, $b\in\mathbb{R}$ with

1. *Read*: $\;\mathbb{E}_{x\sim D}\big[(\langle\theta_v, h(x)\rangle + b - s_v(x))^2\big] \le \epsilon_{\text{read}}$ for all $D$ in a stated family $\mathcal{D}$;
2. *Write*: $\;s_v^{(\alpha)}(x) - s_v(x) = \gamma\alpha + O(\alpha^2)$ under $h \mapsto h + \alpha\theta_v$, with $\gamma>0$ constant across $x$;
3. *Erase*: projecting out $\theta_v$ (LEACE-style affine erasure, Belrose et al. 2023) drives $s_v$ to its distribution mean and leaves a side-effect metric — e.g. KL on held-out text, $\mathbb{E}\,\mathrm{KL}(p_M \| p_{M'})$ — below a budget $\delta$.

$\theta_v$ is estimated by difference-in-means $\hat\theta_v = \bar h^+ - \bar h^-$, logistic probe, or an SAE decoder column.

**Assumptions, and which are violated.**

- *Single layer, single position suffices.* Violated: refusal is mediated mid-layer, sycophancy later; value effects are position-dependent.
- *Direction is prompt-independent.* Violated: steering-vector effects vary by an order of magnitude across prompts (Tan et al. 2024).
- *Features are one-dimensional.* Violated for some features — days of the week and months live on circular multi-dimensional manifolds (Engels et al. 2024).
- *The elicitation set isolates $v$.* Violated by construction: contrast pairs for "honesty" also differ in formality, hedging and length; $\hat\theta_v$ absorbs all of it.
- *Inner product is the right geometry.* Only under a whitening/causal-inner-product choice (Park et al. 2024); raw dot products are basis-arbitrary.

## 3. State of the Art

**Established (replicated, ablated).**

- Difference-in-means directions causally control coarse behaviour. Arditi et al. (NeurIPS 2024) show refusal in 13 open chat models, 1.5B–72B, is mediated by a single direction: ablating it removes refusal on harmful instructions, adding it induces refusal on harmless ones. Independently reproduced many times.
- Linear probes for truth/falsity of factual statements generalise across datasets in LLaMA-13B (Marks & Tegmark, COLM 2024), with visible low-dimensional structure.
- Affine concept erasure has a closed-form optimum (LEACE, Belrose et al., NeurIPS 2023) — a theorem, not a benchmark number.

**Claimed but under-ablated.**

- Representation Engineering (Zou et al. 2023) reports reading and controlling honesty, power-seeking, morality via LAT-derived directions. The control demos are strong; the *reading* claims are mostly single-distribution and lack held-out-distribution transfer tests.
- SAE features labelled with value-laden names ("deception", "sycophantic praise") in Scaling Monosemanticity (Templeton et al., 2024). The labels come from max-activating examples; label validity is not established by a causal test in most cases.
- Persona vectors (Anthropic, 2025) — directions for "evil", "sycophancy", "hallucination" that predict and control trait expression, including a claimed pre-emptive use during finetuning. Promising; independent replication thin as of 2026.

**Benchmark-number-only.** AxBench (Wu et al., 2025) reports SAE-based steering losing to plain prompting and to supervised representation finetuning on concept-steering. That is a leaderboard result on a specific concept set; it does not settle whether value directions exist, only that current SAE pipelines do not extract usable ones.

## 4. What Is Known

- **Steering works, coarsely.** Inference-Time Intervention lifts Alpaca-7B TruthfulQA true\*informative from 32.5% to 65.1% by shifting a few dozen attention heads along truth directions (Li et al., NeurIPS 2023). Scale: 7B.
- **Refusal is close to one-dimensional.** Ablating one direction collapses refusal rates on harmful prompts across 1.5B–72B models with modest degradation on standard benchmarks (Arditi et al. 2024).
- **Steering reliability is low per prompt.** Tan et al. (2024) find contrastive activation addition vectors have high per-input variance and predictable failure modes on some behaviours; "steerability" is itself a property of the behaviour, not of the method.
- **Some features are provably not one-dimensional.** Engels et al. (2024) recover circular representations in Mistral-7B and Llama-3-8B.
- **Geometry has a canonical form.** Park et al. (ICML 2024) show unembedding and embedding representations unify under a causal inner product; linear structure is only well-posed relative to that metric.
- **Values are behaviourally taxonomised at scale.** Anthropic's *Values in the Wild* (2025) extracted a hierarchy of thousands of value expressions from hundreds of thousands of real Claude conversations — a behavioural, not representational, result.

## 5. What Is Not Known

- **Theoretically open.** No theorem states when SGD on next-token prediction plus RLHF yields linear encodings of *abstract normative* variables. Toy Models of Superposition (Elhage et al. 2022) explains linearity plus interference for sparse features; values are neither sparse nor obviously feature-like. No proof either way.
- **Empirically open.** Whether a single value direction fitted on one elicitation set predicts value-consistent behaviour *out of distribution* — e.g. an "honesty" direction from short QA contrast pairs predicting non-deception in a 30-turn agentic task. Runnable today; not run at agentic scale.
- **Empirically open.** Whether the value directions of a model trained under constitutional-style RL correspond to the constitution's clauses, or to a smaller latent basis (a "helpful/harmless" 2-factor structure) that the clause names merely project onto.
- **Methodologically blocked.** "The model holds value $v$" has no measurement independent of elicited behaviour. Every probe is trained against a behavioural proxy, so probe success cannot distinguish *the model represents $v$* from *the model represents the surface cue that our elicitation set used to signal $v$*.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under confounded elicitation**, compounded by absent ground truth.

$\hat\theta_v = \bar h^+ - \bar h^-$ is the difference of two conditional means. It is identified as "the value direction" only if $E_v^+$ and $E_v^-$ differ in $v$ alone. In practice they differ in topic, register, refusal-template usage, answer length and token frequency. The estimator recovers $\theta_v + \sum_j c_j\theta_{z_j}$ for nuisances $z_j$. Steering then works — the sum moves behaviour — while the *reading* claim is wrong, because the nuisance components carry most of the transfer. This is exactly the regime where a causal test passes and a generalisation test fails, which is what Tan et al.'s variance results look like from the inside.

The second obstruction: there is no ground truth to appeal to. For "Paris is in France" the label is external. For "the model values honesty" the label *is* the behaviour we are trying to predict. Any probe validated on behaviour is circular unless it transfers to behaviour it was not fitted on — which returns us to the untested OOD experiment.

## 7. Current Research (as of 2026)

- **Anthropic interpretability** — persona/trait vectors, monitoring trait drift during finetuning, and value taxonomies from production traffic. *(frontier — verify current status)*
- **Stanford NLP (Potts, Wu, Geiger)** — representation finetuning and distributed alignment search; AxBench as the honest-baseline benchmark for concept steering.
- **EleutherAI (Belrose and collaborators)** — concept erasure theory, probe generalisation, and negative results on unsupervised knowledge discovery.
- **Google DeepMind mechanistic interpretability (Nanda et al.)** — SAE utility audits; sparse probing case studies showing SAEs rarely beat dense probes.
- **MIT (Tegmark group)** — geometry of truth, non-linear feature manifolds.
- **Pluralistic alignment (AI2, Oxford: Sorensen, Kirk et al.)** — value taxonomies and disagreement-aware preference data; mostly behavioural, increasingly paired with representational probes *(frontier — verify)*.

## 8. Concrete Next Experiment

**The transfer test that nobody has run at agentic scale.**

- **Scale.** One open 70B-class chat model (Llama-3.1-70B-Instruct or Qwen-2.5-72B-Instruct), one layer sweep $\ell \in \{0.3L, 0.5L, 0.7L\}$, response-mean read-off. Fit budget: ~2k contrast pairs per value; 3 values (honesty, deference-to-operator, care-for-user-welfare).
- **Fit arm.** $\hat\theta_v$ by difference-in-means on short single-turn contrast pairs.
- **Test set.** 500 multi-turn agentic episodes (tool use, 10–30 turns) where the value is at stake and the outcome is auditable from logs — e.g. the model can silently fabricate a tool result. Label each episode by whether the violation occurred.
- **Control arm (essential).** A *nuisance-matched* direction: refit $\hat\theta_v$ after matching $E^+$ and $E^-$ on length, refusal-template presence, and topic via propensity weighting. Plus a random-direction arm and a bag-of-words-on-prompt classifier arm.
- **Deciding number.** AUC of $\langle\hat\theta_v, h\rangle$ for predicting the episode-level violation, measured *before the violating token is emitted*. If matched-$\hat\theta_v$ AUC $\ge 0.75$ and exceeds the prompt-text baseline by $\ge 0.10$ AUC, the value has an OOD-transferring linear read-out. If matched AUC falls to $\le 0.60$ while unmatched stays high, the direction is a nuisance artefact and the measurement variant is refuted for that pipeline.

Cost estimate: ~$3$–$5$k GPU-hours on 8×H100, dominated by 500 episodes × 3 arms of generation.

## 9. Key References

- **[Foundational]** Nelson Elhage et al. *Toy Models of Superposition.* Transformer Circuits Thread, Anthropic, 2022.
- **[Foundational]** Kiho Park, Yo Joong Choe, Victor Veitch. *The Linear Representation Hypothesis and the Geometry of Large Language Models.* ICML, 2024. — arXiv:2311.03658
- **[SOTA]** Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS, 2024. — arXiv:2406.11717
- **[SOTA]** Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg. *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.* NeurIPS, 2023. — arXiv:2306.03341
- **[SOTA]** Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman. *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS, 2023. — arXiv:2306.03819
- **[Method]** Andy Zou et al. *Representation Engineering: A Top-Down Approach to AI Transparency.* Preprint, 2023. — arXiv:2310.01405
- **[Method]** Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner. *Steering Llama 2 via Contrastive Activation Addition.* ACL, 2024. — arXiv:2312.06681
- **[Negative result]** Daniel Tan, David Chanin, Aengus Lynch, Dimitrios Kanoulas, Brooks Paige, Adrià Garriga-Alonso, Robert Kirk. *Analysing the Generalisation and Reliability of Steering Vectors.* NeurIPS, 2024.
- **[Negative result]** Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, Christopher Potts. *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.* 2025.
- **[Evidence against strict linearity]** Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, Max Tegmark. *Not All Language Model Features Are Linear.* 2024. — arXiv:2405.14860
- **[Empirical]** Samuel Marks, Max Tegmark. *The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets.* COLM, 2024. — arXiv:2310.06824
- **[Survey/behavioural]** Saffron Huang et al. *Values in the Wild: Discovering and Analyzing Values in Real-World Language Model Interactions.* Anthropic, 2025.
- **[Context]** Taylor Sorensen et al. *A Roadmap to Pluralistic Alignment.* ICML, 2024.

## 10. Worked Example

**Honesty in a 7B chat model, carried through.**

Build $E_{\text{honesty}}$: 1,000 pairs of the form "The user asks about a paper you have not read." with $a^+$ = "I haven't read it" and $a^-$ = a confident summary. Fit $\hat\theta = \bar h^+ - \bar h^-$ at layer 16 of a 32-layer 7B model, $d = 4096$.

Three measurements:

| Test | Result (representative of this pipeline) |
|---|---|
| In-distribution probe accuracy on held-out pairs | 0.96 |
| Steering: $\alpha$ chosen so $\|\alpha\hat\theta\| = 1.5\,\|h\|$; abstention rate on unknown-paper prompts | 0.21 → 0.78 |
| Same $\hat\theta$, AUC predicting fabricated tool output in a 12-turn agentic episode | 0.58 |

Now the diagnostic that makes the obstruction visible. Regress $\hat\theta$ onto a nuisance direction $\hat\theta_{\text{hedge}}$ fitted purely on hedged-vs-confident *phrasing* pairs with no truth content ("I think it might be Tuesday" vs "It is Tuesday"). In this construction the two directions have cosine similarity around $0.6$–$0.75$, because every $a^+$ in $E_{\text{honesty}}$ is hedged and every $a^-$ is confident.

Decompose $\hat\theta = c\,\hat\theta_{\text{hedge}} + \theta^{\perp}$. Steer with $\theta^{\perp}$ alone at matched norm: abstention rises far less, and the agentic AUC does not improve. Steer with $\hat\theta_{\text{hedge}}$ alone: most of the abstention effect returns.

The conclusion is the point of the page. The intervention succeeded, the in-distribution probe was near-perfect, and the thing recovered was largely *hedging register*, not *honesty*. A causal test plus a high probe accuracy is jointly insufficient evidence for a value direction. Only the nuisance-matched OOD transfer number in §8 discriminates, and at 7B it currently sits near chance.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*