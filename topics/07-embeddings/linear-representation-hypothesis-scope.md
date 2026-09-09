---
id: 07-embeddings/linear-representation-hypothesis-scope
title: "The Linear Representation Hypothesis Beyond Toy Settings"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# The Linear Representation Hypothesis Beyond Toy Settings

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/linear-representation-hypothesis-scope` · **Status:** open

## 1. Problem Statement

The linear representation hypothesis (LRH) says a neural network encodes human-interpretable concepts as directions in activation space: a concept $c$ has a vector $v_c$ such that the concept's value is an affine function of the activation, and adding $\alpha v_c$ changes the model's behaviour on $c$ and nothing else. The hypothesis was formed on word embeddings and toy models. The open problem is its **scope**: which concepts, in which models, at which layers, are linear in this sense — and what the non-linear remainder looks like.

Three variants, routinely conflated:

- **Measurement.** Given a model, a layer, and a concept, produce a test that decides "linear or not" and does not pass trivially. Linear *decodability* is close to vacuous (§10); the content of LRH is linear *causal mediation*.
- **Method.** Find the directions. Supervised (difference-in-means, probes), unsupervised (PCA, sparse autoencoders), or causal (distributed alignment search).
- **Theory.** Prove when gradient descent on next-token prediction yields linear structure. Existing results assume latent-variable data-generating processes that language does not satisfy.

Solving it means: a stated, falsifiable predicate; a measured fraction of model behaviour it covers at frontier scale; and a characterisation of what the residual is.

## 2. Formal Setting

Let $f$ be a transformer, $h_\ell(x) \in \mathbb{R}^d$ the residual-stream activation at layer $\ell$ on prompt $x$, and $W_U \in \mathbb{R}^{|V| \times d}$ the unembedding. A concept $c$ takes values in $\mathcal{Y}_c$.

**Subspace (decoding) form.** There is $v_c \in \mathbb{R}^d$, $b \in \mathbb{R}$ with
$$\Pr[\,Y_c = 1 \mid x\,] = \sigma(\langle v_c, h_\ell(x)\rangle + b).$$
Measured as: probe accuracy on held-out prompts, reported against a **control task** — the same probe fit to random labels with the same label-count and cardinality (Hewitt & Liang, EMNLP 2019). The reported quantity is *selectivity* = accuracy − control accuracy, not accuracy.

**Intervention (causal) form.** For a counterfactual pair $(x, x')$ differing only in $c$,
$$f\big(h_\ell(x) + \alpha v_c\big) \approx f\big(h_\ell(x')\big), \qquad \text{measured as } \ \Delta_{\text{target}} = \log\frac{p(y'\mid \cdot)}{p(y\mid\cdot)}$$
with a **side-effect** term: $\mathrm{KL}\!\left(f(h+\alpha v_c) \,\|\, f(h)\right)$ restricted to tokens irrelevant to $c$. LRH is the claim that $\Delta_{\text{target}}$ is large while the side-effect KL stays near the KL of a random-direction control of equal norm.

**Geometry.** Inner products are not canonical: the residual stream has no privileged basis and no privileged metric. Park, Choe & Veitch (ICML 2024) define a **causal inner product** $\langle u,v\rangle_C = u^\top M v$ with $M = \mathrm{Cov}(\gamma)^{-1}$ over unembedding vectors $\gamma$, chosen so causally separable concepts are orthogonal. Every orthogonality or "same direction" claim is relative to a choice of $M$; results reported in the Euclidean metric are a different claim.

**Assumptions known to be violated.**
1. *Concepts are binary and independent.* False — concepts are hierarchical and correlated (Park et al., ICLR 2025, model hierarchy as polytopes, not free directions).
2. *One direction per concept.* False for cyclic concepts: day-of-week and month are 2D circles, not lines (Engels et al., ICML 2025).
3. *Isotropy.* Contextual embeddings are strongly anisotropic; a dominant "rogue" direction inflates all cosine similarities (Ethayarajh, EMNLP 2019).
4. *Additivity across layers.* Attention and LayerNorm are non-linear in $h$; LayerNorm rescaling alone makes $\alpha$ non-linear in effect size.
5. *$k \gg d$.* Almost always false — see §10.

## 3. State of the Art

**Established (replicated, with controls or causal tests).**
- Difference-in-means directions steer behaviour. Arditi et al. (NeurIPS 2024) show refusal in chat models is mediated by a single direction: ablating it jailbreaks, adding it induces refusal, across 13 open models from 1.8B to 72B parameters. Reproduced widely.
- Contrastive activation addition (Rimsky et al., ACL 2024) and representation engineering (Zou et al., 2023) produce reliable, sign-consistent behavioural shifts on sycophancy, honesty and hallucination axes.
- Some concepts are *not* linear. Engels et al. (ICML 2025) find irreducible multi-dimensional circular features for weekday and month in Mistral 7B and Llama 3 8B, and show intervention on the circle predicts modular-arithmetic behaviour that a 1D direction does not.

**Claimed but unablated / benchmark-only.**
- "Sparse autoencoders find the model's features." Scaling results exist — Templeton et al. (Anthropic, 2024) at 34M features on Claude 3 Sonnet; Gao et al. (OpenAI, 2024, arXiv:2406.04093) at 16M latents on GPT-4 — but these report reconstruction loss, sparsity and cherry-picked interpretable latents. Downstream utility is unablated or negative: Kantamneni et al. (ICML 2025) find SAE probes do not beat simple baselines across ~100 sparse-probing tasks. Heap et al. (2025) find SAEs yield interpretable-looking latents on *randomly initialised* transformers, which removes interpretability-of-latents as evidence for LRH.
- "Truth is a direction." Marks & Tegmark (COLM 2024) find high-accuracy, transferable truth directions in LLaMA-13B, but generalisation across datasets is partial and the causal test is on a narrow factual set.

**Theory SOTA.** Jiang et al. (ICML 2024, arXiv:2403.03867) derive linear representations from a latent-variable next-token model with implicit-bias arguments; Arora et al. (TACL 2016) derive PMI-linearity under a random-walk discourse model. Both assume generative structure real corpora do not have.

## 4. What Is Known

- **Analogy arithmetic is weaker than advertised.** `king − man + woman ≈ queen` holds largely because the query word is excluded from the nearest-neighbour search; without exclusion, accuracy collapses (Levy & Goldberg, CoNLL 2014; Nissim et al., *Computational Linguistics* 2020). Ethayarajh et al. (ACL 2019) and Allen & Hospedales (ICML 2019) show the linear analogy identity requires a co-occurrence-shifted-PMI condition that most word pairs fail.
- **Superposition is real in toy models.** Elhage et al. (2022) show a 2-layer ReLU model packs $n > d$ features into $d$ dimensions as near-orthogonal directions, with predictable polytope geometry.
- **World models are linearly probed.** Othello-GPT's board state is linearly decodable when the probe target is *mine/theirs* rather than *black/white* (Nanda et al., BlackboxNLP 2023) — a result that flipped an earlier non-linear conclusion (Li et al., ICLR 2023), showing linearity claims are sensitive to concept parameterisation.
- **Space and time are linear-ish.** Gurnee & Tegmark (ICLR 2024) recover world coordinates and event dates from Llama-2 7B/13B/70B with linear probes, accuracy rising with scale and saturating in middle layers.
- **Anisotropy.** In GPT-2's upper layers, mean cosine similarity between random tokens exceeds 0.5 (Ethayarajh, EMNLP 2019), so raw cosine geometry is uninformative without whitening.

## 5. What Is Not Known

- **Methodologically blocked (the main blockage).** There is no agreed, non-vacuous predicate for "concept $c$ is linear in model $M$". Decoding tests are trivially satisfiable (§10); intervention tests depend on an arbitrary metric $M$, an arbitrary $\alpha$, and an arbitrary side-effect budget. No standard reports selectivity against a matched random-direction control at fixed intervention norm.
- **Empirically open.** What *fraction* of behaviourally relevant concepts in a frontier model are linear? Nobody has run a census: a fixed concept inventory, a fixed causal test, applied at 7B / 70B / 400B. Runnable today; unrun.
- **Theoretically open.** No proof that next-token prediction on natural text with a transformer must produce linear concept encoding, and no proof of the converse. The known derivations assume a latent structure assumed, not verified.
- Also open: whether SAE latents correspond to model-internal computational variables at all, given feature absorption and splitting (Chanin et al., 2024).

## 6. Why It Is Hard

**Confounded measurement plus non-identifiability**, jointly.

1. *Vacuity of decodability.* With $k$ labelled prompts in general position and $k \le d+1$, every dichotomy is linearly separable. Typical probing sets have $k \sim 10^3$ and $d \sim 4\times10^3$–$10^4$. The probe's success is a property of the sample size, not the model.
2. *No canonical metric.* "Direction" is meaningful only under a chosen inner product, and the choice is itself the hypothesis. Two papers can disagree on whether concepts are orthogonal purely from a whitening step.
3. *Absent ground truth.* There is no independent list of the model's concepts. The concept inventory is supplied by humans, so "we found linear directions for the concepts we looked for" cannot bound coverage.
4. *Compute.* A causal census over $10^3$ concepts × 80 layers × 3 model scales × counterfactual pairs is $10^6$–$10^7$ forward passes per scale — feasible at 7B, expensive at 400B, impossible for a lab without weights access at frontier scale.

## 7. Current Research (as of 2026)

- **Geometry-first.** Veitch's group (Chicago) on causal inner products, hierarchical polytopes, and categorical concepts.
- **Non-linear features.** Tegmark's group (MIT) on irreducible multi-dimensional features and manifold-valued concepts; circular and toroidal structure for cyclic variables.
- **SAE validation, not scaling.** The field's centre of gravity has shifted from bigger dictionaries to whether latents survive causal tests — random-init controls, feature absorption, downstream-task ablations. *(frontier — verify)* Anthropic and DeepMind interpretability teams both report attribution/transcoder-based circuit work displacing pure SAE feature counts.
- **Causal abstraction.** Geiger, Potts and colleagues (Stanford) on distributed alignment search: search over rotations for a subspace that supports an intervention, which turns linearity into an explicit optimisation with an explicit failure mode (overfitting the rotation).

## 8. Concrete Next Experiment

**A linearity census with matched controls.**

- **Scale.** Llama-3.1 8B and 70B, layers $\{0.25L, 0.5L, 0.75L\}$. Concept inventory: 300 binary concepts, each with 200 minimal counterfactual prompt pairs (differing in exactly the concept), drawn from existing counterfactual suites plus templated generation. ~$10^6$ forward passes; roughly 2 GPU-days on 8×H100 at 8B, ~2 GPU-weeks at 70B.
- **Procedure.** For each concept: fit $v_c$ by difference-in-means on train pairs; whiten with the causal inner product. On held-out pairs, sweep $\alpha$ to the value achieving median target logit-difference $\Delta_{\text{target}} = 2$ nats, then record $\mathrm{KL}_{\text{off-target}}$ on 500 unrelated prompts.
- **Control arm.** Random directions of identical whitened norm, plus the *concept-shuffled* control (direction from a different concept), swept to the same $\alpha$ norm. Also a non-linear ceiling: a rank-2 subspace intervention (as for circular features) on the same pairs.
- **Deciding number.** The **linear coverage fraction** $\rho$ = share of the 300 concepts for which $\Delta_{\text{target}} \ge 2$ nats at $\mathrm{KL}_{\text{off-target}} \le 0.1$ nats, while the random control at the same norm achieves $\Delta_{\text{target}} \le 0.2$ nats. $\rho > 0.8$ at both scales supports LRH as a working assumption; $\rho < 0.4$, especially with the rank-2 arm materially higher, refutes the strong form and localises the residual.

## 9. Key References

- **[Foundational]** T. Mikolov, W. Yih, G. Zweig. *Linguistic Regularities in Continuous Space Word Representations.* NAACL-HLT, 2013.
- **[Foundational]** S. Arora, Y. Li, Y. Liang, T. Ma, A. Risteski. *A Latent Variable Model Approach to PMI-based Word Embeddings.* TACL, 2016. — arXiv:1502.03520
- **[Foundational]** N. Elhage et al. *Toy Models of Superposition.* Transformer Circuits Thread, Anthropic, 2022.
- **[SOTA]** K. Park, Y. J. Choe, V. Veitch. *The Linear Representation Hypothesis and the Geometry of Large Language Models.* ICML, 2024. — arXiv:2311.03658
- **[SOTA]** K. Park, Y. J. Choe, Y. Jiang, V. Veitch. *The Geometry of Categorical and Hierarchical Concepts in Large Language Models.* ICLR, 2025. — arXiv:2406.01506
- **[SOTA]** J. Engels, E. J. Michaud, I. Liao, W. Gurnee, M. Tegmark. *Not All Language Model Features Are Linear.* ICML, 2025. — arXiv:2405.14860
- **[SOTA]** A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, N. Nanda. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS, 2024. — arXiv:2406.11717
- **[Method]** J. Hewitt, P. Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP, 2019.
- **[Method]** H. Cunningham, A. Ewart, L. Riggs, R. Huben, L. Sharkey. *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* ICLR, 2024. — arXiv:2309.08600
- **[Evidence against]** S. Kantamneni, J. Engels, S. Rajamanoharan, M. Tegmark, N. Nanda. *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing.* ICML, 2025. — arXiv:2502.16681
- **[Critique]** M. Nissim, R. van Noord, R. van der Goot. *Fair Is Better than Sensational: Man Is to Doctor as Woman Is to Doctor.* Computational Linguistics, 2020.
- **[Survey]** Y. Belinkov. *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics, 2022.

## 10. Worked Example

**Day-of-week in a 7B model.** Take Mistral-7B, $d = 4096$, layer 16. Prompt template: "Two days after Monday is". Seven concept values.

*Step 1 — the decoding test passes, and means nothing.* Average activations per weekday gives $k = 7$ points in $\mathbb{R}^{4096}$. Since $k \le d+1$ and the points are in general position, **all $2^7 = 128$ binary labellings are linearly separable** — including "Tuesday or Friday vs. rest", which corresponds to no concept. A one-vs-rest probe reporting 100% accuracy on "is-Wednesday" is reporting the dimension count, not the model. The control-task version (random 7-way labels over the same activations) also separates perfectly, so selectivity is $\approx 0$.

*Step 2 — the causal test fails in a structured way.* Fit $v_{\text{day}}$ by regressing the ordinal index $0..6$ on $h_{16}$. Adding $\alpha v_{\text{day}}$ shifts predictions forward through the week and then breaks at the wrap: pushing from Saturday should reach Sunday, but a linear direction extrapolates past the manifold and the next-token distribution degrades into unrelated tokens, with off-target KL rising sharply rather than staying flat.

*Step 3 — the correct object is a circle.* PCA on the seven mean activations puts the bulk of the variance in two components, with the seven points arranged at roughly equal angles. Intervening by *rotating* within that 2D plane reproduces modular arithmetic including the wrap, which the 1D direction cannot (Engels et al., ICML 2025).

**The obstruction made visible.** Steps 1 and 3 point in opposite directions while both are "linear probing succeeded". Decodability was satisfied by a concept the model represents on a circle, because $k \ll d$ makes separability free. Any linearity census that reports probe accuracy — and most do — would score weekday as a confirming case for LRH. Only the intervention test, with a random-direction control at matched norm and an off-target KL budget, separates the two. That is why §8 measures $\rho$ and not accuracy.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*