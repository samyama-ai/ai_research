---
id: 20-interpretability/feature-absorption-sparse-dictionaries
title: "Feature Absorption in Sparse Dictionaries"
topic: 20-interpretability
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Feature Absorption in Sparse Dictionaries

> **Topic:** Interpretability · **ID:** `20-interpretability/feature-absorption-sparse-dictionaries` · **Status:** open

## 1. Problem Statement

A sparse autoencoder (SAE) trained on language-model activations is meant to recover a dictionary of interpretable directions. **Feature absorption** is the failure mode where a latent that appears to encode a general concept ("token starts with the letter L") silently stops firing on a subset of its own extension, because a second, more specific latent ("lion") has absorbed the general direction into itself. The general latent then looks clean under manual inspection — every token it fires on does start with L — while being systematically incomplete. Absorption is not noise; it is a sparsity-optimal reparameterisation. Encoding `lion` with one latent costs $L_0 = 1$; encoding it as `starts-with-L` + `lion-residual` costs $L_0 = 2$.

Three variants, of very different difficulty:

- **Measurement.** Given a trained SAE and a concept $c$, output the fraction of $c$-positive inputs on which the nominal $c$-latent is silent *and* the reconstruction still carries the $c$ direction. Requires a ground-truth $c$ and a way to read "carries the direction" off the reconstruction. Currently defined only for concepts with cheap external labels (first letter, part of speech).
- **Method.** Train a dictionary with comparable reconstruction–sparsity trade-off and materially lower absorption. Partially achieved.
- **Theory.** Characterise when the sparsity-penalised objective has a minimiser that is the true feature set, versus when absorbed reparameterisations strictly dominate. Open.

Solving it means: an SAE training objective under which the recovered dictionary's latents have **extensionally complete** supports, plus a metric that certifies completeness without a labelled concept list.

## 2. Formal Setting

Let $x \in \mathbb{R}^d$ be a residual-stream activation, sampled from the model's activation distribution $\mathcal{D}$ over a token corpus. An SAE with $m$ latents ($m \gg d$) computes

$$ z = \sigma(W_{\text{enc}} x + b_{\text{enc}}), \qquad \hat{x} = W_{\text{dec}}^\top z + b_{\text{dec}}, $$

with $\sigma$ a ReLU, JumpReLU, or TopK gate, and decoder columns $\mathbf{d}_i$ unit-normed. Training minimises

$$ \mathcal{L} = \mathbb{E}_{x\sim\mathcal{D}}\big[\lVert x - \hat{x} \rVert_2^2 + \lambda \, \mathcal{S}(z)\big], \quad \mathcal{S}(z) = \lVert z\rVert_1 \ \text{or}\ \lVert z \rVert_0 .$$

**Concept and probe.** A concept $c$ is a labelled binary function on tokens, $y_c(t) \in \{0,1\}$. Measure its direction by fitting a logistic-regression probe $p_c \in \mathbb{R}^d$, $\lVert p_c\rVert = 1$, on raw activations $x$ — *not* on SAE latents. Probe accuracy on held-out tokens is reported alongside; below ~0.9 the measurement is meaningless.

**Nominal latent.** $i^*(c) = \arg\max_i \ \text{(mean $F_1$ of } \mathbb{1}[z_i > 0] \text{ against } y_c)$, or the top-$k$ set by cosine $\langle \mathbf{d}_i, p_c\rangle$.

**Absorption event.** For a token $t$ with $y_c(t)=1$, absorption is recorded when

1. $z_{i^*}(x_t) = 0$ (the nominal latent is silent), and
2. some other latent $j$ carries the concept direction: $\langle \mathbf{d}_j, p_c \rangle > \tau$ (typically $\tau \approx 0.025$–$0.1$) and its ablation destroys the concept readout, $\ \Delta_j = p_c^\top(\hat{x} - \hat{x}^{\setminus j}) \ge \alpha \cdot p_c^\top \hat{x}$ with $\alpha \approx 0.3$, and
3. $j$ is *token-specific*: it fires on a small, non-$c$-aligned support.

**Absorption rate.** $A_c = \Pr_{t \sim \mathcal{D}}\big[\text{absorption} \mid y_c(t)=1\big]$; a dictionary-level score averages $A_c$ over a concept set (26 letters, in the standard instantiation).

**Assumptions, with the violated ones flagged.**

- *Linear representation*: $c$ is a direction. Known violated for circular/multi-dimensional features (Engels et al., 2024) — day-of-week, month.
- *One latent per concept*: the metric needs a nominal $i^*$. Violated by feature splitting; the standard fix (take top-$k$ by cosine) makes $A_c$ depend on $k$.
- *Probe direction is the ground-truth concept direction*: **known false in general**. The probe is a supervised summary of what is linearly decodable, not of what the model uses.
- *Latents are ranked by an unambiguous score*: violated under shrinkage — $L_1$-trained SAEs systematically underestimate $z_i$, distorting the ablation term $\Delta_j$.

## 3. State of the Art

**Established.** Chanin et al. (2024, arXiv:2409.14507) named and operationalised absorption on the first-letter task, and showed it in Gemma Scope and GPT-2-small SAEs across widths and sparsities. Their central ablation is solid: absorbing latents have small but non-zero cosine with the letter probe, they fire on single tokens, and ablating them removes the letter information that the nominal latent never supplied. Absorption is *not* explained by probe error alone, because the nominal latent's own recall is what drops.

**Established, method side.** Matryoshka SAEs (Bussmann, Leask, Nanda, 2025, arXiv:2503.17547) train nested prefixes of the dictionary with a reconstruction loss at each prefix, so early latents must reconstruct alone and cannot be relieved by later token-specific latents. Reported large reductions in SAEBench absorption at a modest reconstruction cost. This is the strongest existing mitigation.

**Claimed but unablated.** That lower absorption implies better downstream utility. SAEBench (Karvonen et al., 2025, arXiv:2503.09532) reports absorption as one of eight metrics and finds it **partly anti-correlated** with sparse-probing and unlearning scores — no causal ablation links absorption reduction to a task gain. Also unablated: that absorption is the mechanism behind SAEs' underperformance on steering (Wu et al., AxBench, arXiv:2501.17148, 2025) rather than a correlate.

**Benchmark-number-only.** Every published absorption figure comes from the first-letter probe family in SAEBench. No absorption number exists for a concept set that was not chosen because it has cheap token-level labels.

**Theory SOTA** is classical sparse coding, not SAEs: uniqueness of the sparsest representation under spark/coherence conditions (Donoho & Elad, PNAS 2003) and exact dictionary recovery for sparsely-used dictionaries (Spielman, Wang, Wright, COLT 2012). Both assume $\ell_0$-sparse *linear* generation with a fixed dictionary and independent supports. Language features are hierarchical and correlated, so neither applies, and no absorption-specific identifiability theorem exists.

## 4. What Is Known

- Absorption is pervasive at frontier SAE scale. Measured on Gemma-2-2B residual-stream SAEs (Gemma Scope, 16k and 65k latents, layers 0–25) and GPT-2-small, over the 26 first-letter concepts: for many letters the nominal latent's recall drops well below its precision, with per-letter absorption rates spanning roughly a few percent to over half of positive tokens depending on letter, layer and sparsity (Chanin et al., 2024).
- Absorption is **worse at larger dictionary width and at lower $L_0$** — exactly the regime that reconstruction–sparsity Pareto curves reward. This is the key finding: the standard training target selects for absorption.
- It is **not** an architecture artefact. It appears in ReLU, Gated (Rajamanoharan et al., arXiv:2404.16014), JumpReLU (arXiv:2407.14435) and TopK (Gao et al., arXiv:2406.04093, ICLR 2025) SAEs.
- Absorbed latents are typically single-token or few-token, with cosine similarity to the concept probe in the low-tens-of-a-percent range — small enough that a coherence-based dictionary audit would not flag them.
- Dictionaries are not canonical: SAEs of different widths trained on identical data yield latents that are neither refinements nor coarsenings of each other (Leask et al., arXiv:2502.04878, 2025). Absorption is one mechanism for this instability.
- Matryoshka training reduces SAEBench absorption scores substantially on Gemma-2-2B while paying a measurable reconstruction penalty at matched $L_0$.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no absorption metric that does not require a hand-labelled concept. Every number in the literature is conditioned on the first-letter task. Whether absorption rates on letters predict absorption on the features that actually matter for behaviour — syntactic roles, refusal, entity types — is unmeasured, and there is currently no measurement procedure for it.
- **Theoretically open.** No theorem states conditions under which the $\ell_1$- or $\ell_0$-penalised SAE objective has the true hierarchical feature set as a global minimiser. The informal argument that absorption is sparsity-optimal has not been turned into a separation result: nobody has exhibited a generative model plus a proof that every global minimiser at a given $L_0$ budget is absorbed.
- **Empirically open.** Whether reducing absorption improves any downstream interpretability task at fixed reconstruction quality. Runnable today: Matryoshka vs. matched-$L_0$ baseline on circuit discovery or unlearning. Nobody has run it as a controlled comparison.
- **Empirically open.** Whether absorption in a transcoder or cross-layer decomposition (attribution-graph style) differs in kind from residual-stream SAE absorption.

## 6. Why It Is Hard

**Absent ground truth, compounded by an evaluation that does not measure what it names.** The definition requires knowing the true feature set to say a latent's support is incomplete — but recovering the true feature set is the problem the SAE exists to solve. The field escapes this circularity by substituting a supervised probe $p_c$ for the true direction. That substitution imports the probe's own errors into the metric: if the model genuinely has no unified "starts with L" feature and instead uses many letter-adjacent features, the probe still finds a decodable direction, and the SAE will be scored as absorbing a feature that was never there.

Second obstruction: **non-identifiability**. Absorbed and non-absorbed dictionaries can achieve the *same* loss. Where $\mathbf{d}_{\text{lion}} = \mathbf{d}_{\text{L}} + \mathbf{r}$, both parameterisations reconstruct exactly; the absorbed one has lower $L_0$. There is nothing in the objective to break the tie in favour of the interpretable solution, so this is a specification failure, not an optimisation failure — better optimisers make it worse.

Compute is *not* the obstruction: absorption is visible on GPT-2-small SAEs trainable in GPU-hours.

## 7. Current Research (as of 2026)

- **Nested/hierarchical dictionaries.** Matryoshka SAEs (Bussmann, Leask, Nanda) and variants that impose a prefix ordering so general features cannot outsource to specific ones. Actively extended to transcoders *(frontier — verify)*.
- **Metrics beyond first-letter.** Extending SAEBench's absorption probe to multi-token and syntactic concepts; the blocker is label quality, not compute.
- **Attribution graphs / cross-layer transcoders** (Anthropic circuits work, 2025). Absorption reframed as a graph-edge artefact rather than a per-latent property *(frontier — verify)*.
- **Parameter decomposition** approaches (Apollo Research) that decompose weights rather than activations, sidestepping the activation-sparsity objective that creates the incentive *(frontier — verify)*.
- **Open Problems in Mechanistic Interpretability** (Sharkey et al., arXiv:2501.16496, 2025) lists absorption and non-canonicality among the central SAE failures.

## 8. Concrete Next Experiment

**Question.** Does reducing absorption buy anything downstream, or is it a metric optimised for its own sake?

**Scale.** Gemma-2-2B, residual stream layer 12, dictionary width 65k. Train three SAEs to matched $L_0 = 40 \pm 2$ and matched fraction-of-variance-unexplained within 1 percentage point: (a) JumpReLU baseline, (b) Matryoshka, (c) **control arm** — JumpReLU with a decoder-orthogonality penalty $\mu \sum_{i \ne j} \langle \mathbf{d}_i, \mathbf{d}_j\rangle^2$ tuned to match Matryoshka's *absorption* score. Cost: roughly 3 × 100 GPU-hours on A100s, plus eval.

**Evaluation.** Held-out task not used for tuning: SAEBench sparse probing over 20 concepts *and* a causal task — patch-based circuit recovery on the indirect-object-identification analogue in Gemma-2-2B, scored by faithfulness of the recovered latent set at fixed node budget.

**Deciding number.** The **faithfulness gap** $\Delta F = F_{\text{Matryoshka}} - F_{\text{JumpReLU}}$ at a 50-node budget. If $\Delta F \ge 0.05$ (5 points of recovered logit-difference fraction) *and* the orthogonality control (c) does **not** reproduce it despite matching absorption score, absorption reduction is causally useful and the metric tracks something real. If $\Delta F < 0.02$, or if control (c) matches Matryoshka's faithfulness, then the absorption score is a probe artefact and the field should stop treating it as a training target.

## 9. Key References

- **[Foundational]** Bricken, Templeton, Batson, et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread, 2023.
- **[Foundational]** Cunningham, Ewart, Riggs, Huben, Sharkey. *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* ICLR 2024. — arXiv:2309.08600
- **[SOTA / defining paper]** Chanin, Wilken-Smith, Dulka, Bhatnagar, Bloom. *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders.* 2024. — arXiv:2409.14507
- **[SOTA / mitigation]** Bussmann, Leask, Nanda. *Learning Multi-Level Features with Matryoshka Sparse Autoencoders.* 2025. — arXiv:2503.17547
- **[Benchmark]** Karvonen, Rager, Lin, et al. *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability.* 2025. — arXiv:2503.09532
- **[Related failure]** Leask, Bussmann, Pearce, Bloom, Nanda, et al. *Sparse Autoencoders Do Not Find Canonical Units of Analysis.* 2025. — arXiv:2502.04878
- **[Architecture]** Rajamanoharan, Lieberum, Sonnerat, et al. *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders.* 2024. — arXiv:2407.14435
- **[Architecture]** Gao, la Tour, Tillman, et al. *Scaling and Evaluating Sparse Autoencoders.* ICLR 2025. — arXiv:2406.04093
- **[Artifacts]** Lieberum, Rajamanoharan, Conmy, et al. *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.* BlackboxNLP 2024. — arXiv:2408.05147
- **[Negative result]** Wu, Geiger, Arora, et al. *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.* 2025. — arXiv:2501.17148
- **[Theory background]** Spielman, Wang, Wright. *Exact Recovery of Sparsely-Used Dictionaries.* COLT 2012.
- **[Theory background]** Donoho, Elad. *Optimally sparse representation in general (nonorthogonal) dictionaries via $\ell_1$ minimization.* PNAS 100(5), 2003.
- **[Survey]** Sharkey, Chughtai, Batson, et al. *Open Problems in Mechanistic Interpretability.* 2025. — arXiv:2501.16496

## 10. Worked Example

Take the token `" lion"` in Gemma-2-2B, layer 12 residual stream, Gemma Scope 16k SAE. Fit $p_L$, a first-letter-L probe, on raw activations: held-out accuracy ~0.97, so the direction is real and decodable.

Now decompose. Write the projection of the reconstruction onto the probe, $p_L^\top \hat{x} = \sum_i z_i \langle \mathbf{d}_i, p_L\rangle$, and look at the two candidate contributors:

| latent | fires on `" lion"`? | $\langle \mathbf{d}_i, p_L\rangle$ | $z_i$ | contribution to $p_L^\top\hat x$ |
|---|---|---|---|---|
| $i^*$ = "starts with L" | **no** | 0.42 | 0.0 | 0.00 |
| $j$ = "lion" | yes | 0.06 | 9.5 | 0.57 |
| all others | — | — | — | ~0.10 |

(Illustrative magnitudes in the range Chanin et al. report; the structure — silent nominal latent, small-cosine token latent supplying the direction — is the measured finding.)

The letter information is present in $\hat{x}$ and the letter latent supplied none of it. Ablate $j$ and the L-readout collapses by ~85%. On `" lamp"`, by contrast, $i^*$ fires at $z_{i^*} = 6.1$ and no token latent carries L. So $i^*$ has precision 1.0 — every token it fires on starts with L — and recall well under 1.0. Manual inspection of $i^*$'s top activations sees nothing wrong.

**Where the obstruction becomes visible.** Both facts above are equally consistent with a second story: the model has no single "starts with L" feature at layer 12, but instead a family of orthography features, and $p_L$ — a supervised direction fitted to maximise decodability — is a *weighted sum* of them. Under that story the "lion" latent is a true atomic feature that legitimately has a small L-component, and nothing has been absorbed. The observable table is identical in both cases. Deciding between them needs a ground-truth feature set, which is what the SAE was supposed to produce. That is the circularity, and no published experiment breaks it — which is why Section 8 tests the *consequences* of absorption reduction rather than the definition.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*