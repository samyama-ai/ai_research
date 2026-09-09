---
id: 27-multilingual/language-neutral-subspace-separation
title: "Language-Neutral Versus Language-Specific Subspaces"
topic: 27-multilingual
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Language-Neutral Versus Language-Specific Subspaces

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/language-neutral-subspace-separation` · **Status:** methodologically-blocked

## 1. Problem Statement

A multilingual model maps text in many languages into one hidden space. The folk model says that space splits: a **language-neutral** part carrying meaning, and a **language-specific** part carrying surface form (script, morphology, word order, register). Cross-lingual transfer is then explained as "the head reads the neutral part."

The problem is to make that split a real object rather than a metaphor.

- **Measurement variant.** Given a model $f$ and a hidden state $h$, produce a decomposition $h = h_N + h_S$ and a test that decides whether the decomposition is a property of the model or an artifact of the estimator. This is the blocked variant.
- **Method variant.** Given a decomposition, use it: erase $h_S$ and improve zero-shot transfer, or steer $h_S$ and control output language, with gains that survive a matched-capacity control. Partly solved for output-language control, unsolved for transfer.
- **Theory variant.** State conditions on the pretraining distribution and architecture under which a linear language-neutral subspace must exist, is unique, and is identifiable from activations alone. No result either way.

Solving it means: a decomposition that is (a) stable across random seeds and estimator choices, (b) causally load-bearing under intervention, and (c) distinguishable from a decomposition fit to a random control label of matched entropy.

## 2. Formal Setting

Let $\mathcal{L}$ be a set of languages, $\mathcal{S}$ a set of semantic contents, and $x_{\ell,s}$ a sentence in language $\ell \in \mathcal{L}$ expressing content $s \in \mathcal{S}$. A parallel corpus gives $\{x_{\ell,s}\}$ for all $\ell$ at fixed $s$ — this is the only source of paired supervision, and it is the weak point of the whole setup.

Let $h^{(k)}(x) \in \mathbb{R}^d$ be the layer-$k$ hidden state, mean-pooled over tokens or taken at the final token. Write $H^{(k)}_\ell$ for the distribution of $h^{(k)}$ under language $\ell$.

**Neutral subspace.** A projection $P \in \mathbb{R}^{d \times d}$, $P^2 = P$, with $\mathrm{rank}(P) = r$. Define

$$\mathrm{Lang}(P) = \max_{g} \; \mathbb{E}\big[\mathbf{1}\{g(Ph) = \ell\}\big], \qquad \mathrm{Sem}(P) = \mathbb{E}_{s}\big[\mathrm{ret}(Ph_{\ell,s}, Ph_{\ell',s})\big]$$

$\mathrm{Lang}(P)$ is measured as the accuracy of a language classifier $g$ trained on $Ph$ (logistic regression, 20k held-out sentences). $\mathrm{Sem}(P)$ is measured as parallel-sentence retrieval precision@1 within a candidate pool of fixed size $N$ (Tatoeba, $N = 1000$). $P$ is "neutral" when $\mathrm{Lang}(P)$ falls to the majority-class baseline $1/|\mathcal{L}|$ while $\mathrm{Sem}(P)$ is preserved.

**Specific subspace.** $Q = I - P$, evaluated by the mirror condition: $\mathrm{Lang}(Q)$ near ceiling, $\mathrm{Sem}(Q)$ near chance.

**Causal test.** For a patch $h \mapsto Ph + Q\tilde{h}$ where $\tilde h$ is drawn from language $\ell'$, measure the rate at which generation switches to $\ell'$ while task accuracy holds. Call these $\rho_{\text{switch}}$ and $\Delta\text{acc}$.

**Assumptions, and which fail.**

1. *Linearity* — the split is a linear projection. Violated in part: language identity is recoverable from a single hidden state at near-ceiling accuracy by both linear and nonlinear probes, and the gap between them is nonzero, so some language information is not in any linear subspace.
2. *Additivity* — $h = h_N + h_S$ with independent parts. Violated: content and form are statistically dependent (topic distributions differ by corpus and language), so $H_\ell$ and $H_{\ell'}$ differ in content as well as form even on "parallel" data.
3. *Parallel data is semantically identical.* Violated: translationese, register shift, and named-entity substitution mean $s$ is not held fixed. This directly contaminates $\mathrm{Sem}$.
4. *Uniqueness.* Not established. Any $P' = P + \Delta$ with $\Delta$ supported on directions unused by the readout satisfies the same behavioural constraints. This is the non-identifiability at the core of the problem.

## 3. State of the Art

**Established.**

- Mean-centering per language — subtract the language centroid $\mu_\ell$ from $h$ — sharply improves parallel-sentence retrieval and word alignment from mBERT (Libovický, Rosa & Fraser, *On the Language Neutrality of Pre-trained Multilingual Representations*, Findings of EMNLP 2020). This is the strongest and simplest positive result: a rank-$(|\mathcal{L}|-1)$ affine shift carries much of the "language-specific" signal.
- Cross-lingual structure emerges without shared vocabulary. Conneau et al. (*Emerging Cross-lingual Structure in Pretrained Language Models*, ACL 2020) show that shared parameters, not anchor subwords, drive transfer, and that separately trained monolingual BERTs can be linearly aligned.
- Language-specific neurons exist and are causal for output language. Tang et al. (ACL 2024) and Kojima et al. (NAACL 2024) both find that a small fraction (order 1%) of feed-forward neurons are language-selective, and that deactivating or amplifying them changes the generated language.

**Claimed but unablated.**

- That the language-sensitive and language-neutral axes are *distinct sets of dimensions* (Chang, Tu & Bergen, *The Geometry of Multilingual Language Model Representations*, EMNLP 2022). The geometric description is solid; the claim that the neutral part is what the task head uses is not tested by intervention.
- That LLMs "think in English." Wendler et al. (ACL 2024) show logit-lens decoding of Llama-2 mid-layers favours English tokens on translation tasks. Whether this is a pivot language or an artifact of the unembedding matrix's English bias is not settled.
- Concept-erasure methods (INLP, ACL 2020; RLACE, ICML 2022; LEACE, NeurIPS 2023) give provably optimal *linear* erasure of a labelled attribute. Applied to language labels they guarantee only that no linear probe recovers language — not that the model stopped using it.

**Benchmark-only numbers.** XTREME and XTREME-R aggregate scores are the usual evidence for "the neutral subspace works." They are task scores, not measurements of any subspace.

## 4. What Is Known

- Language identity is near-perfectly decodable. Linear probes on mBERT/XLM-R mid-layer states classify language at >95% across 100 languages — measured at 12-layer/768-dim (mBERT base) and 24-layer/1024-dim (XLM-R large) scale.
- Centroid subtraction is a large effect. Libovický et al. report Tatoeba-style retrieval accuracy rising from tens of points to majority-correct after per-language mean removal, at mBERT-base scale.
- Cross-lingual transfer survives disjoint vocabularies with a modest drop (Conneau et al. 2020; K et al., ICLR 2020), at BERT-base scale on XNLI/NER.
- Language-specific neuron sets are small and shallow-biased: concentrated in the first and last few layers, ~1% of FFN neurons, measured on XGLM-2.9B and Llama-2-7B class models.
- Layerwise shape is consistent: language identity is high at the embedding layer, dips in the middle, and rises again near the output. Reproduced independently across mBERT, XLM-R, and decoder LLMs.
- Probe accuracy alone is not evidence of use — control tasks (Hewitt & Liang, EMNLP 2019) show high-capacity probes recover randomly assigned labels almost as well.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted criterion that separates "the model has a language-neutral subspace" from "an estimator fit a subspace to my parallel corpus." No published decomposition reports stability across seeds, across estimators (centroid vs. LEACE vs. SVD of the between-language scatter), and against a matched-entropy control label. Until that triple is reported, every $r$-dimensional claim is unfalsifiable.
- **Theoretically open.** No identifiability theorem. Given only the activation distribution $\{H_\ell\}$, nothing forbids infinitely many projections with identical $(\mathrm{Lang}, \mathrm{Sem})$ profiles. Conditions under which $P$ is unique up to the readout's null space are unproved.
- **Empirically open.** Whether erasing the language-specific part *improves* zero-shot transfer at 70B scale. The experiment is runnable; the published erasure work stops at encoder scale and reports probe accuracy, not downstream transfer.
- **Open.** Whether $r$ grows with $|\mathcal{L}|$ (a per-language direction) or saturates (a shared typological basis).

## 6. Why It Is Hard

Two obstructions, both specific.

**Non-identifiability.** $\mathrm{Lang}$ and $\mathrm{Sem}$ are invariant to adding any directions the downstream readout ignores. The constraint set is a manifold of solutions, not a point. Reported dimensionalities ($r \approx 10$, $r \approx 100$) are properties of the regulariser, not the model, and no paper varies the regulariser and reports the spread.

**Confounded measurement.** The only paired data is translation. Translationese shifts lexical choice and sentence length systematically, so a classifier trained to predict language from $Ph$ can succeed on *content* correlates — corpus topic, entity inventory — after all form has been erased. $\mathrm{Lang}(P)$ therefore has a floor that is not the model's language information. Nobody reports that floor, because estimating it needs same-language, same-content, different-register controls that do not exist at scale.

The consequence: the evaluation does not measure what it names. "Language accuracy after projection" names language, measures language-plus-domain.

## 7. Current Research (as of 2026)

- **Causal localisation.** Extending language-neutron work to steering vectors and activation patching. Dumas et al. (*Separating Tongue from Thought*, 2024) patch language and concept independently and report partial dissociation — the cleanest causal evidence to date. *(frontier — verify: follow-ups at 70B.)*
- **Latent-language debate.** Wendler et al. (EPFL) vs. logit-lens-artifact rebuttals; Schut, Gal & Farquhar's *Do Multilingual LLMs Think in English?* (2025) is the current pole. *(frontier — verify.)*
- **Shared grammatical abstractions.** Brinkmann et al. (NAACL 2025) find shared morphosyntactic feature representations across typologically diverse languages — evidence for a neutral part defined by *feature*, not by parallel data.
- **Erasure at LLM scale.** EleutherAI's LEACE line applied to language labels in decoder models. Mostly unpublished.
- **Surveys.** Hämmerl, Libovický & Fraser, *Understanding Cross-Lingual Alignment* (Findings of ACL 2024) is the current map.

## 8. Concrete Next Experiment

**The identifiability stress test.** Purpose: report the spread of $r$ and of $P$ across estimators and seeds — the number nobody has published.

- **Scale.** One 7B decoder (Llama-3-8B or Qwen-2.5-7B), layers 8/16/24, 20 languages spanning 5 scripts, 50k FLORES-200 parallel sentences. Fits on one A100 in under a day; activation extraction is the only cost.
- **Arms.** Four estimators of the language-specific subspace: (i) per-language centroids, (ii) LEACE on language labels, (iii) top singular directions of between-language scatter, (iv) language-selective FFN neurons. Each fit on 5 disjoint data splits.
- **Control arm (essential).** Repeat every estimator with a *pseudo-language* label: a random 20-way partition of the same sentences, balanced across true languages, with matched label entropy $\log 20$. Any structure found here is estimator artifact.
- **Deciding number.** The mean principal angle $\bar\theta$ between subspaces from different estimators at matched rank $r=32$, minus the same quantity in the control arm. If $\bar\theta_{\text{lang}} < 30°$ while $\bar\theta_{\text{ctrl}} > 70°$, the language-specific subspace is estimator-independent and the object is real. If $\bar\theta_{\text{lang}} > 60°$, the four literatures have been describing four different subspaces and every reported $r$ is meaningless.

Secondary readout: $\Delta\text{acc}$ on XNLI after erasing each subspace. A real neutral/specific split predicts erasure of $Q$ leaves accuracy flat.

## 9. Key References

- **[Foundational]** Pires, Schlinger & Garrette. *How Multilingual is Multilingual BERT?* ACL 2019. — arXiv:1906.01502
- **[Foundational]** Conneau, Wu, Li, Zettlemoyer & Stoyanov. *Emerging Cross-lingual Structure in Pretrained Language Models.* ACL 2020. — arXiv:1911.01464
- **[Foundational]** Libovický, Rosa & Fraser. *On the Language Neutrality of Pre-trained Multilingual Representations.* Findings of EMNLP 2020.
- **[Method]** Ravfogel, Elazar, Gonen, Twiton & Goldberg. *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection.* ACL 2020. — arXiv:2004.07667
- **[Method]** Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff & Biderman. *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS 2023. — arXiv:2306.03819
- **[Method]** Hewitt & Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP 2019. — arXiv:1909.03368
- **[SOTA]** Chang, Tu & Bergen. *The Geometry of Multilingual Language Model Representations.* EMNLP 2022. — arXiv:2205.10964
- **[SOTA]** Wendler, Veselovsky, Monea & West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL 2024. — arXiv:2402.10588
- **[SOTA]** Tang, Wang, Li, et al. *Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models.* ACL 2024. — arXiv:2402.16438
- **[SOTA]** Kojima, Okimura, Iwasawa, Yanaka & Matsuo. *On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons.* NAACL 2024. — arXiv:2404.02431
- **[Survey]** Hämmerl, Libovický & Fraser. *Understanding Cross-Lingual Alignment — A Survey.* Findings of ACL 2024.
- **[Survey]** Philippy, Guo & Haddadan. *Towards a Common Understanding of Contributing Factors for Cross-Lingual Transfer in Multilingual Language Models: A Review.* ACL 2023.

## 10. Worked Example

Take mBERT-base, $d = 768$, layer 8, and three languages: English, German, Turkish. Pool 5k FLORES sentences per language.

1. Compute centroids $\mu_{\text{en}}, \mu_{\text{de}}, \mu_{\text{tr}}$. They span a 2-dimensional affine subspace. Let $Q_{\mu}$ project onto it.
2. Fit a logistic language classifier on raw $h$: accuracy ≈ 0.99. On $(I - Q_\mu)h$: accuracy drops sharply but stays well above the 0.33 baseline — typically 0.7–0.9. Two dimensions do not exhaust language.
3. Now fit LEACE on the same labels. By construction, post-erasure linear language accuracy is exactly 0.33. The erased subspace has rank 2 as well — LEACE removes the whitened between-class means.
4. Compare the two rank-2 subspaces. They differ: $Q_\mu$ is the raw mean span, LEACE's is the mean span in whitened coordinates. The principal angle between them is nonzero whenever the within-language covariance is anisotropic, which it always is.

Here is the obstruction, visible. Both are rank 2. Both are "the language subspace." One leaves a probe at 0.85, the other at 0.33 — the guarantee differs, not the amount of language information in the model. And a nonlinear MLP probe on the LEACE-erased states recovers language at ~0.6, which means the 0.33 was a statement about linear probes, not about the model.

Now run the control: relabel the same 15k sentences with a random 3-way partition and repeat. LEACE again drives its probe to chance and again returns a rank-2 subspace. The procedure cannot tell a real language split from a fabricated one by its own output. Only the comparison across estimators and against the control — Section 8's $\bar\theta$ — can, and that comparison has not been published.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*