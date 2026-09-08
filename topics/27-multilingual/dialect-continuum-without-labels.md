---
id: 27-multilingual/dialect-continuum-without-labels
title: "Dialect Continuum Modeling Without Discrete Labels"
topic: 27-multilingual
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Dialect Continuum Modeling Without Discrete Labels

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/dialect-continuum-without-labels` · **Status:** methodologically-blocked

## 1. Problem Statement

Language variation is continuous. Between Cologne and Vienna, between Casablanca and Baghdad, between rural Alabama and Chicago's South Side, there is no line where one variety stops and another starts. NLP systems nonetheless represent variety as a categorical token: a language ID, a dialect label, a `lang="de-AT"` tag. The problem is to build and evaluate models that condition on a **continuous** representation of variety instead.

Three variants, with different difficulty:

- **Measurement variant.** Given a corpus of utterances with speaker/geographic/social metadata but no dialect labels, recover a continuous variety representation $z \in \mathcal{Z}$ per utterance such that $z$ predicts held-out linguistic behaviour better than any discrete labelling of the same data. Solving this requires an evaluation that does not itself presuppose labels — which is where the problem is currently stuck.
- **Method variant.** Given such a representation, condition generation, ASR, and understanding on $z$ so that performance degrades smoothly, not stepwise, as $z$ moves away from the training mass. Success criterion: worst-case performance over a dense sample of $\mathcal{Z}$, not mean performance over a label set.
- **Theory variant.** State conditions under which $z$ is identifiable from text alone. Continuum structure is recoverable only up to a group of transformations; nobody has characterised that group for realistic generative models of variation.

Solving the problem means: a model that has never seen the label "Swabian" performs on Swabian input within $\epsilon$ of its performance on Standard German, and its own uncertainty estimate tracks the gap.

## 2. Formal Setting

Let $\mathcal{V}$ be a population of speakers. Each speaker $v$ has a latent variety vector $z_v \in \mathbb{R}^d$. An utterance is drawn $x \sim p(\cdot \mid c, z_v)$ where $c$ is content (meaning, topic, register) and $z_v$ is form-conditioning. The core factorisation assumption is that content and variety are separable:

$$p(x \mid c, z) = \sum_{y} p(y \mid c)\, p(x \mid y, z)$$

with $y$ an abstract message. **Known violated:** lexical variation is content-entangled (dialect-specific referents have no standard equivalent), and topic correlates with region (Blodgett et al., EMNLP 2016, show AAE-associated Twitter text differs in topic as well as form).

**Measured quantities.**

- *Aggregate linguistic distance.* For sites $i,j$ with parallel item sets $\{w_i^{(k)}\}_{k=1}^{K}$ (K concepts elicited at both sites), the standard dialectometric quantity is normalised Levenshtein distance averaged over items:
  $$D_{ij} = \frac{1}{K}\sum_{k=1}^{K} \frac{\mathrm{lev}\!\left(w_i^{(k)}, w_j^{(k)}\right)}{\max\!\left(|w_i^{(k)}|,|w_j^{(k)}|\right)}$$
  This is what Gabmap and the Groningen dialectometry line actually compute (Heeringa 2004; Wieling & Nerbonne 2015). Its value depends on $K$ and on which concepts are elicited — the atlas designer's choice, not a property of the varieties.
- *Continuum embedding.* $\hat{Z} = \arg\min_{Z}\sum_{i<j}\left(\|z_i - z_j\| - D_{ij}\right)^2$ — classical MDS. Identified only up to $O(d)$ rotation, reflection, and (under monotone MDS) any monotone rescaling of $\|\cdot\|$.
- *Model dialect gap.* For task metric $m$ and reference variety $z_0$: $\Delta(z) = m(z_0) - m(z)$. Measured in WER for ASR, F1/accuracy for classification, COMET or human rating for generation.
- *Smoothness.* $L = \sup_{z \neq z'} |\Delta(z)-\Delta(z')| / \|z - z'\|$. Reporting $L$ requires a metric on $\mathcal{Z}$ — which requires having solved the measurement variant. This circularity is the methodological block.

**Further assumptions, and their status.** (i) One $z$ per speaker — violated by code-switching and style-shifting, which are within-speaker and audience-dependent. (ii) Metadata (geotag, self-report) is a noisy proxy for $z$ — violated by migration and by online geotags that reflect device location, not the speaker's variety. (iii) $\mathcal{Z}$ is low-dimensional and Euclidean — untested; dialect features spread by contact along social networks, whose geometry is not Euclidean.

## 3. State of the Art

**Established.**

- *Dialectometry* has produced continuous maps for a century of atlas data. Séguy (1971) established that linguistic distance grows sublinearly with geographic distance; the aggregate-Levenshtein method (Heeringa 2004) and its MDS/cluster visualisations are reproduced across Dutch, German, Norwegian, and Bantu data (Wieling & Nerbonne, *Annual Review of Linguistics*, 2015). These are established for *elicited atlas words*, not for running text.
- *Feature-level detection.* Demszky et al. (NAACL 2021) show that individual dialect features (e.g. Indian English *focus* "itself") can be recognised with minimal supervision, giving a continuous per-feature density rather than a label.
- *Rule-based dialect perturbation.* VALUE (Ziems et al., ACL 2022) and Multi-VALUE (Ziems et al., ACL 2023) implement ~189 lexico-syntactic transformation rules covering 50 English dialects, letting any benchmark be re-rendered in a target variety. Established: models lose accuracy under perturbation. Not established: that the perturbed text is what speakers produce — rule application probabilities come from eWAVE attestation ratings, not corpus frequencies.

**Claimed but unablated.**

- That multilingual encoders "already" represent dialect continuously. Probing studies recover geography from embeddings, but no study has ablated topic and orthography as confounds; recovering Bavarian-vs-Hamburg geography from text that also differs in topic is not evidence of a variety subspace.
- That scaling closes the gap. Reported as benchmark numbers on DialectBench (Faisal et al., ACL 2024; 281 varieties, 10 task clusters), not as a controlled study holding data quantity fixed.

**Benchmark-number-only results.** NADI shared tasks (Abdul-Mageed and colleagues, WANLP, 2020–2023) report country-level Arabic dialect ID F1 in the 30–40 range for 21-way classification. This is a number about a *label set imposed on a continuum*, and its ceiling is partly the labelling, not the model.

## 4. What Is Known

- **The gap is large and correlated with resource level.** Kantharuban, Vulić & Korhonen (Findings of EMNLP 2023) measure ASR dialect gaps across languages and find performance disparity correlates with dialect-level economic and demographic factors, not just data volume.
- **Perturbation costs accuracy at the ~5–10 point scale.** VALUE/Multi-VALUE report multi-point GLUE-style drops when standard-English benchmarks are re-rendered into AAE-like and other dialects, at BERT-through-RoBERTa scale and replicated on larger models in the Multi-VALUE paper.
- **Language ID fails first.** Blodgett, Green & O'Connor (EMNLP 2016) found off-the-shelf language identifiers misclassify AAE-associated English tweets as non-English at rates far above the white-aligned control — an error that removes the variety from the corpus before any model sees it. Measured on ~59M geolocated tweets.
- **Dialect harms persist after alignment.** Hofmann et al. (*Nature*, 2024) show LLMs make more negative character/employability judgements about AAE text than about Standard American English text, while showing no overt bias — RLHF suppressed the overt signal and left the covert one. Measured across GPT-2/RoBERTa/T5/GPT-3.5/GPT-4 families.
- **Séguy's law holds empirically.** Aggregate linguistic distance rises roughly with the logarithm or square root of geographic distance across multiple atlases — a genuine continuum regularity, at the scale of a few hundred survey sites per language area.

## 5. What Is Not Known

- **Methodologically blocked (the primary gap).** There is no label-free evaluation of continuum modelling. Every headline metric — dialect-ID F1, per-dialect WER, DialectBench aggregates — requires a partition of $\mathcal{Z}$ into named cells. Asking "does the continuous model beat the discrete one?" and answering it with a per-label table is circular. A held-out-speaker perplexity or interpolation metric would break the circle; none is standardised.
- **Theoretically open.** Identifiability. Under what generative conditions is $z$ recoverable from unlabelled text up to a known transformation group? Nonlinear ICA results give identifiability under auxiliary-variable conditioning (Hyvärinen & Morioka, AISTATS 2016/2017; Khemakhem et al., AISTATS 2020, iVAE), but the required conditioning variable — a true independent index of speaker variety — is exactly what is missing.
- **Empirically open.** Whether conditioning on a continuous $z$ beats a 50-way label at fixed parameter and data budget. Runnable today with Multi-VALUE plus real dialect corpora; nobody has published the controlled comparison.
- **Empirically open.** Whether pretraining data mixture or model scale drives the smoothness constant $L$.

## 6. Why It Is Hard

Three specific obstructions, in order of bite.

1. **Non-identifiability of the embedding.** MDS on $D_{ij}$ fixes $Z$ only up to isometry, and $D_{ij}$ itself is a function of which $K$ items the atlas elicited. Two research groups with different item sets get different continua and no way to adjudicate. There is no ground-truth $z$ anywhere in the pipeline.
2. **The evaluation does not measure what it names.** "Dialect robustness" is scored by averaging over labelled dialect test sets. The average is dominated by whichever varieties happened to get corpora, and it is blind to the interior of the continuum — the transitional speakers who fit no label are exactly the ones excluded from every test set.
3. **Confounded measurement.** Region correlates with topic, register, orthographic convention, transcription practice, and speaker demographics. A measured gap $\Delta(z)$ mixes all of these. Controlling requires parallel content across varieties, which for most of the world does not exist; synthetic perturbation (Multi-VALUE) controls content perfectly but at the cost of the rule set being an idealisation of the variety.

Compute is not the obstruction. A decisive experiment fits on a handful of GPUs.

## 7. Current Research (as of 2026)

- **VarDial workshop series** (Zampieri, Nakov, Scherrer and colleagues) — annual evaluation campaigns on closely-related languages and varieties; steadily moving from flat classification toward hierarchical and graded targets.
- **NADI / Arabic dialect work** (Abdul-Mageed, UBC and collaborators) — country- and province-level Arabic, the largest sustained effort on a genuine continuum.
- **DialectBench** (Faisal, Ahia, Anastasopoulos and colleagues, ACL 2024) — broad-coverage varieties benchmark; the natural substrate for continuum experiments even though its interface is label-based.
- **Dialect-robust evaluation metrics** — Sun, Ziems and colleagues (ACL 2023) on making generation metrics insensitive to dialect while sensitive to quality.
- **Indonesian and Nusantara varieties** (NusaX, Winata et al., EACL 2023) — parallel data across ten local languages, one of the few genuinely parallel resources spanning a related-variety cluster.
- *(frontier — verify)* Continuous speaker-variety conditioning in speech foundation models, using speaker embeddings as a proxy $z$; several groups appear to be pursuing this, but published controlled comparisons against label conditioning are scarce.

## 8. Concrete Next Experiment

**Question.** Does a continuous variety representation beat a discrete label at fixed budget, measured without labels?

**Scale.** One 1.4B-parameter decoder, trained from a fixed 30B-token multilingual base, then continued-pretrained on 2B tokens of German dialect and regiolect text (e.g. Twitter/Mastodon geotagged German plus the non-standard portions of available corpora), covering ≥200 distinct geographic cells. Three arms, identical tokens, identical steps:

- **Arm A (control, discrete).** Condition on a 20-way clustered dialect label derived from geography.
- **Arm B (continuous).** Condition on a 2-D $z$ = MDS embedding of aggregate Levenshtein distances between cells, computed from atlas items only, never from the training text.
- **Arm C (null).** No conditioning.

**The deciding number.** Hold out *all* text from 40 randomly chosen geographic cells, including their labels and their $z$ values. At test time, infer $z$ for a held-out cell by interpolation from its three nearest *retained* neighbours; for Arm A, assign the modal label of those neighbours. Report **held-out-cell perplexity**, and the single decision statistic:

$$\delta = \mathrm{PPL}_A^{\text{held-out}} - \mathrm{PPL}_B^{\text{held-out}}$$

$\delta > 0$ with a bootstrap 95% CI excluding zero, over cells the model never saw, decides for the continuum. This metric uses no dialect labels at evaluation time — that is the point. Cost: roughly 3 × a few hundred GPU-hours.

**Secondary read-out.** Regress $\Delta(z)$ (per-cell perplexity gap versus standard German) on $\|z - \bar z_{\text{train}}\|$. A high $R^2$ in Arm B and a low one in Arm A is direct evidence that the continuum coordinate is the right ruler.

## 9. Key References

- **[Foundational]** Séguy, J. *La relation entre la distance spatiale et la distance lexicale.* Revue de Linguistique Romane, 1971.
- **[Foundational]** Heeringa, W. *Measuring Dialect Pronunciation Differences using Levenshtein Distance.* PhD thesis, University of Groningen, 2004.
- **[Survey]** Wieling, M. & Nerbonne, J. *Advances in Dialectometry.* Annual Review of Linguistics, 2015.
- **[Foundational]** Blodgett, S.L., Green, L. & O'Connor, B. *Demographic Dialectal Variation in Social Media: A Case Study of African-American English.* EMNLP, 2016. — arXiv:1608.08868
- **[SOTA]** Ziems, C., Chen, J., Harris, C., Anderson, J. & Yang, D. *VALUE: Understanding Dialect Disparity in NLU.* ACL, 2022. — arXiv:2204.03031
- **[SOTA]** Ziems, C., Held, W., Yang, J., Dhamala, J., Gupta, R. & Yang, D. *Multi-VALUE: A Framework for Cross-Dialectal English NLP.* ACL, 2023. — arXiv:2212.08011
- **[SOTA]** Faisal, F., Ahia, O., Srivastava, A., Ahuja, K., Chiang, D., Tsvetkov, Y. & Anastasopoulos, A. *DialectBench: An NLP Benchmark for Dialects, Varieties, and Closely-Related Languages.* ACL, 2024. — arXiv:2403.11009
- **[SOTA]** Demszky, D., Sharma, D., Clark, J.H., Prabhakaran, V. & Eisenstein, J. *Learning to Recognize Dialect Features.* NAACL, 2021. — arXiv:2010.12707
- **[SOTA]** Kantharuban, A., Vulić, I. & Korhonen, A. *Quantifying the Dialect Gap and its Correlates Across Languages.* Findings of EMNLP, 2023.
- **[Foundational]** Hofmann, V., Kalluri, P.R., Jurafsky, D. & King, S. *Dialect prejudice predicts AI decisions about people's character, employability, and criminality.* Nature, 2024.
- **[Theory]** Khemakhem, I., Kingma, D.P., Monti, R.P. & Hyvärinen, A. *Variational Autoencoders and Nonlinear ICA: A Unifying Framework.* AISTATS, 2020. — arXiv:1907.04809
- **[Resource]** Winata, G.I. et al. *NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages.* EACL, 2023. — arXiv:2205.15960

## 10. Worked Example

Take four German-speaking sites: Hamburg ($h$), Cologne ($k$), Stuttgart ($s$), Vienna ($w$). Suppose an atlas with $K=200$ elicited concepts yields normalised Levenshtein distances:

| | $k$ | $s$ | $w$ |
|---|---|---|---|
| $h$ | 0.34 | 0.41 | 0.46 |
| $k$ | — | 0.30 | 0.39 |
| $s$ | — | — | 0.27 |

Classical MDS in $d=2$ reproduces these to within about 0.01 and lays the four sites on a gentle arc — the continuum. Now the obstruction, in two moves.

**Move 1: the embedding moves when the item set moves.** Re-run with a phonology-heavy subset ($K=80$). Stuttgart–Vienna typically shrinks (shared Upper German vowel developments) while Hamburg–Cologne grows. If $D_{sw}$ drops 0.27 → 0.19 and $D_{hk}$ rises 0.34 → 0.40, the MDS solution reorders the arc: Stuttgart, previously between Cologne and Vienna, now sits nearly on top of Vienna. Both configurations fit their own $D$ matrix perfectly. Nothing in the data says which is *the* continuum, because $D$ is a property of the questionnaire.

**Move 2: the model gap cannot be attributed.** Train Arm A and Arm B from §8 on these four sites, hold out Stuttgart. Suppose held-out perplexity is 41.2 (labels, modal neighbour = "Southern") and 38.6 (continuous, interpolated $z$). $\delta = 2.6$ — the continuum wins. But under the phonology-heavy embedding, Stuttgart's interpolated $z$ lands next to Vienna instead, and the same computation gives 40.9, so $\delta = 0.3$, inside noise.

The measured advantage of continuous conditioning ranges from decisive to nil depending on a choice — which items enter $D$ — made before any model was trained, and for which there is no ground truth. That is the block. The §8 experiment is worth running precisely because it forces the choice into the open: fix the item set in advance, preregister it, and report $\delta$ under at least two item sets. If $\delta$'s sign is stable across item sets, the continuum claim survives its own non-identifiability. If it flips, the field has been measuring the questionnaire.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*