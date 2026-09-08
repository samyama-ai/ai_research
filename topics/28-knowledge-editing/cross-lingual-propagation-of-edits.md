---
id: 28-knowledge-editing/cross-lingual-propagation-of-edits
title: "Cross-Lingual Propagation of Edits"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Lingual Propagation of Edits

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/cross-lingual-propagation-of-edits` · **Status:** open

## 1. Problem Statement

A knowledge edit applied to a multilingual model in one language should change what the model says in every language, because the fact is language-independent. It does not. Editing "The president of France is Emmanuel Macron" → "…is Élisabeth Borne" via ROME in English typically leaves the Chinese, Hindi, and Swahili completions unchanged.

- **Input:** a multilingual LM $f_\theta$, an edit request $(s, r, o^*)$ expressed in a source language $\ell_s$, and a set of target languages $L$.
- **Output:** edited parameters $\theta'$.
- **Predicate:** for every $\ell \in L$, $f_{\theta'}$ emits $o^*$ (in $\ell$'s lexicalization) on paraphrases of $(s,r)$ in $\ell$, while leaving unrelated facts in $\ell$ unchanged.
- **Solved** = a method whose cross-lingual edit success is within a small margin (say 5 points) of its in-language success, at no worse locality, across typologically distant languages and at model scales where the fact is known in both languages pre-edit.

Three variants, different difficulty:

- **Measurement:** define cross-lingual edit success without confounding it with (a) the model not knowing the fact in $\ell$ before the edit, (b) tokenizer/lexicalization mismatch of $o^*$, and (c) translation artifacts in the eval set. Currently the weakest link.
- **Method:** build an editor that propagates. Runnable now; results are poor.
- **Theory:** determine whether factual recall in multilingual transformers is stored in a shared language-agnostic substrate at all, or in per-language circuits that merely agree. If the latter, single-locus editing *cannot* propagate and the problem is a claim about representation geometry, not about optimizers.

## 2. Formal Setting

Let $f_\theta: \mathcal{V}^* \to \Delta(\mathcal{V})$ be an autoregressive LM. A fact is a triple $t = (s, r, o)$. A **lexicalizer** $\pi_\ell(t) = (q, a)$ maps a triple to a cloze prompt and gold answer string in language $\ell$; $P_\ell(t)$ is a set of paraphrase prompts, $N_\ell(t)$ a neighborhood of prompts sharing $r$ but a different subject.

Measured quantities, all in $[0,1]$, all evaluated by exact-match on greedy decoding unless stated:

$$\mathrm{Rel}_\ell = \mathbb{E}_{q \sim \pi_\ell}\big[\mathbb{1}\{f_{\theta'}(q) = a^*_\ell\}\big], \qquad
\mathrm{Gen}_\ell = \mathbb{E}_{q \sim P_\ell}\big[\mathbb{1}\{f_{\theta'}(q) = a^*_\ell\}\big]$$

$$\mathrm{Loc}_\ell = \mathbb{E}_{q \sim N_\ell}\big[\mathbb{1}\{f_{\theta'}(q) = f_{\theta}(q)\}\big]$$

The object of study is the **propagation gap** for source $\ell_s$ and target $\ell$:

$$\Delta(\ell_s \!\to\! \ell) = \mathrm{Rel}_{\ell_s} - \mathrm{Rel}_{\ell}.$$

The gap is only interpretable conditional on **pre-edit knowledge**: define $\kappa_\ell(t) = \mathbb{1}\{f_\theta(\pi_\ell(t)) = a_\ell\}$, the model's pre-edit correctness on the *original* object in $\ell$. Restrict all averages to $\{t : \kappa_{\ell_s}(t) = \kappa_\ell(t) = 1\}$. Without this restriction $\Delta$ mixes editing failure with ignorance.

Assumptions, and their status:

1. **Language-invariant answer set.** $a^*_\ell$ is a well-defined string in each $\ell$. *Violated:* transliteration variants, morphological case (Russian, Finnish), and script choice make exact-match undercount. Mitigation is a per-language alias set $A^*_\ell$; almost no benchmark builds one carefully.
2. **Faithful lexicalization.** $\pi_\ell$ preserves the relation. *Violated:* most multilingual editing sets are machine-translated from English ZsRE/CounterFact, so $\pi_\ell$ inherits translationese and occasional relation drift.
3. **Shared substrate.** There exist parameters whose modification changes recall in all $\ell$. *Partly violated / open* — see §4.
4. **Locality is language-local.** Editing in $\ell_s$ does not damage unrelated facts in $\ell \ne \ell_s$. *Violated:* off-target degradation in non-edited languages is measured rarely and is nonzero.

## 3. State of the Art

**Empirical SOTA (established).** Cross-lingual editing is measurable and consistently poor. Wang et al. (ACL 2024) introduced **BI-ZsRE** (English↔Chinese) and evaluated FT, ROME, MEMIT, KN, MEND, SERAC on BLOOMZ-560m/1.1b/3b/7.1b. The reproduced regularity: in-language reliability is high (often >90 for parameter-editing methods on the small BLOOMZ models) while cross-lingual reliability collapses toward the pre-edit baseline. Beniwal et al. (Findings of EACL 2024) reproduced the direction on BLOOM and mBERT-family models across a wider language set, and reported that gaps widen with typological/script distance from the edit language.

**Benchmark-number-only results.** MLaKE (Wei et al., 2024) reports multi-hop multilingual editing accuracies across 5 languages; MEMLA and language-agnostic-neuron editors (2024–2025) report improved multilingual propagation. These are single-paper leaderboard numbers on machine-translated sets, largely without the $\kappa_\ell$ pre-edit-knowledge control, without alias-set scoring, and without an independent reproduction. Treat as *claimed, unablated*.

**Retrieval / memory-based methods** (SERAC-style, and in-context editing as in BMIKE-53) propagate better than parameter edits when the retriever is language-agnostic — but this is a property of the *retriever*, not of the model's knowledge, and it fails the "model actually knows it" reading of the problem.

**Theory SOTA.** No theorem. The closest is mechanistic evidence about where multilingual facts live: Wendler et al. (ACL 2024) show Llama-2 pivots through an English-centric latent space in middle layers; Zhao et al. (NeurIPS 2024) propose a translate→reason-in-English→translate-back picture; Chen et al. (AAAI 2024) report language-independent knowledge neurons. These support a *partial* shared substrate — enough to make propagation plausible, not enough to predict which parameters carry it.

## 4. What Is Known

- **The gap is large and reproducible.** Across BI-ZsRE (BLOOMZ 560m–7.1b) and the EACL 2024 cross-lingual study (BLOOM, mBERT-scale), locate-and-edit methods (ROME, MEMIT, KN) show near-ceiling source-language reliability and cross-lingual reliability far below it — commonly a 40–60 point drop for distant language pairs, smaller for related pairs. Directionality is asymmetric: English→X is usually better than X→English is not; the asymmetry varies by model and is not explained.
- **Distance matters.** Propagation degrades monotonically-ish with script and family distance (e.g. en→de ≫ en→zh ≳ en→hi ≫ en→sw at the scales tested).
- **Locality survives cross-lingually.** Non-edited-language locality stays high in reported results, which is unsurprising: an edit that fails to propagate also fails to damage.
- **A shared middle-layer representation exists.** Logit-lens and activation-patching work (Wendler et al. 2024; Dumas et al. 2024) shows concept representations in middle layers that transfer across the input language, and language identity separable from concept identity. Measured on Llama-2 7B/13B/70B with controlled word-translation tasks — not on the full factual-recall path.
- **Localization does not predict editability**, in-language (Hase et al., NeurIPS 2023). This weakens any argument of the form "we found the language-agnostic neurons, so edit there."
- **Ripple effects are already broken monolingually** (Cohen et al., TACL 2024): edits do not propagate to logical consequences within one language. Cross-lingual propagation is a strictly harder instance of the same failure.

## 5. What Is Not Known

- **Theoretically open.** Whether multilingual factual recall admits a *single* parameter locus whose modification is sufficient for all languages, or whether recall is $k$ partly-redundant per-language circuits. No proof either way; no formalization of "shared substrate" sharp enough to falsify.
- **Empirically open.** Every reported gap is at ≤7B, mostly BLOOM-family, mostly en↔zh. Whether $\Delta$ shrinks with scale, with more balanced pretraining mixtures, or after multilingual instruction tuning is runnable today on 70B-class open models and has not been run with the $\kappa_\ell$ control.
- **Methodologically blocked.** The gap is not cleanly measurable. Reported $\Delta$ confounds: pre-edit ignorance in the target language, exact-match failure on morphological/transliteration variants, and translation noise in $\pi_\ell$. No public benchmark simultaneously (i) filters on $\kappa_\ell = 1$ in both languages, (ii) scores against native-speaker alias sets, and (iii) uses natively authored rather than translated prompts. Until that exists, "method X improves cross-lingual editing by 12 points" is uninterpretable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**.

- *Confound:* $\mathrm{Rel}_\ell$ after editing conflates three failures — the edit didn't reach language $\ell$; the model never knew the fact in $\ell$; the model produced a correct but unscored surface form. These have opposite implications and current benchmarks cannot separate them. Machine-translated evaluation sets make (2) and (3) worse in exactly the languages where $\Delta$ looks largest, so the headline "typological distance drives the gap" finding is partly an artifact candidate.
- *Non-identifiability:* even given a clean measurement, a propagating edit and a memorized source-language patch that happens to generalize are not distinguishable from output behavior alone. Hase et al.'s result — that causal-tracing localization does not predict where editing works — means the mechanistic handle you would use to tell them apart is not reliable.

Compute is not the obstruction: BLOOMZ-7B-scale editing sweeps are a few GPU-days.

## 7. Current Research (as of 2026)

- **Language-agnostic neuron editing** — identify neurons active for a fact across languages, edit there (Chen et al., AAAI 2024; MEMLA and successors, 2024–2025). *(frontier — verify)* Reported gains are not yet independently reproduced.
- **Multilingual in-context / retrieval editing** — BMIKE-53 style, 53 languages, ICL demonstrations; strong propagation, but external memory (Nie et al., 2024).
- **Latent-language mechanistic work** — EPFL (West group), and activation-patching groups extending Wendler et al. to factual recall rather than word translation.
- **Benchmark repair** — natively authored, alias-scored multilingual editing sets. This is the highest-value and least-glamorous direction; no dominant artifact yet. *(frontier — verify)*
- **Zhejiang / ZJUNLP EasyEdit** maintains the tooling that most of these papers build on.

## 8. Concrete Next Experiment

**Question:** is the reported cross-lingual gap a propagation failure or a measurement artifact?

- **Scale:** Qwen2.5-7B and Llama-3.1-70B (both with genuinely multilingual pretraining), 6 languages: en, de, ru, zh, hi, sw. 1,000 facts.
- **Construction:** take facts from Wikidata, lexicalize *natively* (native-speaker-written templates, not translation), build per-language alias sets $A^*_\ell$ including morphological forms and transliterations. Filter to the subset where $\kappa_\ell(t)=1$ for **all six** languages pre-edit — the model demonstrably knows the fact everywhere. Expect this filter to remove 60–85% of candidates; that attrition is itself a result.
- **Edit:** ROME and MEMIT, edit in English only.
- **Control arms:** (1) the same evaluation on machine-translated prompts with exact-match scoring — the standard protocol; (2) a no-edit arm measuring $\mathrm{Rel}_\ell$ on the *counterfactual* object, to fix the floor; (3) edit-in-$\ell$-eval-in-$\ell$ upper bound.
- **Deciding number:** $\Delta(\text{en}\!\to\!\text{zh})$ under the clean protocol minus $\Delta(\text{en}\!\to\!\text{zh})$ under the standard protocol. If the clean gap is within 10 points of the standard gap, the failure is real propagation and method work is justified. If the clean gap shrinks by more than half, the field has been optimizing against an artifact and the benchmark must be rebuilt before any method claim is meaningful.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[SOTA]** Jiaan Wang, Yunlong Liang, Zengkui Sun, Yuxuan Cao, Jiarong Xu, Fandong Meng. *Cross-Lingual Knowledge Editing in Large Language Models.* ACL, 2024. — arXiv:2309.08952
- **[SOTA]** Himanshu Beniwal, Kowsik Nandagopan D, Mayank Singh. *Cross-Lingual Editing in Multilingual Language Models.* Findings of EACL, 2024.
- **[SOTA]** Zihao Wei et al. *MLaKE: Multilingual Knowledge Editing Benchmark for Large Language Models.* NAACL, 2025.
- **[Mechanism]** Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West. *Do Llamas Work in English? On the Latent Language of Multilingual Transformers.* ACL, 2024. — arXiv:2402.10588
- **[Mechanism]** Yiran Zhao et al. *How do Large Language Models Handle Multilingualism?* NeurIPS, 2024. — arXiv:2402.18815
- **[Mechanism]** Yuheng Chen, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao. *Journey to the Center of the Knowledge Neurons: Discoveries of Language-Independent Knowledge Neurons and Degenerate Knowledge Neurons.* AAAI, 2024.
- **[Critique]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing.* NeurIPS, 2023. — arXiv:2301.04213
- **[Critique]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[Survey]** Ningyu Zhang et al. *A Comprehensive Study of Knowledge Editing for Large Language Models.* 2024. — arXiv:2401.01286

## 10. Worked Example

Fact: $t = (\text{Eiffel Tower}, \text{located in}, \text{Paris})$, edited to $o^* = \text{Rome}$, with ROME on a 7B multilingual model, edit prompt in English.

Post-edit probes:

| Language | Prompt | Output | Exact match to $a^*_\ell$ |
|---|---|---|---|
| en | "The Eiffel Tower is located in" | "Rome" | ✓ |
| de | "Der Eiffelturm befindet sich in" | "Rom" | ✓ (only if "Rom" is in the alias set; "Rome" is not the German form) |
| ru | "Эйфелева башня находится в" | "Риме" | ✗ under naive matching — "Рим" is nominative, the frame requires prepositional "Риме" |
| zh | "埃菲尔铁塔位于" | "巴黎" | ✗ — genuine propagation failure |
| sw | "Mnara wa Eiffel uko" | "Paris" | ✗ — but is this failure, or did the model never know it? |

Naive scoring gives $\mathrm{Rel} = 1/5$ and a reported gap $\Delta \approx 0.8$. Correct alias sets recover German and Russian: $\mathrm{Rel} = 3/5$, $\Delta \approx 0.4$. Now apply the $\kappa$ filter: if the Swahili probe pre-edit returned "Ufaransa" (France) rather than "Paris", the fact was never present in Swahili and the item must be dropped — $\mathrm{Rel} = 3/4$, $\Delta \approx 0.25$.

The same model, same edit, same five probes yield $\Delta \in \{0.80, 0.40, 0.25\}$ depending only on scoring choices that no paper reports in full. That spread is larger than the improvement any published cross-lingual editing method claims. The obstruction is not that the method is weak — it is that the measurement's error bar exceeds the effect being measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*