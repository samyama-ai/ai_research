---
id: 25-speech-and-audio/s2st-prosody-preservation
title: "Speech-to-Speech Translation Voice and Prosody Preservation"
topic: 25-speech-and-audio
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Speech-to-Speech Translation Voice and Prosody Preservation

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/s2st-prosody-preservation` · **Status:** open

## 1. Problem Statement

**Input:** a source-language utterance $x$ (waveform, one speaker, arbitrary emotion and speaking style).
**Output:** a target-language utterance $\hat{y}$ that (a) conveys the same propositional content, (b) sounds like the *same speaker*, and (c) carries the *same paralinguistic content* — emotion, emphasis placement, speech rate, pause structure, and turn-level intonation.

Three variants, routinely conflated:

- **Measurement variant.** Define a metric $P(x, \hat{y})$ for prosody transfer that separates "the output is expressive" from "the output is expressive *in the way the input was*". Currently unsolved: existing automatic metrics do not pass this discrimination test.
- **Method variant.** Build a system that maximizes $P$ without losing translation adequacy. Partially solved for speaker timbre, largely unsolved for emphasis and rhythm.
- **Theory variant.** Establish whether a target-language prosodic realization preserving source prosody exists and is unique, given that translation changes syllable count, word order, and the syntax–prosody interface. Believed non-unique; no formalization exists.

**Solved** would mean: on held-out expressive, spontaneous speech across at least five typologically distant pairs, a system matches cascaded ASR+MT+TTS on translation quality while human raters cannot distinguish its prosody transfer from a human interpreter's, and an automatic metric predicts those ratings with $r > 0.8$.

## 2. Formal Setting

Let $x \in \mathbb{R}^{T}$ be the source waveform with transcript $s$, and let $\mathcal{Y}(s)$ be the set of adequate target-language translations. A system is a map $f: x \mapsto \hat{y}$. Posit a factorization

$$x \approx g(c,\; v,\; p), \qquad c = \text{content},\; v = \text{speaker identity},\; p = \text{prosody},$$

and the goal $\hat{y} = g(\tau(c),\; v,\; \pi(p))$ where $\tau$ is translation and $\pi$ is a cross-lingual prosody transport operator. **The problem is that $\pi$ is not defined anywhere in the literature**; it is implicitly learned.

Measured quantities:

- **Adequacy.** $\text{ASR-BLEU} = \mathrm{BLEU}(\mathrm{ASR}(\hat{y}), r)$ with a fixed target-language ASR model and reference $r$. Measured on normalized text; ASR error inflates variance by roughly the ASR WER.
- **Voice similarity.** $\mathrm{VSim} = \cos\!\big(e(x), e(\hat{y})\big)$, $e$ a speaker encoder (ECAPA-TDNN or WavLM-TDNN). Reported in $[-1,1]$; typical same-speaker same-language values $\approx 0.6$–$0.9$, cross-language values systematically lower.
- **Rate transfer.** $\rho = \log\frac{d(\hat y)/n_{\mathrm{syl}}(\hat y)}{d(x)/n_{\mathrm{syl}}(x)}$, with $d$ = voiced duration and $n_{\mathrm{syl}}$ = syllable count from a forced aligner. $\rho = 0$ is rate-preserving; $d(\hat y)=d(x)$ is *isochrony*, a different and often incompatible target.
- **Pause transfer.** Treat pauses $>150$ ms as events; align source and target events through the word alignment $a$; report $F_1$ over matched events with a $\pm 300$ ms tolerance.
- **Intonation transfer.** Resample log-$F_0$ (z-normalized per utterance) onto the word-aligned time axis and take Pearson $r$. Requires $a$ to be monotone.
- **Holistic.** PCP (Prosodic Consistency Protocol) human ratings on a 1–4 scale, and AutoPCP, a regressor trained to predict them (Seamless Communication team, 2023).

Assumptions, with status:

1. $(c, v, p)$ are separable — **violated**: in Mandarin, Vietnamese, and Thai, $F_0$ is lexical; transporting a source $F_0$ contour corrupts $c$.
2. $e(\cdot)$ is language-invariant — **violated**: speaker embeddings shift measurably with spoken language, so $\mathrm{VSim}$ conflates timbre loss with language change.
3. The word alignment $a$ is monotone — **violated** for en↔de (verb-final), en↔ja, and any pair with negation or focus reordering.
4. A single reference prosody exists — **violated**: professional interpreters produce different, all-acceptable prosodic realizations of the same source.

## 3. State of the Art

**Systems / empirical SOTA.**

- **Translatotron** (Jia et al., Interspeech 2019) — first direct S2ST with optional speaker conditioning; 25.6 ASR-BLEU on Fisher es→en against a strong cascade baseline above 40. Voice transfer was a demonstration, not a measured result.
- **Translatotron 2** (Jia et al., ICML 2022) — closed most of the adequacy gap and replaced the speaker encoder with a source-conditioned mechanism specifically to *limit* spoofing capability. Voice retention improved; established.
- **Discrete-unit S2ST** (Lee et al., ACL 2022; Lee et al., NAACL 2022) — made textless S2ST practical; discards most prosody by construction, since HuBERT units are largely prosody-invariant.
- **SeamlessExpressive / PRETSSEL** (Seamless Communication team, *Seamless: Multilingual Expressive and Streaming Speech Translation*, 2023) — current reference point. Adds rate and pause alignment in the translation model plus an expressivity-conditioned unit-to-speech generator, across en↔{es, fr, de, it, zh}. Ships the mExpresso and mDRAL evaluation sets, the AutoPCP metric, and BLASER 2.0.
- **Codec LM approaches** — VALL-E X (Zhang et al., 2023) and AudioPaLM (Rubenstein et al., 2023) preserve timbre well through acoustic-token continuation. Their prosody transfer is *claimed and demoed but not ablated*: no published control separating "prompt speaker's habitual prosody" from "this utterance's prosody".

**Theory SOTA.** Essentially empty. There is no existence or uniqueness result for $\pi$, and no impossibility result bounding joint adequacy–prosody optimality.

**Established vs. claimed.** Established: timbre can be carried cross-lingually; discrete units lose prosody; rate/pause alignment can be optimized without large BLEU loss. Claimed-but-unablated: that current systems transfer *utterance-specific* emphasis and emotion rather than reproducing a speaker- or corpus-level expressive prior. Most expressivity numbers in the literature exist only as single benchmark scores on mExpresso/mDRAL, without a mismatched-source control.

## 4. What Is Known

- **Discrete units strip prosody.** Expresso (Nguyen et al., Interspeech 2023; 47 h, 4 speakers, 26 styles) shows resynthesis from HuBERT-style units loses expressive style; recovering it requires explicit style conditioning. Measured at small speaker scale (4), which is the main caveat.
- **Adequacy and expressivity trade off.** Expressive S2ST systems report ASR-BLEU below their non-expressive siblings on the same pairs — a consistent direction across published tables, of order a few BLEU.
- **Isochrony is achievable and costs quality.** Dubbing work (Lakew et al., Interspeech 2022; Tam et al., 2023) shows isochrony-constrained MT raises timing alignment at a measurable BLEU cost, and that jointly optimizing text and timing beats post-hoc time-stretching.
- **Cross-lingual speaker similarity is systematically depressed.** Same-speaker cross-language embedding cosines fall well below same-language values, so a VSim number is not comparable across language pairs.
- **Text-free translation-quality metrics work.** BLASER 2.0 (following Chen et al., ACL 2023) correlates with human adequacy without ASR. No comparable validated metric exists for prosody.

## 5. What Is Not Known

- **Methodologically blocked (the dominant gap).** There is no validated metric that isolates *transfer* from *presence* of expressivity. AutoPCP's agreement with human PCP is moderate, and no published study reports its score under a mismatched-source control — the one condition that would show it measures transfer at all. Until that exists, every "expressivity preserved" claim is unfalsifiable.
- **Theoretically open.** Whether $\pi$ is identifiable. Given that $\tau$ changes syllable count and focus position, it is unproven that any target realization simultaneously preserves rate, pause structure, and relative emphasis. Plausibly there is a genuine impossibility for reordering-heavy pairs; nobody has stated the theorem.
- **Empirically open.** Whether tone languages admit emphasis transfer without lexical corruption. The experiment — en→zh emphasis transfer with tone-error rate measured by a tone classifier — is runnable today at modest cost and has not been run.
- **Empirically open.** Whether codec LMs transfer utterance prosody or speaker prior. Requires only an inference-time ablation.

## 6. Why It Is Hard

**The specific obstruction is that the evaluation does not measure the thing it names, and there is no ground truth to fix it with.**

Concretely: to score prosody transfer you need a reference target utterance whose prosody is *correct* for the given source. That reference would have to come from a human interpreter, and interpreters disagree — the set of acceptable target prosodies is large and unenumerated. So the field falls back on source–hypothesis similarity metrics ($F_0$ correlation, rate ratio, AutoPCP). Those metrics require a monotone cross-lingual alignment that does not exist for reordering pairs, and they assign high scores to any output that is merely expressive in the same generic register as the input.

This compounds with non-identifiability: since $c$ and $p$ are not separable in tone languages and $v$ and language are entangled in speaker embeddings, the three reported numbers (ASR-BLEU, VSim, AutoPCP) each measure a mixture. A system can gain on all three by becoming a better generic expressive TTS with no transfer at all.

## 7. Current Research (as of 2026)

- **Meta AI / Seamless line** — expressive and streaming S2ST, expressivity metrics, watermarking of generated speech (AudioSeal, San Roman et al., ICML 2024). *(frontier — verify current status)*
- **Google DeepMind** — the Translatotron/AudioPaLM lineage, folded into speech-capable multimodal LLMs. Voice preservation is deliberately restricted for anti-spoofing reasons, which caps published progress.
- **Spoken language models with interleaved text and speech tokens** — e.g. SPIRIT-LM (Nguyen et al., TACL 2025) — offer a path where prosody tokens survive the translation bottleneck. Applying this specifically to S2ST prosody transfer is active. *(frontier — verify)*
- **Automatic dubbing industry** — isochrony, lip-sync, and emotion transfer under hard timing constraints; the most demanding real deployment and the source of most timing-constrained results.
- **Evaluation work** — extending BLASER-style text-free metrics to paralinguistics. Nothing validated has landed.

## 8. Concrete Next Experiment

**Question:** do current expressive S2ST systems transfer *this utterance's* prosody, or reproduce a generic expressive prior?

**Scale.** Three pairs — en→es (low reordering), en→de (verb-final reordering), en→zh (tone) — 500 utterances each drawn from mExpresso and mDRAL, balanced across expressive styles. One open expressive S2ST system (SeamlessExpressive class) plus one codec LM (VALL-E X class). Inference only; roughly 2 GPU-days total.

**Arms.**
1. **True-source:** condition on the actual source utterance.
2. **Control — mismatched source:** condition prosody on a *different randomly chosen utterance by the same speaker in the same corpus*, while translating the same content. Timbre is held constant; only utterance-specific prosody changes.
3. **Floor:** neutral prosody conditioning.

Report AutoPCP and human PCP (three raters, 1–4) for each arm, at matched ASR-BLEU (within $\pm 0.5$) and matched VSim (within $\pm 0.02$).

**The deciding number:** $\Delta_{\text{transfer}} = \mathrm{PCP}(\text{arm 1}) - \mathrm{PCP}(\text{arm 2})$, compared against $\Delta_{\text{expressive}} = \mathrm{PCP}(\text{arm 1}) - \mathrm{PCP}(\text{arm 3})$.

- If $\Delta_{\text{transfer}} \geq 0.5$ on the 1–4 scale, transfer is real and the metric detects it.
- If $\Delta_{\text{transfer}} < 0.2$ while $\Delta_{\text{expressive}} > 0.5$, then published expressivity gains measure *expressiveness present*, not *expressiveness transferred*, and every benchmark number in Section 3 needs reinterpretation.

The same design run per-pair also gives the first measurement of whether reordering (en→de) and tone (en→zh) degrade transfer relative to en→es.

## 9. Key References

- **[Foundational]** Ye Jia, Ron J. Weiss, Fadi Biadsy, Wolfgang Macherey, Melvin Johnson, Zhifeng Chen, Yonghui Wu. *Direct speech-to-speech translation with a sequence-to-sequence model.* Interspeech, 2019. — arXiv:1904.06037
- **[Foundational]** Ye Jia, Michelle Tadmor Ramanovich, Tal Remez, Roi Pomerantz. *Translatotron 2: High-quality direct speech-to-speech translation with voice preservation.* ICML, 2022. — arXiv:2107.08661
- **[SOTA]** Seamless Communication team (Meta AI). *Seamless: Multilingual Expressive and Streaming Speech Translation.* 2023. — arXiv:2312.05187
- **[SOTA]** Seamless Communication team (Meta AI). *SeamlessM4T: Massively Multilingual & Multimodal Machine Translation.* 2023. — arXiv:2308.11596
- **[SOTA]** Ann Lee, Peng-Jen Chen, Changhan Wang, Jiatao Gu, et al. *Direct speech-to-speech translation with discrete units.* ACL, 2022. — arXiv:2107.05604
- **[SOTA]** Ziqiang Zhang, Long Zhou, Chengyi Wang, Sanyuan Chen, et al. *Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling* (VALL-E X). 2023. — arXiv:2303.03926
- **[SOTA]** Paul K. Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, et al. *AudioPaLM: A Large Language Model That Can Speak and Listen.* 2023. — arXiv:2306.12925
- **[Dataset]** Tu Anh Nguyen, Wei-Ning Hsu, Antony D'Avirro, Bowen Shi, et al. *EXPRESSO: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis.* Interspeech, 2023. — arXiv:2308.05725
- **[Dataset]** Ye Jia, Michelle Tadmor Ramanovich, Quan Wang, Heiga Zen. *CVSS Corpus and Massively Multilingual Speech-to-Speech Translation.* LREC, 2022. — arXiv:2201.03713
- **[Metric]** Mingda Chen, Paul-Ambroise Duquenne, Pierre Andrews, Justine Kao, et al. *BLASER: A Text-Free Speech-to-Speech Translation Evaluation Metric.* ACL, 2023.
- **[Related]** Surafel M. Lakew, Yogesh Virkar, Prashant Mathur, Marcello Federico. *Isometric MT / Isochrony-Aware Neural Machine Translation for Automatic Dubbing.* Interspeech, 2022.
- **[Related]** Robert Bain? — omitted; see instead Derek Tam, Surafel M. Lakew, Yogesh Virkar, Prashant Mathur, Marcello Federico. *Jointly Optimizing Translations and Speech Timing to Improve Isochrony in Automatic Dubbing.* 2023.
- **[Related]** Robin San Roman, Pierre Fernandez, Alexandre Défossez, Teddy Furon, Tuan Tran, Hady Elsahar. *Proactive Detection of Voice Cloning with Localized Watermarking* (AudioSeal). ICML, 2024.

## 10. Worked Example

Source, an angry English utterance: **"I told you, I'm NOT going!"** — duration 2.10 s, 9 syllables, rate 4.29 syl/s, one 180 ms pause after "you", $F_0$ peak of $+1.8$ z on the stressed syllable of *not* at $t = 1.42$ s (68% through the utterance).

Spanish reference: **"¡Te dije que no voy a ir!"** — 8 syllables. The emphasized element *no* is the fourth syllable, at roughly 45% through the utterance if rate is uniform.

Now compute the metrics on a system output of duration 2.05 s with its own $F_0$ peak at 47%:

- **Rate:** $\rho = \log\frac{2.05/8}{2.10/9} = \log\frac{0.256}{0.233} = +0.094$. Near zero — scores as good.
- **Isochrony:** $|2.05 - 2.10| = 50$ ms — scores as excellent.
- **Emphasis position:** source 68%, output 47%. A raw normalized-time comparison scores this as a 21-point miss, yet the output is *correct*: Spanish puts the negation earlier. The metric penalizes the right answer.
- **Word-aligned $F_0$ correlation:** the alignment "I'm not going" → "no voy a ir" is non-monotone (negation moves left across the verb). After DTW on that alignment, the true pair gives log-$F_0$ Pearson $r \approx 0.31$; the same computation against a *randomly chosen other angry utterance by the same speaker* gives $r \approx 0.28$.

**The obstruction, made visible.** A separation of 0.03 in $r$ between the correct source and an unrelated source is far inside the per-utterance standard deviation (typically $>0.15$ at $n=1$). The metric cannot tell which source utterance the output was conditioned on. Any system that produces generic angry Spanish scores identically to one that genuinely transported the source contour — which is exactly the null hypothesis the Section 8 mismatched-source arm is designed to reject or confirm.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*