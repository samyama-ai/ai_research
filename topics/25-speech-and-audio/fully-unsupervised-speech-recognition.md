---
id: 25-speech-and-audio/fully-unsupervised-speech-recognition
title: "Unsupervised Speech Recognition Without Any Text"
topic: 25-speech-and-audio
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Unsupervised Speech Recognition Without Any Text

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/fully-unsupervised-speech-recognition` · **Status:** partially-solved

## 1. Problem Statement

Build a system that maps speech audio to a discrete symbol sequence that is *the language's own* phone or word inventory, using **only** unlabelled audio — no paired transcripts, no unpaired text corpus, no pronunciation lexicon, no grapheme-to-phoneme (G2P) rules, no cross-lingual seed from a written language.

The published "unsupervised ASR" line (wav2vec-U and descendants) does not solve this. It removes *paired* supervision but keeps an unpaired text corpus, a phonemizer, and usually a language model. Those are text. The problem here is the strictly harder one, and it splits three ways:

- **Method variant.** Given audio only, produce a segmentation and a discrete labelling that a native reader could align to orthography with a small key. Solving it means the alignment key is short (a few hundred utterances) and everything else transfers.
- **Measurement variant.** Decide what number scores such a system. WER is undefined without an orthographic anchor. Candidates — ABX discriminability, phone-normalized mutual information (PNMI), unit bitrate, spoken-language-model probes — each measure something adjacent to, but not identical to, "recognises speech."
- **Theory variant.** Is the map from acoustics to the language's phone inventory identifiable from the audio marginal alone, up to relabelling? Or is it identifiable only up to a coarser equivalence, so that no amount of audio fixes it?

## 2. Formal Setting

Let $X \in \mathcal{X}$ be an utterance waveform with distribution $P_X$, and let $Y = (y_1,\dots,y_T) \in \mathcal{A}^{*}$ be its latent phone sequence over the language's inventory $\mathcal{A}$, $|\mathcal{A}| = K$ (39 for the reduced TIMIT set, 41 for the wav2vec-U LibriSpeech setting). The generative model is $P_{X} = \sum_{y} P_{Y}(y)\, P_{X\mid Y}(\cdot \mid y)$, with $P_Y$ the phonotactic language model and $P_{X\mid Y}$ the acoustic channel.

**Measured quantities.**

- *Frame representation.* $z_t = f_\theta(X)_t \in \mathbb{R}^{d}$, $d=768$ or $1024$, at 50 Hz, from a frozen self-supervised encoder (wav2vec 2.0, HuBERT). Measured as the layer activation; the layer index is a tuned hyperparameter, not derived.
- *Segmentation.* $s = (b_1 < \dots < b_M)$, boundary indices. Measured against forced alignments by boundary F1 with a 20 ms tolerance.
- *Unit quality.* For discovered units $u$ and reference phones $y$ on force-aligned frames,
$$\mathrm{PNMI} = \frac{I(u;y)}{H(y)} \in [0,1].$$
Measured on held-out force-aligned data — which requires a supervised aligner, so **the metric itself consumes the supervision the method disclaims**.
- *Discriminability.* ABX error: for triphone tokens $a,b$ with $a\ne b$ and a second instance $x$ of $a$,
$$\mathrm{ABX} = \Pr\big[\, d(x,b) \le d(x,a) \,\big],$$
$d$ = DTW-averaged frame distance. Reported within- and across-speaker.
- *Recognition.* $\mathrm{PER}$ / $\mathrm{WER}$ after applying a map $\pi: \mathcal{U} \to \mathcal{A}$. Without text, $\pi$ is unavailable, so WER is only reportable under an *oracle* $\pi$ fitted on held-out alignments.

**The identifiability question.** Let $g: \mathcal{X} \to \mathcal{U}^{*}$ be the learned transcriber. Text-based unsupervised ASR picks $g$ by distribution matching, $\min_g D\big(g_\\# P_X \;\|\; P_Y\big)$, typically with $D$ a GAN discriminator loss. Delete $P_Y$ and the objective is empty: for any bijection $\pi$ of $\mathcal{U}$, $\pi \circ g$ is exactly as good. The best available substitute is a self-consistency objective — maximise $I(u; \text{future } u)$, minimise bitrate $H(u)/\text{sec}$ subject to an ABX floor — which is invariant to $\pi$ by construction.

**Assumptions, and which are violated.**

1. *A discrete latent of size $\approx K$ exists and is acoustically separable.* Violated for reduced vowels and flaps; English /t/ surfaces as [t], [ɾ], [ʔ] with distinct acoustics.
2. *Segment durations are informative and roughly stationary.* Violated in spontaneous speech; wav2vec-U's segmentation degrades sharply off read speech.
3. *Speaker and channel factor out of $z_t$.* Partly violated — ABX across-speaker error is consistently 1.5–2$\times$ within-speaker error for the same representation.
4. *The unpaired text is drawn from the same $P_Y$ as the audio's transcripts.* This is the assumption the present problem deletes entirely, and it is the load-bearing one.

## 3. State of the Art

**Empirical SOTA, with text (established).** wav2vec-U (Baevski, Hsu, Conneau, Auli, NeurIPS 2021) matched frozen wav2vec 2.0 features to a phonemized unpaired text corpus with a GAN, then self-trained: TIMIT PER 11.3% in the matched setting, and LibriSpeech test-other WER 5.9% with self-training and a language model — roughly the 2016 supervised DeepSpeech 2 number, reached with zero paired data. Independently reproduced; the ablations (feature layer, segmentation, GAN smoothness penalty) are in the paper. wav2vec-U 2.0 (Liu, Hsu, Baevski, Auli, SLT 2022) removed the k-means/PCA/segmentation preprocessing and trained end-to-end from raw features with comparable or better WER. REBORN (Tseng et al., NeurIPS 2024) replaced heuristic segmentation with an RL-learned boundary policy alternating with the GAN, improving over wav2vec-U 2.0 on LibriSpeech, TIMIT and MLS.

**Empirical SOTA, without text (established, but on different metrics).** The ZeroSpeech benchmarks (Dunbar et al., ASRU 2017; Interspeech 2021) score text-free systems by ABX, bitrate, and spoken-LM probes (sWUGGY, sBLIMP) — never by WER. HuBERT-style units (Hsu et al., TASLP 2021) plus k-means give the best ABX and the best downstream spoken language models (GSLM, Lakhotia et al., TACL 2021). Unsupervised word segmentation on discovered units (Kamper, TASLP 2023) recovers word boundaries well above chance but far below forced alignment.

**Claimed but unablated.** Several papers describe their pipelines as "unsupervised speech recognition" while phonemizing text with a hand-built G2P and tuning the feature layer on a labelled dev set. Neither cost is ablated: nobody has reported wav2vec-U's WER when the layer, the GAN checkpoint, and the number of clusters are all chosen without touching labels. Claims that decipherment-style methods (Klejch et al., Interspeech 2022) extend to genuinely unwritten languages exist as benchmark numbers on written languages held out from training, not as demonstrations on unwritten ones.

## 4. What Is Known

- **Text-free supervision is not the bottleneck for features.** HuBERT-base layer-9 k-means with 500 clusters reaches PNMI in the 0.6–0.7 range against forced-aligned LibriSpeech-960h phones — most phone identity is already linearly present in unlabelled representations.
- **A modest text corpus closes the gap to supervised.** LibriSpeech-960h audio + the LibriSpeech LM text corpus, zero pairs: 5.9% WER test-other (wav2vec-U + self-training).
- **Segmentation quality is the dominant error source in the GAN pipeline.** REBORN's gains come entirely from replacing the boundary heuristic, holding features and text fixed.
- **The GAN objective is unstable at scale.** wav2vec-U requires unsupervised model selection by a heuristic combining generator loss and LM agreement; runs with identical hyperparameters differ by several WER points.
- **Across-speaker ABX error is 1.5–2$\times$ within-speaker error** for every representation on ZeroSpeech 2017/2021 — speaker information is never fully removed.
- **The gap is asymmetric.** Adding ~10 minutes of paired audio to a text-free unit system recovers most of the recognition performance; adding 100$\times$ more audio to a text-free system does not.

## 5. What Is Not Known

- **Theoretically open.** Whether $g$ is identifiable from $P_X$ alone up to a bijection of $\mathcal{A}$. No impossibility theorem and no identifiability theorem exists for the speech case. The nearest result is negative-by-analogy: Søgaard, Ruder, Vulić (ACL 2018) showed the isomorphism assumption underlying unsupervised bilingual dictionary induction fails for distant language pairs — the same assumption unsupervised ASR relies on.
- **Methodologically blocked.** There is no text-free metric that means "recognises speech." PNMI and ABX both require forced alignments, i.e. a supervised aligner. Spoken-LM probes (sWUGGY) require a lexicon. So the field cannot currently score the thing it names without importing the supervision it claims to avoid.
- **Empirically open.** Nobody has run the scale test: does ABX/PNMI keep improving from 60k hours to 1M hours of unlabelled monolingual audio, or does it saturate? The compute exists; the run has not been reported.
- **Empirically open.** Whether a non-text anchor — images, video, articulatory or physiological signals — suffices to break the labelling symmetry at ASR-grade accuracy. Visually grounded speech (Harwath, Torralba, Glass, NeurIPS 2016) breaks it partially, for concrete nouns only.

## 6. Why It Is Hard

**Non-identifiability, not compute.** The training signal in text-free learning is invariant to permutations of the label set. Any $\pi \circ g$ has identical loss, so the objective cannot select the language's inventory over $39! \approx 2\times10^{46}$ relabellings. Text supplies the symmetry-breaking prior $P_Y$; delete it and the symmetry is exact, not approximate. More audio does not shrink an exactly invariant orbit.

**The evaluation does not measure what it names.** "Unsupervised ASR" is scored by WER, but WER is only defined once $\pi$ is fixed by labels. Every published WER for a text-free system is therefore an *oracle-mapped* number, which upper-bounds what a deployable text-free system could achieve and does not measure whether the system could have found $\pi$ itself.

**Absent ground truth for the target languages.** The languages where this matters are unwritten. For those, no forced aligner, no lexicon, and no test set exist — so the metric collapses along with the supervision.

## 7. Current Research (as of 2026)

- **Segmentation-as-policy.** REBORN's iterative RL boundary learning (NTU / Hung-yi Lee's group with Shao-Hua Sun) is the strongest current direction inside the GAN framework; extensions to text-free self-consistency objectives are underway *(frontier — verify)*.
- **Massively multilingual pretraining as an anchor.** MMS (Pratap et al., 2023, Meta) covers 1000+ languages with small paired sets; the open question is whether the shared phone space it induces can serve as $P_Y$ for a language with no text at all *(frontier — verify)*.
- **Textless spoken language modelling.** GSLM descendants at Meta AI and Inria/ENS treat discovered units as the end product, sidestepping orthography. This is the honest text-free track; it does not yet report anything WER-like.
- **Grounding as symmetry-breaking.** Visually- and video-grounded speech (MIT, Illinois) for concrete vocabulary; coverage remains the limit.
- **Benchmark reform.** Work on evaluations that do not presuppose an aligner — bitrate-constrained ABX, unit-based lexicon induction *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Can the label-permutation symmetry be broken by phonotactic self-consistency alone, or does it need text?

**Scale.** LibriSpeech-960h audio, HuBERT-large layer-18 features, 500 k-means units. Train a text-free segmenter+labeller into a $K=41$ inventory by maximising a permutation-invariant objective: unit-sequence LM log-likelihood minus a bitrate penalty, with duration and boundary regularisers. ~8 GPU-days on 8 A100s — small enough for one lab.

**Arms.**
1. *Text-free* (the test arm): objective above, $\pi$ never seen.
2. *Control — text arm:* identical architecture and features, wav2vec-U GAN against phonemized LibriSpeech LM text.
3. *Control — oracle map:* text-free arm, then $\pi$ fitted on 100 held-out force-aligned utterances (the "short key").

**Deciding number.** PER on LibriSpeech dev-clean under a *frequency-matched* $\pi$ — $\pi$ chosen by matching unit unigram frequencies to a published English phone frequency table, using no audio-side labels. If arm 1 with frequency-matched $\pi$ lands within **5 PER points** of arm 3's oracle-mapped PER, the symmetry is breakable without text and the problem is empirically open, not blocked. If the gap exceeds 20 points — the outcome I expect — frequency structure alone is insufficient and the anchoring must come from higher-order phonotactics or a non-text modality.

## 9. Key References

- **[Foundational]** Alexei Baevski, Wei-Ning Hsu, Alexis Conneau, Michael Auli. *Unsupervised Speech Recognition.* NeurIPS, 2021. — arXiv:2105.11084
- **[Foundational]** Da-Rong Liu, Kuan-Yu Chen, Hung-yi Lee, Lin-shan Lee. *Completely Unsupervised Phoneme Recognition by Adversarially Learning Mapping Relationships from Speech to Phonemes.* ICASSP, 2018.
- **[Foundational]** Chih-Kuan Yeh, Jianshu Chen, Chengzhu Yu, Dong Yu. *Unsupervised Speech Recognition via Segmental Empirical Output Distribution Matching.* ICLR, 2019.
- **[SOTA]** Alexander H. Liu, Wei-Ning Hsu, Michael Auli, Alexei Baevski. *Towards End-to-end Unsupervised Speech Recognition.* IEEE SLT, 2022.
- **[SOTA]** Liang-Hsuan Tseng, En-Pei Hu, Cheng-Han Chiang, Yuan Tseng, Hung-yi Lee, Lin-shan Lee, Shao-Hua Sun. *REBORN: Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR.* NeurIPS, 2024.
- **[SOTA]** Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.* IEEE/ACM TASLP, 2021.
- **[Benchmark]** Ewan Dunbar et al. *The Zero Resource Speech Challenge 2017.* ASRU, 2017.
- **[Benchmark]** Tu Anh Nguyen et al. *The Zero Resource Speech Benchmark 2021: Metrics and Baselines for Unsupervised Spoken Language Modeling.* Interspeech / NeurIPS SAS workshop, 2021.
- **[Related]** Kushal Lakhotia et al. *On Generative Spoken Language Modeling from Raw Audio.* TACL, 2021.
- **[Related]** David Harwath, Antonio Torralba, James Glass. *Unsupervised Learning of Spoken Language with Visual Context.* NeurIPS, 2016.
- **[Related]** Ondřej Klejch, Electra Wallington, Peter Bell. *Deciphering Speech: a Zero-Resource Approach to Cross-Lingual Transfer in ASR.* Interspeech, 2022.
- **[Related]** Anders Søgaard, Sebastian Ruder, Ivan Vulić. *On the Limitations of Unsupervised Bilingual Dictionary Induction.* ACL, 2018.
- **[Survey]** Herman Kamper. *Word Segmentation on Discovered Phone Units with Dynamic Programming and Self-Supervised Scoring.* IEEE/ACM TASLP, 2023.

## 10. Worked Example

Take LibriSpeech dev-clean, HuBERT-base layer 9, $k$-means with $K=41$ clusters. Frame-level PNMI against forced-aligned phones: about 0.65 — the units carry roughly two-thirds of the phone entropy. Fit the oracle map $\pi^{*}$ by majority phone per cluster on held-out alignments and you get a usable phone recogniser.

Now delete the alignments and try to recover $\pi$ from the audio alone.

*Attempt: unigram frequency matching.* English phone unigram frequencies from a large corpus are roughly /ə/ 8.9%, /n/ 6.9%, /t/ 6.6%, /ɪ/ 6.3%, /s/ 4.7%, /d/ 4.2%, /l/ 4.0%, /ð/ 3.9%, /r/ 3.8%, /m/ 3.1%. Sort the 41 cluster frequencies and match rank-to-rank. The top three ranks are separated by 2.0 and 0.3 percentage points; the cluster frequency estimates on 5.4 hours of dev-clean have a standard error of roughly 0.1–0.2 points, and cluster impurity shifts each estimate by more than that. Ranks 6–10 span 4.2% down to 3.1% — five phones inside a 1.1-point band. Rank matching there is a coin flip.

*The arithmetic.* Suppose frequency matching pins the 6 most frequent phones correctly and leaves the remaining 35 unresolved. The residual ambiguity is $35! \approx 1.0\times10^{40}$ orderings. Each wrong assignment costs its phone's full token mass, so with 35 of 41 phones randomly permuted the expected phone accuracy is roughly the mass of the 6 fixed phones — about 37% — against a PER near 25% (accuracy ~75%) under $\pi^{*}$. That is a 38-point accuracy gap arising from *nothing but the choice of labels*: the acoustic model is bit-identical in both cases.

**What this makes visible.** The failure is not representation quality — PNMI 0.65 says the units are good. It is that the objective is flat across $10^{40}$ labellings and the only signal that is not flat, unigram frequency, has a resolution of about one percentage point where the phones are spaced a tenth of that apart. Higher-order phonotactics would break the tie, but higher-order phonotactics estimated from *text* is exactly what wav2vec-U uses and what this problem forbids. Roughly 100 aligned utterances collapse the $10^{40}$ orbit to a single point. That is the price of the anchor, and no quantity of unlabelled audio has been shown to substitute for it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*