---
id: 24-multimodal/blind-baseline-benchmark-validity
title: "Blind Baselines Beating Multimodal Benchmarks"
topic: 24-multimodal
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Blind Baselines Beating Multimodal Benchmarks

> **Topic:** Multimodal Models · **ID:** `24-multimodal/blind-baseline-benchmark-validity` · **Status:** partially-solved

## 1. Problem Statement

A multimodal benchmark claims to measure whether a model uses a non-text modality (image, video, audio, embodied observation) to answer a question. A **blind baseline** is a model given the question and answer options but not the modality. When the blind baseline scores far above chance — sometimes above the multimodal model it is meant to bound — the benchmark's headline number is not evidence of the capability it names.

Three variants, of very different difficulty:

- **Measurement.** Given a benchmark $B$ and a model family, estimate how much of the achievable score is attributable to the visual input. Requires a definition of "blind score" that is not an artifact of how the modality was removed.
- **Method.** Build benchmarks (or debias existing ones) such that the blind ceiling is provably near chance, without destroying the naturalness or difficulty of the task.
- **Theory.** Characterise when a finite dataset of $(x, q, a)$ triples admits a blind predictor of accuracy $\ge \alpha$, and whether any balancing procedure can drive the blind Bayes accuracy to chance while keeping the label distribution non-degenerate.

Solving it means: for a stated benchmark, a reported number $\nu$ with a confidence interval saying what fraction of the score requires the modality, plus a construction procedure that certifies $\nu$ close to 1 in advance.

## 2. Formal Setting

Benchmark $B = \{(x_i, q_i, \mathcal{O}_i, a_i)\}_{i=1}^{n}$: modality input $x_i \in \mathcal{X}$, question $q_i$, option set $\mathcal{O}_i$ ($|\mathcal{O}_i| = k$ for multiple choice), gold answer $a_i \in \mathcal{O}_i$. A sighted model is $f: \mathcal{X} \times \mathcal{Q} \to \mathcal{A}$; a blind model is $g: \mathcal{Q} \to \mathcal{A}$.

Measured quantities:

$$\mathrm{Acc}(f) = \frac{1}{n}\sum_i \mathbb{1}[f(x_i,q_i)=a_i], \qquad \mathrm{Acc}_{\mathrm{blind}} = \sup_{g \in \mathcal{G}} \frac{1}{n}\sum_i \mathbb{1}[g(q_i)=a_i]$$

where $\mathcal{G}$ is the *searched* blind class — in practice a handful of prompted LLMs, or one fine-tuned question-only model. $\mathrm{Acc}_{\mathrm{blind}}$ is a lower bound on the blind Bayes accuracy $\alpha^\star = \mathbb{E}_q[\max_a \Pr(a \mid q)]$, never an estimate of it.

**Visual necessity index** (headroom-normalised, so it is not inflated by an easy benchmark):

$$\nu(B, f) = \frac{\mathrm{Acc}(f) - \mathrm{Acc}_{\mathrm{blind}}}{1 - \mathrm{Acc}_{\mathrm{blind}}} \in (-\infty, 1]$$

$\nu \le 0$ means blindness costs nothing. Chance is $c = 1/k$ for balanced $k$-way choice; for open-ended answers $c$ is undefined and must be replaced by a prior-answer baseline ("most frequent answer per question type", Antol et al. 2015).

Information-theoretic target: the benchmark is *visually grounded* iff $I(A; X \mid Q) > 0$ at appreciable magnitude. The empirical estimator of this is exactly the blind/sighted gap, so the same confound applies.

**Assumptions, and which fail in practice:**

1. *Masking $x$ is a clean intervention.* Violated: dropping the image shifts the input distribution off the model's instruction-tuning manifold; some models refuse, which understates $\mathrm{Acc}_{\mathrm{blind}}$.
2. *$q$ carries no image content.* Violated routinely — captions, OCR'd text and answer options ("Which of these is a Golgi apparatus?") leak the answer.
3. *$\mathcal{G}$ is rich enough that $\mathrm{Acc}_{\mathrm{blind}} \approx \alpha^\star$.* Violated; the gap is unbounded and unmeasured.
4. *No contamination.* Violated: public benchmarks appear in pretraining corpora, so a blind LLM may retrieve rather than guess.
5. *Items are i.i.d.* Violated: benchmarks are built by small annotator pools, so artifacts are correlated across items and per-item bootstrap CIs understate variance.

## 3. State of the Art

**Established (ablated, reproduced):**

- Question-only baselines on VQA v1 reach ~48.8% overall and ~78% on yes/no questions (Antol et al., ICCV 2015; Agrawal et al., EMNLP 2016). Balancing by construction — every question paired with two images with different answers — is the only intervention shown to collapse this: VQA v2 (Goyal et al., CVPR 2017) and Yin-and-Yang (Zhang et al., CVPR 2016) both cut blind accuracy substantially while keeping questions natural.
- Blindfold baselines for Embodied QA are competitive with the full navigation-plus-vision agent (Anand et al., ViGIL @ NeurIPS 2018); single-modality baselines match or beat multimodal models on vision-and-language navigation and QA tasks (Thomason et al., NAACL 2019).
- The same failure in text-only NLI: hypothesis-only baselines reach ~67% on SNLI against a 34% majority class (Poliak et al., \*SEM 2018; Gururangan et al., NAACL 2018). This established the diagnostic protocol multimodal work inherited.

**Claimed but only partially ablated:**

- MMStar (Chen et al., NeurIPS D&B 2024) reports that strong text-only LLMs exceed random on several LVLM benchmarks and that some LVLMs *lose little* when the image is withheld. The finding is robust in direction; the specific per-benchmark deltas are single-run prompted numbers, not fine-tuned blind ceilings.
- MMMU-Pro (Yue et al., 2024) filters out questions four text-only LLMs can answer and expands option sets from 4 to 10, dropping model scores by roughly 17–27 points relative to MMMU. This is a benchmark number: the filter removes items *this* blind class solved, not items that are blind-solvable in principle.
- "Eyes Wide Shut?" (Tong et al., CVPR 2024) shows CLIP-blind pairs that MLLMs cannot distinguish — evidence that even the sighted pathway is weak, which is a different claim from the blind baseline claim and is often conflated with it.

No method certifies a blind ceiling in advance. Debiasing is post-hoc filtering everywhere except VQA v2-style paired construction.

## 4. What Is Known

- VQA v1 → v2 (~1.1M questions, 200k COCO images): the balanced construction dropped a question-only LSTM from ~48.8% to ~44%, and dropped the sighted-model gain attributable to priors; the language-prior exploit did not survive pairing. Scale: full benchmark, multiple independently trained models.
- Binary abstract-scene VQA: blind question-only accuracy ~64% collapses toward 50% on balanced complementary pairs (Zhang et al. 2016, tens of thousands of pairs).
- SNLI hypothesis-only: 67% vs 34% majority, 570k pairs, reproduced by two independent groups in the same year.
- MMMU (11.5k questions, 30 subjects): four-way-and-up multiple choice with a random baseline near 22–26%; text-only LLMs score well above that, which is why MMMU-Pro exists. The MMMU-Pro option expansion to 10 choices and text-only filtering produced double-digit score drops on GPT-4o-class models.
- Cross-modal interaction is measurable but small: Hessel & Lee (EMNLP 2020) show that on several multimodal datasets, an empirical multimodal-interaction estimator finds statistically detectable but modest cross-modal signal — and that naive gap comparisons overstate it.

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted definition of "the blind score". Masking, black-image substitution, shuffled-image substitution and text-only prompting give different numbers on the same benchmark, and nobody has shown which estimates $\alpha^\star$. Nor is there an agreed contamination-free protocol, so a high blind score cannot be attributed to dataset artifacts versus memorisation.
- **Theoretically open.** No result bounds the gap between the searched blind class $\mathcal{G}$ and $\alpha^\star$, and no theorem says whether a balancing procedure exists that drives $\alpha^\star \to 1/k$ for open-ended, expert-authored items (VQA v2's pairing works only because answers are short and image pairs are findable).
- **Empirically open.** The runnable-but-unrun experiment: fine-tune a large blind model directly on each major multimodal benchmark's training split and report $\mathrm{Acc}_{\mathrm{blind}}$ under that stronger $\mathcal{G}$. Everything published uses prompted zero/few-shot blind models, which is the weakest possible probe.

## 6. Why It Is Hard

The obstruction is **non-identifiability of the blind ceiling under a confounded intervention**. $\mathrm{Acc}_{\mathrm{blind}}$ is a supremum over an unbounded function class, estimated by evaluating two or three prompted LLMs; the estimate can only move up as models improve, so today's "clean" benchmark is tomorrow's blind-solvable one, and no finite experiment certifies cleanliness. Compounding it: removing the image is not a do-operator on the modality alone — it also moves the input off-distribution for an instruction-tuned model, so a low blind score is equally consistent with "no artifact" and with "the blind model refused". And contamination means $\Pr(a \mid q)$ measured on a public benchmark is not the artifact distribution the benchmark author created. Three unmeasured quantities, one observed number.

## 7. Current Research (as of 2026)

- **Construction-time balancing beyond VQA v2** — generating paired items so the answer flips with the image, now via generative image editing rather than retrieval. *(frontier — verify)*
- **Filter-then-harden pipelines** (MMMU-Pro lineage, Yue and collaborators at Yale/Waterloo/CMU): text-only filtering plus option-set expansion plus vision-only screenshot input.
- **Per-item necessity scoring** rather than benchmark-level averages — reporting the fraction of items with a positive blind/sighted delta, which MMStar-style audits (Shanghai AI Lab / CUHK) have pushed toward.
- **Interaction estimators** as a replacement for the gap statistic, following Hessel & Lee; still rarely used in LVLM leaderboards. *(frontier — verify)*
- **Contamination-controlled blind baselines** using post-cutoff or privately held items. Held-out variants exist for a few benchmarks; systematic adoption does not. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does the published blind baseline understate the true blind ceiling by enough to change benchmark rankings?

- **Scale.** Five benchmarks (MMMU, MathVista, ScienceQA-IMG, MMBench, a video QA set), full validation splits, ~2k–11k items each. Fine-tune one 7–8B open LLM per benchmark on the *training* split with images stripped (question + options + answer only), ~3 epochs, single 8×A100 node-day per benchmark. Total ≈ 5 node-days.
- **Control arms.** (a) The published prompted zero-shot text-only baseline. (b) The same fine-tuned model with questions shuffled against answers (measures memorisation of the answer distribution, not the artifact). (c) A held-out private set of 300 newly authored items per benchmark, matched in format, guaranteed contamination-free.
- **Deciding number.** $\Delta = \mathrm{Acc}_{\mathrm{blind}}^{\text{fine-tuned}} - \mathrm{Acc}_{\mathrm{blind}}^{\text{prompted}}$ on the public split, minus the same difference on the private split. If $\Delta_{\text{public}} - \Delta_{\text{private}} > 5$ points, the published baselines are understating the ceiling for artifact reasons (not contamination) and every $\nu$ in the literature is too high. If it is under 2 points, prompted blind baselines are adequate and the community can stop arguing about the probe.

## 9. Key References

- **[Foundational]** S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, D. Parikh. *VQA: Visual Question Answering.* ICCV, 2015. — arXiv:1505.00468
- **[Foundational]** Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, D. Parikh. *Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering.* CVPR, 2017. — arXiv:1612.00837
- **[Foundational]** P. Zhang, Y. Goyal, D. Summers-Stay, D. Batra, D. Parikh. *Yin and Yang: Balancing and Answering Binary Visual Questions.* CVPR, 2016. — arXiv:1511.05099
- **[Foundational]** A. Agrawal, D. Batra, D. Parikh. *Analyzing the Behavior of Visual Question Answering Models.* EMNLP, 2016. — arXiv:1606.07356
- **[Method]** A. Anand, E. Belilovsky, K. Kastner, H. Larochelle, A. Courville. *Blindfold Baselines for Embodied QA.* ViGIL Workshop, NeurIPS, 2018. — arXiv:1811.05013
- **[Method]** J. Thomason, D. Gordon, Y. Bisk. *Shifting the Baseline: Single Modality Performance on Visual Navigation & QA.* NAACL, 2019.
- **[Method]** A. Poliak, J. Naradowsky, A. Haldar, R. Rudinger, B. Van Durme. *Hypothesis Only Baselines in Natural Language Inference.* \*SEM, 2018.
- **[Method]** S. Gururangan, S. Swayamdipta, O. Levy, R. Schwartz, S. R. Bowman, N. A. Smith. *Annotation Artifacts in Natural Language Inference Data.* NAACL, 2018.
- **[Measurement]** J. Hessel, L. Lee. *Does my multimodal model learn cross-modal interactions? It's harder to tell than you might think!* EMNLP, 2020. — arXiv:2010.06572
- **[SOTA]** L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, F. Zhao. *Are We on the Right Way for Evaluating Large Vision-Language Models?* (MMStar). NeurIPS Datasets & Benchmarks, 2024. — arXiv:2403.20330
- **[SOTA]** X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, et al. *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI.* CVPR, 2024. — arXiv:2311.16502
- **[SOTA]** X. Yue, T. Zheng, Y. Ni, et al. *MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark.* 2024. — arXiv:2409.02813
- **[Related]** S. Tong, Z. Liu, Y. Zhai, Y. Ma, Y. LeCun, S. Xie. *Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs.* CVPR, 2024. — arXiv:2401.06209
- **[Survey]** A. Torralba, A. A. Efros. *Unbiased Look at Dataset Bias.* CVPR, 2011.

## 10. Worked Example

Take a 4-option MMMU-style subject slice, $n = 500$, $k = 4$, so $c = 0.25$.

- Sighted model: $\mathrm{Acc}(f) = 0.58$.
- Published prompted text-only baseline: $\mathrm{Acc}_{\mathrm{blind}} = 0.38$.

$$\nu = \frac{0.58 - 0.38}{1 - 0.38} = 0.32$$

Reported conclusion: about a third of the headroom is visual. Now run the fine-tuned blind arm and suppose it reaches $0.47$ — plausible, because option sets in expert exams carry strong distractor style cues (one long specific option, three short generic ones).

$$\nu' = \frac{0.58 - 0.47}{1 - 0.47} = 0.21$$

The visual contribution fell by a third, and the model never saw an image in either blind arm. Now the obstruction: which number is right? $0.38$ and $0.47$ are both lower bounds on $\alpha^\star$. A larger blind model next year may hit $0.52$, giving $\nu'' = 0.125$. Nothing in the benchmark changed. The reported "visual reasoning" score is a function of the strength of the probe that nobody standardised.

Worse, the ranking is not stable. If model $A$ scores $0.58$ and model $B$ scores $0.55$, $A$ leads by 3 points. Restricted to the $\sim 210$ items the fine-tuned blind model *fails*, suppose $A$ gets $0.31$ and $B$ gets $0.36$ — the ordering inverts. The leaderboard was reporting, in large part, which model better exploits the same option-set artifacts the blind model exploits. That inversion is measurable today, on a single node-day, and has not been published for any major LVLM leaderboard.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*