---
id: 24-multimodal/contamination-free-multimodal-evaluation
title: "Contamination-Free Multimodal Evaluation"
topic: 24-multimodal
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Contamination-Free Multimodal Evaluation

> **Topic:** Multimodal Models · **ID:** `24-multimodal/contamination-free-multimodal-evaluation` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a multimodal model $f$ trained on an undisclosed corpus $D$ of image–text (and video/audio–text) pairs, and a benchmark $B = \{(x_i, q_i, a_i)\}$ of image $x_i$, question $q_i$, answer $a_i$, decide whether the score $S(f, B)$ measures generalization or memory of $B$-correlated content in $D$.

Three variants, of very different difficulty:

- **Measurement.** Define and estimate a contamination quantity for image–text pairs. Unlike text, there is no exact-match primitive: an image can be re-encoded, cropped, watermarked, upsampled, or *described in prose* elsewhere in $D$. The unit of overlap is undefined.
- **Method.** Build a benchmark or a decontamination procedure whose scores are provably (or auditably) uncontaminated — held-out-in-time, private, or procedurally generated.
- **Theory.** State conditions under which a black-box test can certify $B \not\subset D$ (or bound the score inflation) without access to $D$ or to model logits.

Solved would mean: for a frontier model with a closed corpus, reporting $S(f,B)$ together with a defensible interval $[S_{\text{clean}}^-, S_{\text{clean}}^+]$ on what the score would have been had $B$-related content been absent from $D$, with the interval validated against a known-contamination control.

## 2. Formal Setting

Let $D = \{(u_j, t_j)\}_{j=1}^{N}$ be the training corpus ($u$ image, $t$ text), $N \approx 10^9$–$10^{10}$ for open sets such as LAION-5B (5.85B pairs; Schuhmann et al., NeurIPS 2022) and unknown for frontier models.

**Overlap.** For an embedding $\phi$ (CLIP image tower, DINOv2, or a perceptual hash) and threshold $\tau$, the image-side contamination rate is
$$c_{\mathrm{img}}(\tau) = \frac{1}{|B|}\sum_{i=1}^{|B|} \mathbf{1}\!\left[\max_{j\le N} \mathrm{sim}\big(\phi(x_i), \phi(u_j)\big) > \tau\right].$$
Measured by an approximate-nearest-neighbour index over $N$ vectors; cost is $O(N)$ index build plus $O(|B|\log N)$ queries. $\tau$ is a free parameter, and the estimate is monotone but not calibrated in $\tau$: no value of $\tau$ corresponds to "the model saw this".

**Semantic (cross-modal) contamination.** The answer $a_i$ may be recoverable from text alone:
$$c_{\mathrm{txt}} = \Pr_{f_{\text{LM}}}\big[f_{\text{LM}}(q_i) = a_i\big] - r,$$
with $r$ the random-choice baseline. This is the leakage MMStar formalizes as *multimodal leakage* $\mathrm{ML}$ alongside *multimodal gain* $\mathrm{MG} = S(f\mid x) - S(f\mid \varnothing)$ (Chen et al., NeurIPS 2024).

**Inflation.** The target quantity is causal:
$$\Delta = \mathbb{E}\big[S(f_D, B)\big] - \mathbb{E}\big[S(f_{D \setminus \mathcal{N}(B)}, B)\big],$$
where $\mathcal{N}(B)$ is the set of $B$-related training points. Estimating $\Delta$ requires retraining under ablation of $\mathcal{N}(B)$ — $\ge 2$ pretraining runs at frontier scale.

**Assumptions and their violations.**
- *$D$ is inspectable* — false for GPT-class, Gemini-class and Claude-class models.
- *Contamination is a set-membership property* — false: a paraphrased caption or a textbook passage discussing the same figure inflates $S$ without any near-duplicate image.
- *Exchangeability of benchmark example order*, used by Oren et al. (ICLR 2024) for black-box contamination proofs — requires log-likelihoods, which frontier multimodal APIs do not expose, and requires that $B$ have a canonical order that was itself memorized.
- *$\phi$-similarity tracks memorization* — unvalidated; a $\tau$ that flags ImageNet-scale duplicates also flags distinct photos of the same landmark.

## 3. State of the Art

**Established.**
- *Overlap auditing with reported effect size.* The CLIP paper (Radford et al., ICML 2021) ran duplicate detection between WIT and 35 downstream sets: median detected overlap 2.2%, and only a handful of datasets showed statistically significant accuracy differences between the overlapping and clean splits, with shifts under about 1 point. This is the single most-cited evidence that image-side overlap is small in effect — but it was measured on the authors' own corpus with their own detector.
- *Visual-dependency auditing.* MMStar (Chen et al., NeurIPS 2024) showed LLMs given only the question text exceed the random baseline on several multimodal benchmarks, and built a 1,500-sample set filtered for visual necessity.
- *Harder re-renderings.* MMMU-Pro (Yue et al., 2024) adds a 10-option setting and a vision-only setting where the question is rendered into the image; reported accuracy drops of roughly 17–27 points relative to MMMU across evaluated models.
- *Dedup at web scale is feasible.* LAION-2B de-duplication analyses (Webster et al., 2023) found large near-duplicate clusters; Lee et al. (ACL 2022) established that text dedup changes memorization behaviour.

**Claimed but unablated.**
- That MMMU-Pro / MMStar / LiveBench-style refreshes *remove* contamination. What is shown is that scores drop. A score drop is jointly explained by contamination removal, increased difficulty, and format shift; no published multimodal work separates the three.
- That perceptual-hash decontamination of a benchmark is sufficient. Reported as a pipeline step, never ablated against a retrained control.

**Benchmark-number-only.** All "contamination-resistant multimodal benchmark" claims — MMMU-Pro, MMStar, live/rolling multimodal sets — exist as leaderboard deltas. None reports a $\Delta$ estimated against a retrained model.

**Text-side theory SOTA.** Oren et al. (ICLR 2024) give a sound exchangeability test that provably bounds false-positive rate for detecting a *test set* in a black-box LM; Golchin & Surdeanu (ICLR 2024) give a guided-instruction heuristic. Neither has a validated image analogue.

## 4. What Is Known

- **Text-only leakage is large on ostensibly visual benchmarks.** MMStar's audit (2024, on six benchmarks including MMMU, ScienceQA, MMBench, at 7B–GPT-4V scale) shows text-only LLMs beating chance on a substantial share of items; the paper's remedy discards the majority of candidate items to reach 1,500 visually necessary ones.
- **Reformatting costs 17–27 points.** MMMU-Pro vs MMMU, measured on frontier models circa 2024.
- **Membership inference barely works at pretraining scale.** Duan et al. (COLM 2024) find MIA on LLM pretraining data near chance (AUC ~0.5–0.6) once candidate and member distributions are matched — the same confound (temporal/topical shift) applies to image MIA.
- **Concept frequency dominates "zero-shot" performance.** Udandarao et al. (NeurIPS 2024) show multimodal downstream accuracy is log-linear in pretraining concept frequency across 34 models and 5 corpora — an exponential data requirement. Contamination is the tail of this same curve, not a separate phenomenon.
- **Train-test similarity is not the whole story for CLIP.** Mayilvahanan et al. (ICLR 2024) pruned LAION examples most similar to ImageNet-class test sets and found CLIP's zero-shot performance largely retained — the strongest existing *negative* evidence for image-side inflation, at LAION-200M-pruning scale.
- **Text benchmarks do inflate.** GSM1k (Zhang et al., 2024) found up to ~13-point drops for some model families on a freshly-collected GSM8K clone, with others showing none. No multimodal equivalent of GSM1k exists.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted definition of "this benchmark image was in training". Overlap is threshold-dependent, cross-modal leakage means the image need not appear at all, and the causal $\Delta$ is unmeasurable without corpus access. Every published multimodal contamination number is a proxy with no validation against ground truth.
- **Empirically open.** Whether a controlled retraining ablation ($\mathcal{N}(B)$ removed) at 1B-image scale shows $\Delta > 2$ points on MMMU-style items. Runnable today for ~$10^5$ GPU-hours; not run.
- **Empirically open.** Whether MMMU-Pro's 17–27 point drop is contamination or difficulty. Decidable by giving the *same* models the same items in both formats with matched difficulty controls.
- **Theoretically open.** Whether any black-box test over image inputs can certify non-membership with bounded false positives when the model exposes only sampled text. Oren et al.'s exchangeability argument does not transfer: image benchmarks have no canonical ordering that could be memorized, and log-probabilities are unavailable.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth.** Validating a contamination detector requires a corpus where membership is known. Frontier corpora are closed; open corpora (LAION, DataComp) are paired with models far weaker than the ones whose scores are disputed. The detector is therefore always evaluated on the wrong model.
2. **Non-identifiability across modalities.** $\Delta$ is confounded by concept frequency (Udandarao et al., 2024). Removing $\mathcal{N}(B)$ also removes concept mass; the ablated model is worse for two reasons and the design cannot separate them without a frequency-matched replacement set.
3. **The evaluation does not measure what it names.** A "multimodal reasoning" score that a text-only LLM partly reproduces is measuring text priors. Contamination and visual-necessity failure are entangled: fixing the first without the second changes the number for the wrong reason.

Compute is a fourth, secondary cost: the clean estimate needs paired pretraining runs, not fine-tuning.

## 7. Current Research (as of 2026)

- **Dynamic and live benchmarks.** LiveBench (White et al., ICLR 2025) and LiveCodeBench (Jain et al., ICLR 2025) use post-cutoff items; multimodal analogues with rolling image collection are being assembled by academic consortia *(frontier — verify)*.
- **Private held-out splits with API-side auditing.** Held by benchmark maintainers, scored on submission. Reduces leakage but destroys reproducibility and is vulnerable to repeated-query fitting.
- **Procedural generation.** DyVal (Zhu et al., ICLR 2024) for text; rendered-scene and synthetic-diagram generators for vision give provably novel items at the cost of distribution shift away from real images.
- **Canary insertion in open corpora.** DataComp/OpenCLIP-community efforts to insert marked images pre-pretraining and measure recall *(frontier — verify)*. This is the only route to ground truth.
- **Replication/memorization measurement in generative vision.** Somepalli et al. (CVPR 2023) and Carlini et al. (USENIX Security 2023) give extraction-based memorization evidence for diffusion models; adapting extraction as a contamination signal for discriminative VLM evaluation is active.

## 8. Concrete Next Experiment

**Canary-ablation pretraining pair.**

- **Scale.** Two CLIP-style or 3B-parameter VLM pretraining runs on DataComp-1B (1.4B pairs), identical seeds, schedules, and token budgets. Roughly $2 \times 5\times10^4$ A100-hours.
- **Treatment arm.** Inject $\mathcal{N}(B)$: for 500 held-out MMMU-style items, insert (a) the exact image with its answer in the caption, (b) a re-encoded/cropped variant, (c) a text-only prose description of the figure and answer, 100 items each plus 200 untouched controls.
- **Control arm.** Identical run with $\mathcal{N}(B)$ replaced by frequency-matched decoys — same concepts, same counts, no answer content. This is what makes $\Delta$ separable from concept frequency.
- **Deciding number.** $\Delta_c$ = accuracy gap between arms on the 300 injected items, per injection channel. If $\Delta_{\text{prose-only}} \ge 5$ points while image-side overlap detection at any $\tau$ recalls $<20\%$ of those items, the field's entire image-dedup practice is measuring the wrong channel, and the status stays methodologically blocked. If $\Delta_{\text{prose-only}} < 1$ point, image-side dedup is sufficient and the problem downgrades to empirically open.

Secondary readout: the ROC of each proxy detector (perceptual hash, CLIP-$\tau$, Min-K% Prob on caption text, MMStar's $\mathrm{ML}$) against the now-known membership labels. This is the first honest AUC for a multimodal contamination detector.

## 9. Key References

- **[Foundational]** Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. — arXiv:2103.00020 (§5 overlap analysis)
- **[Foundational]** Oren, Meister, Chatterji, Ladhak, Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024. — arXiv:2310.17623
- **[SOTA]** Chen et al. *Are We on the Right Way for Evaluating Large Vision-Language Models?* NeurIPS 2024 (MMStar). — arXiv:2403.20330
- **[SOTA]** Yue et al. *MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark.* 2024. — arXiv:2409.02813
- **[SOTA]** Yue et al. *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI.* CVPR 2024. — arXiv:2311.16502
- **[Evidence]** Udandarao et al. *No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance.* NeurIPS 2024. — arXiv:2404.04125
- **[Evidence]** Mayilvahanan, Wiedemer, Rusak, Bethge, Brendel. *Does CLIP's Generalization Performance Mainly Stem from High Train-Test Similarity?* ICLR 2024. — arXiv:2310.09562
- **[Evidence]** Duan et al. *Do Membership Inference Attacks Work on Large Language Models?* COLM 2024. — arXiv:2402.07841
- **[Evidence]** Zhang et al. *A Careful Examination of Large Language Model Performance on Grade School Arithmetic.* NeurIPS 2024 (GSM1k). — arXiv:2405.00332
- **[Method]** White et al. *LiveBench: A Challenging, Contamination-Limited LLM Benchmark.* ICLR 2025. — arXiv:2406.19314
- **[Method]** Zhu et al. *DyVal: Dynamic Evaluation of Large Language Models for Reasoning Tasks.* ICLR 2024. — arXiv:2309.17167
- **[Method]** Somepalli, Singla, Goldblum, Geiping, Goldstein. *Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models.* CVPR 2023. — arXiv:2212.03860
- **[Corpus]** Schuhmann et al. *LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models.* NeurIPS 2022 Datasets & Benchmarks. — arXiv:2210.08402
- **[Survey]** Xu, Wang, Ye et al. *Benchmark Data Contamination of Large Language Models: A Survey.* 2024. — arXiv:2406.04244

## 10. Worked Example

Take one MMMU item: a labelled diagram of a lac operon with the question "Which segment binds the repressor?", four options, answer *B*.

**Step 1 — image-side audit.** Query CLIP-ViT-L embeddings of the diagram against a LAION-2B index. Top-1 cosine similarity: 0.94. At $\tau = 0.95$ (a common dedup threshold), the item is *clean*. At $\tau = 0.90$, it is *contaminated*, and so are 40 other textbook diagrams that share layout conventions but different content. The label flips on a parameter nobody can calibrate. $c_{\mathrm{img}}(0.95) = 0.4\%$, $c_{\mathrm{img}}(0.90) = 6.1\%$ on a 500-item sample — a 15× swing.

**Step 2 — text-only probe.** Give a text-only 8B LLM the question and options, no image. It answers *B*. Random baseline is 25%; across the 500-item sample the text-only model scores 34%, i.e. $\mathrm{ML} \approx 9$ points recoverable without vision. The diagram never had to be in $D$: the phrase "operator binds the repressor" appears in every genetics textbook on the web.

**Step 3 — the gap.** Suppose the VLM scores 62% on the sample. Decompose: 25 points chance, 9 points text prior, 28 points unexplained. The audit at $\tau = 0.95$ says 0.4% image overlap, so the standard report is "negligible contamination". But step 2 already shows a 9-point channel that step 1 cannot see, and nothing in the pipeline bounds how much of the remaining 28 points is memorized answer text paired with a *different* rendering of the same diagram.

**The obstruction, made visible.** The measured quantity ($c_{\mathrm{img}}$) is threshold-unstable and blind to the dominant channel; the quantity that matters ($\Delta$) needs a retrained control that no one has run. That is why the status is methodologically blocked and not merely empirically open — running more evaluations on more benchmarks does not shrink the interval, because there is no interval defined.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*