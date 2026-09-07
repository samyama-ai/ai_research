---
id: 24-multimodal/compositional-generalization-bounds-contrastive
title: "Compositional Generalization Bounds for Contrastive Pretraining"
topic: 24-multimodal
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compositional Generalization Bounds for Contrastive Pretraining

> **Topic:** Multimodal Models · **ID:** `24-multimodal/compositional-generalization-bounds-contrastive` · **Status:** open

## 1. Problem Statement

A CLIP-style model is trained on image–text pairs by an InfoNCE objective. At test time it is asked about *combinations* of attributes, objects and relations that never co-occurred in training: "a red cube on top of a blue sphere" when training saw red spheres and blue cubes but never that arrangement.

The problem: **give a non-vacuous generalization bound on error over unseen combinations, as a function of the training distribution's combinatorial coverage, the encoder architecture, and the contrastive objective.**

Three variants, with different difficulty:

- **Theory variant.** Prove a bound of the form: if the pretraining support covers each factor value $k$ times and satisfies a stated connectivity condition, then error on held-out combinations is at most $\epsilon(n, k, \text{coverage})$. Existing compositional-generalization theorems assume an identifiable generative model and a decoder-based objective; none applies to InfoNCE on web text.
- **Measurement variant.** Decide empirically whether a given model generalizes compositionally at all. Blocked by the fact that "unseen combination" is undefined without knowing what LAION-2B contains.
- **Method variant.** Change the objective, the negatives, or the data mixture so that the bound improves. Currently done by benchmark-chasing rather than by optimizing a quantity with a proof attached.

Solving it means: a bound that (a) is computable from measurable properties of a real pretraining corpus, and (b) is tighter than the trivial bound of 1 at $n \approx 10^9$ pairs.

## 2. Formal Setting

**Factors.** A latent $z = (z_1,\dots,z_K) \in \mathcal{Z} = \prod_{k=1}^K \mathcal{Z}_k$, $|\mathcal{Z}_k| = m_k$. Combination space size $M = \prod_k m_k$. Images $x = g(z)$, captions $t = h(z)$ for (unknown, non-injective) renderers $g, h$.

**Measured as:** in practice $z$ is recovered by parsing captions with a dependency parser into (object, attribute, relation) tuples; $K$ and $m_k$ are then properties of the parser's vocabulary, not of nature. This is the first place the formalism detaches from measurement.

**Support and coverage.** Training distribution $p_{\text{tr}}$ on $\mathcal{Z}$, support $S = \{z : p_{\text{tr}}(z) > 0\}$. Define marginal coverage
$$c_k(v) = \sum_{z \in S,\ z_k = v} n\,p_{\text{tr}}(z), \qquad c_{\min} = \min_{k,v} c_k(v),$$
the expected count of training pairs containing factor value $v$. **Measured as:** substring/synonym counts over the pretraining captions — the procedure Udandarao et al. (2024) used over LAION-400M/2B and DataComp.

**Compositional error.** With encoders $f_I, f_T$ and score $s(x,t) = \langle f_I(x), f_T(t)\rangle / \tau$,
$$\mathcal{E}_{\text{comp}} = \mathbb{E}_{z \sim q}\big[\mathbb{1}\{\arg\max_{t' \in \mathcal{T}(z)} s(g(z), t') \neq h(z)\}\big], \quad \operatorname{supp}(q) \cap S = \emptyset,$$
where $\mathcal{T}(z)$ is a candidate set of hard negatives differing from $h(z)$ in exactly one factor. **Measured as:** accuracy on Winoground / ARO / SugarCrepe items, which fix $|\mathcal{T}| = 2$.

**Objective.** InfoNCE with batch $B$:
$$\mathcal{L} = -\frac{1}{B}\sum_{i} \log \frac{e^{s(x_i,t_i)}}{\sum_{j=1}^{B} e^{s(x_i,t_j)}}.$$

**The target bound** has the shape $\mathcal{E}_{\text{comp}} \le \hat{\mathcal{L}} + \Phi(K, m, c_{\min}, \text{arch}) + o(1)$, with $\Phi$ decaying in $c_{\min}$ rather than in $|S|/M$.

**Assumptions, and which are violated.**

| Assumption | Status |
|---|---|
| Factors independent under $p_{\text{tr}}$ | **Violated.** Web captions have heavy attribute–object correlation ("yellow banana"). |
| $g$ injective / invertible | **Violated.** Occlusion, viewpoint, crops. |
| Support connectivity: any $z$ reachable from $S$ by single-factor moves | **Unknown**, and untested on real corpora. |
| Caption $h(z)$ describes all $K$ factors | **Violated.** Alt-text is partial and often non-descriptive. |
| Negatives are i.i.d. from the marginal | **Violated by design** in hard-negative training. |

## 3. State of the Art

**Theory SOTA (established).**
- Wiedemer et al., *Compositional Generalization from First Principles* (NeurIPS 2023): sufficient conditions — **compositional support** plus a *compositional* (slot-wise) decoder — under which an autoencoder-style model provably generalizes to unseen combinations. Applies to decoders, not to InfoNCE.
- Wiedemer et al., *Provable Compositional Generalization for Object-Centric Learning* (ICLR 2024): identifiability of slot representations under invertibility + slot identifiability.
- Zimmermann et al., *Contrastive Learning Inverts the Data Generating Process* (ICML 2021): InfoNCE with hyperspherical latents recovers $z$ up to orthogonal transform — an identifiability result, **not** a generalization bound over unseen combinations.
- Daunhawer et al., *Identifiability Results for Multimodal Contrastive Learning* (ICLR 2023): multimodal contrastive learning identifies only the **shared** latent content across modalities; modality-specific factors are provably not recovered.
- Arora et al. (ICML 2019) and HaoChen et al. (NeurIPS 2021) give downstream *linear-probe* bounds for contrastive learning. These are i.i.d. bounds; neither has a term for combinatorial extrapolation.

**No published theorem bounds $\mathcal{E}_{\text{comp}}$ for InfoNCE-trained dual encoders on non-uniform, correlated support.** That is the gap.

**Empirical SOTA (benchmark numbers only, largely unablated).**
- Winoground (Thrush et al., CVPR 2022): every model tested scored below chance (25%) on group score; strongest ~10%. Later work showed a large fraction of items need commonsense or are visually ambiguous, so the number partly measures item difficulty, not compositionality.
- ARO (Yuksekgonul et al., ICLR 2023): CLIP near chance on relation/order tasks; the paper's own diagnosis is that the contrastive objective plus retrieval-style data lets a bag-of-words solution win.
- SugarCrepe (Hsieh et al., NeurIPS 2023): showed prior negative-generation pipelines were exploitable by a **blind, text-only** model; de-biased negatives cut apparent gains from hard-negative finetuning. This is the strongest established result in the empirical column, because it is a control, not a leaderboard row.

## 4. What Is Known

- **Frequency, not composition, predicts zero-shot accuracy.** Udandarao et al. (NeurIPS 2024) found a **log-linear** relation between a concept's pretraining frequency and zero-shot accuracy on it, across 34 models and 5 pretraining sets (CC-3M through LAION-2B, up to $\sim2\times10^9$ pairs). Linear improvement needs exponentially more data. Measured at the $10^7$–$10^9$ pair scale.
- **CLIP behaves like a bag of words on relations.** ARO: CLIP ViT-B/32 scores near 50% (chance) on Visual Genome relation ordering, at 400M pairs.
- **Benchmark gains from hard negatives shrink under controls.** SugarCrepe: a text-only model exploiting distributional artifacts beat some VLMs on prior compositionality suites; re-benchmarking on de-biased negatives removed much of the reported NegCLIP-style advantage.
- **Multimodal contrastive learning cannot identify modality-specific factors** (Daunhawer et al., ICLR 2023) — a hard negative result: if a factor appears in images but not captions, no amount of data fixes it.
- **Scaling does not fix it monotonically.** Winoground-style group scores stayed near or below chance from ViT-B/32 to the largest OpenCLIP models trained on LAION-2B.
- **Provable compositional generalization exists**, but only for decoder architectures on synthetic data with $K \le 5$ factors (Wiedemer et al., 2023/2024) — roughly the dSprites/Shapes3D scale, $M \sim 10^5$.

## 5. What Is Not Known

- **Theoretically open.** No bound on $\mathcal{E}_{\text{comp}}$ for InfoNCE dual encoders. Unknown whether the compositional-support condition of Wiedemer et al. is *necessary* for the contrastive case, or whether a weaker connectivity condition suffices. Unknown whether a non-vacuous bound is possible at all without architectural constraints — plausibly this is an instance of the uniform-convergence failure Nagarajan & Kolter (NeurIPS 2019) identified.
- **Empirically open.** Nobody has trained a matched model family with **deliberately held-out combinations** at $\ge 10^8$ pairs. The experiment is runnable (~$10^4$ GPU-hours); the obstacle is that it requires re-filtering a web corpus, not a new algorithm.
- **Methodologically blocked.** "Unseen combination" is not measurable on LAION/DataComp: to certify that "red cube on blue sphere" never appeared you must search 2B captions **and** 2B images, and captions are incomplete descriptions. Every published compositionality number is therefore an upper bound on the true novelty of the test item.

## 6. Why It Is Hard

**Primary obstruction: the test set's novelty is unverifiable (absent ground truth on the training support).** A compositional-generalization bound is a statement about $\operatorname{supp}(q) \cap S = \emptyset$. On web-scale corpora $S$ is measured by caption string matching, which has both false negatives (image shows the combination, caption doesn't say so) and false positives (synonyms). Udandarao et al.'s own frequency estimates rely on this pipeline. So the quantity the bound conditions on cannot currently be measured to better than a large, unquantified error.

**Secondary obstructions.**
- **Confounded measurement.** Benchmark scores mix compositional failure with hard-negative artifacts (SugarCrepe) and with item ambiguity (Winoground post-hoc analyses).
- **Non-identifiability.** Daunhawer et al. show contrastive learning discards modality-specific content by construction, so part of the observed failure is a property of the objective's *optimum*, not of finite-sample generalization. A bound on estimation error would not explain it.
- **Compute.** The clean experiment needs held-out-combination pretraining runs at $\ge 10^8$ pairs, repeated across coverage levels — a grid, not a run.

## 7. Current Research (as of 2026)

- **Identifiability-first theory.** Groups around Bethge/Brendel (Tübingen), Schölkopf (MPI-IS), and Locatello (ISTA) continue extending provable compositional generalization from decoders to encoder-only and contrastive objectives. *(frontier — verify which results are published vs. preprint.)*
- **Data-centric bounds.** Follow-ups to Udandarao et al. attempt to predict downstream accuracy from concept-frequency statistics, effectively an empirical bound with no proof. *(frontier — verify.)*
- **Controlled synthetic testbeds** (procedurally generated scenes with known $z$) as the only setting where $S$ is known exactly; the open question is whether conclusions transfer to web data.
- **Objective modifications** — hard negatives, captioning losses (CoCa-style), sigmoid losses (SigLIP) — evaluated on benchmarks, not against a bound.

## 8. Concrete Next Experiment

**Question:** does compositional error depend on *marginal* coverage $c_{\min}$ (theory's prediction) or on *joint* combination frequency (Udandarao's prediction)?

**Scale.** Build a synthetic-plus-real corpus: 100M image–text pairs from a filtered DataComp subset, restricted to a parsed vocabulary of $K=3$ factor types (object $m_1=200$, attribute $m_2=30$, spatial relation $m_3=8$), $M = 48{,}000$ combinations. Certify support by *rendering-side* filtering: use an open-vocabulary detector to verify the image content, not just the caption. Hold out 5,000 combinations exactly (both caption and detected image content). Train ViT-B/16 CLIP, 32 epochs — ~1,500 A100-hours per arm, 4 arms.

**Arms.**
1. **Held-out** — the 5,000 combinations removed.
2. **Control (matched-size)** — same number of pairs removed at random, combinations intact. This is the arm that separates compositional failure from data reduction.
3. **Coverage-sweep** — held-out, but with $c_{\min}$ boosted 4× by upsampling rare factor values, total pairs held constant.
4. **Slot-structured encoder** — arm 1 with a slot-attention image encoder, testing whether the Wiedemer compositional-decoder condition has an encoder analogue.

**Deciding number.** The accuracy gap on held-out combinations, arm 1 minus arm 2, using a 4-way single-factor-perturbation candidate set. If arm 1 tracks arm 2 within **2 points absolute**, compositional generalization is effectively free at 100M and the interesting bound is about coverage. If the gap exceeds **15 points** and arm 3 closes less than a third of it, marginal coverage is not the controlling variable and any bound in $c_{\min}$ alone is vacuous — which would rule out the natural extension of the existing theorems.

## 9. Key References

- **[Foundational]** Sanjeev Arora, Hrishikesh Khandeparkar, Mikhail Khodak, Orestis Plevrakis, Nikunj Saunshi. *A Theoretical Analysis of Contrastive Unsupervised Representation Learning.* ICML, 2019. — arXiv:1902.09229
- **[Foundational]** Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, Wieland Brendel. *Contrastive Learning Inverts the Data Generating Process.* ICML, 2021. — arXiv:2102.08850
- **[Foundational]** Jeff Z. HaoChen, Colin Wei, Adrien Gaidon, Tengyu Ma. *Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss.* NeurIPS, 2021. — arXiv:2106.04156
- **[SOTA — theory]** Thaddäus Wiedemer, Prasanna Mayilvahanan, Matthias Bethge, Wieland Brendel. *Compositional Generalization from First Principles.* NeurIPS, 2023. — arXiv:2307.05596
- **[SOTA — theory]** Thaddäus Wiedemer, Jack Brady, Alexander Panfilov, Attila Juhos, Matthias Bethge, Wieland Brendel. *Provable Compositional Generalization for Object-Centric Learning.* ICLR, 2024.
- **[SOTA — theory]** Imant Daunhawer, Alice Bizeul, Emanuele Palumbo, Alexander Marx, Julia E. Vogt. *Identifiability Results for Multimodal Contrastive Learning.* ICLR, 2023.
- **[SOTA — empirical]** Vishaal Udandarao, Ameya Prabhu, Adhiraj Ghosh, Yash Sharma, Philip H.S. Torr, Adel Bibi, Samuel Albanie, Matthias Bethge. *No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance.* NeurIPS, 2024. — arXiv:2404.04125
- **[Benchmark]** Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, Candace Ross. *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality.* CVPR, 2022. — arXiv:2204.03162
- **[Benchmark]** Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, James Zou. *When and Why Vision-Language Models Behave Like Bags-of-Words, and What to Do About It?* ICLR, 2023. — arXiv:2210.01936
- **[Benchmark / control]** Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, Ranjay Krishna. *SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality.* NeurIPS, 2023. — arXiv:2306.14610
- **[Related]** Matthew Trager, Pramuditha Perera, Luca Zancato, Alessandro Achille, Parminder Bhatia, Stefano Soatto. *Linear Spaces of Meanings: Compositional Structures in Vision-Language Models.* ICCV, 2023. — arXiv:2302.14383
- **[Related]** Vaishnavh Nagarajan, J. Zico Kolter. *Uniform Convergence May Be Unable to Explain Generalization in Deep Learning.* NeurIPS, 2019.
- **[Survey]** Nikunj Saunshi, Jordan Ash, Surbhi Goel, Dipendra Misra, Cyril Zhang, Sanjeev Arora, Sham Kakade, Akshay Krishnamurthy. *Understanding Contrastive Learning Requires Incorporating Inductive Biases.* ICML, 2022.

## 10. Worked Example

Take $K=3$: object ($m_1 = 200$), color ($m_2 = 20$), relation ($m_3 = 8$). $M = 32{,}000$ combinations.

Apply the Wiedemer-style compositional-support condition naively: every factor value must appear, and single-factor moves must connect the support. With $n = 10^8$ pairs and even coverage, $c_{\min} \approx 10^8 / 200 = 5\times10^5$ — enormously satisfied. The condition predicts generalization.

Now measure the real distribution. Caption frequency in web corpora is Zipfian with exponent near 1. The 200th-ranked object gets roughly $1/(200 \cdot H_{200}) \approx 1/(200 \times 5.9) \approx 0.085\%$ of object mentions. But the *joint* is far worse: attribute–object correlation means "purple giraffe" occurs at essentially zero frequency while its marginal factors are both common. Udandarao et al.'s log-linear law is in **joint concept frequency**, and predicts near-random accuracy for a combination with count $\approx 0$ — regardless of $c_{\min} = 5\times10^5$.

**The two predictions diverge by the full width of the accuracy range**: marginal-coverage theory says "solved at 100M pairs"; frequency scaling says "chance". Both are consistent with observed data, because nobody has held combinations out under control.

Worse, the obstruction bites before the experiment starts. To place "purple giraffe" in $\operatorname{supp}(q)$ you must certify it is absent from $S$. Caption search over 2B strings gives a count of, say, 3. Are those 3 items real? Are there 300 images of purple giraffes whose alt-text says "at the zoo"? With alt-text describing color in a minority of cases, the image-side count could be one to two orders of magnitude higher than the caption-side count, and the direction of the error is unknown per item. So the measured $c_{\text{joint}}$ that both theories are conditioned on carries an unquantified multiplicative error — which is exactly why the decisive experiment in §8 filters on **detector output over images**, not on captions.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*