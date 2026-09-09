---
id: 24-multimodal/multimodal-retrieval-distribution-shift
title: "Robustness of Multimodal Retrieval to Distribution Shift"
topic: 24-multimodal
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Robustness of Multimodal Retrieval to Distribution Shift

> **Topic:** Multimodal Models · **ID:** `24-multimodal/multimodal-retrieval-distribution-shift` · **Status:** open

## 1. Problem Statement

A multimodal retriever embeds queries and candidates from different modalities (text, image, video, audio) into a shared space and ranks candidates by similarity. The question: **how much retrieval quality is lost when the deployment distribution differs from the training distribution, and can that loss be predicted or bounded from quantities measurable before deployment?**

Three variants, with very different difficulty:

- **Measurement.** Define a shift-robustness quantity for *ranking* that is not confounded by in-distribution (ID) accuracy. Classification has *effective robustness* (Taori et al., NeurIPS 2020). Retrieval has no accepted analogue, because the candidate pool — not just the query distribution — moves under shift. Currently **methodologically blocked**.
- **Method.** Build a retriever whose out-of-distribution (OOD) Recall@$k$ drop is smaller than a same-ID-score baseline's. Runnable now; the blocker is that most reported gains vanish once ID score is controlled.
- **Theory.** Bound the drop in ranking metrics as a function of a divergence between train and deployment joint distributions over (query, candidate) pairs. Open: known bounds are on classification risk, not on top-$k$ ranking with a shifting corpus.

Solving it means: given a candidate model and an unlabeled sample of the deployment corpus, predict OOD Recall@1 within a stated interval, and show the predictor holds across shift families it was not fit on.

## 2. Formal Setting

Let $\mathcal{Q}$ be queries, $\mathcal{C}$ candidates. A retriever is a pair of encoders $f_\theta:\mathcal{Q}\to\mathbb{S}^{d-1}$, $g_\theta:\mathcal{C}\to\mathbb{S}^{d-1}$ with score $s_\theta(q,c)=\langle f_\theta(q),g_\theta(c)\rangle$ (unit-normalized, so $s\in[-1,1]$ — this is what CLIP-style models actually compute).

A **retrieval environment** is a triple $E=(P_Q, \Pi, R)$: a query marginal $P_Q$, a corpus $\Pi$ (an empirical set of $N=|\Pi|$ candidates), and a relevance relation $R\subseteq \mathcal{Q}\times\Pi$. Measured quantity:

$$\mathrm{R@}k(\theta; E) \;=\; \mathbb{E}_{q\sim P_Q}\Big[\mathbb{1}\big\{\exists c\in R(q):\ \mathrm{rank}_\theta(c\mid q,\Pi)\le k\big\}\Big],\qquad \mathrm{rank}_\theta(c\mid q,\Pi)=1+\sum_{c'\in\Pi}\mathbb{1}\{s_\theta(q,c')>s_\theta(q,c)\}.$$

In practice $R$ is the caption–image pairing of the eval set, and $N$ is the eval-set size (COCO 5k: $N=5000$; Flickr30k: $N=1000$). **This is a measurement artifact, not a property of the model**: $\mathrm{R@}k$ falls roughly logarithmically in $N$, so any comparison across environments must hold $N$ fixed or report a pool-size-controlled score.

Shift is a pair $E_{\mathrm{id}}\to E_{\mathrm{ood}}$. Decompose it:

$$\underbrace{P_Q^{\mathrm{id}}\neq P_Q^{\mathrm{ood}}}_{\text{query shift}},\qquad \underbrace{\Pi_{\mathrm{id}}\neq\Pi_{\mathrm{ood}}}_{\text{corpus shift}},\qquad \underbrace{R_{\mathrm{id}}\neq R_{\mathrm{ood}}}_{\text{relevance shift}}.$$

Relevance shift is the one classification has no counterpart for: the same caption can be relevant to a different image when the corpus changes.

**Effective robustness**, transposed. Fit $\beta$ on a reference model family via probit-linear regression $\Phi^{-1}(\mathrm{R@}k_{\mathrm{ood}}) = \alpha + \beta\,\Phi^{-1}(\mathrm{R@}k_{\mathrm{id}})$, then

$$\rho(\theta) \;=\; \Phi^{-1}\big(\mathrm{R@}k_{\mathrm{ood}}(\theta)\big) \;-\; \big(\hat\alpha + \hat\beta\,\Phi^{-1}(\mathrm{R@}k_{\mathrm{id}}(\theta))\big).$$

$\rho>0$ means the model beats the accuracy-on-the-line trend (Miller et al., ICML 2021), i.e. gains not explained by ID score.

**Assumptions, and which are violated.**
1. *$R$ is complete.* Violated: COCO/Flickr have one "correct" image per caption; correct-but-unlabeled candidates are scored as errors, and the false-negative rate itself changes under shift.
2. *Query and corpus shift independently.* Violated: real corpora are collected with their queries (a product catalog shifts jointly with its search log).
3. *Probit-linearity of the ID–OOD relation.* Established for ImageNet classification; **assumed, not established** for retrieval, and $\beta$ is not scale-invariant when $N$ differs between environments.
4. *Train/test independence.* Violated at web scale: LAION-style corpora contain near-duplicates of eval images (Mayilvahanan et al., ICLR 2024).

## 3. State of the Art

**Empirical SOTA (established).**
- CLIP-family contrastive pretraining (Radford et al., ICML 2021) gives large zero-shot retrieval and unusually high classification effective robustness versus ImageNet-supervised models.
- **The cause is the pretraining data distribution, not the loss or the language supervision.** Fang et al. (*Data Determines Distributional Robustness in Contrastive Language Image Pre-training*, ICML 2022) trained CLIP and supervised models on matched data and showed the robustness gap tracks the data source. Independently supported by Nguyen et al. (*Quality Not Quantity*, NeurIPS 2022), which found no single source dominates: models trained on YFCC-15M and CC-12M each win on different shifts, and mixing sources can be worse than the better source alone.
- Retrieval-specific shift benchmarks exist: Qiu et al., *Benchmarking Robustness of Multimodal Image-Text Models under Distribution Shift* (TMLR/JDMLR, 2023) apply 17 image and 16 text perturbations and report large drops, with **text perturbation hurting more than image perturbation** for image–text retrieval.
- Text-only analogue is well established and should be treated as the prior: BEIR (Thakur et al., NeurIPS Datasets & Benchmarks 2021) showed dense retrievers beat BM25 in-domain but lose to it on a majority of 18 zero-shot datasets.

**Claimed but unablated.**
- That universal multimodal retrievers (UniIR, Wei et al., ECCV 2024; MMEB/VLM2Vec, Jiang et al., ICLR 2025) generalize to *held-out* tasks. Held-out numbers are reported, but not with ID score matched to a baseline, so it is unclear whether transfer exceeds the accuracy-on-the-line trend.
- That hard-negative mining improves shift robustness. Reported as benchmark deltas; the ID-controlled ablation is largely missing.

**Theory SOTA.** Generic distributionally-robust bounds ($f$-divergence / Wasserstein DRO) apply to the contrastive loss, not to top-$k$ recall with a mutable corpus. No published bound converts a divergence between $E_{\mathrm{id}}$ and $E_{\mathrm{ood}}$ into a $\mathrm{R@}k$ guarantee.

## 4. What Is Known

- **Compositional/relational brittleness is real but was overstated.** ARO (Yuksekgonul et al., ICLR 2023) reported CLIP near chance on word-order and relation probes. SugarCrepe (Hsieh et al., NeurIPS 2023) showed a large part of that gap was a dataset artifact — negatives were detectable by a text-only language model — and CLIP scores rise substantially on de-biased negatives. Both effects are reproduced; the residual gap is smaller than ARO's headline.
- **Perturbation sensitivity is asymmetric.** Qiu et al. (2023), measured on COCO/Flickr30k scale with ViT-B and ViT-L backbones: text-side corruptions degrade retrieval more than comparable image-side corruptions.
- **Duplication inflates apparent robustness.** Mayilvahanan et al. (ICLR 2024) removed LAION train samples highly similar to test data; CLIP's OOD performance held up better than the "memorization" story predicts, but the measurement showed train–test similarity is high enough that it must be controlled explicitly.
- **Data curation dominates at fixed compute.** DataComp (Gadre et al., NeurIPS 2023) fixed architecture and compute and varied only the filtering rule, producing double-digit swings in zero-shot ImageNet and retrieval scores — the strongest evidence that the training distribution, not the objective, sets the shift behaviour.
- **Scale is not a fix by itself.** Across CLIP model sizes, ID and OOD retrieval move together; no public result shows scaling raises $\rho$ (effective robustness) for retrieval.

## 5. What Is Not Known

- **Methodologically blocked.** No agreed effective-robustness metric for ranking. Pool size $N$, incomplete relevance $R$, and corpus shift all move $\mathrm{R@}k$ independently of the encoder, and no published protocol disentangles them. This is the binding constraint: most "robustness" numbers in the literature are not comparable across papers.
- **Empirically open.** Whether *any* intervention (hard negatives, sigmoid loss, caption rewriting, multi-source mixing) yields $\rho>0$ for retrieval at fixed ID score and fixed pool size. Runnable at ~$10^8$ pairs on a few hundred GPU-hours; nobody has published the ID-matched grid.
- **Theoretically open.** A bound of the form $\mathrm{R@}k(\theta;E_{\mathrm{ood}}) \ge \mathrm{R@}k(\theta;E_{\mathrm{id}}) - \psi(D(E_{\mathrm{id}},E_{\mathrm{ood}}), N, \text{margin})$ for a computable divergence $D$. Nothing of this shape exists for top-$k$ ranking.
- **Empirically open.** Whether corpus shift or query shift dominates in deployment. Every public benchmark perturbs both together.

## 6. Why It Is Hard

**The primary obstruction is confounded measurement, with absent ground truth underneath it.**

$\mathrm{R@}k$ is a function of three things a robustness claim wants to hold fixed: the encoder, the corpus size, and the completeness of $R$. Under a natural shift all three move. A model can gain 4 points of OOD R@1 purely because the OOD corpus is smaller or its distractors are less confusable — no encoder property changed. Because relevance labels are one-per-query and unlabeled-but-correct candidates are counted as errors, the *false-negative rate is itself distribution-dependent*: on a corpus with many near-duplicate images, a better model is penalized more. The measured drop is therefore not identifiable as model brittleness versus label incompleteness without re-annotation of the OOD corpus, which nobody does at scale.

Second-order: the ID/OOD split is not clean. Web-scale pretraining corpora overlap every public eval set, so "distribution shift" is measured against data the model has partly seen.

## 7. Current Research (as of 2026)

- **Curation-as-robustness.** DataComp/DataComp-LM lineage (UW, LAION, TRI, Apple) — treating filtering rules as the intervention under fixed compute. Established framing.
- **Universal / instruction-following retrieval.** UniIR (Waterloo), MMEB–VLM2Vec (Waterloo/Salesforce) — LLM-backboned embedders evaluated on held-out multimodal tasks. Robustness claims are benchmark-number-only *(frontier — verify)*.
- **Loss geometry.** SigLIP (Zhai et al., ICCV 2023) replaces softmax contrastive with pairwise sigmoid; better small-batch behaviour is established, effect on $\rho$ is not.
- **Synthetic recaptioning** (e.g. LaCLIP-style / VeCap-style caption rewriting) to reduce text-side brittleness *(frontier — verify whether gains survive ID matching)*.
- **Multimodal RAG retrieval quality under document shift** — active in industry, almost nothing published with controlled pools *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does any training intervention produce $\rho>0$ for multimodal retrieval, once ID score and pool size are controlled?

**Scale.** Train 12 CLIP ViT-B/16 models on DataComp-medium (128M sample scale, ~$3\times10^8$ seen samples each; roughly 300–500 A100-hours total on 8 GPUs per model at this scale). Grid: 4 data mixtures (DataComp-filtered, YFCC-15M, CC-12M, 1:1:1 mix) × 3 objectives (softmax InfoNCE, SigLIP, InfoNCE + hard negatives). Additionally checkpoint each run at 5 points to sweep ID score continuously.

**Control arm.** The 20 public OpenCLIP checkpoints spanning ID R@1 from ~25% to ~60% define the baseline probit-linear trend $(\hat\alpha,\hat\beta)$. Every evaluation uses a **fixed pool size $N=5000$**, sampled by bootstrap from each OOD corpus, averaged over 20 resamples — this removes the $N$ confound. OOD environments: NoCaps (Agrawal et al., ICCV 2019) out-domain split, Flickr30k, and a WIT/Wikipedia-image slice. ID environment: COCO 5k.

**Deciding number.** $\rho$ in probit units for text→image R@1, with a bootstrap 95% CI. **Decision rule:** an intervention counts as improving shift robustness iff its CI lower bound exceeds $0$ by more than $0.05$ probit (≈ +2 R@1 points at the 40% operating point) on **at least two** OOD environments. If no cell clears it, the empirically-open variant resolves negative: retrieval robustness is on the line, and the field should redirect to the corpus-shift and relevance-completeness measurement problem.

**Cost of the label control.** Re-annotate 500 COCO and 500 NoCaps queries with all-relevant-candidates (not one), to estimate the shift-dependent false-negative rate. Without this the $\rho$ estimate is biased by an unknown amount.

## 9. Key References

- **[Foundational]** Alec Radford, Jong Wook Kim, Chris Hallacy, et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021. — arXiv:2103.00020
- **[Foundational]** Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, Ludwig Schmidt. *Measuring Robustness to Natural Distribution Shifts in Image Classification.* NeurIPS, 2020. — arXiv:2007.00644
- **[Foundational]** John Miller, Rohan Taori, Aditi Raghunathan, et al. *Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization.* ICML, 2021. — arXiv:2107.04649
- **[SOTA]** Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, Ludwig Schmidt. *Data Determines Distributional Robustness in Contrastive Language Image Pre-training (CLIP).* ICML, 2022. — arXiv:2205.01397
- **[SOTA]** Thao Nguyen, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, Ludwig Schmidt. *Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP.* NeurIPS, 2022. — arXiv:2208.05516
- **[SOTA]** Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, et al. *DataComp: In search of the next generation of multimodal datasets.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2304.14108
- **[SOTA]** Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. *Sigmoid Loss for Language Image Pre-Training.* ICCV, 2023. — arXiv:2303.15343
- **[Benchmark]** Jielin Qiu, Yi Zhu, Xingjian Shi, et al. *Benchmarking Robustness of Multimodal Image-Text Models under Distribution Shift.* Journal of Data-centric Machine Learning Research, 2023.
- **[Benchmark]** Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2104.08663
- **[Benchmark]** Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, James Zou. *When and Why Vision-Language Models Behave Like Bags-of-Words, and What to Do About It?* ICLR, 2023.
- **[Correction]** Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, Ranjay Krishna. *SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality.* NeurIPS Datasets & Benchmarks, 2023.
- **[Analysis]** Prasanna Mayilvahanan, Thaddäus Wiedemer, Evgenia Rusak, Matthias Bethge, Wieland Brendel. *Does CLIP's Generalization Performance Mainly Stem from High Train-Test Similarity?* ICLR, 2024.
- **[Survey/Systems]** Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, Wenhu Chen. *UniIR: Training and Benchmarking Universal Multimodal Information Retrievers.* ECCV, 2024.

## 10. Worked Example

Take one OpenCLIP ViT-B/32 checkpoint. Measure text→image R@1 on COCO 5k ($N=5000$) and on Flickr30k ($N=1000$). Typical published values for this class of model: COCO ≈ 37%, Flickr30k ≈ 62%.

Naive reading: Flickr30k is +25 points, so the model is *more* robust on the shifted set. That is wrong — the pools differ by 5×.

Correct the pool. Empirically, R@1 for a fixed encoder falls close to log-linearly in $N$; across CLIP checkpoints the slope is roughly $-8$ to $-10$ R@1 points per doubling near the 40% operating point. Subsampling Flickr30k to $N=5000$ is impossible (it has 1000 images), so instead subsample COCO to $N=1000$, 20 bootstrap draws. That lifts COCO R@1 from 37% into the mid-50s. The apparent +25 point "robustness gain" collapses to roughly +5.

Now the residual. Flickr30k is Flickr-sourced; LAION-2B is web-scraped and contains Flickr images. A near-duplicate scan (CLIP-embedding cosine $>0.95$ against the training shard) removes some fraction of Flickr30k test images as effectively seen. Whatever that fraction is, the remaining +5 shrinks further, and the direction of the bias is known but its size is not published for this pair.

Finally the labels. Flickr30k has 5 captions per image and one image per caption. Manually checking 100 COCO queries typically turns up several images that a human would accept as matching the caption but that the metric scores as wrong.

**What the example shows:** a 25-point difference was mostly pool size, partly train–test overlap, and partly label incompleteness. The share attributable to the encoder's brittleness under shift — the quantity the problem is about — is not separable from the other three with any protocol currently in use. That is the obstruction, and it is why Section 8 spends its budget on pool-size control and re-annotation rather than on a new training method.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*