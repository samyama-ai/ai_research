---
id: 07-embeddings/optimal-negative-sampling-retrieval
title: "Optimal Negative Sampling Distribution for Retrieval Embeddings"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Negative Sampling Distribution for Retrieval Embeddings

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/optimal-negative-sampling-retrieval` · **Status:** open

## 1. Problem Statement

Dual-encoder retrievers are trained by contrasting a positive document against $m$ sampled negatives. The negatives are drawn from some distribution $q$ — uniform over the corpus, in-batch, BM25 top-$k$, ANN top-$k$, or a filtered mixture. Every published recipe picks $q$ by hand and validates it on one benchmark.

**The problem:** given a corpus, a query distribution, a model class, and a compute budget, characterize the negative-sampling distribution $q^\star$ that maximizes downstream retrieval quality — and say whether it can be computed or approximated online.

Three variants, different difficulty:

- **Theory variant.** Does $q^\star$ exist and have closed form for the population contrastive objective? Partly answered for *gradient-bias* surrogates (sampled softmax), open for *downstream ranking metrics*.
- **Method variant.** Build a sampler that beats hand-tuned ANN-top-$k$ mining across corpora without per-corpus tuning. Empirically open.
- **Measurement variant.** Separate the *hardness* effect of $q$ from the *false-negative* effect it induces. Methodologically blocked: labels are incomplete, so "false negative" has no ground truth.

Solving it means: a rule mapping (corpus, budget) $\to q$ that dominates ANCE-style mining on BEIR by a margin larger than seed variance, with the mechanism ablated rather than asserted.

## 2. Formal Setting

Queries $x \sim \mathcal{P}_X$, corpus $\mathcal{D}$, $|\mathcal{D}| = N$ ($N = 8.8\times10^6$ for MS MARCO passage). Encoders $f_\theta, g_\theta: \to \mathbb{S}^{d-1}$, score $s_\theta(x,d) = f_\theta(x)^\top g_\theta(d)/\tau$, $\tau$ the temperature. Relevance is a latent binary $R(x,d)\in\{0,1\}$; labels $Y(x,d)$ are observed only for annotated pairs, and $Y=0 \not\Rightarrow R=0$.

Training loss with $m$ negatives $d^-_i \sim q(\cdot\mid x)$:

$$\mathcal{L}(\theta) = -\mathbb{E}_{x, d^+}\left[\log \frac{e^{s_\theta(x,d^+)}}{e^{s_\theta(x,d^+)} + \sum_{i=1}^{m} w_i\, e^{s_\theta(x,d^-_i)}}\right],\quad w_i = \frac{p_0(d^-_i)}{q(d^-_i\mid x)}$$

with $p_0$ the reference (uniform or corpus-frequency) negative law. Setting $w_i \equiv 1$ — what nearly every implementation does — makes the objective a *different* objective, not an importance-corrected estimate of the same one.

**Measured quantities.**

- *Hardness* of $q$ at step $t$: $H_t = \mathbb{E}_{q}[s_{\theta_t}(x,d^-)]$, logged directly from the training batch.
- *False-negative rate* $\phi = \Pr_{d^-\sim q}[R(x,d^-)=1]$. Not observable; estimated by cross-encoder score threshold or by human adjudication on a sample of $\sim10^3$ pairs.
- *Gradient bias* of the sampled objective relative to full softmax over $\mathcal{D}$: $b_t = \|\nabla_\theta\hat{\mathcal{L}}_m - \nabla_\theta\mathcal{L}_{\text{full}}\|_2$. Computable exactly only at $N \lesssim 10^6$ with a full score matrix; otherwise Monte-Carlo estimated.
- *Objective*: nDCG@10 or Recall@100 on held-out judgments; BEIR average over 13–18 tasks.
- *Budget*: $C$ = total forward-backward passes plus index-refresh cost. ANCE-style mining re-encodes $\mathcal{D}$ every $T$ steps, adding $\lceil S/T\rceil \cdot N$ document encodings.

$q^\star(C) = \arg\max_q \ \mathbb{E}[\text{nDCG@10}(\theta_q)] \ \text{s.t. cost}(q) \le C$.

**Assumptions, and where they break.**

1. *Sampled negatives are true negatives* ($\phi = 0$). Violated: the harder $q$ is, the larger $\phi$. This is the central coupling.
2. *Labels are complete.* Violated: MS MARCO has $\approx 1.1$ labeled positives per query against a corpus where dozens are relevant.
3. *Evaluation distribution equals training query distribution.* Violated by construction in BEIR, which is the zero-shot transfer benchmark.
4. *In-batch negatives are i.i.d. from $p_0$.* Violated whenever batches are topic-clustered (TAS-B does this deliberately).
5. *The optimum is stationary in $\theta$.* Violated: hardness that helps at convergence destabilizes early training.

## 3. State of the Art

**Theory SOTA.** For *sampled softmax*, Blanc & Rendle (ICML 2018) prove the bias-minimizing proposal is $q(d) \propto \exp(s_\theta(x,d))$ — i.e. exactly the model's own softmax, which is what sampling was meant to avoid — and give a quadratic-kernel sampler admitting sublinear draws. Ma & Collins (EMNLP 2018) give consistency and asymptotic-variance results for ranking-based NCE. Arora et al. (ICML 2019) bound downstream error by contrastive loss under a latent-class model, with a collision term that *grows* with $m$; Ash et al. (AISTATS 2022) sharpen this to a non-monotone, U-shaped dependence on $m$ with a finite optimum. None of these targets nDCG on a real corpus, and all assume $\phi = 0$.

**Empirical SOTA.** ANCE (Xiong et al., ICLR 2021) — asynchronously refreshed ANN index, negatives from the current top-$k$. RocketQA (Qu et al., NAACL 2021) — cross-encoder-denoised hard negatives plus cross-batch negatives. TAS-B (Hofstätter et al., SIGIR 2021) — topic-clustered batches with balanced margin sampling. SimANS (Zhou et al., EMNLP 2022 industry track) — sample *ambiguous* negatives near the positive's score rather than the hardest. NV-Retriever (Moreira et al., 2024) — positive-aware thresholding: discard mined negatives scoring above $\sim95\%$ of the positive's score.

**Established vs. claimed.** Established: hard negatives beat random negatives at fixed budget, reproduced across ANCE, RocketQA, STAR/ADORE, TAS-B, E5, BGE. Claimed but unablated: that the *specific shape* of each sampler is what carries its gain. TAS-B, SimANS, and NV-Retriever each change sampler, batch construction, distillation, and data simultaneously; the reported deltas are benchmark numbers, not isolated sampler effects. No published result holds all of {teacher, data, batch size, schedule, seed count} fixed and varies only $q$ across more than two settings.

## 4. What Is Known

- **Hard negatives help, at scale.** DPR (Karpukhin et al., EMNLP 2020) with BM25 hard negatives + in-batch reaches 78.4% top-20 on NQ; random negatives alone lose several points. ANCE reports MS MARCO dev MRR@10 $\approx 0.330$ versus $\approx 0.30$ for BM25-negative baselines, at BERT-base scale, $8.8$M passages.
- **Denoising matters more than hardness.** RocketQA's ablation moves MRR@10 from $\approx 0.333$ (cross-batch negatives) to $\approx 0.364$ once a cross-encoder filters mined negatives — a larger delta than mining itself contributed. Same scale.
- **Most negatives are inert.** Cai et al. (2020) find, on ImageNet-scale instance discrimination, that removing the easiest $\sim95\%$ of negatives changes accuracy little; the hardest few percent carry the gradient.
- **More negatives is not monotonically better.** Ash et al. (AISTATS 2022) show a U-shape in $m$ both theoretically and empirically; Awasthi et al. (ICML 2022) show that under a matched latent-class assumption more negatives do *not* hurt — the two results differ in assumption, not in arithmetic.
- **Uniformity/alignment decomposition.** Wang & Isola (ICML 2020) show the InfoNCE limit as $m\to\infty$ decomposes into alignment plus hypersphere uniformity; $q$ controls which term dominates.
- **Batch size is a confound.** SimCLR (Chen et al., ICML 2020) gains from $256 \to 8192$ batch; any sampler compared across different effective $m$ is confounded with this.

## 5. What Is Not Known

- **Theoretically open.** No characterization of $q^\star$ for a *ranking* metric (nDCG@10, Recall@100) rather than a likelihood surrogate. No theory that jointly models hardness and $\phi>0$; debiased contrastive learning (Chuang et al., NeurIPS 2020) assumes a known class prior $\tau^+$, which for retrieval is neither known nor constant across queries. No proof of whether $q^\star$ is stationary or must be scheduled in $t$.
- **Empirically open.** The clean factorial — sampler $\times$ corpus $\times$ model scale, everything else fixed, $\ge 5$ seeds — has never been run. Cost, not difficulty, is the reason: each cell is a full retriever training run plus index refreshes.
- **Methodologically blocked.** $\phi$ cannot be measured. Judging a mined negative "false" requires relevance ground truth the benchmarks do not have; cross-encoder proxies are trained on the same incomplete labels, so the estimate inherits the bias it is meant to correct.

## 6. Why It Is Hard

**Non-identifiability under a single knob.** Every sampler exposes essentially one control — how far up the model's own ranking to draw from. Turning it up raises hardness $H$ (helps) and raises false-negative rate $\phi$ (hurts) *simultaneously and monotonically*. The observed downstream metric is a composition $M(H(k), \phi(k))$; with only $M$ observed and $\phi$ unmeasurable, the two partial derivatives are not separately identified. This is why the literature's optimum keeps moving — top-1000 sampling for ANCE, ambiguous-band for SimANS, 95%-threshold for NV-Retriever — without contradiction: each is a different point on the same unresolved trade-off, under a different label-noise level.

Second obstruction: **compute**. A single MS MARCO retriever run at BERT-base is on the order of $10^2$ GPU-hours with periodic re-indexing of $8.8$M passages; a $6\times4\times5$ factorial is $\sim10^4$ GPU-hours. Third: **the evaluation does not measure what it names** — BEIR nDCG@10 with shallow pools rewards agreement with the pooling systems, and mined-negative denoising partly optimizes for that agreement rather than for relevance.

## 7. Current Research (as of 2026)

- **Positive-aware and score-band mining** — NVIDIA (NV-Retriever line), Microsoft/Baidu (SimANS line): thresholding relative to positive score rather than absolute rank. Now default in several open embedding recipes.
- **Distillation replacing sampling** — TAS-B, RocketQAv2, Dragon (Lin et al., EMNLP Findings 2023): a cross-encoder supplies soft targets over mined candidates, which sidesteps the hard/false-negative binary. This is the most active line and arguably dissolves rather than solves the problem.
- **Synthetic queries and LLM-labeled negatives** — E5-Mistral, Gecko, and successors use an LLM to both generate queries and adjudicate mined negatives, giving a $\phi$ estimate independent of the human pool *(frontier — verify: no published agreement study against human adjudication at $n\ge10^3$)*.
- **Theory of negatives under label noise** — scattered follow-ups to Chuang et al. and Robinson et al. (ICLR 2021); no result yet targets ranking metrics.

## 8. Concrete Next Experiment

**Question:** is the observed sampler optimum driven by hardness or by false-negative rate?

**Scale.** MS MARCO passage, $8.8$M documents, $502{,}939$ training queries; BERT-base bi-encoder, batch 64, $m = 31$ mined negatives, $\sim100$k steps, index refresh every 10k steps. 5 seeds per arm. $\approx 8$ arms $\times$ 5 seeds $\approx 3$k GPU-hours on A100/H100 class hardware.

**Design.** Sweep mining rank band $k \in \{$top-25, 100–200, 500–1000, uniform$\}$ — this varies $H$ and $\phi$ together. Then, at each band, add an arm with an **oracle-cleaned** negative pool: adjudicate every mined candidate for a fixed 10k-query subset with three independent human judgments (not a cross-encoder), and drop the ones judged relevant. This decouples the two: cleaned arms have the band's hardness with $\phi \approx 0$.

**Control arm.** ANCE-style top-1000 mining, uncleaned, same seeds, same budget, same index-refresh schedule.

**Deciding number.** $\Delta = \text{nDCG@10}_{\text{cleaned}} - \text{nDCG@10}_{\text{uncleaned}}$ at the hardest band (top-25), on TREC DL 2019/2020, averaged over 5 seeds. If $\Delta > 0.02$ with seed s.d. $< 0.005$, the sampler optimum is a false-negative artifact and $q^\star$ is "hardest available, then clean" — a labeling problem, not a sampling problem. If $\Delta < 0.005$, false negatives are benign and the optimum is genuinely a hardness optimum, making the theory variant the live question.

## 9. Key References

- **[Foundational]** Gutmann, Hyvärinen. *Noise-contrastive estimation: A new estimation principle for unnormalized statistical models.* AISTATS, 2010.
- **[Foundational]** van den Oord, Li, Vinyals. *Representation Learning with Contrastive Predictive Coding.* 2018. — arXiv:1807.03748
- **[Theory]** Blanc, Rendle. *Adaptive Sampled Softmax with Kernel Based Sampling.* ICML, 2018. — arXiv:1712.00527
- **[Theory]** Ma, Collins. *Noise Contrastive Estimation and Negative Sampling for Conditional Models: Consistency and Statistical Efficiency.* EMNLP, 2018. — arXiv:1809.01812
- **[Theory]** Arora, Khandeparkar, Khodak, Plevrakis, Saunshi. *A Theoretical Analysis of Contrastive Unsupervised Representation Learning.* ICML, 2019. — arXiv:1902.09229
- **[Theory]** Ash, Goel, Krishnamurthy, Misra. *Investigating the Role of Negatives in Contrastive Representation Learning.* AISTATS, 2022. — arXiv:2106.09943
- **[Theory]** Awasthi, Dikkala, Kamath. *Do More Negative Samples Necessarily Hurt in Contrastive Learning?* ICML, 2022.
- **[Theory]** Wang, Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML, 2020. — arXiv:2005.10242
- **[Method]** Chuang, Robinson, Lin, Torralba, Jegelka. *Debiased Contrastive Learning.* NeurIPS, 2020. — arXiv:2007.00224
- **[Method]** Robinson, Chuang, Sra, Jegelka. *Contrastive Learning with Hard Negative Samples.* ICLR, 2021. — arXiv:2010.04592
- **[SOTA]** Karpukhin, Oğuz, Min, Lewis, Wu, Edunov, Chen, Yih. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP, 2020. — arXiv:2004.04906
- **[SOTA]** Xiong, Xiong, Li, Tang, Liu, Bennett, Ahmed, Overwijk. *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval.* ICLR, 2021. — arXiv:2007.00808
- **[SOTA]** Qu, Ding, Liu, Liu, Lv, Zhao, Zhang, She, Wang, Yu, Wu, Wang. *RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.* NAACL, 2021. — arXiv:2010.08191
- **[SOTA]** Hofstätter, Lin, Yang, Lin, Hanbury. *Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling.* SIGIR, 2021. — arXiv:2104.06967
- **[SOTA]** Zhan, Mao, Liu, Guo, Zhang, Ma. *Optimizing Dense Retrieval Model Training with Hard Negatives.* SIGIR, 2021. — arXiv:2104.08051
- **[SOTA]** Moreira, Osmulski, Xu, Ak, Schifferer, Oldridge. *NV-Retriever: Improving Text Embedding Models with Effective Hard-Negative Mining.* 2024. — arXiv:2407.15831
- **[Benchmark]** Thakur, Reimers, Rücklé, Srivastava, Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks, 2021. — arXiv:2104.08663

## 10. Worked Example

Take MS MARCO passage, a BERT-base bi-encoder at MRR@10 $= 0.31$ after a first training round. Mine negatives from its own ANN index.

Two bands, same $m = 31$, same everything else:

| Band | Mean negative score $H$ | Est. $\phi$ (cross-encoder $>$ positive $\times 0.95$) | Reported dev MRR@10 |
|---|---|---|---|
| ANN top-25 | high | $\approx 0.30$ | $\approx 0.30$ |
| ANN 100–1000 | medium | $\approx 0.08$ | $\approx 0.33$ |

The top-25 band is *harder* and performs *worse*. The standard reading is "too hard hurts optimization." But consider the gradient. For a mined negative that is in fact relevant, the InfoNCE gradient pushes $g(d^-)$ directly away from $f(x)$ with weight $\alpha_i = e^{s(x,d_i^-)}/(e^{s(x,d^+)} + \sum_j e^{s(x,d_j^-)})$. At $\tau = 0.05$ and a top-25 negative scoring within $0.02$ cosine of the positive, $e^{\Delta s} = e^{-0.02/0.05} = 0.67$ — the false negative absorbs roughly 40% as much gradient weight as the positive receives. With $\phi \approx 0.30$ across 31 negatives, about 9 documents per step are being pushed away from a query they answer.

So the top-25 arm has two candidate explanations — optimization instability from hardness, or systematic mislabeling from $\phi$ — that predict the *same* MRR@10. The $\phi$ column above is itself produced by a cross-encoder trained on MS MARCO's incomplete labels, so it cannot arbitrate: if the cross-encoder inherits the pool's blind spots, it under-counts exactly the false negatives that matter most. That is the obstruction. Only the human-adjudicated cleaned arm in §8 breaks the tie, and it costs $\sim3\times10^5$ judgments to build.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*