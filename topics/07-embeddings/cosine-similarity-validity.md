---
id: 07-embeddings/cosine-similarity-validity
title: "Cosine Similarity Validity in Learned Embedding Spaces"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cosine Similarity Validity in Learned Embedding Spaces

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/cosine-similarity-validity` · **Status:** open

## 1. Problem Statement

Cosine similarity is the default read-out of nearly every embedding system: retrieval indexes, RAG rankers, dedup pipelines, clustering, drift monitors. The problem is whether that read-out is *valid* — whether $\cos(u,v)$ is a monotone proxy for the semantic relation the system is meant to expose, or an artifact of the geometry the training objective happened to leave behind.

Three variants, with very different difficulty:

- **Measurement variant.** Given a model $f$ and a task-defined ground-truth relation $R$, quantify how much of the variance in $\cos(f(x),f(y))$ is attributable to $R$ versus to nuisance geometry (anisotropy, a handful of high-magnitude coordinates, frequency effects). Solving this means a diagnostic that predicts, *before* deployment, whether cosine will rank correctly on a new corpus.
- **Method variant.** Produce a similarity function — cosine after a learned or closed-form transform, or a different form entirely — that is provably invariant to the nuisance directions and no worse on downstream retrieval. Whitening, all-but-the-top, and contrastive uniformity terms are partial answers.
- **Theory variant.** State conditions on the training objective under which cosine in the learned space is order-isomorphic to a target relation. Steck et al. (2024) show the negative side for linear models: regularization choice alone changes the induced cosine ordering. The positive side — a nontrivial sufficient condition for a deep encoder — is open.

A solution to the measurement variant is a statistic; to the method variant, an algorithm with an ablation; to the theory variant, a theorem.

## 2. Formal Setting

Encoder $f_\theta:\mathcal{X}\to\mathbb{R}^d$, embeddings $u=f_\theta(x)$, $v=f_\theta(y)$, and

$$\cos(u,v)=\frac{\langle u,v\rangle}{\|u\|_2\|v\|_2}.$$

**Anisotropy.** Measured, not assumed: sample $N$ pairs $(x,y)$ i.i.d. from the deployment corpus and report the empirical mean

$$\bar{c} = \frac{1}{N}\sum_{i=1}^{N}\cos\big(f(x_i),f(y_i)\big).$$

An isotropic space gives $\bar c\approx 0$; $\bar c$ near 1 means cosine has almost no dynamic range left for semantics. Report $\bar c$ per layer and per corpus — it is not a property of the model alone.

**Nuisance decomposition.** Write $u=\mu+\tilde u$ with $\mu=\mathbb{E}[f(x)]$ the corpus mean. Then

$$\langle u,v\rangle=\|\mu\|^2+\langle\mu,\tilde v\rangle+\langle\tilde u,\mu\rangle+\langle\tilde u,\tilde v\rangle,$$

so when $\|\mu\|^2$ dominates $\mathrm{tr}\,\mathrm{Cov}(f)$, cosine is mostly measuring the common mean. Define the **mean-dominance ratio** $\rho=\|\mu\|_2^2/\big(\|\mu\|_2^2+\mathrm{tr}\,\mathrm{Cov}(f)\big)\in[0,1]$, estimated from the same sample.

**Per-coordinate contribution.** For coordinate $k$, $\pi_k = \mathbb{E}\big[u_k v_k / (\|u\|\|v\|)\big] / \bar c$. Rogue dimensions are those with $\pi_k$ far above $1/d$.

**Validity target.** With a ground-truth relation $R$ (human similarity ratings, relevance labels, known duplicate pairs), validity is Spearman $\rho_s$ between $\cos$ and $R$, or nDCG@10 for retrieval. Both are measured on a held-out set drawn from the *deployment* distribution, not from STS-B.

**Assumptions and their violations.**
- *Isotropy* — violated. Contextual encoders are strongly anisotropic (Ethayarajh 2019; Gao et al. 2019).
- *Coordinate exchangeability* (no single dimension dominates) — violated. Timkey & van Schijndel (2021) find a small number of outlier dimensions driving most of the inner product in GPT-2 and BERT.
- *Norm carries no signal* — violated in both directions. Norm correlates with token frequency and with information content, so normalizing discards a real signal in some tasks and removes a confound in others.
- *Basis identifiability* — violated. For any invertible $A$, the dot-product model $\langle Au, A^{-\top}v\rangle$ is unchanged while $\cos$ changes arbitrarily; cosine is only invariant to orthogonal $A$.

## 3. State of the Art

**Theory SOTA (established).** Steck, Ekanadham & Kallus, *Is Cosine-Similarity of Embeddings Really About Similarity?* (WWW '24 Companion, arXiv:2403.05440), analyze regularized linear matrix factorization. Two objectives with the same optimal reconstruction $\hat X = UV^\top$ but different regularization yield different left/right factor scalings, hence different cosine orderings — and one variant admits *arbitrary* cosine similarities. This is a proof, in a closed-form model, that cosine ordering is not determined by the fitted function.

**Empirical SOTA (established).** Post-hoc isotropization: all-but-the-top mean-and-top-PC removal (Mu & Viswanath, ICLR 2018), BERT-flow (Li et al., EMNLP 2020), whitening (Su et al. 2021; Huang et al., Findings EMNLP 2021), and standardization / rogue-dimension removal (Timkey & van Schijndel, EMNLP 2021). Contrastive training with an explicit uniformity term (Wang & Isola, ICML 2020; SimCSE, Gao et al., EMNLP 2021) largely removes the need for post-hoc fixes on STS.

**Claimed but unablated.** That modern instruction-tuned retrieval embedders (E5, GTE, Qwen3-Embedding class) have "solved" the anisotropy problem. Their MTEB leaderboard numbers are strong, but the reported evidence is benchmark score only — $\bar c$, $\rho$, and $\pi_k$ are rarely published per model, and MTEB's tasks are largely in-domain to the training mixtures. The claim that cosine validity transfers to an out-of-distribution corpus exists as a benchmark number, not an ablation.

## 4. What Is Known

- **Anisotropy is severe and layer-dependent.** Ethayarajh (EMNLP 2019) reports that in GPT-2's last layer, two randomly sampled words have expected cosine similarity near 1.0, while lower layers are far less anisotropic; ELMo and BERT show the same trend with smaller magnitude. Scale: base-size models, word-level, Wikipedia/WSJ-scale samples.
- **Degeneration is a property of the softmax objective.** Gao et al. (ICLR 2019) show maximum-likelihood language-model training pushes output embeddings of rare tokens into a narrow cone; demonstrated on WMT and language modelling at Transformer-base scale.
- **A few dimensions dominate.** Timkey & van Schijndel (EMNLP 2021) show that in GPT-2 a single dimension can account for a majority of the expected cosine similarity, and that standardizing or removing the top few dimensions substantially improves Spearman correlation with human word-similarity judgments — the largest gains on the layers where anisotropy is worst. Scale: GPT-2, BERT, RoBERTa, XLNet, base size.
- **Cosine of centered vectors is Pearson correlation.** Zhelezniak et al. (NAACL 2019) make the identity explicit and show it changes which similarity estimator is appropriate for STS.
- **Skip-gram geometry is not spherical.** Mimno & Thompson (EMNLP 2017) show SGNS embeddings occupy a narrow cone whose direction is set by the negative-sampling distribution, not by semantics.
- **Post-hoc whitening helps a lot on STS and less elsewhere.** Whitening lifts BERT-CLS Spearman on STS from near-baseline to competitive with fine-tuned sentence encoders; the same transform does not reliably improve retrieval nDCG, and this asymmetry is reproduced across the flow/whitening papers.

## 5. What Is Not Known

- **Theoretically open.** No sufficient condition on a deep encoder's objective under which $\cos$ is order-isomorphic to a stated target relation. Steck et al. give the negative result for linear factorization; the corresponding positive theorem — even for a two-layer encoder trained with InfoNCE — has no proof either way. Likewise open: whether any similarity function computable from $f$ alone can be made invariant to the full invertible-linear ambiguity without extra supervision.
- **Empirically open.** Whether the diagnostics $(\bar c,\rho,\pi_k)$ *predict* downstream retrieval degradation on out-of-distribution corpora. The experiment is runnable today on public models and BEIR; nobody has published the correlation at frontier embedder scale (7B-parameter embedders, 10+ domains).
- **Methodologically blocked.** "Semantic similarity" has no distribution-free ground truth. STS human ratings are a specific annotation protocol, not the relation a retrieval system needs; a model can be *more* valid for retrieval and score *worse* on STS. Until validity is defined relative to a named decision, the measurement is underdetermined.

## 6. Why It Is Hard

Two named obstructions.

**Non-identifiability.** The training loss constrains $f$ only up to a transformation group larger than the one cosine is invariant to. Dot-product and softmax losses are invariant to $u\mapsto Au$, $v\mapsto A^{-\top}v$ for any invertible $A$; cosine is invariant only to orthogonal $A$. So the loss cannot pin down the cosine ordering, and no amount of additional training data fixes it — this is the mechanism behind Steck et al.'s arbitrary-similarity construction, not an artifact of their linear setting.

**Evaluation that does not measure what it names.** STS benchmarks are the standard validity evidence, but STS pairs are short, single-sentence, and topically balanced; deployment corpora are long, domain-skewed, and near-duplicate-heavy. A whitening transform that adds Spearman on STS can lose nDCG on BEIR because centering destroys the topical-magnitude signal retrieval uses. The benchmark's name says "semantic similarity"; what it measures is agreement with one annotation protocol on one sentence distribution.

## 7. Current Research (as of 2026)

- **Objective-level isotropy** — contrastive and uniformity-regularized training, now standard in E5/GTE/Qwen3-Embedding-style recipes (Microsoft Research, Alibaba DAMO, Qwen team). Treats geometry as a training constraint rather than a post-hoc fix.
- **Geometry diagnostics** — IsoScore (Rudman, Gurrola & Eickhoff, Findings NAACL 2022) and successors, aiming at an anisotropy measure that is not confounded by dimensionality. *(frontier — verify whether an IsoScore-to-retrieval correlation has been published.)*
- **Matryoshka and dimension-truncatable embeddings** (Kusupati et al., NeurIPS 2022) — raise a fresh version of the question: is cosine at 64 dims measuring the same relation as cosine at 1024?
- **Interpretability-side work on outlier features** in transformer activations, connecting rogue dimensions to attention-sink and massive-activation phenomena. *(frontier — verify the link to embedding-space cosine specifically.)*

## 8. Concrete Next Experiment

**Question.** Do cheap geometry diagnostics predict where cosine retrieval fails out of domain?

**Scale.** 12 public embedders spanning 33M–7B parameters (MiniLM, BGE-base, E5-large, GTE-large, and 7B-class instruction embedders) × all 15 BEIR corpora. Cost: one forward pass over each corpus, roughly 500 GPU-hours on A100s — feasible for one lab-week.

**Per (model, corpus) measure.** $\bar c$ over $N=10^5$ random document pairs; $\rho$ (mean-dominance); $\pi_{(1)}$, the single largest per-coordinate contribution; and nDCG@10.

**Control arm.** The same models with a whitening transform fitted on that corpus's documents (no labels used), scored on the same queries. Whitening sets $\bar c\to 0$ and $\rho\to 0$ by construction, so it isolates whether the diagnostics' predictive power is about geometry or about the model.

**Deciding number.** The partial Spearman correlation between $\pi_{(1)}$ and the whitening-induced nDCG@10 gain, controlling for model size. If $|\rho_s| \ge 0.5$ across the 180 cells, rogue-dimension mass is a deployable pre-flight test for cosine validity. If $|\rho_s| < 0.2$, the diagnostics are decorative and validity must be measured with labels, which pushes the problem back to the methodologically blocked branch.

## 9. Key References

- **[SOTA]** Harald Steck, Chaitanya Ekanadham, Nathan Kallus. *Is Cosine-Similarity of Embeddings Really About Similarity?* WWW '24 Companion, 2024. — arXiv:2403.05440
- **[Foundational]** Kawin Ethayarajh. *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.* EMNLP 2019.
- **[Foundational]** Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-Yan Liu. *Representation Degeneration Problem in Training Natural Language Generation Models.* ICLR 2019.
- **[Foundational]** David Mimno, Laure Thompson. *The Strange Geometry of Skip-Gram with Negative Sampling.* EMNLP 2017.
- **[SOTA]** William Timkey, Marten van Schijndel. *All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality.* EMNLP 2021.
- **[Foundational]** Jiaqi Mu, Pramod Viswanath. *All-but-the-Top: Simple and Effective Postprocessing for Word Representations.* ICLR 2018.
- **[SOTA]** Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, Lei Li. *On the Sentence Embeddings from Pre-trained Language Models.* EMNLP 2020.
- **[SOTA]** Tianyu Gao, Xingcheng Yao, Danqi Chen. *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* EMNLP 2021.
- **[Foundational]** Tongzhou Wang, Phillip Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* ICML 2020.
- **[Foundational]** Vitalii Zhelezniak, Aleksandar Savkov, April Shen, Nils Hammerla. *Correlation Coefficients and Semantic Textual Similarity.* NAACL 2019.
- **[Survey]** Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers. *MTEB: Massive Text Embedding Benchmark.* EACL 2023.
- **[Survey]** William Rudman, Nate Gurrola, Carsten Eickhoff. *IsoScore: Measuring the Uniformity of Embedding Space Utilization.* Findings of ACL 2022.

## 10. Worked Example

Take a $d=768$ encoder whose embeddings on a legal-document corpus have corpus mean $\mu$ with $\|\mu\|_2 = 9.0$ and covariance trace $\mathrm{tr}\,\mathrm{Cov}(f) = 4.0$ (so per-document deviation norm $\approx 2.0$). Then

$$\rho = \frac{81}{81+4} = 0.953,\qquad \bar c \approx \frac{\|\mu\|^2}{\|\mu\|^2+\mathrm{tr}\,\mathrm{Cov}} = 0.95.$$

Two documents on genuinely different topics land at $\cos = 0.94$; a true near-duplicate pair lands at $\cos = 0.97$. The whole semantic signal lives in a 0.03-wide band at the top of the range.

Now add one rogue coordinate $k$ with $\mu_k = 7.0$ (most of $\|\mu\|$) and per-document standard deviation $0.8$ driven by document length, not topic. Its contribution to the numerator is $\mathbb{E}[u_kv_k]=49\pm$ noise out of $\approx 85$ total, so $\pi_k \approx 0.58$ — one of 768 dimensions supplies 58% of the similarity, and it encodes length.

Rank consequence: a long, off-topic document with $u_k=8.6$ beats a short on-topic one with $u_k=5.4$, because the $u_kv_k$ term swings by $\approx 3.2\cdot 7.0 = 22$ against a topical signal whose whole dynamic range is $\approx 2.5$. Centering and whitening removes it and the ordering flips.

The obstruction is visible here: *both* orderings are consistent with the training loss. The pre-whitening space and the post-whitening space differ by an invertible linear map, under which a dot-product objective with a matched output head is invariant. Nothing in the model's fit says which ordering is the right one — only an external, task-specific label set does, and that is exactly what the deployment corpus lacks.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*