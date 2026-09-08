---
id: 27-multilingual/cross-lingual-retrieval-no-parallel-data
title: "Cross-Lingual Retrieval Without Parallel Corpora"
topic: 27-multilingual
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Lingual Retrieval Without Parallel Corpora

> **Topic:** Multilingual & Low-Resource · **ID:** `27-multilingual/cross-lingual-retrieval-no-parallel-data` · **Status:** open

## 1. Problem Statement

Given a query in language $L_q$ and a document collection in a different language $L_d$, return a ranked list of documents relevant to the query — **without** any bitext, seed dictionary, or human relevance judgment linking $L_q$ and $L_d$ at training time.

Three variants, routinely conflated:

- **Method.** Build a retriever whose only cross-lingual supervision is the co-occurrence structure of two monolingual corpora. Solving it means matching a bitext-supervised retriever (LaBSE-class, mE5-class) on a held-out language pair.
- **Measurement.** Certify that a given system had no parallel signal. Web-scale corpora contain incidental translations, code-switched pages, and translated boilerplate. "No parallel corpora" is currently a claim about the training recipe, not a verified property of the data.
- **Theory.** Under what conditions is the alignment between two monolingual embedding distributions *identifiable*? Distributional isomorphism is not guaranteed and is empirically false for typologically distant, domain-mismatched pairs.

The measurement variant blocks the method variant: no negative result about unsupervised CLIR is interpretable while the contamination rate is unmeasured.

## 2. Formal Setting

Let $\mathcal{Q}\subset \Sigma_{L_q}^*$ and $\mathcal{D}\subset \Sigma_{L_d}^*$. A retriever is a pair of encoders $f_\theta:\mathcal{Q}\to\mathbb{R}^k$, $g_\theta:\mathcal{D}\to\mathbb{R}^k$ with score $s(q,d)=\langle f_\theta(q),g_\theta(d)\rangle$. Ranking is by $s$; the reported quantity is

$$\mathrm{nDCG@10}(\theta)=\frac{1}{|\mathcal{Q}_{\text{test}}|}\sum_{q}\frac{\sum_{i=1}^{10}\frac{2^{r(q,d_{(i)})}-1}{\log_2(i+1)}}{\mathrm{IDCG@10}(q)},$$

with $r$ the graded human judgment, $d_{(i)}$ the $i$-th ranked document. Recall@100 is the usual second number; it is the one that matters when the retriever feeds a reader.

**Training data.** $C_{L_q}, C_{L_d}$ monolingual corpora, $|C|$ in tokens. Define the **incidental parallelism** of the pair as

$$\pi = \frac{\\#\{(x,y)\in C_{L_q}\times C_{L_d} : \mathrm{sim}_{\text{trans}}(x,y)>\tau\}}{|C_{L_q}|/\ell},$$

i.e. mined translation pairs per sentence, at bitext-mining threshold $\tau$ (LASER/LaBSE margin score) and mean sentence length $\ell$. $\pi$ is what "unsupervised" implicitly asserts is $0$. **It is never measured.**

**Assumptions and their status.**
1. *Distributional isomorphism*: there exists near-orthogonal $W$ with $W X_{L_q}\approx X_{L_d}$ for monolingual embedding matrices $X$. **Violated** for distant pairs and mismatched domains (Søgaard et al., ACL 2018).
2. *Corpus independence*: $C_{L_q}\perp C_{L_d}$. **Violated** — web crawls share translated pages and English text leaks into every language's shard (Blevins & Zettlemoyer, EMNLP 2022).
3. *Relevance is language-invariant*: $r(q,d)$ does not depend on which language expresses $q$. **Approximately violated** — MIRACL-style judgments are collected per-language over per-language collections, so cross-lingual $r$ is usually inferred, not judged.
4. *Shared subword vocabulary carries no lexical supervision*. Violated for related scripts; anchor identical strings are a de facto dictionary.

## 3. State of the Art

**Established (ablated, reproduced):**
- Unsupervised cross-lingual word embeddings (MUSE adversarial + Procrustes refinement, Conneau et al., ICLR 2018; VecMap self-learning, Artetxe et al., ACL 2018) reach supervised-parity bilingual lexicon induction on close pairs, and VecMap converges where MUSE diverges. Both degrade catastrophically off-domain.
- Multilingual encoders trained with **no explicit cross-lingual objective** (mBERT, XLM-R) still yield nonzero zero-shot cross-lingual transfer (Pires et al., ACL 2019; K et al., ICLR 2020; Conneau et al., ACL 2020 "Emerging Cross-lingual Structure"), and shared vocabulary is *not* required (Artetxe et al., ACL 2020; Dufter & Schütze, EMNLP 2020).
- Off-the-shelf multilingual encoders are **weak retrievers**. Litschko et al. (ECIR 2021; IR Journal 2022) show unsupervised mBERT/XLM-R CLIR underperforms simple CLWE-based query translation plus BM25 on several pairs.

**Systems SOTA, bitext-supervised (the ceiling to beat):** LaBSE (Feng et al., ACL 2022, 109 languages, ~6B translation pairs), multilingual E5 (Wang et al., 2024), BGE-M3 (Chen et al., 2024). All consume large bitext or MT-generated pseudo-bitext (mMARCO, SWIM-IR).

**Benchmark-number-only results:** most "unsupervised CLIR" scores on CLIRMatrix (Sun & Duh, EMNLP 2020) and NeuCLIR come from encoders whose pretraining corpora were never audited for $\pi$. The claim "no parallel data" is a recipe description; no paper reports a contamination measurement. Treat these as unablated.

## 4. What Is Known

- **MIRACL** (Zhang et al., TACL 2023), 18 languages, ~77M passages: BM25 averages nDCG@10 ≈ 0.39 on the dev languages, mDPR ≈ 0.41, and their hybrid ≈ 0.55 — i.e. a dense multilingual retriever trained on English MS MARCO barely beats lexical matching, and the two are complementary rather than substitutable. This is the honest scale of the field: hundreds of millions of passages, sub-0.6 nDCG@10.
- **Mr. TyDi** (Zhang, Ma, Lin, MRL 2021), 11 languages: same ordering, BM25 competitive with mDPR on low-resource languages (sw, te).
- **Unsupervised BLI fails on hard pairs.** Søgaard et al. (ACL 2018) report near-zero P@1 for MUSE on English–Estonian/Finnish/Greek under domain or algorithm mismatch, against non-trivial supervised accuracy on the same pairs.
- **A tiny seed dictionary erases the gap.** Vulić et al. (EMNLP 2019): 500–1,000 translation pairs recover or beat fully unsupervised methods on most language pairs — the unsupervised regime buys almost nothing over the cheapest possible supervision.
- **Contamination is real and causal.** Blevins & Zettlemoyer (EMNLP 2022) find non-English text in nominally English pretraining corpora at rates sufficient to explain a substantial share of "zero-shot" transfer; removing it reduces transfer.
- **Evaluation artifacts inflate scores.** Glavaš et al. (ACL 2019) show CLWE rankings reverse under proper evaluation; BLI accuracy does not predict downstream CLIR.

## 5. What Is Not Known

- **Methodologically blocked (primary).** There is no accepted procedure to certify $\pi\approx 0$ for a web-scale corpus. Mining bitext requires a bitext-trained model (LASER/LaBSE), so the audit instrument is itself supervised — circularity. Consequence: no published unsupervised-CLIR result can be attributed to the method rather than the leakage.
- **Empirically open.** No one has trained a retrieval-scale encoder ($\geq$300M params, $\geq$100B tokens) on aggressively decontaminated monolingual corpora and measured the CLIR drop. The experiment is affordable ($10^4$ GPU-hours) and unrun.
- **Theoretically open.** No identifiability theorem for unsupervised alignment of *contextual* representation distributions. The static-embedding case has only negative results (isomorphism failure) and no matching positive condition. Whether a non-degenerate condition on $(C_{L_q},C_{L_d})$ implies unique recoverable alignment up to isometry is unproven either way.
- **Open at the interface.** Whether the residual gap is representational (no shared space) or *calibration* — score distributions per language differ, so a single global threshold misranks — is untested.

## 6. Why It Is Hard

**Non-identifiability compounded by unmeasurable contamination.** Two obstructions, both concrete:

1. *Non-identifiability.* Without anchors, the alignment $W$ is determined only up to symmetries of the embedding distribution. If $X_{L_q}$ is approximately rotationally symmetric in some subspace, infinitely many $W$ fit equally well and the objective cannot prefer the semantically correct one. Adversarial objectives get stuck in these; this is the mechanism behind MUSE's divergence on distant pairs.
2. *Circular audit.* Detecting parallelism needs a cross-lingual model. So the null hypothesis ("this system had no parallel signal") is untestable with current instruments. Every negative result is confounded and every positive result is unattributable.

Compute is not the obstruction here — a 300M-parameter encoder is cheap. The obstruction is that the experiment's independent variable cannot currently be set.

## 7. Current Research (as of 2026)

- **Data-provenance auditing** — Dolma/DataComp-style corpus documentation, per-shard language ID, near-duplicate and translation detection at crawl scale. The infrastructure now exists to measure $\pi$; nobody has published the number for a CLIR-relevant pair *(frontier — verify)*.
- **LLM-generated pseudo-bitext** (SWIM-IR-style synthetic multilingual training data, Google Research). Sidesteps the problem rather than solving it — the LLM is itself bitext-trained, so supervision is laundered, not removed.
- **Unsupervised dense-retrieval objectives** (Contriever-style inverse cloze / contrastive spans) extended multilingually; strong monolingually, unverified cross-lingually without leakage *(frontier — verify)*.
- **Generative retrieval and multi-vector late interaction** (ColBERT-X lineage, JHU/HLTCOE) for CLIR, mostly with translated training data.
- **Alignment theory** — optimal-transport and Gromov-Wasserstein formulations of unsupervised alignment; theory exists for point clouds, not for contextual encoders.

## 8. Concrete Next Experiment

**Question.** Is unsupervised cross-lingual retrieval a real capability, or a contamination artifact?

**Scale.** Two encoders, 300M parameters, identical architecture, tokenizer, seed, and token budget (100B tokens), each trained on English + Swahili + Telugu monolingual data.
- **Arm A (control):** standard CC-100/Dolma-style shards, unfiltered.
- **Arm B (treatment):** the same shards after (i) sentence-level language ID at $p>0.99$, dropping foreign-language sentences; (ii) removal of documents whose URL differs only by a locale path segment from a document in another shard; (iii) removal of any sentence pair with LaBSE margin score $>1.06$ (the standard mining threshold) across shards. Report the fraction removed — this is the first published estimate of $\pi$.

Neither arm sees bitext, dictionaries, or relevance labels. Evaluate zero-shot on MIRACL sw and te with English queries (cross-lingual setting), plus a bitext-supervised reference arm for the ceiling.

**Deciding number.** $\Delta = \mathrm{nDCG@10}(A) - \mathrm{nDCG@10}(B)$ on sw, averaged over 3 seeds.
- $\Delta \leq 0.02$ → cross-lingual retrieval survives decontamination; the capability is structural and the theory question becomes the live one.
- $\Delta \geq 0.08$ (roughly 20% relative at the ~0.40 operating point) → published unsupervised CLIR is substantially a leakage artifact, and every benchmark-only claim in §3 must be re-reported with a $\pi$ estimate.

Cost: ~2 × 4,000 A100-hours plus filtering. Nothing about it requires new methods.

## 9. Key References

- **[Foundational]** A. Conneau, G. Lample, M. Ranzato, L. Denoyer, H. Jégou. *Word Translation Without Parallel Data.* ICLR 2018. — arXiv:1710.04087
- **[Foundational]** M. Artetxe, G. Labaka, E. Agirre. *A Robust Self-Learning Method for Fully Unsupervised Cross-Lingual Mappings of Word Embeddings.* ACL 2018.
- **[Negative result]** A. Søgaard, S. Ruder, I. Vulić. *On the Limitations of Unsupervised Bilingual Dictionary Induction.* ACL 2018.
- **[Negative result]** I. Vulić, G. Glavaš, R. Reichart, A. Korhonen. *Do We Really Need Fully Unsupervised Cross-Lingual Embeddings?* EMNLP 2019.
- **[Evaluation]** G. Glavaš, R. Litschko, S. Ruder, I. Vulić. *How to (Properly) Evaluate Cross-Lingual Word Embeddings.* ACL 2019.
- **[Confound]** T. Blevins, L. Zettlemoyer. *Language Contamination Helps Explain the Cross-lingual Capabilities of English Pretrained Models.* EMNLP 2022.
- **[Analysis]** A. Conneau, S. Wu, H. Li, L. Zettlemoyer, V. Stoyanov. *Emerging Cross-lingual Structure in Pretrained Language Models.* ACL 2020.
- **[Analysis]** P. Dufter, H. Schütze. *Identifying Elements Essential for BERT's Multilinguality.* EMNLP 2020.
- **[CLIR SOTA/empirical]** R. Litschko, I. Vulić, S. P. Ponzetto, G. Glavaš. *On Cross-Lingual Retrieval with Multilingual Text Encoders.* Information Retrieval Journal, 2022.
- **[Benchmark]** X. Zhang et al. *MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages.* TACL 2023.
- **[Benchmark]** X. Zhang, X. Ma, J. Lin. *Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval.* MRL Workshop, EMNLP 2021.
- **[Benchmark]** S. Sun, K. Duh. *CLIRMatrix: A Massively Large Collection of Bilingual and Multilingual Datasets for Cross-Lingual Information Retrieval.* EMNLP 2020.
- **[Supervised ceiling]** F. Feng, Y. Yang, D. Cer, N. Arivazhagan, W. Wang. *Language-agnostic BERT Sentence Embedding.* ACL 2022.

## 10. Worked Example

**Setup.** English↔Swahili. Suppose the Swahili shard is $|C_{sw}| = 3\times10^{8}$ tokens — about $1.5\times10^{7}$ sentences at $\ell=20$.

**Contamination accounting.** Take a conservative leak rate of 0.1% of sentences being English (well below rates observed by Blevins & Zettlemoyer for nominally monolingual crawls):

$$1.5\times10^{7}\times 0.001 = 1.5\times10^{4}\ \text{English sentences inside the Swahili shard.}$$

Most are boilerplate. Assume only 5% sit adjacent to their Swahili rendering — cookie notices, product descriptions, mission statements on bilingual NGO and government pages, which dominate Swahili web text:

$$1.5\times10^{4}\times 0.05 = 750\ \text{de facto translation pairs.}$$

**Why that is enough.** Vulić et al. (EMNLP 2019) found 500–1,000 seed pairs sufficient to match or beat fully unsupervised alignment. The incidental leakage in a corpus everyone calls monolingual is the same order as the supervision that closes the gap. And 0.1% is the optimistic figure; at 1% the shard contains 7,500 pairs — a seed dictionary by any other name.

**The obstruction, made visible.** Run VecMap on this shard and get, say, nDCG@10 = 0.31 on MIRACL-sw with English queries. Three explanations are observationally equivalent:
1. distributional structure alone aligns the spaces;
2. the 750 incidental pairs act as anchors and self-learning bootstraps from them;
3. shared strings ("Tanzania", "COVID-19", digits, URLs) act as an identical-string seed dictionary.

To separate them you must remove (2), which requires *finding* the pairs, which requires LaBSE, which was trained on 6B translation pairs. The audit instrument embeds the thing being audited. That circularity — not compute, not model capacity — is why the problem is open, and why §8's decontamination arm, with its removal fraction reported, is the experiment that matters.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*