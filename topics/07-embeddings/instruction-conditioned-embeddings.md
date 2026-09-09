---
id: 07-embeddings/instruction-conditioned-embeddings
title: "Instruction Conditioning versus Task-Specific Embedding Models"
topic: 07-embeddings
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Instruction Conditioning versus Task-Specific Embedding Models

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/instruction-conditioned-embeddings` · **Status:** empirically-open

## 1. Problem Statement

A text embedding model maps a string to a vector; relevance is a fixed similarity (usually cosine). **Instruction conditioning** adds a natural-language task description $I$ to the input, so the same encoder produces different vectors for the same text under different tasks. The claim under test: one instruction-conditioned encoder can match a family of separately fine-tuned, task-specific encoders, at a fraction of the storage and serving cost.

Three distinct variants, routinely conflated:

- **Measurement.** Does an instruction change the embedding *because of its semantic content*, or because it is a task-identifying token prefix? A model that responds only to the prefix's identity is a task-routed mixture, not an instruction follower. No benchmark cleanly separates these.
- **Method.** Is there a training recipe under which conditioning gains do not decay as the number of tasks grows, and under which unseen instructions transfer?
- **Theory.** Given a fixed embedding dimension $d$, how many distinct relevance orderings over a corpus can a conditioned encoder realize? Conditioning cannot buy unlimited capacity — the question is the rate.

Solved would mean: a single $d$-dimensional conditioned encoder within $\epsilon$ of per-task fine-tuned encoders on held-out tasks *and* held-out instruction phrasings, with the instruction-semantics channel isolated by ablation.

## 2. Formal Setting

Corpus $\mathcal{X}$, instruction set $\mathcal{I}$, encoder $f_\theta: \mathcal{I}\times\mathcal{X}\to\mathbb{S}^{d-1}$. Score $s_\theta(I,q,x)=\langle f_\theta(I,q), f_\theta(\emptyset,x)\rangle$ (asymmetric: instructions are typically prepended to queries only, e.g. E5-Mistral, NV-Embed).

A task $t$ is a distribution $\mathcal{D}_t$ over $(q, x^+, x^-_{1:k})$ plus a reference instruction $I_t$. Utility measured as nDCG@10 on a held-out split:

$$U(t,\theta,I) = \mathbb{E}_{\mathcal{D}_t}\big[\mathrm{nDCG@10}(s_\theta(I,\cdot,\cdot))\big].$$

**Conditioning gap** (the quantity the field implicitly claims is small):

$$\Delta_{\text{gap}}(t) = U(t, \theta^{\star}_t, \emptyset) - U(t, \theta_{\text{shared}}, I_t),$$

with $\theta^\star_t$ trained on $\mathcal{D}_t$ alone at matched parameter count and matched total tokens seen.

**Instruction sensitivity**, the measurement that separates semantics from routing. Let $\pi(I)$ be a paraphrase preserving meaning and $\sigma(I)$ a *counterfactual* instruction that inverts the relevance predicate (e.g. "find documents that contradict" for "find documents that support"):

$$S(t) = U(t,\theta,I_t) - U(t,\theta,\sigma(I_t)), \qquad R(t) = \big|U(t,\theta,I_t) - U(t,\theta,\pi(I_t))\big|.$$

Instruction following requires $S \gg 0$ **and** $R \approx 0$. Task routing gives $S\approx 0$ with $U(t,\theta,I_t)$ still high — the failure mode benchmarks miss. FollowIR's p-MRR measures a related pairwise rank change under instruction edits.

**Capacity.** With $n$ documents and $m$ instructions, the score matrix $M\in\mathbb{R}^{mn\times n}$ has rank $\le d$. The number of distinct top-$k$ orderings realizable is bounded through the sign rank of the thresholded relevance matrix — conditioning increases the *number of rows*, not the rank.

Assumptions known to be violated in practice: (i) instructions are i.i.d. samples from $\mathcal{I}$ — in reality one canonical string per dataset, so $|\mathcal{I}|\approx$ number of training datasets; (ii) $\mathcal{D}_t$ held-out splits are independent of training data — MTEB/BEIR test sets leak into web-scale synthetic training corpora; (iii) documents are encoded instruction-free, so the index is task-invariant — several systems break this and re-encode per task, which silently removes the cost argument that motivates conditioning.

## 3. State of the Art

**Established.**
- **INSTRUCTOR** (Su et al., Findings of ACL 2023): single encoder, 330 datasets (MEDI), instruction prefixes on both sides; reported gains on 70 tasks including unseen ones. Established that instruction prefixes are *trainable and non-harmful*.
- **TART** (Asai et al., Findings of ACL 2023): task-aware retrieval with instructions; introduced multi-task instruction retrieval and the BEIR-Instruct style evaluation.
- **E5-Mistral** (Wang et al., ACL 2024) and **NV-Embed** (Lee et al., ICLR 2025): LLM-initialized embedders with per-task instructions; MTEB averages near $66$–$69$ at 7B scale. These are benchmark numbers on MTEB, and instruction content is not ablated against a task-ID token in either paper.
- **FollowIR** (Weller et al., NAACL 2025 / arXiv 2024): built on TREC Robust04, Common Core 17, News 21 with narrative-length instructions. Finding: most standard retrievers, including instruction-trained ones, score **negative or near-zero p-MRR** — they do not use instruction semantics.
- **Promptriever** (Weller et al., ICLR 2025): trains on ~490k instruction-augmented MS MARCO instances; reports large p-MRR gains on FollowIR while holding standard BEIR performance roughly flat.

**Claimed but unablated.** That instruction conditioning *replaces* per-task fine-tuning. No published head-to-head holds parameters, tokens, and data constant between a conditioned shared model and per-task models. The comparison in practice is a 7B conditioned model against a 110M task-specific baseline.

**Benchmark-number-only.** MTEB/MMTEB leaderboard placements. MMTEB (Enevoldsen et al., ICLR 2025) expands to 500+ tasks but still supplies one instruction per task, so it cannot measure $R$ or $S$.

## 4. What Is Known

- Instruction prefixes help most on *symmetric/asymmetric disambiguation* (STS vs. retrieval), where the gain is largely encoding a similarity type, not a task semantics. Reported INSTRUCTOR-XL (1.5B) average improvement over prior SOTA is a few points across 70 tasks — measured at 1.5B, 330 training datasets.
- **Negation and exclusion fail.** On FollowIR's three TREC collections, retrievers below ~7B show p-MRR near zero; instruction-*trained* models before Promptriever were not reliably positive. Measured at 100M–7B.
- **Paraphrase brittleness is real.** Reordering or rewording a task instruction moves MTEB task scores by roughly 1–3 nDCG points on several dense embedders — i.e. $R$ is not $\approx 0$. Measured at 7B on MTEB subsets.
- **Rank capacity is provably finite.** Weller et al. (2025), *On the Theoretical Limitations of Embedding-Based Retrieval*, tie the number of realizable top-$k$ document sets to embedding rank and show a small synthetic corpus (LIMIT) that SOTA embedders at $d\le 4096$ fail on despite trivial relevance structure.
- **Per-task fine-tuning still wins in-domain.** GPL-style domain adaptation (Wang et al., NAACL 2022) yields multi-point nDCG@10 BEIR gains over a general encoder at 110M scale — no instruction-conditioned model has been shown to close this at matched size.

## 5. What Is Not Known

- **Empirically open.** $\Delta_{\text{gap}}$ at matched parameters and matched tokens. Runnable today; nobody has run the matched-compute grid. Likewise the scaling of $\Delta_{\text{gap}}$ with the number of training tasks $|\mathcal{T}|$ — whether conditioning gains saturate, plateau, or degrade past $10^3$ tasks.
- **Methodologically blocked.** Separating instruction *semantics* from task *routing*. Requires counterfactual instruction pairs with verified gold relabeled relevance at scale; only FollowIR and InstructIR-scale sets exist, in the low thousands of judgments, all in English news/web.
- **Theoretically open.** Whether conditioning changes the effective dimension. Is the achievable set of relevance orderings for a conditioned encoder at dimension $d$ strictly larger than for $m$ unconditioned encoders at dimension $d/m$? No proof either way; the sign-rank machinery bounds a single matrix, not a conditioned family.

## 6. Why It Is Hard

**Confounded measurement, primarily.** Instruction gains and training-data gains arrive together. Every instruction-conditioned model was also trained on a broader mixture than its baseline, so the reported delta is a sum of two effects with no published decomposition. Replacing $I_t$ with an arbitrary opaque token `[TASK_37]` is the one-line control that would decompose it, and it is essentially absent from the literature.

**Absent ground truth for counterfactual instructions.** Measuring $S$ requires knowing the correct ranking under an instruction *nobody wrote the dataset for*. Relabeling is human-expensive: FollowIR's instruction-conditioned judgments cost expert annotator time per topic, which is why it covers three collections rather than fifty.

**Non-identifiability.** A model with high $U(t,\theta,I_t)$ and $S\approx 0$ is observationally identical to a true instruction follower under every standard leaderboard metric. The evaluation does not measure the thing it names.

**Compute, secondarily.** The matched-compute grid ($|\mathcal{T}|$ task-specific runs versus one shared run, repeated across scales) is $O(|\mathcal{T}|)$ training runs, which is why it stays unrun even though each run is cheap.

## 7. Current Research (as of 2026)

- **Instruction-negative training.** Promptriever-style synthetic instructions with hard instruction-negatives (JHU/Samaya). Direction: make $S>0$ by construction. Open question is whether it generalizes past the training instruction distribution.
- **Unified generative + embedding models.** GritLM (Muennighoff et al., 2024) shows one model can do both; the untested claim is that generative instruction-following capacity transfers to the embedding head.
- **Multi-vector and late-interaction conditioning** as a capacity escape from the rank bound implied by Weller et al. (2025) *(frontier — verify)*.
- **Instruction-following IR benchmarks** — InstructIR (Oh et al., 2024), MAIR, ExcluIR — expanding counterfactual coverage; still English-heavy and small.
- **Task-vector / adapter routing** as the explicit control arm: per-task LoRA over a frozen backbone, which makes routing-vs-semantics a design choice rather than a confound *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** One backbone at 0.5B and one at 7B. $|\mathcal{T}|=16$ retrieval tasks spanning BEIR + FollowIR's three TREC collections. Fixed token budget $B=2\times10^9$ training tokens *per arm*.

**Arms.**
1. **Shared-conditioned:** one encoder, all 16 tasks, natural-language $I_t$ prepended to queries.
2. **Control arm — shared-routed:** identical model, identical data, identical schedule, but $I_t$ replaced by an opaque learned token `[TASK_k]` carrying zero semantic content.
3. **Task-specific:** 16 encoders, $B/16$ tokens each, no instruction.

**Evaluation.** Held-out splits of the 16 tasks, plus 4 fully held-out tasks with *never-seen* instructions, plus counterfactual instructions $\sigma(I_t)$ from FollowIR.

**The deciding number.** The **semantic margin**

$$\Delta_{\text{sem}} = \overline{U}(\text{arm 1}) - \overline{U}(\text{arm 2})$$

on the 4 held-out tasks, in nDCG@10. If $\Delta_{\text{sem}} \le 1.0$ point, instruction conditioning is task routing with extra steps and per-task adapters are the better engineering choice. If $\Delta_{\text{sem}} \ge 3.0$ points *and* arm 1's $S>0$ on FollowIR, natural-language conditioning carries real transfer. Cost estimate: about 20 training runs, roughly 3k A100-hours total at 0.5B plus 8k at 7B.

## 9. Key References

- **[Foundational]** Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, Tao Yu. *One Embedder, Any Task: Instruction-Finetuned Text Embeddings.* Findings of ACL, 2023. — arXiv:2212.09741
- **[Foundational]** Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, Wen-tau Yih. *Task-aware Retrieval with Instructions.* Findings of ACL, 2023. — arXiv:2211.09260
- **[SOTA]** Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle Lo, Arman Cohan, Benjamin Van Durme, Dawn Lawrie, Luca Soldaini. *FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions.* NAACL, 2025. — arXiv:2403.15246
- **[SOTA]** Orion Weller, Benjamin Van Durme, Dawn Lawrie, Ashwin Paranjape, Yuhao Zhang, Jack Hessel. *Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models.* ICLR, 2025. — arXiv:2409.11136
- **[SOTA]** Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei. *Improving Text Embeddings with Large Language Models.* ACL, 2024. — arXiv:2401.00368
- **[Theory]** Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee. *On the Theoretical Limitations of Embedding-Based Retrieval.* 2025. (LIMIT benchmark; rank/sign-rank bounds on realizable top-$k$ sets.)
- **[Survey/Benchmark]** Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023. — arXiv:2210.07316
- **[Survey/Benchmark]** Kenneth Enevoldsen et al. *MMTEB: Massive Multilingual Text Embedding Benchmark.* ICLR, 2025.
- **[Context]** Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets & Benchmarks, 2021. — arXiv:2104.08663
- **[Context]** Kexin Wang, Nandan Thakur, Nils Reimers, Iryna Gurevych. *GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval.* NAACL, 2022. — arXiv:2112.07577

## 10. Worked Example

Take TREC Robust04 topic 336, "Black Bear Attacks", with the narrative restriction: *relevant only if the document describes an attack on a human*. FollowIR supplies the base query, the narrative instruction, and relevance judgments under both the loose and the narrative-restricted predicate.

Run a 7B instruction-conditioned embedder three ways over the ~528k-document collection:

| Input | nDCG@20 (illustrative, order-of-magnitude from FollowIR-scale results) |
|---|---|
| query only | 0.42 |
| query + narrative instruction | 0.43 |
| query + **inverted** narrative ("attacks *not* involving humans") | 0.43 |

The first two rows are the number that gets reported: instructions "help", marginally. The third row is the one that matters. The inverted instruction demands a nearly disjoint relevant set, yet the ranking barely moves — $S \approx 0.00$. Computing p-MRR over the judged pairs gives a value near zero or slightly negative, which is exactly the FollowIR headline result for pre-Promptriever retrievers.

The obstruction is visible in the table. Rows 1 and 2 are consistent with *both* "the model reads the instruction" and "the instruction is an inert prefix that shifts the query embedding a little". Only row 3 distinguishes them, and row 3 requires relevance judgments under an instruction the assessors never saw — 200+ expert judgments for this single topic. Multiply by 50 topics and 16 tasks and the measurement cost, not the training cost, is what has kept $\Delta_{\text{sem}}$ unmeasured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*