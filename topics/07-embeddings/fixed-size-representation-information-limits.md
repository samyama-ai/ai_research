---
id: 07-embeddings/fixed-size-representation-information-limits
title: "Information-Theoretic Limits of Fixed-Size Text Representations"
topic: 07-embeddings
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Information-Theoretic Limits of Fixed-Size Text Representations

> **Topic:** Embeddings & Representations · **ID:** `07-embeddings/fixed-size-representation-information-limits` · **Status:** open

## 1. Problem Statement

A text embedder maps a variable-length token sequence to a fixed vector $e: \mathcal{V}^* \to \mathbb{R}^d$. The question: **how much task-relevant information can a $d$-dimensional vector carry about a document of length $L$, and where is the true ceiling versus a training artifact?**

Three variants that are routinely conflated:

- **Measurement.** Given a deployed encoder, estimate the bits it actually retains. Operationalized by inversion (can text be reconstructed?) or by task sufficiency (does downstream accuracy saturate?). Currently the weakest leg.
- **Method.** Find the encoder/decoder pair that maximizes retained task information at fixed $d$. This is a rate–distortion optimization with a learned, non-convex encoder.
- **Theory.** Prove a lower bound on $d$ required for a stated retrieval or reconstruction guarantee over a corpus of size $n$. This is the only variant where sharp results exist, and they are combinatorial (rank, sign-rank, JL-type), not statistical.

**Solved** would mean: a bound $d \ge f(n, L, \varepsilon, \text{task})$ that is tight to a constant, plus a constructive encoder that meets it on natural language rather than on adversarial worst-case corpora.

## 2. Formal Setting

Let $X \sim P_X$ be a document over vocabulary $\mathcal{V}$, $|X| = L$ tokens. Let $Z = e(X) \in \mathbb{R}^d$, quantized to $b$ bits per coordinate, so the raw channel width is $R = db$ bits.

**Source entropy, as measured.** $H(X)$ is estimated by a language model's cross-entropy on held-out text: $\hat{H} = -\frac{1}{|C|}\sum \log_2 p_\theta(c_i \mid c_{<i})$ in bits per character. This is an *upper* bound on $H(X)$, tight only to the extent $p_\theta$ is good.

**Usable capacity under a similarity metric.** Retrieval reads $Z$ only through $\langle z_q, z_d\rangle$ with resolution $\varepsilon$ (score gaps below $\varepsilon$ are indistinguishable after noise/quantization). The number of $\varepsilon$-distinguishable directions on $S^{d-1}$ gives
$$C_{\text{usable}} \;\approx\; d \log_2 (1/\varepsilon) \;\ll\; R .$$

**Task sufficiency.** For task $T$ with label $Y$, define retained information $I(Z;Y)$ and the deficiency $\Delta(d) = I(X;Y) - I(Z;Y) \ge 0$ by data processing. Measured in practice as an accuracy gap against a full-text cross-encoder, not as a mutual information — $I(Z;Y)$ is not directly estimable at these dimensions.

**Retrieval feasibility.** Given a binary relevance matrix $A \in \{0,1\}^{m \times n}$ ($m$ queries, $n$ documents), $A$ is *realizable* at dimension $d$ if there exist $q_i, x_j \in \mathbb{R}^d$ with $\langle q_i, x_j\rangle > \tau_i$ iff $A_{ij}=1$. The minimal such $d$ is the row-wise-thresholdable rank, sandwiched by sign-rank.

**Assumptions, and which are violated.**
- *Stationary, in-distribution corpus.* Violated: retrieval indices drift; embedders are evaluated on distributions unlike their contrastive training mix.
- *Isotropic use of the $d$ dimensions.* Violated. Contextual embeddings are strongly anisotropic — Ethayarajh (EMNLP 2019) measured average cosine similarity of random words near $0.99$ in upper GPT-2 layers, so effective dimension is far below $d$.
- *Relevance is a fixed binary matrix.* Violated: relevance is query-intent-dependent and annotator-noisy, so the sign-rank bound is stated over an idealized object.
- *Distortion is symmetric.* Violated: false negatives at rank 1 cost far more than score noise deep in the list.

## 3. State of the Art

**Theory SOTA (established).**
- Johnson–Lindenstrauss with optimality: $k = \Theta(\varepsilon^{-2}\log n)$ is necessary and sufficient for pairwise distance preservation — Larsen & Nelson (FOCS 2017), matching the 1984 upper bound. This bounds *geometry preservation*, not task sufficiency.
- Alon & Klartag (FOCS 2017) give optimal bit complexity for approximate inner products, tying dimension to bits rather than coordinates.
- Weller et al., *On the Theoretical Limitations of Embedding-Based Retrieval* (2025, arXiv:2508.21038), import sign-rank/communication-complexity lower bounds: for any $d$ there is a top-$k$ relevance pattern no single-vector embedder can represent. Established as a theorem; the *extrapolated* critical corpus size for $d=512$ comes from a polynomial fit to small-$d$ free-embedding optimization, and is a fit, not a bound.

**Empirical SOTA.**
- Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022) — nested prefixes trained jointly; reported up to $14\times$ smaller embeddings at equal ImageNet-1K accuracy. Independently reproduced and now standard in production embedders.
- OpenAI `text-embedding-3-large` truncated to 256 dims reportedly outscores the 1536-dim `ada-002` on MTEB. **Vendor-reported benchmark number, not an ablation** — the two models differ in data and training, so it does not isolate dimension.
- ColBERT (Khattab & Zaharia, SIGIR 2020) and multi-vector successors sidestep the single-vector bound entirely; the LIMIT results in Weller et al. show multi-vector and cross-encoder arms solving instances where single-vector arms collapse.
- Binary and int8 embedding quantization (Cohere, HuggingFace, 2024): ~$32\times$ storage reduction at claimed >95% retained nDCG. Claimed on MTEB; corpus sizes are far below the regime where the dimension bound bites, so the claim is untested where it matters.

## 4. What Is Known

- **Inversion recovers a lot.** Morris et al., *Text Embeddings Reveal (Almost) As Much As Text* (EMNLP 2023, arXiv:2310.06816): from 768-dim GTR-base embeddings, ~92% of 32-token inputs are reconstructed **exactly**; recovery degrades sharply with length. Scale: single encoder, 32–128 token inputs.
- **Text entropy is ~0.6–1.3 bits/char** (Shannon 1951); modern neural compressors reach roughly 0.9 bits/char on enwik-class corpora, and Delétang et al. (ICLR 2024, arXiv:2309.10668) show LLMs as competitive general-purpose compressors.
- **Low dimensions fail as indices grow.** Reimers & Gurevych, *The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes* (ACL 2021): dense retrieval quality falls with index size, and the fall is steeper at low $d$ — the failure is a function of $n$, not just of $d$.
- **Single-vector retrieval fails on tiny adversarial corpora.** On LIMIT (Weller et al. 2025), strong embedders score below ~20% recall@100 on a 46-document instance whose relevance pattern exceeds their representable rank. Scale: 46 and 50k document variants.
- **Anisotropy shrinks effective dimension.** Gao et al. (ICLR 2019) and Li et al. (EMNLP 2020) established the representation-degeneration/narrow-cone phenomenon independently.

## 5. What Is Not Known

- **Theoretically open.** No lower bound on $d$ for *natural-language* relevance matrices. All lower bounds are worst-case combinatorial; whether real corpora have low thresholdable rank is unproven either way. Also open: whether the $\varepsilon^{-2}\log n$ JL rate is the right currency for retrieval at all, given retrieval needs only order, not distances.
- **Empirically open.** The scaling law $L^*(d)$ — the document length at which reconstruction or task sufficiency breaks — has never been measured across a $d$-sweep with training held fixed. Runnable today at ~$10^2$ GPU-days; nobody has run it.
- **Methodologically blocked.** $I(Z;Y)$ is not estimable at $d \in [256, 4096]$ with available samples; all mutual-information estimators degrade badly in this regime. So "bits retained" is currently a *proxy* (inversion BLEU, downstream accuracy) with no calibration to bits.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability**. Every reported dimension effect varies $d$ together with model, training data, and contrastive batch composition; MTEB rankings therefore cannot separate "the vector is too small" from "the encoder was trained badly at that size." And the two candidate explanations for failure — an information ceiling versus a decoder that cannot read what is there — are not identifiable from the encoder alone: a failed inversion is evidence about the *pair*, not about $Z$. Add that MTEB corpora are $10^4$–$10^6$ documents while the theoretical break points are conjectured at $10^6$–$10^7$, so the standard benchmark suite is evaluated below the regime it is used to make claims about.

## 7. Current Research (as of 2026)

- **Matryoshka + quantization stacks** as the practical answer to the capacity/cost tradeoff; now default in Nomic, OpenAI, Cohere and open Qwen-embedding lines.
- **Multi-vector / late interaction revival** (ColBERT lineage, ColPali-style extensions) explicitly motivated by the single-vector rank bound — Google DeepMind and Stanford IR are the visible groups.
- **Embedding inversion and privacy** (Cornell/Morris lineage), now the main empirical instrument for measuring retained information.
- **Rank-theoretic retrieval bounds** following Weller et al. 2025; tightening the fit-based extrapolation into a real bound on natural corpora is the open thread. *(frontier — verify)*
- **Compression-as-evaluation**: using LLM compression rates to calibrate what an embedding "should" hold. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Is the length ceiling of fixed-size embeddings information-theoretic or architectural?

**Setup.** Train an encoder family at $d \in \{64, 128, 256, 512, 1024, 2048\}$ with *everything else identical*: same backbone (a 0.5B open model), same 10B-token contrastive corpus, same schedule. For each $d$, train a strong inverter (vec2text-style, iterative refinement, 50 steps) to reconstruct inputs at $L \in \{16, 32, 64, 128, 256, 512\}$ tokens. Cost: ~120 A100-days.

**Control arm.** A *bit-matched classical codec*: compress the same text with a neural LM arithmetic coder to exactly $db$ bits (truncating), then decode. This arm has no learned semantics and no anisotropy — it measures what the bit budget alone can do.

**The deciding number.** $L^{*}(d)$ = token length at which exact-match reconstruction crosses 50%, and specifically the **ratio $\rho = L^{*}_{\text{embed}}(d) \,/\, L^{*}_{\text{codec}}(d)$ at $d=1024$**.

- $\rho \gtrsim 0.5$ and $L^*_{\text{embed}}$ growing linearly in $d$ ⇒ the ceiling is the bit budget; further gains require more bits, and the field should stop tuning encoders for length.
- $\rho \lesssim 0.1$ with $L^*_{\text{embed}}$ plateauing while $d$ grows ⇒ the ceiling is architectural, and the headroom claim is quantified for the first time.

## 9. Key References

- **[Foundational]** Claude E. Shannon. *Prediction and Entropy of Printed English.* Bell System Technical Journal, 1951.
- **[Foundational]** William B. Johnson, Joram Lindenstrauss. *Extensions of Lipschitz mappings into a Hilbert space.* Contemporary Mathematics, 1984.
- **[Foundational]** Naftali Tishby, Fernando Pereira, William Bialek. *The Information Bottleneck Method.* Allerton, 1999. — arXiv:physics/0004057
- **[Theory SOTA]** Kasper Green Larsen, Jelani Nelson. *Optimality of the Johnson-Lindenstrauss Lemma.* FOCS, 2017. — arXiv:1609.02094
- **[Theory SOTA]** Noga Alon, Bo'az Klartag. *Optimal Compression of Approximate Inner Products and Dimension Reduction.* FOCS, 2017.
- **[SOTA]** Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee. *On the Theoretical Limitations of Embedding-Based Retrieval.* 2025. — arXiv:2508.21038
- **[SOTA]** Aditya Kusupati et al. *Matryoshka Representation Learning.* NeurIPS, 2022. — arXiv:2205.13147
- **[SOTA]** John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, Alexander M. Rush. *Text Embeddings Reveal (Almost) As Much As Text.* EMNLP, 2023. — arXiv:2310.06816
- **[Empirical]** Nils Reimers, Iryna Gurevych. *The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes.* ACL, 2021.
- **[Empirical]** Omar Khattab, Matei Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR, 2020. — arXiv:2004.12832
- **[Empirical]** Kawin Ethayarajh. *How Contextual are Contextualized Word Representations?* EMNLP, 2019. — arXiv:1909.00512
- **[Empirical]** Grégoire Delétang et al. *Language Modeling Is Compression.* ICLR, 2024. — arXiv:2309.10668
- **[Survey]** Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers. *MTEB: Massive Text Embedding Benchmark.* EACL, 2023. — arXiv:2210.07316

## 10. Worked Example

Take a 768-dim float32 embedding, the GTR-base setting from Morris et al.

**Raw channel width.** $R = 768 \times 32 = 24{,}576$ bits.

**Usable width under similarity resolution.** Retrieval distinguishes scores only to about $\varepsilon = 0.01$ in cosine, so $C_{\text{usable}} \approx 768 \log_2 100 \approx 5{,}100$ bits.

**Source side.** A 32-token passage is about 144 characters; at 0.9 bits/char that is $\approx 130$ bits. A 512-token passage is $\approx 2{,}070$ bits.

**Prediction.** Both fit inside 5,100 bits with room to spare. Naive capacity accounting says exact inversion of a 512-token passage should be easy.

**Observation.** Exact reconstruction is ~92% at 32 tokens (130 bits) and collapses well before 512 tokens. So the achieved rate is roughly $130 / 24{,}576 \approx 0.5\%$ of raw width, or ~2.5% of the similarity-limited width.

**The obstruction, made visible.** There is a 40–200× gap between capacity accounting and achieved reconstruction, and *the experiment as run cannot say what closes it*. Three explanations fit the same data: (a) anisotropy — the narrow cone means effective dimension is a small fraction of 768, so $C_{\text{usable}}$ is overstated; (b) the contrastive objective discards token order and function words deliberately, so the bits are not lost but never encoded; (c) the inverter is weak. Nothing in the measurement separates them, because the only instrument — the decoder — is confounded with the thing being measured. That is why the bit-matched codec control in §8 is the load-bearing part of the design, not a nicety.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*