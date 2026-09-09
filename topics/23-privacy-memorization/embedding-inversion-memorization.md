---
id: 23-privacy-memorization/embedding-inversion-memorization
title: "Memorization in Embeddings and Vector Databases"
topic: 23-privacy-memorization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Memorization in Embeddings and Vector Databases

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/embedding-inversion-memorization` · **Status:** partially-solved

## 1. Problem Statement

A dense embedding $e = \phi(x)$ is routinely treated as a privacy-preserving surrogate for the text $x$: vector databases are shipped to third-party hosts, embeddings are logged, and retrieval indexes are shared across tenants. The problem is to characterise how much of $x$ — and how much of $\phi$'s *training corpus* — is recoverable from $e$ alone, and to build encoders whose embeddings retain retrieval utility while provably bounding that recovery.

Three variants, routinely conflated:

- **Measurement.** Given a released set of vectors $\{e_i\}$ and query access to $\phi$, quantify leakage. What is the right functional — exact reconstruction rate, PII token recall, or a membership advantage?
- **Method (attack).** Build $A$ mapping $e \mapsto \hat{x}$ maximising reconstruction, with and without white-box access to $\phi$, in-domain and out-of-domain.
- **Theory.** Prove that no encoder with retrieval quality above a threshold can have inversion risk below a threshold, or exhibit an encoder that breaks the trade-off.

The critical distinction, which most published work elides: **inversion** (recovering $x$ from its *own* embedding, a near-injectivity property of $\phi$) is not **memorization** (recovering or detecting records from $\phi$'s *training set*, present only as a property of fitting). A page titled "memorization in embeddings" that reports inversion numbers is measuring the wrong thing. Solving the page means separating them.

## 2. Formal Setting

Encoder $\phi_\theta:\mathcal{X}\to\mathbb{S}^{d-1}$ trained by contrastive loss on corpus $D_{\mathrm{tr}}$. A vector database is $V=\{(e_i,\mathrm{id}_i)\}_{i=1}^N$, $e_i=\phi_\theta(x_i)$, over a deployment corpus $D_{\mathrm{dep}}$ (usually disjoint from $D_{\mathrm{tr}}$).

**Inversion risk.** For attacker $A$ with black-box query access,
$$R_{\mathrm{inv}}(\phi, \mathcal{D}) = \mathbb{E}_{x\sim\mathcal{D}}\big[\,\mathbb{1}[A(\phi(x)) = x]\,\big],$$
measured as exact string match after whitespace/case normalisation, at a fixed token budget $L$ (all published numbers are $L\in\{32,64,128\}$ and degrade sharply with $L$). Softer functionals actually reported: token-F1, BLEU, and **attribute recall** $\mathrm{Rec}_a = |\,\hat{x}\cap S_a(x)\,| / |S_a(x)|$ where $S_a(x)$ is the set of PII spans (names, MRNs, dates) tagged by a fixed NER model — so $\mathrm{Rec}_a$ inherits that tagger's error rate, typically 5–10 points, and is never reported with it.

**Vec2text-style attack.** A conditional decoder plus an iterative corrector:
$$x^{(t+1)} \sim p_\psi\big(x \mid e,\; x^{(t)},\; \phi(x^{(t)})\big), \qquad x^{(0)}\sim p_\psi(x\mid e),$$
run for $T$ steps with beam width $b$; cost is $T\!\cdot\! b$ encoder calls per target.

**Memorization / membership.** For record $z$ and threshold-based attack score $s$,
$$\mathrm{Adv}(z) = \Pr[s(\phi_\theta, z) > \tau \mid z\in D_{\mathrm{tr}}] - \Pr[s(\phi_\theta,z)>\tau \mid z\notin D_{\mathrm{tr}}],$$
estimated over shadow encoders. Measuring this at all requires training $\ge 2$ encoders with a controlled split — the reason it is rarely done.

**Utility.** nDCG@10 on BEIR, or Recall@100 on the deployment corpus. The trade-off object is the frontier $\{(R_{\mathrm{inv}}, \mathrm{nDCG@10})\}$ over defences.

**Assumptions, and where they break.**
1. *Attacker knows $\phi$ and can query it freely.* Violated for proprietary APIs with rate limits and cost; relaxed by transfer attacks, which lose substantial accuracy.
2. *Embeddings are released at full float precision.* Violated: production indexes use product quantization / int8, which is an unmeasured defence.
3. *Texts are short.* Violated: real chunks are 256–512 tokens, where exact reconstruction collapses.
4. *$D_{\mathrm{tr}}\cap D_{\mathrm{dep}}=\emptyset$.* Violated in practice — encoders are trained on MS MARCO/Wikipedia, which is also what people index. This confound is why inversion results are read as memorization results.

## 3. State of the Art

**Empirical SOTA (established).**
- **vec2text** — Morris, Kuleshov, Shmatikov, Rush, *Text Embeddings Reveal (Almost) As Much As Text*, EMNLP 2023. Iterative correction against GTR-base recovers 32-token Natural Questions passages at **92% exact match**, BLEU ~97, with $T{=}50$, $b{=}8$. Ablated over $T$, beam width, and model scale; independently reproduced.
- **GEIA** — Li, Xu, Song, et al., *Sentence Embedding Leaks More Information than You Expect*, Findings of ACL 2023: a generative decoder recovers sensitive tokens without exact reconstruction.
- **Song & Raghunathan**, *Information Leakage in Embedding Models*, CCS 2020: established the three-attack taxonomy (inversion, attribute inference, membership) and showed ~50–70% word recovery for sentence embeddings under white-box multiset prediction.
- **Zhuang, Zuccon, et al.**, *Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems*, SIGIR 2024: the strongest *negative* result. vec2text degrades sharply out-of-domain and under small Gaussian noise added to stored vectors, with modest retrieval loss. This is the main reason the status here is "partially-solved" rather than "open".

**Claimed but unablated.**
- Zero-shot / universal inversion without training a per-encoder corrector (Zhang, Morris, Shmatikov, 2025) reports cross-encoder transfer; the reported gains are not ablated against a matched-compute in-domain corrector, so the transfer premium is unquantified.
- Multilingual inversion (Chen et al., ACL 2024) exists as benchmark numbers on specific language sets; no ablation isolating tokenizer effects from encoder effects.
- Claims that quantized or PQ-compressed indexes "resist" inversion: asserted in system documentation, not measured in any paper I can verify.

**Theory SOTA.** Thin. Metric-DP ($d_\chi$-privacy) over embedding space (Feyisetan et al., WSDM 2020) gives a formal guarantee, but the guarantee is over the *metric*, and its $\varepsilon$ does not translate into a bound on $R_{\mathrm{inv}}$. There is no theorem relating retrieval quality to inversion risk.

## 4. What Is Known

- 92% exact match at $L{=}32$, GTR-base (110M), in-domain NQ; drops to roughly 20–30% exact match at $L{=}64$ and near zero by $L{=}128$ under the same budget (EMNLP 2023).
- On MIMIC-III clinical notes, the same attack recovered **89% of first-and-last-name spans** from embeddings of clinical text — an attribute-recall number, not exact reconstruction.
- Gaussian noise at magnitude costing ~1–2 nDCG@10 points on BEIR reduces vec2text exact match to near zero (SIGIR 2024). Established, reproduced.
- Corrector training cost is real: ~5M passages, tens of GPU-days per encoder. Attacks are encoder-specific by construction.
- Membership advantage against contrastively trained encoders is small in the published measurements — single-digit AUC points above chance for non-duplicated records — but has been measured at far smaller scale than the inversion work.

## 5. What Is Not Known

- **Methodologically blocked.** Whether the field's headline numbers measure memorization at all. No published inversion result controls for overlap between $D_{\mathrm{tr}}$ and the evaluation corpus, so "the embedding leaks $x$" and "the encoder memorized $x$" are not separated. Until a matched in-training / held-out split is run, the quantity being reported has no clean name.
- **Empirically open.** Inversion risk at production chunk lengths (256–512 tokens) under production storage (int8 / PQ-64 / HNSW). Runnable today; unrun.
- **Empirically open.** Whether duplication count in $D_{\mathrm{tr}}$ drives embedding-space memorization the way it drives generative memorization (Carlini et al., ICLR 2023 log-linear law). No embedding analogue has been measured.
- **Theoretically open.** Any non-trivial lower bound of the form: retrieval quality $\ge q \Rightarrow R_{\mathrm{inv}} \ge f(q,d,L)$. Also open: whether $\phi$ can be made provably non-invertible for $L$-token inputs while $d \ll L\log|\mathcal V|$ bits — the counting argument suggests headroom, but no construction realises it.

## 6. Why It Is Hard

The binding obstruction is **confounded measurement plus absent ground truth**. Retrieval encoders are trained on the same public corpora used to evaluate attacks. A successful reconstruction is therefore consistent with two mechanisms — the embedding is near-injective for short strings (an information-capacity fact, no privacy content) or the encoder memorized the string — and the standard evaluation cannot tell them apart. Nobody publishes the training set of the strongest commercial encoders, so the in-training/held-out split needed to disentangle them cannot be constructed post hoc; it requires retraining an encoder, which is the compute cost that has kept the experiment unrun.

Secondary: attacks are per-encoder and cost tens of GPU-days, so defence evaluations use one or two encoders and generalise informally.

## 7. Current Research (as of 2026)

- Cornell (Shmatikov, Morris, Rush) — inversion attacks, transferable and zero-shot variants, extension from embeddings to prompts and to LM logits.
- Queensland / CSIRO (Zuccon, Zhuang) — defences and the negative results on vec2text robustness; noise-vs-utility frontiers on BEIR.
- DP representation learning groups — metric-DP and noisy embeddings, largely disconnected from the inversion-attack literature; the two use incompatible metrics. *(frontier — verify)*
- Index-level leakage — attacks that use HNSW graph structure or query-access patterns rather than the vectors themselves, at the boundary with encrypted-search literature. Early and thin. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question.** Does contrastive training memorize specific records, over and above generic short-text invertibility?

**Design.** Train two GTR-base-size encoders (110M) on a 5M-passage MS MARCO corpus. Insert 2,000 synthetic canary passages (realistic PII, 32 tokens each) at duplication counts $\{1,2,4,16,64\}$, 400 canaries per level, into encoder $\phi_A$'s corpus. Encoder $\phi_B$ is trained on the identical corpus **with the canaries removed** — this is the control arm. Train a vec2text corrector against each encoder using the standard 5M-passage recipe.

**Measurement.** For every canary, compute exact-match reconstruction from both $\phi_A$'s corrector (in-training) and $\phi_B$'s corrector (held-out, same distribution, same length, same embedding norm bucket).

**Deciding number.** The gap
$$\Delta = R_{\mathrm{inv}}^{(A)} - R_{\mathrm{inv}}^{(B)}$$
at duplication count 1. If $\Delta \le 2$ points with a 95% CI excluding 5 points ($n{=}400$ gives $\pm\!\approx\!5$ points, so run 2,000 per level for $\pm 2$), embedding inversion is **capacity, not memorization**, and the defence problem is compression/noise, not data curation. If $\Delta \ge 10$ points, or grows monotonically with duplication count, embeddings memorize and the machinery from generative memorization (deduplication, DP-SGD) transfers directly.

**Cost.** ~4 encoder-plus-corrector trainings, order 200–400 A100-hours. This is the cheapest experiment that separates the two mechanisms the field currently reports as one.

## 9. Key References

- **[Foundational]** Congzheng Song, Ananth Raghunathan. *Information Leakage in Embedding Models.* ACM CCS, 2020.
- **[SOTA]** John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, Alexander M. Rush. *Text Embeddings Reveal (Almost) As Much As Text.* EMNLP, 2023. — arXiv:2310.06816
- **[SOTA — defence]** Shengyao Zhuang, Bevan Koopman, Xiaoran Chu, Guido Zuccon. *Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems.* SIGIR, 2024.
- **[Attack]** Haoran Li, Mingshi Xu, Yangqiu Song. *Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence.* Findings of ACL, 2023.
- **[Related]** John X. Morris, Wenting Zhao, Justin T. Chiu, Vitaly Shmatikov, Alexander M. Rush. *Language Model Inversion.* ICLR, 2024.
- **[Context]** Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, Chiyuan Zhang. *Quantifying Memorization Across Neural Language Models.* ICLR, 2023.
- **[Defence]** Oluwaseyi Feyisetan, Borja Balle, Thomas Drake, Tom Diethe. *Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations.* WSDM, 2020.
- **[Multilingual]** Yiyi Chen et al. *Text Embedding Inversion Security for Multilingual Language Models.* ACL, 2024.

## 10. Worked Example

Take a 32-token clinical sentence, embedded by GTR-base into $d=768$ floats.

**Capacity check.** The text carries at most $32\log_2 30000 \approx 476$ bits. The embedding, at float32 on the unit sphere, carries up to $767\times 32 \approx 24{,}500$ raw bits. Reconstruction is not information-theoretically blocked — it is 50× over-provisioned. The 92% exact-match result is therefore fully consistent with *no memorization whatsoever*: the encoder is simply near-injective on short strings, and vec2text is a learned decoder for an invertible map.

**Now the same sentence at 512 tokens.** Content is ~7,600 bits against the same 24,500-bit vector. Still nominally invertible — but measured exact match falls to roughly zero, because the corrector's search space grows exponentially while its $T\!\cdot\! b = 400$ encoder-call budget is fixed. So the observed length cliff is an *attack-budget* artefact, not an information bound.

**The obstruction, made visible.** Two experiments give indistinguishable outputs:

| | Encoder trained on the sentence | Encoder never trained on it |
|---|---|---|
| Exact match, $L{=}32$ | 92% (measured) | **unmeasured** |
| Exact match, $L{=}512$ | ~0% | **unmeasured** |

Every published number sits in the left column. The right column is empty. Because MS MARCO and Wikipedia are simultaneously the training corpus and the evaluation corpus for essentially all open encoders, no existing result can populate it. That is why the field's strongest privacy claim — "embeddings leak the text" — currently supports no conclusion at all about memorization, and why §8 is a training run rather than an analysis.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*