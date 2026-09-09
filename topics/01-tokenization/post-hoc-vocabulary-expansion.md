---
id: 01-tokenization/post-hoc-vocabulary-expansion
title: "Post-Hoc Vocabulary Expansion Without Retraining"
topic: 01-tokenization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Post-Hoc Vocabulary Expansion Without Retraining

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/post-hoc-vocabulary-expansion` · **Status:** partially-solved

## 1. Problem Statement

A pretrained LM ships with a frozen tokenizer. On a domain it was not tokenized for — Telugu, Lean 4 proof scripts, DNA, chemical SMILES — its fertility (tokens per byte) is 2–5× worse than a domain-fit tokenizer, so inference cost, context consumption and per-byte loss all inflate. **Vocabulary expansion** adds $m$ new tokens and must supply their input embeddings and output (unembedding) rows.

Three variants, routinely conflated:

- **Measurement.** Given an expanded model, decide whether it is better than the original. Blocked by the fact that perplexity is not comparable across tokenizers; bits-per-byte is, but downstream-quality-per-FLOP is what practitioners actually want.
- **Method.** Produce $E'_{\text{new}}, U'_{\text{new}}$ that recover the original model's per-byte loss with a *training budget of zero* (strict) or $\ll$ pretraining compute (practical). Strictly-zero is the interesting version; every deployed system currently cheats with continued pretraining.
- **Theory.** Is the target — the embedding a new token *would have had* if it had been in the vocabulary from step 0 — identifiable at all from the frozen model plus a corpus? No known answer.

Solved means: for a 7B-parameter model, add 30k tokens, get $\geq$ 1.5× fertility reduction on the target domain, and lose $\leq$ 0.01 bits-per-byte relative to the unexpanded model, using $<10^{-3}$ of pretraining FLOPs.

## 2. Formal Setting

Tokenizers $T: \Sigma^* \to V^*$ and $T': \Sigma^* \to V'^*$ over byte strings, with $V \subset V'$, $m = |V'| - |V|$. Model parameters $\theta = (E, \theta_{\text{body}}, U)$ with $E \in \mathbb{R}^{|V| \times d}$, $U \in \mathbb{R}^{|V| \times d}$ (tied iff $U = E$).

**Fertility**, measured on a held-out target corpus $D \subset \Sigma^*$ of raw bytes:
$$\phi(T, D) = \frac{\sum_{x \in D} |T(x)|}{\sum_{x \in D} |x|_{\text{bytes}}}$$
Reciprocal $1/\phi$ is bytes-per-token. Report on bytes, not words — word counts are undefined for Chinese and for code.

**Bits-per-byte**, the only cross-tokenizer-comparable loss:
$$\mathrm{BPB}(\theta, T, D) = \frac{-1}{\ln 2 \sum_x |x|_{\text{bytes}}} \sum_{x \in D} \sum_{i=1}^{|T(x)|} \ln p_\theta\big(T(x)_i \mid T(x)_{<i}\big)$$
This is a valid comparison only if $T$ is injective on $D$ and the model places no mass on strings outside the tokenizer's image — the second condition is *violated in practice*: multiple token sequences decode to the same string, so $\sum_x p_\theta$ over strings exceeds 1 and BPB is an upper bound on true per-byte code length, not an equality.

**The initialization problem.** Choose $f: V'\setminus V \to \mathbb{R}^d \times \mathbb{R}^d$ using only $(\theta, T, T', D)$ and no gradient steps through $\theta_{\text{body}}$. Practical relaxation: allow $B$ tokens of continued pretraining, $B \ll N_{\text{pretrain}}$; the quantity of interest is the curve $\mathrm{BPB}(B)$, not a single point.

**Assumptions and their status.**
- *Embeddings compose additively*: $e(v') \approx \sum_i w_i\, e(v_i)$ for the old-tokenizer decomposition $T(v') = (v_1,\dots,v_k)$. Approximately true for input embeddings, measurably worse for unembeddings (Hewitt 2021; Minixhofer et al. 2024).
- *The body is tokenizer-agnostic*. Violated: attention heads specialize to positional token-boundary patterns induced by the original segmentation.
- *New embeddings should be distributionally matched to old ones*. Only weakly justified; it is a regularizer against out-of-distribution norms, not a correctness argument.
- *Tied embeddings halve the problem*. False in the direction that matters — tying forces one vector to serve two roles with different optimal geometry.

## 3. State of the Art

**Established (ablated, reproduced):**
- **Distribution-matched random init** (Hewitt, *Initializing New Word Embeddings for Pretrained Language Models*, 2021): sample new rows from $\mathcal{N}(\mu, \Sigma)$ fit to existing embedding rows. Beats zero-init and $\mathcal{N}(0, 0.02^2 I)$ reliably; the effect is that new-token logits do not start anomalously large or small.
- **Subword-mean init**: $e(v') = \frac{1}{k}\sum_i e(v_i)$ over the old-tokenizer decomposition. Strong, nearly free baseline; Yamaguchi, Villavicencio & Aletras (*How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?*, 2024) find it competitive with far more elaborate schemes under small adaptation budgets.
- **Auxiliary-embedding alignment**: WECHSEL (Minixhofer, Pfeiffer & Vulić, NAACL 2022) and FOCUS (Dobler & de Melo, EMNLP 2023) initialize new rows as similarity-weighted combinations of overlapping-token rows, with weights from fastText embeddings in the target language. Both reproduce; both need a target-language corpus for the auxiliary model.
- **Hypernetwork init**: ZeTT (Minixhofer, Ponti & Vulić, *Zero-Shot Tokenizer Transfer*, NeurIPS 2024) trains a hypernetwork mapping a token's string to $(e, u)$, then applies it to an arbitrary new tokenizer with no target-side training. This is the strongest strictly-zero-shot method known.

**Claimed but unablated:** that vocabulary expansion *improves quality* rather than only cost. Most reports (e.g. Chinese-LLaMA, Cui, Yang & Yao 2023) confound expansion with tens of billions of tokens of continued pretraining on target-language text; the pretraining alone would explain the gain. Almost nothing isolates the expansion.

**Benchmark-number-only:** cross-lingual downstream deltas after expansion. Reported as accuracy tables on XNLI/Belebele with a single seed and no compute-matched control.

## 4. What Is Known

- **Fertility gains are large and real.** Chinese-LLaMA expanded LLaMA's 32,000-token vocabulary to 49,953 and roughly halved the token count on Chinese text (7B/13B scale, 2023). Yamaguchi et al. (2024) report cross-lingual inference speedups in the 1.7× range from adaptation at 7B scale.
- **Vocabulary size in the original model is systematically too small.** Tao et al., *Scaling Laws with Vocabulary* (NeurIPS 2024), predict a compute-optimal vocabulary near 216k for a Llama-2-70B-class budget against its actual 32k — evidence that the expansion demand is structural, not a niche fix.
- **Tokenizer choice matters but is not dominant at fixed compute.** Ali et al., *Tokenizer Choice For LLM Training: Negligible or Crucial?* (NAACL Findings 2024), at 2.8B parameters across 24 tokenizers: downstream deltas of a few points, and multilingual tokenizers cost up to ~68% more compute when mismatched.
- **Existing rare tokens are already broken.** Land & Bartolo, *Fishing for Magikarp* (EMNLP 2024), automatically detect under-trained tokens across many open models — hundreds to thousands per model. Newly added tokens start in exactly that state, so the failure mode of expansion is a *known* pathology.
- **Output rows are harder than input rows.** Consistently reported across WECHSEL/FOCUS/ZeTT: initialization quality gaps show up first in generation, not in classification probing.

## 5. What Is Not Known

- **Theoretically open.** Whether the "counterfactual embedding" is identifiable. Given frozen $\theta_{\text{body}}$ and a corpus, is there a unique $(e, u)$ minimizing expected BPB, and does it coincide with the embedding a from-scratch run would have found? No non-identifiability proof, no identifiability proof. Related: no known lower bound on the BPB penalty of any zero-training initializer as a function of $m/|V|$.
- **Empirically open.** The compute-matched ablation: expansion + $B$ tokens of continued pretraining versus *no expansion* + the same $B$ tokens on the same data. Runnable today at 7B for well under $50k. Essentially nobody publishes it, so the field does not know how much of the reported quality gain is the vocabulary at all.
- **Methodologically blocked.** Comparing models with different tokenizers on *generation* quality. BPB fixes the likelihood comparison, but sampling quality, calibration and instruction-following have no tokenizer-invariant metric; token-level entropy is not comparable, and per-byte normalization of a sampled sequence is not a proper score.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus absent ground truth**, not compute.

Every deployed expansion bundles three changes at once: (i) the segmentation of the target corpus, (ii) $m \cdot 2d$ new trainable parameters, (iii) continued pretraining on target-domain data. Each alone improves target-domain loss. The published designs vary all three together, so the marginal contribution of (i) is unestimated.

Ground truth is absent in a strict sense: the correct value of a new row is defined only relative to a from-scratch training run that was never performed, and that run's embedding is determined only up to the symmetries of the loss (rotation in the residual stream, scale traded against LayerNorm gain). So even a from-scratch reference does not give a target vector to regress against — only a target *loss*. Any method evaluated by cosine similarity to a reference embedding is measuring the wrong thing.

## 7. Current Research (as of 2026)

- **Hypernetwork tokenizer transfer** — Minixhofer, Ponti, Vulić (Cambridge/Edinburgh lineage): amortize initialization over tokenizers rather than solve per-tokenizer.
- **Dynamic and boundary-free tokenization** — retrofitting word-level or byte-level segmentation onto trained subword models, sidestepping the expansion problem by removing the fixed vocabulary (Feher, Vulić & Minixhofer, 2024). Byte-latent and hierarchical-patch architectures push in the same direction. *(frontier — verify which of these hold at $\geq$ 7B under compute-matched controls.)*
- **Trans-tokenization for low-resource languages** — Remy et al. (COLM 2024), token-translation via word alignment.
- **Vocabulary curriculum / growth during pretraining** — adding tokens on a schedule rather than post hoc. *(frontier — verify.)*
- **Under-trained-token auditing** as a release check, downstream of Land & Bartolo.

## 8. Concrete Next Experiment

**Question:** does vocabulary expansion contribute anything beyond the continued pretraining it is always bundled with?

**Scale.** One 7B open-weights base model (Llama-3.1-8B or Qwen-2.5-7B). Target domain: Telugu web text, 2B bytes held-in, 50M bytes held-out. Adaptation budget fixed at $B = 4 \times 10^9$ *bytes* consumed — bytes, not tokens, so arms are data-matched, with FLOPs logged separately.

**Arms.**
1. **Control:** original tokenizer, continued pretraining on the same 4 GB.
2. **Expanded, mean init:** +30k Telugu tokens, subword-mean init, same 4 GB.
3. **Expanded, FOCUS init:** identical but FOCUS initialization.
4. **Expanded, zero training:** ZeTT-style or FOCUS init, $B = 0$.
5. **Frozen baseline:** no expansion, no training.

Three seeds. Report both data-matched and FLOP-matched points, since arm 1 processes ~2.5× more tokens for the same bytes.

**Deciding number.** $\Delta\mathrm{BPB} = \mathrm{BPB}(\text{arm 2}) - \mathrm{BPB}(\text{arm 1})$ on held-out Telugu bytes.
- $\Delta < -0.01$ bits/byte: expansion adds real modelling capability; the method problem is worth pushing.
- $|\Delta| \le 0.01$: expansion is a pure inference-cost optimization. That is still valuable, but the entire quality literature around it is confounded and should be re-read as cost work.
- $\Delta > +0.01$: expansion is actively harmful at this budget, and the reported gains come from continued pretraining alone.

Secondary: arm 4's $\Delta\mathrm{BPB}$ against arm 5 quantifies the true zero-shot penalty — currently the least reliably reported number in the area.

## 9. Key References

- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. — arXiv:1508.07909
- **[Foundational]** Ofir Press, Lior Wolf. *Using the Output Embedding to Improve Language Models.* EACL 2017. — arXiv:1608.05859
- **[Method]** John Hewitt. *Initializing New Word Embeddings for Pretrained Language Models.* Technical report, Stanford, 2021.
- **[Method]** Benjamin Minixhofer, Fabian Paischer, Navid Rekabsaz. *WECHSEL: Effective Initialization of Subword Embeddings for Cross-lingual Transfer of Monolingual Language Models.* NAACL 2022. — arXiv:2112.06598
- **[Method]** Konstantin Dobler, Gerard de Melo. *FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models.* EMNLP 2023. — arXiv:2305.14481
- **[SOTA]** Benjamin Minixhofer, Edoardo Maria Ponti, Ivan Vulić. *Zero-Shot Tokenizer Transfer.* NeurIPS 2024. — arXiv:2405.07883
- **[Empirical]** Atsuki Yamaguchi, Aline Villavicencio, Nikolaos Aletras. *How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?* 2024. — arXiv:2406.11477
- **[Empirical]** Yiming Cui, Ziqing Yang, Xin Yao. *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca.* 2023. — arXiv:2304.08177
- **[Empirical]** Mehdi Ali et al. *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL 2024. — arXiv:2310.08754
- **[Scaling]** Chaofan Tao et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS 2024. — arXiv:2407.13623
- **[Diagnostic]** Sander Land, Max Bartolo. *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models.* EMNLP 2024. — arXiv:2405.05417
- **[Related]** François Remy et al. *Trans-Tokenization and Cross-lingual Vocabulary Transfers: Language Adaptation of LLMs for Low-Resource NLP.* COLM 2024.

## 10. Worked Example

Take Llama-3.1-8B ($|V| = 128{,}256$, $d = 4096$, untied $E$ and $U$) and 1 MB of Telugu Wikipedia.

**Step 1 — measure the problem.** Telugu is UTF-8 3 bytes/char. The base tokenizer emits roughly one token per 1.5–2 bytes on Telugu, so $\phi \approx 0.55$ tokens/byte against $\approx 0.25$ for English. Concretely: 1,000,000 bytes → ~550,000 tokens, versus ~180,000 for a 30k-token Telugu-fit vocabulary. Fertility ratio $\approx 3\times$.

**Step 2 — the parameter cost.** Adding $m = 30{,}000$ tokens costs $30{,}000 \times 4096 \times 2 = 2.46 \times 10^8$ parameters — 3% of the model, all of it untrained.

**Step 3 — the obstruction becomes visible.** Initialize with subword-mean. A new token `▁ప్రభుత్వం` ("government") decomposes into 6 old tokens; its input embedding is their mean. Now compute BPB on held-out Telugu:

| Arm | tokens for 1 MB | BPB (illustrative) | notes |
|---|---|---|---|
| Base, no expansion | 550k | 1.42 | reference |
| Expanded, mean init, $B=0$ | 180k | 2.9–4.1 | new rows never seen by the body |
| Expanded, +4 GB CPT | 180k | ~1.05 | 3.1× cheaper inference |
| Base, +4 GB CPT | 550k | **unreported** | the missing control |

The zero-training arm gets *worse* per byte even though it uses 3× fewer tokens: each new token must now carry ~5.5 bytes of information through a vector the body has never conditioned on, and the loss per token rises faster than the token count falls. Mean-init preserves direction but not the norm/logit calibration the body expects, and the unembedding row is worse still — the mean of six subword output rows systematically under-predicts the merged token, because output rows encode "what comes next given this prefix", which does not average.

The row that decides the field's central claim is row 4. It is a single 4 GB continued-pretraining run at 8B scale — days on one node — and it is the number almost nobody reports.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*