---
id: 01-tokenization/cross-tokenizer-distillation
title: "Cross-Tokenizer Distillation and Logit Alignment"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Cross-Tokenizer Distillation and Logit Alignment

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/cross-tokenizer-distillation` · **Status:** open

## 1. Problem Statement

A teacher model $T$ and a student model $S$ define distributions over *different* token vocabularies produced by *different* tokenizers. Standard knowledge distillation minimizes a divergence between per-position next-token distributions, which requires the two models to segment the same string identically. When $V_T \neq V_S$, that per-position correspondence does not exist: the teacher may emit `_tokeniz|ation` where the student emits `_token|iz|ation`, and no bijection between logit coordinates exists.

**Input:** teacher $T$ (weights or logit access), student architecture with vocabulary $V_S$, a corpus $\mathcal{D}$ of strings, a compute budget.
**Output:** student parameters $\theta$.
**Objective:** match the teacher's *string-level* distribution, not its token-level one.

Three variants, routinely conflated:

- **Measurement.** Given two models with different tokenizers, compute a divergence between their induced distributions over strings that is finite, comparable across vocabulary pairs, and not dominated by segmentation entropy. Currently ill-posed for arbitrary pairs.
- **Method.** Build a training loss that transfers more than the hard-label cross-entropy baseline (sequence-level KD on teacher samples), at matched compute.
- **Theory.** Characterize when a string-level distribution representable under $V_T$ is representable under $V_S$ at a given parameter count, and bound the transfer gap induced by vocabulary mismatch alone.

Solved means: a loss that beats sequence-level KD by a margin larger than seed variance, across at least three unrelated tokenizer pairs, with an ablation isolating the alignment mechanism from the extra teacher data it consumes.

## 2. Formal Setting

A tokenizer is a map $\tau: \Sigma^* \to V^*$ with detokenizer $\tau^{-1}$ satisfying $\tau^{-1}(\tau(x)) = x$. An autoregressive model over $V$ induces a distribution on token sequences $p(v_{1:n})$ and, by pushforward, a distribution on strings:

$$p^{\mathrm{str}}(x) = \sum_{v \in \tau^{-1}[x]} p(v_{1:|v|}),$$

where $\tau^{-1}[x]$ is the set of all token sequences detokenizing to $x$. For deterministic BPE the *canonical* segmentation has probability mass, but non-canonical segmentations are assigned nonzero probability by the model and are never trained on — the "tokenization mismatch" of Cao & Rimell (EMNLP 2021).

The quantity we actually want is string-level KL:

$$D_{\mathrm{KL}}\!\left(p_T^{\mathrm{str}} \,\|\, p_S^{\mathrm{str}}\right) = \sum_{x \in \Sigma^*} p_T^{\mathrm{str}}(x)\log\frac{p_T^{\mathrm{str}}(x)}{p_S^{\mathrm{str}}(x)}.$$

**How it is measured in practice.** The marginalization is intractable (exponentially many segmentations), so implementations use one of:

1. **Sequence-level KD** — sample $x^{(i)} \sim p_T^{\mathrm{str}}$, minimize $-\sum_i \log p_S(\tau_S(x^{(i)}))$. Unbiased for the canonical path only; ignores teacher uncertainty entirely.
2. **Alignment-based** — compute a matching $\pi$ between teacher positions $1..n_T$ and student positions $1..n_S$ from character spans (dynamic programming over byte offsets, or minimum-edit-distance over token strings as in DSKD), then apply KL at matched positions. Measured cost: one DP pass, $O(n_T n_S)$ per sequence.
3. **Distribution-free** — sort both logit vectors descending, truncate/pad to common length $k$, and take an $\ell_1$ / Wasserstein-1 distance (Universal Logit Distillation, Boizard et al., 2024):
$$\mathcal{L}_{\mathrm{ULD}} = \sum_{t} \big\| \mathrm{sort}(p_T(\cdot\mid x_{<t})) - \mathrm{sort}(p_S(\cdot\mid x_{<t})) \big\|_1 .$$
4. **Likelihood matching** — evaluate both models on the *same* strings and match $\log p^{\mathrm{str}}$ over shared chunks, e.g. Approximate Likelihood Matching (Minixhofer, Vulić & Ponti, 2025), which restricts to spans where both tokenizers agree on boundaries.

**Assumptions and their violations.**

- *Bijective span alignment exists.* Violated whenever token boundaries cross (`_tokeniz` vs `_token|iz`); byte-level fallbacks and unicode normalization differences (NFC vs NFKC, LLaMA's `▁` handling) break byte offsets outright.
- *Sorted logits carry transferable information.* Violated when vocabularies differ in size ($32{,}000$ vs $256{,}000$): sorted tails are dominated by vocabulary-size-dependent mass, so $\mathcal{L}_{\mathrm{ULD}}$ conflates "different beliefs" with "different vocabulary size".
- *Canonical segmentation carries nearly all mass.* Approximately true for trained models but unquantified above 1B parameters.
- *Teacher and student see the same context.* Violated by differing chat templates and special tokens.

Compute is measured in student-forward-equivalent FLOPs, including the teacher forward passes; alignment methods that need full teacher logits over the student's own samples cost roughly $2\times$ hard-label KD per step.

## 3. State of the Art

**Established.**

- Sequence-level KD on teacher-generated text (Kim & Rush, EMNLP 2016) is tokenizer-agnostic by construction and remains the baseline that all cross-tokenizer losses must beat. It reliably works; it is the control arm.
- On-policy distillation with student-generated sequences (GKD, Agarwal et al., ICLR 2024; MiniLLM with reverse KL, Gu et al., ICLR 2024) beats offline KD at fixed data budget — but both assume a *shared* vocabulary.
- Vocabulary/embedding transfer without distillation works: WECHSEL (Minixhofer et al., NAACL 2022) and Zero-Shot Tokenizer Transfer (Minixhofer, Vulić & Ponti, NeurIPS 2024) retrofit a new tokenizer onto a trained model via embedding initialization or a hypernetwork, recovering most task accuracy with orders of magnitude less compute than retraining.

**Claimed but under-ablated.**

- ULD (Boizard et al., 2024) reports gains over hard-label fine-tuning for teacher/student pairs with disjoint vocabularies on summarization and instruction-following. The ablation that is missing everywhere: sorted-logit matching versus *the same teacher data with hard labels only*, at matched steps and matched teacher-token budget. Without it, part of the gain is the extra teacher forward passes.
- DSKD (Zhang et al., EMNLP 2024) projects teacher and student hidden states into a shared space with a cross-model attention mechanism and uses minimum-edit-distance token alignment. Gains are reported at the 0.1–7B scale on instruction benchmarks; the alignment component has not been isolated from the projection component in an independent reproduction.
- Multi-teacher fusion across tokenizers (FuseLLM, Wan et al., ICLR 2024) aligns teacher distributions by token-level matching before fusing. Reported as benchmark numbers; the alignment quality itself is not measured.
- Approximate Likelihood Matching (2025) is the strongest current framing — it defines the objective at the string level rather than patching token correspondence — but as of this writing exists as a single-group result.

There is **no theory SOTA**: no bound relates vocabulary mismatch to achievable distillation gap.

## 4. What Is Known

- Same-tokenizer distillation works and the numbers are stable: DistilBERT retains ~97% of BERT-base GLUE score with 40% fewer parameters (Sanh et al., 2019, 66M student).
- Reverse-KL distillation (MiniLLM) improves over forward KL for instruction-following, verified at 120M–13B student scales (ICLR 2024).
- Non-canonical segmentations are systematically mis-scored: models assign them far lower probability than their true string mass, so single-path likelihoods underestimate $p^{\mathrm{str}}$ (Cao & Rimell, EMNLP 2021; measured on GPT-2-scale LMs, where marginalizing over segmentations changes perplexity by a measurable margin).
- Subword regularization (Kudo, ACL 2018) shows models *can* be trained to be robust across segmentations, at some cost in canonical-path perplexity — evidence that segmentation invariance is trainable, not free.
- Vocabulary size interacts with optimal model size (Tao et al., 2024, "Scaling Laws with Vocabulary"): compute-optimal vocabulary grows with model size, so teacher and student *should* have different vocabularies — which makes this problem structural, not incidental.
- Tokenizer transfer costs are small relative to pretraining: ZeTT-style hypernetworks reach near-original accuracy on a swapped tokenizer at ~1e-3 of pretraining compute (NeurIPS 2024, 1–7B models).

## 5. What Is Not Known

- **Methodologically blocked.** There is no accepted way to *measure* cross-tokenizer distribution distance. Every reported number is a downstream benchmark score, so improvements in "alignment" cannot be separated from improvements in fine-tuning. Until $D_{\mathrm{KL}}(p_T^{\mathrm{str}} \| p_S^{\mathrm{str}})$ has a tractable estimator with known bias, the field is optimizing a proxy of unknown fidelity.
- **Empirically open.** Whether any cross-tokenizer logit loss beats sequence-level KD at matched *total* compute, at ≥7B student scale, across ≥3 tokenizer pairs. All published comparisons hold steps or data fixed, not FLOPs including teacher inference.
- **Theoretically open.** No bound of the form: for student capacity $C$ and vocabularies $(V_T, V_S)$ with mismatch measure $m$, the achievable $D_{\mathrm{KL}}$ is at least $f(m, C)$. Also open: whether the sorted-logit (optimal-transport) objective is a consistent estimator of *any* string-level divergence — it is currently a heuristic with no identified population target.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the target combined with a confounded control**.

Non-identifiability: a token-level distribution does not determine a unique string-level distribution without summing over exponentially many segmentations, and the inverse map is many-to-one. Two students with identical $p^{\mathrm{str}}$ can have arbitrarily different logits under a different vocabulary, so any position-wise logit loss penalizes differences that do not exist at the string level. Sorted-logit matching removes the coordinate problem but destroys the identity of the tokens being compared — it matches the *shape* of the distribution, not its content, and the shape is a function of $|V|$.

Confounded control: every cross-tokenizer method consumes teacher forward passes that the hard-label baseline does not. Reported gains therefore mix (i) alignment signal, (ii) extra teacher compute, (iii) implicit label smoothing from soft targets. No published ablation separates the three.

## 7. Current Research (as of 2026)

- **Likelihood-space objectives** — matching $\log p^{\mathrm{str}}$ on agreed spans rather than logits on aligned positions (Minixhofer/Vulić/Ponti line at Cambridge/Edinburgh). Most principled current direction.
- **Optimal-transport losses over vocabulary embeddings** — replacing the sorted-$\ell_1$ surrogate with a Wasserstein distance under a cost derived from token-string edit distance or embedding similarity *(frontier — verify)*.
- **Tokenizer-free students** — byte-level and dynamic-patch architectures (Byte Latent Transformer, Pagnoni et al., Meta, 2024) sidestep $V_S$ entirely; distilling a BPE teacher into a byte student is the cleanest test case and is being attempted *(frontier — verify)*.
- **Model merging across vocabularies** — FuseChat-style pipelines in open-source labs, mostly evaluated by leaderboard score.
- **Speculative decoding with mismatched drafters** — an industrial driver: a shared-vocabulary constraint currently blocks reusing small models across families.

## 8. Concrete Next Experiment

**Question:** does any cross-tokenizer logit loss beat sequence-level KD at matched total FLOPs?

**Scale.** Teacher: a 7B instruction-tuned model with a 32k SentencePiece vocabulary. Students: 1.3B, trained from a fixed public checkpoint, three vocabularies — 32k identical to teacher (upper bound), 128k byte-level BPE (mismatched), 256-symbol byte-level (extreme mismatch). Budget: $2\times10^{20}$ student-equivalent FLOPs per arm, **teacher inference counted in the budget**.

**Arms.**
1. *Control:* sequence-level KD on teacher samples, hard labels only.
2. ULD sorted-logit loss.
3. Span-aligned KL (byte-offset DP alignment, KL on aligned positions).
4. Approximate likelihood matching on agreed spans.
5. *Oracle:* shared-vocabulary token-level KL (32k student).

**Deciding number.** Held-out **string-level negative log-likelihood on teacher-generated text, in nats per byte** — vocabulary-independent, therefore comparable across arms. Report the gap to the oracle arm, $\Delta = \mathrm{NLL}_{\text{arm}} - \mathrm{NLL}_{\text{oracle}}$, with 3 seeds.

**Decision rule:** a cross-tokenizer loss is real if it closes ≥25% of the control-to-oracle gap in nats/byte, with the improvement exceeding $2\sigma$ of seed variance. If no arm beats the control by more than seed noise, the honest conclusion is that current logit alignment contributes nothing beyond teacher data, and the field should move to likelihood-space or tokenizer-free students.

## 9. Key References

- **[Foundational]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS Deep Learning Workshop, 2015. — arXiv:1503.02531
- **[Foundational]** Yoon Kim, Alexander M. Rush. *Sequence-Level Knowledge Distillation.* EMNLP, 2016. — arXiv:1606.07947
- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Taku Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL, 2018. — arXiv:1804.10959
- **[Measurement]** Kris Cao, Laura Rimell. *You should evaluate your language model on marginal likelihood over tokenisations.* EMNLP, 2021.
- **[SOTA]** Nicolas Boizard, Kevin El Haddad, Céline Hudelot, Pierre Colombo. *Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs.* 2024.
- **[SOTA]** Songming Zhang et al. *Dual-Space Knowledge Distillation for Large Language Models.* EMNLP, 2024.
- **[SOTA]** Benjamin Minixhofer, Edoardo Ponti, Ivan Vulić. *Zero-Shot Tokenizer Transfer.* NeurIPS, 2024. — arXiv:2405.07883
- **[SOTA]** Benjamin Minixhofer, Ivan Vulić, Edoardo Ponti. *Cross-Tokenizer Distillation via Approximate Likelihood Matching.* 2025.
- **[SOTA]** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang. *MiniLLM: Knowledge Distillation of Large Language Models.* ICLR, 2024. — arXiv:2306.08543
- **[SOTA]** Rishabh Agarwal et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.* ICLR, 2024. — arXiv:2306.13649
- **[Related]** Fanqi Wan et al. *Knowledge Fusion of Large Language Models.* ICLR, 2024.
- **[Related]** Benjamin Minixhofer, Fabian Paischer, Navid Rekabsaz. *WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer.* NAACL, 2022. — arXiv:2112.06598
- **[Related]** Artidoro Pagnoni et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* Meta AI, 2024.
- **[Survey]** Xiaohan Xu et al. *A Survey on Knowledge Distillation of Large Language Models.* 2024.

## 10. Worked Example

Teacher: LLaMA-family, 32k SentencePiece. Student: GPT-2-family, 50k byte-level BPE. Context: `"The tokenization"`.

Segmentations:

| Model | Tokens |
|---|---|
| Teacher | `▁The` `▁token` `ization` (3 tokens) |
| Student | `The` `Ġtoken` `ization` (3 tokens) |

Byte offsets happen to align here, so span-aligned KL is defined. Now extend the context by one word — `"The tokenization of"`:

| Model | Tokens |
|---|---|
| Teacher | `▁The` `▁token` `ization` `▁of` |
| Student | `The` `Ġtoken` `ization` `Ġof` |

Still aligned. Now use `"The tokeniser"`:

| Model | Tokens |
|---|---|
| Teacher | `▁The` `▁token` `iser` |
| Student | `The` `Ġtoken` `iser` |

Aligned again — which is why small hand-picked examples make the method look fine. Now `"detokenization"`:

| Model | Tokens |
|---|---|
| Teacher | `▁de` `token` `ization` |
| Student | `det` `oken` `ization` |

Byte boundaries: teacher $\{2, 7, 14\}$, student $\{3, 7, 14\}$. Position 1 covers bytes 0–2 for the teacher and 0–3 for the student. There is no matching. Any DP alignment must either merge positions 1–2 on both sides (losing the per-position distribution, since $p(\text{token}\mid \text{de})$ has no student counterpart) or drop the span.

**The number.** On 10k random English Wikipedia sentences, measure the fraction of token boundaries shared by both tokenizers. For a SentencePiece-32k / byte-BPE-50k pair this is roughly 60–75% on plain English prose and drops sharply on code and non-Latin script — for Devanagari or Chinese, where one tokenizer is byte-fallback and the other has native subwords, shared-boundary rates fall below 20%.

Consequences, made concrete:

- **Span-aligned KL** trains on the aligned ~70% of English positions and is nearly blind on code and non-Latin text — exactly the distribution regions where teacher knowledge is most valuable. The loss silently reweights the training distribution.
- **Sorted-logit ULD** avoids alignment but compares a 32k-dimensional sorted vector to a 50k-dimensional one. If the teacher's top-1 mass is $0.62$ and the student's is $0.55$, $\ell_1$ registers a gap of at least $0.14$ — but part of that gap is that the student spreads mass over 18k more coordinates, not that it disagrees about the next character. The loss cannot tell those apart.
- **Sequence-level KD** loses all soft-target information but has zero alignment bias.

The obstruction is visible: each method is undefined, biased, or uninformative precisely where the vocabularies differ most — and that is the only region where the problem exists at all.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*