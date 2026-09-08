---
id: 01-tokenization/continual-vocabulary-adaptation
title: "Continual Vocabulary Adaptation Under Distribution Shift"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continual Vocabulary Adaptation Under Distribution Shift

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/continual-vocabulary-adaptation` · **Status:** open

## 1. Problem Statement

A deployed language model carries a vocabulary $V_0$ frozen at pretraining time. The text it later sees drifts: new languages, new code libraries, new proper nouns, new markup, new date formats. The tokenizer does not drift with it. Rare or unseen strings fragment into long byte-level sequences, which inflates inference cost, degrades in-context length budgets, and pushes the model onto segmentations it never trained on.

**The problem:** given a pretrained model $(\theta_0, V_0)$, a stream of shifted data, and a compute budget far below retraining, produce an updated $(\theta_t, V_t)$ that improves loss and cost on the new distribution *without* losing performance on the old one — repeatedly, over many adaptation rounds, not once.

Three variants that are usually conflated:

- **Measurement.** Define a single quantity that trades compression against quality against forgetting, and is comparable across two models with *different vocabularies*. Cross-entropy in nats/token is not comparable across tokenizers; nats/byte is, but does not price the token-count savings that motivate the change.
- **Method.** Choose which tokens to add, which to prune, how to initialize their embeddings and unembeddings, and how much continued pretraining to spend, so that quality recovers.
- **Theory.** Bound the loss increase from a vocabulary swap as a function of the segmentation distance between $V_{t-1}$ and $V_t$; and determine whether repeated adaptation is stable (converges) or drifts (each round costs more than the last).

Solving it means: an adaptation procedure that, over $\geq 5$ sequential shifts, keeps the compute-normalized quality on the union of all distributions at or above a full-retrain control, at $<5\%$ of retrain cost.

## 2. Formal Setting

Let $\Sigma$ be the byte alphabet and $\mathcal{D}_t$ a distribution over strings $x \in \Sigma^*$ at round $t$. A tokenizer is a pair $T_t = (V_t, \mathrm{seg}_t)$ with $V_t \subset \Sigma^+$ and a segmentation map $\mathrm{seg}_t : \Sigma^* \to V_t^*$. A model $M_t = (\theta_t, T_t)$ assigns $p_{\theta_t}(\mathrm{seg}_t(x))$.

**Compression (measured).** Fertility on a held-out sample $S \sim \mathcal{D}_t$:
$$\phi_t(\mathcal{D}) = \frac{\sum_{x \in S} |\mathrm{seg}_t(x)|}{\sum_{x \in S} |x|_{\text{bytes}}} \quad \text{(tokens per byte).}$$
Report on a fixed byte budget ($\geq 10^7$ bytes), not a fixed token budget — a token budget makes the metric self-referential.

**Quality (measured, tokenizer-invariant).** Bits per byte:
$$\mathrm{BPB}_t(\mathcal{D}) = \frac{-\sum_{x\in S}\log_2 p_{\theta_t}(\mathrm{seg}_t(x))}{\sum_{x\in S}|x|_{\text{bytes}}}.$$
This is the only likelihood number comparable across vocabularies, and it is exact only if $\mathrm{seg}_t$ is deterministic; under stochastic segmentation it is an upper bound on $-\log_2 p(x)$ by Jensen.

**Forgetting.** With $\mathcal{D}_0$ the original distribution, $F_t = \mathrm{BPB}_t(\mathcal{D}_0) - \mathrm{BPB}_0(\mathcal{D}_0)$, plus a downstream $\Delta$ on a fixed task suite held constant across rounds.

**Cost.** Serving cost is proportional to tokens, so the deployment objective is a scalarization
$$J_t = \underbrace{\mathrm{BPB}_t(\mathcal{D}_t)}_{\text{quality}} + \lambda \underbrace{\phi_t(\mathcal{D}_t)}_{\text{spend}} + \mu \underbrace{\max(0, F_t)}_{\text{regression}},$$
with $\lambda$ set by the price ratio of a token to a unit of quality, and $C_t$ = adaptation FLOPs, budgeted as a fraction $\rho$ of the original pretraining FLOPs.

**Assumptions, and which fail.**
1. *Segmentation is deterministic and cheap.* Holds for BPE/Unigram merges; **violated** by subword regularization (Kudo, 2018) and by any learned segmenter.
2. *Adding a token leaves other tokens' distributions unchanged.* **Violated** — BPE merge order is global; inserting a merge re-segments unrelated strings.
3. *The embedding matrix is the only vocabulary-dependent parameter.* **Violated** in tied-embedding models, where the change also perturbs the output head and its logit scale.
4. *$\mathcal{D}_t$ is observable before adaptation.* **Violated** in deployment: the shift is detected from traffic that has already been mis-tokenized, so the sample is filtered by the model's own failure mode.

## 3. State of the Art

**Established (with ablations).**
- **Cross-lingual embedding initialization.** WECHSEL (Minixhofer et al., NAACL 2022) initializes new embeddings as bilingual-dictionary-weighted combinations of old ones; FOCUS (Dobler & de Melo, EMNLP 2023) does the same using overlapping-token similarity, no dictionary. Both beat random initialization at matched continued-pretraining budgets, ablated in the original papers.
- **Continued pretraining recipes.** Ibrahim et al. (TMLR 2024) show LR re-warming plus replay of 1–5% original data recovers near-from-scratch quality on a distribution shift at 405M and 10B scale — but with the vocabulary held *fixed*.
- **Vocabulary size scaling.** Tao et al. (NeurIPS 2024) show compute-optimal vocabulary grows sublinearly with parameters; most open models are under-vocabularied at their size.

**Claimed but not independently ablated.**
- Chinese-LLaMA (Cui et al., 2023) extends LLaMA's 32,000-token vocabulary to 49,953 and reports large Chinese gains; the vocabulary change and the 120GB of continued pretraining are not separated.
- EEVE (Kim et al., 2024) reports a staged parameter-freezing schedule for Korean vocabulary expansion; the stage ordering is ablated only within the paper.
- Zero-Shot Tokenizer Transfer (Minixhofer et al., NeurIPS 2024) trains a hypernetwork mapping any tokenizer to embeddings, recovering most of the original accuracy with no retraining. Strong result; reproductions at 7B+ across more than a handful of tokenizers are thin.

**Benchmark-number-only.** Cross-lingual vocabulary adaptation inference speedups — Yamaguchi et al. (Findings of EMNLP 2024) report up to ~2.7× generation speedup for target-language adaptation. The number is a throughput measurement on specific language/model pairs, not a general law.

**Nothing above is sequential.** Every published method is a single adaptation step. Round-2-and-beyond is essentially unstudied.

## 4. What Is Known

- **Fragmentation is expensive and unequal.** Ahia et al. (EMNLP 2023) measure up to ~5× token-count differences for the same content across languages under one tokenizer; cost and latency scale with it.
- **Tokenizer quality transfers to downstream quality.** Rust et al. (ACL 2021) show a language-specific tokenizer recovers a substantial part of the gap between mBERT and a monolingual model at BERT-base scale (110M).
- **Initialization matters more than it should.** FOCUS and WECHSEL report several-point downstream differences versus random init at ~100M–1B scale with fixed small adaptation budgets; the gap narrows as continued-pretraining tokens grow.
- **Intrinsic tokenizer metrics correlate only weakly with quality.** Zouhar et al. (ACL 2023) propose Rényi efficiency as a better predictor than fertility; the correlation is real but modest, and Uzan et al. (ACL 2024) show inference-method choice alone moves intrinsic scores.
- **Vocabulary swaps are recoverable in one shot.** Multiple groups recover near-baseline quality after a swap with $\rho \sim 10^{-2}$ of pretraining compute (order 10B tokens for a 7B model).

## 5. What Is Not Known

- **Empirically open.** Whether repeated adaptation is stable. No published study runs $\geq 3$ sequential vocabulary changes on one checkpoint and measures whether recovery cost per round is flat, growing, or shrinking. The experiment is runnable today on a 1B model for a few thousand GPU-hours; nobody has published it.
- **Empirically open.** Whether token *pruning* (removing dead tokens to keep $|V|$ fixed) is free. Pruning is what makes continual adaptation bounded rather than monotonically growing, and its forgetting cost is unmeasured.
- **Theoretically open.** No bound of the form $\Delta\mathrm{BPB} \leq f(d(T_{t-1}, T_t))$ for any segmentation distance $d$. There is not even an agreed $d$.
- **Theoretically open.** Whether embedding initialization can be optimal in any decision-theoretic sense, or whether all methods are heuristics on an ill-posed inverse problem.
- **Methodologically blocked.** The joint objective $J_t$ has no agreed $\lambda$. Papers report fertility and accuracy separately and pick whichever improved. Without a fixed exchange rate between a saved token and a lost accuracy point, "better adaptation" is not a well-ordered claim.

## 6. Why It Is Hard

**Confounded measurement, primarily.** Every published adaptation bundles three interventions — new tokens, new embeddings, and continued pretraining on new data — and reports one number. The continued pretraining alone typically explains most of the gain (Gururangan et al., ACL 2020, established domain-adaptive pretraining works with the *original* vocabulary). Isolating the vocabulary's contribution requires a matched-data, matched-compute control arm that almost no paper runs.

**Non-identifiability of the initialization target.** A new token $v$ has no pretraining gradient history. Its "correct" embedding is defined only relative to the post-adaptation model, which does not exist yet. WECHSEL and FOCUS each impose a different prior; there is no ground truth against which to score them.

**Compute cost of the only decisive experiment.** The question is about round 5, not round 1, so the cost is 5× a full adaptation plus a full-retrain control — and it must be run at a scale where forgetting is visible (forgetting is weaker at small scale, so 100M-parameter results may not transfer).

## 7. Current Research (as of 2026)

- **Tokenizer-agnostic and byte-level models** as a way to sidestep the problem: byte- and patch-level architectures remove $V$ entirely. Trades the vocabulary problem for a sequence-length problem; whether the trade is favorable at frontier scale is unsettled *(frontier — verify)*.
- **Hypernetwork tokenizer transfer**, following ZeTT (Minixhofer, Ponti and collaborators, Edinburgh/Cambridge lineage) — extending to sequential rather than one-shot transfer *(frontier — verify)*.
- **Larger vocabularies by default** following Tao et al. (2024); several 2025-era open models ship 128k–256k vocabularies, which shrinks but does not remove drift.
- **Superword tokenization** (SuperBPE, Liu et al., 2025), which allows tokens to cross whitespace and changes the fragmentation profile of exactly the rare-entity strings that drive drift.
- **Domain-adaptive tokenization** in code and biomedical settings, where identifier and nomenclature churn is fast and measurable.

## 8. Concrete Next Experiment

**The sequential-adaptation stability test.**

- **Scale.** One 1.4B-parameter decoder, pretrained on 100B tokens with a 32k BPE vocabulary. Five sequential shifts, each 4B tokens: (1) Python, (2) Rust, (3) German, (4) clinical notes, (5) 2026 web news. Vocabulary held at 32k throughout — each round adds 4k domain tokens and prunes the 4k lowest-usage tokens. Replay 2% of the original mix.
- **Arms.** (a) Adapt with vocabulary change, FOCUS init. (b) **Control:** identical data, identical compute, vocabulary *frozen* at $V_0$. (c) **Upper-bound control:** retrain from scratch on the union at round 5. (d) Random-init ablation of (a).
- **Cost.** ~20B adaptation tokens per arm; roughly 3–5k A100-hours total.
- **The deciding number.** Recovery cost per round $R_k$ = adaptation tokens needed for arm (a) to reach arm (b)'s round-$k$ BPB on $\mathcal{D}_k$, at equal $F_k \leq 0.01$ bits/byte on $\mathcal{D}_0$. **If $R_5 / R_1 \leq 1.2$, continual vocabulary adaptation is stable and the engineering problem is open-but-tractable. If $R_5/R_1 \geq 2$, vocabulary churn compounds and the field should default to frozen large vocabularies or byte-level models.**

Secondary readout: $\phi$ savings on $\mathcal{D}_5$ from arm (a) versus (b), which prices the whole exercise.

## 9. Key References

- **[Foundational]** Sennrich, Haddow, Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL, 2018. — arXiv:1804.10959
- **[Foundational]** Gururangan et al. *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.* ACL, 2020. — arXiv:2004.10964
- **[SOTA]** Minixhofer, Paischer, Rekabsaz. *WECHSEL: Effective Initialization of Subword Embeddings for Cross-lingual Transfer of Monolingual Language Models.* NAACL, 2022. — arXiv:2112.06598
- **[SOTA]** Dobler, de Melo. *FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models.* EMNLP, 2023. — arXiv:2305.14481
- **[SOTA]** Minixhofer, Ponti, Vulić. *Zero-Shot Tokenizer Transfer.* NeurIPS, 2024. — arXiv:2405.07883
- **[SOTA]** Ibrahim et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024. — arXiv:2403.08763
- **[Analysis]** Rust, Pfeiffer, Vulić, Ruder, Gurevych. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL, 2021. — arXiv:2012.15613
- **[Analysis]** Zouhar et al. *Tokenization and the Noiseless Channel.* ACL, 2023.
- **[Analysis]** Ahia et al. *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP, 2023.
- **[Analysis]** Tao et al. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024. — arXiv:2407.13623
- **[Applied]** Yamaguchi, Villavicencio, Aletras. *An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Language Model Inference.* Findings of EMNLP, 2024.
- **[Survey]** Wu et al. *Continual Learning for Large Language Models: A Survey.* 2024. — arXiv:2402.01364

## 10. Worked Example

A 7B model with $|V_0| = 32{,}000$ serves a code assistant. A new library, `polars`, appears in traffic. Its identifiers (`scan_parquet`, `with_columns`, `lazyframe`) fragment.

**Measured.** On 10 MB of `polars` code, $\phi = 0.42$ tokens/byte, versus $0.28$ on `pandas` code. On 4.2M bytes of user traffic that is 1.76M tokens rather than 1.18M — **+49% serving cost** on that slice.

**The fix.** Add 512 `polars` tokens, prune 512 tokens with zero traffic counts. New $\phi = 0.30$. Savings: 0.12 tokens/byte × 4.2M bytes = **504k tokens per 10 MB served**. At $0.30 per million output tokens that is $0.15 per 10 MB — real but small.

**The cost.** After the swap, BPB on the original mix rises from 0.712 to 0.744 (+0.032). Recovering it takes ~8B tokens of continued pretraining with 2% replay: roughly $6 \times 7\text{e}9 \times 8\text{e}9 = 3.4\times10^{20}$ FLOPs, order 1,000 A100-hours.

**Where the obstruction becomes visible.** The break-even is 1,000 GPU-hours against $0.015 per MB served — about 20 TB of `polars` traffic. Fine, if `polars` traffic is durable. But three months later `polars` is displaced, its 512 tokens are dead weight, and round 2 begins from a checkpoint that has already absorbed one swap. **Nobody has measured whether round 2 costs the same 8B tokens or 16B.** That single unmeasured ratio — not the initialization method, not the fertility gain — decides whether continual vocabulary adaptation is a viable operational practice or a one-time trick.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*