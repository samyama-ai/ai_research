---
id: 01-tokenization/bpe-merge-ordering-optimality
title: "Optimality Gap of BPE Merge Ordering"
topic: 01-tokenization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimality Gap of BPE Merge Ordering

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/bpe-merge-ordering-optimality` · **Status:** open

## 1. Problem Statement

Byte-Pair Encoding (BPE) builds a vocabulary by repeatedly merging the most frequent adjacent symbol pair. The merge sequence is chosen greedily, one step at a time, with no lookahead. The question is how much is lost by that greed.

Three variants, which are routinely conflated:

- **Measurement variant.** For a fixed corpus and merge budget $K$, how far is greedy BPE's *compression* from the best achievable by any length-$K$ merge sequence? Solving this means computing or tightly bounding the ratio on real corpora.
- **Theory variant.** What is the worst-case approximation ratio of greedy BPE against the optimal merge sequence, and is finding the optimum NP-hard / APX-hard? Solving this means a matching upper and lower bound.
- **Method variant — the one that matters for language models.** Does the compression gap translate into a *downstream* gap? Concretely: is there a merge sequence $\mu$ of the same budget $K$ whose trained model beats greedy BPE's on bits-per-byte and on task accuracy, at fixed parameters, data and compute? Solving this means exhibiting such a $\mu$ (or proving greedy is within $\epsilon$).

Status is **open** in all three, but for different reasons (§5).

## 2. Formal Setting

Corpus $\mathcal{D}$ is a multiset of words $w$ with counts $c_w$, over base alphabet $\Sigma$ (bytes). A **merge sequence** is $\mu = (\mu_1,\dots,\mu_K)$, $\mu_t = (x,y)$ with $x,y$ in the vocabulary $V_{t-1} = \Sigma \cup \{\mu_1,\dots,\mu_{t-1}\}$. Applying $\mu$ left-to-right, greedily and non-overlapping within each word, gives token sequences $T_\mu(w)$.

**Tokenized length**, measured by counting tokens after running the actual tokenizer (not by a closed form):
$$\ell(\mu) \;=\; \sum_{w} c_w\,|T_\mu(w)|, \qquad \ell(\varnothing) = \sum_w c_w |w|.$$

**Compression utility** and **compression optimality gap**:
$$u(\mu) = \ell(\varnothing) - \ell(\mu), \qquad \gamma_K = \frac{u(\mu^{\mathrm{BPE}}_K)}{\max_{|\mu|=K} u(\mu)} \in (0,1].$$

Note $\mu^{\mathrm{BPE}}$ selects $\mu_t = \arg\max_{(x,y)} \sum_w c_w \cdot \\#\{\text{occurrences of } (x,y) \text{ in } T_{\mu_{<t}}(w)\}$ — the *occurrence count*, which in standard implementations (`subword-nmt`, HuggingFace `tokenizers`) counts overlapping positions and therefore is not equal to the tokens actually saved (§10).

**Downstream objective**, normalized per byte so it is comparable across tokenizers:
$$\mathrm{BPB}(\mu) = \frac{1}{|\mathcal{D}_{\text{eval}}|_{\text{bytes}}}\sum_{i} -\log_2 p_\theta\!\left(t_i \mid t_{<i}\right), \quad \theta = \arg\min_\theta \mathcal{L}(\theta; T_\mu(\mathcal{D}_{\text{train}})).$$
**Downstream gap** $\Delta_K = \mathrm{BPB}(\mu^{\mathrm{BPE}}_K) - \min_{|\mu|=K}\mathrm{BPB}(\mu)$.

Assumptions, and which fail in practice:

1. *Vocabulary size $=|\Sigma|+K$.* Holds for pure BPE; violated by pre-tokenization regex splits, byte fallback, special tokens, and vocabulary pruning (Picky BPE).
2. *Compression is the objective.* Assumed by the theory literature; empirically contested (§3, §4).
3. *Fixed training FLOPs across arms.* Violated by construction — a tokenizer that emits fewer tokens sees fewer training steps at fixed token budget, or more data at fixed step count. Every honest comparison must fix one and report the other.
4. *Deterministic encoding.* Violated by BPE-dropout and by inference-time re-tokenization mismatches.

## 3. State of the Art

**Theory SOTA (established).**
- Zouhar et al., *A Formal Perspective on Byte-Pair Encoding* (Findings of ACL 2023), formalize BPE as greedy maximization of $u(\mu)$, show the objective is not submodular in general, and give a curvature-dependent approximation guarantee plus an $O(N\log M)$ training algorithm in place of the naive $O(NM)$.
- Kozma & Voderholzer, *Theoretical Analysis of Byte-Pair Encoding* (2024), prove the optimal-pair-encoding problem is APX-complete and that greedy BPE's worst-case approximation ratio lies in $[0.333,\,0.625]$ — a factor-2 window that is still not closed.
- Whittington, Bachmann & Pimentel, *Tokenisation is NP-Complete* (ACL 2025), prove NP-completeness of both direct and bottom-up optimal tokenization.

**Empirical SOTA (mixed evidence).**
- Gallé (EMNLP-IJCNLP 2019) reports that BPE's benefit tracks sequence shortening in NMT — compression as proxy.
- Schmidt et al., *Tokenization Is More Than Compression* (EMNLP 2024), introduce PathPiece, a tokenizer that minimizes token count given a vocabulary, and find it does **not** dominate BPE downstream across 64 trained LMs. This is the strongest direct evidence that $\gamma_K$ and $\Delta_K$ decouple.
- SuperBPE (Liu et al., 2025) reports 27% fewer tokens and +4.0% average over 30 tasks at 8B scale by allowing merges across whitespace. **Claimed but not ablated against a compression-matched control** — the gain could be compression, could be the changed segmentation prior.

**Benchmark-number-only results.** Rényi-efficiency selection (Zouhar et al., *Tokenization and the Noiseless Channel*, ACL 2023) correlates with BLEU across tokenizers, but the correlation is measured over tokenizer *families*, not over merge orderings at fixed family and budget — it is not evidence about $\Delta_K$.

## 4. What Is Known

- **Hardness.** Optimal tokenization is NP-complete (Whittington et al., 2025); optimal pair encoding is APX-complete (Kozma & Voderholzer, 2024). So exact $\max_{|\mu|=K} u(\mu)$ is out of reach for corpus-scale instances.
- **Worst-case ratio bracketed, not pinned.** $0.333 \le \gamma^{\text{worst}} \le 0.625$.
- **Typical-case ratio is much better than worst case.** On natural-language corpora the measured shortfall of greedy versus stronger search is single-digit percent in token count, not 2–3×; no independently reproduced number exists at $K = 50{,}000$ on a multi-billion-token corpus.
- **Compression differences of the size BPE variants produce are weakly predictive downstream.** Schmidt et al. (2024), 350M-parameter models, find no reliable monotone relation between corpus token count and downstream accuracy across their tokenizer grid.
- **Tokenization changes do move real numbers.** Dagan, Synnaeve & Rozière (ICML 2024) show tokenizer choice materially changes code-generation performance and inference cost during domain adaptation — the effect is not noise, even if compression is the wrong summary statistic.
- **Theory says tokenization matters at all.** Rajaraman, Jiao & Ramchandran (2024) prove that on $k$-th-order Markov sources, transformers with a suitable tokenizer approach the source entropy rate while character-level models collapse toward the unigram bound.

## 5. What Is Not Known

- **Theoretically open.** The exact worst-case ratio of greedy BPE inside $[0.333, 0.625]$. Also open: whether $u(\mu)$ admits a constant-factor-better polynomial approximation than greedy, and whether the gap shrinks under a realistic source model (e.g. Zipfian word frequencies) rather than adversarial strings.
- **Empirically open.** The typical-case $\gamma_K$ at production scale. Beam search or lookahead-$b$ merge selection over a 10B-token corpus at $K=50{,}000$ is runnable on a few hundred CPU-hours; nobody has published the resulting ratio.
- **Methodologically blocked.** $\Delta_K$ itself. There is no agreed protocol for comparing two tokenizers at matched compute: per-token loss is not comparable across vocabularies, bits-per-byte is comparable but changes the effective sequence length and thus the attention budget per byte, and downstream accuracy is confounded by vocabulary-size effects on the embedding parameter count. Until the control arm is defined, a measured $\Delta_K$ is not interpretable.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement plus absent ground truth**, not compute.

- No ground truth: $\max_{|\mu|=K} u(\mu)$ is APX-complete, so the denominator of $\gamma_K$ is unknown even for a single corpus. Every reported "gap" is a gap against another heuristic.
- Confounding: changing the merge order changes token count, which changes tokens-per-document, which changes the number of gradient updates per epoch, the effective context in bytes, and the embedding/output matrix cost. Four levers move together; a single training run cannot attribute the difference.
- Objective mismatch: greedy BPE optimizes occurrence count, the theory optimizes token count, and the model cares about $\mathrm{BPB}$ after training. These are three different functions, and the counting used at training time is not even an unbiased estimate of the first (§10).

## 7. Current Research (as of 2026)

- **Exact and approximate optimal tokenization.** Follow-ups to Whittington et al. and Kozma & Voderholzer on ILP/dynamic-programming formulations and tighter greedy bounds; ETH Zürich / Cambridge (Cotterell, Pimentel) and TU Berlin (Kozma) are the visible groups. *(frontier — verify)*
- **Vocabulary refinement after merging.** Picky BPE (Chizhov et al., EMNLP 2024) removes intermediate tokens made redundant by later merges — a direct attack on greedy's non-recoverable early commitments.
- **Superword and boundary-relaxed vocabularies.** SuperBPE and related work at AI2 / UW, expanding the merge search space rather than improving the search. *(frontier — verify)*
- **Inference-time decoupling.** Uzan et al. (ACL 2024) show the encoding algorithm can be swapped independently of the merge list, which means merge-ordering quality and segmentation quality can in principle be measured separately. Underexploited.
- **Tokenizer-free and byte-level models** (byte latent / dynamic patching) sidestep the problem; if they win at scale, $\Delta_K$ becomes moot. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** does a non-greedy merge ordering at the same budget buy anything downstream?

- **Scale.** Corpus: 30B tokens of FineWeb-Edu. Vocabulary $K = 32{,}000$ merges over byte alphabet, identical pre-tokenization regex in all arms. Models: 1.4B parameters, trained for a fixed **20B-byte** budget (not token budget — bytes are the invariant across arms), 3 seeds per arm. About 6 × 1.4B-model runs; ~2,000 A100-hours total.
- **Arms.**
  1. *Control:* standard greedy BPE.
  2. *Treatment:* beam-search merge selection, beam width 8, scoring each candidate by realized token savings after two-step lookahead (this fixes the occurrence-count bias of §10 as well as adding lookahead).
- **Instrumentation.** Report $\ell(\mu)$ on a held-out 1B-byte split for both arms, giving the compression ratio $r = \ell(\mu^{\text{beam}})/\ell(\mu^{\mathrm{BPE}})$. Report $\mathrm{BPB}$ on the same held-out split, plus accuracy on HellaSwag, ARC-e, MMLU.
- **Deciding number.** $\Delta = \mathrm{BPB}(\mu^{\mathrm{BPE}}) - \mathrm{BPB}(\mu^{\text{beam}})$, in bits per byte, with a seed-variance error bar. **If $\Delta < 0.002$ bits/byte while $r \le 0.97$** (i.e. ≥3% better compression buys under 0.2% of a typical ~1.0 bits/byte model), the compression gap is downstream-irrelevant and the method variant is closed in the negative. If $\Delta > 0.01$ bits/byte, greedy merge ordering is leaving real capability on the table and merge search becomes a live research target.

## 9. Key References

- **[Foundational]** Philip Gage. *A New Algorithm for Data Compression.* The C Users Journal, 1994.
- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. — arXiv:1508.07909
- **[Theory SOTA]** Vilém Zouhar, Clara Meister, Juan Luis Gastaldi, Li Du, Tim Vieira, Mrinmaya Sachan, Ryan Cotterell. *A Formal Perspective on Byte-Pair Encoding.* Findings of ACL 2023. — arXiv:2306.16837
- **[Theory SOTA]** László Kozma, Johannes Voderholzer. *Theoretical Analysis of Byte-Pair Encoding.* 2024. — arXiv:2411.08671
- **[Theory SOTA]** Philip Whittington, Gregor Bachmann, Tiago Pimentel. *Tokenisation is NP-Complete.* ACL 2025. — arXiv:2412.15210
- **[Empirical SOTA]** Craig W. Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omri Uzan, Yuval Pinter, Chris Tanner. *Tokenization Is More Than Compression.* EMNLP 2024. — arXiv:2402.18376
- **[Empirical]** Matthias Gallé. *Investigating the Effectiveness of BPE: The Power of Shorter Sequences.* EMNLP-IJCNLP 2019.
- **[Empirical]** Gautier Dagan, Gabriel Synnaeve, Baptiste Rozière. *Getting the Most Out of Your Tokenizer for Pre-training and Domain Adaptation.* ICML 2024. — arXiv:2402.01035
- **[Empirical]** Omri Uzan, Craig W. Schmidt, Chris Tanner, Yuval Pinter. *Greed is All You Need: An Evaluation of Tokenizer Inference Methods.* ACL 2024.
- **[Method]** Pavel Chizhov, Catherine Arnett, Elizaveta Korotkova, Ivan P. Yamshchikov. *BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training.* EMNLP 2024.
- **[Method]** Ivan Provilkov, Dmitrii Emelianenko, Elena Voita. *BPE-Dropout: Simple and Effective Subword Regularization.* ACL 2020. — arXiv:1910.13267
- **[Theory]** Nived Rajaraman, Jiantao Jiao, Kannan Ramchandran. *Toward a Theory of Tokenization in LLMs.* 2024. — arXiv:2404.08335
- **[Survey]** Vilém Zouhar, Clara Meister, Juan Luis Gastaldi, Li Du, Mrinmaya Sachan, Ryan Cotterell. *Tokenization and the Noiseless Channel.* ACL 2023. — arXiv:2306.16842

## 10. Worked Example

A three-word corpus, budget $K=2$, showing greedy losing on its own objective — and losing for the *implementation's* reason, not an adversarial one.

Corpus (word → count): `aaa` → 10, `bc` → 15, `de` → 14. Initial length $\ell(\varnothing) = 10\cdot3 + 15\cdot2 + 14\cdot2 = 88$ symbols.

Pair occurrence counts as computed by standard BPE trainers (overlapping positions within a word):

| pair | occurrence count | tokens actually saved if merged |
|---|---|---|
| `(a,a)` | $10 \times 2 = 20$ | $10$ |
| `(b,c)` | $15$ | $15$ |
| `(d,e)` | $14$ | $14$ |

`aaa` contains `(a,a)` at two positions, so it is counted twice — but merging left-to-right non-overlapping turns `a a a` into `aa a`, saving only **one** token per occurrence of the word.

**Greedy BPE.** Step 1 picks `(a,a)` (count 20, the maximum) and saves 10. Step 2 picks `(b,c)` (count 15) and saves 15. Total $u = 25$, $\ell = 63$.

**Optimal at $K=2$.** Merge `(b,c)` and `(d,e)`: $u = 29$, $\ell = 59$.

$$\gamma_2 = 25/29 = 0.862.$$

Greedy emits 63 tokens where 59 suffice — **6.8% more tokens**, and it is not recoverable: the budget is spent.

What the example makes visible:

- The gap here is not caused by lack of lookahead alone. It is caused by the **selection statistic being biased**: occurrence count over-weights self-overlapping pairs relative to realized savings. Every mainstream BPE trainer has this behavior, and it fires on real data wherever repeated characters occur (`www`, `---`, `aaa`, indentation runs).
- The gap is budget-dependent. At $K=3$ all three merges fit and $\gamma_3 = 1$. So any measured $\gamma_K$ is a statement about the budget, not about BPE.
- Nothing here says the 6.8% token difference changes a trained model's bits-per-byte by any amount. That is exactly the unmeasured quantity of §8 — the obstruction is that the compression number is easy and the number anyone cares about is not.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*