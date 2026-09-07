---
id: 01-tokenization/bpe-merge-optimality-gap
title: "Optimality Gap of BPE Merges"
topic: 01-tokenization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimality Gap of BPE Merges

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/bpe-merge-optimality-gap` · **Status:** partially-solved

## 1. Problem Statement

Byte-Pair Encoding (BPE) builds a subword vocabulary by repeatedly merging the most frequent adjacent symbol pair in a training corpus. The merge rule is greedy and myopic: it never reconsiders, and it optimizes the immediate one-step reduction in token count.

Three distinct questions are routinely conflated:

- **Theory variant.** How far below the optimum can the greedy merge sequence fall on its own stated objective — minimizing corpus token count under a budget of $k$ merges? Is the optimum computable, and is the greedy ratio tight?
- **Measurement variant.** Given a corpus and a budget, can the optimum actually be computed or bounded at realistic scale ($k \approx 10^5$, corpus $\approx 10^{12}$ bytes), so that the empirical gap is a measured number rather than a worst-case bound?
- **Method / consequence variant.** Does closing the compression gap improve a language model? A tokenizer that is optimal for compression is not known to be optimal for next-token prediction, and the two objectives are not known to be monotonically related.

Solving the problem means: (a) a tight constant for the greedy approximation ratio on the token-count objective; (b) a measured gap on a real corpus at production vocabulary size; and (c) a controlled training run showing whether that gap transfers to bits-per-byte at fixed compute.

## 2. Formal Setting

Corpus $\mathcal{C}$ is a multiset of word types $w$ with counts $n_w$, each $w$ a string over base alphabet $\Sigma$ (bytes, $|\Sigma|=256$). Merges do not cross a pre-tokenizer boundary — that is an assumption, not a fact about language (see below).

A merge sequence is $\mu = (p_1,\dots,p_k)$, $p_t = (x_t,y_t)$ a pair of current symbols. Applying $p_t$ rewrites every left-to-right, non-overlapping occurrence of $x_t y_t$ as a new symbol $x_ty_t$. Write $T_t(w)$ for the token sequence of $w$ after $t$ merges.

**Corpus token count** (measured by running the tokenizer and summing):
$$L(\mu) = \sum_{w} n_w \, |T_k(w)|.$$

**Objective / merge utility**, the quantity greedy maximizes one step at a time:
$$U(\mu) = L(\emptyset) - L(\mu) = \sum_{t=1}^{k} c_t(p_t), \qquad c_t(p) = \\#\{\text{non-overlapping occurrences of } p \text{ after } t-1 \text{ merges}\}.$$

**Optimality gap:**
$$\gamma_k = \frac{U(\mu^{\text{greedy}})}{\max_{|\mu|=k} U(\mu)} \in (0,1], \qquad \text{or in absolute terms } \Delta_k = L(\mu^{\text{greedy}}) - \min_{|\mu|=k} L(\mu).$$

**Compression rate**, the reported form: $\rho = L(\mu)/B$ tokens per byte, with $B=\sum_w n_w \,\mathrm{bytes}(w)$.

**Downstream quantity**, the only one that matters for models: bits per byte at fixed compute $C$,
$$\mathrm{BPB} = \frac{\rho \cdot \bar{\ell}}{\ln 2}, \qquad \bar{\ell} = \text{mean cross-entropy in nats per token on held-out text}.$$
Per-token perplexity is *not* comparable across tokenizers; only $\mathrm{BPB}$ is.

**Assumptions, and which are violated.**
1. *Pre-tokenization boundaries are fixed.* Violated in practice: GPT-2-style regex splits, digit-grouping rules and whitespace handling change $\rho$ by more than most merge-order effects.
2. *Corpus counts are the deployment distribution.* Violated: tokenizers trained on web English are used on code, other languages and long digits.
3. *Non-overlapping left-to-right occurrence counting.* An approximation; the true maximum matching for self-overlapping pairs (`aa` in `aaaaa`) differs.
4. *Token count is a proxy for model loss.* Not established (§4).

## 3. State of the Art

**Theory SOTA (established).** Zouhar et al., *A Formal Perspective on Byte-Pair Encoding* (Findings of ACL 2023), give the first formal treatment: BPE training is a greedy maximization of $U$, the objective is not submodular in general, and greedy carries a constant-factor approximation guarantee (reported as $\ge 0.37\cdot\mathrm{OPT}$), with an $O(N\log M)$ implementation. Kozma and Voderholzer, *Theoretical Analysis of Byte-Pair Encoding* (arXiv preprint, 2024), prove the optimal-merge-sequence problem is APX-complete and tighten the greedy constant above $0.37$. Whittington, Bachmann and Pimentel, *Tokenisation is NP-Complete* (arXiv preprint, 2024), prove NP-completeness for both the direct (choose a vocabulary of size $k$) and bottom-up (choose a merge sequence) formulations.

**Empirical SOTA.** No published exact optimum at production scale. Best available: local-search and beam variants over the merge sequence, and SaGe / PathPiece-style vocabulary constructions that optimize token count directly rather than greedily (Schmidt et al., *Tokenization Is More Than Compression*, EMNLP 2024). Reported compression improvements over greedy BPE are single-digit percentages of $\rho$.

**Claimed but unablated.** That better compression yields better models. Gallé (EMNLP-IJCNLP 2019) reports a correlation between shorter sequences and MT quality; Goldman et al. (Findings of ACL 2024) report the correlation holds for generative tasks and weakens for classification. Schmidt et al. report that a tokenizer explicitly minimizing token count did **not** dominate downstream — evidence against the transfer assumption. These are benchmark numbers, not causal ablations with tokenizer held as the sole varied factor at matched compute.

## 4. What Is Known

- **Hardness.** Optimal tokenization is NP-complete; the merge-sequence variant is APX-complete (2024). So an exact optimum at $k=10^5$ is out of reach absent structure.
- **Greedy is a constant-factor approximation** on $U$, with the published constant in the $0.37$–$0.6$ range depending on formulation. This is a worst-case bound over adversarial corpora, not a statement about English.
- **Measured compression gaps are small.** Search-based and non-greedy vocabularies beat greedy BPE by roughly $1$–$5\%$ in tokens-per-byte at vocabularies of $32$k–$64$k on English web corpora — a few percent, not a factor.
- **Compression gains do not reliably transfer.** Schmidt et al. trained dozens of models at $350$M–$2.4$B parameters and found the minimal-token-count tokenizer was not the best downstream. Bostrom and Durrett (Findings of EMNLP 2020) found unigram-LM vocabularies beat BPE on downstream tasks at RoBERTa-base scale ($\approx125$M) *despite* similar compression — the ranking flips depending on which quantity you score.
- **Inference-time segmentation matters separately.** Uzan et al., *Greed Is All You Need* (ACL 2024), show that for a fixed vocabulary, the choice of inference-time segmentation algorithm changes token counts and downstream scores.

## 5. What Is Not Known

- **Theoretically open.** The tight greedy approximation constant on $U$. Upper and lower bounds do not meet. Also open: whether a PTAS is excluded beyond APX-completeness for the *bottom-up* variant under realistic Zipfian corpus assumptions.
- **Empirically open.** The actual $\gamma_k$ on a real corpus at $k \ge 32{,}000$. Nobody has published a certified upper bound on $\max_\mu U(\mu)$ (e.g. via LP/Lagrangian relaxation) for a production corpus, so the measured gap is unbounded from above by anything but heuristic search.
- **Methodologically blocked.** Whether the gap *matters*. The link from $\Delta_k$ to $\mathrm{BPB}$ at fixed compute has no established functional form, and comparing two tokenizers requires two full pretraining runs whose difference is at or below seed noise (§10).

## 6. Why It Is Hard

Three named obstructions.

1. **Non-identifiability of the argmin.** Many merge sequences achieve the same $U$ while producing *different segmentations of the same word* (§10). The objective does not pin down the tokenizer, so "optimal BPE" names a set, not an object, and different elements of that set may train differently.
2. **Absent certificate.** With no computable upper bound on $\max_\mu U(\mu)$ at scale, any reported gap is $U(\text{greedy})/U(\text{best found})$ — a lower bound on $\gamma_k$ that a better search would move. Reported gaps measure search effort, not distance from optimum.
3. **Confounded downstream measurement.** Changing the tokenizer changes tokens per byte, so at fixed token budget it changes the data seen, and at fixed data it changes sequence length, FLOPs per byte, and the position-embedding regime simultaneously. An evaluation that reports per-token perplexity does not measure compression quality at all — it rewards making tokens longer.

## 7. Current Research (as of 2026)

- Formal-language and complexity analyses of tokenization (Cotterell's group at ETH Zürich; Pimentel and collaborators) — extending NP/APX results to inference-time segmentation and to multilingual vocabularies.
- Objective-driven vocabulary construction: SaGe, PathPiece, and successors (Pinter's group, BGU) — optimize a stated objective instead of greedy merging, then measure downstream.
- Tokenizer-aware scaling laws: fitting $\mathrm{BPB}(C, \rho)$ so a compression change can be priced in compute-equivalent terms *(frontier — verify)*.
- Byte-level and tokenizer-free models (byte-latent / dynamic-patching architectures) which, if they close the gap with tokenized models, dissolve the problem rather than solve it *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** Does closing the greedy merge gap change bits-per-byte at fixed compute?

- **Scale.** Corpus: 20B bytes of English web text (deduplicated). Vocabulary $k=32{,}000$. Models: 1.4B parameters, trained on 28B bytes, $\approx 4$ epochs-free, identical architecture, data order and hyperparameters.
- **Arms.** (A) Standard greedy BPE. (B) Best-found non-greedy merge sequence — beam search over merges, beam width 64, plus 10⁵ steps of local swap search, keeping the sequence with the lowest $L$. (C) Control arm: greedy BPE with a different *tie-breaking rule* and a different training-corpus subsample of the same size, which changes the vocabulary without changing the objective value. Three seeds per arm.
- **Deciding number.** $\delta = \mathrm{BPB}_A - \mathrm{BPB}_B$ on held-out text, compared against $\sigma_{\text{control}}$, the seed-and-arm-C spread. Report also $\Delta\rho = (\rho_A-\rho_B)/\rho_A$.
- **Decision rule.** If $\Delta\rho \ge 2\%$ but $|\delta| < 2\sigma_{\text{control}}$, the compression gap is real and downstream-inert: the theory variant is decoupled from the method variant and further merge-order optimization is not worth compute. If $\delta \ge 2\sigma_{\text{control}}$, the gap transfers, and the certified-upper-bound work in §5 becomes the priority.

## 9. Key References

- **[Foundational]** Philip Gage. *A New Algorithm for Data Compression.* The C Users Journal, 1994.
- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. — arXiv:1508.07909
- **[SOTA — theory]** Vilém Zouhar, Clara Meister, Juan Luis Gastaldi, Li Du, Mrinmaya Sachan, Ryan Cotterell. *A Formal Perspective on Byte-Pair Encoding.* Findings of ACL 2023. — arXiv:2306.16837
- **[SOTA — theory]** László Kozma, Johannes Voderholzer. *Theoretical Analysis of Byte-Pair Encoding.* arXiv preprint, 2024.
- **[SOTA — theory]** Philip Whittington, Gregor Bachmann, Tiago Pimentel. *Tokenisation Is NP-Complete.* arXiv preprint, 2024.
- **[SOTA — empirical]** Craig W. Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omri Uzan, Yuval Pinter, Chris Tanner. *Tokenization Is More Than Compression.* EMNLP 2024. — arXiv:2402.18376
- **[Empirical]** Kaj Bostrom, Greg Durrett. *Byte Pair Encoding Is Suboptimal for Language Model Pretraining.* Findings of EMNLP 2020. — arXiv:2004.03720
- **[Empirical]** Omer Goldman, Avi Caciularu, Matan Eyal, Kris Cao, Idan Szpektor, Reut Tsarfaty. *Unpacking Tokenization: Evaluating Text Compression and Its Correlation with Model Performance.* Findings of ACL 2024. — arXiv:2403.06265
- **[Empirical]** Omri Uzan, Craig W. Schmidt, Chris Tanner, Yuval Pinter. *Greed Is All You Need: An Evaluation of Tokenizer Inference Methods.* ACL 2024.
- **[Survey/context]** Matthias Gallé. *Investigating the Effectiveness of BPE: The Power of Shorter Sequences.* EMNLP-IJCNLP 2019.
- **[Context]** Taku Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL 2018. — arXiv:1804.10959

## 10. Worked Example

**Part 1 — the argmin is not a tokenizer (hand-checkable).** Corpus of three word types: `ab` ×5, `bc` ×5, `abc` ×4. Initial token count $L(\emptyset)=5(2)+5(2)+4(3)=32$.

Pair counts: $c(\texttt{ab}) = 5+4 = 9$, $c(\texttt{bc}) = 5+4 = 9$. Exact tie. Budget $k=2$.

- Path A: merge `ab` (utility 9, $L=23$), then `bc` (utility 5) → $L=18$.
- Path B: merge `bc` (utility 9, $L=23$), then `ab` (utility 5) → $L=18$.
- Alternative: `ab` then `abc` → $9+4=13$, $L=19$. Worse.

Both optima give $U=14$ and the *same vocabulary* $\{\texttt{ab},\texttt{bc}\}$ — but `abc` segments as `[ab][c]` under A and `[a][bc]` under B. Identical objective value, different input sequences to the model. The objective cannot distinguish them; only a training run can, and no training run has been used to break such ties.

**Part 2 — the gap is inside the noise.** Anchor: GPT-2 BPE compresses English at roughly $4$ characters per token, so $\rho_A \approx 0.25$ tokens/byte. Suppose arm B in §8 improves compression by $3\%$: $\rho_B = 0.2425$. Take $\bar\ell_A = 2.80$ nats/token. Then
$$\mathrm{BPB}_A = 0.25 \times 2.80 / \ln 2 = 1.010 \ \text{bits/byte}.$$
Arm B's tokens are longer, so its per-token loss must rise. If it rises by $2\%$ ($\bar\ell_B = 2.856$), $\mathrm{BPB}_B = 0.2425 \times 2.856/\ln 2 = 0.999$ — a gain of $0.011$ bits/byte. If it rises by the full $3\%$, the gain is exactly zero. If it rises by $4\%$, arm B is worse.

Seed-to-seed spread in bits/byte at the 1B scale is on the order of $0.005$. So the entire decision turns on whether $\bar\ell$ rises by $2\%$ or $4\%$ — a quantity nobody has predicted from tokenizer statistics — and the effect is roughly two standard deviations wide. That is the obstruction: a $3\%$ compression win, which is already at the optimistic end of what non-greedy search buys, produces a downstream signal of the same order as training noise, and its sign is not determined by the compression number alone.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*