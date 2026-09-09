---
id: 01-tokenization/optimal-tokenizer-construction-hardness
title: "NP-Hardness Boundary of Optimal Tokenizer Construction"
topic: 01-tokenization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# NP-Hardness Boundary of Optimal Tokenizer Construction

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/optimal-tokenizer-construction-hardness` · **Status:** partially-solved

## 1. Problem Statement

Given a corpus and a vocabulary budget, find the tokenizer that is optimal — and say exactly how hard that is.

Three variants, routinely conflated:

- **Theory variant.** Fix the objective as *compression*: minimize the number of tokens the corpus encodes to, subject to $|V| \le k$. Decide the complexity of each formalization (direct vocabulary selection; bottom-up merge sequence), and bound the approximation ratio of greedy BPE. Solving it means a matching hardness result and approximation bound.
- **Method variant.** Produce an algorithm that returns a certified near-optimal vocabulary at production scale ($k \approx 10^5$, corpus $\ge 10^{11}$ bytes) within a training-run-sized compute budget.
- **Measurement variant.** Establish that compression is the objective worth optimizing. If the tokenizer that minimizes token count does not minimize downstream loss at matched compute, the hardness result constrains the wrong problem.

The theory variant is largely settled (§4). The method and measurement variants are not.

## 2. Formal Setting

Let $\Sigma$ be the base alphabet (bytes, or Unicode characters). A corpus is a multiset of words $D = \{(w_i, c_i)\}_{i=1}^{n}$, $w_i \in \Sigma^+$, $c_i \in \mathbb{N}$ the count — measured by pre-tokenizing raw text with a fixed regex splitter (GPT-2/GPT-4 style) and counting types. Pairs never cross word boundaries; this is how every production BPE trainer actually works.

**Direct tokenization.** Choose $V \subseteq \Sigma^+$ with $\Sigma \subseteq V$ and $|V \setminus \Sigma| \le k$. Each $w$ is segmented by the *best* segmentation into elements of $V$; let $\tau_V(w)$ be that minimum number of pieces (computable in $O(|w|\cdot L)$ by dynamic programming, $L$ the longest token). Objective:

$$\mathcal{C}(V) \;=\; \sum_{i=1}^{n} c_i\, \tau_V(w_i), \qquad \text{minimize } \mathcal{C}(V).$$

**Bottom-up tokenization.** Choose an ordered merge sequence $\mu = (\langle x_1,y_1\rangle, \dots, \langle x_k,y_k\rangle)$; the segmentation is *forced* by applying merges in order, left-to-right, to exhaustion. Objective $\mathcal{C}(\mu)$ is the same sum over the resulting piece counts. The feasible set is strictly smaller than the direct one: a token is only reachable if both halves are already tokens.

**Measured quantities.**
- Compression: tokens per UTF-8 byte, $\rho = \mathcal{C}(V)/\sum_i c_i |w_i|_{\text{bytes}}$, measured on a *held-out* corpus, not the training corpus.
- Downstream quality: bits per byte, $\text{bpb} = \frac{1}{\ln 2}\cdot \frac{\sum \text{NLL}_{\text{tokens}}}{\\#\text{bytes}}$ — the only loss comparable across tokenizers, since token-level perplexity is not.
- Compute matching: fixed non-embedding parameters $N$ and fixed training *bytes*, not fixed training tokens (fixing tokens hands the better-compressing tokenizer more data).

**Assumptions, and which are violated.** (i) Objective = token count — violated: the model's loss is not a function of sequence length alone. (ii) Word-boundary independence — violated by SuperBPE-style superword tokenizers and by whitespace-free scripts. (iii) Held-out corpus matches the training corpus distribution — violated for code, multilingual, and post-2024 web mixes. (iv) Vocabulary cost is uniform in $k$ — violated: embedding/softmax parameters scale as $k \cdot d$, so $k$ trades against $N$.

## 3. State of the Art

**Theory SOTA (established).**
- Whittington, Bachmann & Pimentel, *Tokenisation is NP-Complete* (ACL 2025; arXiv:2412.15210): both direct and bottom-up tokenization are NP-complete, by reduction from max-2-SAT.
- Kozma & Voderholzer, *Theoretical Analysis of Byte-Pair Encoding* (2024; arXiv:2411.08671): optimal BPE (compression-utility maximization) is APX-complete; greedy BPE is a $0.333$-approximation of the optimal *utility* (tokens saved), and no better than $0.625$.
- Zouhar et al., *A Formal Perspective on Byte-Pair Encoding* (Findings of ACL 2023): the earlier, weaker guarantee, $\tfrac{1}{\sigma(\sigma-1)}(1-e^{-1})$-style bounds via submodularity-adjacent arguments; superseded by Kozma–Voderholzer.
- Lineage: Storer & Szymanski, *Data Compression via Textual Substitution* (JACM 1982) — macro-scheme compression NP-complete; Charikar et al., *The Smallest Grammar Problem* (IEEE Trans. Inf. Theory 2005) — NP-hard, $O(\log(n/g))$-approximable. Gallé (EMNLP 2019) connected BPE to this line.

**Empirical SOTA (mixed).**
- Greedy BPE (Sennrich et al. 2016) and UnigramLM (Kudo 2018) remain the deployed methods; no production model uses a certified-optimal vocabulary.
- PathPiece (Schmidt et al., *Tokenization Is More Than Compression*, EMNLP 2024) directly minimizes corpus token count via DP over segmentations. **Established by ablation:** across ~64 trained models up to 350M parameters, minimizing token count did *not* systematically improve downstream accuracy.
- **Claimed but unablated:** that Rényi efficiency (Zouhar et al., *Tokenization and the Noiseless Channel*, ACL 2023) is a better tokenizer-selection criterion than compression. Reported as a Spearman correlation with BLEU ($\approx 0.78$) over a set of MT tokenizers — a benchmark correlation across confounded tokenizers, not a controlled intervention. Cognetta, Zouhar, Moon & Okazaki (*Two Counterexamples to Tokenization and the Noiseless Channel*, LREC-COLING 2024) exhibit tokenizers with high Rényi efficiency and poor downstream behavior.

## 4. What Is Known

- **NP-completeness of both formalizations**, direct and bottom-up (Whittington et al., ACL 2025). The decision problem "is there $V$, $|V|\le k$, with $\mathcal{C}(V)\le t$?" is NP-complete even over a bounded alphabet.
- **APX-completeness** of optimal BPE compression utility, i.e. no PTAS unless P = NP (Kozma & Voderholzer 2024).
- **Greedy is a constant-factor approximation**: ratio in $[0.333, 0.625]$ for utility. The true worst-case constant is not pinned down — a factor-1.9 window.
- **Direct $\ne$ bottom-up.** The merge-reachability constraint costs real compression; §10 exhibits a factor-1.98 instance at $k=1$.
- **Compression gains are small at production scale, and do not track quality.** Schmidt et al. (EMNLP 2024) report that PathPiece's token-count reductions over BPE (single-digit percent, English, 32k–64k vocab) produced no consistent downstream gain across 64 models at 350M parameters.
- **Tokenization is not cosmetic.** Rajaraman, Jiao & Ramchandran (*Toward a Theory of Tokenization in LLMs*, 2024; arXiv:2404.08335) show unigram transformers on $k$th-order Markov sources are near-optimal *with* tokenization and provably far from optimal without it — so the vocabulary matters even if compression is the wrong proxy for how.

## 5. What Is Not Known

- **Theoretically open.** The exact greedy approximation constant between $0.333$ and $0.625$. Whether the *held-out* (generalization) version — minimize expected tokens per byte under the data distribution, not on the training corpus — is harder, easier, or differently structured. Whether restricting to natural-language-like inputs (bounded suffix-tree branching, Zipfian type frequencies) admits a PTAS; all hardness constructions use adversarial strings that do not occur in text.
- **Empirically open.** The size of the optimality gap on real corpora. Nobody has computed a certified-optimal vocabulary at $k = 32{,}000$ on even a 1 GB corpus, so the number "greedy BPE leaves $x\%$ of compression on the table at production scale" does not exist. Runnable today with ILP/branch-and-bound on a reduced instance.
- **Methodologically blocked.** The objective itself. There is no agreed loss $\mathcal{L}(V)$ over vocabularies whose minimization is known to minimize downstream bpb at matched compute. Until that exists, "optimal tokenizer" names a quantity we cannot define, and the complexity results apply to a proxy that PathPiece's ablation already suggests is the wrong one.

## 6. Why It Is Hard

The binding obstruction is **an evaluation that does not measure the thing it names**, compounded by **absent ground truth**.

- The hardness results are about $\mathcal{C}(V)$. The quantity anyone cares about is bpb at fixed compute. The only controlled experiment linking them (Schmidt et al. 2024, 350M scale) found the link absent. So the field has a sharp theorem about a quantity of unproven relevance.
- Measuring the true objective requires a pretraining run per candidate vocabulary. At 350M parameters that is affordable for ~64 arms; at 7B it is not. There is no cheap, validated surrogate — Rényi efficiency has published counterexamples.
- **Non-identifiability:** vocabulary size, compression rate, embedding parameter count, and effective training bytes move together. Changing $V$ changes the model's parameter count and its data budget at once. Isolating "vocabulary quality" requires simultaneously matching bytes, non-embedding parameters, and FLOPs — rarely done.
- Even the tractable-looking subproblem is large: certifying optimality over $\binom{|S|}{k}$ candidate vocabularies, where $S$ is the set of frequent substrings ($|S| \sim 10^7$ for 1 GB) and $k \sim 3\times10^4$.

## 7. Current Research (as of 2026)

- **Complexity refinement.** Kozma (Saarland/MPI-INF) and collaborators on tightening the greedy constant and parameterized complexity of BPE. Whittington, Bachmann (ETH Zürich) and Pimentel (ETH/Cambridge) on hardness of tokenizer variants including tokenizer inference. *(frontier — verify)*
- **Foundations.** Gastaldi, Cotterell et al., *The Foundations of Tokenization: Statistical and Computational Concerns* (ICLR 2025) — tokenizers as stochastic maps, consistency and injectivity conditions; the most serious attempt at a well-posed objective.
- **Objective replacement.** Superword vocabularies (Liu et al., *SuperBPE*, 2025) drop the word-boundary constraint and report both better compression and better downstream results — evidence the constraint, not the search, is where the losses are.
- **Inference-side optimality.** Uzan, Schmidt, Tanner & Pinter, *Greed Is All You Need: An Evaluation of Tokenizer Inference Methods* (ACL 2024) — decoupling the vocabulary from the segmentation algorithm applied at inference.

## 8. Concrete Next Experiment

**Question:** does closing the compression-optimality gap improve downstream loss at all?

- **Scale.** Corpus: 2 GB of English + code (fixed byte budget). Vocabulary $k = 8{,}192$. Restrict candidates to the $|S| = 2\times10^6$ most frequent substrings of length $\le 16$. Solve direct tokenization by ILP (set-cover-style formulation, CP-SAT or Gurobi, 72 h wall clock) to a **certified LP gap $\le 1\%$**. If exact optimality is out of reach, report the certified bound — the bound is the deliverable.
- **Arms** (4 tokenizers): (A) certified near-optimal direct vocabulary; (B) **control:** greedy BPE, same $k$, same corpus; (C) UnigramLM, same $k$; (D) BPE at $k$ tuned so its *compression* matches arm A, isolating compression from vocabulary identity.
- **Models.** 350M non-embedding parameters, 4 seeds per arm, trained on an identical **byte** budget (~20 GB of text, so token counts differ by design). Embedding parameters equalized by padding to a common $k_{\max}$.
- **Deciding number.** Held-out **bits per byte**, arm A minus arm B, averaged over seeds. Precondition for the test to be meaningful: arm A must beat arm B by $\ge 2\%$ on held-out tokens/byte. Then:
  - $\Delta\text{bpb} \le 0.005$ (below seed noise, $\sigma \approx 0.003$ at this scale) → compression-optimality is not the objective; the NP-hardness result is about a proxy, and §5's methodological block is the real problem.
  - $\Delta\text{bpb} \ge 0.02$ → the gap matters, and approximate-optimal vocabulary construction becomes a live engineering target.

Cost estimate: ~2,000 A100-hours for the 16 pretraining runs plus the ILP. This is the smallest version that has both a certified optimum and a compute-matched control.

## 9. Key References

- **[Foundational]** Storer, J. A. & Szymanski, T. G. *Data Compression via Textual Substitution.* Journal of the ACM 29(4), 1982.
- **[Foundational]** Charikar, M., Lehman, E., Liu, D., Panigrahy, R., Prabhakaran, M., Sahai, A. & Shelat, A. *The Smallest Grammar Problem.* IEEE Transactions on Information Theory 51(7), 2005.
- **[Foundational]** Sennrich, R., Haddow, B. & Birch, A. *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. — arXiv:1508.07909
- **[Foundational]** Kudo, T. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL 2018. — arXiv:1804.10959
- **[SOTA]** Whittington, P., Bachmann, G. & Pimentel, T. *Tokenisation Is NP-Complete.* ACL 2025. — arXiv:2412.15210
- **[SOTA]** Kozma, L. & Voderholzer, J. *Theoretical Analysis of Byte-Pair Encoding.* 2024. — arXiv:2411.08671
- **[SOTA]** Zouhar, V., Meister, C., Gastaldi, J. L., Du, L., Vieira, T., Sachan, M. & Cotterell, R. *A Formal Perspective on Byte-Pair Encoding.* Findings of ACL 2023. — arXiv:2306.16837
- **[SOTA]** Schmidt, C. W., Reddy, V., Zhang, H., Alameddine, A., Uzan, O., Pinter, Y. & Tanner, C. *Tokenization Is More Than Compression.* EMNLP 2024. — arXiv:2402.18376
- **[SOTA]** Zouhar, V., Meister, C., Gastaldi, J. L., Du, L., Sachan, M. & Cotterell, R. *Tokenization and the Noiseless Channel.* ACL 2023.
- **[SOTA]** Cognetta, M., Zouhar, V., Moon, S. & Okazaki, N. *Two Counterexamples to Tokenization and the Noiseless Channel.* LREC-COLING 2024.
- **[SOTA]** Rajaraman, N., Jiao, J. & Ramchandran, K. *Toward a Theory of Tokenization in LLMs.* 2024. — arXiv:2404.08335
- **[Survey]** Gastaldi, J. L., Terilla, J., Malagutti, L., DuSell, B., Vieira, T. & Cotterell, R. *The Foundations of Tokenization: Statistical and Computational Concerns.* ICLR 2025.
- **[Survey]** Gallé, M. *Investigating the Effectiveness of BPE: The Power of Shorter Sequences.* EMNLP-IJCNLP 2019.
- **[Context]** Uzan, O., Schmidt, C. W., Tanner, C. & Pinter, Y. *Greed Is All You Need: An Evaluation of Tokenizer Inference Methods.* ACL 2024.

## 10. Worked Example

**Corpus.** Two word types: `aaa` with count 200, `aab` with count 1. Base token count $= 200\cdot3 + 1\cdot3 = 603$. Budget: **one** new vocabulary item, $k=1$.

**Bottom-up (BPE).** Pair frequencies: $\text{freq}(aa) = 200 \cdot 1 + 1 \cdot 1 = 201$ under non-overlapping left-to-right counting; $\text{freq}(ab) = 1$. Greedy merges `aa`. Segmentations become `aa|a` and `aa|b`:

$$\mathcal{C}(\mu) = 200\cdot 2 + 1\cdot 2 = 402.$$

This is also the *optimal* bottom-up solution at $k=1$: `ab` gives $200\cdot3 + 1\cdot2 = 602$.

**Direct.** Candidate vocabularies of size 1: `aa` → 402; `ab` → 602; `aab` → $200\cdot3+1\cdot1 = 601$; **`aaa`** → $200\cdot1 + 1\cdot3 = 203$.

$$\frac{\mathcal{C}_{\text{bottom-up}}^{\star}}{\mathcal{C}_{\text{direct}}^{\star}} = \frac{402}{203} = 1.98.$$

**What this makes visible.** The factor-2 loss is not a search failure — bottom-up *optimum* is 402. `aaa` is unreachable because building it needs `aa` first, and the budget is one item. The obstruction is the feasible set, set by the formalization, not by the algorithm's greed. Every deployed tokenizer takes the bottom-up (BPE) restriction; the NP-hardness results say both restricted and unrestricted versions are NP-complete, so lifting the restriction buys compression but no tractability.

**Scaling the same instance.** On 1 GB of English with $k = 32{,}000$, the candidate substring set has $|S| \approx 10^7$; the direct problem is a choice among $\binom{10^7}{3.2\times10^4}$ vocabularies. Greedy BPE finishes in minutes. Nobody has certified how far its output sits from the optimum at that scale — and Schmidt et al.'s 350M-parameter ablation says that even if the gap is 5%, we cannot currently predict whether closing it changes bpb by anything at all. That is the live problem, not the complexity class.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*