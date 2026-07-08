---
id: 01-tokenization/multilingual-bpe-fairness
title: "Cross-Lingual Fertility Parity under a Joint BPE Vocab Budget"
topic: 01-tokenization
status: in-progress
source: "ERA V5 · Session 2 assignment (multilingual BPE tokenizer on India Wikipedia)"
due: 2026-07-11
score_metric: "1000 / (X4 - X1), Xl = tokens/word per language, English X1 <= 1.2"
first_added: 2026-07
last_update: 2026-07
provenance: worked-from-first-principles
---

# Cross-Lingual Fertility Parity under a Joint BPE Vocab Budget

> **Topic:** Tokenization · **Source:** ERA V5 · Session 2 · **Status:** in-progress · **Metric:** `1000/(X4-X1)`

## 1. Problem Statement
Take the **India** Wikipedia article in English, Hindi, Telugu, and one more language. Design **one BPE
tokenizer with a joint vocabulary of 10,000 tokens** for all languages such that, per language,
`Xl = (total tokens to encode that page) / (its word count)` with **English X1 <= 1.2**. Sort the four
ratios; **score = 1000 / (X4 - X1)** where X4 is the largest ratio and X1 the smallest. Deliverable: a
hosted **widget** showing the ratios/stats/score and offering the token list for download. Graders
**re-run the tokenizer** to confirm the numbers.

## 2. First-Principles Formalization
`Xl` is **fertility** — tokens per word. (Checked: if the denominator were unique word *types* or a fixed
5000, English's ratio would be ~2.3-3.1, impossible under the <=1.2 cap; only tokens-per-running-word lands
English near 1.1-1.2. The "say 5000 words" in the prompt is illustrative — the real English page has 10,121
words.) **The word-count definition must match the grader's** (whitespace split assumed) — the top scoring
risk.

Score `1000/(X4-X1)` is **maximized by minimizing the fertility *spread*** across the four languages — i.e.
this is a **cross-lingual fertility *parity*** objective under a fixed joint vocab budget, with a hard cap
`X1(en) <= 1.2`. It decomposes into a **vocab-allocation** problem: pull the worst language (Telugu —
Brahmic, agglutinative) *down* by giving it more merges; let English *rise* toward its 1.2 cap (over-serving
the easy language only widens the gap). Strategic lever: the 4th language should land *between* en and te and
extend neither end (a Latin-script pick like Spanish is score-optimal; a harder Brahmic would raise X4).

## 3. Prior Art / SOTA
- **Fertility / the multilingual tokenization tax** — author's own [paper20 token-cost-ledger](https://github.com/samyama-ai/token-cost-ledger) (removable coding tax vs irreducible G2P-entropy floor); Rust 2021 (fertility), Zouhar et al. 2023 (tokenization info-theory), Ács 2019.
- **Balanced multilingual vocab** — temperature/α exponential smoothing of language sampling (XLM-R, Conneau et al. 2020); vocabulary allocation across languages (Zheng et al. 2021, "Allocating Large Vocabulary Capacity").
- **Char vs byte pre-tokenization** — byte-level BPE (GPT-2, Radford 2019) is script-unfair to Brahmic (each char = 3 bytes); char/unicode-level BPE is required here (measured below).

## 4. Approach
1. **Char/unicode-level BPE** (HF `tokenizers`, Whitespace pre-tokenizer) — not byte-level.
2. **Weighted training corpus**: oversample per language to steer merge allocation. Search sampling weights
   `w_en, w_hi, w_te, w_es` to **minimize (X4 - X1)** subject to `X1(en) <= 1.2` at fixed 10k vocab.
3. Telugu's page is tiny (2,511 words) — oversampling it is essential, not optional.
4. **Honesty rail:** widget numbers must equal the real tokenizer applied to the real pages (graders re-run).

## 5. Results
Baselines (joint 10k, uniform corpus):

| training | en | hi | te | es | gap | score | en<=1.2 |
|---|---|---|---|---|---|---|---|
| char-level, uniform | 1.431 | 1.315 | 2.045 | 1.347 | 0.730 | 1370 | ❌ 1.431 |
| byte-level, uniform | 1.398 | 3.533 | 5.767 | 1.316 | 4.451 | 225 | ❌ |

Lessons: byte-level is catastrophic for Indic; uniform char-level still violates the English cap *and* leaves
Telugu the outlier. **Pre-tokenizer fix:** use `WhitespaceSplit` (whitespace-only), NOT `Whitespace`
(which also splits punctuation and inflates fertility) — must match the grader's `\S+` word count.

**Weighted-allocation search (char-level, WhitespaceSplit, joint 10k):**
- Natural balance point (no cap) = all four ≈ **1.4**: `en 1.42 / hi 1.34 / te 1.35 / es 1.38`, gap **0.078** — but English 1.42 violates its 1.2 cap.
- **The cap forces English below the balance point**, which requires giving English ~40%+ of the weight share; that starves the others. English is a *step function* (only 3,774 unique words → snaps from 1.42 to ~1.0; cannot sit at 1.19).
- **English and Telugu directly compete for the 10k budget:** feed Telugu to ≈1.0 and English pops back to 1.36 (infeasible); satisfy English (≤1.2) and Telugu is stuck at ≈1.95. Telugu's tiny 2,511-word page is the fragile bottleneck.
- **Best FEASIBLE (en≤1.2), joint corpus-weighting:** weights `en20 hi8 te16 es8` → `en 1.142 / hi 1.471 / te 1.949 / es 1.488`, gap **0.807**, **score ≈ 1,239**.
- **Diagnostic:** at V=20k the tension evaporates (all ≈1.0–1.07, gap 0.072). **The gap is budget-scarcity-induced.**

**Next-session unlock:** *script-disjoint vocab allocation* — train per-language BPE with explicit budgets and
union the vocabularies. English (Latin) and Telugu (Brahmic) live in disjoint code-point spaces, so separate
budgets stop them competing — should satisfy en≤1.2 AND pull Telugu down, beating ~1,239. (Also the H1 artifact.)

## 6. Paper Hooks / Open Questions
- **H1 — the fertility-parity allocation frontier** *(status: to gate; now with measured evidence)*: for a
  fixed joint vocab budget V, the min cross-lingual fertility gap is **budget-governed** — measured here it
  falls from 0.81 (V=10k, capped) toward 0.07 (V=20k). There is a **critical budget** above which parity is
  ~free. A tool that computes this frontier for any language set is an effect-agnostic artifact linking
  paper20's *tax* to a *fairness-allocation* frontier. **Gate vs Zheng 2021 (vocab-capacity allocation) /
  α-sampling / paper20 before treating as novel.**
- **H2 — the cap-induced gap** *(now measured)*: a per-language fertility cap below the natural balance point
  *manufactures* the gap — satisfying it forces a budget reallocation that starves other languages. Cleanly
  visible here as a **two-language competition** (English cap vs Telugu bottleneck) that is unsatisfiable at
  10k but trivial at 20k. Candidate clean statement: the min gap under a cap `c` and budget `V` is
  `max(0, balance(V) - c)` plus a starvation term. **Gate before claiming.**
- **H3 — script-disjoint allocation beats corpus-temperature** *(to test next session)*: because scripts
  occupy disjoint code-point spaces, per-script vocab budgets decouple languages that joint corpus-weighting
  couples. If this beats α-sampling on the parity frontier, that's a concrete, testable contribution.

## 7. Artifact / Submission
Code: `llm_ws/era-v5/session-02-multilingual-bpe/` (`train.py`, `data/`, saved `tokenizer.json` / `tokens.txt`
/ `results.json`). Deliverable: static widget (ratios, stats, score, token download) on Netlify — **not yet
built**. Current honest self-score: **≈ 1,239** (joint corpus-weighting, feasible). Resubmission allowed →
improve via H3 (script-disjoint allocation) before final submit. **Due 2026-07-11.**

## 8. References
- paper20 token-cost-ledger (author) — `github.com/samyama-ai/token-cost-ledger`
- Conneau et al. 2020, *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R, α-sampling)
- Zheng et al. 2021, *Allocating Large Vocabulary Capacity for Cross-lingual Language Model Pre-training*
- Zouhar et al. 2023, *A Formal Perspective on Byte-Pair Encoding*
- Radford et al. 2019 (GPT-2, byte-level BPE)
