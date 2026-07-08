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
Telugu the outlier. Weighted-allocation search: **TODO (next step).**

## 6. Paper Hooks / Open Questions
- **H1 — the fertility-parity allocation frontier** *(status: to gate)*: for a fixed joint vocab budget V and
  a language set, what is the minimum achievable cross-lingual fertility gap, and its **irreducible floor**?
  A tool that computes this Pareto frontier for any language set is an effect-agnostic artifact. Connects
  paper20's *tax measurement* to a *fairness-allocation* frontier. **Must gate vs Zheng 2021 / α-sampling
  before treating as novel.**
- **H2 — cap-induced gap** *(observation)*: the `X1<=1.2` cap can *force* a minimum gap (over-serving English
  is mandatory to satisfy it, which pushes X1 down). Is there a clean statement of the gap a per-language cap
  imposes under a shared budget?

## 7. Artifact / Submission
Code: `llm_ws/era-v5/session-02-multilingual-bpe/`. Deliverable: static widget (ratios, stats, score, token
download) hosted on Netlify. Self-score: **TBD after optimization.**

## 8. References
- paper20 token-cost-ledger (author) — `github.com/samyama-ai/token-cost-ledger`
- Conneau et al. 2020, *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R, α-sampling)
- Zheng et al. 2021, *Allocating Large Vocabulary Capacity for Cross-lingual Language Model Pre-training*
- Zouhar et al. 2023, *A Formal Perspective on Byte-Pair Encoding*
- Radford et al. 2019 (GPT-2, byte-level BPE)
