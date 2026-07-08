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

**H3 — script-disjoint allocation (CONFIRMED, score 2,336):** train per *script group* and union the vocabs.
Key subtlety discovered: naive **per-language** union fails for **same-script** languages — English and
Spanish are both Latin, so unioning two Latin merge lists lets whichever has priority hijack the other
(measured: en-priority → es 1.95; es-priority → en 2.05). Correct structure = 3 *script* groups: joint
**Latin** (en oversampled + es), disjoint **Devanagari** (hi), disjoint **Telugu** (te), unioned losslessly.
Result: `en 1.180 / hi 1.608 / te 1.577 / es 1.607`, gap **0.428**, **score 2,336** (1.9×). This independently
reproduces **Chung et al. 2020** (language-clustered vocabularies) and **XLM-V** (de-emphasize cross-script sharing).

**H4 — whole-word budget reclamation (CONFIRMED, score 2,430):** diagnosed that the H3 tokenizer wastes
**~17% of its 10k slots** on intermediate BPE merges that never surface as final tokens, while whole-word
coverage is low (hi 21%, es 17%). Reclaim them: train the union at a *reduced* budget and spend the freed
~650 slots on explicit whole-word merges for the highest-fertility languages. Pulls the max cluster from
~1.61 → 1.59: `en 1.180 / hi 1.582 / te 1.591 / es 1.577`, gap **0.411**, **score 2,430** (2×). This is the
**Picky BPE** (Chizhov 2024) / **BPE-knockout** (Bauwens 2024) idea applied to a parity objective. Still one
valid BPE tokenizer (graders re-run).

**Parity-aware BPE (principled, `experiments/parity_bpe.py`) — the real answer to the problem.**
Implemented the Foroutan et al. (ACL 2026) idea from scratch: at each merge step, help the currently
*worst-compressed* language (merge its most valuable pair) instead of the globally most-frequent pair.
One ordinary BPE tokenizer; verified against HuggingFace (max Δ 0.0000).
- **Pure parity (no cap): PERFECT fairness — all four converge to en=hi=te=es=1.390, gap 0.000.** They
  move in lockstep as |V| grows (e.g. |V|=6000 → all 1.670). Nobody pays a token tax.
- **With en≤1.2 cap: gap 0.367** (en 1.197 / hi=te=es 1.564 — the other three perfectly balanced), which
  beats the post-hoc H4 (0.411) AND is principled, not a bolt-on.
- **Insight (H2, now sharp):** the English cap *manufactures* the inequity. Remove it → perfect parity,
  and total tokenization is *cheaper* (forcing English to 1.2 wastes shared budget, pushing everyone else
  up to 1.56). The assignment scores you on mitigating an inequity its own constraint creates.

**Is the parity fertility a floor? (info-theoretic, `experiments/parity_bpe.py` V-sweep + Rényi).**
- **No — 1.390 is just the V=10k operating point.** Pure-parity common fertility keeps dropping with budget
  (V=4k→1.90, 8k→1.51, 10k→1.39, 14k→1.24, 20k→1.05) and **parity holds at every budget** (spread ~0.0002
  throughout). The irreducible floor for whitespace-word units is **1.0** (each word = one token), reached at
  **V ≈ 11,416 = the number of unique word types** across the four pages. Below that, the common fertility is
  set by Zipf coverage of the word distribution; going *below* 1.0 needs superword (cross-whitespace) tokens.
- **Fertility parity ≠ information parity.** At equal fertility (hi=te=es≈1.58), Zouhar's Rényi efficiency
  (α=2.5, correlates w/ downstream) is NOT equal — te 0.70 / hi 0.54 / es 0.49 / en 0.48 — and bits-per-word
  differ sharply (te 16.5 / es 14.8 / hi 14.3 / en 12.1). Equalizing token *count* leaves the *information*
  cost unequal, and actually over-serves Telugu's tiny page (near-full word coverage → high efficiency) while
  under-serving the large en/es pages. **Observation, gate before claiming** vs Zouhar 2023 (noiseless channel)
  / Arnett 2025 — likely known that length≠quality, but the *inversion* under fertility-parity is worth a check.

**Does parity hold at scale / off the India pages? (`experiments/scale.py`, 14 langs fetched).**
- **Yes, robustly.** Pure parity-aware BPE over the *same* article in **12 whitespace languages / 5 scripts**
  (Latin ×6, Cyrillic, Arabic, Brahmic ×4) holds all twelve to spread **0.0014** @ V=10k (all 2.109–2.110);
  no language starved — Indonesian (2,050 words) = German (23,420 words) to 3 decimals. Common fertility is
  budget-governed: N=4→1.39, 6→1.56, 12→2.11; 12@20k→1.71.
- **Scope limit — the metric breaks for CJK.** Chinese ~57 chars / "\\S+ word", Japanese ~55: scriptio-continua
  scripts have almost no whitespace, so tokens-per-\\S+-word is meaningless. The assignment's fairness framework
  silently assumes whitespace-delimited languages; a char-normalized denominator is needed for zh/ja/th.

**VoCap-style allocation (tested):** a marginal-utility water-filling allocator (attack-the-worst-language,
after Zheng 2021) independently converges to the same base budget as the hand grid (Lat ~6250 / hi ~1000 /
te ~2050) → score **2,429 ≈ 2,430**. Confirms we sit *on* the allocation frontier; the reclamation step, not the
base split, is the lever. (`experiments/vocap.py`.)

**Ceiling (measured):** within "one BPE tokenizer, 10k, en≤1.2, these 4 pages", the frontier is ~2,400 safe /
~2,600 at the cap-edge (en 1.198, risky). Normalization (NFC/NFD/NFKC) does **not** help. The gap is
structural: en is forced low by the cap while hi/te/es cluster at ~1.6 (budget-limited). A larger jump needs a
different tokenizer family — **Unigram LM** (Bostrom & Durrett 2020: BPE is suboptimal) — which the "must be
BPE" assignment forbids.

## 6. Paper Hooks / Open Questions — GATED 2026-07-08 (adversarial web check done)
**Verdict: no GREEN. Every angle is RED after gating — the field converged on this exact problem in 2024–2026.**
The honest value of this session: independently *reproducing* SOTA (a good understanding check), a clean
pedagogical artifact, and a cataloged prior-art map. Honesty-over-reach per the runbook — negatives are fine.

- **H1 — parity is budget-governed** → **RED.** [[arnett2025inequities|Arnett et al., NeurIPS 2025]] trains
  ~7,000 monolingual tokenizers over 97 languages and maps how token-premium disparity varies with vocab size
  and pre-tokenizer — the frontier we observed, done comprehensively.
- **H2 — a per-language cap manufactures the gap** → not a paper. It is an artifact of *this assignment's*
  artificial hard cap + fixed joint budget, not a general phenomenon.
- **H3 — script-disjoint allocation** → **RED.** [[chung2020clustered|Chung et al. 2020]] (language-clustered
  vocabularies) and [[liang2023xlmv|XLM-V]] already combine per-cluster vocabularies / de-emphasize cross-script
  sharing. We reinvented it.
- **H4 — parity-aware reclamation / attack-the-worst-language** → **RED.**
  [[foroutan2026parityaware|Parity-Aware BPE, Foroutan et al., ACL 2026]] is our exact objective **and** method:
  at every merge it maximizes the compression gain of the currently worst-compressed language (= our water-fill /
  VoCap-style "attack the worst"), integrated into the merge objective. Our H3+H4+VoCap post-hoc pipeline is an
  approximation of it. Slot-reclamation itself = [[chizhov2024pickybpe|Picky BPE]] / [[bauwens2024bpeknockout|BPE-knockout]].
- **The one narrow YELLOW (not worth a paper alone):** the *fixed-hard-joint-budget + hard per-language cap*
  regime differs from prior parity work, which assumes *per-language* budgets (Arnett) or trades global
  compression (Parity-Aware BPE). Reclamation as a parity lever under a hard joint cap is mildly distinct — but
  incremental and pre-empted in spirit. Not pursued.

**If we want to actually push past 2,430:** adopt Parity-Aware BPE's integrated merge objective (they have
[code](https://github.com/swiss-ai/parity-aware-bpe)) instead of our post-hoc union+reclamation — the SOTA method,
not a novel one.

## 7. Artifact / Submission
Code: `llm_ws/era-v5/session-02-multilingual-bpe/` — progression `train.py` (naive, 1,239) → `train_h3.py`
(script-disjoint union, 2,336) → `train_h4.py` (+ whole-word reclamation, 2,430) → **`train_h5.py`
(parity-aware BPE, principled, 2,511 — DEPLOYED)**. `experiments/parity_bpe.py` = the pure-parity (gap 0)
version; `experiments/{optimize,augment,vocap}.py` = sweeps. `build_widget.py` publishes into `session-2/`.
**Deployed:** https://era-v5.netlify.app/session-2/ — shows ratios/stats/score, downloads
tokenizer.json/tokens.txt, and **re-tokenizes all four pages live in-browser**. H5: en 1.182 / hi=te=es 1.580,
gap 0.398. Verified exact vs HuggingFace. **Due 2026-07-11.**

## 8. References
Full BibTeX (with per-paper reuse verdicts): [`references.bib`](./references.bib) — also imported into **Zotero**.

Reusable techniques (verdict):
- **Chizhov et al. 2024**, *BPE Gets Picky* (Picky BPE) — remove wasted intermediate tokens → **powers H4** (2,336→2,430).
- **Bauwens & Delobelle 2024**, *BPE-knockout* — post-hoc merge pruning with reconnection (same family as H4).
- **Zheng et al. 2021**, *Allocating Large Vocabulary Capacity* (VoCap, code) — principled per-script budgets (could replace our grid).
- **Bostrom & Durrett 2020**, *BPE is Suboptimal* — Unigram LM lowers fertility; the big lever IF BPE constraint relaxed.

Validates our approach:
- **Chung et al. 2020**, *Language-Clustered Vocabularies* (EMNLP) — combine per-cluster vocabs = our script-disjoint union.
- **Liang et al. 2023**, *XLM-V* — de-emphasize cross-script token sharing at 1M-vocab scale.
- **Petrov et al. 2023**, *Tokenizers Introduce Unfairness Between Languages* (NeurIPS) — frames the fairness objective.
- **Al Kautsar & Koto 2025**, *Parallel Tokenizers* — monolingual-then-align; "naturally improves fertility balance".

Background: paper20 token-cost-ledger (author, `github.com/samyama-ai/token-cost-ledger`); Conneau et al. 2020
(XLM-R α-sampling); Zouhar et al. 2023; Radford et al. 2019 (GPT-2 byte-level BPE).
