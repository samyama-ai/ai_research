---
id: 01-tokenization/speculative-decoding-vocabulary-mismatch
title: "Speculative Decoding With Mismatched Draft Vocabularies"
topic: 01-tokenization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Speculative Decoding With Mismatched Draft Vocabularies

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/speculative-decoding-vocabulary-mismatch` · **Status:** partially-solved

## 1. Problem Statement

Speculative decoding accelerates autoregressive generation by having a cheap draft model propose $\gamma$ tokens that an expensive target model verifies in one forward pass. The standard algorithm (Leviathan et al., ICML 2023; Chen et al., 2023) assumes draft and target share a vocabulary, so that $p(\cdot\mid x)$ and $q(\cdot\mid x)$ are distributions over the *same* finite set and the accept ratio $\min\{1, q(x)/p(x)\}$ is well defined.

In practice the natural drafter is an off-the-shelf small model with a *different* tokenizer: Llama-3 8B (128,256 tokens) drafting for Qwen2.5-72B (151,936), or GPT-2 drafting for anything. The ratio $q/p$ is then a type error, and the coupling that makes speculative decoding *lossless* has no direct analogue.

Three variants, of different difficulty:

- **Method.** Give an algorithm that, for arbitrary tokenizer pairs, produces samples exactly from the target model's distribution *over strings* while achieving a wall-clock speedup $>1$. Partially solved.
- **Measurement.** Define the acceptance rate when the two models do not segment the same string identically — "tokens accepted" is not comparable across tokenizers, and per-token throughput is not the quantity users care about. Weakly defined.
- **Theory.** Characterize the maximum achievable acceptance (equivalently the minimum number of target calls) as a function of a divergence between two *string-level* distributions induced by different segmentations, and give a matching optimal coupling. Open.

Solving it means: for a given (target, drafter) pair with disjoint-ish vocabularies, a verified-lossless algorithm whose speedup is within a small constant of the shared-vocabulary speedup for a drafter of the same quality and cost.

## 2. Formal Setting

Let $\Sigma^*$ be the set of byte strings. Target model $M_q$ has vocabulary $V_q$ and detokenizer $\sigma_q: V_q^* \to \Sigma^*$; drafter $M_p$ has $V_p, \sigma_p$. Each induces a distribution over strings by pushforward:
$$Q(s) = \sum_{t \in V_q^*: \sigma_q(t)=s} \prod_{i} q(t_i \mid t_{<i}), \qquad P(s) = \sum_{u \in V_p^*: \sigma_p(u)=s} \prod_i p(u_i \mid u_{<i}).$$
**Losslessness** is the requirement that the sampler's output law equals $Q$ on $\Sigma^*$ — not on $V_q^*$. This is the correct target because users consume text.

**Quantities as measured.**

- **Block efficiency** $\tau$: target-model forward passes per generated *character* (or per target token, if reported, but then state it). $\tau^{-1}$ is the only tokenizer-neutral acceptance statistic.
- **Cost ratio** $c$: measured wall-clock time of one drafter step divided by one target step, at the deployed batch size — not FLOP ratio. For a heterogeneous drafter $c$ also includes detokenize/retokenize cost, typically $10^{-5}$–$10^{-4}$ s per block on CPU, non-negligible when $c \approx 0.05$.
- **Leviathan speedup** for shared vocabularies with i.i.d. acceptance probability $\alpha$:
$$S(\alpha,\gamma,c) = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}.$$
- **Canonicity gap.** Let $\mathrm{enc}_q: \Sigma^* \to V_q^*$ be the target's BPE encoder. A drafted string $s$ is *boundary-safe* if for every continuation $s'$, $\mathrm{enc}_q(s s')$ has $\mathrm{enc}_q(s)$ as a prefix. Measure the boundary-unsafe fraction $\beta$ empirically by encoding sampled prefixes.

**Assumptions and their status.**

1. *Acceptance is i.i.d. across positions* — assumed by $S(\alpha,\gamma,c)$; **violated**: acceptance is strongly autocorrelated (bursty), so $S$ overestimates variance-free gain and mis-predicts optimal $\gamma$.
2. *Detokenize–retokenize is an identity on model outputs* — **violated** whenever the target can emit non-canonical token sequences (it can, with nonzero probability) and by byte-fallback/UTF-8 splits.
3. *$V_p \cap V_q$ carries most probability mass* — **often violated** across tokenizer families; string-identical entries may differ in byte-level escaping (`Ġ` vs `▁`) so the naive string intersection undercounts.
4. *Target and draft prompts are the same tokens* — vacuously false here; prompt segmentation differs, which changes $q$ itself.

## 3. State of the Art

**Established (algorithm + proof).** Timor et al., *Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies* (ICML 2025) gives three algorithms with proofs of exactness w.r.t. the target: **String-Level Exact Match (SLEM)**, which drafts a string with $M_p$, retokenizes with $\mathrm{enc}_q$ and verifies token-blockwise, accepting only up to the longest exactly matching prefix; **Token-Level Intersection (TLI)**, which restricts the drafter to $V_p \cap V_q$ and renormalizes, then runs standard speculative sampling; and a string-level rejection variant. TLI is the practically robust one: it never *increases* target calls relative to no speculation, and its correctness does not depend on boundary-safety. These are shipped in HuggingFace `transformers` as Universal Assisted Generation.

**Established (systems).** Self-drafting methods sidestep the problem entirely by construction — Medusa (Cai et al., ICML 2024), EAGLE/EAGLE-2/EAGLE-3 (Li et al., ICML 2024 / EMNLP 2024 / 2025), Lookahead decoding (Fu et al., ICML 2024) — all reuse the target's own vocabulary and hidden states. EAGLE-3 is the empirical SOTA for single-model acceleration ($\approx$4–6.5× reported on MT-Bench-style workloads). This is the reason the mismatched-vocabulary problem is *underexplored*: the highest-throughput answer is to not have a separate drafter.

**Claimed but unablated.** Reported speedups for heterogeneous-vocabulary drafting cluster in the 1.5–2.8× range on summarization and code benchmarks, versus 2–3× for a vocabulary-matched drafter of similar size. These are single-paper benchmark numbers; there is no independent reproduction that isolates *vocabulary mismatch* from drafter quality, because the two are confounded in every published comparison (a different tokenizer means a different pretraining run).

**Adjacent.** Cross-tokenizer distillation — Minixhofer et al., *Zero-Shot Tokenizer Transfer* (NeurIPS 2024) and follow-on approximate-likelihood-matching work (2025) — can manufacture a vocabulary-matched drafter from a mismatched one. This converts an inference problem into a one-off training problem and is the most likely practical resolution *(frontier — verify current results)*.

## 4. What Is Known

- **Optimality under matched vocabularies.** Leviathan et al. (ICML 2023) prove the accept/reject scheme is exact, and single-draft acceptance equals $1 - D_{TV}(p,q)$. SpecTr (Sun et al., NeurIPS 2023) shows the multi-draft version is an optimal-transport problem and gives a近-optimal scheme; the $k$-draft optimum is not achieved by naive independent sampling.
- **Vocabulary overlap is large within families, small across.** Llama-3 (128,256) and Qwen2.5 (151,936) share on the order of half their entries after byte-normalization; GPT-2 (50,257) against Llama-3 shares far less. Reported overlap figures vary by 10+ points depending on whether byte-escaping is normalized — a measurement, not a modeling, discrepancy.
- **TLI's safety property.** Restricting to $V_p \cap V_q$ and renormalizing preserves exactness and cannot make throughput worse than plain autoregressive decoding, up to drafter overhead (Timor et al., ICML 2025).
- **Distillation raises $\alpha$ substantially.** DistillSpec (Zhou et al., ICLR 2024) reports 10–45% additional latency reduction over an undistilled drafter, at matched vocabulary, across T5 and decoder-only models up to 11B — evidence that drafter–target *distributional* alignment, not just size, drives $\alpha$.
- **Retrieval drafting is tokenizer-agnostic in principle.** REST (He et al., NAACL 2024) drafts from a datastore of strings, reporting 1.6–2.4× on 7B/13B models.

## 5. What Is Not Known

- **Theoretically open.** No characterization of the maximum acceptance rate between two string distributions with different segmentations. The matched-vocabulary answer is $1-D_{TV}$; the string-level analogue requires a coupling over $\Sigma^*$ with different action granularities, and no optimal scheme (or impossibility bound) is known. Whether SLEM/TLI are within a constant factor of optimal is unknown.
- **Empirically open.** The clean ablation — one target, and drafters *identical except for tokenizer* — has not been run, because it requires pretraining $k$ matched drafters. At 1B parameters and $\sim$100B tokens this is $\sim$10$^3$ GPU-hours per drafter: affordable, unrun.
- **Methodologically blocked.** "Acceptance rate" has no cross-tokenizer definition. Papers report tokens-per-target-call in the target's vocabulary, which rewards drafters that happen to align with a coarser segmentation. There is no agreed characters-per-target-call convention, so numbers across papers are not comparable.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under confounding**: every measured drop in speedup from a mismatched drafter mixes three causes that cannot be separated with existing checkpoints — (i) genuine coupling loss from segmentation mismatch, (ii) drafter quality, since a differently-tokenized model is a different pretraining run, and (iii) the boundary-safety tax, which forces discarding the last drafted token whenever the target could extend across the drafted string's final boundary. No public model suite varies only the tokenizer, so the effect size of (i) alone has never been measured.

A second, structural obstruction: exactness is defined over strings, but efficient verification wants a per-token accept ratio. The map from a drafted string to target tokens is many-to-one and non-prefix-monotone (BPE is not a prefix code over strings), so the elegant per-token rejection sampler has no drop-in replacement — only conservative prefix matching, which throws away probability mass that a truly optimal coupling would keep.

## 7. Current Research (as of 2026)

- **Intel Labs (Timor, Mamou, Wasserblat and colleagues)** — heterogeneous-vocabulary lossless algorithms and dynamic lookahead $\gamma$; the reference implementation in HuggingFace `transformers`.
- **Cross-tokenizer distillation** — Minixhofer, Vulić, and collaborators; approximate likelihood matching to train a drafter into the target's vocabulary *(frontier — verify)*.
- **Self-drafting** — EAGLE-3 and successors (Li et al.), plus vLLM/SGLang production integrations; the dominant practical direction, which makes the mismatched case a fallback for closed or heavily-quantized targets.
- **Byte-level bridging** — drafting in bytes or in a universal small vocabulary so that any target can verify. Attractive theoretically, penalized by short draft blocks *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question.** How much speedup is lost to segmentation mismatch alone, with drafter quality held fixed?

**Scale.** Train four 1.1B drafters on the identical 100B-token corpus and identical architecture, differing only in tokenizer: (a) the target's own tokenizer, (b) a 32k BPE trained on the same corpus, (c) a 256k multilingual BPE, (d) byte-level. Target: Llama-3.1-70B-Instruct, greedy and $T{=}1$, batch size 1 and 32, on 500 prompts each of MT-Bench, HumanEval, and CNN/DailyMail. Cost: roughly 4,000 A100-hours for pretraining plus 200 for evaluation.

**Control arm.** Drafter (a) — matched vocabulary, same data, same steps. This is the ceiling that isolates tokenizer as the only varying factor.

**Deciding number.** Report **characters generated per target forward pass**, $\tau^{-1}_{\text{char}}$, for each drafter under TLI and SLEM. The decision: the ratio $\tau^{-1}_{\text{char}}(\text{b,c,d}) \,/\, \tau^{-1}_{\text{char}}(\text{a})$. If it exceeds $0.9$, vocabulary mismatch is a second-order effect and the field should stop optimizing for it; if it falls below $0.6$, mismatch is the dominant cost and cross-tokenizer distillation is the right investment. Secondary readout: measured $\beta$, the boundary-unsafe fraction, which should predict the gap.

## 9. Key References

- **[Foundational]** Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML, 2023. — arXiv:2211.17192
- **[Foundational]** C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, J. Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. — arXiv:2302.01318
- **[SOTA]** N. Timor, J. Mamou, D. Korat, M. Berchansky, O. Pereg, M. Wasserblat, T. Galanti, M. Gordon, D. Harel. *Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies.* ICML, 2025. — arXiv:2502.05202
- **[SOTA]** Y. Li, F. Wei, C. Zhang, H. Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* ICML, 2024. — arXiv:2401.15077
- **[Theory]** Z. Sun, A. T. Suresh, J. H. Ro, A. Beirami, H. Jain, F. Yu. *SpecTr: Fast Speculative Decoding via Optimal Transport.* NeurIPS, 2023. — arXiv:2310.15141
- **[Method]** Y. Zhou, K. Lyu, A. S. Rawat, A. K. Menon, A. Rostamizadeh, S. Kumar, J.-F. Kagy, R. Agarwal. *DistillSpec: Improving Speculative Decoding via Knowledge Distillation.* ICLR, 2024. — arXiv:2310.08461
- **[Method]** T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, T. Dao. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* ICML, 2024. — arXiv:2401.10774
- **[Method]** Z. He, Z. Zhong, T. Cai, J. D. Lee, D. He. *REST: Retrieval-Based Speculative Decoding.* NAACL, 2024. — arXiv:2311.08252
- **[Adjacent]** B. Minixhofer, E. M. Ponti, I. Vulić. *Zero-Shot Tokenizer Transfer.* NeurIPS, 2024. — arXiv:2405.07883
- **[Survey]** H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, Z. Sui. *Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding.* ACL Findings, 2024. — arXiv:2401.07851

## 10. Worked Example

Target Llama-3.1-70B ($V_q$ = 128,256), drafter GPT-2-large ($V_p$ = 50,257). Measured drafter cost ratio $c = 0.10$; block $\gamma = 5$. Suppose the matched-vocabulary acceptance would be $\alpha = 0.80$.

Matched baseline:
$$S = \frac{1-0.8^{6}}{(1-0.8)(5\cdot 0.10+1)} = \frac{0.7379}{0.2 \times 1.5} = 2.46\times,$$
with expected accepted tokens per block $= 0.7379/0.2 = 3.69$.

Now the mismatch bites at the boundary. GPT-2 drafts the string `"  total = 12345"`. Llama-3 encodes multi-digit numbers as *individual digit tokens* while GPT-2 merges `123`+`45`. Under SLEM the drafted string is retokenized with $\mathrm{enc}_q$ and matched prefix-wise, and the final chunk is boundary-unsafe: the target's next token could merge into it. The safe rule is to discard the trailing token of every block. That gives $3.69 - 1 = 2.69$ tokens per block at unchanged cost:
$$S' = \frac{2.69}{1.5} = 1.79\times.$$
A 27% loss of the speedup, from *segmentation alone*, before any degradation in $\alpha$.

TLI avoids the trim but pays elsewhere: after byte-normalizing `Ġ` against `▁`, suppose $V_p \cap V_q$ carries 0.72 of GPT-2's probability mass on this workload. Renormalizing onto the intersection perturbs the drafter, dropping $\alpha$ from 0.80 to roughly 0.68, giving $S = (1-0.68^6)/(0.32 \times 1.5) = 1.86\times$.

The obstruction is visible in the arithmetic: both algorithms land near $1.8\times$ against a $2.46\times$ ceiling, but for *different reasons* — a discarded token versus a truncated support. And neither number can be attributed to tokenization with confidence, because GPT-2-large is also simply a worse model than a hypothetical Llama-tokenized drafter of the same size. That is exactly the confound Section 8's control arm removes.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*