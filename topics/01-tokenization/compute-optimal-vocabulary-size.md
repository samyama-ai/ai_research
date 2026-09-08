---
id: 01-tokenization/compute-optimal-vocabulary-size
title: "Compute-Optimal Vocabulary Size"
topic: 01-tokenization
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compute-Optimal Vocabulary Size

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/compute-optimal-vocabulary-size` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed training compute budget $C$ (FLOPs), a corpus, an architecture family, and a tokenizer induction algorithm (BPE, Unigram LM, WordPiece), choose the vocabulary size $V$ that minimizes end-of-training loss measured in a vocabulary-independent unit. Vocabulary is not a free parameter: raising $V$ spends parameters on embedding and unembedding matrices, but shortens the token sequence, so the same corpus costs fewer forward passes. The two effects trade off, and the optimum is interior.

Three variants, in increasing difficulty:

- **Measurement.** Is there a unit in which a $V=32{,}000$ model and a $V=256{,}000$ model can be compared at all? Token-level cross-entropy is not it — it is denominated in different alphabets. Bits-per-byte is the standard answer; whether it is the *right* answer for downstream utility is contested.
- **Method.** Given a target non-vocabulary parameter count $N_{nv}$ or budget $C$, predict $V_{opt}$ without training at that scale. Solved-in-form by Tao et al. (2024); the fitted exponents are not independently reproduced.
- **Theory.** Derive the exponent relating $V_{opt}$ to $N_{nv}$ from properties of the text distribution (Zipf/Heaps behaviour, type-token growth) rather than fitting it. Open.

Solving it means: a predictor $\hat V(C, \text{corpus})$ whose held-out error in bits-per-byte at $\ge 10^{22}$ FLOPs is smaller than the loss spread between adjacent standard vocabulary choices ($32$K, $64$K, $128$K, $256$K).

## 2. Formal Setting

Let the corpus be a byte string of length $B$. A tokenizer $\tau_V$ with vocabulary $\mathcal{V}$, $|\mathcal{V}|=V$, maps it to $T(V)$ tokens. Define the **fertility** (bytes per token, measured directly by encoding a held-out shard):

$$ f(V) \;=\; B / T(V), \qquad f \text{ increasing, concave in } \log V .$$

Split parameters into non-vocabulary and vocabulary parts, with model width $d$:

$$ N = N_{nv} + N_{v}, \qquad N_v = \alpha \, d \, V,\ \alpha \in \{1,2\} \ (\text{tied / untied embeddings}).$$

Compute, using the standard $6N$ per-token approximation (Kaplan et al. 2020):

$$ C \;=\; 6\,(N_{nv} + \alpha d V)\, D , \qquad D = \text{training tokens} = B_{\text{train}}/f(V).$$

The comparable objective is loss **per byte**, obtained from measured token-level cross-entropy $\ell_{\text{tok}}$ (nats/token) on a held-out shard:

$$ \ell_{\text{byte}}(V) \;=\; \ell_{\text{tok}}(V) \,/\, f(V), \qquad \text{bits-per-byte} = \ell_{\text{byte}}/\ln 2 .$$

The decision problem:

$$ V_{opt}(C) \;=\; \arg\min_{V} \; \ell_{\text{byte}}\big(V;\, N_{nv}, D\big) \quad \text{s.t.}\quad 6(N_{nv}+\alpha d V)D \le C .$$

Tao et al. (2024) posit a Chinchilla-style decomposition with a vocabulary-dependent term and predict a power law $V_{opt} \propto N_{nv}^{\gamma}$, fitting $\gamma \approx 0.83$ — sublinear, so the vocabulary share of parameters shrinks with scale, but the absolute optimum grows.

**Assumptions, and which are violated.**
- *$6N$ FLOPs accounting.* Violated: the softmax/logit matmul over $V$ and the embedding lookup have very different arithmetic intensity and memory traffic. At $V=256$K the output projection is a large fraction of measured wall-clock but is counted as ordinary FLOPs. Compute-optimal $\ne$ time-optimal.
- *Fertility independent of the model.* Holds by construction, but $f$ is corpus-dependent; a tokenizer fit on web English has different $f$ on code and on Telugu (Petrov et al. 2023).
- *Loss-per-byte is monotone in downstream quality.* Contested — Goldman et al. (2024) find compression correlates with performance only within limited ranges.
- *Data is not repeated and is not scarce.* Under repetition, shorter sequences change the effective epoch count, and the trade-off shifts.
- *Fixed depth/width.* $N_v$ depends on $d$, so vocabulary and aspect ratio are entangled; papers usually fix the shape.

## 3. State of the Art

**Method SOTA — established in form.** Tao, Guo, Duan, Wang, Wang, Chen, *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies* (NeurIPS 2024, arXiv:2407.13623). Three independent estimators — IsoFLOPs curves, derivative of loss w.r.t. FLOPs, and a parametric fit — agree that $V_{opt}$ grows as a power of $N_{nv}$ with exponent below 1. The *agreement of three estimators on one grid* is the strongest result in the area.

**Claimed but unablated.** The headline extrapolation — Llama-2-70B "should" have used $\approx 216$K rather than $32$K — is an extrapolation several orders of magnitude beyond the fitted grid (models up to a few B parameters). No 70B-scale control has been run. The fitted $\gamma$ has not been reproduced by an independent group on a different corpus.

**Benchmark-number-only results.** Tao et al.'s empirical confirmation is a 3B-parameter pair at matched FLOPs, reporting ARC-Challenge $29.1 \to 32.0$ when vocabulary is raised from 4,096 to the predicted optimum. That is a single benchmark delta on a single seed — indicative, not an ablation.

**Systems/empirical SOTA.** Huang et al., *Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling* (2025, arXiv:2501.16975) decouple input and output vocabulary and scale the *input* $n$-gram vocabulary into the millions, reporting log-linear loss improvement and a small model matching a baseline roughly $2\times$ its size. This breaks the single-$V$ formulation: input and output vocabularies have different cost curves. *(frontier — verify at scale.)*

**Counterpoint SOTA.** Ali et al., *Tokenizer Choice For LLM Training: Negligible or Crucial?* (Findings of NAACL 2024, arXiv:2310.08754) find tokenizer choice materially changes downstream results in multilingual settings; XLM-V (Liang et al., EMNLP 2023, arXiv:2301.10472) pushed to a 1M vocabulary for multilingual MLM with gains — but at encoder scale, not autoregressive compute-optimal scale.

## 4. What Is Known

- **The optimum is interior and grows with scale.** Established on grids up to a few billion non-vocabulary parameters (Tao et al. 2024). Fitted exponent $\gamma \approx 0.83$.
- **Standard vocabularies were undersized relative to that fit.** GPT-2: 50,257; Llama-1/2: 32,000; Llama-3: 128,256; Gemma: 256,000; Qwen-2.5: ~151,000. The industry drift from 32K to 128–256K between 2023 and 2025 is consistent with the prediction, though not caused by it in any documented case.
- **Fertility gains are strongly diminishing.** Doubling $V$ past $\sim$100K buys single-digit percent bytes-per-token on English web text, while embedding parameters grow linearly.
- **Loss is flat near the optimum.** Reported IsoFLOP curves are shallow over roughly a factor of 2–4 in $V$; the penalty for a $2\times$ error is small relative to seed noise at small scale.
- **Compression alone does not predict quality.** Schmidt et al. (EMNLP 2024, arXiv:2402.18376) and Goldman et al. (ACL Findings 2024, arXiv:2403.06265) both show corpus-level compression is an incomplete proxy.
- **Byte-level models are viable but not compute-optimal at matched budgets** under standard architectures; ByT5 (Xue et al., TACL 2022) pays a large sequence-length cost. Byte Latent Transformer (Pagnoni et al., 2024, arXiv:2412.09871) reports matched scaling with dynamic patching — an existence proof that $V$ can be replaced by a learned segmentation, not that it should be.

## 5. What Is Not Known

- **Empirically open.** Does $V_{opt} \propto N_{nv}^{0.83}$ hold at $10^{23}$–$10^{25}$ FLOPs? The experiment is runnable by any frontier lab and has not been published. Nobody has published an IsoFLOP vocabulary sweep at $\ge 30$B parameters.
- **Empirically open.** Corpus dependence: how $\gamma$ and the constant shift for code-heavy, multilingual, or math corpora. No published cross-corpus refit.
- **Theoretically open.** No derivation of $\gamma$ from the text distribution. Heaps' law ($\text{types} \sim B^{\beta}$) plus Zipfian frequency ought to constrain both $f(V)$ and the entropy term, but no theorem connects them to the compute-optimal exponent.
- **Theoretically open.** Whether the optimum is unique. The objective is not known to be quasi-convex in $\log V$; multiple local minima (e.g. character-level and word-level basins) are not excluded.
- **Methodologically blocked.** The right unit. Bits-per-byte equalizes the alphabet but not the *difficulty distribution*: a large vocabulary moves probability mass onto rare types where the model is worst, and per-byte averaging hides that. There is no agreed vocabulary-invariant metric that predicts downstream benchmark deltas.
- **Methodologically blocked.** Input/output vocabulary decoupling makes $V$ a two-dimensional object; no scaling law covers the pair $(V_{\text{in}}, V_{\text{out}})$.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by extrapolation range**. Changing $V$ changes four things at once: parameter count, sequence length, the loss's support, and the effective number of gradient updates per byte. Fixing FLOPs holds only the product constant, not the factors. So the measured $\Delta$ loss between two vocabularies cannot be attributed to vocabulary alone without an ablation grid that costs the same as the sweep itself.

Second obstruction: **the objective is flat near the optimum**, so distinguishing $V_{opt}=128$K from $V_{opt}=256$K requires loss resolution below seed-to-seed variance. At small scale the variance dominates; at large scale, single runs cost millions of dollars and nobody runs seeds. The fit is therefore made where it is cheap and applied where it matters — a 2–3 order-of-magnitude extrapolation, exactly the regime where Kaplan-vs-Chinchilla went wrong once already (Hoffmann et al. 2022).

Third: **evaluation that does not measure what it names.** "Compute-optimal" is defined on training FLOPs, but the practical cost of a large vocabulary is inference memory (the unembedding matrix and logit tensor) and serving latency. A vocabulary that is training-compute-optimal may be inference-pessimal.

## 7. Current Research (as of 2026)

- **Refitting vocabulary scaling laws on non-English and code corpora.** Motivated by Petrov et al. (2023) on cross-lingual token cost. *(frontier — verify.)*
- **Decoupled input/output vocabularies** — ByteDance Seed's Over-Tokenized line (arXiv:2501.16975); the claim that input vocabulary scales log-linearly essentially for free is the most consequential open claim in the topic. *(frontier — verify.)*
- **Tokenizer-free / dynamic segmentation** — BLT (Meta, arXiv:2412.09871), hierarchical byte models. If these hold at scale, $V_{opt}$ becomes a question about patch-entropy thresholds rather than a discrete vocabulary.
- **Post-hoc vocabulary transplant / adaptation** — Dagan et al. (ICML 2024, arXiv:2402.01035) on domain adaptation of tokenizers, and the broader practice of extending vocabularies for new languages after pretraining.
- **Softmax-cost mitigations** (approximate or factorized output layers) that would flatten the $N_v$ term and push $V_{opt}$ upward. Largely a systems literature, not yet joined to the scaling-law literature.

## 8. Concrete Next Experiment

**Question:** does the fitted exponent $\gamma \approx 0.83$ survive one order of magnitude of extrapolation?

**Scale.** Train an IsoFLOP grid at $C = 6\times10^{21}$ FLOPs (roughly a 7B model at Chinchilla ratio) on a fixed 1T-token English+code corpus. Vocabulary arm: $V \in \{32\text{K}, 64\text{K}, 128\text{K}, 256\text{K}, 512\text{K}\}$, BPE, retrained per $V$ on the same shard. For each $V$, adjust $N_{nv}$ downward so total FLOPs match exactly. Two seeds per cell — 10 runs. Then repeat at $C = 6\times10^{20}$ (10 more, cheap) to give a two-point extrapolation check.

**Control arm.** The prediction from Tao et al.'s published fit, extrapolated to this $C$, plus a *parameter-matched* control (same $N$ total, varying $V$, FLOPs allowed to differ) to separate the parameter-allocation effect from the sequence-length effect.

**The deciding number.** The location of the IsoFLOP minimum in $\log_2 V$ at $6\times10^{21}$ FLOPs, measured in bits-per-byte on held-out text, with seed variance reported. If the measured argmin lies within $\pm 0.5$ in $\log_2 V$ of the extrapolated prediction, the law transfers; if it is off by $\ge 1$ (a factor of 2), the published exponent is a small-scale artifact. Secondary readout: the bits-per-byte gap between the argmin and $V=32$K. If that gap is $< 0.005$ bpb — comparable to seed noise — then the whole question is practically moot at this scale and the field should stop optimizing $V$ and start optimizing the segmentation algorithm.

Cost estimate: ~$2\times10^{22}$ FLOPs total, a few hundred H100-days. This is the smallest experiment that touches the actual disputed regime.

## 9. Key References

- **[Foundational]** Rico Sennrich, Barry Haddow, Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. — arXiv:1508.07909
- **[Foundational]** Taku Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL, 2018. — arXiv:1804.10959
- **[Foundational]** Jared Kaplan et al. *Scaling Laws for Neural Language Models.* 2020. — arXiv:2001.08361
- **[Foundational]** Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models.* NeurIPS, 2022. — arXiv:2203.15556
- **[SOTA]** Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, Ngai Wong. *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS, 2024. — arXiv:2407.13623
- **[SOTA]** Hongzhi Huang et al. *Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling.* 2025. — arXiv:2501.16975
- **[Empirical]** Mehdi Ali et al. *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL, 2024. — arXiv:2310.08754
- **[Empirical]** Craig W. Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omri Uzan, Yuval Pinter, Chris Tanner. *Tokenization Is More Than Compression.* EMNLP, 2024. — arXiv:2402.18376
- **[Empirical]** Omer Goldman, Avi Caciularu, Matan Eyal, Kris Cao, Idan Szpektor, Reut Tsarfaty. *Unpacking Tokenization: Evaluating Text Compression and its Correlation with Model Performance.* Findings of ACL, 2024. — arXiv:2403.06265
- **[Empirical]** Davis Liang et al. *XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models.* EMNLP, 2023. — arXiv:2301.10472
- **[Empirical]** Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, Adel Bibi. *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS, 2023. — arXiv:2305.15425
- **[Alternative]** Artidoro Pagnoni et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* 2024. — arXiv:2412.09871
- **[Alternative]** Linting Xue et al. *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.* TACL, 2022. — arXiv:2105.13626
- **[Survey]** Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, Iryna Gurevych. *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL, 2021. — arXiv:2012.15613

## 10. Worked Example

Fix $C = 6\times10^{20}$ FLOPs, untied embeddings ($\alpha=2$), width $d=2048$, corpus $B=4\times10^{12}$ bytes of English web text. Measured fertilities on a held-out shard (typical BPE values):

| $V$ | $f$ (bytes/token) | $N_v = 2dV$ | $N_{nv}$ at fixed $C,D$ |
|---|---|---|---|
| 32,768 | 3.90 | 134M | — |
| 131,072 | 4.35 | 537M | — |
| 524,288 | 4.70 | 2.15B | — |

Take $D = 20$B training tokens. Then $N = C/(6D) = 6\times10^{20}/(1.2\times10^{11}) = 5.0$B parameters total.

- $V=32$K: $N_{nv} = 5.00\text{B} - 0.13\text{B} = 4.87$B. Bytes seen: $20\text{B}\times3.90 = 78.0$GB.
- $V=131$K: $N_{nv} = 4.46$B. Bytes seen: $87.0$GB — **12% more text for the same FLOPs**.
- $V=524$K: $N_{nv} = 2.85$B. Bytes seen: $94.0$GB — 21% more text, but 41% fewer compute-bearing parameters.

The trade is legible: from 32K to 131K you give up 8% of $N_{nv}$ to gain 12% of bytes; from 131K to 524K you give up 36% of $N_{nv}$ to gain 8% more bytes. The turn happens between them, which is what the fitted law says.

**Where the obstruction becomes visible.** Suppose the three runs return held-out bits-per-byte of $0.812$, $0.806$, $0.815$. The spread between the best and worst is $0.009$ bpb. Now run 32K twice with different seeds: published small-scale LM runs routinely differ by $0.003$–$0.006$ bpb at this budget. The measured optimum is roughly one seed-standard-deviation deep. To resolve the argmin to a factor of 2 in $V$ you need $\sim$4 seeds per cell, tripling the cost of the sweep — and this is at $6\times10^{20}$ FLOPs, three orders of magnitude below the scale the conclusion is being applied to. The extrapolation is not cheap to check *because the signal is small exactly where checking is affordable*.

Second visible failure: convert to wall-clock. At $V=524$K the output projection is $2\times2048\times524{,}288 \approx 2.1$B multiply-adds per token versus $0.13$B at 32K. Counted as FLOPs the two are commensurable; measured on an H100 with a large logit tensor and its softmax, the large-vocabulary step is disproportionately memory-bound. A grid that is FLOP-matched is not time-matched, so "compute-optimal $V$" and "the $V$ you should actually train" can differ — and no published sweep reports both axes.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*