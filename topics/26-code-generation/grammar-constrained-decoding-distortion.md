---
id: 26-code-generation/grammar-constrained-decoding-distortion
title: "Grammar-Constrained Decoding Distribution Distortion"
topic: 26-code-generation
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Grammar-Constrained Decoding Distribution Distortion

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/grammar-constrained-decoding-distortion` · **Status:** partially-solved

## 1. Problem Statement

Grammar-constrained decoding (GCD) forces a language model to emit only strings in a formal language $L$ — a JSON schema, a SQL dialect, a Python subset, a tool-call signature. The standard implementation masks, at each step, every token that cannot extend the current prefix to some member of $L$, then renormalizes.

The problem: **local masking does not sample from the model conditioned on $L$.** It samples from a different distribution — one that greedily prefers prefixes with many locally-legal continuations, regardless of whether those prefixes lead anywhere the model considers likely. Output is guaranteed parseable and simultaneously less likely, under the model's own beliefs, than unconstrained output filtered by rejection.

Three variants, with different difficulty:

- **Measurement.** Given a model $p$, grammar $L$, and a GCD algorithm, quantify the divergence between the induced distribution $q$ and the true conditional $p(\cdot \mid L)$. Requires estimating a normalizing constant over an exponentially large set.
- **Method.** Build a decoder that is *sound* (output always in $L$), *unbiased* (samples $p(\cdot \mid L)$), and *cheap* (per-token overhead comparable to masking). Two of three are routine; all three are not.
- **Theory.** Characterize when masking is exactly unbiased, and bound the distortion as a function of grammar and tokenizer structure.

A solution is a decoder with a proof of unbiasedness (or a computable bound on the bias) whose amortized per-token cost stays within a small constant of masked decoding, and which does not degrade downstream task accuracy relative to rejection sampling.

## 2. Formal Setting

Let $V$ be the token vocabulary, $p$ an autoregressive model over $V^*$, and $L \subseteq \Sigma^*$ a language over characters $\Sigma$. Write $\mathcal{T}: V^* \to \Sigma^*$ for detokenization and $L_V = \mathcal{T}^{-1}(L)$ for the token-level lift.

**Target.** The correct object is the conditional
$$p^*(x) = \frac{p(x)\,\mathbf{1}[x \in L_V]}{Z}, \qquad Z = \sum_{y \in L_V} p(y).$$
$Z$ is measured, in practice, only as a Monte Carlo estimate: draw $N$ unconstrained samples, $\hat{Z} = \frac{1}{N}\sum_i \mathbf{1}[x_i \in L_V]$. For a strong model and a permissive JSON grammar $\hat{Z} \approx 0.9$; for a strict Python-subset grammar at temperature 1.0, $\hat{Z}$ can fall below $10^{-2}$, at which point $N = 10^4$ samples give a relative standard error near 10%.

**Masked decoder.** Define the viability predicate $\mathrm{via}(x_{<t}, v) = \mathbf{1}[\exists\, s \in V^*:\ x_{<t} v s \in L_V]$. Masked GCD samples
$$q(x_t \mid x_{<t}) = \frac{p(x_t \mid x_{<t})\,\mathrm{via}(x_{<t}, x_t)}{\sum_{v} p(v \mid x_{<t})\,\mathrm{via}(x_{<t}, v)}.$$

**The gap.** Exact conditioning requires weighting by the *expected future mass*
$$\alpha(x_{<t}, v) = \sum_{s:\, x_{<t}vs \in L_V} p(s \mid x_{<t} v) \in [0,1],$$
i.e. $p^*(x_t \mid x_{<t}) \propto p(x_t\mid x_{<t})\,\alpha(x_{<t},x_t)$. Masking substitutes $\mathbf{1}[\alpha > 0]$ for $\alpha$. So **masking is exactly unbiased iff $\alpha$ is constant over all viable tokens at every reachable prefix** — a condition essentially never met.

**Distortion metrics, as measured.**
- $D_{\mathrm{KL}}(q \parallel p^*)$, estimated from $M$ samples of $q$ as $\frac{1}{M}\sum_i \log\frac{q(x_i)}{p(x_i)} + \log \hat Z$. Both $q(x_i)$ and $p(x_i)$ are exactly computable per-sample; only $\hat Z$ is noisy.
- Per-sample **weight** $w(x) = p(x)/q(x)$; the effective sample size $\mathrm{ESS} = (\sum w_i)^2 / \sum w_i^2$ is the practical diagnostic and needs no $Z$.
- **Task delta**: pass@1 or exact-match under $q$ minus that under rejection sampling from $p$, at matched sample budget.

**Assumptions known to be violated.**
1. *Token-character alignment.* $\mathcal{T}^{-1}$ is many-to-one and BPE merges cross grammar boundaries, so masking on canonical tokenizations alone silently prunes valid strings (Beurer-Kellner et al., ICML 2024). Violated for every BPE tokenizer.
2. *$L$ captures the task.* Real targets are context-sensitive — scope, types, declared-before-use. Context-free $L$ over-admits; over-tight $L$ excludes correct programs.
3. *Bounded length.* $L_V$ is infinite; $Z$ is truncated at the generation cap, so all estimates are of a truncated conditional.
4. *The model is calibrated on $L$.* Fine-tuned code models place near-zero mass outside $L$ for easy prompts, making $\hat Z \approx 1$ and distortion invisible on exactly the benchmarks used to argue GCD is harmless.

## 3. State of the Art

**Established.**
- *Soundness and speed.* Willard & Louf's FSM-indexed masking (Outlines, arXiv:2307.09702) reduced mask construction to an $O(1)$ table lookup per step. XGrammar (Dong et al., 2024) and SynCode (Ugare et al., 2024) extend this to pushdown/LR machinery with reported near-zero end-to-end overhead. These are reproduced and deployed; the claim is speed, not fidelity.
- *Distortion is real and correctable in principle.* Park et al., **"Grammar-Aligned Decoding"** (NeurIPS 2024) prove masked GCD does not sample $p^*$, and give **ASAp** — adaptive sampling with approximate expected futures — which maintains an over-approximation of $\alpha$ that tightens with each sample and converges to $p^*$ in the limit. This is the strongest positive result available.
- *Tokenization misalignment.* Beurer-Kellner et al., "Guiding LLMs the Right Way" (ICML 2024) show naive masking is unsound w.r.t. the character-level language and give DOMINO, a sound token-aligned construction at minimal overhead.
- *Sequential Monte Carlo.* Loula et al. (ICLR 2025) reweight partial sequences with resampling, giving a consistent estimator of $p^*$ under arbitrary (including semantic, non-CFG) constraints.

**Claimed but unablated.** That constrained decoding "improves" code/tool-use accuracy. Tam et al., "Let Me Speak Freely?" (EMNLP 2024, industry track) report the opposite on reasoning-bearing tasks: tightening format constraints degrades accuracy, sometimes sharply. The counter-claim that this is a prompt-format artifact rather than a decoding-distribution effect has not been cleanly ablated — nobody has held prompt fixed and varied only the decoder between mask, ASAp, and rejection.

**Benchmark-number-only.** ASAp's convergence is demonstrated on small grammars (SLIA/CP-style, constrained JSON) with modest models; there is no published measurement of its distortion reduction on a full Python or SQL grammar at 70B scale.

## 4. What Is Known

- Masked GCD is biased. Proof, not conjecture (Park et al., NeurIPS 2024). The bias is exactly the omission of $\alpha$.
- ASAp's per-sample cost grows with the number of prior samples (it caches refuted-future mass by prefix); reported experiments run to hundreds of samples per prompt, not tens of thousands.
- Naive vocabulary masking is *unsound at the character level*: DOMINO's authors exhibit valid strings unreachable under canonical-token masking, with corrected decoding at overhead reported as near-zero versus unconstrained generation (ICML 2024).
- PICARD (Scholak et al., EMNLP 2021) raised text-to-SQL exact-set-match on Spider by roughly 3–8 points over unconstrained beams at T5-3B scale — the canonical demonstration that constraints help when the model is weak relative to the grammar.
- Format restriction can hurt: Tam et al. measured double-digit accuracy drops on GSM8K-style reasoning under strict JSON-schema decoding for several commercial models, versus free-form generation with post-hoc parsing.
- Engineering overhead is solved. XGrammar reports order-of-magnitude reductions in per-token grammar overhead versus earlier CFG engines, at 7B–8B serving scale.

Net: **soundness and speed are solved; fidelity is not.**

## 5. What Is Not Known

- **Theoretically open.** No characterization of grammars/tokenizers for which masking's KL distortion is $o(1)$ in sequence length. No lower bound saying unbiased GCD requires super-constant per-token work — plausible via a reduction from estimating $\alpha$, but unproven.
- **Empirically open.** Whether the Tam et al. accuracy drop is caused by *decoding distortion* or by *prompt-format shift*. Runnable today: hold the prompt fixed, swap only the decoder. Nobody has published it at frontier scale.
- **Empirically open.** The magnitude of $D_{\mathrm{KL}}(q\parallel p^*)$ for real code grammars at 30B+. All published measurements are on toy grammars.
- **Methodologically blocked.** "Distortion" for *semantic* constraints (type-correctness, name resolution). $\alpha$ is uncomputable when membership needs a full type-check, so even the target $p^*$ is only defined up to an oracle nobody runs at decode time.

## 6. Why It Is Hard

The specific obstruction is **an unestimable normalizing constant in the regime where it matters**. Distortion is largest exactly when $Z$ is small — when the model rarely satisfies the grammar unaided. But $\hat{Z}$ by rejection sampling costs $O(1/Z)$ samples, so the ground truth $p^*$ becomes unmeasurable precisely where GCD is most needed. This is a genuine measurement collapse, not a compute inconvenience: at $Z = 10^{-4}$, a 10%-relative estimate needs $\sim 10^6$ full generations per prompt.

Secondary: **the evaluation does not measure what it names.** pass@1 under GCD confounds three effects — validity gain, distortion loss, and grammar mis-specification. A pass@1 improvement is routinely reported as evidence that GCD is faithful; it is evidence of none of the three individually.

## 7. Current Research (as of 2026)

- **Unbiased-by-construction decoders.** ASAp descendants and SMC/twisted-proposal methods (Loula et al.; MIT ProbComp and collaborators) — trading exactness for variance rather than for bias. *(frontier — verify)* extensions to context-sensitive constraints with learned twist functions.
- **Learned future-mass estimators.** Train a small head to predict $\alpha(x_{<t},v)$ and reweight directly; cheap at inference, unproven calibration. *(frontier — verify)*
- **Constraint-aware fine-tuning.** Make $Z \to 1$ so masking becomes vacuous. Deployed informally in tool-use post-training; no public ablation isolating it from distortion.
- **Serving-stack integration.** XGrammar, Outlines, llguidance, SynCode inside vLLM/SGLang — the fidelity question is largely absent from this line of work.

## 8. Concrete Next Experiment

**Question:** does masked GCD lose accuracy relative to *unbiased* conditioning on the same grammar?

**Scale.** One open-weights code model at 30–34B (e.g. a Qwen2.5-Coder-32B-class model), 400 prompts drawn from MBPP+ and Spider (200 each). Grammar: a Python-subset CFG and the Spider SQL grammar. Budget: $10^3$ samples/prompt/arm, temperature 1.0 — about $1.6\times10^6$ generations, feasible on 8×H100 in a few days.

**Arms.**
1. **Control:** rejection sampling from unconstrained $p$, keeping only in-grammar samples. This is exact $p^*$ by construction, and also yields $\hat Z$.
2. Masked GCD (XGrammar/Outlines).
3. ASAp with matched sample budget.

Restrict scoring to prompts where $\hat{Z} \ge 0.05$, so arm 1 gets $\ge 50$ exact samples.

**Deciding number.** $\Delta = \text{pass@1}(\text{arm 2}) - \text{pass@1}(\text{arm 1})$, with 95% bootstrap CI over prompts. If $\Delta \le -2$ points, masking's distortion is a first-order correctness problem and unbiased decoders are mandatory. If $|\Delta| < 0.5$ points with a CI half-width under 1 point, masking is practically benign on real code grammars and the field should stop paying for exactness. Report $\mathrm{ESS}/M$ for arm 2 as the mechanism check: a large $|\Delta|$ with $\mathrm{ESS}/M > 0.5$ would mean the loss is *not* distributional and points at grammar mis-specification instead.

## 9. Key References

- **[Foundational]** Torsten Scholak, Nathan Schucher, Dzmitry Bahdanau. *PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models.* EMNLP 2021. — arXiv:2109.05093
- **[Foundational]** Gabriel Poesia, Oleksandr Polozov, Vu Le, Ashish Tiwari, Gustavo Soares, Christopher Meek, Sumit Gulwani. *Synchromesh: Reliable Code Generation from Pre-trained Language Models.* ICLR 2022. — arXiv:2201.11227
- **[Foundational]** Saibo Geng, Martin Josifoski, Maxime Peyrard, Robert West. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning.* EMNLP 2023. — arXiv:2305.13971
- **[Systems SOTA]** Brandon T. Willard, Rémi Louf. *Efficient Guided Generation for Large Language Models.* 2023. — arXiv:2307.09702
- **[SOTA — fidelity]** Kanghee Park, Jiayu Wang, Taylor Berg-Kirkpatrick, Nadia Polikarpova, Loris D'Antoni. *Grammar-Aligned Decoding.* NeurIPS 2024. — arXiv:2405.21047
- **[SOTA — soundness]** Luca Beurer-Kellner, Marc Fischer, Martin Vechev. *Guiding LLMs the Right Way: Fast, Non-Invasive Constrained Generation.* ICML 2024. — arXiv:2403.06988
- **[SOTA — inference]** João Loula, Benjamin LeBrun, Li Du, et al. *Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo.* ICLR 2025. — arXiv:2504.13139
- **[Empirical counterpoint]** Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, Chieh-Yen Lin, Hung-yi Lee, Yun-Nung Chen. *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models.* EMNLP 2024 (Industry Track). — arXiv:2408.02442
- **[Systems]** Shubham Ugare, Tarun Suresh, Hangoo Kang, Sasa Misailovic, Gagandeep Singh. *SynCode: LLM Generation with Grammar Augmentation.* 2024. — arXiv:2403.01632
- **[Survey/Systems]** Terry Yue Zhuo et al. *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions.* ICLR 2025. — arXiv:2406.15877

## 10. Worked Example

Grammar: JSON values restricted to `{"answer": <int>}` or `{"answer": "<string>"}`. Prompt asks for a numeric answer the model is unsure about.

At the position after `{"answer": `, suppose the model's next-token mass over viable tokens is:

| token | $p$ | $\alpha$ (future mass staying in $L$) | $p\cdot\alpha$ |
|---|---|---|---|
| `"` (opens string) | 0.20 | 0.95 | 0.190 |
| `4` | 0.45 | 0.30 | 0.135 |
| `1` | 0.30 | 0.30 | 0.090 |
| `-` | 0.05 | 0.20 | 0.010 |

All four are viable, so masking renormalizes over $p$ alone: $q(`"`) = 0.20$. The true conditional is $p^*(`"`) = 0.190/0.425 = 0.447$.

Why $\alpha$ differs: after `4`, the model wants to continue `42 (the answer to...` — free text that leaves the grammar. Masking will later force a closing quote or brace the model assigned little mass to, so the *whole branch* is low-probability under $p^*$. After `"`, almost every continuation stays legal.

Distortion at this single step: $D_{\mathrm{KL}}(q\parallel p^*) \approx 0.20\log\frac{0.20}{0.447} + 0.45\log\frac{0.45}{0.318} + 0.30\log\frac{0.30}{0.212} + 0.05\log\frac{0.05}{0.024} \approx 0.113$ nats. Over a 200-token generation with even one-tenth this per step, accumulated KL exceeds 2 nats — a factor of $\sim 8$ in sequence likelihood.

**The obstruction, made visible.** Every $\alpha$ in that table is a sum over all completions. To fill one cell exactly you must marginalize the model over an infinite set; to fill it empirically you sample from `4` until you observe the in-grammar fraction, which at $\alpha = 0.30$ is cheap — but the same measurement at the sequence level requires $\hat Z$, and here $Z = 0.425 \times (\text{same collapse at every subsequent step})$. Three more steps at comparable narrowing and $Z < 10^{-2}$: the reference distribution you need to compare against is now 100 generations per usable sample. The bias is provable in one line; measuring it costs more than the generation itself.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*