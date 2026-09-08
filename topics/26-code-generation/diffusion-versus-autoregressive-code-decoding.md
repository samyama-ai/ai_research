---
id: 26-code-generation/diffusion-versus-autoregressive-code-decoding
title: "Diffusion versus Autoregressive Decoding for Code"
topic: 26-code-generation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Diffusion versus Autoregressive Decoding for Code

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/diffusion-versus-autoregressive-code-decoding` · **Status:** empirically-open

## 1. Problem Statement

Does masked-diffusion (any-order, parallel-refinement) decoding beat left-to-right autoregressive (AR) decoding for program synthesis at matched training compute, matched inference compute, and matched data — and if so, on which sub-class of coding tasks?

Three variants, often conflated:

- **Measurement.** Define a comparison protocol under which "diffusion vs AR for code" is a well-posed question. The hard part is the exchange rate: AR spends one forward pass per token; a masked diffusion model (MDM) spends $T$ full-sequence passes to emit $L$ tokens. Wall-clock, FLOPs, and energy give three different verdicts.
- **Method.** Produce a diffusion code model that dominates a same-FLOP AR baseline on a fixed suite (HumanEval+, MBPP+, BigCodeBench, CRUXEval, SWE-bench Verified) — not just on throughput at a lower accuracy point.
- **Theory.** Prove or refute a separation: a family of code-generation tasks (e.g. those requiring a global constraint fixed late in the sequence — a closing brace count, a forward-referenced type, an import list) on which any-order denoising achieves error $\varepsilon$ with $o(\cdot)$ the sample or compute cost of left-to-right factorization, or a lower bound showing no such separation exists.

Solved would mean: a reproduced, ablated result that fixes the exchange rate and reports a sign on the accuracy-per-FLOP curve, with the mechanism identified (order flexibility vs infilling vs bidirectional attention vs extra compute per token).

## 2. Formal Setting

Let $x = (x_1,\dots,x_L) \in \mathcal{V}^L$ be a tokenized program, $c$ the prompt (docstring, signature, repo context).

**AR model.** $p_\theta^{\mathrm{AR}}(x\mid c) = \prod_{i=1}^{L} p_\theta(x_i \mid x_{<i}, c)$. Exact log-likelihood; $L$ sequential passes with KV cache; measured decode FLOPs $\approx 2NL$ for $N$ non-embedding parameters.

**Masked diffusion.** Forward process masks each token independently with probability $t\in[0,1]$: $q_t(x^{(t)}\mid x)=\prod_i \big[t\,\delta_{\mathrm{[M]}} + (1-t)\,\delta_{x_i}\big]$. The model $p_\theta(x_i \mid x^{(t)}, c)$ predicts masked positions. The training objective is the NELBO

$$\mathcal{L}(\theta) = \mathbb{E}_{t\sim U[0,1]}\,\mathbb{E}_{x^{(t)}\sim q_t(\cdot\mid x)}\left[\frac{1}{t}\sum_{i:\,x_i^{(t)}=\mathrm{[M]}} -\log p_\theta(x_i \mid x^{(t)}, c)\right] \;\ge\; -\log p_\theta(x\mid c),$$

which is an *upper bound* on NLL, not NLL (Sahoo et al., NeurIPS 2024; Shi et al., NeurIPS 2024). Sampling runs $T$ denoising steps, unmasking a schedule-chosen subset each step; decode FLOPs $\approx 2NLT$ without caching, or $\approx 2NL(T/B)$-ish for block-wise variants with block size $B$ and per-block caching.

**Quantities as measured.**
- Accuracy: $\mathrm{pass@}k$ over the EvalPlus-augmented tests, greedy for $k=1$; report the exact harness commit.
- Inference cost: three separate numbers — (i) decode FLOPs from the counted forward passes, (ii) tokens/s on a named accelerator at batch size 1 *and* at the throughput-optimal batch size, (iii) latency to first *valid parse*.
- Training cost: total pretraining FLOPs $C \approx 6ND$ over $D$ tokens, plus any AR-model initialization (adaptation from an AR checkpoint imports AR compute and must be added).
- The decisive statistic: $\Delta = \mathrm{pass@1}_{\text{diff}} - \mathrm{pass@1}_{\text{AR}}$ at matched $C$ and matched decode FLOPs.

**Assumptions, and which fail.** (a) *NELBO is a usable proxy for quality* — fails: the bound gap is unmeasured and varies with the masking schedule, so likelihood comparisons across the two families are not apples-to-apples. (b) *Steps are the cost unit* — fails: MDM passes are not cacheable the way AR passes are, so step-count parity ≠ FLOP parity ≠ wall-clock parity. (c) *Independent per-position decoding* — the factorized reverse step assumes conditional independence across simultaneously unmasked positions, which is false for code (a variable name and its later use are strongly dependent); this is why aggressive parallelism degrades. (d) *Matched data* — most released diffusion code models are adapted from AR checkpoints, so "trained from scratch, matched data" has essentially not been tested at 7B+.

## 3. State of the Art

**Established (reproduced or well-ablated).**
- Masked/absorbing discrete diffusion is the strongest text-diffusion family: SEDD (Lou, Meng, Ermon, ICML 2024) closed much of the perplexity gap to AR; MDLM (Sahoo et al., NeurIPS 2024) and MD4 (Shi et al., NeurIPS 2024) simplified it to a masked-token cross-entropy with a clean bound.
- Block diffusion (BD3-LMs, Arriola et al., ICLR 2025) interpolates AR and diffusion, restores KV caching across blocks, supports variable length, and improves likelihood over pure diffusion — this is the current practical architecture for long code.
- Adapting an AR checkpoint into a diffusion model is cheaper than pretraining one: Dream-7B (Ye et al., 2025) initializes from Qwen2.5-7B.

**Claimed but unablated.**
- Mercury Coder (Inception Labs, 2025) reports on the order of $10^3$ tokens/s on H100 with HumanEval in the high 80s. This is a benchmark number plus a throughput number; no matched-FLOP AR control, no disclosed training data.
- Gemini Diffusion (Google DeepMind, 2025) — announced with coding benchmark and latency figures, no paper, no ablation. *(frontier — verify)*
- LLaDA-8B (Nie et al., 2025) reports competitive general benchmarks against LLaMA3-8B but is weaker on code; the comparison is not FLOP-matched.

**Only-a-benchmark-number:** every public "diffusion beats AR on code" claim. None reports the same team's AR baseline trained on the same data at the same FLOPs.

## 4. What Is Known

- **The compute exchange rate is unfavorable for likelihood.** "Scaling up Masked Diffusion Models on Text" (Nie et al., 2025) finds MDMs need roughly an order of magnitude more training compute (~16× in their setting) to match AR likelihood, while remaining competitive on some downstream tasks. Scale: up to ~1B params, ~100B tokens.
- **Any-order helps on reversal-shaped tasks.** LLaDA-8B (pretrained on ~2.3T tokens) beats GPT-4o on the reversal-poem-completion task, where AR models degrade. Scale: 8B.
- **Order matters and the model is trained on the worst case.** Kim et al. (ICML 2025, *Train for the Worst, Plan for the Best*) show MDMs are trained on the hardest token orderings but can plan a favorable order at inference; adaptive-order inference gave large gains on Sudoku-style puzzles (from single digits to >85% in their reported setting) with no retraining.
- **Code diffusion has a documented decoding-order signature.** DiffuCoder (Apple, 2025, arXiv:2506.20639) shows code generation under an MDM is less left-to-right than prose, that sampling temperature widens the token-order distribution, and that a diffusion-native RL objective (coupled-GRPO) adds a few points on HumanEval/MBPP over the SFT model at 7B.
- **Small diffusion code models existed early and were weak.** CodeFusion (Singh et al., EMNLP 2023), 75M params, matched much larger AR models on top-1 for constrained NL-to-code (Bash, Python, Excel) but does not extend to open-ended synthesis.
- **Parallelism is bounded by dependency.** Accuracy falls monotonically as tokens-unmasked-per-step rises, in every published ablation; the usable parallel factor on code is small (single digits) without a verifier.

## 5. What Is Not Known

- **Empirically open.** No FLOP-matched, data-matched, from-scratch diffusion-vs-AR comparison on code at $\ge$7B. Everything public is either adapted-from-AR (confounded initialization) or unmatched in data. The experiment is runnable today for a few hundred GPU-days.
- **Empirically open.** Whether diffusion's edge, if any, is on *infilling / repo-context editing* rather than function synthesis. AR models with fill-in-the-middle training (Bavarian et al., 2022) already do infilling; nobody has compared FIM-AR against MDM at matched compute on a repository-level edit benchmark.
- **Theoretically open.** No separation theorem. There is no proof that any-order denoising is asymptotically cheaper (or more expressive) than AR for any natural class of programs, and no lower bound ruling it out. The conditional-independence error of a parallel unmasking step has no tight bound for realistic dependency structure.
- **Methodologically blocked.** "Matched inference compute" has no accepted definition across the two families. Step count, FLOPs, and wall-clock rank the methods differently, and no benchmark reports all three. Likelihood comparison is also blocked: MDM gives a bound, AR gives exact NLL.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by non-identifiability of the mechanism**.

Every headline diffusion-code result varies at least four things at once against its AR comparator: initialization (adapted from an AR checkpoint), training data, objective, and serving stack (custom parallel-decode kernels). A win therefore cannot be attributed. If a diffusion 7B beats an AR 7B on HumanEval+, the cause could be bidirectional attention, order flexibility, more effective compute per token ($T$ passes), better data, or a faster kernel that permits more samples per second.

Second obstruction: **the evaluation does not measure what it names.** HumanEval/MBPP are short, self-contained, left-to-right-friendly functions — exactly the regime where AR's inductive bias is free and diffusion's any-order capability is worth nothing. The tasks that would discriminate (long-range constraint satisfaction, repo-scale edits with forward references, patch generation) are dominated by retrieval and context handling, which swamps the decoding-order effect.

## 7. Current Research (as of 2026)

- **Block/semi-AR hybrids.** BD3-LM-style interpolation (Cornell, Kuleshov group) is the direction most likely to ship: KV caching plus local parallelism.
- **Diffusion-native RL for code.** Apple's DiffuCoder line (coupled-GRPO) addresses the fact that GRPO-style objectives assume an AR likelihood the MDM does not provide exactly.
- **Adaptive decoding order / planners.** Choosing which positions to unmask by model confidence or an external planner (Kim et al.; several 2025–2026 follow-ups). Code-specific variants that unmask at parser-determined boundaries. *(frontier — verify)*
- **Production diffusion decoders.** Inception Labs (Mercury), Google DeepMind (Gemini Diffusion), HKU NLP (Dream). Throughput claims are credible; quality-at-matched-compute claims are not yet checkable. *(frontier — verify)*
- **Verifier-guided parallelism.** Using a fast parser/type-checker as the acceptance test for multi-token unmasking, structurally analogous to speculative decoding (Leviathan et al., ICML 2023).

## 8. Concrete Next Experiment

**Scale.** Two models, 1.4B non-embedding params, trained from scratch on the identical 300B-token code-heavy mix (e.g. The Stack v2 subset + permissive NL), identical tokenizer, identical $C \approx 6ND \approx 2.5\times10^{21}$ FLOPs, identical LR schedule and batch size. Cost: roughly 4–6k A100-hours per arm.

**Arms.**
- *Control:* AR transformer with 50% fill-in-the-middle span training (Bavarian et al., 2022) — the strongest honest baseline, because it has infilling too.
- *Treatment:* MDLM-style masked diffusion, same backbone, bidirectional attention.
- *Optional third:* block diffusion, $B=32$.

**Protocol.** Evaluate at three matched decode-FLOP budgets per problem ($1\times$, $4\times$, $16\times$ the AR single-sample cost), spending extra budget on diffusion steps $T$ for the MDM and on best-of-$n$ with a test-free reranker for AR. Suites: HumanEval+/MBPP+ (synthesis), a held-out *infilling* suite built by masking spans in real repo functions, and CRUXEval (execution reasoning).

**Deciding number.** $\Delta_{16\times}^{\mathrm{infill}} = \mathrm{pass@1}_{\text{MDM}} - \mathrm{pass@1}_{\text{FIM-AR}}$ on the infilling suite at the $16\times$ budget. A reproduced $\Delta \ge +3$ points (95% CI excluding 0 over $\ge$800 problems, 3 seeds) establishes a real decoding-paradigm advantage in the regime diffusion is supposed to own. $\Delta \le 0$ there, given the known likelihood penalty, would reduce the diffusion case for code to a *throughput-only* argument — worth building, but not a capability claim.

## 9. Key References

- **[Foundational]** Austin, Johnson, Ho, Tarlow, van den Berg. *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS 2021. — arXiv:2107.03006
- **[Foundational]** Li, Thickstun, Gulrajani, Liang, Hashimoto. *Diffusion-LM Improves Controllable Text Generation.* NeurIPS 2022. — arXiv:2205.14217
- **[Foundational]** Bavarian et al. *Efficient Training of Language Models to Fill in the Middle.* 2022. — arXiv:2207.14255
- **[SOTA-theory]** Lou, Meng, Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* ICML 2024 (best paper runner-up). — arXiv:2310.16834
- **[SOTA-theory]** Sahoo et al. *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024. — arXiv:2406.07524
- **[SOTA-theory]** Shi, Han, Wang, Doucet, Titsias. *Simplified and Generalized Masked Diffusion for Discrete Data.* NeurIPS 2024. — arXiv:2406.04329
- **[SOTA]** Arriola et al. *Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.* ICLR 2025 (oral). — arXiv:2503.09573
- **[SOTA]** Nie et al. *Large Language Diffusion Models (LLaDA).* 2025. — arXiv:2502.09992
- **[SOTA]** Nie et al. *Scaling up Masked Diffusion Models on Text.* ICLR 2025. — arXiv:2410.18514
- **[SOTA-code]** Gong et al. *DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation.* Apple, 2025. — arXiv:2506.20639
- **[Code, early]** Singh, Cambronero, Gulwani, Le, Negreanu, Verbruggen. *CodeFusion: A Pre-trained Diffusion Model for Code Generation.* EMNLP 2023. — arXiv:2310.17680
- **[Analysis]** Kim et al. *Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions.* ICML 2025. — arXiv:2502.06768
- **[Baseline]** Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. — arXiv:2211.17192
- **[Eval]** Liu, Xia, Wang, Zhang. *Is Your Code Generated by ChatGPT Really Correct? (EvalPlus).* NeurIPS 2023. — arXiv:2305.01210

## 10. Worked Example

Take one HumanEval-style function, $L=128$ tokens, $N=7\times10^9$.

- **AR:** 128 sequential passes, decode FLOPs $\approx 2NL = 1.8\times10^{12}$.
- **MDM, $T=128$ (one token per step):** 128 *full-sequence* passes. Without caching, each pass costs $\approx 2NL$, so $\approx 2.3\times10^{14}$ FLOPs — **128× the AR cost for the same output.**
- **MDM, $T=32$ (4 tokens per step):** $\approx 5.7\times10^{13}$ FLOPs, still ~32× AR.

Now the accuracy side. Suppose both models sit at pass@1 $=0.60$ at $T=L$. Push to $T=32$ and the four positions unmasked per step are decoded independently: the reverse step approximates $p(x_{i_1},\dots,x_{i_4}\mid x^{(t)})$ by $\prod_j p(x_{i_j}\mid x^{(t)})$. For code this is exactly wrong at the places that matter — unmasking `def solve(n)` and a later `return solv(n)` in the same step yields a typo no later step can fix once positions are committed. Published ablations show a few points of pass@1 lost per doubling of tokens-per-step.

**Where the obstruction becomes visible.** Vendors report the $T=32$ throughput number (fast) alongside a pass@1 measured on a differently-trained model, and compare against an AR model from another lab trained on other data. On FLOPs the MDM is 32× more expensive; on wall-clock it can still win, because 32 batched full-sequence passes parallelize on an H100 while 128 sequential AR passes are memory-bandwidth bound. Both statements are true, and they point opposite ways. Until a single paper publishes accuracy at *matched decode FLOPs* and at *matched wall-clock*, for models trained from scratch on the same tokens, the comparison has no sign — which is precisely why the status here is empirically open rather than partially solved.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*