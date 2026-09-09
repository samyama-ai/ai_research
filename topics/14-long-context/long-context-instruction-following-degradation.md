---
id: 14-long-context/long-context-instruction-following-degradation
title: "Long-Context Instruction Following Degradation"
topic: 14-long-context
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Context Instruction Following Degradation

> **Topic:** Long Context · **ID:** `14-long-context/long-context-instruction-following-degradation` · **Status:** empirically-open

## 1. Problem Statement

A model given a system instruction ("answer in exactly three bullets", "never cite a source not in the provided documents", "reply in JSON with keys `a`, `b`") obeys it reliably at 2K tokens and unreliably at 200K. The question is why, and whether the loss is separable from the loss in retrieval accuracy that occurs over the same range.

- **Input.** A prompt $x = (I, C, q)$: an instruction set $I$ of $k$ verifiable constraints, a context $C$ of $n$ tokens, a query $q$.
- **Output.** A response $y$ from model $\pi_\theta$.
- **Predicate.** $\mathrm{Sat}(y, I) \in \{0,1\}$ — a programmatic checker, one per constraint, aggregated strictly (all $k$ must hold).
- **Solving it** means either (a) a method that holds $\mathbb{E}[\mathrm{Sat}]$ flat in $n$ out to the advertised window at fixed task difficulty, or (b) a demonstration that instruction adherence and evidence retrieval degrade through a single shared mechanism, so that fixing one fixes the other.

Three variants, different difficulty:

| Variant | Question | Status |
|---|---|---|
| **Measurement** | Can adherence loss be isolated from task-difficulty growth as $n$ grows? | Partly blocked — no accepted difficulty control |
| **Method** | Can training or inference-time intervention flatten $A(n)$? | Empirically open |
| **Theory** | Does a bounded-precision softmax attention layer necessarily lose constraint salience as $n\to\infty$? | Theoretically open |

## 2. Formal Setting

Let $\pi_\theta$ be autoregressive with context window $N$. Fix a task family $\mathcal{T}$ and a filler distribution $\mathcal{F}$ that is *irrelevant by construction* to $\mathcal{T}$ (e.g. Paul Graham essays, unrelated Wikipedia). Build prompts $x_n$ by inserting $n - n_0$ filler tokens into a base prompt of length $n_0$, keeping $I$, $q$, and the gold evidence identical.

**Adherence.** For constraint $j$ with checker $c_j: \mathcal{Y} \to \{0,1\}$,
$$A_j(n) = \mathbb{E}_{x_n \sim \mathcal{D}_n,\, y \sim \pi_\theta(\cdot \mid x_n)}[c_j(y)], \qquad A(n) = \mathbb{E}\Big[\textstyle\prod_{j=1}^k c_j(y)\Big].$$
Measured as a sample mean over $\geq 300$ prompts per $(n, j)$ cell at temperature 0; the binomial standard error at $A=0.8$, 300 samples is $0.023$, so differences below $\sim 5$ points are noise.

**Task accuracy.** $R(n) = \mathbb{E}[\mathbb{1}\{y \text{ answers } q\}]$, scored independently of $I$ (string match or exact-match on an extracted span — not an LLM judge, which is itself length-sensitive).

**Separation statistic.** The object of interest is the difference in slopes,
$$\Delta = \frac{\partial \log A}{\partial \log n} - \frac{\partial \log R}{\partial \log n},$$
estimated by regressing both on $\log n$ over $n \in \{2^{11}, \dots, 2^{17}\}$. $\Delta \approx 0$ means adherence loss is a shadow of retrieval loss. $\Delta < 0$ means adherence degrades *faster* and is a distinct failure mode.

**Instruction distance.** $d = $ token offset from the last instruction token to the generation start. In a system-prompt-first layout $d \approx n$; instruction repetition at the end sets $d \approx 0$. Sweeping $d$ at fixed $n$ separates positional decay from load effects.

**Attention mass on the instruction.** $M_\ell(n) = \frac{1}{|H|}\sum_{h}\sum_{t \in I} \alpha^{(\ell,h)}_{\text{gen},t}$, the fraction of layer-$\ell$ attention the first generated token puts on instruction spans. Trivially $M \leq |I|/n$ by normalization, so the meaningful quantity is the *ratio to uniform*, $M_\ell(n) \cdot n / |I|$.

**Assumptions, and where they break.**
1. *Filler is irrelevant.* Violated — long filler changes the output-length prior and can supply distractor content that a checker like "cite only provided sources" scores as a violation.
2. *Task difficulty is constant in $n$.* Violated for any retrieval-bearing task: more filler means more distractors, so $R(n)$ falls for reasons unrelated to instructions.
3. *Checkers are length-invariant.* Violated for format checkers on free generation — models emit longer answers at long context, and "exactly three bullets" fails more often simply because outputs grow.
4. *Decoding is deterministic.* Held only at $T=0$; most deployed serving stacks are non-deterministic under batching, so replication needs fixed batch composition.

## 3. State of the Art

**Established (independently reproduced).**
- Position sensitivity. *Lost in the Middle* (Liu et al., TACL 2024) — U-shaped accuracy over gold-document position in multi-document QA; reproduced across GPT-3.5, Claude, and open models.
- Claimed windows overstate usable windows. RULER (Hsieh et al., COLM 2024) defines effective length as the longest $n$ at which a model beats a Llama-2-7B 4K baseline; nearly every model tested fell short of its advertised length, several by $4\times$.
- Degradation begins far below the window. *Same Task, More Tokens* (Levy, Jacoby, Goldberg, ACL 2024) isolates input length on a reasoning task with fixed relevant content and finds decline starting around 3K tokens.

**Claimed but unablated.**
- That long-context SFT data "fixes" instruction following. Vendor and open-model reports show gains on IFEval-style suites at length; the ablation separating instruction adherence from retrieval improvement is generally absent.
- That instruction repetition at the end of context recovers adherence. Widely used as a prompting practice; published controlled sweeps over $d$ at fixed $n$ are thin.

**Benchmark numbers only (no mechanism).** HELMET (Yen et al., ICLR 2025) shows synthetic recall correlates poorly with downstream long-context task performance and that model rankings reorder with length. LongBench v2 (Bai et al., 2025), ∞Bench (Zhang et al., ACL 2024), and NoCha (Karpinska et al., 2024) each report large drops at length; none decomposes adherence from retrieval. Multi-IF (He et al., 2024) reports multi-turn instruction-following accuracy falling monotonically across three turns for every model tested — a length-adjacent but not length-controlled result.

## 4. What Is Known

- **Needle retrieval is near-saturated; instruction adherence is not.** Frontier models score >95% on single-needle retrieval at 128K, while RULER's multi-hop and aggregation categories collapse at the same lengths (Hsieh et al., 2024, 13 models, 4K–128K).
- **Onset is early.** Reasoning degradation from length alone appears by ~3K tokens (Levy et al., 2024, FLenQA, models up to GPT-4).
- **Book-scale global tasks are far from solved.** NoCha: 971 true/false claim pairs over books averaging ~127K tokens; the best model reported at 55.8% pair accuracy against ~97% for human readers (Karpinska et al., 2024).
- **Short-context instruction following is high but not saturated.** IFEval (Zhou et al., 2023) strict prompt-level accuracy for strong instruction-tuned models sits in the 80–90% band at prompts under 1K tokens — so there is real headroom to lose.
- **Attention dilution is measurable, not yet causal.** Attention-calibration work (*Found in the Middle*, Hsieh et al., NAACL 2025) shows position bias in attention mass is partly correctable at inference and improves retrieval accuracy; the same lever has not been shown to restore *format* compliance.
- **Length-generalization training helps unevenly.** Models trained with long-context continued pretraining improve on recall categories more than on aggregation/constraint categories (RULER category breakdowns, 2024).

## 5. What Is Not Known

- **Empirically open.** The core separation: nobody has published a length sweep with fixed instruction set, fixed gold evidence, programmatic constraint checkers, and independent task scoring, at $2$K–$128$K, across $\geq 5$ model families. The experiment costs on the order of $10^4$ long-context calls — runnable today, unrun at the right scale.
- **Empirically open.** Whether $A(n)$ depends on $n$ (total load) or on $d$ (instruction distance). These are confounded in every published layout.
- **Methodologically blocked.** No accepted difficulty control. Filler that is truly inert does not exist: inert-looking filler changes output-length priors, and semantically related filler adds distractors. Without a control, any $A(n)$ curve is uninterpretable.
- **Methodologically blocked.** Constraint checkers that are provably length-invariant. "Exactly three bullets" is not the same task when the model's own output prior has shifted.
- **Theoretically open.** Whether bounded-precision softmax attention must lose constraint salience as $n$ grows. Related results exist — bounded-precision transformers are contained in uniform $\mathsf{TC}^0$ (Merrill & Sabharwal, TACL 2023), and softmax attention over $n$ keys with bounded logit range cannot concentrate arbitrarily — but no theorem connects either to a lower bound on instruction-violation rate.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by absent ground truth for difficulty**. When $n$ grows, at least four things change together: attention entropy over a larger key set, the number of distractors for the retrieval sub-task, the model's output-length prior, and the effective distance $d$ from instruction to generation. Every published long-context evaluation varies all four at once. So a drop in $A(n)$ is consistent with attention dilution, with the retrieval sub-task simply getting harder, or with an artifact of the checker.

Second obstruction: **non-identifiability from behavior alone**. A model that violates "cite only provided sources" at 100K may have failed to retrieve the source, or retrieved it and ignored the constraint. The output is identical. Distinguishing them needs either a probe of the intermediate representation or a task design where retrieval is trivially verifiable and separately scored — and the second is exactly what is missing.

Cost is real but secondary: a 128K-token call is roughly $60\times$ the prefill FLOPs of a 2K call, so a 6-point log sweep with 300 samples per cell across 5 models is a few hundred million prefill tokens — thousands of dollars, not millions. This is not a compute-blocked problem.

## 7. Current Research (as of 2026)

- **Evaluation redesign.** HELMET (Princeton NLP: Yen, Gao, Chen) argues for application-centric long-context suites and diverse coverage; LongBench v2 (THU KEG) pushes to 2M-token inputs with human-verified difficulty. Neither yet ships an adherence/retrieval decomposition.
- **Attention-side interventions.** Position-bias calibration and attention re-weighting at inference (Hsieh et al., NAACL 2025) and retrieval-head analyses (Wu et al., 2024, showing a small set of heads is causally responsible for copy-retrieval).
- **Long-output and constraint generation.** LongGenBench-style suites that check constraints spread across a long *generation* rather than a long input *(frontier — verify: this line of work is fast-moving and several similarly named benchmarks exist)*.
- **Architectural.** Hybrid state-space/attention stacks and sparse-attention serving; the open claim that they trade recall for adherence differently from dense attention is untested *(frontier — verify)*.
- **Agentic multi-turn.** Multi-IF-style multi-turn constraint tracking; industrial interest is high because system prompts in agent stacks sit at $d \approx n$ by construction.

## 8. Concrete Next Experiment

**The $\Delta$ sweep.**

- **Scale.** $n \in \{2, 4, 8, 16, 32, 64, 128\}$K tokens; 300 prompts per cell; 5 model families (one frontier closed, two open ≥70B, two open ~8B); $k = 5$ programmatic constraints from the IFEval verifier family (JSON keys, exact bullet count, forbidden word, language, length band). Total $\approx 7 \times 300 \times 5 = 10{,}500$ calls per constraint layout, $\approx 2\times10^8$ prefill tokens.
- **Design.** Gold evidence is a single sentence at a *fixed relative position* (25% depth). Task score $R(n)$ is exact match on a 3-token answer span, so retrieval success is separately observable in the same output that the constraint checkers score. Filler is drawn from a domain disjoint from the evidence.
- **Control arms.** (1) *Difficulty control:* $n$ fixed at 2K, filler replaced by repeated copies of the same 2K block padded to length $n$ — same token count, no new distractors. (2) *Distance control:* at each $n$, a second layout with $I$ duplicated verbatim immediately before the generation, setting $d \approx 0$. (3) *Output-prior control:* score constraints only on responses whose length falls in the 2K-condition interquartile range.
- **Deciding number.** $\Delta = \partial\log A/\partial\log n - \partial\log R/\partial\log n$ over the sweep, with a bootstrap 95% CI. If the CI excludes 0 and $\Delta < -0.05$, instruction adherence is a distinct failure mode and deserves its own training signal. If the CI contains 0, adherence degradation is downstream of retrieval and the field should stop treating it as a separate problem. Secondary decider: if the $d \approx 0$ arm recovers $\geq 80\%$ of the adherence gap, the mechanism is positional, not load-based, and the fix is a prompt-layout change rather than a training change.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Foundational]** Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, Le Hou. *Instruction-Following Evaluation for Large Language Models.* 2023. — arXiv:2311.07911
- **[SOTA]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[SOTA]** Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, Danqi Chen. *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly.* ICLR, 2025. — arXiv:2410.02694
- **[SOTA]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: The Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024. — arXiv:2402.14848
- **[Empirical]** Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, Mohit Iyyer. *One Thousand and One Pairs: A "Novel" Challenge for Long-Context Language Models.* EMNLP, 2024. — arXiv:2406.16264
- **[Empirical]** Yushi Bai, Xin Lv, Jiajie Zhang, et al. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.* ACL, 2024. — arXiv:2308.14508
- **[Empirical]** Xinrong Zhang, Yingfa Chen, Shengding Hu, et al. *∞Bench: Extending Long Context Evaluation Beyond 100K Tokens.* ACL, 2024. — arXiv:2402.13718
- **[Mechanism]** Cheng-Ping Hsieh, Yung-Sung Chuang, et al. *Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization.* NAACL Findings, 2025. — arXiv:2406.16008
- **[Mechanism]** Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu. *Retrieval Head Mechanistically Explains Long-Context Factuality.* 2024. — arXiv:2404.15574
- **[Theory]** William Merrill, Ashish Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision Transformers.* TACL, 2023. — arXiv:2207.00729
- **[Survey]** Saurav Pawar, S.M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Aman Chadha, Amitava Das. *The What, Why, and How of Context Length Extension Techniques in Large Language Models — A Detailed Survey.* 2024. — arXiv:2401.07872

## 10. Worked Example

**Setup.** One constraint: "Respond with exactly one JSON object containing keys `answer` and `page`, and no prose." One task: return a 3-token answer found in a single planted sentence at 25% depth. Filler: unrelated essays.

Suppose a run at $n = 2$K and $n = 64$K gives, for a single model:

| $n$ | $R(n)$ retrieval | $A(n)$ format | Both |
|---|---|---|---|
| 2K | 0.97 | 0.94 | 0.92 |
| 64K | 0.81 | 0.66 | 0.60 |

Slopes over $\log_2 n$ (5 doublings): $\partial\log R/\partial\log_2 n = \ln(0.81/0.97)/5 = -0.036$; $\partial\log A/\partial\log_2 n = \ln(0.66/0.94)/5 = -0.071$. So $\Delta = -0.035$ — adherence falls about twice as fast as retrieval. That looks like a distinct failure mode.

**Where it breaks.** Condition on retrieval success. Among the 81% of 64K cases where $R=1$, format compliance is $0.60/0.81 = 0.74$; at 2K it is $0.92/0.97 = 0.95$. Still a 21-point gap — so far the conclusion survives.

Now apply the output-prior control. At 2K the model's median response is 18 tokens; at 64K it is 34, and 61% of the format violations at 64K are a preamble sentence ("Based on the provided documents, ...") emitted before the JSON. Restrict scoring to responses in the 2K interquartile length band and $A(64\text{K})$ rises to 0.88 — the gap collapses from 21 points to 7, and the 7 is within $2\times$ the binomial standard error at $n=300$.

**The obstruction, made visible.** The measured "instruction-following degradation" was, in this instance, mostly a shift in the model's output-length prior under long input — a different behavior that the checker happens to score as a constraint violation. Whether that shift *is* instruction-following failure or a separate phenomenon is a definitional choice nobody has made, and the answer to $\Delta < 0$ flips depending on which choice you take. That is why the problem is filed as empirically open with a methodologically blocked core: the number is cheap to compute and expensive to interpret.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*