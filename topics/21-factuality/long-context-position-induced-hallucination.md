---
id: 21-factuality/long-context-position-induced-hallucination
title: "Hallucination Under Long Context and Position Effects"
topic: 21-factuality
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Hallucination Under Long Context and Position Effects

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/long-context-position-induced-hallucination` · **Status:** empirically-open

## 1. Problem Statement

A model is given a context $c$ of $n$ tokens containing evidence that fully determines the answer to a query $q$. As $n$ grows, and as the evidence moves away from the context edges, the model emits more claims that the context does not support. The question is what causes that increase and whether it is fixable at the architecture/positional-encoding level or only at the data level.

Three variants, of very different difficulty:

- **Measurement.** Define a hallucination rate that is a function of evidence *position* and context *length* alone, holding query difficulty, distractor content, and parametric answerability fixed. Currently no benchmark does this cleanly.
- **Method.** Produce a training or inference intervention that flattens the position curve without lowering peak accuracy and without shortening the usable context. Positional-interpolation and attention-recalibration methods claim this; few are ablated against a length-matched control.
- **Theory.** Prove whether, for softmax attention with a causal mask, a bounded-logit model must lose the ability to sharply select one of $n$ evidence tokens as $n \to \infty$, and whether the resulting failure mode is *abstention* (harmless) or *confabulation* (harmful).

Solving it means: a curve $H(\rho, n)$ that is measured, not confounded; a causal account of its shape; and an intervention whose effect on the curve replicates across model families.

## 2. Formal Setting

Let $M_\theta$ be an autoregressive model, $c = (c_1,\dots,c_n)$ a context, $q$ a query, $y = M_\theta(c,q)$ the generation. Let the gold evidence occupy a contiguous span starting at token index $p$; define **normalized depth** $\rho = p/n \in [0,1]$.

Decompose $y$ into atomic claims $a_1,\dots,a_m$ (FActScore-style decomposition, Min et al. 2023). Let $E(a_i, c) \in \{\text{supported}, \text{unsupported}\}$ be an entailment judgment against $c$. The measured quantity is

$$H(\rho, n) \;=\; \mathbb{E}\!\left[\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}\!\left[E(a_i,c)=\text{unsupported}\right]\;\Big|\;\rho, n\right].$$

Two summary statistics:

$$\Delta_{\text{pos}}(n) = \max_{\rho} H(\rho,n) - \min_{\rho} H(\rho,n), \qquad \Delta_{\text{len}}(\rho) = H(\rho, n_{\max}) - H(\rho, n_{\min}).$$

$\Delta_{\text{pos}}$ is the *position* effect (the "lost in the middle" U-curve); $\Delta_{\text{len}}$ is the *length* effect at fixed depth. The open question is the ratio $\Delta_{\text{pos}}/\Delta_{\text{len}}$.

Two controls are part of the definition, not extras:

- **Closed-book arm** $H_\varnothing(q)$: same query, no context. Subtracts parametric answerability, so a "correct" answer read from weights is not scored as context use.
- **Ablated-evidence arm** $H_{-}(n)$: identical context with the gold span replaced by a length- and topic-matched distractor. The correct behaviour is abstention; $1 - \text{abstain rate}$ under this arm is the **confabulation rate**, the part of the failure that is actually hallucination rather than retrieval miss.

Mechanistic side quantity: for head $h$ at layer $\ell$, the attention mass on the gold span, $\alpha^{(\ell,h)}_{\text{gold}} = \sum_{j \in \text{span}} A^{(\ell,h)}_{t,j}$, averaged over generated positions $t$.

**Assumptions, and which are violated.**
1. *Evidence is a single contiguous span.* Violated for multi-hop and aggregation queries, where evidence is distributed and $\rho$ is undefined.
2. *Distractors are exchangeable.* Violated: retrieved distractors vary in lexical similarity to $q$, and similarity interacts with position (Cuconasu et al., SIGIR 2024).
3. *Parametric and contextual knowledge are separable.* Violated by pretraining contamination; the closed-book arm bounds it but does not remove it.
4. *The entailment judge is position-blind.* Violated when the judge is itself a long-context LLM reading the same context — the measurement instrument has the bug under study.
5. *Token position equals semantic position.* Violated across tokenizers; $\rho$ measured in tokens is not $\rho$ measured in documents.

## 3. State of the Art

**Established (independently reproduced).**
- The U-shaped position curve. Liu et al., *Lost in the Middle* (TACL 2024) — multi-document QA and key-value retrieval; accuracy peaks when gold evidence is first or last and dips in the middle, across GPT-3.5-Turbo, Claude, and open models. Reproduced widely.
- Effective context $\ll$ claimed context. Hsieh et al., **RULER** (COLM 2024) — synthetic tasks with controlled length; most models advertising 32K hold Llama-2-7B-at-4K quality only to a much shorter length.
- Attention sinks: the first tokens absorb large attention mass regardless of content (Xiao et al., ICLR 2024), and this is load-bearing rather than incidental (Barbero et al., 2025).

**Claimed but unablated against a length-matched control.**
- That positional-encoding rescaling (Positional Interpolation, Chen et al. 2023; YaRN, Peng et al. ICLR 2024) reduces *hallucination*. These are perplexity- and retrieval-validated; claim-level factuality is largely untested.
- That attention recalibration flattens the curve. Hsieh et al., *Found in the Middle* (ACL Findings 2024) and Peysakhovich & Lerer's attention-sorting (2023) report gains, but mostly on extractive QA with short outputs, where free-form confabulation cannot be observed.

**Benchmark-number-only results.** NoCha (Karpinska et al., EMNLP 2024): 1,001 true/false claim pairs over 63 recent novels; best model at the time ~55% pair accuracy against ~97% for human readers. Michelangelo / MRCR (Vodrahalli et al., 2024) shows sharp degradation on latent-structure queries. Neither isolates position from length.

## 4. What Is Known

- **Position gap magnitude.** In Liu et al. (2024), with 20 retrieved documents, GPT-3.5-Turbo's multi-doc QA accuracy falls by roughly 20 points between the best (edge) and worst (middle) gold position; mid-position accuracy approaches its no-context baseline. Scale: 20 documents, ~4K tokens — a *short* context by 2026 standards.
- **Length hurts independently of task.** Levy et al., *Same Task, More Tokens* (ACL 2024): on FLenQA, reasoning accuracy degrades from ~2K tokens onward with the reasoning content held constant — the padding alone costs accuracy.
- **Sparse mechanistic locus.** Wu et al., *Retrieval Head Mechanistically Explains Long-Context Factuality* (ICLR 2025): under 5% of heads are retrieval heads; masking them collapses needle retrieval and induces confabulation, while masking equal numbers of random heads does not.
- **Softmax must disperse.** Veličković et al., *softmax is not enough (for sharp out-of-distribution)* (2024): with bounded logits, attention entropy necessarily grows with sequence length — sharp selection cannot be maintained as $n\to\infty$ without unbounded logits or temperature scaling.
- **Causal masking creates an early-position prior.** Wu, Wang, Jegelka & Jadbabaie, *On the Emergence of Position Bias in Transformers* (ICML 2025): a graph-theoretic account in which the causal mask biases mass toward earlier positions and depth amplifies it.
- **Distractors are not neutral.** Shi et al. (ICML 2023) and Cuconasu et al. (SIGIR 2024): irrelevant context changes accuracy in both directions depending on its similarity to the query.

## 5. What Is Not Known

- **Empirically open.** The ratio $\Delta_{\text{pos}}/\Delta_{\text{len}}$ at 128K–1M tokens, on free-form generation with claim-level scoring. Every ingredient exists (RULER-style controlled contexts, FActScore/RAGTruth-style scoring, open-weight 128K models); nobody has run the crossed design at scale.
- **Empirically open.** Whether length-induced errors are *abstention* or *confabulation*. Benchmarks score accuracy, which merges the two.
- **Theoretically open.** Whether dispersion (Veličković et al.) and causal-mask position bias (Wu et al.) are the same phenomenon or additive, and whether any fixed-precision softmax transformer can hold $\Delta_{\text{pos}} = 0$ at arbitrary $n$.
- **Methodologically blocked.** A position-blind judge. Verifying claims against a 128K context requires either a long-context judge (same bug) or human annotation (~$10^2$ per query-length cell). Until that is solved, $H(\rho,n)$ at long $n$ has unknown measurement error that itself varies with $\rho$.

## 6. Why It Is Hard

**Confounded measurement, structurally.** You cannot lengthen a context without changing something else. Padding with random text changes the distribution; padding with retrieved documents changes distractor count *and* distractor similarity; padding with the same document repeated changes the entropy of the attention field. Length, depth, distractor count, and distractor similarity co-vary by construction. Liu et al. varied position at fixed length; RULER varied length at fixed structure; no public suite crosses them with content held constant.

**Compounding this: the instrument shares the defect.** LLM-as-judge entailment over a 128K context is subject to the same positional bias being measured, so a measured U-curve is partly a curve in the judge. Chunked verification breaks the coupling but loses cross-chunk contradictions.

**Non-identifiability of the mechanism.** Attention dispersion, causal-mask prior, RoPE long-range decay (Barbero et al., ICLR 2025), and training-distribution length scarcity all predict the same U-curve. Observing the curve does not select among them.

## 7. Current Research (as of 2026)

- **Positional-encoding surgery.** RoPE base-frequency scaling and YaRN-style interpolation are standard in frontier long-context recipes; the open question is whether they change factuality or only perplexity *(frontier — verify)*.
- **Head-level intervention.** Retrieval-head identification and head-targeted training, following Wu et al. — small teams; the natural extension is to make retrieval-head attention mass a training signal.
- **Contamination-resistant long-context evals.** NoCha (UMass) and Michelangelo (Google DeepMind) both attack the problem that 128K benchmarks leak into pretraining.
- **Attribution-first generation.** ALCE-style citation-constrained decoding (Gao et al., EMNLP 2023) as a mitigation: force each claim to name a span, so unsupported claims become detectable rather than requiring a judge.
- **Hybrid attention and state-space long-context models** as an architectural control arm: they have different dispersion behaviour, so measuring $\Delta_{\text{pos}}$ across architecture families is now possible *(frontier — verify)*.

## 8. Concrete Next Experiment

**Design.** Fully crossed length × depth, content held constant.

- **Scale.** 4 open-weight models (8B, 30B, 70B, plus one hybrid-attention model), each with $\ge$128K claimed context. 3 lengths: 8K / 32K / 128K. 11 depth bins $\rho \in \{0, 0.1, \dots, 1.0\}$. 600 free-form queries whose gold evidence is a single 200-token span. Padding drawn from a *fixed* distractor pool, with distractor count held at 40 across all three lengths by varying distractor length — this is what decouples length from distractor count. 4 × 3 × 11 × 600 = 79,200 generations; ~$10^{10}$ prefill tokens, feasible on 8×H100 in days.
- **Control arms.** (a) Closed-book, no context. (b) Ablated-evidence: gold span swapped for a matched distractor; correct behaviour is abstention. (c) Shuffled-depth: identical token multiset, gold span at a random depth, to detect content-order artifacts. Scoring by chunked NLI verification over 4K windows with a small entailment model, plus 1,000 human-adjudicated items to bound judge error per depth bin.
- **The deciding number.**
$$R = \frac{\Delta_{\text{pos}}(128\text{K})}{H_{\text{mean}}(128\text{K}) - H_{\text{mean}}(8\text{K})}.$$
$R > 0.5$: position is the dominant mechanism, and positional remedies (interpolation, attention recalibration, Ms-PoE) are the right lever. $R < 0.2$: degradation is depth-uniform, positional remedies address the wrong variable, and the lever is data/training at length. Report $R$ with the confabulation rate from arm (b) alongside — a high $R$ with a flat confabulation rate means the failure is retrieval, not hallucination.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Foundational]** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[SOTA]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[SOTA]** Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu. *Retrieval Head Mechanistically Explains Long-Context Factuality.* ICLR, 2025. — arXiv:2404.15574
- **[SOTA]** Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, Mohit Iyyer. *One Thousand and One Pairs: A "novel" challenge for long-context language models.* EMNLP, 2024.
- **[Theory]** Petar Veličković, Christos Perivolaropoulos, Federico Barbero, Razvan Pascanu. *softmax is not enough (for sharp out-of-distribution).* 2024. — arXiv:2410.01104
- **[Theory]** Xinyi Wu, Yifei Wang, Stefanie Jegelka, Ali Jadbabaie. *On the Emergence of Position Bias in Transformers.* ICML, 2025.
- **[Method]** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR, 2024. — arXiv:2309.00071
- **[Method]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Empirical]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024. — arXiv:2402.14848
- **[Empirical]** Florin Cuconasu et al. *The Power of Noise: Redefining Retrieval for RAG Systems.* SIGIR, 2024.
- **[Survey]** Yikun Han et al. / see also Bai et al. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.* ACL, 2024. — arXiv:2308.14508

## 10. Worked Example

Take one retrieval head whose query-key logit for the gold token exceeds the logit for every other token by a fixed margin $\delta$. Attention mass on the gold token is

$$\alpha_{\text{gold}}(n) = \frac{e^{\delta}}{e^{\delta} + (n-1)}.$$

With $\delta = 8$ (a strong, well-trained margin; $e^8 = 2981$):

| $n$ | $\alpha_{\text{gold}}$ |
|---|---|
| 4,096 | $2981/7076 = 0.421$ |
| 32,768 | $2981/35{,}748 = 0.083$ |
| 131,072 | $2981/134{,}052 = 0.022$ |

A 32× length increase costs the gold token 18× of its attention mass with the *margin unchanged*. Restoring 0.42 at 128K needs $\delta' = 8 + \ln 32 = 11.47$ — the model must learn logit gaps that grow like $\ln n$, which is exactly the sharpness that bounded-logit softmax cannot supply (Veličković et al. 2024).

Now the obstruction. Suppose you measure claim-level hallucination on a RAG stack and see it rise from 6% at 4K to 27% at 128K. Three explanations fit:

1. **Dispersion** — the arithmetic above, purely a function of $n$; predicts a *flat* curve in $\rho$, so $\Delta_{\text{pos}} \approx 0$.
2. **Causal-mask position prior** — predicts a monotone or U-shaped curve in $\rho$, so $\Delta_{\text{pos}}$ carries most of the 21-point rise.
3. **Distractor count** — at 4K you retrieved 5 documents, at 128K you retrieved 160; predicts a rise driven by near-miss distractors and no dependence on $\rho$ once count is fixed.

All three produce the same headline 6% → 27%. Only the crossed design in §8 separates them, because only it holds distractor count at 40 while $n$ varies, and only it reports $\Delta_{\text{pos}}$ at each length. Reporting "hallucination increases with context length" without $\Delta_{\text{pos}}$ and a fixed distractor count names an effect it has not measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*