---
id: 14-long-context/long-context-hallucination-attribution
title: "Long-Context Hallucination Attribution"
topic: 14-long-context
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Context Hallucination Attribution

> **Topic:** Long Context · **ID:** `14-long-context/long-context-hallucination-attribution` · **Status:** methodologically-blocked

## 1. Problem Statement

Given a model that has just produced an unsupported claim while conditioned on 100K–1M tokens of context, decide **where the error came from**: a specific context span, the absence of any supporting span, a parametric prior overriding the context, or the positional/architectural machinery (attention sinks, RoPE extrapolation, KV eviction) that decides which tokens are legible at all.

- **Input:** context $c = (c_1,\dots,c_n)$, $n \gtrsim 10^5$; generation $y$; a flagged span $y_{s:t}$ judged unsupported.
- **Output:** an attribution $a$ over a fixed cause set plus, for context-caused errors, a subset $S \subseteq [n]$ of blamed positions.
- **Predicate:** $a$ is correct if a counterfactual intervention on the named cause flips the error, and interventions on the non-named causes do not.

Three variants, different difficulty:

- **Measurement:** define a ground truth for "the cause". Currently undefined at long context — this is where the problem is blocked.
- **Method:** produce $S$ cheaply. Partially solved for short contexts (ContextCite, Lookback Lens); cost scales badly.
- **Theory:** decide whether the cause is identifiable at all from a single forward pass. Open.

Solving it means: a procedure that, on held-out long-context errors, assigns causes that survive a counterfactual audit at rates well above a length-and-position-matched control.

## 2. Formal Setting

Model $p_\theta(y \mid c)$. Let $\mathcal{V}(y_{s:t}, c) \in \{0,1\}$ be a support verdict: $1$ if some $S \subseteq [n]$ entails $y_{s:t}$. A **contextual hallucination** is $\mathcal{V}=0$ with the flagged span asserted as fact.

**Ablation attribution.** For a mask $m \in \{0,1\}^n$, let $c_m$ be the context with masked spans removed. Define the log-probability drop

$$\Delta(S) = \log p_\theta(y_{s:t} \mid c) - \log p_\theta(y_{s:t} \mid c_{\neg S}).$$

*Measured as:* teacher-forced scoring of the same token string under the ablated context — two forward passes, $O(n^2)$ attention each. ContextCite fits a sparse linear surrogate $\Delta(S) \approx \sum_{i \in S} w_i$ from $k$ random masks; $k = 32$–$256$ in published work, so cost is $k$ long-context forward passes per attributed span.

**Cause decomposition.** Write the flagged logit as a contrast between context-conditioned and context-free distributions,

$$\ell = \log p_\theta(y_{s:t}\mid c) - \log p_\theta(y_{s:t}\mid \varnothing),$$

the quantity context-aware decoding (Shi et al., 2024) amplifies. $\ell \le 0$ with the error present is evidence for a **parametric** cause; $\ell \gg 0$ with $\mathcal{V}=0$ is evidence for **spurious context binding**.

**Positional cause.** Let $\pi$ be a permutation of context chunks. The positional effect on span $y_{s:t}$ is

$$\rho = \mathbb{E}_\pi\!\left[\mathbf{1}\{\text{error present under } \pi\}\right] - \mathbf{1}\{\text{error present under identity}\},$$

estimated over $\sim 20$ permutations that preserve the evidence set but move it in the window.

**Attention-based signal.** Lookback ratio per head $h$, layer $l$, output token $u$:

$$L^{l,h}_u = \frac{\sum_{i \le n} \alpha^{l,h}_{u,i}}{\sum_{i \le n}\alpha^{l,h}_{u,i} + \sum_{n < j < u}\alpha^{l,h}_{u,j}},$$

context mass over context-plus-generated mass. Cheap: one pass, no ablation.

**Assumptions, and which break.**
1. *Removal-based counterfactuals stay on-distribution.* Violated — deleting spans from a 200K-token document produces contexts the model never saw in training; the ROAR critique (Hooker et al., NeurIPS 2019) applies directly.
2. *A unique blamed set exists.* Violated by redundancy: long documents restate facts, so many disjoint $S$ give similar $\Delta(S)$. Non-identifiable.
3. *Support verdict $\mathcal{V}$ is computable.* Violated at scale — verifying "no span in 200K tokens entails this claim" is a $\Theta(n)$ human read; NLI verifiers are trained on sentence pairs, not haystacks.
4. *Attention weight equals causal influence.* Contested since Jain & Wallace (NAACL 2019); attention sinks (Xiao et al., ICLR 2024) put large mass on semantically empty tokens.

## 3. State of the Art

**Established (ablated, reproduced).**
- **ContextCite** (Cohen-Wang et al., NeurIPS 2024): surrogate-model context attribution, validated by *top-$k$ drop* — removing the top-attributed sources collapses the target probability far more than removing random sources. Demonstrated on Llama-3-8B/Mistral-7B at context lengths in the thousands, not $10^5$.
- **Lost in the Middle** (Liu et al., TACL 2024): U-shaped accuracy over evidence position; on multi-document QA the mid-position penalty reaches roughly 20 points at 20–30 documents. Independently reproduced many times. Establishes that *position* is a real cause.
- **RULER** (Hsieh et al., COLM 2024): claimed context lengths overstate effective ones; most evaluated models degrade well before their advertised window.

**Claimed but under-ablated.**
- **Lookback Lens** (Chuang et al., EMNLP 2024): attention-map classifier detects contextual hallucination and transfers across tasks and to a larger model. Detection is demonstrated; *attribution* — naming which span — is not the evaluated quantity.
- Self-citation and post-hoc citation pipelines report high citation-precision numbers on ALCE-style benchmarks. These measure whether a cited span supports the claim, not whether it *caused* the generation. The two are routinely conflated.

**Benchmark-number-only.** RAGTruth (Niu et al., ACL 2024) gives ~18K span-level hallucination annotations, but over short RAG contexts. HALoGEN (Ravichander et al., ACL 2025) decomposes generations into atomic facts and adds an error taxonomy (training-data-recall error vs. absent-knowledge error) — the closest thing to a cause label, still not long-context.

**No SOTA exists** for cause assignment at $n \ge 10^5$. There is no benchmark with cause labels at that length.

## 4. What Is Known

- Position causes errors independently of content: the mid-context penalty (~20 points, 20–30 documents, GPT-3.5/Claude-1 era, TACL 2024) persists under content-preserving permutation.
- Parametric priors override context measurably: models flip toward their prior more often as the context evidence deviates further from it (Wu et al., ClashEval, NeurIPS 2024 D&B) — a directly observed parametric cause.
- Long-context comprehension is genuinely weak, so hallucinations at length are frequent enough to study: on NoCha (Karpinska et al., EMNLP 2024), book-length claim verification, the best model scored about 55% against ~97% for human readers, near chance on a balanced pair task.
- Attention mass is a poor proxy for causal influence in general (Jain & Wallace 2019; Wiegreffe & Pinter 2019), yet a *ratio* of attention mass carries usable signal for detection (Lookback Lens).
- Deletion-based attribution metrics are confounded by distribution shift (Hooker et al., NeurIPS 2019) and saliency methods can pass visual sanity checks while being independent of the model (Adebayo et al., NeurIPS 2018).

## 5. What Is Not Known

- **Methodologically blocked (primary).** No agreed operational definition of "the cause" of a long-context hallucination, and no dataset with cause labels above $10^4$ tokens. Every candidate ground truth — annotator judgment, ablation drop, citation precision — measures a different thing, and their disagreement rate is unmeasured.
- **Theoretically open.** Whether the blamed set is identifiable when the context is redundant. Conjecture: with $r$ paraphrases of the same fact, $\Delta(S)$ is near-flat across all $\binom{r}{1}$ singletons, so no removal-based method can select among them. No proof, no counterexample published.
- **Empirically open.** Whether ContextCite-style surrogates retain top-$k$-drop validity at 128K–1M tokens. The experiment is runnable — $k$ forward passes at 128K on an 8B model is hours on one node — and unrun at that scale.
- **Empirically open.** Whether the cause mix shifts with length: does the parametric-prior share of errors grow, shrink, or stay flat from 4K to 512K? No published curve.

## 6. Why It Is Hard

Three named obstructions.

1. **Absent ground truth at scale.** Establishing $\mathcal{V}=0$ requires certifying that *no* span in 200K tokens supports the claim. That is a linear human read (a 200K-token book is ~8 hours). NoCha's authors used readers who had genuinely read the books — that annotation model does not scale to thousands of labeled errors.
2. **Non-identifiability under redundancy.** Removal attribution is defined on a set function $\Delta$ that is flat under duplicated evidence. Two disjoint blamed sets can be equally consistent with every observation. This is not a measurement-noise problem; the target is not unique.
3. **Confounded counterfactual.** Every ablation changes context *length* and *position* simultaneously with content — and both are known independent causes (§4). A drop after deleting span $i$ may be caused entirely by having shifted span $j$ out of the window's high-attention region. Length-matched masking (replace, do not delete) is the obvious control and is rarely applied.

Compute is a secondary cost, not the blocker: $k=64$ masks $\times$ 128K tokens is affordable. The blocker is that after paying it you still cannot say what the answer should have been.

## 7. Current Research (as of 2026)

- **Cheap causal attribution.** Successors to ContextCite trading surrogate fidelity for passes; gradient- and KV-cache-based approximations that avoid re-encoding the prefix *(frontier — verify)*.
- **Internal-state detectors.** Lookback-Lens-style probes over attention and residual features, extended from detection to span localization (MIT/CSAIL and Meta lines) *(frontier — verify)*.
- **Cause taxonomies.** HALoGEN's error typology being ported to long-context and RAG settings (AI2 and collaborators).
- **Interpretability-grade attribution.** Influence functions on training data (Grosse et al., 2023) as the parametric-cause counterpart to context attribution; context-reliance frameworks such as PECoRe (Sarti et al., ICLR 2024) that pair a detection step with an attribution step.
- **Benchmarks.** Length-scaled successors to RULER and NoCha adding span-level error annotation *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does removal-based context attribution stay valid as context length grows, or does it degrade to position artifact?

- **Scale.** One open 8B-class long-context model (e.g. Llama-3.1-8B-Instruct, 128K). Build 300 items from 60 book-length documents. Each item: a synthesized question whose evidence is a single planted span. Run at $n \in \{4\text{K}, 32\text{K}, 128\text{K}\}$ — the *same* 300 items padded with in-domain filler, so the answer is fixed and only length changes. Evidence position randomized over $\{$start, 25%, 50%, 75%, end$\}$. Collect the subset where the model errs. Attribute each error with ContextCite, $k=64$ masks.
- **Control arm (two, both required).** (a) *Length-matched masking:* replace the candidate span with an equal-token in-domain distractor instead of deleting it, so length and downstream positions are unchanged. (b) *Random-span baseline:* attribute using uniformly sampled spans of the same token count.
- **Deciding number.** Planted-span recall@1 of the attributor under length-matched masking, as a function of $n$. If recall@1 stays within 10 points from 4K to 128K, removal attribution survives scaling and the problem is a method problem. If it drops toward the random-span baseline at 128K while deletion-based masking keeps a high score, the published validity is a length-and-position artifact and the field needs a different ground truth. Cost: $300 \times 3 \times 64$ forward passes, most at short length — order 1–2 GPU-days on one H100.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Foundational]** Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, David Reitter. *Measuring Attribution in Natural Language Generation Models.* Computational Linguistics, 2023.
- **[SOTA]** Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, Aleksander Mądry. *ContextCite: Attributing Model Generation to Context.* NeurIPS, 2024. — arXiv:2409.00729
- **[SOTA]** Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, James Glass. *Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps.* EMNLP, 2024. — arXiv:2407.07071
- **[Benchmark]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[Benchmark]** Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, Mohit Iyyer. *One Thousand and One Pairs: A "novel" challenge for long-context language models.* EMNLP, 2024. — arXiv:2406.16264
- **[Benchmark]** Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song, Tong Zhang. *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.* ACL, 2024. — arXiv:2401.00396
- **[Method]** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Method]** Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, Scott Wen-tau Yih. *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding.* NAACL, 2024. — arXiv:2305.14739
- **[Critique]** Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, Been Kim. *A Benchmark for Interpretability Methods in Deep Neural Networks.* NeurIPS, 2019.
- **[Critique]** Sarthak Jain, Byron C. Wallace. *Attention is not Explanation.* NAACL, 2019.
- **[Survey]** Ashish Ravichander, Shrusti Ghela, David Wadden, Yejin Choi. *HALoGEN: Fantastic LLM Hallucinations and Where to Find Them.* ACL, 2025. — arXiv:2501.08292

## 10. Worked Example

A 190K-token contract corpus. The question: *what is the termination notice period?* Section 12.4 (position ~0.5 of the window) says **60 days**. An appendix at ~0.93 restates it as "the notice period set out in §12.4". The model answers **30 days** — the modal value in contract corpora, so a plausible parametric prior.

Run the three probes.

| Probe | Measurement | Value |
|---|---|---|
| Ablation, delete §12.4 | $\Delta$ on the "30 days" span | $-0.04$ nats |
| Ablation, delete appendix | $\Delta$ | $-0.03$ nats |
| Context-free contrast | $\ell = \log p(\cdot\mid c) - \log p(\cdot\mid\varnothing)$ | $-0.6$ nats |
| Permutation, §12.4 moved to position 0.05 | error recurs? | no, answers 60 days |

Read it. Deletion of either evidence span changes almost nothing, because the model was not using either — consistent with a parametric cause, and $\ell < 0$ agrees. But the permutation probe says moving the *same* span to the front fixes the answer, which is a positional cause. Both are true and they are not separable: the prior wins *because* the evidence sits mid-window where it is under-attended, and the evidence is under-attended *only* for a claim the prior already covers.

Now the redundancy trap. Delete §12.4 **and** the appendix together: $\Delta = -0.9$ nats — an order of magnitude larger than the sum of the singles, $-0.07$. The set function is strongly non-additive, so the linear surrogate ContextCite fits assigns near-zero weight to both spans and the attributor returns "no context cause". And the joint deletion removed 4.1K tokens, shifting every later section's position — the control arm (replace with equal-length distractor) gives $\Delta = -0.5$ nats instead, so roughly 40% of the apparent effect was length, not content.

The obstruction is visible without any modelling assumption: three defensible probes give three different causes for one error, and there is no label to say which is right.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*