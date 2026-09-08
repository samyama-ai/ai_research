---
id: 01-tokenization/cross-vocabulary-ensembling
title: "Token-Level Ensembling Across Heterogeneous Vocabularies"
topic: 01-tokenization
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Token-Level Ensembling Across Heterogeneous Vocabularies

> **Topic:** Tokenization & Vocabulary · **ID:** `01-tokenization/cross-vocabulary-ensembling` · **Status:** partially-solved

## 1. Problem Statement

**Input.** $K$ autoregressive language models $M_1,\dots,M_K$, each with its own tokenizer $\tau_k$ and vocabulary $V_k$, where the $V_k$ are neither equal nor nested (e.g. Llama-3 at $|V|=128{,}256$, Mistral at $32{,}000$, Gemma at $256{,}000$). Weights $w_k$ may be fixed or context-dependent.

**Output.** A single generative process over byte strings that combines the members at every decoding step, not only after full generations are produced.

**Objective.** Beat the best single member on a held-out task at matched inference compute, while remaining a well-defined distribution over strings.

Three variants, routinely conflated:

- **Measurement.** Is the observed gain from combining *knowledge*, or from spending $K\times$ the forward passes? The control is a compute-matched single model, not the mean of the members.
- **Method.** Build a per-step combination rule whose outputs are comparable across $V_k$. Options: project all members into a shared or union vocabulary; marginalize each member down to bytes; or drop to string-level combination (reranking, MBR) and give up per-step steering.
- **Theory.** Is the exact per-step ensemble of tokenized models over a common byte alphabet computable in polynomial time, and does a mixture over string distributions dominate the best member under a stated loss?

The theory variant is the only one that is genuinely open; the method variant is partially solved; the measurement variant is the one most papers get wrong.

## 2. Formal Setting

Let $\Sigma$ be the byte alphabet, $\Sigma^*$ the strings. Each model defines a distribution over token sequences $p_k(t_{1:n})$, $t_i \in V_k$. The **induced string distribution** is the pushforward through detokenization $\mathrm{d}_k: V_k^* \to \Sigma^*$:

$$P_k(s) \;=\; \sum_{t \in \mathrm{d}_k^{-1}(s)} p_k(t).$$

Measurement note: $P_k(s)$ is *not* what a standard `logprobs` API returns. The API returns $p_k(\tau_k(s))$ — the mass on the single canonical tokenization. The gap $P_k(s) - p_k(\tau_k(s))$ is the noncanonical mass, and is nonzero whenever a string admits multiple segmentations.

The two natural ensembles over $\Sigma^*$:

$$P_{\text{mix}}(s) = \sum_k w_k P_k(s), \qquad P_{\text{prod}}(s) \propto \prod_k P_k(s)^{w_k}.$$

For the mixture, the next-byte conditional is exact and local given byte-level conditionals:

$$P_{\text{mix}}(b \mid s_{<i}) = \sum_k \underbrace{\frac{w_k P_k(s_{<i})}{\sum_j w_j P_j(s_{<i})}}_{\tilde w_k(s_{<i})} \, P_k(b \mid s_{<i}),$$

so the weights are a *posterior over members* that updates with evidence. Nothing here requires shared vocabularies. The entire difficulty is computing the two quantities on the right.

**Byte-level conditional.** For a prefix $s_{<i}$, let $C_k(s_{<i}) \subseteq V_k^*$ be the **cover set**: token sequences whose detokenization has $s_{<i}$ as a prefix. Then

$$P_k(b \mid s_{<i}) = \frac{\sum_{t \in C_k(s_{<i}b)} p_k(t)}{\sum_{t \in C_k(s_{<i})} p_k(t)}.$$

Measured cost: one forward pass per distinct prefix-token branch. The branching is bounded by the longest token in $V_k$ (typically 16–32 bytes), so the cover set is finite but the number of passes grows with boundary ambiguity.

**Union-vocabulary alternative.** Define $V_\cup = \bigcup_k V_k$ and a projection $\Pi_k \in \{0,1\}^{|V_\cup| \times |V_k|}$ or a learned $\Pi_k \in \mathbb{R}^{|V_\cup| \times |V_k|}$; ensemble $q(\cdot) = \sum_k w_k \Pi_k p_k(\cdot \mid \text{ctx}_k)$, then sample one token and re-tokenize each member's context.

Assumptions, with violation status:

1. **Members are conditioned on the same string.** Violated after any step where the sampled token is not canonical for some $\tau_k$: re-tokenizing puts $M_k$ on a prefix it would never emit.
2. **Canonical tokenization carries all mass.** Violated by construction; noncanonical mass is what makes byte-level conditioning ill-behaved (tokenization bias).
3. **Members are calibrated on a shared scale.** Violated — temperature and entropy differ per model, so $w_k$ set by validation loss is not the mixture-optimal weight.
4. **Union projection is loss-free.** Violated whenever $t \in V_j \setminus V_k$ has no single-token image in $V_k$; the projection then compares a full-word logit against a subword logit.

## 3. State of the Art

**Established (method, empirical).**
- **Exact byte-level marginalization.** Phan et al., *Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles* (ICML 2025), gives an algorithm converting a tokenized LM into an exact next-byte predictor, and uses it to ensemble models with different tokenizers without any alignment heuristic. This is the only line where correctness is proved rather than asserted; the same group's earlier *Understanding and Mitigating Tokenization Bias in Language Models* (2024) establishes the underlying byte-token representation result.
- **Heterogeneous speculative decoding.** Timor et al., *Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies* (ICML 2025), gives drafter/target pairs with disjoint vocabularies and a provably lossless acceptance rule (string-level exact match; token-level intersection). Losslessness is a theorem; the speedups (reported in the ~1.5–2.8× range) are benchmark numbers.

**Claimed but under-ablated.**
- **EVA** (Xu et al., *Bridging the Gap between Different Vocabularies for LLM Ensemble*, NAACL 2024): learns a token-alignment matrix from overlapping tokens. Gains over the best single member are reported on standard suites; the ablation isolating alignment quality from ensemble-of-any-kind is partial.
- **DeePEn** (Huang et al., *Ensemble Learning for Heterogeneous LLMs with Deep Parallel Collaboration*, NeurIPS 2024): maps each member's distribution into a relative representation space anchored on shared tokens, then searches back. Benchmark-number result.
- **GaC** (Yu et al., EMNLP Findings 2024): treats generation as classification over a union vocabulary; also reports that ensembling only at high-disagreement steps retains most of the gain at a fraction of the cost. The "critical token" claim is the interesting one and is the least ablated.
- **UniTE** (2024): restricts the union to each member's top-$k$, sidestepping full-vocabulary alignment.

**Adjacent, not ensembling.** FuseLLM / FuseChat (Wan et al., ICLR 2024; 2024) align vocabularies with minimum-edit-distance token matching to *distill* several models into one — a train-time, single-model outcome. Zero-Shot Tokenizer Transfer (Minixhofer et al., NeurIPS 2024) makes a model consume a foreign tokenizer via a hypernetwork, which converts the problem to the homogeneous case at the cost of retraining.

## 4. What Is Known

- **Mixture posterior weights are exact and cheap** given byte-level conditionals: the update $\tilde w_k \propto w_k P_k(s_{<i})$ needs only running prefix likelihoods. No approximation is required for the combination rule itself — only for the conditionals.
- **Noncanonical mass is not negligible.** Byte-level analyses of BPE models find that naive re-tokenization of a mid-token prefix yields conditionals that differ substantially from the true byte-marginal; this is the mechanism behind the well-known prompt-boundary ("token healing") failure in FIM tasks, and is what the ICML 2025 exact-byte work removes.
- **Vocabulary overlap is high but structurally biased.** Between two 32k–128k BPE vocabularies trained on similar web corpora, a large majority of tokens by *frequency* are shared (ASCII words, punctuation) while the disjoint tail concentrates in multilingual, code, and long-word tokens — exactly where members disagree and where an ensemble would add value. Any method keyed on overlapping tokens is therefore best-supported where it is least needed.
- **Losslessness across vocabularies is achievable for acceleration.** The heterogeneous speculative-decoding result shows the target distribution can be preserved exactly with a mismatched drafter — proof, at 7B–70B target scale.
- **Union-vocabulary ensembles beat the best single member on published suites** (EVA, DeePEn, GaC), typically by low-single-digit points, measured at 7B–13B members and 2–4 members per ensemble.

## 5. What Is Not Known

- **Theoretically open.** Whether exact next-byte conditionals for a BPE model are computable in time polynomial in the prefix length *without* a bound on maximum token length, and whether the product-of-experts ensemble $P_{\text{prod}}$ over string distributions admits any local (bounded-lookahead) sampler at all. Also open: conditions under which $P_{\text{mix}}$ provably dominates $\max_k P_k$ in held-out log-loss for non-nested vocabularies.
- **Empirically open.** Whether *any* published cross-vocabulary ensemble beats a compute-matched single model. A 3-member 7B ensemble costs ~21B forward FLOPs per token; the honest control is a 21B-class model, and no paper in the list above runs it. The experiment is runnable today.
- **Methodologically blocked.** "Ensemble gain" is not defined at fixed string-level semantics: members re-tokenize the shared prefix differently, so the $K$ models are not conditioned on the same event, and the reported quantity is not a mixture of the members' string distributions. Until every member reports byte-level conditionals, cross-paper comparison of ensemble gains is comparing different objects.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the conditioning event**. At decode step $i$, the ensemble has committed to a byte prefix $s_{<i}$; each member must supply $P_k(\cdot \mid s_{<i})$. But a tokenized model natively supplies $p_k(\cdot \mid t_{1:m})$ for one specific segmentation. Mapping the byte prefix back to tokens is one-to-many, and the choice is not free: canonical re-tokenization silently drops the noncanonical mass, and forcing a mid-token split feeds the model an off-distribution prefix. So the per-step "distribution" being averaged is a function of an arbitrary segmentation choice, not of the string. Two implementations of "the same" ensemble can differ by more than the reported gain.

This compounds with a confounded measurement: $K$ members means $K$ forward passes, and adding parameters improves perplexity on its own. Reported gains of 1–3 points over the best single member sit inside the range a compute-matched scale-up would deliver.

## 7. Current Research (as of 2026)

- **Byte-level and character-level interfaces as the unification layer.** Meta FAIR and collaborators (Phan, Ullrich, Gat, Amos) on exact byte-level probabilities; Vieira et al. on language models over characters. Direction: make the byte marginal the API, at which point cross-vocabulary ensembling reduces to ordinary mixture-of-experts. Cost is the open question.
- **Heterogeneous speculative decoding at production scale**, including drafters shared across model families (Timor et al. and the HuggingFace universal-assisted-generation line). *(frontier — verify)* extension from acceleration to genuine quality ensembling.
- **Selective / critical-token ensembling** — spend the $K\times$ cost only where member entropy or disagreement is high. Post-GaC follow-ups. *(frontier — verify)* whether the selection rule survives a compute-matched control.
- **Tokenizer transfer and grafting** (Minixhofer et al.; Remy et al., COLM 2024) as an alternative that eliminates the problem rather than solving it.
- **Routing instead of ensembling** — pick one member per query. Cheaper, and the correct baseline that ensembling papers should but usually do not report.

## 8. Concrete Next Experiment

**Question.** Does cross-vocabulary token-level ensembling deliver gains that survive a compute-matched control and an exact-conditioning control?

**Scale.** Three open members with disjoint tokenizers at ~7–8B: Llama-3.1-8B ($|V|=128{,}256$), Mistral-7B-v0.3 ($32{,}768$), Qwen2.5-7B ($151{,}936$). Evaluation: held-out byte-level log-loss on a 50M-byte mixed corpus (web, code, non-Latin-script text), plus GSM8K, HumanEval, MMLU. Budget: ~2k A100-hours including the control.

**Arms.**
1. Best single member (per-task).
2. Union-vocabulary ensemble with canonical re-tokenization (the EVA/GaC family).
3. Exact byte-level mixture with posterior weights $\tilde w_k \propto w_k P_k(s_{<i})$ (Phan et al. algorithm).
4. **Control A (compute-matched):** a single ~24B model (e.g. Mistral-Small-24B or Qwen2.5-32B at reduced precision), matched on forward FLOPs per generated byte, not on parameters.
5. **Control B (cheap alternative):** an oracle router that picks the best single member per query — the upper bound on routing.

**Deciding number.** Held-out byte-level log-loss in bits/byte. The ensemble is real only if arm 3 beats **arm 4** by $\ge 0.02$ bits/byte with a paired bootstrap 95% CI excluding zero, *and* beats arm 5. Secondary, and equally informative: the gap between arm 2 and arm 3 measures how much of the published literature's reported gain is an artifact of re-tokenization rather than ensembling. If arm 2 $\approx$ arm 3, the conditioning problem is empirically benign and the field can stop worrying about it; if arm 2 is within noise of arm 1 while arm 3 is not, the union-vocabulary line was measuring the wrong thing.

## 9. Key References

- **[Foundational]** Phan, B., Havasi, M., Muckley, M., Ullrich, K. *Understanding and Mitigating Tokenization Bias in Language Models.* 2024.
- **[SOTA]** Phan, B., Amos, B., Gat, I., Havasi, M., Muckley, M., Ullrich, K. *Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles.* ICML, 2025.
- **[SOTA]** Timor, N., et al. *Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies.* ICML, 2025.
- **[Method]** Xu, Y., Lu, J., Zhang, J. *Bridging the Gap between Different Vocabularies for LLM Ensemble.* NAACL, 2024.
- **[Method]** Huang, Y., et al. *Ensemble Learning for Heterogeneous Large Language Models with Deep Parallel Collaboration.* NeurIPS, 2024.
- **[Method]** Yu, Y., et al. *Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling.* Findings of EMNLP, 2024.
- **[Adjacent]** Wan, F., Huang, X., Cai, D., Quan, X., Bi, W., Shi, S. *Knowledge Fusion of Large Language Models.* ICLR, 2024.
- **[Adjacent]** Minixhofer, B., Ponti, E. M., Vulić, I. *Zero-Shot Tokenizer Transfer.* NeurIPS, 2024.
- **[Adjacent]** Jiang, D., Ren, X., Lin, B. Y. *LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion.* ACL, 2023.
- **[Adjacent]** Vieira, T., et al. *From Language Models over Tokens to Language Models over Characters.* 2024.
- **[Background]** Hinton, G. E. *Training Products of Experts by Minimizing Contrastive Divergence.* Neural Computation, 2002.

## 10. Worked Example

Two members must agree on the next bytes after the prompt `"Compute the definite"`. The continuation is `" integral"`.

- $M_A$ (128k vocab) has `" integral"` as a **single token**.
- $M_B$ (32k vocab) segments it as `" integ"` + `"ral"`.

**Step 1.** The union-vocabulary ensemble samples one token. Suppose it samples `" integ"` from $M_B$'s support (present in $V_B$, absent from $V_A$).

**Step 2.** $M_A$ must now be conditioned on the byte prefix `"...definite integ"`. Its canonical tokenizer maps that string to `[" integ"]`? No — $\tau_A$ is greedy-longest and `" integ"` is not a token of $V_A$; it backs off to, say, `[" int", "eg"]`. $M_A$ is now conditioned on a two-token sequence it would essentially never generate, because in its own distribution the bytes `" integ"` are only ever produced as a prefix of the single token `" integral"`.

**The numbers** (stipulated to make the size of the effect visible; the structure is what matters).

Under $M_A$'s true byte-marginal, mass on the continuation from prefix `" integ"` splits as: `"ral"` $0.87$, `"er"` $0.09$, other $0.04$ — because the cover set $C_A(\text{" integ"})$ is dominated by the single token `" integral"`.

Under naive re-tokenization to `[" int", "eg"]`, $M_A$ returns: `"ral"` $0.31$, `"rity"` $0.14$, `"er"` $0.12$, and $0.43$ scattered — the model is now completing a *different* string-generating process, one in which the writer already chose to break `int|eg`.

**Consequence for the ensemble.** With uniform priors $w_A = w_B = 0.5$ and $M_B$ giving `"ral"` $\to 0.80$:

$$P_{\text{ens}}(\texttt{"ral"}) = \begin{cases} 0.5(0.87) + 0.5(0.80) = 0.835 & \text{exact byte-marginal} \\ 0.5(0.31) + 0.5(0.80) = 0.555 & \text{naive re-tokenization} \end{cases}$$

A $0.28$ absolute swing on a token both models individually consider near-certain, produced entirely by a segmentation choice with no semantic content. In log-loss that is $0.59$ bits on this one byte-boundary — an order of magnitude larger than the $0.02$ bits/byte threshold that Section 8 proposes as the bar for a *real* ensembling gain.

The obstruction is now visible: the reported gain of a union-vocabulary ensemble is a difference of two quantities that are each perturbed by an artifact larger than the difference. Fixing it requires the exact cover-set sum — which costs, at every ambiguous byte boundary, one forward pass per branch, and that cost is precisely what the union-vocabulary approximation was introduced to avoid.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*