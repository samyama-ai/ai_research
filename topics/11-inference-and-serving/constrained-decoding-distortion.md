---
id: 11-inference-and-serving/constrained-decoding-distortion
title: "Constrained Decoding Distortion of Model Distribution"
topic: 11-inference-and-serving
status: partially-solved
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Constrained Decoding Distortion of Model Distribution

> **Topic:** Inference & Serving · **ID:** `11-inference-and-serving/constrained-decoding-distortion` · **Status:** partially-solved

## 1. Problem Statement

Constrained (grammar-guided) decoding forces a language model's output into a formal language $L$ — a JSON Schema, a regex, a CFG for SQL — by masking, at each step, the tokens that cannot extend the current prefix to a member of $L$. The intended semantics is conditioning: sample from $p(\cdot \mid x \in L)$. The implemented semantics is greedy local renormalization, which is a different distribution. The gap is the **distortion**.

Three variants, of different difficulty:

- **Measurement.** Given $p$, $L$, and a constrained sampler $q$, estimate a divergence $D(q \,\|\, p(\cdot\mid L))$ without being able to enumerate $L$. Currently the hardest part.
- **Method.** Build a sampler that is exact (or consistent) for $p(\cdot\mid L)$ at serving latency — sub-millisecond per token, batched, prefix-cache-compatible.
- **Theory.** Characterize when local masking is exact, and bound the distortion as a function of the grammar and the model's mass on $L$.

Solved means: a serving-grade decoder whose output distribution is provably $p(\cdot \mid x \in L)$ (or converges to it with a stated rate), at no more than a small constant factor of unconstrained throughput, plus a measurable statistic that certifies the claim on a real schema.

## 2. Formal Setting

Let $V$ be the token vocabulary, $V^*$ the strings over it, and $p$ an autoregressive model with $p(x) = \prod_{t} p(x_t \mid x_{<t})$ over sequences terminated by EOS. Let $L \subseteq \Sigma^*$ be the target language over *characters*, and $\tilde L \subseteq V^*$ its preimage under detokenization.

The target is the conditional
$$p_L(x) = \frac{p(x)\,\mathbf{1}[x \in \tilde L]}{Z}, \qquad Z = \Pr_{x\sim p}[x \in \tilde L].$$

$Z$ is measured as the empirical rate at which unconstrained sampling produces a valid string — for a strict JSON Schema and an 8B model this is often $10^{-2}$–$10^{-1}$, and can be $<10^{-3}$; that is exactly the regime where rejection sampling dies.

Local masking defines, for a prefix $x_{<t}$, the live set $M(x_{<t}) = \{v : \exists\, s,\ x_{<t}vs \in \tilde L\}$ and
$$q(x_t \mid x_{<t}) = \frac{p(x_t\mid x_{<t})\,\mathbf{1}[x_t \in M(x_{<t})]}{\sum_{v \in M(x_{<t})} p(v \mid x_{<t})}.$$

Write the **expected future grammar mass** $A(x_{<t}) = \Pr_{s\sim p(\cdot\mid x_{<t})}[x_{<t}s \in \tilde L]$. Then
$$p_L(x) \propto q(x)\prod_{t} \frac{A(x_{\le t})}{\;\mathbb{E}_{v\sim q}\,A(x_{<t}v)}\quad\text{— i.e. } q = p_L \text{ iff } A \text{ is constant along all live continuations.}$$
So local masking is exact only when the constraint is *prefix-uninformative*; otherwise it over-weights prefixes that are locally cheap and globally rare. $A$ is not computable in closed form — it is the quantity every method approximates.

Measured quantities: distortion as $\mathrm{KL}(q \| p_L)$ estimated by importance weights $w(x) = 1/\prod_t(\cdot)$; task accuracy under $L$ versus free-form plus a parser; per-token mask-construction latency (µs) and grammar-compilation latency (ms); the **canonicality gap** — mass placed on token sequences that no BPE tokenizer would emit for the same string.

Assumptions known to be violated in practice: (i) that $\tilde L$ is exactly recognizable at token granularity — BPE token boundaries cross grammar boundaries, so masks are computed on a token-level automaton that is an over- or under-approximation; (ii) that each string has one tokenization — it does not, and masking makes non-canonical tokenizations reachable, a distribution the model was never trained on; (iii) that $p$ is unchanged by prompt format — instructing "reply in JSON" shifts $p$ itself, confounding any before/after comparison; (iv) that EOS handling is uniform across engines — it is not.

## 3. State of the Art

**Systems / empirical SOTA.** Outlines (Willard & Louf, 2023) precompiles a regex-to-FSM index giving $O(1)$ amortized mask lookup. XGrammar (Dong et al., 2024) splits the vocabulary into context-independent and context-dependent tokens with a persistent pushdown-automaton stack cache; it *reports* up to two orders of magnitude lower grammar-processing overhead and near-zero end-to-end overhead when overlapped with GPU compute — these are the authors' benchmark numbers on their own harness, not independently ablated. DOMINO (Beurer-Kellner et al., ICML 2024) does non-invasive, subtoken-aware constraint enforcement with claimed near-zero overhead. llguidance, vLLM's and SGLang's structured-output backends are the deployed descendants. Established: mask construction is no longer the bottleneck. Not established: that any of these sample $p_L$.

**Theory / correctness SOTA.** Grammar-Aligned Decoding (Park, Wang, Berg-Kirkpatrick, Polikarpova, D'Antoni, NeurIPS 2024) proves local masking is *not* $p_L$ and gives ASAp, which maintains an under-approximation of $A$ from observed trajectories and provably converges to $p_L$ from above — but needs many samples per output. Sequential Monte Carlo (Lew et al., 2023; Loula et al., ICLR 2025) gives a consistent estimator of $p_L$ with particle reweighting against expensive potentials; adaptive weighted rejection sampling (Lipkin et al., 2025) gives exact per-token sampling from the *locally* masked distribution without materializing the full mask. Koo, Liu & He (COLM 2024) formalize the tokenization/automaton mismatch and give automata constructions that fix detokenization soundness. GUARD (Khalifa et al., ICLR 2025) frames constraint satisfaction plus KL-minimality as an explicit objective.

**Contested empirically.** Tam et al. (EMNLP 2024 Industry) report format restriction degrades reasoning (notably GSM8K-style tasks). The .txt team's rebuttal argues the drop is a prompting and parsing artifact and reports the opposite sign. Both are benchmark-number claims with different prompts; neither has been reproduced by a third party at matched decoding settings.

## 4. What Is Known

- **Local masking $\ne$ conditioning.** Proven, with explicit counterexample grammars (GAD, NeurIPS 2024). ASAp lowers the distortion monotonically on small CFGs with 7B-class models; the gain is measured in likelihood-under-$p$ of the sampled program, on hundreds of samples, not at serving scale.
- **The overhead is solvable.** Grammar mask computation has moved from tens of milliseconds per token (naive per-token CFG check over a 128K vocabulary) to microseconds via FSM/PDA caching (Outlines 2023; XGrammar 2024).
- **Constraints raise syntactic validity to ~100%.** Grammar-constrained decoding takes JSON-Schema conformance from roughly 50–90% (model- and schema-dependent, worse for small models) to essentially 100% by construction; JSONSchemaBench-style evaluations show the residual failures are coverage bugs in the schema-to-automaton compiler, not model errors.
- **Constraints help small models on structured extraction.** Geng et al. (EMNLP 2023 Findings) show grammar-constrained decoding improving closed information extraction and entity disambiguation without finetuning, at up to ~10B scale.
- **Tokenization misalignment is real and fixable in principle.** Koo et al. (COLM 2024) exhibit strings in $L$ that a token-level mask makes unreachable.

## 5. What Is Not Known

- **Methodologically blocked:** there is no accepted estimator of $\mathrm{KL}(q\|p_L)$ for a real schema and a real model. $Z$ is tiny, $A$ is intractable, and importance weights have unbounded variance. Every published "distortion" number is a proxy (mean log-likelihood, task accuracy) whose relation to the divergence is unquantified.
- **Empirically open:** whether the observed accuracy changes under JSON/CFG constraints are caused by *distribution distortion* or by *prompt-induced shift in $p$*. The disentangling experiment (§8) is runnable today on 8B–70B models and has not been run at matched prompts, matched temperature, and matched parsers.
- **Theoretically open:** a distortion bound in terms of grammar structure. No known result of the form $\mathrm{KL}(q\|p_L) \le f(\text{branching}, \text{depth}, \log 1/Z)$. Also open: whether an exact $p_L$ sampler with $O(1)$ amortized per-token cost exists for context-free $L$, or whether exactness provably costs a factor in the number of model calls.
- **Empirically open:** whether non-canonical tokenization, not the conditioning gap, is the dominant error term.

## 6. Why It Is Hard

**Confounded measurement plus absent ground truth.** The reference distribution $p_L$ cannot be sampled: rejection sampling needs $\approx 1/Z$ draws, and $Z < 10^{-3}$ on realistic schemas makes a single ground-truth sample cost thousands of full generations. So the only object you could compare against is the one you cannot obtain. Every substitute — accuracy, validity rate, perplexity — is a projection through a task, and the task itself changes when you change the output format, so the control arm is not the same experiment.

Second obstruction: **non-identifiability of the cause**. A drop in GSM8K under JSON output is consistent with (a) local-masking distortion, (b) the model reasoning worse when it must interleave syntax, (c) the "answer in JSON" instruction moving $p$, (d) the free-form baseline's parser silently scoring differently. These four are confounded in every published comparison.

Third: **exactness costs samples**. ASAp and SMC both buy correctness with repeated trajectories, which is exactly what a serving system with a latency SLO cannot spend.

## 7. Current Research (as of 2026)

- Automata-correct compilers: Koo/Liu/He line at MIT-IBM; llguidance and XGrammar (CMU/NVIDIA/MLC) folding subtoken and canonicality handling into the mask engine.
- Probabilistic-programming samplers: MIT ProbComp (Lew, Loula, Lipkin, Mansinghka, O'Donnell) on SMC and adaptive weighted rejection sampling — consistent estimators with controllable particle budgets.
- Correct-by-construction decoding: UCSD/CMU GAD line, extending ASAp toward amortized $A$ estimation with a learned prefix value head *(frontier — verify)*.
- Serving integration: vLLM and SGLang structured-output backends measuring the throughput cost of grammar masks under continuous batching; the open engineering question is prefix-cache invalidation when the automaton state differs across sequences sharing a prefix *(frontier — verify)*.
- Alignment-side alternative: constraint-aware finetuning so $Z \to 1$ and masking becomes near-vacuous.

## 8. Concrete Next Experiment

**Question.** Is measured accuracy loss under constrained decoding caused by distributional distortion, or by prompt/format shift?

**Scale.** One open-weights 8B instruct model (e.g. Llama-3.1-8B-Instruct) plus one ~70B for scale sanity. 200 JSON Schemas from a public schema benchmark, 500 samples each, temperature 1.0, one A100/H100-node.

**Arms.**
1. *Constrained:* schema-instructed prompt + grammar mask (XGrammar backend).
2. *Control (ground truth):* the **same** prompt, unconstrained, with rejection to $\tilde L$ — keep only valid samples. This is an exact $p_L$ sampler; restrict to the schema subset where the measured $Z \ge 10^{-2}$ so 500 accepted samples cost $\le 5\times10^4$ generations.
3. *Ablation:* free-form prompt, unconstrained, parsed leniently — isolates prompt shift.

**Deciding number.** On the $Z \ge 10^{-2}$ subset, the total-variation distance between arm 1 and arm 2 over the empirical distribution of *schema-level answer values* (a finite, comparable support). **If $\mathrm{TV} \le 0.05$ on the median schema, local masking is empirically benign and the field should spend its effort on tokenization and prompting, not on exact samplers. If $\mathrm{TV} \ge 0.2$, the conditioning gap is the dominant error and ASAp/SMC-class methods are load-bearing.** Report TV as a function of $Z$ — the extrapolation to $Z \ll 10^{-2}$ is the interesting curve.

## 9. Key References

- **[Foundational]** Brandon T. Willard, Rémi Louf. *Efficient Guided Generation for Large Language Models.* Preprint, 2023. — arXiv:2307.09702
- **[Foundational]** Saibo Geng, Martin Josifoski, Maxime Peyrard, Robert West. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning.* EMNLP 2023 (Findings).
- **[SOTA — theory]** Kanghee Park, Jiayu Wang, Taylor Berg-Kirkpatrick, Nadia Polikarpova, Loris D'Antoni. *Grammar-Aligned Decoding.* NeurIPS 2024. — arXiv:2405.21047
- **[SOTA — systems]** Yixin Dong et al. *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models.* Preprint, 2024. — arXiv:2411.15100
- **[SOTA — systems]** Luca Beurer-Kellner, Marc Fischer, Martin Vechev. *Guiding LLMs the Right Way: Fast, Non-Invasive Constrained Generation.* ICML 2024.
- **[SOTA — sampling]** João Loula et al. *Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo.* ICLR 2025.
- **[SOTA — sampling]** Benjamin Lipkin et al. *Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling.* Preprint, 2025.
- **[Foundational — tokenization]** Terry Koo, Frederick Liu, Luheng He. *Automata-Based Constraints for Language Model Decoding.* COLM 2024.
- **[Contested empirical]** Zhi Rui Tam et al. *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models.* EMNLP 2024, Industry Track.
- **[Objective formulation]** Muhammad Khalifa et al. *Guaranteed Generation from Large Language Models.* ICLR 2025.
- **[Benchmark]** Saibo Geng et al. *Generating Structured Outputs from Language Models: Benchmark and Studies* (JSONSchemaBench). Preprint, 2025.

## 10. Worked Example

Grammar: `S → "a" S | "b"`, i.e. $L = \{a^n b\}$. Take a model that, at every step, puts $p(a)=0.1$, $p(b)=0.1$, and $0.8$ on tokens outside the grammar (punctuation, EOS, whatever).

Conditional target. Every string $a^nb$ has $p(a^nb) = 0.1^{n+1}$, so
$$p_L(a^nb) = \frac{0.1^{n+1}}{\sum_{m\ge0}0.1^{m+1}} = 0.9 \cdot 0.1^{n},$$
a geometric with continuation probability $0.1$. Mean length $\approx 1.11$ tokens.

Local masking. At each step the live set is $\{a,b\}$ with masses $0.1,0.1$, renormalizing to $0.5/0.5$. So $q(a^nb) = 0.5^{n+1}$ — a geometric with continuation probability $0.5$. Mean length $2$.

$$\mathrm{KL}(q\|p_L) = \sum_n 0.5^{n+1}\log\frac{0.5^{n+1}}{0.9\cdot 0.1^{n}} \approx 0.80\ \text{nats}.$$

The masked sampler produces strings roughly **twice as long** as conditioning would, and the divergence does not shrink as you add data or scale the model — it is a property of the sampler.

Now make the obstruction visible. $Z = \Pr[x \in L] = \sum_n 0.1^{n+1} \approx 0.111$ here, so rejection sampling is cheap and you can measure the gap directly. Replace the grammar with a JSON Schema of depth 4 and 12 required fields; the same model puts $Z \approx 10^{-3}$ on it. To get 500 ground-truth samples you need $5\times10^5$ generations — about 3 GPU-days at 8B. And the importance weights that would let you skip that have a ratio $q/p_L$ spanning $0.9/0.5 \cdot (0.1/0.5)^n$ — exponentially small in depth, so the effective sample size of a 500-draw weighted estimator collapses to single digits. **The distortion is easy to compute where it does not matter and infeasible to measure where it does.** That, not the mask latency, is the open problem.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*