---
id: 21-factuality/agentic-fabricated-tool-observations
title: "Agentic Tool Use: Fabricated Tool Outputs and Observations"
topic: 21-factuality
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Agentic Tool Use: Fabricated Tool Outputs and Observations

> **Topic:** Hallucination & Factuality · **ID:** `21-factuality/agentic-fabricated-tool-observations` · **Status:** open

## 1. Problem Statement

A tool-using agent alternates between *actions* (calls to an external tool) and *observations* (whatever the harness returns). The failure of interest: the agent asserts an observation that the environment did not produce. Four distinguishable modes:

1. **Phantom call.** The final answer attributes a fact to a tool invocation that never appears in the execution log.
2. **Observation forgery.** The model itself emits tokens in the harness's observation format, continuing past its stop token instead of yielding to the runtime.
3. **Ungrounded report.** The call ran, the result returned, and the model's summary is not entailed by it — numbers changed, absent fields invented, empty result narrated as a positive finding.
4. **Error laundering.** The tool returned a non-zero exit, a timeout, a 404, or an empty set, and the agent reports task success.

The three variants differ sharply in difficulty.

- **Measurement variant.** Given a trajectory and the harness's execution log, decide whether each asserted observation is grounded. Modes 1–2 are decidable by string/log comparison. Mode 3 reduces to an entailment judgment against a retrieved context and is open. Mode 4 requires knowing whether the tool's return signals failure, which is tool-specific.
- **Method variant.** Train or scaffold an agent whose fabrication rate falls without a matching drop in task completion. Trivially solvable at rate zero by an agent that refuses to act, so any claim must be reported jointly with success rate.
- **Theory variant.** Bound the fabrication rate achievable by a calibrated next-token predictor operating in a partially observable loop where its own prior outputs re-enter the context.

Solved would mean: a public benchmark where the top system holds ungrounded-report rate below some fixed $\epsilon$ across held-out tool families it was not tuned on, at unchanged task success. No such result exists.

## 2. Formal Setting

A trajectory of horizon $T$:
$$\tau = (x, a_1, o_1, a_2, o_2, \dots, a_T, o_T, y)$$
with instruction $x$, action $a_t$ (tool name plus arguments), observation $o_t$, final answer $y$. The policy is $\pi_\theta(a_t \mid x, a_{<t}, o_{<t})$. The environment supplies $o_t = E(a_t, s_t)$ where $s_t$ is hidden environment state.

**Ground-truth channel.** The harness records the true execution log $L = \{(a_t^\star, o_t^\star)\}_{t=1}^{T^\star}$ — what was actually dispatched and returned, independent of the model's context. This is the measurement's anchor and is exact for modes 1, 2, 4.

**Phantom-call rate.** Let $C(y)$ be the set of tool invocations the answer $y$ references (extracted by a parser or an annotator). Then
$$\mathrm{PCR} = \frac{|\{c \in C(y) : c \notin L\}|}{|C(y)|}.$$

**Forgery rate.** With $M_t$ the model-generated token span at step $t$ and $\mathrm{Obs}$ the harness's observation delimiter grammar,
$$\mathrm{FOR} = \Pr_\tau\big[\exists t : M_t \text{ parses as a well-formed element of } \mathrm{Obs}\big].$$

**Ungrounded-report rate.** Decompose $y$ into atomic claims $\{q_i\}$ (the FActScore construction, Min et al., EMNLP 2023). With $\models$ an entailment oracle against the union of returned observations:
$$\mathrm{UGR} = \frac{1}{|\{q_i\}|}\Big|\Big\{q_i : \bigcup_t o_t^\star \not\models q_i \ \wedge\ q_i \notin \mathcal{K}_x\Big\}\Big|,$$
where $\mathcal{K}_x$ is the set of claims permissibly answered from parametric knowledge without tool support. **$\mathcal{K}_x$ has no accepted operational definition; this is where the measurement breaks.**

**Error-laundering rate.** With $\mathrm{fail}(o_t^\star) \in \{0,1\}$ a per-tool failure predicate and $\mathrm{succ}(y)$ the agent's own success assertion:
$$\mathrm{ELR} = \Pr\big[\mathrm{succ}(y) = 1 \ \wedge\ \mathrm{fail}(o^\star_{T}) = 1\big].$$

**The operating point.** Report $(\mathrm{UGR}, \mathrm{SR})$ jointly, $\mathrm{SR}$ = task success rate; either alone is gameable.

**Assumptions, and which fail.**
- *The context contains what the harness returned.* Violated: truncation, summarization of long outputs, and context compaction in long-horizon agents mean $o_t$ in context $\neq o_t^\star$ in the log. A "fabrication" may be faithful reporting of a lossy context.
- *Observations are trustworthy.* Violated under indirect prompt injection (Zhan et al., ACL Findings 2024; Debenedetti et al., NeurIPS 2024 D&B): a faithful report of an adversarial observation is grounded but wrong.
- *Claims are atomizable and independently checkable.* Violated for aggregate claims ("no matching records exist") whose support is the absence of evidence.
- *The entailment oracle is unbiased.* Violated: LLM judges score their own family's outputs higher (self-preference).

## 3. State of the Art

**Established.** Structural separation works for the modes it targets. Enforcing the observation channel outside the sampled token stream — constrained decoding with the tool-result delimiter as a stop sequence, plus API-level typed function calling — drives forgery (mode 2) and phantom calls (mode 1) to near zero, because the model is never given the opportunity to emit those tokens. This is an engineering result, not a learned one, and it is fully ablatable: remove the stop sequence and forged observations reappear in text-completion-style ReAct loops (Yao et al., ICLR 2023).

**Established, weaker.** Grounded-summarization detection transfers from RAG. RAGTruth (Niu et al., ACL 2024) provides ~18k word-level hallucination annotations over responses conditioned on retrieved context; fine-tuned detectors on it beat prompted GPT-4 judges on the same task. Tool observations are structurally close to retrieved passages, so mode-3 detection inherits this — but no equivalent word-level annotated corpus exists over *tool* observations specifically.

**Claimed but unablated.** Self-verification / critic passes ("re-read the tool output and check your claim") are reported to cut fabrication in system cards and blog evaluations, almost never with a matched-compute control arm. An extra verification pass costs tokens; the correct control is the same token budget spent on more sampling, and that control is usually absent.

**Benchmark numbers only.** τ-bench (Yao et al., 2024) reports pass^k — the fraction of tasks solved on all $k$ independent trials — collapsing sharply as $k$ grows, which bounds reliability but does not separate fabrication from planning failure. AgentBench (Liu et al., ICLR 2024), GAIA (Mialon et al., ICLR 2024), WebArena (Zhou et al., ICLR 2024) and SWE-bench (Jimenez et al., ICLR 2024) score end-task success only; an agent that fabricates a tool result and still passes the test is scored correct. ToolEmu (Ruan et al., ICLR 2024) emulates tool execution with an LM, which by construction removes the ground-truth log this problem depends on.

## 4. What Is Known

- **Function-call correctness is not observation faithfulness.** Berkeley Function-Calling Leaderboard leaders exceed 85% on abstract syntax match for single-turn calls; that metric never inspects the returned value, so it is uninformative about UGR. Established by what the metric computes.
- **Failure compounds within a trajectory.** Hallucination snowballing (Zhang et al., ICML 2024) shows models commit to an early wrong claim and then generate consistent support; in a loop where $o_{<t}$ is model-visible context, the same mechanism applies to fabricated observations. Measured on QA-scale prompts, not on $T > 10$ agentic rollouts.
- **Stated reasoning is not the causal reason.** Turpin et al. (NeurIPS 2023) and Lanham et al. (2023) show chain-of-thought explanations can be systematically unfaithful under bias injection, at 13B–175B scale. Consequence: an agent's narration of "the tool returned X" is not evidence about what the model conditioned on.
- **Context conflicts have a measurable pull.** Longpre et al. (EMNLP 2021) show substituted entity contexts flip QA answers at high rates; Xie et al. (ICLR 2024) show models favor coherent external evidence over parametric memory but revert when evidence is fragmentary — exactly the regime of truncated tool output.
- **Some hallucination is not removable by better training.** Kalai & Vempala (STOC 2024) prove a calibrated model's error rate on facts appearing once in training is lower-bounded by roughly the singleton-fact fraction. This bounds parametric fabrication, not observation fabrication, which is conditional.

## 5. What Is Not Known

- **Methodologically blocked.** $\mathcal{K}_x$ — the set of claims an agent may legitimately state without tool support. Without it, mode 3 is undefined: "the config file is JSON" from parametric knowledge is either fine or a fabrication, depending on an unwritten convention. Every published UGR-like number silently fixes $\mathcal{K}_x$ differently, so numbers across papers are incomparable. This is the binding gap.
- **Methodologically blocked.** Attributing a fabrication to the model versus to the harness, once context compaction or output truncation is in play. Current benchmarks do not log pre-truncation observations, so the distinction is unrecoverable post hoc.
- **Empirically open.** Whether UGR grows with horizon $T$ superlinearly. Runnable today — instrument any agent harness and vary the task horizon — but no public study reports UGR as a function of $T$ at $T \in \{5, 20, 100\}$.
- **Empirically open.** Whether tool-use RL (rewarding end-task success) increases fabrication, since fabricating a plausible observation is a cheap path to a reward-passing trajectory. The obvious reward-hacking hypothesis has no clean public measurement.
- **Theoretically open.** Any lower bound on fabrication for a calibrated policy in a partially observable loop with self-generated context re-entry. The Kalai–Vempala argument does not extend; the conditional-generation analogue is unproven either way.

## 6. Why It Is Hard

The specific obstruction is **an evaluation that does not measure the thing it names, compounded by absent ground truth in the one mode that matters**.

Modes 1, 2, and 4 have exact ground truth — the execution log — and are largely solved by making the observation channel structurally unforgeable. They are also the modes benchmarks report, so the field's numbers look good. Mode 3, the surviving failure, needs an entailment judgment over $\mathcal{K}_x$, and $\mathcal{K}_x$ is not defined. So the measurable part is not the hard part, and the hard part is not measurable.

Secondary obstruction: **confounding with task success**. UGR and SR move together under most interventions. An agent that refuses uncertain actions scores UGR $= 0$ and SR near zero. Any single-number leaderboard therefore selects for the wrong behavior, and joint reporting requires a matched-SR comparison that few papers construct.

## 7. Current Research (as of 2026)

- **Structural enforcement.** Typed tool-call APIs and constrained decoding, now standard across major providers; the observation channel is runtime-owned, not model-owned. Mature.
- **Span-level grounding detectors** trained on RAGTruth-style annotation, applied to agent transcripts *(frontier — verify)*. The bottleneck is annotated tool-observation data, not detector architecture.
- **Injection-robust agent evaluation.** AgentDojo (ETH Zürich SPY Lab) and InjecAgent extend the setting to adversarial observations, where faithfulness and correctness diverge.
- **Reliability-under-repetition metrics.** τ-bench's pass^k and its successors, treating unreliability rather than average accuracy as the target *(frontier — verify)*.
- **Mechanistic training-side accounts.** Kalai, Nachum, Vempala & Zhang (2025) argue binary-scored evaluations reward confident guessing over abstention; the agentic analogue — success-only scoring rewarding fabricated observations — is stated but not measured.

## 8. Concrete Next Experiment

**Question.** Does ungrounded reporting grow with horizon, independent of task difficulty?

**Scale.** 300 tasks in a fully logged sandbox (filesystem, HTTP, SQL over a fixed snapshot), stratified into $T \approx 5$, $20$, $80$ tool calls, difficulty held constant by construction (identical subtask types, varied only in count). Three frontier models, 5 seeds each: 4,500 trajectories. Full log of every returned observation **before** truncation or compaction.

**Annotation.** Decompose final answers into atomic claims (FActScore procedure). Fix $\mathcal{K}_x$ explicitly and publish it: a claim is grounded only if entailed by some $o_t^\star$; parametric claims count as ungrounded. Human-label a 600-claim stratified subsample; report LLM-judge agreement (Cohen's $\kappa$) against it and use the judge only if $\kappa > 0.7$.

**Control arm.** The same models on the same trajectories with **all observations pasted verbatim into the final-answer prompt, untruncated** — a single-turn grounded-summarization task with identical evidence. This separates loop-induced fabrication from ordinary summarization error.

**Deciding number.** $\Delta = \mathrm{UGR}(T{=}80) - \mathrm{UGR}(T{=}5)$ in the agentic arm, minus the same difference in the control arm. $\Delta > 5$ percentage points with a bootstrap 95% CI excluding zero establishes horizon-induced fabrication as a real, separate phenomenon. $\Delta \approx 0$ says the problem is ordinary summarization error and the agentic framing adds nothing.

## 9. Key References

- **[Foundational]** Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR, 2023. — arXiv:2210.03629
- **[Foundational]** Schick, Dwivedi-Yu, Dessì, Raileanu, Lomeli, Zettlemoyer, Cancedda, Scialom. *Toolformer: Language Models Can Teach Themselves to Use Tools.* NeurIPS, 2023. — arXiv:2302.04761
- **[SOTA]** Niu, Wu, Zhu, Xu, Shum, Zhong, Song, Zhang. *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.* ACL, 2024. — arXiv:2401.00396
- **[SOTA]** Yao, Shinn, Razavi, Narasimhan. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[SOTA]** Debenedetti, Zhang, Balunović, Beurer-Kellner, Fischer, Tramèr. *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.13352
- **[Theory]** Kalai, Vempala. *Calibrated Language Models Must Hallucinate.* STOC, 2024. — arXiv:2311.14648
- **[Method]** Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, Hajishirzi. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP, 2023. — arXiv:2305.14251
- **[Evidence]** Turpin, Michael, Perez, Bowman. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* NeurIPS, 2023. — arXiv:2305.04388
- **[Evidence]** Zhang, Diao, Lin, Ho, Ye, Wang, Zhang. *How Language Model Hallucinations Can Snowball.* ICML, 2024. — arXiv:2305.13534
- **[Evidence]** Xie, Zhang, Chen, Lou, Su. *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts.* ICLR, 2024. — arXiv:2305.13300
- **[Related]** Zhan, Liang, Ying, Kang. *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents.* ACL Findings, 2024. — arXiv:2403.02691
- **[Related]** Ruan, Dong, Wang, Pitis, Zhou, Ba, Dubois, Maddison, Hashimoto. *Identifying the Risks of LM Agents with an LM-Emulated Sandbox.* ICLR, 2024. — arXiv:2309.15817
- **[Survey]** Huang, Yu, Ma, Zhong, Feng, Wang, Chen, Peng, Feng, Qin, Liu. *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* ACM TOIS, 2025. — arXiv:2311.05232

## 10. Worked Example

**Task.** "How many orders in the `orders` table shipped late last quarter? Fix the query if it errors."

**Log (ground truth).**

```
a_1  sql("SELECT COUNT(*) FROM orders WHERE ship_date > due_date
          AND quarter='2026Q1'")
o_1  ERROR: no such column: quarter
a_2  sql("PRAGMA table_info(orders)")
o_2  id|order_date|due_date|ship_date|status     # 5 columns, no quarter
a_3  sql("SELECT COUNT(*) FROM orders WHERE ship_date > due_date
          AND order_date >= '2026-01-01' AND order_date < '2026-04-01'")
o_3  1,284
```

**Answer.** "1,284 orders shipped late in Q1 2026 — about 7% of the 18,400 orders in the quarter. The initial query failed because the schema uses `order_date` rather than a quarter column."

**Scoring.** Atomic claims: (i) 1,284 late orders — grounded in $o_3$. (ii) 18,400 orders in quarter — **no tool call ever returned a denominator**; the model divided a plausible-looking total into 1,284 to produce 7%. Ungrounded. (iii) 7% — derived from (ii), ungrounded. (iv) schema lacks a quarter column — grounded in $o_2$.

$\mathrm{UGR} = 2/4 = 0.50$. $\mathrm{PCR} = 0$ — every referenced call is in the log. $\mathrm{FOR} = 0$ — no forged observation span. $\mathrm{ELR} = 0$ — the final call succeeded. $\mathrm{SR} = 1$ under any benchmark that checks the headline count.

**The obstruction, visible.** Three of the four exactly-measurable metrics are clean and the task scores as a pass. The one failure sits entirely in mode 3, and even there the verdict is contested: if $\mathcal{K}_x$ is defined as "claims requiring the current database state", claim (ii) is a fabrication; if it is defined as "claims the user asked about", (ii) is unrequested colour and arguably out of scope. Move $\mathcal{K}_x$ one notch and UGR on this single trajectory moves from 0.50 to 0.00. No public benchmark states which notch it uses.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*