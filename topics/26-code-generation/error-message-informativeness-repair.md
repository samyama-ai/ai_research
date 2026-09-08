---
id: 26-code-generation/error-message-informativeness-repair
title: "Error-Message Informativeness for Automated Repair"
topic: 26-code-generation
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Error-Message Informativeness for Automated Repair

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/error-message-informativeness-repair` · **Status:** methodologically-blocked

## 1. Problem Statement

A repair loop feeds a failing program plus a diagnostic message — compiler error, stack trace, assertion failure, test diff, linter output — back to a model and asks for a fix. Everyone builds these loops. Nobody can say how much of the fix comes from the message.

- **Input:** a program $p$ that fails, the failing context $c$ (spec, tests, execution trace), and a diagnostic message $m$ produced by some tool $T$.
- **Output:** a scalar $\mathcal{I}(m)$ — the informativeness of $m$ for repair — plus a ranking over candidate message designs.
- **Decision predicate:** given two diagnostic tools $T_1, T_2$, decide whether $T_1$'s messages are more informative than $T_2$'s for a stated repairer population, independent of that population's prior knowledge of the bug.

Three variants, three difficulties:

- **Measurement (the blocked one).** Define $\mathcal{I}$ so it is not just "the repair rate of one model on one benchmark". Today every reported number is confounded by the model's ability to re-derive the diagnosis from the program alone.
- **Method.** Generate or rewrite messages that maximise $\mathcal{I}$. Runnable now; optimising an undefined target.
- **Theory.** Bound the achievable repair rate as a function of the information a message carries about the true edit. No such bound exists.

Solving it means: an estimator of $\mathcal{I}$ that (a) is comparable across repairer models, (b) separates message content from program-recoverable content, and (c) predicts held-out repair gains on unseen bug distributions.

## 2. Formal Setting

Let $\mathcal{P}$ be programs, $\mathcal{S}$ specs, $\mathcal{M}$ messages. A bug instance is $(p, s, e^\star)$: buggy program, spec, and a *ground-truth edit* $e^\star$ (the maintainer's patch or the injected mutation, inverted). A diagnostic tool is $T: \mathcal{P}\times\mathcal{S} \to \mathcal{M}$. A repairer is a policy $\pi(\hat p \mid p, s, m)$.

**Correctness, as measured.** $\mathrm{ok}(\hat p) = 1$ iff $\hat p$ passes the held-out test suite $\mathcal{T}_{\text{hid}}$ disjoint from any tests shown to $\pi$. Everything below inherits the gap between $\mathrm{ok}$ and semantic correctness.

**Repair rate.**
$$R(\pi, T) = \mathbb{E}_{(p,s)\sim \mathcal{D}}\ \mathbb{E}_{\hat p \sim \pi(\cdot\mid p,s,T(p,s))}\big[\mathrm{ok}(\hat p)\big].$$

**Naive informativeness (what the literature reports).** Difference against a message-free arm:
$$\Delta_\pi(T) = R(\pi, T) - R(\pi, \varnothing).$$

**Ablation-corrected informativeness.** Let $\rho$ be a redaction that destroys the message's semantic content but preserves its length, format, and token statistics (e.g. shuffle identifiers to fresh names, permute line numbers, keep the template). Then
$$\mathcal{I}_\pi(T) = R(\pi, T) - R(\pi, \rho \circ T),$$
which removes the "a message is present, so try again" effect from the "this message says where the bug is" effect.

**Policy-free informativeness.** The quantity one actually wants is the mutual information between message and true edit given the program:
$$\mathcal{I}^\star(T) = I\big(M ; E^\star \mid P, S\big), \qquad M = T(P,S),$$
measured in nats. Only the *incremental* information matters: content a strong reader could recompute from $p$ alone contributes nothing. Operationally this is estimated as a compression gap with a fixed reference model $q$:
$$\hat{\mathcal{I}}^\star(T) = \frac{1}{n}\sum_{i=1}^{n} \Big[-\log q(e^\star_i \mid p_i, s_i) + \log q(e^\star_i \mid p_i, s_i, m_i)\Big].$$
Units: nats per bug. This is the one measurable definition that does not move when you swap repairers — but it moves when you swap $q$.

**Assumptions, and which are violated.**

| Assumption | Status |
|---|---|
| $\mathrm{ok}$ approximates correctness | **Violated.** Test-suite overfitting is documented (Smith et al., FSE 2015; Qi et al., ISSTA 2015). |
| $e^\star$ is unique | **Violated.** Many bugs admit non-equivalent acceptable patches. |
| $\mathcal{D}$ is fresh w.r.t. $\pi$'s pretraining | **Violated.** Defects4J, HumanEval, and pre-2023 GitHub issues are in-corpus. |
| Redaction $\rho$ preserves nuisance structure | **Unverified.** No standard $\rho$ exists; this is the core methodological hole. |
| Single-round repair | **Violated by practice.** Agent loops re-invoke $T$; $\mathcal{I}$ compounds across turns. |

## 3. State of the Art

**Established (with ablations).**

- Olausson et al., *Is Self-Repair a Silver Bullet for Code Generation?* (ICLR 2024, arXiv:2306.09896). Under a fixed sampling budget, self-repair gives little or no gain over drawing more i.i.d. samples for GPT-3.5 and GPT-4 on HumanEval/APPS. Substituting an expert human's feedback for GPT-4's own feedback raised the count of repaired programs by ~57%. This is the strongest evidence that the *feedback channel*, not the edit capability, is the bottleneck — and it was obtained with a human, not a metric.
- Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* (ICLR 2024, arXiv:2310.01798). Without external signal, self-correction degrades accuracy. Cleanly ablated, and it generalises the point: informativeness must come from outside the model.

**Claimed but unablated.**

- Chen et al., *Teaching Large Language Models to Self-Debug* (ICLR 2024, arXiv:2304.05128) reports gains of roughly 2–12 accuracy points on Spider, TransCoder, and MBPP with execution feedback and self-explanation. No redaction control, so the split between "the error message told the model something" and "another attempt with a longer prompt" is unmeasured.
- Reflexion (Shinn et al., NeurIPS 2023) and Self-Refine (Madaan et al., NeurIPS 2023) report benchmark deltas; neither isolates message content from retry effects.

**Benchmark-number-only.** SWE-bench resolve rates (Jimenez et al., ICLR 2024, arXiv:2310.06770) and Agentless (Xia et al., arXiv:2407.01489; 27.3% on SWE-bench Lite at ~$0.34/issue with GPT-4o) are end-to-end system scores. They contain no per-message attribution.

**Human-side SOTA.** Becker et al., *Compiler Error Messages Considered Unhelpful* (ITiCSE-WGR 2019) is the reference survey. Denny et al. (ITiCSE 2014) found enhanced syntax error messages produced no significant improvement in novice error rates; Becker (Computer Science Education, 2016) found significant reductions with a different enhancement and cohort. Two decades, contradictory results, no shared metric — the same failure now recurring for machines.

## 4. What Is Known

- **Feedback quality dominates edit capability.** GPT-4 with expert human feedback repaired ~57% more programs than with its own feedback, at HumanEval/APPS scale (~164 and ~5,000 problems respectively).
- **Self-generated feedback has near-zero marginal value on reasoning tasks**, measured at GSM8K/HotpotQA scale on GPT-3.5 and GPT-4.
- **Execution feedback beats no feedback**, but by single-digit to low-double-digit points, at MBPP (500 problems) and Spider scale.
- **Repair with test failures beats repair without**, established across APR literature since Xia, Wei & Zhang (ICSE 2023) on Defects4J (395 bugs, v1.2).
- **Plausible ≠ correct.** Qi et al. (ISSTA 2015) found the large majority of patches produced by then-current generate-and-validate systems on their benchmarks were test-suite-overfitting, not correct. Any $\mathcal{I}$ built on $\mathrm{ok}$ inherits this.
- **Human studies do not transfer.** Readability factors identified for novices (Denny et al., CHI 2021) have never been shown to predict LLM repair gains.

## 5. What Is Not Known

- **Methodologically blocked (primary).** No accepted redaction control $\rho$. Without one, every published $\Delta_\pi(T)$ conflates message content with prompt length, retry count, and the model's own re-derivation of the fault. The measurement does not name what it measures.
- **Methodologically blocked (secondary).** $\hat{\mathcal{I}}^\star$ depends on the reference model $q$. Whether the *ranking* of tools is $q$-invariant is untested.
- **Empirically open.** Does $\hat{\mathcal{I}}^\star(T)$ measured with a small $q$ (e.g. 7B) predict repair-rate gains for a frontier repairer on a held-out bug distribution? Runnable today; unrun.
- **Empirically open.** Rust's diagnostics are widely believed more informative than C++'s. No controlled cross-language measurement exists on matched bug distributions.
- **Theoretically open.** No bound of the form $R(\pi,T) \le f(\mathcal{I}^\star(T), H(E^\star \mid P,S))$ — no rate-distortion statement for repair.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability under a competent reader**. A strong repairer can recompute most of a diagnostic from the source. When a stack trace names line 42 and the model would have inspected line 42 anyway, the message's measured contribution is zero — not because it is uninformative, but because it is redundant *with respect to that reader*. Informativeness is therefore a property of the (tool, reader) pair, not the tool. Two consequences:

1. **The ordering is reader-dependent.** A message that helps a 7B model may be pure redundancy for a frontier model, so tool rankings can invert across scale. No published study reports the same message ablation at two model scales.
2. **The obvious fix — measure against a fixed reference $q$ — replaces reader-dependence with $q$-dependence**, which is unaudited.

Compounding these: absent ground truth ($e^\star$ is not unique), a proxy objective that is known to be gamed (test-suite overfitting), and benchmark contamination that lets $\pi$ succeed with $m$ redacted.

## 7. Current Research (as of 2026)

- **Agent-loop diagnostic design.** SWE-agent-style work on agent-computer interfaces (Yang et al., NeurIPS 2024) treats message formatting as a design variable and reports end-to-end deltas. Attribution to individual message fields remains absent.
- **Compiler-side.** Rust and Elm diagnostics remain the practitioner reference for informativeness; no ML-facing evaluation.
- **LLM-rewritten error messages for humans.** Leinonen et al. (SIGCSE 2023) showed LLM-enhanced programming error messages are rated more readable by novices; the machine-repair analogue is untested. *(frontier — verify)*
- **Information-theoretic prompt attribution** — pointwise-information estimators applied to feedback channels rather than training data. *(frontier — verify)*
- **Verifier-generated feedback** (proof obligations, type errors from refinement checkers) as a higher-$\mathcal{I}^\star$ channel than test failures. *(frontier — verify)*

## 8. Concrete Next Experiment

**The redaction-ladder ablation.**

- **Scale.** 500 bugs: SWE-bench Verified (500 instances), plus a 500-bug mutation set injected into post-cutoff repositories to control contamination. Four repairers spanning 7B / 32B / frontier-small / frontier-large. 5 samples per condition. ≈ 40k repair attempts.
- **Conditions (the ladder).** For each bug: (0) no message; (1) redacted message — same template, same length, identifiers renamed, line numbers permuted; (2) message with location only; (3) location + error class; (4) full message. Fixed token budget across arms, padded.
- **Control arm.** Condition (1). This is what the field is missing: it holds prompt length, format, and retry count fixed while destroying content.
- **Deciding number.** $\mathcal{I}_\pi = R(\text{full}) - R(\text{redacted})$, in resolve-rate points, reported per model scale. **If $\mathcal{I}_\pi < 3$ points for the frontier repairer while exceeding 10 points for the 7B repairer, message informativeness is a small-model phenomenon and diagnostic-design effort should move to verifier channels.** If $\mathcal{I}_\pi > 10$ points at every scale, error-message design is a first-class lever and the $\hat{\mathcal{I}}^\star$ estimator becomes worth calibrating.
- **Secondary readout.** Spearman correlation between $\hat{\mathcal{I}}^\star$ computed with a 7B reference $q$ and per-bug $\mathcal{I}_\pi$ for the frontier repairer. $\rho > 0.5$ would make cheap, repairer-free message evaluation viable.

## 9. Key References

- **[Foundational]** Brett A. Becker et al. *Compiler Error Messages Considered Unhelpful: The Landscape of Text-Based Programming Error Message Research.* ITiCSE-WGR, 2019.
- **[Foundational]** Paul Denny, Andrew Luxton-Reilly, Dave Carpenter. *Enhancing Syntax Error Messages Appears Ineffectual.* ITiCSE, 2014.
- **[SOTA]** Theo X. Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, Armando Solar-Lezama. *Is Self-Repair a Silver Bullet for Code Generation?* ICLR, 2024. — arXiv:2306.09896
- **[SOTA]** Xinyun Chen, Maxwell Lin, Nathanael Schärli, Denny Zhou. *Teaching Large Language Models to Self-Debug.* ICLR, 2024. — arXiv:2304.05128
- **[SOTA]** Jie Huang et al. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR, 2024. — arXiv:2310.01798
- **[Benchmark]** Carlos E. Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770
- **[Benchmark]** Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, Lingming Zhang. *Agentless: Demystifying LLM-based Software Engineering Agents.* 2024. — arXiv:2407.01489
- **[Established]** Edward K. Smith, Earl T. Barr, Claire Le Goues, Yuriy Brun. *Is the Cure Worse Than the Disease? Overfitting in Automated Program Repair.* ESEC/FSE, 2015.
- **[Established]** Zichao Qi, Fan Long, Sara Achour, Martin Rinard. *An Analysis of Patch Plausibility and Correctness for Generate-and-Validate Patch Generation Systems.* ISSTA, 2015.
- **[Related]** Juho Leinonen et al. *Using Large Language Models to Enhance Programming Error Messages.* SIGCSE, 2023.
- **[Survey]** Chunqiu Steven Xia, Yuxiang Wei, Lingming Zhang. *Automated Program Repair in the Era of Large Pre-trained Language Models.* ICSE, 2023.

## 10. Worked Example

A Python bug: an off-by-one in a binary search, `hi = len(a)` should be `hi = len(a) - 1`, failing one test with `IndexError: list index out of range` at line 7.

Two candidate messages:

- $m_1$: `IndexError: list index out of range` (no location).
- $m_2$: `IndexError at bsearch.py:7 in loop guard; a has len 5, index 5 requested; hi initialised at line 3.`

Run a 7B repairer, 20 samples per arm, on this single bug:

| Arm | Resolve rate |
|---|---|
| no message | 6/20 |
| redacted $m_2$ (same template, renamed vars, line numbers shifted) | 7/20 |
| $m_1$ | 11/20 |
| $m_2$ | 17/20 |

$\Delta = 17/20 - 6/20 = +0.55$; ablation-corrected $\mathcal{I} = 17/20 - 7/20 = +0.50$. The redaction arm cost 5 points of the naive delta — small here.

Now the frontier repairer:

| Arm | Resolve rate |
|---|---|
| no message | 19/20 |
| redacted $m_2$ | 19/20 |
| $m_2$ | 20/20 |

$\mathcal{I} = 0.05$, with a 95% Wilson interval spanning zero. The message is not uninformative — it still names the exact line — but the frontier model reads a 12-line binary search and locates the off-by-one unaided. **The obstruction is visible:** $m_2$ has fixed content and fixed $\mathcal{I}^\star$, yet measured informativeness collapses from 0.50 to 0.05 purely by changing the reader. Any evaluation that reports a single repair-rate delta is reporting a property of its repairer, mislabelled as a property of the diagnostic tool. Distinguishing the two needs bugs where the frontier ceiling is not already saturated — which is exactly why the deciding experiment must run at SWE-bench-Verified difficulty, not HumanEval difficulty.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*