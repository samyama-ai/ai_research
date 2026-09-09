---
id: 05-retrieval-and-agents/multi-agent-versus-single-agent-equal-compute
title: "When Multi-Agent Beats Single-Agent at Equal Compute"
topic: 05-retrieval-and-agents
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# When Multi-Agent Beats Single-Agent at Equal Compute

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/multi-agent-versus-single-agent-equal-compute` · **Status:** empirically-open

## 1. Problem Statement

Multi-agent LLM systems — an orchestrator spawning subagents, debate panels, role-specialised pipelines — are routinely reported to beat single-agent baselines. Almost every such comparison spends more inference compute on the multi-agent arm. The open problem is what survives when compute is held fixed.

**Decision predicate.** For a task distribution $\mathcal{D}$, a model family $M$, and a compute budget $C$, does there exist a multi-agent scaffold $A_{\text{multi}}$ whose expected score exceeds that of the *best* single-agent scaffold at the same budget, and if so, over which $(\mathcal{D}, C)$ region?

Three variants, with different difficulty:

- **Measurement.** Define a compute budget that is fair across scaffolds with different parallelism, model sizes, and tool-call costs. Not settled — this is the binding constraint.
- **Method.** Find a scaffold that wins at matched budget on a task family where it currently does not.
- **Theory.** Characterise the task property (decomposability, verifiability, context pressure) that predicts a positive multi-agent gap. No formal characterisation exists.

Solving it means producing accuracy-vs-compute Pareto curves, not single points, with the crossover budget $C^\*$ identified and the task property that moves it named.

## 2. Formal Setting

A task instance $x \sim \mathcal{D}$, scaffold $A$, base model family $M$, score $s(A(x), y^\*) \in [0,1]$.

**Compute, as actually measured.** Three non-equivalent meters; a claim must say which:

$$C_{\text{tok}}(A,x) = \sum_{i} \big( n^{\text{in}}_i \cdot \kappa_{m_i} + n^{\text{out}}_i \cdot \lambda_{m_i} \big)$$

summed over LLM calls $i$, where $m_i$ is the model used and $\kappa, \lambda$ are per-model input/output weights. Setting $\kappa = \lambda = 1$ gives raw tokens; setting them to list prices gives $C_{\$}$; setting them to $\propto$ active parameters gives FLOP-proportional cost. These rank scaffolds differently: a lead-plus-subagent design that runs a large orchestrator over small workers wins on $C_{\text{tok}}$ and loses on $C_{\$}$.

$$C_{\text{wall}}(A,x) = \text{critical-path latency}, \qquad C_{\text{seq}}(A,x) = \max_{\text{path}} \sum_i n^{\text{out}}_i$$

$C_{\text{seq}}$ is the serial-token depth — the part that cannot be parallelised away. Multi-agent systems trade $C_{\text{tok}}$ for $C_{\text{seq}}$; the comparison is only meaningful once you say which resource is scarce.

**Object of study.** The compute-conditioned frontier

$$F_A(C) = \mathbb{E}_{x\sim\mathcal{D}}\big[\, s(A_C(x), y^\*) \,\big], \qquad \Delta(C) = F_{\text{multi}}(C) - \max_{A \in \mathcal{S}_1} F_A(C)$$

where $\mathcal{S}_1$ is the single-agent family (one context, one model, tools, self-consistency, best-of-$n$ with the same verifier the multi-agent arm gets). The crossover is $C^\* = \inf\{C : \Delta(C) > 0\}$. Claiming "multi-agent wins" is claiming $\Delta(C) > 0$ at the operating $C$, with $\mathcal{S}_1$ tuned as hard as $A_{\text{multi}}$.

**Assumptions, and which are violated.**

1. *$\mathcal{S}_1$ is tuned to parity.* Violated almost universally: baselines are usually zero-shot ReAct with a default prompt while the multi-agent arm is prompt-engineered over months.
2. *Score is a proper measurement of the task.* Violated on agentic benchmarks with weak or exploitable verifiers.
3. *Compute is the only scarce resource.* Violated when the real constraint is $C_{\text{wall}}$ or context-window length.
4. *$\Delta$ is monotone in $C$.* Violated: voting-based compound systems are non-monotone in call count (Chen, Zaharia & Zou, NeurIPS 2024).
5. *Held-out validity.* Violated when the scaffold was selected on the same benchmark it is reported on.

## 3. State of the Art

**Empirical SOTA (established).**
- *Parallel sampling scales.* Repeated sampling raises coverage (pass@$k$) log-linearly over four orders of magnitude of $k$ on SWE-bench Lite and GSM8K (Brown et al., "Large Language Monkeys", 2024). This is a single-agent result and is the strongest control arm any multi-agent claim must beat.
- *Test-time compute allocation matters more than the scaffold's shape.* Compute-optimal search allocation beats a naive best-of-$n$ by $\sim 4\times$ at matched budget on MATH (Snell et al., 2024).
- *Cost-controlled evaluation reorders leaderboards.* Kapoor, Stroebl & Narayanan ("AI Agents That Matter", 2024) show that on HumanEval, a simple retry/warming baseline over GPT-4 matches published agent architectures (LDB, LATS, Reflexion) at substantially lower cost, and that most agent papers report accuracy without cost at all.
- *Multi-agent debate is not free lunch.* Wang et al. (ACL 2024, "Rethinking the Bounds of LLM Reasoning") report that a single agent with a strong prompt and self-consistency matches multi-agent discussion on most of their reasoning suite; debate helps mainly when the single-agent prompt is weak.

**Claimed but unablated.**
- Anthropic's engineering report (2025) states its multi-agent research system (Claude Opus 4 lead, Sonnet 4 subagents) beat single-agent Opus 4 by **90.2%** on an internal browsing eval, and that token usage alone explains **80%** of the variance in that eval, with multi-agent runs consuming roughly **15×** the tokens of chat. The compute-matched single-agent arm was not reported. The headline is a compute-unmatched point comparison.
- MetaGPT (ICLR 2024), AutoGen (2023), Magentic-One (Microsoft, 2024), CAMEL (NeurIPS 2023): all report benchmark improvements; none report an accuracy-vs-token Pareto curve against a tuned single-agent arm.
- Mixture-of-Agents (Wang et al., ICLR 2025) reports **65.1%** on AlpacaEval 2.0 with open-source models versus **57.5%** for GPT-4 Omni — a cross-model-family comparison, so it does not isolate the scaffold.

**Theory SOTA.** Chen, Zaharia & Zou (NeurIPS 2024) give the only scaling law for compound inference systems: performance of one-layer voting systems is non-monotone in the number of LLM calls, rising then falling, because easy and hard items respond in opposite directions. Stroebl, Kapoor & Narayanan (2024) show resampling gains collapse under imperfect verifiers — false positives accumulate faster than true ones.

## 4. What Is Known

- **Coverage rises, selection does not.** On SWE-bench Lite with DeepSeek-V2-Coder, pass@1 $\approx 15.9\%$ rises to $\approx 56\%$ at $k=250$ samples; the deliverable score with an automatic verifier lands far below coverage (Brown et al., 2024, 7B–236B scale).
- **Failures are coordination failures.** Cemri et al. (2025, arXiv:2503.13657) hand-annotated 150+ traces from 7 multi-agent systems into 14 failure modes (MAST); a large share are specification, inter-agent misalignment, and verification failures, not base-model capability failures. Independent taxonomy, human-annotated, $\kappa$-validated.
- **Self-correction without external signal does not help.** Huang et al. (ICLR 2024) show intrinsic self-correction degrades GPT-3.5/GPT-4 reasoning accuracy on GSM8K, CommonsenseQA and HotpotQA. Debate-style scaffolds that rely on peer critique inherit this.
- **Token count predicts score.** In the one place it was measured at frontier scale (Anthropic 2025 internal eval), tokens explain 80% of score variance — meaning most reported scaffold gains are inside the noise band of "spent more".
- **Context is a real bottleneck.** Long-context degradation ("lost in the middle", Liu et al., TACL 2024) gives multi-agent a genuine mechanism: $k$ subagents each read $n$ tokens, so no single context exceeds $n$, while the single-agent arm must hold $kn$.

## 5. What Is Not Known

- **Empirically open (dominant).** No published study reports $F_{\text{multi}}(C)$ and $F_{\text{single}}(C)$ as curves over a $\geq 10\times$ budget sweep on the same tasks with the same base model and an equally tuned single-agent arm. The experiment is runnable today for well under $100k. Nobody has run it at frontier scale with the negative result publishable.
- **Methodologically blocked.** Which meter is "equal compute" — $C_{\text{tok}}$, $C_{\$}$, $C_{\text{wall}}$, or $C_{\text{seq}}$ — is unresolved, and the sign of $\Delta$ flips between them for the same system. Until the community fixes a meter (or reports all four), "beats at equal compute" is not a well-formed claim.
- **Theoretically open.** No characterisation of the task property that makes $\Delta(C) > 0$. Candidate: tasks with *wide, independently verifiable* subproblems and a per-subproblem verifier better than chance. No theorem states this, and no lower bound says a single agent with equal tokens cannot simulate any multi-agent protocol by serialising it — the obvious construction costs only context, which is exactly what is scarce.

## 6. Why It Is Hard

**The specific obstruction is confounded measurement plus asymmetric tuning effort.** Every published multi-agent-vs-single comparison varies at least three things at once: total tokens, prompt engineering hours, and (often) the model mix. Because tokens alone explain ~80% of variance on the one frontier eval where it was checked, a scaffold effect of realistic size (a few points) is smaller than the confound.

Second: **the control arm has no canonical form.** "Single-agent at budget $C$" could be one long trajectory, best-of-$n$, self-consistency, or a compute-optimal search policy — and these differ by ~4× in effective compute (Snell et al., 2024). A paper is free to pick the weakest.

Third: **the serialisation argument makes a clean theory statement hard.** A single agent can in principle run every subagent's transcript sequentially in one context. The only thing it cannot do is *forget* — so any true multi-agent advantage must be an advantage of context isolation, not of parallelism, and context effects are precisely the least well-characterised part of transformer behaviour.

## 7. Current Research (as of 2026)

- **Cost-controlled agent evaluation.** Princeton CITP (Kapoor, Stroebl, Narayanan) push accuracy-cost Pareto reporting; HAL (Holistic Agent Leaderboard) reports cost alongside score for agent benchmarks *(frontier — verify current coverage)*.
- **Failure-mode-driven design.** Berkeley Sky Computing (Cemri, Zaharia, Stoica et al.) extend MAST toward automated trace diagnosis.
- **Compound-system scaling laws.** Stanford (Chen, Zou) and Databricks/Berkeley (Zaharia) on when adding calls stops helping.
- **Verifier quality as the limiting factor.** Work following Stroebl et al. on imperfect-verifier resampling; process reward models as the multi-agent selection mechanism.
- **Industrial multi-agent research systems.** Anthropic, OpenAI and Google deep-research products; public reporting is engineering-blog grade, not ablated *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** 300 tasks: 100 GAIA level-2/3 (Mialon et al., ICLR 2024), 100 SWE-bench Verified, 100 wide-fanout retrieval queries (each needing $\geq 15$ independent source lookups). One base model family, one model, fixed decoding temperature. Budget sweep at $C_{\text{tok}} \in \{0.5, 1, 2, 4, 8, 16\} \times 10^5$ tokens per task — a 32× range. Estimated cost at 2026 frontier prices: ~$30–60k including reruns.

**Arms.** (a) Orchestrator + $k$ subagents, $k$ scaled with budget. (b) **Control:** single agent, best-of-$n$ with the *same* verifier the orchestrator uses, $n$ scaled with budget, prompt tuned by the same team for the same wall-clock hours as arm (a) — log the hours. (c) Second control: one long single-agent trajectory with compute-optimal step allocation. Report all four meters.

**The deciding number.** $C^\*_{\text{tok}}$: the smallest token budget at which $\Delta(C) > 0$ with a 95% bootstrap CI excluding zero, per task family. If $C^\*$ does not exist below $1.6 \times 10^6$ tokens/task on any family, multi-agent scaffolding is a latency and context-management technique, not an accuracy technique — and should be reported as such. If $C^\*$ exists only on the wide-fanout family, the decomposability hypothesis is supported and the theory question sharpens to "how wide is wide enough".

## 9. Key References

- **[Foundational]** Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2024. — arXiv:2203.11171
- **[Foundational]** Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., Mordatch, I. *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* ICML 2024. — arXiv:2305.14325
- **[SOTA — theory]** Chen, L., Zaharia, M., Zou, J. *Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems.* NeurIPS 2024. — arXiv:2403.02419
- **[SOTA — control arm]** Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., Mirhoseini, A. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA — allocation]** Snell, C., Lee, J., Xu, K., Kumar, A. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[Methodology]** Kapoor, S., Stroebl, B., Siegel, Z. S., Nadgir, N., Narayanan, A. *AI Agents That Matter.* 2024. — arXiv:2407.01502
- **[Methodology]** Stroebl, B., Kapoor, S., Narayanan, A. *Inference Scaling Flaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[Failure analysis]** Cemri, M., Pan, M. Z., Yang, S., Agrawal, L. A., Chopra, B., Tiwari, R., Keutzer, K., Parameswaran, A., Klein, D., Ramchandran, K., Zaharia, M., Gonzalez, J. E., Stoica, I. *Why Do Multi-Agent LLM Systems Fail?* 2025. — arXiv:2503.13657
- **[Negative result]** Wang, Q., Wang, Z., Su, Y., Tong, H., Song, Y. *Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?* ACL 2024. — arXiv:2402.18272
- **[Negative result]** Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., Zhou, D. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024. — arXiv:2310.01798
- **[Systems]** Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang, S., Awadallah, A., White, R. W., Burger, D., Wang, C. *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.* 2023. — arXiv:2308.08155
- **[Systems]** Hong, S., Zhuge, M., Chen, J., et al. *MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework.* ICLR 2024. — arXiv:2308.00352
- **[Benchmark]** Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., Scialom, T. *GAIA: A Benchmark for General AI Assistants.* ICLR 2024. — arXiv:2311.12983
- **[Mechanism]** Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. — arXiv:2307.03172

## 10. Worked Example

**Task.** "List every company that raised a Series A above \$20M in Q1 2024 in the EU and give each lead investor." Ground truth: 41 companies. Score = F1 over the company set.

**Multi-agent arm.** Orchestrator splits by country, spawns 12 subagents. Each subagent: ~8 searches, ~40k tokens. Orchestrator synthesis: ~60k tokens. Total $C_{\text{tok}} \approx 540{,}000$. Serial depth $C_{\text{seq}} \approx 70{,}000$ (orchestrator + one subagent path). Result: recall 0.83, precision 0.71 (four hallucinated rounds from one subagent, unchecked by the orchestrator), **F1 = 0.77**.

**Naive control.** Single ReAct agent, one context, 60k tokens: recall 0.44, F1 0.53. This is the comparison usually published — and it "proves" multi-agent wins by 24 F1 points while spending 9× the tokens.

**Matched control.** Single agent, 9 independent runs at 60k each ($C_{\text{tok}} = 540{,}000$), union of extracted companies, then a verification pass over the union at 60k more. Recall 0.85, precision 0.79, **F1 = 0.82**. The union-plus-verify single agent wins by 5 F1 points at the same token budget — but takes 10× the serial depth, so $C_{\text{seq}} \approx 600{,}000$ against 70,000.

**The obstruction, visible.** Under $C_{\text{tok}}$, single-agent wins. Under $C_{\text{seq}}$ (latency-bound production), multi-agent wins by a wide margin. Under $C_{\$}$ with a large orchestrator and small subagents, the sign depends on the price ratio $\kappa_{\text{large}}/\kappa_{\text{small}}$. Three defensible meters, three different answers, one experiment. Every published claim of the form "multi-agent beats single-agent" has silently picked one meter and usually the weakest control arm — which is why $\Delta(C)$ remains empirically open rather than settled.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*