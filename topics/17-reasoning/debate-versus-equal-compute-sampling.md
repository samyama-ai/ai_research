---
id: 17-reasoning/debate-versus-equal-compute-sampling
title: "Multi-Agent Debate Versus Equal-Compute Single-Model Sampling"
topic: 17-reasoning
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multi-Agent Debate Versus Equal-Compute Single-Model Sampling

> **Topic:** Reasoning & Inference-Time Compute · **ID:** `17-reasoning/debate-versus-equal-compute-sampling` · **Status:** empirically-open

## 1. Problem Statement

Multi-agent debate (MAD) runs $n$ copies of a language model on the same question, shows each copy the others' answers, and iterates for $r$ rounds before aggregating. It is reported to beat single-pass chain-of-thought. The open question is whether it beats **the best single-model method at the same inference cost**.

- **Input:** a question $x$, a base model $p_\theta$, a token budget $B$.
- **Output:** an answer $\hat{y}$.
- **Decision predicate:** does there exist a task family and budget regime where debate's accuracy at budget $B$ exceeds the best budget-$B$ single-model procedure (repeated sampling + majority vote, best-of-$n$ with a verifier, or one long chain of thought) by a margin that survives a matched-budget control and prompt-strength ablation?

Three variants, different difficulty:

- **Measurement:** define "equal compute" for a procedure whose context grows across rounds. Currently under-specified.
- **Method:** build a debate protocol that wins at matched budget. Empirically open.
- **Theory:** characterise the task property (error independence across agents? asymmetric verification cost?) under which cross-agent conditioning beats i.i.d. sampling. Theoretically open.

## 2. Formal Setting

Let $p_\theta(\cdot \mid c)$ be the model's distribution over completions given context $c$, and $y^\star(x)$ the gold answer. Define per-question accuracy $A(\pi) = \mathbb{E}_x[\mathbb{1}\{\pi(x) = y^\star(x)\}]$ for procedure $\pi$.

**Cost, as actually measured.** Prompt and output tokens are not fungible — prefill is compute-bound, decode is memory-bandwidth-bound. Measure both:

$$C(\pi) = \sum_{\text{calls } i} \left( \alpha \, T^{\text{in}}_i + T^{\text{out}}_i \right), \qquad \alpha \approx 0.1\text{–}0.3$$

with $\alpha$ the empirically fitted prefill:decode cost ratio on the serving stack used. Report $\alpha = 1$ and $\alpha = 0$ as bounds; a claim that flips sign between them is not a result.

**Debate.** Agent $j$ at round $t$ produces $a_j^{(t)} \sim p_\theta(\cdot \mid x, \{a_k^{(t-1)}\}_{k \neq j})$. Final answer is $\mathrm{mode}_j\, a_j^{(r)}$. With per-answer length $L$ and prompt length $P$:

$$C_{\text{MAD}} = nr\,L + \alpha\, n\left[P + (r-1)\big(P + (n-1)L\big)\right]$$

quadratic in $n$ through the context term.

**Self-consistency control.** $k$ i.i.d. samples, majority vote (Wang et al., ICLR 2023): $C_{\text{SC}} = k(L + \alpha P)$. Matched budget sets $k^\star = \max\{k : C_{\text{SC}} \le C_{\text{MAD}}\}$.

**The quantity of interest.** The budget-matched debate gain

$$\Delta(B) = A(\pi_{\text{MAD}} \mid C = B) - \max_{\pi \in \Pi_{\text{single}}} A(\pi \mid C = B)$$

where $\Pi_{\text{single}}$ contains self-consistency, best-of-$n$ under a verifier, and single-chain reasoning with matched output length. The claim under test is $\Delta(B) > 0$ for some $B$.

**Assumptions, and which fail.**

1. *Agents are exchangeable and their errors partly independent.* Violated: identical $\theta$ and identical prompts give highly correlated errors; observed inter-agent agreement on GSM8K-style tasks is far above chance-independent levels.
2. *Answers are extractable and comparable.* Holds for math/multiple-choice, fails for open-ended generation, where the majority-vote aggregator is undefined and the comparison silently changes.
3. *Debate converges to a fixed point.* Violated: models exhibit sycophantic drift — agents abandon correct answers under peer pressure.
4. *$L$ is held constant across arms.* Usually violated; debate transcripts lengthen across rounds, so an unmatched-length comparison confounds reasoning-length scaling with debate.

## 3. State of the Art

**Established (with matched-budget controls).**
- Smit et al., *Should We Be Going MAD? A Look at Multi-Agent Debate Strategies for LLMs* (ICML 2024). The central negative result: across debate protocols, a cost-equivalent single-agent baseline (self-consistency at matched calls) matches or beats debate on most benchmarks tested. Sensitivity to agent count, rounds, and agreement-modulation is larger than the debate-vs-single effect.
- Wang et al., *Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?* (ACL 2024). A single agent with a strong prompt and demonstrations matches multi-agent discussion; the reported debate gain largely disappears once the single-agent prompt is strengthened.

**Claimed but incompletely ablated.**
- Du et al., *Improving Factuality and Reasoning in Language Models through Multiagent Debate* (ICML 2024). Gains on GSM8K, MMLU, and biography factuality — but the primary baseline is single-pass or few-sample self-consistency, not budget-matched sampling at $k^\star \approx nr$ or more.
- Liang et al., *Encouraging Divergent Thinking in LLMs through Multi-Agent Debate* (EMNLP 2024). Degeneration-of-thought framing; benchmark numbers on translation and counter-intuitive arithmetic, without a token-matched sampling arm.
- Chen et al., *ReConcile* (ACL 2024). Uses **heterogeneous** models; gains are confounded with ensembling distinct model families, which is a different claim from debate itself.

**Adjacent and established.** Snell et al. (2024) and Brown et al. (*Large Language Monkeys*, 2024) give the strong single-model baseline that debate must beat: coverage under repeated sampling scales near-log-linearly in $k$ over orders of magnitude, and compute-optimal test-time strategy selection can beat naive scaling substantially.

**Debate-for-oversight is a separate literature.** Khan et al., *Debating with More Persuasive LLMs Leads to More Truthful Answers* (ICML 2024) shows debate helps a weak judge on an information-asymmetric task (QuALITY with hidden passage). Kenton et al. (2024) find debate's advantage over consultancy is real but small and task-dependent. These measure judge accuracy under asymmetry, not raw reasoning accuracy at matched compute.

## 4. What Is Known

- **Self-consistency is a strong, cheap baseline.** Wang et al. (ICLR 2023): +17.9 points on GSM8K with PaLM-540B, +11.0 on SVAMP, +12.2 on AQuA, using 40 samples. Scale: 540B dense model, 2022.
- **Coverage scales with samples.** Brown et al. (2024): on GSM8K and MATH, fraction of problems solved by at least one of $k$ samples rises steadily from $k=1$ to $k=10^4$; a small model with large $k$ can exceed a much larger model at $k=1$. Scale: Llama-3 8B/70B, Gemma, 2024.
- **Debate gains shrink under matched cost.** Smit et al. (ICML 2024) and Wang et al. (ACL 2024), at GPT-3.5/GPT-4-class scale on GSM8K, MMLU, MATH-style sets.
- **Models do not reliably self-correct without external signal.** Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* (ICLR 2024): intrinsic self-correction degrades GSM8K accuracy; apparent gains in prior work came from oracle labels stopping the correction loop. This removes the most common mechanism proposed for debate.
- **Sycophancy is measurable.** Models revise correct answers toward stated peer or user positions (Sharma et al., ICLR 2024, on RLHF'd assistants), giving a concrete negative mechanism for round-over-round debate drift.

## 5. What Is Not Known

- **Empirically open.** No published study sweeps debate and self-consistency over $\ge 3$ orders of magnitude of matched token budget on the same tasks and models, with a reasoning-trained model (o-series / R1-class). Every matched-budget comparison to date sits at $k \lesssim 40$ and pre-reasoning-RL models. It is runnable today; nobody has run it at the right scale.
- **Empirically open.** Whether debate wins on tasks with *asymmetric verification* (one agent holds evidence another lacks) rather than symmetric self-play. Khan et al. suggest yes for judge accuracy; the matched-compute reasoning version is unrun.
- **Theoretically open.** No characterisation of when conditioning agent $j$ on $\{a_k\}_{k\ne j}$ increases the probability of the correct mode relative to i.i.d. sampling. Informally the answer must depend on the error-correlation structure $\rho$ between agents, but no bound of the form "$\Delta > 0$ only if $\rho < \rho^\star(n, r)$" exists.
- **Methodologically blocked.** "Equal compute" for open-ended generation. Without an extractable answer, majority vote is undefined, so the control arm cannot be constructed — debate is then compared to whatever aggregator the authors chose, which is not a controlled comparison.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by a moving control arm**, not compute.

1. **The baseline is a maximum over a family, not a fixed method.** $\max_{\pi \in \Pi_{\text{single}}}$ requires tuning the sampling temperature, sample count, verifier, and prompt of the control. Papers proposing debate tune debate and leave the control at defaults; Wang et al. (ACL 2024) showed the gap closes when the control's prompt is tuned. Any single reported $\Delta$ is an upper bound on the true one.
2. **Cost is not a scalar.** The $\alpha$ ratio between prefill and decode is stack-dependent, and debate is prefill-heavy while sampling is decode-heavy with shared-prefix caching. The same experiment can show debate as 1.4× cheaper or 3× more expensive depending on batching and KV-cache reuse.
3. **Length is a hidden treatment.** Debate transcripts grow; more tokens of reasoning improves accuracy on its own. Unless output length per final answer is matched, debate's gain is partly just longer thinking.

## 7. Current Research (as of 2026)

- **Compute-optimal test-time scaling.** Extending Snell et al.'s framing to select among debate, sampling, and long-CoT per-question by difficulty. Groups: Berkeley/Google DeepMind, Stanford (Hashimoto, Zou lines). *(frontier — verify)*
- **Debate as scalable oversight.** UK AI Safety Institute, NYU (Bowman), Google DeepMind (Kenton, Irving) — protocol design where the judge is genuinely weaker than debaters. Distinct objective from accuracy-at-budget.
- **Heterogeneous ensembles and routing.** Replacing self-debate with mixed-model committees, where diversity is architectural rather than sampled. Reported gains here are ensembling, not debate.
- **Debate on top of reasoning-RL models.** Whether MAD adds anything over a model already trained to produce long self-checking chains is the live question; early reports suggest sharply diminished returns. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** Two open-weights models per class — one instruction-tuned (Llama-3.3-70B-class), one reasoning-trained (R1-distill-70B-class). Four task sets: GSM8K (saturating), MATH-500 (hard, verifiable), GPQA-Diamond (knowledge-bound), and one information-asymmetric set (QuALITY with the passage hidden from the judge). $\ge 500$ questions each, 5 seeds.

**Budget grid.** Seven token budgets per question, log-spaced: $B \in \{2, 6, 20, 60, 200, 600, 2000\} \times 10^3$ tokens, measured as $C(\pi)$ with $\alpha$ fitted on the actual serving stack and reported additionally at $\alpha \in \{0, 1\}$.

**Arms.**
- Debate: $(n, r)$ swept over $\{2,3,5\} \times \{2,3,4\}$, best configuration per budget.
- **Control arm (the one that matters):** self-consistency at $k^\star$, with temperature tuned per budget on a held-out split, using the *same* prompt that the debate agents receive in round 1 and also a separately tuned strong prompt. Report both.
- Secondary controls: best-of-$n$ under a same-family reward model; single chain with output length matched to total debate decode tokens.

**Deciding number.** $\Delta^\star = \max_B \big[ A_{\text{MAD}}(B) - A_{\text{SC-tuned}}(B) \big]$, with a 95% cluster-bootstrap CI over questions and seeds. **Decision rule:** if the CI for $\Delta^\star$ excludes 0 and $\Delta^\star \ge 2$ accuracy points on any non-asymmetric task, debate has a real matched-compute advantage. If the CI contains 0 at every budget on symmetric tasks while excluding 0 on the asymmetric task, the conclusion is that debate's value is *information asymmetry*, not deliberation — which reclassifies the whole literature.

**Cost estimate.** ~$4\times10^9$ generated tokens; roughly 3–6 GPU-days on 8×H100 with vLLM per model. Cheap. The reason it is unrun is that it is a control experiment, not a method paper.

## 9. Key References

- **[Foundational]** Du, Li, Torralba, Tenenbaum, Mordatch. *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* ICML, 2024. — arXiv:2305.14325
- **[Foundational]** Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou. *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR, 2023. — arXiv:2203.11171
- **[Foundational]** Irving, Christiano, Amodei. *AI Safety via Debate.* 2018. — arXiv:1805.00899
- **[SOTA / negative result]** Smit, Duckworth, Grinsztajn, Barrett, Pretorius. *Should We Be Going MAD? A Look at Multi-Agent Debate Strategies for LLMs.* ICML, 2024. — arXiv:2311.17371
- **[SOTA / negative result]** Wang, Wang, Wang, Zhao, Chen, Zhang, Chang. *Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?* ACL, 2024. — arXiv:2402.18272
- **[SOTA]** Snell, Lee, Xu, Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[SOTA]** Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** Khan, Hughes, Valentine, Ruis, Sachan, Radhakrishnan, Grefenstette, Bowman, Rocktäschel, Perez. *Debating with More Persuasive LLMs Leads to More Truthful Answers.* ICML, 2024. — arXiv:2402.06782
- **[Mechanism]** Huang, Chen, Mishra, Zheng, Yu, Song, Zhou. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR, 2024. — arXiv:2310.01798
- **[Mechanism]** Kenton, Siegel, Kramár, Brown-Cohen, Albanie, Bulian, Agarwal, Lindner, Tang, Goodman, Irving. *On Scalable Oversight with Weak LLMs Judging Strong LLMs.* 2024. — arXiv:2407.04622
- **[Related]** Liang, He, Jiao, Wang, Wang, Wang, Yang, Tu, Shi. *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate.* EMNLP, 2024. — arXiv:2305.19118
- **[Related]** Chen, Saha, Bansal. *ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs.* ACL, 2024. — arXiv:2309.13007
- **[Survey]** Guo, Chen, Wang, Chang, Pei, Yang, Chen, Cheng, Zhang. *Large Language Model based Multi-Agents: A Survey of Progress and Challenges.* IJCAI, 2024. — arXiv:2402.01680

## 10. Worked Example

**Setting.** GSM8K, one 70B model. Measured per-call sizes: prompt $P = 150$ tokens, chain-of-thought answer $L = 250$ tokens. Take $\alpha = 1$ (worst case for debate) and $\alpha = 0.2$ (a realistic serving ratio).

**Debate, $n = 3$, $r = 3$.** Round 1: 3 calls, in $150$, out $250$. Rounds 2–3: each agent's prompt carries the two peer answers, so in $= 150 + 2(250) = 650$, out $250$.

$$C_{\text{MAD}}(\alpha{=}1) = 3(150{+}250) + 6(650{+}250) = 1200 + 5400 = 6600$$
$$C_{\text{MAD}}(\alpha{=}0.2) = 3(30{+}250) + 6(130{+}250) = 840 + 2280 = 3120$$

**Matched self-consistency.** $C_{\text{SC}} = k(L + \alpha P)$, so
- $\alpha = 1$: $k^\star = \lfloor 6600 / 400 \rfloor = 16$
- $\alpha = 0.2$: $k^\star = \lfloor 3120 / 280 \rfloor = 11$

**The obstruction, made visible.** The typical published comparison is debate ($nr = 9$ generations) against self-consistency at $k = 3$ or $k = 5$ — matched on *number of agents*, not tokens. Two things go wrong at once:

1. **The control is under-budgeted.** The honest control is $k = 16$, not $k = 3$. Self-consistency's accuracy curve is still climbing between $k=3$ and $k=16$; on GSM8K at 70B scale that segment is typically worth several points. A 4-point debate "gain" over $k=3$ can be a 1-point loss against $k=16$.
2. **The budget itself moves with the stack.** $k^\star$ shifts from 16 to 11 purely by changing $\alpha$ from 1 to 0.2 — a 45% swing in the control's sample count, driven by KV-cache reuse and batching, not by anything about reasoning. With shared-prefix caching, self-consistency's 16 samples share one 150-token prefill, pushing its effective $k^\star$ higher still; debate's per-agent contexts diverge after round 1 and cannot share.

So the sign of $\Delta$ at this budget is determined by two choices — the control's $k$ and the serving stack's $\alpha$ — that no paper in the debate literature reports. That is the gap: not that debate has been shown to fail, but that the comparison as run does not measure what it names.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*