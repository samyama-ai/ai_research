---
id: 26-code-generation/repair-versus-resample-equal-compute
title: "Repair versus Resample Under Equal Compute"
topic: 26-code-generation
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Repair versus Resample Under Equal Compute

> **Topic:** Code Generation & Program Synthesis · **ID:** `26-code-generation/repair-versus-resample-equal-compute` · **Status:** empirically-open

## 1. Problem Statement

Given a fixed inference budget and a program synthesis task with an executable test suite, is it better to spend the budget on **repair** (feed a failing program and its error trace back to the model, iterate) or on **resampling** (draw fresh independent programs at temperature $T$ and filter)? Almost every agentic coding system assumes repair wins. The comparison is rarely run with the budgets equalized.

Three variants, with different difficulty:

- **Measurement.** Define a budget unit under which the two policies are comparable at all. Tokens, FLOPs, wall-clock and dollars rank the policies differently, because repair reuses a KV prefix and resampling parallelizes. Choosing the unit largely chooses the winner. This variant is **methodologically blocked**.
- **Method.** Find a policy that allocates a budget $C$ between the two — including switching mid-run — and beats both pure arms. Empirically open.
- **Theory.** Prove when conditioning on a failure trace raises the per-sample success probability by more than it costs in correlation between successive attempts. Open, and the standard toy models do not capture it.

Solving it means: a curve of success versus budget for both arms, on the same tasks, same model, same verifier, with the crossover budget $C^\*$ named and shown to shift predictably with task difficulty and feedback quality.

## 2. Formal Setting

Task $x$, program space $\mathcal{Y}$, model $p_\theta$. A **verifier** $V:\mathcal{Y}\to\{0,1\}$ is the visible test suite; **ground truth** $V^\*$ is the held-out suite. Measured as: `pytest` exit status on the given tests, and on EvalPlus / hidden SWE-bench `FAIL_TO_PASS` tests respectively.

**Resample arm.** Draw $y_1,\dots,y_k \stackrel{iid}{\sim} p_\theta(\cdot\mid x)$, return any $y_i$ with $V(y_i)=1$. Success probability

$$\Pr[\text{solve}] = 1-(1-q)^k,\qquad q=\Pr_{y\sim p_\theta(\cdot|x)}[V(y)=1].$$

Measured as coverage over $n\ge 100$ draws with the unbiased pass@$k$ estimator of Chen et al. (2021), $1-\binom{n-c}{k}/\binom{n}{k}$.

**Repair arm.** A chain $y_{t+1}\sim p_\theta(\cdot \mid x, y_t, f(y_t))$, where $f$ is feedback: raw stderr, failing assertion, or a model-written critique. Per-step success $q_t = \Pr[V(y_{t+1})=1 \mid V(y_t)=0]$, measured as the fraction of chains alive at depth $t$ that pass at $t+1$.

**Budget.** Prefill tokens $a$, decode tokens $b$, per-token costs $\kappa_a \ll \kappa_b$ (measured: prefill FLOPs $\approx 2Pa$, decode $\approx 2Pb$ for $P$ non-embedding parameters; or the provider's cached/uncached input and output prices).

$$C_{\text{resample}}(k)=k(\kappa_a a_0+\kappa_b b),\qquad C_{\text{repair}}(T)=\sum_{t=1}^{T}\big(\kappa_a (a_0+t(b+|f|)) + \kappa_b b\big).$$

Repair's prefill grows linearly in depth, so its total cost is $O(T^2)$ in prefill and $O(T)$ in decode. The decision quantity is

$$\Delta(C)=\Pr[\text{solve}\mid \text{repair}, C]-\Pr[\text{solve}\mid \text{resample}, C],$$

and $C^\*=\inf\{C:\Delta(C)<0\}$.

**Assumptions, and which are violated.**
1. *$V=V^\*$ (tests are sound).* Violated: EvalPlus (Liu et al., NeurIPS 2023) shows HumanEval's original tests inflate pass@1 by roughly 15 points for several models; APR patch-overfitting is the same failure (Qi et al., ISSTA 2015).
2. *Resamples are i.i.d.* Holds by construction, but the effective diversity collapses at low $T$.
3. *Repair steps are Markov in $y_t$.* Violated when the transcript keeps all prior attempts; the chain conditions on its own failures and often re-emits them.
4. *$q$ is constant across a task set.* Badly violated — $q$ is bimodal (near 0 or near 1), which is exactly what makes the aggregate curve uninformative.

## 3. State of the Art

**Established (properly ablated).**
- Olausson et al., *Is Self-Repair a Silver Bullet for Code Generation?* (ICLR 2024) is the one paper that runs this comparison correctly. Plotting pass rate against **total tokens sampled** rather than against number of repair rounds, self-repair gives little or no gain over i.i.d. resampling for GPT-3.5 on HumanEval and APPS; for GPT-4 the gain is real but small. The gain is attributable to feedback quality, not to the repair mechanism: GPT-4 feedback given to GPT-3.5 lifts it substantially, and human-written feedback repairs far more programs than GPT-4's own.
- Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* (ICLR 2024) shows the same on reasoning: intrinsic self-correction without an oracle signal degrades accuracy, and at matched sample budget self-consistency beats multi-round self-correction.
- Brown et al., *Large Language Monkeys* (2024): coverage rises log-linearly in $\log k$ over four orders of magnitude; DeepSeek-Coder-V2-Instruct on SWE-bench Lite goes from 15.9% at one sample to 56% at 250 — above the then-best single-attempt system.

**Claimed but unablated.** Self-Debug (Chen et al., ICLR 2024), Reflexion (Shinn et al., NeurIPS 2023), ChatRepair (Xia & Zhang, 2023) and AlphaCodium (Ridnik et al., 2024) all report repair gains against a **single-sample** baseline, not against a resample baseline of equal cost. These are benchmark numbers, not ablations of the mechanism.

**Systems SOTA.** Snell et al. (2024) fit a compute-optimal allocation between sequential revision and parallel sampling on MATH, and report that the optimal ratio moves with problem difficulty — easy problems favour sequential, hard ones favour parallel. The code-generation analogue has not been fit.

## 4. What Is Known

- Repair helps most where the verifier signal is informative. Olausson et al.: replacing model self-feedback with human feedback increases the number of programs repaired by a large margin on the same model — the mechanism's ceiling is set by feedback, not by decoding.
- Resampling's returns are predictable. Coverage $\approx 1-e^{-ak^{b}}$ fits well over $k\in[1,10^4]$ (Brown et al., 2024, Llama-3 and Gemma families, 8B–70B).
- Selection, not generation, is the binding constraint at large $k$. On SWE-bench Lite, coverage of 56% at $k=250$ collapses to about 43% once an automatic verifier must pick the patch (Brown et al., 2024).
- With an imperfect verifier, resampling has a ceiling and can get *worse* with more samples as false positives accumulate (Stroebl et al., *Inference Scaling Flaws*, 2024).
- Budget reallocation across model sizes matters as much as across policies: Hassid et al. (COLM 2024) find many small-model samples beat one large-model sample at fixed cost on HumanEval/MBPP.
- Overfitting to visible tests is old and quantified: generate-and-validate APR systems produced plausible-but-incorrect patches for the majority of "fixed" defects (Qi et al., ISSTA 2015).

## 5. What Is Not Known

- **Methodologically blocked.** No agreed budget unit. Repair with prefix caching costs a fraction of its nominal token count; resampling costs more tokens but finishes in one round-trip. Papers that report tokens, dollars, and latency reach different conclusions from the same runs. Until the unit is fixed, $\Delta(C)$ is not a well-defined number.
- **Empirically open.** The crossover $C^\*$ as a function of task difficulty, model scale and feedback type has never been measured on a repository-scale benchmark. Runnable today; roughly $10^4$–$10^5$ GPU-hours with an open model.
- **Empirically open.** Whether repair adds *any* coverage beyond the support of the i.i.d. sampler — i.e. whether repair reaches programs resampling would never emit — or merely reweights. Testable by set-difference of solved tasks at matched budget.
- **Theoretically open.** No characterization of when conditioning on a trace increases $q_t$ enough to beat the loss from inter-attempt correlation. Bandit and branching-process models exist; none is calibrated to measured $q_t$.

## 6. Why It Is Hard

**Confounded measurement, three ways.**

1. *The unit is a free parameter.* Prefill is ~10–50× cheaper per token than decode on batched inference, and cached prefill is cheaper still. Repair's cost is prefill-heavy, resampling's is decode-heavy. Any paper picks a ratio, usually implicitly, and the ratio determines the sign of $\Delta$.
2. *Bimodal $q$.* Aggregate pass@$k$ averages tasks with $q\approx 0.9$ (both arms win at $k=1$) and $q\approx 10^{-3}$ (neither arm wins in budget). The tasks that discriminate are a thin band, and benchmark-level means hide it. Conditioning on the band requires estimating $q$ per task first, which costs more than the experiment.
3. *The verifier is the experiment.* Repair reads $V$'s output; resampling only uses $V$ to select. So repair's advantage is entangled with $V$'s informativeness, and both arms' measured success is entangled with $V\neq V^\*$. With weak tests, repair looks better because it overfits to them faster.

## 7. Current Research (as of 2026)

- **Compute-optimal test-time scaling.** Extending Snell et al.'s sequential/parallel allocation from math to code and to agentic trajectories. Google DeepMind, Berkeley. *(frontier — verify)*
- **Verifier-aware scaling limits.** Following Stroebl et al., characterizing the false-positive ceiling and its interaction with repair. Princeton.
- **Execution-feedback RL.** Training the repair step directly rather than prompting it, which changes $q_t$ and therefore moves $C^\*$; this is the main reason older ablations may not transfer to 2026 reasoning models. *(frontier — verify)*
- **Benchmark hygiene.** SWE-bench Verified and EvalPlus-style test strengthening, which narrows the $V\neq V^\*$ gap that currently confounds every repair result.

## 8. Concrete Next Experiment

**Scale.** SWE-bench Verified (500 tasks) plus APPS-Competition (1,000 tasks, for a hard tail). One open-weights model of ~30B and one of ~200B+ class, run locally so FLOPs are directly countable. Budget grid $C\in\{1,2,4,\dots,256\}$ single-sample-equivalents.

**Arms.**
- *Control:* i.i.d. resampling at $T=0.8$, best-of-$k$ selected by the visible tests, ties broken at random.
- *Treatment A:* repair chains of depth $T$, raw execution feedback, one chain.
- *Treatment B:* $\sqrt{C}$ chains of depth $\sqrt{C}$ (the mixed policy).
- Cost accounted **three times**: decode-token count, measured FLOPs, and measured wall-clock with prefix caching on.

**Deciding number.** $C^\*$ — the budget at which the control arm overtakes Treatment A on **held-out** pass rate ($V^\*$, not $V$) — reported separately under each of the three cost units. If $C^\*$ is stable within a factor of 2 across units, the field's ambiguity is harmless and repair/resample is settled by a single curve. If $C^\*$ varies by more than 10× across units, the measurement problem is the real result, and no existing repair claim is interpretable.

Secondary: $|S_{\text{repair}}\setminus S_{\text{resample}}|$ at matched $C$ — tasks repair solves that resampling never does. If this is under 2% of tasks, repair is reweighting, not exploring.

## 9. Key References

- **[Foundational]** M. Chen et al. *Evaluating Large Language Models Trained on Code.* 2021. — arXiv:2107.03374
- **[SOTA]** T. Olausson, J. Inala, C. Wang, J. Gao, A. Solar-Lezama. *Is Self-Repair a Silver Bullet for Code Generation?* ICLR 2024. — arXiv:2306.09896
- **[SOTA]** B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. Le, C. Ré, A. Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* 2024. — arXiv:2407.21787
- **[SOTA]** C. Snell, J. Lee, K. Xu, A. Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* 2024. — arXiv:2408.03314
- **[Contra]** J. Huang, X. Chen, S. Mishra, H. Zheng, A. Yu, X. Song, D. Zhou. *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024. — arXiv:2310.01798
- **[Contra]** B. Stroebl, S. Kapoor, A. Narayanan. *Inference Scaling Flaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[Method]** X. Chen, M. Lin, N. Schärli, D. Zhou. *Teaching Large Language Models to Self-Debug.* ICLR 2024. — arXiv:2304.05128
- **[Method]** N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, S. Yao. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023. — arXiv:2303.11366
- **[Measurement]** J. Liu, C. Xia, Y. Wang, L. Zhang. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.* NeurIPS 2023. — arXiv:2305.01210
- **[Foundational, APR]** Z. Qi, F. Long, S. Achour, M. Rinard. *An Analysis of Patch Plausibility and Correctness for Generate-and-Validate Patch Generation Systems.* ISSTA 2015.
- **[Benchmark]** C. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, K. Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. — arXiv:2310.06770

## 10. Worked Example

One task, plausible measured values. Prompt $a_0=1{,}500$ tokens, program $b=400$ tokens, feedback $|f|=200$ tokens. Base success $q=0.20$.

**Resample, $k=4$.** Decode $= 1{,}600$ tokens. Prefill $= 6{,}000$. Success $=1-0.8^4=59.0\%$.

**Repair, depth $T=4$.** Decode $=1{,}600$. Prefill $=\sum_{t=0}^{3}(1500+600t)=9{,}600$. Suppose measured per-step $q_t = 0.20, 0.12, 0.08, 0.06$ (decaying, as chains that fail twice tend to keep failing). Success $=1-(0.8)(0.88)(0.92)(0.94)=39.2\%$.

Now change only the accounting unit:

| Unit | Resample | Repair | Winner |
|---|---|---|---|
| Decode tokens | 1,600 | 1,600 | tie on cost, resample on success |
| Total tokens | 7,600 | 11,200 | resample |
| FLOPs, prefill 1/50 of decode | 1,720 | 1,792 | resample |
| Wall-clock, 4-way parallel, cached prefix | 1× latency | 4× latency | resample |

Repair loses on every unit here — but flip one measured number. If feedback is informative and $q_t = 0.20, 0.35, 0.30, 0.25$ (what a good stack trace buys), repair reaches $1-(0.8)(0.65)(0.70)(0.75)=72.7\%$ against resampling's 59.0%, and wins on decode tokens and FLOPs while still losing on latency.

**The obstruction, visible.** The sign of $\Delta$ is set by the $q_t$ decay profile, which is a property of the *feedback channel*, not of the model or the policy. No published code-repair paper reports $q_t$ per depth. Every reported repair gain is consistent with both tables above.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*