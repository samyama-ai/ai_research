---
id: 05-retrieval-and-agents/compounding-error-rates-multi-step-agents
title: "Compounding Error Rates in Multi-Step Agents"
topic: 05-retrieval-and-agents
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Compounding Error Rates in Multi-Step Agents

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/compounding-error-rates-multi-step-agents` · **Status:** empirically-open

## 1. Problem Statement

An LLM agent executes a trajectory of $H$ dependent steps (tool call, observation, reasoning update). End-to-end success falls as $H$ grows. The problem is to characterize *how* it falls, and whether the decay rate is a fixed property of the model or something an architecture can change.

Three variants, routinely conflated:

- **Measurement.** Given trajectories, estimate the per-step error rate $\epsilon$ and the recovery probability $\rho$ separately from end-to-end success. Currently there is no agreed estimator, because "a step was wrong" has no ground truth on open-ended tasks.
- **Method.** Build an agent whose success decays sub-geometrically in $H$ — i.e. whose effective horizon grows faster than $1/\epsilon$ — via verification, checkpointing, or state externalization. Claimed by many systems; ablated against a matched-compute control by almost none.
- **Theory.** Prove whether self-verification by the same model that generated the step can lower the asymptotic decay exponent, or whether errors and their detection are correlated enough that it cannot.

**Solved** would mean: a measurement protocol that recovers $(\epsilon, \rho)$ identifiably from logs, plus a demonstration that some intervention reduces the fitted decay exponent at fixed inference compute.

## 2. Formal Setting

A task is a POMDP with horizon $H$. At step $t$ the agent emits action $a_t \sim \pi_\theta(\cdot \mid c_t)$ where the context $c_t = (g, a_1, o_1, \dots, a_{t-1}, o_{t-1})$ holds the goal $g$ and history. Let $S_t \in \{\text{ok}, \text{err}\}$ be a latent per-step correctness variable.

**Per-step error rate**, measured as the fraction of steps a step-level oracle (human annotator or programmatic invariant) marks as deviating from every optimal continuation:
$$\epsilon_t = \Pr[S_t = \text{err} \mid S_{1:t-1} = \text{ok}].$$

**Recovery probability**, measured as the fraction of trajectories that reach a marked-err step and still terminate in task success:
$$\rho = \Pr[\text{success} \mid \exists t: S_t = \text{err}].$$

**Independent-error baseline.** If $\epsilon_t \equiv \epsilon$ and errors are absorbing ($\rho = 0$),
$$\Pr[\text{success}] = (1-\epsilon)^H = e^{-\alpha H}, \qquad \alpha = -\log(1-\epsilon).$$
The measured quantity is the fitted **decay exponent** $\hat\alpha$ from a regression of $\log \widehat{\Pr}[\text{success}]$ on $H$ across task bins. The **50% horizon** is $H_{1/2} = \log 2/\hat\alpha$.

**With recovery**, model the trajectory as a two-state chain with absorbing failure. Success becomes
$$\Pr[\text{success}] = \big(1 - \epsilon(1-\rho)\big)^{H},$$
so only the product $\epsilon(1-\rho)$ is identified from end-to-end data. This is the central non-identifiability.

**Cost formulation (imitation-learning form).** If the agent has per-step disagreement $\epsilon$ with an expert under its *own* state distribution and the task cost is bounded in $[0,1]$ per step, DAgger's reduction gives regret $O(\epsilon H)$; a behavior-cloned policy measured under the *expert's* distribution gives only $O(\epsilon H^2)$ (Ross, Gordon & Bagnell, AISTATS 2011).

**Assumptions and their violations.**
- *Stationary $\epsilon_t$* — violated. Long contexts degrade retrieval of mid-context facts (Liu et al., TACL 2024), so $\epsilon_t$ rises with $t$ independent of task difficulty.
- *Conditional independence of errors* — violated. Hallucination snowballing: a model commits to a wrong claim, then defends it (Zhang et al., ICML 2024).
- *$H$ known and task-intrinsic* — violated. Agents choose their own $H$; failed agents often run longer, inducing selection bias in any per-step estimate.
- *Verifier independence from generator* — violated whenever the same model verifies its own step.

## 3. State of the Art

**Theory SOTA.** Ross & Bagnell (AISTATS 2010) and DAgger (AISTATS 2011) remain the only sharp compounding-error results with matching lower bounds: the $\Theta(\epsilon H^2)$ behavior-cloning cost blowup, reduced to $O(\epsilon H)$ under on-policy data collection. These are *established*. No analogue exists for LLM agents where the "expert" is undefined and $\epsilon$ is not measurable.

Stroebl, Kapoor & Narayanan (2024, arXiv:2411.17501) prove that with an imperfect verifier of non-zero false-positive rate, resampling-based inference scaling saturates and can degrade — a bound on the *recovery* term $\rho$. Chen, Zaharia & Zou (NeurIPS 2024, arXiv:2403.02419) show that compound systems of $k$ LLM calls have non-monotone performance in $k$: gains on easy items, losses on hard ones. Both established analytically with small-scale empirical support.

**Empirical SOTA.** METR's time-horizon work (Kwa et al., 2025, arXiv:2503.14499) fits a logistic curve of success against *human task duration* and reports the 50%-success horizon doubling roughly every 7 months from 2019 to 2025. This is the best-fit trend line to date, not a mechanism; and it measures horizon in human minutes, not agent steps.

**Claimed but unablated.** Reflexion (Shinn et al., NeurIPS 2023), ReAct (Yao et al., ICLR 2023) and the large family of self-critique loops report end-to-end gains, but almost none hold *total inference tokens* fixed against a best-of-$n$ control. Where the comparison has been made, much of the gain is a compute effect. Agent scaffolds reporting SWE-bench Verified above 65% report a benchmark number only; per-step error rates are not published.

## 4. What Is Known

- **Compositional depth decays accuracy fast.** Dziri et al. (NeurIPS 2023, "Faith and Fate") show GPT-4 multi-digit multiplication and Einstein-puzzle accuracy falls from near-perfect at depth 1 to single digits by 4–5 composition steps, and that accuracy is predicted by the number of required computation-graph nodes — consistent with geometric, not polynomial, decay.
- **Multi-turn degradation is large and model-general.** Laban et al. (2025, arXiv:2505.06120) report an average ~39% drop across 15 models when a single-turn instruction is sharded across turns, with most of the loss attributable to increased variance ("unreliability"), not lower ceiling. Scale: 200k+ simulated conversations.
- **Agent–human gaps on long tasks are order-of-magnitude.** GAIA: humans 92%, GPT-4 with plugins 15% (Mialon et al., ICLR 2024). WebArena: humans 78.24%, GPT-4 agent 14.41% (Zhou et al., ICLR 2024). OSWorld: humans 72.36%, best agent 12.24% at release (Xie et al., NeurIPS 2024). The gap widens with the number of required steps in all three.
- **Failure modes are enumerable.** Cemri et al. (2025, arXiv:2503.13657) hand-annotate 200+ multi-agent traces into a 14-mode taxonomy (MAST); inter-annotator agreement Cohen's $\kappa \approx 0.88$. Establishes that step-level annotation is *possible*, at roughly one expert-hour per trace.
- **Verifier quality caps recovery.** Resampling gains vanish once the verifier's false-positive rate exceeds the generator's per-attempt success rate (Stroebl et al., 2024).

## 5. What Is Not Known

- **Methodologically blocked:** separating $\epsilon$ from $\rho$. End-to-end success identifies only $\epsilon(1-\rho)$ (§2). Without a step-level oracle there is no estimator, and on open-ended tasks no oracle exists. This is the binding gap.
- **Empirically open:** whether $\hat\alpha$ is falling across model generations *at fixed $\epsilon$*. METR's horizon doubling is consistent with two very different stories — better per-step accuracy, or better recovery — and nobody has run the annotation study that distinguishes them.
- **Empirically open:** whether any scaffold reduces $\hat\alpha$ at matched total inference tokens. Every published reflection/critique result the author is aware of varies compute and architecture together.
- **Theoretically open:** whether same-model self-verification can reduce the asymptotic exponent at all, or whether generator–verifier error correlation forces $\rho$ below a bound depending only on $\epsilon$. No proof either way.

## 6. Why It Is Hard

**Non-identifiability, compounded by absent step-level ground truth.** Two agents with $(\epsilon = 0.10, \rho = 0.50)$ and $(\epsilon = 0.05, \rho = 0.0)$ are indistinguishable from any amount of end-to-end benchmark data: both give $(0.95)^H$. They demand opposite engineering — the first needs a better base model, the second needs a checkpoint-and-retry loop. Benchmarks report the one number that cannot tell them apart.

Breaking the tie requires labeling individual steps as errors. On open-ended tasks "wrong step" is not well defined: a detour that eventually succeeds is not an error, and correctness of step $t$ depends on the continuation. MAST-style annotation costs ~1 expert-hour per trajectory, so a study powered to resolve a 2× difference in $\rho$ across 4 models is a four-figure hour count.

Second obstruction: **horizon is endogenous.** $H$ is chosen by the agent, so conditioning on it selects on failure. Fitting $\hat\alpha$ against agent step count is biased; fitting against human task minutes (METR's choice) avoids the bias but no longer measures steps.

## 7. Current Research (as of 2026)

- **Horizon-scaling measurement.** METR continues extending the time-horizon methodology to messier, less-clean task suites; the open question they name is whether the doubling trend holds off-distribution *(frontier — verify)*.
- **Process supervision as an oracle proxy.** Process reward models (Lightman et al., ICLR 2024, "Let's Verify Step by Step") give step-level labels cheaply on math, where ground truth exists. Extending PRMs to tool-use trajectories is active at OpenAI, DeepMind and several academic groups *(frontier — verify)*.
- **Failure taxonomies and trace datasets.** Berkeley/UC groups behind MAST are building annotated multi-agent trace corpora.
- **Evaluation hygiene.** Kapoor et al. ("AI Agents That Matter", 2024) push cost-controlled agent evaluation — the exact control arm §8 requires.
- **Environment-side checkpointing.** Sandbox snapshots and transactional tool APIs that make errors non-absorbing by construction; largely industrial, little published ablation.

## 8. Concrete Next Experiment

**Question decided:** does a self-critique scaffold reduce the decay exponent $\hat\alpha$, or only raise $\rho$ at the cost of more compute?

**Scale.** 300 tasks from a step-instrumented environment where a programmatic oracle exists — WebArena or a SWE-bench-Verified subset with per-step unit-test invariants — binned into 5 horizon bins ($H \approx$ 3, 6, 12, 24, 48 oracle-minimal steps), 60 tasks per bin, 8 rollouts each = 12,000 trajectories per arm.

**Arms.**
1. *Treatment:* ReAct + per-step self-critique.
2. *Control (the one usually missing):* plain ReAct with best-of-$n$ resampling, $n$ tuned so **total inference tokens match arm 1 within 5%**.
3. *Floor:* plain ReAct, single sample.

**Measurement.** Label each step with the programmatic oracle to get $\hat\epsilon$ per arm; estimate $\hat\rho$ as the fraction of oracle-flagged-err trajectories that still succeed. Fit $\log \widehat{\Pr}[\text{success}] = -\hat\alpha H + b$ across the 5 bins.

**The deciding number.** $\Delta\hat\alpha = \hat\alpha_{\text{control}} - \hat\alpha_{\text{treatment}}$. If $\Delta\hat\alpha > 0$ with a bootstrap 95% CI excluding zero, self-critique genuinely changes the compounding rate. If the CI includes zero while the treatment's intercept $b$ is higher, the scaffold buys a constant-factor gain and the doubling-horizon trend is a base-model property, not an architecture one. With 60 tasks/bin and 8 rollouts, the design resolves $\Delta\hat\alpha \approx 0.02$ nats/step — the difference between a 34-step and a 46-step 50% horizon at $\hat\alpha = 0.02$.

## 9. Key References

- **[Foundational]** Stéphane Ross, Geoffrey Gordon, J. Andrew Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS, 2011. — arXiv:1011.0686
- **[Foundational]** Stéphane Ross, J. Andrew Bagnell. *Efficient Reductions for Imitation Learning.* AISTATS, 2010.
- **[SOTA]** Thomas Kwa et al. *Measuring AI Ability to Complete Long Tasks.* METR, 2025. — arXiv:2503.14499
- **[SOTA]** Benedikt Stroebl, Sayash Kapoor, Arvind Narayanan. *Inference Scaling Flaws: The Limits of LLM Resampling with Imperfect Verifiers.* 2024. — arXiv:2411.17501
- **[SOTA]** Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Ion Stoica, Matei Zaharia, James Zou. *Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems.* NeurIPS, 2024. — arXiv:2403.02419
- **[Evidence]** Nouha Dziri et al. *Faith and Fate: Limits of Transformers on Compositionality.* NeurIPS, 2023. — arXiv:2305.18654
- **[Evidence]** Muru Zhang, Ofir Press, William Merrill, Alisa Liu, Noah A. Smith. *How Language Model Hallucinations Can Snowball.* ICML, 2024. — arXiv:2305.13534
- **[Evidence]** Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville. *LLMs Get Lost in Multi-Turn Conversation.* 2025. — arXiv:2505.06120
- **[Evidence]** Mert Cemri et al. *Why Do Multi-Agent LLM Systems Fail?* 2025. — arXiv:2503.13657
- **[Benchmark]** Shuyan Zhou et al. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR, 2024. — arXiv:2307.13854
- **[Benchmark]** Grégoire Mialon et al. *GAIA: A Benchmark for General AI Assistants.* ICLR, 2024. — arXiv:2311.12983
- **[Benchmark]** Tianbao Xie et al. *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments.* NeurIPS, 2024. — arXiv:2404.07972
- **[Method]** Hunter Lightman et al. *Let's Verify Step by Step.* ICLR, 2024. — arXiv:2305.20050
- **[Survey]** Lei Wang et al. *A Survey on Large Language Model based Autonomous Agents.* Frontiers of Computer Science, 2024. — arXiv:2308.11432

## 10. Worked Example

Take WebArena's reported GPT-4 agent success of 14.41% against humans at 78.24%. Suppose the average task needs $H = 12$ oracle-minimal steps.

Fit the absorbing model:
$$(1-\epsilon)^{12} = 0.1441 \;\Rightarrow\; \epsilon = 1 - 0.1441^{1/12} = 1 - e^{-0.1614} = 0.149.$$
So a 14.9% per-step error rate "explains" the benchmark. Now fit the recovery model with $\rho = 0.6$:
$$\big(1 - \epsilon(1-0.6)\big)^{12} = 0.1441 \;\Rightarrow\; \epsilon(0.4) = 0.149 \;\Rightarrow\; \epsilon = 0.373.$$

Same benchmark number; per-step error rates differing by 2.5×. Nothing in the WebArena score distinguishes a fairly accurate agent that cannot recover from a sloppy agent that recovers well. The two imply different fixes and different forecasts: at $H = 48$, both give $0.1441^4 = 0.043\%$ — but a checkpointing intervention that lifts $\rho$ from 0.6 to 0.9 moves the second agent to $(1 - 0.373 \times 0.1)^{48} = 16.0\%$, a 370× gain, while doing nothing at all for the first ($\rho$ is already 0 and there is no err-state to recover from — the intervention has no purchase).

Making the obstruction visible: the decision of whether to spend the engineering budget on a better base model or on transactional checkpointing hinges entirely on $\rho$, and $\rho$ is exactly the quantity end-to-end evaluation cannot see. Every number in this calculation is derived from one published scalar; the identifiability failure is not a subtlety of the model, it is arithmetic.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*