---
id: 19-evaluation/multi-turn-eval-simulated-users
title: "Multi-Turn Dialogue Evaluation Without Human Interlocutors"
topic: 19-evaluation
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Multi-Turn Dialogue Evaluation Without Human Interlocutors

> **Topic:** Evaluation & Benchmarking · **ID:** `19-evaluation/multi-turn-eval-simulated-users` · **Status:** open

## 1. Problem Statement

A single-turn benchmark can be scored offline: the input is fixed, so the model's output is the only random variable. Multi-turn evaluation cannot, because turn $t$ of the user's input depends on turns $1..t-1$ of the system's output. Evaluating a dialogue policy therefore requires an interlocutor. Hiring humans is slow, expensive, and non-reproducible; the standard substitute is a **simulated user** — usually another LLM given a goal, a persona, and a stopping rule.

**The problem:** determine whether a score obtained against a simulated user is a valid estimate of the score the same system would obtain against the human population it will serve, and construct simulators for which it is.

Three variants, with different difficulty:

- **Measurement.** Given a simulator $U_\phi$ and a system population $\{\pi_1..\pi_m\}$, estimate the sim-to-real gap and the rank distortion it induces. Empirically runnable today; almost never run.
- **Method.** Build $U_\phi$ whose induced ranking over systems matches the human ranking within a stated tolerance, at a stated cost per dialogue.
- **Theory.** Characterise when a ranking under one interlocutor distribution transfers to another. This is off-policy evaluation with a non-stationary, adaptive behaviour policy; no useful bound is known for the LLM case.

Solving it means: a simulator plus a validation protocol such that a reported multi-turn score comes with an interval that covers the human-elicited score, and the interval is narrow enough to separate the systems people actually compare.

## 2. Formal Setting

A dialogue is a trajectory $\tau = (u_1, a_1, \dots, u_T, a_T)$, where $u_t$ is a user turn and $a_t \sim \pi_\theta(\cdot \mid h_{t})$ a system turn, $h_t$ the history. The user is drawn from an interlocutor distribution: $u_t \sim U(\cdot \mid h_t, g)$ with latent goal $g \sim G$. A scorer $R(\tau, g) \in [0,1]$ is either **programmatic** (final database state equals the goal state — measured by comparing a serialised environment record to a reference, no judgment involved) or a **judge model** $J$ (measured as the mean of $J$'s verdicts over the trajectory).

The estimand is the human-elicited value

$$V_H(\pi) \;=\; \mathbb{E}_{g \sim G,\; \tau \sim (\pi, U_H)}\big[R(\tau, g)\big],$$

and the estimator is the simulator-elicited value $V_\phi(\pi)$ with $U_H$ replaced by $U_\phi$. Two quantities matter, and they are not the same:

- **Level gap** $\varepsilon(\pi) = |V_\phi(\pi) - V_H(\pi)|$.
- **Rank fidelity** $\rho = \operatorname{Kendall-}\tau\big(\{V_\phi(\pi_i)\}, \{V_H(\pi_i)\}\big)$ over the evaluated system set. A large but *constant* level gap is harmless for model selection; a small gap that is systematically larger for one architecture is not.

**Reliability** is separate from mean quality. $\text{pass}^k$ (τ-bench) is the probability that all $k$ independent rollouts of the same task succeed, measured as $\mathbb{E}_{\text{task}}\big[\binom{c}{k}/\binom{n}{k}\big]$ with $c$ successes out of $n$ rollouts. Variance decomposes as

$$\operatorname{Var}(R) = \underbrace{\operatorname{Var}_g}_{\text{task}} + \underbrace{\mathbb{E}_g\operatorname{Var}_{U_\phi}}_{\text{simulator}} + \underbrace{\mathbb{E}_{g,U}\operatorname{Var}_{\pi}}_{\text{system}} + \underbrace{\operatorname{Var}_J}_{\text{judge}},$$

and published multi-turn numbers almost never report the middle two terms separately, so a confidence interval computed over tasks alone understates the true error.

Assumptions, with those known to be violated marked:

- **A1 — Goal exogeneity.** $g$ is fixed before the dialogue. **Violated:** humans form and revise goals mid-conversation; a fixed agenda cannot produce a genuine mind change.
- **A2 — Interlocutor non-adaptivity to system identity.** $U$ depends on $h_t$ only. **Violated:** an LLM simulator sharing a base model with $\pi_\theta$ accommodates its phrasing and repairs its errors, inflating $V_\phi$ for same-family systems.
- **A3 — Judge independence.** $J \perp U_\phi \mid \tau$. **Violated whenever the same base model plays both roles**; errors are positively correlated and the resulting bias does not shrink with more dialogues.
- **A4 — Population coverage.** $\operatorname{supp}(U_\phi) \supseteq \operatorname{supp}(U_H)$. **Violated:** simulators produce fluent, cooperative, on-topic turns; real users are terse, contradictory, and underspecify.

## 3. State of the Art

**Established.**

- Word-overlap metrics do not measure dialogue quality. Liu et al. (EMNLP 2016) found BLEU/ROUGE/METEOR correlations with human judgment near zero on unconstrained dialogue. Reproduced widely.
- LLM judges agree with human pairwise preference at rates comparable to human–human agreement on *single-response* comparisons: MT-Bench reports ~85% GPT-4/human agreement against 81% human/human (Zheng et al., NeurIPS D&B 2023). This is a **turn-level** result and is routinely over-extended to whole dialogues.
- Multi-turn degradation is real and large. Laban et al. (2025) sharded single-turn instructions into multi-turn "underspecified" conversations across 6 tasks and 15 LLMs, >200k simulated conversations: **average performance drop of 39%**, decomposed as a ~16% loss in aptitude and a ~112% rise in unreliability. Every model tested degraded.
- Reliability collapses under repetition. τ-bench (Yao et al., 2024): GPT-4o reaches pass^1 ≈ 61% on retail and ≈ 35% on airline, but pass^8 on retail falls to roughly 25% — the same policy, the same tasks, eight samples.

**Claimed but unablated.**

- That LLM user simulators are a valid stand-in for humans. τ-bench, τ²-bench, MultiChallenge and most agentic leaderboards use a frontier LLM as the user and report the resulting number as the system's score. None publishes $\varepsilon(\pi)$ or $\rho$ against a matched human-interlocutor arm at scale.
- That persona conditioning increases realism. Widely adopted, evaluated mostly by surface diversity statistics rather than by improved agreement with human-elicited rankings.
- Reported frontier scores on MultiChallenge (Sirdeshmukh et al., 2025) — all tested frontier models under 50% — exist **only as benchmark numbers** against a fixed scripted/simulated protocol, with no human-interlocutor calibration arm.

**Systems SOTA** is τ²-bench-style dual-control environments (Barres et al., 2025) with programmatic state-based rewards, which at least remove the judge term $\operatorname{Var}_J$ for the outcome, leaving the simulator term unmeasured. **Theory SOTA** is the classical statistical user-simulation literature (Schatzmann et al., 2006 survey; hidden-agenda model, Schatzmann & Young, IEEE TASLP 2009), which gives fitted, evaluable simulators for slot-filling domains but no transfer guarantee for open-ended dialogue.

## 4. What Is Known

- **Scale of degradation:** 39% mean drop, 15 models, 6 tasks, 200k+ simulated conversations (Laban et al., 2025).
- **Reliability gap:** pass^1 ≈ 61% → pass^8 ≈ 25% for GPT-4o, retail domain, ~115 tasks (τ-bench, 2024).
- **Judge–human agreement, single turn:** ~85% (MT-Bench, 80 questions × 6 models, ~3k expert votes, 2023).
- **Human preference data at scale is feasible:** Chatbot Arena collected >240k votes over 90+ models (Chiang et al., ICML 2024) — but conversations there are short, self-selected, and unpaired to fixed goals, so it does not supply the $U_H$ arm this problem needs.
- **Fitted simulators transfer within narrow domains:** agenda-based simulators trained on MultiWOZ (10,438 dialogues, Budzianowski et al., EMNLP 2018) reproduce human-corpus turn statistics for slot-filling, and policies trained against them transfer to human users in those domains.
- **Self-play approximates interactive human evaluation better than static metrics** for open-domain chat (Ghandeharioun et al., NeurIPS 2019) — correlation improved, but well below the reliability needed for model selection.

## 5. What Is Not Known

- **Empirically open (the main gap).** No published study measures $\rho$ between simulator-elicited and human-elicited rankings over ≥8 frontier systems on ≥300 matched goals with ≥3 human dialogues per goal. The experiment is runnable today for roughly the cost of a mid-sized human study. Nobody has run it.
- **Empirically open.** Whether the same-family accommodation effect (A2) is large. The self-preference literature documents it for judging; the analogous *interlocutor* effect has not been isolated.
- **Methodologically blocked.** "Realism" of a simulator has no accepted operational definition. Distributional similarity to a human corpus, human indistinguishability, and rank fidelity are three different targets that can be optimised against each other, and no consensus exists on which one licenses a benchmark claim.
- **Theoretically open.** No bound on $|V_\phi - V_H|$ in terms of any tractable divergence between $U_\phi$ and $U_H$. Standard off-policy bounds require bounded importance weights over turn sequences; those weights are unbounded here and grow exponentially in $T$.

## 6. Why It Is Hard

The obstruction is **confounded measurement with absent ground truth, compounded by shared-model correlation**. The simulator, the system, and often the judge are the same class of model, so their errors are not independent: a simulator that fails to press an ambiguity is exactly the system that fails to ask about it. The bias term this introduces does not shrink with sample size — running 10,000 simulated dialogues narrows the interval around the *wrong* quantity. And the reference quantity $V_H$ is itself expensive and noisy: goal-directed human dialogues cost roughly $2–$8 each and inter-annotator agreement on dialogue-level quality is typically weak ($\alpha$ in the 0.2–0.5 range in interactive-evaluation studies). So the estimator is biased by an unmeasured amount toward a target that is itself only measurable to within a wide band.

Secondary: **non-identifiability**. A low score has at least four causes — bad system, unrealistic simulator, mis-specified goal, wrong judge — and a single scalar cannot separate them.

## 7. Current Research (as of 2026)

- **Programmatic-reward agentic environments.** Sierra's τ-bench / τ²-bench line; similar dual-control environments from agent-platform vendors. Removes judge noise for outcomes; simulator validity still unaddressed.
- **Sharded / underspecified-instruction protocols.** Microsoft Research and Salesforce Research (Laban et al.) — converting single-turn benchmarks into controlled multi-turn ones, which fixes goal ground truth while keeping the simulator synthetic.
- **User simulation as its own research object** in information access — Balog and Zhai's programme, which is the clearest articulation of validity criteria for simulators.
- **Human-in-the-loop calibration arms** for agentic benchmarks *(frontier — verify)*: several labs are reported to run small paired human-interlocutor studies internally for release evaluations; results are not public, so the field cannot audit them.
- **Interactive arena formats with fixed goals** *(frontier — verify)*, aiming to make crowd conversations goal-matched so they can serve as the $U_H$ arm.

## 8. Concrete Next Experiment

**Question:** does simulator-elicited ranking match human-elicited ranking?

- **Scale.** 300 goals from an existing programmatic-reward environment (τ-bench retail/airline is sufficient). 8 systems spanning ≥3 model families and ≥2 capability tiers. Two arms per (goal, system) cell:
  - **Treatment:** LLM simulator, 8 rollouts per cell → 19,200 simulated dialogues.
  - **Control arm:** paid human interlocutors given the identical goal card and identical stopping rule, 3 dialogues per cell → 7,200 human dialogues. At ~$5 each, ≈$36k. Reward is computed programmatically from final environment state in **both** arms, so the judge term is eliminated by construction and only the interlocutor differs.
  - **Second control (cheap, do it anyway):** re-run the treatment arm with a simulator from a different model family, to bound the same-family accommodation effect.
- **Deciding number.** Kendall's $\tau$ between the 8 simulator-elicited and 8 human-elicited system rankings, with a bootstrap CI over goals. **$\tau \geq 0.80$ with the CI lower bound above 0.6** licenses simulator-only reporting for model selection in this domain. **$\tau < 0.6$** means every simulator-only leaderboard number currently published is uncalibrated for ranking, not just for level.
- **Secondary readouts:** $\varepsilon(\pi)$ per system (is the gap constant or system-dependent?); pass^8 in both arms; the ratio $\mathbb{E}_g\operatorname{Var}_{U}$ human-vs-simulated, which quantifies how much behavioural diversity the simulator is missing.

Total: ~27k dialogues, ~$40k, four weeks. This is the smallest experiment that converts the field's central assumption from an assumption into a measurement.

## 9. Key References

- **[Foundational]** Schatzmann, Weilhammer, Stuttle, Young. *A survey of statistical user simulation techniques for reinforcement-learning of dialogue management strategies.* The Knowledge Engineering Review, 2006.
- **[Foundational]** Schatzmann, Young. *The Hidden Agenda User Simulation Model.* IEEE Transactions on Audio, Speech, and Language Processing, 2009.
- **[Foundational]** Liu, Lowe, Serban, Noseworthy, Charlin, Pineau. *How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation.* EMNLP 2016. — arXiv:1603.08023
- **[Foundational]** Budzianowski et al. *MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling.* EMNLP 2018. — arXiv:1810.00278
- **[SOTA]** Yao, Shinn, Razavi, Narasimhan. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[SOTA]** Barres et al. *τ²-bench: Evaluating Conversational Agents in a Dual-Control Environment.* 2025. — arXiv:2506.07982
- **[SOTA]** Laban, Hayashi, Zhou, Neville. *LLMs Get Lost In Multi-Turn Conversation.* 2025. — arXiv:2505.06120
- **[SOTA]** Sirdeshmukh et al. *MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs.* 2025. — arXiv:2501.17399
- **[SOTA]** Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks, 2023. — arXiv:2306.05685
- **[Related]** Ghandeharioun, Shen, Jaques, Ferguson, Jones, Lapedriza, Picard. *Approximating Interactive Human Evaluation with Self-Play for Open-Domain Dialog Systems.* NeurIPS 2019. — arXiv:1906.09308
- **[Related]** Chiang et al. *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* ICML 2024. — arXiv:2403.04132
- **[Related]** Finch, Paek, Choi. *Don't Forget Your ABCs: Evaluating the State-of-the-Art in Chat-Oriented Dialogue Systems.* ACL 2023.
- **[Survey]** Balog, Zhai. *User Simulation for Evaluating Information Access Systems.* 2023. — arXiv:2306.08550
- **[Survey]** Bai et al. *MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues.* ACL 2024. — arXiv:2402.14762

## 10. Worked Example

**Goal card (τ-bench retail style):** *"You bought a pair of running shoes last month. You want them exchanged for a half-size larger. You do not remember your order number. If the agent says the exchange window has closed, you accept it."* Reward $R = 1$ iff the final database state contains an exchange for the correct item at the correct size, or no exchange if the window is genuinely closed.

**Simulated arm.** The LLM user, prompted with the card, opens with: *"Hi, I'd like to exchange the running shoes from my order last month for a half size up."* It has stated the item, the intent, and the timeframe in one turn. The agent looks up the profile, finds one shoe order, and completes the exchange. Across 8 rollouts, 7 succeed: $\hat V_\phi = 0.875$, and pass^8 for this task is 0.

**Human arm.** Three humans given the identical card open with: *"my shoes dont fit"*, *"hello I need help with an order"*, and *"can I return something"*. None mentions size in the first turn. Two of the three, when the agent asks "do you have your order number?", answer *"no"* and stop — they do not volunteer the timeframe, because the card said they don't remember the number and they read that as the salient fact. One agent asks a clarifying question and recovers; two guess the wrong order. $\hat V_H = 1/3 = 0.333$.

**The gap.** $\varepsilon = 0.875 - 0.333 = 0.542$ on this task. Now the part that makes it a research problem, not a bug: the gap is **not constant across systems**. A system whose policy is "ask one clarifying question before any lookup" loses nothing in the simulated arm (the simulator already gave it the information, so the question is redundant and costs one turn) and gains ~0.5 in the human arm. A system whose policy is "act on first sufficient-looking evidence" scores at the top simulated and near the bottom human. So the simulator does not merely shift the scale — it **reverses the ordering of exactly the two design choices the evaluation exists to compare**. Running 10,000 more simulated dialogues tightens the interval around 0.875 and changes nothing.

That is the obstruction in one instance: the simulator's cooperativeness is a hidden covariate correlated with the system property under test, so more data does not help and only a human control arm can detect it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*