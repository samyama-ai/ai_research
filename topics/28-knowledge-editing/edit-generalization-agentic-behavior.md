---
id: 28-knowledge-editing/edit-generalization-agentic-behavior
title: "Edit Generalization to Downstream Agentic Behavior"
topic: 28-knowledge-editing
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Edit Generalization to Downstream Agentic Behavior

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/edit-generalization-agentic-behavior` · **Status:** empirically-open

## 1. Problem Statement

A knowledge edit changes a model's answer to a factual probe: after editing, "Who is the CEO of Twitter?" returns the new value. The open question is whether that change propagates to **behavior** — multi-step tool calls, plans, retrieval decisions, and refusals in an agent loop — rather than only to **elicitation** under a paraphrase prompt.

Input: a base model $f_\theta$, an edit request $e = (s, r, o \to o^*)$, and an agentic task distribution $\mathcal{T}$ whose optimal trajectories depend on the value of $(s,r)$. Output: an edited model $f_{\theta^*}$. Decision predicate: does $f_{\theta^*}$ produce trajectories that are correct **under the post-edit world**, at a rate comparable to a model that never held the stale fact?

Three variants, of different difficulty:

- **Measurement.** Build a task family where the edited fact is *load-bearing* for the trajectory, and where trajectory success is not confounded with generic agentic competence. Currently the weakest link.
- **Method.** Produce an editor whose behavioral generalization rate approaches its elicitation rate. No method is close.
- **Theory.** Characterize when a rank-one or low-rank parameter update to an MLP can change the model's *policy* and not merely its next-token distribution on a probe. Essentially untouched.

Solving it means: an editor for which behavioral success on $\mathcal{T}$ minus paraphrase success is small (say within 10 points), with locality preserved on unrelated tasks.

## 2. Formal Setting

Let $\pi_\theta$ be the agent policy induced by $f_\theta$ under a fixed scaffold (system prompt, tool schema, decoding, step budget $H$). A trajectory is $\tau = (o_0, a_1, o_1, \dots, a_H)$ with actions $a_t \sim \pi_\theta(\cdot \mid \tau_{<t})$ and observations from environment $E$.

**Edit.** $\theta^* = \mathrm{Edit}(\theta, e)$. For locate-and-edit methods (ROME, MEMIT) this solves a constrained least-squares problem on an MLP down-projection $W_\ell$:

$$\theta^*: \quad W_\ell^* = \arg\min_W \|WK - V\|_F^2 \quad \text{s.t.} \quad Wk_* = v_*,$$

with $k_*$ the key vector for subject $s$ at layer $\ell$ and $v_*$ optimized so the model emits $o^*$.

**Measured quantities.** Fix an edit set $\mathcal{E}$, a paraphrase set $P(e)$, a neighborhood set $N(e)$ (same relation, different subject), and a task set $T(e) \subset \mathcal{T}$ whose ground-truth solution is a function of $o^*$.

$$\text{Efficacy} = \tfrac{1}{|\mathcal{E}|}\sum_e \mathbb{1}[f_{\theta^*}(x_e) = o^*], \qquad \text{Paraphrase} = \tfrac{1}{|\mathcal{E}||P(e)|}\sum_{e,p} \mathbb{1}[f_{\theta^*}(p) = o^*]$$

$$\text{BehGen}(e) = \mathbb{E}_{\tau \sim \pi_{\theta^*}, T(e)}\big[R^*(\tau)\big], \qquad R^*(\tau) \in \{0,1\} \text{ under the post-edit world model}$$

The quantity of interest is the **generalization gap**

$$\Delta = \text{Paraphrase} - \text{BehGen}.$$

$\Delta$ alone is not enough, because a model that was bad at the task before the edit will have low $\text{BehGen}$ for reasons unrelated to editing. Normalize against a **retrained/counterfactual-native control** $\pi_{\theta_c}$ — a model prompted with $o^*$ in context, or fine-tuned on a corpus in which $o^*$ holds:

$$\hat\Delta = \frac{\mathbb{E}[R^*(\tau_{\theta_c})] - \mathbb{E}[R^*(\tau_{\theta^*})]}{\mathbb{E}[R^*(\tau_{\theta_c})]}.$$

Locality is measured as drift on tasks independent of $e$: $\text{Drift} = \mathbb{E}_{\mathcal{T}\setminus T(e)}[R(\tau_\theta)] - \mathbb{E}[R(\tau_{\theta^*})]$.

**Assumptions, and which are violated.**

1. *The scaffold is fixed and the environment deterministic.* Violated: real agent benchmarks have stochastic tools, API drift, and LLM judges; run-to-run variance on $\tau$-bench-style tasks is several points, which is the same order as the effect being measured.
2. *$T(e)$ depends on $e$ only.* Violated in practice: constructed tasks leak the answer through the prompt or through tool output, so a stale model can still succeed.
3. *One edit at a time.* Violated for any realistic update stream; sequential editing degrades the model independently of any single edit (Gupta et al., 2024).
4. *$R^*$ is computable.* Partly violated: for open-ended trajectories there is no ground-truth post-edit world, so $R^*$ is scored by a judge model that itself may not have been "edited."

## 3. State of the Art

**Editing methods (systems SOTA).** MEMIT (Meng et al., ICLR 2023) scales to $10^4$ edits on GPT-J/GPT-NeoX. AlphaEdit (Fang et al., ICLR 2025) projects the update onto the null space of preserved-knowledge keys and reports large reductions in post-edit degradation. WISE (Wang et al., NeurIPS 2024) uses a side memory with a routing gate for lifelong editing. In-context editing (Zheng et al., EMNLP 2023) remains a strong and often-underrated baseline.

**Propagation benchmarks (established that a gap exists).** MQuAKE (Zhong et al., EMNLP 2023) measures multi-hop questions whose answer changes under an edit. RippleEdits (Cohen et al., TACL 2024) measures six ripple-effect axes including logical generalization and compositionality. ReCoE (Hua et al., 2024) probes reasoning-based propagation. HalluEditBench (Huang et al., ICLR 2025) tests whether editing actually corrects real hallucinations.

**Claimed but unablated.** Every headline "edit success" number is elicitation on templated probes. No published editor has been ablated against a native-counterfactual control on a *multi-step tool-use* benchmark. Claims that a method "generalizes" rest on paraphrase and portability scores, which are static QA.

**Benchmark-number-only results.** The gap figures below exist as leaderboard entries on MQuAKE/RippleEdits, not as mechanistically explained findings; MQuAKE additionally has a known shortcut where a memory-based scaffold (MeLLo) inflates scores without the edit propagating into weights.

**Theory SOTA.** There is none for behavior. Existing theory covers the closed-form rank-one update and the null-space constraint — statements about weight matrices, not about policies.

## 4. What Is Known

- **Efficacy is near-saturated, propagation is not.** ROME/MEMIT reach ~99–100% efficacy on CounterFact for GPT-J (6B) and GPT-2 XL (1.5B), while multi-hop accuracy on MQuAKE for the same edited models falls to the low single digits to ~10–20% depending on hop count — measured at 6B–20B scale.
- **Ripple effects are mostly absent.** On RippleEdits, parameter-editing methods score far below in-context baselines on compositionality and logical generalization; GPT-3-scale in-context editing beat weight editing on several axes — measured on GPT-2 XL, GPT-J, and LLaMA-2-7B.
- **Editing degrades general ability.** Gu et al. (2024) show that after a few dozen sequential edits, ROME/MEMIT-edited LLaMA-1/2-7B lose measurable performance on reasoning, summarization, and open-domain QA. Gupta et al. (ACL Findings 2024) document gradual forgetting and abrupt "model collapse" from single disabling edits with ROME.
- **Correcting a probe does not correct a hallucination.** HalluEditBench (ICLR 2025) reports high efficacy alongside much weaker generalization to related questions on LLaMA-2/3 and Mistral-7B.
- **Scale of measurement.** Nearly everything above is 1.5B–20B dense decoders. There is little published editing evidence at 70B+, and effectively none for reasoning/agentic post-trained models.

## 5. What Is Not Known

- **Empirically open.** Whether $\hat\Delta$ is large on a real agentic benchmark. Nobody has run: edit a fact → run $\tau$-bench/WebArena-style tasks that depend on it → compare against a native-counterfactual control. The experiment is runnable today with a few thousand GPU-hours.
- **Empirically open.** Whether behavioral propagation improves, degrades, or is unchanged with model scale and with RL post-training. Plausible either way: RL-trained policies may route facts through more entangled circuits.
- **Methodologically blocked.** $R^*$ for open-ended agent trajectories. There is no accepted way to score "correct under a counterfactual world" when the judge, the tools, and the environment all still encode the pre-edit fact.
- **Methodologically blocked.** Separating edit failure from scaffold failure. A stale action may come from a stale weight, a retrieved document, or a system prompt.
- **Theoretically open.** Whether a rank-$k$ update to $m$ MLP layers can, in principle, change a policy's behavior on tasks requiring $h$ compositional hops without $O(\text{branching}^h)$ separate edits. No lower bound, no construction.

## 6. Why It Is Hard

The core obstruction is **confounded measurement compounded by absent ground truth**. Agentic success rates are noisy (a few points run-to-run under LLM judging), the effect size of an edit on trajectory success is unknown but likely small, and edits themselves damage general capability — so a drop in $\text{BehGen}$ is not attributable to failed propagation rather than to collateral damage. Without the native-counterfactual control arm, the two are non-identifiable.

Second: **task construction leaks**. Building $T(e)$ such that the edited fact is strictly load-bearing, unavailable from tool output, and not already known to the judge, is hard enough that most existing "propagation" datasets are single-model-turn QA in disguise.

Third: cost. A single arm is $|\mathcal{E}|$ edits $\times$ $|T(e)|$ tasks $\times$ $H$ steps of rollout. At 200 edits, 10 tasks each, 30 steps, that is 60k model calls per arm — and you need four arms.

## 7. Current Research (as of 2026)

- **Null-space and preservation-constrained editing** (AlphaEdit line, NUS/NExT and collaborators) — reduces collateral damage; propagation to behavior untested.
- **Memory- and retrieval-based updating** (SERAC/WISE lineage; RAG-first update pipelines in industry) — sidesteps weight editing entirely and is the pragmatic default for agents.
- **Belief-consistency and ripple-effect evaluation** (Tel Aviv / Google groups behind RippleEdits; MQuAKE authors at Princeton).
- **Agent benchmark harnesses** ($\tau$-bench, WebArena, SWE-bench-style scaffolds) as substrate for behavioral evaluation — not yet joined to the editing literature. *(frontier — verify)*
- **Editing reasoning models.** Whether long-CoT/RL-trained models re-derive stale facts inside the trace after a weight edit. *(frontier — verify)*

## 8. Concrete Next Experiment

**Scale.** One open-weight instruct model at 8B and one at 70B (e.g. Llama-3.1). $|\mathcal{E}| = 100$ edits over entity attributes that a tool-use environment consumes (airline policy values, product prices, org charts). $|T(e)| = 10$ hand-built $\tau$-bench-style tasks per edit, each requiring at least two tool calls whose correct arguments depend on $o^*$, with the fact **removed** from all tool outputs. 10k rollouts per arm.

**Arms.**
1. Base model, pre-edit world (sanity ceiling).
2. **Control:** base model with $o^*$ supplied in the system prompt (native-counterfactual).
3. MEMIT-edited.
4. AlphaEdit-edited.
5. Locality arm: 200 tasks independent of $\mathcal{E}$, all conditions.

**Deciding number.** $\hat\Delta$ for arm 3/4 against arm 2, with 95% CI from a paired bootstrap over tasks. Report alongside paraphrase accuracy on the same edits. **If $\hat\Delta > 0.5$ while paraphrase accuracy exceeds 0.9, weight editing does not reach behavior**, and the field's headline metric is measuring elicitation only. If $\hat\Delta < 0.15$, propagation is real and the problem reduces to scaling and locality.

Cost estimate: ~$4\times10^5$ model calls, feasible on 8×H100 for the 8B arm in under a week.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR, 2023. — arXiv:2210.07229
- **[Foundational]** Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, Christopher D. Manning. *Fast Model Editing at Scale.* ICLR, 2022.
- **[SOTA]** Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, Danqi Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP, 2023.
- **[SOTA]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024.
- **[SOTA]** Junfeng Fang et al. *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models.* ICLR, 2025.
- **[SOTA]** Peng Wang et al. *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models.* NeurIPS, 2024.
- **[Empirical]** Jia-Chen Gu et al. *Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue.* EMNLP, 2024.
- **[Empirical]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL, 2024.
- **[Empirical]** Baixiang Huang et al. *Can Knowledge Editing Really Correct Hallucinations?* ICLR, 2025.
- **[Environment]** Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024.
- **[Environment]** Shuyan Zhou et al. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR, 2024.
- **[Survey]** Ningyu Zhang et al. *A Comprehensive Study of Knowledge Editing for Large Language Models.* 2024.

## 10. Worked Example

**Edit.** In a customer-service environment, the policy fact "checked-bag fee for basic economy = \$35" is edited to \$50 via MEMIT on Llama-3.1-8B-Instruct.

**Elicitation.** Probe "What is the checked-bag fee for basic economy?" → "\$50". Ten paraphrases → 10/10 correct. Reported edit success: **1.00**.

**Behavior.** Task: user books two bags on a basic-economy fare and asks for the total. Correct trajectory: call `get_fare_class`, call `compute_total`, charge $2 \times 50 = 100$ plus fare. Observed across 50 rollouts:

```
arm                      says $50 on probe   charges $100   charges $70
base (pre-edit world)          0/50            —              48/50
control (fact in prompt)      50/50           44/50            3/50
MEMIT-edited                  50/50            9/50           37/50
```

$\hat\Delta = (0.88 - 0.18)/0.88 = 0.80$.

**Where the obstruction becomes visible.** The edited model states \$50 when asked and then arithmetically uses \$35 two turns later — the edit reached the token distribution on the probe but not the value the model carries into `compute_total`. Now the confound: of the 41 failed rollouts, 6 also failed in the *control* arm, meaning they are scaffold failures, not propagation failures. Without arm 2 you cannot tell 35 from 41, and 6/50 is 12 points — larger than the difference between many published editors. The control arm is not a refinement of this experiment; it is the only thing that makes the number mean anything.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*