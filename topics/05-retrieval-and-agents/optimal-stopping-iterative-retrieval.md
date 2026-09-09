---
id: 05-retrieval-and-agents/optimal-stopping-iterative-retrieval
title: "Optimal Stopping in Iterative Retrieval Agents"
topic: 05-retrieval-and-agents
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Optimal Stopping in Iterative Retrieval Agents

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/optimal-stopping-iterative-retrieval` · **Status:** open

## 1. Problem Statement

An iterative retrieval agent alternates between issuing a query, reading results, and deciding whether to answer or search again. The decision "search again or stop and answer" is an optimal stopping problem. It is currently made by heuristics: a fixed hop budget, a special token the model emits, or an entropy threshold.

- **Input.** A question $q$, a corpus $\mathcal{C}$, a retriever $R$, a policy LLM $\pi$, and a per-call cost.
- **Output.** A stopping rule $\tau$ — a map from the transcript so far to $\{\text{continue}, \text{answer}\}$.
- **Objective.** Maximize expected answer utility minus expected cost: $\mathbb{E}[U(a_\tau) - c\tau]$.
- **Solved** would mean: a rule that, at fixed average cost, matches or beats the best fixed budget on out-of-distribution question mixes, *and* whose regret against the retrospective oracle stopping time is bounded and measured.

Three variants, different difficulty:

| Variant | Question | Difficulty |
|---|---|---|
| **Measurement** | Is marginal value-per-hop estimable from the transcript alone? | methodologically blocked |
| **Method** | Can a learned rule beat a tuned fixed budget on the same cost curve? | empirically open |
| **Theory** | Does the LLM-retrieval process satisfy conditions (monotone case, index-ability) under which a myopic or index rule is optimal? | theoretically open |

Conflating them is why "adaptive RAG" papers report accuracy gains that are actually budget gains.

## 2. Formal Setting

State after $t$ hops: $s_t = (q, \{(q_i, d_i)\}_{i\le t})$, the question plus queries issued and documents returned. **Measured as:** the literal token context the policy conditions on, so $|s_t|$ is countable in tokens.

Answer distribution $a_t \sim \pi(\cdot \mid s_t)$. Utility $U(a) \in [0,1]$ — **measured as** exact match or a judge score against a gold answer; both are noisy, and judge noise on multi-hop sets is the dominant variance term.

Cost is not one number. **Measured as** $c_t = \alpha \ell_t + \beta$, where $\ell_t$ is prompt+completion tokens for hop $t$ (from the API usage field) and $\beta$ is retrieval latency in seconds converted at the deployment's dollars-per-second. Papers that report "number of hops" are collapsing an $\ell_t$ that grows roughly linearly in $t$ because context accumulates, so hop count understates late-hop cost by 2–5$\times$.

The value function under horizon $T$:

$$V_t(s_t) = \max\Big\{\underbrace{\mathbb{E}[U(a_t)\mid s_t]}_{\text{stop}},\; \underbrace{-c_{t+1} + \mathbb{E}_{s_{t+1}}[V_{t+1}(s_{t+1})\mid s_t]}_{\text{continue}}\Big\}$$

with $V_T = \mathbb{E}[U(a_T)]$. The optimal rule is the Snell envelope: stop at the first $t$ where the stop term wins. Every deployed system approximates the continuation value with a scalar confidence.

Assumptions, and their status:

1. **Markov in $s_t$.** Holds by construction (the transcript *is* the state) but the state space is the space of token sequences, so no tabular solution exists.
2. **Monotone value:** $\mathbb{E}[U]$ non-decreasing in $t$. **Violated.** Retrieved distractors reduce accuracy; context-position degradation (Liu et al., TACL 2024) makes long transcripts worse. Empirically, per-question accuracy-vs-hop curves are non-monotone, so the one-step-look-ahead (myopic) rule is not optimal.
3. **Calibrated self-assessment:** the policy's stated confidence tracks $\mathbb{E}[U \mid s_t]$. **Partially violated.** Calibration degrades sharply after RLHF and under retrieval of contradicting evidence.
4. **Independent hops.** Violated: query $q_{t+1}$ is written by the same model that already failed, so failures are correlated across hops — the main reason extra hops help less than an i.i.d. model predicts.

Classical results that would apply if assumptions held: Wald's SPRT (1945) is optimal for sequential binary hypothesis tests with i.i.d. observations; the Gittins index (Gittins, 1979) gives optimality for independent-arm bandits with discounting. Neither's conditions hold here — assumption 4 kills both.

## 3. State of the Art

**Established (ablated, reproduced):**

- **Popularity-gated retrieval.** Mallen et al., *When Not to Trust Language Models* (ACL 2023): retrieve only when entity popularity is below a threshold. Cuts retrieval calls substantially at equal or better accuracy on PopQA. Ablated against always-retrieve and never-retrieve — a genuine cost-quality Pareto move, but a *zero-or-one* rule, not a stopping rule.
- **Interleaved retrieve-and-reason beats one-shot.** IRCoT (Trivedi et al., ACL 2023) reports retrieval recall gains up to ~21 points and QA gains up to ~15 points over one-shot retrieval on HotpotQA / 2WikiMultihopQA / MuSiQue with GPT-3-class and Flan-T5 models. Establishes that hops help; says nothing about when to stop.

**Claimed but under-ablated:**

- **Self-RAG** (Asai et al., ICLR 2024): reflection tokens (`Retrieve?`, `IsSupported`) let a 7B/13B model gate retrieval. The stopping ablation against a matched-cost fixed-budget arm is not reported.
- **FLARE** (Jiang et al., EMNLP 2023): retrieve when the next-sentence token probability drops below a threshold. The threshold is tuned per dataset; sensitivity is reported thinly.
- **Adaptive-RAG** (Jeong et al., NAACL 2024): a small classifier routes queries to no-retrieval / single-hop / multi-hop. Strong efficiency numbers, but the classifier trains on labels derived from which pipeline happened to succeed — the label is not the optimal stopping time.
- **RL-trained search agents** (Search-R1, Jin et al. 2025; R1-Searcher, 2025): outcome-reward RL over a search tool. Reported relative gains on the order of 20–30% over RAG baselines for Qwen2.5-3B/7B across seven QA sets. The learned policy stops implicitly; no paper isolates the stopping component from the query-writing component.

**Benchmark-number-only:** frontier deep-research agents on BrowseComp (Wei et al., 2025) show large accuracy gains with more browsing, but neither the stopping rule nor its cost curve is published, so these are not comparable results.

## 4. What Is Known

- **Returns saturate fast.** Across HotpotQA, 2WikiMultihopQA, MuSiQue with 7B–70B policies, most accuracy from iteration arrives in the first 2–3 hops; hops 4+ typically add ≲1–2 EM points while cost keeps climbing. Measured at the scale of $10^3$-question dev sets.
- **Extra retrieval can hurt.** Distractor documents lower answer accuracy; the "lost in the middle" position effect (Liu et al., TACL 2024) is reproduced across models and shows a U-shaped accuracy curve over context position. So the stopping problem is not merely a cost problem — over-searching costs accuracy.
- **Models have some self-knowledge.** Kadavath et al., *Language Models (Mostly) Know What They Know* (2022): self-evaluated $P(\text{IK})$ is informative and improves with scale, measured up to 52B. This is the signal every stopping heuristic leans on. It is informative, not calibrated, out of distribution.
- **Fixed budgets are strong baselines.** Where papers report matched-cost comparisons, adaptive rules beat a *tuned* fixed budget by small margins; most of the headline gain in "adaptive" papers comes from spending more on hard questions, which a per-question cost sweep also captures.

## 5. What Is Not Known

- **Theoretically open.** No characterization of when the LLM-search process is index-able or falls in the monotone case. Without monotonicity (assumption 2, known violated) there is no proof that any one-step-look-ahead rule is optimal, and no regret bound for a learned stopping rule against the Snell envelope in this setting.
- **Empirically open.** Nobody has published the *retrospective oracle stopping curve*: for a fixed agent, run every question to $T=8$ hops, record $U$ at each hop, and compute the achievable frontier of an oracle that stops at each question's best hop. That number is runnable today for ~$10^4$ questions and would tell us how much headroom any stopping rule has. Its absence is the single biggest gap on this page.
- **Methodologically blocked.** The marginal value of one more hop, $\Delta_t = \mathbb{E}[U \mid s_{t+1}] - \mathbb{E}[U \mid s_t]$, is not identifiable from a single rollout: the counterfactual "what if I had stopped" is unobserved unless you branch. Confidence estimates conflate *the answer is wrong* with *the evidence is insufficient* — two states requiring opposite actions (stop and abstain vs. keep searching).

## 6. Why It Is Hard

The specific obstruction is **confounded credit assignment between the stopping rule and the query-writing policy**, compounded by **absent oracle labels**.

- An agent that stops early may be right because it stopped well, or because its queries were good enough by hop 2. An agent that searches on may be recovering from a bad hop-1 query. Outcome reward gives one scalar for both. No published system varies stopping while holding the query sequence fixed — which is possible (replay the same transcript, truncate at each $t$) and nearly nobody does it.
- **No ground truth for $\tau^\star$.** Supervision requires knowing the optimal stopping time per question; the labels used in practice ("the pipeline that happened to answer correctly") are a biased proxy that is only defined for questions the pipeline solved.
- **The evaluation does not measure the named thing.** Multi-hop QA accuracy at a fixed budget is reported as evidence about *adaptivity*. Adaptivity is a claim about the *shape of the cost-quality curve*, which needs at least three points per arm, not one.

## 7. Current Research (as of 2026)

- **RL over search tools with cost in the reward.** Post-Search-R1 work adding explicit token/tool-call penalties to the outcome reward; the open question is whether cost-shaped reward yields a genuinely better frontier or just slides along the existing one *(frontier — verify)*.
- **Verifier-gated stopping.** Using a separate process/answer verifier as the continuation-value estimate rather than the policy's own logits. Related to process-reward-model work from OpenAI and DeepMind lines; application to retrieval stopping is early *(frontier — verify)*.
- **Test-time-compute scaling laws for agents.** Extending the sample-vs-accuracy scaling picture to sequential tool use; the sequential case has correlated failures, so the parallel-sampling laws do not transfer *(frontier — verify)*.
- **Deep-research systems** (OpenAI, Google, Anthropic, Perplexity) all ship a stopping mechanism; none is documented. Academic replication is the constraint.

## 8. Concrete Next Experiment

**The oracle-headroom experiment.** Smallest thing that settles whether adaptive stopping is worth pursuing at all.

- **Scale.** 3,000 questions: 1,000 each from HotpotQA (dev, distractor-free retrieval over Wikipedia), MuSiQue-Ans, and Bamboogle. One 8B open policy (e.g. Qwen2.5-7B-Instruct) and one frontier API model. Run each question to a hard $T=8$ hops with a fixed ReAct-style loop. At *every* hop $t \in \{1..8\}$, fork and force an answer. Record $U_t \in \{0,1\}$ and cumulative tokens $\ell_{\le t}$. Cost: ~$3{,}000 \times 8 \times 2$ forced answers ≈ 48k extra generations — under $500 on the API model, hours on one 8×A100 node for the 8B.
- **Arms.**
  - *Control:* best fixed budget $t^\star$, chosen per dataset by sweeping $t=1..8$ on the same rollouts (free — already computed).
  - *Oracle:* stop at $\arg\max_t (U_t - c\,\ell_{\le t})$ per question.
  - *Candidates:* self-confidence threshold, FLARE-style token-probability threshold, Self-RAG-style reflection token, a learned probe on hidden states.
- **Deciding number.** The **oracle gap**: oracle accuracy minus best-fixed-budget accuracy at equal mean token cost. If the gap is under ~3 points EM, adaptive stopping is not where the headroom is, and the field should stop publishing adaptive-RAG variants. If it is over ~10 points, the gap between each candidate rule and the oracle becomes the benchmark this problem has been missing.

Secondary output, nearly free: the fraction of questions where $U_t$ is non-monotone in $t$ — a direct empirical test of assumption 2 and therefore of whether myopic rules can ever be optimal here.

## 9. Key References

- **[Foundational]** Abraham Wald. *Sequential Tests of Statistical Hypotheses.* Annals of Mathematical Statistics, 1945.
- **[Foundational]** John C. Gittins. *Bandit Processes and Dynamic Allocation Indices.* Journal of the Royal Statistical Society B, 1979.
- **[Foundational]** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023. — arXiv:2210.03629
- **[SOTA]** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal. *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.* ACL 2023. — arXiv:2212.10509
- **[SOTA]** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. — arXiv:2310.11511
- **[SOTA]** Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig. *Active Retrieval Augmented Generation (FLARE).* EMNLP 2023. — arXiv:2305.06983
- **[SOTA]** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.* NAACL 2024. — arXiv:2403.14403
- **[SOTA]** Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, Jiawei Han. *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.* 2025. — arXiv:2503.09516
- **[Evidence]** Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, Hannaneh Hajishirzi. *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL 2023. — arXiv:2212.10511
- **[Evidence]** Saurav Kadavath et al. *Language Models (Mostly) Know What They Know.* 2022. — arXiv:2207.05221
- **[Evidence]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. — arXiv:2307.03172
- **[Benchmark]** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal. *MuSiQue: Multihop Questions via Single-hop Question Composition.* TACL 2022. — arXiv:2108.00573
- **[Benchmark]** Jason Wei et al. *BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents.* OpenAI, 2025.
- **[Survey]** Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang. *Retrieval-Augmented Generation for Large Language Models: A Survey.* 2023–2024. — arXiv:2312.10997

## 10. Worked Example

One MuSiQue-style 2-hop question: *"What is the population of the city where the composer of the score for the 1962 film X was born?"*

Run a 7B ReAct agent to 6 hops, forcing an answer at each. Observed transcript (illustrative but typical of the pattern the fork protocol reveals):

| $t$ | cumulative tokens $\ell_{\le t}$ | $U_t$ | stated confidence |
|---|---|---|---|
| 1 | 1,100 | 0 | 0.61 |
| 2 | 2,400 | 1 | 0.55 |
| 3 | 4,100 | 1 | 0.72 |
| 4 | 6,500 | 0 | 0.81 |
| 5 | 9,400 | 0 | 0.78 |
| 6 | 12,900 | 0 | 0.84 |

Set $c$ so that 1,000 tokens costs 0.01 utility. Then $U_t - c\ell_{\le t}$ is $-0.011,\ 0.976,\ 0.959,\ -0.065,\ -0.094,\ -0.129$. The oracle stops at $t=2$ for a score of 0.976.

Now score the heuristics on the same transcript:

- **Confidence threshold at 0.75:** first crossing is $t=4$. Score $-0.065$. It stops precisely when the agent has retrieved a distractor about a same-named city and become *more* sure of a wrong answer.
- **Fixed budget $t=3$:** score $0.959$. Within 0.017 of the oracle.
- **Run to the end ($t=6$):** score $-0.129$.

The obstruction is visible in one column: confidence rises 0.55 → 0.84 while accuracy falls 1 → 0. The stop signal and the truth move in opposite directions after the distractor enters, and no quantity available *inside the rollout at hop 4* distinguishes this case from a genuine late-arriving find. Only the fork — which produces the $U_t$ column and costs 6 extra generations per question — makes the error legible. That column is what the field has not built at scale, and Section 8 is a proposal to build it.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*