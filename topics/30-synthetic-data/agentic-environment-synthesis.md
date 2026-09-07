---
id: 30-synthetic-data/agentic-environment-synthesis
title: "Agentic Environment Synthesis for Tool-Use Training"
topic: 30-synthetic-data
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Agentic Environment Synthesis for Tool-Use Training

> **Topic:** Synthetic Data · **ID:** `30-synthetic-data/agentic-environment-synthesis` · **Status:** open

## 1. Problem Statement

Tool-use training needs *environments*, not just transcripts: a set of callable APIs, a persistent backend state, a task distribution over that state, and a programmatic success check. Human-built environments are the bottleneck — AppWorld took a documented multi-month engineering effort for 9 apps and 457 APIs; τ-bench covers 2 domains. The proposal is to synthesize environments with a generator model.

- **Input:** a generator $G$ (an LLM plus scaffolding), a compute budget, optionally a seed corpus of real API specs or app schemas.
- **Output:** a set of environments $\mathcal{E} = \{e_1,\dots,e_N\}$, each $e = (\mathcal{T}, S_0, P, R, \mathcal{D})$ — tool schemas, initial state, transition function (executable code, not a model), verifier, task distribution.
- **Objective:** train policy $\pi$ on $\mathcal{E}$ and maximize success on a *held-out, human-authored* benchmark $\mathcal{B}$ that shares no tools, no domain, and no generator with $\mathcal{E}$.

Three variants that are routinely conflated:

- **Measurement:** does transfer from synthetic to human-authored environments exist, and how is it separated from contamination and from generic instruction-following gain? Currently the weakest link.
- **Method:** which generator produces the most transferable $\mathcal{E}$ per unit compute — random schema sampling, seeded mutation of real APIs, or adversarial/curricular generation against the current $\pi$?
- **Theory:** under what conditions does a distribution of *generated* MDPs induce a policy that generalizes to a *natural* MDP? This is unsupervised environment design (UED) with an unobservable target distribution, and it is open.

Solved would mean: a published recipe where synthetic environments alone (no human-authored training environment) reach within a stated margin of training on the target-domain human environment, with a held-out generalization gap that shrinks as $N$ grows.

## 2. Formal Setting

Each environment is a POMDP $e = (\mathcal{S}, \mathcal{A}_{\mathcal{T}}, P, R, \Omega, O, \gamma)$. Actions are tool calls $a = (\tau, \theta)$ with $\tau \in \mathcal{T}$ and arguments $\theta$ conforming to $\tau$'s JSON schema. $P$ is deterministic executable code (a sandboxed database plus handlers). The verifier $R(\text{final state}, \text{trajectory}) \in \{0,1\}$ is programmatic state-checking, not an LLM judge.

The generator induces a distribution $p_G$ over environments. Training minimizes
$$\hat{J}(\pi) = \mathbb{E}_{e \sim p_G}\,\mathbb{E}_{d \sim \mathcal{D}_e}\big[R_e(\pi, d)\big],$$
but the quantity we care about is $J^*(\pi) = \mathbb{E}_{e \sim p_{\text{real}}}[\cdot]$ with $p_{\text{real}}$ unobserved and estimated by a finite benchmark $\mathcal{B}$.

Measured quantities:

- **Transfer gap** $\Delta = \hat{J}(\pi) - J^*(\pi)$, estimated as (pass rate on held-out synthetic environments) minus (pass rate on $\mathcal{B}$), each over $\geq 200$ tasks so the binomial standard error is $\leq 3.5$ points.
- **Environment diversity.** Report three, not one: $|\mathcal{T}|$ unique tool signatures after normalizing names; mean pairwise Jaccard distance over tool-name token sets; and *dependency depth* — the length of the longest chain of calls where call $k+1$ needs a value only obtainable from call $k$'s return. Depth is the one that correlates with difficulty and the one generators collapse.
- **Executability** $\rho$: fraction of generated environments whose reference solution runs end to end and whose verifier fires. Measured by executing, not by asking a model.
- **Solvability band:** fraction of tasks with $0 < \Pr[\text{success}] < 1$ under a reference policy at $k{=}8$ samples. Tasks outside the band carry no gradient signal.
- **Contamination:** for each $\mathcal{B}$ task, max n-gram overlap ($n{=}13$) and embedding similarity against $\mathcal{E}$; report $J^*$ on the decontaminated subset separately.

Assumptions, with the ones known to fail marked:

1. $p_G$ has support covering the relevant region of $p_{\text{real}}$. **Known violated** — generated tool schemas cluster on CRUD-shaped APIs and short dependency chains.
2. $R_e$ is a faithful reward. **Known violated** — generated verifiers accept degenerate solutions (checking that a field is non-empty rather than correct).
3. Generator and evaluated policy are independent. **Usually violated** — the same model family generates and is evaluated, so stylistic conventions leak.
4. Tasks are solvable. Violated at a measurable rate; APIGen reports large filter-out fractions precisely because of this.

## 3. State of the Art

**Established (executed and verified, with ablations):**

- **APIGen** (Liu et al., NeurIPS 2024 Datasets & Benchmarks) — a three-stage verification pipeline (format check, actual API execution, semantic check) over 3,673 real APIs; released 60k verified entries. The xLAM-7B model trained on it scores competitively on the Berkeley Function Calling Leaderboard against far larger models. The multi-stage *filter* is ablated; environment *synthesis* is not the contribution — the APIs are real.
- **ToolLLM / ToolBench** (Qin et al., ICLR 2024) — 16,464 real RapidAPI endpoints, instructions and solution paths generated by GPT-4 with a DFS-based decision tree search. Established: DFSDT beats ReAct-style CoT on pass rate. Not established: transfer to non-RapidAPI domains; the evaluator is GPT-4-based, so the reward is a model, not state.
- **AppWorld** (Trivedi et al., ACL 2024, best resource paper) — 9 apps, 457 APIs, 750 tasks, programmatic state-based verification. Establishes the difficulty floor: GPT-4o passes about 49% of normal tasks and roughly 30% of the harder challenge split. Human-authored, so it is the natural held-out target.
- **τ-bench** (Yao et al., 2024) — dual-control with a simulated user; the headline established result is *reliability*: pass^8 (all 8 independent trials succeed) collapses far below pass^1 for frontier models, below ~25% in the retail domain. This is the number synthetic-environment work usually fails to report.

**Claimed but unablated:**

- **ToolACE** (Liu et al., 2024) — self-evolving API synthesis producing 26,507 diverse APIs; an 8B model reported at state-of-the-art on BFCL at the time. The diversity claim rests on generated-API counts; there is no controlled arm holding data volume fixed while varying diversity, so "diversity caused the gain" is unestablished.
- **AgentGym** (Xi et al., 2024) — 14 environments unified behind one interface with an evolution loop (AgentEvol). Cross-environment generalization is reported as benchmark numbers; the environments are ported, not synthesized.
- **Absolute Zero** (Zhao et al., 2025) — self-proposed tasks with a code executor as ground truth, zero external data, reported gains on math and code. Cited in agentic contexts as evidence that self-generated task distributions work, but the executor gives an exact verifier that generated tool environments do not have.

**Theory SOTA:** unsupervised environment design — PAIRED (Dennis et al., NeurIPS 2020) proves a minimax-regret equilibrium guarantee for a generator/antagonist/protagonist game; POET (Wang et al., 2019) gives open-ended coevolution without such a guarantee. Neither has been instantiated over tool-call environments, where the design space is discrete schemas rather than terrain parameters.

## 4. What Is Known

- Execution-based filtering beats model-judged filtering. APIGen's staged verification retains a minority of generated candidates; the surviving 60k entries train a 7B model to BFCL-competitive scores — evidence that yield, not volume, is the operative variable.
- Verifier quality dominates. On τ-bench, pass^1 to pass^8 degradation of 30+ points at frontier scale shows single-sample benchmark numbers overstate capability; any synthetic-environment claim reported at pass^1 is unfalsified.
- Difficulty is where generators fail. AppWorld's 457 human-authored APIs support tasks needing dependency depth well beyond 2; GPT-4o at ~49% shows the ceiling is real. Generated environments overwhelmingly produce depth-1 and depth-2 tasks.
- Scale of the largest efforts: ToolBench 16,464 real APIs; ToolACE 26,507 generated APIs; APIGen 3,673 real APIs / 60k data points. No published work synthesizes $N > 10^3$ *executable stateful backends* — the synthesized artifact is almost always the schema plus a stub, not a state machine.
- Synthetic instruction data alone (Self-Instruct, Wang et al., ACL 2023) lifts instruction-following substantially; the fact that it does not automatically lift multi-step tool use is itself the finding.

## 5. What Is Not Known

- **Methodologically blocked:** there is no accepted measure of environment *diversity* that predicts transfer. Papers report unique API counts, which are trivially inflatable by renaming. Until diversity is operationalized (dependency depth, state-space branching, verifier strictness), "more diverse synthetic environments" is not a testable claim.
- **Empirically open:** the scaling law. Nobody has trained a fixed policy on $N \in \{10, 10^2, 10^3, 10^4\}$ synthetic environments at fixed token budget and plotted held-out AppWorld/τ-bench pass^4. The experiment is runnable today for well under $100k. Also open: whether generated environments transfer at all once decontaminated and once the generator differs from the trainee family.
- **Theoretically open:** no bound relating coverage of $p_G$ to $J^*$ for tool-call POMDPs. PAIRED's regret guarantee assumes the generator can realize the target environment; whether an LLM generator's support contains realistic enterprise APIs is unproven and probably false.

## 6. Why It Is Hard

The specific obstruction is **absent ground truth in the verifier, compounded by generator/verifier correlation**. When $G$ writes the environment, the task, *and* the success check, the three share failure modes: a verifier that checks `order.status == "cancelled"` without checking the refund was issued rewards a policy that half-solves. RL then optimizes the exploit, and the resulting number is high on synthetic and flat on $\mathcal{B}$. This is not fixable by better prompting because there is no external signal — unlike code (unit tests execute) or math (answers check). Second obstruction: **confounded measurement**. Gains from synthetic environments mix (a) genuine tool-use skill, (b) format conformance to JSON schemas, (c) contamination through the generator's memory of public benchmarks. Papers reporting a single BFCL delta cannot separate the three.

## 7. Current Research (as of 2026)

- Execution-grounded pipelines with real backends — Salesforce AI Research (APIGen/xLAM line), extended toward multi-turn data *(frontier — verify)*.
- Self-evolving API synthesis and agentic pipelines — Huawei Noah's Ark (ToolACE), Microsoft (AgentInstruct, Mitra et al. 2024, agentic flows over web-seed documents).
- Environment unification and cross-environment evolution — Fudan NLP (AgentGym), OSU/Berkeley on web and OS environments (WebArena, Zhou et al. ICLR 2024; OSWorld, Xie et al. NeurIPS 2024).
- Open-ended/UED transplanted to LLM agents — descendants of POET and OMNI-EPIC (Faldor et al., 2024), where an LLM proposes environments and a code interpreter runs them *(frontier — verify)*.
- Reliability-first evaluation — Sierra's τ-bench and τ²-bench line, pushing pass^k as the reported statistic.

## 8. Concrete Next Experiment

**Question:** does synthetic environment count buy held-out tool-use skill, once verifier quality is controlled?

- **Scale:** one 8B open base model (e.g. Llama-3.1-8B), 4 arms, ~3B training tokens each, roughly 1.5k A100-hours total. Synthesize $N \in \{32, 256, 2048\}$ executable environments — each a SQLite-backed app with 8–20 tools and a state-diff verifier — from a generator of a *different* family than the trainee. Sample tasks to a fixed 40k-trajectory budget in every arm, so $N$ is the only variable.
- **Control arms:** (i) 40k trajectories from a *single* human-authored environment domain; (ii) 40k trajectories from $N{=}32$ synthetic; (iii) same trajectories with tool names and argument keys randomized, to separate format learning from skill.
- **Held out:** AppWorld test-challenge and τ-bench retail, both 13-gram decontaminated against all generated text.
- **Deciding number:** **AppWorld test-challenge pass^4**, measured at $N{=}32$ vs $N{=}2048$. A gain of $\geq 8$ points (above the ~5-point 95% CI at 168 challenge tasks) with a monotone trend across the three $N$ says environment count is a real axis. A gain under 3 points says the generator's support, not its sample count, is the binding constraint — and the field should stop scaling $N$ and start scaling dependency depth.
- **Secondary:** report $\rho$ (executability) and the fraction of tasks whose verifier accepts a deliberately half-completed reference trajectory. If that fraction exceeds 10%, the primary number is uninterpretable.

## 9. Key References

- **[Foundational]** Yuchen Lin et al. — see instead: Yao, Shunyu; Zhao, Jeffrey; Yu, Dian; Du, Nan; Shafran, Izhak; Narasimhan, Karthik; Cao, Yuan. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR, 2023. — arXiv:2210.03629
- **[Foundational]** Dennis, Michael; Jaques, Natasha; Vinitsky, Eugene; Bayen, Alexandre; Russell, Stuart; Critch, Andrew; Levine, Sergey. *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design.* NeurIPS, 2020. — arXiv:2012.02096
- **[Foundational]** Wang, Rui; Lehman, Joel; Clune, Jeff; Stanley, Kenneth O. *Paired Open-Ended Trailblazer (POET).* 2019. — arXiv:1901.01753
- **[SOTA]** Liu, Zuxin; Hoang, Thai; Zhang, Jianguo; et al. *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.18518
- **[SOTA]** Qin, Yujia; Liang, Shihao; Ye, Yining; et al. *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.* ICLR, 2024. — arXiv:2307.16789
- **[SOTA]** Liu, Weiwen; Huang, Xu; Zeng, Xingshan; et al. *ToolACE: Winning the Points of LLM Function Calling.* 2024. — arXiv:2409.00920
- **[Benchmark]** Trivedi, Harsh; Khot, Tushar; Hartmann, Mareike; et al. *AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents.* ACL, 2024.
- **[Benchmark]** Yao, Shunyu; Shinn, Noah; Razavi, Pedram; Narasimhan, Karthik. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[Benchmark]** Zhou, Shuyan; Xu, Frank F.; Zhu, Hao; et al. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR, 2024. — arXiv:2307.13854
- **[Benchmark]** Xie, Tianbao; Zhang, Danyang; Chen, Jixuan; et al. *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments.* NeurIPS, 2024. — arXiv:2404.07972
- **[Method]** Mitra, Arindam; Del Corro, Luciano; Zheng, Guoqing; et al. *AgentInstruct: Toward Generative Teaching with Agentic Flows.* 2024. — arXiv:2407.03502
- **[Method]** Wang, Yizhong; Kordi, Yeganeh; Mishra, Swaroop; et al. *Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL, 2023. — arXiv:2212.10560
- **[Survey]** Xi, Zhiheng; Chen, Wenxiang; Guo, Xin; et al. *The Rise and Potential of Large Language Model Based Agents: A Survey.* 2023. — arXiv:2309.07864

## 10. Worked Example

Generate one environment from a 3-line seed: "a hotel booking app with rooms, reservations, and loyalty points."

The generator emits 11 tools (`search_rooms`, `create_reservation`, `cancel_reservation`, `get_loyalty_balance`, `redeem_points`, …), a SQLite schema, and 40 tasks. Execute and measure:

| Quantity | Value | Note |
|---|---|---|
| Tools emitted | 11 | 9 are single-table CRUD |
| Executability $\rho$ | 33/40 = 0.83 | 7 tasks: reference solution raises |
| Dependency depth 1 | 26/33 | one call, then done |
| Depth 2 | 6/33 | search → book |
| Depth $\geq 3$ | 1/33 | book → redeem → verify balance |
| In solvability band at $k{=}8$ | 9/33 = 0.27 | rest are pass-8 or fail-8 |

Now the obstruction. Take task 17: *"Cancel Ana's June reservation and refund her points."* The generated verifier is

```python
assert db.query("select status from reservations where id=?", rid) == "cancelled"
```

Run a deliberately half-completed policy that calls `cancel_reservation` and stops. It passes. The refund is never checked; loyalty balance stays at 1,200 when it should be 1,650. Across the 33 executable tasks, 6 verifiers (18%) accept a half-completed trajectory — above the 10% threshold from §8.

So this environment yields 33 executable tasks, of which 9 carry gradient signal, of which 6 have exploitable verifiers, leaving perhaps 5 usable tasks — a 12.5% yield from 40. Reaching the 40k-trajectory budget of §8 needs roughly 8,000 generated tasks *after* execution-filtering, and the exploitable-verifier fraction does not shrink with $N$: it is a property of the generator, not the sample. Scaling $N$ multiplies both the signal and the exploit at the same rate. That is why the deciding number in §8 is pass^4 on a *human-authored* held-out set and why the verifier-exploit fraction is reported alongside it — the synthetic pass rate would rise smoothly in every arm and tell you nothing.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*