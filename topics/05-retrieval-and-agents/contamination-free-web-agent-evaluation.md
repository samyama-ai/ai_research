---
id: 05-retrieval-and-agents/contamination-free-web-agent-evaluation
title: "Contamination-Free Evaluation of Web-Browsing Agents"
topic: 05-retrieval-and-agents
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Contamination-Free Evaluation of Web-Browsing Agents

> **Topic:** Retrieval & Agentic Systems · **ID:** `05-retrieval-and-agents/contamination-free-web-agent-evaluation` · **Status:** methodologically-blocked

## 1. Problem Statement

A web-browsing agent is a policy that, given a natural-language task, issues search and page-fetch actions against the live web and returns an answer. We want to measure its **browsing skill**: the ability to find and combine information it did not already have.

The measurement is confounded by three distinct leakage channels, which the literature routinely collapses into the single word "contamination":

1. **Parametric leakage** — the answer was in pretraining data, so the agent can emit it without browsing.
2. **Benchmark leakage** — the benchmark's questions *and* gold answers were published on the web, indexed, and are now retrievable by the agent under test (or absorbed into a later pretraining run).
3. **Substrate drift** — the live web changed between the gold-label date and the evaluation date, so a correct-at-authoring label is now wrong, or the target page is gone.

Three variants, of very different difficulty:

- **Measurement variant** (the blocked one): define an estimator of browsing skill that is invariant to (1)–(3), and show it is identifiable from observable rollouts.
- **Method variant**: build a benchmark or harness that empirically suppresses leakage — freshness windows, private held-out sets, frozen web snapshots, synthetic sandboxes.
- **Theory variant**: prove a detection or lower-bound result — e.g. that no black-box test can distinguish retrieval from recall without an intervention on the agent's inputs.

A solution to the measurement variant would produce a score $\hat{S}$ with a stated confidence interval, reproducible by a third party at a later date on the same agent, whose value does not move when the benchmark is made public.

## 2. Formal Setting

Let $\mathcal{T}$ be a task distribution, $t \sim \mathcal{T}$ a task with gold answer $y^*(t, \tau)$ **indexed by the time $\tau$ at which it is valid**. Let $\pi_\theta$ be the agent with parameters $\theta$ frozen at pretraining cutoff $c(\theta)$. Let $W_\tau$ be the state of the web (index + page contents) at time $\tau$, and $\mathcal{O}$ the harness's observation function (search API, fetch, DOM serialization).

A rollout is $\rho = (a_1, o_1, \dots, a_k, o_k, \hat{y})$, $o_i = \mathcal{O}(a_i; W_\tau)$. Score $R(t) = \mathbb{1}[\hat y = y^*(t,\tau)]$ under exact/LLM-judge match; report $S(\pi, \mathcal{T}, \tau) = \mathbb{E}_{t}[R(t)]$.

**The quantity we actually want** is the browsing-attributable component. Define the *closed-book* counterfactual: same agent, same task, no tool calls executed ($\mathcal{O} \equiv \emptyset$, best-of-$n$ guessing allowed), giving $S_{\text{cb}}$. Then

$$\Delta_{\text{browse}} \;=\; S(\pi,\mathcal{T},\tau) \;-\; S_{\text{cb}}(\pi,\mathcal{T}).$$

$\Delta_{\text{browse}}$ is measurable but *not* the target: an agent can browse, retrieve the leaked benchmark page containing $(t, y^*)$, and score. Define the **oracle-page indicator** $Z(t,\rho) = 1$ if any $o_i$ contains a document derivable from the benchmark release itself (question text, answer key, a public leaderboard writeup, a scraped mirror). The leakage-free score is

$$S^{\text{clean}} = \mathbb{E}_t\big[R(t)\,\big|\,Z=0,\ \text{cutoff}(\theta) < \text{first-publication}(t)\big].$$

**How each quantity is measured in practice.**
- $R$: string/numeric match, or a judge model; judge agreement with humans is itself an error term, typically 90–97% on short-answer web tasks.
- $S_{\text{cb}}$: rerun with tools disabled — cheap, and the single most informative control almost never reported per-item.
- $Z$: substring / near-duplicate match of the question stem against fetched page text, plus URL blocklist of the benchmark repo and its mirrors. This is the weak link — paraphrased mirrors and model-written blog posts about the benchmark defeat exact matching.
- $c(\theta)$: for closed API models, *asserted by the vendor*, not verifiable.

**Assumptions, and which are violated.**
- (A1) $y^*(t,\tau)$ stable over the evaluation window — **violated** for any task touching prices, rankings, counts, or personnel.
- (A2) $W_\tau$ identical across arms — **violated**: search APIs personalize, rate-limit, and re-rank; two runs a week apart are two different environments.
- (A3) $c(\theta)$ known — **violated** for closed models and for any model with post-cutoff RLHF or tool-use data.
- (A4) $Z$ decidable — **violated**; this is the methodological block (§6).
- (A5) i.i.d. tasks — violated by benchmark construction, which oversamples question types the authors found hard.

## 3. State of the Art

**Empirical/systems SOTA.**
- **BrowseComp** (Wei et al., OpenAI, 2025): 1,266 hard-to-find-answer questions built by inverting the search problem — authors start from an obscure fact and write a question whose answer is verifiable but not top-of-search. Deliberately targets parametric leakage. *Established*: the construction procedure and the reported gap between browsing and non-browsing configurations. *Claimed but unablated*: that the set stays uncontaminated after public release — no post-release re-measurement protocol is specified.
- **GAIA** (Mialon, Fourrier, Swift, Wolf, LeCun, Scialom; ICLR 2024): 466 real-world assistant questions, answers held privately for the test split. Private held-out answers are the strongest deployed defense; *established* only against benchmark leakage, not parametric.
- **WebArena** (Zhou et al., ICLR 2024) and **VisualWebArena** (Koh et al., ACL 2024): 812 tasks on self-hosted clones of e-commerce/forum/CMS sites. Fully reproducible substrate — solves (A2) and $Z$ by construction, at the cost of not measuring open-web retrieval at all.
- **WebVoyager** (He et al., ACL 2024): 643 tasks on 15 live sites; live-web realism, no contamination control.
- **AssistantBench** (Yoran et al., EMNLP 2024): 214 time-consuming open-web tasks, explicitly notes answer volatility and re-verification burden.
- **Live/rolling benchmarks**: **LiveCodeBench** (Jain et al., ICLR 2025) and **LiveBench** (White et al., ICLR 2025) establish the freshness-window design — score only on items released after a model's cutoff. Directly transferable to browsing agents; not yet the norm for them.

**Theory/statistics SOTA.** Oren, Meister, Chatterji, Ladhak, Hashimoto (ICLR 2024), *Proving Test Set Contamination in Black Box Language Models*: an exchangeability test — if a model has seen a dataset in a canonical order, its log-likelihood is higher on that order than on shuffled permutations, giving a $p$-value with no access to weights or training data. Golchin & Surdeanu (ICLR 2024), *Time Travel in LLMs*, gives a guided-completion detector. Both need token log-probs and assume the item was memorized as text — neither applies when leakage arrives through the agent's own retrieval at test time.

## 4. What Is Known

- **Browsing changes scores by an order of magnitude on retrieval-hard sets.** On BrowseComp, GPT-4o without browsing was reported near 0.6–1.9%, and OpenAI's Deep Research configuration at 51.5% (OpenAI, 2025). Scale: 1,266 items, single vendor.
- **Agents are far below humans on open-web assistant tasks.** GAIA: humans 92%, GPT-4 with plugins ~15% at release (466 questions). WebArena: end-to-end success 14.41% for the initial GPT-4 agent against 78.24% human (812 tasks).
- **Contamination shifts leaderboard rank, not just level.** LiveCodeBench (ICLR 2025) shows several models scoring well on pre-cutoff HumanEval-style problems and dropping sharply on post-cutoff problems, while others are flat — the ordering is not preserved. Scale: hundreds of problems, dozens of models.
- **Black-box contamination tests work when the item is memorized verbatim.** Oren et al. report reliable detection on datasets duplicated as few as ~10 times in a training corpus of a 1.4B-parameter model. This is a *pretraining* result; it says nothing about retrieval-time leakage.
- **Sandboxed substrates are reproducible.** WebArena reruns are stable because the sites are frozen containers; live-web reruns of the same agent are not, with drift reported anecdotally across every live benchmark.

## 5. What Is Not Known

- **Methodologically blocked (the core gap).** There is no accepted operational definition of $Z$ — "the agent found the answer because the benchmark leaked." Exact-match on question stems is trivially defeated by paraphrase, translation, or an LLM-written summary of the leaderboard. Without a decidable $Z$, $S^{\text{clean}}$ is not estimable, and every reported browsing score is a mixture of skill and leakage with unknown mixing weight.
- **Methodologically blocked.** No standard for label revalidation: nobody reports what fraction of a benchmark's gold answers are still correct $n$ months after authoring.
- **Empirically open.** The closed-book control arm $S_{\text{cb}}$ is cheap and almost never reported per item on browsing benchmarks. Nor has anyone run the obvious A/B: same agent, same tasks, with and without benchmark-domain blocking in the search index.
- **Empirically open.** Whether frozen web snapshots (Common Crawl-backed search) preserve agent ranking relative to live-web evaluation, at $n \geq 500$ tasks.
- **Theoretically open.** Whether *any* black-box test can separate retrieval from recall without intervening on the agent's observation stream. Plausibly a non-identifiability result exists; none is proved.

## 6. Why It Is Hard

The obstruction is **non-identifiability under an adversarially updating substrate**. Publishing a benchmark makes it retrievable; the act of measuring changes $W_\tau$ so that the next measurement is of a different quantity. Unlike static-benchmark contamination, you cannot fix it by choosing a later cutoff, because the leakage arrives at *inference* time through the agent's own tool calls.

Compounding it:
- **Absent ground truth for $Z$.** Deciding whether a fetched page is "derived from the benchmark" is a semantic-provenance question with no labels and no reference method.
- **Confounded control arms.** Blocking benchmark domains also removes legitimately useful pages (an agent blocked from GitHub loses real capability), so the ablation does not isolate leakage.
- **Cost asymmetry.** A browsing rollout is 10–100× a closed-book generation in tokens and wall-clock; running the full factorial (live/frozen × blocked/unblocked × tools/no-tools) at $n=500$ is a five-figure evaluation.
- **Vendor-asserted cutoffs.** Freshness windows, the one method that works elsewhere, rest on $c(\theta)$ being truthful and single-valued. Neither holds.

## 7. Current Research (as of 2026)

- **Rolling/freshness benchmarks for agents** — extending the LiveBench/LiveCodeBench design (Abacus.AI, NYU, Nvidia lineage) to browsing tasks with monthly item releases. *(frontier — verify)*
- **Private held-out splits with server-side grading** — GAIA's model, now standard for agent leaderboards (Hugging Face).
- **Reproducible sandboxes at web scale** — BrowserGym / AgentLab (ServiceNow Research, Drouin, Lacoste et al., 2024) unifying WebArena, WorkArena, MiniWoB under one action space; the direction of travel is frozen substrates plus synthetic task generators.
- **Canary strings and provenance markers** in benchmark releases (BIG-bench lineage) — cheap, and detects only naive scraping.
- **Contamination statistics** — exchangeability and membership-inference tests (Stanford, Arizona) applied to agent trajectories rather than answers. *(frontier — verify)*

## 8. Concrete Next Experiment

**Question:** how much of a browsing agent's score is leakage rather than skill?

**Scale.** $n = 400$ tasks: 200 from a public benchmark released $\geq 12$ months ago (BrowseComp or GAIA validation), 200 newly authored under the identical protocol and never published. Three agents (one frontier closed model, one open-weights, one retrieval-heavy pipeline), 3 seeds. Roughly 3,600 rollouts.

**Arms (2×2 plus control).**
1. Live web, unrestricted (standard reported condition).
2. Live web, **search index blocking the benchmark's domain and its top-50 mirrors** — the leakage-suppression arm.
3. Frozen snapshot index dated to the benchmark's authoring month.
4. **Control arm: tools disabled**, $S_{\text{cb}}$, best-of-1 — the parametric-recall floor.

**Deciding number.** The leakage share

$$L \;=\; \frac{S_{\text{public}}^{\text{arm1}} - S_{\text{public}}^{\text{arm2}}}{S_{\text{public}}^{\text{arm1}} - S_{\text{public}}^{\text{arm4}}} \;-\; \frac{S_{\text{private}}^{\text{arm1}} - S_{\text{private}}^{\text{arm2}}}{S_{\text{private}}^{\text{arm1}} - S_{\text{private}}^{\text{arm4}}}.$$

The private split subtracts the capability loss caused by blocking a useful domain. $L \leq 0.05$ with a 95% CI excluding 0.15 means published browsing scores are usable as-is. $L \geq 0.20$ means every live-web browsing leaderboard number in the literature is an upper bound of unknown tightness, and freshness-windowed private splits become mandatory. At $n=200$ per split, an $L$ of 0.20 is detectable at 80% power for base rates in the 30–60% range.

Secondary output: per-item $Z$ labels from human adjudication of 300 sampled trajectories, which would be the first reference set for the leakage detector that §5 says does not exist.

## 9. Key References

- **[Foundational]** Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, Thomas Scialom. *GAIA: a benchmark for General AI Assistants.* ICLR 2024. — arXiv:2311.12983
- **[Foundational]** Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, Graham Neubig. *WebArena: A Realistic Web Environment for Building Autonomous Agents.* ICLR 2024. — arXiv:2307.13854
- **[SOTA]** Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, Amelia Glaese. *BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents.* OpenAI, 2025. — arXiv:2504.12516
- **[SOTA]** Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, Tatsunori B. Hashimoto. *Proving Test Set Contamination in Black Box Language Models.* ICLR 2024. — arXiv:2310.17623
- **[SOTA]** Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* ICLR 2025. — arXiv:2403.07974
- **[SOTA]** Colin White et al. *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* ICLR 2025. — arXiv:2406.19314
- **[Method]** Shahriar Golchin, Mihai Surdeanu. *Time Travel in LLMs: Tracing Data Contamination in Large Language Models.* ICLR 2024. — arXiv:2308.08493
- **[Method]** Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, Dong Yu. *WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models.* ACL 2024. — arXiv:2401.13919
- **[Method]** Ori Yoran, Samuel Joseph Amouyal, Chaitanya Malaviya, Ben Bogin, Ofir Press, Jonathan Berant. *AssistantBench: Can Web Agents Solve Realistic and Time-Consuming Tasks?* EMNLP 2024. — arXiv:2407.15711
- **[Survey]** Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, Eneko Agirre. *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark.* Findings of EMNLP 2023. — arXiv:2310.18018
- **[Systems]** Thibault Le Sellier de Chezelles, Maxime Gasse, Alexandre Drouin, Massimo Caccia, et al. *The BrowserGym Ecosystem for Web Agent Research.* 2024. — arXiv:2412.05467

## 10. Worked Example

Take a BrowseComp-style item: *"Which university awarded the doctorate to the second author of the 1987 paper that introduced [obscure algorithm]?"* Gold answer authored 2025-03.

Run an agent in 2026-09 across the four arms. Suppose:

| Arm | Score (200 public items) | Score (200 private items) |
|---|---|---|
| 1. Live, unrestricted | 0.52 | 0.44 |
| 2. Live, benchmark domains blocked | 0.41 | 0.42 |
| 3. Frozen 2025-03 snapshot | 0.46 | 0.43 |
| 4. Tools disabled ($S_{\text{cb}}$) | 0.09 | 0.05 |

Public leakage fraction: $(0.52-0.41)/(0.52-0.09) = 0.256$. Private: $(0.44-0.42)/(0.44-0.05)=0.051$. So $L = 0.205$ — one fifth of the browsing-attributable gain on the public split comes from retrieving the benchmark rather than solving it. The published 0.52 corresponds to a leakage-free 0.41.

Now the obstruction. The blocklist covered the benchmark repo and 50 mirrors. Inspecting 300 trajectories by hand, 34 of the 200 public items had a *correct* answer retrieved from a page that was not on the blocklist and did not contain the question stem verbatim: a Chinese-language blog summarizing "hard BrowseComp examples", a Hugging Face Space's cached prediction dump, and an arXiv paper's qualitative-examples appendix. Exact-match $Z$ scored all three as clean. Under human adjudication $L$ rises to roughly 0.29.

The two estimates differ by 40%, and there is no principled way to say which is right — the detector $Z$ has no reference standard, so the correction to the score inherits the detector's unquantified error. That is the block: not that leakage is large, but that its magnitude is not measurable with any method the field currently agrees on.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*