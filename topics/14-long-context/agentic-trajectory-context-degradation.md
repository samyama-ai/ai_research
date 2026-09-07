---
id: 14-long-context/agentic-trajectory-context-degradation
title: "Long-Context Performance Under Multi-Turn Agentic Trajectories"
topic: 14-long-context
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Long-Context Performance Under Multi-Turn Agentic Trajectories

> **Topic:** Long Context · **ID:** `14-long-context/agentic-trajectory-context-degradation` · **Status:** empirically-open

## 1. Problem Statement

Long-context evaluation measures a model reading a context someone else wrote. An agent reads a context it wrote itself: its own tool calls, tool outputs, retries, failed hypotheses, and user corrections, appended turn after turn. The question is whether the second case degrades faster than the first, and if so, why.

Three variants, different difficulty:

- **Measurement.** Define a degradation curve for agentic context that is not confounded by task difficulty growing with turn index. Currently there is no accepted estimator. This is the binding variant.
- **Method.** Given a budget $B$ tokens, choose what to keep in the context at turn $t$ (raw history, summary, external memory, re-retrieval) to maximize task success. Compaction, memory files, and sub-agent delegation are deployed but not ablated against a matched no-compaction control.
- **Theory.** Is there a model of self-generated context in which error accumulates super-linearly in turn count — i.e. does conditioning on one's own erroneous output provably compound? No such model exists for transformers.

Solving it means: an estimator that separates *context length* effects from *turn count* effects from *self-authorship* effects, plus one intervention that beats the no-intervention control at fixed compute.

## 2. Formal Setting

A trajectory is $\tau = (o_0, a_1, o_1, \dots, a_T, o_T)$ where $a_t$ is a model action (text plus tool call) and $o_t$ is an environment observation. The context at turn $t$ is $c_t = \phi(o_0, a_1, o_1, \dots, a_{t-1}, o_{t-1})$, with $\phi$ the **context policy** (identity = full history; otherwise compaction).

Measured quantities:

- **Context length** $L_t = |c_t|$ in tokens, from the tokenizer, logged per request. Includes system prompt and tool schemas.
- **Turn depth** $t$: number of assistant messages, counted from the request log.
- **Self-authored fraction**
$$\sigma_t = \frac{\sum_{i<t}|a_i|}{L_t},$$
tokens the model itself emitted over total context. Tool outputs count as environment, not self. In practice $\sigma_t \in [0.2, 0.6]$ for coding agents; measure it, do not assume it.
- **Per-turn correctness** $y_t \in \{0,1\}$: requires a step-level oracle. Available only in environments with programmatic state checks (database final-state equality in $\tau$-bench, test suites in SWE-bench, DOM/state assertions in WebArena).
- **Degradation curve** $D(\ell) = \mathbb{E}[y_t \mid L_t \approx \ell]$, estimated by binning requests by context length.

The confound: $L_t$, $t$, and task residual difficulty are jointly determined by the policy's own behavior. A run is long *because* it is going badly. So the naive $\hat{D}(\ell)$ is a selection-biased estimate; the causal quantity is
$$D^{\mathrm{causal}}(\ell) = \mathbb{E}\big[y_t \mid \mathrm{do}(L_t = \ell), \text{state } s_t\big],$$
which requires holding the underlying environment state fixed while varying the context that describes it.

**Assumptions and their violations.** (i) *Turn-level independence of errors* — violated; an agent that mis-reads a file at $t=3$ conditions on that at $t=4$. (ii) *Stationary task difficulty across turns* — violated by construction. (iii) *$L_t$ is the causal variable* — likely wrong; distractor count, self-contradiction count, and KV-cache reuse pattern all covary with $L_t$. (iv) *A step oracle exists* — violated for open-ended tasks; only end-state oracles are reliable.

## 3. State of the Art

**Established (independently reproduced).**

- Positional degradation in retrieval-style long context: *Lost in the Middle* (Liu et al., TACL 2024) — U-shaped accuracy over gold-document position, reproduced widely.
- Effective context $\ll$ claimed context: RULER (Hsieh et al., COLM 2024) and NoLiMa (Modarressi et al., ICML 2025). NoLiMa removes literal lexical overlap and finds most models fall below half their short-context score by 32K.
- Multi-turn degradation exists and is large: *LLMs Get Lost in Multi-Turn Conversation* (Laban et al., 2025) — sharding a single-turn instruction across turns costs ~39% average performance over 15 models, decomposed into a small aptitude loss and a large reliability loss.

**Claimed but unablated.**

- Context compaction/summarization in production agent harnesses (Anthropic, OpenAI, Cursor, Devin) is universally deployed and, to public knowledge, never compared against a matched full-history control at equal token spend on a step-oracle benchmark. The claim "compaction preserves task performance" is an engineering assertion, not a measured result.
- "Context rot" as a named phenomenon circulates in engineering write-ups with the mechanism (distractor accumulation, self-conditioning, attention dilution) unidentified.

**Benchmark numbers only, no mechanism.** $\tau$-bench (Yao et al., 2024) reports $\mathrm{pass}^k$ collapsing far faster than $\mathrm{pass}@1$ across 8 trials — a consistency failure over multi-turn interaction, but the paper does not attribute it to context length. TheAgentCompany (Xu et al., 2024) reports best-agent full completion around 24% on long consequential tasks; failure modes are catalogued, not causally decomposed.

## 4. What Is Known

- Degradation starts far below advertised limits. FLenQA (Levy, Jacoby, Goldberg, ACL 2024) isolates reasoning from retrieval by padding the same two-hop task; accuracy falls starting at roughly 3K tokens of input, well before any architectural limit, at GPT-4-class scale.
- The multi-turn penalty is mostly *variance*, not *ceiling*. Laban et al. (2025), 15 models × 6 generation tasks, ~200k simulated conversations: best-case performance drops modestly while spread between best and worst run roughly doubles. Reliability is the failing quantity.
- Literal-match benchmarks overstate long-context ability. NoLiMa (ICML 2025): 10 of 12 tested models score below 50% of their own short-context baseline at 32K; GPT-4o falls from 99.3% to 69.7% between short context and 32K.
- Effective length is model-specific and much shorter than claimed. RULER at 4K–128K: only a minority of models advertising $\geq$32K hold the baseline threshold at 32K.
- Attention-sink work (Xiao et al., ICLR 2024) shows streaming stability is achievable *for perplexity* over millions of tokens while task accuracy is not preserved — fluency and competence decouple.

None of these were measured on self-authored context. Every one uses externally supplied documents.

## 5. What Is Not Known

- **Empirically open (the core gap).** Whether, at fixed $L$ and fixed underlying task state, self-authored trajectory context degrades performance more than an equal-length human-authored or retrieved context. The experiment is runnable today on $\tau$-bench or SWE-bench with a step oracle; nobody has published the paired comparison.
- **Empirically open.** Whether compaction at budget $B$ beats truncation and beats full history, at matched token spend. Trivial to run, not run.
- **Methodologically blocked.** $D^{\mathrm{causal}}(\ell)$ requires intervening on context while holding environment state fixed. There is no accepted procedure for constructing a counterfactual context that describes the *same* state at a different length without also changing information content.
- **Methodologically blocked.** Step-level correctness $y_t$ for open-ended agentic work. Without it, only trajectory-terminal signals exist, and those cannot localize the degradation.
- **Theoretically open.** No bound relating error accumulation to $\sigma_t$ (self-authored fraction). Whether autoregressive self-conditioning induces super-linear compounding under any realistic assumption on the transformer is unproven either way.

## 6. Why It Is Hard

The specific obstruction is **selection-biased measurement**: context length is an outcome of the policy, not an input to it. Long trajectories are the ones that went wrong, so $\hat{D}(\ell)$ measures "how hard were the tasks that survived to length $\ell$" and calls it "how well does the model use $\ell$ tokens". Every agentic long-context number published to date has this defect.

Two aggravating factors. (1) **Absent step ground truth** — you cannot localize a failure to a turn without a per-turn oracle, and building one requires programmatic environment state, which restricts the study to a handful of environments. (2) **Non-identifiability of the cause** — $L_t$, $t$, $\sigma_t$, distractor count, and residual difficulty are collinear along a natural trajectory; separating them requires synthetic intervention, which then raises the question of whether the synthetic context is representative.

## 7. Current Research (as of 2026)

- **Multi-turn evaluation.** $\tau$-bench and $\tau^2$-bench (Sierra), MINT (Wang et al., ICLR 2024), TheAgentCompany (CMU) — all push toward long-horizon, step-checkable environments. None yet reports a length-controlled arm. *(frontier — verify)*
- **Context engineering as a first-class object.** Compaction, structured note-taking, sub-agent context isolation in production harnesses; described publicly by Anthropic and others as engineering practice, without controlled ablation. *(frontier — verify)*
- **Multi-turn RL on trajectories.** ArCHer (Zhou et al., ICML 2024) and successors optimize over turns rather than tokens; an open question is whether RL-trained agents learn context hygiene implicitly.
- **Mechanistic long-context work.** Attention sinks, KV eviction, and retrieval-head analysis — applied to document context, not to self-authored context.

## 8. Concrete Next Experiment

**Question.** At fixed context length and fixed environment state, does self-authored context cost more accuracy than externally authored context of the same length?

**Scale.** $\tau$-bench retail + airline (about 165 tasks), 3 frontier models, 8 seeds. Roughly $165 \times 3 \times 8 \times 4$ arms $\approx$ 16k trajectories; at ~50k tokens mean context, order $10^9$ tokens. Days on a single API budget, not a training run.

**Arms, all evaluated at the same decision points with the same database-state oracle:**

1. **Control:** full self-authored history (the standard agent).
2. **Paraphrase-matched:** at each decision point, replace the self-authored history with a *third-model paraphrase* of identical information content, token count within $\pm 3\%$. Isolates authorship from information.
3. **Padded-oracle:** minimal correct state description plus irrelevant filler to the same token count. Isolates length from content.
4. **Compacted:** harness summarization to 25% of control length.

**Deciding number.** $\Delta = \mathrm{acc}(\text{arm 2}) - \mathrm{acc}(\text{arm 1})$ at contexts in the 40–80K bin. If $\Delta \geq 5$ points with 95% CI excluding zero, self-authorship is a real and separable cause of degradation and context hygiene is a mechanism-level intervention. If $|\Delta| < 2$ points while arm 3 also degrades, the cause is length alone and compaction is the whole story.

## 9. Key References

- **[Foundational]** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL, 2024. — arXiv:2307.03172
- **[Foundational]** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR, 2023. — arXiv:2210.03629
- **[SOTA]** Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville. *LLMs Get Lost in Multi-Turn Conversation.* 2025. — arXiv:2505.06120
- **[SOTA]** Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, Hinrich Schütze. *NoLiMa: Long-Context Evaluation Beyond Literal Matching.* ICML, 2025. — arXiv:2502.05167
- **[SOTA]** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM, 2024. — arXiv:2404.06654
- **[SOTA]** Shunyu Yao, Noah Shinn, Pedram Razavi, Karthik Narasimhan. *$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024. — arXiv:2406.12045
- **[Empirical]** Mosh Levy, Alon Jacoby, Yoav Goldberg. *Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models.* ACL, 2024. — arXiv:2402.14848
- **[Empirical]** Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, Heng Ji. *MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback.* ICLR, 2024. — arXiv:2309.10691
- **[Empirical]** Frank F. Xu, Yufan Song, Boxuan Li, et al. *TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks.* 2024. — arXiv:2412.14161
- **[Method]** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. — arXiv:2309.17453
- **[Survey]** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR, 2024. — arXiv:2310.06770

## 10. Worked Example

A retail-support agent, 40 turns, ending in a wrong refund. Logged: $L_{40} = 78{,}400$ tokens, $\sigma_{40} = 0.44$ (34.5k tokens the model wrote itself), 11 tool calls, 3 of which returned errors that were retried verbatim.

Naive analysis: bin all trajectory decision points by $L_t$. Suppose success is 0.81 in the 0–20K bin and 0.44 in the 60–80K bin. Reported as "37-point degradation from 20K to 80K".

Now check the selection. Count how tasks reach the 60–80K bin: in a typical $\tau$-bench run, a large majority of successful tasks terminate under 25K tokens, so the 60–80K bin is populated almost entirely by tasks that already failed a step. Write $p$ for the fraction of that bin whose trajectory contains an earlier error. If $p = 0.7$ and conditional-on-earlier-error success is 0.25 regardless of length, then the length-free prediction for the bin is
$$0.7 \times 0.25 + 0.3 \times 0.81 = 0.418,$$
against the observed 0.44. The measured "37-point degradation" is fully explained without any context-length effect at all. The residual attributable to length is $0.44 - 0.418 = 0.022$ — two points, inside noise at $n \sim 200$.

This is the obstruction made concrete: the headline number and the null hypothesis are numerically indistinguishable on observational trajectory data. Arm 2 of §8 — same state, same length, different authorship — is the cheapest way to break the tie, because it fixes $p$ by construction.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*