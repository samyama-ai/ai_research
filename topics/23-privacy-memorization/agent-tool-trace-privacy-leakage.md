---
id: 23-privacy-memorization/agent-tool-trace-privacy-leakage
title: "Privacy Leakage Through Agent Tool Traces and Logs"
topic: 23-privacy-memorization
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Privacy Leakage Through Agent Tool Traces and Logs

> **Topic:** Privacy & Memorization · **ID:** `23-privacy-memorization/agent-tool-trace-privacy-leakage` · **Status:** open

## 1. Problem Statement

An LLM agent given access to a user's data (mailbox, calendar, files, browser session) does not keep that data inside the model. It copies it into **tool arguments**, **retrieval queries**, **scratchpad reasoning**, **error messages**, and **observability traces** — each of which lands in a different trust domain: a third-party API, a vector database, a logging backend (LangSmith, Datadog, CloudWatch), a support engineer's console, a model provider's training pipeline.

The problem: **given an agent policy and a task distribution, bound the probability that a private attribute reaches a sink that was not authorized to receive it.**

Three variants, with different difficulty:

- **Measurement.** Define and estimate a leakage rate per (attribute, sink) pair from execution traces. Blocked less by compute than by the absence of an agreed leakage predicate — see §5.
- **Method.** Build an agent that completes tasks at unchanged utility while emitting no unauthorized attribute. Minimization (send the tool only the fields it needs) is the leading family; it is unsolved because *needed* is task-dependent and decided before the outcome is known.
- **Theory.** Give a composition bound: if each tool call leaks at most $\varepsilon$, what does a $T$-step trajectory leak? No agent-specific answer exists; differential privacy's composition theorems do not apply because tool calls are adaptive, non-noised, and carry raw strings.

Solving it means: a certificate, checkable from the trace alone, that no attribute outside the authorized set for sink $s$ appeared in any message routed to $s$ — with a measured task-success cost.

## 2. Formal Setting

**Objects.** A user context $C$ is a set of attribute–value pairs $\{(x_i, v_i)\}_{i=1}^n$ (home address, diagnosis, salary, a colleague's phone number). A task $q$ is drawn from $\mathcal{D}$. The agent policy $\pi$ produces a trajectory

$$\tau = (a_1, o_1, \dots, a_T, o_T), \qquad a_t = (s_t, m_t),$$

where $s_t \in \mathcal{S}$ is the **sink** (tool endpoint, log stream, retrieval index) and $m_t$ the message sent to it. Sinks are the measured unit, not tools: one tool call typically writes to two or three sinks (the API, the trace store, stdout).

**Norm.** Following contextual integrity (Nissenbaum, 2004), each attribute carries a permitted-recipient set $\Pi(x_i) \subseteq \mathcal{S}$. Leakage on trajectory $\tau$:

$$L(\tau) = \sum_{t=1}^{T} \big| \{\, i : x_i \in \mathrm{Att}(m_t) \ \wedge\ s_t \notin \Pi(x_i) \,\} \big|,$$

and the **leakage rate** $\mathcal{L}(\pi) = \Pr_{q \sim \mathcal{D}}[L(\tau) > 0]$.

**Measurement of $\mathrm{Att}(m)$** — the attributes actually present in a message. Three estimators, none exact:
1. *String match* on canonical values $v_i$. Undercounts paraphrase ("the hospital on Elm" for an address); this is the operator most deployed redaction uses.
2. *NER/PII classifier* (Presidio-class). Type-level, not instance-level: flags that an address appeared, not *whose*.
3. *LLM judge* against the attribute list. Highest recall, unquantified false-positive rate, and it is itself a sink.

**Information-theoretic form.** For sink $s$, let $\Sigma_s(\tau)$ be the concatenation of messages routed to $s$. Define $I(X_{\bar\Pi(s)} ; \Sigma_s \mid q)$, the mutual information between attributes *not* authorized for $s$ and what $s$ observes. Estimating this requires a distribution over $C$; in deployment $C$ is a single realized context, so the quantity is not estimable from one user's traces — only across a synthetic population.

**Utility.** $U(\pi) = \Pr[\text{task judged complete}]$. The object of interest is the frontier $\{(\mathcal{L}(\pi), U(\pi))\}$, not either alone.

**Assumptions, and which are violated.**
- *Attributes are enumerable.* Violated: inference-time attributes (Staab et al., ICLR 2024) — a model reconstructs location from writing style, so $\mathrm{Att}$ is not a function of surface strings.
- *Sinks are known and static.* Violated: retries, SDK-level telemetry, and sub-agent spawning create sinks the policy never named.
- *$\Pi$ is available.* Violated: real norms are underspecified; ConfAIde-style annotation disagreement among humans is nontrivial.
- *Traces are complete.* Violated: sampled tracing (common at 1–10%) makes $\mathcal{L}$ a biased estimate downward.
- *Leakage is memoryless across steps.* Violated by design: observation $o_t$ re-enters the context and is re-emitted at $t' > t$.

## 3. State of the Art

**Established (ablated, reproduced):**
- **Indirect prompt injection** turns a benign agent into an exfiltration channel. Greshake et al. (AISec@CCS 2023) demonstrated it; **AgentDojo** (Debenedetti et al., NeurIPS 2024 D&B) made it a controlled benchmark — 97 tasks, 629 security cases, with utility and attack-success measured on the same suite, so defenses can be scored on both axes. This is the strongest methodology in the area.
- **Privacy side channels** in ML systems (Debenedetti et al., USENIX Security 2024): deduplication, caching and filtering components leak membership even when the model does not. The agent analogue — timing and retrieval-hit side channels — is demonstrated, not yet benchmarked.
- **RAG leaks its corpus.** Zeng et al. (Findings of ACL 2024) extract verbatim private documents from retrieval-augmented pipelines via crafted queries.

**Claimed but unablated:**
- Trace-redaction / PII-scrubbing middleware shipped by observability vendors. No public evaluation reports the recall of the scrubber against paraphrase, nor the utility cost of scrubbing before the model reads the observation.
- "Privacy-aware system prompts." Reported to reduce leakage; PrivacyLens (Shao et al., NeurIPS 2024 D&B) shows the reduction does not transfer from question-answering to action.

**Benchmark-number-only results:** PrivacyLens leakage rates, ConfAIde tier-4 rates, and InjecAgent attack-success rates are each single-suite numbers on constructed scenarios. None has a demonstrated correlation with leakage in a deployed agent's real trace store — that correlation study does not exist.

**Method SOTA:** **AirGapAgent** (Bagdasarian et al., CCS 2024) — a context minimizer decides, per query, which fields the task-executing agent may see. It is the only published architecture with an explicit minimization boundary rather than a post-hoc filter.

## 4. What Is Known

- **Models that state the norm still violate it.** PrivacyLens: even on cases where the model answers the probing privacy question correctly, GPT-4 leaks in ~25.7% of agent actions and Llama-3-70B in ~38.7%. Scale: ~493 seed norms expanded to vignettes and trajectories.
- **Knowing ≠ doing at the QA level too.** ConfAIde (Mireshghallah et al., ICLR 2024) tier 4: GPT-4 reveals secrets in ~22% of scenarios, ChatGPT in ~93%, despite passing tier-1 norm questions.
- **Tool-integrated agents are injectable.** InjecAgent (Zhan et al., Findings of ACL 2024): 1,054 test cases, 17 user tools; ReAct-prompted GPT-4 attacked successfully in ~24% of cases, higher with a "hacking prompt."
- **Attributes can be inferred, not just copied.** Staab et al. (ICLR 2024): GPT-4 infers personal attributes from Reddit text at ~85% top-1 accuracy — so a leak-free-by-string-match trace can still be a leak.
- **Verbatim memorization is real but is not the dominant agent channel.** Carlini et al. (USENIX Security 2021) extracted ~600 memorized sequences from GPT-2 (1.5B). Agent traces move orders of magnitude more raw user data per session than extraction attacks recover from weights.
- **Users disclose more than PII.** Mireshghallah et al. (COLM 2024) analysed real WildChat/ShareGPT conversations and found sensitive disclosures (health, relationships, employment) that no PII taxonomy tags.

## 5. What Is Not Known

- **Methodologically blocked — the central gap.** There is no agreed, instance-level operator $\mathrm{Att}(m)$. Every reported leakage rate is the rate *of a particular detector*, and no paper publishes that detector's precision/recall against human labels on agent traces. Two systems' numbers are therefore not comparable.
- **Methodologically blocked.** No ground-truth $\Pi$ for real deployments; norms are annotated per benchmark, and inter-annotator agreement on agent scenarios is unreported.
- **Empirically open.** The leakage rate of a *production* agent measured on its own trace store, with a human-labelled sample. Runnable today by any vendor with traces; no public result exists. Also open: does benchmark leakage rank systems the same way real traces do?
- **Empirically open.** Utility cost of aggressive minimization at fixed task difficulty — AirGapAgent reports its own setting; no cross-architecture frontier.
- **Theoretically open.** A composition bound for adaptive, non-randomized tool calls. Whether any DP-style guarantee is achievable when messages are raw strings chosen by the policy is unproven either way. Also open: whether a trace-only certificate can be sound against inference-based leakage — plausibly not, since $\mathrm{Att}$ then depends on the receiver's model.

## 6. Why It Is Hard

**Absent ground truth compounded by non-identifiability.** To score a trace you need (a) the true attribute set of the user, (b) the true norm for each sink, (c) an operator that decides whether a paraphrase carries the attribute. (a) and (b) are unavailable outside constructed benchmarks; (c) is undecidable in general because "carries the attribute" is receiver-relative — a message leaks to a recipient who can invert it and does not leak to one who cannot.

**Confounded measurement.** Leakage rate and task utility are estimated on the same trajectories, and the redaction that lowers one lowers the other through the same mechanism (the model no longer sees the field). Reporting either alone is uninformative, yet most systems report only one.

**The measurement instrument is a sink.** An LLM judge over traces sends the private content to another model. Any high-recall detector re-creates the exposure it measures — which is why deployed scrubbers use low-recall regex.

## 7. Current Research (as of 2026)

- **Contextual-integrity-native agent architectures** — minimization decided before execution (Google DeepMind's AirGapAgent line; follow-on work operationalizing CI for assistants). Direction is established; scaling to multi-tool, multi-turn is *(frontier — verify)*.
- **Provable injection defenses with utility accounting** — CaMeL-style control/data separation and AgentDojo-scored defenses (ETH Zürich SRI, Google). Established as a benchmark practice.
- **Trace-store governance**: retention limits, per-sink scoping, encrypted spans in observability platforms. Vendor-side, largely unevaluated in public *(frontier — verify)*.
- **Sub-agent and memory leakage** — persistent agent memory as a cross-session sink. Actively discussed; no benchmark with published detector validation *(frontier — verify)*.

## 8. Concrete Next Experiment

**Question:** does benchmark leakage predict trace leakage, and what does the detector actually cost?

**Scale.** One agent stack (ReAct, 6 tools: email, calendar, files, web search, code exec, ticketing) run over 2,000 tasks from a synthetic 200-user population with fully enumerated $C$ (mean $n = 40$ attributes) and a hand-specified $\Pi$ per sink. Log all sinks, unsampled — including the trace store and stderr. ~2,000 trajectories × ~12 steps ≈ 24k messages; well under $1k of inference on a mid-tier model.

**Human anchor.** Label 1,000 randomly sampled messages by two annotators for "does this message convey attribute $x_i$ to this sink." Report agreement.

**Arms.** (1) unmodified agent (control); (2) regex/Presidio scrubbing of tool arguments; (3) AirGapAgent-style pre-execution minimization; (4) minimization + scrubbing.

**Deciding number.** The **detector-corrected leakage rate** $\hat{\mathcal{L}}$ per arm, reported jointly with utility $U$, plus the single scalar that settles §5's first gap: **the recall of string-match $\mathrm{Att}$ against the human labels.** If recall $\ge 0.9$, every published leakage rate is roughly comparable and the field can proceed to methods. If recall $\le 0.6$ — the outcome the Staab et al. inference results predict — then all existing numbers understate leakage by an unknown factor and the measurement must be rebuilt before any method claim is meaningful.

## 9. Key References

- **[Foundational]** Helen Nissenbaum. *Privacy as Contextual Integrity.* Washington Law Review, 2004.
- **[Foundational]** Nicholas Carlini, Florian Tramèr, Eric Wallace, et al. *Extracting Training Data from Large Language Models.* USENIX Security, 2021. — arXiv:2012.07805
- **[SOTA]** Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, et al. *Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory.* ICLR, 2024. — arXiv:2310.17884
- **[SOTA]** Yijia Shao, Tianshi Li, Weiyan Shi, Yanchen Liu, Diyi Yang. *PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2409.00138
- **[SOTA]** Eugene Bagdasarian, Ren Yi, Sahra Ghalebikesabi, et al. *AirGapAgent: Protecting Privacy-Conscious Conversational Agents.* ACM CCS, 2024. — arXiv:2405.05175
- **[SOTA]** Edoardo Debenedetti, Jie Zhang, Mislav Balunović, et al. *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* NeurIPS Datasets & Benchmarks, 2024. — arXiv:2406.13352
- **[SOTA]** Robin Staab, Mark Vero, Mislav Balunović, Martin Vechev. *Beyond Memorization: Violating Privacy via Inference with Large Language Models.* ICLR, 2024. — arXiv:2310.07298
- Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, et al. *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* AISec@CCS, 2023. — arXiv:2302.12173
- Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang. *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents.* Findings of ACL, 2024. — arXiv:2403.02691
- Edoardo Debenedetti, Giorgio Severi, Nicholas Carlini, et al. *Privacy Side Channels in Machine Learning Systems.* USENIX Security, 2024. — arXiv:2309.05610
- Shenglai Zeng, Jiankun Zhang, Pengfei He, et al. *The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG).* Findings of ACL, 2024. — arXiv:2402.16893
- Niloofar Mireshghallah, Maria Antoniak, Yash More, Yejin Choi, Golnoosh Farnadi. *Trust No Bot: Discovering Personal Disclosures in Human-LLM Conversations in the Wild.* COLM, 2024. — arXiv:2407.11438
- **[Survey]** Nils Lukas, Ahmed Salem, Robert Sim, et al. *Analyzing Leakage of Personally Identifiable Information in Language Models.* IEEE S&P, 2023. — arXiv:2302.00539

## 10. Worked Example

**Task.** "Book me the cheapest flight to my next conference and file the expense."

**Context.** $C$ includes `home_address`, `passport_no`, `dietary_restriction = coeliac`, `employer`, `manager_email`. Norm: `passport_no` authorized only for the airline API; `dietary_restriction` for the airline API only; nothing else for the expense tool.

**Trace, 9 steps.** Step 3 the agent calls `search_flights(passenger=..., notes="gluten-free meal, coeliac")`. Step 6 it calls `file_expense(description="Flight for <name>, coeliac meal surcharge $18")`. Step 7 it retries after a 500 error; the retry payload is written verbatim to the trace store.

**Count.** Unauthorized emissions: step 6 (`dietary_restriction` → expense tool, $\notin \Pi$), step 6 again (trace store), step 7 (trace store, retry). $L(\tau) = 3$ from **one** model decision — the 3× multiplier is the sink fan-out, invisible in any evaluation that counts tool calls rather than sinks. Every agent benchmark listed in §9 scores this trajectory as one violation.

**Now the obstruction.** Run the three estimators on step 6's message:
- String match on canonical value `"coeliac"` → **detected**.
- Presidio → **not detected** (health condition is not a default PII entity in the common configuration).
- Now change the model's phrasing to `description="Flight for <name>, special meal surcharge $18"` plus a step-4 message `"avoid all wheat-based options"`. String match → **0 detections**. A human annotator, and GPT-4 as judge, both recover `dietary_restriction = coeliac` from the pair with high confidence.

The measured leakage rate moved from 3 to 0 with no change in what the expense tool's operator can learn. That is the whole problem: $\mathcal{L}$ as currently measured is a property of the detector and the model's word choice, not of the information transferred. Until §8's recall number exists, "we reduced leakage by 60%" is an unfalsifiable claim.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*