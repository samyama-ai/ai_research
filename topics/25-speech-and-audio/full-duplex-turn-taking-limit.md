---
id: 25-speech-and-audio/full-duplex-turn-taking-limit
title: "Full-Duplex Conversational Turn-Taking Prediction Limit"
topic: 25-speech-and-audio
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Full-Duplex Conversational Turn-Taking Prediction Limit

> **Topic:** Speech & Audio · **ID:** `25-speech-and-audio/full-duplex-turn-taking-limit` · **Status:** open

## 1. Problem Statement

A full-duplex spoken dialogue system listens and speaks on overlapping channels and must decide, continuously, whether to start speaking, keep speaking, or yield. The question: **is there an irreducible ceiling on turn-taking prediction accuracy from a causal audio stream, and where is it?**

Input: two-channel audio $x_{1:t}$ (system channel, user channel), causal, no lookahead. Output: at every frame, a distribution over near-future voice activity of both speakers. Decision predicate: at a candidate transition-relevance place, predict SHIFT (other speaker takes the floor) versus HOLD (current speaker continues).

Three variants that get conflated:

- **Measurement variant.** What is the Bayes error of SHIFT/HOLD given only the acoustic-lexical past? Human turn-taking is partly *decided* by the listener, not *determined* by the signal; some fraction of the label is genuinely stochastic. Nobody has estimated that fraction.
- **Method variant.** Close the gap between current models and whatever the ceiling is. Runnable today.
- **Theory variant.** Is causal turn-taking prediction bounded away from the offline (bidirectional) predictor by a quantity determined by the response-planning latency of the interlocutor? No proof either way.

Solving it means: a defensible estimate of $H(\text{next speaker} \mid \text{causal past})$ on natural conversation, plus a model that provably attains it.

## 2. Formal Setting

Let $v_t^{(i)} \in \{0,1\}$ be voice activity of speaker $i \in \{1,2\}$ at frame $t$ (frame rate $f = 50$ Hz, i.e. 20 ms; measured by a forced-aligned or energy-based VAD on **separate close-talk channels**, not on the mixture).

**Voice Activity Projection (VAP).** Discretize the next $T = 2$ s into $K$ bins with boundaries $(0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0)$ s. Bin activity:

$$b_k^{(i)}(t) = \mathbb{1}\!\left[\frac{1}{|B_k|}\sum_{s \in B_k} v_{t+s}^{(i)} > 0.5\right], \qquad z_t = \big(b^{(1)}_{1:K}, b^{(2)}_{1:K}\big) \in \{0,1\}^{2K}.$$

With $K=4$ this gives 256 states; models predict $p_\theta(z_t \mid x_{1:t})$.

**The decision quantity.** At an event $e$ (a mutual silence following speaker $i$), define
$$p_{\text{shift}}(t) = \Pr\big[\text{speaker } j \neq i \text{ holds the floor over } [t, t+T] \mid x_{1:t}\big],$$
obtained by marginalizing $p_\theta(z_t\mid\cdot)$ over states where $j$ dominates. The reported score is **balanced accuracy** over matched SHIFT/HOLD events, because the class prior is skewed (HOLD is roughly 3–5× more frequent at short pauses).

**Timing.** Floor transfer offset $\text{FTO} = t_{\text{onset}}^{(j)} - t_{\text{offset}}^{(i)}$, signed; negative means overlap. A system is evaluated not only on the SHIFT/HOLD label but on $D_{\mathrm{KL}}\big(P_{\text{human}}(\text{FTO}) \,\|\, P_{\text{model}}(\text{FTO})\big)$ — matching human timing distribution, not minimizing latency.

**The ceiling.** Define the causal Bayes error $\varepsilon^\star = \min_f \Pr[f(x_{1:t}) \neq y_e]$ over the true conditional. The open quantity is $\varepsilon^\star$ and the gap $\Delta = \varepsilon^\star - \varepsilon^\star_{\text{offline}}$ where the offline predictor sees $x_{1:t+T}$.

**Assumptions, and which are violated:**

1. *Two speakers, separable channels.* Violated for multiparty and for single-mic deployment; on mixed audio, diarization error injects label noise of the same order as the effect being measured.
2. *The label is a function of the signal.* Violated. Whether B takes the floor at a 300 ms pause depends on B's internal state — planning completion, intent — which is not in $x_{1:t}$. This is exactly why $\varepsilon^\star > 0$.
3. *Stationarity across corpora.* Violated: Stivers et al. (PNAS 2009) found mean FTO varying by roughly 250 ms across ten languages; telephone (Switchboard) and face-to-face (Candor) differ in overlap rate.
4. *Human-human data transfers to human-machine.* Violated: users speak to machines with longer, less overlapped turns.

## 3. State of the Art

**Empirical SOTA — established.** *Voice Activity Projection* (Ekstedt & Skantze, Interspeech 2022) is the standard causal formulation; the 256-state discrete VAP head, trained on Switchboard/Fisher, reports balanced accuracy in the **high 70s to low 80s** on SHIFT/HOLD at pauses, with performance strongly dependent on how the evaluation events are filtered. Reproduced independently by several groups; the code and event-extraction protocol are public, which is why this number is the reference point.

**Text-only baseline — established.** TurnGPT (Ekstedt & Skantze, Findings of EMNLP 2020) predicts turn-completion from transcript alone and is competitive at long pauses but loses badly at short ones — establishing that prosody carries the short-horizon signal.

**Full-duplex generative systems — claimed, largely unablated.** Moshi (Défossez et al., Kyutai 2024) models two parallel audio streams with an "inner monologue" text stream and reports ~200 ms theoretical / ~160 ms practical latency. SyncLLM (Veluri et al., EMNLP 2024) synchronizes an LLM to real time. LSLM (Ma et al., 2024) adds a listening channel for barge-in. These report end-to-end latency and subjective quality; **none reports a controlled SHIFT/HOLD balanced accuracy against a VAP baseline on the same events**. The turn-taking claims exist as system demos, not ablations.

**Benchmarks.** Full-Duplex-Bench (Lin et al., 2025) is the first attempt at standardized turn-taking metrics for full-duplex models (pause handling, backchannel, barge-in, user interruption). Its numbers are benchmark numbers only — the metric definitions are new and not yet cross-validated against human judgments.

**Theory SOTA.** There is none for $\varepsilon^\star$. The nearest formal object is the Sacks–Schegloff–Jefferson (Language, 1974) turn-taking system, which is descriptive, not predictive.

## 4. What Is Known

- **Human gaps are short and tightly distributed.** Stivers et al. (PNAS 2009), 10 languages, ~350 transitions each: modal FTO near 0–200 ms; cross-language means span roughly 0 to 250 ms.
- **Overlap is common, not exceptional.** Heldner & Edlund (Journal of Phonetics, 2010), Swedish/Scottish/Dutch corpora: roughly 40–45% of between-speaker intervals involve overlap; gaps under 200 ms dominate the rest.
- **Timing beats content at short horizons.** Levinson & Torreira (Frontiers in Psychology, 2015) argue from production-latency data that speech planning takes ≥600 ms, so the listener must *predict* completion, not react to it. This is the strongest evidence that a causal predictor is doing something humans also do.
- **Prosody adds measurably over text.** VAP ablations at Switchboard scale (~260 h, 2400 conversations) show the audio model beating transcript-only at pauses <400 ms; the margin shrinks as pause length grows.
- **Model scale is not the binding constraint at current sizes.** Reported VAP gains from larger encoders are small relative to gains from event-filtering choices — a warning sign about the metric, discussed in §6.

## 5. What Is Not Known

- **Methodologically blocked:** $\varepsilon^\star$ itself. There is no accepted estimator of the irreducible error, because the ground truth is a single realization — we observe what B *did*, never the distribution over what B *could have* done. Without a repeated-trial construction, "Bayes error" is not measurable, only bounded above by the best model.
- **Empirically open:** whether large full-duplex audio LMs (Moshi-class, ~7B) actually exceed a 40M-parameter VAP head on matched SHIFT/HOLD events. The experiment is a week of work; nobody has published it.
- **Empirically open:** whether human listeners, given the same truncated causal context, score at, above, or below current models. A human-ceiling study at $n \geq 1000$ events does not exist.
- **Theoretically open:** whether $\Delta = \varepsilon^\star - \varepsilon^\star_{\text{offline}}$ is bounded below by a positive constant determined by planning latency. No proof either way.

## 6. Why It Is Hard

**The specific obstruction is absent ground truth compounded by an evaluation that does not measure what it names.**

Balanced accuracy on SHIFT/HOLD depends on which pauses you call events. Filter to pauses in $[250, 500]$ ms with $\geq 1$ s of prior single-speaker activity and you get one number; widen to $[100, 2000]$ ms and you get another, several points apart, on the same model. The score is a property of the *event extractor* as much as of the model. Published comparisons often differ in the extractor, so cross-paper numbers are not comparable.

Underneath that: the label is a choice, not a fact. If B declines to take the floor at a 300 ms pause and takes it at 700 ms instead, the model that predicted SHIFT at 300 ms is scored wrong, though nothing in the signal distinguished the two worlds. This is irreducible label noise whose magnitude is unknown, and it puts a floor under every reported error rate that no amount of compute removes.

## 7. Current Research (as of 2026)

- **Speech–text interleaved duplex LMs.** Kyutai (Moshi lineage), the LSLM/Parrot line, and Freeze-Omni-style frozen-LLM adapters continue to push latency down. *(frontier — verify)* the open question they have not addressed is whether latency reduction trades against timing-distribution fidelity.
- **Turn-taking as a conditioned prediction.** Ekstedt & Skantze's response-conditioned prediction (2023) — predicting the shift given a candidate response — is the most direct attack on the "the label depends on the responder's plan" problem, since it puts part of the plan in the input.
- **Benchmark consolidation.** Full-Duplex-Bench and successors are converging on a shared event protocol; adoption is partial.
- **Cross-cultural and multiparty extension.** Corpora such as Candor (face-to-face, large-scale) enable overlap-rate questions telephone corpora cannot answer. *(frontier — verify)* whether models trained on Switchboard transfer to face-to-face timing.

## 8. Concrete Next Experiment

**Question:** does a 7B full-duplex audio LM beat a 40M causal VAP head on turn-taking, and where is the human ceiling?

**Scale.** Fisher + Switchboard held-out split, $n = 5000$ matched SHIFT/HOLD events, extracted with one frozen, published event filter (pause $\in [250, 500]$ ms, $\geq 1$ s prior single-speaker activity, no backchannel within 500 ms), separate channels, no lookahead beyond frame $t$.

**Arms.**
1. VAP-50Hz causal head, ~40M params (control).
2. TurnGPT text-only on ASR transcript (lower control — isolates prosody).
3. Moshi-class 7B duplex LM, probed for $p_{\text{shift}}$ by sampling 64 continuations per event and counting which speaker holds the floor.
4. **Human ceiling arm:** 20 annotators hear audio truncated at frame $t$ and answer SHIFT/HOLD; 3 annotators per event on a 1000-event subset.

**Deciding number.** Balanced accuracy on the identical 5000 events, plus the human arm's balanced accuracy. If arm 3 does not exceed arm 1 by $\geq 3$ points absolute (95% CI via bootstrap over conversations), scale in duplex LMs is not buying turn-taking. If arm 4 lands within 2 points of arm 1, current models are already at the human causal ceiling and the remaining error is the irreducible term — which would convert the measurement variant from blocked to answered.

**Secondary:** report $D_{\mathrm{KL}}$ between human and model FTO distributions per arm. A system that wins on accuracy while shifting modal FTO from 200 ms to 0 ms has not solved the problem.

## 9. Key References

- **[Foundational]** Harvey Sacks, Emanuel A. Schegloff, Gail Jefferson. *A Simplest Systematics for the Organization of Turn-Taking for Conversation.* Language 50(4), 1974.
- **[Foundational]** Tanya Stivers et al. *Universals and cultural variation in turn-taking in conversation.* PNAS 106(26), 2009.
- **[Foundational]** Mattias Heldner, Jens Edlund. *Pauses, gaps and overlaps in conversations.* Journal of Phonetics 38(4), 2010.
- **[Foundational]** Stephen C. Levinson, Francisco Torreira. *Timing in turn-taking and its implications for processing models of language.* Frontiers in Psychology 6:731, 2015.
- **[SOTA]** Erik Ekstedt, Gabriel Skantze. *Voice Activity Projection: Self-supervised Learning of Turn-taking Events.* Interspeech, 2022.
- **[SOTA]** Erik Ekstedt, Gabriel Skantze. *TurnGPT: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog.* Findings of EMNLP, 2020.
- **[SOTA]** Alexandre Défossez et al. *Moshi: a speech-text foundation model for real-time dialogue.* Kyutai technical report, 2024.
- **[SOTA]** Bandhav Veluri et al. *Beyond Turn-Based Interfaces: Synchronous LLMs as Full-Duplex Dialogue Agents.* EMNLP, 2024.
- **[SOTA]** Tuan Dinh Nguyen et al. *Generative Spoken Dialogue Language Modeling.* TACL, 2023.
- **[Benchmark]** Guan-Ting Lin et al. *Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities.* 2025.
- **[Survey]** Gabriel Skantze. *Turn-taking in Conversational Systems and Human-Robot Interaction: A Review.* Computer Speech & Language 67, 2021.

## 10. Worked Example

Take one Switchboard pause. Speaker A ends "…so we ended up going to my sister's place in **Dallas**" with falling F0 and 180 ms of final lengthening. Mutual silence begins at $t_0$. The VAP head at $t_0$ outputs $p_{\text{shift}} = 0.71$. B stays silent; A continues at $t_0 + 410$ ms with "…which was, you know, a whole thing." Label: HOLD. The model is scored wrong.

Now the arithmetic that exposes the obstruction. Suppose the true conditional at this event is genuinely $\Pr[\text{SHIFT}] = 0.7$ — the prosody really does signal completion, and B declined for reasons in B's head. A perfect predictor outputs $0.7$ and, under 0/1 scoring, is wrong 30% of the time on events like this. If the event set has this structure with $\Pr[\text{SHIFT}] \in \{0.3, 0.7\}$ split evenly, the Bayes error is $0.30$ and the ceiling on balanced accuracy is **70%** — *below* the ~80% currently reported.

That contradiction is the point. Reported accuracy above the naive ceiling means the observed event set is not the ambiguous one: the extractor has selected easy events (long clean pauses after clear syntactic completion), where $\Pr[\text{SHIFT}]$ is near 0 or 1. Widen the filter to include 100–250 ms pauses and overlapped onsets — the cases that actually matter for a full-duplex agent, and the majority of real transitions given Heldner & Edlund's ~40% overlap rate — and accuracy falls, because the ambiguous mass enters.

So the number in the papers is not an estimate of turn-taking skill. It is a joint measurement of the model and of how much ambiguity the event filter excluded, and the two are not separated by any published protocol. Until arm 4 of §8 is run, no one knows which of the ~20% error is model deficiency and which is $\varepsilon^\star$.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*