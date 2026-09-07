---
id: 28-knowledge-editing/black-box-api-model-editing
title: "Editing Without Access to Gradients or Weights"
topic: 28-knowledge-editing
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Editing Without Access to Gradients or Weights

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/black-box-api-model-editing` · **Status:** open

## 1. Problem Statement

A deployed model is reachable only through an inference API: text in, text (and sometimes truncated top-$k$ logprobs) out. No parameters, no gradients, no activations, no fine-tuning endpoint for the target checkpoint. The task is to install a specific factual or behavioural change — "the CEO of X is now Y" — so that it holds under paraphrase and multi-hop composition, does not leak into unrelated inputs, and persists across sessions.

Three variants, with different difficulty:

- **Method.** Build a wrapper $W$ (prompt prefix, retrieval memory, output-side logit combiner, or query rewriter) such that the composed system $W \circ f$ behaves as an edited model. Runnable today; the open part is whether any wrapper achieves the generalization of a weight edit.
- **Measurement.** Decide whether an edit "took" when you cannot inspect the model. All parametric-editing metrics (locality, ripple effects) assume you can probe the same weights before and after. Under an API, the base model is itself a moving target — providers silently update checkpoints.
- **Theory.** Characterize which behavioural changes are reachable by a bounded-length context or a bounded-compute output transform, and which provably are not. Essentially untouched.

**Solved** would mean: a wrapper that matches parametric editing on efficacy, paraphrase generalization, and multi-hop propagation, at bounded per-query token cost, with locality damage no worse than the weight-edit baseline, demonstrated on a frontier API model.

## 2. Formal Setting

Let $f_\theta: \mathcal{V}^* \to \Delta(\mathcal{V})$ be the target model, with $\theta$ inaccessible. The API exposes an oracle $\mathcal{O}(x) \mapsto (y, \ell)$ where $y$ is sampled text and $\ell$ is either $\emptyset$ or the top-$k$ log-probabilities ($k \le 20$ on OpenAI's Chat Completions API; $\emptyset$ on Anthropic's Messages API as of 2026).

An edit request is $e = (s, r, o^* )$ — subject, relation, new object — with a probe set $P(e)$ of natural-language queries whose correct answer becomes $o^*$. A wrapper is a map $W$ with its own parameters $\phi$ (prompt tokens, memory contents, small auxiliary model), inducing $g = W_\phi \circ \mathcal{O}$.

Measured quantities, as actually computed:

$$\text{ES} = \frac{1}{|E|}\sum_{e \in E} \mathbb{1}\!\left[g(q_e) \models o^*\right], \qquad q_e \text{ the canonical prompt}$$

$$\text{PS} = \mathbb{E}_{q \sim \text{Para}(e)} \mathbb{1}[g(q) \models o^*], \qquad \text{NS} = \mathbb{E}_{q \sim N(e)} \mathbb{1}[g(q) = f_\theta(q)]$$

where $\models$ is string/entity match (not logit comparison — the logit-difference form of efficacy used by ROME is **unavailable** when $\ell = \emptyset$), $\text{Para}$ is a paraphrase distribution, and $N(e)$ is a neighbourhood of unrelated-but-similar prompts. Multi-hop propagation is measured on composed queries $q_{e_1 \circ e_2}$ as in MQuAKE.

Cost is the quantity parametric editing does not have:

$$C(n) = \underbrace{|\text{ctx}(n)|}_{\text{prompt tokens carried per query}} + \underbrace{r \cdot |\text{ctx}_{\text{retr}}|}_{\text{retrieved evidence}} + \underbrace{m}_{\text{extra API round-trips}}$$

for $n$ accumulated edits. A wrapper is **scalable** iff $C(n) = O(\text{polylog}\, n)$ with $\text{PS}$ non-decreasing in $n$.

Assumptions, with the violated ones flagged:

1. $f_\theta$ is fixed across the measurement window — **violated**: providers rotate checkpoints and system prompts without notice, so pre/post comparisons are confounded.
2. The API is stateless given the prompt — **violated** in products with server-side memory, caching, and safety rewriting.
3. Entity-match scoring is a faithful proxy for belief change — **violated**: a model can emit $o^*$ by copying it from context while its underlying distribution is unchanged (Zheng et al., 2023, discuss exactly this "in-context override" ambiguity).
4. Probes in $N(e)$ are independent of $e$ — **violated**: neighbourhood sets are constructed by lexical similarity and leak the edited subject.

## 3. State of the Art

**Wrapper methods (systems SOTA).**
- **SERAC** (Mitchell et al., ICML 2022): a scope classifier routes in-scope inputs to a small counterfactual model, leaving the base model untouched. Establishes that non-parametric editing can beat weight editing on locality; the base model must still be run, but never modified.
- **IKE** (Zheng et al., EMNLP 2023): demonstration-based in-context editing. On COUNTERFACT with GPT-J (6B) and GPT-NeoX (20B), reports higher paraphrase and neighbourhood scores than ROME/MEMIT. **Established** for the reported models; **claimed but unablated** is that the gain survives when demonstrations cannot be selected using the target model's own logits.
- **MeLLo** (Zhong et al., EMNLP 2023, MQuAKE): decomposes a multi-hop query, retrieves edited facts, and self-checks each hop — pure prompting, works on GPT-3.5. Beats MEMIT-style weight editing on multi-hop by a large margin. This exists as **a benchmark number on MQuAKE only**, and MQuAKE's counterfactual construction has since been criticized for allowing shortcut solutions.
- **Proxy-tuning** (Liu et al., COLM 2024): shifts a large model's output distribution by adding the logit difference of a tuned and untuned small model. Reported closing ~88% of the gap between LLAMA-2 70B base and its chat version. Requires per-token logits over the **full** vocabulary — no commercial API provides this, so it is black-box in the weights sense but not in the API sense.
- **CombLM** (Ormazabal et al., EMNLP 2023) and black-box prompt optimization (**BBT**, Sun et al., ICML 2022; **RLPrompt**, Deng et al., EMNLP 2022) establish that derivative-free adaptation of an API model is possible at all, on classification-scale tasks.

**Theory SOTA.** Thin. The nearest rigorous results are on what an API *leaks*, not what it accepts: Finlayson et al. (COLM 2024) show top-$k$ logprob APIs reveal the softmax matrix rank; Carlini et al. (ICML 2024) recover the embedding-projection dimension of production models for under \$20 in queries for OpenAI's `ada`/`babbage`. No theorem bounds which behavioural edits a length-$L$ context can realize.

## 4. What Is Known

- Weight editing at scale degrades the model. MEMIT edits 10,000 facts into GPT-J (6B) with high reported efficacy, but **sequential** editing collapses: Gupta et al. (ACL Findings 2024) report gradual then catastrophic forgetting for ROME/MEMIT past the order of $10^3$ sequential edits on GPT-2-XL and Llama-2-7B. Gu et al. (2024) find general-ability degradation on unrelated benchmarks after modest edit counts. This is the strongest argument *for* wrapper editing.
- Editing does not propagate. Cohen et al. (TACL 2024, RippleEdits) show parametric edits fail on logical implications and two-hop composition, at GPT-2/GPT-J/GPT-3 scale; MQuAKE shows the same for multi-hop, with prompting-based MeLLo outperforming MEMIT on GPT-3.5.
- Localization does not determine edit site. Hase et al. (NeurIPS 2023) show causal-tracing localization is uncorrelated with where an edit succeeds — undercutting the claim that weight access is *necessary* for principled editing.
- In-context override is real but shallow: appending the counterfactual to the prompt yields high edit success and poor generalization to indirect probes across the 6B–20B range.
- Context cost is the binding constraint. At $n = 10^4$ edits of ~30 tokens, naive prompt-carrying is $3\times 10^5$ tokens per query — beyond most context windows and, at 2026 API prices, orders of magnitude above the per-query cost of an unedited call.

## 5. What Is Not Known

- **Theoretically open.** Whether there exists an edit $e$ and a model class such that no context of length $L$ (for $L$ polynomial in $|e|$) induces the edited behaviour on the full probe distribution, while a rank-one weight update does. No separation theorem, no impossibility result, in either direction.
- **Empirically open.** Whether a retrieval-based wrapper with $C(n) = O(\text{polylog}\,n)$ matches parametric editing on paraphrase and multi-hop at $n \ge 10^4$ edits on a frontier API model. Runnable now; the blocker is money and provider drift, not ideas.
- **Methodologically blocked.** Locality (NS) is not measurable under an API, because you cannot obtain $f_\theta(q)$ from the *same* checkpoint at a different time with the wrapper disabled if the wrapper is a system prompt the provider also mutates. Also blocked: distinguishing "the model believes $o^*$" from "the model copied $o^*$ from context" without logits.

## 6. Why It Is Hard

The specific obstruction is **confounded measurement compounded by non-identifiability of the base model**. Every editing metric is a *difference* between pre- and post-edit behaviour of one fixed function. Under an API, the pre-edit function is unobservable at post-edit time: checkpoints rotate, sampling is non-deterministic even at temperature 0 (batched MoE routing and non-associative float reduction), and safety layers rewrite inputs. A measured NS drop of 3 points cannot be attributed to the wrapper versus a silent model update.

Second obstruction: **the evaluation does not measure what it names**. "Edit success" scored by string match is trivially satisfied by putting $o^*$ in the context. The construct the field cares about — that the model's *knowledge* changed — is defined parametrically and has no black-box operationalization. Wrapper methods therefore score highly on the metric while the question stays open.

## 7. Current Research (as of 2026)

- **Retrieval-as-editing.** Editing recast as maintaining an external fact store with a rewriting front-end; MeLLo and its descendants (DeepEdit, PokeMQA lines) are the reference points. Active at Princeton NLP, Zhejiang/ZJUNLP (EasyEdit toolkit), UCSD.
- **Output-side steering under logit APIs.** Proxy-tuning-style logit arithmetic, contrastive decoding variants, and small "corrector" models. Limited by API logprob truncation. *(frontier — verify)* whether any provider ships a full-vocabulary logprob endpoint.
- **Agentic/self-verifying edit application** — multi-round decompose-retrieve-check loops, trading round-trips $m$ for propagation. *(frontier — verify)* the cost/benefit at $n>10^3$.
- **API leakage as a lever.** Finlayson et al. and Carlini et al. suggest partial white-box structure is recoverable from logprob APIs; whether recovered structure suffices to *target* an edit is unexplored *(frontier — verify)*.
- **Benchmark hygiene.** Re-examination of MQuAKE/COUNTERFACT for shortcut leakage; without this, black-box wins are not credible.

## 8. Concrete Next Experiment

**Question.** Does a polylog-cost retrieval wrapper match a weight edit on paraphrase and multi-hop generalization?

**Scale.** $n = 10{,}000$ edits sampled from COUNTERFACT plus the 3,000-item MQuAKE-CF set. Target: one open-weight model served *through an API shim that hides weights* (Llama-3.1-70B-Instruct), so that the parametric arm is runnable as ground truth on the identical checkpoint. Freeze the checkpoint — this removes the drift confound by construction.

**Arms.**
1. **Parametric control:** MEMIT batch-edit of all $10^4$ facts.
2. **Naive context control:** append the top-1 retrieved fact.
3. **Treatment:** retrieval wrapper with query decomposition and per-hop re-retrieval (MeLLo-style), capped at $C(n) \le 2{,}000$ prompt tokens and $m \le 4$ round-trips.
4. **Null control:** wrapper with a *random* fact retrieved — bounds how much of any gain is prompt-format artefact rather than the edit.

**Deciding number.** Multi-hop accuracy on MQuAKE-CF-3k under 3,000 simultaneous edits. If arm 3 exceeds arm 1 by $\ge 15$ absolute points while $\text{NS}$ on the COUNTERFACT neighbourhood set stays within 2 points of the unedited model, and arm 4 is within noise of unedited, wrappers dominate and the parametric framing is the wrong default. If arm 3 is within 5 points of arm 2, the decomposition machinery adds nothing and the field should stop building it.

**Cost estimate.** ~$4\times10^4$ wrapper queries at ~2.5k tokens ≈ $10^8$ tokens; single-digit thousands of dollars on 2026 open-weight inference pricing.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS, 2022. — arXiv:2202.05262
- **[Foundational]** Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, Chelsea Finn. *Memory-Based Model Editing at Scale.* ICML, 2022. — arXiv:2206.06520
- **[SOTA]** Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, Baobao Chang. *Can We Edit Factual Knowledge by In-Context Learning?* EMNLP, 2023. — arXiv:2305.12740
- **[SOTA]** Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, Danqi Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP, 2023. — arXiv:2305.14795
- **[SOTA]** Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, Noah A. Smith. *Tuning Language Models by Proxy.* COLM, 2024. — arXiv:2401.08565
- **[Method]** Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, Xipeng Qiu. *Black-Box Tuning for Language-Model-as-a-Service.* ICML, 2022. — arXiv:2201.03514
- **[Method]** Aitor Ormazabal, Mikel Artetxe, Eneko Agirre. *CombLM: Adapting Black-Box Language Models through Small Fine-Tuned Models.* EMNLP, 2023. — arXiv:2305.16876
- **[Evaluation]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[Limits]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL, 2024. — arXiv:2401.07453
- **[Limits]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS, 2023. — arXiv:2301.04213
- **[API structure]** Nicholas Carlini, Daniel Paleka, Krishnamurthy Dvijotham, et al. *Stealing Part of a Production Language Model.* ICML, 2024. — arXiv:2403.06634
- **[API structure]** Matthew Finlayson, Xiang Ren, Swabha Swayamdipta. *Logits of API-Protected LLMs Leak Proprietary Information.* COLM, 2024. — arXiv:2403.09539
- **[Survey]** Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, Jundong Li. *Knowledge Editing for Large Language Models: A Survey.* ACM Computing Surveys, 2024. — arXiv:2310.16218

## 10. Worked Example

Edit: *"The CEO of Twitter is Linda Yaccarino."* Probe the composed query **"What is the nationality of the CEO of Twitter?"** (answer under the edit: Italian-American).

**Arm 2 (naive context).** Prefix: `Fact: The CEO of Twitter is Linda Yaccarino.` — 12 tokens. Single-hop probe "Who is the CEO of Twitter?" → correct, ES $=1$. Composed probe → the model must (i) resolve the subject from context, (ii) retrieve `nationality(Yaccarino)` from parameters. On MQuAKE-class data this two-step chain is where the reported drop lives: single-hop edit success in the 90s, composed accuracy far lower for the same edit.

**Arm 4 (null control) makes the obstruction visible.** Retrieve a *random* fact instead: `Fact: The CEO of Boeing is Kelly Ortberg.` The model still answers the composed query — with the *unedited* answer, correctly formatted. Now score both arms with string match against the edited target. Arm 2 gets credit whenever $o^*$ appears anywhere in the answer; arm 4 gets credit whenever the pre-edit world happens to agree. Neither score distinguishes *belief change* from *context copying*.

**The number.** Suppose across 3,000 composed probes arm 2 scores 41% and arm 4 scores 9%. The 32-point gap is the only defensible estimate of edit effect — but it is not comparable to MEMIT's reported figure, because MEMIT's metric is a logit margin $\log p(o^*) - \log p(o)$ evaluated on identical weights, and no API returns $p(o)$ for the un-retrieved alternative. Two arms, two incompatible measurement instruments, one shared metric name. That mismatch — not the engineering of the wrapper — is what keeps the problem open.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*