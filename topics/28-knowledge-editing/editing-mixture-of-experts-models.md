---
id: 28-knowledge-editing/editing-mixture-of-experts-models
title: "Editing Mixture-of-Experts Models"
topic: 28-knowledge-editing
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Editing Mixture-of-Experts Models

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/editing-mixture-of-experts-models` · **Status:** empirically-open

## 1. Problem Statement

Locate-and-edit methods (ROME, MEMIT) write a fact into a dense MLP by solving a rank-one or low-rank least-squares update against a key covariance estimated from a general corpus. Sparse Mixture-of-Experts (MoE) models replace that MLP with $E$ experts and a router that activates $k \ll E$ of them per token. The edit target is therefore no longer a single matrix but a *routing-conditioned* set of matrices, and whether the edit fires on a new prompt depends on a discrete, non-differentiable decision the editor does not control.

Three variants, of different difficulty:

- **Measurement.** Does the standard edit suite — efficacy, paraphrase generalization, locality, ripple effects — remain valid when a failure can be caused by the router sending a paraphrase to an unedited expert rather than by the edit failing? Current benchmarks report one number and cannot separate these.
- **Method.** Given an MoE checkpoint and an edit request, produce a parameter delta that generalizes across paraphrases without degrading unrelated behaviour, at compute comparable to dense MEMIT. Sub-questions: edit one expert, all top-$k$ experts, the shared expert (DeepSeekMoE-style), or the router itself.
- **Theory.** Is the routing-conditioned key covariance $C_i = \mathbb{E}[kk^\top \mid \text{token routed to } i]$ well-conditioned enough for the MEMIT normal equations to be solvable at realistic sample budgets? This has a clean answer that depends only on $E$, $k$, hidden width $d$, and corpus size — see §10.

Solved = an MoE editor matching dense-model paraphrase generalization (≈90% on CounterFact-style paraphrase sets at 6B scale) on a 100B+-total-parameter MoE, with locality no worse than the dense baseline, and with an explicit report of route drift.

## 2. Formal Setting

**Model.** Layer $\ell$ of an MoE transformer computes
$$h^{\ell+1} = h^{\ell} + \sum_{i \in S_\ell(h^\ell)} g_i(h^\ell)\, E_i^\ell(h^\ell), \qquad S_\ell(h) = \operatorname{top-}k\big(W_r^\ell h\big),$$
with $E$ routed experts, router $W_r^\ell \in \mathbb{R}^{E \times d}$, gate weights $g_i$ from a softmax over the selected logits. Each expert is a gated MLP with input $W^{i}_{\text{in}}$ and output $W^{i}_{\text{out}} \in \mathbb{R}^{d \times d_{\text{ff}}}$.

**Edit request.** $e = (s, r, o \to o^\ast)$; prompt template $p(s,r)$; paraphrase set $P(e)$; neighbourhood set $N(e)$ of prompts about distinct subjects with the same relation.

**Measured quantities.**
- *Efficacy* $\mathrm{ES} = \mathbb{1}[\,P_{\theta'}(o^\ast \mid p) > P_{\theta'}(o \mid p)\,]$, averaged over edits.
- *Paraphrase generalization* $\mathrm{PS} = \mathbb{E}_{p' \sim P(e)}\,\mathbb{1}[P_{\theta'}(o^\ast\mid p') > P_{\theta'}(o\mid p')]$.
- *Locality* $\mathrm{NS} = \mathbb{E}_{p'' \sim N(e)}\,\mathbb{1}[P_{\theta'}(o\mid p'') > P_{\theta'}(o^\ast\mid p'')]$.
- *Routing coverage* (MoE-specific) $\;c(e) = \mathbb{E}_{p' \sim P(e)}\,\mathbb{1}[\,i^\ast \in S_\ell(h^\ell(p'))\,]$, where $i^\ast$ is the edited expert and $h^\ell$ is taken at the subject's last token. Measured by a forward pass logging router argmax sets.
- *Route drift* $\delta = 1 - \frac{1}{|T|}\sum_{t \in T}\frac{|S_\ell^{\theta'}(t) \cap S_\ell^{\theta}(t)|}{k}$ over a held-out token set $T$, computed layer-wise. Measured directly from router outputs before and after the edit.
- *Expert key covariance* $C_i = \frac{1}{|T_i|}\sum_{t \in T_i} k_t k_t^\top$ where $T_i$ is the subset of corpus tokens routed to expert $i$ and $k_t \in \mathbb{R}^{d_{\text{ff}}}$ is the expert's inner activation.

**The MEMIT update, transposed to an expert.** For target key $k_\ast$, target value $v_\ast$:
$$\Delta^i = (v_\ast - W^{i}_{\text{out}} k_\ast)\,\frac{(C_i^{-1}k_\ast)^\top}{(C_i^{-1}k_\ast)^\top k_\ast}.$$

**Assumptions, and which are violated.**
1. *The MLP is the sole write path for the fact.* Violated in MoE: the value is a gate-weighted sum over $k$ experts plus, in DeepSeekMoE/DeepSeek-V3, an always-on shared expert.
2. *$C_i$ is invertible at practical corpus size.* Violated for large-$E$, small-$k$ models — §10 gives the arithmetic.
3. *Routing is stable under the edit.* Not established; $\delta > 0$ is expected because the edit changes $h^{\ell+1}$, which is the router input at every later layer.
4. *Expert assignment is a function of semantics.* Contradicted by the Mixtral analysis, which found routing correlated with syntax/token identity more than with topic.

## 3. State of the Art

**Established (dense models only).** ROME (Meng et al., NeurIPS 2022) and MEMIT (Meng et al., ICLR 2023) are the reference locate-and-edit methods; MEMIT scales to 10,000 simultaneous edits on GPT-J 6B and GPT-NeoX 20B. MEND (Mitchell et al., ICLR 2022) and SERAC (Mitchell et al., ICML 2022) are the hypernetwork and memory-based alternatives. GRACE (Hartvigsen et al., NeurIPS 2023) and WISE (Wang et al., NeurIPS 2024) are the lifelong-editing SOTA. None of these papers report an MoE backbone.

**Claimed but unablated.** MEMoE (Wang & Li, 2024) *uses* a mixture-of-experts adapter as the editing mechanism on a dense model. This is the reverse of the present problem and is routinely miscited as evidence that MoE models are editable. No paper establishes that a MoE-adapter editor transfers to a MoE backbone.

**Benchmark-number-only.** EasyEdit (Wang et al., 2024) and the ZsRE/CounterFact leaderboards report editing scores; where MoE checkpoints (Mixtral, OLMoE, Qwen-MoE) appear at all, the numbers are single-configuration runs without a route-drift or coverage ablation, so a low score cannot be attributed between "the edit did not take" and "the paraphrase went elsewhere".

**Interpretability substrate.** Geva et al. (EMNLP 2021) established MLPs as key–value memories — the premise locate-and-edit rests on. Hase et al. (NeurIPS 2023) showed localization does not predict where an edit succeeds, which weakens the transfer of "find the expert" intuitions.

## 4. What Is Known

- **Dense baseline.** MEMIT on GPT-J 6B reaches near-ceiling efficacy with paraphrase scores in the high-80s to low-90s on CounterFact for single edits, degrading with batch size; ROME on GPT-2 XL 1.5B is similar at $n=1$.
- **Sequential editing collapses.** Gupta et al. (Findings of ACL 2024) show gradual then catastrophic forgetting as sequential edits accumulate on GPT-2 XL and Llama-2 7B — measured at hundreds to thousands of edits, dense only.
- **Ripple effects are largely unhandled.** Cohen et al. (TACL 2024) find that edits which pass efficacy fail on logically entailed consequences; success rates on composition/two-hop subsets are far below single-hop.
- **Routing is not topical.** The Mixtral 8×7B report (Jiang et al., 2024; $E=8$, $k=2$, 32 layers, 47B total / 13B active) found no clear domain specialization in expert assignment, but did find above-chance consecutive-token repetition of expert choice. OLMoE (Muennighoff et al., 2024; $E=64$, $k=8$, 6.9B/1.3B active) reports stronger domain and vocabulary specialization, so specialization is architecture-dependent, not universal.
- **Sparsity ratios are extreme at frontier scale.** DeepSeek-V3 (2024): 671B total, 37B active, 256 routed experts plus 1 shared expert per layer, $k=8$, $d=7168$. Each routed expert sees $\approx 3.1\%$ of tokens under balanced load.

## 5. What Is Not Known

- **Empirically open.** Whether any dense editor (ROME/MEMIT/MEND/GRACE/WISE) retains its dense-model paraphrase generalization on Mixtral-class or DeepSeek-class MoEs. The experiment is runnable today on open weights; nobody has published it with the required controls.
- **Empirically open.** Whether writing the edit into all $k$ activated experts, into the shared expert, or into the router beats single-expert editing. Four arms, one GPU-week each at 7B-active scale.
- **Methodologically blocked.** There is no accepted decomposition of edit failure into *write failure* vs *routing miss*. Without $c(e)$ reported alongside PS, MoE editing scores are uninterpretable. This page's §2 gives a candidate definition; it is not standard.
- **Theoretically open.** No bound relating route drift $\delta$ to the edit norm $\|\Delta\|_F$. A Lipschitz-style bound on router-logit perturbation would give a design rule ("keep $\|\Delta\|$ below the router margin") but the router margin distribution at depth is uncharacterized.

## 6. Why It Is Hard

The specific obstruction is **non-identifiability of the write target compounded by a rank-deficient covariance estimate**.

1. *Non-identifiability.* The fact's contribution at layer $\ell$ is $\sum_{i\in S} g_i E_i(h)$. Any reallocation across the $k$ active experts producing the same sum is observationally equivalent on the edit prompt but produces different behaviour on paraphrases that route differently. The edit prompt does not identify which allocation to choose; the objective is underdetermined.
2. *Rank deficiency.* MEMIT needs $C_i^{-1}$. Under balanced load each expert receives a $k/E$ fraction of tokens, so a corpus of $M$ tokens yields $Mk/E$ samples for a $d_{\text{ff}} \times d_{\text{ff}}$ covariance. At DeepSeek-V3 ratios this is a factor-of-32 sample reduction against dense, and load-balancing losses used in training deliberately keep the ratio near uniform, so no expert can be counted on to be data-rich.
3. *Discrete failure mode.* Route drift is a step function of a continuous perturbation. Small edits are safe until they cross a router margin, then flip an entire expert selection. Gradient-based editors get no signal about the approaching boundary.

"Hard because MoEs are popular" is not the obstruction; the obstruction is that the least-squares problem dense editing solves is both underdetermined and ill-conditioned in the sparse setting.

## 7. Current Research (as of 2026)

- **Expert-level attribution.** Extending causal tracing to identify which expert carries a fact, as the prerequisite for locate-and-edit. Groups working on MoE interpretability at AI2 (OLMoE) and academic interpretability labs. *(frontier — verify)*
- **Router-side editing.** Editing $W_r$ so a paraphrase family is deterministically routed to an edited expert, trading locality for coverage. *(frontier — verify)*
- **Memory-based sidestep.** SERAC/GRACE/WISE-style external memories are architecture-agnostic and avoid the routing problem entirely; the open question is whether they scale to lifelong editing without an unbounded memory. This is the pragmatic favourite.
- **Shared-expert editing.** DeepSeekMoE's always-on shared expert is a routing-free write target; whether it has the capacity to hold many edits is untested. *(frontier — verify)*
- **Benchmark work.** Adding MoE checkpoints and routing telemetry to EasyEdit-style harnesses.

## 8. Concrete Next Experiment

**Question.** Is MoE editing failure caused by the write or by the router?

**Scale.** OLMoE-1B-7B ($E=64$, $k=8$, 16 layers) and Mixtral-8×7B ($E=8$, $k=2$). 1,000 CounterFact edits, applied one at a time (not batched), 5 paraphrases and 10 neighbourhood prompts each. Roughly 200 A100-hours total — small enough for one lab.

**Arms.**
1. *Control (dense):* MEMIT on Llama-2 7B, same 1,000 edits. Establishes the reference PS.
2. *Single-expert:* MEMIT applied to the top-1 expert at the critical layer, with $C_i$ from tokens routed to that expert.
3. *All-active:* the same update distributed over all $k$ active experts, gate-weighted.
4. *Router-frozen oracle:* apply arm 2, then at evaluation time **force** the router to select $i^\ast$ on every paraphrase.

**Deciding number.** $\mathrm{PS}(\text{arm 4}) - \mathrm{PS}(\text{arm 2})$, the *routing gap*.
- Gap $> 15$ points: the failure is routing. Effort should go to router-side editing or to memory-based methods.
- Gap $< 5$ points with arm 2 also well below arm 1: the failure is the write itself — expert capacity or $C_i$ conditioning — and the fix is a better estimator, not better routing.

Report $c(e)$ and layer-wise $\delta$ for every arm. Without them the result is not interpretable.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[Foundational]** Noam Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR 2017. — arXiv:1701.06538
- **[Foundational]** Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy. *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP 2021. — arXiv:2012.14913
- **[SOTA]** Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, Christopher D. Manning. *Fast Model Editing at Scale.* ICLR 2022. — arXiv:2110.11309
- **[SOTA]** Thomas Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi. *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors.* NeurIPS 2023. — arXiv:2211.11031
- **[SOTA]** Peng Wang et al. *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models.* NeurIPS 2024. — arXiv:2405.14768
- **[Architecture]** Albert Q. Jiang et al. *Mixtral of Experts.* 2024. — arXiv:2401.04088
- **[Architecture]** Damai Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL 2024. — arXiv:2401.06066
- **[Architecture]** Niklas Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models.* 2024. — arXiv:2409.02060
- **[Architecture]** William Fedus, Barret Zoph, Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* JMLR 2022. — arXiv:2101.03961
- **[Critique]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023. — arXiv:2301.04213
- **[Critique]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL 2024. — arXiv:2401.07453
- **[Critique]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024. — arXiv:2307.12976
- **[Survey]** Yunzhi Yao et al. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. — arXiv:2305.13172
- **[Adjacent, often miscited]** Renzhi Wang, Piji Li. *MEMoE: Enhancing Model Editing with Mixture of Experts Adaptors.* 2024. — MoE *as editor*, dense backbone; not evidence about editing MoE models.

## 10. Worked Example

**The covariance budget, DeepSeek-V3 geometry.** MEMIT estimates $C = \mathbb{E}[kk^\top]$ from a Wikipedia sample — the released implementation uses on the order of $10^5$ token samples per layer. Take $M = 100{,}000$, $E = 256$, $k = 8$, $d = 7168$.

Balanced-load token count reaching a given expert:
$$M_i = M \cdot \frac{k}{E} = 100{,}000 \times \frac{8}{256} = 3{,}125.$$

The sample covariance $C_i = \frac{1}{M_i}\sum k_t k_t^\top$ has rank at most $\min(M_i, d_{\text{ff}})$. Even taking the conservative case $d_{\text{ff}} = d = 7168$:
$$\operatorname{rank}(C_i) \le 3{,}125 < 7{,}168.$$

$C_i$ is singular. $C_i^{-1}k_\ast$ in the MEMIT update does not exist. The minimum corpus for a full-rank estimate is
$$M_{\min} = d \cdot \frac{E}{k} = 7168 \times 32 \approx 229{,}000 \text{ tokens},$$
and a *well-conditioned* estimate needs roughly an order of magnitude more, $\sim 2.3\times 10^6$ tokens routed through that layer — per expert, per layer, and re-collected whenever the router changes. Mixtral is milder: $E/k = 4$, $M_i = 25{,}000$ against $d = 4096$, full rank but conditioned on a quarter of the dense sample.

**The routing ceiling, Mixtral geometry.** Suppose the subject's last-token representation at layer 15 routes to $S = \{3, 6\}$ on the canonical prompt, and the edit is written into expert 3 alone. If across the paraphrase set expert 3 is re-selected on 12 of 20 paraphrases, then $c(e) = 0.60$ and
$$\mathrm{PS} \le c(e) + (1 - c(e))\cdot \mathrm{PS}_{\text{base}} \approx 0.60 + 0.40 \times 0.05 = 0.62,$$
against a dense GPT-J MEMIT reference near $0.90$. The edit is perfect on every prompt where it fires and invisible on the rest.

**What the example makes visible.** Both failures produce the same headline number — a paraphrase score around 0.6 — and the standard benchmark cannot tell them apart. One is a linear-algebra problem fixed by more data or a shrinkage estimator; the other is a discrete routing problem that more data will never fix. That is why §8's arm 4 (router-frozen oracle) is the load-bearing control: it is the only cheap way to separate the two.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*