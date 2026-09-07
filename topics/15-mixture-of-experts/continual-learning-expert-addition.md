---
id: 15-mixture-of-experts/continual-learning-expert-addition
title: "Continual Learning by Expert Addition"
topic: 15-mixture-of-experts
status: open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continual Learning by Expert Addition

> **Topic:** Mixture of Experts · **ID:** `15-mixture-of-experts/continual-learning-expert-addition` · **Status:** open

## 1. Problem Statement

A sparse Mixture-of-Experts (MoE) model offers an apparently clean route to continual learning: when a new data distribution arrives, add new experts, train only those, freeze everything else, and let the router send old inputs to old experts. Old parameters never move, so catastrophic forgetting should be structurally impossible.

It is not. Three variants, of different difficulty:

- **Method variant.** Given a pretrained MoE $M_0$ and a stream of distributions $D_1,\dots,D_T$, produce $M_T$ by adding $k_t$ experts per task such that (i) loss on $D_{<t}$ does not degrade, (ii) loss on $D_t$ matches a from-scratch specialist, (iii) total parameters grow sublinearly in $T$. No published method achieves all three at LLM scale.
- **Measurement variant.** Frozen weights do not imply frozen behaviour: the router is retrained (or must be), so old tokens get routed to new experts. Distinguishing *forgetting by weight drift* from *forgetting by routing drift* is not standard practice in reported results.
- **Theory variant.** Under what conditions on distribution overlap does a router provably stop allocating new experts — i.e. when does capacity growth terminate — and what is the regret against an oracle that knows the task boundaries?

Solving it means: a growth rule with a stopping criterion, a forgetting bound that survives router retraining, and a demonstration that adding experts beats replay at matched *total* FLOPs and matched *inference* FLOPs.

## 2. Formal Setting

Let the MoE layer have expert set $E_t = \{f_1,\dots,f_{N_t}\}$ and router $g_{\theta_t}: \mathbb{R}^d \to \Delta^{N_t-1}$. Top-$k$ routing gives

$$y(x) = \sum_{i \in \mathrm{TopK}(g_{\theta_t}(x))} \frac{g_{\theta_t}(x)_i}{\sum_{j \in \mathrm{TopK}} g_{\theta_t}(x)_j} f_i(x).$$

After task $t$, define quantities **as measured**:

- **Forgetting** $F_t = \frac{1}{t-1}\sum_{s<t}\big(\mathcal{L}_{D_s}(M_t) - \mathcal{L}_{D_s}(M_s)\big)$, with $\mathcal{L}$ the held-out token-level cross-entropy in nats, on a *fixed* validation shard drawn before any training (not resampled per task).
- **Routing drift** $R_t = \frac{1}{|X_s|}\sum_{x \in X_s} \mathbb{1}[\mathrm{TopK}_t(x) \neq \mathrm{TopK}_s(x)]$ — fraction of old-task tokens whose selected expert set changed. Measured per layer; layer-averaged numbers hide the fact that drift concentrates in later layers.
- **Weight-drift forgetting** $F_t^{\text{w}}$: forgetting measured with the router *rolled back* to $\theta_s$ and new experts masked out. If experts are truly frozen, $F_t^{\text{w}} = 0$ exactly. Then $F_t - F_t^{\text{w}}$ is forgetting attributable to routing alone. Almost no paper reports this decomposition.
- **Growth** $G_T = N_T/N_0$; **active-parameter growth** is separate, since top-$k$ keeps inference FLOPs flat while memory grows linearly.

Assumptions usually made, and their status:

1. *Task boundaries are known.* Violated in continual pretraining, where distribution shift is gradual (web crawl over time).
2. *Distributions are disjoint enough that a router can separate them.* Violated: code and math share tokens; measured expert-domain mutual information in decoder MoEs is low, with routing correlating more with token identity than with domain (Jiang et al., Mixtral, 2024).
3. *Freezing an expert freezes its function.* Violated whenever the shared attention blocks, layer norms, or the embedding are trainable — the expert's *inputs* change even if its weights do not.
4. *Adding capacity is free at inference.* False for memory-bound serving: $N_T$ experts must be resident or paged.

## 3. State of the Art

**Established.**

- *Sparse upcycling* (Komatsuzaki et al., ICLR 2023): a dense checkpoint can be converted into an MoE by copying the FFN into $N$ experts and continuing training; this beats both the dense checkpoint and a from-scratch MoE within a bounded additional-compute budget, on T5 and ViT. This is the mechanism most expert-addition work stands on, and it is ablated.
- *Branch-Train-Merge* (Li et al., 2022) and *Branch-Train-MiX* (Sukhbaatar et al., 2024): train domain experts fully in parallel from a common seed (Llama-2 7B in BTX), then combine into one MoE and finetune the router. Embarrassingly parallel expert training works; it is the clearest existence proof that experts can be added after the fact.
- *Lifelong-MoE* (Chen et al., ICML 2023): expand experts per pretraining stage, freeze old experts and their gating dimensions, add output-level regularization. Reports reduced forgetting versus dense continual pretraining at up to ~1B-scale models.

**Claimed but unablated.** Most continual-MoE papers report end-of-stream average accuracy without (a) the routing-drift decomposition of §2, (b) a matched-total-FLOPs replay baseline, or (c) a matched-memory baseline. Claims that "freezing experts eliminates forgetting" are typically supported by a benchmark number, not by $F_t^{\text{w}}$.

**Benchmark-only.** Class-incremental results — e.g. SEED (Rypeść et al., ICLR 2024), which selectively trains one expert per task — exist as CIFAR-100/ImageNet-Subset numbers with small backbones. Whether the ranking transfers to LLM continual pretraining is untested.

**Theory SOTA.** Li, Jian, Wang et al., *Theory on Mixture-of-Experts in Continual Learning* (ICLR 2025), analyze an overparameterized linear MoE in a continual stream and show, under cluster-separation assumptions, that the router converges, that expert exploration *terminates* after a finite number of tasks, and that this yields vanishing forgetting. This is the only rigorous termination result; its assumptions (linear experts, well-separated clusters, no shared trunk) are exactly the ones violated in §2.

## 4. What Is Known

- Dense continual pretraining forgetting is largely mitigable *without* new experts: Ibrahim et al. (2024) show LR re-warming plus ~5% replay of the original mixture recovers near-joint-training performance across a 300B-token English→German shift at 405M and 10B scale. This is the baseline expert addition must beat, and it is strong.
- EWC-style regularization (Kirkpatrick et al., PNAS 2017) reduces but does not eliminate forgetting, and degrades as $T$ grows — the parameter-isolation motivation for expert addition.
- Expert specialization in decoder LLM MoEs is weak by domain: Mixtral 8x7B (47B total, 13B active) shows no clear topic-to-expert assignment, with routing showing positional/syntactic structure instead. DeepSeekMoE (Dai et al., ACL 2024) improves specialization by fine-graining experts (64+ small experts, shared always-on experts), measured as larger degradation when a specialist expert is ablated.
- Fine-grained + shared-expert designs measurably raise expert redundancy resistance; DeepSeekMoE 16B reports comparable quality to LLaMA2 7B with ~40% of the compute.
- Router load-balancing losses actively fight specialization: they push assignment toward uniform, which is the opposite of the per-task partition continual learning wants. This tension is documented, not resolved.

## 5. What Is Not Known

- **Theoretically open.** Whether expert-exploration termination survives (i) nonlinear experts, (ii) a shared trainable trunk, (iii) overlapping (non-clustered) distributions. No forgetting bound exists that accounts for router drift under a growing expert set.
- **Empirically open.** Whether expert addition beats replay at matched total training FLOPs and matched serving memory, at $\geq$7B scale over $\geq$5 sequential domains. Every ingredient exists; nobody has published the head-to-head. This is the central gap.
- **Methodologically blocked.** "Forgetting" in MoE is not well defined until the routing/weight decomposition is standard. Two systems with identical $F_t$ can have $R_t = 0.02$ and $R_t = 0.6$; only the first is actually parameter-isolated. Also blocked: growth stopping criteria, since there is no accepted measure of "this distribution is already covered by existing experts".

## 6. Why It Is Hard

Two specific obstructions.

1. **Confounded measurement.** The router must be trained on new data to use new experts, which necessarily changes routing for old data. So the one property that motivates the method — frozen weights ⇒ no forgetting — is broken by the mechanism that makes new capacity reachable. Reported forgetting therefore mixes two causes that require different fixes (weight drift → freezing; routing drift → router replay or frozen gate dimensions), and the literature does not separate them.
2. **Non-identifiability of "when to add".** Given a new batch, deciding whether existing experts already cover it requires a coverage statistic. Loss is not one: high loss can mean novel distribution *or* underfit-but-covered. Without ground truth for "novelty", growth rules are tuned against the final benchmark, which is exactly the leak that makes continual-learning results non-transferable.

Compute cost is secondary but real: the decisive experiment is a $\geq$7B model over $\geq$5 domains with 3–4 arms, on the order of $10^{22}$–$10^{23}$ FLOPs.

## 7. Current Research (as of 2026)

- **Parameter-efficient expert addition.** LoRA-based experts added per task with frozen base (LoRAMoE, Dou et al., ACL 2024) — cheap growth, but the router still drifts.
- **Language/domain extension.** Adding experts to extend an LLM to new languages while freezing the original, with router-level regularization (MoE-LPR line of work, 2024–2025) — reported to preserve original-language ability. *(frontier — verify the matched-FLOPs baseline.)*
- **Upcycling at frontier scale.** Dense→MoE conversion is now standard practice inside major labs; the continual variant (repeated upcycling across pretraining stages) is being explored. *(frontier — verify.)*
- **Theory.** Follow-ups to the ICLR 2025 MoE-continual-learning analysis, relaxing separation assumptions. *(frontier — verify.)*
- **Fine-grained/shared-expert architectures** (DeepSeek line) as a substrate where addition is less disruptive, because each expert carries less of the function.

## 8. Concrete Next Experiment

**Scale.** Seed: an 8-expert, top-2 MoE upcycled from a 7B dense model. Stream: 5 domains (code, math, biomedical, legal, multilingual), 30B tokens each, held-out shard fixed in advance.

**Arms** (all at matched *total* training FLOPs):
- **A (control-1, replay):** dense continual pretraining, LR re-warm + 5% replay — the Ibrahim et al. recipe.
- **B (control-2, joint):** train on the union of all 5 domains — the upper bound.
- **C (expert addition):** +2 experts per domain (8→18), old experts and their gate rows frozen, router trained on new data only.
- **D (C + router replay):** as C, but 5% of old-domain tokens replayed *through the router only*, expert weights still frozen.

**Report** $F_5$, $F_5^{\text{w}}$, $R_5$ per layer, and serving memory.

**Deciding number.** $F_5$ for arm C or D, in nats, on the fixed old-domain shards, against arm A at equal total FLOPs. If $\min(F_C, F_D) \geq F_A$ while $G_5 = 2.25$ — i.e. addition costs 2.25× memory and buys no less forgetting — expert addition is not a forgetting fix and should be reframed as a *capacity* method. Secondary: if $F_C^{\text{w}} \approx 0$ but $F_C \gg 0$, forgetting is entirely routing drift, and the field's freezing arguments are measuring the wrong thing.

## 9. Key References

- **[Foundational]** Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR, 2017. — arXiv:1701.06538
- **[Foundational]** Rusu et al. *Progressive Neural Networks.* 2016. — arXiv:1606.04671
- **[Foundational]** Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017. — arXiv:1612.00796
- **[Foundational]** Yoon et al. *Lifelong Learning with Dynamically Expandable Networks.* ICLR, 2018. — arXiv:1708.01547
- **[SOTA]** Komatsuzaki et al. *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints.* ICLR, 2023. — arXiv:2212.05055
- **[SOTA]** Chen et al. *Lifelong Language Pretraining with Distribution-Specialized Experts.* ICML, 2023. — arXiv:2305.12281
- **[SOTA]** Sukhbaatar et al. *Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM.* 2024. — arXiv:2403.07816
- **[SOTA]** Li et al. *Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models.* 2022. — arXiv:2208.03306
- **[SOTA]** Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.* ACL, 2024. — arXiv:2401.06066
- **[Theory]** Li, Jian, Wang et al. *Theory on Mixture-of-Experts in Continual Learning.* ICLR, 2025. — arXiv:2406.16437
- **[Baseline]** Ibrahim et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024. — arXiv:2403.08763
- **[Survey]** Wang et al. *A Comprehensive Survey of Continual Learning: Theory, Method and Application.* IEEE TPAMI, 2024. — arXiv:2302.00487
- **[Survey]** De Lange et al. *A Continual Learning Survey: Defying Forgetting in Classification Tasks.* IEEE TPAMI, 2022.

## 10. Worked Example

Take an 8-expert top-2 MoE, 32 layers, and one added domain (code). Add 2 experts, freeze experts 1–8 and their gate rows, train only the two new gate rows and the two new experts on 30B code tokens.

The freezing argument says old behaviour is preserved. Check it. The router logit vector for an old token $x$ was $z \in \mathbb{R}^8$; it is now $[z, z_9, z_{10}]$ with $z_9, z_{10}$ trained. Top-2 over the old experts is preserved only if

$$\max(z_9, z_{10}) < z_{(2)},$$

where $z_{(2)}$ is the second-largest old logit. Nothing in the training objective enforces this — the new gate rows are optimized to win on code tokens, and English and code share a large token overlap (identifiers, digits, punctuation, whitespace).

Concretely: suppose on held-out English, 12% of tokens have a new expert in the top-2 at layer 24. Softmax renormalization means those tokens get a *fraction* of their output from an expert that never saw English. Even a modest per-token perturbation compounds over 32 layers. A plausible measured outcome: $F^{\text{w}} = 0.000$ nats (freezing works exactly, as promised) but $F = 0.06$ nats on the English shard — a ~6% perplexity regression — with $R = 0.12$.

The obstruction is now visible. The forgetting is 100% routing-attributable, so every parameter-isolation argument in the literature is satisfied and the model still forgot. And the fix is not more freezing — it is replaying old data through the router, which reintroduces exactly the replay buffer expert addition was supposed to make unnecessary. Meanwhile memory grew 1.25× for one domain; at 5 domains, 2.25×, while an 8-expert dense-replay baseline grew 1.0×. The comparison that decides the method is memory-versus-forgetting at matched FLOPs, and it has not been published.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*