---
id: 28-knowledge-editing/continual-pretraining-versus-editing
title: "Continual Pretraining Versus Editing Tradeoff"
topic: 28-knowledge-editing
status: empirically-open
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Continual Pretraining Versus Editing Tradeoff

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/continual-pretraining-versus-editing` · **Status:** empirically-open

## 1. Problem Statement

A deployed model must absorb $m$ new facts arriving over time. Two families of update exist:

- **Editing** — a targeted parameter or memory intervention (ROME, MEMIT, MEND, GRACE, LoRA patches) costing $O(m)$ small operations, no new corpus.
- **Continual pretraining (CPT)** — resume the pretraining objective on a fresh token stream containing the new facts, with replay of old data.

The problem: **for a given update budget, which family gives more retained, generalizing knowledge per FLOP, and where is the crossover in $m$?**

Three variants, different difficulty:

- **Measurement.** Define a single quantity on which an edit of 100 facts and a 10B-token CPT run are comparable. No accepted such quantity exists; editing is scored on rewrite/paraphrase/locality triples, CPT on perplexity and downstream benchmarks.
- **Method.** Build a hybrid updater that dominates both arms across $m \in [1, 10^6]$.
- **Theory.** Prove a bound relating the number of facts durably installable to parameters touched, tokens spent, or the rank of the update.

Solved would mean: a published crossover curve $m^\star(N, C)$ — the fact count above which CPT beats editing at equal compute — reproduced at two model scales with a shared metric.

## 2. Formal Setting

Model $f_\theta$, $\theta \in \mathbb{R}^N$. Fact set $\mathcal{E} = \{(s_i, r_i, o_i^{\text{old}} \to o_i^{\text{new}})\}_{i=1}^m$. Pretraining corpus $\mathcal{D}_{\text{old}}$; update corpus $\mathcal{D}_{\text{new}}$ of $D$ tokens expressing $\mathcal{E}$.

**Efficacy** (measured as top-1 greedy decode on a held-out prompt template *not* used to construct the update):
$$\mathrm{Eff} = \frac{1}{m}\sum_{i=1}^m \mathbb{1}\!\left[\arg\max_o p_{\theta'}(o \mid s_i, r_i) = o_i^{\text{new}}\right].$$

**Generalization** $\mathrm{Gen}$: the same indicator averaged over $k \ge 5$ paraphrases plus multi-hop compositions $(s_i, r_i, r')$ — the ripple-effect set of Cohen et al. (2024).

**Locality / drift**: on a control set $\mathcal{C}$ of unrelated prompts,
$$\Delta_{\text{loc}} = \mathbb{E}_{x \sim \mathcal{C}}\left[\mathrm{KL}\big(p_\theta(\cdot\mid x)\,\|\,p_{\theta'}(\cdot\mid x)\big)\right],$$
measured token-averaged over $\ge 10^5$ tokens of held-out pretraining data, plus a general-ability panel (MMLU, GSM8K, HellaSwag) reported as absolute point delta.

**Compute**, the common currency. CPT: $C_{\text{CPT}} \approx 6 N D_{\text{eff}}$ where $D_{\text{eff}} = D(1+\rho)$ and $\rho$ is the replay fraction. Editing: $C_{\text{edit}} = m \cdot c_{\text{edit}}$, where $c_{\text{edit}}$ counts forward/backward passes *plus* the one-time covariance estimation $C_{\text{cov}}$ used by ROME/MEMIT (typically $10^5$–$10^6$ Wikipedia tokens of forward passes), which is routinely excluded from reported costs.

**Objective.** With a durability horizon of $T$ subsequent updates,
$$U(\text{method}) = \frac{\mathrm{Gen}^{(T)} - \lambda\,\Delta^{(T)}_{\text{loc}}}{C/N},\qquad m^\star = \min\{m : U(\text{CPT}) > U(\text{edit})\}.$$

**Assumptions, and which are violated.**

1. *Facts are independent atoms.* Violated: entailment means one edit should propagate; ripple-effect benchmarks show it does not.
2. *$\Delta_{\text{loc}} \approx 0$ implies no damage.* Violated: sequential editing degrades abilities that unrelated-prompt KL does not detect (Gu et al., 2024).
3. *Edit cost is $m$-independent.* Violated: MEMIT batches amortize; sequential single edits do not, and error compounds.
4. *$\mathcal{D}_{\text{new}}$ contains each fact at frequency sufficient for acquisition.* Violated: long-tail facts need many co-occurrences (Kandpal et al., 2023), so CPT on one mention of a fact often fails to install it at all.

## 3. State of the Art

**Editing, established.** MEMIT (Meng et al., ICLR 2023) installs ~10,000 edits into GPT-J (6B) and GPT-NeoX (20B) with high rewrite success — this is established and independently reproduced. ROME's rank-one causal-tracing-derived update (Meng et al., NeurIPS 2022) is established as a method; the *localization inference* behind it is not — Hase et al. (NeurIPS 2023) showed edit success is largely independent of the layer causal tracing identifies.

**Editing, claimed but unablated.** That large-$m$ editing is a substitute for retraining. Gupta et al. (Findings of ACL 2024) report gradual then catastrophic forgetting under sequential editing; the counterclaims that batched MEMIT avoids this are benchmark numbers on CounterFact/zsRE, not ablations against a compute-matched CPT arm.

**CPT, established.** Ibrahim et al. (TMLR 2024) show that LR re-warming + re-decaying + ~5% replay of the original corpus matches full retraining on the union of old and new data, at 405M and 10B scale, across a distribution shift (English → German, and Pile → SlimPajama). This is the strongest compute-matched result in either family.

**Neither arm has been run against the other at equal FLOPs on a shared metric.** Ovadia et al. (2023) compare fine-tuning with retrieval and find retrieval wins on knowledge injection, but do not include parameter editing or a replay-tuned CPT arm. That comparison is the gap this page names.

## 4. What Is Known

- **Capacity.** Allen-Zhu & Li (2024, *Physics of Language Models 3.3*) measure ~2 bits of factual knowledge per parameter after sufficient exposure, across models from 10M to 1B parameters. A 7B model therefore has room for $\sim 10^{10}$ bits; capacity is not the binding constraint for $m \sim 10^4$.
- **Exposure.** Kandpal et al. (ICML 2023) find QA accuracy scales roughly log-linearly with the number of pretraining documents mentioning the entity; models up to 176B (BLOOM) still fail on facts seen few times. Single-mention CPT is unreliable.
- **Replay dose.** 1% replay is insufficient, 5% roughly suffices at 10B scale (Ibrahim et al., TMLR 2024). This is the only well-calibrated hyperparameter in the whole comparison.
- **Editing degradation.** Gu et al. (2024) show measurable drops on general-ability benchmarks after even a few dozen ROME/MEMIT edits on GPT-2 XL and LLaMA-1 7B, with locality metrics still reporting success.
- **Ripple failure.** Cohen et al. (TACL 2024) report that edited models answer the direct query correctly but fail two-hop consequences at high rates on RIPPLEEDITS — GPT-J/GPT-2 XL scale.
- **Non-parametric alternative.** SERAC (Mitchell et al., ICML 2022) and GRACE (Hartvigsen et al., NeurIPS 2023) achieve high edit retention by *not* touching $\theta$, which sidesteps the tradeoff rather than resolving it.

## 5. What Is Not Known

- **Empirically open (primary).** The crossover $m^\star$. Nobody has run compute-matched editing versus CPT arms on the same fact set, same model, same evaluation. The experiment is runnable today on a 7B model for well under $10^{21}$ FLOPs; it has not been run.
- **Empirically open.** Durability of edits under *subsequent* legitimate training. Whether an edit survives a later CPT pass, an RLHF stage, or quantization is unmeasured at scale.
- **Methodologically blocked.** A single utility comparable across arms. Editing metrics presuppose a fact list; CPT metrics presuppose a corpus. Until $\mathrm{Gen}$ and $\Delta_{\text{loc}}$ are defined on the same held-out set, "which wins" is not a well-posed question.
- **Theoretically open.** Any bound of the form: a rank-$r$ update to $L$ layers can durably install at most $g(r, L, N)$ mutually-consistent facts without exceeding drift $\epsilon$. No proof either way; no non-trivial lower bound on edits required per fact.

## 6. Why It Is Hard

The obstruction is **confounded measurement compounded by asymmetric cost accounting**.

- Edit success is scored on prompts derived from the same template used to construct the edit. This measures template-matched retrieval, not knowledge — an evaluation that does not measure the thing it names. CPT is scored on perplexity, which is dominated by fluency and barely moves for $m = 10^3$ new facts inside $10^9$ tokens.
- Reported editing cost omits $C_{\text{cov}}$ and omits the search over layer/hyperparameters. Reported CPT cost omits the data-curation pass that isolates $\mathcal{D}_{\text{new}}$. Both omissions run in the direction that flatters the reporting authors' method.
- Non-identifiability: a model that answers correctly after CPT may have acquired the fact, or memorized the update corpus's surface form. Distinguishing these needs held-out paraphrases written by someone who has not seen $\mathcal{D}_{\text{new}}$ — expensive and rarely done.

## 7. Current Research (as of 2026)

- **Batched and lifelong editing** — EasyEdit/KnowEdit ecosystem (Zhejiang University group: Zhang, Yao, Chen, Zhang) continues to standardize editing evaluation; the benchmark still lacks a CPT arm.
- **Replay-scheduling for CPT** — Mila/EleutherAI-adjacent work following Ibrahim et al.; extension to instruction-tuned checkpoints is active *(frontier — verify)*.
- **Memory-layer and retrieval-hybrid updates** — treating updates as writes to an explicit key-value store rather than to $\theta$ (descendants of SERAC and GRACE). Reported as strictly better on durability, weaker on multi-hop *(frontier — verify)*.
- **Edit durability under later training** — small literature; the "edits are erased by subsequent fine-tuning" claim circulates but I know of no compute-matched published measurement *(frontier — verify)*.

## 8. Concrete Next Experiment

**Scale.** One 7B open-weights base model (Llama-3-8B or Qwen2.5-7B), one $m = 2{,}000$ fact set drawn from post-cutoff Wikidata diffs (so no leakage), one shared evaluation harness.

**Arms, matched at $C = 3\times10^{19}$ FLOPs** (≈ 600M tokens at $6ND$ for $N=7\text{B}$):

1. **Edit arm.** MEMIT, batched, all 2,000 facts. Charge $C_{\text{cov}}$ and hyperparameter search into the budget; spend any FLOP surplus on more covariance data.
2. **CPT arm.** 30M tokens of synthetic multi-paraphrase text covering the 2,000 facts (≈ 15 mentions each, per Kandpal's exposure finding), plus 570M tokens of replay from a Pile/SlimPajama proxy, LR re-warm to $3\times10^{-5}$ and cosine re-decay.
3. **Control arm (required).** Identical CPT recipe on 600M tokens of pure replay, no new facts. This isolates how much of any general-ability change is CPT itself rather than the update.
4. **Retrieval reference.** Frozen base model + oracle retrieval of the 2,000 facts, $C \approx 0$ training FLOPs.

**Evaluation.** $\mathrm{Gen}$ on 5 held-out paraphrases plus 2-hop compositions written before either arm runs; $\Delta_{\text{loc}}$ as token-averaged KL on 200k held-out tokens; MMLU/GSM8K/HellaSwag absolute deltas relative to arm 3.

**The deciding number.** $\mathrm{Gen}$ on the **2-hop composition set**, with the constraint $|\Delta\text{MMLU}| \le 1.0$ point versus the control arm. If the edit arm reaches within 5 points of the CPT arm's 2-hop $\mathrm{Gen}$ under that drift constraint, editing dominates at $m = 2{,}000$ and $m^\star > 2{,}000$. If it falls more than 5 points short, $m^\star < 2{,}000$ and the field's operating assumption — that editing scales to thousands of facts — is wrong at the only scale that matters.

## 9. Key References

- **[Foundational]** Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[SOTA — editing]** Meng, Sharma, Andonian, Belinkov, Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[SOTA — CPT]** Ibrahim, Thérien, Gupta, Richter, Anthony, Lesort, Belilovsky, Rish. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR, 2024. — arXiv:2403.08763
- **[Foundational]** Mitchell, Lin, Bosselut, Finn, Manning. *Fast Model Editing at Scale.* ICLR 2022. — arXiv:2110.11309
- **[Foundational]** Mitchell, Lin, Bosselut, Manning, Finn. *Memory-Based Model Editing at Scale.* ICML 2022. — arXiv:2206.06520
- **[Result]** Hartvigsen, Sankaranarayanan, Palangi, Kim, Ghassemi. *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors.* NeurIPS 2023. — arXiv:2211.11031
- **[Result]** Gupta, Rao, Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* Findings of ACL 2024. — arXiv:2401.07453
- **[Result]** Gu, Xu, Ma, Lu, Ling, Chang, Peng. *Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue.* EMNLP 2024. — arXiv:2401.04700
- **[Result]** Cohen, Biran, Yoran, Globerson, Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL, 2024. — arXiv:2307.12976
- **[Result]** Hase, Bansal, Kim, Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing.* NeurIPS 2023. — arXiv:2301.04213
- **[Result]** Kandpal, Deng, Roberts, Wallace, Raffel. *Large Language Models Struggle to Learn Long-Tail Knowledge.* ICML 2023. — arXiv:2211.08411
- **[Result]** Allen-Zhu, Li. *Physics of Language Models 3.3: Knowledge Capacity Scaling Laws.* 2024. — arXiv:2404.05405
- **[Comparison]** Ovadia, Brief, Mishaeli, Elisha. *Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs.* EMNLP 2024. — arXiv:2312.05934
- **[Survey]** Yao, Wang, Tian, Cheng, Wang, Zhang, Ni, Chen. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. — arXiv:2305.13172
- **[Benchmark]** Jang, Ye, Yang, Shin, Han, Kim, Choi, Seo. *Towards Continual Knowledge Learning of Language Models.* ICLR 2022. — arXiv:2110.03215

## 10. Worked Example

Update a 7B model with **one** fact: the 2026 holder of an office changes from $o^{\text{old}}$ to $o^{\text{new}}$.

**Edit arm.** ROME on layer 5 MLP: a rank-one update, $\sim 2\times$ hidden-dim $\times$ 4·hidden-dim $\approx 1.2\times10^8$ changed weight entries out of $7\times10^9$ (1.7%). Optimization is ~25 gradient steps on one prompt: $\approx 25 \times 6 \times 7\times10^9 \times 30 \text{ tokens} \approx 3\times10^{13}$ FLOPs. But the second-moment statistic $C = \mathbb{E}[kk^\top]$ needs a forward pass over $\sim10^5$ Wikipedia tokens: $2 \times 7\times10^9 \times 10^5 = 1.4\times10^{15}$ FLOPs — **47× the edit itself**, and normally reported as free because it is amortized over a batch. At $m=1$ it is not amortized.

**CPT arm.** To hit ~15 mentions, write 15 paraphrases ≈ 600 tokens; add 5% replay floor, which at any sane batch size means at least ~1M tokens of replay to avoid an LR-warmup shock: $6 \times 7\times10^9 \times 10^6 = 4.2\times10^{16}$ FLOPs — **30× the edit arm including covariance**.

**Where the obstruction becomes visible.** Both arms now answer "Who holds office $X$?" with $o^{\text{new}}$; $\mathrm{Eff}=1$ for both. Ask the 2-hop question — "Which party does the current holder of $X$ belong to?" — and the edited model typically still answers with $o^{\text{old}}$'s party, because the rank-one write installed a key→value association, not a revision of the entity graph (this is the RIPPLEEDITS failure mode). The CPT model may answer correctly *if* the paraphrase set happened to state the party, and otherwise fails identically.

So at $m=1$: editing is 30× cheaper and equally good on the metric everyone reports, and both fail the metric nobody reports. The reported win comes from the cheap metric. Scale to $m=2{,}000$ and the CPT arm's cost is nearly flat (replay dominates, $4.2\times10^{16} \to \sim10^{17}$) while the edit arm's grows roughly linearly to $\sim 6\times10^{16}$ plus compounding drift. The crossover is plausibly in the low thousands — which is exactly the regime nobody has measured.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*