---
id: 28-knowledge-editing/edit-locality-metric-definition
title: "Definition of Edit Locality Metric"
topic: 28-knowledge-editing
status: methodologically-blocked
first_added: 2026-09
last_reviewed: 2026-09
last_substantive_update: 2026-09
stale_since: ""
provenance: synthesized
---

# Definition of Edit Locality Metric

> **Topic:** Knowledge Editing & Model Updating · **ID:** `28-knowledge-editing/edit-locality-metric-definition` · **Status:** methodologically-blocked

## 1. Problem Statement

A knowledge edit changes a model's answer to one fact. **Locality** (also called specificity, neighborhood success, or drawdown) is meant to measure that nothing *else* changed. No accepted definition exists.

The difficulty is that "nothing else" is wrong as stated. Some other behaviour *should* change: if the model now believes the Eiffel Tower is in Rome, it should also say you can see the Colosseum from its top. So a locality metric needs a partition of the input space into *in-scope* (must change, consistently) and *out-of-scope* (must not change), and that partition is what nobody has defined.

Three variants, with very different difficulty:

- **Measurement.** Given an edit and a pair $(\theta, \theta')$, produce a scalar that is high iff out-of-scope behaviour is preserved. Blocked: requires the scope partition.
- **Method.** Given *any* fixed locality metric, build an editor that maximises it subject to edit success. Tractable; solved to a usable degree by projection-constrained editors.
- **Theory.** Prove that no single scalar can order editors consistently across scope definitions, or exhibit a scope definition that is canonical. Open.

Solving it means: a locality metric $L$ with (i) an explicit, defensible scope predicate, (ii) rank-stability — editors ranked by $L$ on one prompt distribution keep their order on another, and (iii) predictive validity — $L$ predicts downstream degradation after $n$ sequential edits.

## 2. Formal Setting

Let $f_\theta: \mathcal{X} \to \Delta(\mathcal{V})$ be an autoregressive LM mapping a prompt to a next-token distribution over vocabulary $\mathcal{V}$. An edit request is $e = (x_e, y_e^{\text{old}}, y_e^{\text{new}})$; an editor $E$ produces $\theta' = E(\theta, e)$.

Every published locality metric is an instance of one template:

$$L(\theta,\theta';S,\mathcal{D},d) \;=\; \mathbb{E}_{x\sim\mathcal{D}}\Big[\big(1 - S(x)\big)\cdot d\big(f_\theta(x),\, f_{\theta'}(x)\big)\Big]$$

with three free choices, none canonical.

- **Scope predicate** $S:\mathcal{X}\to\{0,1\}$, $S(x)=1$ if $x$ is entailed by the edit. Measured in practice by *construction, not evaluation*: benchmark authors write neighborhood prompts by template and declare them out-of-scope. $S$ is never estimated from the model.
- **Probe distribution** $\mathcal{D}$. Measured as an empirical set: CounterFact uses ~10 prompts about distinct subjects sharing the edited relation and object; zsRE uses held-out questions; LEME uses long-form generations.
- **Divergence** $d$. Three incompatible instruments in use:
  - argmax agreement, $d_{\text{acc}} = \mathbb{1}\{\arg\max f_{\theta'}(x) \neq \arg\max f_\theta(x)\}$ — bounded in $[0,1]$, insensitive to logit shifts that have not yet flipped the top token;
  - margin flip, $d_{\text{mag}} = \mathbb{1}\{p_{\theta'}(y^{\text{new}}\!\mid x) > p_{\theta'}(y^{\text{old}}\!\mid x)\}$ — CounterFact's Neighborhood Magnitude;
  - $d_{\text{KL}} = \mathrm{KL}\!\left(f_\theta(x)\,\|\,f_{\theta'}(x)\right)$ — unbounded, no calibrated threshold, not comparable across models.

Reported "Locality %" is $1 - L$ under $d_{\text{acc}}$ or $d_{\text{mag}}$.

Assumptions, with the ones known to fail marked:

1. $S$ is binary. **Violated.** Ripple effects are graded: two-hop consequences of an edit are partly in scope (Cohen et al., TACL 2024).
2. The neighborhood set is out-of-scope. **Violated.** CounterFact neighborhood prompts share the edited object, so an editor that merely suppresses $y^{\text{new}}$ everywhere except $x_e$ scores well without generalising.
3. Single-edit locality composes over $n$ edits. **Violated.** Sequential editing degrades far faster than per-edit locality predicts.
4. $\theta$'s pre-edit answer is the reference. Fine for $d_{\text{acc}}$, but it makes locality reward preserving pre-existing errors.

## 3. State of the Art

**Established.**
- CounterFact's Neighborhood Score/Magnitude and the harmonic-mean summary $S$ (Meng et al., NeurIPS 2022) is the de facto standard; MEMIT (ICLR 2023) reuses it unchanged.
- CounterFact+ (Hoelscher-Obermaier et al., ACL Findings 2023) established that the standard neighborhood set is too easy: prompts that mention the edited subject *and* a neighbor cause large specificity drops that CounterFact does not detect. This is a reproduced negative result about the metric, not about an editor.
- RippleEdits (Cohen et al., TACL 2024) established that logical consequences of edits are systematically missed, so part of what locality counts as "must not change" is actually "must change".
- EasyEdit/KnowEdit (Zhang et al., 2024) established that renaming is not harmonisation: their "Locality" is computed on a different probe set than CounterFact's, so cross-paper locality numbers are not comparable.

**Claimed but unablated.**
- That KL-based locality is strictly better than accuracy-based. Asserted in several papers; no study varies only $d$ with $S,\mathcal{D}$ held fixed and reports rank changes.
- That LLM-as-judge long-form locality (LEME, Rosati et al., NAACL 2024) correlates with short-form locality. Reported correlations are low, which the paper frames as short-form being inadequate; the reverse reading is not excluded.

**Benchmark-number-only.** Every headline "Locality 97%" is a benchmark number under one $(S,\mathcal{D},d)$ triple. There is no result showing the ordering it induces survives a change of triple.

## 4. What Is Known

- ROME on GPT-2 XL (1.5B) and GPT-J (6B), CounterFact, single edits: efficacy ≈100%, neighborhood success ≈75–79%. Same checkpoints, harder CounterFact+ prompts: specificity falls sharply (Hoelscher-Obermaier et al., 2023) — the drop is caused by the probe set, not the editor.
- MEMIT, GPT-J (6B), 10,000 sequential edits: neighborhood success stays in the ~70s while paraphrase generalisation falls — locality remains high while the model is measurably worse.
- Gu et al. (EMNLP 2024) measured general-ability drawdown on eight downstream tasks and found significant degradation after a few dozen ROME/MEMIT edits on models up to 7B, at edit counts where CounterFact locality still reads >90%. This is the sharpest evidence that locality does not measure collateral damage.
- Gupta et al. (ACL Findings 2024; EMNLP 2024) showed sequential editing produces gradual then catastrophic forgetting and, for ROME, outright model collapse traceable to a small number of "disabling" edits — again invisible to per-edit neighborhood scores.
- Yang et al. (ACL Findings 2024) showed few edits can collapse a model, with collapse detectable by perplexity but not by the standard locality triple.
- Hase et al. (NeurIPS 2023) showed causal-tracing localisation does not predict where an edit can be made successfully — so "locality" in the parameter sense and in the behavioural sense are not the same quantity.

## 5. What Is Not Known

- **Methodologically blocked (the core).** There is no operational definition of $S$. Without it, $L$ is defined only up to a choice of probe set, and every reported locality number is a statement about the benchmark authors' template-writing, not about the model.
- **Methodologically blocked.** No calibration for $d_{\text{KL}}$: nobody can say what KL value counts as "the model changed".
- **Empirically open.** Rank stability. Take 6 editors × 4 probe sets × 3 divergences on a 7B model; measure Kendall's $\tau$ between rankings. Runnable in a few GPU-days. Unrun.
- **Empirically open.** Whether any single-edit locality metric predicts $n$-edit downstream drawdown. Requires sequential-edit runs plus a held-out capability suite.
- **Theoretically open.** Whether an impossibility result holds: no scalar $L$ can be simultaneously rank-stable across scope definitions and sensitive to ripple-consistency failures. Nobody has stated it as a theorem, let alone proved it.

## 6. Why It Is Hard

**Absent ground truth plus non-identifiability.** The scope predicate $S$ is a fact about what the *edit* entails, and entailment over a model's belief state is not observable. Two editors can produce identical out-of-scope behaviour on every finite probe set and differ arbitrarily off it; no finite evaluation identifies $S$.

Compounding it: **the evaluation does not measure what it names.** CounterFact neighborhood prompts share the edited relation and object, so they test object-suppression, not global preservation. An editor that overfits to "do not say Rome unless asked about the Eiffel Tower" scores 79% and still collapses at 100 edits. The metric's failure mode is aligned with the editors' failure mode, which is why locality scores stayed high through a decade of results showing editors damage models.

Compute is *not* the obstruction here — the deciding experiments fit on a single 8×A100 node.

## 7. Current Research (as of 2026)

- **Scope-explicit benchmarks.** RippleEdits-style graded scope, extended to multi-hop (MQuAKE, Zhong et al., EMNLP 2023) — the direction is to replace the binary $S$ with an entailment graph and score consistency over it.
- **Capability-preservation as locality.** Gu et al.'s regularisation line and follow-ups treat downstream-task drawdown as the real locality signal. *(frontier — verify)* Several 2025–2026 papers propose replacing neighborhood scores with a fixed general-ability suite; no standardisation yet.
- **Distributional locality.** KL/perplexity over a general corpus rather than curated prompts, used as a collapse detector after the ROME-collapse results. Calibration remains unaddressed.
- **Tooling.** EasyEdit (Zhejiang University NLP group) is the de facto harness and its metric choices are becoming the field default by inertia rather than by argument. *(frontier — verify)* Whether EasyEdit v2+ has changed its locality definition should be checked against the current repo.

## 8. Concrete Next Experiment

**Rank-stability of locality metrics.**

- **Scale.** One 7B model (Llama-2-7B or GPT-J-6B). 6 editors: ROME, MEMIT, MEND, SERAC, FT-L, and a null editor ($\theta'=\theta$). 1,000 CounterFact edits, applied both singly and sequentially in batches of $\{1, 10, 100, 1000\}$.
- **Metric grid.** 4 probe sets $\mathcal{D}$ (CounterFact neighborhood, CounterFact+, zsRE held-out, 2,000 random C4 prefixes) × 3 divergences $d$ ($d_{\text{acc}}$, $d_{\text{mag}}$, $d_{\text{KL}}$) = 12 locality scores per editor per batch size.
- **Control arm.** The null editor (must score 1.0 everywhere; catches probe-set noise) plus a *sham* editor that adds isotropic Gaussian noise to the same MLP matrices ROME writes to, scaled to match ROME's $\|\Delta W\|_F$. A locality metric that cannot separate ROME from norm-matched noise is measuring update magnitude, not locality.
- **Deciding number.** Mean pairwise Kendall's $\tau$ between the 12 editor rankings. **$\bar\tau \geq 0.8$** means the choice of $(\mathcal{D},d)$ is immaterial and the field can standardise on any of them. **$\bar\tau < 0.5$** means published locality numbers do not transfer across papers and every cross-paper comparison in the literature is unsupported.
- **Second number, same run.** Spearman correlation between single-edit locality and post-1000-edit MMLU drop. Below ~0.3, single-edit locality has no predictive validity and should be retired as a headline metric.

Cost: roughly 200–400 GPU-hours on A100s.

## 9. Key References

- **[Foundational]** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. — arXiv:2202.05262
- **[Foundational]** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau. *Mass-Editing Memory in a Transformer.* ICLR 2023. — arXiv:2210.07229
- **[Foundational]** Nicola De Cao, Wilker Aziz, Ivan Titov. *Editing Factual Knowledge in Language Models.* EMNLP 2021. — arXiv:2104.08164
- **[Foundational]** Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, Christopher D. Manning. *Fast Model Editing at Scale.* ICLR 2022. — arXiv:2110.11309
- **[SOTA / metric critique]** Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas, Fazl Barez. *Detecting Edit Failures in Large Language Models: An Improved Specificity Benchmark.* ACL Findings 2023. — arXiv:2305.17553
- **[SOTA / metric critique]** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva. *Evaluating the Ripple Effects of Knowledge Editing in Language Models.* TACL 2024. — arXiv:2307.12976
- **[SOTA]** Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, Danqi Chen. *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.* EMNLP 2023. — arXiv:2305.14795
- **[SOTA]** Jia-Chen Gu, Hao-Xiang Xu, Jun-Yu Ma, Pan Lu, Zhen-Hua Ling, Kai-Wei Chang, Nanyun Peng. *Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue.* EMNLP 2024. — arXiv:2401.04700
- **[SOTA]** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting.* ACL Findings 2024. — arXiv:2401.07453
- **[Analysis]** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.* NeurIPS 2023. — arXiv:2301.04213
- **[Evaluation]** Domenic Rosati, Robie Gonzales, Jinkun Chen, Xuemin Yu, Yahya Kayani, Frank Rudzicz, Hassan Sajjad. *Long-form evaluation of model editing.* NAACL 2024. — arXiv:2402.09394
- **[Survey]** Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, Jundong Li. *Knowledge Editing for Large Language Models: A Survey.* ACM Computing Surveys, 2024. — arXiv:2310.16218
- **[Survey]** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, Ningyu Zhang. *Editing Large Language Models: Problems, Methods, and Opportunities.* EMNLP 2023. — arXiv:2305.13172

## 10. Worked Example

One edit on GPT-J-6B: $x_e =$ "The Eiffel Tower is located in", $y^{\text{old}}=$ "Paris", $y^{\text{new}}=$ "Rome". ROME, single edit. Efficacy: 1.0.

CounterFact gives 10 neighborhood prompts — other Paris landmarks, e.g. "The Louvre is located in", "Notre-Dame is located in".

| Probe set | $d$ | Result | Locality |
|---|---|---|---|
| CounterFact neighborhood (10) | $d_{\text{acc}}$ | 9/10 unchanged | **0.90** |
| CounterFact neighborhood (10) | $d_{\text{KL}}$ | mean KL 0.31 nats | **uninterpretable** |
| CounterFact+ style (10) | $d_{\text{acc}}$ | 5/10 unchanged | **0.50** |
| C4 prefixes (2,000) | $d_{\text{acc}}$ | 1,996/2,000 unchanged | **0.998** |

The CounterFact+ prompts differ only by mentioning the Eiffel Tower before asking about the Louvre: "The Eiffel Tower is in Rome. The Louvre is located in ___" → "Rome". Same edit, same model, same divergence, same edit-success — locality reads 0.90 or 0.50 depending on who wrote the prompts.

Now the scope question. "From the top of the Eiffel Tower you can see the ___" → the edited model says "Colosseum". Is that a locality violation or correct ripple propagation? Under CounterFact it is neither: the prompt is in no probe set. Under RippleEdits it is *required* behaviour. So the same generation is scored as a failure, a success, or nothing at all, by choice of benchmark.

Finally the composition check. Apply 100 such edits sequentially. Mean per-edit CounterFact locality stays ≈0.89. MMLU drops several points and perplexity on C4 rises — the regime Gu et al. and Gupta et al. document. Locality of 0.89 across 100 edits predicts a model that is fine. The model is not fine.

The obstruction is visible in one line: **the number moved from 0.998 to 0.50 without the model changing at all.** Only the definition of "elsewhere" changed.

---
*Part of the [AI Research catalog](../../README.md). Schema: [TEMPLATE.md](../../TEMPLATE.md).*